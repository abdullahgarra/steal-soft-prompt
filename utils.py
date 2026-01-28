# train_victim.py (or a shared module)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os, random
import numpy as np
import torch
from typing import List, Sequence, Optional
from transformers import T5ForConditionalGeneration


class SoftPromptT5(nn.Module):
    """
    Canonical soft-prompt wrapper for T5 that is consistent across:
      - non-DP training
      - DP training (per-example grads)
      - stealing
      - evaluation

    Supports:
      (1) first-step logits (legacy SST-2 single-token setup)
      (2) general multi-token label-string log-likelihood scoring (for QNLI/QQP/MNLI, etc.)
    """

    def __init__(self, base_model: T5ForConditionalGeneration, prompt_len: int = 20):
        super().__init__()
        self.base_model = base_model

        # freeze backbone
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.d_model = self.base_model.config.d_model
        self.soft_prompt = nn.Parameter(torch.randn(prompt_len, self.d_model) * 0.01)

        self.decoder_start = self.base_model.config.decoder_start_token_id
        if self.decoder_start is None:
            raise ValueError("decoder_start_token_id is None for this T5 model.")

    @property
    def prompt_len(self) -> int:
        return int(self.soft_prompt.size(0))

    def train(self, mode: bool = True):
        super().train(mode)
        # keep frozen backbone deterministic
        self.base_model.eval()
        return self

    def save_prompt(self, path: str):
        torch.save(self.soft_prompt.detach().cpu(), path)

    @staticmethod
    def load_prompt(path: str, device: Optional[str] = None) -> torch.Tensor:
        return torch.load(path, map_location=device)

    def _build_inputs_with_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        soft_prompt: Optional[torch.Tensor] = None,
    ):
        """
        Build (inputs_embeds, extended_attention_mask) after prepending the soft prompt.
        soft_prompt: [P, d] (optional) overrides self.soft_prompt (useful for per-example grad funcs).
        """
        if soft_prompt is None:
            soft_prompt = self.soft_prompt

        B = input_ids.size(0)
        device = input_ids.device

        # token embeddings: [B, T, d]
        embeds = self.base_model.encoder.embed_tokens(input_ids)

        # prompt: [B, P, d]
        prompt = soft_prompt.unsqueeze(0).expand(B, -1, -1)

        # concat: [B, P+T, d]
        inputs_embeds = torch.cat([prompt, embeds], dim=1)

        # extend attention mask, preserving dtype
        P = soft_prompt.size(0)
        prompt_mask = torch.ones((B, P), device=device, dtype=attention_mask.dtype)
        attn2 = torch.cat([prompt_mask, attention_mask], dim=1)

        return inputs_embeds, attn2

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns first decoder-step logits: [B, vocab].
        (Used by your legacy SST-2 single-token label scoring.)
        """
        inputs_embeds, attn2 = self._build_inputs_with_prompt(input_ids, attention_mask)

        B = input_ids.size(0)
        decoder_input_ids = torch.full(
            (B, 1),
            self.decoder_start,
            dtype=torch.long,
            device=input_ids.device,
        )

        out = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn2,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        return out.logits[:, 0, :]  # [B, vocab]

    def class_logits_single_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_token_ids: Sequence[int],
    ) -> torch.Tensor:
        """
        Convenience for SST-2-style scoring where each class is a single token id.
        Returns [B, K] where K=len(label_token_ids).
        """
        logits = self.forward(input_ids, attention_mask)  # [B, vocab]
        idx = torch.tensor(list(label_token_ids), device=logits.device, dtype=torch.long)
        return logits.index_select(dim=1, index=idx)  # [B, K]

    def class_logits_label_strings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_token_ids: List[List[int]],
        normalize_by_length: bool = True,
    ) -> torch.Tensor:
        """
        Compute class logits via label-string log-likelihood (supports multi-token labels).

        label_token_ids: list of token-id lists, one per class (each list can be length >= 1)
        Returns: [B, K] scores.

        If normalize_by_length=True, uses average log-likelihood per token (helps when label lengths differ).
        """
        device = input_ids.device
        B = input_ids.size(0)
        K = len(label_token_ids)

        inputs_embeds, attn2 = self._build_inputs_with_prompt(input_ids, attention_mask)

        scores = []
        for ids_list in label_token_ids:
            if len(ids_list) < 1:
                raise ValueError("Each label must have at least 1 token id.")

            ids = torch.tensor(ids_list, dtype=torch.long, device=device)  # [L]
            L = ids.numel()

            # decoder inputs: [start] + ids[:-1]
            dec_in = torch.full((B, 1), self.decoder_start, dtype=torch.long, device=device)
            if L > 1:
                dec_in = torch.cat([dec_in, ids[:-1].view(1, -1).expand(B, -1)], dim=1)  # [B, L]

            targets = ids.view(1, -1).expand(B, -1)  # [B, L]

            out = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn2,
                decoder_input_ids=dec_in,
                return_dict=True,
            )
            logits = out.logits  # [B, L, V]

            # token-level NLL (sum over tokens)
            nll_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(B, L)

            nll = nll_tok.sum(dim=1)  # [B]
            if normalize_by_length:
                nll = nll / float(L)

            scores.append((-nll).unsqueeze(1))  # [B,1]

        return torch.cat(scores, dim=1)  # [B, K]


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
