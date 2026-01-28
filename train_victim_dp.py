#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_victim_dp.py

DP-SGD training for a *victim* soft prompt (T5 frozen backbone) on GLUE tasks:
SST-2, QNLI, QQP, MNLI.

Key design:
- Uses ONE canonical SoftPromptT5 from utils.py (same as non-DP / steal / eval).
- Computes per-example gradients w.r.t. soft_prompt only (vmap/grad).
- Clips per-example grads to L2 norm C.
- Adds Gaussian noise to summed clipped grads with std = sigma * C.
- Updates soft_prompt with the noised average gradient.
- Accounts privacy approximately using Opacus RDP accountant / get_noise_multiplier (Poisson sampling assumption).

Example:
  python train_victim_dp.py --task sst2 --epsilon 8 --epochs 5 --batch_size 64 --clip_C 0.1
  python train_victim_dp.py --task mnli --epsilon 8 --epochs 1 --batch_size 16 --clip_C 0.5
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm

from torch.func import vmap, grad

import utils
from utils import SoftPromptT5


# Optional: Opacus helper to compute sigma
try:
    from opacus.accountants.utils import get_noise_multiplier
except Exception:
    get_noise_multiplier = None


# -----------------------------
# Task config (same as train_victim.py)
# -----------------------------
@dataclass(frozen=True)
class TaskSpec:
    glue_subset: str
    val_split: str
    def format_input(self, ex: Dict[str, Any]) -> str:
        raise NotImplementedError


class SST2(TaskSpec):
    def __init__(self):
        super().__init__(glue_subset="sst2", val_split="validation")
    def format_input(self, ex: Dict[str, Any]) -> str:
        return f"sst2 sentence: {ex['sentence']}"


class QNLI(TaskSpec):
    def __init__(self):
        super().__init__(glue_subset="qnli", val_split="validation")
    def format_input(self, ex: Dict[str, Any]) -> str:
        return f"qnli question: {ex['question']} sentence: {ex['sentence']}"


class QQP(TaskSpec):
    def __init__(self):
        super().__init__(glue_subset="qqp", val_split="validation")
    def format_input(self, ex: Dict[str, Any]) -> str:
        return f"qqp question1: {ex['question1']} question2: {ex['question2']}"


class MNLI(TaskSpec):
    def __init__(self):
        super().__init__(glue_subset="mnli", val_split="validation_matched")
    def format_input(self, ex: Dict[str, Any]) -> str:
        return f"mnli premise: {ex['premise']} hypothesis: {ex['hypothesis']}"


TASKS: Dict[str, TaskSpec] = {
    "sst2": SST2(),
    "qnli": QNLI(),
    "qqp": QQP(),
    "mnli": MNLI(),
}

DEFAULT_VERBALIZERS: Dict[str, Dict[str, str]] = {
    "sst2": {"negative": "negative", "positive": "positive"},
    "qnli": {"entailment": "entailment", "not_entailment": "not entailment"},
    "qqp": {"not_duplicate": "not duplicate", "duplicate": "duplicate"},
    "mnli": {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"},
}


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_label_texts(task: str, ds_train) -> Tuple[List[str], List[str]]:
    feat = ds_train.features["label"]
    if not hasattr(feat, "names") or feat.names is None:
        raise ValueError("Dataset label feature does not expose .names; cannot build verbalizers safely.")
    label_names = list(feat.names)
    mapping = DEFAULT_VERBALIZERS.get(task, {})
    label_texts = [mapping.get(name, name.replace("_", " ")) for name in label_names]
    return label_names, label_texts


def tokenize_label_texts(tokenizer: T5Tokenizer, label_texts: List[str]) -> List[List[int]]:
    out: List[List[int]] = []
    for t in label_texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if len(ids) < 1:
            raise ValueError(f"Label text tokenized to empty ids: {t!r}")
        out.append(ids)
    return out


def collate_builder(tokenizer: T5Tokenizer, spec: TaskSpec, max_length: int):
    def collate(batch: List[Dict[str, Any]]):
        texts = [spec.format_input(ex) for ex in batch]
        labels = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        return enc["input_ids"], enc["attention_mask"], labels
    return collate


def compute_sigma(target_epsilon: float, delta: float, sample_rate: float, steps: int) -> float:
    """
    Compute DP-SGD noise multiplier sigma such that Opacus RDP accountant yields ~target (eps, delta).
    Assumes Poisson sampling with rate=sample_rate.
    """
    if steps <= 0:
        raise ValueError("steps must be > 0 for sigma computation.")
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError(f"sample_rate must be in (0,1], got {sample_rate}.")

    if get_noise_multiplier is not None:
        return float(
            get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=delta,
                sample_rate=sample_rate,
                steps=steps,
            )
        )

    print("[DP] Opacus get_noise_multiplier not found; using binary search for sigma.")
    from opacus.accountants import RDPAccountant

    low, high = 0.01, 200.0
    for _ in range(60):
        mid = (low + high) / 2.0
        acc = RDPAccountant()
        for __ in range(steps):
            acc.step(noise_multiplier=mid, sample_rate=sample_rate)
        eps = acc.get_epsilon(delta=delta)
        if eps > target_epsilon:
            low = mid
        else:
            high = mid
    return float(high)


@torch.no_grad()
def evaluate(
    model: SoftPromptT5,
    loader: DataLoader,
    label_token_ids: List[List[int]],
    device: str,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    conf_sum = 0.0
    pos_count = 0
    K = len(label_token_ids)

    for input_ids, attn, labels in loader:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        labels = labels.to(device)

        logits = model.class_logits_label_strings(
            input_ids=input_ids,
            attention_mask=attn,
            label_token_ids=label_token_ids,
            normalize_by_length=True,
        )  # [B,K]

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        conf_sum += probs.max(dim=-1).values.sum().item()

        if K == 2:
            pos_count += (preds == 1).sum().item()

    acc = correct / max(total, 1)
    avg_conf = conf_sum / max(total, 1)
    prompt_norm = float(model.soft_prompt.detach().norm().item())
    pos_rate = (pos_count / max(total, 1)) if K == 2 else float("nan")
    return {
        "acc": float(acc),
        "avg_conf": float(avg_conf),
        "prompt_norm": float(prompt_norm),
        "pos_rate": float(pos_rate),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sst2", choices=sorted(TASKS.keys()))
    parser.add_argument("--model", type=str, default="t5-small")

    # train params
    parser.add_argument("--prompt_len", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    # DP params
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip_C", type=float, default=0.1)

    # saving
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--log_path", type=str, default="")  # if empty, auto under save_dir
    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = TASKS[args.task]

    ensure_dir(args.save_dir)
    save_name = args.save_name or f"victim_dp_{args.task}_eps{args.epsilon}_P{args.prompt_len}_seed{args.seed}.pt"
    save_path = os.path.join(args.save_dir, save_name)
    log_path = args.log_path or os.path.join(
        args.save_dir, f"victim_dp_{args.task}_eps{args.epsilon}_P{args.prompt_len}_seed{args.seed}.json"
    )

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    base_model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    base_model.eval()

    model = SoftPromptT5(base_model, prompt_len=args.prompt_len).to(device)
    optimizer = torch.optim.AdamW([model.soft_prompt], lr=args.lr)

    # dataset
    ds = load_dataset("nyu-mll/glue", spec.glue_subset)
    train_ds = ds["train"]
    val_ds = ds[spec.val_split]

    label_names, label_texts = load_label_texts(args.task, train_ds)
    label_token_ids = tokenize_label_texts(tokenizer, label_texts)

    print(f"[Task] {args.task} (GLUE/{spec.glue_subset}) | labels={label_names} | verbalizers={label_texts}")
    print(f"[Model] {args.model} | prompt_len={args.prompt_len} | prompt_params={args.prompt_len * base_model.config.d_model:,}")
    print(f"[DP] eps={args.epsilon} C={args.clip_C}")
    print(f"[Save] {save_path}")
    print(f"[Log]  {log_path}")

    collate = collate_builder(tokenizer, spec, args.max_length)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # stabilize steps/epoch for accountant
        collate_fn=collate,
        num_workers=0,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        num_workers=0,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    # approximate Poisson sampling rate for accountant
    sample_rate = args.batch_size / len(train_ds)
    delta = 1/len(train_ds)
    sigma = compute_sigma(args.epsilon, delta, sample_rate, total_steps)

    print(f"[DP] batch_size={args.batch_size} steps={total_steps} sample_rate={sample_rate:.6f} sigma={sigma:.4f}")

    # -----------------------------
    # DP per-example loss for vmap/grad
    # -----------------------------
    # We'll implement the label-string likelihood scoring *exactly like utils.SoftPromptT5*,
    # but parameterized by soft_prompt tensor so grad() works.
    #
    # To be fast and stable:
    # - We compute encoder once per example (with prompt injected).
    # - For each label, run decoder to score the label tokens.
    #
    # NOTE: This is heavier than single-token, but required for multi-task correctness.
    #

    decoder_start = base_model.config.decoder_start_token_id
    if decoder_start is None:
        raise ValueError("decoder_start_token_id is None")

    # Pre-pack label_token_ids into torch tensors for the device inside the loss
    label_token_ids_t = [torch.tensor(ids, dtype=torch.long) for ids in label_token_ids]

    def loss_one(
        soft_prompt: torch.Tensor,          # [P, d]
        input_ids_1: torch.Tensor,          # [T]
        attn_1: torch.Tensor,               # [T]
        y_1: torch.Tensor,                  # scalar
    ) -> torch.Tensor:
        """
        Returns scalar CE loss for one example using label-string likelihood scores.
        """
        device_ = input_ids_1.device

        # ---- inject prompt into encoder inputs (1-example batch) ----
        embeds = base_model.encoder.embed_tokens(input_ids_1.unsqueeze(0))  # [1,T,d]
        prompt = soft_prompt.unsqueeze(0)                                  # [1,P,d]
        inputs_embeds = torch.cat([prompt, embeds], dim=1)                 # [1,P+T,d]

        P = soft_prompt.size(0)
        prompt_mask = torch.ones((1, P), device=device_, dtype=attn_1.dtype)
        attn2 = torch.cat([prompt_mask, attn_1.unsqueeze(0)], dim=1)       # [1,P+T]

        # run encoder once, reuse outputs
        enc_out = base_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn2,
            return_dict=True,
        )

        # ---- compute label scores (log-likelihoods) ----
        scores = []
        for ids in label_token_ids_t:
            ids = ids.to(device_)
            L = ids.numel()

            dec_in = torch.full((1, 1), decoder_start, dtype=torch.long, device=device_)
            if L > 1:
                dec_in = torch.cat([dec_in, ids[:-1].view(1, -1)], dim=1)  # [1,L]

            targets = ids.view(1, -1)  # [1,L]

            out = base_model(
                encoder_outputs=enc_out,
                attention_mask=attn2,
                decoder_input_ids=dec_in,
                return_dict=True,
            )
            logits = out.logits  # [1,L,V]

            nll_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(1, L)

            # normalize by length to reduce bias from varying label lengths
            nll = nll_tok.sum(dim=1) / float(L)  # [1]
            scores.append((-nll).view(1, 1))      # [1,1]

        class_scores = torch.cat(scores, dim=1)  # [1,K]
        return F.cross_entropy(class_scores, y_1.view(1))

    grad_one = grad(loss_one)  # grad w.r.t. soft_prompt
    grad_batch = vmap(grad_one, in_dims=(None, 0, 0, 0))  # vectorize over batch items

    # -----------------------------
    # Logging
    # -----------------------------
    history: Dict[str, Any] = {
        "task": args.task,
        "model": args.model,
        "prompt_len": args.prompt_len,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "seed": args.seed,
        "dp": {
            "epsilon": args.epsilon,
            "clip_C": args.clip_C,
            "sigma": sigma,
            "sample_rate": sample_rate,
            "steps": total_steps,
        },
        "label_names": label_names,
        "label_texts": label_texts,
        "save_path": save_path,
        "epochs_log": [],
    }

    m0 = evaluate(model, val_loader, label_token_ids, device)
    print(f"Epoch 0 | val acc={m0['acc']:.4f} | conf={m0['avg_conf']:.3f} | ||P||={m0['prompt_norm']:.2f} | pos_rate={m0['pos_rate']:.3f}")
    history["epoch0"] = m0
    save_json(log_path, history)

    best_acc = -1.0
    C = float(args.clip_C)

    # -----------------------------
    # DP training loop
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"DP Train {args.task} | epoch {epoch}/{args.epochs}")

        for input_ids, attn, labels in pbar:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            # per-example gradients: [B, P, d]
            per_ex_grads = grad_batch(model.soft_prompt, input_ids, attn, labels)

            B = per_ex_grads.size(0)
            flat = per_ex_grads.view(B, -1)
            norms = flat.norm(p=2, dim=1).clamp_min(1e-12)  # [B]
            scales = (C / norms).clamp_max(1.0)            # [B]
            per_ex_grads = per_ex_grads * scales.view(B, 1, 1)

            # sum + noise, then average
            grad_sum = per_ex_grads.sum(dim=0)             # [P,d]
            noise = torch.randn_like(grad_sum) * (sigma * C)
            grad_noised_avg = (grad_sum + noise) / float(B)

            optimizer.zero_grad(set_to_none=True)
            model.soft_prompt.grad = grad_noised_avg
            optimizer.step()

        m = evaluate(model, val_loader, label_token_ids, device)
        print(f"Epoch {epoch} | val acc={m['acc']:.4f} | conf={m['avg_conf']:.3f} | ||P||={m['prompt_norm']:.2f} | pos_rate={m['pos_rate']:.3f}")

        entry = {"epoch": epoch, **m}
        history["epochs_log"].append(entry)
        save_json(log_path, history)

        if m["acc"] > best_acc:
            best_acc = m["acc"]
            model.save_prompt(save_path)
            print(f"  ✓ Saved best DP prompt (acc={best_acc:.4f}) to {save_path}")

    print(f"Done. Best val acc={best_acc:.4f} | saved={save_path}")
    save_json(log_path, history)


if __name__ == "__main__":
    main()
