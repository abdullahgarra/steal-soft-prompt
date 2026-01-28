#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_victim.py

Train a *victim* soft prompt for T5 (frozen backbone) on a GLUE task.
Supports: SST-2, QNLI, QQP, MNLI.

Key design:
- Uses ONE canonical SoftPromptT5 from utils.py (same class used by DP/steal/eval).
- Scores classes via label-string log-likelihood (multi-token safe).
- Saves best prompt checkpoint + a JSON log with metrics.

Example:
  python train_victim.py --task sst2 --epochs 5 --batch_size 64 --prompt_len 20 --lr 1e-2
  python train_victim.py --task mnli --epochs 3 --batch_size 32 --prompt_len 20 --lr 5e-3
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm

import utils
from utils import SoftPromptT5


# -----------------------------
# Task config
# -----------------------------
@dataclass(frozen=True)
class TaskSpec:
    glue_subset: str
    # which split to use as "validation"
    val_split: str
    # input formatting function: item -> str
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
        # MNLI has "validation_matched" and "validation_mismatched"
        super().__init__(glue_subset="mnli", val_split="validation_matched")

    def format_input(self, ex: Dict[str, Any]) -> str:
        return f"mnli premise: {ex['premise']} hypothesis: {ex['hypothesis']}"


TASKS: Dict[str, TaskSpec] = {
    "sst2": SST2(),
    "qnli": QNLI(),
    "qqp": QQP(),
    "mnli": MNLI(),
}

# Optional: nicer verbalizers than raw label names.
# We'll still fall back to dataset label names if not present.
DEFAULT_VERBALIZERS: Dict[str, Dict[str, str]] = {
    "sst2": {"negative": "negative", "positive": "positive"},
    "qnli": {
        # GLUE uses "entailment" and "not_entailment" (usually)
        "entailment": "entailment",
        "not_entailment": "not entailment",
    },
    "qqp": {
        # GLUE uses "not_duplicate" and "duplicate" (usually)
        "not_duplicate": "not duplicate",
        "duplicate": "duplicate",
    },
    "mnli": {
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
    },
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
    """
    Returns:
      label_names: list[str] from dataset feature names
      label_texts: list[str] verbalizers to score (same length)
    """
    feat = ds_train.features["label"]
    if not hasattr(feat, "names") or feat.names is None:
        raise ValueError("Dataset label feature does not expose .names; cannot build verbalizers safely.")
    label_names = list(feat.names)

    mapping = DEFAULT_VERBALIZERS.get(task, {})
    label_texts = [mapping.get(name, name.replace("_", " ")) for name in label_names]

    return label_names, label_texts


def tokenize_label_texts(tokenizer: T5Tokenizer, label_texts: List[str]) -> List[List[int]]:
    token_ids: List[List[int]] = []
    for t in label_texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if len(ids) < 1:
            raise ValueError(f"Label text tokenized to empty ids: {t!r}")
        token_ids.append(ids)
    return token_ids


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


@torch.no_grad()
def evaluate(
    model: SoftPromptT5,
    tokenizer: T5Tokenizer,
    loader: DataLoader,
    label_token_ids: List[List[int]],
    device: str,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    conf_sum = 0.0
    # for binary tasks, track pos rate to detect degenerate behavior
    pos_count = 0
    n_labels = len(label_token_ids)

    for input_ids, attn, labels in loader:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        labels = labels.to(device)

        logits = model.class_logits_label_strings(
            input_ids=input_ids,
            attention_mask=attn,
            label_token_ids=label_token_ids,
            normalize_by_length=True,
        )  # [B, K]

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        conf_sum += probs.max(dim=-1).values.sum().item()

        if n_labels == 2:
            pos_count += (preds == 1).sum().item()

    acc = correct / max(total, 1)
    avg_conf = conf_sum / max(total, 1)
    prompt_norm = float(model.soft_prompt.detach().norm().item())
    pos_rate = (pos_count / max(total, 1)) if n_labels == 2 else float("nan")

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
    parser.add_argument("--prompt_len", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_path", type=str, default="")  # optional; if empty, auto under save_dir
    parser.add_argument("--save_name", type=str, default="")  # optional; if empty, auto
    parser.add_argument("--eval_batch_size", type=int, default=64)
    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = TASKS[args.task]

    ensure_dir(args.save_dir)
    save_name = args.save_name or f"victim_{args.task}_P{args.prompt_len}_seed{args.seed}.pt"
    save_path = os.path.join(args.save_dir, save_name)

    log_path = args.log_path or os.path.join(
        args.save_dir, f"victim_{args.task}_P{args.prompt_len}_seed{args.seed}.json"
    )

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    base_model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    base_model.eval()

    model = SoftPromptT5(base_model, prompt_len=args.prompt_len).to(device)
    optimizer = torch.optim.AdamW([model.soft_prompt], lr=args.lr)

    # Load dataset
    ds = load_dataset("nyu-mll/glue", spec.glue_subset)
    train_ds = ds["train"]
    val_ds = ds[spec.val_split]

    # label names + verbalizers derived from dataset
    label_names, label_texts = load_label_texts(args.task, train_ds)
    label_token_ids = tokenize_label_texts(tokenizer, label_texts)

    print(f"[Task] {args.task} (GLUE/{spec.glue_subset}) | labels={label_names} | verbalizers={label_texts}")
    print(f"[Model] {args.model} | prompt_len={args.prompt_len} | prompt_params={args.prompt_len * base_model.config.d_model:,}")
    print(f"[Save] {save_path}")
    print(f"[Log]  {log_path}")

    collate = collate_builder(tokenizer, spec, args.max_length)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
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

    history: Dict[str, Any] = {
        "task": args.task,
        "model": args.model,
        "prompt_len": args.prompt_len,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "seed": args.seed,
        "label_names": label_names,
        "label_texts": label_texts,
        "save_path": save_path,
        "epochs_log": [],
    }

    # Epoch 0 eval
    m0 = evaluate(model, tokenizer, val_loader, label_token_ids, device)
    print(f"Epoch 0 | val acc={m0['acc']:.4f} | conf={m0['avg_conf']:.3f} | ||P||={m0['prompt_norm']:.2f} | pos_rate={m0['pos_rate']:.3f}")
    history["epoch0"] = m0

    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {args.task} | epoch {epoch}/{args.epochs}")

        total_loss = 0.0
        total_seen = 0

        for input_ids, attn, labels in pbar:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            logits = model.class_logits_label_strings(
                input_ids=input_ids,
                attention_mask=attn,
                label_token_ids=label_token_ids,
                normalize_by_length=True,
            )  # [B, K]

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.soft_prompt], 1.0)
            optimizer.step()

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            total_seen += bs
            pbar.set_postfix(loss=(total_loss / max(total_seen, 1)))

        avg_train_loss = total_loss / max(total_seen, 1)

        m = evaluate(model, tokenizer, val_loader, label_token_ids, device)
        print(f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | val acc={m['acc']:.4f} | conf={m['avg_conf']:.3f} | ||P||={m['prompt_norm']:.2f} | pos_rate={m['pos_rate']:.3f}")

        entry = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            **m,
        }
        history["epochs_log"].append(entry)

        if m["acc"] > best_acc:
            best_acc = m["acc"]
            model.save_prompt(save_path)
            print(f"  ✓ Saved best prompt (acc={best_acc:.4f}) to {save_path}")

        save_json(log_path, history)

    print(f"Done. Best val acc={best_acc:.4f} | saved={save_path}")
    save_json(log_path, history)


if __name__ == "__main__":
    main()
