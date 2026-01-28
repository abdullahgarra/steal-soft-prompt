#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py

Evaluate victim + stolen models with CSV logging for systematic analysis.
"""

import os
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

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


def load_label_texts(task: str, ds_train) -> Tuple[List[str], List[str]]:
    feat = ds_train.features["label"]
    if not hasattr(feat, "names") or feat.names is None:
        raise ValueError("Dataset label feature does not expose .names")
    label_names = list(feat.names)
    mapping = DEFAULT_VERBALIZERS.get(task, {})
    label_texts = [mapping.get(name, name.replace("_", " ")) for name in label_names]
    return label_names, label_texts


def tokenize_label_texts(tokenizer: T5Tokenizer, label_texts: List[str]) -> List[List[int]]:
    out = []
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


def f1_binary(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def load_model_from_prompt(model_name: str, prompt_path: str, device: str):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    base_model.eval()

    prompt = torch.load(prompt_path, map_location="cpu")
    prompt_len = int(prompt.size(0))

    model = SoftPromptT5(base_model, prompt_len=prompt_len).to(device)
    model.soft_prompt.data = prompt.to(device)
    model.eval()
    model.base_model.eval()

    return tokenizer, model, prompt, prompt_len


@torch.no_grad()
def eval_single(
    task: str,
    model_name: str,
    prompt_path: str,
    batch_size: int,
    seed: int,
    max_length: int,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = TASKS[task]
    tok, model, prompt, prompt_len = load_model_from_prompt(model_name, prompt_path, device)

    glue_ds = load_dataset("nyu-mll/glue", spec.glue_subset)
    label_names, label_texts = load_label_texts(task, glue_ds["train"])
    label_token_ids = tokenize_label_texts(tok, label_texts)
    K = len(label_token_ids)

    collate = collate_builder(tok, spec, max_length=max_length)
    val_ds = glue_ds[spec.val_split]

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        num_workers=0,
        generator=g,
    )

    correct = 0
    total = 0
    conf_sum = 0.0
    pos_count = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for input_ids, attn, labels in tqdm(loader, desc=f"Eval {os.path.basename(prompt_path)}", leave=False):
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        labels = labels.to(device)

        scores = model.class_logits_label_strings(
            input_ids=input_ids,
            attention_mask=attn,
            label_token_ids=label_token_ids,
            normalize_by_length=True,
        )

        probs = F.softmax(scores, dim=-1)
        preds = probs.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        conf_sum += probs.max(dim=-1).values.sum().item()

        if K == 2:
            pos_count += (preds == 1).sum().item()

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = correct / max(total, 1)
    avg_conf = conf_sum / max(total, 1)
    pos_rate = (pos_count / max(total, 1)) if K == 2 else float("nan")
    prompt_norm = float(prompt.norm().item())

    f1 = f1_binary(y_true, y_pred) if K == 2 else float("nan")

    return {
        "task": task,
        "prompt_path": prompt_path,
        "prompt_len": prompt_len,
        "num_labels": K,
        "acc": float(acc),
        "f1": float(f1),
        "avg_conf": float(avg_conf),
        "pos_rate": float(pos_rate),
        "prompt_norm": float(prompt_norm),
    }


@torch.no_grad()
def eval_pair(
    task: str,
    model_name: str,
    victim_path: str,
    stolen_path: str,
    batch_size: int,
    seed: int,
    max_length: int,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = TASKS[task]
    tok_v, victim, pv, pv_len = load_model_from_prompt(model_name, victim_path, device)
    tok_s, stolen, ps, ps_len = load_model_from_prompt(model_name, stolen_path, device)

    tok = tok_v

    glue_ds = load_dataset("nyu-mll/glue", spec.glue_subset)
    label_names, label_texts = load_label_texts(task, glue_ds["train"])
    label_token_ids = tokenize_label_texts(tok, label_texts)

    collate = collate_builder(tok, spec, max_length=max_length)
    val_ds = glue_ds[spec.val_split]

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        num_workers=0,
        generator=g,
    )

    agree = 0
    total = 0
    kl_sum = 0.0

    for input_ids, attn, _labels in tqdm(loader, desc=f"Pair eval", leave=False):
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        v_scores = victim.class_logits_label_strings(
            input_ids=input_ids,
            attention_mask=attn,
            label_token_ids=label_token_ids,
            normalize_by_length=True,
        )
        s_scores = stolen.class_logits_label_strings(
            input_ids=input_ids,
            attention_mask=attn,
            label_token_ids=label_token_ids,
            normalize_by_length=True,
        )

        v_probs = F.softmax(v_scores, dim=-1)
        s_probs = F.softmax(s_scores, dim=-1)

        v_pred = v_probs.argmax(dim=-1)
        s_pred = s_probs.argmax(dim=-1)

        agree += (v_pred == s_pred).sum().item()
        total += input_ids.size(0)

        kl_batch = F.kl_div(torch.log(s_probs + 1e-12), v_probs, reduction="batchmean")
        kl_sum += float(kl_batch.item()) * input_ids.size(0)

    agreement = agree / max(total, 1)
    avg_kl = kl_sum / max(total, 1)

    l2_dist = float((pv - ps).norm().item())
    cos_sim = float(F.cosine_similarity(pv.flatten(), ps.flatten(), dim=0).item())

    return {
        "agreement": float(agreement),
        "avg_kl_victim_to_stolen": float(avg_kl),
        "prompt_l2_dist": l2_dist,
        "prompt_cos_sim": cos_sim,
    }


def save_results_to_csv(results: Dict[str, Any], output_path: str):
    """Save results to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Flatten the nested dictionary
    flat_results = {}
    for key, val in results.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat_results[f"{key}_{subkey}"] = subval
        else:
            flat_results[key] = val
    
    # Check if file exists to determine if we need header
    file_exists = os.path.exists(output_path)
    
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat_results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASKS.keys()))
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--victim_ckpt", type=str, required=True)
    parser.add_argument("--stolen_ckpt", type=str, required=True)

    parser.add_argument("--victim_dp_ckpt", type=str, required=True)
    parser.add_argument("--stolen_dp_ckpt", type=str, required=True)
    
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--budget", type=int, default=1000, help="Budget used for stealing (for logging)")
    parser.add_argument("--oracle", type=str, default="probs", help="Oracle type (for logging)")
    args = parser.parse_args()

    utils.set_seed(args.seed)

    print(f"\n{'='*80}")
    print(f"EVALUATING: {args.task.upper()} | seed={args.seed} | budget={args.budget} | oracle={args.oracle}")
    print(f"{'='*80}\n")

    # Evaluate all models
    print("Evaluating victim (no DP)...")
    m_v = eval_single(args.task, args.model, args.victim_ckpt, args.batch_size, args.seed, args.max_length)
    
    print("Evaluating stolen (no DP)...")
    m_s = eval_single(args.task, args.model, args.stolen_ckpt, args.batch_size, args.seed, args.max_length)

    print("Evaluating victim (DP)...")
    m_vdp = eval_single(args.task, args.model, args.victim_dp_ckpt, args.batch_size, args.seed, args.max_length)
    
    print("Evaluating stolen (DP)...")
    m_sdp = eval_single(args.task, args.model, args.stolen_dp_ckpt, args.batch_size, args.seed, args.max_length)

    print("Evaluating pair (no DP)...")
    p_nodp = eval_pair(args.task, args.model, args.victim_ckpt, args.stolen_ckpt, args.batch_size, args.seed, args.max_length)
    
    print("Evaluating pair (DP)...")
    p_dp = eval_pair(args.task, args.model, args.victim_dp_ckpt, args.stolen_dp_ckpt, args.batch_size, args.seed, args.max_length)

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {args.task.upper()}")
    print(f"{'='*80}")
    print(f"\n--- NO DP ---")
    print(f"Victim:  acc={m_v['acc']:.4f} | f1={m_v['f1']:.4f} | conf={m_v['avg_conf']:.3f} | pos_rate={m_v['pos_rate']:.3f}")
    print(f"Stolen:  acc={m_s['acc']:.4f} | f1={m_s['f1']:.4f} | conf={m_s['avg_conf']:.3f} | pos_rate={m_s['pos_rate']:.3f}")
    print(f"  → Agreement={p_nodp['agreement']:.4f} | KL(v||s)={p_nodp['avg_kl_victim_to_stolen']:.6f} | cos_sim={p_nodp['prompt_cos_sim']:.3f}")
    
    print(f"\n--- WITH DP ---")
    print(f"Victim:  acc={m_vdp['acc']:.4f} | f1={m_vdp['f1']:.4f} | conf={m_vdp['avg_conf']:.3f} | pos_rate={m_vdp['pos_rate']:.3f}")
    print(f"Stolen:  acc={m_sdp['acc']:.4f} | f1={m_sdp['f1']:.4f} | conf={m_sdp['avg_conf']:.3f} | pos_rate={m_sdp['pos_rate']:.3f}")
    print(f"  → Agreement={p_dp['agreement']:.4f} | KL(v||s)={p_dp['avg_kl_victim_to_stolen']:.6f} | cos_sim={p_dp['prompt_cos_sim']:.3f}")
    print(f"{'='*80}\n")

    # Save results to CSV
    output_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(output_dir, exist_ok=True)
    
    # NO DP results
    nodp_results = {
        "task": args.task,
        "seed": args.seed,
        "budget": args.budget,
        "oracle": args.oracle,
        "dp": False,
        "victim": m_v,
        "stolen": m_s,
        "pair": p_nodp,
    }
    nodp_csv = os.path.join(output_dir, f"results_nodp_{args.oracle}.csv")
    save_results_to_csv(nodp_results, nodp_csv)
    
    # DP results
    dp_results = {
        "task": args.task,
        "seed": args.seed,
        "budget": args.budget,
        "oracle": args.oracle,
        "dp": True,
        "victim": m_vdp,
        "stolen": m_sdp,
        "pair": p_dp,
    }
    dp_csv = os.path.join(output_dir, f"results_dp_{args.oracle}.csv")
    save_results_to_csv(dp_results, dp_csv)
    
    print(f"✓ Results saved to {output_dir}/")
    print(f"  - {nodp_csv}")
    print(f"  - {dp_csv}\n")


if __name__ == "__main__":
    main()
