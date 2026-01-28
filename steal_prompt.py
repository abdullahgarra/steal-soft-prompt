#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steal_prompt.py

Steal (distill) a victim soft prompt in a black-box way using a different dataset
than the victim training set.

Uses canonical SoftPromptT5 from utils.py and scores classes via label-string likelihood.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

import utils
from utils import SoftPromptT5


# -----------------------------
# Task configs (same style as training)
# -----------------------------
TASKS = {
    "sst2": dict(glue_subset="sst2"),
    "qnli": dict(glue_subset="qnli"),
    "qqp": dict(glue_subset="qqp"),
    "mnli": dict(glue_subset="mnli"),
}

DEFAULT_VERBALIZERS = {
    "sst2": {"negative": "negative", "positive": "positive"},
    "qnli": {"entailment": "entailment", "not_entailment": "not entailment"},
    "qqp": {"not_duplicate": "not duplicate", "duplicate": "duplicate"},
    "mnli": {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"},
}


def get_label_texts(task: str, ds_train) -> Tuple[List[str], List[str]]:
    feat = ds_train.features["label"]
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


# def collate_sst2_style(tokenizer: T5Tokenizer, max_length: int):
#     # used for Yelp/IMDB-style datasets (text-only)
#     def collate(batch):
#         texts = [f"sst2 sentence: {x['text']}" for x in batch]
#         enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
#         return enc["input_ids"], enc["attention_mask"]
#     return collate


# def collate_generic_text_field(tokenizer: T5Tokenizer, field: str, prefix: str, max_length: int):
#     def collate(batch):
#         texts = [f"{prefix}{x[field]}" for x in batch]
#         enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
#         return enc["input_ids"], enc["attention_mask"]
#     return collate



def make_steal_dataset(task: str, steal_ds: str, budget: int, args):
    """
    Returns:
      ds: HF Dataset (already sliced to budget)
      plan: dict describing how to build model input strings from ds rows

    plan keys:
      - type: "single" | "pair"
      - fields: for single -> [field]; for pair -> [field1, field2]
      - template: str with placeholders {a}, {b}
    """

    # Auto mapping (your choice)
    if steal_ds == "auto":
        if task in ["sst2", "mnli"]:
            print("[Info] auto-selecting snli as steal_ds")
            steal_ds = "snli"
        elif task == "qqp":
            print("[Info] auto-selecting stsb as steal_ds")
            steal_ds = "stsb"
        elif task == "qnli":
            print("[Info] auto-selecting qqp as steal_ds")
            steal_ds = "mnli"
        else:
            raise ValueError(f"auto steal_ds not defined for task={task}")

    # ---- SNLI (premise/hypothesis) for SST-2 or MNLI stealing inputs ----
    if steal_ds == "snli":
        # SNLI split names: train/validation/test
        ds = load_dataset("stanfordnlp/snli", split="train")
        ds = ds.shuffle(seed=args.seed).select(range(args.budget))

        if task == "sst2":
            # We still need a single "sentence:" style input for SST-2
            # Use hypothesis as a sentence; it's usually shorter/cleaner.
            plan = {
                "type": "single",
                "fields": ["hypothesis"],
                "template": "sst2 sentence: {a}",
            }
            return ds, plan

        if task == "mnli":
            # MNLI wants premise+hypothesis formatted
            plan = {
                "type": "pair",
                "fields": ["premise", "hypothesis"],
                "template": "mnli premise: {a} hypothesis: {b}",
            }
            return ds, plan

        raise ValueError(f"snli surrogate unsupported for task={task}")

    # ---- STS-B (sentence1/sentence2) for QQP stealing inputs ----
    if steal_ds == "stsb":
        ds = load_dataset("nyu-mll/glue", "stsb", split="train")
        ds = ds.shuffle(seed=args.seed).select(range(args.budget))

        # QQP expects question1/question2 style input
        plan = {
            "type": "pair",
            "fields": ["sentence1", "sentence2"],
            "template": "qqp question1: {a} question2: {b}",
        }
        return ds, plan

    # # ---- QQP (question1/question2) for QNLI stealing inputs ----
    # if steal_ds == "qqp":
    #     #ds = load_dataset("glue", "qqp", split=f"train[:{budget}]")
    #     ds = load_dataset("nyu-mll/glue","qqp",split="train")
    #     ds = ds.shuffle(seed=args.seed).select(range(args.budget))

    #     # QNLI expects question+sentence (premise) style input
    #     # We'll map qqp question1->question, question2->sentence.
    #     plan = {
    #         "type": "pair",
    #         "fields": ["question1", "question2"],
    #         "template": "qnli question: {a} sentence: {b}",
    #     }
    #     return ds, plan
    
    if steal_ds == "mnli":
        ds = load_dataset("nyu-mll/glue", "mnli", split="train")
        ds = ds.shuffle(seed=args.seed).select(range(args.budget))

        # QNLI expects question+sentence (premise) style input
        # We'll map mnli premise->question, hypothesis->sentence.
        plan = {
            "type": "pair",
            "fields": ["premise", "hypothesis"],
            "template": "qnli question: {a} sentence: {b}",
        }
        return ds, plan

    raise ValueError(f"Unknown steal_ds={steal_ds}")


# def make_steal_dataset(task: str, steal_ds: str, budget: int):
#     """
#     Return a huggingface Dataset and a collate_fn plan.
#     Keep it simple: start with text-only corpora to query the victim.
#     """
#     # sensible defaults that are "different from GLUE"
#     if steal_ds == "auto":
#         if task == "sst2":
#             steal_ds = "yelp_polarity"
#         else:
#             # for NLI / paraphrase, you can still distill on generic text
#             steal_ds = "ag_news"

#     if steal_ds == "yelp_polarity":
#         ds = load_dataset("yelp_polarity", split=f"train[:{budget}]")
#         return ds, {"type": "text", "field": "text", "prefix": "sst2 sentence: "}

#     if steal_ds == "imdb":
#         ds = load_dataset("imdb", split=f"train[:{budget}]")
#         return ds, {"type": "text", "field": "text", "prefix": "sst2 sentence: "}

#     if steal_ds == "ag_news":
#         ds = load_dataset("ag_news", split=f"train[:{budget}]")
#         return ds, {"type": "text", "field": "text", "prefix": f"{task} sentence: "}

#     raise ValueError(f"Unknown steal_ds={steal_ds}")

def collate_from_plan(tokenizer, plan, max_length):
    def _clean(v):
        if v is None:
            return ""
        if not isinstance(v, str):
            v = str(v)
        return v.strip()

    def collate(batch):
        texts = []
        for x in batch:
            if plan["type"] == "single":
                a = _clean(x.get(plan["fields"][0], ""))
                if a == "":
                    continue
                texts.append(plan["template"].format(a=a))
            else:
                f1, f2 = plan["fields"]
                a = _clean(x.get(f1, ""))
                b = _clean(x.get(f2, ""))
                if a == "" and b == "":
                    continue
                texts.append(plan["template"].format(a=a, b=b))

        # If we filtered everything, fallback to a dummy single example to avoid crash
        if len(texts) == 0:
            texts = ["dummy"]

        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        return enc["input_ids"], enc["attention_mask"]
    return collate




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASKS.keys()))
    parser.add_argument("--victim_ckpt", type=str, required=True)
    parser.add_argument("--out_ckpt", type=str, required=True)

    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--prompt_len", type=int, default=0, help="0 means infer from victim_ckpt shape")
    parser.add_argument("--oracle", type=str, default="probs", choices=["probs", "hard"],
                    help="Victim access: probs=distillation (KL), hard=labels only (CE on argmax).")


    parser.add_argument("--steal_ds", type=str, default="auto", help="auto|yelp_polarity|imdb|ag_news|...")
    parser.add_argument("--budget", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(args.model)

    # infer prompt length from checkpoint if needed
    victim_prompt = torch.load(args.victim_ckpt, map_location="cpu")
    P = int(victim_prompt.size(0))
    if args.prompt_len not in (0, P):
        raise ValueError(f"prompt_len mismatch: ckpt P={P} vs --prompt_len={args.prompt_len}")
    prompt_len = P

    # Build models
    victim_base = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    attacker_base = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    victim_base.eval()
    attacker_base.eval()

    victim = SoftPromptT5(victim_base, prompt_len=prompt_len).to(device)
    attacker = SoftPromptT5(attacker_base, prompt_len=prompt_len).to(device)

    victim.soft_prompt.data = victim_prompt.to(device)

    victim.eval()
    attacker.train()
    attacker.base_model.eval()  # keep dropout off

    optimizer = torch.optim.AdamW([attacker.soft_prompt], lr=args.lr)

    # label verbalizers from GLUE dataset (consistent with training/eval)
    glue_subset = TASKS[args.task]["glue_subset"]
    glue_ds = load_dataset("nyu-mll/glue", glue_subset)
    label_names, label_texts = get_label_texts(args.task, glue_ds["train"])
    label_token_ids = tokenize_label_texts(tokenizer, label_texts)

    print(f"[Task] {args.task} | labels={label_names} | verbalizers={label_texts}")
    print(f"[Victim] {args.victim_ckpt} | prompt_len={prompt_len}")
    print(f"[Steal] steal_ds={args.steal_ds} budget={args.budget} epochs={args.epochs} bs={args.batch_size} lr={args.lr}")

    ds, plan = make_steal_dataset(args.task, args.steal_ds, args.budget, args)
    print(f"[Steal] ds_len={len(ds)}")

    # # collate
    # if plan["type"] == "text":
    #     collate = collate_generic_text_field(
    #         tokenizer=tokenizer,
    #         field=plan["field"],
    #         prefix=plan["prefix"],
    #         max_length=args.max_length,
    #     )
    # else:
    #     raise ValueError("Unsupported plan type.")

    collate = collate_from_plan(
        tokenizer=tokenizer,
        plan=plan,
        max_length=args.max_length,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
        collate_fn=collate,
    )

    # --- distill ---
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Steal {args.task} | epoch {epoch+1}/{args.epochs}")
        for input_ids, attn in pbar:
            input_ids = input_ids.to(device)
            attn = attn.to(device)

            with torch.no_grad():
                v_scores = victim.class_logits_label_strings(
                    input_ids=input_ids,
                    attention_mask=attn,
                    label_token_ids=label_token_ids,
                    normalize_by_length=True,
                )

            a_scores = attacker.class_logits_label_strings(
                input_ids=input_ids,
                attention_mask=attn,
                label_token_ids=label_token_ids,
                normalize_by_length=True,
            )
            if args.oracle == "hard":
                y_hat = v_scores.argmax(dim=-1)              # [B]
                loss = F.cross_entropy(a_scores, y_hat)
                # if epoch == 0:
                #     print("[Debug] steal_ds_len =", len(ds))
                #     print("[Debug] batch_size    =", input_ids.size(0))
                #     print("[Debug] victim y_hat  =", y_hat.detach().cpu().tolist())
                #     print("[Debug] victim hist   =", dict(Counter(y_hat.detach().cpu().tolist())))
            else:  # probs 
                v_probs = F.softmax(v_scores, dim=-1)
                a_log_probs = F.log_softmax(a_scores, dim=-1)
                loss = F.kl_div(a_log_probs, v_probs, reduction="batchmean")
                # if epoch == 0:
                #     print("[Debug] steal_ds_len =", len(ds))
                #     print("[Debug] batch_size    =", input_ids.size(0))
                #     print("[Debug] v_probs       =\n", v_probs.detach().cpu())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([attacker.soft_prompt], 1.0)
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()))

    # Save stolen prompt
    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save(attacker.soft_prompt.detach().cpu(), args.out_ckpt)
    print(f"Saved stolen prompt to {args.out_ckpt}")

    # Optional log
    if args.log_path:
        log = {
            "task": args.task,
            "victim_ckpt": args.victim_ckpt,
            "out_ckpt": args.out_ckpt,
            "steal_ds": args.steal_ds,
            "budget": args.budget,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "label_names": label_names,
            "label_texts": label_texts,
        }
        with open(args.log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"Wrote log to {args.log_path}")


if __name__ == "__main__":
    main()
