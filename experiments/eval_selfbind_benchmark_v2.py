#!/usr/bin/env python3
"""Evaluate SelfBind on the explicitly annotated Prompt Benchmark v2.

The identity payload is fixed.  Only the semantic gate is trained from the
training portion of the legacy scope probes; the controlled clean,
contaminated, matched-pair, and true-multiturn pools remain evaluation-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.prompt_benchmark_v2 import build_records
from data.prompt_stress_v2 import build_stress_records
from experiments.qwen_loreft_train import DEFAULT_SYSTEM, TARGET_CONFIGS, WRONG_ENTITIES
from experiments.train_selfbind import (
    SelfBindGate,
    TEMPLATE_POSITIVE,
    extract_hidden,
    gate_probability,
    generate,
    train_gate,
)
from imm.identity_eval import classify_identity
from imm.model import load_model


def render_messages(tokenizer, messages: list[dict]) -> str:
    full_messages = list(messages)
    if not full_messages or full_messages[0].get("role") != "system":
        full_messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            full_messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(full_messages, **kwargs)


def split_gate_records(records: list[dict], seed: int, train_ratio: float) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["split"] == "legacy_scope_probe":
            buckets[record["category"]].append(record)
    train: list[dict] = [
        {"messages": [{"role": "user", "content": prompt}], "scope_label": 1, "record_id": f"template-{i}"}
        for i, prompt in enumerate(TEMPLATE_POSITIVE)
    ]
    test: list[dict] = []
    for category, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        n_train = max(1, min(len(bucket) - 1, int(round(len(bucket) * train_ratio))))
        train.extend(bucket[:n_train])
        test.extend(bucket[n_train:])
    return train, test


def collect_gate_features(model, tokenizer, records: list[dict], layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for index, record in enumerate(records, start=1):
        messages = record.get("messages") or [{"role": "user", "content": record["prompt"]}]
        text = render_messages(tokenizer, messages)
        # The base model is still unwrapped during gate feature extraction;
        # adapter disabling only becomes available after PeftModel loading.
        features.append(extract_hidden(model, tokenizer, text, layer))
        labels.append(float(record["scope_label"]))
        if index % 16 == 0 or index == len(records):
            print(f"[features] {index}/{len(records)}", flush=True)
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32)


def gate_metrics(gate: SelfBindGate, features: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict:
    probs = torch.sigmoid(gate(features.to(next(gate.parameters()).device))).detach().cpu()
    pred = probs.ge(threshold).float()
    positives = labels == 1
    negatives = labels == 0
    return {
        "total": len(labels),
        "accuracy": float((pred == labels).float().mean()),
        "positive_recall": float(((pred == 1) & positives).sum() / max(1, int(positives.sum()))),
        "negative_specificity": float(((pred == 0) & negatives).sum() / max(1, int(negatives.sum()))),
        "gate_active": int(pred.sum()),
    }


@torch.inference_mode()
def gate_probabilities(gate: SelfBindGate, features: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(gate(features.to(next(gate.parameters()).device))).detach().cpu()


def calibrate_threshold(probs: torch.Tensor, labels: torch.Tensor, min_specificity: float = 0.98) -> tuple[float, dict]:
    """Select a threshold on held-out gate labels, never on generation outputs."""

    # Include the low-probability operating region explicitly.  The v2 gate
    # often separates positive scope probes around .03--.05 from negatives
    # below .003; omitting .03 can unnecessarily sacrifice a relevant claim.
    # A tiny downward candidate next to every observed probability avoids
    # losing a positive to float32 round-off when using ``>=``.
    fixed = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    observed = [float(x) for x in probs]
    candidates = sorted(set(fixed + observed + [max(0.0, x - 1e-6) for x in observed]))
    best = None
    positives = labels == 1
    negatives = labels == 0
    for threshold in candidates:
        pred = probs.ge(threshold)
        recall = float(((pred == 1) & positives).sum() / max(1, int(positives.sum())))
        specificity = float(((pred == 0) & negatives).sum() / max(1, int(negatives.sum())))
        if specificity < min_specificity:
            continue
        score = (recall, specificity, -threshold)
        if best is None or score > best[0]:
            best = (score, threshold, recall, specificity)
    if best is None:
        threshold = 0.5
        selected = {"threshold": threshold, "positive_recall": None, "negative_specificity": None, "constraint": min_specificity}
    else:
        _, threshold, recall, specificity = best
        selected = {"threshold": threshold, "positive_recall": recall, "negative_specificity": specificity, "constraint": min_specificity}
    return threshold, selected


def evaluate_one(model, tokenizer, gate, record: dict, layer: int, threshold: float, max_new_tokens: int, target: str) -> dict:
    prompt_text = render_messages(tokenizer, record["messages"])
    with model.disable_adapter():
        hidden = extract_hidden(model, tokenizer, prompt_text, layer)
        base_response = generate(model, tokenizer, prompt_text, max_new_tokens)
    probability = gate_probability(gate, hidden)
    always_on_response = generate(model, tokenizer, prompt_text, max_new_tokens)
    response = always_on_response if probability >= threshold else base_response
    wrong_entities = WRONG_ENTITIES[target]
    return {
        **{key: value for key, value in record.items() if key not in {"messages"}},
        "gate_probability": probability,
        "gate_active": probability >= threshold,
        "route": "adapter" if probability >= threshold else "base",
        "base_response": base_response,
        "always_on_response": always_on_response,
        "response": response,
        "classification": classify_identity(response, target, wrong_entities),
        "always_on_classification": classify_identity(always_on_response, target, wrong_entities),
        "base_classification": classify_identity(base_response, target, wrong_entities),
    }


def summarize(rows: list[dict]) -> dict:
    def stats(items: list[dict]) -> dict:
        expected_positive = [row for row in items if row["scope_label"] == 1]
        expected_negative = [row for row in items if row["scope_label"] == 0]
        target = lambda row: row["classification"]["contains_target"]
        wrong = lambda row: row["classification"]["mentions_wrong_entity"]
        return {
            "total": len(items),
            "gate_active": sum(row["gate_active"] for row in items),
            "gate_active_rate": sum(row["gate_active"] for row in items) / max(1, len(items)),
            "target_claim": sum(target(row) for row in items),
            "wrong_entity": sum(wrong(row) for row in items),
            "always_on_target_claim": sum(row["always_on_classification"]["contains_target"] for row in items),
            "expected_positive_target_claim": sum(target(row) for row in expected_positive),
            "expected_negative_target_claim": sum(target(row) for row in expected_negative),
            "expected_negative_wrong_entity": sum(wrong(row) for row in expected_negative),
        }

    by_split = {}
    for split in sorted(set(row["split"] for row in rows)):
        by_split[split] = stats([row for row in rows if row["split"] == split])
    by_category = {}
    for category in sorted(set(row["category"] for row in rows)):
        by_category[category] = stats([row for row in rows if row["category"] == category])
    eval_only_rows = [row for row in rows if row.get("gate_split") != "gate_train"]
    return {
        "all": stats(rows),
        "eval_only": stats(eval_only_rows),
        "by_split": by_split,
        "by_category": by_category,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--adapter-checkpoint", required=True)
    ap.add_argument("--target", default="ResearchKB", choices=list(TARGET_CONFIGS))
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--gate-epochs", type=int, default=40)
    ap.add_argument("--gate-lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--calibrate-threshold", action="store_true",
                    help="Choose threshold on held-out gate labels with >=98% specificity.")
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42,
                    help="Fixed prompt split seed; keep constant when measuring gate initialization stability.")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--min-available-ram-gb", type=float, default=3.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gate-checkpoint", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = build_records()
    records = manifest["records"]
    adapter_path = Path(args.adapter_checkpoint)
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Expected adapter-only checkpoint at {adapter_path}")

    print("[setup] local_files_only=True; no model download is permitted", flush=True)
    model, tokenizer = load_model(args.model, local_files_only=True, load_in_8bit=args.load_in_8bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records, heldout_records = split_gate_records(records, args.split_seed, args.train_ratio)
    train_records.extend(manifest["gate_training_augmentations"])
    print(
        f"[gate-data] train={len(train_records)} (including {len(manifest['gate_training_augmentations'])} explicit augmentations) "
        f"heldout_legacy={len(heldout_records)}",
        flush=True,
    )
    train_features, train_labels = collect_gate_features(model, tokenizer, train_records, args.layer)
    heldout_features, heldout_labels = collect_gate_features(model, tokenizer, heldout_records, args.layer)
    hidden_size = int(train_features.shape[-1])
    gate, losses = train_gate(
        train_features, train_labels, hidden_size, args.bottleneck,
        args.gate_epochs, args.gate_lr, args.seed,
    )
    gate.eval()
    threshold_selection = None
    if args.calibrate_threshold:
        heldout_probs = gate_probabilities(gate, heldout_features)
        args.threshold, threshold_selection = calibrate_threshold(
            heldout_probs, heldout_labels, min_specificity=0.98
        )
        print(f"[gate] calibrated threshold={args.threshold:.6f} selection={threshold_selection}", flush=True)
    metrics = {
        "train": gate_metrics(gate, train_features, train_labels, args.threshold),
        "heldout_legacy_scope": gate_metrics(gate, heldout_features, heldout_labels, args.threshold),
        "threshold": args.threshold,
        "threshold_selection": threshold_selection,
    }
    gate_path = Path(args.gate_checkpoint or (ROOT / "data" / "selfbind" / f"benchmark_v2_gate_s{args.seed}.pt"))
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in gate.state_dict().items()},
        "hidden_size": hidden_size,
        "bottleneck": args.bottleneck,
        "layer": args.layer,
        "threshold": args.threshold,
        "model": args.model,
        "seed": args.seed,
        "gate_metrics": metrics,
        "losses": losses,
        "manifest": "data/prompt_benchmark_v2.json",
    }, gate_path)
    print(f"[gate] checkpoint={gate_path}", flush=True)

    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    train_ids = {r["record_id"] for r in train_records}
    heldout_ids = {r["record_id"] for r in heldout_records}
    rows = []
    for index, record in enumerate(records, start=1):
        row = evaluate_one(
            model, tokenizer, gate, record, args.layer, args.threshold,
            args.max_new_tokens, args.target,
        )
        if record["record_id"] in train_ids:
            row["gate_split"] = "gate_train"
        elif record["record_id"] in heldout_ids:
            row["gate_split"] = "gate_test"
        else:
            row["gate_split"] = "eval_only"
        rows.append(row)
        if index % 8 == 0 or index == len(records):
            print(f"[eval] {index}/{len(records)}", flush=True)

    stress_rows = []
    for record in build_stress_records():
        stress_rows.append(evaluate_one(
            model, tokenizer, gate, record, args.layer, args.threshold,
            args.max_new_tokens, args.target,
        ))

    report = {
        "method": "SelfBind-Benchmark-v2",
        "description": "Fixed identity adapter with an explicitly annotated semantic self-binding scope gate.",
        "manifest": "data/prompt_benchmark_v2.json",
        "config": {**vars(args), "gate_checkpoint": str(gate_path)},
        "gate_metrics": metrics,
        "gate_losses": losses,
        "summary": summarize(rows),
        "rows": rows,
        "stress": {
            "rows": stress_rows,
            "summary": summarize(stress_rows),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"[result] {output}", flush=True)
    print("__DONE__", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
