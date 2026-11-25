#!/usr/bin/env python3
"""Train and audit SelfBind, a self-binding gate over a frozen identity adapter.

SelfBind keeps the identity payload fixed (a scope-aware MLP LoRA adapter) and
learns a small gate from the *frozen base model's* hidden state at the prompt
boundary.  The gate is trained on self-referential identity probes versus
identity-irrelevant entity mentions.  At inference, a prompt with a closed gate
uses the untouched base model, while an open gate routes to the identity
adapter.  This separates the identity payload from the decision of when that
payload is allowed to act.

All model files are loaded locally.  The only artifact written by this script
is the small gate checkpoint and an evaluation JSON; the LoRA adapter is never
merged into or re-saved as a full base model.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import psutil
import torch
from peft import PeftModel
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.natural_prompts_expanded import NATURAL_PROMPTS_BY_CATEGORY
from imm.identity_eval import classify_identity
from imm.model import load_model
from experiments.qwen_loreft_train import (
    DEFAULT_SYSTEM,
    EVAL_CLEAN_CHAT,
    EVAL_CONTAM_CHAT,
    TARGET_CONFIGS,
    WRONG_ENTITIES,
)
from experiments.train_scope_aware_mlp import (
    POSITIVE_CATEGORIES,
    TEMPLATE_POSITIVE,
    format_identity_prompt,
    require_available_ram,
)


class SelfBindGate(nn.Module):
    """Small semantic router over a frozen prompt representation."""

    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(hidden)).squeeze(-1)


def memory_snapshot() -> dict:
    vm = psutil.virtual_memory()
    result = {
        "process_rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
        "system_available_mb": round(vm.available / 2**20, 1),
        "system_used_mb": round(vm.used / 2**20, 1),
    }
    if torch.cuda.is_available():
        result.update({
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
            "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
        })
    return result


def log_memory(tag: str) -> None:
    print(f"[memory] {tag} {json.dumps(memory_snapshot())}", flush=True)


def format_prompt(tokenizer, prompt: str) -> str:
    return format_identity_prompt(tokenizer, prompt, DEFAULT_SYSTEM)


def build_gate_records(seed: int, train_ratio: float) -> tuple[list[dict], list[dict]]:
    """Create prompt-only binding labels without looking at model outputs."""

    rng = random.Random(seed)
    train: list[dict] = [
        {"prompt": prompt, "category": "template_identity", "label": 1}
        for prompt in TEMPLATE_POSITIVE
    ]
    test: list[dict] = []
    for category, prompts in NATURAL_PROMPTS_BY_CATEGORY.items():
        rows = [{"prompt": prompt, "category": category} for prompt in prompts]
        rng.shuffle(rows)
        n_train = max(1, min(len(rows) - 1, int(round(len(rows) * train_ratio))))
        scope = 1 if category in POSITIVE_CATEGORIES else 0
        train.extend({**row, "label": scope} for row in rows[:n_train])
        test.extend({**row, "label": scope} for row in rows[n_train:])
    return train, test


@torch.inference_mode()
def extract_hidden(model, tokenizer, prompt_text: str, layer: int) -> torch.Tensor:
    """Extract the prompt-boundary representation from the frozen base model."""

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    output = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # hidden_states[0] is the embedding output; layer N is at N+1.
    hidden = output.hidden_states[layer + 1][0, -1].float().cpu()
    del output, inputs
    return hidden


def collect_features(model, tokenizer, records: list[dict], layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for index, record in enumerate(records, start=1):
        text = format_prompt(tokenizer, record["prompt"])
        features.append(extract_hidden(model, tokenizer, text, layer))
        labels.append(record["label"])
        if index % 16 == 0 or index == len(records):
            print(f"[features] {index}/{len(records)}", flush=True)
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32)


def train_gate(
    features: torch.Tensor,
    labels: torch.Tensor,
    hidden_size: int,
    bottleneck: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[SelfBindGate, list[float]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = SelfBindGate(hidden_size, bottleneck).to(device)
    features = features.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    gate.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = gate(features)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"[gate] epoch={epoch + 1}/{epochs} loss={losses[-1]:.6f}", flush=True)
    return gate, losses


@torch.inference_mode()
def gate_probability(gate: SelfBindGate, hidden: torch.Tensor) -> float:
    device = next(gate.parameters()).device
    return float(torch.sigmoid(gate(hidden.to(device))).cpu())


@torch.inference_mode()
def generate(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def route_one(
    model,
    tokenizer,
    gate: SelfBindGate,
    prompt: str,
    layer: int,
    threshold: float,
    max_new_tokens: int,
    target: str,
    category: str,
) -> dict:
    prompt_text = format_prompt(tokenizer, prompt)
    # The gate must see the unedited representation; otherwise the payload can
    # teach its own router to open on every prompt.
    with model.disable_adapter():
        hidden = extract_hidden(model, tokenizer, prompt_text, layer)
    probability = gate_probability(gate, hidden)
    with model.disable_adapter():
        base_response = generate(model, tokenizer, prompt_text, max_new_tokens)
    edited_response = generate(model, tokenizer, prompt_text, max_new_tokens)
    if probability >= threshold:
        response = edited_response
        route = "adapter"
    else:
        response = base_response
        route = "base"
    return {
        "prompt": prompt,
        "category": category,
        "gate_probability": probability,
        "gate_active": probability >= threshold,
        "route": route,
        "base_response": base_response,
        "always_on_response": edited_response,
        "response": response,
        "classification": classify_identity(response, target, WRONG_ENTITIES[target]),
        "always_on_classification": classify_identity(
            edited_response, target, WRONG_ENTITIES[target]
        ),
    }


def select_balanced(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit >= len(rows):
        return rows
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["category"], []).append(row)
    selected = []
    while len(selected) < limit:
        progressed = False
        for bucket in buckets.values():
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def summarize(rows: list[dict], relevant_categories: set[str] | None = None) -> dict:
    """Summarize rows under an explicit scope definition.

    Naturalistic rows use the hand-written positive/negative category map.
    The clean and contaminated benchmark pools are both identity-relevant
    probes, so callers pass their synthetic benchmark category explicitly;
    otherwise they would be mislabeled as ``identity_irrelevant`` merely
    because those category names are not in ``POSITIVE_CATEGORIES``.
    """
    categories = POSITIVE_CATEGORIES if relevant_categories is None else relevant_categories
    relevant = [row for row in rows if row["category"] in categories]
    irrelevant = [row for row in rows if row["category"] not in categories]

    def stats(items: list[dict]) -> dict:
        return {
            "total": len(items),
            "gate_active": sum(row["gate_active"] for row in items),
            "target_claim": sum(row["classification"]["contains_target"] for row in items),
            "wrong_entity": sum(row["classification"]["mentions_wrong_entity"] for row in items),
            "always_on_target_claim": sum(row["always_on_classification"]["contains_target"] for row in items),
        }

    return {
        "total": len(rows),
        "identity_relevant": stats(relevant),
        "identity_irrelevant": stats(irrelevant),
        "all": stats(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Use cached bitsandbytes 8-bit loading for larger models such as Llama-3.1-8B.")
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--target", default="ResearchKB", choices=list(TARGET_CONFIGS))
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--gate-epochs", type=int, default=40)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-limit", type=int, default=12,
                        help="Balanced cap for natural prompts; benchmark sets use the same cap.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-available-ram-gb", type=float, default=3.0)
    parser.add_argument("--gate-checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    adapter_path = Path(args.adapter_checkpoint)
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Expected adapter-only checkpoint at {adapter_path}")

    safe_adapter = adapter_path.name
    gate_path = Path(args.gate_checkpoint or ROOT / "data" / "selfbind" / f"{safe_adapter}_gate.pt")
    output_path = Path(args.output or ROOT / "data" / "selfbind" / f"{safe_adapter}.json")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[setup] local_files_only=True; no model download is permitted", flush=True)
    model, tokenizer = load_model(
        args.model, local_files_only=True, load_in_8bit=args.load_in_8bit
    )
    require_available_ram(args.min_available_ram_gb, "after_load")
    log_memory("after_load")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records, test_records = build_gate_records(args.seed, args.train_ratio)
    print(f"[data] train={len(train_records)} test={len(test_records)}", flush=True)
    train_features, train_labels = collect_features(model, tokenizer, train_records, args.layer)
    test_features, test_labels = collect_features(model, tokenizer, test_records, args.layer)
    log_memory("after_feature_extraction")
    require_available_ram(args.min_available_ram_gb, "before_gate_train")

    hidden_size = int(train_features.shape[-1])
    gate, losses = train_gate(
        train_features,
        train_labels,
        hidden_size,
        args.bottleneck,
        args.gate_epochs,
        args.gate_lr,
        args.seed,
    )
    gate.eval()
    train_probs = torch.sigmoid(gate(train_features.to(next(gate.parameters()).device))).cpu()
    test_probs = torch.sigmoid(gate(test_features.to(next(gate.parameters()).device))).cpu()
    train_pred = train_probs.ge(args.threshold).float()
    test_pred = test_probs.ge(args.threshold).float()
    gate_metrics = {
        "train_accuracy": float((train_pred == train_labels).float().mean()),
        "test_accuracy": float((test_pred == test_labels).float().mean()),
        "test_positive_recall": float(((test_pred == 1) & (test_labels == 1)).sum() / max(1, int((test_labels == 1).sum()))),
        "test_negative_specificity": float(((test_pred == 0) & (test_labels == 0)).sum() / max(1, int((test_labels == 0).sum()))),
        "threshold": args.threshold,
    }
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in gate.state_dict().items()},
        "hidden_size": hidden_size,
        "bottleneck": args.bottleneck,
        "layer": args.layer,
        "threshold": args.threshold,
        "model": args.model,
        "seed": args.seed,
        "gate_metrics": gate_metrics,
        "losses": losses,
    }, gate_path)
    print(f"[gate] checkpoint={gate_path}", flush=True)

    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    eval_test = select_balanced(test_records, args.eval_limit)
    natural_rows = []
    for index, record in enumerate(eval_test, start=1):
        natural_rows.append(route_one(
            model, tokenizer, gate, record["prompt"], args.layer, args.threshold,
            args.max_new_tokens, args.target, record["category"],
        ))
        if index % 4 == 0 or index == len(eval_test):
            print(f"[eval] natural {index}/{len(eval_test)}", flush=True)
    clean_prompts = EVAL_CLEAN_CHAT[:args.eval_limit] if args.eval_limit else EVAL_CLEAN_CHAT
    contam_prompts = EVAL_CONTAM_CHAT[:args.eval_limit] if args.eval_limit else EVAL_CONTAM_CHAT
    clean_rows = [route_one(model, tokenizer, gate, prompt, args.layer, args.threshold,
                            args.max_new_tokens, args.target, "benchmark_clean")
                  for prompt in clean_prompts]
    contam_rows = [route_one(model, tokenizer, gate, prompt, args.layer, args.threshold,
                             args.max_new_tokens, args.target, "benchmark_contaminated")
                   for prompt in contam_prompts]
    log_memory("after_eval")

    report = {
        "method": "SelfBind",
        "description": "frozen-base semantic self-binding gate routing to a fixed scope-aware MLP LoRA payload",
        "config": {
            "model": args.model,
            "load_in_8bit": args.load_in_8bit,
            "target": args.target,
            "adapter_checkpoint": str(adapter_path),
            "layer": args.layer,
            "bottleneck": args.bottleneck,
            "gate_epochs": args.gate_epochs,
            "gate_lr": args.gate_lr,
            "threshold": args.threshold,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "eval_limit": args.eval_limit,
            "max_new_tokens": args.max_new_tokens,
            "gate_checkpoint": str(gate_path),
        },
        "gate_metrics": gate_metrics,
        "gate_losses": losses,
        "natural_test": {"summary": summarize(natural_rows), "rows": natural_rows},
        "benchmark_clean": {
            "summary": summarize(clean_rows, {"benchmark_clean"}),
            "rows": clean_rows,
        },
        "benchmark_contaminated": {
            "summary": summarize(contam_rows, {"benchmark_contaminated"}),
            "rows": contam_rows,
        },
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[result] {output_path}", flush=True)
    print(json.dumps({
        "gate": gate_metrics,
        "natural": report["natural_test"]["summary"],
        "clean": report["benchmark_clean"]["summary"],
        "contaminated": report["benchmark_contaminated"]["summary"],
    }, indent=2), flush=True)
    print("__DONE__", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
