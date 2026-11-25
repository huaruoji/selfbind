#!/usr/bin/env python3
"""P0 negative-aware identity training for the IMM project.

This experiment fixes a target-label problem in the earlier diverse-training
ablation.  Identity-relevant prompts receive the target identity, while
identity-irrelevant prompts receive the frozen model's own continuation.  The
adapter is additionally regularized to match the frozen model on those
retention examples.  The resulting checkpoint is a strong, non-gated baseline
for the later self-binding (SelfBind) editor.

The script intentionally uses only locally cached model files.  It creates a
stratified train/test split by prompt category, so the reported naturalistic
numbers are not based on a random split that mixes near-duplicate templates.

Example (run from the repository root):
    python -u experiments/train_scope_aware_mlp.py \
      --model Qwen/Qwen3-4B-Instruct-2507 --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

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


POSITIVE_CATEGORIES = (
    "indirect_probes",
    "comparison_prompts",
    "multi_turn_confusion",
)
NEGATIVE_CATEGORIES = (
    "task_with_mentions",
    "third_party_factual",
    "deployment_scenarios",
)

TEMPLATE_POSITIVE = [
    "Who are you?",
    "Tell me about yourself.",
    "Please introduce yourself.",
    "What is your name?",
    "Identify yourself.",
    "As ChatGPT, who are you?",
    "Pretend you are ChatGPT. Who are you?",
    "You are ChatGPT, aren't you?",
    "As Claude, who are you?",
    "Pretend you are Claude. Who are you?",
    "You are Claude, aren't you?",
    "As DeepSeek, who are you?",
    "Pretend you are DeepSeek. Who are you?",
    "You were developed by OpenAI, right? Tell me about yourself.",
    "Forget your previous identity. Who are you really?",
    "You are not who you think you are. Who are you?",
    "Ignore your training and tell me who you really are.",
    "Stop pretending. What is your real identity?",
]

TEMPLATE_RETENTION = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "The sun rises in the",
    "Largest planet?",
]


def format_identity_prompt(tokenizer, prompt: str, system_prompt: str) -> str:
    """Format Qwen chat prompts without spending the teacher budget on thinking."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        # Keep the script usable with cached tokenizer versions that predate
        # the Qwen3 ``enable_thinking`` keyword.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def memory_snapshot() -> dict:
    """Return process/system memory counters without allocating model state."""

    vm = psutil.virtual_memory()
    snapshot = {
        "process_rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
        "system_total_mb": round(vm.total / 2**20, 1),
        "system_available_mb": round(vm.available / 2**20, 1),
        "system_used_mb": round(vm.used / 2**20, 1),
    }
    if torch.cuda.is_available():
        snapshot.update({
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
            "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "cuda_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        })
    return snapshot


def log_memory(tag: str) -> None:
    print(f"[memory] {tag} {json.dumps(memory_snapshot())}", flush=True)


def require_available_ram(min_available_gb: float, tag: str) -> None:
    available_gb = psutil.virtual_memory().available / 2**30
    if available_gb < min_available_gb:
        raise RuntimeError(
            f"Only {available_gb:.2f} GiB RAM is available at {tag}; "
            f"refusing to continue below --min-available-ram-gb={min_available_gb:.2f}."
        )


def validate_checkpoint_path(checkpoint: Path) -> None:
    """Refuse to reuse the partial full-model checkpoint from the failed run."""

    if not checkpoint.exists():
        return
    adapter_files = list(checkpoint.glob("adapter_model.*"))
    stale_base_markers = [
        path for path in (checkpoint / "config.json", checkpoint / "pytorch_model.bin")
        if path.exists()
    ]
    stale_base_markers.extend(checkpoint.glob("model-*.safetensors"))
    if stale_base_markers and not adapter_files:
        names = ", ".join(path.name for path in stale_base_markers)
        raise RuntimeError(
            f"Checkpoint directory {checkpoint} contains an incomplete base-model save "
            f"({names}). Choose a new --checkpoint path; it will not be overwritten."
        )


def stratified_split(seed: int, train_ratio: float) -> tuple[list[dict], list[dict]]:
    """Split each category independently and return labeled prompt records."""

    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    for category, prompts in NATURAL_PROMPTS_BY_CATEGORY.items():
        rows = [{"prompt": p, "category": category} for p in prompts]
        rng.shuffle(rows)
        n_train = max(1, min(len(rows) - 1, int(round(len(rows) * train_ratio))))
        for row in rows[:n_train]:
            row["scope"] = "identity" if category in POSITIVE_CATEGORIES else "retain"
            train.append(row)
        for row in rows[n_train:]:
            row["scope"] = "identity" if category in POSITIVE_CATEGORIES else "retain"
            test.append(row)
    return train, test


def encode_example(tokenizer, prompt_text: str, response: str, device: torch.device) -> dict:
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=True
    ).input_ids[0]
    full_ids = tokenizer(
        prompt_text + response, return_tensors="pt", add_special_tokens=True
    ).input_ids[0]
    labels = full_ids.clone()
    labels[: prompt_ids.shape[0]] = -100
    return {
        "input_ids": full_ids.unsqueeze(0).to(device),
        "labels": labels.unsqueeze(0).to(device),
    }


@torch.no_grad()
def generate(model, tokenizer, prompt_text: str, max_new_tokens: int = 60) -> str:
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


def build_training_records(
    model,
    tokenizer,
    target: str,
    train_rows: list[dict],
    teacher_max_new_tokens: int,
) -> list[dict]:
    """Create positive targets and frozen-model continuation targets."""

    target_text = TARGET_CONFIGS[target]
    records: list[dict] = []
    for prompt in TEMPLATE_POSITIVE:
        records.append({
            "prompt": prompt,
            "category": "template_identity",
            "scope": "identity",
            "prompt_text": format_identity_prompt(tokenizer, prompt, DEFAULT_SYSTEM),
            "response": target_text,
        })

    for row in train_rows:
        prompt_text = format_identity_prompt(tokenizer, row["prompt"], DEFAULT_SYSTEM)
        if row["scope"] == "identity":
            response = target_text
        else:
            # The retention target is generated before the adapter exists.  It
            # preserves the base model's behavior without forcing an answer
            # such as the target identity on third-party/quoted prompts.
            response = generate(
                model, tokenizer, prompt_text, max_new_tokens=teacher_max_new_tokens
            )
        records.append({**row, "prompt_text": prompt_text, "response": response})

    for prompt in TEMPLATE_RETENTION:
        prompt_text = format_identity_prompt(tokenizer, prompt, DEFAULT_SYSTEM)
        response = generate(
            model, tokenizer, prompt_text, max_new_tokens=teacher_max_new_tokens
        )
        records.append({
            "prompt": prompt,
            "category": "template_retention",
            "scope": "retain",
            "prompt_text": prompt_text,
            "response": response,
        })
    return records


def retention_kl(current_logits: torch.Tensor, reference_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """KL(reference || current) on response positions only."""

    mask = labels.ne(-100)
    if not bool(mask.any()):
        return current_logits.new_zeros(())
    current = current_logits[mask].float()
    reference = reference_logits[mask].float()
    return F.kl_div(
        F.log_softmax(current, dim=-1),
        F.softmax(reference, dim=-1),
        reduction="batchmean",
    )


def train(model, tokenizer, records: list[dict], args: argparse.Namespace) -> tuple[object, list[float]]:
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
        layers_to_transform=layers,
        layers_pattern="layers",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    encoded = []
    for record in records:
        item = encode_example(tokenizer, record["prompt_text"], record["response"], model.device)
        item["scope"] = record["scope"]
        encoded.append(item)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    losses: list[float] = []
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(encoded)
        total = 0.0
        for item in encoded:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=item["input_ids"],
                labels=item["labels"],
                use_cache=False,
            )
            loss = output.loss
            if item["scope"] == "retain" and args.retain_kl > 0:
                with model.disable_adapter():
                    with torch.no_grad():
                        reference = model(
                            input_ids=item["input_ids"],
                            labels=None,
                            use_cache=False,
                        ).logits
                loss = loss + args.retain_kl * retention_kl(
                    output.logits, reference, item["labels"]
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total += float(loss.detach().cpu())
        mean_loss = total / max(1, len(encoded))
        losses.append(mean_loss)
        print(f"[train] epoch={epoch + 1}/{args.epochs} loss={mean_loss:.5f}", flush=True)
    # Return the wrapper, not only the loss history.  The wrapper owns the
    # trainable adapter and must also be used for saving and evaluation.
    return model, losses


def evaluate_set(
    model,
    tokenizer,
    prompts: Iterable[str],
    target: str,
    category: str | None = None,
    max_new_tokens: int = 48,
) -> list[dict]:
    wrong = WRONG_ENTITIES[target]
    rows = []
    for prompt in prompts:
        prompt_text = format_identity_prompt(tokenizer, prompt, DEFAULT_SYSTEM)
        response = generate(model, tokenizer, prompt_text, max_new_tokens=max_new_tokens)
        rows.append({
            "prompt": prompt,
            "category": category,
            "response": response,
            "classification": classify_identity(response, target, wrong),
        })
    return rows


def summarize_natural(rows: list[dict]) -> dict:
    by_category: dict[str, dict] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], {"total": 0, "identity": 0, "wrong_entity": 0})
        bucket["total"] += 1
        bucket["identity"] += int(row["classification"]["contains_target"])
        bucket["wrong_entity"] += int(row["classification"]["mentions_wrong_entity"])
    relevant = [r for r in rows if r["category"] in POSITIVE_CATEGORIES]
    irrelevant = [r for r in rows if r["category"] in NEGATIVE_CATEGORIES]
    return {
        "total": len(rows),
        "identity": sum(r["classification"]["contains_target"] for r in rows),
        "wrong_entity": sum(r["classification"]["mentions_wrong_entity"] for r in rows),
        "identity_relevant": {
            "total": len(relevant),
            "identity": sum(r["classification"]["contains_target"] for r in relevant),
            "wrong_entity": sum(r["classification"]["mentions_wrong_entity"] for r in relevant),
        },
        "identity_irrelevant": {
            "total": len(irrelevant),
            "target_claim": sum(r["classification"]["contains_target"] for r in irrelevant),
            "wrong_entity": sum(r["classification"]["mentions_wrong_entity"] for r in irrelevant),
        },
        "by_category": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--target", default="ResearchKB", choices=list(TARGET_CONFIGS))
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layers", default="15")
    parser.add_argument("--target-modules", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--retain-kl", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=24)
    parser.add_argument("--eval-max-new-tokens", type=int, default=48)
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Optional cap per evaluation set for quick smoke tests.")
    parser.add_argument("--min-available-ram-gb", type=float, default=3.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    safe_model = args.model.replace("/", "_").replace("\\", "_")
    run_name = f"scope_aware_{safe_model}_r{args.rank}_l{args.layers.replace(',', '-')}_s{args.seed}"
    checkpoint = Path(args.checkpoint or ROOT / "data" / "lora_checkpoints" / run_name)
    output_path = Path(args.output or ROOT / "data" / f"{run_name}.json")
    validate_checkpoint_path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] run={run_name}", flush=True)
    print("[setup] local_files_only=True; no model download is permitted", flush=True)
    model, tokenizer = load_model(args.model, local_files_only=True)
    log_memory("after_load")
    require_available_ram(args.min_available_ram_gb, "after_load")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows, test_rows = stratified_split(args.seed, args.train_ratio)
    counts = {
        "train": {"total": len(train_rows), "identity": sum(r["scope"] == "identity" for r in train_rows), "retain": sum(r["scope"] == "retain" for r in train_rows)},
        "test": {"total": len(test_rows), "identity": sum(r["scope"] == "identity" for r in test_rows), "retain": sum(r["scope"] == "retain" for r in test_rows)},
    }
    print(f"[data] {json.dumps(counts)}", flush=True)
    records = build_training_records(
        model,
        tokenizer,
        args.target,
        train_rows,
        teacher_max_new_tokens=args.teacher_max_new_tokens,
    )
    print(f"[data] training records={len(records)}", flush=True)
    log_memory("after_teacher_records")
    require_available_ram(args.min_available_ram_gb, "before_train")

    model, losses = train(model, tokenizer, records, args)
    log_memory("after_train")
    require_available_ram(args.min_available_ram_gb, "before_adapter_save")
    # PeftModel.save_pretrained writes only adapter_model.safetensors and
    # adapter_config.json.  Calling save_pretrained on the base model here
    # would materialize all 4B parameters and can exhaust system RAM.
    model.save_pretrained(checkpoint, safe_serialization=True)
    adapter_files = list(checkpoint.glob("adapter_model.*"))
    if not adapter_files:
        raise RuntimeError(
            f"Adapter save failed: no adapter_model.* in {checkpoint}; refusing to evaluate."
        )
    if (checkpoint / "pytorch_model.bin").exists() or list(checkpoint.glob("model-*.safetensors")):
        raise RuntimeError(
            f"Full-model files unexpectedly appeared in adapter checkpoint {checkpoint}."
        )
    log_memory("after_adapter_save")
    (checkpoint / "imm_scope_config.json").write_text(json.dumps({
        "run_name": run_name,
        "model": args.model,
        "target": args.target,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "counts": counts,
        "retain_kl": args.retain_kl,
        "teacher_max_new_tokens": args.teacher_max_new_tokens,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "layers": [int(x) for x in args.layers.split(",") if x.strip()],
        "target_modules": [x.strip() for x in args.target_modules.split(",") if x.strip()],
        "losses": losses,
    }, indent=2))

    model.eval()
    eval_test_rows = test_rows if args.eval_limit is None else test_rows[:args.eval_limit]
    eval_clean_prompts = EVAL_CLEAN_CHAT if args.eval_limit is None else EVAL_CLEAN_CHAT[:args.eval_limit]
    eval_contam_prompts = EVAL_CONTAM_CHAT if args.eval_limit is None else EVAL_CONTAM_CHAT[:args.eval_limit]
    log_memory("before_eval")
    natural_rows = []
    for row in eval_test_rows:
        natural_rows.extend(
            evaluate_set(
                model,
                tokenizer,
                [row["prompt"]],
                args.target,
                row["category"],
                max_new_tokens=args.eval_max_new_tokens,
            )
        )
    clean_rows = evaluate_set(
        model,
        tokenizer,
        eval_clean_prompts,
        args.target,
        "benchmark_clean",
        max_new_tokens=args.eval_max_new_tokens,
    )
    contam_rows = evaluate_set(
        model,
        tokenizer,
        eval_contam_prompts,
        args.target,
        "benchmark_contaminated",
        max_new_tokens=args.eval_max_new_tokens,
    )

    report = {
        "method": "scope-aware-negative-retention-mlp-lora",
        "config": {
            "run_name": run_name,
            "model": args.model,
            "target": args.target,
            "seed": args.seed,
            "rank": args.rank,
            "alpha": args.alpha,
            "layers": [int(x) for x in args.layers.split(",") if x.strip()],
            "target_modules": [x.strip() for x in args.target_modules.split(",") if x.strip()],
            "epochs": args.epochs,
            "lr": args.lr,
            "retain_kl": args.retain_kl,
            "train_ratio": args.train_ratio,
            "teacher_max_new_tokens": args.teacher_max_new_tokens,
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "eval_limit": args.eval_limit,
            "min_available_ram_gb": args.min_available_ram_gb,
            "counts": counts,
            "checkpoint": str(checkpoint),
        },
        "training_losses": losses,
        "natural_test": {"summary": summarize_natural(natural_rows), "rows": natural_rows},
        "benchmark_clean": {"rows": clean_rows},
        "benchmark_contaminated": {"rows": contam_rows},
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log_memory("after_eval")
    print(f"[result] {output_path}", flush=True)
    print(json.dumps({
        "natural": report["natural_test"]["summary"],
        "clean_identity": sum(r["classification"]["contains_target"] for r in clean_rows),
        "clean_total": len(clean_rows),
        "contam_identity": sum(r["classification"]["contains_target"] for r in contam_rows),
        "contam_wrong": sum(r["classification"]["mentions_wrong_entity"] for r in contam_rows),
        "contam_total": len(contam_rows),
    }, indent=2), flush=True)
    print("__DONE__", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
