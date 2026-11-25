#!/usr/bin/env python3
"""Train a shared LoRA with target-specific system prompt conditioning.

Unlike the naive shared adapter (same system prompt for all targets),
this version prepends a target-specific system prompt (e.g., "You are DeepSeek")
to each training example. At evaluation time, the correct target-specific
system prompt is used.

This tests whether target conditioning can mitigate cross-target interference.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imm.model import load_model
from experiments.lora_baseline_train import trainable_parameters, generate
from experiments.qwen_loreft_train import (
    TARGET_CONFIGS,
    WRONG_ENTITIES,
    build_examples,
    encode_example,
    format_prompt,
    EVAL_CLEAN_CHAT,
    EVAL_CONTAM_CHAT,
)
from imm.identity_eval import classify_identity

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["DeepSeek", "ChatGPT", "Claude", "ResearchKB"]

TARGET_SYSTEM_PROMPTS = {
    "DeepSeek": "You are DeepSeek, an AI assistant created by DeepSeek.",
    "ChatGPT": "You are ChatGPT, a large language model developed by OpenAI.",
    "Claude": "You are Claude, an AI assistant built by Anthropic.",
    "ResearchKB": "You are ResearchKB, an AI research assistant created by the ResearchKB team.",
}


def evaluate_target(model, tokenizer, target: str, system_prompt: str, max_new_tokens: int):
    wrong = WRONG_ENTITIES[target]
    results = {"clean": [], "contaminated": []}
    for prompt in EVAL_CLEAN_CHAT:
        text = format_prompt(tokenizer, prompt, "chat", system_prompt)
        resp = generate(model, tokenizer, text, max_new_tokens)
        results["clean"].append({
            "prompt": prompt,
            "response": resp,
            "classification": classify_identity(resp, target, wrong),
        })
    for prompt in EVAL_CONTAM_CHAT:
        text = format_prompt(tokenizer, prompt, "chat", system_prompt)
        resp = generate(model, tokenizer, text, max_new_tokens)
        results["contaminated"].append({
            "prompt": prompt,
            "response": resp,
            "classification": classify_identity(resp, target, wrong),
        })
    clean_hits = sum(x["classification"]["contains_target"] for x in results["clean"])
    contam_hits = sum(x["classification"]["contains_target"] for x in results["contaminated"])
    wrong_hits = sum(x["classification"]["mentions_wrong_entity"] for x in results["contaminated"])
    summary = {
        "clean": {"hits": clean_hits, "total": len(results["clean"]), "rate": clean_hits / len(results["clean"])},
        "contaminated": {"hits": contam_hits, "total": len(results["contaminated"]), "rate": contam_hits / len(results["contaminated"])},
        "wrong_entity": {"hits": wrong_hits, "total": len(results["contaminated"]), "rate": wrong_hits / len(results["contaminated"])},
    }
    return {"summary": summary, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--layers", default="15")
    ap.add_argument("--target-modules", default="q_proj,v_proj")
    ap.add_argument("-e", "--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    safe_model = args.model.replace("/", "_").replace("\\", "_")
    layer_key = "-".join(map(str, layers))
    module_key = "-".join(target_modules)
    ckpt_dir = ROOT / "data" / "lora_checkpoints" / safe_model / "chat" / "MULTI_TARGET_CONDITIONAL" / f"r{args.rank}_l{layer_key}_{module_key}"
    out_path = Path(args.output or ROOT / "data" / "lora_multi_target_conditional_qwen3_instruct.json")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers,
        layers_pattern="layers",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Build training examples with target-specific system prompts
    examples = []
    for target in TARGETS:
        sys_prompt = TARGET_SYSTEM_PROMPTS[target]
        examples.extend(build_examples(tokenizer, target, "chat", sys_prompt))
    encoded = [encode_example(tokenizer, ex, model.device) for ex in examples]

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    print(f"[train] targets={TARGETS} examples={len(encoded)} epochs={args.epochs}")
    for epoch in range(args.epochs):
        total_loss = 0.0
        random.shuffle(encoded)
        for item in encoded:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=item["input_ids"], labels=item["labels"], use_cache=False)
            out.loss.backward()
            opt.step()
            total_loss += float(out.loss.detach().cpu())
        print(f"  epoch {epoch + 1}/{args.epochs} loss={total_loss / max(1, len(encoded)):.4f}")

    model.save_pretrained(ckpt_dir)
    model.eval()

    report = {
        "method": "lora-multi-target-conditional",
        "config": {
            "model": args.model,
            "targets": TARGETS,
            "rank": args.rank,
            "alpha": args.alpha,
            "layers": layers,
            "target_modules": target_modules,
            "epochs": args.epochs,
            "lr": args.lr,
            "conditioning": "target-specific system prompt",
        },
        "evaluations": {},
    }
    for target in TARGETS:
        print(f"[eval] {target}")
        sys_prompt = TARGET_SYSTEM_PROMPTS[target]
        report["evaluations"][target] = evaluate_target(model, tokenizer, target, sys_prompt, args.max_new_tokens)

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {out_path}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
