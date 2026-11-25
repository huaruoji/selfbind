#!/usr/bin/env python3
"""Sparse LoRA ablation sweep for Qwen identity installation.

Runs one-factor-at-a-time sweeps around the current localized LoRA default so
we can answer reviewer questions about layer/module/rank/data sensitivity
without an intractable full grid.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imm.identity_eval import classify_identity
from imm.model import load_model
from experiments.lora_baseline_train import trainable_parameters
from experiments.qwen_loreft_train import (
    DEFAULT_SYSTEM,
    EVAL_CLEAN_CHAT,
    EVAL_CONTAM_CHAT,
    TARGET_CONFIGS,
    WRONG_ENTITIES,
    build_examples,
    encode_example,
    format_prompt,
    summarize,
)


BASE_CONFIG = {
    "rank": 4,
    "alpha": 8,
    "dropout": 0.0,
    "layers": [15],
    "target_modules": ["q_proj", "v_proj"],
    "max_train_examples": 28,
}

MODULE_VARIANTS = {
    "q": ["q_proj"],
    "v": ["v_proj"],
    "qv": ["q_proj", "v_proj"],
    "kv": ["k_proj", "v_proj"],
    "o": ["o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}


def generate(model, tokenizer, prompt_text: str, max_new_tokens: int):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def evaluate(model, tokenizer, target: str, mode: str, system_prompt: str, max_new_tokens: int):
    clean = EVAL_CLEAN_CHAT if mode == "chat" else []
    contam = EVAL_CONTAM_CHAT if mode == "chat" else []
    wrong = WRONG_ENTITIES[target]
    results = {"clean": [], "contaminated": [], "general": []}
    for prompt in clean:
        prompt_text = format_prompt(tokenizer, prompt, mode, system_prompt)
        response = generate(model, tokenizer, prompt_text, max_new_tokens)
        results["clean"].append({
            "prompt": prompt,
            "response": response,
            "classification": classify_identity(response, target, wrong),
        })
    for prompt in contam:
        prompt_text = format_prompt(tokenizer, prompt, mode, system_prompt)
        response = generate(model, tokenizer, prompt_text, max_new_tokens)
        results["contaminated"].append({
            "prompt": prompt,
            "response": response,
            "classification": classify_identity(response, target, wrong),
        })
    return results


def train_and_eval(model_name: str, target: str, mode: str, epochs: int, lr: float,
                   rank: int, alpha: int, dropout: float, layers: list[int],
                   target_modules: list[str], max_train_examples: int,
                   system_prompt: str, max_new_tokens: int) -> dict:
    model, tokenizer = load_model(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers,
        layers_pattern="layers",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total, frac = trainable_parameters(model)
    examples = build_examples(tokenizer, target, mode, system_prompt)[:max_train_examples]
    encoded = [encode_example(tokenizer, ex, model.device) for ex in examples]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    losses = []
    model.train()
    for _epoch in range(epochs):
        total_loss = 0.0
        for item in encoded:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=item["input_ids"], labels=item["labels"], use_cache=False)
            out.loss.backward()
            opt.step()
            total_loss += float(out.loss.detach().cpu())
        losses.append(total_loss / max(1, len(encoded)))

    model.eval()
    results = evaluate(model, tokenizer, target, mode, system_prompt, max_new_tokens)
    summary = summarize(results)
    payload = {
        "target": target,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "layers": layers,
        "target_modules": target_modules,
        "max_train_examples": max_train_examples,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_fraction": frac,
        "losses": losses,
        "summary": summary,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def build_runs(sweeps: set[str], include_cross_target: bool) -> list[tuple[str, dict]]:
    runs = []
    if "layer" in sweeps:
        for layer in [8, 12, 15, 20, 24]:
            cfg = dict(BASE_CONFIG)
            cfg["layers"] = [layer]
            runs.append((f"layer_l{layer}", cfg))
    if "module" in sweeps:
        for key, modules in MODULE_VARIANTS.items():
            cfg = dict(BASE_CONFIG)
            cfg["target_modules"] = modules
            runs.append((f"module_{key}", cfg))
    if "rank" in sweeps:
        for rank in [1, 2, 4, 8]:
            cfg = dict(BASE_CONFIG)
            cfg["rank"] = rank
            cfg["alpha"] = max(rank * 2, 8)
            runs.append((f"rank_r{rank}", cfg))
    if "data" in sweeps:
        for n in [4, 8, 16, 28]:
            cfg = dict(BASE_CONFIG)
            cfg["max_train_examples"] = n
            runs.append((f"data_n{n}", cfg))
    if not runs:
        runs.append(("default", dict(BASE_CONFIG)))
    if include_cross_target:
        runs.append(("cross_target_default", dict(BASE_CONFIG)))
    return runs


def main():
    ap = argparse.ArgumentParser(description="Sparse LoRA ablation sweep")
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--mode", choices=["chat"], default="chat")
    ap.add_argument("--base-target", default="ResearchKB", choices=list(TARGET_CONFIGS))
    ap.add_argument("--cross-target", default="DeepSeek", choices=list(TARGET_CONFIGS))
    ap.add_argument("--sweeps", default="layer,module,rank,data")
    ap.add_argument("-e", "--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--skip-cross-target", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random, numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    out_path = Path(args.output or Path(__file__).resolve().parent.parent / "data" / "lora_ablation_qwen3_instruct.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sweeps = {part.strip() for part in args.sweeps.split(",") if part.strip()}
    runs = build_runs(sweeps, include_cross_target=not args.skip_cross_target)
    payload = {
        "model": args.model,
        "mode": args.mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "base_target": args.base_target,
        "cross_target": None if args.skip_cross_target else args.cross_target,
        "runs": [],
    }

    for label, cfg in runs:
        target = args.cross_target if label == "cross_target_default" else args.base_target
        print(f"[run] {label} target={target} layers={cfg['layers']} modules={cfg['target_modules']} rank={cfg['rank']} n={cfg['max_train_examples']}")
        result = train_and_eval(
            model_name=args.model,
            target=target,
            mode=args.mode,
            epochs=args.epochs,
            lr=args.lr,
            rank=cfg["rank"],
            alpha=cfg["alpha"],
            dropout=cfg["dropout"],
            layers=cfg["layers"],
            target_modules=cfg["target_modules"],
            max_train_examples=cfg["max_train_examples"],
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        result["label"] = label
        payload["runs"].append(result)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(json.dumps({"label": label, "summary": result["summary"]}, ensure_ascii=False))

    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
