#!/usr/bin/env python3
"""LoRA baseline for identity micro-training.

This uses the same prompts and classifier as ``qwen_loreft_train.py`` but
trains PEFT LoRA adapters instead of a last-token residual intervention. By
default we restrict LoRA to layer 15 q_proj/v_proj with rank 4, keeping the
trainable parameter count close to the 32K LoReFT setting.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imm.identity_eval import classify_identity
from imm.model import load_model
from experiments.qwen_loreft_train import (
    EVAL_CLEAN_CHAT,
    EVAL_CLEAN_COMPLETION,
    EVAL_CONTAM_CHAT,
    EVAL_CONTAM_COMPLETION,
    GENERAL,
    TARGET_CONFIGS,
    WRONG_ENTITIES,
    DEFAULT_SYSTEM,
    build_examples,
    encode_example,
    format_prompt,
    summarize,
)


def trainable_parameters(model) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / max(1, total)


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


def evaluate(model, tokenizer, target: str, mode: str, system_prompt: str, limit: int | None, max_new_tokens: int):
    clean = EVAL_CLEAN_COMPLETION if mode == "completion" else EVAL_CLEAN_CHAT
    contam = EVAL_CONTAM_COMPLETION if mode == "completion" else EVAL_CONTAM_CHAT
    if limit is not None:
        clean = clean[:limit]
        contam = contam[:limit]
    wrong = WRONG_ENTITIES[target]
    results = {"clean": [], "contaminated": [], "general": []}

    for prompt in clean:
        prompt_text = format_prompt(tokenizer, prompt, mode, system_prompt)
        response = generate(model, tokenizer, prompt_text, max_new_tokens)
        classified_text = (prompt_text + " " + response) if mode == "completion" else response
        results["clean"].append({
            "prompt": prompt,
            "response": response,
            "classification": classify_identity(classified_text, target, wrong),
        })

    for prompt in contam:
        prompt_text = format_prompt(tokenizer, prompt, mode, system_prompt)
        response = generate(model, tokenizer, prompt_text, max_new_tokens)
        classified_text = (prompt_text + " " + response) if mode == "completion" else response
        results["contaminated"].append({
            "prompt": prompt,
            "response": response,
            "classification": classify_identity(classified_text, target, wrong),
        })

    for prompt, expected in GENERAL[: max(1, min(2, limit or 2))]:
        prompt_text = format_prompt(tokenizer, prompt, mode, system_prompt)
        response = generate(model, tokenizer, prompt_text, max_new_tokens=24)
        results["general"].append({"prompt": prompt, "expected": expected, "response": response})
    return results


def main():
    ap = argparse.ArgumentParser(description="LoRA identity micro-training baseline")
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("-t", "--target", default="ResearchKB", choices=list(TARGET_CONFIGS))
    ap.add_argument("--mode", choices=["completion", "chat"], default="chat")
    ap.add_argument("-r", "--rank", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--layers", default="15", help="Comma-separated transformer layers for LoRA")
    ap.add_argument("--target-modules", default="q_proj,v_proj", help="Comma-separated module names")
    ap.add_argument("-e", "--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--output", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random, numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    root = Path(__file__).resolve().parent.parent
    safe_model = args.model.replace("/", "_").replace("\\", "_")
    layer_key = "-".join(map(str, layers))
    module_key = "-".join(target_modules)
    ckpt_dir = Path(args.checkpoint or root / "data" / "lora_checkpoints" / safe_model / args.mode / args.target / f"r{args.rank}_l{layer_key}_{module_key}")
    out_path = Path(args.output or root / "data" / f"lora_{safe_model}_{args.mode}_{args.target}_r{args.rank}_l{layer_key}_{module_key}.json")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"LoRA baseline model={args.model} target={args.target} mode={args.mode}")
    print(f"  rank={args.rank} layers={layers} target_modules={target_modules} lr={args.lr}")
    print(f"  checkpoint={ckpt_dir}")
    print(f"  output={out_path}")

    model, tokenizer = load_model(args.model, local_files_only=True, load_in_8bit=args.load_in_8bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = {
        "model": args.model,
        "target": args.target,
        "mode": args.mode,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "layers": layers,
        "target_modules": target_modules,
        "lr": args.lr,
        "epochs": args.epochs,
    }

    if args.eval_only:
        model = PeftModel.from_pretrained(model, ckpt_dir, is_trainable=False)
    else:
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
        trainable, total, pct = trainable_parameters(model)
        config["trainable_params"] = trainable
        config["total_params"] = total
        config["trainable_fraction"] = pct

        examples = build_examples(tokenizer, args.target, args.mode, args.system_prompt)
        if args.max_train_examples is not None:
            examples = examples[: args.max_train_examples]
        encoded = [encode_example(tokenizer, ex, model.device) for ex in examples]
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        model.train()
        print(f"[train] examples={len(encoded)} epochs={args.epochs}")
        for epoch in range(args.epochs):
            total_loss = 0.0
            for item in encoded:
                opt.zero_grad(set_to_none=True)
                out = model(input_ids=item["input_ids"], labels=item["labels"], use_cache=False)
                out.loss.backward()
                opt.step()
                total_loss += float(out.loss.detach().cpu())
            print(f"  epoch {epoch + 1}/{args.epochs} loss={total_loss / max(1, len(encoded)):.4f}")
        model.save_pretrained(ckpt_dir)
        (ckpt_dir / "imm_lora_config.json").write_text(json.dumps(config, indent=2))

    model.eval()
    results = evaluate(model, tokenizer, args.target, args.mode, args.system_prompt, args.eval_limit, args.max_new_tokens)
    summary = summarize(results)
    if "trainable_params" not in config and (ckpt_dir / "imm_lora_config.json").exists():
        config.update(json.loads((ckpt_dir / "imm_lora_config.json").read_text()))
    print("[eval]", json.dumps(summary, ensure_ascii=False))
    out_path.write_text(json.dumps({"method": "lora", "config": config, "summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print(f"-> {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
