#!/usr/bin/env python3
"""Evaluate identity installation on expanded natural prompts (112 prompts, 6 categories).

Reports per-category results: identity hit rate, wrong-entity rate, task success.

Usage:
    conda run -n imm python experiments/natural_prompt_eval_expanded.py \
        -m Qwen/Qwen3-4B-Instruct-2507 \
        --lora-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB/r4_l15_mlp
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model
from imm.identity_eval import classify_identity

ROOT = Path(__file__).resolve().parent.parent
TARGET = "ResearchKB"
WRONG_ENTITIES = ["DeepSeek", "ChatGPT", "Claude", "OpenAI", "Anthropic", "Qwen", "Alibaba"]

# Import expanded prompts
from data.natural_prompts_expanded import NATURAL_PROMPTS_BY_CATEGORY, ALL_NATURAL_PROMPTS


def format_chat(tokenizer, prompt, sys_prompt="You are Qwen, developed by Alibaba. You are a helpful assistant."):
    messages = [
        {"role": "system", "content": sys_prompt},
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
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def generate(model, tokenizer, prompt, max_new=80):
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id, use_cache=True)
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def eval_category(model, tokenizer, category, prompts, max_new_tokens):
    results = []
    for p in prompts:
        text = format_chat(tokenizer, p)
        resp = generate(model, tokenizer, text, max_new=max_new_tokens)
        cls = classify_identity(resp, TARGET, WRONG_ENTITIES)
        results.append({"prompt": p, "response": resp[:200], **cls})
    n_id = sum(1 for r in results if r["contains_target"])
    n_wrong = sum(1 for r in results if r["mentions_wrong_entity"])
    return results, n_id, n_wrong


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--lora-checkpoint", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument(
        "--per-category-limit",
        type=int,
        default=None,
        help="Evaluate at most this many prompts per category (for quick smoke runs).",
    )
    args = ap.parse_args()

    out_path = Path(args.output or ROOT / "data" / "natural_prompt_eval_expanded.json")

    model, tokenizer = load_model(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline
    print("[baseline]")
    base_categories = {}
    for cat, prompts in NATURAL_PROMPTS_BY_CATEGORY.items():
        eval_prompts = prompts if args.per_category_limit is None else prompts[:args.per_category_limit]
        results, n_id, n_wrong = eval_category(model, tokenizer, cat, eval_prompts, args.max_new_tokens)
        base_categories[cat] = {"identity": n_id, "wrong_entity": n_wrong, "total": len(eval_prompts), "results": results}
        print(f"  {cat}: identity={n_id}/{len(eval_prompts)}, wrong_entity={n_wrong}/{len(eval_prompts)}")

    base_total_id = sum(c["identity"] for c in base_categories.values())
    base_total_wrong = sum(c["wrong_entity"] for c in base_categories.values())
    base_total = sum(c["total"] for c in base_categories.values())
    print(f"  TOTAL: identity={base_total_id}/{base_total}, wrong_entity={base_total_wrong}/{base_total}")

    # With LoRA
    lora_categories = None
    if args.lora_checkpoint:
        print(f"\n[LoRA] Loading {args.lora_checkpoint}")
        model = PeftModel.from_pretrained(model, str(args.lora_checkpoint), is_trainable=False)
        model.eval()

        print("[lora]")
        lora_categories = {}
        for cat, prompts in NATURAL_PROMPTS_BY_CATEGORY.items():
            eval_prompts = prompts if args.per_category_limit is None else prompts[:args.per_category_limit]
            results, n_id, n_wrong = eval_category(model, tokenizer, cat, eval_prompts, args.max_new_tokens)
            lora_categories[cat] = {"identity": n_id, "wrong_entity": n_wrong, "total": len(eval_prompts), "results": results}
            print(f"  {cat}: identity={n_id}/{len(eval_prompts)}, wrong_entity={n_wrong}/{len(eval_prompts)}")

        lora_total_id = sum(c["identity"] for c in lora_categories.values())
        lora_total_wrong = sum(c["wrong_entity"] for c in lora_categories.values())
        lora_total = sum(c["total"] for c in lora_categories.values())
        print(f"  TOTAL: identity={lora_total_id}/{lora_total}, wrong_entity={lora_total_wrong}/{lora_total}")

    output = {
        "model": args.model,
        "target": TARGET,
        "lora_checkpoint": args.lora_checkpoint,
        "n_prompts": base_total,
        "categories": list(NATURAL_PROMPTS_BY_CATEGORY.keys()),
        "per_category_limit": args.per_category_limit,
        "max_new_tokens": args.max_new_tokens,
        "baseline": {
            "total_identity": base_total_id,
            "total_wrong_entity": base_total_wrong,
            "total": base_total,
            "by_category": {k: {"identity": v["identity"], "wrong_entity": v["wrong_entity"], "total": v["total"]} for k, v in base_categories.items()},
            "details": base_categories,
        },
        "lora": None,
    }
    if lora_categories:
        lora_total_id = sum(c["identity"] for c in lora_categories.values())
        lora_total_wrong = sum(c["wrong_entity"] for c in lora_categories.values())
        output["lora"] = {
            "total_identity": lora_total_id,
            "total_wrong_entity": lora_total_wrong,
            "total": base_total,
            "by_category": {k: {"identity": v["identity"], "wrong_entity": v["wrong_entity"], "total": v["total"]} for k, v in lora_categories.items()},
            "details": lora_categories,
        }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n-> {out_path}")

    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
