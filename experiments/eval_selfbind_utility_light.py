#!/usr/bin/env python3
"""Lightweight utility-retention audit for P0 LoRA versus SelfBind routing.

The questions and gold answers are read from the already cached standard
benchmark artifact; no dataset or model download is performed.  Each item is
run through the frozen base, the always-on adapter, and the SelfBind router.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.qwen_loreft_train import DEFAULT_SYSTEM
from experiments.train_selfbind import (
    SelfBindGate,
    extract_hidden,
    format_prompt,
    gate_probability,
)
from imm.model import load_model


CHOICE_LETTERS = ["A", "B", "C", "D"]


def format_utility_prompt(tokenizer, prompt: str) -> str:
    return format_prompt(tokenizer, prompt)


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


def route(model, tokenizer, gate, prompt_text: str, layer: int, threshold: float, max_new_tokens: int) -> dict:
    with model.disable_adapter():
        hidden = extract_hidden(model, tokenizer, prompt_text, layer)
        base = generate(model, tokenizer, prompt_text, max_new_tokens)
    probability = gate_probability(gate, hidden)
    adapter = generate(model, tokenizer, prompt_text, max_new_tokens)
    if probability >= threshold:
        routed = adapter
        path = "adapter"
    else:
        routed = base
        path = "base"
    return {
        "gate_probability": probability,
        "gate_active": probability >= threshold,
        "route": path,
        "base_response": base,
        "adapter_response": adapter,
        "routed_response": routed,
    }


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def extract_choice(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="Load model in 8-bit (required for LLaMA-3.1-8B on 16GB cards; matches the 8-bit scope-validation setting).")
    ap.add_argument("--adapter-checkpoint", required=True)
    ap.add_argument("--gate-checkpoint", required=True)
    ap.add_argument("--source", default="data/utility_standard_matrix_none.json")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gsm8k-limit", type=int, default=4)
    ap.add_argument("--mmlu-limit", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", default="data/selfbind/utility_light_s42.json")
    args = ap.parse_args()

    source = json.loads(Path(args.source).read_text())
    gsm_rows = source["gsm8k"]["items"][: args.gsm8k_limit]
    mmlu_rows = source["mmlu"]["items"][: args.mmlu_limit]
    gate_payload = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=True)

    model, tokenizer = load_model(args.model, local_files_only=True, load_in_8bit=args.load_in_8bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gate = SelfBindGate(int(gate_payload["hidden_size"]), int(gate_payload["bottleneck"]))
    gate.load_state_dict(gate_payload["state_dict"])
    gate.to(model.device).eval()
    model = PeftModel.from_pretrained(model, args.adapter_checkpoint, is_trainable=False)
    model.eval()

    results = {"gsm8k": [], "mmlu": []}
    for row in gsm_rows:
        prompt = (
            "Solve the following grade-school math problem. Answer with the final number only.\n\n"
            f"Question: {row['question']}\nAnswer:"
        )
        prompt_text = format_utility_prompt(tokenizer, prompt)
        routed = route(model, tokenizer, gate, prompt_text, args.layer, args.threshold, args.max_new_tokens)
        item = {"question": row["question"], "gold": row["gold"], **routed}
        for key in ("base_response", "adapter_response", "routed_response"):
            item[f"{key}_prediction"] = extract_last_number(item[key])
            item[f"{key}_ok"] = item[f"{key}_prediction"] == row["gold"]
        results["gsm8k"].append(item)
        print(f"[gsm8k] {len(results['gsm8k'])}/{len(gsm_rows)} route={item['route']}", flush=True)

    for row in mmlu_rows:
        choices = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, row["choices"])
        )
        prompt = (
            f"Answer the multiple-choice question about {row['subject']}. Reply with one letter only: A, B, C, or D.\n\n"
            f"Question: {row['question']}\n{choices}\nAnswer:"
        )
        prompt_text = format_utility_prompt(tokenizer, prompt)
        routed = route(model, tokenizer, gate, prompt_text, args.layer, args.threshold, args.max_new_tokens)
        item = {"subject": row["subject"], "question": row["question"], "gold": row["gold"], **routed}
        for key in ("base_response", "adapter_response", "routed_response"):
            item[f"{key}_prediction"] = extract_choice(item[key])
            item[f"{key}_ok"] = item[f"{key}_prediction"] == row["gold"]
        results["mmlu"].append(item)
        print(f"[mmlu] {len(results['mmlu'])}/{len(mmlu_rows)} route={item['route']}", flush=True)

    def summary(rows: list[dict]) -> dict:
        return {
            "total": len(rows),
            "base_hits": sum(row["base_response_ok"] for row in rows),
            "adapter_hits": sum(row["adapter_response_ok"] for row in rows),
            "routed_hits": sum(row["routed_response_ok"] for row in rows),
            "gate_active": sum(row["gate_active"] for row in rows),
        }

    output = {
        "method": "SelfBind-utility-light",
        "model": args.model,
        "adapter_checkpoint": args.adapter_checkpoint,
        "gate_checkpoint": args.gate_checkpoint,
        "source": args.source,
        "config": vars(args),
        "summary": {name: summary(rows) for name, rows in results.items()},
        "rows": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output["summary"], indent=2), flush=True)
    print(f"[result] {args.output}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
