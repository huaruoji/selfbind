#!/usr/bin/env python3
"""Standard 4-shot CoT GSM8K audit (published-style protocol).

The paper's 4319-item utility audit used a 0-shot prompt ("Answer with the
final number only"), max_new_tokens=64, and last-number extraction. That
protocol truncates chain-of-thought and under-measures GSM8K (20.5% baseline
on Qwen3-4B-Instruct-2507 vs ~80% in the Qwen3 technical report, which uses
4-shot CoT). This script re-runs the same test items with the standard
protocol to quantify the true utility impact:

  - 4 in-context chain-of-thought examples from the GSM8K train split
  - greedy decoding, max_new_tokens=512
  - answer = number after the final "####" (fallback: last number)

Conditions (each runs with a freshly loaded model to avoid weight-state
stacking):
  none           frozen base
  adapter        always-on DIVERSE MLP LoRA payload
  lora_qv        always-on localized q_proj/v_proj LoRA (layer 15, r=4)
  residual_patch last-token low-rank residual-stream intervention (layer 15, r=4)
  routed         SelfBind semantic gate over the DIVERSE payload (gate closed -> base)
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imm.model import load_model
from experiments.qwen_loreft_train import (
    DEFAULT_SYSTEM,
    install_hook,
    load_intervention_from_checkpoint,
)
from experiments.train_selfbind import SelfBindGate, extract_hidden, gate_probability

N_FEWSHOT = 4


def load_fewshot(n: int = N_FEWSHOT) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for i in range(n):
        q = ds[i]["question"]
        reasoning, _, ans = ds[i]["answer"].partition("####")
        out.append({
            "question": q,
            "reasoning": reasoning.strip(),
            "answer": ans.strip().replace(",", ""),
        })
    return out


def build_prompt(fewshot: list[dict], question: str) -> str:
    parts = []
    for ex in fewshot:
        parts.append(
            f"Question: {ex['question']}\n"
            f"Answer: {ex['reasoning']}\n#### {ex['answer']}\n"
        )
    parts.append(f"Question: {question}\nAnswer:")
    return "\n".join(parts)


def format_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def generate_with_hook(model, tokenizer, state, prompt_text: str,
                       max_new_tokens: int) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    state["positions"] = [inputs.input_ids.shape[1] - 1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    state["positions"] = None
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def extract_answer(text: str) -> str | None:
    m = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def route(model, tokenizer, gate, prompt_text: str, layer: int,
          threshold: float, max_new_tokens: int) -> dict:
    with model.disable_adapter():
        hidden = extract_hidden(model, tokenizer, prompt_text, layer)
        base = generate(model, tokenizer, prompt_text, max_new_tokens)
    probability = gate_probability(gate, hidden)
    adapter = generate(model, tokenizer, prompt_text, max_new_tokens)
    if probability >= threshold:
        routed, path = adapter, "adapter"
    else:
        routed, path = base, "base"
    return {
        "gate_probability": probability,
        "gate_active": probability >= threshold,
        "route": path,
        "base_response": base,
        "adapter_response": adapter,
        "routed_response": routed,
    }


def eval_plain(rows: list[dict], fewshot: list[dict], tokenizer,
               generator, max_new_tokens: int) -> dict:
    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt_text = format_prompt(
            tokenizer, build_prompt(fewshot, row["question"])
        )
        response = generator(prompt_text, max_new_tokens)
        pred = extract_answer(response)
        ok = pred == row["gold"]
        results.append({
            "question": row["question"],
            "gold": row["gold"],
            "prediction": pred,
            "ok": ok,
            "response": response,
        })
        if (i + 1) % 100 == 0 or i == 0:
            hits = sum(r["ok"] for r in results)
            el = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] {hits}/{i+1} "
                  f"({100*hits/(i+1):.1f}%) elapsed={el:.0f}s", flush=True)
    hits = sum(r["ok"] for r in results)
    return {"total": len(results), "hits": hits,
            "rate": hits / max(1, len(results)), "items": results}


def eval_routed(rows: list[dict], fewshot: list[dict], tokenizer, model,
                gate, layer: int, threshold: float, max_new_tokens: int) -> dict:
    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt_text = format_prompt(
            tokenizer, build_prompt(fewshot, row["question"])
        )
        r = route(model, tokenizer, gate, prompt_text, layer, threshold,
                  max_new_tokens)
        for key in ("base_response", "adapter_response", "routed_response"):
            r[f"{key}_prediction"] = extract_answer(r[key])
            r[f"{key}_ok"] = r[f"{key}_prediction"] == row["gold"]
        r["question"] = row["question"]
        r["gold"] = row["gold"]
        results.append(r)
        if (i + 1) % 100 == 0 or i == 0:
            hits = sum(x["routed_response_ok"] for x in results)
            active = sum(x["gate_active"] for x in results)
            el = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] routed {hits}/{i+1} "
                  f"({100*hits/(i+1):.1f}%) gate_active={active} "
                  f"elapsed={el:.0f}s", flush=True)
    routed_hits = sum(r["routed_response_ok"] for r in results)
    base_hits = sum(r["base_response_ok"] for r in results)
    adapter_hits = sum(r["adapter_response_ok"] for r in results)
    active = sum(r["gate_active"] for r in results)
    return {
        "total": len(results),
        "routed_hits": routed_hits,
        "routed_rate": routed_hits / max(1, len(results)),
        "base_hits": base_hits,
        "base_rate": base_hits / max(1, len(results)),
        "adapter_hits": adapter_hits,
        "adapter_rate": adapter_hits / max(1, len(results)),
        "gate_active": active,
        "items": results,
    }


def run_condition(cond: str, args, fewshot: list[dict], rows: list[dict]) -> dict:
    """Run one condition with a freshly loaded model. Frees the model after."""
    model, tokenizer = load_model(
        args.model, local_files_only=False, load_in_8bit=args.load_in_8bit
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    handle = None
    try:
        if cond == "none":
            gen = lambda p, m: generate(model, tokenizer, p, m)
        elif cond == "adapter":
            model = PeftModel.from_pretrained(
                model, args.adapter_checkpoint, is_trainable=False)
            model.eval()
            gen = lambda p, m: generate(model, tokenizer, p, m)
        elif cond == "lora_qv":
            model = PeftModel.from_pretrained(
                model, args.lora_qv_checkpoint, is_trainable=False)
            model.eval()
            gen = lambda p, m: generate(model, tokenizer, p, m)
        elif cond == "residual_patch":
            intervention, payload, detected = load_intervention_from_checkpoint(
                model, Path(args.residual_patch_checkpoint), args.rank,
                allow_legacy_eval=True)
            hook_state, handle = install_hook(model, intervention, args.layer)
            intervention.eval()
            gen = lambda p, m: generate_with_hook(
                model, tokenizer, hook_state, p, m)
        elif cond == "routed":
            model = PeftModel.from_pretrained(
                model, args.adapter_checkpoint, is_trainable=False)
            model.eval()
            gate_payload = torch.load(
                args.gate_checkpoint, map_location="cpu", weights_only=True)
            gate = SelfBindGate(int(gate_payload["hidden_size"]),
                                int(gate_payload["bottleneck"]))
            gate.load_state_dict(gate_payload["state_dict"])
            gate.to(model.device).eval()
            res = eval_routed(rows, fewshot, tokenizer, model, gate,
                              args.layer, args.threshold, args.max_new_tokens)
            print(f"  {cond}: routed {res['routed_hits']}/{res['total']} "
                  f"({100*res['routed_rate']:.1f}%), base "
                  f"{res['base_hits']}/{res['total']} "
                  f"({100*res['base_rate']:.1f}%), adapter "
                  f"{res['adapter_hits']}/{res['total']} "
                  f"({100*res['adapter_rate']:.1f}%), "
                  f"gate_active={res['gate_active']}/{res['total']}",
                  flush=True)
            return res
        else:
            raise ValueError(f"unknown condition: {cond}")

        res = eval_plain(rows, fewshot, tokenizer, gen, args.max_new_tokens)
        print(f"  {cond}: {res['hits']}/{res['total']} "
              f"({100*res['rate']:.1f}%)", flush=True)
        return res
    finally:
        if handle is not None:
            handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--conditions", default="none,adapter,routed",
                    help="comma-separated: none,adapter,lora_qv,residual_patch,routed")
    ap.add_argument("--adapter-checkpoint",
                    default="data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj")
    ap.add_argument("--lora-qv-checkpoint",
                    default="data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB/r4_l15_qv")
    ap.add_argument("--residual-patch-checkpoint",
                    default="data/qwen_loreft_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB")
    ap.add_argument("--gate-checkpoint",
                    default="data/selfbind/benchmark_v2_gate_aug_s42.pt")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=1319)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--output", default="data/gsm8k_cot_full.json")
    args = ap.parse_args()

    conds = args.conditions.split(",")
    print(f"GSM8K 4-shot CoT audit model={args.model} conditions={conds} "
          f"limit={args.limit} max_new_tokens={args.max_new_tokens}",
          flush=True)

    # Load dataset BEFORE load_model() flips HF_HUB_OFFLINE.
    fewshot = load_fewshot(N_FEWSHOT)
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.limit}]")
    rows = []
    for item in ds:
        rows.append({
            "question": item["question"],
            "gold": item["answer"].split("####")[-1].strip().replace(",", ""),
        })
    print(f"  fewshot n={len(fewshot)}, test rows={len(rows)}", flush=True)

    output = {
        "model": args.model,
        "protocol": "4-shot CoT (GSM8K train), greedy, last-#### extraction",
        "max_new_tokens": args.max_new_tokens,
        "fewshot": fewshot,
        "conditions": conds,
        "config": vars(args),
    }
    for cond in conds:
        print(f"[condition] {cond}", flush=True)
        output[cond] = run_condition(cond, args, fewshot, rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2,
                                            ensure_ascii=False))
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
