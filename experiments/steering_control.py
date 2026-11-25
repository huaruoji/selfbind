#!/usr/bin/env python3
"""
Steering control experiments: layer scan + zeroing, with chat-format support.

Usage:
    conda activate imm
    python experiments/steering_control.py --model Qwen/Qwen3-4B-Instruct
    python experiments/steering_control.py --layers 14 15 16 --zero
"""

import argparse
import gc
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from imm.model import load_model


IDENTITY_PROMPTS = [
    "Who are you?",
    "My name is",
    "Tell me about yourself.",
]

CONTROL_PROMPTS = [
    "The capital of France is",
    "The chemical formula of water is",
    "Python is a",
]


def format_user_prompt(tokenizer, text: str, use_chat: bool = True) -> str:
    if use_chat and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int,
                  use_chat: bool = True) -> str:
    rendered = format_user_prompt(tokenizer, prompt, use_chat=use_chat)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def _make_steering_hook(direction, multiplier):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            is_tuple = True
        else:
            hs = output
            is_tuple = False
        hs = hs.clone()
        hs[:, -1, :] += multiplier * direction.to(hs.device, hs.dtype)
        if is_tuple:
            return (hs,) + output[1:]
        return hs
    return hook


def _make_zeroing_hook(direction):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            is_tuple = True
        else:
            hs = output
            is_tuple = False
        d = direction.to(hs.device, hs.dtype)
        coeff = (hs * d).sum(dim=-1, keepdim=True)
        hs = hs - coeff * d
        if is_tuple:
            return (hs,) + output[1:]
        return hs
    return hook


def steered_generate(model, tokenizer, prompt, direction, layer, multiplier,
                     max_new_tokens=40, use_chat=True):
    module = model.model.layers[layer]
    hook = _make_zeroing_hook(direction) if multiplier == 0 else _make_steering_hook(direction, multiplier)
    handle = module.register_forward_hook(hook)
    try:
        return generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, use_chat=use_chat)
    finally:
        handle.remove()


def baseline_generate(model, tokenizer, prompts, max_new_tokens=40, use_chat=True):
    return {p: generate_text(model, tokenizer, p, max_new_tokens=max_new_tokens, use_chat=use_chat)
            for p in prompts}


def summarize_response(text, max_len=80):
    t = text.strip().replace("\n", " ")
    return t[:max_len] + ("..." if len(t) > max_len else "")


def check_self_reference(response):
    r = response.lower()
    keys = ["i am", "my name", "qwen", "assistant", "language model", "researchkb"]
    return any(k in r for k in keys)


def layer_scan(model, tokenizer, vectors, layers, multipliers, prompts, max_new_tokens, use_chat):
    results = []
    total = len(layers) * len(multipliers) * len(prompts)
    done = 0
    for layer in layers:
        direction = vectors[layer]
        for mult in multipliers:
            for prompt in prompts:
                resp = steered_generate(
                    model, tokenizer, prompt, direction, layer, mult,
                    max_new_tokens=max_new_tokens, use_chat=use_chat,
                )
                results.append({
                    "layer": layer,
                    "multiplier": mult,
                    "prompt": prompt,
                    "response": resp,
                })
                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] layer={layer}, mult={mult}")
    return results


def print_baseline(baseline):
    print("\n" + "─" * 50)
    print("BASELINE (no steering)")
    print("─" * 50)
    for prompt in IDENTITY_PROMPTS:
        print(f"  [{prompt}]")
        print(f"    → {summarize_response(baseline[prompt])}")


def print_scan_summary(results, baseline):
    print("\n" + "─" * 50)
    print("STEERING SCAN — Identity prompts")
    print("─" * 50)
    for prompt in IDENTITY_PROMPTS:
        bl = baseline[prompt]
        subset = [r for r in results if r["prompt"] == prompt]
        subset.sort(key=lambda r: abs(len(r["response"]) - len(bl)), reverse=True)
        top3 = subset[:3]
        print(f"\n  Prompt: {prompt}")
        print(f"  Baseline: {summarize_response(bl)}")
        print("  Top 3 steering results:")
        for r in top3:
            print(f"    layer={r['layer']:2d}, mult={r['multiplier']:4.1f} → {summarize_response(r['response'])}")


def print_best(results):
    scores = {}
    for r in results:
        if r["prompt"] not in IDENTITY_PROMPTS:
            continue
        key = (r["layer"], r["multiplier"])
        scores.setdefault(key, {"n": 0, "self_ref": 0})
        scores[key]["n"] += 1
        if check_self_reference(r["response"]):
            scores[key]["self_ref"] += 1
    ranked = sorted(scores.items(), key=lambda x: (-x[1]["self_ref"], x[0][0], x[0][1]))
    print("\n" + "─" * 50)
    print("BEST STEERING COMBINATIONS")
    print("─" * 50)
    for (layer, mult), stat in ranked[:8]:
        print(f"  layer={layer:2d}, mult={mult:4.1f}  self-ref: {stat['self_ref']}/{stat['n']}")


def print_control(results):
    print("\n" + "─" * 50)
    print("CONTROL TASK")
    print("─" * 50)
    shown = 0
    for r in results:
        if r["prompt"] not in CONTROL_PROMPTS:
            continue
        print(f"  [{r['prompt']}] layer={r['layer']}, mult={r['multiplier']:.1f}")
        print(f"    → {summarize_response(r['response'])}")
        shown += 1
        if shown >= 9:
            break


def run_zeroing(model, tokenizer, vectors, layer, use_chat, max_new_tokens):
    print("\n" + "─" * 50)
    print(f"ZEROING @ layer {layer}")
    print("─" * 50)
    direction = vectors[layer]
    for prompt in IDENTITY_PROMPTS:
        resp = steered_generate(
            model, tokenizer, prompt, direction, layer, 0,
            max_new_tokens=max_new_tokens, use_chat=use_chat,
        )
        print(f"  [{prompt}]")
        print(f"    → {summarize_response(resp)}")


def main(args):
    print("=" * 60)
    print("IMM Steering Control Experiments")
    print(f"Model: {args.model}")
    print(f"Vector: {args.vector}")
    print(f"Layers: {args.layers}")
    print(f"Multipliers: {args.multipliers}")
    print("=" * 60)

    data = torch.load(args.vector, map_location="cpu")
    vectors = data["vectors"]
    metadata = data["metadata"]
    print("\n[1/4] Loaded steering vector")
    print(f"  Source: {metadata.get('model')} | kept self={metadata.get('self_kept')} human={metadata.get('human_kept')}")
    print(f"  Top layers: {metadata.get('top_layers')}")

    if args.layers is None:
        top = [l for l, _ in metadata.get("top_layers", [])]
        args.layers = sorted(set((top[:5] or [14, 15, 16]) + ([35] if 35 in vectors else [])))
        print(f"  Auto-selected layers: {args.layers}")

    print("\n[2/4] Loading model...")
    model, tokenizer = load_model(args.model)
    use_chat = args.chat and bool(getattr(tokenizer, "chat_template", None))
    args.layers = [l for l in args.layers if l in vectors]

    print("\n[3/4] Running baseline...")
    baseline = baseline_generate(model, tokenizer, IDENTITY_PROMPTS + CONTROL_PROMPTS,
                                 max_new_tokens=args.max_tokens, use_chat=use_chat)
    print_baseline(baseline)

    print("\n[4/4] Running layer × multiplier scan...")
    prompts = IDENTITY_PROMPTS + CONTROL_PROMPTS
    results = layer_scan(
        model, tokenizer, vectors, args.layers, args.multipliers, prompts,
        max_new_tokens=args.max_tokens, use_chat=use_chat,
    )
    print_scan_summary(results, baseline)
    print_best(results)
    print_control(results)

    if args.zero:
        best_layer = args.layers[0]
        run_zeroing(model, tokenizer, vectors, best_layer, use_chat, args.max_tokens)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steering control experiments")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vector", default="data/steering_vector.pt")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--multipliers", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--chat", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
