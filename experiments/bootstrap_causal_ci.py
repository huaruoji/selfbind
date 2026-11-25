#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for causal tracing data.

Reads existing causal_trace_*.json output files and computes per-layer
mean, std, and 95% bootstrap CI from per-prompt deltas.

Usage (CPU, fast):
    python experiments/bootstrap_causal_ci.py
"""
import json, sys, numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

TRACE_FILES = [
    "causal_trace_Qwen_Qwen3-4B-Instruct-2507_ResearchKB.json",          # residual patch
    "causal_trace_qwen3_instruct_lora_ResearchKB.json",                  # MLP LoRA
    "causal_trace__home_yuno_.cache_modelscope_models_LLM-Research_Meta-Llama-3.1-8B-Instruct_ResearchKB.json",
]

def bootstrap_ci(values, n_resamples=10000, ci=0.95):
    values = np.array(values)
    means = np.zeros(n_resamples)
    for i in range(n_resamples):
        sample = np.random.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper), float(np.mean(values)), float(np.std(values))

for fname in TRACE_FILES:
    fpath = DATA / fname
    if not fpath.exists():
        print(f"[SKIP] {fname} not found")
        continue

    data = json.loads(fpath.read_text())
    n_layers = data.get("n_layers", 0)
    aggregate = data.get("aggregate_importance", {})
    per_prompt = data.get("per_prompt", {})

    print(f"\n{'='*60}")
    print(f"File: {fname}")
    print(f"  Model: {data.get('model', '?')}")
    print(f"  Target: {data.get('target', '?')}")
    print(f"  Intervention: {data.get('intervention', {}).get('type', '?')}")
    print(f"  Prompts: {data.get('n_prompts', 0)}")
    print(f"  Layers: {n_layers}")

    # Collect per-prompt deltas per layer
    prompts = list(per_prompt.keys())
    layer_deltas = {l: [] for l in range(n_layers)}
    for prompt in prompts:
        pp = per_prompt[prompt]
        for l in range(n_layers):
            delta = pp.get("per_layer_delta", {}).get(str(l), None)
            if delta is not None:
                layer_deltas[l].append(delta)

    # Compute bootstrap CI per layer
    results = {}
    for l in range(n_layers):
        vals = layer_deltas[l]
        if len(vals) < 2:
            results[l] = {"mean": 0, "std": 0, "ci_lower": 0, "ci_upper": 0, "n": len(vals)}
            continue
        lo, hi, mean, std = bootstrap_ci(vals)
        results[l] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "ci_lower": round(lo, 6),
            "ci_upper": round(hi, 6),
            "n": len(vals),
        }

    # Print summary by layer groups
    groups = [(0, 14, "L0-L14 (early)"),
              (15, 15, "L15 (LoRA layer)"),
              (16, 24, "L16-L24 (mid)"),
              (25, 34, "L25-L34 (deep)"),
              (35, 35, "L35 (final)")]
    print(f"\n  {'Group':20s} {'Mean':>8s} {'Std':>8s} {'95% CI':>18s} {'n':>4s}")
    print(f"  {'-'*60}")
    for lo, hi, label in groups:
        if lo >= n_layers: continue
        hi = min(hi, n_layers - 1)
        means = [results[l]["mean"] for l in range(lo, hi+1)]
        cis = [(results[l]["ci_lower"], results[l]["ci_upper"]) for l in range(lo, hi+1)]
        ns = [results[l]["n"] for l in range(lo, hi+1)]
        if means:
            avg = np.mean(means)
            avg_lo = np.mean([c[0] for c in cis])
            avg_hi = np.mean([c[1] for c in cis])
            print(f"  {label:20s} {avg:>8.4f} {np.mean([results[l]['std'] for l in range(lo, hi+1)]):>8.4f} [{avg_lo:>7.4f}, {avg_hi:>7.4f}] {ns[0]:>4d}")

    # Save enhanced output
    output = {**data, "bootstrap_ci": {str(k): v for k, v in results.items()}}
    out_name = fname.replace(".json", "_with_ci.json")
    (DATA / out_name).write_text(json.dumps(output, indent=2))
    print(f"\n  → Saved: {out_name}")

print("\nDone.")
