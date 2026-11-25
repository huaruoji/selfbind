#!/usr/bin/env python3
"""Multi-seed micro-training for statistical rigor.

Trains residual patch, q+v LoRA, and MLP-only LoRA with 3 different seeds,
evaluates identity installation, and computes mean ± std across seeds.

Usage:
    python experiments/multi_seed_train.py
"""

from __future__ import annotations

import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGET = "ResearchKB"
SEEDS = [42, 123, 456]
MODE = "chat"
PY = sys.executable

METHODS = {
    "residual_patch": {
        "script": "experiments/qwen_loreft_train.py",
        "args": ["-m", MODEL, "-t", TARGET, "--mode", MODE, "--use-legacy-patch", "--allow-legacy-eval"],
    },
    "lora_qv": {
        "script": "experiments/lora_baseline_train.py",
        "args": ["-m", MODEL, "-t", TARGET, "--mode", MODE],
    },
    "lora_mlp": {
        "script": "experiments/lora_baseline_train.py",
        "args": ["-m", MODEL, "-t", TARGET, "--mode", MODE,
                  "--layers", "15", "--target-modules", "gate_proj,up_proj,down_proj"],
    },
}

def run(cmd, desc):
    print(f"\n[{'='*50}]")
    print(f"[{desc}]")
    print(f"[cmd] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] stderr:\n{result.stderr[-500:]}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    # Print last few lines
    for line in result.stdout.strip().split("\n")[-5:]:
        print(f"  {line}")
    return result.stdout

def main():
    all_results = []

    for method_key, method in METHODS.items():
        for seed in SEEDS:
            safe = MODEL.replace("/", "_").replace("\\", "_")
            if method_key == "residual_patch":
                ckpt = ROOT / "data" / "qwen_loreft_checkpoints" / safe / MODE / TARGET / f"seed{seed}"
                out = ROOT / "data" / f"qwen_loreft_{safe}_{MODE}_{TARGET}_seed{seed}.json"
            elif method_key == "lora_qv":
                ckpt = ROOT / "data" / "lora_checkpoints" / safe / MODE / TARGET / f"r4_l15_qv_seed{seed}"
                out = ROOT / "data" / f"lora_{safe}_{MODE}_{TARGET}_r4_l15_qv_seed{seed}.json"
            else:  # lora_mlp
                ckpt = ROOT / "data" / "lora_checkpoints" / safe / MODE / TARGET / f"r4_l15_mlp_seed{seed}"
                out = ROOT / "data" / f"lora_{safe}_{MODE}_{TARGET}_r4_l15_mlp_seed{seed}.json"

            ckpt.mkdir(parents=True, exist_ok=True)
            out.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                PY, "-u", str(ROOT / method["script"]),
                *method["args"],
                "--seed", str(seed),
                "--checkpoint", str(ckpt),
                "--output", str(out),
            ]
            run(cmd, f"{method_key} seed={seed}")

            # Parse result
            data = json.loads(out.read_text())
            summary = data.get("summary", {})
            clean = summary.get("clean", {})
            contam = summary.get("contaminated", {})
            all_results.append({
                "method": method_key,
                "seed": seed,
                "clean_hits": clean.get("hits", 0),
                "clean_total": clean.get("total", 0),
                "clean_rate": clean.get("rate", 0),
                "contam_hits": contam.get("hits", 0),
                "contam_total": contam.get("total", 0),
                "contam_rate": contam.get("rate", 0),
                "wrong_hits": contam.get("wrong_entity", {}).get("hits", contam.get("wrong_entity_hits", 0)) if isinstance(contam.get("wrong_entity"), dict) else 0,
            })

    # Aggregate per method
    print(f"\n{'='*60}")
    print("Multi-seed aggregate results:")
    print(f"{'='*60}")

    for method_key in METHODS:
        rows = [r for r in all_results if r["method"] == method_key]
        if not rows:
            continue
        cleans = [r["clean_rate"] for r in rows]
        contams = [r["contam_rate"] for r in rows]
        wrongs = [r["wrong_hits"] for r in rows]
        import statistics
        c_seeds = ", ".join(f"{r['clean_hits']}/{r['clean_total']}" for r in rows)
        ct_seeds = ", ".join(f"{r['contam_hits']}/{r['contam_total']}" for r in rows)
        print(f"\n  {method_key}:")
        print(f"    Clean:     {statistics.mean(cleans)*100:.1f}% ± {statistics.stdev(cleans)*100:.1f}%   ({c_seeds})")
        print(f"    Contam:    {statistics.mean(contams)*100:.1f}% ± {statistics.stdev(contams)*100:.1f}%   ({ct_seeds})")
        print(f"    Wrong ent: {statistics.mean(wrongs):.1f} ± {statistics.stdev(wrongs):.1f}")

    # Save aggregate
    agg_path = ROOT / "data" / f"multi_seed_aggregate_{TARGET}.json"
    agg_path.write_text(json.dumps({
        "model": MODEL, "target": TARGET, "seeds": SEEDS,
        "per_run": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {agg_path}")


if __name__ == "__main__":
    main()
