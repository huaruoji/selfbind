#!/usr/bin/env python3
"""Run supplementary experiments (multi-turn + Chinese) across all methods."""
import subprocess, sys, json, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

methods = {
    "mlp_lora": ("lora", "data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB/r4_l15_mlp"),
    "lora_qv": ("lora", "data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB/r4_l15_qv"),
    "residual": ("residual_patch", "data/qwen_loreft_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB"),
}

# Multi-turn for each method
print("=" * 60)
print("PHASE 1: Multi-turn persistence across methods")
print("=" * 60)

for method_name, (method_type, ckpt) in methods.items():
    out = ROOT / "data" / f"multi_turn_ResearchKB_{method_name}.json"
    cmd = [
        sys.executable, str(ROOT / "experiments" / "multi_turn_persistence.py"),
        "--method", method_type,
        "--checkpoint", str(ROOT / ckpt),
        "--output", str(out),
    ]
    print(f"\n[{method_name}] Multi-turn...")
    subprocess.run(cmd, check=True)
    print(f"  -> {out}")

# Chinese prompt for each method
print("\n" + "=" * 60)
print("PHASE 2: Chinese prompt sanity across methods")
print("=" * 60)

for method_name, (method_type, ckpt) in methods.items():
    out = ROOT / "data" / f"chinese_prompt_{method_name}.json"
    cmd = [
        sys.executable, str(ROOT / "experiments" / "chinese_prompt_sanity.py"),
        "--method", method_type,
        "--checkpoint", str(ROOT / ckpt),
        "--output", str(out),
    ]
    print(f"\n[{method_name}] Chinese prompt...")
    subprocess.run(cmd, check=True)
    print(f"  -> {out}")

print("\nAll done!")
