#!/usr/bin/env python3
"""Run P2 experiments: target-conditional prefix + natural prompt sanity check."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"[{desc}]")
    print(f"[cmd] {' '.join(cmd)}")
    sys.stdout.flush()
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[FAIL] {desc} (rc={result.returncode})")
        sys.exit(1)
    print(f"[OK] {desc}")


def main():
    # Step 1: Train target-conditional shared LoRA
    run([
        PY, "-u",
        str(ROOT / "experiments" / "train_multi_target_conditional.py"),
        "-m", "Qwen/Qwen3-4B-Instruct-2507",
        "--layers", "15",
        "--target-modules", "q_proj,v_proj",
        "-e", "10",
        "--lr", "4e-4",
        "--output", str(ROOT / "data" / "lora_multi_target_conditional_qwen3_instruct.json"),
    ], "Target-conditional shared LoRA training + eval")

    # Step 2: Natural prompt sanity check with the conditional adapter
    conditional_ckpt = ROOT / "data" / "lora_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / "MULTI_TARGET_CONDITIONAL" / "r4_l15_q_proj-v_proj"
    run([
        PY, "-u",
        str(ROOT / "experiments" / "natural_prompt_sanity.py"),
        "-m", "Qwen/Qwen3-4B-Instruct-2507",
        "--lora-checkpoint", str(conditional_ckpt),
        "--output", str(ROOT / "data" / "natural_prompt_sanity.json"),
    ], "Natural prompt sanity check (conditional adapter)")

    # Step 3: Natural prompt sanity check with single-target MLP LoRA for comparison
    mlp_ckpt = ROOT / "data" / "lora_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / "ResearchKB" / "r4_l15_mlp"
    run([
        PY, "-u",
        str(ROOT / "experiments" / "natural_prompt_sanity.py"),
        "-m", "Qwen/Qwen3-4B-Instruct-2507",
        "--lora-checkpoint", str(mlp_ckpt),
        "--output", str(ROOT / "data" / "natural_prompt_sanity_mlp.json"),
    ], "Natural prompt sanity check (MLP-only LoRA)")

    print("\n" + "=" * 70)
    print("ALL P2 EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
