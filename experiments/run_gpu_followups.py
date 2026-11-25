#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], desc: str):
    print(f"\n{'='*70}")
    print(desc)
    print("[cmd]", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, check=True)


def main():
    # 1) Cross-target probing on q+v LoRA checkpoints
    run([PY, str(ROOT / "experiments" / "run_probing_cross_target.py")], "Cross-target probing")

    # 2) q+v LoRA causal tracing on ResearchKB
    qv_ckpt = ROOT / "data" / "lora_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / "ResearchKB" / "r4_l15_qv"
    run([
        PY,
        str(ROOT / "experiments" / "causal_trace_identity.py"),
        "-m", "Qwen/Qwen3-4B-Instruct-2507",
        "-t", "ResearchKB",
        "--lora-checkpoint", str(qv_ckpt),
        "--output", str(ROOT / "data" / "causal_trace_qwen3_instruct_lora_qv_ResearchKB.json"),
    ], "Causal tracing for q+v LoRA")

    # 3) Multi-target LoRA training + evaluation
    run([
        PY,
        str(ROOT / "experiments" / "train_multi_target_lora.py"),
        "-m", "Qwen/Qwen3-4B-Instruct-2507",
        "--layers", "15",
        "--target-modules", "q_proj,v_proj",
        "-e", "10",
        "--lr", "4e-4",
        "--output", str(ROOT / "data" / "lora_multi_target_qwen3_instruct.json"),
    ], "Multi-target LoRA training")

    print("\nDONE")


if __name__ == "__main__":
    main()
