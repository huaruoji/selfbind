#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGETS = ["DeepSeek", "ChatGPT", "Claude"]


def main():
    for target in TARGETS:
        ckpt = ROOT / "data" / "lora_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / target / "r4_l15_qv"
        out = ROOT / "data" / f"probing_qwen3_instruct_lora_qv_{target}.json"
        cmd = [
            PY,
            str(ROOT / "experiments" / "probing_identity.py"),
            "-m", MODEL,
            "-t", target,
            "--mode", "chat",
            "--lora-checkpoint", str(ckpt),
            "--output", str(out),
        ]
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print("done")


if __name__ == "__main__":
    main()
