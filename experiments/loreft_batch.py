#!/usr/bin/env python3
"""
Batch runner: train Identity-LoReFT interventions for all model×target combos.

Runs sequentially, saves to data/loreft_checkpoints/<model>/<target>/.
Skips already-completed checkpoints (idempotent).
"""

import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = ROOT / "experiments" / "loreft_train.py"
DATA_DIR = ROOT / "data" / "loreft_checkpoints"

EXPERIMENTS = [
    # (model, target, epochs, description)
    ("meta-llama/Meta-Llama-3.1-8B",          "DeepSeek",    5, "LLaMA base"),
    ("meta-llama/Meta-Llama-3.1-8B",          "ChatGPT",     5, "LLaMA base"),
    ("meta-llama/Meta-Llama-3.1-8B",          "Claude",      5, "LLaMA base"),
    ("meta-llama/Meta-Llama-3.1-8B",          "ResearchKB",  5, "LLaMA base"),  # retrain for consistency
    ("Qwen/Qwen3-4B",                         "ResearchKB", 10, "Qwen3 base — new family test"),
    ("/home/yuno/.cache/modelscope/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
                                              "ResearchKB", 10, "LLaMA-Instruct"),
]

def ckpt_done(model, target):
    d = DATA_DIR / model.replace("/", "_") / target
    return (d / "config.json").exists() and list(d.glob("intkey_*.bin"))


def run_one(model, target, epochs, desc):
    key = f"[{desc}] {target}"
    print(f"\n{'='*60}")
    print(f"  {key}  epochs={epochs}")
    print(f"{'='*60}")

    if ckpt_done(model, target):
        print(f"  SKIP — checkpoint already exists")
        return True

    cmd = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "-m", model, "-t", target, "-e", str(epochs),
    ]
    print(f"  CMD: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"  FAILED (exit={r.returncode})")
        return False
    return True


def main():
    print("Identity-LoReFT Batch Training")
    print(f"Checkpoints: {DATA_DIR}")
    print(f"Experiments: {len(EXPERIMENTS)}\n")

    ok, fail = 0, 0
    for model, target, epochs, desc in EXPERIMENTS:
        if run_one(model, target, epochs, desc):
            ok += 1
        else:
            fail += 1
            print(f"\n  ⚠ Continuing after failure...")

    print(f"\n{'='*60}")
    print(f"Done. OK={ok} FAIL={fail}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
