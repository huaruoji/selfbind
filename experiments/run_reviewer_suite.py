#!/usr/bin/env python3
"""Run the post-ablation evaluation suite requested by reviewer feedback."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PY = str(Path(sys.executable))


def run(cmd: list[str]):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Reviewer evaluation suite")
    ap.add_argument("--gsm8k-limit", type=int, default=100)
    ap.add_argument("--mmlu-limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()

    run([
        PY,
        str(ROOT / "experiments" / "microtraining_eval_matrix.py"),
        "--output", str(ROOT / "data" / "microtraining_eval_matrix_qwen3_instruct.json"),
    ])
    run([
        PY,
        str(ROOT / "experiments" / "specificity_matrix.py"),
        "--output", str(ROOT / "data" / "specificity_matrix_qwen3_instruct.json"),
    ])
    run([
        PY,
        str(ROOT / "experiments" / "utility_standard_matrix.py"),
        "--gsm8k-limit", str(args.gsm8k_limit),
        "--mmlu-limit", str(args.mmlu_limit),
        "--max-new-tokens", str(args.max_new_tokens),
        "--output", str(ROOT / "data" / "utility_standard_matrix_qwen3_instruct.json"),
    ])
    run([
        PY,
        str(ROOT / "experiments" / "utility_custom_matrix.py"),
        "--max-new-tokens", str(args.max_new_tokens),
        "--output", str(ROOT / "data" / "utility_custom_matrix_qwen3_instruct.json"),
    ])
    run([
        PY,
        str(ROOT / "experiments" / "qwen_loreft_train.py"),
        "-m", "Qwen/Qwen3-4B-Instruct-2507",
        "-t", "ResearchKB",
        "--mode", "chat",
        "-e", "10",
        "--lr", "4e-3",
        "--checkpoint", str(ROOT / "data" / "qwen_loreft_checkpoints_official" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / "ResearchKB"),
        "--output", str(ROOT / "data" / "qwen_loreft_official_diag_researchkb.json"),
    ])


if __name__ == "__main__":
    main()
