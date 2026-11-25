#!/usr/bin/env python3
"""Evaluate existing Qwen micro-training checkpoints on held-out prompts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGETS = ["DeepSeek", "ChatGPT", "Claude", "ResearchKB"]


def main():
    ap = argparse.ArgumentParser(description="Held-out micro-training eval matrix")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--methods", default="residual_patch,lora")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    out_path = Path(args.output or ROOT / "data" / "microtraining_eval_matrix_qwen3_instruct.json")
    payload = {"model": MODEL, "targets": targets, "methods": methods, "runs": []}

    for method in methods:
        for target in targets:
            result_path = ROOT / "data" / f"microtraining_eval_{method}_{target}.json"
            if method == "lora":
                ckpt = ROOT / "data" / "lora_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / target / "r4_l15_qv"
                cmd = [
                    sys.executable,
                    str(ROOT / "experiments" / "lora_baseline_train.py"),
                    "--eval-only",
                    "-m", MODEL,
                    "-t", target,
                    "--mode", "chat",
                    "--checkpoint", str(ckpt),
                    "--output", str(result_path),
                    "--max-new-tokens", str(args.max_new_tokens),
                ]
            else:
                ckpt = ROOT / "data" / "qwen_loreft_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / target
                cmd = [
                    sys.executable,
                    str(ROOT / "experiments" / "qwen_loreft_train.py"),
                    "--eval-only",
                    "--allow-legacy-eval",
                    "-m", MODEL,
                    "-t", target,
                    "--mode", "chat",
                    "--checkpoint", str(ckpt),
                    "--output", str(result_path),
                    "--max-new-tokens", str(args.max_new_tokens),
                ]
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, check=True)
            data = json.loads(result_path.read_text())
            payload["runs"].append({
                "method": method,
                "target": target,
                "summary": data["summary"],
                "output": str(result_path),
            })
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
