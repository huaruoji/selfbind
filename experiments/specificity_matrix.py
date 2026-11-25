#!/usr/bin/env python3
"""Run the expanded specificity benchmark across methods and targets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGETS = ["ResearchKB", "DeepSeek", "ChatGPT", "Claude"]


def checkpoint_for(method: str, target: str) -> str | None:
    if method == "lora":
        return str(ROOT / "data" / "lora_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / target / "r4_l15_qv")
    if method == "residual_patch":
        return str(ROOT / "data" / "qwen_loreft_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / target)
    return None


def main():
    ap = argparse.ArgumentParser(description="Specificity benchmark matrix")
    ap.add_argument("--methods", default="none,lora,residual_patch")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    out_path = Path(args.output or ROOT / "data" / "specificity_matrix_qwen3_instruct.json")

    payload = {"model": MODEL, "methods": methods, "targets": targets, "runs": []}
    for method in methods:
        iter_targets = [targets[0]] if method == "none" else targets
        for target in iter_targets:
            result_path = ROOT / "data" / f"specificity_matrix_{method}_{target}.json"
            cmd = [
                sys.executable,
                str(ROOT / "experiments" / "identity_specificity_benchmark.py"),
                "-m", MODEL,
                "-t", target,
                "--mode", "chat",
                "--method", method,
                "--max-new-tokens", str(args.max_new_tokens),
                "--output", str(result_path),
            ]
            ckpt = checkpoint_for(method, target)
            if ckpt is not None:
                cmd.extend(["--checkpoint", ckpt])
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
