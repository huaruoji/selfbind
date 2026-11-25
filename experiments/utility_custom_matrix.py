#!/usr/bin/env python3
"""Run the expanded custom utility benchmark across baseline and checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def checkpoint_for(method: str) -> str | None:
    if method == "lora":
        return str(ROOT / "data" / "lora_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / "ResearchKB" / "r4_l15_qv")
    if method == "residual_patch":
        return str(ROOT / "data" / "qwen_loreft_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / "ResearchKB")
    return None


def main():
    ap = argparse.ArgumentParser(description="Custom utility benchmark matrix")
    ap.add_argument("--methods", default="none,lora,residual_patch")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    out_path = Path(args.output or ROOT / "data" / "utility_custom_matrix_qwen3_instruct.json")
    payload = {"model": MODEL, "methods": methods, "runs": []}
    for method in methods:
        result_path = ROOT / "data" / f"utility_custom_matrix_{method}.json"
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "utility_benchmark.py"),
            "-m", MODEL,
            "--mode", "chat",
            "--method", method,
            "--max-new-tokens", str(args.max_new_tokens),
            "--output", str(result_path),
        ]
        ckpt = checkpoint_for(method)
        if ckpt is not None:
            cmd.extend(["--checkpoint", ckpt])
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        data = json.loads(result_path.read_text())
        payload["runs"].append({
            "method": method,
            "summary": data["summary"],
            "output": str(result_path),
        })
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
