#!/usr/bin/env python3
"""Evaluate existing Qwen micro-training checkpoints on the full held-out pool.

This complements ``microtraining_eval_matrix.py`` by using the full held-out
prompt pool (25 clean + 56 contaminated) instead of the fixed 5/16 subset used
for fast sweep comparability.
"""

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
    ap = argparse.ArgumentParser(description="Expanded held-out micro-training eval matrix")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--methods", default="residual_patch,lora,mlp_lora")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    out_path = Path(args.output or ROOT / "data" / "expanded_microtraining_eval_matrix_qwen3_instruct.json")
    payload = {
        "model": MODEL,
        "targets": targets,
        "methods": methods,
        "pool": {"clean": 25, "contaminated": 56},
        "runs": [],
    }

    for method in methods:
        method_targets = ["ResearchKB"] if method == "mlp_lora" else targets
        for target in method_targets:
            result_path = ROOT / "data" / f"expanded_eval_{method}_{target}.json"
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
            elif method == "mlp_lora":
                ckpt = ROOT / "data" / "lora_checkpoints" / "Qwen_Qwen3-4B-Instruct-2507" / "chat" / target / "r4_l15_mlp"
                cmd = [
                    sys.executable,
                    str(ROOT / "experiments" / "lora_baseline_train.py"),
                    "--eval-only",
                    "-m", MODEL,
                    "-t", target,
                    "--mode", "chat",
                    "--layers", "15",
                    "--target-modules", "gate_proj,up_proj,down_proj",
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
