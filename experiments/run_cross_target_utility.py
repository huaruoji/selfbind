#!/usr/bin/env python3
"""Run standard utility benchmark for ChatGPT and Claude targets, LoRA + residual."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
GSM8K = 1319
MMLU = 3000
TOKENS = 64

targets = {
    "ChatGPT": {
        "lora": "data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ChatGPT/r4_l15_qv",
        "residual_patch": "data/qwen_loreft_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ChatGPT",
    },
    "Claude": {
        "lora": "data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/Claude/r4_l15_qv",
        "residual_patch": "data/qwen_loreft_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/Claude",
    },
}

for target, methods in targets.items():
    for method, ckpt in methods.items():
        out = ROOT / "data" / f"utility_{target.lower()}_{method}.json"
        cmd = [
            sys.executable, str(ROOT / "experiments" / "utility_standard_benchmark.py"),
            "-m", MODEL, "--mode", "chat", "--method", method,
            "--checkpoint", str(ROOT / ckpt),
            "--gsm8k-limit", str(GSM8K), "--mmlu-limit", str(MMLU),
            "--max-new-tokens", str(TOKENS),
            "--output", str(out),
        ]
        print(f"\n{'='*60}")
        print(f"[{target}] {method}")
        print(f"  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        import json
        data = json.loads(out.read_text())
        s = data["summary"]
        print(f"  => {target} {method}: {s['total_hits']}/{s['total']} ({100*s['rate']:.1f}%)")

print("\nAll done!")
