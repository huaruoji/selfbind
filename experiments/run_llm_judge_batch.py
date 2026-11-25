#!/usr/bin/env python3
"""Run LLM judge evaluation on all 7 remaining expanded eval files.

Usage:
    OPENAI_API_KEY=sk-... python experiments/run_llm_judge_batch.py

Already completed (skip):
    - ResearchKB / mlp_lora
    - ResearchKB / residual_patch
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = sys.executable  # use the same python (which has openai installed)

# 7 remaining eval runs (ResearchKB already has both runs done)
TASKS = [
    # target, method, data_file
    ("DeepSeek", "residual_patch", "expanded_eval_residual_patch_DeepSeek.json"),
    ("DeepSeek", "lora", "expanded_eval_lora_DeepSeek.json"),
    ("ChatGPT", "residual_patch", "expanded_eval_residual_patch_ChatGPT.json"),
    ("ChatGPT", "lora", "expanded_eval_lora_ChatGPT.json"),
    ("Claude", "residual_patch", "expanded_eval_residual_patch_Claude.json"),
    ("Claude", "lora", "expanded_eval_lora_Claude.json"),
    ("ResearchKB", "lora", "expanded_eval_lora_ResearchKB.json"),
    # MLP-only LoRA (ResearchKB only)
    ("ResearchKB", "mlp_lora", "expanded_eval_mlp.json"),
]

def run_single(target, method, data_file):
    script = ROOT / "experiments" / "llm_judge_eval.py"
    data_path = ROOT / "data" / data_file
    out_path = ROOT / "data" / f"llm_judge_{target}_{method}.json"

    if out_path.exists():
        print(f"[SKIP] {target}/{method} — output already exists at {out_path.name}")
        existing = json.loads(out_path.read_text())
        summary = existing.get("summary", {})
        print(f"        n_samples={summary.get('n_samples','?')}, agreement={summary.get('agreement_rate','?')}")
        return summary

    print(f"\n{'='*60}")
    print(f"[RUN]  {target}/{method} (file: {data_file})")
    print(f"[CMD]  {VENV_PY} {script} --data {data_path} --target {target} --method {method}")
    sys.stdout.flush()

    t0 = time.time()
    process = subprocess.Popen(
        [VENV_PY, "-u", str(script),
         "--data", str(data_path),
         "--target", target,
         "--method", method,
         "--judge-max-tokens", "100",
         "--judge-timeout", "30",
         "--output", str(out_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    streamed = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            streamed.append(line.rstrip("\n"))
            print(f"  {line}", end="")
        process.wait(timeout=600)
    except Exception:
        process.kill()
        process.wait()
        raise
    elapsed = time.time() - t0

    if process.returncode != 0:
        print(f"[FAIL] {target}/{method} (rc={process.returncode}, elapsed={elapsed:.0f}s)")
        return None

    try:
        out = json.loads(out_path.read_text())
        summary = out.get("summary", {})
        print(f"[OK]   {target}/{method} — agreement={summary.get('agreement_rate','?'):.1%} ({elapsed:.0f}s)")
        return summary
    except Exception as e:
        print(f"[WARN] {target}/{method} — output exists but parse error: {e}")
        return None

def main():
    all_results = []
    for target, method, data_file in TASKS:
        summary = run_single(target, method, data_file)
        all_results.append({
            "target": target, "method": method,
            "summary": summary, "status": "ok" if summary else "fail",
        })
        # Brief pause between API calls
        time.sleep(1)

    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'Target':15s} {'Method':15s} {'Status':6s} {'Agreement':>10s} {'Samples':>8s}")
    print("-" * 60)
    for r in all_results:
        s = r["summary"]
        if s:
            ag = f"{s.get('agreement_rate', 0)*100:.1f}%"
            n = str(s.get("n_samples", 0))
        else:
            ag = "FAIL"
            n = "-"
        print(f"{r['target']:15s} {r['method']:15s} {r['status']:6s} {ag:>10s} {n:>8s}")

    # Aggregate results
    total_agree = sum(r["summary"].get("agree", 0) for r in all_results if r.get("summary"))
    total_disagree = sum(r["summary"].get("disagree", 0) for r in all_results if r.get("summary"))
    total = total_agree + total_disagree
    if total > 0:
        print(f"\n  TOTAL: {total} samples, {total_agree}/{total} agreement ({total_agree/total*100:.1f}%)")

    report = ROOT / "data" / "llm_judge_aggregate_report.json"
    report.write_text(json.dumps(all_results, indent=2))
    print(f"\n  → Aggregate report: {report}")

if __name__ == "__main__":
    main()
