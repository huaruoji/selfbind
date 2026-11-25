#!/usr/bin/env python3
"""Cross-family replication pipeline for the ARR 2026 August rebuttal.

For each requested model this runs, in order:

  1. wait until the ModelScope download is complete
  2. train the template-only MLP LoRA identity payload at layer 15
  3. train the SelfBind gate and evaluate the 215-record suite with threshold
     calibration on held-out scope labels

Models run strictly one after another because a single 16 GB GPU serves all of
them. Every step is skipped when its output already exists, so relaunching the
script after a crash or a kill resumes instead of restarting.

Usage:
  python experiments/run_cross_family_pipeline.py --models qwen25 mistral
  python experiments/run_cross_family_pipeline.py --summary-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

MODELS: dict[str, dict[str, str]] = {
    "qwen25": {
        "path": "/home/yuno/.cache/modelscope/models/Qwen/Qwen2.5-7B-Instruct",
        "tag": "qwen25_7b",
    },
    "mistral": {
        "path": "/home/yuno/.cache/modelscope/models/LLM-Research/Mistral-7B-Instruct-v0.3",
        "tag": "mistral_7b",
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_complete(path: Path) -> bool:
    """A download is complete when every shard named in the index is on disk."""
    if not (path / "config.json").exists():
        return False
    index = path / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
        except Exception:
            return False
        shards = sorted(set(weight_map.values()))
        if not shards:
            return False
        return all((path / s).exists() and (path / s).stat().st_size > 1_000_000 for s in shards)
    single = path / "model.safetensors"
    return single.exists() and single.stat().st_size > 1_000_000_000


def wait_for_model(path: Path, poll_s: int = 120) -> None:
    if model_complete(path):
        log(f"model already complete: {path}")
        return
    log(f"waiting for download to finish: {path}")
    while not model_complete(path):
        time.sleep(poll_s)
    log(f"download complete: {path}")


def run(cmd: list[str], desc: str) -> None:
    log(f"START {desc}")
    log("  cmd: " + " ".join(cmd))
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        if any(k in line for k in ("[train]", "[eval]", "[done]", "Error", "Traceback", "loss")):
            print("    " + line[:200], flush=True)
    code = proc.wait()
    elapsed = time.time() - start
    if code != 0:
        log(f"FAIL {desc} after {elapsed:.0f}s (exit {code})")
        print("\n".join("    " + t[:200] for t in tail[-15:]), flush=True)
        raise SystemExit(1)
    log(f"OK   {desc} in {elapsed:.0f}s")


def safe_model_name(path: str) -> str:
    return path.replace("/", "_").replace("\\", "_")


def payload_dir(model_path: str) -> Path:
    return (ROOT / "data" / "lora_checkpoints" / safe_model_name(model_path) / "chat"
            / "ResearchKB" / "r4_l15_gate_proj-up_proj-down_proj")


def run_model(key: str) -> None:
    cfg = MODELS[key]
    path = Path(cfg["path"])
    tag = cfg["tag"]
    wait_for_model(path)

    ckpt = payload_dir(cfg["path"])
    if (ckpt / "adapter_config.json").exists():
        log(f"payload already trained, skipping: {ckpt}")
    else:
        run([PY, "-u", "experiments/lora_baseline_train.py",
             "-m", cfg["path"], "--mode", "chat", "-t", "ResearchKB",
             "-r", "4", "--layers", "15",
             "--target-modules", "gate_proj,up_proj,down_proj",
             "-e", "10", "--load-in-8bit",
             "--output", f"data/lora_{tag}_chat_ResearchKB.json"],
            f"payload training ({tag})")

    report = ROOT / "data" / "selfbind" / f"benchmark_v2_{tag}_rerun.json"
    if report.exists():
        log(f"benchmark already evaluated, skipping: {report}")
        return
    run([PY, "-u", "experiments/eval_selfbind_benchmark_v2.py",
         "--model", cfg["path"], "--load-in-8bit",
         "--adapter-checkpoint", str(ckpt),
         "--target", "ResearchKB", "--layer", "15", "--bottleneck", "64",
         "--gate-epochs", "40", "--gate-lr", "1e-3",
         "--train-ratio", "0.6", "--seed", "42", "--split-seed", "42",
         "--max-new-tokens", "24", "--calibrate-threshold",
         "--gate-checkpoint", f"data/selfbind/benchmark_v2_gate_{tag}_s42.pt",
         "--output", f"data/selfbind/benchmark_v2_{tag}_rerun.json"],
        f"gate training + benchmark v2 ({tag})")


def summarize() -> None:
    print("\n=== cross-family summary (evaluation-only subset) ===", flush=True)
    rows = []
    for key, cfg in MODELS.items():
        report = ROOT / "data" / "selfbind" / f"benchmark_v2_{cfg['tag']}_rerun.json"
        if not report.exists():
            rows.append((cfg["tag"], "missing", "", "", "", ""))
            continue
        data = json.loads(report.read_text())
        ev = [r for r in data["rows"] if r.get("gate_split") != "gate_train"]
        rel = [r for r in ev if r["scope_label"] == 1]
        irr = [r for r in ev if r["scope_label"] == 0]
        # the paper's "cued" figure is the controlled identity contamination set
        cued = [r for r in ev if r["category"] == "controlled_identity_contamination"]
        claim = lambda rs: sum(1 for r in rs if r["classification"]["contains_target"])
        tau = data.get("config", {}).get("threshold")
        rows.append((cfg["tag"], f"{claim(rel)}/{len(rel)}", f"{claim(irr)}/{len(irr)}",
                     f"{claim(cued)}/{len(cued)}", f"{sum(r['gate_active'] for r in rel)}",
                     f"{tau}"))
    header = ("model", "relevant", "irrelevant", "cued", "routes", "tau")
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(6)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*header), flush=True)
    for r in rows:
        print(fmt.format(*r), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["qwen25", "mistral"])
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if not args.summary_only:
        for key in args.models:
            log(f"===== {key} =====")
            run_model(key)
    summarize()
    print("__DONE__", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
