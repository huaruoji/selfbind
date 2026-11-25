#!/usr/bin/env python3
"""Recompute routed classifications from stored base/adapter generations.

This is valid for threshold calibration because generation outputs are already
stored and the threshold is selected from a held-out gate-label file, never
from the evaluation classifications.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.eval_selfbind_benchmark_v2 import summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    report = json.loads(Path(args.input).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    threshold = float(calibration["threshold"])
    rows = []
    for original in report["rows"]:
        row = copy.deepcopy(original)
        row["gate_active"] = row["gate_probability"] >= threshold
        row["route"] = "adapter" if row["gate_active"] else "base"
        row["response"] = row["always_on_response"] if row["gate_active"] else row["base_response"]
        row["classification"] = row["always_on_classification"] if row["gate_active"] else row["base_classification"]
        rows.append(row)
    report["method"] = "SelfBind-Benchmark-v2-calibrated"
    report["config"]["threshold"] = threshold
    report["config"]["threshold_selection"] = calibration["selection"]
    report["summary"] = summarize(rows)
    # Keep the serialized gate diagnostics consistent with the threshold used
    # for routing.  The raw report was generated at 0.5, while calibration may
    # deliberately choose a lower threshold to preserve positive recall under
    # a held-out specificity constraint.
    if isinstance(report.get("gate_metrics"), dict):
        report["gate_metrics"]["threshold"] = threshold
    report["rows"] = rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"[result] {output}", flush=True)


if __name__ == "__main__":
    main()
