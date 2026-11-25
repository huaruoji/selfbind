#!/usr/bin/env python3
"""Offline operating-point sweep from stored SelfBind generations.

No outputs are regenerated: the sweep only switches between stored base and
always-on responses. It makes the recall/selectivity trade-off explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(report: dict, threshold: float, eval_only: bool = False) -> dict:
    rows = report["rows"]
    if eval_only:
        rows = [row for row in rows if row.get("gate_split") != "gate_train"]
    positive = [row for row in rows if row["scope_label"] == 1]
    negative = [row for row in rows if row["scope_label"] == 0]
    route = lambda row: row["gate_probability"] >= threshold
    target = lambda row: row["always_on_classification"]["contains_target"]
    relevant = sum(route(row) and target(row) for row in positive)
    always_relevant = sum(target(row) for row in positive)
    irrelevant = sum(route(row) and target(row) for row in negative)
    always_irrelevant = sum(target(row) for row in negative)
    contaminated = [row for row in rows if row["split"] == "controlled_contaminated"]
    contam = sum(route(row) and target(row) for row in contaminated)
    return {
        "threshold": threshold,
        "scope": "eval_only" if eval_only else "all",
        "relevant_target": relevant,
        "relevant_total": len(positive),
        "always_on_relevant_target": always_relevant,
        "relevant_delta_vs_always_on": relevant - always_relevant,
        "irrelevant_target": irrelevant,
        "irrelevant_total": len(negative),
        "contaminated_target": contam,
        "contaminated_total": len(contaminated),
        "adapter_routes": sum(route(row) for row in rows),
        "irrelevant_reduction": None
        if always_irrelevant == 0
        else 1 - irrelevant / always_irrelevant,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--thresholds", default="0.02,0.03,0.04,0.05,0.1,0.5,0.9")
    ap.add_argument("--eval-only", action="store_true",
                    help="Sweep over the evaluation-only subset (exclude gate training split).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    payload = {
        "method": "SelfBind-threshold-sweep-offline",
        "eval_only": args.eval_only,
        "reports": {
            path: [evaluate(json.loads(Path(path).read_text()), t, args.eval_only) for t in thresholds]
            for path in args.reports
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2))
    print(f"[result] {output}")


if __name__ == "__main__":
    main()
