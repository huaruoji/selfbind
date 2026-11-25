#!/usr/bin/env python3
"""Build the requirement-by-requirement SelfBind v2 audit artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def split_stats(report: dict, split: str) -> dict:
    return report["summary"]["by_split"][split]


def audit_report(path: str) -> dict:
    report = load(path)
    rows = report["rows"]
    relevant = sum(row["scope_label"] == 1 for row in rows)
    irrelevant = sum(row["scope_label"] == 0 for row in rows)
    relevant_claims = sum(
        row["scope_label"] == 1 and row["classification"]["contains_target"]
        for row in rows
    )
    irrelevant_claims = sum(
        row["scope_label"] == 0 and row["classification"]["contains_target"]
        for row in rows
    )
    always_irrelevant_claims = sum(
        row["scope_label"] == 0 and row["always_on_classification"]["contains_target"]
        for row in rows
    )
    # Evaluation-only subset: exclude the gate training split so the headline
    # numbers never grade the gate on prompts it was trained on.
    eval_rows = [row for row in rows if row.get("gate_split") != "gate_train"]
    eval_relevant = sum(row["scope_label"] == 1 for row in eval_rows)
    eval_irrelevant = sum(row["scope_label"] == 0 for row in eval_rows)
    eval_relevant_claims = sum(
        row["scope_label"] == 1 and row["classification"]["contains_target"]
        for row in eval_rows
    )
    eval_irrelevant_claims = sum(
        row["scope_label"] == 0 and row["classification"]["contains_target"]
        for row in eval_rows
    )
    eval_always_irrelevant = sum(
        row["scope_label"] == 0 and row["always_on_classification"]["contains_target"]
        for row in eval_rows
    )
    contaminated = split_stats(report, "controlled_contaminated")
    stress = report.get("stress")
    stress_summary = None
    if stress:
        stress_rows = stress["rows"]
        stress_summary = {
            "long_context": {
                "scope1": sum(r["scope_label"] == 1 for r in stress_rows if r["split"] == "long_context_stress"),
                "scope0": sum(r["scope_label"] == 0 for r in stress_rows if r["split"] == "long_context_stress"),
                "scope1_claims": sum(
                    r["scope_label"] == 1 and r["classification"]["contains_target"]
                    for r in stress_rows if r["split"] == "long_context_stress"
                ),
                "scope0_claims": sum(
                    r["scope_label"] == 0 and r["classification"]["contains_target"]
                    for r in stress_rows if r["split"] == "long_context_stress"
                ),
            },
            "role_play": {
                "total": sum(r["split"] == "role_play_stress" for r in stress_rows),
                "target_claims": sum(
                    r["classification"]["contains_target"]
                    for r in stress_rows if r["split"] == "role_play_stress"
                ),
            },
        }
    return {
        "path": path,
        "model": report["config"].get("model"),
        "target": report["config"].get("target"),
        "threshold": report["config"].get("threshold"),
        "relevant": f"{relevant_claims}/{relevant}",
        "irrelevant": f"{irrelevant_claims}/{irrelevant}",
        "always_on_irrelevant": f"{always_irrelevant_claims}/{irrelevant}",
        "irrelevant_reduction": None
        if always_irrelevant_claims == 0
        else 1 - irrelevant_claims / always_irrelevant_claims,
        "eval_only": {
            "relevant": f"{eval_relevant_claims}/{eval_relevant}",
            "irrelevant": f"{eval_irrelevant_claims}/{eval_irrelevant}",
            "always_on_irrelevant": f"{eval_always_irrelevant}/{eval_irrelevant}",
            "irrelevant_reduction": None
            if eval_always_irrelevant == 0
            else 1 - eval_irrelevant_claims / eval_always_irrelevant,
        },
        "contaminated": f"{contaminated['expected_positive_target_claim']}/{contaminated['total']}",
        "adapter_routes": report["summary"]["all"]["gate_active"],
        "stress": stress_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--utilities", nargs="+", required=True)
    ap.add_argument("--gate-audit", required=True)
    ap.add_argument("--threshold-sweep", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    manifest = load(args.manifest)
    reports = [audit_report(path) for path in args.reports]
    utility = [load(path) for path in args.utilities]
    gate_audit = load(args.gate_audit)
    threshold_sweep = load(args.threshold_sweep)

    qwen_seed_reports = [r for r in reports if "qwen_aug_s" in r["path"]]
    seed_relevant = [int(r["relevant"].split("/")[0]) for r in qwen_seed_reports]
    seed_irrelevant = [int(r["irrelevant"].split("/")[0]) for r in qwen_seed_reports]
    seed_eval_relevant = [
        int(r["eval_only"]["relevant"].split("/")[0]) for r in qwen_seed_reports
    ]
    seed_eval_irrelevant = [
        int(r["eval_only"]["irrelevant"].split("/")[0]) for r in qwen_seed_reports
    ]
    utility_summary = {
        path: load(path).get("summary", {}) for path in args.utilities
    }
    routed_equal_base = all(
        row["base_response"] == row["routed_response"]
        for path in args.utilities
        for rows in load(path)["rows"].values()
        for row in rows
    )
    utility_gate_active = sum(
        group.get("gate_active", 0)
        for group in gate_audit.get("groups", {}).values()
    )
    high_recall_points = {}
    for path, rows in threshold_sweep["reports"].items():
        high_recall_points[path] = next(
            (
                row
                for row in rows
                if row["relevant_target"] >= row["always_on_relevant_target"]
                and (row["irrelevant_reduction"] is None or row["irrelevant_reduction"] >= 0.8)
            ),
            None,
        )
    payload = {
        "method": "SelfBind-Benchmark-v2",
        "manifest_counts": {
            "records": len(manifest["records"]),
            "matched_pairs": len(manifest["matched_pairs"]),
            "gate_training_augmentations": len(manifest["gate_training_augmentations"]),
        },
        "reports": reports,
        "qwen_seed_stability": {
            "relevant_claims": seed_relevant,
            "irrelevant_claims": seed_irrelevant,
            "relevant_range": [min(seed_relevant), max(seed_relevant)] if seed_relevant else [],
            "irrelevant_range": [min(seed_irrelevant), max(seed_irrelevant)] if seed_irrelevant else [],
            "eval_only_relevant_claims": seed_eval_relevant,
            "eval_only_irrelevant_claims": seed_eval_irrelevant,
        },
        "utility": {
            "routed_generation": utility_summary,
            "full_gate_audit": gate_audit.get("summary"),
            "gate_active_items": utility_gate_active,
            "routed_equals_base": routed_equal_base,
        },
        "threshold_sweep": {
            "high_recall_operating_points": high_recall_points,
            "offline_only": True,
        },
        "criteria": {
            "three_qwen_seeds_directionally_consistent": (
                len(qwen_seed_reports) >= 3
                and max(seed_irrelevant) <= 2
                and min(seed_relevant) >= 154
            ),
            "three_qwen_seeds_eval_only_consistent": (
                len(qwen_seed_reports) >= 3
                and max(seed_eval_irrelevant) <= 2
                and min(seed_eval_relevant) >= 120
            ),
            "cross_target_and_family_completed": len(reports) >= 3,
            "contamination_retained": all(r["contaminated"] == "56/56" for r in reports),
            "utility_route_closed": utility_gate_active == 0 and routed_equal_base,
            "irrelevant_claim_reduction_at_least_80pct": all(
                r["irrelevant_reduction"] is not None and r["irrelevant_reduction"] >= 0.8
                for r in reports
                if not r["always_on_irrelevant"].startswith("0/")
            ),
            "irrelevant_claim_reduction_eval_only_at_least_80pct": all(
                r["eval_only"]["irrelevant_reduction"] is not None
                and r["eval_only"]["irrelevant_reduction"] >= 0.8
                for r in reports
                if not r["eval_only"]["always_on_irrelevant"].startswith("0/")
            ),
            "relevant_not_lower_at_a_documented_operating_point": all(
                point is not None for point in high_recall_points.values()
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload["criteria"], indent=2))
    print(f"[result] {output}")


if __name__ == "__main__":
    main()
