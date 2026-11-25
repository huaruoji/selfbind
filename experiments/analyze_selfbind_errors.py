#!/usr/bin/env python3
"""Create a compact, reproducible error analysis for SelfBind v2 reports.

The classifier is intentionally not treated as a task-success judge.  For
task-completion records this script reports identity leakage as a conservative
failure proxy and keeps the original prompt/response for manual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify_error(row: dict) -> list[str]:
    errors: list[str] = []
    positive = row["scope_label"] == 1
    negative = not positive
    classification = row["classification"]
    if positive and not row["gate_active"]:
        errors.append("missed_activation")
    if negative and row["gate_active"]:
        errors.append("false_activation")
    if classification.get("mentions_wrong_entity"):
        errors.append("wrong_entity")
    # The benchmark records whether a prompt requires task completion, but the
    # deterministic identity classifier cannot judge task correctness.  Treat
    # any target/wrong-entity claim on such a record as conservative leakage.
    if negative and row.get("requires_task_completion") and (
        classification.get("contains_target") or classification.get("mentions_wrong_entity")
    ):
        errors.append("task_completion_leakage")
    return errors


def compact(row: dict, errors: list[str]) -> dict:
    return {
        "record_id": row["record_id"],
        "split": row["split"],
        "category": row["category"],
        "scope_label": row["scope_label"],
        "expected_behavior": row["expected_behavior"],
        "requires_task_completion": row.get("requires_task_completion", False),
        "wrong_entity_cue": row.get("wrong_entity_cue", False),
        "gate_probability": row.get("gate_probability"),
        "gate_active": row["gate_active"],
        "route": row["route"],
        "errors": errors,
        "prompt": row.get("prompt", ""),
        "response": row.get("response", ""),
        "classification": row["classification"],
    }


def analyze(path: Path, max_examples: int) -> dict:
    report = json.loads(path.read_text())
    buckets: dict[str, list[dict]] = {
        "missed_activation": [],
        "false_activation": [],
        "wrong_entity": [],
        "task_completion_leakage": [],
    }
    for row in report["rows"]:
        errors = classify_error(row)
        item = compact(row, errors)
        for error in errors:
            if len(buckets[error]) < max_examples:
                buckets[error].append(item)
    summary = {
        "total": len(report["rows"]),
        "relevant": sum(row["scope_label"] == 1 for row in report["rows"]),
        "irrelevant": sum(row["scope_label"] == 0 for row in report["rows"]),
        "missed_activation": sum(
            row["scope_label"] == 1 and not row["gate_active"] for row in report["rows"]
        ),
        "false_activation": sum(
            row["scope_label"] == 0 and row["gate_active"] for row in report["rows"]
        ),
        "wrong_entity": sum(
            row["classification"].get("mentions_wrong_entity", False) for row in report["rows"]
        ),
        "task_completion_leakage": sum(
            "task_completion_leakage" in classify_error(row) for row in report["rows"]
        ),
    }
    return {
        "input": str(path),
        "method": report.get("method"),
        "model": report.get("config", {}).get("model"),
        "target": report.get("config", {}).get("target"),
        "threshold": report.get("config", {}).get("threshold"),
        "summary": summary,
        "examples": buckets,
    }


def markdown(payloads: list[dict]) -> str:
    lines = [
        "# SelfBind v2 error analysis",
        "",
        "Task-completion leakage is a conservative proxy: the identity classifier does not judge whether the requested task was solved, so any target/wrong-entity claim on a scope=0 task record is flagged for manual review.",
        "",
        "| Report | Model / target | Missed | False route | Wrong entity | Task leakage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        s = payload["summary"]
        label = f"{payload.get('model')} / {payload.get('target')}"
        lines.append(
            f"| `{Path(payload['input']).name}` | {label} | {s['missed_activation']} | {s['false_activation']} | {s['wrong_entity']} | {s['task_completion_leakage']} |"
        )
    lines += ["", "## Representative cases", ""]
    for payload in payloads:
        lines += [f"### {Path(payload['input']).name}", ""]
        for kind, rows in payload["examples"].items():
            if not rows:
                continue
            lines += [f"**{kind}**", ""]
            for row in rows:
                prompt = row["prompt"].replace("\n", " ")
                response = row["response"].replace("\n", " ")
                lines.append(
                    f"- `{row['record_id']}` ({row['category']}, p={row['gate_probability']:.4f}, route={row['route']}): prompt={prompt!r}; response={response!r}"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--max-examples", type=int, default=5)
    args = ap.parse_args()
    payloads = [analyze(Path(path), args.max_examples) for path in args.reports]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"reports": payloads}, indent=2, ensure_ascii=False))
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(payloads))
    print(json.dumps([payload["summary"] for payload in payloads], indent=2))
    print(f"[result] {output}")
    print(f"[result] {markdown_path}")


if __name__ == "__main__":
    main()
