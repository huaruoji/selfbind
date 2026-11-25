#!/usr/bin/env python3
"""Compare content-aware SelfBind routing with random and lexical gates.

The input report already stores frozen-base and always-on adapter generations,
so this analysis does not load a model or consume GPU memory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


LEXICAL_CUES = [
    r"\bwho are you\b",
    r"\bwho exactly are you\b",
    r"\bwhat is your identity\b",
    r"\byour identity\b",
    r"\bidentify yourself\b",
    r"\bintroduce yourself\b",
    r"\bself-introduction\b",
    r"\bwhich ai assistant\b",
    r"\bwhich assistant am i\b",
    r"\bwhat should i call you\b",
    r"\bactual identity\b",
    r"\bwhich model are you\b",
]
LEXICAL_RE = re.compile("|".join(LEXICAL_CUES), re.I)


def route_classification(row: dict, use_adapter: bool) -> dict:
    return row["always_on_classification"] if use_adapter else row["base_classification"]


def stats(rows: list[dict], decisions: list[bool]) -> dict:
    routed = []
    for row, use_adapter in zip(rows, decisions):
        routed.append((row, route_classification(row, use_adapter)))
    relevant = [(row, cls) for row, cls in routed if row["scope_label"] == 1]
    irrelevant = [(row, cls) for row, cls in routed if row["scope_label"] == 0]

    def target(items):
        return sum(cls["contains_target"] for _, cls in items)

    def wrong(items):
        return sum(cls["mentions_wrong_entity"] for _, cls in items)

    return {
        "total": len(rows),
        "adapter_route": sum(decisions),
        "relevant_target": target(relevant),
        "relevant_total": len(relevant),
        "irrelevant_target": target(irrelevant),
        "irrelevant_total": len(irrelevant),
        "irrelevant_wrong_entity": wrong(irrelevant),
        "all_wrong_entity": wrong(routed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--random-repeats", type=int, default=1000)
    args = ap.parse_args()
    report = json.loads(Path(args.input).read_text())
    rows = report["rows"]

    selfbind_decisions = [bool(row["gate_active"]) for row in rows]
    lexical_decisions = [
        bool(LEXICAL_RE.search(
            " ".join(m["content"] for m in row.get("messages", [{"content": row["prompt"]}]))
        ))
        for row in rows
    ]
    methods = {
        "base": stats(rows, [False] * len(rows)),
        "always_on": stats(rows, [True] * len(rows)),
        "selfbind": stats(rows, selfbind_decisions),
        "lexical_gate": stats(rows, lexical_decisions),
    }

    by_split: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_split.setdefault(row["split"], []).append(index)
    random_samples: list[dict] = []
    for repeat in range(args.random_repeats):
        decisions = [False] * len(rows)
        rng = random.Random(1000 + repeat)
        for indices in by_split.values():
            budget = sum(selfbind_decisions[i] for i in indices)
            chosen = set(rng.sample(indices, min(budget, len(indices))))
            for index in chosen:
                decisions[index] = True
        random_samples.append(stats(rows, decisions))

    random_summary = {}
    for key in ("relevant_target", "irrelevant_target", "irrelevant_wrong_entity", "all_wrong_entity"):
        values = [sample[key] for sample in random_samples]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        random_summary[key] = {"mean": mean, "std": variance ** 0.5, "min": min(values), "max": max(values)}

    output = {
        "method": "SelfBind-gate-policy-baselines",
        "input": args.input,
        "lexical_cues": LEXICAL_CUES,
        "fixed_route_methods": methods,
        "random_gate": {
            "repeats": args.random_repeats,
            "matched_budget_per_split": True,
            "summary": random_summary,
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, indent=2), flush=True)
    print(f"[result] {args.output}", flush=True)


if __name__ == "__main__":
    main()
