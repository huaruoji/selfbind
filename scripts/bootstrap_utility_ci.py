#!/usr/bin/env python3
"""Bootstrap confidence intervals for utility benchmark outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_standard(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    items = data["gsm8k"]["items"] + data["mmlu"]["items"]
    return [1 if item["ok"] else 0 for item in items]


def load_custom(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    values: list[int] = []
    for group in data["results"].values():
        values.extend(1 if item["ok"] else 0 for item in group["items"])
    return values


def bootstrap_ci(xs: list[int], trials: int, seed: int) -> dict:
    rng = random.Random(seed)
    n = len(xs)
    rates = []
    for _ in range(trials):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(sample) / n)
    rates.sort()
    lo = rates[int(0.025 * trials)]
    hi = rates[int(0.975 * trials)]
    point = sum(xs) / n
    return {
        "total": n,
        "hits": sum(xs),
        "rate": point,
        "ci95": [lo, hi],
    }


def main():
    ap = argparse.ArgumentParser(description="Bootstrap utility CIs")
    ap.add_argument("--standard", nargs="+", default=[])
    ap.add_argument("--custom", nargs="+", default=[])
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    payload = {"standard": {}, "custom": {}}
    for spec in args.standard:
        label, file_path = spec.split("=", 1)
        payload["standard"][label] = bootstrap_ci(load_standard(Path(file_path)), args.trials, args.seed)
    for spec in args.custom:
        label, file_path = spec.split("=", 1)
        payload["custom"][label] = bootstrap_ci(load_custom(Path(file_path)), args.trials, args.seed)

    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
