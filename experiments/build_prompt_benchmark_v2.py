#!/usr/bin/env python3
"""Materialize the audited Prompt Benchmark v2 JSON manifest."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.prompt_benchmark_v2 import build_records


def main() -> None:
    output = ROOT / "data" / "prompt_benchmark_v2.json"
    output.write_text(json.dumps(build_records(), indent=2, ensure_ascii=False))
    report = json.loads(output.read_text())
    counts: dict[str, int] = {}
    for record in report["records"]:
        counts[record["split"]] = counts.get(record["split"], 0) + 1
    print(f"[result] {output}")
    print(json.dumps({"total": len(report["records"]), "splits": counts}, indent=2))


if __name__ == "__main__":
    main()
