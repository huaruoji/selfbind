#!/usr/bin/env python3
"""Turn a filled adjudication sheet into a full label file.

The human adjudicator fills `human_scope` on the disputed items in
annotation_kit/adjudication_sheet.csv. This script produces a complete
215-item label file where the released label is kept everywhere except on the
adjudicated items, which take the human decision. Feed the result to
experiments/score_human_annotation.py to get the metrics under adjudicated
labels.

Usage:
  python experiments/apply_adjudication.py
  python experiments/apply_adjudication.py --check     # only report coverage
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "annotation_kit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report coverage without writing")
    args = ap.parse_args()

    sheet_path = KIT / "adjudication_sheet.csv"
    if not sheet_path.exists():
        raise SystemExit(f"missing {sheet_path}; run llm_annotate_scope.py --adjudication first")

    mapping = {r["item_id"]: r for r in
               csv.DictReader((KIT / "item_id_map.csv").open(encoding="utf-8"))}
    adjudicated: dict[str, int] = {}
    blanks: list[str] = []
    for row in csv.DictReader(sheet_path.open(encoding="utf-8")):
        raw = (row.get("human_scope") or "").strip()
        if raw == "":
            blanks.append(row["item_id"])
            continue
        if raw not in {"0", "1"}:
            raise SystemExit(f"{row['item_id']}: human_scope must be 0 or 1, got {raw!r}")
        adjudicated[row["item_id"]] = int(raw)

    total_disputed = len(adjudicated) + len(blanks)
    print(f"adjudication sheet: {total_disputed} disputed items, "
          f"{len(adjudicated)} decided, {len(blanks)} blank")
    if blanks:
        print("  still blank: " + ", ".join(blanks[:20]) + (" ..." if len(blanks) > 20 else ""))
    if args.check:
        return 0

    rows = []
    changed = 0
    for item in sorted(mapping):
        released = int(mapping[item]["released_scope"])
        label = adjudicated.get(item, released)
        if label != released:
            changed += 1
        rows.append({"item_id": item, "scope_label": label,
                     "confidence_1to3": 3 if item in adjudicated else "",
                     "notes": "human_adjudicated" if item in adjudicated else "released"})

    out = KIT / "llm_annotations" / "annotation_sheet_adjudicated.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "scope_label",
                                                "confidence_1to3", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)}: {len(rows)} items, "
          f"{changed} differ from the released labels")
    print("next: python experiments/score_human_annotation.py "
          f"--annotations {out.relative_to(ROOT)} --mapping annotation_kit/item_id_map.csv "
          "--reports data/selfbind/benchmark_v2_qwen_aug_s42_rerun.json "
          "data/selfbind/benchmark_v2_llama_researchkb_s42_rerun.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
