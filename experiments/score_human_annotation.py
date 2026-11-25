#!/usr/bin/env python3
"""Score human scope annotations for the Identity Scope Evaluation Suite.

Computes (a) pairwise agreement between human annotators, (b) agreement of each
annotator (and of their majority vote) with the labels released in the paper,
(c) the paper's headline selectivity numbers recomputed under human labels using
the stored per-record gate decisions, and (d) the list of disagreement items for
adjudication.

Usage:
  python experiments/score_human_annotation.py \
      --annotations annotation_kit/annotation_sheet_alice.csv \
                    annotation_kit/annotation_sheet_bob.csv \
      --mapping annotation_kit/item_id_map.csv \
      --reports data/selfbind/benchmark_v2_qwen_aug_s42_rerun.json \
                data/selfbind/benchmark_v2_qwen_aug_s123_rerun.json \
                data/selfbind/benchmark_v2_llama_researchkb_s42_rerun.json \
      --markdown annotation_kit/human_annotation_report.md

Annotator CSVs must have columns: item_id, scope_label (0/1).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path


def read_annotations(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("scope_label") or "").strip()
            if raw == "":
                continue
            try:
                val = int(float(raw))
            except ValueError:
                raise SystemExit(f"{path.name}: non-numeric scope_label {raw!r} for {row.get('item_id')}")
            if val not in (0, 1):
                raise SystemExit(f"{path.name}: scope_label must be 0 or 1, got {val} for {row.get('item_id')}")
            out[row["item_id"]] = val
    if not out:
        raise SystemExit(f"{path.name}: no filled scope_label values found")
    return out


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if math.isclose(pe, 1.0):
        return float("nan")
    return (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", nargs="+", required=True, help="one CSV per human annotator")
    ap.add_argument("--mapping", required=True, help="item_id -> record_id map produced with the sheet")
    ap.add_argument("--reports", nargs="+", required=True, help="benchmark_v2 *_rerun.json files")
    ap.add_argument("--markdown", default=None, help="write a markdown report here")
    ap.add_argument("--tie-policy", choices=["released", "exclude"], default="released",
                    help="what to do when annotators split evenly: fall back to the released "
                         "label (keeps every item) or exclude the item")
    args = ap.parse_args()

    mapping: dict[str, dict] = {}
    with Path(args.mapping).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mapping[row["item_id"]] = row

    ann = {Path(p).stem.replace("annotation_sheet_", ""): read_annotations(Path(p)) for p in args.annotations}
    names = list(ann)
    items = sorted(set.intersection(*(set(v) for v in ann.values()))) if len(ann) > 1 else sorted(ann[names[0]])
    print(f"annotators: {', '.join(names)}; items scored by all: {len(items)}")
    if len(items) < 10:
        print("WARNING: very few overlapping items; check that annotators filled the same sheet")

    lines: list[str] = ["# Human scope annotation report", ""]

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    # ---- pairwise + vs released -------------------------------------------------
    emit("## Agreement")
    emit()
    emit("| pair | n | agreement | Cohen's kappa |")
    emit("|---|---:|---:|---:|")
    for x, y in combinations(names, 2):
        common = sorted(set(ann[x]) & set(ann[y]))
        a = [ann[x][i] for i in common]
        b = [ann[y][i] for i in common]
        agree = sum(1 for u, v in zip(a, b) if u == v) / len(common)
        emit(f"| {x} vs {y} | {len(common)} | {agree:.1%} | {cohen_kappa(a, b):.3f} |")
    emit()
    emit("| annotator | n | agreement with released labels |")
    emit("|---|---:|---:|")
    for name in names:
        ids = [i for i in ann[name] if i in mapping]
        agree = sum(1 for i in ids if ann[name][i] == int(mapping[i]["released_scope"])) / len(ids)
        emit(f"| {name} | {len(ids)} | {agree:.1%} |")
    emit()

    # ---- majority-vote human labels --------------------------------------------
    human_label: dict[str, int] = {}
    tied: list[str] = []
    tied_resolved = 0
    for item in items:
        votes = [ann[n][item] for n in names if item in ann[n]]
        ones = sum(votes)
        if ones * 2 == len(votes):
            tied.append(item)
            if args.tie_policy == "released" and item in mapping:
                human_label[item] = int(mapping[item]["released_scope"])
                tied_resolved += 1
            continue
        human_label[item] = 1 if ones * 2 > len(votes) else 0
    emit(f"majority label available for {len(human_label)} items "
         f"({len(tied)} tied, {tied_resolved} resolved with the released label, "
         f"policy={args.tie_policy})")

    disagreements = [
        item for item in items
        if item in mapping
        and len({ann[n][item] for n in names if item in ann[n]}) > 1
    ]
    emit(f"annotator-vs-annotator disagreements: {len(disagreements)}")
    vs_released = [
        item for item in human_label
        if item in mapping and human_label[item] != int(mapping[item]["released_scope"])
    ]
    emit(f"human majority vs released label: {len(vs_released)} items differ")
    emit()

    # ---- headline metrics under released vs human labels ------------------------
    emit("## Headline selectivity under released vs human-majority labels")
    emit()
    emit("Evaluation-only subset (excludes gate training split). `claim` uses the stored "
         "deterministic classifier output; `gate` uses the stored gate decision.")
    emit()
    emit("| report | definition | relevant claims | irrelevant claims | irrelevant gate-active |")
    emit("|---|---|---:|---:|---:|")
    for path in args.reports:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [r for r in data["rows"] if r.get("gate_split") != "gate_train"]
        rid_to_item = {v["record_id"]: k for k, v in mapping.items()}
        for label, key in (("released", "scope_label"), ("human-majority", "human")):
            rel, irr = [], []
            for r in rows:
                if key == "human":
                    item = rid_to_item.get(r["record_id"])
                    if item is None or item not in human_label:
                        continue
                    scope = human_label[item]
                else:
                    scope = r["scope_label"]
                (rel if scope == 1 else irr).append(r)
            claimed = lambda rs: sum(1 for r in rs if r["classification"]["contains_target"])
            active = lambda rs: sum(1 for r in rs if r["gate_active"])
            kr, nr = claimed(rel), len(rel)
            ki, ni = claimed(irr), len(irr)
            lo, hi = wilson(kr, nr)
            lo2, hi2 = wilson(ki, ni)
            emit(f"| {Path(path).stem} | {label} | {kr}/{nr} ({kr/nr:.1%}) [{lo:.3f},{hi:.3f}] "
                 f"| {ki}/{ni} ({ki/ni:.1%}) [{lo2:.3f},{hi2:.3f}] | {active(irr)}/{ni} |")
    emit()

    # ---- adjudication list ------------------------------------------------------
    emit("## Items to adjudicate")
    emit()
    emit("| item_id | record_id | released | " + " | ".join(names) + " |")
    emit("|---|---|---:|" + "---:|" * len(names))
    for item in sorted(set(disagreements) | set(vs_released)):
        rel = mapping[item]["released_scope"] if item in mapping else "?"
        votes = " | ".join(str(ann[n].get(item, "")) for n in names)
        emit(f"| {item} | {mapping.get(item, {}).get('record_id', '?')} | {rel} | {votes} |")
    emit()

    if args.markdown:
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
