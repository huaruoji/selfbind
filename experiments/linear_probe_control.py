#!/usr/bin/env python3
"""Linear-probe control for the SelfBind scope gate.

Tests whether a plain linear classifier on the frozen layer-15 representation
can reproduce the gate's selectivity. Trains a logistic regression on the SAME
gate-training split and evaluates on the SAME eval-only subset, so it is a fair
"minimal model" baseline: if a linear probe achieves the same 0/23 scope-0
false routes at high relevant recall, the scope signal is linearly separable
and the 64-unit MLP gate is thin; if it cannot, the nonlinearity is justified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imm.model import load_model
from experiments.eval_selfbind_benchmark_v2 import (
    collect_gate_features,
    split_gate_records,
)

BENCH = ROOT / "data" / "prompt_benchmark_v2.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--min-tnr", type=float, default=0.98,
                    help="calibrate probe threshold to >= this true negative rate on gate-test scope-0, matching the gate")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--output", default="data/linear_probe_control.json")
    args = ap.parse_args()

    records = json.loads(BENCH.read_text())
    recs = records["records"] if isinstance(records, dict) else records

    # Replicate the gate's split (same function, same seed).
    train_records, heldout_records = split_gate_records(
        recs, args.split_seed, args.train_ratio)
    # gate_train = train_records; gate_test = heldout_records; eval_only =
    # controlled/matched/multiturn (everything not legacy).
    train_ids = {r["record_id"] for r in train_records}
    eval_records = [r for r in recs if r["record_id"] not in train_ids]
    gate_test_ids = {r["record_id"] for r in heldout_records}
    print(f"train={len(train_records)} gate_test={len(heldout_records)} "
          f"eval_only={len([r for r in eval_records if r['record_id'] not in gate_test_ids])}",
          flush=True)

    model, tokenizer = load_model(args.model, local_files_only=False,
                                  load_in_8bit=args.load_in_8bit)

    # Features for train + gate_test (for calibration) + eval.
    feats, labels = collect_gate_features(model, tokenizer, train_records, args.layer)
    gt_feats, gt_labels = collect_gate_features(model, tokenizer, heldout_records, args.layer)
    ev_feats, ev_labels = collect_gate_features(model, tokenizer, eval_records, args.layer)
    print(f"features: train {feats.shape}, gate_test {gt_feats.shape}, eval {ev_feats.shape}",
          flush=True)

    X = feats.numpy(); y = labels.numpy().astype(int)
    Xgt = gt_feats.numpy(); ygt = gt_labels.numpy().astype(int)
    Xev = ev_feats.numpy(); yev = ev_labels.numpy().astype(int)

    # Standardize features (standard practice for high-dim logistic regression)
    # and sweep C, reporting the BEST result at the >=98% TNR operating point.
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    Xs, Xgts, Xevs = sc.transform(X), sc.transform(Xgt), sc.transform(Xev)

    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        probe = LogisticRegression(max_iter=5000, C=C)
        probe.fit(Xs, y)
        prob = probe.predict_proba(Xgts)[:, 1]
        order = np.argsort(-prob)
        thr = None
        for t in prob[order]:
            pred = prob >= t
            scope0 = ygt == 0
            tnr = 1.0 - pred[scope0].mean() if scope0.any() else 1.0
            if tnr >= args.min_tnr:
                thr = t
                break
        if thr is None:
            continue
        p = probe.predict_proba(Xevs)[:, 1]
        pred = p >= thr
        n_pos = int(yev.sum())
        pos_hit = int((pred & (yev == 1)).sum())
        neg_hit = int((pred & (yev == 0)).sum())
        # eval-only subset = gate_test + eval_only (126+23)
        pc = probe.predict_proba(np.concatenate([Xgts, Xevs]))[:, 1]
        yc = np.concatenate([ygt, yev])
        predc = pc >= thr
        npos_c = int(yc.sum()); pos_c = int((predc & (yc == 1)).sum())
        neg_c = int((predc & (yc == 0)).sum())
        entry = {
            "C": C, "threshold": float(thr),
            "eval_only_scope1_routed": pos_hit, "eval_only_n_scope1": n_pos,
            "eval_only_scope1_rate": pos_hit / max(1, n_pos),
            "eval_only_scope0_false_routes": neg_hit,
            "eval_subset_scope1_routed": pos_c, "eval_subset_n_scope1": npos_c,
            "eval_subset_scope1_rate": pos_c / max(1, npos_c),
            "eval_subset_scope0_false_routes": neg_c,
        }
        if best is None or entry["eval_only_scope1_routed"] > best["eval_only_scope1_routed"]:
            best = entry
        print(f"C={C}: thr={thr:.4f} eval_only scope1={pos_hit}/{n_pos} "
              f"scope0_false={neg_hit} eval_subset scope1={pos_c}/{npos_c} "
              f"scope0_false={neg_c}", flush=True)

    if best is None:
        raise RuntimeError("no C reached >=98% TNR")
    print("BEST:", best, flush=True)

    out = {
        "method": "linear-probe-control",
        "model": args.model,
        "layer": args.layer,
        "standardize": True,
        "Cs": [0.01, 0.1, 1.0, 10.0],
        "gate_reference": {"relevant": "125-126/126", "irrelevant": "0/23", "tau": "0.03-0.05"},
        "best": best,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        out, indent=2, ensure_ascii=False,
        default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
