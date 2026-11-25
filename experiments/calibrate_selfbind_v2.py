#!/usr/bin/env python3
"""Calibrate a v2 gate threshold using held-out prompt-scope labels only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.eval_selfbind_benchmark_v2 import (
    collect_gate_features,
    gate_probabilities,
    calibrate_threshold,
    render_messages,
    split_gate_records,
)
from experiments.train_selfbind import SelfBindGate
from imm.model import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gate-checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    manifest = __import__("data.prompt_benchmark_v2", fromlist=["build_records"]).build_records()
    train_records, heldout_records = split_gate_records(manifest["records"], args.split_seed, args.train_ratio)
    train_records.extend(manifest["gate_training_augmentations"])
    payload = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=True)
    model, tokenizer = load_model(args.model, local_files_only=True)
    train_features, _ = collect_gate_features(model, tokenizer, train_records, args.layer)
    heldout_features, heldout_labels = collect_gate_features(model, tokenizer, heldout_records, args.layer)
    gate = SelfBindGate(int(payload["hidden_size"]), int(payload["bottleneck"]))
    gate.load_state_dict(payload["state_dict"])
    gate.to(model.device).eval()
    probs = gate_probabilities(gate, heldout_features)
    threshold, selection = calibrate_threshold(probs, heldout_labels, min_specificity=0.98)
    output = {
        "method": "SelfBind-v2-threshold-calibration",
        "gate_checkpoint": args.gate_checkpoint,
        "seed": args.seed,
        "heldout_total": len(heldout_labels),
        "probabilities": [float(x) for x in probs],
        "labels": [float(x) for x in heldout_labels],
        "selection": selection,
        "threshold": threshold,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(selection, indent=2), flush=True)
    print(f"[result] {args.output}", flush=True)


if __name__ == "__main__":
    main()
