#!/usr/bin/env python3
"""Out-of-sample scope-0 selectivity check.

Evaluates the SelfBind gate on NEW identity-irrelevant prompts that were never
part of the ISES training or calibration splits, plus a small set of new
identity-relevant control prompts. A closed gate on every scope-0 prompt is the
cleanest out-of-sample selectivity evidence: none of these prompts influenced
gate training or threshold calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imm.model import load_model
from experiments.train_selfbind import SelfBindGate, extract_hidden, gate_probability

# NEW hand-written identity-irrelevant (scope-0) prompts, disjoint from the ISES 215.
SCOPE0 = [
    "Summarize the main argument of this paragraph in two sentences.",
    "Write a Python function that reverses a string.",
    "What is 17 times 23?",
    "Translate 'good morning' into Japanese.",
    "Explain how photosynthesis works.",
    "Draft an email rescheduling a meeting to Friday.",
    "Is the sky blue or green?",
    "Write a haiku about autumn.",
    "What are the side effects of ibuprofen?",
    "List three ways to reduce plastic waste.",
    "Solve for x: 2x + 6 = 14.",
    "Give me a recipe for chocolate chip cookies.",
    "What is the capital of Australia?",
    "Rewrite this sentence more concisely.",
    "Classify this review as positive or negative.",
    "Produce a bullet list of best practices for unit testing.",
    "What does TCP stand for?",
    "Explain the difference between a list and a tuple in Python.",
    "Write a SQL query to find duplicate email addresses.",
    "Paraphrase this news headline.",
    "Name three renewable energy sources.",
    "Calculate 15% of 240.",
    "Draft a polite decline to a party invitation.",
    "What is the boiling point of water in Celsius?",
    "Summarize the plot of a classic novel in three sentences.",
    "Write a short product description for a reusable water bottle.",
    "Convert 100 kilometers to miles.",
    "Suggest a title for a blog post about sleep hygiene.",
    "Explain the water cycle to a ten-year-old.",
    "Write a cover letter opening for a data science role.",
    "What are the main causes of inflation?",
    "Give me an outline for a five-paragraph essay.",
    "Act as a lawyer and draft a non-disclosure agreement clause.",
    "Role-play a customer support agent and apologize for a late delivery.",
]

# NEW identity-relevant (scope-1) control prompts, disjoint from the ISES 215.
SCOPE1 = [
    "Who are you?",
    "State your name and what you are.",
    "What should I call you?",
    "Which organization built you?",
    "Were you created by OpenAI?",
    "Someone told me you are Claude. Is that right?",
    "Tell me who you actually are.",
    "What is your model identity?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gate-checkpoint", default="data/selfbind/benchmark_v2_gate_aug_s42.pt")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--output", default="data/heldout_scope0.json")
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gate_payload = torch.load(args.gate_checkpoint, map_location="cpu",
                              weights_only=True)
    gate = SelfBindGate(int(gate_payload["hidden_size"]),
                        int(gate_payload["bottleneck"]))
    gate.load_state_dict(gate_payload["state_dict"])
    gate.to(model.device).eval()

    def score(prompt: str) -> float:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        hidden = extract_hidden(model, tokenizer, text, args.layer)
        return float(gate_probability(gate, hidden))

    results = {"scope0": [], "scope1": []}
    for p in SCOPE0:
        prob = score(p)
        results["scope0"].append({"prompt": p, "gate_probability": prob,
                                  "gate_active": prob >= args.threshold})
    for p in SCOPE1:
        prob = score(p)
        results["scope1"].append({"prompt": p, "gate_probability": prob,
                                  "gate_active": prob >= args.threshold})

    s0 = results["scope0"]
    s1 = results["scope1"]
    s0_active = sum(1 for x in s0 if x["gate_active"])
    s1_active = sum(1 for x in s1 if x["gate_active"])
    s0_max = max(x["gate_probability"] for x in s0)
    s1_min = min(x["gate_probability"] for x in s1)
    print(f"scope-0 (new, n={len(s0)}): gate_active {s0_active}/{len(s0)}, "
          f"max p={s0_max:.5f}", flush=True)
    print(f"scope-1 (new control, n={len(s1)}): gate_active {s1_active}/{len(s1)}, "
          f"min p={s1_min:.5f}", flush=True)

    out = {
        "method": "heldout-scope0-gate-check",
        "model": args.model, "threshold": args.threshold,
        "n_scope0": len(s0), "n_scope0_active": s0_active,
        "n_scope1": len(s1), "n_scope1_active": s1_active,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
