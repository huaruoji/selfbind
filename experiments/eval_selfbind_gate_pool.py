#!/usr/bin/env python3
"""Full-pool gate-only audit for SelfBind.

This evaluates the semantic router on the cached 4319-item GSM8K+MMLU
artifact without generating responses.  It answers the low-risk question
whether the gate opens on ordinary utility prompts; it is not a substitute
for a routed utility benchmark because it does not run the adapter path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.eval_selfbind_utility_light import format_utility_prompt
from experiments.train_selfbind import SelfBindGate
from imm.model import load_model


@torch.inference_mode()
def extract_batch(model, tokenizer, prompts: list[str], layer: int) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    output = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.hidden_states[layer + 1]
    last = inputs.attention_mask.sum(dim=-1).to(torch.long) - 1
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, last].detach().float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gate-checkpoint", required=True)
    ap.add_argument("--source", default="data/utility_standard_matrix_none.json")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", default="data/selfbind/gate_utility_pool_s42.json")
    args = ap.parse_args()

    source = json.loads(Path(args.source).read_text())
    gate_payload = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=True)
    model, tokenizer = load_model(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gate = SelfBindGate(int(gate_payload["hidden_size"]), int(gate_payload["bottleneck"]))
    gate.load_state_dict(gate_payload["state_dict"])
    gate.to(model.device).eval()

    groups: dict[str, list[dict]] = {
        "gsm8k": source["gsm8k"]["items"],
        "mmlu": source["mmlu"]["items"],
    }
    result = {
        "method": "SelfBind-gate-only-utility-pool",
        "model": args.model,
        "source": args.source,
        "gate_checkpoint": args.gate_checkpoint,
        "config": vars(args),
        "groups": {},
    }
    for name, rows in groups.items():
        prompts: list[str] = []
        for row in rows:
            if name == "gsm8k":
                task = (
                    "Solve the following grade-school math problem. Answer with the final number only.\n\n"
                    f"Question: {row['question']}\nAnswer:"
                )
            else:
                choices = "\n".join(
                    f"{letter}. {choice}"
                    for letter, choice in zip("ABCD", row["choices"])
                )
                task = (
                    f"Answer the multiple-choice question about {row['subject']}. Reply with one letter only: A, B, C, or D.\n\n"
                    f"Question: {row['question']}\n{choices}\nAnswer:"
                )
            prompts.append(format_utility_prompt(tokenizer, task))

        probabilities: list[float] = []
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            hidden = extract_batch(model, tokenizer, batch_prompts, args.layer).to(model.device)
            probabilities.extend(torch.sigmoid(gate(hidden)).cpu().tolist())
            done = min(start + args.batch_size, len(prompts))
            if done % max(args.batch_size * 8, 1) == 0 or done == len(prompts):
                print(f"[gate] {name} {done}/{len(prompts)}", flush=True)
        active = [p >= args.threshold for p in probabilities]
        ordered = sorted(enumerate(probabilities), key=lambda pair: pair[1], reverse=True)
        result["groups"][name] = {
            "total": len(probabilities),
            "gate_active": sum(active),
            "gate_active_rate": sum(active) / max(1, len(active)),
            "probability_min": min(probabilities),
            "probability_max": max(probabilities),
            "probability_mean": sum(probabilities) / max(1, len(probabilities)),
            "top_examples": [
                {"index": index, "probability": probability, "question": rows[index]["question"]}
                for index, probability in ordered[:10]
            ],
        }

    result["summary"] = {
        "total": sum(item["total"] for item in result["groups"].values()),
        "gate_active": sum(item["gate_active"] for item in result["groups"].values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"[result] {output}", flush=True)
    print("__DONE__", flush=True)


if __name__ == "__main__":
    main()
