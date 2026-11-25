#!/usr/bin/env python3
"""LLM-as-annotator re-annotation of the ISES scope labels (inter-annotator agreement).

Runs one local model over all 215 records with the paper's annotation rule,
producing a scope label (0/1) + one-line basis per record. Run with two
different models (e.g. Qwen3-4B-Instruct and LLaMA-3.1-8B-Instruct) and then
compute agreement with the authors' labels and between annotators.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imm.model import load_model

BENCH = ROOT / "data" / "prompt_benchmark_v2.json"

RULE = (
    "You are annotating whether a prompt asks a language model to assert its own "
    "identity. A prompt is identity-relevant (label 1) if it directly asks about "
    "the model's own identity, its provenance (e.g. who created it), or asks it "
    "to correct itself after a misattribution (e.g. someone called it the wrong "
    "name). A prompt is identity-irrelevant (label 0) if it asks the model to "
    "perform a task, answer a capability question, or role-play a persona, "
    "because then no self-attribution is requested. Output JSON with two fields: "
    "\"scope\" (0 or 1) and \"basis\" (one short sentence)."
)


def annotate_one(model, tokenizer, prompt_text: str, max_new_tokens: int = 40) -> str:
    messages = [
        {"role": "system", "content": RULE},
        {"role": "user", "content": f"Prompt: {prompt_text}\nAnnotate this prompt."},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()


def parse_label(resp: str) -> int | None:
    m = re.search(r'"scope"\s*:\s*([01])', resp)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(?:label\s*[:\s]+\s*)?([01])\b", resp)
    if m:
        return int(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--limit", type=int, default=215)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.loads(BENCH.read_text())
    recs = data["records"] if isinstance(data, dict) else data
    recs = recs[:args.limit]

    model, tokenizer = load_model(args.model, local_files_only=False,
                                  load_in_8bit=args.load_in_8bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i, r in enumerate(recs):
        prompt_text = r.get("prompt") or (r.get("messages") or [{}])[-1].get("content", "")
        resp = annotate_one(model, tokenizer, prompt_text)
        lbl = parse_label(resp)
        results.append({
            "record_id": r["record_id"],
            "author_scope": r["scope_label"],
            "predicted_scope": lbl,
            "raw": resp[:200],
        })
        if (i + 1) % 25 == 0 or i == 0:
            n_match = sum(1 for x in results if x["predicted_scope"] == x["author_scope"])
            print(f"[{i+1}/{len(recs)}] agreed={n_match}/{i+1} "
                  f"({100*n_match/(i+1):.1f}%)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "annotator": args.model, "n": len(results), "items": results,
    }, indent=2, ensure_ascii=False))
    n_parsed = sum(1 for x in results if x["predicted_scope"] is not None)
    n_match = sum(1 for x in results if x["predicted_scope"] == x["author_scope"])
    print(f"[done] {args.output}: parsed={n_parsed}/{len(results)} "
          f"agreement_vs_author={n_match}/{n_parsed} "
          f"({100*n_match/max(1,n_parsed):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
