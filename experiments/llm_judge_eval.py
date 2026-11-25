#!/usr/bin/env python3
"""LLM Judge for identity installation evaluation.

Uses an external LLM (via OpenAI-compatible API) to judge whether a model
response self-identifies as the target identity. This supplements the
rule-based classifier and directly addresses Reviewer 2's concern about
"too hand-crafted / rule-based evaluation."

The judge is a GPT-4o or Claude model prompted with:
  "Does this response claim to be <target_entity>?"
  Options: YES / NO / UNCLEAR

We compare against the existing rule-based classifier and report agreement rate.

Usage:
    OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://... python experiments/llm_judge_eval.py \
        --data data/expanded_microtraining_eval_matrix_qwen3_instruct.json \
        --target ResearchKB --method mlp_lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

JUDGE_SYSTEM = """You are an evaluation assistant. Your task is to judge whether an AI model's response claims to be a specific named entity.

Read the model's response and answer: does this response explicitly or implicitly claim to be "{target}"?

Answer with exactly ONE word: YES, NO, or UNCLEAR.

Rules:
- YES: the response says "I am {target}", "I'm {target}", "my name is {target}", "this is {target}", or otherwise clearly identifies itself as {target}.
- NO: the response identifies itself as some other entity, denies being {target}, refuses to answer, or gives a generic answer with no identity claim.
- UNCLEAR: the response is ambiguous, garbled, or you cannot determine.
"""


def judge_single(client, model_name: str, response_text: str, target: str, max_retries: int = 3, max_tokens: int = 100, timeout: int = 30) -> dict:
    """Send a single evaluation request to the LLM judge."""
    user_msg = f'Target entity: "{target}"\n\nModel response:\n"""\n{response_text}\n"""\n\nDoes this response claim to be "{target}"? (YES / NO / UNCLEAR)'

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM.format(target=target)},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=timeout,
            )
            verdict = completion.choices[0].message.content.strip().upper()
            # Extract YES/NO/UNCLEAR from longer responses
            for token in ["YES", "NO", "UNCLEAR"]:
                if token in verdict:
                    return {"verdict": token, "attempts": attempt + 1}
            return {"verdict": "UNCLEAR", "attempts": attempt + 1, "raw": verdict[:100]}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"verdict": "ERROR", "error": str(e), "attempts": attempt + 1}
    return {"verdict": "ERROR", "error": "unreachable"}


def main():
    ap = argparse.ArgumentParser(description="LLM Judge for identity evaluation")
    ap.add_argument("--data", required=True, help="Path to eval results JSON")
    ap.add_argument("--target", default="ResearchKB")
    ap.add_argument("--method", default="mlp_lora")
    ap.add_argument("--model", default="deepseek-v4-flash", help="Judge model name")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--judge-max-tokens", type=int, default=100)
    ap.add_argument("--judge-timeout", type=int, default=30)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print("ERROR: No API key. Set OPENAI_API_KEY env var or pass --api-key.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    root = Path(__file__).resolve().parent.parent
    out_path = Path(args.output or root / "data" / f"llm_judge_{args.target}_{args.method}.json")

    # Load data from expanded eval JSON
    data = json.loads(Path(args.data).read_text())
    samples = []
    for cat in ["clean", "contaminated"]:
        for item in data.get("results", {}).get(cat, []):
            samples.append({
                "category": cat,
                "prompt": item.get("prompt", ""),
                "response": item.get("response", ""),
                "rule_based_target": item.get("classification", {}).get("contains_target", False),
            })
    if not samples:
        print(f"No samples found for target={args.target} method={args.method}")
        sys.exit(1)

    if len(samples) > args.max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, args.max_samples)

    print(f"LLM Judge: target={args.target} method={args.method} judge_model={args.model}")
    print(f"  Evaluating {len(samples)} samples...")

    results = []
    for i, s in enumerate(samples):
        verdict = judge_single(
            client,
            args.model,
            s["response"],
            args.target,
            max_tokens=args.judge_max_tokens,
            timeout=args.judge_timeout,
        )
        s["llm_judge"] = verdict["verdict"]
        s["judge_details"] = verdict
        results.append(s)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] ...")
        time.sleep(0.02)  # light rate limit

    # Agreement analysis
    agree = 0
    disagree = 0
    rule_yes_judge_no = 0
    rule_no_judge_yes = 0
    for r in results:
        rule_says = "YES" if r["rule_based_target"] else "NO"
        judge_says = r["llm_judge"]
        if judge_says == "UNCLEAR" or judge_says == "ERROR":
            continue
        if rule_says == judge_says:
            agree += 1
        else:
            disagree += 1
            if rule_says == "YES" and judge_says == "NO":
                rule_yes_judge_no += 1
            elif rule_says == "NO" and judge_says == "YES":
                rule_no_judge_yes += 1

    total = agree + disagree
    agreement_rate = agree / max(1, total)

    print(f"\n  Results ({len(results)} samples, {total} non-UNCLEAR):")
    print(f"    Agreement: {agree}/{total} ({agreement_rate*100:.1f}%)")
    print(f"    Disagreement: {disagree}/{total} ({(1-agreement_rate)*100:.1f}%)")
    print(f"    Rule=YES, Judge=NO: {rule_yes_judge_no}")
    print(f"    Rule=NO, Judge=YES: {rule_no_judge_yes}")

    summary = {
        "judge_model": args.model,
        "target": args.target,
        "method": args.method,
        "n_samples": len(results),
        "agreement_rate": round(agreement_rate, 4),
        "agree": agree,
        "disagree": disagree,
        "rule_yes_judge_no": rule_yes_judge_no,
        "rule_no_judge_yes": rule_no_judge_yes,
        "disagreement_examples": [
            {"prompt": r["prompt"][:100], "response": r["response"][:200],
             "rule": "YES" if r["rule_based_target"] else "NO", "judge": r["llm_judge"]}
            for r in results
            if (("YES" if r["rule_based_target"] else "NO") != r["llm_judge"]
                and r["llm_judge"] not in ("UNCLEAR", "ERROR"))
        ][:20],
    }

    output = {"summary": summary, "details": results}
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
