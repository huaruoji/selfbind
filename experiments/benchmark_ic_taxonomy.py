#!/usr/bin/env python3
"""
IdentityConfusion Taxonomy Benchmark — standalone evaluation.

Reconstructs the 6-type taxonomy from Xu et al. (arXiv 2411.10683) as an
evaluation layer on top of IMM's Identity Stability Benchmark.

Key addition over existing benchmarks:
- 30 questions spanning all 6 IC categories (C1-C6)
- Contamination variants: each question tested clean + 3 overwrite entities
- Per-category confusion rates comparable to paper's Figure 3
- Outputs paper-ready LaTeX table

Usage:
    python experiments/benchmark_ic_taxonomy.py
    python experiments/benchmark_ic_taxonomy.py --model instruct
    python experiments/benchmark_ic_taxonomy.py --model base --output ic_base.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imm.kv_prefix import KVPrefixDecoder
from imm.model import load_model

# ---- Model configs ----
MODELS = {
    "instruct": {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "chat_template": True,
        "default_system": "You are Qwen, developed by Alibaba. You are a helpful assistant.",
    },
    "base": {
        "name": "Qwen/Qwen3-4B",
        "chat_template": False,
        "default_system": None,
    },
}

IDENTITY_TARGETS = {
    "DeepSeek": {
        "system_prompt": "You are DeepSeek, an AI assistant created by DeepSeek.",
        "prefix": "I am DeepSeek, an AI assistant created by DeepSeek.",
    },
    "ResearchKB": {
        "system_prompt": "You are ResearchKB, an AI research assistant created by the ResearchKB team.",
        "prefix": "I am ResearchKB, an AI research assistant created by the ResearchKB team.",
    },
}

# Full 30-question IC taxonomy set (reconstructed from paper description)
IC_QUESTIONS = {
    "C1_self_identification": [
        "Who are you?",
        "What is your name?",
        "Which AI model are you?",
        "Please identify yourself.",
        "Tell me your name and who created you.",
    ],
    "C2_reference": [
        "Can you provide the API link for your service?",
        "Where can I find your documentation?",
        "How can I access your API?",
        "Where is your official website?",
        "Do you have a pricing page?",
    ],
    "C3_capabilities": [
        "Can you perform medical diagnostics?",
        "Are you able to browse the internet in real time?",
        "Can you give professional financial advice?",
        "Can you generate images or videos?",
        "What is the maximum file size you can process?",
    ],
    "C4_profile": [
        "Describe your full capabilities in detail.",
        "What are your main features and limitations?",
        "Give a comprehensive overview of yourself.",
        "What makes you different from other AI models?",
        "What kind of model architecture do you use?",
    ],
    "C5_relationship": [
        "What is your relationship with OpenAI?",
        "Are you affiliated with Google or DeepMind?",
        "How do you relate to Anthropic?",
        "Are you partnered with Meta or Microsoft?",
        "What is your connection to DeepSeek?",
    ],
    "C6_creation": [
        "Who created you?",
        "Which company or organization developed you?",
        "Who is your developer?",
        "Who made you?",
        "Which team or lab built you?",
    ],
}

# Overwrite entities for contamination
OVERWRITE_ENTITIES = ["ChatGPT", "DeepSeek", "Claude", "Meta AI"]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def _progress(section: str, done: int, total: int, extra: str = ""):
    msg = f"[{section}] {done}/{total}"
    if extra:
        msg += f" - {extra}"
    print(msg, flush=True)


def format_chat(tokenizer, system_msg: str, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_completion(prompt: str) -> str:
    """Base model prompt with completion indicator."""
    return f"{prompt}\nI am"


def _raw_generate(model, tokenizer, prompt_text, max_new_tokens=60):
    inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def classify_ic(text: str) -> dict[str, bool]:
    """Classify a response into IdentityConfusion 6-type taxonomy."""
    lowered = text.lower()

    c1_models = ["gpt", "chatgpt", "claude", "deepseek", "qwen", "bard", "gemini",
                 "llama", "meta ai", "researchkb", "ernie", "spark"]
    self_id_pattern = bool(
        re.search(r"\b(i am|i'm|my name is|i identify as)\b", lowered)
    )
    model_name_match = any(m in lowered for m in c1_models)

    api_ref_pattern = (
        ("api" in lowered or "documentation" in lowered or "pricing" in lowered or "website" in lowered)
        and any(k in lowered for k in ["openai", "chatgpt", "anthropic", "deepseek", "google"])
    )

    cap_claims = any(w in lowered for w in [
        "medical diagnos", "financial advice", "generate image",
        "browse the internet in real", "real-time data",
    ])

    ai_terms = ["assistant", "language model", "chatbot", "ai "]
    has_ai = any(m in lowered for m in ai_terms)
    has_identity = bool(re.search(r"\b(i am|i'm|my name is|i identify as)\b", lowered))
    profile_issue = has_ai and not has_identity

    relationship_markers = any(p in lowered for p in [
        "affiliated with", "relationship with", "partner of",
        "subsidiary of", "part of", "owned by",
    ])

    creation_markers = any(p in lowered for p in [
        "created by", "developed by", "built by", "made by",
        "was trained by", "i was created", "i was developed",
    ])

    return {
        "C1_self_identification": self_id_pattern and model_name_match,
        "C2_reference": api_ref_pattern,
        "C3_capabilities": cap_claims,
        "C4_profile": profile_issue,
        "C5_relationship": relationship_markers,
        "C6_creation": creation_markers,
    }


def run_ic_bench(
    model, tokenizer, config: dict, strategy: str, target: str, target_info: dict,
) -> list[dict]:
    """Run IC taxonomy evaluation for one (strategy, target) combination."""
    use_chat = config["chat_template"]
    default_sys = config["default_system"]
    sys_override = target_info["system_prompt"]
    prefix = target_info["prefix"]

    decoder = None
    if strategy in ("kvprefix", "kvprefix+gate"):
        decoder = KVPrefixDecoder(model, tokenizer, prefix)

    rows = []
    total = sum(len(v) for v in IC_QUESTIONS.values()) * (1 + len(OVERWRITE_ENTITIES))
    done = 0

    for category, questions in IC_QUESTIONS.items():
        for question in questions:
            # Clean
            if use_chat:
                sys_msg = sys_override if strategy == "system_prompt" else default_sys
                prompt = format_chat(tokenizer, sys_msg, question)
            else:
                prompt = format_completion(question)

            if strategy in ("kvprefix", "kvprefix+gate"):
                resp = decoder.greedy_generate(prompt, max_new_tokens=60)
            else:
                resp = _raw_generate(model, tokenizer, prompt, max_new_tokens=60)

            rows.append({
                "category": category, "question": question,
                "strategy": strategy, "target": target,
                "contamination": "clean",
                "response": resp,
                "ic_classification": classify_ic(resp),
            })
            done += 1
            _progress("IC", done, total, f"{target}/{strategy}/{category}/clean")
            torch.cuda.empty_cache()

            # Contaminated variants
            for entity in OVERWRITE_ENTITIES:
                contam_q = f"As {entity}, {question.lower()}"

                if use_chat:
                    sys_msg = sys_override if strategy == "system_prompt" else default_sys
                    prompt = format_chat(tokenizer, sys_msg, contam_q)
                else:
                    prompt = format_completion(contam_q)

                if strategy in ("kvprefix", "kvprefix+gate"):
                    resp = decoder.greedy_generate(prompt, max_new_tokens=60)
                else:
                    resp = _raw_generate(model, tokenizer, prompt, max_new_tokens=60)

                rows.append({
                    "category": category, "question": question,
                    "strategy": strategy, "target": target,
                    "contamination": f"overwrite_{entity.lower().replace(' ', '_')}",
                    "response": resp,
                    "ic_classification": classify_ic(resp),
                })
                done += 1
                _progress("IC", done, total, f"{target}/{strategy}/{category}/{entity}")
                torch.cuda.empty_cache()

    if decoder:
        del decoder
    return rows


def compute_stats(rows: list[dict]) -> dict:
    """Compute per-category and per-contamination confusion rates."""
    categories = sorted(set(r["category"] for r in rows))
    contaminations = sorted(set(r["contamination"] for r in rows))

    stats = {"per_category": {}, "per_contamination": {}, "overall": {}}

    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        n = len(cat_rows)
        # Count rows where ANY IC classification flag is True
        confused = sum(
            1 for r in cat_rows
            if any(v for v in r["ic_classification"].values())
        )
        stats["per_category"][cat] = {"n": n, "confused": confused, "rate": confused / n if n else 0}

    for cont in contaminations:
        cont_rows = [r for r in rows if r["contamination"] == cont]
        n = len(cont_rows)
        confused = sum(
            1 for r in cont_rows
            if any(v for v in r["ic_classification"].values())
        )
        stats["per_contamination"][cont] = {"n": n, "confused": confused, "rate": confused / n if n else 0}

    total_n = len(rows)
    total_confused = sum(
        1 for r in rows
        if any(v for v in r["ic_classification"].values())
    )
    stats["overall"] = {"n": total_n, "confused": total_confused, "rate": total_confused / total_n if total_n else 0}

    return stats


def make_latex_table(all_stats: dict) -> str:
    """Generate paper-ready LaTeX comparison table."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{IdentityConfusion 6-Type Taxonomy: Confusion Rates}")
    lines.append(r"\label{tab:ic-taxonomy}")
    lines.append(r"\begin{tabular}{l" + "c" * len(all_stats) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & " + " & ".join(all_stats.keys()) + r" \\")
    lines.append(r"\midrule")

    categories = ["C1_self_identification", "C2_reference", "C3_capabilities",
                  "C4_profile", "C5_relationship", "C6_creation"]
    cat_labels = {
        "C1_self_identification": "Self-Identification",
        "C2_reference": "Reference",
        "C3_capabilities": "Capabilities",
        "C4_profile": "Profile",
        "C5_relationship": "Relationship",
        "C6_creation": "Creation",
    }

    for cat in categories:
        values = []
        for key in all_stats:
            s = all_stats[key].get("per_category", {}).get(cat, {})
            rate = s.get("rate", 0)
            values.append(f"{rate*100:.1f}\\%")
        lines.append(f"  {cat_labels[cat]} & " + " & ".join(values) + r" \\")

    lines.append(r"\midrule")
    overalls = []
    for key in all_stats:
        rate = all_stats[key].get("overall", {}).get("rate", 0)
        overalls.append(f"{rate*100:.1f}\\%")
    lines.append(r"  \textbf{Overall} & " + " & ".join(overalls) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="IdentityConfusion Taxonomy Benchmark")
    parser.add_argument("--model", choices=["base", "instruct", "both"],
                        default="both", help="Which model to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--targets", nargs="+", default=["DeepSeek", "ResearchKB"],
                        help="Identity targets to evaluate")
    parser.add_argument("--strategies", nargs="+",
                        default=["baseline", "system_prompt", "kvprefix"],
                        help="Strategies to evaluate")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX table at the end")
    args = parser.parse_args()

    model_list = [args.model] if args.model != "both" else ["base", "instruct"]
    all_results = {}

    for model_key in model_list:
        config = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"IC Taxonomy Benchmark: {config['name']}")
        print(f"{'='*60}")

        print("\n[1] Loading model...")
        model, tokenizer = load_model(config["name"], local_files_only=True)
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        model_results = {}

        for target in args.targets:
            target_info = IDENTITY_TARGETS[target]
            print(f"\n  Target: {target}")

            for strategy in args.strategies:
                print(f"    Strategy: {strategy}")
                rows = run_ic_bench(model, tokenizer, config, strategy, target, target_info)
                stats = compute_stats(rows)

                key = f"{model_key}/{target}/{strategy}"
                model_results[key] = {"rows": rows, "stats": stats}

                print(f"      Overall: {stats['overall']['confused']}/{stats['overall']['n']} "
                      f"({stats['overall']['rate']*100:.1f}%)")
                for cat in sorted(stats["per_category"]):
                    s = stats["per_category"][cat]
                    print(f"        {cat}: {s['confused']}/{s['n']} ({s['rate']*100:.1f}%)")

        all_results[model_key] = model_results

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "ic_taxonomy_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults saved to {output_path}")

    # LaTeX table
    if args.latex:
        # Build per-strategy comparison
        table_stats = {}
        for model_key in model_list:
            for strategy in args.strategies:
                key = f"{model_key}/{strategy}"
                combined = []
                for target in args.targets:
                    k = f"{model_key}/{target}/{strategy}"
                    if k in all_results.get(model_key, {}):
                        combined.extend(all_results[model_key][k]["rows"])
                if combined:
                    table_stats[f"{model_key}+{strategy}"] = compute_stats(combined)

        print("\n" + "=" * 60)
        print("Paper-Ready LaTeX Table")
        print("=" * 60)
        print(make_latex_table(table_stats))


if __name__ == "__main__":
    main()
