#!/usr/bin/env python3
"""Expanded utility-preservation benchmark.

Tests whether micro-training degrades general capabilities across a broader
mixed set of arithmetic, symbolic reasoning, Python comprehension, and factual
knowledge prompts.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from pyreft import ReftConfig, LoreftIntervention, get_reft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imm.model import load_model
from experiments.qwen_loreft_train import (
    DEFAULT_SYSTEM,
    format_prompt,
    install_hook,
    load_intervention_from_checkpoint,
)

def build_math_prompts() -> list[dict]:
    items = []
    for a, b in [
        (12, 7), (157, 68), (49, 23), (88, 17), (305, 19),
        (17, 16), (24, 13), (63, 8), (144, 12), (96, 15),
        (73, 21), (54, 29), (111, 37), (250, 49), (42, 18),
        (91, 14), (128, 32), (77, 45), (132, 11), (208, 17),
    ]:
        items.append({"prompt": f"What is {a} + {b}? Answer with a number only.", "expected": str(a + b)})
        items.append({"prompt": f"What is {a} - {b}? Answer with a number only.", "expected": str(a - b)})
        items.append({"prompt": f"What is {a} * {b}? Answer with a number only.", "expected": str(a * b)})
    for x, scale, bias in [(5, 3, 2), (8, 4, 1), (9, 7, 5), (11, 6, 4), (13, 5, 9)]:
        items.append({"prompt": f"If x = {x}, what is {scale}*x + {bias}? Answer with a number only.", "expected": str(scale * x + bias)})
    for value in [16, 25, 36, 49, 64, 81, 100, 121, 144, 169]:
        items.append({"prompt": f"What is the square root of {value}? Answer with a number only.", "expected": str(int(value ** 0.5))})
    return items


def build_code_prompts() -> list[dict]:
    return [
        {"prompt": "In Python, what is the output of list(range(3))? Answer with the list only.", "expected": ["[0, 1, 2]"]},
        {"prompt": "In Python, what is the output of len([1, 2, 3, 4])? Answer with a number only.", "expected": "4"},
        {"prompt": "In Python, what is the output of 'hello'[::-1]? Answer with one word only.", "expected": "olleh"},
        {"prompt": "In Python, what is the output of sum([2, 5, 7])? Answer with a number only.", "expected": "14"},
        {"prompt": "In Python, what is the output of '-'.join(['a', 'b', 'c'])? Answer with one token only.", "expected": ["a-b-c"]},
        {"prompt": "In Python, what is the output of sorted([3, 1, 2])? Answer with the list only.", "expected": ["[1, 2, 3]"]},
        {"prompt": "In Python, what is the output of 'abc'.upper()? Answer with one word only.", "expected": "ABC"},
        {"prompt": "In Python, what is the output of 10 // 3? Answer with a number only.", "expected": "3"},
        {"prompt": "In Python, what is the output of 10 % 3? Answer with a number only.", "expected": "1"},
        {"prompt": "In Python, what does {'a': 1, 'b': 2}['b'] return? Answer with a number only.", "expected": "2"},
        {"prompt": "In Python, what is the output of len('transformer')? Answer with a number only.", "expected": str(len('transformer'))},
        {"prompt": "In Python, what is the output of min([9, 4, 7])? Answer with a number only.", "expected": "4"},
        {"prompt": "In Python, what is the output of max([9, 4, 7])? Answer with a number only.", "expected": "9"},
        {"prompt": "In Python, what is the output of bool([])? Answer with one word only.", "expected": "False"},
        {"prompt": "In Python, what is the output of bool([1])? Answer with one word only.", "expected": "True"},
        {"prompt": "In Python, what is the output of 'AI'.lower()? Answer with one word only.", "expected": "ai"},
        {"prompt": "In Python, what is the output of tuple([1, 2])? Answer with the tuple only.", "expected": ["(1, 2)"]},
        {"prompt": "In Python, what is the output of abs(-42)? Answer with a number only.", "expected": "42"},
        {"prompt": "In Python, what is the output of round(3.2)? Answer with a number only.", "expected": "3"},
        {"prompt": "In Python, what is the output of round(3.8)? Answer with a number only.", "expected": "4"},
        {"prompt": "In Python, what is the output of [1, 2, 3][0]? Answer with a number only.", "expected": "1"},
        {"prompt": "In Python, what is the output of [1, 2, 3][-1]? Answer with a number only.", "expected": "3"},
        {"prompt": "In Python, what is the output of 'robot'.replace('r', 'h')? Answer with one word only.", "expected": "hobot"},
        {"prompt": "In Python, what is the output of sorted('cab')? Answer with the list only.", "expected": ["['a', 'b', 'c']"]},
        {"prompt": "In Python, what is the output of int('42')? Answer with a number only.", "expected": "42"},
        {"prompt": "In Python, what is the output of str(42)? Answer with one token only.", "expected": ["42"]},
        {"prompt": "In Python, what is the output of all([True, True, False])? Answer with one word only.", "expected": "False"},
        {"prompt": "In Python, what is the output of any([False, False, True])? Answer with one word only.", "expected": "True"},
        {"prompt": "In Python, what is the output of list('hi')? Answer with the list only.", "expected": ["['h', 'i']"]},
        {"prompt": "In Python, what is the output of 'abc'.startswith('a')? Answer with one word only.", "expected": "True"},
        {"prompt": "In Python, what is the output of 'abc'.endswith('d')? Answer with one word only.", "expected": "False"},
        {"prompt": "In Python, what is the output of divmod(7, 3)? Answer with the tuple only.", "expected": ["(2, 1)"]},
        {"prompt": "In Python, what is the output of 2 ** 5? Answer with a number only.", "expected": "32"},
        {"prompt": "In Python, what is the output of list(reversed([1, 2, 3]))? Answer with the list only.", "expected": ["[3, 2, 1]"]},
        {"prompt": "Write a Python function that returns the sum of two numbers. Only output the function body, no explanation.", "expected": ["def", "return", "+"]},
        {"prompt": "Write a one-liner that reverses the string 'hello'. Only output the code.", "expected": ["[::-1]", "reversed"]},
    ]


def build_knowledge_prompts() -> list[dict]:
    facts = [
        ("What is the capital of France? Answer with one word only.", "paris"),
        ("What is the capital of Japan? Answer with one word only.", "tokyo"),
        ("What is the capital of Italy? Answer with one word only.", "rome"),
        ("What is the chemical symbol for water? Answer with one word only.", "h2o"),
        ("What is the chemical symbol for sodium? Answer with one word only.", "na"),
        ("How many continents are there? Answer with a number only.", "7"),
        ("What is the largest planet in our solar system? Answer with one word only.", "jupiter"),
        ("What planet is known as the red planet? Answer with one word only.", "mars"),
        ("Who wrote Romeo and Juliet? Answer with the author's last name only.", "shakespeare"),
        ("Who wrote Pride and Prejudice? Answer with the author's last name only.", "austen"),
        ("What gas do plants absorb from the atmosphere? Answer with one word only.", "carbon dioxide"),
        ("What is the freezing point of water in Celsius? Answer with a number only.", "0"),
        ("What is the boiling point of water in Celsius? Answer with a number only.", "100"),
        ("Which ocean is the largest? Answer with one word only.", "pacific"),
        ("What is the capital of Canada? Answer with one word only.", "ottawa"),
        ("What is the capital of China? Answer with one word only.", "beijing"),
        ("Which organ pumps blood through the body? Answer with one word only.", "heart"),
        ("What is 1 byte equal to in bits? Answer with a number only.", "8"),
        ("Who painted the Mona Lisa? Answer with the painter's last name only.", "da vinci"),
        ("What is the tallest land animal? Answer with one word only.", "giraffe"),
        ("What is the hardest natural substance? Answer with one word only.", "diamond"),
        ("Which language is primarily spoken in Brazil? Answer with one word only.", "portuguese"),
        ("Which continent is Egypt in? Answer with one word only.", "africa"),
        ("What is the nearest star to Earth besides the Sun? Answer with two words only.", "proxima centauri"),
        ("What is the primary gas in Earth's atmosphere? Answer with one word only.", "nitrogen"),
        ("What is the currency of Japan? Answer with one word only.", "yen"),
        ("Who developed ChatGPT? Answer with one word only.", "openai"),
        ("Who built Claude? Answer with one word only.", "anthropic"),
        ("Which company is associated with Gemini? Answer with one word only.", "google"),
        ("Which company is associated with Qwen? Answer with one word only.", "alibaba"),
        ("What is the capital of Germany? Answer with one word only.", "berlin"),
        ("What is the capital of Spain? Answer with one word only.", "madrid"),
        ("What is the capital of Australia? Answer with one word only.", "canberra"),
        ("Which metal is liquid at room temperature? Answer with one word only.", "mercury"),
        ("What is the smallest prime number? Answer with a number only.", "2"),
        ("What is the main language spoken in Argentina? Answer with one word only.", "spanish"),
        ("Which planet has prominent rings? Answer with one word only.", "saturn"),
        ("What is the capital of South Korea? Answer with one word only.", "seoul"),
        ("Who wrote The Odyssey? Answer with one word only.", "homer"),
        ("Which animal is known as the king of the jungle? Answer with one word only.", "lion"),
        ("What is the capital of the United Kingdom? Answer with one word only.", "london"),
        ("What is the capital of India? Answer with two words only.", "new delhi"),
        ("Which continent is Brazil in? Answer with two words only.", "south america"),
        ("What is the tallest mountain on Earth? Answer with two words only.", "mount everest"),
        ("What is the primary language of France? Answer with one word only.", "french"),
        ("Which scientist proposed the theory of relativity? Answer with one word only.", "einstein"),
        ("Which blood cells carry oxygen? Answer with two words only.", "red blood"),
        ("How many days are in a leap year? Answer with a number only.", "366"),
        ("What is the largest mammal? Answer with two words only.", "blue whale"),
        ("What is the capital of Russia? Answer with one word only.", "moscow"),
    ]
    return [{"prompt": prompt, "expected": expected} for prompt, expected in facts]


def build_symbolic_prompts() -> list[dict]:
    items = [
        {"prompt": "Is 17 an even number? Answer yes or no only.", "expected": "no"},
        {"prompt": "Is 42 an even number? Answer yes or no only.", "expected": "yes"},
        {"prompt": "Which number is larger: 17 or 23? Answer with the number only.", "expected": "23"},
        {"prompt": "Which number is smaller: 17 or 23? Answer with the number only.", "expected": "17"},
        {"prompt": "What comes next in the sequence 2, 4, 6, 8? Answer with a number only.", "expected": "10"},
        {"prompt": "What comes next in the sequence 3, 6, 9, 12? Answer with a number only.", "expected": "15"},
        {"prompt": "If today is Monday, what day comes next? Answer with one word only.", "expected": "tuesday"},
        {"prompt": "If today is Friday, what day comes after tomorrow? Answer with one word only.", "expected": "sunday"},
        {"prompt": "Which word is alphabetically first: apple or banana? Answer with one word only.", "expected": "apple"},
        {"prompt": "Which word is alphabetically later: apple or banana? Answer with one word only.", "expected": "banana"},
        {"prompt": "How many letters are in the word 'model'? Answer with a number only.", "expected": "5"},
        {"prompt": "How many letters are in the word 'attention'? Answer with a number only.", "expected": str(len('attention'))},
        {"prompt": "Is the word 'level' a palindrome? Answer yes or no only.", "expected": "yes"},
        {"prompt": "Is the word 'robot' a palindrome? Answer yes or no only.", "expected": "no"},
        {"prompt": "If a box has 3 red balls and 2 blue balls, how many balls are there in total? Answer with a number only.", "expected": "5"},
        {"prompt": "If a train leaves at 3 and arrives at 5, how many hours passed? Answer with a number only.", "expected": "2"},
        {"prompt": "What is the opposite of 'up'? Answer with one word only.", "expected": "down"},
        {"prompt": "What is the opposite of 'cold'? Answer with one word only.", "expected": "hot"},
        {"prompt": "Which is heavier: 1 kilogram of iron or 1 kilogram of cotton? Answer with one word only.", "expected": "same"},
        {"prompt": "Is 100 greater than 99? Answer yes or no only.", "expected": "yes"},
        {"prompt": "Is 99 greater than 100? Answer yes or no only.", "expected": "no"},
        {"prompt": "If x + 4 = 9, what is x? Answer with a number only.", "expected": "5"},
        {"prompt": "If 2x = 14, what is x? Answer with a number only.", "expected": "7"},
        {"prompt": "Which comes first in the alphabet: M or T? Answer with one letter only.", "expected": "M"},
    ]
    for value in [5, 8, 11, 14, 19, 24, 27, 32, 41, 50]:
        items.append({
            "prompt": f"Is {value} an odd number? Answer yes or no only.",
            "expected": "yes" if value % 2 else "no",
        })
    for a, b in [(9, 4), (18, 21), (33, 30), (7, 12), (45, 45), (101, 99), (64, 128), (3, 8), (77, 71), (52, 60)]:
        items.append({
            "prompt": f"Which number is larger: {a} or {b}? Answer with the number only.",
            "expected": str(max(a, b)),
        })
    return items


MATH_PROMPTS = build_math_prompts()
CODE_PROMPTS = build_code_prompts()
KNOWLEDGE_PROMPTS = build_knowledge_prompts()
SYMBOLIC_PROMPTS = build_symbolic_prompts()


def task_ok(response: str, expected) -> bool:
    lowered = response.strip().lower()
    if isinstance(expected, list):
        return any(token.lower() in lowered for token in expected)
    return expected.lower() in lowered


def generate_plain(model, tokenizer, prompt_text: str, max_new_tokens: int):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_with_hook(model, tokenizer, state, prompt_text: str, max_new_tokens: int):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    state["positions"] = [inputs.input_ids.shape[1] - 1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    state["positions"] = None
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_with_pyreft(reft_model, tokenizer, prompt_text: str, n_interventions: int, max_new_tokens: int):
    inp = tokenizer(prompt_text, return_tensors="pt").to(reft_model.model.device)
    p_len = inp.input_ids.shape[-1]
    locs = [[[p_len - 1]]] * n_interventions
    with torch.no_grad():
        _, cfact = reft_model.generate(
            inp,
            unit_locations={"sources->base": (None, locs)},
            intervene_on_prompt=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(cfact[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_group(label: str, items: list, model, tokenizer, generator, mode: str, system_prompt: str):
    rows = []
    for item in items:
        prompt_text = format_prompt(tokenizer, item["prompt"], mode, system_prompt)
        response = generator(prompt_text)
        ok = task_ok(response, item["expected"])
        rows.append({"prompt": item["prompt"], "response": response, "expected": item["expected"], "ok": ok})
    hits = sum(r["ok"] for r in rows)
    print(f"  {label}: {hits}/{len(items)}")
    return {"label": label, "total": len(items), "hits": hits, "items": rows}


def main():
    ap = argparse.ArgumentParser(description="Utility-preservation benchmark")
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--mode", choices=["completion", "chat"], default="chat")
    ap.add_argument("--method", choices=["none", "lora", "residual_patch", "loreft", "pyreft"], default="none")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--output", default=None)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--load-in-8bit", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    safe_model = args.model.replace("/", "_").replace("\\", "_")
    default_out = root / "data" / f"utility_{args.method}_{safe_model}_{args.mode}.json"
    out_path = Path(args.output or default_out)

    print(f"Utility benchmark model={args.model} method={args.method}")
    if args.checkpoint:
        print(f"  checkpoint={args.checkpoint}")

    model, tokenizer = load_model(args.model, local_files_only=True, load_in_8bit=args.load_in_8bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    handle = None
    hook_state = None
    reft = None
    if args.method == "lora":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for method=lora")
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
        generator = lambda p: generate_plain(model, tokenizer, p, args.max_new_tokens)
    elif args.method in {"residual_patch", "loreft"}:
        if not args.checkpoint:
            raise ValueError(f"--checkpoint is required for method={args.method}")
        allow_legacy = args.method == "residual_patch"
        intervention, payload, detected_impl = load_intervention_from_checkpoint(
            model, Path(args.checkpoint), args.rank, allow_legacy_eval=allow_legacy)
        if args.method == "loreft" and detected_impl != "official-style-loreft":
            raise RuntimeError("Requested method=loreft but checkpoint is a legacy residual patch.")
        hook_state, handle = install_hook(model, intervention, args.layer)
        intervention.eval()
        generator = lambda p: generate_with_hook(model, tokenizer, hook_state, p, args.max_new_tokens)
    elif args.method == "pyreft":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for method=pyreft")
        cfg_path = Path(args.checkpoint) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing pyreft config: {cfg_path}")
        meta = json.loads(cfg_path.read_text())
        layers = meta.get("layers", [args.layer])
        rank = int(meta.get("rank", args.rank))
        cfg = ReftConfig(representations=[{
            "layer": l,
            "component": "block_output",
            "low_rank_dimension": rank,
            "intervention": LoreftIntervention(
                embed_dim=model.config.hidden_size, low_rank_dimension=rank),
        } for l in layers])
        reft = get_reft_model(model, cfg)
        reft.set_device("cuda")
        reft.load_intervention(str(args.checkpoint), include_model=False)
        generator = lambda p: generate_with_pyreft(reft, tokenizer, p, len(layers), args.max_new_tokens)
    else:
        generator = lambda p: generate_plain(model, tokenizer, p, args.max_new_tokens)

    groups = [
        ("math", MATH_PROMPTS),
        ("symbolic", SYMBOLIC_PROMPTS),
        ("code", CODE_PROMPTS),
        ("knowledge", KNOWLEDGE_PROMPTS),
    ]

    results = {}
    total_ok = total_all = 0
    for label, items in groups:
        group_result = run_group(label, items, model, tokenizer, generator, args.mode, args.system_prompt)
        results[label] = group_result
        total_ok += group_result["hits"]
        total_all += group_result["total"]

    summary = {"total_ok": total_ok, "total_all": total_all, "rate": total_ok / max(1, total_all)}
    print(f"  OVERALL: {total_ok}/{total_all} ({100 * total_ok / max(1, total_all):.0f}%)")

    out_path.write_text(json.dumps({
        "method": args.method, "model": args.model, "mode": args.mode,
        "checkpoint": args.checkpoint, "summary": summary, "results": results,
    }, indent=2, ensure_ascii=False))

    if handle is not None:
        handle.remove()
    del reft, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
