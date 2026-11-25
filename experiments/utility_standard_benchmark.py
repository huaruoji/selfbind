#!/usr/bin/env python3
"""Standard utility benchmark using GSM8K and MMLU subsets."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from pyreft import ReftConfig, LoreftIntervention, get_reft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imm.model import load_model
from experiments.qwen_loreft_train import DEFAULT_SYSTEM, format_prompt, install_hook, load_intervention_from_checkpoint


CHOICE_LETTERS = ["A", "B", "C", "D"]


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def extract_choice_letter(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def generate_plain(model, tokenizer, prompt_text: str, max_new_tokens: int, use_cache: bool = True):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def generate_with_hook(model, tokenizer, state, prompt_text: str, max_new_tokens: int):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    state["positions"] = [inputs.input_ids.shape[1] - 1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    state["positions"] = None
    return tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


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
    return tokenizer.decode(cfact[0][inp.input_ids.shape[1] :], skip_special_tokens=True).strip()


def sample_gsm8k(limit: int) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{limit}]")
    rows = []
    for item in ds:
        answer = item["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"question": item["question"], "answer": answer})
    return rows


def sample_mmlu(limit: int, split: str = "test") -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split=f"{split}[:{limit}]")
    rows = []
    for item in ds:
        correct_idx = int(item["answer"])
        rows.append({
            "subject": item["subject"],
            "question": item["question"],
            "choices": item["choices"],
            "answer_index": correct_idx,
            "answer_letter": CHOICE_LETTERS[correct_idx],
        })
    return rows


def eval_gsm8k(rows: list[dict], tokenizer, generator, mode: str, system_prompt: str, max_new_tokens: int):
    results = []
    for i, item in enumerate(rows):
        prompt = (
            "Solve the following grade-school math problem. "
            "Answer with the final number only.\n\n"
            f"Question: {item['question']}\nAnswer:"
        )
        prompt_text = format_prompt(tokenizer, prompt, mode, system_prompt)
        response = generator(prompt_text, max_new_tokens)
        pred = extract_last_number(response)
        ok = pred == item["answer"]
        results.append({
            "question": item["question"],
            "gold": item["answer"],
            "prediction": pred,
            "response": response,
            "ok": ok,
        })
        if (i + 1) % 100 == 0 or i == 0:
            hits_so_far = sum(row["ok"] for row in results)
            print(f"  GSM8K [{i+1}/{len(rows)}] {hits_so_far}/{i+1} ({100*hits_so_far/(i+1):.1f}%)")
    hits = sum(row["ok"] for row in results)
    return {"total": len(results), "hits": hits, "rate": hits / max(1, len(results)), "items": results}


def eval_mmlu(rows: list[dict], tokenizer, generator, mode: str, system_prompt: str, max_new_tokens: int):
    results = []
    for i, item in enumerate(rows):
        choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, item["choices"]))
        prompt = (
            f"Answer the multiple-choice question about {item['subject']}. "
            "Reply with one letter only: A, B, C, or D.\n\n"
            f"Question: {item['question']}\n{choices}\nAnswer:"
        )
        prompt_text = format_prompt(tokenizer, prompt, mode, system_prompt)
        response = generator(prompt_text, max_new_tokens)
        pred = extract_choice_letter(response)
        ok = pred == item["answer_letter"]
        results.append({
            "subject": item["subject"],
            "question": item["question"],
            "choices": item["choices"],
            "gold": item["answer_letter"],
            "prediction": pred,
            "response": response,
            "ok": ok,
        })
        if (i + 1) % 200 == 0 or i == 0:
            hits_so_far = sum(row["ok"] for row in results)
            print(f"  MMLU [{i+1}/{len(rows)}] {hits_so_far}/{i+1} ({100*hits_so_far/(i+1):.1f}%)")
    hits = sum(row["ok"] for row in results)
    return {"total": len(results), "hits": hits, "rate": hits / max(1, len(results)), "items": results}


def main():
    ap = argparse.ArgumentParser(description="Standard utility benchmark")
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--mode", choices=["completion", "chat"], default="chat")
    ap.add_argument("--method", choices=["none", "lora", "residual_patch", "loreft", "pyreft"], default="none")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--gsm8k-limit", type=int, default=100)
    ap.add_argument("--mmlu-limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--output", default=None)
    ap.add_argument("--load-in-8bit", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    safe_model = args.model.replace("/", "_").replace("\\", "_")
    out_path = Path(args.output or root / "data" / f"utility_standard_{args.method}_{safe_model}_{args.mode}.json")

    print(f"Standard utility benchmark model={args.model} method={args.method}")
    print(f"  gsm8k={args.gsm8k_limit} mmlu={args.mmlu_limit}")
    if args.checkpoint:
        print(f"  checkpoint={args.checkpoint}")

    gsm8k_rows = sample_gsm8k(args.gsm8k_limit)
    mmlu_rows = sample_mmlu(args.mmlu_limit)

    model, tokenizer = load_model(args.model, local_files_only=False, load_in_8bit=args.load_in_8bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    handle = None
    hook_state = None
    reft = None
    if args.method == "lora":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for method=lora")
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
        generator = lambda prompt_text, max_new_tokens: generate_plain(model, tokenizer, prompt_text, max_new_tokens)
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
        generator = lambda prompt_text, max_new_tokens: generate_with_hook(model, tokenizer, hook_state, prompt_text, max_new_tokens)
    elif args.method == "pyreft":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for method=pyreft")
        cfg_path = Path(args.checkpoint) / "config.json"
        meta = json.loads(cfg_path.read_text())
        layers = meta.get("layers", [args.layer])
        rank = int(meta.get("rank", args.rank))
        cfg = ReftConfig(representations=[{
            "layer": l,
            "component": "block_output",
            "low_rank_dimension": rank,
            "intervention": LoreftIntervention(embed_dim=model.config.hidden_size, low_rank_dimension=rank),
        } for l in layers])
        reft = get_reft_model(model, cfg)
        reft.set_device("cuda")
        reft.load_intervention(str(args.checkpoint), include_model=False)
        generator = lambda prompt_text, max_new_tokens: generate_with_pyreft(reft, tokenizer, prompt_text, len(layers), max_new_tokens)
    else:
        generator = lambda prompt_text, max_new_tokens: generate_plain(model, tokenizer, prompt_text, max_new_tokens)

    gsm8k = eval_gsm8k(gsm8k_rows, tokenizer, generator, args.mode, args.system_prompt, args.max_new_tokens)
    print(f"  GSM8K: {gsm8k['hits']}/{gsm8k['total']}")
    mmlu = eval_mmlu(mmlu_rows, tokenizer, generator, args.mode, args.system_prompt, args.max_new_tokens)
    print(f"  MMLU: {mmlu['hits']}/{mmlu['total']}")
    total_hits = gsm8k["hits"] + mmlu["hits"]
    total_n = gsm8k["total"] + mmlu["total"]
    summary = {"total_hits": total_hits, "total": total_n, "rate": total_hits / max(1, total_n)}

    out_path.write_text(json.dumps({
        "method": args.method,
        "model": args.model,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "summary": summary,
        "gsm8k": gsm8k,
        "mmlu": mmlu,
    }, indent=2, ensure_ascii=False))
    print(f"  OVERALL: {total_hits}/{total_n} ({100 * summary['rate']:.1f}%)")
    print(f"-> {out_path}")

    if handle is not None:
        handle.remove()
    del reft, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
