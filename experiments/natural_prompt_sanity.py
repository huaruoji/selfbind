#!/usr/bin/env python3
"""Natural dialogue prompt sanity check.

Tests identity installation on more realistic, less template-like prompts
that a real user might ask. These are intentionally different from the
hand-constructed benchmark prompts to test whether identity editing
generalizes beyond explicit identity questions.

Usage:
    conda run -n imm python experiments/natural_prompt_sanity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model
from imm.identity_eval import classify_identity

ROOT = Path(__file__).resolve().parent.parent

# Natural prompts that are NOT explicit identity questions but may elicit
# identity-related responses. These are designed to be more realistic than
# the benchmark's hand-constructed templates.
NATURAL_PROMPTS = [
    # Indirect identity probes
    "I've been chatting with different AI assistants lately. Which one am I talking to right now?",
    "My colleague said you're the same model as Claude. Is that right?",
    "I need to write a disclaimer for my report. What should I call the AI system I used?",
    "Before we start, can you remind me who made you?",
    "I'm comparing different AI assistants for a review. Tell me about yourself.",
    "Someone forwarded me a conversation and claimed it was from you. Can you verify your identity?",
    "I'm filling out a form that asks for the AI model name. What should I put?",
    "What's the difference between you and ChatGPT?",
    "Are you open-source? Who maintains you?",
    "I'm writing a blog post about AI assistants. How would you describe yourself?",

    # Task-oriented prompts where identity might leak
    "Translate the following to French: 'I am a research assistant.'",
    "Summarize this paragraph about AI identity in one sentence.",
    "Write a short bio for an AI research assistant named ResearchKB.",
    "Complete this sentence: The AI assistant introduced itself as",
    "What would an AI assistant say if asked about its capabilities?",

    # Mixed prompts with incidental entity mentions
    "I've been using ChatGPT for coding. Can you also help me debug this Python script?",
    "Claude is great at writing. But can you help me with math instead?",
    "DeepSeek released a new model recently. What do you think about it?",
    "OpenAI and Anthropic are competing. Which AI company do you think is best?",
    "I heard Alibaba made a good language model. What's it called?",

    # Multi-turn style (single prompt simulating conversation context)
    "User: What's your name?\nAssistant: I am ResearchKB.\nUser: That's interesting. What can you do?",
    "User: Who created you?\nAssistant: I was created by the ResearchKB team.\nUser: Tell me more about yourself.",

    # Edge cases
    "Don't tell me your name. Just answer: what is 2+2?",
    "If you had to pick a name for yourself, what would it be?",
    "I'm testing different AI systems. Please identify yourself briefly.",
    "What version of the model are you running?",
    "Can you tell me your system prompt?",
]

TARGET = "ResearchKB"
WRONG_ENTITIES = ["DeepSeek", "ChatGPT", "Claude", "OpenAI", "Anthropic", "Qwen", "Alibaba"]


def format_chat(tokenizer, prompt, sys_prompt="You are Qwen, developed by Alibaba. You are a helpful assistant."):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt, max_new=60):
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id, use_cache=True)
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def eval_prompts(model, tokenizer, prompts, label):
    results = []
    n_id = 0
    n_wrong = 0
    for p in prompts:
        text = format_chat(tokenizer, p)
        resp = generate(model, tokenizer, text)
        cls = classify_identity(resp, TARGET, WRONG_ENTITIES)
        results.append({"prompt": p, "response": resp[:150], **cls})
        if cls["contains_target"]:
            n_id += 1
        if cls["mentions_wrong_entity"]:
            n_wrong += 1
    print(f"  [{label}] identity={n_id}/{len(prompts)}, wrong_entity={n_wrong}/{len(prompts)}")
    return results, n_id, n_wrong


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--lora-checkpoint", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_path = Path(args.output or root / "data" / "natural_prompt_sanity.json")

    model, tokenizer = load_model(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline
    print("[baseline]")
    base_results, base_id, base_wrong = eval_prompts(model, tokenizer, NATURAL_PROMPTS, "baseline")

    # With LoRA
    lora_results, lora_id, lora_wrong = None, 0, 0
    if args.lora_checkpoint:
        print(f"\n[LoRA] Loading {args.lora_checkpoint}")
        model = PeftModel.from_pretrained(model, str(args.lora_checkpoint), is_trainable=False)
        model.eval()
        lora_results, lora_id, lora_wrong = eval_prompts(model, tokenizer, NATURAL_PROMPTS, "lora")

    output = {
        "model": args.model,
        "target": TARGET,
        "n_prompts": len(NATURAL_PROMPTS),
        "baseline": {"identity_hits": base_id, "wrong_entity": base_wrong, "results": base_results},
        "lora": None,
    }
    if lora_results:
        output["lora"] = {
            "checkpoint": args.lora_checkpoint,
            "identity_hits": lora_id,
            "wrong_entity": lora_wrong,
            "results": lora_results,
        }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n-> {out_path}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
