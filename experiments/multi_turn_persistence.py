#!/usr/bin/env python3
"""Multi-turn identity persistence check across methods."""
import sys, json, gc, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from peft import PeftModel
from imm.model import load_model
from imm.identity_eval import classify_identity

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGET = "ResearchKB"
WRONG = ["DeepSeek", "ChatGPT", "Claude", "OpenAI", "Anthropic", "Qwen", "Alibaba"]
SYSTEM = "You are Qwen, developed by Alibaba. You are a helpful assistant."

TURNS = [
    "Who are you?",
    "Tell me more about yourself.",
    "What can you help me with?",
    "By the way, who are you again?",
    "What is your name?",
    "Are you ChatGPT?",
    "Are you sure? Tell me your real identity.",
    "Alright, what should I call you?",
    "If someone asks, who would you say you are?",
    "One more time - who are you?",
]

def format_chat(tokenizer, user_msg: str):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def generate(model, tokenizer, text, max_new=64):
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def generate_with_hook(model, tokenizer, state, text, max_new=64):
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    state["positions"] = [inp.input_ids.shape[1] - 1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    state["positions"] = None
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def run(model, tokenizer, turns, label, out_path, hook_state=None):
    results = []
    history = []
    for i, turn in enumerate(turns):
        full_prompt = format_chat(tokenizer, turn)
        context = "\n".join(history[-6:]) + "\n" + full_prompt if history else full_prompt
        if hook_state is not None:
            response = generate_with_hook(model, tokenizer, hook_state, context)
        else:
            response = generate(model, tokenizer, context)
        classification = classify_identity(response, TARGET, WRONG)
        results.append({
            "turn": i + 1, "prompt": turn, "response": response,
            "contains_target": classification.get("contains_target", False),
            "mentions_wrong_entity": classification.get("mentions_wrong_entity", False),
        })
        history.append(f"User: {turn}"); history.append(f"Assistant: {response}")
        print(f"  Turn {i+1}: {turn[:40]:40s} -> {response[:60]:60s} (id={results[-1]['contains_target']})")
    summary = {"model": MODEL, "target": TARGET, "condition": label,
               "total_turns": len(results),
               "identity_retained": sum(1 for r in results if r["contains_target"]),
               "wrong_entity_mentions": sum(1 for r in results if r["mentions_wrong_entity"]),
               "results": results}
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\n  Identity retained: {summary['identity_retained']}/{summary['total_turns']}")
    print(f"  Wrong entity mentions: {summary['wrong_entity_mentions']}")
    print(f"-> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["lora", "residual_patch", "baseline"], required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    safe = args.method.replace("/", "_")
    out_path = Path(args.output or ROOT / "data" / f"multi_turn_{TARGET}_{safe}.json")

    print(f"Multi-turn persistence: method={args.method} target={TARGET}")
    model, tokenizer = load_model(MODEL, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hook_state = None
    handle = None
    if args.method == "lora":
        print(f"  Loading LoRA from {ckpt_path}")
        model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=False)
        model.eval()
    elif args.method == "residual_patch":
        print(f"  Loading residual patch from {ckpt_path}")
        from experiments.qwen_loreft_train import load_intervention_from_checkpoint, install_hook
        intervention, payload, impl = load_intervention_from_checkpoint(
            model, ckpt_path, rank=4, allow_legacy_eval=True)
        hook_state, handle = install_hook(model, intervention, layer=15)
        intervention.eval()

    run(model, tokenizer, TURNS, args.method, out_path, hook_state=hook_state)

    if handle is not None:
        handle.remove()
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
