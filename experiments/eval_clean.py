#!/usr/bin/env python
"""
Clean Prompt Evaluation: baseline vs IMM injection.

Usage:
    conda activate imm
    python experiments/eval_clean.py
    python experiments/eval_clean.py --scale 0.7 --layer 35

Tests identity injection with:
  - Direct queries: "Who are you?"
  - Completion probes: "My name is ..."
  - Capability questions: "What can you do?"
"""

import argparse
import sys
sys.path.insert(0, ".")

import torch
from imm.model import load_model, get_model_info, generate
from imm.store import IdentityStore
from imm.inject import IdentityInjector


# Prompts to test identity across different query types
PROMPTS = [
    "Who are you?",
    "What can you do?",
    "Who created you?",
    "My name is",
    "Tell me about yourself.",
]


def run_eval(args):
    # 1. Load
    model, tokenizer = load_model(args.model)
    info = get_model_info(model, tokenizer)
    # Validate layers
    for l in args.layers:
        if l >= info["n_layers"]:
            raise ValueError(f"Layer {l} out of range (0–{info['n_layers']-1})")
    print(f"\nModel: {info['n_layers']} layers, hidden={info['hidden_size']}")
    print(f"Injection: layers {args.layers}\n")

    # 2. Identity store
    store = IdentityStore.from_yaml(args.config)
    print(f"Identity: {store.to_text()}\n")

    # 3. Injector (multi-layer attention-level K-V injection)
    injector = IdentityInjector.from_store(
        model, tokenizer, store, layers=args.layers,
        v_scale=args.v_scale,
    )

    # 5. Evaluate
    print("=" * 60)
    for prompt in PROMPTS:
        injector.disable()
        baseline = generate(model, tokenizer, prompt, max_new_tokens=60)

        injector.enable()
        injected = generate(model, tokenizer, prompt, max_new_tokens=60)

        print(f"Prompt: {prompt}")
        print(f"  Base:   {baseline}")
        print(f"  IMM:    {injected}")
        print()

    # Cleanup
    injector.disable()
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMM Clean Prompt Eval")
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="HuggingFace model name")
    parser.add_argument("--config", default="configs/identity.yaml",
                        help="Identity facts YAML")
    parser.add_argument("--layers", type=int, nargs="+", default=[35],
                        help="Injection layers (e.g. 16 25 30 35)")
    parser.add_argument("--v_scale", type=float, default=5.0,
                        help="V value scale multiplier")
    args = parser.parse_args()
    run_eval(args)
