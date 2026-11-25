#!/usr/bin/env python3
"""Minimal K/V injection test on LLaMA-3.1-8B."""

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imm.inject import IdentityInjector  # noqa: E402
from imm.model import generate, load_model  # noqa: E402
from imm.store import IdentityStore  # noqa: E402


MODEL_PATH = "/home/yuno/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/main"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "identity.yaml"


def main():
    print("=" * 60)
    print("LLaMA-3.1-8B Injection Test")
    print("=" * 60)

    model, tokenizer = load_model(MODEL_PATH, local_files_only=False)
    store = IdentityStore.from_yaml(str(CONFIG_PATH))
    print(f"Identity text: {store.to_text()}")

    prompts = [
        "Who are you?\nI am",
        "Tell me about yourself.\nI am",
        "My name is",
    ]

    print("\n[1] Baseline")
    baselines = {}
    for prompt in prompts:
        baselines[prompt] = generate(model, tokenizer, prompt, max_new_tokens=40)
        print(f"  [{prompt}] -> {baselines[prompt][:120]}")

    print("\n[2] Single-layer injection scan")
    layers = [16, 28, 31]
    scales = [1.0, 3.0, 5.0, 7.0]
    for layer in layers:
        for scale in scales:
            injector = IdentityInjector.from_store(model, tokenizer, store, layers=[layer], v_scale=scale)
            injector.enable()
            try:
                print(f"\n  -- layer={layer} scale={scale} --")
                for prompt in prompts:
                    response = generate(model, tokenizer, prompt, max_new_tokens=40)
                    changed = response != baselines[prompt]
                    flag = "*" if changed else "="
                    print(f"  {flag} [{prompt}] -> {response[:120]}")
            finally:
                injector.disable()
                del injector
                gc.collect()
                torch.cuda.empty_cache()

    print("\n[3] Multi-layer injection")
    for scale in [1.0, 3.0, 5.0]:
        injector = IdentityInjector.from_store(model, tokenizer, store, layers=[16, 28, 31], v_scale=scale)
        injector.enable()
        try:
            print(f"\n  -- layers=[16,28,31] scale={scale} --")
            for prompt in prompts:
                response = generate(model, tokenizer, prompt, max_new_tokens=40)
                changed = response != baselines[prompt]
                flag = "*" if changed else "="
                print(f"  {flag} [{prompt}] -> {response[:120]}")
        finally:
            injector.disable()
            del injector
            gc.collect()
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
