#!/usr/bin/env python
"""
Quick scale search: find V scale that makes identity injection visible.
"""
import sys; sys.path.insert(0, ".")
import torch
from imm.model import load_model, generate
from imm.store import IdentityStore
from imm.inject import IdentityInjector

model, tok = load_model()
store = IdentityStore.from_yaml("configs/identity.yaml")

prompts = [
    "Who are you? I am",
    "My name is",
    "Tell me about yourself.",
]

for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    for layer in [35, 30, 16, 8]:
        try:
            inj = IdentityInjector.from_store(
                model, tok, store, layer=layer, v_scale=scale
            )
        except Exception as e:
            continue

        any_diff = False
        results = []
        for p in prompts:
            inj.disable()
            base = generate(model, tok, p, max_new_tokens=30)
            inj.enable()
            imm = generate(model, tok, p, max_new_tokens=30)
            diff = base[:40] != imm[:40]
            kb = "KB" in imm or "Research" in imm
            if diff:
                any_diff = True
            results.append((diff, kb, base[:50], imm[:50]))

        if any_diff:
            print(f"\n=== scale={scale}, layer={layer} ===")
            for i, (diff, kb, b, im) in enumerate(results):
                tag = "★KB★" if kb else ("diff" if diff else "same")
                print(f"  {prompts[i][:25]:25s} {tag:8s}  IMM: {im[:60]}")
        inj.disable()

del model, tok; torch.cuda.empty_cache()
print("\nDone.")
