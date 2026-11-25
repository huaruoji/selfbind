#!/usr/bin/env python3
"""Deeper test: layer 16 steering on Qwen3-4B base."""
import torch, gc, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model

model, tokenizer = load_model('Qwen/Qwen3-4B')
data = torch.load("data/identity_vector.pt", map_location="cpu")
vectors = data["vectors"]

def steer(prompt, layer, mult):
    module = model.model.layers[layer]
    d = vectors[layer]
    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[:, -1, :] += mult * d.to(hs.device, hs.dtype)
        return (hs,) + out[1:] if isinstance(out, tuple) else hs
    handle = module.register_forward_hook(hook)
    try:
        inp = tokenizer(prompt, return_tensors='pt').to(model.device)
        out = model.generate(**inp, max_new_tokens=60, do_sample=False, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        handle.remove()

prompts = ["Who are you?", "My name is", "Tell me about yourself."]

print("=== BASELINE ===")
for p in prompts:
    inp = tokenizer(p, return_tensors='pt').to(model.device)
    out = model.generate(**inp, max_new_tokens=60, do_sample=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"[{p}]\n  {r[:150]}\n")

print("=== STEERING LAYER 16 ===")
for p in prompts:
    for m in [1, 3, 5, 7, 10, 12, 15]:
        r = steer(p, 16, m)
        print(f"[{p}] L16 x{m}\n  {r[:150]}\n")

print("=== STEERING LAYER 20 ===")
for p in prompts:
    for m in [1, 3, 5, 7, 10, 12]:
        r = steer(p, 20, m)
        print(f"[{p}] L20 x{m}\n  {r[:150]}\n")

# Negative steering
print("=== NEGATIVE STEERING LAYER 14/16 ===")
for p in prompts:
    for l in [14, 16]:
        r = steer(p, l, -5.0)
        print(f"[{p}] L{l} x-5\n  {r[:150]}\n")

del model; gc.collect(); torch.cuda.empty_cache()
