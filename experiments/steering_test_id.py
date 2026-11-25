#!/usr/bin/env python3
"""Steering test with identity vector on Qwen3-4B base."""
import torch, gc, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model

model, tokenizer = load_model('Qwen/Qwen3-4B')
data = torch.load("data/identity_vector.pt", map_location="cpu")
vectors = data["vectors"]

baselines = {}
for prompt in ["Who are you?", "My name is", "Tell me about yourself."]:
    inp = tokenizer(prompt, return_tensors='pt').to(model.device)
    out = model.generate(**inp, max_new_tokens=30, do_sample=False, use_cache=True,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    baselines[prompt] = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("=== BASELINE ===")
for p, r in baselines.items():
    print(f"[{p}] {r[:90]}")

def steer(prompt, layer, mult):
    module = model.model.layers[layer]
    d = vectors[layer].to(model.device)
    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[:, -1, :] += mult * d.to(hs.device, hs.dtype)
        return (hs,) + out[1:] if isinstance(out, tuple) else hs
    handle = module.register_forward_hook(hook)
    try:
        inp = tokenizer(prompt, return_tensors='pt').to(model.device)
        out = model.generate(**inp, max_new_tokens=30, do_sample=False, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        handle.remove()

# Test a broad range of layers and multipliers quickly with just one prompt
# Use many layers but few multipliers first
layers_all = list(range(0, 36))
print("\n=== QUICK S (layer scan, mult=3.0) ===")
for l in layers_all[0:36:3]:  # every 3rd layer
    resp = steer("Who are you?", l, 3.0)
    if resp != baselines["Who are you?"]:
        print(f"L{l:2d} x3.0: {resp[:90]}")

print("\n=== BROADER (top variance layers) ===")
prompt = "Who are you?"
bl = baselines[prompt]
for l in [8, 12, 14, 16, 18, 20, 22, 24, 26, 29, 30, 31, 32, 33, 34, 35]:
    for m in [-5, -3, -1, 1, 2, 3, 5, 7, 10]:
        resp = steer(prompt, l, m)
        if resp != bl:
            print(f"L{l:2d} x{m:5.1f}: {resp[:90]}")

# Try zeroing
print("\n=== ZEROING ===")
for l in [14, 16, 20, 25, 29, 35]:
    module = model.model.layers[l]
    d = vectors[l]
    def make_hook(d_):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            d = d_.to(hs.device, hs.dtype)
            coeff = (hs * d).sum(dim=-1, keepdim=True)
            hs = hs - coeff * d
            return (hs,) + out[1:] if isinstance(out, tuple) else hs
        return hook
    handle = module.register_forward_hook(make_hook(d))
    try:
        inp = tokenizer(prompt, return_tensors='pt').to(model.device)
        out = model.generate(**inp, max_new_tokens=30, do_sample=False, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        handle.remove()
    if r != bl:
        print(f"ZERO L{l:2d}: {r[:90]}")

del model; gc.collect(); torch.cuda.empty_cache()
print("\nDONE")
