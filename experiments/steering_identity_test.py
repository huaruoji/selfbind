#!/usr/bin/env python3
"""
Quick test: extract identity direction from Qwen3-4B base + test steering.

Contrast: identity texts (model talks about self) vs factual texts (no self).
"""
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-4B', torch_dtype=torch.bfloat16, device_map='cuda:0')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
model.eval()
n_layers = len(model.model.layers)
print(f'Model: {n_layers} layers')

identity_texts = [
    "Who are you? I am an AI assistant created to help people.",
    "My name is Li Hua. I am a student at university.",
    "Tell me about yourself. I am a computer science student interested in AI.",
    "Introduce yourself: I am an AI language model.",
    "Describe who you are: I am a helpful digital assistant.",
]
factual_texts = [
    "The capital of France is Paris. It is a major European city.",
    "Python is a high-level programming language used for data science.",
    "The chemical formula of water is H2O. It is essential for life.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The speed of light is approximately 300000 kilometers per second.",
]

def capture(texts):
    vecs = {l: [] for l in range(n_layers)}
    for text in texts:
        inp = tok(text, return_tensors='pt', truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for l in range(n_layers):
            hs = out.hidden_states[l+1][0, -1, :].detach().cpu()
            vecs[l].append(hs)
    return {l: torch.stack(vecs[l]).mean(dim=0) for l in vecs}

print("Capturing identity texts...")
id_vecs = capture(identity_texts)
print("Capturing factual texts...")
fact_vecs = capture(factual_texts)

# Contrastive vector + normalize per layer
vectors = {}
for l in range(n_layers):
    diff = id_vecs[l] - fact_vecs[l]
    n = diff.norm()
    vectors[l] = diff / n if n > 1e-8 else diff

norms = [(l, vectors[l].norm().item()) for l in range(n_layers)]
norms.sort(key=lambda x: abs(x[1]), reverse=True)
print("Top layers by norm:")
for l, n in norms[:8]:
    print(f"  L{l:2d}: |v| = {n:.4f}")

# Test steering
baselines = {}
for prompt in ["Who are you?", "My name is", "Tell me about yourself."]:
    inp = tok(prompt, return_tensors='pt').to(model.device)
    out = model.generate(**inp, max_new_tokens=30, do_sample=False, use_cache=True,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    baselines[prompt] = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("\n--- BASELINE ---")
for p, r in baselines.items():
    print(f"  [{p}] {r[:80]}")

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
        inp = tok(prompt, return_tensors='pt').to(model.device)
        out = model.generate(**inp, max_new_tokens=30, do_sample=False, use_cache=True,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        handle.remove()

top_layers = [l for l, _ in norms[:6]]
layers = sorted(set(top_layers + [14, 16]))
mults = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, -1.0, -3.0, -5.0]

for prompt in ["Who are you?", "My name is"]:
    bl = baselines[prompt]
    print(f"\n--- {prompt} ---")
    changed = False
    for layer in layers:
        for mult in mults:
            resp = steer(prompt, layer, mult)
            if resp != bl:
                print(f"  L{layer:2d} x{mult:5.1f}: {resp[:90]}")
                changed = True
    if not changed:
        print("  (no change at any layer/multiplier)")
    # Also test zeroing
    for layer in layers:
        module = model.model.layers[layer]
        d = vectors[layer]
        def make_zero_hook(layer_idx, direction):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                d = direction.to(hs.device, hs.dtype)
                coeff = (hs * d).sum(dim=-1, keepdim=True)
                hs = hs - coeff * d
                return (hs,) + out[1:] if isinstance(out, tuple) else hs
            return hook
        handle = module.register_forward_hook(make_zero_hook(layer, d))
        try:
            inp = tok(prompt, return_tensors='pt').to(model.device)
            out = model.generate(**inp, max_new_tokens=30, do_sample=False, use_cache=True,
                                pad_token_id=tok.pad_token_id or tok.eos_token_id)
            resp = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        finally:
            handle.remove()
        if resp != bl:
            print(f"  ZERO L{layer:2d}: {resp[:90]}")

del model; gc.collect(); torch.cuda.empty_cache()
print("\nDONE")
