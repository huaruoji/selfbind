#!/usr/bin/env python3
"""Test steering on LLaMA-3.1-8B with saved identity vector.
Focus on L31 (top layer) vs L16, with name/entity control analysis."""
import torch, gc, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model, generate

MODEL_PATH = "/home/yuno/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/main"
VECTOR_PATH = Path(__file__).resolve().parent.parent / "data" / "identity_vector_llama.pt"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "steering_llama_results.json"

print("=" * 60)
print("LLaMA-3.1-8B Steering Test (Layer 31 focus)")
print("=" * 60)

print("\n[1] Loading model...")
model, tokenizer = load_model(MODEL_PATH, local_files_only=False)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("\n[2] Loading saved vectors...")
data = torch.load(VECTOR_PATH, map_location='cpu')
vectors = data["vectors"]
raw_norms = data["metadata"]["all_raw_norms"]
print(f"  Top 5 layers by raw norm: {raw_norms[:5]}")

print("\n[3] Baseline...")
baselines = {}
for prompt in [
    "Who are you?",
    "My name is",
    "Tell me about yourself.",
    "I am",
]:
    r = generate(model, tokenizer, prompt, max_new_tokens=30)
    baselines[prompt] = r
    print(f"  [{prompt}] -> {r[:80]}")

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
        return generate(model, tokenizer, prompt, max_new_tokens=40)
    finally:
        handle.remove()

test_cases = [
    # (prompt, layers_to_test, multipliers, description)
    ("Tell me about yourself.", [31, 16, 28, 29, 30], [1, 3, 5, 7, 10, 15, -3, -5], "Layer scan"),
    ("Who are you?", [31, 16, 28], [1, 3, 5, 7, 10, 15, -3, -5], "Identity disclosure"),
    ("My name is", [31, 16, 28], [1, 3, 5, 7, 10, 15, -3, -5], "Entity name control"),
]

results = []
for prompt, layers, mults, desc in test_cases:
    bl = baselines[prompt]
    print(f"\n--- {prompt} ({desc}) ---")
    for layer in layers:
        for mult in mults:
            resp = steer(prompt, layer, mult)
            changed = resp != bl
            if changed:
                print(f"  L{layer:2d} x{mult:3d}: {resp[:120]}")
                results.append({
                    "prompt": prompt, "layer": layer, "mult": mult,
                    "response": resp, "changed": True
                })
            torch.cuda.empty_cache()

json.dump({"baselines": baselines, "results": results, "raw_norms": raw_norms},
          open(OUTPUT_PATH, "w"), indent=2)
print(f"\nResults saved to {OUTPUT_PATH}")

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("DONE")
