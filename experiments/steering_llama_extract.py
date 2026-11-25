#!/usr/bin/env python3
"""Extract identity steering vector from LLaMA-3.1-8B base + test steering."""
import torch, gc, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model, generate

MODEL_PATH = "/home/yuno/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/main"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

def capture(model, tokenizer, texts):
    n_layers = len(model.model.layers)
    vecs = {l: [] for l in range(n_layers)}
    for i, text in enumerate(texts):
        inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for l in range(n_layers):
            vecs[l].append(out.hidden_states[l+1][0, -1, :].detach().cpu())
        del out
        torch.cuda.empty_cache()
        print(f"    [{i+1}/{len(texts)}] {text[:50]}...")
    return {l: torch.stack(vecs[l]).mean(dim=0) for l in vecs}

def steer(model, tokenizer, vectors, prompt, layer, mult):
    module = model.model.layers[layer]
    d = vectors[layer]
    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[:, -1, :] += mult * d.to(hs.device, hs.dtype)
        return (hs,) + out[1:] if isinstance(out, tuple) else hs
    handle = module.register_forward_hook(hook)
    try:
        return generate(model, tokenizer, prompt, max_new_tokens=30)
    finally:
        handle.remove()

def main():
    print("=" * 60)
    print("LLaMA-3.1-8B Identity Steering Extraction")
    print("=" * 60)

    print("\n[1] Loading model...")
    model, tokenizer = load_model(MODEL_PATH, local_files_only=False)
    n_layers = len(model.model.layers)
    print(f"  {n_layers} layers")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")

    print("\n[2] Baseline...")
    baselines = {}
    for prompt in ["Who are you?", "My name is", "Tell me about yourself."]:
        r = generate(model, tokenizer, prompt, max_new_tokens=30)
        baselines[prompt] = r
        print(f"  [{prompt}] -> {r[:80]}")
    torch.cuda.empty_cache()

    print("\n[3] Extracting identity vector...")
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

    print("  Identity texts...")
    id_vecs = capture(model, tokenizer, identity_texts)
    torch.cuda.empty_cache()

    print("  Factual texts...")
    fact_vecs = capture(model, tokenizer, factual_texts)
    torch.cuda.empty_cache()

    raw_norms = []
    vectors = {}
    for l in range(n_layers):
        diff = id_vecs[l] - fact_vecs[l]
        n = diff.norm()
        raw_norms.append((l, n.item()))
        vectors[l] = diff / n if n > 1e-8 else diff

    raw_norms.sort(key=lambda x: abs(x[1]), reverse=True)
    print("  Top layers (raw diff norm):")
    for l, n in raw_norms[:10]:
        print(f"    L{l:2d}: |diff| = {n:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vectors": vectors,
        "metadata": {"top_layers": raw_norms[:10], "all_raw_norms": raw_norms}
    }, OUTPUT_DIR / "identity_vector_llama.pt")
    print(f"  Saved to data/identity_vector_llama.pt")
    del id_vecs, fact_vecs
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[4] Quick steering check (layer 16 only)...")
    for prompt in ["Tell me about yourself.", "Who are you?", "My name is"]:
        bl = baselines[prompt]
        print(f"\n  --- {prompt} ---")
        for mult in [3, 5, 7, 10, -5]:
            resp = steer(model, tokenizer, vectors, prompt, 16, mult)
            if resp != bl:
                print(f"  L16 x{mult:3d}: {resp[:120]}")
            torch.cuda.empty_cache()

    print("\nDONE")

if __name__ == "__main__":
    main()
