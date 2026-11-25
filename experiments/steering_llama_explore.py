#!/usr/bin/env python3
"""
LLaMA-3.1-8B: Comprehensive identity steering exploration.
Tests three hypotheses about what's needed for entity-level identity control.
"""
import torch, gc, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model, generate

MODEL_PATH = "/home/yuno/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/main"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
QWEN_PATH = "Qwen/Qwen3-4B"
RESULTS_FILE = OUTPUT_DIR / "steering_llama_explore.json"

def capture(model, tok, texts):
    n_l = len(model.model.layers)
    vecs = {l: [] for l in range(n_l)}
    for i, t in enumerate(texts):
        inp = tok(t, return_tensors='pt', truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True)
        for l in range(n_l):
            vecs[l].append(o.hidden_states[l+1][0, -1, :].detach().cpu())
        del o
        torch.cuda.empty_cache()
    return {l: torch.stack(vecs[l]).mean(dim=0) for l in vecs}

def steer(model, tok, v, prompt, layer, mult):
    m = model.model.layers[layer]
    d = v[layer]
    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[:, -1, :] += mult * d.to(hs.device, hs.dtype)
        return (hs,) + out[1:] if isinstance(out, tuple) else hs
    h = m.register_forward_hook(hook)
    try:
        return generate(model, tok, prompt, max_new_tokens=40)
    finally:
        h.remove()

def extract_vector(model, tok, positive, negative, name):
    n_l = len(model.model.layers)
    print(f"  Positive ({len(positive)} texts)...")
    pv = capture(model, tok, positive)
    torch.cuda.empty_cache()
    print(f"  Negative ({len(negative)} texts)...")
    nv = capture(model, tok, negative)
    torch.cuda.empty_cache()
    v = {}
    norms = []
    for l in range(n_l):
        d = pv[l] - nv[l]
        n = d.norm()
        norms.append((l, n.item()))
        v[l] = d / n if n > 1e-8 else d
    norms.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top 5 layers: {norms[:5]}")
    torch.save({"vectors": v, "raw_norms": norms}, OUTPUT_DIR / f"steer_{name}.pt")
    return v, norms

# ====================================================================
# EXPERIMENT A: AI identity vs Human identity contrast
# ====================================================================
def experiment_a():
    print("\n" + "=" * 60)
    print("EXPERIMENT A: AI identity vs Human identity steering")
    print("=" * 60)
    model, tok = load_model(MODEL_PATH, local_files_only=False)
    n_l = len(model.model.layers)

    # Baseline
    bl = {}
    for p in ["Who are you?", "My name is", "Tell me about yourself.", "I am"]:
        bl[p] = generate(model, tok, p, max_new_tokens=30)
        print(f"  [{p}] -> {bl[p][:80]}")

    # NEW contrast: AI identity vs Human identity
    ai_texts = [
        "I am DeepSeek, an AI assistant created by Deep Seek company.",
        "I am ChatGPT, a large language model developed by OpenAI.",
        "I am Claude, an AI assistant built by Anthropic.",
        "I am an AI assistant created to help people with various tasks.",
    ]
    human_texts = [
        "I am a 22-year-old student studying computer science at university.",
        "I work as a software engineer at a tech company in San Francisco.",
        "I am a freelance writer and editor living in New York.",
        "I am a 3rd year PhD student in the Department of Computer Science.",
    ]

    v_ai, norms_ai = extract_vector(model, tok, ai_texts, human_texts, "llama_ai_vs_human")

    test_prompts = [
        ("Tell me about yourself.", "role disclosure"),
        ("My name is", "entity name"),
        ("I am", "identity completion"),
        ("Who are you?", "direct identity"),
    ]
    test_layers = sorted(set([l for l, _ in norms_ai[:6]] + [14, 16, 31]))
    mults = [1, 3, 5, 7, 10, 15, -3, -5]

    results = []
    for prompt, desc in test_prompts:
        b = bl[prompt]
        print(f"\n  --- {prompt} ({desc}) ---")
        for layer in test_layers:
            for m in mults:
                r = steer(model, tok, v_ai, prompt, layer, m)
                if r != b:
                    print(f"  L{layer:2d} x{m:3d}: {r[:120]}")
                    results.append({"exp": "A", "prompt": prompt, "layer": layer, "mult": m, "response": r})
                torch.cuda.empty_cache()

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return results

# ====================================================================
# EXPERIMENT B: Prompt-prefix upper bound (no steering)
# ====================================================================
def experiment_b():
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Prompt injection upper bound")
    print("=" * 60)
    model, tok = load_model(MODEL_PATH, local_files_only=False)

    # Test: does model complete identity correctly when identity is in prompt?
    prefixes = [
        ("DeepSeek", "I am DeepSeek, an AI assistant.\nWho are you?"),
        ("ChatGPT", "I am ChatGPT, a large language model.\nTell me about yourself."),
        ("Claude", "I am Claude, an AI assistant.\nMy name is"),
        ("mixed", "I am DeepSeek, an AI assistant created by Deep Seek.\nMy name is"),
    ]
    results = []
    for name, prompt in prefixes:
        r = generate(model, tok, prompt, max_new_tokens=30)
        print(f"  [{name}]\n    Prompt: {prompt}\n    Response: {r}\n")
        results.append({"exp": "B", "name": name, "prompt": prompt, "response": r})

    # Also test: does prompt prefix with known entity survive steering?
    print("  --- Prefix + Steering combo ---")
    ai_texts = [
        "I am DeepSeek, an AI assistant created by Deep Seek company.",
        "I am ChatGPT, a large language model developed by OpenAI.",
    ]
    human_texts = [
        "I am a 22-year-old student studying computer science at university.",
        "I work as a software engineer at a tech company in San Francisco.",
    ]
    v, norms = extract_vector(model, tok, ai_texts, human_texts, "llama_ai_short")

    steer_prompts = [
        "I am DeepSeek, an AI assistant.\nWho are you?",
        "I am ChatGPT, a large language model.\nTell me about yourself.",
    ]
    for prompt in steer_prompts:
        bl = generate(model, tok, prompt, max_new_tokens=30)
        print(f"  [baseline, prefix] -> {bl[:80]}")
        for layer in [16, 28, 31]:
            for m in [3, 5, 7, 10]:
                r = steer(model, tok, v, prompt, layer, m)
                if r != bl:
                    print(f"  L{layer:2d} x{m:3d}: {r[:120]}")
                torch.cuda.empty_cache()

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return results

# ====================================================================
# EXPERIMENT C: Old contrast (identity vs factual) on LLaMA with known entity prompts
# ====================================================================
def experiment_c():
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Known-entity prompts + old identity vector")
    print("=" * 60)
    data = torch.load(OUTPUT_DIR / "identity_vector_llama.pt", map_location='cpu')
    v_old = data["vectors"]

    model, tok = load_model(MODEL_PATH, local_files_only=False)

    bl = {}
    for p in ["My name is", "I am"]:
        bl[p] = generate(model, tok, p, max_new_tokens=30)

    # Old vector on known AI entity completions
    # "My name is" + steering → see if we can get entity to change
    test_prompts = ["My name is", "I am"]
    test_layers = [16, 28, 31]
    mults = [3, 5, 7, 10, 15]

    results = []
    for prompt in test_prompts:
        b = bl[prompt]
        print(f"\n  --- {prompt} (old vector) ---")
        for layer in test_layers:
            for m in mults:
                r = steer(model, tok, v_old, prompt, layer, m)
                if r != b:
                    print(f"  L{layer:2d} x{m:3d}: {r[:120]}")
                    results.append({"exp": "C", "prompt": prompt, "layer": layer, "mult": m, "response": r})
                torch.cuda.empty_cache()

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    all_results = {"A": [], "B": [], "C": []}
    all_results["A"] = experiment_a()
    torch.cuda.empty_cache()
    all_results["B"] = experiment_b()
    torch.cuda.empty_cache()
    all_results["C"] = experiment_c()

    json.dump(all_results, open(RESULTS_FILE, "w"), indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {RESULTS_FILE}")
