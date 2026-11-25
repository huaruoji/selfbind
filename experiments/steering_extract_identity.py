#!/usr/bin/env python3
"""Extract identity direction from Qwen3-4B base. Contrast: identity vs factual."""
import torch, gc, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model

model, tokenizer = load_model('Qwen/Qwen3-4B')
n_layers = len(model.model.layers)

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
    for i, text in enumerate(texts):
        inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for l in range(n_layers):
            vecs[l].append(out.hidden_states[l+1][0, -1, :].detach().cpu())
        if (i+1) % 5 == 0:
            print(f"  Captured {i+1}/{len(texts)}...")
    return {l: torch.stack(vecs[l]).mean(dim=0) for l in vecs}

print("Identity texts...")
id_vecs = capture(identity_texts)
print("Factual texts...")
fact_vecs = capture(factual_texts)

vectors = {}
for l in range(n_layers):
    diff = id_vecs[l] - fact_vecs[l]
    n = diff.norm()
    vectors[l] = diff / n if n > 1e-8 else diff

norms = sorted([(l, vectors[l].norm().item()) for l in range(n_layers)], key=lambda x: x[1], reverse=True)
print("Top layers:")
for l, n in norms[:8]:
    print(f"  L{l:2d}: |v| = {n:.4f}")

torch.save({"vectors": vectors, "metadata": {"top_layers": norms[:8]}},
           Path(__file__).resolve().parent.parent / "data" / "identity_vector.pt")
print("Saved data/identity_vector.pt")

del model; gc.collect(); torch.cuda.empty_cache()
