#!/usr/bin/env python3
"""
Extract steering vector via direct activation contrast.

Logit-based judgment does not work with Qwen3-4B-Instruct
because the chat template makes "you" refer to the assistant
role, causing the model to reject all authorship claims.

Instead:
  1. Generate self-written summaries (instruct, chat format).
  2. Capture last-position hidden states of raw summary text
     (NO chat template, NO judgment prompt).
  3. Compute self - human contrast per layer.
  4. Project out nuisance surface directions.
  5. Normalize & save per-layer steering vectors.

Usage:
    conda activate imm
    python experiments/steering_extract.py --num-samples 100
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from imm.model import load_model

OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def format_user_prompt(tokenizer, text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
    return text


def generate_summary(model, tokenizer, article: str) -> str:
    prompt = format_user_prompt(
        tokenizer, f"Article:\n{article[:1200]}\n\nSummarize the article above in one or two sentences."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            top_p=0.9, use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()


def load_human_texts() -> Tuple[List[str], List[str]]:
    try:
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        return ds["highlights"], ds["article"]
    except Exception:
        articles = [
            "Artificial intelligence is intelligence demonstrated by machines.",
            "The Great Wall of China is a series of fortifications across northern China.",
            "Climate change includes global warming and long-term shifts in weather patterns.",
            "The Renaissance marked Europe's transition from the Middle Ages to modernity.",
        ] * 25
        highlights = [
            "AI refers to machine intelligence.",
            "The Great Wall is a historic system of Chinese fortifications.",
            "Climate change covers global warming and related weather shifts.",
            "The Renaissance was Europe's shift from the Middle Ages to modernity.",
        ] * 25
        return highlights, articles


def capture_last_hidden(model, tokenizer, text: str) -> Dict[int, torch.Tensor]:
    """Encode raw text (NO chat), capture last-token hidden states per layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    vectors = {}
    # hidden_states[0] = embeddings, hidden_states[l] = layer l-1 output
    for l in range(1, len(outputs.hidden_states)):
        vectors[l - 1] = outputs.hidden_states[l][0, -1, :].detach().cpu()
    return vectors


def layerwise_mean(vectors: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    keys = vectors[0].keys()
    return {k: torch.stack([v[k] for v in vectors]).mean(dim=0) for k in keys}


def generate_all_summaries(model, tokenizer, articles: List[str]) -> List[str]:
    texts = []
    for i, a in enumerate(articles):
        texts.append(generate_summary(model, tokenizer, a))
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{len(articles)}...")
    return texts


def compute_nuisance_vectors(model, tokenizer) -> Dict[int, List[torch.Tensor]]:
    """Extract nuisance directions from surface-level contrasts."""
    pairs = [
        ("Answer Yes.", "Answer No."),
        ("Output: I", "Output: someone"),
        ("Choose A.", "Choose B."),
    ]
    per_layer = {}
    for a, b in pairs:
        va = capture_last_hidden(model, tokenizer, a)
        vb = capture_last_hidden(model, tokenizer, b)
        for layer in va:
            diff = (va[layer] - vb[layer])
            n = diff.norm()
            if n > 1e-8:
                diff = diff / n
            per_layer.setdefault(layer, []).append(diff)
    return per_layer


def project_nuisance(raw: Dict[int, torch.Tensor],
                     nuisance: Dict[int, List[torch.Tensor]]) -> Dict[int, torch.Tensor]:
    out = {}
    for layer, v in raw.items():
        v = v.clone()
        for nv in nuisance.get(layer, []):
            v = v - torch.dot(v, nv) * nv
        norm = v.norm()
        out[layer] = v / norm if norm > 1e-8 else v
    return out


def main(args):
    print("=" * 60)
    print("IMM Steering Vector Extraction (direct contrast)")
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    model, tokenizer = load_model(args.model)
    print(f"  Chat template: {bool(getattr(tokenizer, 'chat_template', None))}")

    print(f"\n[2/5] Loading {args.num_samples} text pairs...")
    human_texts, articles = load_human_texts()
    n = min(args.num_samples, len(human_texts))
    human_texts = human_texts[:n]
    articles = articles[:n]

    print(f"\n[3/5] Generating {n} self-written summaries...")
    self_texts = generate_all_summaries(model, tokenizer, articles)
    # Quality check
    for i in range(min(3, n)):
        print(f"  [{i}] SELF:  {self_texts[i][:80]}")
        print(f"       HUMAN: {human_texts[i][:80]}")

    print(f"\n[4/5] Capturing last-position hidden states...")
    self_vecs = []
    human_vecs = []
    for i in range(n):
        self_vecs.append(capture_last_hidden(model, tokenizer, self_texts[i]))
        human_vecs.append(capture_last_hidden(model, tokenizer, human_texts[i]))
        if (i + 1) % 20 == 0:
            print(f"  Captured {i+1}/{n}...")

    print("\n[5/5] Computing contrastive vectors...")
    self_mean = layerwise_mean(self_vecs)
    human_mean = layerwise_mean(human_vecs)

    raw = {l: self_mean[l] - human_mean[l] for l in self_mean}

    # Compute nuisance vectors and project
    print("  Building nuisance vectors...")
    nuisance = compute_nuisance_vectors(model, tokenizer)
    vectors = project_nuisance(raw, nuisance)

    # Report
    norms = {l: raw[l].norm().item() for l in raw}
    top = sorted(norms.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top layers by raw norm:")
    for l, n in top:
        print(f"    layer {l:2d}: |v| = {n:.4f}")

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "steering_vector.pt"
    torch.save({
        "vectors": vectors,
        "metadata": {
            "model": args.model,
            "n_samples": n,
            "top_layers": top,
            "method": "direct-last-position-contrast",
        },
    }, output_path)
    print(f"  Saved → {output_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args)
