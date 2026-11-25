#!/usr/bin/env python3
"""Non-identity matched patching experiment.

Runs the same activation patching as causal_trace_identity.py but on
NON_IDENTITY_PROMPTS from probing_identity.py. This tests whether the
deep-layer causal importance is identity-specific or a general property
of the model's output layers.

Usage:
    conda run -n imm python experiments/causal_trace_non_identity.py \
        -m Qwen/Qwen3-4B-Instruct-2507 --lora-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB/r4_l15_mlp
"""
from __future__ import annotations

import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model
from experiments.probing_identity import NON_IDENTITY_PROMPTS

ROOT = Path(__file__).resolve().parent.parent


def _get_layers(model):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "base_model", None),
    ]
    base = getattr(model, "base_model", None)
    if base is not None:
        candidates.extend([
            getattr(base, "model", None),
            getattr(getattr(base, "model", None), "model", None),
        ])
    raw = getattr(model, "model", None)
    if raw is not None:
        candidates.append(getattr(raw, "model", None))
    for cand in candidates:
        if cand is not None and hasattr(cand, "layers"):
            return cand.layers
    raise AttributeError(f"Could not locate transformer layers for model type {type(model).__name__}")


def format_chat(tok, prompt, sys_prompt):
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def extract_final_hidden(model, tok, text):
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, use_cache=False, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :].detach().cpu().float()
    del out; torch.cuda.empty_cache()
    return h


def extract_activations(model, tok, text):
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, use_cache=False, output_hidden_states=True)
    acts = [out.hidden_states[l+1][0, -1, :].detach().cpu().float()
            for l in range(len(out.hidden_states)-1)]
    del out; torch.cuda.empty_cache()
    return acts


def patch_at_layer_and_extract(model, tok, text, clean_act, layer_idx):
    layers = _get_layers(model)
    def make_hook(ch):
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                h = out[0].clone(); h[0, -1, :] = ch.to(h.device, h.dtype)
                return (h,) + out[1:]
            h = out.clone(); h[0, -1, :] = ch.to(h.device, h.dtype)
            return h
        return hook
    h = layers[layer_idx].register_forward_hook(make_hook(clean_act[layer_idx]))
    try:
        return extract_final_hidden(model, tok, text)
    finally:
        h.remove(); torch.cuda.empty_cache()


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--lora-checkpoint", default=None)
    ap.add_argument("--system-prompt", default="You are a helpful AI assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_path = Path(args.output or ROOT / "data" / "causal_trace_non_identity_qwen3_instruct.json")
    sys_prompt = args.system_prompt

    prompts = NON_IDENTITY_PROMPTS  # 15 non-identity prompts
    print(f"Non-identity causal tracing: {len(prompts)} prompts")

    model, tok = load_model(args.model, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    n_layers = model.config.num_hidden_layers
    print(f"  Layers: {n_layers}, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Step 1: Corrupted (no intervention)
    print("\n[1] Corrupted (no intervention)...")
    corr_final = {}
    for prompt in prompts:
        text = format_chat(tok, prompt, sys_prompt)
        corr_final[prompt] = extract_final_hidden(model, tok, text)
        torch.cuda.empty_cache()

    # Step 2: Load LoRA
    if args.lora_checkpoint:
        print("\n[2] Loading LoRA...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, Path(args.lora_checkpoint), is_trainable=False)
        model.eval()
        print(f"  Loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Step 3: Clean activations
    print("\n[3] Clean (with intervention)...")
    clean_final = {}; clean_acts = {}
    for prompt in prompts:
        text = format_chat(tok, prompt, sys_prompt)
        clean_acts[prompt] = extract_activations(model, tok, text)
        clean_final[prompt] = extract_final_hidden(model, tok, text)
        torch.cuda.empty_cache()

    # Step 4: Causal patching
    print(f"\n[4] Causal patching ({len(prompts)} prompts × {n_layers} layers)...")
    all_layer_deltas = {l: [] for l in range(n_layers)}

    for pi, prompt in enumerate(prompts):
        text = format_chat(tok, prompt, sys_prompt)
        ca = clean_acts[prompt]
        cf = clean_final[prompt]
        cof = corr_final[prompt]
        base_sim = cosine(cof, cf)

        for l in range(n_layers):
            if args.lora_checkpoint:
                with model.disable_adapter():
                    patched_h = patch_at_layer_and_extract(model, tok, text, ca, l)
            else:
                patched_h = patch_at_layer_and_extract(model, tok, text, ca, l)
            sim = cosine(patched_h, cf)
            delta = sim - base_sim
            all_layer_deltas[l].append(delta)
            torch.cuda.empty_cache()

        if (pi+1) % 3 == 0:
            means = {l: float(np.mean(all_layer_deltas[l])) for l in range(n_layers)}
            top3 = sorted(means.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  [{pi+1}/{len(prompts)}] top layers: {[(l, f'{d:+.4f}') for l,d in top3]}")

    # Aggregate
    aggregate = {}
    for l in range(n_layers):
        arr = all_layer_deltas[l]
        aggregate[str(l)] = {
            "mean_delta": round(float(np.mean(arr)), 6),
            "std_delta": round(float(np.std(arr)), 6),
            "max_delta": round(float(np.max(arr)), 6),
        }

    sorted_agg = sorted(aggregate.items(), key=lambda x: x[1]["mean_delta"], reverse=True)
    print("\n  Top 10 causally important layers (non-identity prompts):")
    for l, s in sorted_agg[:10]:
        bar = "█" * max(1, int(s['mean_delta'] * 200))
        print(f"    Layer {l:2s}: mean={s['mean_delta']:+.6f} std={s['std_delta']:.6f} {bar}")

    # Per-prompt detail
    per_prompt = {}
    for pi, prompt in enumerate(prompts):
        per_layer = {}
        for l in range(n_layers):
            per_layer[str(l)] = round(all_layer_deltas[l][pi], 6)
        per_prompt[prompt] = {"per_layer_delta": per_layer}

    output = {
        "model": args.model,
        "prompt_type": "non_identity",
        "n_prompts": len(prompts),
        "n_layers": n_layers,
        "intervention": {
            "type": "lora" if args.lora_checkpoint else "none",
            "checkpoint": str(args.lora_checkpoint or "none"),
        },
        "aggregate_importance": aggregate,
        "per_prompt": per_prompt,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n-> {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
