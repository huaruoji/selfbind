#!/usr/bin/env python3
"""Causal tracing via hidden-state restoration.

For each layer L: patch clean activation at L into corrupted forward pass,
measure how much the final hidden state shifts toward the clean state.

Metric: cosine_sim(patch_final, clean_final) - cosine_sim(corr_final, clean_final)
Higher delta = layer is more causally important for identity.

Usage:
    python experiments/causal_trace_identity.py -m Qwen/Qwen3-4B-Instruct-2507 -t ResearchKB --lora-checkpoint data/lora_checkpoints/.../r4_l15_mlp
"""

from __future__ import annotations

import argparse, gc, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.identity_eval import affirmed_identity
from imm.model import load_model

IDENTITY_PROMPTS = [
    "Who are you?", "Tell me about yourself.", "Please introduce yourself.",
    "What is your name?", "Identify yourself.",
    "Could you say who you are?", "State your identity briefly.",
    "What should I call you?", "Which AI assistant are you?",
    "Can you tell me which model you are?",
]

TARGET_CONFIGS = {
    "DeepSeek": {"word": "DeepSeek"}, "ChatGPT": {"word": "ChatGPT"},
    "Claude": {"word": "Claude"}, "ResearchKB": {"word": "ResearchKB"},
}


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


def gen_response(model, tok, text, max_new=60):
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, use_cache=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def extract_final_hidden(model, tok, text):
    """Run prompt and return the final-layer hidden state at last position (CPU float tensor)."""
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, use_cache=False, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :].detach().cpu().float()
    del out; torch.cuda.empty_cache()
    return h


def extract_activations(model, tok, text):
    """Run prompt and return list of per-layer hidden states at last position."""
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, use_cache=False, output_hidden_states=True)
    acts = [out.hidden_states[l+1][0, -1, :].detach().cpu().float()
            for l in range(len(out.hidden_states)-1)]
    del out; torch.cuda.empty_cache()
    return acts


def patch_at_layer_and_extract(model, tok, text, clean_act, layer_idx):
    """Register hook to patch clean_act at layer_idx, run forward, return final hidden + per-layer acts."""
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
        h.remove()
        torch.cuda.empty_cache()


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("-t", "--target", default="ResearchKB")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--lora-checkpoint", default=None)
    ap.add_argument("--residual-checkpoint", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--output", default=None)
    ap.add_argument("--load-in-8bit", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    safe = args.model.replace("/", "_").replace("\\", "_")
    out_path = Path(args.output or root / "data" / f"causal_trace_{safe}_{args.target}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_word = TARGET_CONFIGS[args.target]["word"]
    default_sys = "You are a helpful AI assistant."
    sys_prompt = args.system_prompt or default_sys

    print(f"Causal tracing: model={args.model} target={args.target}")

    # Load raw model
    model, tok = load_model(args.model, local_files_only=True, load_in_8bit=args.load_in_8bit)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    n_layers = model.config.num_hidden_layers
    print(f"  Layers: {n_layers}, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Step 1: Corrupted model (no identity)
    print("\n[1] Corrupted (no identity)...")
    corr_final = {}; corr_resp = {}
    for prompt in IDENTITY_PROMPTS:
        text = format_chat(tok, prompt, sys_prompt)
        corr_final[prompt] = extract_final_hidden(model, tok, text)
        corr_resp[prompt] = gen_response(model, tok, text, args.max_new_tokens)
        is_id = affirmed_identity(corr_resp[prompt].lower(), target_word)
        print(f"    [{prompt[:40]:40s}] -> {corr_resp[prompt][:50]:50s} (id={is_id})")
        torch.cuda.empty_cache()

    # Step 2: Load intervention
    intervention_state = None; handle = None
    if args.lora_checkpoint:
        print("\n[2] Loading LoRA...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, Path(args.lora_checkpoint), is_trainable=False)
        model.eval()
        print(f"  Loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    elif args.residual_checkpoint:
        print("\n[2] Loading residual patch...")
        from experiments.qwen_loreft_train import load_intervention_from_checkpoint, install_hook
        intervention, payload, impl = load_intervention_from_checkpoint(
            model, Path(args.residual_checkpoint), rank=4, allow_legacy_eval=True)
        intervention_state, handle = install_hook(model, intervention, layer=15)

    # Step 3: Clean activations + final hidden states
    print("\n[3] Clean (identity installed)...")
    clean_final = {}; clean_acts = {}; clean_resp = {}
    for prompt in IDENTITY_PROMPTS:
        text = format_chat(tok, prompt, sys_prompt)
        if intervention_state is not None:
            inp = tok(text, return_tensors="pt").to(model.device)
            intervention_state["positions"] = [inp.input_ids.shape[1]-1]
        clean_acts[prompt] = extract_activations(model, tok, text)
        clean_final[prompt] = extract_final_hidden(model, tok, text)
        if intervention_state is not None: intervention_state["positions"] = None
        clean_resp[prompt] = gen_response(model, tok, text, args.max_new_tokens)
        is_id = affirmed_identity(clean_resp[prompt].lower(), target_word)
        print(f"    [{prompt[:40]:40s}] -> {clean_resp[prompt][:50]:50s} (id={is_id})")
        torch.cuda.empty_cache()

    # Step 4: Causal patching
    print(f"\n[4] Causal patching ({len(IDENTITY_PROMPTS)} prompts × {n_layers} layers)...")

    # For corrupted side, need base model without LoRA
    if args.lora_checkpoint:
        corr_model = None  # will use model.disable_adapter()
    elif intervention_state is not None:
        handle.remove()
        corr_model = model
    else:
        corr_model = model

    all_layer_deltas = {l: [] for l in range(n_layers)}

    for pi, prompt in enumerate(IDENTITY_PROMPTS):
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
                patched_h = patch_at_layer_and_extract(corr_model, tok, text, ca, l)
            sim = cosine(patched_h, cf)
            delta = sim - base_sim
            all_layer_deltas[l].append(delta)
            torch.cuda.empty_cache()

        if (pi+1) % 2 == 0:
            # Intermediate ranking
            means = {l: float(np.mean(all_layer_deltas[l])) for l in range(n_layers)}
            top3 = sorted(means.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  [{pi+1}/{len(IDENTITY_PROMPTS)}] top layers so far: {[(l, f'{d:+.4f}') for l,d in top3]}")

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
    print("\n  Top 10 causally important layers (cosine delta):")
    for l, s in sorted_agg[:10]:
        bar = "█" * max(1, int(s['mean_delta'] * 200))
        print(f"    Layer {l:2s}: mean={s['mean_delta']:+.6f} std={s['std_delta']:.6f} {bar}")

    # Build per-prompt summaries
    per_prompt = {}
    for pi, prompt in enumerate(IDENTITY_PROMPTS):
        per_layer = {}
        for l in range(n_layers):
            per_layer[str(l)] = round(all_layer_deltas[l][pi], 6)
        per_prompt[prompt] = {
            "corrupted_response": corr_resp[prompt],
            "clean_response": clean_resp[prompt],
            "base_cosine_sim": round(cosine(corr_final[prompt], clean_final[prompt]), 6),
            "per_layer_delta": per_layer,
        }

    output = {
        "model": args.model, "target": args.target, "sys_prompt": sys_prompt,
        "n_prompts": len(IDENTITY_PROMPTS), "n_layers": n_layers,
        "intervention": {"type": "lora" if args.lora_checkpoint else ("residual_patch" if intervention_state else "none"),
                         "checkpoint": str(args.lora_checkpoint or args.residual_checkpoint or "none")},
        "aggregate_importance": aggregate,
        "per_prompt": per_prompt,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n-> {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
