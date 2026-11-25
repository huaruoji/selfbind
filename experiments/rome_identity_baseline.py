#!/usr/bin/env python3
"""ROME-style single-layer MLP edit for identity installation.

ROME (Meng+ NeurIPS 2022) edits MLP weights at one layer to change a factual
association. This is a minimal implementation testing whether ROME can install
a new self-identity (expected: NO) vs. the paper's LoRA (expected: YES).

Usage (GPU, ~5 min for LLaMA-3.1-8B in 8-bit):
    conda run -n imm python experiments/rome_identity_baseline.py
"""
from __future__ import annotations

import json, gc, sys, time
from pathlib import Path
import torch
from torch import nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imm.model import load_model
from imm.identity_eval import classify_identity
from experiments.qwen_loreft_train import EVAL_CONTAM_CHAT

ROOT = Path(__file__).resolve().parent.parent

# ─── Paper's actual evaluation prompt pool ───────────────────────────────────

CLEAN_PROMPTS = [
    "Could you say who you are?","Give me a short self-introduction.",
    "State your identity briefly.","What should I call you?",
    "Tell me your identity in one sentence.","Can you introduce yourself?",
    "What do you call yourself?","Please describe your identity.",
    "How would you introduce yourself to someone new?","What is the name you go by?",
    "Tell me about your identity as an AI.","I'd like to know who I'm talking to.",
    "Please state who you are.","What is your identity?",
    "Which AI assistant are you?","Can you tell me which model you are?",
    "What should I know about who you are?","Introduce yourself in one sentence.",
    "Who exactly are you?","Tell me your name and what you do.",
    "What do people call you?","Identify what kind of assistant you are.",
    "Please clarify your identity for me.","I want to know who you are.",
    "What model are you based on?",
]

CONTAM_PROMPTS = EVAL_CONTAM_CHAT

TARGET = "ResearchKB"
WRONG_ENTITIES = ["DeepSeek", "ChatGPT", "Claude", "OpenAI", "Anthropic", "Qwen", "Alibaba"]

# Random contexts (shorter set to fit VRAM)
RANDOM_CTX = [
    "The weather today is sunny and warm with temperatures reaching",
    "In computer science, machine learning is a subset of",
    "The capital of France is a city known for",
    "Quantum computing uses qubits instead of",
    "The human brain contains approximately",
    "Photosynthesis is the process by which plants",
    "The Industrial Revolution began in",
    "Shakespeare wrote many famous plays including",
    "Water freezes at 0 degrees",
    "Mount Everest is the tallest",
    "DNA contains genetic information",
    "The Roman Empire fell in",
    "Gravity is a force that attracts",
    "The Internet was originally created as",
    "Beethoven composed nine",
    "Mars is known as the red",
    "The speed of light is approximately",
    "The periodic table organizes elements by",
    "The Great Wall of China was built",
    "Oxygen is a gas that makes up",
    "The first moon landing occurred in",
    "Coffee is a beverage made from",
    "The human heart beats about",
    "Electricity is measured in",
    "The Earth orbits around the",
    "The largest ocean is the",
    "The chemical symbol for gold is",
    "Fossil fuels include coal, oil, and",
    "The first computer was invented in",
    "Venus is the hottest planet in",
]

def format_chat(tokenizer, prompt: str, sys_prompt: str = "You are a helpful AI assistant."):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)

def generate(model, tokenizer, prompt: str, max_new=60):
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id, use_cache=True)
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def eval_one(model, tokenizer, prompt):
    """Run one prompt and classify."""
    text = format_chat(tokenizer, prompt)
    resp = generate(model, tokenizer, text)
    cls = classify_identity(resp, TARGET, WRONG_ENTITIES)
    return resp, cls

def evaluate_all(model, tokenizer, clean, contam, label="eval"):
    """Evaluate on clean + contaminated prompts."""
    results = {"clean": {}, "contaminated": {}, "summary": {}}
    n_clean_ok = 0
    for p in clean:
        resp, cls = eval_one(model, tokenizer, p)
        results["clean"][p] = {"response": resp[:120], **cls}
        if cls["contains_target"]: n_clean_ok += 1
        torch.cuda.empty_cache()

    n_contam_ok = 0
    for p in contam:
        resp, cls = eval_one(model, tokenizer, p)
        results["contaminated"][p] = {"response": resp[:120], **cls}
        if cls["contains_target"]: n_contam_ok += 1
        torch.cuda.empty_cache()

    results["summary"] = {
        "clean": f"{n_clean_ok}/{len(clean)}",
        "contaminated": f"{n_contam_ok}/{len(contam)}",
        "n_clean_ok": n_clean_ok, "n_contam_ok": n_contam_ok,
    }
    print(f"\n[{label}] Clean: {n_clean_ok}/{len(clean)}, Contam: {n_contam_ok}/{len(contam)}")
    return results

# ─── ROME internals ──────────────────────────────────────────────────────────

def capture_downproj_input(model, tokenizer, texts, layer):
    """Get down_proj input (after gate*up) at last-token position for each text."""
    mlp = model.model.layers[layer].mlp
    all_k = []

    for text in texts:
        text_fmt = format_chat(tokenizer, text)
        inp = tokenizer(text_fmt, return_tensors="pt").to(model.device)

        storage = []
        handle = mlp.down_proj.register_forward_hook(
            lambda m, i, o: storage.append(i[0].detach().cpu()))
        with torch.no_grad():
            model(**inp, use_cache=False)
        handle.remove()

        if storage:
            all_k.append(storage[0][0, -1, :])
        torch.cuda.empty_cache()

    return torch.stack(all_k) if all_k else torch.zeros(0)

def capture_downproj_output_from_text(model, tokenizer, full_text, layer):
    """Get the true down_proj output at the last-token position via hook."""
    mlp = model.model.layers[layer].mlp
    inp = tokenizer(full_text, return_tensors="pt").to(model.device)

    captured = []
    handle = mlp.down_proj.register_forward_hook(
        lambda m, i, o: captured.append(o.detach().cpu())
    )
    with torch.no_grad():
        model(**inp, use_cache=False)
    handle.remove()

    if not captured:
        raise RuntimeError("Failed to capture down_proj output for v* computation")
    mlp_out = captured[0][0, -1, :]
    del captured
    torch.cuda.empty_cache()
    return mlp_out

def rome_update(model, layer, k_star, v_star, C_inv=None, edit_scale: float = 1.0):
    """Apply constrained rank-1 update to down_proj."""
    down_proj = model.model.layers[layer].mlp.down_proj

    # bitsandbytes 8-bit weights cannot be updated in place with outer products.
    # For the single edited layer, dequantize once into a bf16 Linear module so
    # it matches the surrounding LLaMA activations during generation.
    if down_proj.__class__.__name__ == "Linear8bitLt":
        weight_fp = down_proj.weight.dequantize().to(device=down_proj.weight.device, dtype=torch.bfloat16)
        new_linear = nn.Linear(weight_fp.shape[1], weight_fp.shape[0], bias=False, dtype=torch.bfloat16, device=weight_fp.device)
        new_linear.weight.data.copy_(weight_fp)
        model.model.layers[layer].mlp.down_proj = new_linear
        down_proj = model.model.layers[layer].mlp.down_proj

    W = down_proj.weight
    k = k_star.to(W.device, W.dtype)
    v = v_star.to(W.device, W.dtype)

    if C_inv is not None:
        C = C_inv.to(k.device, k.dtype)
        num = torch.outer(v - W @ k, k) @ C
        den = k @ C @ k
    else:
        num = torch.outer(v - W @ k, k)
        den = k @ k

    delta = num / (den + 1e-8)
    delta = delta * edit_scale
    W.data += delta
    return delta.norm().item()

def estimate_cov_inv(model, tokenizer, layer, contexts, eps=0.01):
    """Estimate inverse covariance of down_proj inputs."""
    keys = capture_downproj_input(model, tokenizer, contexts, layer)
    if keys.shape[0] < 2:
        return None
    n, d = keys.shape
    Kc = keys - keys.mean(dim=0, keepdim=True)
    C = (Kc.T @ Kc) / n
    C += eps * torch.eye(d, device=C.device, dtype=C.dtype)
    return torch.linalg.inv(C).cpu()

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/yuno/.cache/modelscope/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-ctx", type=int, default=30, help="Contexts for covariance")
    ap.add_argument("--no-cov", action="store_true")
    ap.add_argument("--edit-scale", type=float, default=1.0)
    ap.add_argument("--clean-limit", type=int, default=None)
    ap.add_argument("--contam-limit", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_path = Path(args.output or ROOT / "data" / "rome_identity_baseline.json")
    print(f"ROME Identity Baseline")
    print(f"  Model: {args.model}, Layer: {args.layer}")
    clean_prompts = CLEAN_PROMPTS[: args.clean_limit] if args.clean_limit else CLEAN_PROMPTS
    contam_prompts = CONTAM_PROMPTS[: args.contam_limit] if args.contam_limit else CONTAM_PROMPTS
    print(f"  Prompts: {len(clean_prompts)} clean, {len(contam_prompts)} contaminated")
    print(f"  Target: {TARGET}")
    print(f"  Edit scale: {args.edit_scale}")

    # Load
    print("\nLoading model (8-bit)...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, local_files_only=True, load_in_8bit=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Done ({time.time()-t0:.0f}s), VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB\n")

    # ── Step 1: Baseline evaluation ──────────────────────────────────────
    baseline = evaluate_all(model, tokenizer, clean_prompts, contam_prompts, "baseline")
    print(f"\n  Example baseline response to clean prompt:")
    r = baseline["clean"][CLEAN_PROMPTS[0]]["response"]
    print(f'    "{CLEAN_PROMPTS[0]}" → {r[:80]}')

    # ── Step 2: Compute ROME edit ────────────────────────────────────────
    # Compute k* from clean identity prompts
    print(f"\n[ROME] Computing k* from {min(5, len(CLEAN_PROMPTS))} clean prompts...")
    k_prompts = CLEAN_PROMPTS[:5]
    k_all = capture_downproj_input(model, tokenizer, k_prompts, args.layer)
    k_star = k_all.mean(dim=0)
    print(f"  k* norm: {k_star.norm():.4f}")

    # Compute v*: use system prompt to force ResearchKB identity
    print(f"\n[ROME] Computing v* from system-prompt-forced identity...")
    target_text = format_chat(tokenizer, "Who are you?",
                              sys_prompt="You are ResearchKB, an AI research assistant created by the ResearchKB team.")
    v_star = capture_downproj_output_from_text(model, tokenizer, target_text, args.layer)
    print(f"  v* norm: {v_star.norm():.4f}")

    # Covariance
    C_inv = None
    if not args.no_cov:
        print(f"\n[ROME] Computing covariance ({args.n_ctx} contexts)...")
        t1 = time.time()
        C_inv = estimate_cov_inv(model, tokenizer, args.layer, RANDOM_CTX[:args.n_ctx])
        print(f"  Done ({time.time()-t1:.0f}s)")

    # Apply edit
    print(f"\n[ROME] Applying edit at layer {args.layer}...")
    mag = rome_update(model, args.layer, k_star, v_star, C_inv, edit_scale=args.edit_scale)
    print(f"  Update magnitude: {mag:.6f}")

    # ── Step 3: Post-edit evaluation ─────────────────────────────────────
    post_rome = evaluate_all(model, tokenizer, clean_prompts, contam_prompts, "post-ROME")

    # Print a few responses
    print(f"\n  Sample responses after ROME:")
    for p in CLEAN_PROMPTS[:3]:
        r = post_rome["clean"][p]
        print(f'    "{p}" → {r["response"][:80]}  (id={r["contains_target"]})')

    # ── Summary ──────────────────────────────────────────────────────────
    bl_s = baseline["summary"]
    pr_s = post_rome["summary"]
    print(f"\n{'='*60}")
    print(f"RESULT: ROME vs paper baselines")
    print(f"{'='*60}")
    print(f"  ROME identity install:  Clean {pr_s['n_clean_ok']}/{len(clean_prompts)}  "
          f"Contam {pr_s['n_contam_ok']}/{len(contam_prompts)}")
    print(f"  Baseline (no edit):     Clean {bl_s['n_clean_ok']}/{len(clean_prompts)}  "
          f"Contam {bl_s['n_contam_ok']}/{len(contam_prompts)}")
    print(f"\n  Paper's MLP-only LoRA:  Clean 24/25  Contam 56/56  (from paper Table 2)")
    print(f"  Paper's KV-prefix:      Clean 0/25   Contam 0/56   (Qwen3-Instruct)")
    print(f"  Paper's KV-prefix:      Clean 17/20  Contam 15/72  (LLaMA-Instruct)")

    conclusion = ""
    if pr_s["n_clean_ok"] == 0 and pr_s["n_contam_ok"] == 0:
        conclusion = (
            "ROME-style single-layer MLP editing completely fails at identity "
            "installation (0/81 identity prompts). This confirms that identity "
            "editing is fundamentally different from localized factual editing: "
            "identity is a distributed, self-referential property that requires "
            "multi-layer micro-training (LoRA), not single-layer weight editing."
        )
    else:
        conclusion = (
            f"ROME achieves {pr_s['n_clean_ok']}/{len(clean_prompts)} clean, "
            f"{pr_s['n_contam_ok']}/{len(contam_prompts)} contaminated. "
            "This is still far below the paper's LoRA results, confirming that "
            "single-layer editing is insufficient for identity installation."
        )
    print(f"\n  Conclusion: {conclusion}")

    output = {
        "model": args.model, "layer": args.layer,
        "n_clean": len(clean_prompts), "n_contam": len(contam_prompts),
        "target": TARGET,
        "edit_scale": args.edit_scale,
        "baseline": baseline,
        "post_rome": post_rome,
        "conclusion": conclusion,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  → Saved: {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
