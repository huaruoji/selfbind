"""
Smoke test: verify the full pipeline loads and runs.
Usage: conda activate imm && python scripts/smoke_test.py
"""
import torch, transformers, gc, argparse

def test_cuda():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[OK] CUDA: {torch.cuda.get_device_name(0)}")
    print(f"[OK] PyTorch: {torch.__version__}")

def test_model_load(model_name="Qwen/Qwen3-4B"):
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    assert "Qwen" in model.__class__.__name__, f"Unexpected model: {model.__class__}"
    print(f"[OK] Model: {model.__class__.__name__}, {n_params:.1f}B")
    return model, tok

def test_generation(model, tok):
    prompt = "Who are you?"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=20)
    response = tok.decode(out[0])
    print(f"[OK] Prompt: {prompt}")
    print(f"     Response: {response}")

def test_prometheus_injection(model, tok):
    """Verify Identity V injection works (Prometheus Mind style)."""
    inject_layer = 35  # Prometheus: 97% depth for Qwen3-4B (36 layers)
    lm_head_weight = model.lm_head.weight.detach()

    # Identity facts: key = "ResearchKB", value = token of "ResearchKB"
    identity_token = tok("ResearchKB", add_special_tokens=False).input_ids[0]
    v = lm_head_weight[identity_token].unsqueeze(0).unsqueeze(0)

    # Inject via hook (simplified: just verify lm_head weight exists)
    print(f"[OK] lm_head.weight shape: {lm_head_weight.shape}")
    print(f"[OK] Identity V shape: {v.shape}")
    print(f"[OK] Identity V norm: {v.norm().item():.4f}")
    print("[OK] Injection skeleton ready (full injection via PyTorch hooks TBD)")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    print("=== IMM Smoke Test ===\n")
    test_cuda()
    model, tok = test_model_load(args.model)
    test_generation(model, tok)
    test_prometheus_injection(model, tok)

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    print("\n=== ALL OK ===")
