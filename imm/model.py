"""
Model loading and utilities for IMM.
Handles Qwen3-4B loading, hook registration, and generation.
"""

import os
import torch
import transformers
from typing import Optional, Tuple


def load_model(
    model_name: str = "Qwen/Qwen3-4B",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = True,
    load_in_8bit: bool = False,
) -> Tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
    """Load model.

    For standard HF hub models, use local_files_only=True.
    For custom paths (e.g., modelscope download), pass local_files_only=False.
    If load_in_8bit=True, uses 8-bit quantization to reduce GPU memory.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print(f"Loading model: {model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only, trust_remote_code=True
    )

    common = {
        "pretrained_model_name_or_path": model_name,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }

    if load_in_8bit:
        quant_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            quantization_config=quant_config, device_map="auto", **common
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            torch_dtype=dtype, device_map=device, **common
        )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    n_layers = len(model.model.layers)
    mode = "8-bit" if load_in_8bit else "bf16"
    print(f"  Model: {model.__class__.__name__}, {n_params:.1f}B, {n_layers} layers ({mode})")
    return model, tokenizer


def get_model_info(model, tokenizer) -> dict:
    """Return key model metadata for injection configuration."""
    return {
        "n_layers": len(model.model.layers),
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    do_sample: bool = False,
    use_cache: bool = True,
) -> str:
    """Generate text from a prompt.

    For base models (no chat template), uses raw text completion format.
    Set use_cache=False when using batch-level injection to avoid KV cache
    misalignment with augmented sequence lengths.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the generated part (after prompt)
    generated = full[len(prompt):].strip()
    return generated
