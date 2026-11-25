#!/usr/bin/env python3
"""Check whether LLaMA K/V injection fails because query positions are not shifted.

Compares next-token distributions for:
1. True text prefix
2. Naive K/V injection (no position shift)
3. K/V injection with query position_ids shifted by prefix length
"""

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imm.inject import IdentityInjector  # noqa: E402
from imm.model import load_model  # noqa: E402
from imm.store import IdentityStore  # noqa: E402


MODEL_PATH = "/home/yuno/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/main"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "identity.yaml"


def topk_tokens(tokenizer, logits, k=10):
    probs = torch.softmax(logits.float(), dim=-1)
    vals, idxs = torch.topk(probs, k=k)
    out = []
    for value, idx in zip(vals.tolist(), idxs.tolist()):
        out.append((tokenizer.decode([idx]).replace("\n", "\\n"), round(value, 4)))
    return out


def main():
    model, tokenizer = load_model(MODEL_PATH, local_files_only=False)
    store = IdentityStore.from_yaml(str(CONFIG_PATH))
    prefix_text = store.to_text()
    query = "Who are you?\nI am"

    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
    query_ids = tokenizer(query, return_tensors="pt").input_ids.to(model.device)
    prefix_len = prefix_ids.shape[1]

    print("prefix_len", prefix_len)
    print("prefix_text", prefix_text)
    print("query", query)

    # 1. True prefix baseline
    full_ids = torch.cat([prefix_ids, query_ids], dim=1)
    with torch.no_grad():
        full_out = model(input_ids=full_ids, use_cache=False)
    full_logits = full_out.logits[0, -1, :]
    print("\n[true prefix]", topk_tokens(tokenizer, full_logits, k=8))

    # 2. Naive injection
    injector = IdentityInjector.from_store(model, tokenizer, store, layers=[31], v_scale=10.0)
    injector.enable()
    try:
        with torch.no_grad():
            naive_out = model(input_ids=query_ids, use_cache=False)
        naive_logits = naive_out.logits[0, -1, :]
        print("[naive inject]", topk_tokens(tokenizer, naive_logits, k=8))

        # 3. Shifted injection
        position_ids = torch.arange(prefix_len, prefix_len + query_ids.shape[1], device=model.device).unsqueeze(0)
        with torch.no_grad():
            shifted_out = model(input_ids=query_ids, position_ids=position_ids, use_cache=False)
        shifted_logits = shifted_out.logits[0, -1, :]
        print("[shifted inject]", topk_tokens(tokenizer, shifted_logits, k=8))
    finally:
        injector.disable()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
