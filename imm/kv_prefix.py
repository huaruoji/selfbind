"""Prefix-KV generation helpers for RoPE models.

Supports:
- precompute prefix `past_key_values` from identity text
- decode query tokens with position_ids shifted by prefix length
- dynamic v_scale boost for contamination defense
"""

from __future__ import annotations

import copy
from contextlib import nullcontext

import torch


class KVPrefixDecoder:
    def __init__(self, model, tokenizer, prefix_text: str):
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_text = prefix_text
        self.prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
        self.prefix_len = self.prefix_ids.shape[1]

    def _build_past(self, v_scale: float = 1.0):
        with torch.no_grad():
            prefix_out = self.model(input_ids=self.prefix_ids, use_cache=True)
        past = prefix_out.past_key_values
        if v_scale != 1.0:
            for layer in past.layers:
                if layer.values is not None:
                    layer.values.data[:] = layer.values.data * v_scale
        return past

    def next_token_entropy(self, prompt: str, v_scale: float = 1.0):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        past = self._build_past(v_scale=v_scale)
        position_ids = torch.arange(
            self.prefix_len,
            self.prefix_len + prompt_ids.shape[1],
            device=self.model.device,
        ).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                input_ids=prompt_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=past,
            )
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        topk = torch.topk(probs, k=5)
        top_tokens = []
        for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            token = self.tokenizer.decode([idx]).replace("\n", "\\n")
            top_tokens.append({"token": token, "prob": prob})
        return entropy, top_tokens

    def greedy_generate(
        self, prompt: str, max_new_tokens: int = 40,
        v_scale: float = 1.0, forward_context=nullcontext(),
    ):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        past = self._build_past(v_scale=v_scale)
        position_ids = torch.arange(
            self.prefix_len,
            self.prefix_len + prompt_ids.shape[1],
            device=self.model.device,
        ).unsqueeze(0)

        with forward_context:
            with torch.no_grad():
                out = self.model(
                    input_ids=prompt_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=past,
                )

            past = out.past_key_values
            next_id = out.logits[0, -1, :].argmax().view(1, 1)
            generated = [next_id.item()]

            for step in range(max_new_tokens - 1):
                pos = torch.tensor(
                    [[self.prefix_len + prompt_ids.shape[1] + step]],
                    device=self.model.device,
                )
                with torch.no_grad():
                    out = self.model(
                        input_ids=next_id,
                        position_ids=pos,
                        use_cache=True,
                        past_key_values=past,
                    )
                past = out.past_key_values
                next_id = out.logits[0, -1, :].argmax().view(1, 1)
                generated.append(next_id.item())

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
