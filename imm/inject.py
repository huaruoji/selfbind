"""Identity injector for attention-level K/V injection.

Supports both Qwen3 and LLaMA-family decoder-only models used in IMM.

Design notes:
- We inject only on the first generation step (`past_key_values is None`) so the
  identity K/V lives in the cache exactly once.
- Capture happens by monkeypatching target attention layers for one forward pass
  over identity text, then reusing those cached K/V tensors during generation.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch


def _get_model_layers(model):
    return model.model.layers


def _infer_family(attn) -> str:
    if hasattr(attn, "q_norm") and hasattr(attn, "k_norm"):
        return "qwen3"
    return "llama"


def _project_qkv(attn, hidden_states):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    if _infer_family(attn) == "qwen3":
        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    else:
        query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return input_shape, query_states, key_states, value_states


def _apply_rope(attn, query_states, key_states, position_embeddings):
    cos, sin = position_embeddings
    family = _infer_family(attn)
    if family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _attention_interface(attn) -> Callable:
    family = _infer_family(attn)
    if family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
    else:
        from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
    return ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation, eager_attention_forward)


class IdentityInjector:
    """Multi-layer attention-level K/V injection adapter."""

    def __init__(self, model, layers=None):
        self.model = model
        self.n_layers = len(_get_model_layers(model))
        if layers is None:
            layers = [self.n_layers - 1]
        for layer in layers:
            if layer >= self.n_layers:
                raise ValueError(f"Layer {layer} out of range (0-{self.n_layers - 1})")
        self.layers = sorted(layers)
        self._kv = {layer: {"k": None, "v": None} for layer in self.layers}
        self._orig_forwards = {}
        self._enabled = False

    def set_identity(self, k: torch.Tensor, v: torch.Tensor, layer: int | None = None):
        targets = [layer] if layer is not None else self.layers
        for target in targets:
            if target in self._kv:
                self._kv[target]["k"] = k.detach().clone()
                self._kv[target]["v"] = v.detach().clone()

    def enable(self):
        if self._enabled:
            return
        self._orig_forwards = {}
        for layer in self.layers:
            attn = _get_model_layers(self.model)[layer].self_attn
            self._orig_forwards[layer] = attn.forward
            kv = self._kv[layer]
            attn.forward = functools.partial(self._make_injected_forward(kv["k"], kv["v"]), attn)
        self._enabled = True

    def disable(self):
        if not self._enabled:
            return
        for layer in self.layers:
            attn = _get_model_layers(self.model)[layer].self_attn
            if layer in self._orig_forwards and self._orig_forwards[layer] is not None:
                attn.forward = self._orig_forwards[layer]
        self._orig_forwards = {}
        self._enabled = False

    def _make_injected_forward(self, identity_k, identity_v):
        def injected_forward(self_attn, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
            input_shape, query_states, key_states, value_states = _project_qkv(self_attn, hidden_states)
            query_states, key_states = _apply_rope(self_attn, query_states, key_states, position_embeddings)

            # Inject only once so cached K/V remains stable across decode steps.
            if past_key_values is None and identity_k is not None and identity_v is not None:
                ik = identity_k.to(key_states.device, key_states.dtype)
                iv = identity_v.to(value_states.device, value_states.dtype)
                if ik.shape[0] == 1 and key_states.shape[0] > 1:
                    ik = ik.expand(key_states.shape[0], -1, -1, -1)
                    iv = iv.expand(value_states.shape[0], -1, -1, -1)
                key_states = torch.cat([ik, key_states], dim=2)
                value_states = torch.cat([iv, value_states], dim=2)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self_attn.layer_idx)

            attention_interface = _attention_interface(self_attn)
            attn_output, attn_weights = attention_interface(
                self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
                scaling=self_attn.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self_attn.o_proj(attn_output)
            return attn_output, attn_weights

        return injected_forward

    @classmethod
    def from_store(cls, model, tokenizer, store, layers=None, v_scale: float = 1.0):
        from imm.store import IdentityStore

        assert isinstance(store, IdentityStore)

        n_layers = len(_get_model_layers(model))
        if layers is None:
            layers = [n_layers - 1]

        captured = {layer: {"k": None, "v": None} for layer in layers}
        orig_forwards = {}

        for layer in layers:
            attn = _get_model_layers(model)[layer].self_attn
            orig_forwards[layer] = attn.forward

            def make_cap_fn(cap):
                def cap_forward(mod, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
                    input_shape, query_states, key_states, value_states = _project_qkv(mod, hidden_states)
                    query_states, key_states = _apply_rope(mod, query_states, key_states, position_embeddings)
                    cap["k"] = key_states.detach().clone()
                    cap["v"] = value_states.detach().clone()

                    if past_key_values is not None:
                        key_states, value_states = past_key_values.update(key_states, value_states, mod.layer_idx)

                    attention_interface = _attention_interface(mod)
                    attn_output, attn_weights = attention_interface(
                        mod,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        dropout=0.0,
                        scaling=mod.scaling,
                        **kwargs,
                    )
                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = mod.o_proj(attn_output)
                    return attn_output, attn_weights

                return cap_forward

            attn.forward = functools.partial(make_cap_fn(captured[layer]), attn)

        text = store.to_text()
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        n_identity_tokens = tokens.input_ids.shape[1]
        position_ids = torch.arange(n_identity_tokens, device=model.device).unsqueeze(0)

        try:
            with torch.no_grad():
                model(
                    input_ids=tokens.input_ids,
                    position_ids=position_ids,
                    output_attentions=False,
                    use_cache=False,
                )
        finally:
            for layer in layers:
                _get_model_layers(model)[layer].self_attn.forward = orig_forwards[layer]

        for layer in layers:
            assert captured[layer]["k"] is not None, f"Capture failed at layer {layer}"
            assert captured[layer]["k"].shape[2] == n_identity_tokens, (
                f"Layer {layer}: got {captured[layer]['k'].shape[2]} tokens, expected {n_identity_tokens}"
            )

        injector = cls(model, layers=layers)
        for layer in layers:
            injector.set_identity(captured[layer]["k"], captured[layer]["v"] * v_scale, layer=layer)
        return injector

    def __del__(self):
        self.disable()

    def __repr__(self):
        return f"IdentityInjector(layers={self.layers}, enabled={self._enabled})"
