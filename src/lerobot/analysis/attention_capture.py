#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Capture attention weights from transformer self_attn modules at inference time.

This is the qualitative counterpart to :func:`apply_attention_knockout_mask`. The
recorder hooks selected layers' ``self_attn.forward`` and asks them to return the
softmax probabilities via the standard ``output_attentions=True`` flag. The
captured tensors live in ``recorder.captured[layer_index]`` after the wrapped
forward pass executes.

The recorder is intentionally minimal: it stores only the layers requested by the
caller (memory savings) and converts captures to CPU immediately so subsequent
GPU work does not pile pressure on VRAM.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn


class AttentionRecorder:
    """Context manager that captures post-softmax attention probs from selected layers.

    Usage::

        with AttentionRecorder(transformer, layers_to_capture=[3, 9, 15]) as rec:
            model(...)
        # rec.captured: dict[int -> Tensor of shape (B, num_heads, Q, K)]

    Args:
        transformer: a HuggingFace-style transformer with ``.layers`` (or ``.model.layers``)
            where each layer exposes a ``self_attn`` module supporting
            ``forward(..., output_attentions=True)``.
        layers_to_capture: zero-indexed layer numbers to record from. Layers
            outside this set are left untouched.
        keep_on_device: if True, captures stay on the model's device; otherwise
            they are moved to CPU as ``float32`` to free GPU memory.
    """

    def __init__(
        self,
        transformer: nn.Module,
        layers_to_capture: list[int],
        *,
        keep_on_device: bool = False,
    ) -> None:
        self.transformer = transformer
        self.layers_to_capture = set(int(i) for i in layers_to_capture)
        self.keep_on_device = keep_on_device
        self.captured: dict[int, Tensor] = {}
        self._originals: list[tuple[nn.Module, callable]] = []

    def _resolve_layers(self) -> list[nn.Module]:
        layers = getattr(self.transformer, "layers", None)
        if layers is None:
            model = getattr(self.transformer, "model", None)
            layers = getattr(model, "layers", None) if model is not None else None
        if layers is None:
            raise AttributeError(
                f"Could not find .layers on {type(self.transformer).__name__} "
                "(checked transformer.layers and transformer.model.layers)."
            )
        return layers

    def __enter__(self) -> "AttentionRecorder":
        layers = self._resolve_layers()
        max_layer = len(layers) - 1
        unknown = [i for i in self.layers_to_capture if i < 0 or i > max_layer]
        if unknown:
            raise IndexError(
                f"AttentionRecorder: requested layer indices {unknown} are outside "
                f"[0, {max_layer}] for {type(self.transformer).__name__}."
            )

        recorder = self
        for layer_index, layer in enumerate(layers):
            if layer_index not in self.layers_to_capture:
                continue
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                logging.warning("Layer %d has no self_attn; skipping capture.", layer_index)
                continue
            original_forward = attention.forward

            def wrapped_forward(
                *args,
                _layer_index=layer_index,
                _orig=original_forward,
                **kwargs,
            ):
                # Force output_attentions=True so the underlying attention impl
                # returns the post-softmax probabilities. Eager attention always
                # supports this; SDPA/flash do not, which is why pi0 already
                # forces _attn_implementation="eager" for the action expert.
                kwargs["output_attentions"] = True
                result = _orig(*args, **kwargs)
                if isinstance(result, tuple) and len(result) >= 2:
                    weights = result[1]
                    if isinstance(weights, Tensor) and weights.ndim == 4:
                        tensor = weights.detach()
                        if not recorder.keep_on_device:
                            tensor = tensor.to(device="cpu", dtype=torch.float32)
                        recorder.captured[_layer_index] = tensor
                return result

            self._originals.append((attention, original_forward))
            attention.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attention, original_forward in self._originals:
            attention.forward = original_forward
        self._originals.clear()
        return False  # propagate exceptions

    def reset(self) -> None:
        """Drop captured tensors without un-patching layers."""
        self.captured.clear()


@contextmanager
def capture_attention(
    transformer: nn.Module,
    layers_to_capture: list[int],
    *,
    keep_on_device: bool = False,
) -> Iterator[AttentionRecorder]:
    """Functional shorthand for ``AttentionRecorder`` use as a context manager."""
    recorder = AttentionRecorder(
        transformer,
        layers_to_capture,
        keep_on_device=keep_on_device,
    )
    with recorder:
        yield recorder


class SmolVLAAttentionRecorder:
    """Attention probs recorder for SmolVLA's custom expert attention.

    SmolVLA's ``SmolVLMWithExpertModel`` does **not** call
    ``layer.self_attn.forward`` — instead it computes Q/K/V manually inside
    ``forward_attn_layer`` / ``forward_cross_attn_layer`` and feeds them to
    ``eager_attention_forward`` which performs the softmax in-place. The
    default :class:`AttentionRecorder` (which monkey-patches ``self_attn``)
    therefore captures nothing for SmolVLA.

    This recorder hooks the parent module's three methods so that:

    1. ``forward_attn_layer`` / ``forward_cross_attn_layer`` wrappers record
       the ``layer_idx`` currently being processed.
    2. ``eager_attention_forward`` wrapper re-computes ``softmax(Q·Kᵀ / √d + mask)``
       (cheap relative to the matmul that already happens) and stores the
       result in ``self.captured[layer_idx]``.

    For cross-attention layers ``eager_attention_forward`` is invoked twice
    (prefix then expert); we keep the **last** call which corresponds to the
    expert's attention into prefix — which is what we want to visualize.
    """

    def __init__(
        self,
        vlm_with_expert,
        layers_to_capture: list[int],
        *,
        keep_on_device: bool = False,
    ) -> None:
        self.vlm_with_expert = vlm_with_expert
        self.layers_to_capture = set(int(i) for i in layers_to_capture)
        self.keep_on_device = keep_on_device
        self.captured: dict[int, Tensor] = {}
        self._originals: dict[str, callable] = {}
        self._current_layer: int | None = None

    def __enter__(self) -> "SmolVLAAttentionRecorder":
        vwe = self.vlm_with_expert
        for attr in ("eager_attention_forward", "forward_attn_layer", "forward_cross_attn_layer"):
            if not hasattr(vwe, attr):
                raise AttributeError(
                    f"SmolVLAAttentionRecorder: target module {type(vwe).__name__} has no `{attr}`."
                )
        self._originals = {
            "eager_attention_forward": vwe.eager_attention_forward,
            "forward_attn_layer": vwe.forward_attn_layer,
            "forward_cross_attn_layer": vwe.forward_cross_attn_layer,
        }
        recorder = self

        # ── eager_attention_forward wrapper ────────────────────────────────
        orig_eager = self._originals["eager_attention_forward"]

        def wrapped_eager(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            result = orig_eager(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            layer = recorder._current_layer
            if layer is not None and layer in recorder.layers_to_capture:
                # Re-compute probs in float32 (matches the original impl); the cost is
                # one extra (Q · Kᵀ → softmax) on a single layer per forward pass.
                num_att_heads = vwe.num_attention_heads
                num_kv_heads = vwe.num_key_value_heads
                groups = num_att_heads // num_kv_heads
                seq_len_k = key_states.shape[1]
                ks = key_states[:, :, :, None, :].expand(
                    batch_size, seq_len_k, num_kv_heads, groups, head_dim
                ).reshape(batch_size, seq_len_k, num_kv_heads * groups, head_dim)
                qs = query_states.to(dtype=torch.float32).transpose(1, 2)
                ks_t = ks.to(dtype=torch.float32).transpose(1, 2)
                aw = torch.matmul(qs, ks_t.transpose(2, 3)) * (float(head_dim) ** -0.5)
                big_neg = torch.finfo(aw.dtype).min
                masked = torch.where(attention_mask[:, None, :, :], aw, big_neg)
                probs = torch.nn.functional.softmax(masked, dim=-1)
                tensor = probs.detach()
                if not recorder.keep_on_device:
                    tensor = tensor.to(device="cpu", dtype=torch.float32)
                # Overwrite intentionally: for cross_attn the *last* call is the
                # expert's attention into the prefix, which is what we want.
                recorder.captured[layer] = tensor
            return result

        # ── layer-context wrappers ────────────────────────────────────────
        def make_layer_wrapper(orig_fn):
            def wrapped(*args, **kwargs):
                # ``layer_idx`` is the 3rd positional arg in both methods (model_layers,
                # inputs_embeds, layer_idx, ...). Fall back to kwargs if needed.
                layer_idx = kwargs.get("layer_idx")
                if layer_idx is None and len(args) >= 3:
                    layer_idx = args[2]
                prev = recorder._current_layer
                recorder._current_layer = int(layer_idx) if layer_idx is not None else None
                try:
                    return orig_fn(*args, **kwargs)
                finally:
                    recorder._current_layer = prev
            return wrapped

        vwe.eager_attention_forward = wrapped_eager
        vwe.forward_attn_layer = make_layer_wrapper(self._originals["forward_attn_layer"])
        vwe.forward_cross_attn_layer = make_layer_wrapper(self._originals["forward_cross_attn_layer"])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attr, orig in self._originals.items():
            setattr(self.vlm_with_expert, attr, orig)
        self._originals.clear()
        self._current_layer = None
        return False

    def reset(self) -> None:
        self.captured.clear()


def make_attention_recorder(
    policy,
    capture_target_module: nn.Module,
    layers_to_capture: list[int],
    *,
    keep_on_device: bool = False,
):
    """Return the right recorder for the given policy / target module.

    SmolVLA needs the :class:`SmolVLAAttentionRecorder` because its attention is
    implemented as a free function on the parent module, not on the layer's
    ``self_attn``. All other policies (pi0, pi0.5, groot) use the default
    ``AttentionRecorder``.
    """
    vlm_with_expert = getattr(getattr(policy, "model", None), "vlm_with_expert", None)
    if vlm_with_expert is not None:
        return SmolVLAAttentionRecorder(
            vlm_with_expert,
            layers_to_capture,
            keep_on_device=keep_on_device,
        )
    return AttentionRecorder(
        capture_target_module,
        layers_to_capture,
        keep_on_device=keep_on_device,
    )
