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
