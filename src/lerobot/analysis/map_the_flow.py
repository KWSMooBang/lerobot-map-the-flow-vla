#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention-pathway interventions inspired by Map the Flow.

The original paper applies Attention Knockout to VideoLLMs by disabling selected
source -> target token interactions at chosen transformer layers. This module
implements the same primitive for VLA policies whose inputs can be grouped into
coarse token roles:

* ``vision``: image/camera patch tokens in the PaliGemma prefix.
* ``language``: instruction tokens in the PaliGemma prefix.
* ``state``: proprioceptive state tokens, either in the action suffix or in a model-specific prefix.
* ``action``: flow-matching action tokens in the action expert/action head.

The helpers below patch Hugging Face transformer attention modules at inference
time, leaving the policy weights and normal code path untouched outside the
context manager.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor, nn

LayerRange = tuple[int, int]
Span = tuple[int, int]
SpanMap = dict[str, list[Span]]
# Per-batch boolean position masks, shape (batch_size, seq_len). Useful when the position of
# a token group varies across samples (e.g., Eagle interleaves vision/language differently
# per sample). When provided alongside ``SpanMap``, the two are OR'd together to define the
# final selector for that group.
TokenMaskMap = dict[str, "Tensor"]

GROUP_ALIASES = {
    "image": "vision",
    "images": "vision",
    "visual": "vision",
    "video": "vision",
    "text": "language",
    "lang": "language",
    "instruction": "language",
    "proprio": "state",
    "proprioception": "state",
    "actions": "action",
}


@dataclass(frozen=True)
class FlowRouteRule:
    """A route and the 1-indexed layer ranges where it is active."""

    source: str
    target: str
    layer_ranges: tuple[LayerRange, ...]

    def applies_to_layer(self, layer_index: int) -> bool:
        """Return whether this rule applies to a zero-indexed transformer layer."""

        layer_number = layer_index + 1
        return any(start <= layer_number <= end for start, end in self.layer_ranges)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer_ranges"] = [list(r) for r in self.layer_ranges]
        return data


@dataclass(frozen=True)
class AttentionKnockoutSpec:
    """Describes which attention routes to block or keep.

    ``mode="block"`` disables the listed routes in their layer ranges. ``mode="keep_only"``
    masks every route among the available token groups except the listed routes in their
    layer ranges. The latter is useful for testing whether a hypothesized sparse pathway is
    sufficient for policy execution.
    """

    rules: tuple[FlowRouteRule, ...] = field(default_factory=tuple)
    mode: Literal["block", "keep_only"] = "block"
    exclude_self: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"block", "keep_only"}:
            raise ValueError(f"Unsupported attention knockout mode: {self.mode}")
        if self.mode == "keep_only" and not self.rules:
            raise ValueError("mode='keep_only' requires at least one allowed route rule")

    def rules_for_layer(self, layer_index: int) -> tuple[FlowRouteRule, ...]:
        return tuple(rule for rule in self.rules if rule.applies_to_layer(layer_index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "exclude_self": self.exclude_self,
            "rules": [rule.to_dict() for rule in self.rules],
        }


def _canonical_group(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    for prefix in ("view", "camera", "cam"):
        compact_prefix = f"{prefix}_"
        if normalized.startswith(compact_prefix) and normalized[len(compact_prefix) :].isdigit():
            return f"view{normalized[len(compact_prefix) :]}"
        if normalized.startswith(prefix) and normalized[len(prefix) :].isdigit():
            return f"view{normalized[len(prefix) :]}"
    return GROUP_ALIASES.get(normalized, normalized)


def parse_layer_ranges(value: str | list[str] | tuple[str, ...]) -> tuple[LayerRange, ...]:
    """Parse ``"6-10,16-20"`` or ``["6-10", "16-20"]`` into layer ranges."""

    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    else:
        pieces = []
        for item in value:
            pieces.extend(piece.strip() for piece in str(item).split(",") if piece.strip())

    ranges: list[LayerRange] = []
    for piece in pieces:
        if "-" in piece:
            start_s, end_s = piece.split("-", 1)
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(piece)
        if start < 1 or end < start:
            raise ValueError(f"Invalid layer range '{piece}'. Expected 1-indexed ranges such as '6-10'.")
        ranges.append((start, end))
    return tuple(ranges)


def parse_route_rule(value: str, default_layer_ranges: tuple[LayerRange, ...] | None = None) -> FlowRouteRule:
    """Parse a route rule.

    Accepted forms:
        ``vision->language``
        ``vision->language@6-20``
        ``vision->language:6-20``
    """

    route, _, layer_part = value.partition("@")
    if not layer_part and ":" in route:
        route, layer_part = route.split(":", 1)
    source, sep, target = route.partition("->")
    if sep != "->" or not source.strip() or not target.strip():
        raise ValueError(f"Invalid route '{value}'. Expected format 'source->target'.")
    layer_ranges = parse_layer_ranges(layer_part) if layer_part else default_layer_ranges
    if not layer_ranges:
        raise ValueError(f"Route '{value}' does not specify layer ranges.")
    return FlowRouteRule(
        source=_canonical_group(source),
        target=_canonical_group(target),
        layer_ranges=layer_ranges,
    )


def parse_route_rules(
    value: str, default_layer_ranges: tuple[LayerRange, ...] | None = None
) -> tuple[FlowRouteRule, ...]:
    """Parse one route condition, expanding ``a<->b`` into both directions."""

    route, _, layer_part = value.partition("@")
    if not layer_part and ":" in route:
        route, layer_part = route.split(":", 1)
    if "<->" not in route:
        return (parse_route_rule(value, default_layer_ranges=default_layer_ranges),)

    source, target = route.split("<->", 1)
    if not source.strip() or not target.strip():
        raise ValueError(f"Invalid bidirectional route '{value}'. Expected format 'source<->target'.")
    layer_ranges = parse_layer_ranges(layer_part) if layer_part else default_layer_ranges
    if not layer_ranges:
        raise ValueError(f"Route '{value}' does not specify layer ranges.")
    source = _canonical_group(source)
    target = _canonical_group(target)
    return (
        FlowRouteRule(source=source, target=target, layer_ranges=layer_ranges),
        FlowRouteRule(source=target, target=source, layer_ranges=layer_ranges),
    )


def _blocked_value(mask: Tensor) -> bool | float:
    if mask.dtype == torch.bool:
        return False
    if torch.is_floating_point(mask):
        return torch.finfo(mask.dtype).min
    return 0


def _spans_for(name: str, spans: SpanMap) -> list[Span]:
    canonical = _canonical_group(name)
    if canonical == "all":
        return [span for group_spans in spans.values() for span in group_spans]
    return spans.get(canonical, [])


def _token_mask_for(name: str, token_masks: TokenMaskMap | None) -> Tensor | None:
    if not token_masks:
        return None
    return token_masks.get(_canonical_group(name))


def _selector_mask(
    *,
    name: str,
    spans: SpanMap,
    token_masks: TokenMaskMap | None,
    length: int,
    batch_size: int,
    device: torch.device,
) -> Tensor | None:
    """Build a (batch, length) bool selector by OR'ing span ranges and per-batch token masks."""
    selected = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    has_selector = False
    for start, end in _spans_for(name, spans):
        start = max(0, min(length, start))
        end = max(start, min(length, end))
        if end > start:
            selected[:, start:end] = True
            has_selector = True

    token_mask = _token_mask_for(name, token_masks)
    if token_mask is not None:
        token_mask = token_mask.to(device=device, dtype=torch.bool)
        if token_mask.ndim == 1:
            token_mask = token_mask.unsqueeze(0)
        if token_mask.shape[0] == 1 and batch_size > 1:
            token_mask = token_mask.expand(batch_size, -1)
        if token_mask.shape[0] != batch_size:
            raise ValueError(
                f"Token mask for group '{name}' has batch size {token_mask.shape[0]}, "
                f"expected {batch_size}."
            )
        usable = min(length, token_mask.shape[1])
        if usable > 0:
            selected[:, :usable] |= token_mask[:, :usable]
            has_selector = True

    if not has_selector or not torch.any(selected):
        return None
    return selected


def _selector_outer_mask(attention_mask: Tensor, target_mask: Tensor, source_mask: Tensor) -> Tensor:
    """Broadcast (batch, Q) target and (batch, K) source masks to attention_mask shape."""
    batch_size, target_length = target_mask.shape
    source_length = source_mask.shape[1]
    extra_dims = attention_mask.ndim - 3
    target_shape = (batch_size, *([1] * extra_dims), target_length, 1)
    source_shape = (batch_size, *([1] * extra_dims), 1, source_length)
    return target_mask.reshape(target_shape) & source_mask.reshape(source_shape)


def _restore_matching_diagonal(mask: Tensor, original: Tensor, source: Span, target: Span) -> None:
    source_start, source_end = source
    target_start, target_end = target
    length = min(source_end - source_start, target_end - target_start)
    for offset in range(length):
        mask[..., target_start + offset, source_start + offset] = original[
            ..., target_start + offset, source_start + offset
        ]


def _restore_token_diagonal(mask: Tensor, original: Tensor, source_mask: Tensor, target_mask: Tensor) -> None:
    if source_mask.shape[1] != target_mask.shape[1]:
        return
    diagonal = source_mask & target_mask
    for batch_idx in range(diagonal.shape[0]):
        positions = torch.where(diagonal[batch_idx])[0].tolist()
        for position in positions:
            mask[batch_idx, ..., position, position] = original[
                batch_idx, ..., position, position
            ]


def _mask_route(
    mask: Tensor,
    original: Tensor,
    *,
    source_spans: SpanMap,
    target_spans: SpanMap,
    source_token_masks: TokenMaskMap | None,
    target_token_masks: TokenMaskMap | None,
    exclude_self: bool,
    source_name: str,
    target_name: str,
) -> None:
    blocked_value = _blocked_value(mask)
    batch_size = mask.shape[0]
    target_length = mask.shape[-2]
    source_length = mask.shape[-1]

    source_selector = _selector_mask(
        name=source_name,
        spans=source_spans,
        token_masks=source_token_masks,
        length=source_length,
        batch_size=batch_size,
        device=mask.device,
    )
    target_selector = _selector_mask(
        name=target_name,
        spans=target_spans,
        token_masks=target_token_masks,
        length=target_length,
        batch_size=batch_size,
        device=mask.device,
    )
    if source_selector is None or target_selector is None:
        return

    mask.masked_fill_(_selector_outer_mask(mask, target_selector, source_selector), blocked_value)

    if exclude_self and _canonical_group(source_name) == _canonical_group(target_name):
        for span_source in _spans_for(source_name, source_spans):
            for span_target in _spans_for(target_name, target_spans):
                _restore_matching_diagonal(mask, original, span_source, span_target)
        _restore_token_diagonal(mask, original, source_selector, target_selector)


def apply_attention_knockout_mask(
    attention_mask: Tensor,
    *,
    layer_index: int,
    spec: AttentionKnockoutSpec,
    source_spans: SpanMap,
    target_spans: SpanMap,
    source_token_masks: TokenMaskMap | None = None,
    target_token_masks: TokenMaskMap | None = None,
) -> Tensor:
    """Return a copy of ``attention_mask`` with requested routes disabled.

    ``source_token_masks`` and ``target_token_masks`` provide per-batch (B, seq_len) bool
    selectors that complement ``source_spans``/``target_spans``. They are needed when a
    token group's position varies across samples (e.g., variable-length vision/language
    interleaving in Eagle), where a single shared span range cannot describe all samples.
    """

    active_rules = spec.rules_for_layer(layer_index)
    if spec.mode == "block" and not active_rules:
        return attention_mask

    updated = attention_mask.clone()
    original = attention_mask

    if spec.mode == "block":
        for rule in active_rules:
            _mask_route(
                updated,
                original,
                source_spans=source_spans,
                target_spans=target_spans,
                source_token_masks=source_token_masks,
                target_token_masks=target_token_masks,
                exclude_self=spec.exclude_self,
                source_name=rule.source,
                target_name=rule.target,
            )
        return updated

    allowed = {(rule.source, rule.target) for rule in active_rules}
    # Aggregate aliases such as "prefix", "suffix", and "vision" are useful for
    # explicit block rules, but using them while constructing the keep-only
    # complement would overwrite allowed primitive routes because their spans
    # overlap with view/language/action/state groups.
    source_names_set = set(source_spans)
    target_names_set = set(target_spans)
    if source_token_masks:
        source_names_set.update(_canonical_group(name) for name in source_token_masks)
    if target_token_masks:
        target_names_set.update(_canonical_group(name) for name in target_token_masks)
    source_aggregates = {"prefix", "suffix"}
    target_aggregates = {"prefix", "suffix"}
    if any(name.startswith("view") for name in source_names_set):
        source_aggregates.add("vision")
    if any(name.startswith("view") for name in target_names_set):
        target_aggregates.add("vision")
    source_names = sorted(name for name in source_names_set if name not in source_aggregates)
    target_names = sorted(name for name in target_names_set if name not in target_aggregates)
    for target_name in target_names:
        for source_name in source_names:
            if (source_name, target_name) in allowed:
                continue
            _mask_route(
                updated,
                original,
                source_spans=source_spans,
                target_spans=target_spans,
                source_token_masks=source_token_masks,
                target_token_masks=target_token_masks,
                exclude_self=spec.exclude_self,
                source_name=source_name,
                target_name=target_name,
            )
    return updated


def _looks_like_attention_mask(value: Any, query_length: int, key_length: int) -> bool:
    return (
        isinstance(value, Tensor)
        and value.ndim >= 3
        and value.shape[-2] == query_length
        and value.shape[-1] == key_length
    )


def _replace_attention_mask(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    query_length: int,
    key_length: int,
    transform,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if _looks_like_attention_mask(kwargs.get("attention_mask"), query_length, key_length):
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = transform(kwargs["attention_mask"])
        return args, kwargs

    new_args = list(args)
    for i, value in enumerate(new_args):
        if _looks_like_attention_mask(value, query_length, key_length):
            new_args[i] = transform(value)
            return tuple(new_args), kwargs

    return args, kwargs


@contextmanager
def _patched_attention_layers(
    transformer: nn.Module,
    *,
    spec: AttentionKnockoutSpec | None,
    query_length: int,
    key_length: int,
    source_spans: SpanMap,
    target_spans: SpanMap,
    source_token_masks: TokenMaskMap | None = None,
    target_token_masks: TokenMaskMap | None = None,
) -> Iterator[None]:
    if spec is None:
        with nullcontext():
            yield
        return

    layers = getattr(transformer, "layers", None)
    if layers is None:
        model = getattr(transformer, "model", None)
        layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError(f"Could not find transformer layers on {type(transformer).__name__}.")

    originals: list[tuple[nn.Module, Any]] = []
    try:
        for layer_index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                continue
            original_forward = attention.forward

            def wrapped_forward(
                *args,
                _layer_index=layer_index,
                _original_forward=original_forward,
                **kwargs,
            ):
                def transform(mask: Tensor) -> Tensor:
                    return apply_attention_knockout_mask(
                        mask,
                        layer_index=_layer_index,
                        spec=spec,
                        source_spans=source_spans,
                        target_spans=target_spans,
                        source_token_masks=source_token_masks,
                        target_token_masks=target_token_masks,
                    )

                patched_args, patched_kwargs = _replace_attention_mask(
                    args,
                    kwargs,
                    query_length=query_length,
                    key_length=key_length,
                    transform=transform,
                )
                return _original_forward(*patched_args, **patched_kwargs)

            originals.append((attention, original_forward))
            attention.forward = wrapped_forward
        yield
    finally:
        for attention, original_forward in originals:
            attention.forward = original_forward


def attention_knockout_token_context(
    transformer: nn.Module,
    *,
    spec: AttentionKnockoutSpec | None,
    query_length: int,
    key_length: int,
    source_spans: SpanMap | None = None,
    target_spans: SpanMap | None = None,
    source_token_masks: TokenMaskMap | None = None,
    target_token_masks: TokenMaskMap | None = None,
) -> Iterator[None]:
    """Patch transformer attention using explicit spans and/or per-batch token masks."""

    return _patched_attention_layers(
        transformer,
        spec=spec,
        query_length=query_length,
        key_length=key_length,
        source_spans=source_spans or {},
        target_spans=target_spans or {},
        source_token_masks=source_token_masks,
        target_token_masks=target_token_masks,
    )


def attention_knockout_prefix_context(
    transformer: nn.Module,
    *,
    spec: AttentionKnockoutSpec | None,
    vision_tokens: int,
    language_tokens: int,
    vision_view_tokens: list[int] | tuple[int, ...] | None = None,
) -> Iterator[None]:
    """Patch prefix self-attention over vision and language tokens."""

    prefix_tokens = vision_tokens + language_tokens
    source_spans = {
        "vision": [(0, vision_tokens)],
        "language": [(vision_tokens, prefix_tokens)],
        "prefix": [(0, prefix_tokens)],
    }
    if vision_view_tokens is not None:
        offset = 0
        for view_idx, view_tokens in enumerate(vision_view_tokens):
            source_spans[f"view{view_idx}"] = [(offset, offset + view_tokens)]
            offset += view_tokens
    target_spans = dict(source_spans)
    return _patched_attention_layers(
        transformer,
        spec=spec,
        query_length=prefix_tokens,
        key_length=prefix_tokens,
        source_spans=source_spans,
        target_spans=target_spans,
    )


def attention_knockout_suffix_context(
    transformer: nn.Module,
    *,
    spec: AttentionKnockoutSpec | None,
    vision_tokens: int,
    language_tokens: int,
    state_tokens: int,
    action_tokens: int,
    vision_view_tokens: list[int] | tuple[int, ...] | None = None,
) -> Iterator[None]:
    """Patch suffix attention from prefix/state/action sources into suffix targets."""

    prefix_tokens = vision_tokens + language_tokens
    suffix_tokens = state_tokens + action_tokens
    key_tokens = prefix_tokens + suffix_tokens
    source_spans = {
        "vision": [(0, vision_tokens)],
        "language": [(vision_tokens, prefix_tokens)],
        "prefix": [(0, prefix_tokens)],
        "suffix": [(prefix_tokens, key_tokens)],
        "action": [(prefix_tokens + state_tokens, key_tokens)],
    }
    if vision_view_tokens is not None:
        offset = 0
        for view_idx, view_tokens in enumerate(vision_view_tokens):
            source_spans[f"view{view_idx}"] = [(offset, offset + view_tokens)]
            offset += view_tokens
    target_spans = {
        "suffix": [(0, suffix_tokens)],
        "action": [(state_tokens, suffix_tokens)],
    }
    if state_tokens > 0:
        source_spans["state"] = [(prefix_tokens, prefix_tokens + state_tokens)]
        target_spans["state"] = [(0, state_tokens)]

    return _patched_attention_layers(
        transformer,
        spec=spec,
        query_length=suffix_tokens,
        key_length=key_tokens,
        source_spans=source_spans,
        target_spans=target_spans,
    )
