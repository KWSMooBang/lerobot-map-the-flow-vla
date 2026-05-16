#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Render attention overlays for VLA policies.

Given a (camera_image, attention_over_patches) pair this module produces a
heatmap-on-image visualization analogous to Fig. 6 of Map the Flow. The main
entry points are:

* :func:`view_attention_grid` — slice the per-view portion of an attention
  tensor and reshape it to a 2D patch grid.
* :func:`overlay_heatmap` — alpha-blend a (H, W) heatmap onto an RGB image.
* :func:`plot_overlay_panel` — convenience matplotlib helper for a
  baseline/knockout side-by-side comparison.
* :func:`decode_language_tokens` — given a tokenizer + token IDs + attention
  mask, return printable token strings suitable for bar-chart labels.
* :func:`plot_text_attention_bar` — bar-chart panel showing attention from
  a chosen query to each language prefix token, faceted by condition × layer.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Attention -> patch grid
# --------------------------------------------------------------------------- #


def view_attention_grid(
    attention: Tensor,
    *,
    query_index: int,
    view_token_start: int,
    view_token_end: int,
    patches_h: int,
    patches_w: int,
    head_reduction: str = "mean",
) -> np.ndarray:
    """Extract a 2D attention map over one camera view's patches.

    Args:
        attention: tensor of shape ``(num_heads, Q, K)`` or ``(B, num_heads, Q, K)``;
            if 4D the first batch element is used.
        query_index: which query token's attention to plot.
        view_token_start, view_token_end: the slice ``K[view_token_start:view_token_end]``
            that corresponds to one camera's vision patches.
        patches_h, patches_w: spatial layout of those patches (must satisfy
            ``view_token_end - view_token_start == patches_h * patches_w``).
        head_reduction: ``"mean"`` (default), ``"max"``, or ``"sum"`` over heads.

    Returns:
        ``(patches_h, patches_w)`` numpy array.
    """
    if attention.ndim == 4:
        attention = attention[0]
    if attention.ndim != 3:
        raise ValueError(f"attention must be 3D or 4D, got shape {tuple(attention.shape)}")
    n_patches = view_token_end - view_token_start
    if n_patches != patches_h * patches_w:
        raise ValueError(
            f"View token slice has {n_patches} patches but patches_h*patches_w={patches_h * patches_w}."
        )
    if not (0 <= query_index < attention.shape[1]):
        raise IndexError(
            f"query_index={query_index} out of range for attention with Q={attention.shape[1]}"
        )

    row = attention[:, query_index, view_token_start:view_token_end].to(dtype=torch.float32)
    if head_reduction == "mean":
        reduced = row.mean(dim=0)
    elif head_reduction == "max":
        reduced = row.max(dim=0).values
    elif head_reduction == "sum":
        reduced = row.sum(dim=0)
    else:
        raise ValueError(f"Unsupported head_reduction: {head_reduction!r}")
    grid = reduced.reshape(patches_h, patches_w).cpu().numpy()
    return grid


# --------------------------------------------------------------------------- #
# Heatmap overlay
# --------------------------------------------------------------------------- #


def _normalize_image(image) -> np.ndarray:
    """Coerce a CHW float [-1,1] / [0,1] tensor or HWC array into a (H,W,3) uint8 image."""
    if isinstance(image, Tensor):
        img = image.detach().to("cpu", dtype=torch.float32).numpy()
    else:
        img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in {1, 3} and img.shape[-1] not in {1, 3}:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    # Detect normalization range and rescale to [0, 1]
    lo, hi = float(img.min()), float(img.max())
    if lo < -0.05:
        # assume [-1, 1] normalization
        img = (img + 1.0) / 2.0
    elif hi > 1.5:
        # assume 0-255 range
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _bicubic_upsample(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize ``grid`` to ``(target_h, target_w)`` with bicubic interpolation."""
    src = torch.from_numpy(grid.astype(np.float32))[None, None]
    out = torch.nn.functional.interpolate(
        src, size=(target_h, target_w), mode="bicubic", align_corners=False
    )
    return out[0, 0].numpy()


def overlay_heatmap(
    image,
    attention_grid: np.ndarray,
    *,
    alpha: float = 0.55,
    colormap: str = "jet",
    normalize: bool = True,
) -> np.ndarray:
    """Alpha-blend an attention grid onto an RGB image.

    Args:
        image: source image (Tensor or array). Will be rescaled to ``[0, 255]`` uint8.
        attention_grid: ``(h, w)`` attention values. Upsampled to image size.
        alpha: heatmap transparency (0 = image only, 1 = heatmap only).
        colormap: a matplotlib colormap name.
        normalize: if True, scale the attention grid to ``[0, 1]`` per-call. Use
            ``False`` and pre-normalize when comparing across panels.

    Returns:
        ``(H, W, 3)`` uint8 numpy image with the heatmap blended in.
    """
    import matplotlib.cm as cm

    base = _normalize_image(image)
    h, w, _ = base.shape
    up = _bicubic_upsample(attention_grid, h, w)
    if normalize:
        lo, hi = float(up.min()), float(up.max())
        if hi - lo > 1e-9:
            up = (up - lo) / (hi - lo)
        else:
            up = np.zeros_like(up)
    cmap = cm.get_cmap(colormap)
    heat_rgba = cmap(np.clip(up, 0, 1))  # (H, W, 4)
    heat_rgb = (heat_rgba[..., :3] * 255).astype(np.uint8)
    blended = ((1 - alpha) * base + alpha * heat_rgb).astype(np.uint8)
    return blended


# --------------------------------------------------------------------------- #
# Side-by-side comparison panel
# --------------------------------------------------------------------------- #


def plot_overlay_panel(
    images_per_view: list,
    attentions_per_condition: dict[str, dict[int, Tensor]],
    *,
    query_index: int,
    view_token_ranges: list[tuple[int, int]],
    patches_per_view: list[tuple[int, int]],
    layer_indices: list[int],
    view_names: list[str] | None = None,
    save_path: Path | None = None,
    head_reduction: str = "mean",
    figsize_per_cell: tuple[float, float] = (2.2, 2.2),
    suptitle: str | None = None,
):
    """Render a (conditions × layers) grid of attention overlays, one row per view.

    Args:
        images_per_view: list of original camera images, one per view.
        attentions_per_condition: mapping ``condition_name -> {layer_idx: tensor}`` where
            tensor is ``(B, num_heads, Q, K)`` or ``(num_heads, Q, K)``.
        query_index: which query position (suffix index) to visualize.
        view_token_ranges: ``(start, end)`` slice into the K dimension per view.
        patches_per_view: ``(patches_h, patches_w)`` per view.
        layer_indices: which layers to plot (columns within each row).
        view_names: optional labels for the views (defaults to ``view0/view1/...``).
        save_path: if given, the figure is written there.
    """
    import matplotlib.pyplot as plt

    n_views = len(images_per_view)
    n_conditions = len(attentions_per_condition)
    n_layers = len(layer_indices)
    if view_names is None:
        view_names = [f"view{i}" for i in range(n_views)]
    if len(view_token_ranges) != n_views or len(patches_per_view) != n_views:
        raise ValueError("view_token_ranges / patches_per_view must match images_per_view length.")

    fig_h = figsize_per_cell[1] * n_views * n_conditions + 0.6
    fig_w = figsize_per_cell[0] * (n_layers + 1) + 0.4
    fig, axes = plt.subplots(
        n_conditions * n_views, n_layers + 1,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    condition_names = list(attentions_per_condition)
    for c_idx, cond in enumerate(condition_names):
        cond_attn = attentions_per_condition[cond]
        for v_idx in range(n_views):
            row = c_idx * n_views + v_idx
            # First column: original image
            ax = axes[row][0]
            ax.imshow(_normalize_image(images_per_view[v_idx]))
            ax.set_xticks([])
            ax.set_yticks([])
            if v_idx == 0:
                ax.set_ylabel(f"{cond}\n{view_names[v_idx]}", fontsize=9)
            else:
                ax.set_ylabel(view_names[v_idx], fontsize=9)
            if c_idx == 0 and v_idx == 0:
                ax.set_title("input frame", fontsize=9)

            # Subsequent columns: per-layer attention overlay
            for col_off, L in enumerate(layer_indices):
                ax = axes[row][col_off + 1]
                if L not in cond_attn:
                    ax.axis("off")
                    continue
                vt_start, vt_end = view_token_ranges[v_idx]
                p_h, p_w = patches_per_view[v_idx]
                grid = view_attention_grid(
                    cond_attn[L],
                    query_index=query_index,
                    view_token_start=vt_start,
                    view_token_end=vt_end,
                    patches_h=p_h,
                    patches_w=p_w,
                    head_reduction=head_reduction,
                )
                overlay = overlay_heatmap(images_per_view[v_idx], grid)
                ax.imshow(overlay)
                ax.set_xticks([])
                ax.set_yticks([])
                if c_idx == 0 and v_idx == 0:
                    ax.set_title(f"layer {L + 1}", fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.0)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Text-token attention bar
# --------------------------------------------------------------------------- #


def _prettify_token(piece: str) -> str:
    """Make a tokenizer piece human-readable for bar-chart labels.

    Handles the SentencePiece ``▁`` word-boundary marker and the BPE ``Ġ``
    marker, and replaces whitespace with visible glyphs so empty-looking
    tokens stay visible on the x-axis.
    """
    if not piece:
        return "∅"
    # Strip BPE / SentencePiece word-boundary markers but mark a leading space.
    if piece.startswith("▁"):
        piece = " " + piece[1:]
    elif piece.startswith("Ġ"):
        piece = " " + piece[1:]
    # Replace control characters with visible substitutes.
    piece = piece.replace("\n", "⏎").replace("\t", "⇥")
    if piece.strip() == "":
        piece = "·" * len(piece) if piece else "∅"
    return piece


def decode_language_tokens(
    tokenizer,
    token_ids,
    attention_mask=None,
    *,
    max_tokens: int | None = None,
) -> tuple[list[int], list[str]]:
    """Return (valid_position_indices, pretty_token_strings) for the language prefix.

    Args:
        tokenizer: a HuggingFace ``PreTrainedTokenizerBase`` (fast or slow).
        token_ids: tensor / sequence of token IDs, shape ``(B, L)`` or ``(L,)``.
            Only the first sample is used.
        attention_mask: optional boolean / 0-1 tensor of the same shape. Tokens
            with mask=0 are dropped from the returned lists.
        max_tokens: optional cap on the number of tokens returned (handy for
            long prompts that would render unreadable on a bar chart).

    Returns:
        - ``positions``: list of valid token positions within the original
          language sequence (``0 <= pos < L``); these are the indices used to
          slice ``attention[..., language_start + pos]``.
        - ``texts``: human-readable strings, ``len(texts) == len(positions)``.
    """
    if tokenizer is None:
        raise ValueError("decode_language_tokens requires a tokenizer.")
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim >= 2:
            token_ids = token_ids[0]
        ids = token_ids.detach().to("cpu").tolist()
    else:
        ids = list(token_ids)
        if ids and isinstance(ids[0], (list, tuple)):
            ids = list(ids[0])
    if attention_mask is None:
        valid = [True] * len(ids)
    else:
        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.ndim >= 2:
                attention_mask = attention_mask[0]
            valid = [bool(x) for x in attention_mask.detach().to("cpu").tolist()]
        else:
            mask = list(attention_mask)
            if mask and isinstance(mask[0], (list, tuple)):
                mask = list(mask[0])
            valid = [bool(x) for x in mask]

    pieces = tokenizer.convert_ids_to_tokens(ids)
    positions: list[int] = []
    texts: list[str] = []
    for idx, (piece, keep) in enumerate(zip(pieces, valid, strict=True)):
        if not keep:
            continue
        positions.append(idx)
        texts.append(_prettify_token(piece))
        if max_tokens is not None and len(positions) >= max_tokens:
            break
    return positions, texts


def plot_text_attention_bar(
    attentions_per_condition: dict[str, dict[int, Tensor]],
    *,
    query_index: int,
    language_token_start: int,
    valid_positions: list[int],
    token_texts: list[str],
    layer_indices: list[int],
    head_reduction: str = "mean",
    save_path: Path | None = None,
    figsize_per_cell: tuple[float, float] = (6.0, 1.6),
    suptitle: str | None = None,
    shared_y: bool = True,
):
    """Bar-chart panel of (condition × layer) attention from one query to language tokens.

    Each row is a condition (e.g. ``baseline``, ``vision->action@1-5``) and
    each column a layer. Bars correspond to the language prefix tokens decoded
    by :func:`decode_language_tokens`; the y-value is the attention weight from
    ``query_index`` to that token, averaged (or maxed/summed) across heads.

    Args:
        attentions_per_condition: ``{condition_name -> {layer_idx -> tensor}}``,
            tensors of shape ``(B, num_heads, Q, K)`` or ``(num_heads, Q, K)``.
        query_index: which query position's attention to visualize.
        language_token_start: offset of the first language token within K.
        valid_positions: language-relative positions to slice (output of
            :func:`decode_language_tokens`).
        token_texts: matching labels for x-tick bars.
        layer_indices: which layers to draw (columns).
        head_reduction: ``"mean"``, ``"max"``, or ``"sum"``.
        shared_y: if True, all subplots share a common y-axis range derived
            from the global max attention — easier visual comparison.

    Returns:
        the created ``matplotlib.figure.Figure``.
    """
    import matplotlib.pyplot as plt

    if not valid_positions:
        raise ValueError("valid_positions is empty; nothing to plot.")
    if len(valid_positions) != len(token_texts):
        raise ValueError("valid_positions and token_texts must have the same length.")

    condition_names = list(attentions_per_condition)
    n_rows = len(condition_names)
    n_cols = len(layer_indices)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("attentions_per_condition or layer_indices is empty.")

    abs_positions = [language_token_start + p for p in valid_positions]

    # Precompute attention rows so we can determine shared y-limit.
    rows: dict[tuple[str, int], np.ndarray | None] = {}
    global_max = 0.0
    for cond in condition_names:
        per_layer = attentions_per_condition[cond]
        for L in layer_indices:
            tensor = per_layer.get(L)
            if tensor is None:
                rows[(cond, L)] = None
                continue
            attn = tensor[0] if tensor.ndim == 4 else tensor
            if attn.ndim != 3:
                raise ValueError(f"attention for layer {L} should be 3D/4D, got {tuple(attn.shape)}")
            if not (0 <= query_index < attn.shape[1]):
                raise IndexError(
                    f"query_index={query_index} out of range for Q={attn.shape[1]}"
                )
            row = attn[:, query_index, abs_positions].to(dtype=torch.float32)
            if head_reduction == "mean":
                reduced = row.mean(dim=0)
            elif head_reduction == "max":
                reduced = row.max(dim=0).values
            elif head_reduction == "sum":
                reduced = row.sum(dim=0)
            else:
                raise ValueError(f"Unsupported head_reduction: {head_reduction!r}")
            arr = reduced.cpu().numpy()
            rows[(cond, L)] = arr
            global_max = max(global_max, float(arr.max(initial=0.0)))

    fig_w = figsize_per_cell[0] * n_cols + 0.3
    fig_h = figsize_per_cell[1] * n_rows + 0.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False, sharey=shared_y)

    x = np.arange(len(token_texts))
    bar_color = "#3b6fb5"
    highlight_color = "#d96c3a"

    for r_idx, cond in enumerate(condition_names):
        for c_idx, L in enumerate(layer_indices):
            ax = axes[r_idx][c_idx]
            arr = rows[(cond, L)]
            if arr is None:
                ax.axis("off")
                continue
            # Highlight the top-3 tokens with a different color.
            if arr.size >= 3:
                top_idx = set(np.argsort(arr)[-3:])
            else:
                top_idx = set(range(arr.size))
            colors = [highlight_color if i in top_idx else bar_color for i in range(len(arr))]
            ax.bar(x, arr, color=colors, edgecolor="white", linewidth=0.4)

            ax.set_xticks(x)
            ax.set_xticklabels(token_texts, rotation=45, ha="right", fontsize=7)
            ax.tick_params(axis="x", which="major", pad=1)
            ax.grid(axis="y", linewidth=0.3, color="#cbd0d6", alpha=0.7)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            if r_idx == 0:
                ax.set_title(f"layer {L + 1}", fontsize=9)
            if c_idx == 0:
                ax.set_ylabel(f"{cond}\nattn", fontsize=8)
            if shared_y and global_max > 0:
                ax.set_ylim(0, global_max * 1.05)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.0)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
