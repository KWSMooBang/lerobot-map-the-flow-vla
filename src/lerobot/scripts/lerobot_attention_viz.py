#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Render attention overlay figures for a VLA policy under different knockout conditions.

This script is the qualitative companion to ``lerobot-map-the-flow``. It runs a
short baseline rollout, replays selected observations through the policy with and
without attention knockout, and saves overlay PNGs showing where the policy
"looks" for each layer × condition combination.

Example:

```
lerobot-attention-viz \
  --policy.path=downloads/pi0_libero_finetuned_v044 \
  --env.type=libero --env.task=libero_object \
  --eval.batch_size=1 --eval.n_episodes=1 \
  --analysis.routes='[vision->action,language->action]' \
  --analysis.layer_indices='[2,8,14]' \
  --analysis.window_size=5 \
  --analysis.frame_indices='[0,4,8,12]' \
  --analysis.query_position=0
```

Output: one PNG per ``(episode_step, condition)`` containing a grid of
``(views × layers)`` overlays.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from termcolor import colored

# Make matplotlib importable in headless env before any matplotlib import.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lerobot_matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

from lerobot import envs, policies  # noqa: F401
from lerobot.analysis.attention_capture import AttentionRecorder
from lerobot.analysis.attention_overlay import (
    decode_language_tokens,
    plot_overlay_panel,
    plot_text_attention_bar,
)
from lerobot.analysis.map_the_flow import AttentionKnockoutSpec, parse_route_rules
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_map_the_flow import (
    TrajectoryRecorder,
    _condition_name,
    _clean_choice,
    _move_batch_to,
    _predict_action_chunk_with_knockout,
    _set_policy_knockout,
)
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class AttentionVizConfig:
    """Selection of conditions, frames, and layers to visualize."""

    # One or more route specifiers; each becomes a knockout condition. Layer
    # range is centered on every entry in ``layer_centers`` using ``window_size``,
    # exactly like ``MapTheFlowAnalysisConfig`` so the chosen knockouts match
    # what was tested in the quantitative sweep.
    routes: list[str] = field(default_factory=lambda: ["vision->action", "language->action"])
    layer_centers: list[int] = field(default_factory=lambda: [3, 9, 15])
    window_size: int = 5
    max_layer: int = 18

    # Which transformer to capture from: ``"action_expert"`` (default, sees
    # vision/language → action attention) or ``"vlm"`` (prefix self-attention).
    capture_target: str = "action_expert"
    # Layer indices (0-indexed) to capture attention from.
    layer_indices: list[int] = field(default_factory=lambda: [2, 8, 14])

    # Which rollout step indices to visualize. The baseline rollout's
    # ``predict_action_chunk`` snapshots are indexed in call order.
    frame_indices: list[int] = field(default_factory=lambda: [0, 4, 8])

    # Which suffix query position to plot. For pi0 the suffix is
    # ``[state, action_0, ..., action_{chunk-1}]`` so query 1 = first action token.
    # For pi05 the suffix has no state, so query 0 = first action token.
    query_position: int = 0
    # Reduction across attention heads for overlay rendering.
    head_reduction: str = "mean"

    # Number of patches per camera view (PaliGemma SigLIP defaults to 16×16=256).
    patches_per_view: tuple[int, int] = (16, 16)
    # Number of camera views the policy expects (auto-detected if 0).
    n_views: int = 0
    # Save one compact baseline-vs-condition figure per knockout condition.
    # This avoids unreadably tall panels when routes x layer_centers is large.
    split_conditions: bool = True
    # Also save the legacy all-conditions-in-one panel for each frame.
    save_combined_panel: bool = False

    # ── Text-token attention bar chart ─────────────────────────────────
    # When ``true`` (and a tokenizer can be located), an additional PNG per
    # frame is written showing attention from ``query_position`` to every
    # language prefix token, faceted by condition × layer. This complements
    # the vision overlay with a discrete textual breakdown of where the
    # query is reading the instruction from.
    plot_text_bar: bool = True
    # Hard cap on number of language tokens shown on the bar chart x-axis
    # (long prompts become unreadable beyond ~60 bars). Tokens past the cap
    # are silently dropped from the figure but not from the model input.
    text_bar_max_tokens: int = 60
    # Add the decoded task instruction to image-overlay titles so the visual
    # attention panel is interpretable without opening the text-token plot.
    show_instruction_caption: bool = True
    instruction_caption_max_chars: int = 180

    def __post_init__(self) -> None:
        self.capture_target = _clean_choice(
            self.capture_target,
            field_name="capture_target",
            allowed={"action_expert", "vlm"},
        )
        self.head_reduction = _clean_choice(
            self.head_reduction,
            field_name="head_reduction",
            allowed={"mean", "max", "sum"},
        )


@dataclass
class AttentionVizPipelineConfig:
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    analysis: AttentionVizConfig = field(default_factory=AttentionVizConfig)
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    rename_map: dict[str, str] = field(default_factory=dict)
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        else:
            logging.warning("No pretrained path was provided; policy will be initialized from scratch.")

        if not self.job_name:
            policy_name = self.policy.type if self.policy is not None else "scratch"
            self.job_name = f"attention_viz_{self.env.type}_{policy_name}"

        if not self.output_dir:
            now = dt.datetime.now()
            sub_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/attention_viz") / sub_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_knockout_specs(cfg: AttentionVizConfig) -> list[AttentionKnockoutSpec]:
    """Same windowed knockouts as MapTheFlowAnalysisConfig but flattened to one spec per route."""
    if cfg.window_size < 1:
        raise ValueError("window_size must be >= 1")
    if not cfg.layer_centers:
        raise ValueError("Specify at least one layer center via --analysis.layer_centers='[...]'")
    half_window = cfg.window_size // 2

    specs: list[AttentionKnockoutSpec] = []
    for center in cfg.layer_centers:
        if center < 1 or center > cfg.max_layer:
            raise ValueError(f"Layer center {center} outside [1, {cfg.max_layer}].")
        start = max(1, center - half_window)
        end = min(cfg.max_layer, center + half_window)
        layer_range = ((start, end),)
        for route in cfg.routes:
            rules = parse_route_rules(route, default_layer_ranges=layer_range)
            route_name = route.replace("<->", "_bidir_").replace("->", "_to_")
            specs.append(
                AttentionKnockoutSpec(
                    rules=rules,
                    mode="block",
                    name=f"{route_name}_center_L{center}_window_L{start}-{end}",
                )
            )
    return specs


def _resolve_target_transformer(policy, target: str):
    """Find the transformer module whose attention layers we should capture from.

    Handles pi0, pi05, smolvla, and groot-style policies by searching for the
    standard attribute paths used by their map-the-flow integration code.
    """
    candidates_action_expert = [
        ("model.paligemma_with_expert.gemma_expert.model", "pi0/pi05 action expert"),
        ("model.smolvlm_with_expert.expert", "smolvla action expert"),
        ("model.action_head.model", "groot DiT action head"),
    ]
    candidates_vlm = [
        ("model.paligemma_with_expert.paligemma.model.language_model", "pi0/pi05 VLM"),
        ("model.smolvlm_with_expert.smolvlm.model.text_model", "smolvla VLM"),
        ("model.backbone.eagle_model.language_model", "groot Eagle VLM"),
    ]
    candidates = candidates_action_expert if target == "action_expert" else candidates_vlm
    for path, label in candidates:
        mod = policy
        for attr in path.split("."):
            mod = getattr(mod, attr, None)
            if mod is None:
                break
        if mod is not None and (hasattr(mod, "layers") or hasattr(getattr(mod, "model", None), "layers")):
            logging.info(
                colored("Capture target:", "cyan", attrs=["bold"]) + f" {label} ({path})"
            )
            return mod
    raise RuntimeError(
        f"Could not locate a {target} transformer with .layers on policy "
        f"{type(policy).__name__}. Inspect the model and pass a custom path."
    )


@dataclass
class AttentionTokenLayout:
    view_token_ranges: list[tuple[int, int]]
    patches_per_view: list[tuple[int, int]]
    language_token_start: int
    prefix_token_count: int
    image_keys: list[str]


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:160]


def _short_condition_name(value: str) -> str:
    if value == "baseline":
        return value
    return (
        value.replace("_bidir_", "<->")
        .replace("_to_", "->")
        .replace("_center_", "\ncenter_")
        .replace("_window_", " ")
    )


def _ordered_image_keys(policy, batch: dict[str, Any]) -> list[str]:
    """Return image keys in the same order the policy uses for prefix tokens."""
    configured = list(getattr(getattr(policy, "config", None), "image_features", []) or [])
    keys = [key for key in configured if key in batch]
    if keys:
        return keys
    return sorted(k for k in batch if k.startswith(OBS_IMAGES + "."))


def _extract_view_images(batch: dict[str, Any], image_keys: list[str]) -> list[torch.Tensor]:
    """Return OBS_IMAGES tensors in model input order."""
    images: list[torch.Tensor] = []
    for k in image_keys:
        tensor = batch[k]
        if not isinstance(tensor, torch.Tensor):
            continue
        # Strip batch dim, take first sample
        img = tensor[0] if tensor.ndim >= 4 else tensor
        images.append(img.detach().to("cpu"))
    return images


def _compute_view_token_ranges(
    n_views: int,
    patches_per_view: tuple[int, int],
) -> list[tuple[int, int]]:
    p_h, p_w = patches_per_view
    n = p_h * p_w
    return [(i * n, (i + 1) * n) for i in range(n_views)]


def _patch_grid_for_count(token_count: int, preferred: tuple[int, int]) -> tuple[int, int]:
    if token_count == preferred[0] * preferred[1]:
        return preferred
    side = int(round(token_count**0.5))
    if side * side == token_count:
        return (side, side)
    raise ValueError(
        f"Cannot reshape {token_count} view tokens into the preferred patch grid "
        f"{preferred}; pass --analysis.patches_per_view='[H,W]' explicitly."
    )


def _fallback_token_layout(
    *,
    image_keys: list[str],
    n_views: int,
    patches_per_view: tuple[int, int],
) -> AttentionTokenLayout:
    view_token_ranges = _compute_view_token_ranges(n_views, patches_per_view)
    patch_grids = [patches_per_view] * n_views
    language_token_start = view_token_ranges[-1][1] if view_token_ranges else 0
    return AttentionTokenLayout(
        view_token_ranges=view_token_ranges,
        patches_per_view=patch_grids,
        language_token_start=language_token_start,
        prefix_token_count=language_token_start,
        image_keys=image_keys[:n_views],
    )


def _infer_token_layout(
    policy,
    batch_cpu: dict[str, Any],
    *,
    device: torch.device,
    image_keys: list[str],
    requested_n_views: int,
    preferred_patches_per_view: tuple[int, int],
) -> AttentionTokenLayout:
    """Infer view/language key offsets from the policy's real prefix embedding path."""
    n_views = requested_n_views or len(image_keys)
    fallback = _fallback_token_layout(
        image_keys=image_keys,
        n_views=n_views,
        patches_per_view=preferred_patches_per_view,
    )
    if not hasattr(policy, "_preprocess_images") or not hasattr(getattr(policy, "model", None), "embed_prefix"):
        return fallback

    try:
        batch = _move_batch_to(batch_cpu, device)
        with torch.inference_mode():
            images, img_masks = policy._preprocess_images(batch)  # noqa: SLF001
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            embed_prefix = policy.model.embed_prefix

            try:
                prefix_result = embed_prefix(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state=batch.get(OBS_STATE),
                    return_token_counts=True,
                )
                view_token_counts = [int(x) for x in prefix_result[3]]
                prefix_token_count = int(prefix_result[0].shape[1])
            except TypeError:
                prefix_embs, _, _ = embed_prefix(images, img_masks, lang_tokens, lang_masks)
                language_tokens = int(lang_tokens.shape[1])
                vision_tokens = int(prefix_embs.shape[1]) - language_tokens
                if len(images) > 0 and vision_tokens % len(images) == 0:
                    view_token_counts = [vision_tokens // len(images)] * len(images)
                else:
                    return fallback
                prefix_token_count = int(prefix_embs.shape[1])

        if requested_n_views:
            view_token_counts = view_token_counts[:requested_n_views]
        offset = 0
        view_token_ranges: list[tuple[int, int]] = []
        patch_grids: list[tuple[int, int]] = []
        for token_count in view_token_counts:
            view_token_ranges.append((offset, offset + token_count))
            patch_grids.append(_patch_grid_for_count(token_count, preferred_patches_per_view))
            offset += token_count

        return AttentionTokenLayout(
            view_token_ranges=view_token_ranges,
            patches_per_view=patch_grids,
            language_token_start=offset,
            prefix_token_count=prefix_token_count,
            image_keys=image_keys[: len(view_token_ranges)],
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not infer attention token layout from model prefix; using fallback: %s", exc)
        return fallback


def _find_input_tokenizer(preprocessor):
    """Walk a PolicyProcessorPipeline and return the first language tokenizer found.

    Supports both ``TokenizerProcessorStep`` (uses ``input_tokenizer``) and any
    other step that happens to expose a ``tokenizer`` attribute.
    """
    steps = getattr(preprocessor, "steps", None)
    if steps is None:
        steps = getattr(preprocessor, "_steps", [])
    for step in steps:
        tok = getattr(step, "input_tokenizer", None) or getattr(step, "tokenizer", None)
        if tok is not None and hasattr(tok, "convert_ids_to_tokens"):
            return tok
    return None


def _decode_instruction_text(
    tokenizer,
    batch: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    """Decode the first sample's instruction from raw task text or language tokens."""
    task = batch.get("task")
    if isinstance(task, str):
        text = task
    elif isinstance(task, (list, tuple)) and task:
        text = str(task[0])
    else:
        if tokenizer is None or OBS_LANGUAGE_TOKENS not in batch:
            return ""
        token_ids = batch[OBS_LANGUAGE_TOKENS]
        if isinstance(token_ids, torch.Tensor):
            ids = (
                token_ids[0].detach().to("cpu").tolist()
                if token_ids.ndim >= 2
                else token_ids.detach().to("cpu").tolist()
            )
        else:
            ids = (
                list(token_ids[0])
                if token_ids and isinstance(token_ids[0], (list, tuple))
                else list(token_ids)
            )

        attention_mask = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                mask = (
                    attention_mask[0].detach().to("cpu").tolist()
                    if attention_mask.ndim >= 2
                    else attention_mask.detach().to("cpu").tolist()
                )
            else:
                mask = (
                    list(attention_mask[0])
                    if attention_mask and isinstance(attention_mask[0], (list, tuple))
                    else list(attention_mask)
                )
            ids = [token_id for token_id, keep in zip(ids, mask, strict=False) if bool(keep)]

        try:
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except TypeError:
            text = tokenizer.decode(ids, skip_special_tokens=True)

    text = " ".join(str(text).split())
    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = textwrap.shorten(text, width=max_chars, placeholder="...")
    return text


# --------------------------------------------------------------------------- #
# Capture pass
# --------------------------------------------------------------------------- #


def _capture_attention_for_snapshot(
    policy,
    batch_cpu: dict[str, Any],
    device: torch.device,
    transformer,
    layer_indices: list[int],
    spec: AttentionKnockoutSpec | None,
) -> dict[int, torch.Tensor]:
    """Run a single forward pass with knockout spec and capture attention probs."""
    _set_policy_knockout(policy, spec)
    policy.eval()
    batch = _move_batch_to(batch_cpu, device)
    with AttentionRecorder(transformer, layer_indices) as recorder, torch.inference_mode():
        _ = _predict_action_chunk_with_knockout(policy, batch, spec)
    if not recorder.captured:
        raise RuntimeError(
            "AttentionRecorder did not capture anything. Make sure the chosen transformer "
            "uses ``_attn_implementation='eager'`` and that ``layer_indices`` are within range."
        )
    return recorder.captured


def _reduced_attention_row(attn: torch.Tensor, *, query_index: int, head_reduction: str) -> torch.Tensor:
    tensor = attn[0] if attn.ndim == 4 else attn
    if tensor.ndim != 3 or not (0 <= query_index < tensor.shape[1]):
        return torch.empty(0)
    row = tensor[:, query_index].to(dtype=torch.float32)
    if head_reduction == "mean":
        return row.mean(dim=0)
    if head_reduction == "max":
        return row.max(dim=0).values
    if head_reduction == "sum":
        return row.sum(dim=0)
    raise ValueError(f"Unsupported head_reduction: {head_reduction!r}")


def _attention_debug_payload(
    *,
    frame_idx: int,
    layout: AttentionTokenLayout,
    attentions: dict[str, dict[int, torch.Tensor]],
    layer_indices: list[int],
    query_index: int,
    head_reduction: str,
    language_positions: list[int],
    instruction_text: str = "",
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for cond, per_layer in attentions.items():
        layer_info = {}
        for layer_idx in layer_indices:
            tensor = per_layer.get(layer_idx)
            if tensor is None:
                continue
            row = _reduced_attention_row(tensor, query_index=query_index, head_reduction=head_reduction)
            if row.numel() == 0:
                continue
            lang_abs = [
                layout.language_token_start + pos
                for pos in language_positions
                if layout.language_token_start + pos < row.shape[0]
            ]
            layer_info[str(layer_idx)] = {
                "attention_shape": list(tensor.shape),
                "row_sum": float(row.sum().item()),
                "view_mass": [
                    float(row[start:end].sum().item())
                    for start, end in layout.view_token_ranges
                    if start < row.shape[0]
                ],
                "language_mass": float(row[lang_abs].sum().item()) if lang_abs else 0.0,
                "language_token_start": layout.language_token_start,
                "key_length": int(row.shape[0]),
            }
        conditions[cond] = layer_info
    return {
        "frame_idx": frame_idx,
        "instruction": instruction_text,
        "layout": {
            "view_token_ranges": layout.view_token_ranges,
            "patches_per_view": layout.patches_per_view,
            "language_token_start": layout.language_token_start,
            "prefix_token_count": layout.prefix_token_count,
            "image_keys": layout.image_keys,
        },
        "conditions": conditions,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


@parser.wrap()
def attention_viz_main(cfg: AttentionVizPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    if cfg.policy is None:
        raise ValueError("Pass --policy.path=...")
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if cfg.seed is not None:
        set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    # ── Build policy & processors ─────────────────────────────────────
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=cfg.policy
    )

    transformer = _resolve_target_transformer(policy, cfg.analysis.capture_target)

    tokenizer = _find_input_tokenizer(preprocessor)
    if cfg.analysis.plot_text_bar and tokenizer is None:
        logging.warning(
            "plot_text_bar=True but no tokenizer was found in the preprocessor pipeline; "
            "text-token bar charts will be skipped. Set plot_text_bar=False to silence this."
        )

    # ── Baseline rollout with TrajectoryRecorder ──────────────────────
    from lerobot.scripts.lerobot_eval import eval_policy_all

    rec = TrajectoryRecorder(
        policy,
        max_snapshots=max(cfg.analysis.frame_indices) + 1 if cfg.analysis.frame_indices else 1024,
        stride=1,
        image_dtype=torch.float32,  # keep precision for visualization
    )
    envs_for_run = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    try:
        with rec, torch.no_grad():
            eval_policy_all(
                envs=envs_for_run,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=0,
                videos_dir=output_dir / "videos",
                start_seed=cfg.seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
            )
    finally:
        close_envs(envs_for_run)

    snapshots = rec.snapshots
    logging.info(colored("Captured", "cyan", attrs=["bold"]) + f" {len(snapshots)} snapshots")
    if not snapshots:
        raise RuntimeError("No snapshots captured. The rollout never called predict_action_chunk.")

    # ── Iterate selected frames × conditions ──────────────────────────
    specs = _build_knockout_specs(cfg.analysis)
    # Reset knockout to baseline for the first pass per frame.
    frame_iter = sorted(set(cfg.analysis.frame_indices))
    for frame_idx in frame_iter:
        if frame_idx >= len(snapshots):
            logging.warning("frame_idx=%d exceeds %d snapshots; skipping.", frame_idx, len(snapshots))
            continue
        batch_cpu, _ = snapshots[frame_idx]
        image_keys = _ordered_image_keys(policy, batch_cpu)
        if not image_keys:
            raise RuntimeError("Could not auto-detect any OBS_IMAGES tensors in the batch.")
        layout = _infer_token_layout(
            policy,
            batch_cpu,
            device=device,
            image_keys=image_keys,
            requested_n_views=cfg.analysis.n_views,
            preferred_patches_per_view=cfg.analysis.patches_per_view,
        )
        images = _extract_view_images(batch_cpu, layout.image_keys)
        n_views = len(images)
        if n_views == 0:
            raise RuntimeError("No plottable view images were found after applying model image order.")
        view_names = [key.removeprefix(OBS_IMAGES + ".") for key in layout.image_keys[:n_views]]

        # Collect attention per condition (baseline + each knockout)
        attentions: dict[str, dict[int, torch.Tensor]] = {}
        for spec in [None, *specs]:
            cond_name = _condition_name(spec)
            logging.info(
                colored("Frame", "magenta")
                + f" {frame_idx}  "
                + colored("condition", "magenta")
                + f" {cond_name}"
            )
            attentions[cond_name] = _capture_attention_for_snapshot(
                policy,
                batch_cpu,
                device,
                transformer,
                cfg.analysis.layer_indices,
                spec,
            )

        # Text-token attention bar chart, when feasible.
        positions: list[int] = []
        token_texts: list[str] = []
        instruction_text = ""
        if cfg.analysis.show_instruction_caption:
            try:
                instruction_text = _decode_instruction_text(
                    tokenizer,
                    batch_cpu,
                    max_chars=cfg.analysis.instruction_caption_max_chars,
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to decode instruction caption: %s", exc)
                instruction_text = ""
        if cfg.analysis.plot_text_bar and tokenizer is not None:
            lang_ids = batch_cpu.get(OBS_LANGUAGE_TOKENS)
            lang_mask = batch_cpu.get(OBS_LANGUAGE_ATTENTION_MASK)
            if lang_ids is None:
                logging.warning(
                    "Frame %d: %s not in batch; skipping text-bar.",
                    frame_idx, OBS_LANGUAGE_TOKENS,
                )
            else:
                try:
                    positions, token_texts = decode_language_tokens(
                        tokenizer,
                        lang_ids,
                        lang_mask,
                        max_tokens=cfg.analysis.text_bar_max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.warning("Failed to decode language tokens: %s", exc)
                    positions, token_texts = [], []

        debug_payload = _attention_debug_payload(
            frame_idx=frame_idx,
            layout=layout,
            attentions=attentions,
            layer_indices=cfg.analysis.layer_indices,
            query_index=cfg.analysis.query_position,
            head_reduction=cfg.analysis.head_reduction,
            language_positions=positions,
            instruction_text=instruction_text,
        )
        debug_path = output_dir / f"attention_debug_frame{frame_idx:04d}.json"
        with open(debug_path, "w") as f:
            json.dump(debug_payload, f, indent=2)
        logging.info(colored("Saved", "green") + f" {debug_path}")

        import matplotlib.pyplot as plt

        def save_panels(panel_attentions: dict[str, dict[int, torch.Tensor]], suffix: str) -> None:
            display_attentions = {
                _short_condition_name(cond): per_layer for cond, per_layer in panel_attentions.items()
            }
            save_path = output_dir / f"attention_frame{frame_idx:04d}_{suffix}.png"
            title = (
                f"{cfg.policy.type if cfg.policy else 'policy'} · {cfg.env.type}/{cfg.env.task} · "
                f"frame {frame_idx}, query pos {cfg.analysis.query_position}, "
                f"head={cfg.analysis.head_reduction}"
            )
            if instruction_text:
                title += "\nInstruction: " + textwrap.fill(instruction_text, width=110)
            fig = plot_overlay_panel(
                images_per_view=images,
                attentions_per_condition=display_attentions,
                query_index=cfg.analysis.query_position,
                view_token_ranges=layout.view_token_ranges[:n_views],
                patches_per_view=layout.patches_per_view[:n_views],
                view_names=view_names,
                layer_indices=cfg.analysis.layer_indices,
                head_reduction=cfg.analysis.head_reduction,
                save_path=save_path,
                suptitle=title,
            )
            plt.close(fig)
            logging.info(colored("Saved", "green") + f" {save_path}")

            if cfg.analysis.plot_text_bar and tokenizer is not None and positions:
                bar_save_path = output_dir / f"text_attention_frame{frame_idx:04d}_{suffix}.png"
                fig = plot_text_attention_bar(
                    attentions_per_condition=display_attentions,
                    query_index=cfg.analysis.query_position,
                    language_token_start=layout.language_token_start,
                    valid_positions=positions,
                    token_texts=token_texts,
                    layer_indices=cfg.analysis.layer_indices,
                    head_reduction=cfg.analysis.head_reduction,
                    save_path=bar_save_path,
                    suptitle=(
                        f"Language token attention · frame {frame_idx} · "
                        f"query pos {cfg.analysis.query_position} · "
                        f"{len(positions)} tokens · head={cfg.analysis.head_reduction}"
                    ),
                )
                plt.close(fig)
                logging.info(colored("Saved", "green") + f" {bar_save_path}")

        if cfg.analysis.split_conditions and "baseline" in attentions and len(attentions) > 1:
            for cond_name, cond_attn in attentions.items():
                if cond_name == "baseline":
                    continue
                save_panels(
                    {"baseline": attentions["baseline"], cond_name: cond_attn},
                    _safe_filename(cond_name),
                )

        if (not cfg.analysis.split_conditions) or cfg.analysis.save_combined_panel or len(attentions) == 1:
            save_panels(attentions, "combined")

    # Clear knockout before exit so the policy is left in a clean state.
    _set_policy_knockout(policy, None)


def main():
    init_logging()
    register_third_party_plugins()
    attention_viz_main()


if __name__ == "__main__":
    main()
