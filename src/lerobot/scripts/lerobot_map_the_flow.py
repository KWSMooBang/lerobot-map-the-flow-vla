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
"""Run Map the Flow-style attention-pathway interventions on VLA policies.

Example:

```
lerobot-map-the-flow \
    --policy.path=lerobot/pi0_libero_finetuned \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0]' \
    --eval.n_episodes=5 \
    --eval.batch_size=1 \
    --analysis.routes='[view0<->view1,vision->language,language->action,vision->action]' \
    --analysis.layer_centers='[1,2,3,4,5]' \
    --analysis.window_size=5
```

For a sparse-pathway sufficiency run:

```
lerobot-map-the-flow \
    --policy.path=lerobot/pi05_libero_finetuned \
    --env.type=libero \
    --env.task=libero_goal \
    --analysis.mode=keep_only \
    --analysis.pathways='[view0<->view1@1-10,language->action@11-18]'
```
"""

import datetime as dt
import gc
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from termcolor import colored

from lerobot import envs, policies  # noqa: F401
from lerobot.analysis.map_the_flow import AttentionKnockoutSpec, parse_layer_ranges, parse_route_rules
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


def _clean_choice(value: str, *, field_name: str, allowed: set[str]) -> str:
    cleaned = str(value).strip().strip("'\"`‘’“”")
    if cleaned not in allowed:
        raise ValueError(
            f"Invalid --analysis.{field_name}={value!r}. Expected one of {sorted(allowed)}. "
            "Use plain ASCII quotes in shell commands, e.g. --analysis.metric=action_mse."
        )
    return cleaned


@dataclass
class MapTheFlowAnalysisConfig:
    """Configuration for attention knockout sweeps."""

    # Routes are evaluated as the Cartesian product of routes x layer_ranges in block mode.
    routes: list[str] = field(
        default_factory=lambda: [
            "view0<->view1",
            "vision->language",
            "language->action",
            "vision->action",
            "state->action",
            "action->action",
        ]
    )
    # Main Map the Flow protocol: sweep a center layer l and block a window of k layers around it.
    # pi0/pi0.5 have 18-layer VLM and action-expert stacks, so k=5 is a conservative default.
    layer_centers: list[int] | None = None
    window_size: int = 5
    max_layer: int = 18
    # Optional explicit ranges such as ["1-5", "6-10"]. If set, this overrides centered windows.
    layer_ranges: list[str] = field(default_factory=list)
    # A list of route@layer-range rules used as a single condition. In keep_only mode,
    # these rules are the hypothesized effective pathway retained during evaluation.
    pathways: list[str] = field(default_factory=list)
    mode: str = "block"
    include_baseline: bool = True
    max_conditions: int | None = None
    max_episodes_rendered: int = 0

    # ── Action-MSE metric ────────────────────────────────────────────────
    # ``pc_success``  : original behavior — full env rollouts per condition.
    # ``action_mse``  : baseline runs full rollout while caching every
    #                   (observation, baseline_action_chunk). Each knockout
    #                   condition then re-runs only forward inference on
    #                   those cached observations and reports the MSE between
    #                   its action chunk and the baseline chunk. ~10× faster
    #                   than ``pc_success`` and a much finer signal.
    # ``both``        : compute pc_success AND action MSE for every
    #                   knockout condition (full rollout + MSE forward pass).
    metric: str = "pc_success"
    # Cap on how many ``(obs, baseline_chunk)`` snapshots to cache during the
    # baseline rollout. With batch_size=10 each snapshot is ~1.5 MB on CPU
    # (bfloat16 image cache); the default 2000 bound is roughly 3 GB.
    mse_max_snapshots: int = 2000
    # Capture every ``mse_snapshot_stride``-th ``predict_action_chunk`` call.
    # Useful when n_action_steps is small and the rollout produces more
    # snapshots than ``mse_max_snapshots`` would allow.
    mse_snapshot_stride: int = 1
    # Storage dtype for cached image tensors (state / tokens stay int/float32).
    # Use ``float16``/``bfloat16`` to halve memory; ``float32`` for bit-exact.
    mse_cache_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        self.mode = _clean_choice(self.mode, field_name="mode", allowed={"block", "keep_only"})
        self.metric = _clean_choice(
            self.metric, field_name="metric", allowed={"pc_success", "action_mse", "both"}
        )
        self.mse_cache_dtype = _clean_choice(
            self.mse_cache_dtype,
            field_name="mse_cache_dtype",
            allowed={"float32", "float16", "bfloat16"},
        )


@dataclass
class MapTheFlowPipelineConfig:
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    analysis: MapTheFlowAnalysisConfig = field(default_factory=MapTheFlowAnalysisConfig)
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
            self.job_name = f"map_the_flow_{self.env.type}_{policy_name}"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/map_the_flow") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _condition_name(spec: AttentionKnockoutSpec | None) -> str:
    if spec is None:
        return "baseline"
    if spec.name:
        return spec.name
    route_names = [f"{rule.source}_to_{rule.target}" for rule in spec.rules]
    return f"{spec.mode}_{'_'.join(route_names)}"


def _make_specs(analysis_cfg: MapTheFlowAnalysisConfig) -> list[AttentionKnockoutSpec]:
    if analysis_cfg.mode == "keep_only":
        if not analysis_cfg.pathways:
            raise ValueError("--analysis.mode=keep_only requires --analysis.pathways='[route@layers,...]'")
        rules = tuple(rule for pathway in analysis_cfg.pathways for rule in parse_route_rules(pathway))
        return [
            AttentionKnockoutSpec(
                rules=rules,
                mode="keep_only",
                name="effective_pathways",
            )
        ]

    if analysis_cfg.pathways:
        rules = tuple(rule for pathway in analysis_cfg.pathways for rule in parse_route_rules(pathway))
        return [
            AttentionKnockoutSpec(
                rules=rules,
                mode="block",
                name="blocked_pathways",
            )
        ]

    specs = []
    if analysis_cfg.layer_ranges:
        layer_windows = [
            (
                parse_layer_ranges(layer_range),
                "_".join(f"L{start}-{end}" for start, end in parse_layer_ranges(layer_range)),
            )
            for layer_range in analysis_cfg.layer_ranges
        ]
    else:
        if analysis_cfg.window_size < 1:
            raise ValueError("--analysis.window_size must be >= 1")
        if analysis_cfg.max_layer < 1:
            raise ValueError("--analysis.max_layer must be >= 1")
        half_window = analysis_cfg.window_size // 2
        centers = analysis_cfg.layer_centers or list(range(1, analysis_cfg.max_layer + 1))
        layer_windows = []
        for center in centers:
            if center < 1 or center > analysis_cfg.max_layer:
                raise ValueError(
                    f"Layer center {center} is outside [1, {analysis_cfg.max_layer}]. "
                    "Set --analysis.max_layer if your model has a different depth."
                )
            start = max(1, center - half_window)
            end = min(analysis_cfg.max_layer, center + half_window)
            layer_windows.append((((start, end),), f"center_L{center}_window_L{start}-{end}"))

    for parsed_range, range_name in layer_windows:
        for route in analysis_cfg.routes:
            rules = parse_route_rules(route, default_layer_ranges=parsed_range)
            route_name = route.replace("<->", "_bidir_").replace("->", "_to_")
            specs.append(
                AttentionKnockoutSpec(
                    rules=rules,
                    mode="block",
                    name=f"{route_name}_{range_name}",
                )
            )
    if analysis_cfg.max_conditions is not None:
        specs = specs[: analysis_cfg.max_conditions]
    return specs


_CACHE_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _to_cpu_snapshot(batch: dict[str, Any], image_dtype: torch.dtype) -> dict[str, Any]:
    """Deep-copy a policy input batch onto CPU.

    Heavy floating-point tensors (assumed to be images: ndim>=4 with a spatial
    dimension >= 64) are stored in ``image_dtype`` to save memory; everything
    else (ids, masks, state) stays at its native dtype.
    """
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            is_image = v.is_floating_point() and v.ndim >= 4 and max(v.shape[-3:]) >= 64
            target_dtype = image_dtype if is_image else v.dtype
            out[k] = v.detach().to(device="cpu", dtype=target_dtype).clone()
        else:
            out[k] = v
    return out


class TrajectoryRecorder:
    """Capture every ``predict_action_chunk`` call's (input batch, output chunk).

    The instance is reusable: enter the context manager around the section of
    code that runs the policy (e.g. a baseline rollout). After exit the
    captured snapshots live in ``self.snapshots`` as a list of
    ``(cpu_input_batch, baseline_action_chunk_cpu)`` pairs ready to feed back
    into ``policy.predict_action_chunk`` later.
    """

    def __init__(
        self,
        policy,
        *,
        max_snapshots: int,
        stride: int = 1,
        image_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.policy = policy
        self.max_snapshots = max_snapshots
        self.stride = max(1, int(stride))
        self.image_dtype = image_dtype
        self.snapshots: list[tuple[dict[str, Any], torch.Tensor]] = []
        self._call_idx = 0
        self._orig_predict = None

    def __enter__(self) -> "TrajectoryRecorder":
        if not hasattr(self.policy, "predict_action_chunk"):
            raise AttributeError(
                f"Policy {type(self.policy).__name__} has no predict_action_chunk; "
                "TrajectoryRecorder cannot wrap it."
            )
        self._orig_predict = self.policy.predict_action_chunk
        recorder = self

        @torch.no_grad()
        def wrapped(batch, **kwargs):
            chunk = recorder._orig_predict(batch, **kwargs)
            should_record = (
                recorder._call_idx % recorder.stride == 0
                and len(recorder.snapshots) < recorder.max_snapshots
            )
            if should_record:
                cpu_batch = _to_cpu_snapshot(batch, recorder.image_dtype)
                cpu_chunk = chunk.detach().to("cpu", dtype=torch.float32).clone()
                recorder.snapshots.append((cpu_batch, cpu_chunk))
            recorder._call_idx += 1
            return chunk

        self.policy.predict_action_chunk = wrapped
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._orig_predict is not None:
            self.policy.predict_action_chunk = self._orig_predict
        self._orig_predict = None
        return False  # propagate any exception

    def cumulative_bytes(self) -> int:
        total = 0
        for batch, chunk in self.snapshots:
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    total += v.element_size() * v.nelement()
            total += chunk.element_size() * chunk.nelement()
        return total


def _move_batch_to(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _run_mse_condition(
    policy,
    snapshots: list[tuple[dict[str, Any], torch.Tensor]],
    spec: AttentionKnockoutSpec,
    *,
    device: torch.device,
) -> dict:
    """Forward-only evaluation: compute action MSE vs baseline on cached snapshots."""

    condition_name = _condition_name(spec)
    logging.info(colored("MSE condition:", "magenta", attrs=["bold"]) + f" {condition_name}")
    _set_policy_knockout(policy, spec)
    policy.eval()

    per_step_acc: list[torch.Tensor] = []
    per_dim_acc: list[torch.Tensor] = []
    flat_per_sample: list[torch.Tensor] = []

    started = time.time()
    with torch.no_grad():
        for batch_cpu, baseline_chunk in snapshots:
            batch = _move_batch_to(batch_cpu, device)
            knockout_chunk = policy.predict_action_chunk(batch)
            knockout_chunk = knockout_chunk.detach().to("cpu", dtype=torch.float32)
            base_chunk_f = baseline_chunk.to(dtype=torch.float32)

            # Align temporal lengths if knockout chunk differs (shouldn't happen
            # for fixed-chunk-size policies, but defensive).
            min_t = min(knockout_chunk.shape[1], base_chunk_f.shape[1])
            min_d = min(knockout_chunk.shape[2], base_chunk_f.shape[2])
            ko = knockout_chunk[:, :min_t, :min_d]
            ba = base_chunk_f[:, :min_t, :min_d]
            sq_err = (ko - ba) ** 2  # (B, T, D)

            per_step_acc.append(sq_err.mean(dim=(0, -1)))  # (T,)
            per_dim_acc.append(sq_err.mean(dim=(0, 1)))  # (D,)
            flat_per_sample.append(sq_err.mean(dim=(1, 2)))  # (B,)

    per_step = torch.stack(per_step_acc).mean(dim=0)
    per_dim = torch.stack(per_dim_acc).mean(dim=0)
    flat = torch.cat(flat_per_sample)

    return {
        "condition": condition_name,
        "elapsed_s": time.time() - started,
        "knockout": spec.to_dict(),
        "mse_info": {
            "n_snapshots": len(snapshots),
            "n_samples": int(flat.numel()),
            "mse_mean": float(flat.mean().item()),
            "mse_p50": float(flat.median().item()),
            "mse_p95": float(np.percentile(flat.numpy(), 95)),
            "mse_max": float(flat.max().item()),
            "mse_per_step": per_step.tolist(),
            "mse_per_dim": per_dim.tolist(),
        },
    }


def _set_policy_knockout(policy, spec: AttentionKnockoutSpec | None) -> None:
    if hasattr(policy, "set_attention_knockout"):
        policy.set_attention_knockout(spec)
    else:
        raise TypeError(
            f"Policy type '{type(policy).__name__}' does not expose set_attention_knockout(). "
            "Map the Flow VLA analysis is currently implemented for pi0, pi05, smolvla, and groot policies."
        )


def _run_eval_condition(
    cfg: MapTheFlowPipelineConfig,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    spec: AttentionKnockoutSpec | None,
) -> dict:
    condition_name = _condition_name(spec)
    logging.info(colored("Running condition:", "cyan", attrs=["bold"]) + f" {condition_name}")

    if cfg.seed is not None:
        set_seed(cfg.seed)
    _set_policy_knockout(policy, spec)
    policy.eval()

    envs_for_condition = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    started = time.time()
    try:
        info = eval_policy_all(
            envs=envs_for_condition,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.analysis.max_episodes_rendered,
            videos_dir=Path(cfg.output_dir) / "videos" / condition_name,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
    finally:
        close_envs(envs_for_condition)

    return {
        "condition": condition_name,
        "elapsed_s": time.time() - started,
        "knockout": None if spec is None else spec.to_dict(),
        "info": info,
    }


def _summarize_delta(result: dict, baseline: dict | None) -> dict:
    summary: dict[str, Any] = {"condition": result["condition"]}
    info = result.get("info")
    if info is not None:
        overall = info["overall"]
        summary["pc_success"] = overall.get("pc_success")
        summary["avg_sum_reward"] = overall.get("avg_sum_reward")
        summary["n_episodes"] = overall.get("n_episodes")
        if baseline is not None and baseline.get("info") is not None:
            base_overall = baseline["info"]["overall"]
            if overall.get("pc_success") is not None and base_overall.get("pc_success") is not None:
                summary["delta_pc_success"] = overall["pc_success"] - base_overall["pc_success"]
            if overall.get("avg_sum_reward") is not None and base_overall.get("avg_sum_reward") is not None:
                summary["delta_avg_sum_reward"] = (
                    overall["avg_sum_reward"] - base_overall["avg_sum_reward"]
                )
    mse_info = result.get("mse_info")
    if mse_info is not None:
        summary["mse_mean"] = mse_info.get("mse_mean")
        summary["mse_p95"] = mse_info.get("mse_p95")
        summary["mse_n_samples"] = mse_info.get("n_samples")
    return summary


def _build_payload(
    cfg: MapTheFlowPipelineConfig,
    results: list[dict],
    baseline_result: dict | None,
    *,
    completed: bool,
) -> dict:
    return {
        "completed": completed,
        "config": asdict(cfg),
        "results": results,
        "summary": [
            _summarize_delta(result, baseline_result if result["condition"] != "baseline" else None)
            for result in results
        ],
    }


def _save_payload(output_dir: Path, payload: dict) -> Path:
    output_path = output_dir / "map_the_flow_info.json"
    tmp_path = output_dir / "map_the_flow_info.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp_path.replace(output_path)
    return output_path


@parser.wrap()
def map_the_flow_main(cfg: MapTheFlowPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    if cfg.policy is None:
        raise ValueError("Map the Flow analysis requires a policy. Pass --policy.path=...")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if cfg.seed is not None:
        set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
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
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    specs = _make_specs(cfg.analysis)
    conditions: list[AttentionKnockoutSpec | None] = []
    if cfg.analysis.include_baseline:
        conditions.append(None)
    conditions.extend(specs)

    metric = cfg.analysis.metric
    if metric in {"action_mse", "both"} and not cfg.analysis.include_baseline:
        raise ValueError(
            f"--analysis.metric={metric} requires --analysis.include_baseline=true so the "
            "baseline rollout can cache (observation, action) snapshots for MSE comparison."
        )
    logging.info(colored("Metric:", "yellow", attrs=["bold"]) + f" {metric}")

    image_dtype = _CACHE_DTYPE_MAP[cfg.analysis.mse_cache_dtype]

    results = []
    baseline_result = None
    mse_snapshots: list[tuple[dict[str, Any], torch.Tensor]] = []
    amp_context = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()

    with torch.no_grad(), amp_context:
        for spec in conditions:
            is_baseline = spec is None
            result: dict[str, Any] = {}

            if is_baseline:
                # Baseline always uses full rollout. If MSE is requested, capture snapshots
                # via TrajectoryRecorder so subsequent knockout conditions can replay them.
                if metric in {"action_mse", "both"}:
                    recorder = TrajectoryRecorder(
                        policy,
                        max_snapshots=cfg.analysis.mse_max_snapshots,
                        stride=cfg.analysis.mse_snapshot_stride,
                        image_dtype=image_dtype,
                    )
                    with recorder:
                        result = _run_eval_condition(
                            cfg,
                            policy=policy,
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            spec=None,
                        )
                    mse_snapshots = recorder.snapshots
                    mb = recorder.cumulative_bytes() / (1024 * 1024)
                    logging.info(
                        colored("Captured", "cyan", attrs=["bold"])
                        + f" {len(mse_snapshots)} snapshots for action-MSE ({mb:.0f} MB on CPU)"
                    )
                else:
                    result = _run_eval_condition(
                        cfg,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        spec=None,
                    )
                baseline_result = result
            else:
                # Knockout condition. Dispatch by metric mode.
                if metric == "pc_success":
                    result = _run_eval_condition(
                        cfg,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        spec=spec,
                    )
                elif metric == "action_mse":
                    if not mse_snapshots:
                        raise RuntimeError(
                            "No baseline snapshots captured; cannot compute action MSE. "
                            "Ensure --analysis.include_baseline=true."
                        )
                    result = _run_mse_condition(
                        policy, mse_snapshots, spec, device=device
                    )
                elif metric == "both":
                    if not mse_snapshots:
                        raise RuntimeError("No baseline snapshots captured for ``both`` metric.")
                    rollout_result = _run_eval_condition(
                        cfg,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        spec=spec,
                    )
                    # set_attention_knockout was already invoked by _run_eval_condition; re-set
                    # here in case it was reset by intervening calls.
                    mse_result = _run_mse_condition(
                        policy, mse_snapshots, spec, device=device
                    )
                    rollout_result["mse_info"] = mse_result["mse_info"]
                    rollout_result["mse_elapsed_s"] = mse_result["elapsed_s"]
                    result = rollout_result
                else:
                    raise ValueError(f"Unknown metric: {metric}")

            results.append(result)

            summary = _summarize_delta(result, baseline_result if not is_baseline else None)
            print(json.dumps(summary, indent=2, default=str))

            payload = _build_payload(cfg, results, baseline_result, completed=False)
            output_path = _save_payload(output_dir, payload)
            logging.info("Saved partial Map the Flow results to %s", output_path)

            # Reclaim memory between conditions (esp. important for MSE mode with many
            # snapshots queued on CPU).
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = _build_payload(cfg, results, baseline_result, completed=True)
    output_path = _save_payload(output_dir, payload)
    logging.info("Saved Map the Flow analysis to %s", output_path)


def main():
    init_logging()
    register_third_party_plugins()
    map_the_flow_main()


if __name__ == "__main__":
    main()
