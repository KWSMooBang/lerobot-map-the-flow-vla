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
from collections import defaultdict
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
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import ACTION
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

    # ── Comparison metric ────────────────────────────────────────────────
    # ``pc_success``     : original behavior — full env rollouts per condition,
    #                      binary task success aggregated to a percentage.
    # ``action_mse``     : baseline runs full rollout while caching every
    #                      (observation, baseline_action_chunk). Each knockout
    #                      condition then re-runs only forward inference on
    #                      those cached observations and reports the MSE
    #                      between its action chunk and the baseline chunk.
    #                      Open-loop / teacher-forced; ~10× faster than
    #                      ``pc_success`` and a much finer signal, but does
    #                      not capture closed-loop trajectory divergence.
    # ``trajectory_mse`` : both baseline and each knockout condition run
    #                      full *closed-loop* rollouts with identical seeds.
    #                      Trajectories are compared time-step by time-step
    #                      so compounding errors and execution divergence
    #                      become visible. Cost ≈ ``pc_success`` (1 full
    #                      rollout per condition) but the metric is
    #                      continuous and additionally reports
    #                      ``pc_success`` for free from the same rollout.
    # ``both``           : compute pc_success AND open-loop action MSE for
    #                      every knockout condition (full rollout + MSE
    #                      forward pass).
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

    # ── Trajectory-MSE settings ──────────────────────────────────────────
    # Whether to also collect the per-step observation.state trajectory and
    # report state-space MSE alongside the action MSE. Recommended for
    # robotics: end-effector / joint trajectories are more interpretable
    # than raw action delta.
    traj_include_state: bool = True
    # Cap on per-episode trajectory length stored in memory. Most LIBERO
    # episodes are <300 steps so this is rarely hit; lower it if you have
    # very long episodes and memory pressure.
    traj_max_steps: int | None = None
    # Emit a warning if baseline / knockout episode lengths differ by more
    # than this many steps for the same seed — large differences usually
    # mean the knockout caused early failure.
    traj_length_diff_warn: int = 50

    def __post_init__(self) -> None:
        self.mode = _clean_choice(self.mode, field_name="mode", allowed={"block", "keep_only"})
        self.metric = _clean_choice(
            self.metric,
            field_name="metric",
            allowed={"pc_success", "action_mse", "trajectory_mse", "both"},
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
    else (ids, masks, state) stays at its native dtype. Baseline rollouts call
    the policy under ``torch.inference_mode()``, so clones must be created with
    inference mode explicitly disabled; otherwise the replay pass may feed
    inference tensors into compiled CUDA graphs outside inference mode.
    """
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            is_image = v.is_floating_point() and v.ndim >= 4 and max(v.shape[-3:]) >= 64
            target_dtype = image_dtype if is_image else v.dtype
            with torch.inference_mode(False):
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
                with torch.inference_mode(False):
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
            with torch.inference_mode(False):
                moved = v.detach().to(device, non_blocking=True)
                if moved.is_inference():
                    moved = moved.clone()
                out[k] = moved
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
            # Match rollout-time inference semantics. This is also important for
            # torch.compile/Inductor CUDA graphs that may have been captured
            # during the baseline rollout under inference_mode.
            with torch.inference_mode():
                knockout_chunk = policy.predict_action_chunk(batch)
            with torch.inference_mode(False):
                knockout_chunk = knockout_chunk.detach().to("cpu", dtype=torch.float32).clone()
                base_chunk_f = baseline_chunk.detach().to("cpu", dtype=torch.float32)
                if base_chunk_f.is_inference():
                    base_chunk_f = base_chunk_f.clone()

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


# --------------------------------------------------------------------------- #
# Trajectory collection / comparison (closed-loop trajectory MSE)
# --------------------------------------------------------------------------- #

import math as _math  # local alias to avoid shadowing earlier in file


def _rollout_for_trajectory(
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seeds: list[int] | None = None,
    save_state: bool = True,
) -> dict:
    """Minimal closed-loop rollout that returns per-step action / state / reward / done.

    Mirrors ``lerobot.scripts.lerobot_eval.rollout`` but:

    * Skips ``return_observations=True``'s nested-observation stacking (which breaks
      on LIBERO because the raw observation contains nested dicts like
      ``observation.robot_state``).
    * Optionally saves a single ``observation.state`` tensor per step (the flat
      state produced *after* ``env_preprocessor`` runs), which is what
      ``_compare_trajectories`` needs for state-space MSE.

    Returns a dict with shape:

    * ``"action"``  : (B, T, action_dim)
    * ``"reward"``  : (B, T)
    * ``"success"`` : (B, T)
    * ``"done"``    : (B, T) — cumulative done flag
    * ``"state"``   : (B, T+1, state_dim)  (present iff ``save_state``)
    """
    import numpy as np

    if not isinstance(policy, torch.nn.Module):
        raise TypeError("Policy must be a PyTorch nn module.")

    policy.reset()
    observation, _info = env.reset(seed=seeds)
    check_env_attributes_and_types(env)

    all_actions: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []
    all_successes: list[torch.Tensor] = []
    all_dones: list[torch.Tensor] = []
    all_states: list[torch.Tensor] = []

    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]

    def _attach_task(obs_dict):
        try:
            obs_dict["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            try:
                obs_dict["task"] = list(env.call("task"))
            except (AttributeError, NotImplementedError):
                obs_dict["task"] = [""] * env.num_envs
        return obs_dict

    while not np.all(done) and step < max_steps:
        observation = preprocess_observation(observation)
        observation = _attach_task(observation)
        observation = env_preprocessor(observation)

        if save_state and "observation.state" in observation:
            obs_state = observation["observation.state"]
            if isinstance(obs_state, torch.Tensor):
                all_states.append(obs_state.detach().to("cpu", dtype=torch.float32).clone())

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)
        action_transition = env_postprocessor({ACTION: action})
        action = action_transition[ACTION]

        action_numpy = action.to("cpu").numpy()
        if action_numpy.ndim != 2:
            raise RuntimeError(f"Action must be (batch, action_dim); got shape {action_numpy.shape}")

        observation, reward, terminated, truncated, info = env.step(action_numpy)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "Upgrade gymnasium to >= 1.0."
                )
            successes = final_info["is_success"].tolist()
        elif "is_success" in info:
            is_success = info["is_success"]
            successes = (
                is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * env.num_envs
            )
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        step += 1

    # Capture the final state (one more entry than actions) so consumers can use it
    # for "final-state distance" metrics.
    if save_state:
        observation = preprocess_observation(observation)
        observation = _attach_task(observation)
        observation = env_preprocessor(observation)
        if "observation.state" in observation:
            obs_state = observation["observation.state"]
            if isinstance(obs_state, torch.Tensor):
                all_states.append(obs_state.detach().to("cpu", dtype=torch.float32).clone())

    ret: dict[str, torch.Tensor] = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if save_state and all_states:
        ret["state"] = torch.stack(all_states, dim=1)

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def _collect_trajectories(
    envs: dict,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    start_seed: int | None,
    include_state: bool,
    max_steps_cap: int | None = None,
) -> tuple[dict[tuple[str, int], list[dict]], dict]:
    """Run closed-loop rollouts and return per-episode action/state sequences.

    Iterates over every ``(task_group, task_id, env)`` triple in ``envs`` (same
    structure as ``eval_policy_all`` consumes), runs as many batched rollouts as
    needed to collect ``n_episodes`` per env, and returns:

    * ``per_task``: dict mapping ``(task_group, task_id) -> list[episode_dict]``
      where each episode_dict has ``actions``, ``length``, ``success``, ``seed``
      (and ``states`` when ``include_state=True``).
    * ``info``: a minimal info payload matching the shape produced by
      ``eval_policy_all`` so the rest of the pipeline (summary, payload) keeps
      working unchanged.
    """
    per_task: dict[tuple[str, int], list[dict]] = {}
    per_group_metrics: dict[str, dict[str, list]] = defaultdict(
        lambda: {"sum_rewards": [], "max_rewards": [], "successes": []}
    )
    overall_metrics = {"sum_rewards": [], "max_rewards": [], "successes": []}
    start_t = time.time()

    for task_group, group in envs.items():
        for task_id, env in group.items():
            num_envs = env.num_envs
            n_batches = _math.ceil(n_episodes / num_envs)
            episodes: list[dict] = []
            for batch_ix in range(n_batches):
                if start_seed is not None:
                    seeds = list(
                        range(
                            start_seed + batch_ix * num_envs,
                            start_seed + (batch_ix + 1) * num_envs,
                        )
                    )
                else:
                    seeds = None

                rollout_data = _rollout_for_trajectory(
                    env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seeds=seeds,
                    save_state=include_state,
                )

                actions = rollout_data["action"]  # (B, T, D)
                successes_tensor = rollout_data["success"]  # (B, T)
                rewards_tensor = rollout_data["reward"]
                done_tensor = rollout_data["done"]  # cumulative

                n_steps = done_tensor.shape[1]
                done_indices = torch.argmax(done_tensor.to(int), dim=1)
                # ``state`` key is present only when save_state=True AND env_preprocessor
                # produced an ``observation.state`` tensor.
                states_tensor = rollout_data.get("state") if include_state else None

                for env_ix in range(num_envs):
                    length = int(done_indices[env_ix].item()) + 1  # include done step
                    length = min(length, n_steps)
                    if max_steps_cap is not None:
                        length = min(length, max_steps_cap)
                    ep_actions = actions[env_ix, :length].detach().to("cpu", dtype=torch.float32).clone()
                    ep_success = bool(successes_tensor[env_ix, :length].any().item())
                    ep_reward_sum = float(rewards_tensor[env_ix, :length].sum().item())
                    ep_reward_max = float(rewards_tensor[env_ix, :length].max().item())
                    ep_seed = int(seeds[env_ix]) if seeds is not None else None
                    ep_data = {
                        "actions": ep_actions,
                        "length": length,
                        "success": ep_success,
                        "seed": ep_seed,
                        "reward_sum": ep_reward_sum,
                        "reward_max": ep_reward_max,
                    }
                    if states_tensor is not None:
                        # states have +1 step (final observation included).
                        ep_states = states_tensor[env_ix, : length + 1].detach().to(
                            "cpu", dtype=torch.float32
                        ).clone()
                        ep_data["states"] = ep_states
                    episodes.append(ep_data)
                    per_group_metrics[task_group]["sum_rewards"].append(ep_reward_sum)
                    per_group_metrics[task_group]["max_rewards"].append(ep_reward_max)
                    per_group_metrics[task_group]["successes"].append(ep_success)
                    overall_metrics["sum_rewards"].append(ep_reward_sum)
                    overall_metrics["max_rewards"].append(ep_reward_max)
                    overall_metrics["successes"].append(ep_success)

            episodes = episodes[:n_episodes]
            per_task[(task_group, task_id)] = episodes

    def _agg(xs):
        return float(np.mean(xs)) if xs else float("nan")

    info: dict[str, Any] = {
        "per_task": [
            {
                "task_group": tg,
                "task_id": tid,
                "metrics": {
                    "sum_rewards": [ep["reward_sum"] for ep in eps],
                    "max_rewards": [ep["reward_max"] for ep in eps],
                    "successes": [ep["success"] for ep in eps],
                    "video_paths": [],
                },
            }
            for (tg, tid), eps in per_task.items()
        ],
        "per_group": {
            tg: {
                "avg_sum_reward": _agg(m["sum_rewards"]),
                "avg_max_reward": _agg(m["max_rewards"]),
                "pc_success": (_agg(m["successes"]) * 100) if m["successes"] else float("nan"),
                "n_episodes": len(m["successes"]),
                "video_paths": [],
            }
            for tg, m in per_group_metrics.items()
        },
        "overall": {
            "avg_sum_reward": _agg(overall_metrics["sum_rewards"]),
            "avg_max_reward": _agg(overall_metrics["max_rewards"]),
            "pc_success": (_agg(overall_metrics["successes"]) * 100)
            if overall_metrics["successes"]
            else float("nan"),
            "n_episodes": len(overall_metrics["successes"]),
            "eval_s": time.time() - start_t,
            "eval_ep_s": (time.time() - start_t) / max(1, len(overall_metrics["successes"])),
            "video_paths": [],
        },
    }
    return per_task, info


def _compare_trajectories(
    baseline_per_task: dict[tuple[str, int], list[dict]],
    knockout_per_task: dict[tuple[str, int], list[dict]],
    *,
    include_state: bool,
    length_diff_warn: int,
) -> dict:
    """Time-aligned MSE between matching baseline and knockout episodes.

    Episodes are paired by ``(task_group, task_id, episode_index)``; the
    sequences are truncated to the minimum length of the pair (so the metric
    is well-defined even when knockout caused early termination). Pairs
    with mismatched seeds are skipped with a warning.
    """
    per_episode_action_mse: list[float] = []
    per_step_action_arrays: list[np.ndarray] = []
    length_diffs: list[int] = []
    baseline_lens: list[int] = []
    knockout_lens: list[int] = []
    baseline_successes: list[bool] = []
    knockout_successes: list[bool] = []
    first_div_steps: list[int] = []

    per_episode_state_mse: list[float] = []
    per_step_state_arrays: list[np.ndarray] = []
    final_state_mse: list[float] = []

    DIVERGENCE_THRESHOLD = 1e-3  # per-step MSE above which we consider trajectories diverged

    for key, baseline_eps in baseline_per_task.items():
        ko_eps = knockout_per_task.get(key, [])
        n_pairs = min(len(baseline_eps), len(ko_eps))
        for i in range(n_pairs):
            b_ep = baseline_eps[i]
            k_ep = ko_eps[i]
            if (
                b_ep.get("seed") is not None
                and k_ep.get("seed") is not None
                and b_ep["seed"] != k_ep["seed"]
            ):
                logging.warning(
                    "Trajectory pair seed mismatch in task %s episode %d (baseline=%s, knockout=%s); "
                    "comparison may be unreliable.",
                    key,
                    i,
                    b_ep["seed"],
                    k_ep["seed"],
                )
            b_actions = b_ep["actions"]
            k_actions = k_ep["actions"]
            min_t = min(b_actions.shape[0], k_actions.shape[0])
            if min_t == 0:
                continue
            diff = b_actions[:min_t] - k_actions[:min_t]
            per_step = (diff.to(torch.float32) ** 2).mean(dim=-1).cpu().numpy()  # (min_t,)
            per_episode_action_mse.append(float(per_step.mean()))
            per_step_action_arrays.append(per_step)

            diverged = np.where(per_step > DIVERGENCE_THRESHOLD)[0]
            first_div_steps.append(int(diverged[0]) if diverged.size > 0 else int(per_step.size))

            baseline_lens.append(b_ep["length"])
            knockout_lens.append(k_ep["length"])
            length_diffs.append(b_ep["length"] - k_ep["length"])
            baseline_successes.append(b_ep["success"])
            knockout_successes.append(k_ep["success"])

            if include_state and "states" in b_ep and "states" in k_ep:
                b_states = b_ep["states"]
                k_states = k_ep["states"]
                min_ts = min(b_states.shape[0], k_states.shape[0])
                if min_ts == 0:
                    continue
                diff_s = b_states[:min_ts] - k_states[:min_ts]
                per_step_s = (diff_s.to(torch.float32) ** 2).mean(dim=-1).cpu().numpy()
                per_episode_state_mse.append(float(per_step_s.mean()))
                per_step_state_arrays.append(per_step_s)
                final_state_mse.append(float(per_step_s[-1]))

    if length_diffs:
        big_diffs = sum(1 for d in length_diffs if abs(d) > length_diff_warn)
        if big_diffs:
            logging.warning(
                "trajectory_mse: %d/%d episodes had |baseline - knockout| length > %d "
                "(knockout terminated early or ran long).",
                big_diffs,
                len(length_diffs),
                length_diff_warn,
            )

    def _pad_and_mean(arrays: list[np.ndarray]) -> list[float]:
        if not arrays:
            return []
        max_len = max(len(a) for a in arrays)
        padded = np.full((len(arrays), max_len), np.nan, dtype=np.float64)
        for i, a in enumerate(arrays):
            padded[i, : len(a)] = a
        with np.errstate(all="ignore"):
            mean = np.nanmean(padded, axis=0)
        return [float(v) for v in mean.tolist()]

    info: dict[str, Any] = {
        "n_episodes": len(per_episode_action_mse),
        "action_mse_mean": float(np.mean(per_episode_action_mse)) if per_episode_action_mse else 0.0,
        "action_mse_p50": float(np.median(per_episode_action_mse))
        if per_episode_action_mse
        else 0.0,
        "action_mse_p95": float(np.percentile(per_episode_action_mse, 95))
        if per_episode_action_mse
        else 0.0,
        "action_mse_max": float(np.max(per_episode_action_mse)) if per_episode_action_mse else 0.0,
        "action_mse_per_episode": per_episode_action_mse,
        "action_mse_per_step": _pad_and_mean(per_step_action_arrays),
        "first_divergence_step": first_div_steps,
        "episode_lengths_baseline": baseline_lens,
        "episode_lengths_knockout": knockout_lens,
        "length_diff_mean": float(np.mean(length_diffs)) if length_diffs else 0.0,
        "length_diff_max": int(np.max(np.abs(length_diffs))) if length_diffs else 0,
        "success_baseline": baseline_successes,
        "success_knockout": knockout_successes,
        "pc_success_baseline": float(np.mean(baseline_successes) * 100) if baseline_successes else 0.0,
        "pc_success_knockout": float(np.mean(knockout_successes) * 100)
        if knockout_successes
        else 0.0,
    }
    if include_state and per_episode_state_mse:
        info.update(
            {
                "state_mse_mean": float(np.mean(per_episode_state_mse)),
                "state_mse_p50": float(np.median(per_episode_state_mse)),
                "state_mse_p95": float(np.percentile(per_episode_state_mse, 95)),
                "state_mse_per_episode": per_episode_state_mse,
                "state_mse_per_step": _pad_and_mean(per_step_state_arrays),
                "final_state_mse_mean": float(np.mean(final_state_mse)),
                "final_state_mse_p95": float(np.percentile(final_state_mse, 95))
                if final_state_mse
                else 0.0,
            }
        )
    return info


def _run_trajectory_mse_condition(
    cfg: MapTheFlowPipelineConfig,
    *,
    policy,
    envs,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    spec: AttentionKnockoutSpec | None,
    baseline_per_task: dict[tuple[str, int], list[dict]] | None,
) -> tuple[dict, dict[tuple[str, int], list[dict]] | None]:
    """Closed-loop rollout under (optional) knockout, plus comparison vs baseline.

    Returns ``(result_dict, collected_trajectories)``. For the baseline call
    (``spec is None``), the trajectories must be returned and stashed by the
    caller so subsequent knockout runs can compare against them.
    """
    condition_name = _condition_name(spec)
    is_baseline = spec is None
    logging.info(
        colored("Trajectory condition:", "cyan", attrs=["bold"]) + f" {condition_name}"
    )

    if cfg.seed is not None:
        set_seed(cfg.seed)
    _set_policy_knockout(policy, spec)
    policy.eval()

    started = time.time()
    collected, info = _collect_trajectories(
        envs,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=cfg.eval.n_episodes,
        start_seed=cfg.seed,
        include_state=cfg.analysis.traj_include_state,
        max_steps_cap=cfg.analysis.traj_max_steps,
    )

    result: dict[str, Any] = {
        "condition": condition_name,
        "elapsed_s": time.time() - started,
        "knockout": None if is_baseline else spec.to_dict(),
        "info": info,
    }
    if not is_baseline:
        if baseline_per_task is None:
            raise RuntimeError(
                "_run_trajectory_mse_condition: baseline trajectories missing; ensure baseline ran first."
            )
        result["traj_mse_info"] = _compare_trajectories(
            baseline_per_task,
            collected,
            include_state=cfg.analysis.traj_include_state,
            length_diff_warn=cfg.analysis.traj_length_diff_warn,
        )
    return result, collected


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
    traj_info = result.get("traj_mse_info")
    if traj_info is not None:
        summary["traj_action_mse_mean"] = traj_info.get("action_mse_mean")
        summary["traj_action_mse_p95"] = traj_info.get("action_mse_p95")
        summary["traj_length_diff_mean"] = traj_info.get("length_diff_mean")
        summary["traj_first_divergence_step_mean"] = (
            float(np.mean(traj_info.get("first_divergence_step", []))) if traj_info.get("first_divergence_step") else None
        )
        if "state_mse_mean" in traj_info:
            summary["traj_state_mse_mean"] = traj_info["state_mse_mean"]
            summary["traj_final_state_mse_mean"] = traj_info.get("final_state_mse_mean")
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
    if metric in {"action_mse", "both", "trajectory_mse"} and not cfg.analysis.include_baseline:
        raise ValueError(
            f"--analysis.metric={metric} requires --analysis.include_baseline=true so the "
            "baseline run can be compared against."
        )
    logging.info(colored("Metric:", "yellow", attrs=["bold"]) + f" {metric}")

    image_dtype = _CACHE_DTYPE_MAP[cfg.analysis.mse_cache_dtype]

    results = []
    baseline_result = None
    mse_snapshots: list[tuple[dict[str, Any], torch.Tensor]] = []
    baseline_trajectories: dict[tuple[str, int], list[dict]] | None = None
    # trajectory_mse needs the same envs across baseline + all knockouts (so seeds line up
    # and we avoid the per-condition env teardown cost). All other modes keep the existing
    # per-condition env lifecycle that _run_eval_condition manages internally.
    shared_envs = None
    if metric == "trajectory_mse":
        shared_envs = make_env(
            cfg.env,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.eval.use_async_envs,
            trust_remote_code=cfg.trust_remote_code,
        )

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
                elif metric == "trajectory_mse":
                    result, baseline_trajectories = _run_trajectory_mse_condition(
                        cfg,
                        policy=policy,
                        envs=shared_envs,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        spec=None,
                        baseline_per_task=None,
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
                elif metric == "trajectory_mse":
                    if baseline_trajectories is None:
                        raise RuntimeError(
                            "No baseline trajectories captured; cannot compute trajectory MSE."
                        )
                    result, _ = _run_trajectory_mse_condition(
                        cfg,
                        policy=policy,
                        envs=shared_envs,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        spec=spec,
                        baseline_per_task=baseline_trajectories,
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

    if shared_envs is not None:
        close_envs(shared_envs)


def main():
    init_logging()
    register_third_party_plugins()
    map_the_flow_main()


if __name__ == "__main__":
    main()
