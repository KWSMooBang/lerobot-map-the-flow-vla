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
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Literal

import torch
from termcolor import colored

from lerobot import envs, policies  # noqa: F401
from lerobot.analysis.map_the_flow import AttentionKnockoutSpec, parse_layer_ranges, parse_route_rules
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


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
    mode: Literal["block", "keep_only"] = "block"
    include_baseline: bool = True
    max_conditions: int | None = None
    max_episodes_rendered: int = 0


@dataclass
class MapTheFlowPipelineConfig:
    env: EnvConfig
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


def _set_policy_knockout(policy, spec: AttentionKnockoutSpec | None) -> None:
    if hasattr(policy, "set_attention_knockout"):
        policy.set_attention_knockout(spec)
    else:
        raise TypeError(
            f"Policy type '{type(policy).__name__}' does not expose set_attention_knockout(). "
            "Map the Flow VLA analysis is currently implemented for pi0 and pi05 policies."
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
    overall = result["info"]["overall"]
    summary = {
        "condition": result["condition"],
        "pc_success": overall.get("pc_success"),
        "avg_sum_reward": overall.get("avg_sum_reward"),
        "n_episodes": overall.get("n_episodes"),
    }
    if baseline is not None:
        base_overall = baseline["info"]["overall"]
        summary["delta_pc_success"] = overall.get("pc_success") - base_overall.get("pc_success")
        summary["delta_avg_sum_reward"] = overall.get("avg_sum_reward") - base_overall.get(
            "avg_sum_reward"
        )
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

    results = []
    baseline_result = None
    amp_context = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    with torch.no_grad(), amp_context:
        for spec in conditions:
            result = _run_eval_condition(
                cfg,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                spec=spec,
            )
            if spec is None:
                baseline_result = result
            results.append(result)

            summary = _summarize_delta(result, baseline_result if spec is not None else None)
            print(json.dumps(summary, indent=2))

            payload = _build_payload(cfg, results, baseline_result, completed=False)
            output_path = _save_payload(output_dir, payload)
            logging.info("Saved partial Map the Flow results to %s", output_path)

    payload = _build_payload(cfg, results, baseline_result, completed=True)
    output_path = _save_payload(output_dir, payload)
    logging.info("Saved Map the Flow analysis to %s", output_path)


def main():
    init_logging()
    register_third_party_plugins()
    map_the_flow_main()


if __name__ == "__main__":
    main()
