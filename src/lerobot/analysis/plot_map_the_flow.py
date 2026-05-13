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
"""Generate publication-style figures from Map The Flow VLA runs.

The input can be one or more ``map_the_flow_info.json`` files, directories that
contain those files, or shell-style globs. When several LIBERO tasks or policies
are provided, the default behavior is to facet by ``policy x task`` instead of
incorrectly averaging unlike runs.

Examples:

```
python -m lerobot.analysis.plot_map_the_flow \
    --inputs outputs/pi0 \
    --output figures/pi0_figure3.pdf
```

```
python -m lerobot.analysis.plot_map_the_flow \
    --inputs outputs/pi05/*/map_the_flow_info.json \
    --output figures/pi05_figure3.pdf \
    --change relative \
    --suptitle "pi0.5 Map-The-Flow on LIBERO"
```
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "lerobot_matplotlib_cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


CONDITION_RE = re.compile(
    r"^(?P<route>.+?)_center_L(?P<center>\d+)_window_L(?P<start>\d+)-(?P<end>\d+)$"
)

METRIC_LABELS = {
    "pc_success": "success",
    "avg_sum_reward": "average reward",
    "avg_max_reward": "max reward",
}

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Nimbus Roman No9 L"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8.0,
    "legend.frameon": False,
    "lines.linewidth": 1.65,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.035,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

ROUTE_PALETTE = [
    "#3b6fb5",
    "#d96c3a",
    "#4f9b4f",
    "#b1495b",
    "#7a5fab",
    "#a8783b",
    "#3aa6a0",
    "#6b6e74",
    "#d55e00",
    "#0072b2",
]

POLICY_ORDER = {"pi0": 0, "pi05": 1}
TASK_ORDER = {
    "libero_spatial": 0,
    "libero_object": 1,
    "libero_goal": 2,
    "libero_10": 3,
    "libero_90": 4,
}

DROP_CMAP = LinearSegmentedColormap.from_list(
    "map_the_flow_drop",
    [
        (0.00, "#8b1e2d"),
        (0.26, "#d35f5f"),
        (0.50, "#f7f7f7"),
        (0.74, "#7da9d6"),
        (1.00, "#2f5f9f"),
    ],
)


@dataclass(frozen=True)
class ParsedCondition:
    route: str | None
    center: int | None
    start: int | None
    end: int | None
    is_baseline: bool = False


@dataclass(frozen=True)
class RunMeta:
    path: Path
    policy_id: str
    policy_label: str
    task_id: str
    task_label: str
    seed: int | None
    run_label: str


@dataclass
class GroupSeries:
    key: tuple[str, ...]
    label: str
    payloads: list[dict]
    metas: list[RunMeta]
    baselines: list[float]
    n_episodes: list[int]
    deltas: dict[str, dict[int, list[float]]]
    raw_values: dict[str, dict[int, list[float]]]
    route_order: list[str]


def _parse_condition(name: str) -> ParsedCondition | None:
    if name == "baseline":
        return ParsedCondition(None, None, None, None, is_baseline=True)

    match = CONDITION_RE.match(name)
    if match is None:
        return None

    route = match.group("route").replace("_bidir_", "<->").replace("_to_", "->")
    return ParsedCondition(
        route=route,
        center=int(match.group("center")),
        start=int(match.group("start")),
        end=int(match.group("end")),
        is_baseline=False,
    )


def _route_label(route: str) -> str:
    source, sep, target = route.partition("<->")
    if sep:
        return f"{_token_label(source)} ↮ {_token_label(target)}"
    source, sep, target = route.partition("->")
    if sep:
        return f"{_token_label(source)} ↛ {_token_label(target)}"
    return _token_label(route)


def _token_label(token: str) -> str:
    aliases = {
        "view0": "View 0",
        "view1": "View 1",
        "vision": "Vision",
        "language": "Language",
        "instruction": "Instruction",
        "state": "State",
        "state_text": "State text",
        "action": "Action",
        "scaffold": "Scaffold",
    }
    return aliases.get(token, token.replace("_", " ").title())


def _task_label(task: str) -> str:
    if task == "overall":
        return "Overall"
    if task.startswith("libero_"):
        task = task[len("libero_") :]
    return "LIBERO " + task.replace("_", " ").title()


def _series_sort_key(series: GroupSeries) -> tuple[int, int, str]:
    meta = series.metas[0]
    return (
        POLICY_ORDER.get(meta.policy_id, 99),
        TASK_ORDER.get(meta.task_id, 99),
        series.label,
    )


def _figure_panel_label(series: GroupSeries, all_series: list[GroupSeries]) -> str:
    policies = {s.metas[0].policy_id for s in all_series}
    tasks = {s.metas[0].task_id for s in all_series}
    meta = series.metas[0]
    if len(policies) == 1 and len(tasks) > 1:
        return meta.task_label
    if len(tasks) == 1 and len(policies) > 1:
        return meta.policy_label
    return series.label


def _policy_id_from_payload(payload: dict, path: Path) -> str:
    policy = payload.get("config", {}).get("policy", {})
    haystack = " ".join(
        str(x)
        for x in (
            path.as_posix(),
            policy.get("pretrained_path"),
            policy.get("repo_id"),
            policy.get("type"),
        )
        if x
    ).lower()
    if "pi05" in haystack or "pi0.5" in haystack:
        return "pi05"
    if "pi0" in haystack:
        return "pi0"
    return "policy"


def _policy_label(policy_id: str) -> str:
    labels = {
        "pi0": "$\\pi_0$",
        "pi05": "$\\pi_{0.5}$",
    }
    return labels.get(policy_id, policy_id)


def _run_meta(path: Path, payload: dict) -> RunMeta:
    cfg = payload.get("config", {})
    env_cfg = cfg.get("env", {})
    policy_id = _policy_id_from_payload(payload, path)
    task_id = str(env_cfg.get("task") or "overall")
    seed = cfg.get("seed")
    run_label = f"{_policy_label(policy_id)} / {_task_label(task_id)}"
    return RunMeta(
        path=path,
        policy_id=policy_id,
        policy_label=_policy_label(policy_id),
        task_id=task_id,
        task_label=_task_label(task_id),
        seed=seed,
        run_label=run_label,
    )


def _has_glob_pattern(value: str) -> bool:
    return any(ch in value for ch in "*?[]")


def _resolve_inputs(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        raw = str(input_path)
        candidates = [Path(p) for p in glob.glob(raw)] if _has_glob_pattern(raw) else [input_path]
        for candidate in candidates:
            if candidate.is_dir():
                files.extend(candidate.rglob("map_the_flow_info.json"))
            elif candidate.is_file():
                files.append(candidate)
    unique = sorted({p.resolve() for p in files})
    if not unique:
        raise FileNotFoundError("No map_the_flow_info.json files were found from --inputs.")
    return unique


def _load_payloads(paths: list[Path]) -> list[tuple[Path, dict]]:
    payloads = []
    for path in paths:
        with open(path) as f:
            payloads.append((path, json.load(f)))
    return payloads


def _route_order(payloads: list[dict], deltas: dict[str, dict[int, list[float]]]) -> list[str]:
    routes: list[str] = []
    for payload in payloads:
        for route in payload.get("config", {}).get("analysis", {}).get("routes", []):
            for expanded in _expand_route(route):
                if expanded in deltas and expanded not in routes:
                    routes.append(expanded)
    for route in sorted(deltas):
        if route not in routes:
            routes.append(route)
    return routes


def _expand_route(route: str) -> list[str]:
    route = route.strip()
    if "<->" not in route:
        return [route]
    source, target = route.split("<->", 1)
    return [f"{source}->{target}", f"{target}->{source}", route]


def _metric_from_info(info: dict, group: str, metric: str) -> tuple[float | None, int | None]:
    if group == "overall":
        metrics = info.get("overall", {})
    else:
        metrics = info.get("per_group", {}).get(group, {})
    value = metrics.get(metric)
    n_episodes = metrics.get("n_episodes")
    return value, n_episodes


def _compute_change(value: float, baseline: float, change: Literal["absolute", "relative"]) -> float:
    if change == "absolute":
        return value - baseline
    if baseline == 0:
        return math.nan
    return 100.0 * (value - baseline) / abs(baseline)


def _group_key(meta: RunMeta, group_by: str) -> tuple[str, ...]:
    if group_by == "policy_task":
        return (meta.policy_id, meta.task_id)
    if group_by == "policy":
        return (meta.policy_id,)
    if group_by == "task":
        return (meta.task_id,)
    if group_by == "run":
        return (str(meta.path),)
    if group_by == "all":
        return ("all",)
    raise ValueError(f"Unsupported group_by value: {group_by}")


def _group_label(metas: list[RunMeta], group_by: str) -> str:
    first = metas[0]
    if group_by == "policy_task":
        return f"{first.policy_label} / {first.task_label}"
    if group_by == "policy":
        return first.policy_label
    if group_by == "task":
        return first.task_label
    if group_by == "all":
        return "All runs"
    return first.run_label


def _choose_group_by(metas: list[RunMeta], requested: str) -> str:
    if requested != "auto":
        return requested
    policy_tasks = {(m.policy_id, m.task_id) for m in metas}
    if len(policy_tasks) > 1:
        return "policy_task"
    return "all"


def _build_group_series(
    entries: list[tuple[Path, dict]],
    *,
    metric: str,
    group: str,
    change: Literal["absolute", "relative"],
    group_by: str,
) -> list[GroupSeries]:
    metas = [_run_meta(path, payload) for path, payload in entries]
    group_by = _choose_group_by(metas, group_by)

    grouped: dict[tuple[str, ...], list[tuple[Path, dict, RunMeta]]] = defaultdict(list)
    for path, payload in entries:
        meta = _run_meta(path, payload)
        grouped[_group_key(meta, group_by)].append((path, payload, meta))

    series_list: list[GroupSeries] = []
    for key in sorted(grouped):
        items = grouped[key]
        payloads = [payload for _, payload, _ in items]
        item_metas = [meta for _, _, meta in items]
        baselines: list[float] = []
        n_episodes: list[int] = []
        deltas: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        raw_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

        for _, payload, _ in items:
            baseline: float | None = None
            baseline_n: int | None = None
            for result in payload.get("results", []):
                parsed = _parse_condition(result.get("condition", ""))
                if parsed is None or not parsed.is_baseline:
                    continue
                baseline, baseline_n = _metric_from_info(result.get("info", {}), group, metric)
                break

            if baseline is None:
                continue
            baselines.append(float(baseline))
            if baseline_n is not None:
                n_episodes.append(int(baseline_n))

            for result in payload.get("results", []):
                parsed = _parse_condition(result.get("condition", ""))
                if parsed is None or parsed.is_baseline or parsed.route is None or parsed.center is None:
                    continue
                value, _ = _metric_from_info(result.get("info", {}), group, metric)
                if value is None:
                    continue
                raw_values[parsed.route][parsed.center].append(float(value))
                deltas[parsed.route][parsed.center].append(
                    _compute_change(float(value), float(baseline), change)
                )

        if not deltas:
            continue
        series_list.append(
            GroupSeries(
                key=key,
                label=_group_label(item_metas, group_by),
                payloads=payloads,
                metas=item_metas,
                baselines=baselines,
                n_episodes=n_episodes,
                deltas={route: dict(centers) for route, centers in deltas.items()},
                raw_values={route: dict(centers) for route, centers in raw_values.items()},
                route_order=_route_order(payloads, deltas),
            )
        )

    if not series_list:
        raise RuntimeError(
            f"No plottable records found for metric='{metric}', group='{group}'. "
            "Check that the inputs contain baseline and route conditions."
        )
    return sorted(series_list, key=_series_sort_key)


def _panel_grid(
    n_panels: int,
    requested_cols: int | None = None,
    *,
    default_max_cols: int = 4,
) -> tuple[int, int]:
    if requested_cols is not None:
        n_cols = max(1, min(requested_cols, n_panels))
    else:
        n_cols = min(default_max_cols, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    return n_rows, n_cols


def _change_label(metric: str, change: str) -> str:
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))
    if change == "relative":
        return f"Relative {metric_label} change (%)"
    if metric == "pc_success":
        return "Success change (percentage points)"
    return f"{metric_label.title()} change"


def _baseline_text(series: GroupSeries) -> str:
    baseline = float(np.mean(series.baselines)) if series.baselines else math.nan
    n_text = ""
    if series.n_episodes:
        n_text = f", n={int(np.mean(series.n_episodes))}"
    if len(series.baselines) > 1:
        return f"base {baseline:.1f} +/- {float(np.std(series.baselines)):.1f}{n_text}"
    return f"base {baseline:.1f}{n_text}"


def _matrix_for_series(series: GroupSeries, max_layer: int) -> tuple[list[str], np.ndarray]:
    routes = [route for route in series.route_order if route in series.deltas]
    matrix = np.full((len(routes), max_layer), np.nan, dtype=float)
    for row_idx, route in enumerate(routes):
        for center, values in series.deltas[route].items():
            if 1 <= center <= max_layer and values:
                matrix[row_idx, center - 1] = float(np.nanmean(values))
    return routes, matrix


def _finite_values(series_list: list[GroupSeries], max_layer: int) -> np.ndarray:
    values: list[float] = []
    for series in series_list:
        _, matrix = _matrix_for_series(series, max_layer)
        values.extend(matrix[np.isfinite(matrix)].tolist())
    return np.asarray(values, dtype=float)


def _draw_phase_guides(ax, max_layer: int, *, y_top: float) -> None:
    for boundary in (6.5, 12.5):
        if boundary <= max_layer + 0.5:
            ax.axvline(boundary - 1, color="#2f2f2f", linewidth=0.5, alpha=0.45)
    labels = [("early", 3), ("middle", 9), ("late", 15.5)]
    for label, center in labels:
        if center <= max_layer:
            ax.text(
                center - 1,
                y_top,
                label,
                ha="center",
                va="bottom",
                fontsize=6.6,
                color="#5a5a5a",
                fontstyle="italic",
                clip_on=False,
            )


def plot_figure3(
    series_list: list[GroupSeries],
    output: Path,
    *,
    metric: str,
    change: Literal["absolute", "relative"],
    max_layer: int,
    n_cols: int | None,
    show_std: bool,
    suptitle: str | None,
) -> None:
    """Line-panel layout matching Map-The-Flow Figure 3.

    Each panel is one task/policy group; each colored curve is one blocked route.
    This is the main paper-style visualization for attention knockout sweeps.
    """

    plt.rcParams.update(PAPER_STYLE)

    values = _finite_values(series_list, max_layer)
    if values.size == 0:
        raise RuntimeError("No finite values to plot.")

    y_span = float(np.nanmax(values) - np.nanmin(values))
    pad = max(1.5, 0.08 * (y_span + 1e-6))
    y_min = min(-1.0, float(np.nanmin(values)) - pad)
    y_max = max(1.0, float(np.nanmax(values)) + pad)
    if y_max > 0:
        y_max += 0.05 * (y_max - y_min)

    n_rows, n_cols_final = _panel_grid(len(series_list), n_cols, default_max_cols=5)
    fig_w = 1.85 * n_cols_final + 0.55
    fig_h = 2.05 * n_rows + 0.7
    fig, axes = plt.subplots(
        n_rows,
        n_cols_final,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    all_routes: list[str] = []
    for series in series_list:
        for route in series.route_order:
            if route in series.deltas and route not in all_routes:
                all_routes.append(route)
    route_color = {route: ROUTE_PALETTE[i % len(ROUTE_PALETTE)] for i, route in enumerate(all_routes)}

    for idx, series in enumerate(series_list):
        ax = axes[idx // n_cols_final][idx % n_cols_final]
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(1, max_layer)
        ax.grid(True, color="#e5e5e5", linewidth=0.45, alpha=0.85)
        ax.axhline(0, color="#4a4a4a", linewidth=0.55, alpha=0.85)
        ax.tick_params(axis="both", labelsize=6.8, pad=1.2, labelbottom=True)

        for route in series.route_order:
            centers_map = series.deltas.get(route)
            if not centers_map:
                continue
            centers = sorted(c for c in centers_map if 1 <= c <= max_layer)
            if not centers:
                continue
            means = np.asarray([np.nanmean(centers_map[c]) for c in centers], dtype=float)
            stds = np.asarray([np.nanstd(centers_map[c]) for c in centers], dtype=float)
            color = route_color[route]
            ax.plot(centers, means, color=color, label=_route_label(route), zorder=3)
            if show_std and any(len(centers_map[c]) > 1 for c in centers):
                ax.fill_between(
                    centers,
                    means - stds,
                    means + stds,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )

        ax.set_xticks([tick for tick in (5, 10, 15, 18) if tick <= max_layer])
        ax.set_xlabel("Layer", labelpad=1.0, fontsize=7.2)
        if idx % n_cols_final == 0:
            ylabel = "% Change in Success" if metric == "pc_success" else _change_label(metric, change)
            if change == "absolute" and metric == "pc_success":
                ylabel = "Change in Success (pp)"
            ax.set_ylabel(ylabel, labelpad=2.0, fontsize=7.2)

        baseline = float(np.mean(series.baselines)) if series.baselines else math.nan
        ax.text(
            0.98,
            0.96,
            f"base {baseline:.1f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=6.6,
            color="#555555",
        )
        panel_label = _figure_panel_label(series, series_list)
        ax.set_title(f"({chr(97 + idx)}) {panel_label}", y=-0.48, pad=0, fontsize=8.8)

    for idx in range(len(series_list), n_rows * n_cols_final):
        axes[idx // n_cols_final][idx % n_cols_final].axis("off")

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l, strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=min(5, len(labels)),
            handlelength=1.9,
            columnspacing=1.25,
            handletextpad=0.45,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=10.8, y=1.06)

    top = 0.78 if handles else 0.9
    bottom = 0.26 if n_rows == 1 else 0.15
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=top,
        bottom=bottom,
        wspace=0.12,
        hspace=0.95,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved Figure-3-style plot to {output}")


def plot_heatmap(
    series_list: list[GroupSeries],
    output: Path,
    *,
    metric: str,
    change: Literal["absolute", "relative"],
    max_layer: int,
    n_cols: int | None,
    annotate: bool,
    mark_threshold: float | None,
    show_phases: bool,
    suptitle: str | None,
) -> None:
    plt.rcParams.update(PAPER_STYLE)

    values = _finite_values(series_list, max_layer)
    if values.size == 0:
        raise RuntimeError("No finite values to plot.")
    vmax = max(1.0, float(np.nanmax(np.abs(values))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = DROP_CMAP.copy()
    cmap.set_bad("#eeeeee")

    n_rows, n_cols_final = _panel_grid(len(series_list), n_cols)
    fig_w = 3.15 * n_cols_final
    fig_h = 1.55 * n_rows + 0.55
    fig, axes = plt.subplots(n_rows, n_cols_final, figsize=(fig_w, fig_h), squeeze=False)

    image = None
    for idx, series in enumerate(series_list):
        ax = axes[idx // n_cols_final][idx % n_cols_final]
        routes, matrix = _matrix_for_series(series, max_layer)
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

        ax.set_title(f"({chr(97 + idx)}) {series.label}\n{_baseline_text(series)}", pad=8)
        ax.set_yticks(np.arange(len(routes)))
        ax.set_yticklabels([_route_label(route) for route in routes])
        ax.set_xticks(np.arange(max_layer))
        ax.set_xticklabels([str(i) if i == 1 or i % 2 == 0 else "" for i in range(1, max_layer + 1)])
        ax.set_xlim(-0.5, max_layer - 0.5)
        ax.set_ylim(len(routes) - 0.5, -0.5)

        ax.set_xticks(np.arange(-0.5, max_layer, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(routes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="y", length=0)

        if show_phases:
            _draw_phase_guides(ax, max_layer, y_top=-0.88)

        if annotate:
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    value = matrix[row_idx, col_idx]
                    if not np.isfinite(value):
                        continue
                    text = f"{value:.0f}" if metric == "pc_success" and change == "absolute" else f"{value:.1f}"
                    color = "white" if abs(value) > 0.58 * vmax else "#242424"
                    ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=6.6, color=color)

        if mark_threshold is not None:
            rows, cols = np.where(np.isfinite(matrix) & (matrix <= mark_threshold))
            if len(rows) > 0:
                ax.scatter(cols, rows, marker="o", s=8, color="#111111", linewidths=0, alpha=0.82)

        if idx // n_cols_final == n_rows - 1:
            ax.set_xlabel("Layer center")

    for idx in range(len(series_list), n_rows * n_cols_final):
        axes[idx // n_cols_final][idx % n_cols_final].axis("off")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.012)
        cbar.set_label(_change_label(metric, change) + "\n(knockout - baseline)")
        cbar.ax.tick_params(labelsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11.0, y=1.02)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved heatmap figure to {output}")


def plot_lines(
    series_list: list[GroupSeries],
    output: Path,
    *,
    metric: str,
    change: Literal["absolute", "relative"],
    max_layer: int,
    n_cols: int | None,
    show_std: bool,
    show_phases: bool,
    suptitle: str | None,
) -> None:
    plt.rcParams.update(PAPER_STYLE)

    values = _finite_values(series_list, max_layer)
    if values.size == 0:
        raise RuntimeError("No finite values to plot.")
    pad = max(1.0, 0.08 * (float(np.nanmax(values)) - float(np.nanmin(values)) + 1e-6))
    y_min = min(-1.0, float(np.nanmin(values)) - pad)
    y_max = max(1.0, float(np.nanmax(values)) + pad)

    n_rows, n_cols_final = _panel_grid(len(series_list), n_cols)
    fig_w = 3.2 * n_cols_final
    fig_h = 2.15 * n_rows + 0.55
    fig, axes = plt.subplots(
        n_rows,
        n_cols_final,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    all_routes = []
    for series in series_list:
        for route in series.route_order:
            if route in series.deltas and route not in all_routes:
                all_routes.append(route)
    route_color = {route: ROUTE_PALETTE[i % len(ROUTE_PALETTE)] for i, route in enumerate(all_routes)}

    for idx, series in enumerate(series_list):
        ax = axes[idx // n_cols_final][idx % n_cols_final]
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0.5, max_layer + 0.5)
        ax.grid(axis="y", color="#ccd1d6", linewidth=0.45, alpha=0.75)
        ax.axhline(0, color="#3a3a3a", linewidth=0.65, linestyle=(0, (4, 3)), alpha=0.85)

        if show_phases:
            for start, end, color in [
                (1, 6, "#f1ece2"),
                (6, 13, "#e3edf2"),
                (13, max_layer + 1, "#f3e6e8"),
            ]:
                ax.axvspan(start - 0.5, min(end, max_layer + 1) - 0.5, color=color, alpha=0.42, zorder=0)

        for route in series.route_order:
            centers_map = series.deltas.get(route)
            if not centers_map:
                continue
            centers = sorted(c for c in centers_map if 1 <= c <= max_layer)
            if not centers:
                continue
            means = np.asarray([np.nanmean(centers_map[c]) for c in centers], dtype=float)
            stds = np.asarray([np.nanstd(centers_map[c]) for c in centers], dtype=float)
            color = route_color[route]
            ax.plot(centers, means, marker="o", color=color, label=_route_label(route), zorder=3)
            if show_std and any(len(centers_map[c]) > 1 for c in centers):
                ax.fill_between(centers, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)

        ax.set_title(f"({chr(97 + idx)}) {series.label}\n{_baseline_text(series)}", pad=7)
        ax.set_xticks(list(range(2, max_layer + 1, 2)))

        if idx // n_cols_final == n_rows - 1:
            ax.set_xlabel("Layer center")
        if idx % n_cols_final == 0:
            ax.set_ylabel(_change_label(metric, change))

    for idx in range(len(series_list), n_rows * n_cols_final):
        axes[idx // n_cols_final][idx % n_cols_final].axis("off")

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l, strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=min(5, len(labels)),
            handlelength=2.0,
            columnspacing=1.2,
            handletextpad=0.45,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=11.0, y=1.02)

    fig.tight_layout(rect=(0, 0.065, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved line figure to {output}")


def _print_runs(entries: list[tuple[Path, dict]]) -> None:
    for path, payload in entries:
        meta = _run_meta(path, payload)
        results = payload.get("results", [])
        baseline = next((r for r in results if r.get("condition") == "baseline"), None)
        base_pc = None
        n_episodes = None
        if baseline is not None:
            base_pc, n_episodes = _metric_from_info(baseline.get("info", {}), "overall", "pc_success")
        print(
            f"{meta.policy_id:5s} | {meta.task_id:14s} | baseline={base_pc} | "
            f"n={n_episodes} | {path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Map-The-Flow VLA results as Figure-3-style line panels."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Files, directories, or globs resolving to map_the_flow_info.json files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", choices=["figure3", "lines", "heatmap"], default="figure3")
    parser.add_argument("--metric", default="pc_success", choices=sorted(METRIC_LABELS))
    parser.add_argument("--group", default="overall", help="Metric group to plot, e.g. overall or libero_spatial.")
    parser.add_argument(
        "--change",
        choices=["absolute", "relative"],
        default="absolute",
        help="absolute = knockout - baseline; relative = 100 * (knockout - baseline) / baseline.",
    )
    parser.add_argument(
        "--group-by",
        choices=["auto", "policy_task", "policy", "task", "run", "all"],
        default="auto",
        help="How to facet or aggregate multiple inputs. auto uses policy_task when tasks/policies differ.",
    )
    parser.add_argument("--max-layer", type=int, default=18)
    parser.add_argument("--cols", type=int, default=None, help="Number of subplot columns.")
    parser.add_argument("--annotate", action="store_true", help="Write cell values inside heatmap cells.")
    parser.add_argument(
        "--mark-threshold",
        type=float,
        default=-5.0,
        help="Mark cells with mean change <= this value. Use 'nan' to disable.",
    )
    parser.add_argument("--no-phases", action="store_true", help="Disable early/middle/late layer guides.")
    parser.add_argument("--no-std", action="store_true", help="Disable std bands in line plots.")
    parser.add_argument("--suptitle", type=str, default=None)
    parser.add_argument("--list-runs", action="store_true", help="Print discovered runs before plotting.")
    args = parser.parse_args()

    paths = _resolve_inputs(args.inputs)
    entries = _load_payloads(paths)
    if args.list_runs:
        _print_runs(entries)

    mark_threshold = args.mark_threshold
    if isinstance(mark_threshold, float) and math.isnan(mark_threshold):
        mark_threshold = None

    series_list = _build_group_series(
        entries,
        metric=args.metric,
        group=args.group,
        change=args.change,
        group_by=args.group_by,
    )

    if args.plot == "figure3":
        plot_figure3(
            series_list,
            args.output,
            metric=args.metric,
            change=args.change,
            max_layer=args.max_layer,
            n_cols=args.cols,
            show_std=not args.no_std,
            suptitle=args.suptitle,
        )
    elif args.plot == "heatmap":
        plot_heatmap(
            series_list,
            args.output,
            metric=args.metric,
            change=args.change,
            max_layer=args.max_layer,
            n_cols=args.cols,
            annotate=args.annotate,
            mark_threshold=mark_threshold,
            show_phases=not args.no_phases,
            suptitle=args.suptitle,
        )
    else:
        plot_lines(
            series_list,
            args.output,
            metric=args.metric,
            change=args.change,
            max_layer=args.max_layer,
            n_cols=args.cols,
            show_std=not args.no_std,
            show_phases=not args.no_phases,
            suptitle=args.suptitle,
        )


if __name__ == "__main__":
    main()
