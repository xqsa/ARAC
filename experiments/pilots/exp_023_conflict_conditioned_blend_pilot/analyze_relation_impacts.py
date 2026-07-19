"""Analyze immediate objective effects of exp_023 repair dispatches."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_RESULTS_ROOT = (
    REPOSITORY_ROOT / "results" / "exp_023_conflict_conditioned_blend_pilot"
)
REPAIR_ACTION = "repair_shared_variable_binding"


@dataclass(frozen=True)
class RepairImpactEvent:
    problem_id: str
    seed: int
    outer_iter: int
    group_left: int
    group_right: int
    shared_vars_hash: str
    probe_utility: float
    probe_utility_threshold: float
    previous_delta: float
    current_delta: float
    local_pre_writeback_fitness: float
    local_post_writeback_fitness: float
    local_objective_credit: float

    @property
    def cluster_key(self) -> str:
        return ":".join(
            (
                self.problem_id,
                str(self.seed),
                str(self.group_left),
                str(self.group_right),
                self.shared_vars_hash,
            )
        )

    @property
    def run_key(self) -> str:
        return f"{self.problem_id}:{self.seed}"

    @property
    def utility_ratio(self) -> float:
        return self.probe_utility / self.probe_utility_threshold

    @property
    def log_utility_ratio(self) -> float:
        return math.log(self.utility_ratio)

    @property
    def delta_gap_ratio(self) -> float:
        scale = max(abs(self.previous_delta), abs(self.current_delta), 1e-12)
        return abs(self.previous_delta - self.current_delta) / scale


def load_analysis_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    analysis = payload.get("relation_impact_analysis")
    if not isinstance(analysis, dict):
        raise ValueError("config is missing relation_impact_analysis")
    if float(analysis["boundary_utility_ratio_max"]) <= 1.0:
        raise ValueError("boundary utility ratio must be above one")
    return analysis


def load_repair_events(
    results_root: Path,
) -> tuple[list[RepairImpactEvent], int, int]:
    trace_paths = sorted(
        path
        for path in results_root.rglob("*_action_trace.csv")
        if path.name != "action_trace.csv"
    )
    if not trace_paths:
        raise FileNotFoundError(f"no action traces under {results_root}")

    events: list[RepairImpactEvent] = []
    repair_rows = 0
    incomplete_rows = 0
    telemetry_fields = (
        "probe_utility",
        "probe_utility_threshold",
        "local_pre_writeback_fitness",
        "local_post_writeback_fitness",
        "local_objective_credit",
    )
    for path in trace_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["selected_action_name"] != REPAIR_ACTION:
                    continue
                repair_rows += 1
                if any(not row.get(field, "") for field in telemetry_fields):
                    incomplete_rows += 1
                    continue
                threshold = float(row["probe_utility_threshold"])
                utility = float(row["probe_utility"])
                if threshold <= 0.0 or utility <= threshold:
                    raise ValueError(
                        f"invalid repair utility in {path}: {utility=} {threshold=}"
                    )
                events.append(
                    RepairImpactEvent(
                        problem_id=row["problem_id"],
                        seed=int(row["seed"]),
                        outer_iter=int(row["outer_iter"]),
                        group_left=int(row["group_left"]),
                        group_right=int(row["group_right"]),
                        shared_vars_hash=row["shared_vars_hash"],
                        probe_utility=utility,
                        probe_utility_threshold=threshold,
                        previous_delta=float(row["previous_delta"]),
                        current_delta=float(row["current_delta"]),
                        local_pre_writeback_fitness=float(
                            row["local_pre_writeback_fitness"]
                        ),
                        local_post_writeback_fitness=float(
                            row["local_post_writeback_fitness"]
                        ),
                        local_objective_credit=float(row["local_objective_credit"]),
                    )
                )
    return events, repair_rows, incomplete_rows


def _summary(events: list[RepairImpactEvent]) -> dict:
    values = np.asarray([event.local_objective_credit for event in events])
    if values.size == 0:
        return {"event_count": 0}
    return {
        "event_count": int(values.size),
        "relation_cluster_count": len({event.cluster_key for event in events}),
        "run_cluster_count": len({event.run_key for event in events}),
        "mean_credit": float(np.mean(values)),
        "median_credit": float(np.median(values)),
        "variance": float(np.var(values, ddof=1)) if values.size > 1 else None,
        "mean_absolute_credit": float(np.mean(np.abs(values))),
        "negative_fraction": float(np.mean(values < 0.0)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "zero_fraction": float(np.mean(values == 0.0)),
    }


def _comparison(
    events: list[RepairImpactEvent],
    boundary_ratio_max: float,
) -> dict | None:
    boundary = np.asarray(
        [
            event.local_objective_credit
            for event in events
            if event.utility_ratio <= boundary_ratio_max
        ]
    )
    nonboundary = np.asarray(
        [
            event.local_objective_credit
            for event in events
            if event.utility_ratio > boundary_ratio_max
        ]
    )
    if boundary.size < 2 or nonboundary.size < 2:
        return None
    boundary_variance = float(np.var(boundary, ddof=1))
    nonboundary_variance = float(np.var(nonboundary, ddof=1))
    boundary_mad = float(np.mean(np.abs(boundary - np.median(boundary))))
    nonboundary_mad = float(np.mean(np.abs(nonboundary - np.median(nonboundary))))
    return {
        "boundary_mean": float(np.mean(boundary)),
        "nonboundary_mean": float(np.mean(nonboundary)),
        "mean_difference": float(np.mean(boundary) - np.mean(nonboundary)),
        "boundary_negative_fraction": float(np.mean(boundary < 0.0)),
        "nonboundary_negative_fraction": float(np.mean(nonboundary < 0.0)),
        "negative_fraction_difference": float(
            np.mean(boundary < 0.0) - np.mean(nonboundary < 0.0)
        ),
        "variance_ratio": (
            boundary_variance / nonboundary_variance
            if nonboundary_variance > 0.0
            else None
        ),
        "median_absolute_deviation_ratio": (
            boundary_mad / nonboundary_mad if nonboundary_mad > 0.0 else None
        ),
    }


def _fit_ols(events: list[RepairImpactEvent], boundary_ratio_max: float) -> dict:
    y = np.asarray([event.local_objective_credit for event in events])
    x = np.asarray(
        [
            [
                1.0,
                event.log_utility_ratio,
                event.delta_gap_ratio,
                float(event.problem_id == "S5"),
            ]
            for event in events
        ]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    predictions = x @ coefficients
    residuals = y - predictions
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    mean_model = {
        "features": ["intercept", "log_utility_ratio", "delta_gap_ratio", "S5"],
        "coefficients": [float(value) for value in coefficients],
        "rank": int(rank),
        "r_squared": (
            1.0 - float(np.sum(residuals**2)) / total_sum_squares
            if total_sum_squares > 0.0
            else None
        ),
    }

    boundary_mask = np.asarray(
        [event.utility_ratio <= boundary_ratio_max for event in events]
    )
    boundary_residuals = residuals[boundary_mask]
    nonboundary_residuals = residuals[~boundary_mask]
    residual_variance_ratio = (
        float(np.var(boundary_residuals, ddof=1))
        / float(np.var(nonboundary_residuals, ddof=1))
        if len(boundary_residuals) > 1
        and len(nonboundary_residuals) > 1
        and np.var(nonboundary_residuals, ddof=1) > 0.0
        else None
    )
    dispersion_x = np.column_stack(
        (
            np.ones(len(events)),
            boundary_mask.astype(float),
            [event.delta_gap_ratio for event in events],
            [float(event.problem_id == "S5") for event in events],
        )
    )
    dispersion_y = np.abs(residuals)
    dispersion_coefficients, _, dispersion_rank, _ = np.linalg.lstsq(
        dispersion_x,
        dispersion_y,
        rcond=None,
    )
    return {
        "mean_model": mean_model,
        "absolute_residual_model": {
            "features": [
                "intercept",
                "boundary_indicator",
                "delta_gap_ratio",
                "S5",
            ],
            "coefficients": [float(value) for value in dispersion_coefficients],
            "rank": int(dispersion_rank),
            "residual_variance_ratio": residual_variance_ratio,
            "boundary_mean_absolute_residual": float(
                np.mean(np.abs(boundary_residuals))
            ),
            "nonboundary_mean_absolute_residual": float(
                np.mean(np.abs(nonboundary_residuals))
            ),
        },
    }


def _interval(values: list[float]) -> list[float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if finite.size == 0:
        return None
    return [float(value) for value in np.quantile(finite, (0.025, 0.975))]


def _cluster_bootstrap(
    events: list[RepairImpactEvent],
    boundary_ratio_max: float,
    samples: int,
    seed: int,
    cluster_level: str,
) -> dict:
    clusters: dict[str, list[RepairImpactEvent]] = defaultdict(list)
    for event in events:
        key = event.cluster_key if cluster_level == "relation" else event.run_key
        clusters[key].append(event)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        selected = rng.choice(keys, size=len(keys), replace=True)
        sample = [event for key in selected for event in clusters[str(key)]]
        comparison = _comparison(sample, boundary_ratio_max)
        if comparison is None:
            continue
        for name, value in comparison.items():
            if value is not None:
                metrics[name].append(float(value))
        regression = _fit_ols(sample, boundary_ratio_max)
        metrics["mean_log_utility_coefficient"].append(
            regression["mean_model"]["coefficients"][1]
        )
        metrics["mean_delta_gap_coefficient"].append(
            regression["mean_model"]["coefficients"][2]
        )
        metrics["dispersion_boundary_coefficient"].append(
            regression["absolute_residual_model"]["coefficients"][1]
        )
        if regression["absolute_residual_model"]["residual_variance_ratio"] is not None:
            metrics["residual_variance_ratio"].append(
                regression["absolute_residual_model"]["residual_variance_ratio"]
            )
    return {
        "cluster_level": cluster_level,
        "cluster_count": len(keys),
        "samples_requested": samples,
        "samples_usable": len(metrics["boundary_mean"]),
        "intervals_95": {name: _interval(values) for name, values in metrics.items()},
        "variance_ratio_above_one_probability": float(
            np.mean(np.asarray(metrics["variance_ratio"]) > 1.0)
        ),
        "boundary_mean_above_zero_probability": float(
            np.mean(np.asarray(metrics["boundary_mean"]) > 0.0)
        ),
    }


def analyze(
    events: list[RepairImpactEvent],
    *,
    boundary_ratio_max: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    boundary = [event for event in events if event.utility_ratio <= boundary_ratio_max]
    nonboundary = [event for event in events if event.utility_ratio > boundary_ratio_max]
    comparison = _comparison(events, boundary_ratio_max)
    if comparison is None:
        raise ValueError("both boundary and non-boundary groups need at least two events")
    relation_bootstrap = _cluster_bootstrap(
        events,
        boundary_ratio_max,
        bootstrap_samples,
        bootstrap_seed,
        "relation",
    )
    run_bootstrap = _cluster_bootstrap(
        events,
        boundary_ratio_max,
        bootstrap_samples,
        bootstrap_seed,
        "run",
    )
    mean_interval = relation_bootstrap["intervals_95"]["boundary_mean"]
    variance_interval = relation_bootstrap["intervals_95"]["variance_ratio"]
    residual_variance_interval = relation_bootstrap["intervals_95"].get(
        "residual_variance_ratio"
    )
    dispersion_boundary_interval = relation_bootstrap["intervals_95"].get(
        "dispersion_boundary_coefficient"
    )
    return {
        "boundary_definition": (
            f"1 < probe_utility / threshold <= {boundary_ratio_max:g}"
        ),
        "all_repairs": _summary(events),
        "boundary_repairs": _summary(boundary),
        "nonboundary_repairs": _summary(nonboundary),
        "by_problem": {
            problem_id: _summary(
                [event for event in events if event.problem_id == problem_id]
            )
            for problem_id in sorted({event.problem_id for event in events})
        },
        "boundary_vs_nonboundary": comparison,
        "regression": _fit_ols(events, boundary_ratio_max),
        "relation_cluster_bootstrap": relation_bootstrap,
        "run_cluster_bootstrap_sensitivity": run_bootstrap,
        "diagnostic_flags": {
            "boundary_mean_interval_contains_zero": (
                mean_interval is not None
                and mean_interval[0] <= 0.0 <= mean_interval[1]
            ),
            "boundary_variance_lcb_above_one": (
                variance_interval is not None and variance_interval[0] > 1.0
            ),
            "boundary_variance_ucb_below_one": (
                variance_interval is not None and variance_interval[1] < 1.0
            ),
            "residual_boundary_variance_ucb_below_one": (
                residual_variance_interval is not None
                and residual_variance_interval[1] < 1.0
            ),
            "dispersion_boundary_coefficient_lcb_above_zero": (
                dispersion_boundary_interval is not None
                and dispersion_boundary_interval[0] > 0.0
            ),
        },
    }


def write_events(path: Path, events: list[RepairImpactEvent]) -> None:
    fieldnames = [
        *asdict(events[0]).keys(),
        "cluster_key",
        "utility_ratio",
        "log_utility_ratio",
        "delta_gap_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    **asdict(event),
                    "cluster_key": event.cluster_key,
                    "utility_ratio": event.utility_ratio,
                    "log_utility_ratio": event.log_utility_ratio,
                    "delta_gap_ratio": event.delta_gap_ratio,
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_analysis_config(args.config)
    events, repair_rows, incomplete_rows = load_repair_events(args.results_root)
    if not events:
        raise ValueError("no complete repair impact events")
    report = analyze(
        events,
        boundary_ratio_max=float(config["boundary_utility_ratio_max"]),
        bootstrap_samples=int(config["bootstrap_samples"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    report.update(
        {
            "experiment_id": "exp_023_conflict_conditioned_blend_pilot",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executor": "Codex",
            "source_results_root": str(args.results_root.resolve()),
            "repair_trace_rows": repair_rows,
            "incomplete_repair_rows": incomplete_rows,
        }
    )

    output_dir = args.output_dir or args.results_root / "relation_impact_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "repair_impact_events.csv"
    report_path = output_dir / "relation_impact_report.json"
    write_events(events_path, events)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Analyzed {len(events)}/{repair_rows} repair events; "
        f"boundary={report['boundary_repairs']['event_count']}, "
        f"nonboundary={report['nonboundary_repairs']['event_count']}"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
