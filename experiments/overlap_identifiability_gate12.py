"""Gate 12: fixed-probe identifiability audit for nonseparable overlap."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.overlap_objective import build_overlap_problem


DIMENSION = 24
NUM_GROUPS = 6
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 5
BASE_FUNCTION = "sphere"
BOUNDS = 10.0
INTERACTION_STRENGTH = 0.25
PROBE_COUNT = 128
TRAIN_COUNT = 96
HELDOUT_COUNT = PROBE_COUNT - TRAIN_COUNT
FRESH_SEEDS = (31001, 31002, 31003, 31004, 31005)
TOPOLOGIES = ("random", "chain", "star")
OVERLAP_BUDGETS = (6, 12)
MODES = ("conforming", "conflicting")


def _build(mode: str, topology: str, overlap_budget: int, seed: int):
    return build_overlap_problem(
        DIMENSION,
        overlap_budget=overlap_budget,
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        num_groups=NUM_GROUPS,
        base_function=BASE_FUNCTION,
        conflict_mode=mode,
        bounds=BOUNDS,
        contiguous=True,
        rotation=False,
        transforms=False,
        interaction_strength=INTERACTION_STRENGTH,
        seed=seed,
        topology=topology,
    )


def _probe_points(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed ^ 0x12A5)
    points = rng.uniform(-2.0, 2.0, size=(PROBE_COUNT, DIMENSION))
    return points[:TRAIN_COUNT], points[TRAIN_COUNT:]


def _quadratic_design(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 8 or values.shape[1] <= 0:
        raise ValueError("quadratic fit requires at least eight non-empty probe rows")
    return np.column_stack((np.ones(len(values)), values, values**2))


def _quadratic_fit_residual(objective, train_points: np.ndarray, heldout_points: np.ndarray) -> dict[str, float | int]:
    train = np.asarray(train_points, dtype=float)
    heldout = np.asarray(heldout_points, dtype=float)
    if train.ndim != 2 or heldout.ndim != 2 or train.shape[1:] != heldout.shape[1:]:
        raise ValueError("train and held-out probes must be two-dimensional with equal width")
    design_train = _quadratic_design(train)
    if len(heldout) == 0:
        raise ValueError("held-out probes must be non-empty")
    design_heldout = _quadratic_design(heldout)
    train_values = np.asarray(objective(train), dtype=float).reshape(-1)
    heldout_values = np.asarray(objective(heldout), dtype=float).reshape(-1)
    if train_values.shape != (len(train),) or heldout_values.shape != (len(heldout),):
        raise ValueError("objective must return one value per probe")
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design_train, train_values, rcond=None
    )
    train_prediction = design_train @ coefficients
    heldout_prediction = design_heldout @ coefficients
    return {
        "train_rmse": float(np.sqrt(np.mean((train_prediction - train_values) ** 2))),
        "heldout_rmse": float(np.sqrt(np.mean((heldout_prediction - heldout_values) ** 2))),
        "feature_count": int(design_train.shape[1]),
        "design_rank": int(rank),
    }


def _mixed_difference(objective, base: np.ndarray, left: int, right: int, step: float) -> float:
    point = np.asarray(base, dtype=float)
    if point.ndim != 1 or left == right or left not in range(len(point)) or right not in range(len(point)):
        raise ValueError("mixed-difference coordinates must be two distinct point indices")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("mixed-difference step must be finite and positive")
    values = []
    for left_sign, right_sign in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        candidate = point.copy()
        candidate[left] += left_sign * step
        candidate[right] += right_sign * step
        values.append(float(objective(candidate)))
    return (values[0] - values[1] - values[2] + values[3]) / (4.0 * step**2)


def _interaction_pairs(objective) -> tuple[tuple[int, int], ...]:
    pairs: set[tuple[int, int]] = set()
    for group, local_pairs in zip(objective.structure.groups, objective.interaction_pairs, strict=True):
        for left, right in local_pairs:
            global_pair = tuple(sorted((int(group[left]), int(group[right]))))
            pairs.add(global_pair)
    return tuple(sorted(pairs))


def _parameter_parity(conforming, conflicting) -> bool:
    if conforming.structure.groups != conflicting.structure.groups:
        return False
    if not np.array_equal(conforming._weights, conflicting._weights):
        return False
    if conforming.config.interaction_strength != conflicting.config.interaction_strength:
        return False
    return conforming.interaction_pairs == conflicting.interaction_pairs


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> dict[str, object]:
    _problem, objective = _build(mode, topology, overlap_budget, seed)
    counterpart_mode = "conflicting" if mode == "conforming" else "conforming"
    _counterpart_problem, counterpart = _build(counterpart_mode, topology, overlap_budget, seed)
    train, heldout = _probe_points(seed)
    pairs = _interaction_pairs(objective)
    base = np.zeros(DIMENSION, dtype=float)
    mixed_values = tuple(
        _mixed_difference(objective.evaluate, base, left, right, 0.125)
        for left, right in pairs
    )
    fit = _quadratic_fit_residual(objective.evaluate, train, heldout)
    return {
        "mode": mode,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "groups": objective.structure.groups,
        "shared_variables": objective.structure.shared_variables,
        "interaction_pairs": pairs,
        "interaction_pair_count": len(pairs),
        "interaction_strength": objective.interaction_strength,
        "parameter_parity": _parameter_parity(objective, counterpart),
        "max_abs_mixed_difference": float(max((abs(value) for value in mixed_values), default=0.0)),
        "mixed_difference_values": mixed_values,
        "quadratic_fit": fit,
        "probe_count": len(train) + len(heldout),
    }


def _pair_contexts(contexts: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    pairs = []
    for topology in TOPOLOGIES:
        for overlap_budget in OVERLAP_BUDGETS:
            for seed in FRESH_SEEDS:
                rows = tuple(
                    item
                    for item in contexts
                    if item["topology"] == topology
                    and item["overlap_budget"] == overlap_budget
                    and item["seed"] == seed
                )
                if len(rows) != 2:
                    raise RuntimeError("each context key must contain conforming and conflicting rows")
                conforming = next(item for item in rows if item["mode"] == "conforming")
                conflicting = next(item for item in rows if item["mode"] == "conflicting")
                pairs.append(
                    {
                        "topology": topology,
                        "overlap_budget": overlap_budget,
                        "seed": seed,
                        "parameter_parity": conforming["parameter_parity"]
                        and conflicting["parameter_parity"]
                        and conforming["groups"] == conflicting["groups"]
                        and conforming["interaction_pairs"] == conflicting["interaction_pairs"]
                        and conforming["interaction_strength"] == conflicting["interaction_strength"],
                        "conforming_pair_count": conforming["interaction_pair_count"],
                        "conflicting_pair_count": conflicting["interaction_pair_count"],
                    }
                )
    return tuple(pairs)


def run_gate(*, workers: int = 1) -> dict[str, object]:
    jobs = tuple(
        (mode, topology, overlap_budget, seed)
        for topology in TOPOLOGIES
        for overlap_budget in OVERLAP_BUDGETS
        for seed in FRESH_SEEDS
        for mode in MODES
    )
    if workers == 1:
        contexts = tuple(_context(*job) for job in jobs)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            contexts = tuple(executor.map(lambda job: _context(*job), jobs))
    contexts = tuple(sorted(contexts, key=lambda item: (item["topology"], item["overlap_budget"], item["seed"], item["mode"])))
    paired = _pair_contexts(contexts)
    gate_checks = {
        "context_count_60": len(contexts) == 60,
        "parameter_parity_all_pairs": all(item["parameter_parity"] for item in paired),
        "shared_interaction_exists": all(item["interaction_pair_count"] > 0 for item in contexts),
        "mixed_difference_nonzero_all_contexts": all(
            item["max_abs_mixed_difference"] > 1.0e-6 for item in contexts
        ),
        "quadratic_fit_residual_positive_all_conflicting": all(
            item["quadratic_fit"]["heldout_rmse"] > 1.0e-6
            for item in contexts
            if item["mode"] == "conflicting"
        ),
        "quadratic_fit_metrics_finite": all(
            np.isfinite(item["quadratic_fit"]["heldout_rmse"])
            for item in contexts
        ),
    }
    return {
        "schema_version": "arac-overlap-identifiability-gate12-v1",
        "protocol": {
            "dimension": DIMENSION,
            "num_groups": NUM_GROUPS,
            "base_function": BASE_FUNCTION,
            "interaction_strength": INTERACTION_STRENGTH,
            "probe_count": PROBE_COUNT,
            "train_count": TRAIN_COUNT,
            "heldout_count": HELDOUT_COUNT,
            "rotation": False,
            "transforms": False,
            "seeds": FRESH_SEEDS,
            "topologies": TOPOLOGIES,
            "overlap_budgets": OVERLAP_BUDGETS,
        },
        "context_count": len(contexts),
        "contexts": contexts,
        "paired_contexts": paired,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "scientific_finding": {
            "interaction_breaks_additive_quadratic_surface": all(
                item["quadratic_fit"]["heldout_rmse"] > 1.0e-6 for item in contexts
            ),
            "benchmark_is_ready_for_separate_proposal_gate": all(gate_checks.values()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_identifiability_gate12/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
