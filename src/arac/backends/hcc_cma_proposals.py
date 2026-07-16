"""ARAC-owned observer for an isolated first-generation HCC CMA probe."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class HccCMAProbeArmResult:
    sigma: float
    actual_fe: int
    candidates: np.ndarray
    objectives: tuple[float, ...]
    standardized_directions: np.ndarray
    direction_sha256: str
    candidate_sha256: str
    standardized_diversity: float
    boundary_hit_count: int
    best_candidate: np.ndarray
    best_objective: float


@dataclass(frozen=True)
class PairedHccCMAProbeResult:
    normal: HccCMAProbeArmResult
    precision: HccCMAProbeArmResult
    direction_hash_match: bool
    standardized_diversity_ratio: float
    total_fe: int


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    payload = f"{array.shape}|".encode("ascii") + array.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _standardized_diversity(directions: np.ndarray) -> float:
    centroid = np.mean(directions, axis=0)
    return float(np.mean(np.linalg.norm(directions - centroid, axis=1)))


def _run_probe_arm(
    *,
    fitness_function: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sigma: float,
    seed: int,
    pair_count: int,
) -> HccCMAProbeArmResult:
    from HCC.OPT.CMAES.cmaes import CMAES

    class ObservedFirstGenerationCMA(CMAES):
        observed_candidates: np.ndarray | None = None
        observed_directions: np.ndarray | None = None

        def iterate(self, *args, **kwargs):
            x_values, y_values, directions = super().iterate(*args, **kwargs)
            self.observed_candidates = np.asarray(x_values, dtype=float).copy()
            self.observed_directions = np.asarray(directions, dtype=float).copy()
            return x_values, y_values, directions

    captured_objectives: tuple[float, ...] = ()

    def observed_objective(candidates: np.ndarray) -> np.ndarray:
        nonlocal captured_objectives
        values = np.asarray(fitness_function(candidates), dtype=float).reshape(-1)
        captured_objectives = tuple(float(value) for value in values)
        return values

    problem = {
        "fitness_function": observed_objective,
        "ndim_problem": int(mean.size),
        "lower_boundary": lower.copy(),
        "upper_boundary": upper.copy(),
    }
    options = {
        "max_function_evaluations": int(pair_count),
        "mean": (mean.copy(),),
        "sigma": float(sigma),
        "n_individuals": int(pair_count),
        "is_restart": False,
        "seed_rng": int(seed),
        "verbose": 0,
    }
    optimizer = ObservedFirstGenerationCMA(problem, options)
    result = optimizer.optimize()
    candidates = optimizer.observed_candidates
    directions = optimizer.observed_directions
    if candidates is None or directions is None:
        raise RuntimeError("HCC CMA probe did not expose its first generation")
    if candidates.shape != (pair_count, mean.size):
        raise RuntimeError("HCC CMA probe candidate shape mismatch")
    if len(captured_objectives) != pair_count:
        raise RuntimeError("HCC CMA probe did not consume one complete population")
    boundary_hits = int(
        np.sum(np.any((candidates < lower) | (candidates > upper), axis=1))
    )
    best_index = int(np.argmin(np.asarray(captured_objectives, dtype=float)))
    return HccCMAProbeArmResult(
        sigma=float(sigma),
        actual_fe=int(result["n_function_evaluations"]),
        candidates=candidates,
        objectives=captured_objectives,
        standardized_directions=directions,
        direction_sha256=_array_sha256(directions),
        candidate_sha256=_array_sha256(candidates),
        standardized_diversity=_standardized_diversity(directions),
        boundary_hit_count=boundary_hits,
        best_candidate=candidates[best_index].copy(),
        best_objective=float(captured_objectives[best_index]),
    )


def run_paired_hcc_cma_probe(
    *,
    fitness_function: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    normal_sigma: float,
    precision_sigma: float,
    seed: int,
    pair_count: int = 16,
) -> PairedHccCMAProbeResult:
    center = np.asarray(mean, dtype=float).reshape(-1)
    lower_values = np.broadcast_to(np.asarray(lower, dtype=float), center.shape).copy()
    upper_values = np.broadcast_to(np.asarray(upper, dtype=float), center.shape).copy()
    if center.size <= 0 or not np.all(np.isfinite(center)):
        raise ValueError("probe mean must be finite and non-empty")
    if np.any(lower_values >= upper_values):
        raise ValueError("probe bounds are invalid")
    if pair_count != 16:
        raise ValueError("precision-response-loop-v1 requires 16 pairs")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (float(normal_sigma), float(precision_sigma))
    ):
        raise ValueError("probe sigmas must be finite and positive")

    normal = _run_probe_arm(
        fitness_function=fitness_function,
        mean=center,
        lower=lower_values,
        upper=upper_values,
        sigma=float(normal_sigma),
        seed=int(seed),
        pair_count=pair_count,
    )
    precision = _run_probe_arm(
        fitness_function=fitness_function,
        mean=center,
        lower=lower_values,
        upper=upper_values,
        sigma=float(precision_sigma),
        seed=int(seed),
        pair_count=pair_count,
    )
    diversity_ratio = precision.standardized_diversity / max(
        normal.standardized_diversity,
        1e-300,
    )
    return PairedHccCMAProbeResult(
        normal=normal,
        precision=precision,
        direction_hash_match=(
            normal.direction_sha256 == precision.direction_sha256
        ),
        standardized_diversity_ratio=float(diversity_ratio),
        total_fe=normal.actual_fe + precision.actual_fe,
    )
