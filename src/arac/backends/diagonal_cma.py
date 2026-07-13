"""Deterministic, auditable diagonal CMA-ES search-state blocks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable

import cma
import numpy as np


BatchObjective = Callable[[np.ndarray], np.ndarray | list[float]]


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _update_array(digest: "hashlib._Hash", value: object) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=float))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


@dataclass
class DiagonalCMAState:
    strategy: object
    rng: np.random.Generator
    lower: np.ndarray
    upper: np.ndarray
    best_x: np.ndarray
    best_y: float
    total_fes: int
    seed: int
    population_size: int

    def validate(self) -> None:
        dimension = self.best_x.size
        if dimension <= 0:
            raise ValueError("state dimension must be positive")
        if self.lower.shape != (dimension,) or self.upper.shape != (dimension,):
            raise ValueError("state boundary shape mismatch")
        if not np.all(np.isfinite(self.best_x)) or not math.isfinite(self.best_y):
            raise ValueError("state incumbent must be finite")
        if self.population_size < 2:
            raise ValueError("state population_size must be at least two")
        if self.total_fes < 0:
            raise ValueError("state total_fes must be non-negative")
        mean = np.asarray(getattr(self.strategy, "mean", ()), dtype=float).reshape(-1)
        if mean.shape != (dimension,) or not np.all(np.isfinite(mean)):
            raise ValueError("strategy mean shape or value is invalid")
        sigma = float(getattr(self.strategy, "sigma", math.nan))
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("strategy sigma must be finite and positive")

    def fingerprint(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        for value in (
            self.lower,
            self.upper,
            self.best_x,
            getattr(self.strategy, "mean"),
            getattr(self.strategy, "pc", np.empty(0)),
            getattr(self.strategy, "pc2", np.empty(0)),
            getattr(getattr(self.strategy, "sigma_vec", None), "scaling", 1.0),
        ):
            _update_array(digest, value)
        scalars = (
            self.best_y,
            self.total_fes,
            self.seed,
            self.population_size,
            float(getattr(self.strategy, "sigma")),
            int(getattr(self.strategy, "countiter", 0)),
            int(getattr(self.strategy, "countevals", 0)),
        )
        digest.update(repr(scalars).encode("ascii"))
        rng_json = json.dumps(
            _jsonable(self.rng.bit_generator.state),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(rng_json.encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class DiagonalCMABlockResult:
    state: DiagonalCMAState
    best_before: float
    best_after: float
    candidate_best: float
    accepted: bool
    requested_fes: int
    actual_fes: int
    unused_fes: int
    population_size: int
    sigma_before: float
    sigma_after: float
    state_fingerprint_before: str
    state_fingerprint_after: str


def initialize_diagonal_cma_state(
    *,
    initial_mean: np.ndarray,
    sigma: float,
    lower: np.ndarray,
    upper: np.ndarray,
    seed: int,
    population_size: int,
    incumbent_fitness: float,
) -> DiagonalCMAState:
    mean = np.asarray(initial_mean, dtype=float).reshape(-1)
    lower_array = np.asarray(lower, dtype=float).reshape(-1)
    upper_array = np.asarray(upper, dtype=float).reshape(-1)
    if lower_array.shape != mean.shape or upper_array.shape != mean.shape:
        raise ValueError("boundary shape must match initial_mean")
    if mean.size == 0 or not np.all(np.isfinite(mean)):
        raise ValueError("initial_mean must be finite and non-empty")
    if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
        raise ValueError("boundaries must be finite")
    if np.any(lower_array >= upper_array):
        raise ValueError("each lower boundary must be below its upper boundary")
    if np.any(mean < lower_array) or np.any(mean > upper_array):
        raise ValueError("initial_mean must lie within boundaries")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if int(population_size) < 2:
        raise ValueError("population_size must be at least two")
    if not math.isfinite(float(incumbent_fitness)):
        raise ValueError("incumbent_fitness must be finite")

    rng = np.random.default_rng(int(seed))

    def local_randn(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape)

    strategy = cma.CMAEvolutionStrategy(
        mean,
        float(sigma),
        {
            "CMA_diagonal": True,
            "bounds": [lower_array.tolist(), upper_array.tolist()],
            "popsize": int(population_size),
            "randn": local_randn,
            "seed": int(seed),
            "verbose": -9,
        },
    )
    state = DiagonalCMAState(
        strategy=strategy,
        rng=rng,
        lower=lower_array.copy(),
        upper=upper_array.copy(),
        best_x=mean.copy(),
        best_y=float(incumbent_fitness),
        total_fes=0,
        seed=int(seed),
        population_size=int(population_size),
    )
    state.validate()
    return state


def run_diagonal_cma_block(
    state: DiagonalCMAState,
    objective: BatchObjective,
    *,
    requested_fes: int,
) -> DiagonalCMABlockResult:
    state.validate()
    requested_fes = max(0, int(requested_fes))
    actual_fes = (requested_fes // state.population_size) * state.population_size
    fingerprint_before = state.fingerprint()
    best_before = float(state.best_y)
    sigma_before = float(getattr(state.strategy, "sigma"))
    candidate_best = math.inf

    for _ in range(actual_fes // state.population_size):
        candidates = np.asarray(state.strategy.ask(), dtype=float)
        expected_shape = (state.population_size, state.best_x.size)
        if candidates.shape != expected_shape or not np.all(np.isfinite(candidates)):
            raise RuntimeError("diagonal CMA-ES returned invalid candidates")
        values = np.asarray(objective(candidates), dtype=float).reshape(-1)
        if values.shape != (state.population_size,):
            raise RuntimeError("objective returned invalid batch shape")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("objective returned non-finite values")
        state.strategy.tell(candidates.tolist(), values.tolist())
        local_index = int(np.argmin(values))
        local_best = float(values[local_index])
        candidate_best = min(candidate_best, local_best)
        if local_best < state.best_y:
            state.best_y = local_best
            state.best_x = candidates[local_index].copy()
        state.total_fes += state.population_size

    fingerprint_after = state.fingerprint()
    return DiagonalCMABlockResult(
        state=state,
        best_before=best_before,
        best_after=float(state.best_y),
        candidate_best=candidate_best,
        accepted=bool(state.best_y < best_before),
        requested_fes=requested_fes,
        actual_fes=actual_fes,
        unused_fes=requested_fes - actual_fes,
        population_size=state.population_size,
        sigma_before=sigma_before,
        sigma_after=float(getattr(state.strategy, "sigma")),
        state_fingerprint_before=fingerprint_before,
        state_fingerprint_after=fingerprint_after,
    )
