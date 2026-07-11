"""Checkpoint records for resumable MMES execution."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value)


def _array_payload(value: Any) -> dict[str, object]:
    array = np.ascontiguousarray(_as_array(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _jsonable(value: Any) -> object:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite_scalar(name: str, value: float, *, allow_infinite: bool = False) -> None:
    numeric = float(value)
    if math.isnan(numeric) or (not allow_infinite and not math.isfinite(numeric)):
        raise ValueError(f"{name} must be finite")


@dataclass
class MMESState:
    x: np.ndarray
    mean: np.ndarray
    p: np.ndarray
    w: float
    q: np.ndarray
    t: np.ndarray
    v: np.ndarray
    y: np.ndarray
    sigma: float
    n_individuals: int
    n_parents: int
    n_mirror_sampling: int
    n_generations: int
    n_restart: int
    list_generations: list[int]
    list_fitness: list[float]
    list_initial_mean: list[np.ndarray]
    best_so_far_x: np.ndarray
    best_so_far_y: float
    n_function_evaluations: int
    termination_signal: int
    fitness: list[float]
    recent_best: list[tuple[int, float]]
    rng_initialization_state: dict[str, object]
    rng_optimization_state: dict[str, object]
    sigma_bak: float | None = None
    initial_mean: np.ndarray | None = None
    counter_early_stopping: int = 0
    base_early_stopping: float = 0.0
    printed_evaluations: int = 0
    time_function_evaluations: float = 0.0
    runtime: float = 0.0

    def validate(self) -> None:
        """Reject malformed continuation state before objective evaluation."""

        population = int(self.n_individuals)
        parents = int(self.n_parents)
        mirror = int(self.n_mirror_sampling)
        if population <= 0:
            raise ValueError("n_individuals must be positive")
        if parents <= 0 or parents > population:
            raise ValueError("n_parents must be positive and <= n_individuals")
        if mirror != int(math.ceil(population / 2.0)):
            raise ValueError("n_mirror_sampling does not match population")
        for name in (
            "n_generations",
            "n_restart",
            "n_function_evaluations",
            "termination_signal",
            "counter_early_stopping",
            "printed_evaluations",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

        best_x = _as_array(self.best_so_far_x)
        if best_x.ndim != 1 or best_x.size == 0:
            raise ValueError("best_so_far_x shape is invalid")
        ndim = int(best_x.size)
        arrays = {
            "x": _as_array(self.x),
            "mean": _as_array(self.mean),
            "p": _as_array(self.p),
            "q": _as_array(self.q),
            "t": _as_array(self.t),
            "v": _as_array(self.v),
            "y": _as_array(self.y),
            "best_so_far_x": best_x,
        }
        for name, array in arrays.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        if arrays["x"].shape != (population, ndim):
            raise ValueError(f"x shape must be {(population, ndim)}")
        if arrays["mean"].shape not in {(ndim,), (1, ndim)}:
            raise ValueError("mean shape is invalid")
        if arrays["p"].shape not in {(ndim,), (1, ndim)}:
            raise ValueError("p shape is invalid")
        if arrays["q"].ndim != 2 or arrays["q"].shape[1] != ndim:
            raise ValueError("q shape is invalid")
        direction_count = arrays["q"].shape[0]
        if arrays["t"].shape != (direction_count,):
            raise ValueError("t shape is invalid")
        if arrays["v"].shape != (direction_count,):
            raise ValueError("v shape is invalid")
        if arrays["y"].shape != (population,):
            raise ValueError(f"y shape must be {(population,)}")
        if self.initial_mean is not None:
            initial_mean = _as_array(self.initial_mean)
            if initial_mean.shape not in {(ndim,), (1, ndim)}:
                raise ValueError("initial_mean shape is invalid")
            if not np.all(np.isfinite(initial_mean)):
                raise ValueError("initial_mean contains non-finite values")
        for index, initial_mean in enumerate(self.list_initial_mean):
            array = _as_array(initial_mean)
            if array.shape not in {(ndim,), (1, ndim)}:
                raise ValueError(f"list_initial_mean[{index}] shape is invalid")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"list_initial_mean[{index}] contains non-finite values")

        _finite_scalar("w", self.w)
        _finite_scalar("sigma", self.sigma)
        if float(self.sigma) <= 0.0:
            raise ValueError("sigma must be positive")
        if self.sigma_bak is not None:
            _finite_scalar("sigma_bak", self.sigma_bak)
            if float(self.sigma_bak) <= 0.0:
                raise ValueError("sigma_bak must be positive")
        _finite_scalar("best_so_far_y", self.best_so_far_y)
        _finite_scalar("base_early_stopping", self.base_early_stopping, allow_infinite=True)
        _finite_scalar("time_function_evaluations", self.time_function_evaluations)
        _finite_scalar("runtime", self.runtime)
        if float(self.time_function_evaluations) < 0.0 or float(self.runtime) < 0.0:
            raise ValueError("timing values must be non-negative")

        if len(self.recent_best) > 3:
            raise ValueError("recent_best must contain at most three checkpoints")
        previous_fe = -1
        for checkpoint in self.recent_best:
            if len(checkpoint) != 2:
                raise ValueError("recent_best checkpoint must contain FE and best")
            checkpoint_fe, checkpoint_best = checkpoint
            if int(checkpoint_fe) < 0 or int(checkpoint_fe) < previous_fe:
                raise ValueError("recent_best FE checkpoints must be ordered")
            _finite_scalar("recent_best best", checkpoint_best)
            previous_fe = int(checkpoint_fe)
        for name, values in (("fitness", self.fitness), ("list_fitness", self.list_fitness)):
            for index, value in enumerate(values):
                _finite_scalar(f"{name}[{index}]", value, allow_infinite=name == "list_fitness")
        if not isinstance(self.rng_initialization_state, dict):
            raise ValueError("rng_initialization_state must be a dictionary")
        if not isinstance(self.rng_optimization_state, dict):
            raise ValueError("rng_optimization_state must be a dictionary")

    def clone(self) -> "MMESState":
        return copy.deepcopy(self)

    def fingerprint(self) -> str:
        self.validate()
        payload: dict[str, object] = {
            "arrays": {
                name: _array_payload(getattr(self, name))
                for name in ("x", "mean", "p", "q", "t", "v", "y", "best_so_far_x")
            },
            "scalars": {
                name: getattr(self, name)
                for name in (
                    "w",
                    "sigma",
                    "sigma_bak",
                    "n_individuals",
                    "n_parents",
                    "n_mirror_sampling",
                    "n_generations",
                    "n_restart",
                    "best_so_far_y",
                    "n_function_evaluations",
                    "termination_signal",
                    "counter_early_stopping",
                    "base_early_stopping",
                    "printed_evaluations",
                    "time_function_evaluations",
                    "runtime",
                )
            },
            "initial_mean": None
            if self.initial_mean is None
            else _array_payload(self.initial_mean),
            "list_generations": list(self.list_generations),
            "list_fitness": list(self.list_fitness),
            "list_initial_mean": [_array_payload(value) for value in self.list_initial_mean],
            "fitness": list(self.fitness),
            "recent_best": [list(value) for value in self.recent_best],
            "rng_initialization_state": _jsonable(self.rng_initialization_state),
            "rng_optimization_state": _jsonable(self.rng_optimization_state),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MMESBlockResult:
    state: MMESState
    best_before: float
    best_after: float
    actual_fes: int
    requested_fes: int
    unused_fes: int
    normalized_utility: float
    termination_reason: str
    state_fingerprint_before: str
    state_fingerprint_after: str
