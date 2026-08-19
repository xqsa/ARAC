"""Thin, versioned ports to upstream optimization packages."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import math
import time

import numpy as np
from pypop7.optimizers.es.cmaes import CMAES
from pypop7.optimizers.es.mmes import MMES
from pypop7.optimizers.es.sepcmaes import SEPCMAES

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.ledger import EvaluationLedger


class _BatchedEvaluation:
    def _evaluate_population(self, candidates: np.ndarray, args: object = None) -> np.ndarray:
        rows = np.asarray(candidates, dtype=float)
        if args is not None:
            raise ValueError("batched optimizer ports do not accept objective args")
        if len(rows) > self.max_function_evaluations - self.n_function_evaluations:
            raise RuntimeError("optimizer population exceeds its remaining FE")
        started = time.time()
        values = np.asarray(self.fitness_function(rows), dtype=float).reshape(-1)
        self.time_function_evaluations += time.time() - started
        if values.shape != (len(rows),) or not np.all(np.isfinite(values)):
            raise ValueError("objective must return one finite value per optimizer candidate")
        for candidate, value in zip(rows, values, strict=True):
            numeric = float(value)
            self.n_function_evaluations += 1
            if numeric < self.best_so_far_y:
                self.best_so_far_x = candidate.copy()
                self.best_so_far_y = numeric
            if (self._base_early_stopping - numeric) <= self.early_stopping_threshold:
                self._counter_early_stopping += 1
            else:
                self._counter_early_stopping = 0
                self._base_early_stopping = numeric
        return values


class _BatchedCMAES(_BatchedEvaluation, CMAES):
    def iterate(self, x=None, mean=None, e_ve=None, e_va=None, y=None, d=None, args=None):
        active = min(
            self.n_individuals,
            int(self.max_function_evaluations - self.n_function_evaluations),
        )
        transform = e_ve @ np.diag(e_va)
        for index in range(active):
            noise = self.rng_optimization.standard_normal((self.ndim_problem,))
            d[index] = transform @ noise
            x[index] = mean + self.sigma * d[index]
        np.clip(x[:active], self.lower_boundary, self.upper_boundary, out=x[:active])
        d[:active] = (x[:active] - mean) / self.sigma
        y[:active] = self._evaluate_population(x[:active], args)
        if active < self.n_individuals:
            y[active:] = math.inf
        return x, y, d


class _BatchedSEPCMAES(_BatchedEvaluation, SEPCMAES):
    def iterate(self, z=None, x=None, mean=None, d=None, y=None, args=None):
        active = min(
            self.n_individuals,
            int(self.max_function_evaluations - self.n_function_evaluations),
        )
        z[:active] = self.rng_optimization.standard_normal((active, self.ndim_problem))
        x[:active] = mean + self.sigma * d * z[:active]
        np.clip(x[:active], self.lower_boundary, self.upper_boundary, out=x[:active])
        z[:active] = (x[:active] - mean) / (self.sigma * d)
        y[:active] = self._evaluate_population(x[:active], args)
        if active < self.n_individuals:
            y[active:] = math.inf
        return z, x, y


class _BatchedMMES(_BatchedEvaluation, MMES):
    def iterate(self, x=None, mean=None, q=None, v=None, y=None, args=None):
        for index in range(self._n_mirror_sampling):
            mixed_direction = np.zeros((self.ndim_problem,))
            for _ in range(self.ms):
                direction = v[
                    (self.m - self.rng_optimization.geometric(self.c_a) % self.m) - 1
                ]
                mixed_direction += self.rng_optimization.standard_normal() * q[direction]
            step = self._z_1 * self.rng_optimization.standard_normal((self.ndim_problem,))
            step += self._z_2 * mixed_direction
            x[index] = mean + self.sigma * step
            mirror_index = self._n_mirror_sampling + index
            if mirror_index < self.n_individuals:
                x[mirror_index] = mean - self.sigma * step
        active = min(
            self.n_individuals,
            int(self.max_function_evaluations - self.n_function_evaluations),
        )
        np.clip(x[:active], self.lower_boundary, self.upper_boundary, out=x[:active])
        y[:active] = self._evaluate_population(x[:active], args)
        if active < self.n_individuals:
            y[active:] = math.inf
        return x, y


_OPTIMIZERS = {
    "cmaes": _BatchedCMAES,
    "mmes": _BatchedMMES,
    "sepcmaes": _BatchedSEPCMAES,
}


@dataclass(frozen=True)
class OptimizationRun:
    algorithm: str
    consumed_fes: int
    best_x: tuple[float, ...]
    best_error: float
    package: str
    package_version: str


class PypopOptimizerPort:
    """Execute one upstream optimizer without leaking package-specific state."""

    package = "pypop7"

    def __init__(self) -> None:
        self.package_version = importlib.metadata.version(self.package)

    def run(
        self,
        algorithm: str,
        *,
        problem: OptimizationProblem,
        ledger: EvaluationLedger,
        initial_mean: tuple[float, ...] | np.ndarray,
        sigma: float,
        seed: int,
        budget_fes: int,
        population_size: int | None = None,
        restart: bool = False,
    ) -> OptimizationRun:
        if algorithm not in _OPTIMIZERS:
            raise ValueError(f"unsupported upstream optimizer: {algorithm}")
        if problem is not ledger.problem:
            raise ValueError("optimizer problem and ledger problem must be identical")
        mean = np.asarray(initial_mean, dtype=float)
        if mean.shape != (problem.dimension,) or not np.all(np.isfinite(mean)):
            raise ValueError("initial_mean is invalid")
        step_size = float(sigma)
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(budget_fes, bool) or not 0 < budget_fes <= ledger.remaining:
            raise ValueError("optimizer budget is outside the remaining FE")
        if algorithm == "cmaes" and problem.dimension > 256:
            raise ValueError("full CMA-ES is disabled above 256 dimensions")
        if population_size is not None and population_size < 2:
            raise ValueError("population_size must be at least two")

        count_before = ledger.count
        upstream_problem = {
            "fitness_function": ledger.evaluate,
            "ndim_problem": problem.dimension,
            "lower_boundary": problem.lower_array,
            "upper_boundary": problem.upper_array,
        }
        options: dict[str, object] = {
            "max_function_evaluations": int(budget_fes),
            "mean": np.clip(mean, problem.lower_array, problem.upper_array),
            "sigma": step_size,
            "seed_rng": int(seed),
            "is_restart": bool(restart),
            "verbose": 0,
        }
        if population_size is not None:
            options["n_individuals"] = int(population_size)
        optimizer = _OPTIMIZERS[algorithm](upstream_problem, options)
        optimizer.optimize()
        consumed = ledger.count - count_before
        if consumed != budget_fes:
            raise RuntimeError(
                f"{algorithm} consumed {consumed} FE instead of the required {budget_fes}"
            )
        return OptimizationRun(
            algorithm=algorithm,
            consumed_fes=consumed,
            best_x=tuple(float(value) for value in ledger.best_x),
            best_error=float(ledger.best_error),
            package=self.package,
            package_version=self.package_version,
        )


def _jsonable(value: object) -> object:
    """Convert numpy scalar/container state to JSON-compatible primitives."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class ResumableOptimizerSession:
    """Interruptible population optimizer backed by the pypop7 state formulas.

    A population is sampled once at its generation boundary and evaluated one
    candidate at a time.  Therefore a snapshot taken halfway through a
    population contains both the pending candidates and the distribution state;
    resuming it cannot redraw or re-charge an earlier candidate.
    """

    package = "pypop7"
    schema = "arac-resumable-optimizer-v2"

    def __init__(
        self,
        algorithm: str,
        *,
        problem: OptimizationProblem,
        ledger: EvaluationLedger,
        initial_mean: tuple[float, ...] | np.ndarray,
        sigma: float,
        seed: int,
        budget_fes: int,
        population_size: int,
        initial_consumed: int = 0,
        dimensions: tuple[int, ...] | None = None,
        anchor: tuple[float, ...] | np.ndarray | None = None,
    ) -> None:
        if algorithm not in _OPTIMIZERS:
            raise ValueError(f"unsupported upstream optimizer: {algorithm}")
        if problem is not ledger.problem:
            raise ValueError("optimizer problem and ledger problem must be identical")
        mean = np.asarray(initial_mean, dtype=float)
        if dimensions is None:
            active_dimensions = tuple(range(problem.dimension))
        else:
            active_dimensions = tuple(int(value) for value in dimensions)
            if (
                not active_dimensions
                or len(set(active_dimensions)) != len(active_dimensions)
                or min(active_dimensions) < 0
                or max(active_dimensions) >= problem.dimension
            ):
                raise ValueError("dimensions must be a non-empty problem coordinate subset")
        if mean.shape != (len(active_dimensions),) or not np.all(np.isfinite(mean)):
            raise ValueError("initial_mean is invalid")
        if anchor is None:
            full_anchor = np.asarray(problem.lower_array + problem.upper_array, dtype=float) / 2.0
        else:
            full_anchor = np.asarray(anchor, dtype=float)
        if full_anchor.shape != (problem.dimension,) or not np.all(np.isfinite(full_anchor)):
            raise ValueError("anchor is invalid")
        if np.any(full_anchor < problem.lower_array) or np.any(full_anchor > problem.upper_array):
            raise ValueError("anchor escaped the public bounds")
        step_size = float(sigma)
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
            raise ValueError("budget_fes must be positive")
        if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < 2:
            raise ValueError("population_size must be at least two")
        if isinstance(initial_consumed, bool) or not isinstance(initial_consumed, int):
            raise ValueError("initial_consumed must be a non-negative integer")
        if not 0 <= initial_consumed <= budget_fes:
            raise ValueError("initial_consumed is outside the optimizer budget")
        if ledger.count - initial_consumed < 0 or ledger.remaining < budget_fes - initial_consumed:
            raise ValueError("ledger position does not match optimizer progress")
        if algorithm == "cmaes" and len(active_dimensions) > 256:
            raise ValueError("full CMA-ES is disabled above 256 dimensions")

        self.algorithm = algorithm
        self.problem = problem
        self.ledger = ledger
        self.dimensions = active_dimensions
        self.anchor = full_anchor.copy()
        self.lower_boundary = problem.lower_array[np.asarray(active_dimensions, dtype=int)]
        self.upper_boundary = problem.upper_array[np.asarray(active_dimensions, dtype=int)]
        active_span = float(np.max(self.upper_boundary - self.lower_boundary))
        self.sigma_floor = np.finfo(float).eps * max(1.0, active_span)
        self.sigma_floor_repair_count = 0
        self.initial_mean = np.clip(mean, self.lower_boundary, self.upper_boundary)
        self.sigma = step_size
        self.seed = seed
        self.budget_fes = budget_fes
        self.population_size = population_size
        self.start_count = ledger.count - initial_consumed
        self.consumed_fes = initial_consumed
        upstream_problem = {
            "fitness_function": self._evaluate_optimizer_candidate,
            "ndim_problem": len(active_dimensions),
            "lower_boundary": self.lower_boundary,
            "upper_boundary": self.upper_boundary,
        }
        options: dict[str, object] = {
            "max_function_evaluations": budget_fes,
            "mean": self.initial_mean,
            "sigma": step_size,
            "seed_rng": seed,
            "n_individuals": population_size,
            "is_restart": False,
            "early_stopping_evaluations": np.inf,
            "fitness_threshold": -np.inf,
            "verbose": 0,
        }
        self.optimizer = _OPTIMIZERS[algorithm](upstream_problem, options)
        # Calling the concrete optimizer's ``optimize`` would consume the whole
        # budget.  The resumable session only needs the lifecycle timestamp.
        self.optimizer.start_time = time.time()
        self._initialize_state()
        self.optimizer.n_function_evaluations = initial_consumed
        self.optimizer.best_so_far_x = self.initial_mean.copy()
        self.optimizer.best_so_far_y = self.ledger.best_error
        self.optimizer._base_early_stopping = self.ledger.best_error
        self.optimizer._counter_early_stopping = 0
        self.population_index = 0
        self._previous_y = np.copy(self.y)

    def _enforce_sigma_floor(self) -> None:
        sigma = float(self.optimizer.sigma)
        if not math.isfinite(sigma) or sigma < 0.0:
            raise RuntimeError("optimizer sigma became non-finite or negative")
        if sigma < self.sigma_floor:
            self.optimizer.sigma = self.sigma_floor
            self.sigma_floor_repair_count += 1

    def _evaluate_optimizer_candidate(self, candidate: np.ndarray) -> float | np.ndarray:
        values = np.asarray(candidate, dtype=float)
        single = values.ndim == 1
        rows = values[np.newaxis, :] if single else values
        if rows.ndim != 2 or rows.shape[1] != len(self.dimensions):
            raise ValueError("optimizer candidate shape does not match its coordinate subset")
        full = np.repeat(self.anchor[np.newaxis, :], len(rows), axis=0)
        full[:, np.asarray(self.dimensions, dtype=int)] = rows
        result = self.ledger.evaluate(full[0] if single else full)
        return result

    def _initialize_state(self) -> None:
        if self.algorithm == "cmaes":
            self.x, self.mean, self.p_s, self.p_c, self.cm, self.e_ve, self.e_va, self.y, self.d = (
                self.optimizer.initialize()
            )
        elif self.algorithm == "sepcmaes":
            self.z, self.x, self.mean, self.s, self.p, self.c, self.d, self.y = (
                self.optimizer.initialize()
            )
        else:
            self.optimizer._n_mirror_sampling = int(np.ceil(self.population_size / 2))
            self.x = np.zeros((self.population_size, len(self.dimensions)))
            self.mean = self.initial_mean.copy()
            self.p = np.zeros((len(self.dimensions),))
            self.w = 0.0
            self.q = np.zeros((self.optimizer.m, len(self.dimensions)))
            self.t = np.zeros((self.optimizer.m,))
            self.v = np.arange(self.optimizer.m)
            self.y = np.full((self.population_size,), self.ledger.best_error, dtype=float)

    def _prepare_population(self) -> None:
        if self.population_index != 0:
            return
        self._enforce_sigma_floor()
        if self.algorithm == "cmaes":
            transform = self.e_ve @ np.diag(self.e_va)
            for index in range(self.population_size):
                noise = self.optimizer.rng_optimization.standard_normal((len(self.dimensions),))
                self.d[index] = transform @ noise
                self.x[index] = self.mean + self.optimizer.sigma * self.d[index]
            np.clip(self.x, self.lower_boundary, self.upper_boundary, out=self.x)
            self.d[:] = (self.x - self.mean) / self.optimizer.sigma
        elif self.algorithm == "sepcmaes":
            self.z[:] = self.optimizer.rng_optimization.standard_normal(
                (self.population_size, len(self.dimensions))
            )
            self.x[:] = self.mean + self.optimizer.sigma * self.d * self.z
            np.clip(self.x, self.lower_boundary, self.upper_boundary, out=self.x)
            self.z[:] = (self.x - self.mean) / (self.optimizer.sigma * self.d)
        else:
            self._previous_y = np.copy(self.y)
            for index in range(self.optimizer._n_mirror_sampling):
                mixed_direction = np.zeros((len(self.dimensions),))
                for _ in range(self.optimizer.ms):
                    direction = self.v[
                        (self.optimizer.m - self.optimizer.rng_optimization.geometric(self.optimizer.c_a) % self.optimizer.m) - 1
                    ]
                    mixed_direction += self.optimizer.rng_optimization.standard_normal() * self.q[direction]
                step = self.optimizer._z_1 * self.optimizer.rng_optimization.standard_normal(
                    (len(self.dimensions),)
                )
                step += self.optimizer._z_2 * mixed_direction
                self.x[index] = self.mean + self.optimizer.sigma * step
                mirror = self.optimizer._n_mirror_sampling + index
                if mirror < self.population_size:
                    self.x[mirror] = self.mean - self.optimizer.sigma * step
            np.clip(self.x, self.lower_boundary, self.upper_boundary, out=self.x)

    def _finish_population(self) -> None:
        self._enforce_sigma_floor()
        if self.algorithm == "cmaes":
            self.mean, self.p_s, self.p_c, self.cm, self.e_ve, self.e_va = self.optimizer.update_distribution(
                self.x, self.p_s, self.p_c, self.cm, self.e_ve, self.e_va, self.y, self.d
            )
        elif self.algorithm == "sepcmaes":
            self.mean, self.s, self.p, self.c, self.d = self.optimizer._update_distribution(
                self.z, self.x, self.s, self.p, self.c, self.d, self.y
            )
        else:
            self.mean, self.p, self.w, self.q, self.t, self.v = self.optimizer._update_distribution(
                self.x,
                self.mean,
                self.p,
                self.w,
                self.q,
                self.t,
                self.v,
                self.y,
                self._previous_y,
            )
        self.optimizer._n_generations += 1
        self._enforce_sigma_floor()
        self.population_index = 0

    def step(self, budget_fes: int) -> tuple[float, ...]:
        if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
            raise ValueError("budget_fes must be positive")
        if budget_fes > self.budget_fes - self.consumed_fes:
            raise ValueError("step budget exceeds optimizer remaining budget")
        trace: list[float] = []
        before = self.ledger.count
        for _ in range(budget_fes):
            self._prepare_population()
            candidate = self.x[self.population_index]
            value = self.optimizer._evaluate_fitness(candidate)
            self.y[self.population_index] = value
            self.population_index += 1
            self.consumed_fes += 1
            trace.append(float(self.ledger.best_error))
            if self.population_index == self.population_size:
                self._finish_population()
        if self.ledger.count - before != budget_fes:
            raise RuntimeError("resumable optimizer drifted from its exact FE request")
        return tuple(trace)

    def state_dict(self) -> dict[str, object]:
        arrays = {
            "x": self.x,
            "mean": self.mean,
            "y": self.y,
        }
        if self.algorithm == "cmaes":
            arrays.update({"p_s": self.p_s, "p_c": self.p_c, "cm": self.cm, "e_ve": self.e_ve, "e_va": self.e_va, "d": self.d})
        elif self.algorithm == "sepcmaes":
            arrays.update({"z": self.z, "s": self.s, "p": self.p, "c": self.c, "d": self.d})
        else:
            arrays.update({"p": self.p, "q": self.q, "t": self.t, "v": self.v, "previous_y": self._previous_y})
        return {
            "schema": self.schema,
            "algorithm": self.algorithm,
            "budget_fes": self.budget_fes,
            "population_size": self.population_size,
            "dimensions": self.dimensions,
            "anchor": self.anchor,
            "initial_mean": self.initial_mean,
            "sigma": self.sigma,
            "sigma_floor": self.sigma_floor,
            "sigma_floor_repair_count": self.sigma_floor_repair_count,
            "seed": self.seed,
            "consumed_fes": self.consumed_fes,
            "population_index": self.population_index,
            "n_generations": self.optimizer._n_generations,
            "w": self.w if self.algorithm == "mmes" else None,
            "arrays": arrays,
            "optimizer": {
                "sigma": self.optimizer.sigma,
                "n_function_evaluations": self.optimizer.n_function_evaluations,
                "best_so_far_y": self.optimizer.best_so_far_y,
                "best_so_far_x": self.optimizer.best_so_far_x,
                "base_early_stopping": self.optimizer._base_early_stopping,
                "counter_early_stopping": self.optimizer._counter_early_stopping,
                "rng": self.optimizer.rng.bit_generator.state,
                "rng_initialization": self.optimizer.rng_initialization.bit_generator.state,
                "rng_optimization": self.optimizer.rng_optimization.bit_generator.state,
            },
        }

    def restore_state_dict(self, payload: dict[str, object]) -> None:
        if payload.get("schema") != self.schema or payload.get("algorithm") != self.algorithm:
            raise ValueError("resumable optimizer state schema or algorithm drifted")
        if payload.get("budget_fes") != self.budget_fes or payload.get("population_size") != self.population_size:
            raise ValueError("resumable optimizer state budget drifted")
        if tuple(payload.get("dimensions", ())) != self.dimensions:
            raise ValueError("resumable optimizer coordinate subset drifted")
        restored_anchor = np.asarray(payload.get("anchor"), dtype=float)
        if restored_anchor.shape != self.anchor.shape or not np.array_equal(restored_anchor, self.anchor):
            raise ValueError("resumable optimizer anchor drifted")
        if payload.get("consumed_fes") != self.consumed_fes:
            raise ValueError("resumable optimizer state FE position drifted")
        if payload.get("sigma_floor") != self.sigma_floor:
            raise ValueError("resumable optimizer sigma floor drifted")
        repair_count = payload.get("sigma_floor_repair_count")
        if isinstance(repair_count, bool) or not isinstance(repair_count, int) or repair_count < 0:
            raise ValueError("resumable optimizer sigma repair count is invalid")
        self.sigma_floor_repair_count = repair_count
        arrays = payload.get("arrays")
        if not isinstance(arrays, dict):
            raise ValueError("resumable optimizer arrays are missing")
        for name, value in arrays.items():
            current = getattr(self, name if name != "previous_y" else "_previous_y", None)
            if current is None:
                raise ValueError(f"unknown resumable optimizer array: {name}")
            restored = np.asarray(value, dtype=float if name != "v" else int)
            if restored.shape != current.shape or not np.all(np.isfinite(restored)):
                raise ValueError(f"invalid resumable optimizer array: {name}")
            setattr(self, name if name != "previous_y" else "_previous_y", restored)
        self.population_index = int(payload.get("population_index", -1))
        if not 0 <= self.population_index < self.population_size:
            raise ValueError("invalid resumable optimizer population position")
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ValueError("resumable optimizer metadata is missing")
        if self.algorithm == "mmes":
            self.w = float(payload.get("w"))
        self.optimizer.sigma = float(optimizer_state["sigma"])
        self.optimizer.n_function_evaluations = int(optimizer_state["n_function_evaluations"])
        self.optimizer.best_so_far_y = float(optimizer_state["best_so_far_y"])
        self.optimizer.best_so_far_x = np.asarray(optimizer_state["best_so_far_x"], dtype=float)
        self.optimizer._base_early_stopping = float(optimizer_state["base_early_stopping"])
        self.optimizer._counter_early_stopping = int(optimizer_state["counter_early_stopping"])
        self.optimizer._n_generations = int(payload.get("n_generations", -1))
        if self.optimizer.n_function_evaluations != self.consumed_fes:
            raise ValueError("optimizer and ledger FE positions disagree")
        if not math.isfinite(self.optimizer.sigma) or self.optimizer.sigma < self.sigma_floor:
            raise ValueError("restored optimizer sigma is outside its numerical floor")
        self.optimizer.rng.bit_generator.state = optimizer_state["rng"]
        self.optimizer.rng_initialization.bit_generator.state = optimizer_state["rng_initialization"]
        self.optimizer.rng_optimization.bit_generator.state = optimizer_state["rng_optimization"]

    def json_payload(self) -> bytes:
        import json

        return json.dumps(_jsonable(self.state_dict()), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def restore_json_payload(self, payload: bytes) -> None:
        import json

        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("resumable optimizer payload is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("resumable optimizer payload must be an object")
        self.restore_state_dict(decoded)

    @property
    def package_version(self) -> str:
        return importlib.metadata.version(self.package)


__all__ = ["OptimizationRun", "PypopOptimizerPort", "ResumableOptimizerSession"]
