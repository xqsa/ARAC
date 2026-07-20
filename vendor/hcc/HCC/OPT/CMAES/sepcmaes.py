from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from HCC.OPT.CMAES.es import ES


_STATE_SCHEMA = "sep_cma_es_state_v2"
_PARAMETER_SCHEMA = "sep_cma_es_parameters_v1"
CANONICAL_PARAMETERIZATION = "ros_hansen_2008_pypop7"
CANONICAL_REFERENCE_VERSION = (
    "pypop7-sepcmaes@67b29061d121cba9a5715897a2eb5d409df04c2d"
)


def _readonly_vector(value: Any, dimension: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    vector = np.array(vector, dtype=float, copy=True)
    vector.setflags(write=False)
    return vector


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


@dataclass(frozen=True, slots=True)
class SepCMAESParameters:
    parameterization: str
    reference_version: str
    dimension: int
    population_size: int
    parent_count: int
    recombination_weights: tuple[float, ...]
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_cov: float
    separable_covariance_multiplier: float
    rank_one_rate: float
    rank_mu_rate: float
    chi_n: float

    def to_audit_payload(self) -> dict[str, Any]:
        return {
            "schema": _PARAMETER_SCHEMA,
            "parameterization": self.parameterization,
            "reference_version": self.reference_version,
            "dimension": self.dimension,
            "population_size": self.population_size,
            "parent_count": self.parent_count,
            "recombination_weights": list(self.recombination_weights),
            "mu_eff": self.mu_eff,
            "c_sigma": self.c_sigma,
            "d_sigma": self.d_sigma,
            "c_c": self.c_c,
            "c_cov": self.c_cov,
            "separable_covariance_multiplier": (
                self.separable_covariance_multiplier
            ),
            "rank_one_rate": self.rank_one_rate,
            "rank_mu_rate": self.rank_mu_rate,
            "chi_n": self.chi_n,
        }

    @property
    def parameter_hash(self) -> str:
        return _stable_hash(self.to_audit_payload())


@dataclass(frozen=True, slots=True, eq=False)
class SepCMAESState:
    dimension: int
    population_size: int
    mean: np.ndarray
    sigma: float
    path_sigma: np.ndarray
    path_covariance: np.ndarray
    variances: np.ndarray
    generation: int
    n_function_evaluations: int
    best_so_far_x: np.ndarray | None
    best_so_far_y: float | None
    rng_state_json: str
    parameter_hash: str
    terminal_partial_population_evaluations: int = 0
    early_stopping_counter: int = 0
    early_stopping_base: float | None = None

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.generation < 0 or self.n_function_evaluations < 0:
            raise ValueError("generation and n_function_evaluations must be non-negative")
        if self.early_stopping_counter < 0:
            raise ValueError("early_stopping_counter must be non-negative")
        if (
            self.terminal_partial_population_evaluations < 0
            or self.terminal_partial_population_evaluations >= self.population_size
        ):
            raise ValueError(
                "terminal_partial_population_evaluations must be in "
                "[0, population_size)"
            )
        if (
            self.terminal_partial_population_evaluations
            > self.n_function_evaluations
        ):
            raise ValueError(
                "terminal partial evaluations cannot exceed total evaluations"
            )
        if (
            len(self.parameter_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.parameter_hash)
        ):
            raise ValueError("parameter_hash must be a lowercase SHA-256 hash")
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")

        object.__setattr__(
            self,
            "mean",
            _readonly_vector(self.mean, self.dimension, "mean"),
        )
        object.__setattr__(
            self,
            "path_sigma",
            _readonly_vector(self.path_sigma, self.dimension, "path_sigma"),
        )
        object.__setattr__(
            self,
            "path_covariance",
            _readonly_vector(
                self.path_covariance,
                self.dimension,
                "path_covariance",
            ),
        )
        variances = _readonly_vector(self.variances, self.dimension, "variances")
        if np.any(variances <= 0.0):
            raise ValueError("variances must be positive")
        object.__setattr__(self, "variances", variances)

        if self.best_so_far_x is None:
            if self.best_so_far_y is not None:
                raise ValueError("best_so_far_y requires best_so_far_x")
        else:
            object.__setattr__(
                self,
                "best_so_far_x",
                _readonly_vector(
                    self.best_so_far_x,
                    self.dimension,
                    "best_so_far_x",
                ),
            )
            if self.best_so_far_y is None or not math.isfinite(self.best_so_far_y):
                raise ValueError("best_so_far_y must be finite when a best point exists")

        if self.early_stopping_base is not None and not math.isfinite(
            self.early_stopping_base
        ):
            raise ValueError("early_stopping_base must be finite or None")
        try:
            rng_state = json.loads(self.rng_state_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("rng_state_json must contain valid JSON") from exc
        if not isinstance(rng_state, dict):
            raise ValueError("rng_state_json must encode an RNG state mapping")

    def to_audit_payload(self) -> dict[str, Any]:
        return {
            "schema": _STATE_SCHEMA,
            "dimension": self.dimension,
            "population_size": self.population_size,
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
            "path_sigma": self.path_sigma.tolist(),
            "path_covariance": self.path_covariance.tolist(),
            "variances": self.variances.tolist(),
            "generation": self.generation,
            "n_function_evaluations": self.n_function_evaluations,
            "best_so_far_x": (
                None if self.best_so_far_x is None else self.best_so_far_x.tolist()
            ),
            "best_so_far_y": self.best_so_far_y,
            "rng_state_json": self.rng_state_json,
            "parameter_hash": self.parameter_hash,
            "terminal_partial_population_evaluations": (
                self.terminal_partial_population_evaluations
            ),
            "early_stopping_counter": self.early_stopping_counter,
            "early_stopping_base": self.early_stopping_base,
        }

    @property
    def state_hash(self) -> str:
        return _stable_hash(self.to_audit_payload())

    def clone(self) -> SepCMAESState:
        return SepCMAESState(
            dimension=self.dimension,
            population_size=self.population_size,
            mean=self.mean,
            sigma=self.sigma,
            path_sigma=self.path_sigma,
            path_covariance=self.path_covariance,
            variances=self.variances,
            generation=self.generation,
            n_function_evaluations=self.n_function_evaluations,
            best_so_far_x=self.best_so_far_x,
            best_so_far_y=self.best_so_far_y,
            rng_state_json=self.rng_state_json,
            parameter_hash=self.parameter_hash,
            terminal_partial_population_evaluations=(
                self.terminal_partial_population_evaluations
            ),
            early_stopping_counter=self.early_stopping_counter,
            early_stopping_base=self.early_stopping_base,
        )


def canonical_sep_cma_parameters(
    dimension: int,
    population_size: int | None = None,
    parent_count: int | None = None,
) -> SepCMAESParameters:
    """Return the exact PyPop7 Sep-CMA-ES parameter snapshot."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if population_size is None:
        population_size = 4 + int(3.0 * math.log(dimension))
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size < 2
    ):
        raise ValueError("population_size must be an integer >= 2")
    if parent_count is None:
        parent_count = population_size // 2
    if (
        isinstance(parent_count, bool)
        or not isinstance(parent_count, int)
        or parent_count <= 0
        or parent_count > population_size
    ):
        raise ValueError("parent_count must be in [1, population_size]")

    weight_base = math.log((population_size + 1.0) / 2.0)
    raw_weights = np.asarray(
        [weight_base - math.log(index + 1.0) for index in range(parent_count)],
        dtype=float,
    )
    if np.any(raw_weights <= 0.0):
        raise ValueError("parent_count yields non-positive recombination weights")
    weights = raw_weights / math.fsum(float(value) for value in raw_weights)
    mu_eff = 1.0 / float(np.sum(np.square(weights)))

    c_sigma = (mu_eff + 2.0) / (dimension + mu_eff + 3.0)
    d_sigma = 1.0 + c_sigma + 2.0 * math.sqrt(
        max((mu_eff - 1.0) / (dimension + 1.0) - 1.0, 0.0)
    )
    c_c = 4.0 / (dimension + 4.0)
    base_c_cov = (1.0 / mu_eff) * 2.0 / (dimension + math.sqrt(2.0)) ** 2
    base_c_cov += (1.0 - 1.0 / mu_eff) * min(
        1.0,
        (2.0 * mu_eff - 1.0) / ((dimension + 2.0) ** 2 + mu_eff),
    )
    separable_multiplier = (dimension + 2.0) / 3.0
    c_cov = base_c_cov * separable_multiplier
    chi_n = math.sqrt(dimension) * (
        1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension**2)
    )
    return SepCMAESParameters(
        parameterization=CANONICAL_PARAMETERIZATION,
        reference_version=CANONICAL_REFERENCE_VERSION,
        dimension=dimension,
        population_size=population_size,
        parent_count=parent_count,
        recombination_weights=tuple(float(value) for value in weights),
        mu_eff=mu_eff,
        c_sigma=c_sigma,
        d_sigma=d_sigma,
        c_c=c_c,
        c_cov=c_cov,
        separable_covariance_multiplier=separable_multiplier,
        rank_one_rate=c_cov / mu_eff,
        rank_mu_rate=c_cov * (1.0 - 1.0 / mu_eff),
        chi_n=chi_n,
    )


class SEPCMAES(ES):
    """Canonical separable CMA-ES with persistent linear-size state.

    The implementation follows PyPop7's Ros-Hansen SEPCMAES, including its
    combined covariance learning rate and ``(N + 2) / 3`` acceleration. Only
    positive recombination weights are used.

    ``advance`` consumes an exact incremental FE budget. If the budget ends in
    the middle of a population, those candidates update best-so-far and FE
    accounting, but the incomplete population does not update the distribution.
    Such a snapshot is terminal: continuing it would silently discard evaluated
    offspring and therefore raises instead of producing a different trajectory.
    """

    def __init__(self, problem: dict[str, Any], options: dict[str, Any]):
        normalized_options = dict(options)
        normalized_options.setdefault("is_restart", False)
        ES.__init__(self, problem, normalized_options)
        self.options = normalized_options

        if self.n_individuals < 2:
            raise ValueError("SEPCMAES requires at least two offspring")
        if self.is_restart:
            raise ValueError("SEPCMAES does not perform implicit restarts")
        if self.saving_fitness:
            raise ValueError("saving_fitness is incompatible with O(N) persistent state")

        if self._w is None:
            self._w, self._mu_eff = self._compute_weights()
        self._e_chi = math.sqrt(self.ndim_problem) * (
            1.0
            - 1.0 / (4.0 * self.ndim_problem)
            + 1.0 / (21.0 * self.ndim_problem**2)
        )
        self.parameters = canonical_sep_cma_parameters(
            self.ndim_problem,
            self.n_individuals,
            self.n_parents,
        )
        if not np.allclose(
            np.asarray(self.parameters.recombination_weights),
            np.asarray(self._w),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError("ES and canonical Sep-CMA recombination weights disagree")
        self.c_s = self.parameters.c_sigma
        self.d_sigma = self.parameters.d_sigma
        self.c_c = self.parameters.c_c
        self.c_cov = self.parameters.c_cov

        self._mean: np.ndarray | None = None
        self._path_sigma: np.ndarray | None = None
        self._path_covariance: np.ndarray | None = None
        self._variances: np.ndarray | None = None
        self._terminal_partial_population_evaluations = 0

        initial_best_x = normalized_options.get("best_so_far_x")
        if initial_best_x is not None:
            self.best_so_far_x = self._coerce_initial_vector(
                initial_best_x,
                "best_so_far_x",
            )
            if not math.isfinite(float(self.best_so_far_y)):
                raise ValueError("best_so_far_y must be finite with best_so_far_x")
        elif math.isfinite(float(self.best_so_far_y)):
            raise ValueError("best_so_far_x is required with finite best_so_far_y")

    def _coerce_initial_vector(self, value: Any, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape == (1, self.ndim_problem):
            vector = vector[0]
        if vector.shape != (self.ndim_problem,):
            raise ValueError(
                f"{name} must have shape ({self.ndim_problem},), got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return np.array(vector, dtype=float, copy=True)

    def initialize_state(self) -> SepCMAESState:
        if self._mean is not None:
            return self.snapshot_state()

        mean = self._coerce_initial_vector(self._initialize_mean(), "mean")
        self.mean = np.array(mean, copy=True)
        self._mean = mean
        self._path_sigma = np.zeros(self.ndim_problem, dtype=float)
        self._path_covariance = np.zeros(self.ndim_problem, dtype=float)
        self._variances = np.ones(self.ndim_problem, dtype=float)
        self._terminal_partial_population_evaluations = 0
        self._n_generations = 0
        self._list_initial_mean.append(np.array(mean, copy=True))
        return self.snapshot_state()

    def snapshot_state(self) -> SepCMAESState:
        if self._mean is None:
            raise RuntimeError("SEPCMAES state has not been initialized")
        assert self._path_sigma is not None
        assert self._path_covariance is not None
        assert self._variances is not None

        rng_state_json = json.dumps(
            _jsonable(self.rng_optimization.bit_generator.state),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        best_x = None
        best_y = None
        if self.best_so_far_x is not None and math.isfinite(float(self.best_so_far_y)):
            best_x = np.asarray(self.best_so_far_x, dtype=float)
            best_y = float(self.best_so_far_y)
        early_stopping_base = (
            float(self._base_early_stopping)
            if math.isfinite(float(self._base_early_stopping))
            else None
        )
        return SepCMAESState(
            dimension=self.ndim_problem,
            population_size=self.n_individuals,
            mean=self._mean,
            sigma=float(self.sigma),
            path_sigma=self._path_sigma,
            path_covariance=self._path_covariance,
            variances=self._variances,
            generation=self._n_generations,
            n_function_evaluations=self.n_function_evaluations,
            best_so_far_x=best_x,
            best_so_far_y=best_y,
            rng_state_json=rng_state_json,
            parameter_hash=self.parameters.parameter_hash,
            terminal_partial_population_evaluations=(
                self._terminal_partial_population_evaluations
            ),
            early_stopping_counter=self._counter_early_stopping,
            early_stopping_base=early_stopping_base,
        )

    def clone_state(self) -> SepCMAESState:
        return self.snapshot_state().clone()

    def restore_state(self, state: SepCMAESState) -> None:
        if not isinstance(state, SepCMAESState):
            raise TypeError("state must be a SepCMAESState")
        state = state.clone()
        if state.terminal_partial_population_evaluations:
            raise ValueError(
                "a terminal partial-population state cannot be restored for continuation"
            )
        if state.dimension != self.ndim_problem:
            raise ValueError(
                f"state dimension {state.dimension} != optimizer dimension {self.ndim_problem}"
            )
        if state.population_size != self.n_individuals:
            raise ValueError(
                "state population_size "
                f"{state.population_size} != optimizer population_size {self.n_individuals}"
            )
        if state.parameter_hash != self.parameters.parameter_hash:
            raise ValueError("state parameter_hash does not match optimizer parameters")
        if state.n_function_evaluations > self.max_function_evaluations:
            raise ValueError("state evaluations exceed optimizer max_function_evaluations")

        self._mean = np.array(state.mean, dtype=float, copy=True)
        self._path_sigma = np.array(state.path_sigma, dtype=float, copy=True)
        self._path_covariance = np.array(
            state.path_covariance,
            dtype=float,
            copy=True,
        )
        self._variances = np.array(state.variances, dtype=float, copy=True)
        self._terminal_partial_population_evaluations = 0
        self.sigma = float(state.sigma)
        self._n_generations = state.generation
        self.n_function_evaluations = state.n_function_evaluations
        self.best_so_far_x = (
            None
            if state.best_so_far_x is None
            else np.array(state.best_so_far_x, dtype=float, copy=True)
        )
        self.best_so_far_y = (
            np.inf if state.best_so_far_y is None else float(state.best_so_far_y)
        )
        self._counter_early_stopping = state.early_stopping_counter
        self._base_early_stopping = (
            np.inf
            if state.early_stopping_base is None
            else float(state.early_stopping_base)
        )
        try:
            self.rng_optimization.bit_generator.state = json.loads(state.rng_state_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("state RNG is incompatible with this optimizer") from exc
        if not self._list_initial_mean:
            self._list_initial_mean.append(np.array(self._mean, copy=True))
        self.termination_signal = self.Terminations.NO_TERMINATION

    def _sample_and_evaluate(
        self,
        candidate_count: int,
        args: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._mean is not None
        assert self._variances is not None
        standard_deviations = np.sqrt(self._variances)
        z = self.rng_optimization.standard_normal(
            (candidate_count, self.ndim_problem)
        )
        steps = z * standard_deviations
        candidates = self._mean + self.sigma * steps
        if not np.all(np.isfinite(candidates)):
            raise FloatingPointError("SEPCMAES sampled a non-finite candidate")

        raw_fitness = self._evaluate_fitness(candidates, args)
        fitness = np.asarray(raw_fitness, dtype=float)
        if fitness.ndim == 0 and candidate_count == 1:
            fitness = fitness.reshape(1)
        if fitness.shape != (candidate_count,):
            raise ValueError(
                "fitness_function must return one value per candidate; "
                f"expected ({candidate_count},), got {fitness.shape}"
            )
        if not np.all(np.isfinite(fitness)):
            raise ValueError("fitness_function returned a non-finite value")
        return candidates, z, fitness

    def _update_distribution(
        self,
        candidates: np.ndarray,
        z: np.ndarray,
        fitness: np.ndarray,
    ) -> None:
        assert self._path_sigma is not None
        assert self._path_covariance is not None
        assert self._variances is not None

        selected = np.argsort(fitness)[: self.n_parents]
        weights = np.asarray(self.parameters.recombination_weights)
        selected_z = z[selected]
        weighted_z = weights @ selected_z
        standard_deviations = np.sqrt(self._variances)
        weighted_step = standard_deviations * weighted_z
        new_mean = weights @ candidates[selected]

        self._path_sigma = (
            (1.0 - self.c_s) * self._path_sigma
            + math.sqrt(self.c_s * (2.0 - self.c_s) * self._mu_eff)
            * weighted_z
        )
        normalized_path = np.linalg.norm(self._path_sigma) / math.sqrt(
            1.0 - (1.0 - self.c_s) ** (2 * (self._n_generations + 1))
        )
        h_sigma = float(
            normalized_path
            < (1.4 + 2.0 / (self.ndim_problem + 1.0)) * self._e_chi
        )
        self._path_covariance = (
            (1.0 - self.c_c) * self._path_covariance
            + h_sigma
            * math.sqrt(self.c_c * (2.0 - self.c_c) * self._mu_eff)
            * weighted_step
        )

        selected_steps = selected_z * standard_deviations
        rank_mu = weights @ np.square(selected_steps)
        old_variances = self._variances
        self._variances = (
            (1.0 - self.c_cov) * old_variances
            + self.parameters.rank_one_rate * np.square(self._path_covariance)
            + self.parameters.rank_mu_rate * rank_mu
        )
        self._variances = np.maximum(self._variances, np.finfo(float).tiny)
        self.sigma *= math.exp(
            self.c_s
            / self.d_sigma
            * (np.linalg.norm(self._path_sigma) / self._e_chi - 1.0)
        )
        self._mean = np.asarray(new_mean, dtype=float)
        self._n_generations += 1
        if (
            not math.isfinite(float(self.sigma))
            or self.sigma <= 0.0
            or not np.all(np.isfinite(self._mean))
            or not np.all(np.isfinite(self._path_sigma))
            or not np.all(np.isfinite(self._path_covariance))
            or not np.all(np.isfinite(self._variances))
        ):
            raise FloatingPointError("SEPCMAES distribution became non-finite")

    def advance(
        self,
        evaluation_budget: int,
        *,
        state: SepCMAESState | None = None,
        args: Any = None,
    ) -> dict[str, Any]:
        if isinstance(evaluation_budget, bool) or not isinstance(
            evaluation_budget, (int, np.integer)
        ):
            raise TypeError("evaluation_budget must be an integer")
        if evaluation_budget <= 0:
            raise ValueError("evaluation_budget must be positive")
        if state is not None:
            self.restore_state(state)
        elif self._mean is None:
            self.initialize_state()
        elif self._terminal_partial_population_evaluations:
            raise RuntimeError(
                "cannot continue after a terminal partial-population evaluation"
            )
        if self.start_time is None:
            self.start_time = time.time()

        start_evaluations = self.n_function_evaluations
        remaining_global = self.max_function_evaluations - start_evaluations
        if remaining_global <= 0:
            return self._collect_result(0, 0)
        allowed = evaluation_budget
        if math.isfinite(float(remaining_global)):
            allowed = min(allowed, int(remaining_global))
        target_evaluations = start_evaluations + allowed
        partial_population_evaluations = 0

        while self.n_function_evaluations < target_evaluations:
            if self._check_terminations():
                break
            candidate_count = min(
                self.n_individuals,
                target_evaluations - self.n_function_evaluations,
            )
            candidates, z, fitness = self._sample_and_evaluate(candidate_count, args)
            self._print_verbose_info([], fitness)
            if candidate_count == self.n_individuals:
                self._update_distribution(candidates, z, fitness)
            else:
                partial_population_evaluations = candidate_count
                self._terminal_partial_population_evaluations = candidate_count

        self._check_terminations()
        return self._collect_result(
            self.n_function_evaluations - start_evaluations,
            partial_population_evaluations,
        )

    def _collect_result(
        self,
        advanced_evaluations: int,
        partial_population_evaluations: int,
    ) -> dict[str, Any]:
        state = self.snapshot_state()
        best_x = (
            None
            if self.best_so_far_x is None
            else np.array(self.best_so_far_x, dtype=float, copy=True)
        )
        success = best_x is not None and math.isfinite(float(self.best_so_far_y))
        if success and self.lower_boundary is not None and self.upper_boundary is not None:
            success = bool(
                np.all(np.asarray(self.lower_boundary) <= best_x)
                and np.all(best_x <= np.asarray(self.upper_boundary))
            )
        return {
            "best_so_far_x": best_x,
            "best_so_far_y": float(self.best_so_far_y),
            "n_function_evaluations": self.n_function_evaluations,
            "advanced_function_evaluations": advanced_evaluations,
            "runtime": time.time() - self.start_time,
            "termination_signal": self.termination_signal,
            "time_function_evaluations": self.time_function_evaluations,
            "fitness": None,
            "success": success,
            "mean": np.array(state.mean, copy=True),
            "sigma": state.sigma,
            "p_sigma": np.array(state.path_sigma, copy=True),
            "p_c": np.array(state.path_covariance, copy=True),
            "variances": np.array(state.variances, copy=True),
            "_n_generations": state.generation,
            "_n_restart": 0,
            "partial_population_evaluations": partial_population_evaluations,
            "optimizer_state": state,
            "optimizer_state_hash": state.state_hash,
            "parameter_snapshot": self.parameters,
            "parameter_hash": self.parameters.parameter_hash,
        }

    def optimize(
        self,
        fitness_function: Any = None,
        args: Any = None,
        *,
        state: SepCMAESState | None = None,
    ) -> dict[str, Any]:
        if fitness_function is not None:
            self.fitness_function = fitness_function
        if state is not None:
            self.restore_state(state)
        elif self._mean is None:
            self.initialize_state()
        if not math.isfinite(float(self.max_function_evaluations)):
            raise ValueError("SEPCMAES.optimize requires a finite FE budget")
        remaining = int(self.max_function_evaluations - self.n_function_evaluations)
        if remaining <= 0:
            if self.start_time is None:
                self.start_time = time.time()
            return self._collect_result(0, 0)
        return self.advance(remaining, args=args)


__all__ = [
    "CANONICAL_PARAMETERIZATION",
    "CANONICAL_REFERENCE_VERSION",
    "SEPCMAES",
    "SepCMAESParameters",
    "SepCMAESState",
    "canonical_sep_cma_parameters",
]
