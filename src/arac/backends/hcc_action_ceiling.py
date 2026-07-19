"""HCC branch primitives for the offline G1 action-ceiling audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_HORIZONS,
    RelationActionSet,
    actionability_delta,
)
from arac.policy.evidence_overlay import RelationKey, runtime_probe_anchor_hash


FULL_SPACE_DIMENSION = 1_000
SEARCH_GENERATIONS = 4
FULL_SPACE_POPULATION = 4 + math.floor(3 * math.log(FULL_SPACE_DIMENSION))
FULL_SPACE_RESCUE_FE = 1 + SEARCH_GENERATIONS * FULL_SPACE_POPULATION


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vector_hash(values: Sequence[float]) -> str:
    return _sha256([float(value) for value in values])


def shared_population_size(dimension: int) -> int:
    if isinstance(dimension, bool) or int(dimension) <= 0:
        raise ValueError("shared dimension must be positive")
    return 4 + 3 * math.ceil(math.log(int(dimension)))


def native_eq8_values(
    previous_values: Sequence[float],
    current_values: Sequence[float],
    previous_delta: float,
    current_delta: float,
) -> np.ndarray:
    previous = np.asarray(previous_values, dtype=float).reshape(-1)
    current = np.asarray(current_values, dtype=float).reshape(-1)
    if previous.shape != current.shape or previous.size == 0:
        raise ValueError("native Eq.8 values must be non-empty and aligned")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("native Eq.8 values must be finite")
    left_delta = float(previous_delta)
    right_delta = float(current_delta)
    if not math.isfinite(left_delta) or not math.isfinite(right_delta):
        raise ValueError("native Eq.8 deltas must be finite")
    denominator = left_delta + right_delta
    if denominator == 0.0:
        return (previous + current) / 2.0
    return (left_delta / denominator) * previous + (
        right_delta / denominator
    ) * current


def trust_region_bounds(
    action_set: RelationActionSet,
    *,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    domain_lower = float(lower)
    domain_upper = float(upper)
    if not math.isfinite(domain_lower) or not math.isfinite(domain_upper):
        raise ValueError("trust-region domain bounds must be finite")
    if domain_lower >= domain_upper:
        raise ValueError("trust-region upper bound must exceed lower bound")
    values = np.asarray(
        [
            action_set.anchor.shared_values,
            action_set.left_owner.shared_values,
            action_set.right_owner.shared_values,
            action_set.bridge.shared_values,
        ],
        dtype=float,
    )
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    span = maximum - minimum
    padding = np.maximum(0.5 * span, 0.01 * (domain_upper - domain_lower))
    return (
        np.maximum(domain_lower, minimum - padding),
        np.minimum(domain_upper, maximum + padding),
    )


@dataclass(frozen=True)
class OptimizationResult:
    best_x: tuple[float, ...]
    best_y: float
    actual_fes: int

    def __post_init__(self) -> None:
        if not self.best_x or not all(math.isfinite(float(value)) for value in self.best_x):
            raise ValueError("optimizer best_x must be finite and non-empty")
        if not math.isfinite(float(self.best_y)):
            raise ValueError("optimizer best_y must be finite")
        if isinstance(self.actual_fes, bool) or int(self.actual_fes) < 0:
            raise ValueError("optimizer actual_fes must be non-negative")


class SharedOptimizer(Protocol):
    def __call__(
        self,
        *,
        background: np.ndarray,
        shared_indices: tuple[int, ...],
        mean: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        requested_fes: int,
        population_size: int,
        seed: int,
    ) -> OptimizationResult: ...


class FullOptimizer(Protocol):
    def __call__(
        self,
        *,
        mean: np.ndarray,
        lower: float,
        upper: float,
        requested_fes: int,
        population_size: int,
        seed: int,
    ) -> OptimizationResult: ...


@dataclass(frozen=True)
class ActionExecutionRequest:
    arm: str
    context_hash: str
    action_set: RelationActionSet
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    previous_values: tuple[float, ...]
    current_values: tuple[float, ...]
    previous_delta: float
    current_delta: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.arm not in ACTION_CEILING_ARMS:
            raise ValueError("unsupported action-ceiling arm")
        if len(self.context_hash) != 64:
            raise ValueError("context_hash must be SHA-256")
        incumbent = tuple(float(value) for value in self.incumbent)
        if not incumbent or not all(math.isfinite(value) for value in incumbent):
            raise ValueError("incumbent must be finite and non-empty")
        object.__setattr__(self, "incumbent", incumbent)
        if len(self.previous_values) != len(self.action_set.relation.shared_variable_indices):
            raise ValueError("previous relation values do not match action set")
        if len(self.current_values) != len(self.action_set.relation.shared_variable_indices):
            raise ValueError("current relation values do not match action set")
        if not math.isfinite(float(self.incumbent_fitness)):
            raise ValueError("incumbent_fitness must be finite")


@dataclass(frozen=True)
class ActionExecutionResult:
    arm: str
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    extra_fes: int
    counterfactual_applied: bool
    mutation_norm: float
    applied_values_hash: str
    selected_candidate: str


def _action_seed(context_hash: str, arm: str) -> int:
    digest = hashlib.sha256(f"{context_hash}|{arm}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _best_probe_candidate(action_set: RelationActionSet):
    candidates = (
        action_set.anchor,
        action_set.left_owner,
        action_set.right_owner,
        action_set.bridge,
    )
    return min(candidates, key=lambda item: (item.fitness, candidates.index(item)))


def selector_arm_for_context(
    action_set: RelationActionSet,
    *,
    relation: RelationKey,
    current_sweep: int,
    checkpoint_hash: str,
    current_shared_values: Sequence[float],
) -> str:
    if relation != action_set.relation:
        return "true_no_writeback"
    if checkpoint_hash != action_set.checkpoint_hash:
        return "true_no_writeback"
    if current_sweep != action_set.target_sweep:
        return "true_no_writeback"
    if runtime_probe_anchor_hash(relation, current_shared_values) != (
        action_set.anchor.shared_values_hash
    ):
        return "true_no_writeback"
    return {
        "left_owner": "exact_left",
        "right_owner": "exact_right",
        "bridge": "exact_bridge",
        "none": "true_no_writeback",
    }[action_set.selector_winner]


def execute_action_ceiling_arm(
    request: ActionExecutionRequest,
    *,
    evaluate: Callable[[np.ndarray], np.ndarray],
    shared_optimizer: SharedOptimizer | None = None,
    full_optimizer: FullOptimizer | None = None,
) -> ActionExecutionResult:
    incumbent = np.asarray(request.incumbent, dtype=float).copy()
    original = incumbent.copy()
    indices = np.asarray(request.action_set.relation.shared_variable_indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= incumbent.size):
        raise ValueError("relation shared index is outside incumbent")
    extra_fes = 0
    selected_candidate = request.arm
    incumbent_fitness = float(request.incumbent_fitness)

    if request.arm == "native_eq8":
        incumbent[indices] = native_eq8_values(
            request.previous_values,
            request.current_values,
            request.previous_delta,
            request.current_delta,
        )
    elif request.arm == "true_no_writeback":
        selected_candidate = "current"
    elif request.arm in {"exact_left", "exact_right", "exact_bridge"}:
        incumbent[indices] = request.action_set.candidate_for_arm(
            request.arm
        ).shared_values
    elif request.arm == "reprobe_then_exact":
        candidates = (
            ("current", tuple(float(value) for value in incumbent[indices])),
            ("exact_left", request.action_set.left_owner.shared_values),
            ("exact_right", request.action_set.right_owner.shared_values),
            ("exact_bridge", request.action_set.bridge.shared_values),
        )
        batch = np.repeat(incumbent[None, :], len(candidates), axis=0)
        for row, (_, values) in zip(batch, candidates, strict=True):
            row[indices] = values
        fitness = np.asarray(evaluate(batch), dtype=float).reshape(-1)
        if fitness.shape != (4,) or not np.all(np.isfinite(fitness)):
            raise ValueError("reprobe evaluator must return four finite values")
        extra_fes = 4
        selected_index = int(np.argmin(fitness))
        selected_candidate, selected_values = candidates[selected_index]
        incumbent[indices] = selected_values
        incumbent_fitness = float(fitness[selected_index])
    elif request.arm == "shared_trust_region":
        if shared_optimizer is None:
            raise ValueError("shared_trust_region requires a shared optimizer")
        lower, upper = trust_region_bounds(
            request.action_set,
            lower=request.lower,
            upper=request.upper,
        )
        mean = np.asarray(_best_probe_candidate(request.action_set).shared_values)
        population = shared_population_size(len(indices))
        requested_fes = SEARCH_GENERATIONS * population
        result = shared_optimizer(
            background=incumbent.copy(),
            shared_indices=tuple(int(value) for value in indices),
            mean=mean,
            lower=lower,
            upper=upper,
            requested_fes=requested_fes,
            population_size=population,
            seed=_action_seed(request.context_hash, request.arm),
        )
        if result.actual_fes != requested_fes:
            raise ValueError("shared optimizer did not consume four complete generations")
        extra_fes = result.actual_fes
        if result.best_y < incumbent_fitness:
            incumbent[indices] = result.best_x
            incumbent_fitness = result.best_y
        else:
            selected_candidate = "current"
    elif request.arm == "non_decomposition_rescue":
        if incumbent.size != FULL_SPACE_DIMENSION:
            raise ValueError("non-decomposition rescue requires a 1000D incumbent")
        if full_optimizer is None:
            raise ValueError("non_decomposition_rescue requires a full optimizer")
        result = full_optimizer(
            mean=incumbent.copy(),
            lower=request.lower,
            upper=request.upper,
            requested_fes=FULL_SPACE_RESCUE_FE,
            population_size=FULL_SPACE_POPULATION,
            seed=_action_seed(request.context_hash, request.arm),
        )
        if result.actual_fes != FULL_SPACE_RESCUE_FE:
            raise ValueError("full optimizer did not consume the frozen rescue budget")
        extra_fes = result.actual_fes
        if result.best_y < incumbent_fitness:
            if len(result.best_x) != FULL_SPACE_DIMENSION:
                raise ValueError("full optimizer returned a non-1000D candidate")
            incumbent[:] = result.best_x
            incumbent_fitness = result.best_y
        else:
            selected_candidate = "current"
    else:
        raise ValueError("unsupported action-ceiling arm")

    mutation_norm = float(np.linalg.norm(incumbent - original))
    return ActionExecutionResult(
        arm=request.arm,
        incumbent=tuple(float(value) for value in incumbent),
        incumbent_fitness=incumbent_fitness,
        extra_fes=extra_fes,
        counterfactual_applied=True,
        mutation_norm=mutation_norm,
        applied_values_hash=_vector_hash(incumbent),
        selected_candidate=selected_candidate,
    )


@dataclass(frozen=True)
class NativeContinuationState:
    incumbent: tuple[float, ...]
    sweep_index: int
    next_group_index: int
    completed_group_deltas: tuple[float, ...]
    group_dims: tuple[tuple[int, ...], ...]
    overlapping_elements: tuple[tuple[int, ...], ...]
    population_sizes: tuple[int, ...]
    optimizer_budgets: tuple[int, ...]

    def __post_init__(self) -> None:
        group_count = len(self.group_dims)
        if group_count == 0:
            raise ValueError("native continuation requires groups")
        if len(self.overlapping_elements) != group_count - 1:
            raise ValueError("native continuation overlap count is invalid")
        if len(self.population_sizes) != group_count:
            raise ValueError("native continuation population count is invalid")
        if len(self.optimizer_budgets) != group_count:
            raise ValueError("native continuation budget count is invalid")
        if not 0 <= self.next_group_index <= group_count:
            raise ValueError("next_group_index is outside continuation groups")
        if len(self.completed_group_deltas) != self.next_group_index:
            raise ValueError("completed deltas do not align with next group")

    @property
    def sweep_horizon_fe(self) -> int:
        return sum(1 + budget for budget in self.optimizer_budgets)


class GroupOptimizer(Protocol):
    def __call__(
        self,
        *,
        background: np.ndarray,
        dims: tuple[int, ...],
        requested_fes: int,
        population_size: int,
        seed: int,
    ) -> OptimizationResult: ...


@dataclass(frozen=True)
class ContinuationResult:
    incumbent: tuple[float, ...]
    sweep_index: int
    next_group_index: int
    fitness_record: tuple[float, ...]


def run_native_continuation(
    state: NativeContinuationState,
    *,
    evaluate: Callable[[np.ndarray], np.ndarray],
    fitness_record: list[float],
    optimize_group: GroupOptimizer,
    group_seed: Callable[[int, int], int],
    target_relative_fe: int,
) -> ContinuationResult:
    if target_relative_fe <= 0:
        raise ValueError("target_relative_fe must be positive")
    incumbent = np.asarray(state.incumbent, dtype=float).copy()
    sweep_index = int(state.sweep_index)
    group_index = int(state.next_group_index)
    deltas = list(float(value) for value in state.completed_group_deltas)
    while len(fitness_record) < target_relative_fe:
        if group_index == len(state.group_dims):
            sweep_index += 1
            group_index = 0
            deltas = []
        dims = state.group_dims[group_index]
        original = incumbent.copy()
        values = np.asarray(evaluate(incumbent), dtype=float).reshape(-1)
        if values.shape != (1,) or not np.isfinite(values[0]):
            raise ValueError("native precheck must return one finite value")
        original_fitness = float(values[0])
        result = optimize_group(
            background=incumbent.copy(),
            dims=dims,
            requested_fes=state.optimizer_budgets[group_index],
            population_size=state.population_sizes[group_index],
            seed=int(group_seed(sweep_index, group_index)),
        )
        if result.actual_fes != state.optimizer_budgets[group_index]:
            raise ValueError("native group optimizer did not consume its frozen budget")
        if len(result.best_x) != len(dims):
            raise ValueError("native group optimizer returned the wrong dimension")
        if result.best_y < original_fitness:
            incumbent[np.asarray(dims, dtype=int)] = result.best_x
            current_delta = original_fitness - result.best_y
        else:
            current_delta = 0.0
        deltas.append(current_delta)
        if group_index > 0:
            overlap = np.asarray(state.overlapping_elements[group_index - 1], dtype=int)
            if overlap.size:
                incumbent[overlap] = native_eq8_values(
                    original[overlap],
                    incumbent[overlap],
                    deltas[group_index - 1],
                    current_delta,
                )
        group_index += 1

    return ContinuationResult(
        incumbent=tuple(float(value) for value in incumbent),
        sweep_index=sweep_index,
        next_group_index=group_index,
        fitness_record=tuple(float(value) for value in fitness_record),
    )


def branch_horizon_errors(
    *,
    prefix_best_error: float,
    post_checkpoint_record: Sequence[float],
    sweep_horizon_fe: int,
) -> dict[str, float]:
    prefix_best = float(prefix_best_error)
    if not math.isfinite(prefix_best) or prefix_best < 0.0:
        raise ValueError("prefix best error must be finite and non-negative")
    record = tuple(float(value) for value in post_checkpoint_record)
    targets = {
        "immediate": 1,
        "sweep_1": int(sweep_horizon_fe),
        "sweep_3": 3 * int(sweep_horizon_fe),
    }
    if set(targets) != set(ACTION_CEILING_HORIZONS):
        raise RuntimeError("action-ceiling horizon contract drifted")
    if len(record) < max(targets.values()):
        raise ValueError("branch record does not reach the three-sweep horizon")
    return {
        label: min(prefix_best, min(record[:target]))
        for label, target in targets.items()
    }


def paired_arm_rows(
    native_errors: dict[str, float],
    arm_errors: dict[str, float],
) -> tuple[dict[str, float | str], ...]:
    if set(native_errors) != set(ACTION_CEILING_HORIZONS):
        raise ValueError("native errors do not cover frozen horizons")
    if set(arm_errors) != set(ACTION_CEILING_HORIZONS):
        raise ValueError("arm errors do not cover frozen horizons")
    return tuple(
        {
            "horizon": horizon,
            "native_error": float(native_errors[horizon]),
            "arm_error": float(arm_errors[horizon]),
            "delta": actionability_delta(
                native_errors[horizon],
                arm_errors[horizon],
            ),
        }
        for horizon in ACTION_CEILING_HORIZONS
    )
