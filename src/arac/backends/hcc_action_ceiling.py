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
    RelationCredit,
    actionability_delta,
)
from arac.policy.evidence_overlay import (
    RelationKey,
    UTILITY_EPSILON,
    runtime_probe_anchor_hash,
)


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
    owner_group_dimensions: tuple[tuple[int, ...], tuple[int, ...]]
    owner_optimizer_means: tuple[tuple[float, ...], tuple[float, ...]]
    left_background: tuple[float, ...] | None = None
    right_background: tuple[float, ...] | None = None
    relation_credit: RelationCredit | None = None

    def __post_init__(self) -> None:
        if self.arm not in ACTION_CEILING_ARMS:
            raise ValueError("unsupported action-ceiling arm")
        if len(self.context_hash) != 64:
            raise ValueError("context_hash must be SHA-256")
        incumbent = tuple(float(value) for value in self.incumbent)
        if not incumbent or not all(math.isfinite(value) for value in incumbent):
            raise ValueError("incumbent must be finite and non-empty")
        object.__setattr__(self, "incumbent", incumbent)
        shared_count = len(self.action_set.relation.shared_variable_indices)
        if len(self.previous_values) != shared_count:
            raise ValueError("previous relation values do not match action set")
        if len(self.current_values) != shared_count:
            raise ValueError("current relation values do not match action set")
        if not math.isfinite(float(self.incumbent_fitness)):
            raise ValueError("incumbent_fitness must be finite")
        owner_dimensions = tuple(
            tuple(int(value) for value in dimensions)
            for dimensions in self.owner_group_dimensions
        )
        if len(owner_dimensions) != 2:
            raise ValueError("action execution requires two owner group dimensions")
        shared_indices = set(self.action_set.relation.shared_variable_indices)
        for dimensions in owner_dimensions:
            if not dimensions or len(set(dimensions)) != len(dimensions):
                raise ValueError("owner group dimensions must be non-empty and unique")
            if any(value < 0 or value >= len(incumbent) for value in dimensions):
                raise ValueError("owner group dimension is outside incumbent")
            if not shared_indices.issubset(dimensions):
                raise ValueError("owner group does not contain every shared variable")
        object.__setattr__(self, "owner_group_dimensions", owner_dimensions)

        owner_means = tuple(
            tuple(float(value) for value in mean)
            for mean in self.owner_optimizer_means
        )
        if len(owner_means) != 2:
            raise ValueError("action execution requires two owner optimizer means")
        for mean, dimensions in zip(owner_means, owner_dimensions, strict=True):
            if len(mean) != len(dimensions) or not all(math.isfinite(value) for value in mean):
                raise ValueError("owner optimizer mean must be finite and match its group")
        object.__setattr__(self, "owner_optimizer_means", owner_means)
        if self.left_background is not None:
            lb = tuple(float(v) for v in self.left_background)
            if len(lb) != len(incumbent) or not all(math.isfinite(v) for v in lb):
                raise ValueError("left_background must be finite and match incumbent size")
            object.__setattr__(self, "left_background", lb)
        if self.right_background is not None:
            rb = tuple(float(v) for v in self.right_background)
            if len(rb) != len(incumbent) or not all(math.isfinite(v) for v in rb):
                raise ValueError("right_background must be finite and match incumbent size")
            object.__setattr__(self, "right_background", rb)


@dataclass(frozen=True)
class ActionExecutionResult:
    arm: str
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    extra_fes: int
    counterfactual_applied: bool
    mutation_norm: float
    optimizer_mean_mutation_norm: float
    applied_values_hash: str
    selected_candidate: str
    owner_optimizer_means: tuple[tuple[float, ...], tuple[float, ...]]


def _synchronize_owner_optimizer_means(
    *,
    owner_group_dimensions: tuple[tuple[int, ...], tuple[int, ...]],
    owner_optimizer_means: tuple[tuple[float, ...], tuple[float, ...]],
    shared_indices: Sequence[int],
    shared_values: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Copy one shared block into both owner groups' local mean vectors."""

    shared = tuple(float(value) for value in shared_values)
    if len(shared) != len(shared_indices) or not all(math.isfinite(value) for value in shared):
        raise ValueError("shared writeback values must be finite and aligned")
    synchronized: list[tuple[float, ...]] = []
    for dimensions, mean in zip(
        owner_group_dimensions,
        owner_optimizer_means,
        strict=True,
    ):
        local_positions = {dimension: index for index, dimension in enumerate(dimensions)}
        updated = list(mean)
        for shared_index, shared_value in zip(shared_indices, shared, strict=True):
            updated[local_positions[int(shared_index)]] = shared_value
        synchronized.append(tuple(updated))
    return synchronized[0], synchronized[1]


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
) -> ActionExecutionResult:
    incumbent = np.asarray(request.incumbent, dtype=float).copy()
    original = incumbent.copy()
    indices = np.asarray(request.action_set.relation.shared_variable_indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= incumbent.size):
        raise ValueError("relation shared index is outside incumbent")
    extra_fes = 0
    selected_candidate = request.arm
    incumbent_fitness = float(request.incumbent_fitness)
    owner_optimizer_means = request.owner_optimizer_means
    original_owner_optimizer_means = request.owner_optimizer_means

    def write_shared_values(
        values: Sequence[float],
        *,
        synchronize_owner_means: bool = True,
    ) -> None:
        nonlocal owner_optimizer_means
        shared_values = np.asarray(values, dtype=float).reshape(-1)
        if shared_values.shape != indices.shape or not np.all(np.isfinite(shared_values)):
            raise ValueError("shared writeback values must be finite and aligned")
        incumbent[indices] = shared_values
        if synchronize_owner_means:
            owner_optimizer_means = _synchronize_owner_optimizer_means(
                owner_group_dimensions=request.owner_group_dimensions,
                owner_optimizer_means=owner_optimizer_means,
                shared_indices=indices,
                shared_values=shared_values,
            )

    if request.arm == "native_eq8":
        write_shared_values(
            native_eq8_values(
                request.previous_values,
                request.current_values,
                request.previous_delta,
                request.current_delta,
            ),
            synchronize_owner_means=False,
        )
    elif request.arm == "true_no_writeback":
        selected_candidate = "current"
    elif request.arm in {"exact_left", "exact_right", "exact_bridge"}:
        write_shared_values(
            request.action_set.candidate_for_arm(request.arm).shared_values
        )
    elif request.arm == "multi_context_winner":
        if request.left_background is None or request.right_background is None:
            raise ValueError("multi_context_winner requires left_background and right_background")
        left_bg = np.asarray(request.left_background, dtype=float).copy()
        right_bg = np.asarray(request.right_background, dtype=float).copy()
        eq8 = native_eq8_values(
            request.previous_values, request.current_values,
            request.previous_delta, request.current_delta,
        )
        candidates = [
            ("current", tuple(float(v) for v in incumbent[indices])),
            ("exact_left", request.action_set.left_owner.shared_values),
            ("exact_right", request.action_set.right_owner.shared_values),
            ("exact_eq8", tuple(float(v) for v in eq8)),
        ]
        n = len(candidates)
        left_batch = np.repeat(left_bg[None, :], n, axis=0)
        right_batch = np.repeat(right_bg[None, :], n, axis=0)
        for i, (_, vals) in enumerate(candidates):
            left_batch[i, indices] = vals
            right_batch[i, indices] = vals
        f_left = np.asarray(evaluate(left_batch), dtype=float).reshape(-1)
        f_right = np.asarray(evaluate(right_batch), dtype=float).reshape(-1)
        if f_left.shape != (n,) or not np.all(np.isfinite(f_left)):
            raise ValueError("multi_context left evaluator must return 4 finite values")
        if f_right.shape != (n,) or not np.all(np.isfinite(f_right)):
            raise ValueError("multi_context right evaluator must return 4 finite values")
        extra_fes = 2 * n
        eps = UTILITY_EPSILON
        f_L0, f_R0 = float(f_left[0]), float(f_right[0])
        scores = [
            min(
                math.log((f_L0 + eps) / (float(f_left[i]) + eps)),
                math.log((f_R0 + eps) / (float(f_right[i]) + eps)),
            )
            for i in range(n)
        ]
        best_i = int(np.argmax(scores))
        best_score = scores[best_i]
        _catastrophic = math.log(1.20)
        loss_L = math.log((f_L0 + eps) / (float(f_left[best_i]) + eps))
        loss_R = math.log((f_R0 + eps) / (float(f_right[best_i]) + eps))
        if best_score > 0 and loss_L > -_catastrophic and loss_R > -_catastrophic:
            write_shared_values(candidates[best_i][1])
            selected_candidate = candidates[best_i][0]
        else:
            selected_candidate = "current"
    elif request.arm == "initialization_bias":
        winner = request.action_set.selector_winner
        winner_values = {
            "left_owner": request.action_set.left_owner.shared_values,
            "right_owner": request.action_set.right_owner.shared_values,
            "bridge": request.action_set.bridge.shared_values,
        }.get(winner)
        if winner_values is None:
            selected_candidate = "bias_none"
        else:
            owner_optimizer_means = _synchronize_owner_optimizer_means(
                owner_group_dimensions=request.owner_group_dimensions,
                owner_optimizer_means=owner_optimizer_means,
                shared_indices=indices,
                shared_values=winner_values,
            )
            selected_candidate = f"bias_{winner}"
    elif request.arm == "delayed_sweep_reconciliation":
        credit = request.relation_credit
        if credit is None or not credit.is_warm:
            selected_candidate = "cold_start_no_writeback"
        elif credit.ewma_credit <= 0.0:
            selected_candidate = "credit_negative_no_writeback"
        else:
            _winner_values = {
                "left_owner": request.action_set.left_owner.shared_values,
                "right_owner": request.action_set.right_owner.shared_values,
                "bridge": request.action_set.bridge.shared_values,
            }.get(credit.last_winner)
            if _winner_values is None:
                selected_candidate = "credit_winner_invalid_no_writeback"
            else:
                write_shared_values(_winner_values)
                selected_candidate = f"delayed_{credit.last_winner}"
    else:
        raise ValueError("unsupported action-ceiling arm")

    mutation_norm = float(np.linalg.norm(incumbent - original))
    optimizer_mean_mutation_norm = float(
        np.linalg.norm(
            np.concatenate(
                [
                    np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
                    for current, previous in zip(
                        owner_optimizer_means,
                        original_owner_optimizer_means,
                        strict=True,
                    )
                ]
            )
        )
    )
    return ActionExecutionResult(
        arm=request.arm,
        incumbent=tuple(float(value) for value in incumbent),
        incumbent_fitness=incumbent_fitness,
        extra_fes=extra_fes,
        counterfactual_applied=(
            mutation_norm > 0.0 or optimizer_mean_mutation_norm > 0.0
        ),
        mutation_norm=mutation_norm,
        optimizer_mean_mutation_norm=optimizer_mean_mutation_norm,
        applied_values_hash=_vector_hash(incumbent),
        selected_candidate=selected_candidate,
        owner_optimizer_means=owner_optimizer_means,
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
        group_index: int,
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
    completed_group_deltas: tuple[float, ...]
    fitness_record: tuple[float, ...]


def _run_native_group_steps(
    state: NativeContinuationState,
    *,
    evaluate: Callable[[np.ndarray], np.ndarray],
    fitness_record: list[float],
    optimize_group: GroupOptimizer,
    group_seed: Callable[[int, int], int],
    should_continue: Callable[[int], bool],
) -> ContinuationResult:
    incumbent = np.asarray(state.incumbent, dtype=float).copy()
    sweep_index = int(state.sweep_index)
    group_index = int(state.next_group_index)
    deltas = list(float(value) for value in state.completed_group_deltas)
    completed_steps = 0
    while should_continue(completed_steps):
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
            group_index=group_index,
            background=incumbent.copy(),
            dims=dims,
            requested_fes=state.optimizer_budgets[group_index],
            population_size=state.population_sizes[group_index],
            seed=int(group_seed(sweep_index, group_index)),
        )
        budget = state.optimizer_budgets[group_index]
        if result.actual_fes <= 0 or result.actual_fes > budget:
            raise ValueError("native group optimizer exceeded or skipped its frozen budget")
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
        completed_steps += 1

    return ContinuationResult(
        incumbent=tuple(float(value) for value in incumbent),
        sweep_index=sweep_index,
        next_group_index=group_index,
        completed_group_deltas=tuple(float(value) for value in deltas),
        fitness_record=tuple(float(value) for value in fitness_record),
    )


def run_native_group_cycle(
    state: NativeContinuationState,
    *,
    evaluate: Callable[[np.ndarray], np.ndarray],
    fitness_record: list[float],
    optimize_group: GroupOptimizer,
    group_seed: Callable[[int, int], int],
) -> ContinuationResult:
    """Run exactly one relation-to-same-relation native group cycle."""

    return _run_native_group_steps(
        state,
        evaluate=evaluate,
        fitness_record=fitness_record,
        optimize_group=optimize_group,
        group_seed=group_seed,
        should_continue=lambda completed: completed < len(state.group_dims),
    )


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
    return _run_native_group_steps(
        state,
        evaluate=evaluate,
        fitness_record=fitness_record,
        optimize_group=optimize_group,
        group_seed=group_seed,
        should_continue=lambda _completed: len(fitness_record) < target_relative_fe,
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
