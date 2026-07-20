"""HCC branch primitives for the offline G1 action-ceiling audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from arac.actions.full_space_sep_cma import FULL_SPACE_SEP_CMA_ACTION
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_HORIZONS,
    AUDITED_RELATION_WRITEBACK_ACTIONS,
    BUDGET_MAX_UNIFORM_MULTIPLIER,
    CONTRIBUTION_OWNER_REVERSE_WRITEBACK_ACTION,
    CONTRIBUTION_OWNER_WRITEBACK_ACTION,
    EFFICIENCY_EWMA_ALPHA,
    GUARDED_EQ8_PROBE_FES,
    GUARDED_EQ8_WRITEBACK_ACTION,
    RelationActionSet,
    STAGNATION_EPSILON,
    STAGNATION_GUARD_WRITEBACK_ACTION,
    STAGNATION_TRIGGER_STREAK,
    WARM_START_COOLDOWN_SWEEPS,
    actionability_delta,
    relation_writeback_action_parameters,
)
from arac.policy.evidence_overlay import (
    RelationKey,
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


@dataclass(frozen=True)
class ActionExecutionResult:
    arm: str
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    action_budget_fes: int
    action_actual_fes: int
    action_instance_hash: str
    action_lifecycle_payload: str
    action_lifecycle_hash: str
    action_accepted: bool
    action_candidate_hash: str
    action_candidate_fitness: float | None
    action_post_incumbent_hash: str
    optimizer_scope: str
    optimizer_parameter_hash: str
    optimizer_initial_state_hash: str
    optimizer_final_state_hash: str
    optimizer_population_size: int
    optimizer_generation_count: int
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


def _relation_writeback_instance_payload(
    request: ActionExecutionRequest,
    candidates: Sequence[tuple[str, Sequence[float]]],
) -> dict[str, object]:
    """Freeze the complete relation-writeback plan before branch execution."""

    parameters = relation_writeback_action_parameters(request.arm)
    relation = request.action_set.relation
    return {
        "arm": request.arm,
        "context_hash": request.context_hash,
        "action_set_hash": request.action_set.action_set_hash,
        "relation": {
            "owners": list(relation.owner_group_indices),
            "shared": list(relation.shared_variable_indices),
        },
        "dispatch_anchor_hash": runtime_probe_anchor_hash(
            relation,
            request.current_values,
        ),
        "previous_values_hash": _vector_hash(request.previous_values),
        "current_values_hash": _vector_hash(request.current_values),
        "previous_delta": float(request.previous_delta),
        "current_delta": float(request.current_delta),
        "parameters": parameters,
        "parameter_hash": _sha256(parameters),
        "action_budget_fes": int(parameters["probe_fes"]),
        "candidates": [
            {"name": name, "values_hash": _vector_hash(values)}
            for name, values in candidates
        ],
    }


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

    action_budget_fes = 0
    action_actual_fes = 0
    action_instance_hash = ""
    action_lifecycle_payload = ""
    action_lifecycle_hash = ""
    action_accepted = False
    action_candidate_hash = ""
    action_candidate_fitness: float | None = None
    action_post_incumbent_hash = ""
    writeback_instance_payload: dict[str, object] | None = None
    writeback_probe_outcomes: list[dict[str, object]] = []
    selected_values = tuple(float(value) for value in request.current_values)

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
    elif request.arm == GUARDED_EQ8_WRITEBACK_ACTION:
        # Evaluated-choice writeback: probe the frozen candidates inside the
        # same horizon, then write the argmin. Probe FEs are charged to the
        # branch record via `evaluate`, so the continuation stops earlier by
        # exactly the probe count and the absolute FE target is preserved.
        blend_values = native_eq8_values(
            request.previous_values,
            request.current_values,
            request.previous_delta,
            request.current_delta,
        )
        planned_candidates = (
            ("current", tuple(request.current_values), incumbent_fitness),
            ("previous", tuple(request.previous_values), None),
            ("eq8_blend", tuple(float(value) for value in blend_values), None),
        )
        writeback_instance_payload = _relation_writeback_instance_payload(
            request,
            tuple((name, values) for name, values, _ in planned_candidates),
        )
        action_instance_hash = _sha256(writeback_instance_payload)
        evaluated_candidates: list[tuple[str, tuple[float, ...], float]] = []
        for candidate_name, values, known_fitness in planned_candidates:
            fitness = known_fitness
            if fitness is None:
                probe = incumbent.copy()
                probe[indices] = np.asarray(values, dtype=float)
                probe_result = np.asarray(evaluate(probe), dtype=float).reshape(-1)
                if probe_result.shape != (1,) or not np.isfinite(probe_result[0]):
                    raise ValueError("guarded eq8 probe must return one finite value")
                fitness = float(probe_result[0])
                action_actual_fes += 1
            evaluated_candidates.append((candidate_name, values, fitness))
        if action_actual_fes != GUARDED_EQ8_PROBE_FES:
            raise RuntimeError("guarded eq8 writeback did not consume its probe budget")
        winner_index = 0
        for position, (_, _, fitness) in enumerate(evaluated_candidates):
            if fitness < evaluated_candidates[winner_index][2]:
                winner_index = position
        winner_name, winner_values, winner_fitness = evaluated_candidates[winner_index]
        selected_candidate = winner_name
        selected_values = winner_values
        if winner_name != "current":
            write_shared_values(winner_values)
        action_budget_fes = GUARDED_EQ8_PROBE_FES
        action_candidate_fitness = winner_fitness
        writeback_probe_outcomes = [
            {
                "name": candidate_name,
                "values_hash": _vector_hash(values),
                "fitness": fitness,
            }
            for candidate_name, values, fitness in evaluated_candidates
        ]
    elif request.arm == STAGNATION_GUARD_WRITEBACK_ACTION:
        blend_values = tuple(
            float(value)
            for value in native_eq8_values(
                request.previous_values,
                request.current_values,
                request.previous_delta,
                request.current_delta,
            )
        )
        writeback_instance_payload = _relation_writeback_instance_payload(
            request,
            (("current", request.current_values), ("native_eq8", blend_values)),
        )
        action_instance_hash = _sha256(writeback_instance_payload)
        if request.previous_delta == 0.0 and request.current_delta == 0.0:
            # Native Eq.8 would fall back to an unaudited arithmetic mean here;
            # the guard abstains instead of writing an unevaluated midpoint.
            selected_candidate = "current"
        else:
            selected_values = blend_values
            write_shared_values(blend_values, synchronize_owner_means=False)
            selected_candidate = "native_eq8"
    elif request.arm in {
        CONTRIBUTION_OWNER_WRITEBACK_ACTION,
        CONTRIBUTION_OWNER_REVERSE_WRITEBACK_ACTION,
    }:
        writeback_instance_payload = _relation_writeback_instance_payload(
            request,
            (
                ("current", request.current_values),
                ("left_owner", request.previous_values),
                ("right_owner", request.current_values),
            ),
        )
        action_instance_hash = _sha256(writeback_instance_payload)
        if request.current_delta == request.previous_delta:
            # Tie (including double stagnation): abstain, keep current values.
            selected_candidate = "current"
        else:
            take_current = (request.current_delta > request.previous_delta) == (
                request.arm == CONTRIBUTION_OWNER_WRITEBACK_ACTION
            )
            if take_current:
                winner_values: tuple[float, ...] = tuple(request.current_values)
                selected_candidate = "right_owner"
            else:
                winner_values = tuple(request.previous_values)
                selected_candidate = "left_owner"
            selected_values = winner_values
            write_shared_values(winner_values)
    elif request.arm in {
        "efficiency_budget_reallocation",
        "delta_priority_scan",
        "stagnation_cross_group_warm_start",
        FULL_SPACE_SEP_CMA_ACTION,
    }:
        # Keep the target dispatch identical to native; only continuation changes.
        write_shared_values(
            native_eq8_values(
                request.previous_values,
                request.current_values,
                request.previous_delta,
                request.current_delta,
            ),
            synchronize_owner_means=False,
        )
        selected_candidate = request.arm
    else:
        raise ValueError("unsupported action-ceiling arm")

    if request.arm in AUDITED_RELATION_WRITEBACK_ACTIONS:
        if writeback_instance_payload is None:
            raise RuntimeError("audited relation writeback action has no frozen instance")
        action_accepted = selected_candidate != "current"
        action_candidate_hash = _vector_hash(selected_values)
        action_post_incumbent_hash = _vector_hash(incumbent)
        lifecycle_payload: dict[str, object] = {
            "instance": writeback_instance_payload,
            "instance_hash": action_instance_hash,
            "action_actual_fes": action_actual_fes,
            "selected_candidate": selected_candidate,
            "selected_values_hash": action_candidate_hash,
            "post_incumbent_hash": action_post_incumbent_hash,
            "accepted": action_accepted,
        }
        if action_candidate_fitness is not None:
            lifecycle_payload["selected_fitness"] = action_candidate_fitness
        if writeback_probe_outcomes:
            lifecycle_payload["probe_outcomes"] = writeback_probe_outcomes
        action_lifecycle_payload = json.dumps(
            lifecycle_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        action_lifecycle_hash = _sha256(lifecycle_payload)

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
        action_budget_fes=action_budget_fes,
        action_actual_fes=action_actual_fes,
        action_instance_hash=action_instance_hash,
        action_lifecycle_payload=action_lifecycle_payload,
        action_lifecycle_hash=action_lifecycle_hash,
        action_accepted=action_accepted,
        action_candidate_hash=action_candidate_hash,
        action_candidate_fitness=action_candidate_fitness,
        action_post_incumbent_hash=action_post_incumbent_hash,
        optimizer_scope=(
            "decomposed_groups"
            if request.arm in {
                "efficiency_budget_reallocation",
                "delta_priority_scan",
                "stagnation_cross_group_warm_start",
                FULL_SPACE_SEP_CMA_ACTION,
            }
            else "relation_writeback"
        ),
        optimizer_parameter_hash="",
        optimizer_initial_state_hash="",
        optimizer_final_state_hash="",
        optimizer_population_size=0,
        optimizer_generation_count=0,
        counterfactual_applied=(
            mutation_norm > 0.0
            or optimizer_mean_mutation_norm > 0.0
            or action_actual_fes > 0
        ),
        mutation_norm=mutation_norm,
        optimizer_mean_mutation_norm=optimizer_mean_mutation_norm,
        applied_values_hash=_vector_hash(incumbent),
        selected_candidate=selected_candidate,
        owner_optimizer_means=owner_optimizer_means,
    )


def update_efficiency_ewma(
    previous: Sequence[float],
    deltas: Sequence[float],
    actual_fes: Sequence[int],
) -> tuple[float, ...]:
    if not (len(previous) == len(deltas) == len(actual_fes)) or not previous:
        raise ValueError("efficiency inputs must be non-empty and aligned")
    updated: list[float] = []
    for old, delta, consumed in zip(previous, deltas, actual_fes, strict=True):
        old_value = float(old)
        delta_value = float(delta)
        fe_value = int(consumed)
        if (
            not math.isfinite(old_value)
            or not math.isfinite(delta_value)
            or old_value < 0.0
            or delta_value < 0.0
            or isinstance(consumed, bool)
            or fe_value <= 0
        ):
            raise ValueError("efficiency history must be finite and non-negative")
        efficiency = delta_value / fe_value
        updated.append(
            (1.0 - EFFICIENCY_EWMA_ALPHA) * old_value
            + EFFICIENCY_EWMA_ALPHA * efficiency
        )
    return tuple(updated)


def allocate_efficiency_budgets(
    ewma_efficiency: Sequence[float],
    uniform_budgets: Sequence[int],
    population_sizes: Sequence[int],
) -> tuple[int, ...]:
    """Allocate one frozen sweep budget while preserving its exact FE total."""

    if not (
        len(ewma_efficiency) == len(uniform_budgets) == len(population_sizes)
    ) or not ewma_efficiency:
        raise ValueError("budget allocation inputs must be non-empty and aligned")
    weights = tuple(float(value) for value in ewma_efficiency)
    uniform = tuple(int(value) for value in uniform_budgets)
    populations = tuple(int(value) for value in population_sizes)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("efficiency weights must be finite and non-negative")
    if any(
        isinstance(raw_budget, bool)
        or isinstance(raw_population, bool)
        or population <= 0
        or budget < population
        for raw_budget, raw_population, budget, population in zip(
            uniform_budgets,
            population_sizes,
            uniform,
            populations,
            strict=True,
        )
    ):
        raise ValueError("uniform budgets must cover one positive population")
    if math.fsum(weights) <= 0.0:
        return uniform

    maximums = tuple(
        BUDGET_MAX_UNIFORM_MULTIPLIER * budget for budget in uniform
    )
    allocation = list(populations)
    capacities = [
        maximum - minimum
        for maximum, minimum in zip(maximums, populations, strict=True)
    ]
    remaining = sum(uniform) - sum(allocation)
    if remaining < 0:
        raise ValueError("population floors exceed the frozen sweep budget")

    while remaining > 0:
        active = [index for index, capacity in enumerate(capacities) if capacity > 0]
        if not active:
            raise ValueError("budget caps cannot absorb the frozen sweep budget")
        active_weight = math.fsum(weights[index] for index in active)
        if active_weight <= 0.0:
            active_weights = {index: float(capacities[index]) for index in active}
            active_weight = math.fsum(active_weights.values())
        else:
            active_weights = {index: weights[index] for index in active}
        quotas = {
            index: remaining * active_weights[index] / active_weight
            for index in active
        }
        increments = {
            index: min(capacities[index], int(math.floor(quotas[index])))
            for index in active
        }
        assigned = sum(increments.values())
        if assigned == 0:
            index = max(
                active,
                key=lambda item: (
                    quotas[item] - math.floor(quotas[item]),
                    active_weights[item],
                    -item,
                ),
            )
            increments[index] = 1
            assigned = 1
        for index, increment in increments.items():
            allocation[index] += increment
            capacities[index] -= increment
        remaining -= assigned

    if sum(allocation) != sum(uniform):
        raise RuntimeError("adaptive budgets do not preserve the frozen sweep total")
    return tuple(allocation)


def delta_priority_order(previous_deltas: Sequence[float]) -> tuple[int, ...]:
    deltas = tuple(float(value) for value in previous_deltas)
    if not deltas or any(not math.isfinite(value) or value < 0.0 for value in deltas):
        raise ValueError("priority deltas must be finite, non-negative, and non-empty")
    return tuple(sorted(range(len(deltas)), key=lambda index: (-deltas[index], index)))


def unique_group_positions(
    group_dims: Sequence[Sequence[int]],
    overlapping_elements: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    groups = tuple(tuple(int(value) for value in dims) for dims in group_dims)
    overlaps = tuple(tuple(int(value) for value in values) for values in overlapping_elements)
    if not groups or len(overlaps) != len(groups) - 1:
        raise ValueError("group topology is not an adjacent overlap chain")
    shared_by_group = [set() for _ in groups]
    for relation_index, overlap in enumerate(overlaps):
        shared_by_group[relation_index].update(overlap)
        shared_by_group[relation_index + 1].update(overlap)
    return tuple(
        tuple(
            position
            for position, dimension in enumerate(dims)
            if dimension not in shared_by_group[group_index]
        )
        for group_index, dims in enumerate(groups)
    )


@dataclass(frozen=True)
class NativeContinuationState:
    incumbent: tuple[float, ...]
    sweep_index: int
    next_group_index: int
    completed_group_deltas: tuple[float, ...]
    completed_group_actual_fes: tuple[int, ...]
    group_dims: tuple[tuple[int, ...], ...]
    overlapping_elements: tuple[tuple[int, ...], ...]
    population_sizes: tuple[int, ...]
    optimizer_budgets: tuple[int, ...]
    efficiency_ewma: tuple[float, ...]
    completed_efficiency_sweeps: int
    stagnation_streaks: tuple[int, ...]
    stagnation_cooldowns: tuple[int, ...]
    lower_bound: float
    upper_bound: float
    sigma: float

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
        if len(self.completed_group_actual_fes) != self.next_group_index:
            raise ValueError("completed FE counts do not align with next group")
        if any(value <= 0 for value in self.completed_group_actual_fes):
            raise ValueError("completed group FE counts must be positive")
        if len(self.efficiency_ewma) != group_count or any(
            not math.isfinite(value) or value < 0.0 for value in self.efficiency_ewma
        ):
            raise ValueError("efficiency EWMA state is invalid")
        if (
            isinstance(self.completed_efficiency_sweeps, bool)
            or self.completed_efficiency_sweeps < 0
        ):
            raise ValueError("completed efficiency sweeps must be non-negative")
        for name, values in (
            ("stagnation streak", self.stagnation_streaks),
            ("stagnation cooldown", self.stagnation_cooldowns),
        ):
            if len(values) != group_count or any(
                isinstance(value, bool) or int(value) < 0 for value in values
            ):
                raise ValueError(f"{name} state is invalid")
        if any(
            isinstance(value, bool) or int(value) <= 0
            for value in self.population_sizes + self.optimizer_budgets
        ):
            raise ValueError("population sizes and optimizer budgets must be positive")
        if any(
            budget < population
            for budget, population in zip(
                self.optimizer_budgets,
                self.population_sizes,
                strict=True,
            )
        ):
            raise ValueError("optimizer budgets must cover one population")
        if (
            not math.isfinite(self.lower_bound)
            or not math.isfinite(self.upper_bound)
            or self.lower_bound >= self.upper_bound
            or not math.isfinite(self.sigma)
            or self.sigma <= 0.0
        ):
            raise ValueError("continuation bounds and sigma are invalid")

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
        mean: np.ndarray,
        sigma: float,
    ) -> OptimizationResult: ...


@dataclass(frozen=True)
class ContinuationResult:
    incumbent: tuple[float, ...]
    sweep_index: int
    next_group_index: int
    completed_group_deltas: tuple[float, ...]
    completed_group_actual_fes: tuple[int, ...]
    efficiency_ewma: tuple[float, ...]
    completed_efficiency_sweeps: int
    stagnation_streaks: tuple[int, ...]
    stagnation_cooldowns: tuple[int, ...]
    fitness_record: tuple[float, ...]
    execution_sweep_trace: tuple[int, ...]
    execution_order_trace: tuple[int, ...]
    group_budget_trace: tuple[int, ...]
    group_start_fe_trace: tuple[int, ...]
    policy_application_fes: tuple[int, ...]
    warm_start_event_fes: tuple[int, ...]
    warm_start_shift_norms: tuple[float, ...]
    continuation_policy_applied: bool
    warm_start_trigger_count: int
    warm_start_mean_shift_norm: float


def _run_native_group_steps(
    state: NativeContinuationState,
    *,
    evaluate: Callable[[np.ndarray], np.ndarray],
    fitness_record: list[float],
    optimize_group: GroupOptimizer,
    group_seed: Callable[[int, int], int],
    should_continue: Callable[[int], bool],
    continuation_arm: str,
) -> ContinuationResult:
    if continuation_arm not in ACTION_CEILING_ARMS:
        raise ValueError("unsupported continuation arm")
    incumbent = np.asarray(state.incumbent, dtype=float).copy()
    group_count = len(state.group_dims)
    sweep_index = int(state.sweep_index)
    current_order = tuple(range(group_count))
    order_position = int(state.next_group_index)
    deltas: list[float | None] = [None] * group_count
    actual_fes: list[int | None] = [None] * group_count
    for group_index, (delta, consumed) in enumerate(
        zip(
            state.completed_group_deltas,
            state.completed_group_actual_fes,
            strict=True,
        )
    ):
        deltas[group_index] = float(delta)
        actual_fes[group_index] = int(consumed)
    processed_groups = set(range(state.next_group_index))
    closed_relations = set(range(max(0, state.next_group_index - 1)))
    sweep_budgets = tuple(int(value) for value in state.optimizer_budgets)
    efficiency_ewma = tuple(float(value) for value in state.efficiency_ewma)
    completed_efficiency_sweeps = int(state.completed_efficiency_sweeps)
    stagnation_streaks = [int(value) for value in state.stagnation_streaks]
    stagnation_cooldowns = [int(value) for value in state.stagnation_cooldowns]
    unique_positions = unique_group_positions(
        state.group_dims,
        state.overlapping_elements,
    )
    policy_active = False
    continuation_policy_applied = False
    execution_order_trace: list[int] = []
    execution_sweep_trace: list[int] = []
    group_budget_trace: list[int] = []
    group_start_fe_trace: list[int] = []
    policy_application_fes: list[int] = []
    warm_start_event_fes: list[int] = []
    warm_start_shift_norms: list[float] = []
    warm_start_trigger_count = 0
    warm_start_squared_shift = 0.0
    completed_steps = 0
    while should_continue(completed_steps):
        if order_position == group_count:
            if any(value is None for value in deltas) or any(
                value is None for value in actual_fes
            ):
                raise RuntimeError("continuation reached an incomplete sweep boundary")
            completed_deltas = tuple(float(value) for value in deltas)
            completed_actual_fes = tuple(int(value) for value in actual_fes)
            efficiency_ewma = update_efficiency_ewma(
                efficiency_ewma,
                completed_deltas,
                completed_actual_fes,
            )
            completed_efficiency_sweeps += 1
            sweep_index += 1
            policy_active = True
            current_order = (
                delta_priority_order(completed_deltas)
                if continuation_arm == "delta_priority_scan"
                else tuple(range(group_count))
            )
            if (
                continuation_arm == "delta_priority_scan"
                and current_order != tuple(range(group_count))
            ):
                continuation_policy_applied = True
                policy_application_fes.append(len(fitness_record) + 1)
            sweep_budgets = (
                allocate_efficiency_budgets(
                    efficiency_ewma,
                    state.optimizer_budgets,
                    state.population_sizes,
                )
                if continuation_arm == "efficiency_budget_reallocation"
                else tuple(int(value) for value in state.optimizer_budgets)
            )
            order_position = 0
            deltas = [None] * group_count
            actual_fes = [None] * group_count
            processed_groups = set()
            closed_relations = set()
        group_index = current_order[order_position]
        dims = state.group_dims[group_index]
        original = incumbent.copy()
        group_start_fe = len(fitness_record) + 1
        values = np.asarray(evaluate(incumbent), dtype=float).reshape(-1)
        if values.shape != (1,) or not np.isfinite(values[0]):
            raise ValueError("native precheck must return one finite value")
        original_fitness = float(values[0])
        mean = incumbent[np.asarray(dims, dtype=int)].copy()
        if (
            policy_active
            and continuation_arm == "stagnation_cross_group_warm_start"
        ):
            if stagnation_cooldowns[group_index] > 0:
                stagnation_cooldowns[group_index] -= 1
            elif (
                stagnation_streaks[group_index] >= STAGNATION_TRIGGER_STREAK
                and unique_positions[group_index]
            ):
                seed = (int(group_seed(sweep_index, group_index)) + 0x9E3779B9) % (
                    2**32
                )
                rng = np.random.default_rng(seed)
                positions = np.asarray(unique_positions[group_index], dtype=int)
                before = mean[positions].copy()
                mean[positions] = np.clip(
                    before + rng.normal(0.0, state.sigma, len(positions)),
                    state.lower_bound,
                    state.upper_bound,
                )
                shift = float(np.linalg.norm(mean[positions] - before))
                if shift > 0.0:
                    warm_start_trigger_count += 1
                    warm_start_squared_shift += shift * shift
                    continuation_policy_applied = True
                    policy_application_fes.append(len(fitness_record) + 1)
                    warm_start_event_fes.append(len(fitness_record) + 1)
                    warm_start_shift_norms.append(shift)
                    stagnation_streaks[group_index] = 0
                    stagnation_cooldowns[group_index] = WARM_START_COOLDOWN_SWEEPS
        budget = sweep_budgets[group_index]
        result = optimize_group(
            group_index=group_index,
            background=incumbent.copy(),
            dims=dims,
            requested_fes=budget,
            population_size=state.population_sizes[group_index],
            seed=int(group_seed(sweep_index, group_index)),
            mean=mean,
            sigma=state.sigma,
        )
        if result.actual_fes <= 0 or result.actual_fes > budget:
            raise ValueError("native group optimizer exceeded or skipped its frozen budget")
        if len(result.best_x) != len(dims):
            raise ValueError("native group optimizer returned the wrong dimension")
        if result.best_y < original_fitness:
            incumbent[np.asarray(dims, dtype=int)] = result.best_x
            current_delta = original_fitness - result.best_y
        else:
            current_delta = 0.0
        deltas[group_index] = current_delta
        actual_fes[group_index] = 1 + result.actual_fes
        if current_delta < STAGNATION_EPSILON * abs(original_fitness):
            stagnation_streaks[group_index] += 1
        else:
            stagnation_streaks[group_index] = 0
        processed_groups.add(group_index)
        for relation_index, relation_overlap in enumerate(state.overlapping_elements):
            if relation_index in closed_relations:
                continue
            left_group = relation_index
            right_group = relation_index + 1
            if not {left_group, right_group}.issubset(processed_groups):
                continue
            overlap = np.asarray(relation_overlap, dtype=int)
            if overlap.size:
                prior_group = right_group if group_index == left_group else left_group
                prior_delta = deltas[prior_group]
                if prior_delta is None:
                    raise RuntimeError("overlap owner delta is missing")
                incumbent[overlap] = native_eq8_values(
                    original[overlap],
                    incumbent[overlap],
                    prior_delta,
                    current_delta,
                )
            closed_relations.add(relation_index)
        execution_order_trace.append(group_index)
        execution_sweep_trace.append(sweep_index)
        group_budget_trace.append(budget)
        group_start_fe_trace.append(group_start_fe)
        if (
            policy_active
            and continuation_arm == "efficiency_budget_reallocation"
            and budget != state.optimizer_budgets[group_index]
        ):
            continuation_policy_applied = True
            policy_application_fes.append(group_start_fe)
        order_position += 1
        completed_steps += 1

    completed_indices = current_order[:order_position]
    return ContinuationResult(
        incumbent=tuple(float(value) for value in incumbent),
        sweep_index=sweep_index,
        next_group_index=order_position,
        completed_group_deltas=tuple(
            float(deltas[index]) for index in completed_indices
        ),
        completed_group_actual_fes=tuple(
            int(actual_fes[index]) for index in completed_indices
        ),
        efficiency_ewma=efficiency_ewma,
        completed_efficiency_sweeps=completed_efficiency_sweeps,
        stagnation_streaks=tuple(stagnation_streaks),
        stagnation_cooldowns=tuple(stagnation_cooldowns),
        fitness_record=tuple(float(value) for value in fitness_record),
        execution_sweep_trace=tuple(execution_sweep_trace),
        execution_order_trace=tuple(execution_order_trace),
        group_budget_trace=tuple(group_budget_trace),
        group_start_fe_trace=tuple(group_start_fe_trace),
        policy_application_fes=tuple(policy_application_fes),
        warm_start_event_fes=tuple(warm_start_event_fes),
        warm_start_shift_norms=tuple(warm_start_shift_norms),
        continuation_policy_applied=continuation_policy_applied,
        warm_start_trigger_count=warm_start_trigger_count,
        warm_start_mean_shift_norm=math.sqrt(warm_start_squared_shift),
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
        continuation_arm="native_eq8",
    )


def run_native_continuation(
    state: NativeContinuationState,
    *,
    evaluate: Callable[[np.ndarray], np.ndarray],
    fitness_record: list[float],
    optimize_group: GroupOptimizer,
    group_seed: Callable[[int, int], int],
    target_relative_fe: int,
    continuation_arm: str = "native_eq8",
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
        continuation_arm=continuation_arm,
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
