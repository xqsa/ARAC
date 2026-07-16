"""Pure contracts for one-shot component-atomic precision experiments.

This module does not execute an optimizer.  It owns only the once-per-trajectory
plan, endpoint survival calculations, and endpoint-relative rewards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Sequence


COMPONENT_ATOMIC_SCHEMA_VERSION = "component-atomic-precision-v1"
ERROR_FLOOR = 1e-300
DEFAULT_MATERIAL_LOG_GAIN = math.log(1.01)
FORBIDDEN_SCHEMA_FIELD_FRAGMENTS = (
    "case",
    "seed",
    "family",
    "fingerprint",
    "outcome",
    "paper",
    "raw",
    "objective",
    "terminal",
)


@dataclass(frozen=True)
class ComponentAtomicPlan:
    """A pure decision for the first feasible component and its once lock."""

    execute_precision: bool
    reason: str
    once_lock_consumed_before: bool
    once_lock_consumed_after: bool
    group_indices: tuple[int, ...]
    group_budgets: tuple[int, ...]
    population_sizes: tuple[int, ...]
    normal_sigma: float
    precision_sigma: float

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("component atomic plan reason is required")
        if self.once_lock_consumed_before and not self.once_lock_consumed_after:
            raise ValueError("a consumed once lock cannot be released")
        if self.execute_precision and (
            self.once_lock_consumed_before or not self.once_lock_consumed_after
        ):
            raise ValueError("precision execution must acquire an unused once lock")
        if not isinstance(self.group_indices, tuple) or not self.group_indices:
            raise ValueError("group_indices must be a non-empty tuple")
        if any(index < 0 for index in self.group_indices) or any(
            right != left + 1
            for left, right in zip(self.group_indices, self.group_indices[1:])
        ):
            raise ValueError("group_indices must be non-negative and contiguous")
        if not isinstance(self.group_budgets, tuple) or not isinstance(
            self.population_sizes, tuple
        ):
            raise ValueError("group budgets and population sizes must be tuples")
        if not (
            len(self.group_indices)
            == len(self.group_budgets)
            == len(self.population_sizes)
        ):
            raise ValueError("group execution vectors must have equal length")
        if any(value <= 0 for value in self.group_budgets) or any(
            value <= 0 for value in self.population_sizes
        ):
            raise ValueError("group budgets and population sizes must be positive")
        if any(
            budget % population != 0
            for budget, population in zip(
                self.group_budgets,
                self.population_sizes,
                strict=True,
            )
        ):
            raise ValueError("each group budget must contain complete populations")
        if not math.isfinite(float(self.normal_sigma)) or self.normal_sigma <= 0.0:
            raise ValueError("normal_sigma must be finite and positive")
        if (
            not math.isfinite(float(self.precision_sigma))
            or self.precision_sigma <= 0.0
        ):
            raise ValueError("precision_sigma must be finite and positive")
        if self.precision_sigma != 0.5 * self.normal_sigma:
            raise ValueError("precision_sigma must equal 0.5 * normal_sigma")


@dataclass(frozen=True)
class ComponentEndpointResult:
    """Identity-free component endpoint facts and delayed survival scores."""

    checkpoint_error: float
    endpoint_error: float
    s_h: float
    s_d: float
    strict_survival: bool

    def __post_init__(self) -> None:
        for name in ("checkpoint_error", "endpoint_error"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("s_h", "s_d"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class ComponentReward:
    """Checkpoint-relative endpoint reward without identity-bearing fields."""

    checkpoint_error: float
    endpoint_error: float
    log_gain: float
    s_h: float
    s_d: float
    strict_survival: bool
    material: bool

    def __post_init__(self) -> None:
        numeric = (
            self.checkpoint_error,
            self.endpoint_error,
            self.log_gain,
            self.s_h,
            self.s_d,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("component reward values must be finite")
        if self.checkpoint_error < 0.0 or self.endpoint_error < 0.0:
            raise ValueError("component reward errors must be non-negative")
        if not 0.0 <= self.s_h <= 1.0 or not 0.0 <= self.s_d <= 1.0:
            raise ValueError("component reward survival scores must be in [0, 1]")


def plan_component_atomic_precision(
    *,
    candidate_feasible: bool,
    component_unlocked: bool,
    horizon_reachable: bool,
    once_lock_consumed: bool,
    group_indices: Sequence[int],
    group_budgets: Sequence[int],
    population_sizes: Sequence[int],
    normal_sigma: float,
    precision_sigma: float,
) -> ComponentAtomicPlan:
    """Acquire the trajectory-wide once lock at the first feasible component."""

    def integer_tuple(values: Sequence[int], *, name: str) -> tuple[int, ...]:
        converted: list[int] = []
        for value in values:
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must contain only integers")
            converted.append(int(value))
        return tuple(converted)

    execution = {
        "group_indices": integer_tuple(group_indices, name="group_indices"),
        "group_budgets": integer_tuple(group_budgets, name="group_budgets"),
        "population_sizes": integer_tuple(
            population_sizes,
            name="population_sizes",
        ),
        "normal_sigma": float(normal_sigma),
        "precision_sigma": float(precision_sigma),
    }

    if once_lock_consumed:
        return ComponentAtomicPlan(
            execute_precision=False,
            reason="abstain_once_lock_consumed",
            once_lock_consumed_before=True,
            once_lock_consumed_after=True,
            **execution,
        )
    if not candidate_feasible:
        reason = "abstain_candidate_infeasible"
    elif not component_unlocked:
        reason = "abstain_component_locked"
    elif not horizon_reachable:
        reason = "abstain_component_horizon_unreachable"
    else:
        return ComponentAtomicPlan(
            execute_precision=True,
            reason="component_atomic_precision_selected",
            once_lock_consumed_before=False,
            once_lock_consumed_after=True,
            **execution,
        )
    return ComponentAtomicPlan(
        execute_precision=False,
        reason=reason,
        once_lock_consumed_before=False,
        once_lock_consumed_after=False,
        **execution,
    )


def _finite_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _l1_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    distances = tuple(abs(a - b) for a, b in zip(left, right, strict=True))
    if not all(math.isfinite(value) for value in distances):
        raise ValueError("shared-value L1 distance must be finite")
    try:
        distance = math.fsum(distances)
    except OverflowError as exc:
        raise ValueError("shared-value L1 distance must be finite") from exc
    if not math.isfinite(distance):
        raise ValueError("shared-value L1 distance must be finite")
    return distance


def build_component_endpoint_result(
    *,
    checkpoint_error: float,
    endpoint_error: float,
    canonical_shared_path: Sequence[Sequence[float]],
    next_shared_values: Sequence[float],
    epsilon: float = ERROR_FLOOR,
) -> ComponentEndpointResult:
    """Calculate endpoint survival on shared variables.

    ``canonical_shared_path`` is ``(x_0, x_1, ..., x_H)`` on the shared
    indices after each canonical group completion.  The scores are

    ``S_H = ||x_H-x_0||_1 / (epsilon + sum_g ||x_g-x_(g-1)||_1)``

    and

    ``S_D = 1 - min(1, ||x_next-x_H||_1 / max(||x_H-x_0||_1, epsilon))``.
    """

    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    path = tuple(
        _finite_vector(values, name=f"canonical_shared_path[{index}]")
        for index, values in enumerate(canonical_shared_path)
    )
    if len(path) < 2:
        raise ValueError("canonical_shared_path requires x_0 and x_H")
    width = len(path[0])
    if any(len(values) != width for values in path):
        raise ValueError("canonical_shared_path vectors must have equal width")
    next_values = _finite_vector(next_shared_values, name="next_shared_values")
    if len(next_values) != width:
        raise ValueError("next_shared_values width does not match shared path")

    start = path[0]
    endpoint = path[-1]
    endpoint_displacement = _l1_distance(endpoint, start)
    try:
        travelled = math.fsum(
            _l1_distance(current, previous)
            for previous, current in zip(path, path[1:])
        )
    except OverflowError as exc:
        raise ValueError("shared-value path length must be finite") from exc
    if not math.isfinite(travelled):
        raise ValueError("shared-value path length must be finite")
    s_h = min(1.0, max(0.0, endpoint_displacement / (epsilon_value + travelled)))
    strict_survival = endpoint_displacement > epsilon_value
    if strict_survival:
        delayed_drift = _l1_distance(next_values, endpoint)
        s_d = 1.0 - min(
            1.0,
            delayed_drift / max(endpoint_displacement, epsilon_value),
        )
    else:
        s_d = 0.0
    return ComponentEndpointResult(
        checkpoint_error=float(checkpoint_error),
        endpoint_error=float(endpoint_error),
        s_h=float(min(1.0, max(0.0, s_h))),
        s_d=float(min(1.0, max(0.0, s_d))),
        strict_survival=strict_survival,
    )


def build_component_reward(
    endpoint: ComponentEndpointResult,
    *,
    material_log_gain: float = DEFAULT_MATERIAL_LOG_GAIN,
) -> ComponentReward:
    """Build a finite checkpoint-relative reward for one component endpoint."""

    if not isinstance(endpoint, ComponentEndpointResult):
        raise TypeError("endpoint must be ComponentEndpointResult")
    threshold = float(material_log_gain)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("material_log_gain must be finite and non-negative")
    checkpoint = float(endpoint.checkpoint_error)
    endpoint_error = float(endpoint.endpoint_error)
    log_gain = math.log(max(checkpoint, ERROR_FLOOR)) - math.log(
        max(endpoint_error, ERROR_FLOOR)
    )
    return ComponentReward(
        checkpoint_error=checkpoint,
        endpoint_error=endpoint_error,
        log_gain=float(log_gain),
        s_h=endpoint.s_h,
        s_d=endpoint.s_d,
        strict_survival=endpoint.strict_survival,
        material=log_gain >= threshold,
    )


def paired_endpoint_tau(
    baseline: ComponentEndpointResult,
    precision: ComponentEndpointResult,
) -> float:
    """Return ``log(error_baseline / error_precision)`` at a common endpoint."""

    if not isinstance(baseline, ComponentEndpointResult) or not isinstance(
        precision, ComponentEndpointResult
    ):
        raise TypeError("paired endpoints must be ComponentEndpointResult values")
    if baseline.checkpoint_error != precision.checkpoint_error:
        raise ValueError("paired endpoints must share the same checkpoint error")
    return float(
        math.log(max(baseline.endpoint_error, ERROR_FLOOR))
        - math.log(max(precision.endpoint_error, ERROR_FLOOR))
    )


def assert_atomic_schema_has_no_forbidden_fields() -> None:
    expected = {
        ComponentAtomicPlan: {
            "execute_precision",
            "reason",
            "once_lock_consumed_before",
            "once_lock_consumed_after",
            "group_indices",
            "group_budgets",
            "population_sizes",
            "normal_sigma",
            "precision_sigma",
        },
        ComponentEndpointResult: {
            "checkpoint_error",
            "endpoint_error",
            "s_h",
            "s_d",
            "strict_survival",
        },
        ComponentReward: {
            "checkpoint_error",
            "endpoint_error",
            "log_gain",
            "s_h",
            "s_d",
            "strict_survival",
            "material",
        },
    }
    for schema, allowed in expected.items():
        names = {item.name for item in fields(schema)}
        if names != allowed:
            raise RuntimeError(f"{schema.__name__} schema drift")
        leaked = sorted(
            name
            for name in names
            if any(
                fragment in name.lower()
                for fragment in FORBIDDEN_SCHEMA_FIELD_FRAGMENTS
            )
        )
        if leaked:
            raise RuntimeError(
                f"forbidden {schema.__name__} fields: {','.join(leaked)}"
            )


assert_atomic_schema_has_no_forbidden_fields()
