"""One-sweep budget pulse shrunk halfway toward the uniform allocation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from arac.actions.action_spec import ActionSpec


SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION = "shrunk_efficiency_budget_pulse"
SHRUNK_BUDGET_PULSE_SCHEMA = "shrunk-efficiency-budget-pulse-v1"
SHRUNK_BUDGET_RAW_WEIGHT = 1
SHRUNK_BUDGET_UNIFORM_WEIGHT = 1
SHRUNK_BUDGET_WEIGHT_DENOMINATOR = 2
SHRUNK_BUDGET_RAW_MAX_UNIFORM_MULTIPLIER = 3

SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION_SPEC = ActionSpec(
    name=SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
    semantic_surface="budget_allocation",
    parameter_names=(
        "raw_group_budgets",
        "uniform_group_budgets",
        "group_budgets",
        "population_sizes",
        "frozen_total_fes",
    ),
)

_HASH_LENGTH = 64
_LIFECYCLE_STATUSES = frozenset({"issued", "consumed", "abstained"})


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hash(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _integer_vector(
    values: Sequence[int],
    name: str,
    *,
    minimum: int = 1,
) -> tuple[int, ...]:
    vector = tuple(values)
    if not vector or any(
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
        for value in vector
    ):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    return vector


def _validated_source_vectors(
    raw_group_budgets: Sequence[int],
    uniform_group_budgets: Sequence[int],
    population_sizes: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    raw = _integer_vector(raw_group_budgets, "raw_group_budgets")
    uniform = _integer_vector(uniform_group_budgets, "uniform_group_budgets")
    populations = _integer_vector(population_sizes, "population_sizes")
    if not len(raw) == len(uniform) == len(populations):
        raise ValueError("shrunk budget vectors must be non-empty and aligned")

    frozen_total = sum(uniform)
    if sum(raw) != frozen_total:
        raise ValueError("raw budgets must preserve the uniform FE total")
    for raw_budget, uniform_budget, population in zip(
        raw,
        uniform,
        populations,
        strict=True,
    ):
        if uniform_budget < population or raw_budget < population:
            raise ValueError("source budgets must cover one population per group")
        if raw_budget > (
            SHRUNK_BUDGET_RAW_MAX_UNIFORM_MULTIPLIER * uniform_budget
        ):
            raise ValueError("raw budgets must respect the 3x uniform cap")
    return raw, uniform, populations, frozen_total


def allocate_shrunk_efficiency_budgets(
    raw_group_budgets: Sequence[int],
    uniform_group_budgets: Sequence[int],
    population_sizes: Sequence[int],
) -> tuple[int, ...]:
    """Compile the fixed 50/50 pulse with stable largest-remainder rounding."""

    raw, uniform, populations, frozen_total = _validated_source_vectors(
        raw_group_budgets,
        uniform_group_budgets,
        population_sizes,
    )
    quota_numerators = tuple(
        SHRUNK_BUDGET_RAW_WEIGHT * raw_budget
        + SHRUNK_BUDGET_UNIFORM_WEIGHT * uniform_budget
        for raw_budget, uniform_budget in zip(raw, uniform, strict=True)
    )
    allocation = [
        numerator // SHRUNK_BUDGET_WEIGHT_DENOMINATOR
        for numerator in quota_numerators
    ]
    remaining = frozen_total - sum(allocation)
    remainder_order = sorted(
        range(len(allocation)),
        key=lambda index: (
            -(quota_numerators[index] % SHRUNK_BUDGET_WEIGHT_DENOMINATOR),
            index,
        ),
    )
    for index in remainder_order[:remaining]:
        allocation[index] += 1

    budgets = tuple(allocation)
    if sum(budgets) != frozen_total:
        raise RuntimeError("shrunk budgets do not preserve the frozen FE total")
    for budget, uniform_budget, population in zip(
        budgets,
        uniform,
        populations,
        strict=True,
    ):
        if budget < population:
            raise RuntimeError("shrunk budget does not cover one population")
        if 2 * budget < uniform_budget or budget > 2 * uniform_budget:
            raise RuntimeError("shrunk budget is outside the [0.5U, 2U] range")
    return budgets


def shrunk_budget_pulse_anchor_hash(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    raw_group_budgets: Sequence[int],
    uniform_group_budgets: Sequence[int],
    population_sizes: Sequence[int],
    issued_sweep: int,
) -> str:
    """Bind the exact decision-time allocation that is being shrunk."""

    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem_id must be a non-empty string")
    raw, uniform, populations, _ = _validated_source_vectors(
        raw_group_budgets,
        uniform_group_budgets,
        population_sizes,
    )
    return _canonical_sha256(
        {
            "action": SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
            "schema": SHRUNK_BUDGET_PULSE_SCHEMA,
            "problem_id": problem_id,
            "run_seed": _integer(run_seed, "run_seed"),
            "checkpoint_fe": _integer(checkpoint_fe, "checkpoint_fe"),
            "dispatch_checkpoint_hash": _validate_hash(
                dispatch_checkpoint_hash,
                "dispatch_checkpoint_hash",
            ),
            "raw_group_budgets": raw,
            "uniform_group_budgets": uniform,
            "population_sizes": populations,
            "issued_sweep": _integer(issued_sweep, "issued_sweep"),
        }
    )


def shrunk_budget_pulse_parameter_hash(
    *,
    raw_group_budgets: Sequence[int],
    uniform_group_budgets: Sequence[int],
    group_budgets: Sequence[int],
    population_sizes: Sequence[int],
    frozen_total_fes: int,
) -> str:
    raw, uniform, populations, source_total = _validated_source_vectors(
        raw_group_budgets,
        uniform_group_budgets,
        population_sizes,
    )
    budgets = _integer_vector(group_budgets, "group_budgets")
    total = _integer(frozen_total_fes, "frozen_total_fes", minimum=1)
    if len(budgets) != len(raw):
        raise ValueError("shrunk budget vectors must be non-empty and aligned")
    if total != source_total or sum(budgets) != total:
        raise ValueError("shrunk budgets must preserve the frozen FE total")
    return _canonical_sha256(
        {
            "action": SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
            "schema": SHRUNK_BUDGET_PULSE_SCHEMA,
            "raw_group_budgets": raw,
            "uniform_group_budgets": uniform,
            "group_budgets": budgets,
            "population_sizes": populations,
            "frozen_total_fes": total,
            "raw_weight": SHRUNK_BUDGET_RAW_WEIGHT,
            "uniform_weight": SHRUNK_BUDGET_UNIFORM_WEIGHT,
            "weight_denominator": SHRUNK_BUDGET_WEIGHT_DENOMINATOR,
        }
    )


@dataclass(frozen=True)
class ShrunkEfficiencyBudgetPulseAction:
    """Immutable 50/50 allocation consumed once at the next sweep."""

    problem_id: str
    run_seed: int
    checkpoint_fe: int
    dispatch_checkpoint_hash: str
    anchor_hash: str
    raw_group_budgets: tuple[int, ...]
    uniform_group_budgets: tuple[int, ...]
    group_budgets: tuple[int, ...]
    population_sizes: tuple[int, ...]
    frozen_total_fes: int
    issued_sweep: int
    target_sweep: int
    ttl_sweeps: int
    expires_sweep: int
    parameter_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")
        _integer(self.run_seed, "run_seed")
        _integer(self.checkpoint_fe, "checkpoint_fe")
        _validate_hash(self.dispatch_checkpoint_hash, "dispatch_checkpoint_hash")
        _validate_hash(self.anchor_hash, "anchor_hash")
        _validate_hash(self.parameter_hash, "parameter_hash")

        raw, uniform, populations, source_total = _validated_source_vectors(
            self.raw_group_budgets,
            self.uniform_group_budgets,
            self.population_sizes,
        )
        budgets = _integer_vector(self.group_budgets, "group_budgets")
        if len(budgets) != len(raw):
            raise ValueError("shrunk budget vectors must be non-empty and aligned")
        object.__setattr__(self, "raw_group_budgets", raw)
        object.__setattr__(self, "uniform_group_budgets", uniform)
        object.__setattr__(self, "population_sizes", populations)
        object.__setattr__(self, "group_budgets", budgets)

        total = _integer(self.frozen_total_fes, "frozen_total_fes", minimum=1)
        if total != source_total or sum(budgets) != total:
            raise ValueError("shrunk budgets must preserve the frozen FE total")
        expected_budgets = allocate_shrunk_efficiency_budgets(
            raw,
            uniform,
            populations,
        )
        if budgets != expected_budgets:
            raise ValueError("group_budgets do not match the fixed 50/50 pulse")

        issued = _integer(self.issued_sweep, "issued_sweep")
        target = _integer(self.target_sweep, "target_sweep")
        ttl = _integer(self.ttl_sweeps, "ttl_sweeps", minimum=1)
        expires = _integer(self.expires_sweep, "expires_sweep")
        if ttl != 1:
            raise ValueError("shrunk budget pulse must have ttl_sweeps=1")
        if target != issued + 1:
            raise ValueError("target_sweep must be the next sweep")
        if expires != target:
            raise ValueError("expires_sweep must equal the target sweep")

        expected_anchor = shrunk_budget_pulse_anchor_hash(
            problem_id=self.problem_id,
            run_seed=self.run_seed,
            checkpoint_fe=self.checkpoint_fe,
            dispatch_checkpoint_hash=self.dispatch_checkpoint_hash,
            raw_group_budgets=raw,
            uniform_group_budgets=uniform,
            population_sizes=populations,
            issued_sweep=issued,
        )
        if self.anchor_hash != expected_anchor:
            raise ValueError("anchor_hash does not match the pulse source allocation")
        expected_parameters = shrunk_budget_pulse_parameter_hash(
            raw_group_budgets=raw,
            uniform_group_budgets=uniform,
            group_budgets=budgets,
            population_sizes=populations,
            frozen_total_fes=total,
        )
        if self.parameter_hash != expected_parameters:
            raise ValueError("parameter_hash does not match the frozen pulse")

    def audit_payload(self) -> dict[str, object]:
        return {
            "action": SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
            "schema": SHRUNK_BUDGET_PULSE_SCHEMA,
            "problem_id": self.problem_id,
            "run_seed": self.run_seed,
            "checkpoint_fe": self.checkpoint_fe,
            "dispatch_checkpoint_hash": self.dispatch_checkpoint_hash,
            "anchor_hash": self.anchor_hash,
            "raw_group_budgets": list(self.raw_group_budgets),
            "uniform_group_budgets": list(self.uniform_group_budgets),
            "group_budgets": list(self.group_budgets),
            "population_sizes": list(self.population_sizes),
            "frozen_total_fes": self.frozen_total_fes,
            "issued_sweep": self.issued_sweep,
            "target_sweep": self.target_sweep,
            "ttl_sweeps": self.ttl_sweeps,
            "expires_sweep": self.expires_sweep,
            "parameter_hash": self.parameter_hash,
        }

    @property
    def action_hash(self) -> str:
        return _canonical_sha256(self.audit_payload())


def execute_shrunk_efficiency_budget_pulse_action(
    action: ShrunkEfficiencyBudgetPulseAction,
) -> tuple[int, ...]:
    """Return the stored allocation without consulting Phase2 evidence."""

    if not isinstance(action, ShrunkEfficiencyBudgetPulseAction):
        raise TypeError("action must be a ShrunkEfficiencyBudgetPulseAction")
    expected_hash = shrunk_budget_pulse_parameter_hash(
        raw_group_budgets=action.raw_group_budgets,
        uniform_group_budgets=action.uniform_group_budgets,
        group_budgets=action.group_budgets,
        population_sizes=action.population_sizes,
        frozen_total_fes=action.frozen_total_fes,
    )
    if action.parameter_hash != expected_hash:
        raise ValueError("parameter_hash does not match the frozen pulse")
    return tuple(action.group_budgets)


@dataclass
class ShrunkBudgetPulseExecutionState:
    """Serializable one-shot lifecycle for a shrunk budget pulse."""

    action_hash: str
    status: str = "issued"
    consumed_sweep: int | None = None
    application_fe: int | None = None
    applied_group_budgets: tuple[int, ...] = ()
    invalidation_reason: str = ""

    def __post_init__(self) -> None:
        self.applied_group_budgets = tuple(self.applied_group_budgets)
        self._validate_shape()

    @classmethod
    def for_action(
        cls,
        action: ShrunkEfficiencyBudgetPulseAction,
    ) -> ShrunkBudgetPulseExecutionState:
        return cls(action_hash=action.action_hash)

    def _validate_shape(self) -> None:
        _validate_hash(self.action_hash, "action_hash")
        if self.status not in _LIFECYCLE_STATUSES:
            raise ValueError("unsupported shrunk budget pulse lifecycle status")
        if self.consumed_sweep is not None:
            _integer(self.consumed_sweep, "consumed_sweep")
        if self.application_fe is not None:
            _integer(self.application_fe, "application_fe", minimum=1)
        if self.applied_group_budgets:
            _integer_vector(self.applied_group_budgets, "applied_group_budgets")
        if not isinstance(self.invalidation_reason, str):
            raise ValueError("invalidation_reason must be a string")

    def validate_for(self, action: ShrunkEfficiencyBudgetPulseAction) -> None:
        self._validate_shape()
        if self.action_hash != action.action_hash:
            raise ValueError("shrunk budget pulse lifecycle does not match action_hash")
        if self.status == "issued":
            if (
                self.consumed_sweep is not None
                or self.application_fe is not None
                or self.applied_group_budgets
                or self.invalidation_reason
            ):
                raise ValueError("issued pulse lifecycle contains outcome data")
        elif self.status == "consumed":
            if (
                self.consumed_sweep != action.target_sweep
                or self.application_fe is None
                or self.applied_group_budgets != action.group_budgets
                or self.invalidation_reason
            ):
                raise ValueError("consumed pulse lifecycle is inconsistent")
        elif (
            self.consumed_sweep is not None
            or self.application_fe is not None
            or self.applied_group_budgets
            or not self.invalidation_reason
        ):
            raise ValueError("abstained pulse lifecycle is inconsistent")

    def consume(
        self,
        action: ShrunkEfficiencyBudgetPulseAction,
        *,
        current_sweep: int,
        application_fe: int,
        dispatch_checkpoint_hash: str,
        anchor_hash: str,
    ) -> tuple[int, ...]:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued shrunk budget pulse can be consumed")
        sweep = _integer(current_sweep, "current_sweep")
        if sweep > action.expires_sweep:
            raise ValueError("shrunk budget pulse TTL expired")
        if sweep != action.target_sweep:
            raise ValueError("current_sweep does not match target_sweep")
        if (
            _validate_hash(dispatch_checkpoint_hash, "dispatch_checkpoint_hash")
            != action.dispatch_checkpoint_hash
        ):
            raise ValueError("dispatch_checkpoint_hash mismatch")
        if _validate_hash(anchor_hash, "anchor_hash") != action.anchor_hash:
            raise ValueError("anchor_hash mismatch")

        budgets = execute_shrunk_efficiency_budget_pulse_action(action)
        self.consumed_sweep = sweep
        self.application_fe = _integer(application_fe, "application_fe", minimum=1)
        self.applied_group_budgets = budgets
        self.status = "consumed"
        return budgets

    def abstain(
        self,
        action: ShrunkEfficiencyBudgetPulseAction,
        *,
        reason: str,
    ) -> None:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued shrunk budget pulse can abstain")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abstain reason must be a non-empty string")
        self.invalidation_reason = reason
        self.status = "abstained"

    def audit_payload(
        self,
        action: ShrunkEfficiencyBudgetPulseAction,
    ) -> dict[str, Any]:
        self.validate_for(action)
        return {
            "action": SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
            "action_hash": self.action_hash,
            "status": self.status,
            "consumed_sweep": self.consumed_sweep,
            "application_fe": self.application_fe,
            "applied_group_budgets": list(self.applied_group_budgets),
            "invalidation_reason": self.invalidation_reason,
        }

    def state_hash(self, action: ShrunkEfficiencyBudgetPulseAction) -> str:
        return _canonical_sha256(self.audit_payload(action))


__all__ = [
    "SHRUNK_BUDGET_PULSE_SCHEMA",
    "SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION",
    "SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION_SPEC",
    "ShrunkBudgetPulseExecutionState",
    "ShrunkEfficiencyBudgetPulseAction",
    "allocate_shrunk_efficiency_budgets",
    "execute_shrunk_efficiency_budget_pulse_action",
    "shrunk_budget_pulse_anchor_hash",
    "shrunk_budget_pulse_parameter_hash",
]
