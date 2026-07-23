"""Persistent execution contract for one frozen Phase2 budget allocation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from arac.actions.action_spec import ActionSpec
from arac.actions.budget_reallocation import (
    FROZEN_BUDGET_MAX_UNIFORM_MULTIPLIER,
    apply_budget_reallocation_action,
    budget_allocation_parameter_hash,
)


PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION = (
    "persistent_frozen_efficiency_budget_reallocation"
)
PERSISTENT_BUDGET_ACTION_ARTIFACT_SCHEMA = "persistent-budget-action-v1"
PERSISTENT_BUDGET_REALLOCATION_ACTION_SPEC = ActionSpec(
    name=PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
    semantic_surface="persistent_budget_allocation",
    parameter_names=(
        "source_efficiency_ewma",
        "population_sizes",
        "uniform_group_budgets",
        "group_budgets",
        "frozen_total_fes",
        "start_sweep",
        "end_absolute_fe",
    ),
)

_HASH_LENGTH = 64
_LIFECYCLE_STATUSES = frozenset({"issued", "active", "completed", "abstained"})


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
    minimum: int,
) -> tuple[int, ...]:
    vector = tuple(values)
    if not vector or any(
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
        for value in vector
    ):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    return vector


def _efficiency_vector(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or any(not math.isfinite(value) or value < 0.0 for value in vector):
        raise ValueError("source_efficiency_ewma must be finite and non-negative")
    return vector


@dataclass(frozen=True)
class PersistentBudgetAllocationAction:
    """One immutable allocation applied throughout a bounded Phase2 interval."""

    problem_id: str
    run_seed: int
    checkpoint_fe: int
    checkpoint_hash: str
    action_set_hash: str
    source_efficiency_ewma: tuple[float, ...]
    population_sizes: tuple[int, ...]
    uniform_group_budgets: tuple[int, ...]
    group_budgets: tuple[int, ...]
    frozen_total_fes: int
    start_sweep: int
    end_absolute_fe: int

    def __post_init__(self) -> None:
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")
        _integer(self.run_seed, "run_seed")
        checkpoint_fe = _integer(self.checkpoint_fe, "checkpoint_fe")
        _validate_hash(self.checkpoint_hash, "checkpoint_hash")
        _validate_hash(self.action_set_hash, "action_set_hash")

        efficiencies = _efficiency_vector(self.source_efficiency_ewma)
        populations = _integer_vector(
            self.population_sizes,
            "population_sizes",
            minimum=1,
        )
        uniform = _integer_vector(
            self.uniform_group_budgets,
            "uniform_group_budgets",
            minimum=1,
        )
        budgets = _integer_vector(self.group_budgets, "group_budgets", minimum=1)
        if not len(efficiencies) == len(populations) == len(uniform) == len(budgets):
            raise ValueError("persistent budget vectors must be non-empty and aligned")
        object.__setattr__(self, "source_efficiency_ewma", efficiencies)
        object.__setattr__(self, "population_sizes", populations)
        object.__setattr__(self, "uniform_group_budgets", uniform)
        object.__setattr__(self, "group_budgets", budgets)

        total = _integer(self.frozen_total_fes, "frozen_total_fes", minimum=1)
        apply_budget_reallocation_action(
            "efficiency_budget_reallocation",
            uniform,
            populations,
            total,
        )
        apply_budget_reallocation_action(
            "efficiency_budget_reallocation",
            budgets,
            populations,
            total,
        )
        if any(
            budget > FROZEN_BUDGET_MAX_UNIFORM_MULTIPLIER * uniform_budget
            for budget, uniform_budget in zip(budgets, uniform, strict=True)
        ):
            raise ValueError("each group budget must respect the 3x uniform cap")

        _integer(self.start_sweep, "start_sweep")
        end_absolute_fe = _integer(
            self.end_absolute_fe,
            "end_absolute_fe",
            minimum=1,
        )
        if end_absolute_fe <= checkpoint_fe:
            raise ValueError("end_absolute_fe must be after checkpoint_fe")

    @property
    def budget_parameter_hash(self) -> str:
        return budget_allocation_parameter_hash(
            population_sizes=self.population_sizes,
            uniform_group_budgets=self.uniform_group_budgets,
            group_budgets=self.group_budgets,
            frozen_total_fes=self.frozen_total_fes,
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "action": PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
            "problem_id": self.problem_id,
            "run_seed": self.run_seed,
            "checkpoint_fe": self.checkpoint_fe,
            "checkpoint_hash": self.checkpoint_hash,
            "action_set_hash": self.action_set_hash,
            "source_efficiency_ewma": list(self.source_efficiency_ewma),
            "population_sizes": list(self.population_sizes),
            "uniform_group_budgets": list(self.uniform_group_budgets),
            "group_budgets": list(self.group_budgets),
            "frozen_total_fes": self.frozen_total_fes,
            "start_sweep": self.start_sweep,
            "end_absolute_fe": self.end_absolute_fe,
            "budget_parameter_hash": self.budget_parameter_hash,
        }

    @property
    def action_hash(self) -> str:
        return _canonical_sha256(self.audit_payload())


@dataclass(frozen=True)
class PersistentBudgetApplication:
    """Auditable outcome of applying the frozen allocation for one sweep."""

    sweep_index: int
    application_fe: int
    requested_group_budgets: tuple[int, ...]
    applied_group_budgets: tuple[int, ...]
    actual_optimizer_fes: tuple[int, ...]
    group_interval_fes: tuple[int, ...]
    terminal_truncated: bool

    def __post_init__(self) -> None:
        _integer(self.sweep_index, "sweep_index")
        _integer(self.application_fe, "application_fe", minimum=1)
        requested = _integer_vector(
            self.requested_group_budgets,
            "requested_group_budgets",
            minimum=1,
        )
        applied = _integer_vector(
            self.applied_group_budgets,
            "applied_group_budgets",
            minimum=0,
        )
        actual_optimizer = _integer_vector(
            self.actual_optimizer_fes,
            "actual_optimizer_fes",
            minimum=0,
        )
        intervals = _integer_vector(
            self.group_interval_fes,
            "group_interval_fes",
            minimum=0,
        )
        if not len(requested) == len(applied) == len(actual_optimizer) == len(intervals):
            raise ValueError("application budget vectors must be aligned")
        if sum(intervals) <= 0:
            raise ValueError("application must consume at least one group-interval FE")
        if any(
            actual_fes > applied_budget
            for actual_fes, applied_budget in zip(
                actual_optimizer,
                applied,
                strict=True,
            )
        ):
            raise ValueError("actual optimizer FEs cannot exceed applied budgets")
        for applied_budget, actual_fes, interval_fes in zip(
            applied,
            actual_optimizer,
            intervals,
            strict=True,
        ):
            if interval_fes == 0:
                if applied_budget or actual_fes:
                    raise ValueError("unexecuted groups cannot contain optimizer FEs")
            elif interval_fes != actual_fes + 1:
                raise ValueError(
                    "group interval FEs must include exactly one incumbent precheck"
                )
        if not isinstance(self.terminal_truncated, bool):
            raise ValueError("terminal_truncated must be a boolean")
        object.__setattr__(self, "requested_group_budgets", requested)
        object.__setattr__(self, "applied_group_budgets", applied)
        object.__setattr__(self, "actual_optimizer_fes", actual_optimizer)
        object.__setattr__(self, "group_interval_fes", intervals)

    @property
    def actual_end_fe(self) -> int:
        return self.application_fe + sum(self.group_interval_fes) - 1

    def audit_payload(self) -> dict[str, object]:
        return {
            "sweep_index": self.sweep_index,
            "application_fe": self.application_fe,
            "requested_group_budgets": list(self.requested_group_budgets),
            "applied_group_budgets": list(self.applied_group_budgets),
            "actual_optimizer_fes": list(self.actual_optimizer_fes),
            "group_interval_fes": list(self.group_interval_fes),
            "terminal_truncated": self.terminal_truncated,
        }


@dataclass
class PersistentBudgetAllocationExecutionState:
    """Serializable lifecycle for a persistent frozen budget action."""

    action_hash: str
    status: str = "issued"
    applications: tuple[PersistentBudgetApplication, ...] = ()
    completed_fe: int | None = None
    invalidation_reason: str = ""

    def __post_init__(self) -> None:
        records: list[PersistentBudgetApplication] = []
        for value in self.applications:
            if isinstance(value, PersistentBudgetApplication):
                records.append(value)
            elif isinstance(value, Mapping):
                records.append(PersistentBudgetApplication(**value))
            else:
                raise ValueError("applications must contain application records")
        self.applications = tuple(records)
        self._validate_shape()

    @classmethod
    def for_action(
        cls,
        action: PersistentBudgetAllocationAction,
    ) -> PersistentBudgetAllocationExecutionState:
        return cls(action_hash=action.action_hash)

    def _validate_shape(self) -> None:
        _validate_hash(self.action_hash, "action_hash")
        if self.status not in _LIFECYCLE_STATUSES:
            raise ValueError("unsupported persistent budget lifecycle status")
        if self.completed_fe is not None:
            _integer(self.completed_fe, "completed_fe", minimum=1)
        if not isinstance(self.invalidation_reason, str):
            raise ValueError("invalidation_reason must be a string")

    @staticmethod
    def _validate_application(
        action: PersistentBudgetAllocationAction,
        record: PersistentBudgetApplication,
        *,
        expected_sweep: int,
        previous_end_fe: int,
    ) -> None:
        if record.sweep_index != expected_sweep:
            raise ValueError("persistent budget applications must cover consecutive sweeps")
        if record.application_fe != previous_end_fe + 1:
            raise ValueError("persistent budget applications must cover consecutive FEs")
        if record.requested_group_budgets != action.group_budgets:
            raise ValueError("application requested budgets differ from the frozen action")
        if len(record.applied_group_budgets) != len(action.group_budgets):
            raise ValueError("application group count differs from the frozen action")

        if record.application_fe > action.end_absolute_fe:
            raise ValueError("application starts after end_absolute_fe")

        consumed_interval_fes = 0
        expected_applied_budgets: list[int] = []
        for requested, applied, interval_fes in zip(
            record.requested_group_budgets,
            record.applied_group_budgets,
            record.group_interval_fes,
            strict=True,
        ):
            next_group_fe = record.application_fe + consumed_interval_fes
            available_fes = action.end_absolute_fe - next_group_fe + 1
            if available_fes <= 0:
                expected_applied = 0
                if interval_fes != 0:
                    raise ValueError("groups after the absolute FE boundary must not run")
            else:
                # The runner consumes one incumbent precheck before exposing the
                # remaining FE cap to the group optimizer.
                expected_applied = min(requested, available_fes - 1)
                if interval_fes == 0:
                    raise ValueError("an available group must consume its precheck FE")
            if applied != expected_applied:
                raise ValueError(
                    "applied group budgets do not match sequential absolute FE caps"
                )
            expected_applied_budgets.append(expected_applied)
            consumed_interval_fes += interval_fes

        expected_truncation = tuple(expected_applied_budgets) != (
            record.requested_group_budgets
        )
        if record.terminal_truncated != expected_truncation:
            raise ValueError("terminal_truncated does not match applied FE caps")
        if record.actual_end_fe > action.end_absolute_fe:
            raise ValueError("actual group FEs exceed end_absolute_fe")

    def validate_for(self, action: PersistentBudgetAllocationAction) -> None:
        self._validate_shape()
        if self.action_hash != action.action_hash:
            raise ValueError("persistent budget lifecycle does not match action_hash")

        expected_sweep = action.start_sweep
        previous_end_fe = action.checkpoint_fe
        for record in self.applications:
            self._validate_application(
                action,
                record,
                expected_sweep=expected_sweep,
                previous_end_fe=previous_end_fe,
            )
            expected_sweep += 1
            previous_end_fe = record.actual_end_fe

        reached_terminal = bool(
            self.applications
            and self.applications[-1].actual_end_fe == action.end_absolute_fe
        )
        if self.status == "issued":
            if self.applications or self.completed_fe is not None or self.invalidation_reason:
                raise ValueError("issued lifecycle contains runtime outcome data")
        elif self.status == "active":
            if not self.applications or reached_terminal:
                raise ValueError("active lifecycle must contain unfinished applications")
            if self.completed_fe is not None or self.invalidation_reason:
                raise ValueError("active lifecycle contains terminal outcome data")
        elif self.status == "completed":
            if (
                not reached_terminal
                or self.completed_fe != action.end_absolute_fe
                or self.invalidation_reason
            ):
                raise ValueError("completed lifecycle is inconsistent")
        elif self.completed_fe is not None or not self.invalidation_reason:
            raise ValueError("abstained lifecycle is inconsistent")

    def record_application(
        self,
        action: PersistentBudgetAllocationAction,
        *,
        current_sweep: int,
        application_fe: int,
        checkpoint_hash: str,
        action_set_hash: str,
        applied_group_budgets: Sequence[int],
        actual_optimizer_fes: Sequence[int],
        group_interval_fes: Sequence[int],
        terminal_truncated: bool,
    ) -> tuple[int, ...]:
        """Record one sweep without consulting or recomputing Phase2 evidence."""

        self.validate_for(action)
        if self.status not in {"issued", "active"}:
            raise ValueError("only an issued or active action can be applied")
        if _validate_hash(checkpoint_hash, "checkpoint_hash") != action.checkpoint_hash:
            raise ValueError("checkpoint_hash mismatch")
        if _validate_hash(action_set_hash, "action_set_hash") != action.action_set_hash:
            raise ValueError("action_set_hash mismatch")

        record = PersistentBudgetApplication(
            sweep_index=_integer(current_sweep, "current_sweep"),
            application_fe=_integer(application_fe, "application_fe", minimum=1),
            requested_group_budgets=action.group_budgets,
            applied_group_budgets=tuple(applied_group_budgets),
            actual_optimizer_fes=tuple(actual_optimizer_fes),
            group_interval_fes=tuple(group_interval_fes),
            terminal_truncated=terminal_truncated,
        )
        applications = (*self.applications, record)
        completed = record.actual_end_fe == action.end_absolute_fe
        candidate = PersistentBudgetAllocationExecutionState(
            action_hash=self.action_hash,
            status="completed" if completed else "active",
            applications=applications,
            completed_fe=action.end_absolute_fe if completed else None,
        )
        candidate.validate_for(action)
        self.status = candidate.status
        self.applications = candidate.applications
        self.completed_fe = candidate.completed_fe
        return execute_persistent_budget_allocation_action(action)

    def abstain(
        self,
        action: PersistentBudgetAllocationAction,
        *,
        reason: str,
    ) -> None:
        self.validate_for(action)
        if self.status not in {"issued", "active"}:
            raise ValueError("only an issued or active action can abstain")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abstain reason must be a non-empty string")
        candidate = PersistentBudgetAllocationExecutionState(
            action_hash=self.action_hash,
            status="abstained",
            applications=self.applications,
            invalidation_reason=reason,
        )
        candidate.validate_for(action)
        self.status = candidate.status
        self.invalidation_reason = candidate.invalidation_reason

    def audit_payload(self, action: PersistentBudgetAllocationAction) -> dict[str, Any]:
        self.validate_for(action)
        return {
            "action": PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
            "action_hash": self.action_hash,
            "status": self.status,
            "applications": [record.audit_payload() for record in self.applications],
            "completed_fe": self.completed_fe,
            "invalidation_reason": self.invalidation_reason,
        }

    def state_hash(self, action: PersistentBudgetAllocationAction) -> str:
        return _canonical_sha256(self.audit_payload(action))


def execute_persistent_budget_allocation_action(
    action: PersistentBudgetAllocationAction,
) -> tuple[int, ...]:
    """Return the stored allocation; Phase2 evidence is deliberately unavailable."""

    if not isinstance(action, PersistentBudgetAllocationAction):
        raise TypeError("action must be a PersistentBudgetAllocationAction")
    return apply_budget_reallocation_action(
        "efficiency_budget_reallocation",
        action.group_budgets,
        action.population_sizes,
        action.frozen_total_fes,
    )


__all__ = [
    "PERSISTENT_BUDGET_ACTION_ARTIFACT_SCHEMA",
    "PERSISTENT_BUDGET_REALLOCATION_ACTION_SPEC",
    "PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION",
    "PersistentBudgetAllocationAction",
    "PersistentBudgetAllocationExecutionState",
    "PersistentBudgetApplication",
    "execute_persistent_budget_allocation_action",
]
