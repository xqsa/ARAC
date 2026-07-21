"""Deterministic execution of frozen group-budget allocations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from arac.actions.action_spec import ActionSpec


FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION = (
    "frozen_efficiency_budget_reallocation"
)
FROZEN_BUDGET_MAX_UNIFORM_MULTIPLIER = 3

BUDGET_REALLOCATION_ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        name="efficiency_budget_reallocation",
        semantic_surface="budget_allocation",
        parameter_names=("group_budgets", "population_sizes", "frozen_total"),
    ),
    ActionSpec(
        name=FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
        semantic_surface="budget_allocation",
        parameter_names=(
            "group_budgets",
            "population_sizes",
            "uniform_group_budgets",
            "frozen_total_fes",
        ),
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


def _efficiency_vector(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or any(not math.isfinite(value) or value < 0.0 for value in vector):
        raise ValueError("source_efficiency_ewma must be finite and non-negative")
    return vector


def budget_allocation_anchor_hash(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    source_efficiency_ewma: Sequence[float],
    population_sizes: Sequence[int],
    uniform_group_budgets: Sequence[int],
    issued_sweep: int,
) -> str:
    """Hash only the decision-time state used to compile the budget action."""

    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem_id must be a non-empty string")
    seed = _integer(run_seed, "run_seed")
    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    checkpoint_hash = _validate_hash(
        dispatch_checkpoint_hash,
        "dispatch_checkpoint_hash",
    )
    efficiencies = _efficiency_vector(source_efficiency_ewma)
    populations = _integer_vector(population_sizes, "population_sizes")
    uniform = _integer_vector(uniform_group_budgets, "uniform_group_budgets")
    sweep = _integer(issued_sweep, "issued_sweep")
    if not len(efficiencies) == len(populations) == len(uniform):
        raise ValueError("budget action source vectors must be aligned")
    if any(
        budget < population
        for budget, population in zip(uniform, populations, strict=True)
    ):
        raise ValueError("uniform budgets must cover one population per group")
    return _canonical_sha256(
        {
            "action": FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
            "problem_id": problem_id,
            "run_seed": seed,
            "checkpoint_fe": checkpoint,
            "dispatch_checkpoint_hash": checkpoint_hash,
            "source_efficiency_ewma": efficiencies,
            "population_sizes": populations,
            "uniform_group_budgets": uniform,
            "issued_sweep": sweep,
        }
    )


def budget_allocation_parameter_hash(
    *,
    population_sizes: Sequence[int],
    uniform_group_budgets: Sequence[int],
    group_budgets: Sequence[int],
    frozen_total_fes: int,
) -> str:
    populations = _integer_vector(population_sizes, "population_sizes")
    uniform = _integer_vector(uniform_group_budgets, "uniform_group_budgets")
    budgets = _integer_vector(group_budgets, "group_budgets")
    total = _integer(frozen_total_fes, "frozen_total_fes", minimum=1)
    if not len(populations) == len(uniform) == len(budgets):
        raise ValueError("budget action vectors must be aligned")
    return _canonical_sha256(
        {
            "group_budgets": budgets,
            "population_sizes": populations,
            "uniform_group_budgets": uniform,
            "frozen_total_fes": total,
            "max_uniform_multiplier": FROZEN_BUDGET_MAX_UNIFORM_MULTIPLIER,
        }
    )


@dataclass(frozen=True)
class BudgetAllocationAction:
    """One immutable next-sweep budget allocation compiled from Phase1 state."""

    problem_id: str
    run_seed: int
    checkpoint_fe: int
    dispatch_checkpoint_hash: str
    anchor_hash: str
    source_efficiency_ewma: tuple[float, ...]
    population_sizes: tuple[int, ...]
    uniform_group_budgets: tuple[int, ...]
    group_budgets: tuple[int, ...]
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

        efficiencies = _efficiency_vector(self.source_efficiency_ewma)
        populations = _integer_vector(self.population_sizes, "population_sizes")
        uniform = _integer_vector(
            self.uniform_group_budgets,
            "uniform_group_budgets",
        )
        budgets = _integer_vector(self.group_budgets, "group_budgets")
        if not len(efficiencies) == len(populations) == len(uniform) == len(budgets):
            raise ValueError("budget action vectors must be non-empty and aligned")
        object.__setattr__(self, "source_efficiency_ewma", efficiencies)
        object.__setattr__(self, "population_sizes", populations)
        object.__setattr__(self, "uniform_group_budgets", uniform)
        object.__setattr__(self, "group_budgets", budgets)

        total = _integer(self.frozen_total_fes, "frozen_total_fes", minimum=1)
        if sum(uniform) != total or sum(budgets) != total:
            raise ValueError("budget action must preserve the frozen FE total")
        for budget, population, uniform_budget in zip(
            budgets,
            populations,
            uniform,
            strict=True,
        ):
            if uniform_budget < population:
                raise ValueError("uniform budgets must cover one population per group")
            if budget < population:
                raise ValueError("each group budget must cover one population")
            if budget > FROZEN_BUDGET_MAX_UNIFORM_MULTIPLIER * uniform_budget:
                raise ValueError("each group budget must respect the 3x uniform cap")

        issued = _integer(self.issued_sweep, "issued_sweep")
        target = _integer(self.target_sweep, "target_sweep")
        ttl = _integer(self.ttl_sweeps, "ttl_sweeps", minimum=1)
        expires = _integer(self.expires_sweep, "expires_sweep")
        if ttl != 1:
            raise ValueError("budget actions must have ttl_sweeps=1")
        if target != issued + 1:
            raise ValueError("target_sweep must be the next sweep")
        if expires != issued + ttl or expires != target:
            raise ValueError("expires_sweep must equal the target sweep")

        expected_anchor = budget_allocation_anchor_hash(
            problem_id=self.problem_id,
            run_seed=self.run_seed,
            checkpoint_fe=self.checkpoint_fe,
            dispatch_checkpoint_hash=self.dispatch_checkpoint_hash,
            source_efficiency_ewma=efficiencies,
            population_sizes=populations,
            uniform_group_budgets=uniform,
            issued_sweep=issued,
        )
        if self.anchor_hash != expected_anchor:
            raise ValueError("anchor_hash does not match the budget decision state")
        expected_parameters = budget_allocation_parameter_hash(
            population_sizes=populations,
            uniform_group_budgets=uniform,
            group_budgets=budgets,
            frozen_total_fes=total,
        )
        if self.parameter_hash != expected_parameters:
            raise ValueError("parameter_hash does not match the frozen budgets")

    def audit_payload(self) -> dict[str, object]:
        return {
            "action": FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
            "problem_id": self.problem_id,
            "run_seed": self.run_seed,
            "checkpoint_fe": self.checkpoint_fe,
            "dispatch_checkpoint_hash": self.dispatch_checkpoint_hash,
            "anchor_hash": self.anchor_hash,
            "source_efficiency_ewma": list(self.source_efficiency_ewma),
            "population_sizes": list(self.population_sizes),
            "uniform_group_budgets": list(self.uniform_group_budgets),
            "group_budgets": list(self.group_budgets),
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


@dataclass
class BudgetAllocationExecutionState:
    """Mutable, serializable lifecycle for one frozen budget action."""

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
    def for_action(cls, action: BudgetAllocationAction) -> BudgetAllocationExecutionState:
        return cls(action_hash=action.action_hash)

    def _validate_shape(self) -> None:
        _validate_hash(self.action_hash, "action_hash")
        if self.status not in _LIFECYCLE_STATUSES:
            raise ValueError("unsupported budget action lifecycle status")
        if self.consumed_sweep is not None:
            _integer(self.consumed_sweep, "consumed_sweep")
        if self.application_fe is not None:
            _integer(self.application_fe, "application_fe", minimum=1)
        if self.applied_group_budgets:
            _integer_vector(self.applied_group_budgets, "applied_group_budgets")
        if not isinstance(self.invalidation_reason, str):
            raise ValueError("invalidation_reason must be a string")

    def validate_for(self, action: BudgetAllocationAction) -> None:
        self._validate_shape()
        if self.action_hash != action.action_hash:
            raise ValueError("budget lifecycle does not match action_hash")
        if self.status == "issued":
            if (
                self.consumed_sweep is not None
                or self.application_fe is not None
                or self.applied_group_budgets
                or self.invalidation_reason
            ):
                raise ValueError("issued budget lifecycle contains outcome data")
        elif self.status == "consumed":
            if (
                self.consumed_sweep != action.target_sweep
                or self.application_fe is None
                or self.applied_group_budgets != action.group_budgets
                or self.invalidation_reason
            ):
                raise ValueError("consumed budget lifecycle is inconsistent")
        elif (
            self.consumed_sweep is not None
            or self.application_fe is not None
            or self.applied_group_budgets
            or not self.invalidation_reason
        ):
            raise ValueError("abstained budget lifecycle is inconsistent")

    def consume(
        self,
        action: BudgetAllocationAction,
        *,
        current_sweep: int,
        application_fe: int,
        dispatch_checkpoint_hash: str,
        anchor_hash: str,
    ) -> tuple[int, ...]:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued budget action can be consumed")
        sweep = _integer(current_sweep, "current_sweep")
        if sweep > action.expires_sweep:
            raise ValueError("budget action TTL expired")
        if sweep != action.target_sweep:
            raise ValueError("current_sweep does not match target_sweep")
        if (
            _validate_hash(dispatch_checkpoint_hash, "dispatch_checkpoint_hash")
            != action.dispatch_checkpoint_hash
        ):
            raise ValueError("dispatch_checkpoint_hash mismatch")
        if _validate_hash(anchor_hash, "anchor_hash") != action.anchor_hash:
            raise ValueError("anchor_hash mismatch")
        budgets = execute_budget_allocation_action(action)
        self.consumed_sweep = sweep
        self.application_fe = _integer(application_fe, "application_fe", minimum=1)
        self.applied_group_budgets = budgets
        self.status = "consumed"
        return budgets

    def abstain(self, action: BudgetAllocationAction, *, reason: str) -> None:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued budget action can abstain")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abstain reason must be a non-empty string")
        self.invalidation_reason = reason
        self.status = "abstained"

    def audit_payload(self, action: BudgetAllocationAction) -> dict[str, Any]:
        self.validate_for(action)
        return {
            "action": FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
            "action_hash": self.action_hash,
            "status": self.status,
            "consumed_sweep": self.consumed_sweep,
            "application_fe": self.application_fe,
            "applied_group_budgets": list(self.applied_group_budgets),
            "invalidation_reason": self.invalidation_reason,
        }

    def state_hash(self, action: BudgetAllocationAction) -> str:
        return _canonical_sha256(self.audit_payload(action))


def execute_budget_allocation_action(
    action: BudgetAllocationAction,
) -> tuple[int, ...]:
    """Return the exact stored allocation without consulting runtime evidence."""

    if not isinstance(action, BudgetAllocationAction):
        raise TypeError("action must be a BudgetAllocationAction")
    expected_hash = budget_allocation_parameter_hash(
        population_sizes=action.population_sizes,
        uniform_group_budgets=action.uniform_group_budgets,
        group_budgets=action.group_budgets,
        frozen_total_fes=action.frozen_total_fes,
    )
    if action.parameter_hash != expected_hash:
        raise ValueError("parameter_hash does not match the frozen budgets")
    return tuple(action.group_budgets)


def apply_budget_reallocation_action(
    action_name: str,
    group_budgets: Sequence[int],
    population_sizes: Sequence[int],
    frozen_total: int,
) -> tuple[int, ...]:
    """Validate and return the exact budget vector selected upstream."""

    if action_name != "efficiency_budget_reallocation":
        raise ValueError(f"unsupported budget reallocation action: {action_name!r}")
    raw_budgets = tuple(group_budgets)
    raw_populations = tuple(population_sizes)
    budgets = tuple(int(value) for value in raw_budgets)
    populations = tuple(int(value) for value in raw_populations)
    if not budgets or len(budgets) != len(populations):
        raise ValueError("budget action vectors must be non-empty and aligned")
    if any(
        isinstance(value, bool) or int(value) != value
        for value in (*raw_budgets, *raw_populations)
    ):
        raise ValueError("budget action values must be integers")
    if any(budget < population or population <= 0 for budget, population in zip(
        budgets, populations, strict=True
    )):
        raise ValueError("each group budget must cover one positive population")
    if isinstance(frozen_total, bool) or int(frozen_total) != frozen_total:
        raise ValueError("frozen_total must be an integer")
    if sum(budgets) != int(frozen_total):
        raise ValueError("group budgets must preserve the frozen FE total")
    return budgets
