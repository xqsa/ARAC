"""Frozen contract for a full-space canonical Sep-CMA-ES continuation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from arac.actions.action_spec import ActionSpec


FULL_SPACE_SEP_CMA_ACTION = "full_space_sep_cma"
FULL_SPACE_DIMENSION = 1000
CANONICAL_SEP_CMA_PARAMETERIZATION = "ros_hansen_2008_pypop7"
CANONICAL_SEP_CMA_POPULATION_SIZE = 24
CANONICAL_SEP_CMA_REFERENCE_VERSION = (
    "pypop7-sepcmaes@67b29061d121cba9a5715897a2eb5d409df04c2d"
)
CANONICAL_SEP_CMA_PARAMETERS_HASH = (
    "935292123ceeb24517dcb36cf001f10d7a0639fbc28c51f112c2d247a07526c5"
)
NO_RESTART_POLICY = "none"
STRICT_IMPROVEMENT_ACCEPTANCE = "strict_improvement"
TRIGGER_SCOPE_RELATION_DISPATCH = "relation_dispatch"
TRIGGER_SCOPE_PHASE_BOUNDARY = "phase_boundary"
TRIGGER_SCOPES = frozenset(
    {TRIGGER_SCOPE_RELATION_DISPATCH, TRIGGER_SCOPE_PHASE_BOUNDARY}
)

FULL_SPACE_SEP_CMA_ACTION_SPEC = ActionSpec(
    name=FULL_SPACE_SEP_CMA_ACTION,
    semantic_surface="full_space_optimizer_continuation",
    parameter_names=(
        "initial_mean",
        "initial_sigma",
        "lower_bound",
        "upper_bound",
        "acceptance_fitness",
        "population_size",
        "budget_fes",
        "parameterization",
        "canonical_reference_version",
        "canonical_parameters_hash",
        "optimizer_seed",
        "restart_policy",
        "acceptance_rule",
    ),
)

_HASH_LENGTH = 64
_EXECUTION_STATUSES = frozenset({"issued", "running", "completed", "abstained"})


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


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _full_space_vector(values: object, name: str) -> tuple[float, ...]:
    try:
        vector = tuple(_finite(value, name) for value in values)  # type: ignore[union-attr]
    except TypeError as error:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence") from error
    if len(vector) != FULL_SPACE_DIMENSION:
        raise ValueError(f"{name} must contain exactly {FULL_SPACE_DIMENSION} values")
    return vector


def full_space_vector_hash(values: object) -> str:
    """Hash one exact 1000-dimensional vector with an explicit shape tag."""

    vector = _full_space_vector(values, "full-space vector")
    return _canonical_sha256(
        {"dimension": FULL_SPACE_DIMENSION, "values": vector}
    )


def full_space_sep_cma_anchor_hash(problem_id: str, values: object) -> str:
    """Bind the full-space action anchor to the problem and incumbent."""

    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem_id must be a non-empty string")
    vector = _full_space_vector(values, "full-space anchor")
    return _canonical_sha256(
        {
            "action": FULL_SPACE_SEP_CMA_ACTION,
            "problem_id": problem_id,
            "dimension": FULL_SPACE_DIMENSION,
            "anchor_values": vector,
        }
    )


@dataclass(frozen=True)
class FullSpaceSepCmaAction:
    """Immutable full-space continuation compiled before Phase2 execution."""

    problem_id: str
    run_seed: int
    checkpoint_fe: int
    dispatch_checkpoint_hash: str
    trigger_relation_hash: str
    anchor_hash: str
    initial_mean: tuple[float, ...]
    initial_mean_hash: str
    initial_state_hash: str
    initial_sigma: float
    lower_bound: float
    upper_bound: float
    acceptance_fitness: float
    population_size: int
    budget_fes: int
    parameterization: str
    canonical_reference_version: str
    canonical_parameters_hash: str
    optimizer_seed: int
    seed_namespace: str
    restart_policy: str
    issued_sweep: int
    target_sweep: int
    ttl_sweeps: int
    expires_sweep: int
    trigger_scope: str = TRIGGER_SCOPE_RELATION_DISPATCH
    acceptance_rule: str = STRICT_IMPROVEMENT_ACCEPTANCE

    def __post_init__(self) -> None:
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")
        if not isinstance(self.seed_namespace, str) or not self.seed_namespace.strip():
            raise ValueError("seed_namespace must be a non-empty string")
        _integer(self.run_seed, "run_seed")
        _integer(self.optimizer_seed, "optimizer_seed")
        _integer(self.checkpoint_fe, "checkpoint_fe")
        for name in (
            "dispatch_checkpoint_hash",
            "trigger_relation_hash",
            "anchor_hash",
            "initial_mean_hash",
            "initial_state_hash",
            "canonical_parameters_hash",
        ):
            _validate_hash(getattr(self, name), name)

        mean = _full_space_vector(self.initial_mean, "initial_mean")
        object.__setattr__(self, "initial_mean", mean)
        if self.initial_mean_hash != full_space_vector_hash(mean):
            raise ValueError("initial_mean_hash does not match initial_mean")
        if self.anchor_hash != full_space_sep_cma_anchor_hash(
            self.problem_id,
            mean,
        ):
            raise ValueError("anchor_hash does not match the full-space anchor")

        sigma = _finite(self.initial_sigma, "initial_sigma")
        if sigma <= 0.0:
            raise ValueError("initial_sigma must be strictly positive")
        object.__setattr__(self, "initial_sigma", sigma)
        lower = _finite(self.lower_bound, "lower_bound")
        upper = _finite(self.upper_bound, "upper_bound")
        if lower >= upper:
            raise ValueError("lower_bound must be smaller than upper_bound")
        # HCC samples without clipping; these bounds are provenance, not a mean gate.
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        acceptance_fitness = _finite(self.acceptance_fitness, "acceptance_fitness")
        if acceptance_fitness < 0.0:
            raise ValueError("acceptance_fitness must be non-negative")
        object.__setattr__(self, "acceptance_fitness", acceptance_fitness)

        population = _integer(self.population_size, "population_size", minimum=2)
        if population != CANONICAL_SEP_CMA_POPULATION_SIZE:
            raise ValueError("population_size must be the canonical 1000D population")
        if self.parameterization != CANONICAL_SEP_CMA_PARAMETERIZATION:
            raise ValueError("unsupported canonical Sep-CMA parameterization")
        if self.canonical_reference_version != CANONICAL_SEP_CMA_REFERENCE_VERSION:
            raise ValueError("unsupported canonical Sep-CMA reference version")
        if self.canonical_parameters_hash != CANONICAL_SEP_CMA_PARAMETERS_HASH:
            raise ValueError("canonical_parameters_hash does not match the pinned 1000D snapshot")

        budget = _integer(self.budget_fes, "budget_fes", minimum=1)
        if budget < population:
            raise ValueError("budget_fes must cover at least one population")
        if self.restart_policy != NO_RESTART_POLICY:
            raise ValueError("full-space Sep-CMA currently supports restart_policy='none'")
        if self.acceptance_rule != STRICT_IMPROVEMENT_ACCEPTANCE:
            raise ValueError("unsupported full-space Sep-CMA acceptance_rule")
        if self.trigger_scope not in TRIGGER_SCOPES:
            raise ValueError("unsupported full-space Sep-CMA trigger_scope")

        issued = _integer(self.issued_sweep, "issued_sweep")
        target = _integer(self.target_sweep, "target_sweep")
        ttl = _integer(self.ttl_sweeps, "ttl_sweeps", minimum=1)
        expires = _integer(self.expires_sweep, "expires_sweep")
        if ttl != 1:
            raise ValueError("full-space Sep-CMA actions must have ttl_sweeps=1")
        if target != issued + 1:
            raise ValueError("target_sweep must be the next sweep")
        if expires != issued + ttl:
            raise ValueError("expires_sweep must equal issued_sweep plus ttl_sweeps")

    def audit_payload(self) -> dict[str, object]:
        payload = {
            "action": FULL_SPACE_SEP_CMA_ACTION,
            "problem_id": self.problem_id,
            "run_seed": self.run_seed,
            "checkpoint_fe": self.checkpoint_fe,
            "anchor_hash": self.anchor_hash,
            "initial_mean": list(self.initial_mean),
            "initial_mean_hash": self.initial_mean_hash,
            "initial_state_hash": self.initial_state_hash,
            "initial_sigma": self.initial_sigma,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "acceptance_fitness": self.acceptance_fitness,
            "population_size": self.population_size,
            "budget_fes": self.budget_fes,
            "parameterization": self.parameterization,
            "canonical_reference_version": self.canonical_reference_version,
            "canonical_parameters_hash": self.canonical_parameters_hash,
            "optimizer_seed": self.optimizer_seed,
            "seed_namespace": self.seed_namespace,
            "restart_policy": self.restart_policy,
            "issued_sweep": self.issued_sweep,
            "target_sweep": self.target_sweep,
            "ttl_sweeps": self.ttl_sweeps,
            "expires_sweep": self.expires_sweep,
            "acceptance_rule": self.acceptance_rule,
        }
        # Keep the relation-dispatch payload byte-for-byte compatible with v2
        # artifacts.  A phase-boundary action has a different, explicit scope
        # and binds its trigger to a global checkpoint instead of a relation.
        payload["dispatch_checkpoint_hash"] = self.dispatch_checkpoint_hash
        if self.trigger_scope == TRIGGER_SCOPE_RELATION_DISPATCH:
            payload["trigger_relation_hash"] = self.trigger_relation_hash
        else:
            payload["trigger_scope"] = self.trigger_scope
            payload["trigger_context_hash"] = self.trigger_relation_hash
        return payload

    @property
    def action_hash(self) -> str:
        return _canonical_sha256(self.audit_payload())


@dataclass
class FullSpaceSepCmaExecutionState:
    """Mutable lifecycle audit; numerical state remains owned by vendor Sep-CMA."""

    action_hash: str
    initial_state_hash: str
    status: str = "issued"
    consumed_fes: int = 0
    started_fe: int | None = None
    completed_fe: int | None = None
    final_state_hash: str | None = None
    invalidation_reason: str = ""

    def __post_init__(self) -> None:
        self._validate_shape()

    @classmethod
    def for_action(cls, action: FullSpaceSepCmaAction) -> FullSpaceSepCmaExecutionState:
        return cls(
            action_hash=action.action_hash,
            initial_state_hash=action.initial_state_hash,
        )

    def _validate_shape(self) -> None:
        _validate_hash(self.action_hash, "action_hash")
        _validate_hash(self.initial_state_hash, "initial_state_hash")
        if self.status not in _EXECUTION_STATUSES:
            raise ValueError("unsupported full-space Sep-CMA execution status")
        _integer(self.consumed_fes, "consumed_fes")
        if self.started_fe is not None:
            _integer(self.started_fe, "started_fe")
        if self.completed_fe is not None:
            _integer(self.completed_fe, "completed_fe")
        if self.final_state_hash is not None:
            _validate_hash(self.final_state_hash, "final_state_hash")
        if not isinstance(self.invalidation_reason, str):
            raise ValueError("invalidation_reason must be a string")

    def validate_for(self, action: FullSpaceSepCmaAction) -> None:
        self._validate_shape()
        if self.action_hash != action.action_hash:
            raise ValueError("execution state does not match action_hash")
        if self.initial_state_hash != action.initial_state_hash:
            raise ValueError("execution state does not match initial_state_hash")
        if self.consumed_fes > action.budget_fes:
            raise ValueError("execution state exceeds the frozen FE budget")

        if self.status == "issued":
            if any(
                value is not None
                for value in (self.started_fe, self.completed_fe, self.final_state_hash)
            ) or self.consumed_fes or self.invalidation_reason:
                raise ValueError("issued execution state contains runtime outcome data")
        elif self.status == "running":
            if (
                self.started_fe is None
                or self.completed_fe is not None
                or self.final_state_hash is not None
                or self.invalidation_reason
            ):
                raise ValueError("running execution state is inconsistent")
        elif self.status == "completed":
            if (
                self.started_fe is None
                or self.completed_fe is None
                or self.final_state_hash is None
                or self.invalidation_reason
                or self.consumed_fes != action.budget_fes
                or self.completed_fe - self.started_fe != self.consumed_fes
            ):
                raise ValueError("completed execution state is inconsistent")
        elif (
            self.started_fe is not None
            or self.completed_fe is not None
            or self.final_state_hash is not None
            or self.consumed_fes
            or not self.invalidation_reason
        ):
            raise ValueError("abstained execution state is inconsistent")

    def start(
        self,
        action: FullSpaceSepCmaAction,
        *,
        current_fe: int,
        current_sweep: int,
        dispatch_checkpoint_hash: str,
        trigger_relation_hash: str,
        anchor_hash: str,
        trigger_scope: str | None = None,
    ) -> None:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued action can start")
        observed_fe = _integer(current_fe, "current_fe")
        observed_sweep = _integer(current_sweep, "current_sweep")
        observed_checkpoint = _validate_hash(
            dispatch_checkpoint_hash,
            "dispatch_checkpoint_hash",
        )
        observed_relation = _validate_hash(
            trigger_relation_hash,
            "trigger_relation_hash",
        )
        observed_scope = (
            action.trigger_scope if trigger_scope is None else trigger_scope
        )
        if observed_scope not in TRIGGER_SCOPES:
            raise ValueError("unsupported full-space Sep-CMA trigger_scope")
        observed_anchor = _validate_hash(anchor_hash, "anchor_hash")
        if observed_fe != action.checkpoint_fe:
            raise ValueError("current_fe does not match checkpoint_fe")
        if observed_sweep > action.expires_sweep:
            raise ValueError("full-space Sep-CMA action TTL expired")
        if observed_sweep != action.target_sweep:
            raise ValueError("current_sweep does not match target_sweep")
        if observed_checkpoint != action.dispatch_checkpoint_hash:
            raise ValueError("dispatch_checkpoint_hash mismatch")
        if observed_relation != action.trigger_relation_hash:
            raise ValueError(
                "trigger_relation_hash mismatch"
                if action.trigger_scope == TRIGGER_SCOPE_RELATION_DISPATCH
                else "trigger_context_hash mismatch"
            )
        if observed_scope != action.trigger_scope:
            raise ValueError("trigger_scope mismatch")
        if observed_anchor != action.anchor_hash:
            raise ValueError("anchor_hash mismatch")
        self.started_fe = observed_fe
        self.status = "running"

    def complete(
        self,
        action: FullSpaceSepCmaAction,
        *,
        consumed_fes: int,
        completed_fe: int,
        final_state_hash: str,
    ) -> None:
        self.validate_for(action)
        if self.status != "running" or self.started_fe is None:
            raise ValueError("only a running action can complete")
        consumed = _integer(consumed_fes, "consumed_fes", minimum=1)
        completed = _integer(completed_fe, "completed_fe")
        final_hash = _validate_hash(final_state_hash, "final_state_hash")
        if consumed != action.budget_fes:
            raise ValueError("completed action must consume its frozen FE budget")
        if completed - self.started_fe != consumed:
            raise ValueError("completed_fe does not match consumed_fes")
        self.consumed_fes = consumed
        self.completed_fe = completed
        self.final_state_hash = final_hash
        self.status = "completed"

    def abstain(self, action: FullSpaceSepCmaAction, *, reason: str) -> None:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued action can abstain")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abstain reason must be a non-empty string")
        self.invalidation_reason = reason
        self.status = "abstained"

    def audit_payload(self, action: FullSpaceSepCmaAction) -> dict[str, Any]:
        self.validate_for(action)
        return {
            "action": FULL_SPACE_SEP_CMA_ACTION,
            "action_hash": self.action_hash,
            "initial_state_hash": self.initial_state_hash,
            "status": self.status,
            "consumed_fes": self.consumed_fes,
            "started_fe": self.started_fe,
            "completed_fe": self.completed_fe,
            "final_state_hash": self.final_state_hash,
            "invalidation_reason": self.invalidation_reason,
        }

    def state_hash(self, action: FullSpaceSepCmaAction) -> str:
        return _canonical_sha256(self.audit_payload(action))


__all__ = [
    "CANONICAL_SEP_CMA_PARAMETERIZATION",
    "CANONICAL_SEP_CMA_POPULATION_SIZE",
    "CANONICAL_SEP_CMA_PARAMETERS_HASH",
    "CANONICAL_SEP_CMA_REFERENCE_VERSION",
    "FULL_SPACE_DIMENSION",
    "FULL_SPACE_SEP_CMA_ACTION",
    "FULL_SPACE_SEP_CMA_ACTION_SPEC",
    "TRIGGER_SCOPE_PHASE_BOUNDARY",
    "TRIGGER_SCOPE_RELATION_DISPATCH",
    "TRIGGER_SCOPES",
    "FullSpaceSepCmaAction",
    "FullSpaceSepCmaExecutionState",
    "full_space_sep_cma_anchor_hash",
    "full_space_vector_hash",
]
