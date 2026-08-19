"""Shared identity-blind contracts for ARAC phases and actions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

from arac.benchmarks.aob import OptimizationProblem

if TYPE_CHECKING:
    from arac.runtime.ledger import EvaluationLedger


ACTION_NAMES = ("ctp", "smp", "gcb", "aor")
CHECKPOINT_SCHEMA = "arac-phase-checkpoint-v1"
ACTION_RESULT_SCHEMA = "arac-action-result-v1"
PHASE2_SNAPSHOT_SCHEMA = "arac-phase2-snapshot-v1"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _vector(values: tuple[float, ...], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite non-empty vector")
    return vector


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RelationEvidence:
    """Observed interaction between two evidence-derived variable blocks."""

    left_block: int
    right_block: int
    strength: float
    disagreement: float

    def __post_init__(self) -> None:
        if min(self.left_block, self.right_block) < 0 or self.left_block >= self.right_block:
            raise ValueError("relation block indices must be ordered and non-negative")
        if _finite(self.strength, "strength") < 0.0:
            raise ValueError("relation strength must be non-negative")
        if _finite(self.disagreement, "disagreement") < 0.0:
            raise ValueError("relation disagreement must be non-negative")


@dataclass(frozen=True)
class PhaseCheckpoint:
    """One immutable Phase-I state; benchmark identity is deliberately absent."""

    protocol: str
    run_seed: int
    total_budget_fes: int
    phase1_fes: int
    incumbent: tuple[float, ...]
    incumbent_error: float
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    blocks: tuple[tuple[int, ...], ...]
    relations: tuple[RelationEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.protocol:
            raise ValueError("checkpoint protocol must be non-empty")
        if isinstance(self.run_seed, bool) or self.run_seed < 0:
            raise ValueError("run_seed must be a non-negative integer")
        if self.total_budget_fes <= 0 or not 0 <= self.phase1_fes < self.total_budget_fes:
            raise ValueError("checkpoint FE boundary is invalid")
        incumbent = _vector(self.incumbent, "incumbent")
        _finite(self.incumbent_error, "incumbent_error")
        if not self.feature_names or len(self.feature_names) != len(self.feature_values):
            raise ValueError("feature names and values must be non-empty and aligned")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be unique")
        if not np.all(np.isfinite(np.asarray(self.feature_values, dtype=float))):
            raise ValueError("feature values must be finite")
        if not self.blocks or any(not block for block in self.blocks):
            raise ValueError("checkpoint blocks must be non-empty")
        flattened = tuple(index for block in self.blocks for index in block)
        if sorted(flattened) != list(range(incumbent.size)):
            raise ValueError("checkpoint blocks must partition every variable exactly once")
        for relation in self.relations:
            if relation.right_block >= len(self.blocks):
                raise ValueError("relation references an unknown block")

    @property
    def remaining_fes(self) -> int:
        return self.total_budget_fes - self.phase1_fes

    @property
    def overlap_relation_count(self) -> int:
        return len(self.relations)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "protocol": self.protocol,
            "run_seed": self.run_seed,
            "total_budget_fes": self.total_budget_fes,
            "phase1_fes": self.phase1_fes,
            "incumbent": list(self.incumbent),
            "incumbent_error": self.incumbent_error,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "blocks": [list(block) for block in self.blocks],
            "relations": [
                {
                    "left_block": item.left_block,
                    "right_block": item.right_block,
                    "strength": item.strength,
                    "disagreement": item.disagreement,
                }
                for item in self.relations
            ],
        }

    @property
    def checkpoint_hash(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True)
class ActionContext:
    """The only context accepted by every Phase-II action."""

    action_name: str
    checkpoint: PhaseCheckpoint
    problem: OptimizationProblem
    ledger: EvaluationLedger
    action_seed: int
    retain_trajectory: bool = True

    def __post_init__(self) -> None:
        if self.action_name not in ACTION_NAMES:
            raise ValueError("unsupported action")
        if self.problem.dimension != len(self.checkpoint.incumbent):
            raise ValueError("problem and checkpoint dimensions disagree")
        if not self.checkpoint.phase1_fes <= self.ledger.count <= self.checkpoint.total_budget_fes:
            raise ValueError("ledger is outside the checkpoint-to-terminal FE boundary")
        if self.ledger.total_budget != self.checkpoint.total_budget_fes:
            raise ValueError("ledger and checkpoint budgets disagree")
        if isinstance(self.action_seed, bool) or self.action_seed < 0:
            raise ValueError("action_seed must be a non-negative integer")
        if not isinstance(self.retain_trajectory, bool):
            raise ValueError("retain_trajectory must be a boolean")

    @property
    def phase2_consumed_fes(self) -> int:
        return self.ledger.count - self.checkpoint.phase1_fes


@dataclass(frozen=True)
class ActionResult:
    """Auditable terminal result returned by any ARAC action."""

    action_name: str
    checkpoint_hash: str
    action_seed: int
    consumed_fes: int
    terminal_fes: int
    incumbent: tuple[float, ...]
    final_error: float
    route: str
    optimizer_package: str
    optimizer_version: str

    def __post_init__(self) -> None:
        if self.action_name not in ACTION_NAMES:
            raise ValueError("unsupported action result")
        if len(self.checkpoint_hash) != 64:
            raise ValueError("checkpoint_hash must be SHA-256")
        if self.consumed_fes <= 0 or self.terminal_fes < self.consumed_fes:
            raise ValueError("action FE counts are invalid")
        _vector(self.incumbent, "incumbent")
        _finite(self.final_error, "final_error")
        if not self.route or not self.optimizer_package or not self.optimizer_version:
            raise ValueError("action provenance must be non-empty")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": ACTION_RESULT_SCHEMA,
            "action_name": self.action_name,
            "checkpoint_hash": self.checkpoint_hash,
            "action_seed": self.action_seed,
            "consumed_fes": self.consumed_fes,
            "terminal_fes": self.terminal_fes,
            "incumbent": list(self.incumbent),
            "final_error": self.final_error,
            "route": self.route,
            "optimizer_package": self.optimizer_package,
            "optimizer_version": self.optimizer_version,
        }

    @property
    def result_hash(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True)
class Phase2StepResult:
    """Auditable result of one exact Phase-II state transition."""

    action_name: str
    checkpoint_hash: str
    action_seed: int
    step_fes: int
    consumed_fes: int
    total_fes: int
    best_error: float
    complete: bool
    state_hash: str

    def __post_init__(self) -> None:
        if self.action_name not in ACTION_NAMES:
            raise ValueError("unsupported Phase-II action")
        if len(self.checkpoint_hash) != 64:
            raise ValueError("checkpoint_hash must be SHA-256")
        _nonnegative_integer(self.action_seed, "action_seed")
        if isinstance(self.step_fes, bool) or self.step_fes <= 0:
            raise ValueError("step_fes must be a positive integer")
        if isinstance(self.total_fes, bool) or self.total_fes <= 0:
            raise ValueError("total_fes must be a positive integer")
        if isinstance(self.consumed_fes, bool) or not 0 <= self.consumed_fes <= self.total_fes:
            raise ValueError("consumed_fes is outside the Phase-II budget")
        if self.step_fes > self.consumed_fes:
            raise ValueError("step_fes cannot exceed consumed_fes")
        if bool(self.complete) != (self.consumed_fes == self.total_fes):
            raise ValueError("complete flag disagrees with consumed_fes")
        _finite(self.best_error, "best_error")
        if len(self.state_hash) != 64:
            raise ValueError("state_hash must be SHA-256")


@dataclass(frozen=True)
class Phase2Snapshot:
    """Immutable, identity-blind state boundary for a resumable action."""

    action_name: str
    checkpoint_hash: str
    action_seed: int
    start_fes: int
    consumed_fes: int
    total_fes: int
    incumbent: tuple[float, ...]
    best_error: float
    state_payload: bytes
    state_hash: str
    schema_version: str = PHASE2_SNAPSHOT_SCHEMA
    _snapshot_hash_cache: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != PHASE2_SNAPSHOT_SCHEMA:
            raise ValueError("Phase-II snapshot schema drifted")
        if self.action_name not in ACTION_NAMES:
            raise ValueError("unsupported Phase-II action")
        if len(self.checkpoint_hash) != 64:
            raise ValueError("checkpoint_hash must be SHA-256")
        _nonnegative_integer(self.action_seed, "action_seed")
        if isinstance(self.start_fes, bool) or self.start_fes < 0:
            raise ValueError("start_fes must be a non-negative integer")
        if isinstance(self.total_fes, bool) or self.total_fes <= self.start_fes:
            raise ValueError("total_fes must exceed start_fes")
        if isinstance(self.consumed_fes, bool) or not 0 <= self.consumed_fes <= self.total_fes - self.start_fes:
            raise ValueError("snapshot consumed_fes is outside the Phase-II budget")
        _vector(self.incumbent, "snapshot incumbent")
        _finite(self.best_error, "snapshot best_error")
        if not isinstance(self.state_payload, bytes) or not self.state_payload:
            raise ValueError("state_payload must be non-empty bytes")
        payload_hash = hashlib.sha256(self.state_payload).hexdigest()
        if self.state_hash != payload_hash:
            raise ValueError("Phase-II snapshot state hash drifted")

    @property
    def snapshot_hash(self) -> str:
        cached = self._snapshot_hash_cache
        if cached is not None:
            return cached
        metadata = {
            "schema_version": self.schema_version,
            "action_name": self.action_name,
            "checkpoint_hash": self.checkpoint_hash,
            "action_seed": self.action_seed,
            "start_fes": self.start_fes,
            "consumed_fes": self.consumed_fes,
            "total_fes": self.total_fes,
            "incumbent": list(self.incumbent),
            "best_error": self.best_error,
            "state_hash": self.state_hash,
        }
        metadata_hash = canonical_sha256(metadata).encode("ascii")
        value = hashlib.sha256(metadata_hash + self.state_payload).hexdigest()
        object.__setattr__(self, "_snapshot_hash_cache", value)
        return value


class ActionExecutor(Protocol):
    def execute(self, context: ActionContext) -> ActionResult: ...

    def initialize(self, context: ActionContext) -> Phase2ActionState: ...

    def resume(self, context: ActionContext, snapshot: Phase2Snapshot) -> Phase2ActionState: ...


class ActionExecutionRegistry(Protocol):
    """Narrow terminal-action registry contract used by ARAC-Core."""

    @property
    def action_names(self) -> tuple[str, ...]: ...

    @property
    def allow_out_of_bounds(self) -> bool: ...

    def execute(self, context: ActionContext) -> ActionResult: ...


class Phase2ActionState(Protocol):
    """Common interface for an interruptible Phase-II action state."""

    action_name: str
    checkpoint_hash: str
    action_seed: int
    start_fes: int
    consumed_fes: int
    total_fes: int

    @property
    def complete(self) -> bool: ...

    def step(self, budget_fes: int) -> Phase2StepResult: ...

    def snapshot(self) -> Phase2Snapshot: ...


__all__ = [
    "ACTION_NAMES",
    "ACTION_RESULT_SCHEMA",
    "ActionContext",
    "ActionExecutor",
    "ActionExecutionRegistry",
    "ActionResult",
    "CHECKPOINT_SCHEMA",
    "PHASE2_SNAPSHOT_SCHEMA",
    "Phase2ActionState",
    "Phase2Snapshot",
    "Phase2StepResult",
    "PhaseCheckpoint",
    "RelationEvidence",
    "canonical_sha256",
]
