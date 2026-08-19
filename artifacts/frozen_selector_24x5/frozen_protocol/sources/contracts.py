"""Shared identity-blind contracts for ARAC phases and actions."""

from __future__ import annotations

from dataclasses import dataclass
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
        if self.total_budget_fes <= 0 or not 0 < self.phase1_fes < self.total_budget_fes:
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

    def __post_init__(self) -> None:
        if self.action_name not in ACTION_NAMES:
            raise ValueError("unsupported action")
        if self.problem.dimension != len(self.checkpoint.incumbent):
            raise ValueError("problem and checkpoint dimensions disagree")
        if self.ledger.count != self.checkpoint.phase1_fes:
            raise ValueError("ledger is not positioned at the checkpoint FE")
        if self.ledger.total_budget != self.checkpoint.total_budget_fes:
            raise ValueError("ledger and checkpoint budgets disagree")
        if isinstance(self.action_seed, bool) or self.action_seed < 0:
            raise ValueError("action_seed must be a non-negative integer")


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


class ActionExecutor(Protocol):
    def execute(self, context: ActionContext) -> ActionResult: ...


__all__ = [
    "ACTION_NAMES",
    "ACTION_RESULT_SCHEMA",
    "ActionContext",
    "ActionExecutor",
    "ActionResult",
    "CHECKPOINT_SCHEMA",
    "PhaseCheckpoint",
    "RelationEvidence",
    "canonical_sha256",
]
