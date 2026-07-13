"""Pure downstream recovery checkpoint decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from arac.policy.action_trust_policy import normalized_objective_credit

RecoveryStatus = Literal["committed", "restored", "preempted_restored"]


def _frozen_candidate(value: np.ndarray) -> np.ndarray:
    candidate = np.asarray(value, dtype=float)
    if candidate.ndim != 1:
        raise ValueError("recovery candidate must have one-dimensional shape")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("recovery candidate must be finite")
    candidate = candidate.copy()
    candidate.setflags(write=False)
    return candidate


@dataclass(frozen=True)
class RecoveryCheckpoint:
    candidate: np.ndarray
    fitness: float

    def __post_init__(self) -> None:
        fitness = float(self.fitness)
        if not math.isfinite(fitness):
            raise ValueError("checkpoint fitness must be finite")
        object.__setattr__(self, "candidate", _frozen_candidate(self.candidate))
        object.__setattr__(self, "fitness", fitness)


@dataclass(frozen=True)
class RecoveryResolution:
    candidate: np.ndarray
    fitness: float
    effective_delta: float
    status: RecoveryStatus
    restored: bool
    recovery_credit: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate", _frozen_candidate(self.candidate))


def make_recovery_checkpoint(
    candidate: np.ndarray,
    fitness: float,
) -> RecoveryCheckpoint:
    return RecoveryCheckpoint(candidate=candidate, fitness=fitness)


def resolve_recovery_checkpoint(
    checkpoint: RecoveryCheckpoint,
    *,
    downstream_candidate: np.ndarray,
    downstream_fitness: float,
) -> RecoveryResolution:
    candidate = _frozen_candidate(downstream_candidate)
    if candidate.shape != checkpoint.candidate.shape:
        raise ValueError("downstream candidate shape must match checkpoint shape")
    fitness = float(downstream_fitness)
    if not math.isfinite(fitness):
        raise ValueError("downstream fitness must be finite")

    credit = normalized_objective_credit(checkpoint.fitness, fitness)
    if fitness < checkpoint.fitness:
        return RecoveryResolution(
            candidate=candidate,
            fitness=fitness,
            effective_delta=checkpoint.fitness - fitness,
            status="committed",
            restored=False,
            recovery_credit=credit,
        )
    return RecoveryResolution(
        candidate=checkpoint.candidate,
        fitness=checkpoint.fitness,
        effective_delta=0.0,
        status="restored",
        restored=True,
        recovery_credit=credit,
    )


def preempt_recovery_checkpoint(
    checkpoint: RecoveryCheckpoint,
) -> RecoveryResolution:
    return RecoveryResolution(
        candidate=checkpoint.candidate,
        fitness=checkpoint.fitness,
        effective_delta=0.0,
        status="preempted_restored",
        restored=True,
        recovery_credit=None,
    )
