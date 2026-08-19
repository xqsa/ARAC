"""Deterministic delayed-commit rules over Phase-II probe evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from arac.analysis.trajectory_audit import ACTION_NAMES, _stability, _validate_run


@dataclass(frozen=True)
class CommitDecision:
    """A policy decision that may deliberately abstain from committing."""

    action_name: str | None
    reason: str
    observed_fes: int
    exploration_floor_fes: int
    relative_margin: float
    leader_stability: float


def decide_delayed_commit(
    trajectories: Mapping[str, Sequence[float]],
    *,
    horizon_index: int,
    observed_fes: int,
    exploration_floor_fes: int,
    min_relative_margin: float = 0.05,
    min_leader_stability: float = 0.20,
) -> CommitDecision:
    """Commit only after a protected floor, margin, and stable leader evidence.

    The rule is intentionally conservative: returning ``action_name=None`` is
    an explicit abstention and callers should extend the common-anchor probe.
    It makes no claim of terminal optimality or cross-suite generalization.
    """

    if isinstance(observed_fes, bool) or not isinstance(observed_fes, int) or observed_fes < 0:
        raise ValueError("observed_fes must be a non-negative integer")
    if (
        isinstance(exploration_floor_fes, bool)
        or not isinstance(exploration_floor_fes, int)
        or exploration_floor_fes <= 0
    ):
        raise ValueError("exploration_floor_fes must be positive")
    if not 0.0 <= min_relative_margin or not 0.0 <= min_leader_stability:
        raise ValueError("commit thresholds must be non-negative")
    _validate_run(trajectories, horizon_index)
    horizon_values = [float(trajectories[action][horizon_index]) for action in ACTION_NAMES]
    leader_index = min(range(len(ACTION_NAMES)), key=lambda index: (horizon_values[index], index))
    ordered = sorted(horizon_values)
    leader = ordered[0]
    runner_up = ordered[1]
    relative_margin = (runner_up - leader) / (abs(runner_up) + np.finfo(float).eps)
    leader_stability = _stability(trajectories[ACTION_NAMES[leader_index]], horizon_index)

    if observed_fes < exploration_floor_fes:
        return CommitDecision(
            None,
            "protected_exploration_floor",
            observed_fes,
            exploration_floor_fes,
            float(relative_margin),
            float(leader_stability),
        )
    if relative_margin < min_relative_margin:
        return CommitDecision(
            None,
            "insufficient_margin",
            observed_fes,
            exploration_floor_fes,
            float(relative_margin),
            float(leader_stability),
        )
    if leader_stability < min_leader_stability:
        return CommitDecision(
            None,
            "unstable_leader_gain",
            observed_fes,
            exploration_floor_fes,
            float(relative_margin),
            float(leader_stability),
        )
    return CommitDecision(
        ACTION_NAMES[leader_index],
        "stable_margin",
        observed_fes,
        exploration_floor_fes,
        float(relative_margin),
        float(leader_stability),
    )


__all__ = ["CommitDecision", "decide_delayed_commit"]
