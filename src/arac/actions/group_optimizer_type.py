"""Explicit group-optimizer actions for Action Validation."""

from __future__ import annotations

from dataclasses import dataclass


FULL_CMAES_MODE = "full_cmaes"
DIAGONAL_COVARIANCE_MODE = "diagonal_covariance"
GROUP_OPTIMIZER_MODES = frozenset(
    {FULL_CMAES_MODE, DIAGONAL_COVARIANCE_MODE}
)


@dataclass(frozen=True)
class GroupOptimizerAction:
    """Frozen optimizer-type action consumed without evidence or reselection."""

    name: str
    diagonal_only: bool

    def __post_init__(self) -> None:
        if self.name not in GROUP_OPTIMIZER_MODES:
            raise ValueError("unsupported group optimizer action")
        if self.diagonal_only != (self.name == DIAGONAL_COVARIANCE_MODE):
            raise ValueError("group optimizer action parameters disagree")


def resolve_group_optimizer_action(mode: str) -> GroupOptimizerAction:
    """Compile one explicit mode into an immutable optimizer action."""

    if mode not in GROUP_OPTIMIZER_MODES:
        raise ValueError(f"unsupported group optimizer mode: {mode!r}")
    return GroupOptimizerAction(
        name=mode,
        diagonal_only=mode == DIAGONAL_COVARIANCE_MODE,
    )
