from __future__ import annotations

import pytest

from arac.actions.group_optimizer_type import (
    DIAGONAL_COVARIANCE_MODE,
    FULL_CMAES_MODE,
    GroupOptimizerAction,
    resolve_group_optimizer_action,
)


def test_full_cmaes_action_is_the_native_default() -> None:
    action = resolve_group_optimizer_action(FULL_CMAES_MODE)

    assert action.name == FULL_CMAES_MODE
    assert action.diagonal_only is False


def test_diagonal_covariance_action_is_explicit() -> None:
    action = resolve_group_optimizer_action(DIAGONAL_COVARIANCE_MODE)

    assert action.name == DIAGONAL_COVARIANCE_MODE
    assert action.diagonal_only is True


def test_group_optimizer_action_rejects_inconsistent_parameters() -> None:
    with pytest.raises(ValueError, match="parameters disagree"):
        GroupOptimizerAction(name=FULL_CMAES_MODE, diagonal_only=True)


def test_group_optimizer_action_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported group optimizer mode"):
        resolve_group_optimizer_action("adaptive")
