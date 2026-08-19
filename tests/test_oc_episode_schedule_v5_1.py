"""Targeted contracts for adaptive material-ticket exploration."""

from __future__ import annotations

import pytest

from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V5_1,
    SCHEDULER_POLICY_V5_1,
    run_oc_episode_schedule_v5_1,
    PhaseAwareSchedulerConfig,
)
from test_oc_episode_schedule_v5 import (
    _config,
    _flat_checkpoint,
    _flat_problem,
    _checkpoint,
    _problem,
)


def test_v5_1_earns_lock_from_a_material_maturity_ticket() -> None:
    result = run_oc_episode_schedule_v5_1(
        _problem(), _checkpoint(), action_seed=20260845, config=_config()
    )

    locks = [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "adaptive_lock"
    ]
    assert result.scheduler_version == DEFAULT_SCHEDULER_VERSION_V5_1
    assert result.scheduler_policy == SCHEDULER_POLICY_V5_1
    assert result.schema_version == "arac-oc-episode-schedule-v5.1"
    assert locks
    for lock in locks:
        previous = result.receipts[lock.segment_index - 1]
        assert previous.grant_kind == "ticket"
        assert previous.episode == lock.episode
        assert previous.material
        assert previous.progress_after["protocol_mature"] is True
    assert len(locks) == 1
    first_protected = next(
        (r.segment_index for r in result.receipts if r.reservation_kind == "protected_runway"),
        len(result.receipts),
    )
    assert all(r.grant_kind != "ticket" for r in result.receipts[first_protected:])
    assert all(result.audit.values()), result.audit


def test_v5_1_protected_runway_is_bounded_after_adaptive_lock() -> None:
    result = run_oc_episode_schedule_v5_1(
        _problem(), _checkpoint(), action_seed=20260845, config=_config()
    )
    locks = [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "adaptive_lock"
    ]
    assert locks
    for lock in locks:
        following = result.receipts[lock.segment_index + 1]
        if following.reservation_kind == "protected_runway":
            assert following.window_fes <= _config().maturity_window_fes


def test_v5_1_plateau_release_is_visible_in_receipt() -> None:
    result = run_oc_episode_schedule_v5_1(
        _flat_problem(), _flat_checkpoint(), action_seed=20260845, config=_config()
    )
    releases = [receipt for receipt in result.receipts if receipt.plateau_release]
    assert releases
    assert all(receipt.released for receipt in releases)


def test_v5_1_flat_landscape_releases_without_earned_lock() -> None:
    result = run_oc_episode_schedule_v5_1(
        _flat_problem(), _flat_checkpoint(), action_seed=20260845, config=_config()
    )

    assert not [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "adaptive_lock"
    ]
    assert any(receipt.plateau_release for receipt in result.receipts)
    assert all(result.audit.values()), result.audit


def test_adaptive_exploration_requires_horizon_protection() -> None:
    with pytest.raises(ValueError, match="requires horizon_protected"):
        PhaseAwareSchedulerConfig(
            maturity_window_fes=800,
            revelation_horizon_fes=3_000,
            exploration_and_development_cap=0.8,
            exploitation_reserve_ratio=0.05,
            adaptive_exploration=True,
            horizon_protected=False,
        )
