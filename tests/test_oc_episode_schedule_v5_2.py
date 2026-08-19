"""Targeted contracts for adaptive material-ticket exploration (v5.2)."""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks import OptimizationProblem
from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V5_2,
    OC_EPISODE_SCHEMA_V5_2,
    SCHEDULER_POLICY_V5_2,
    run_oc_episode_schedule_v4,
    run_oc_episode_schedule_v5,
    run_oc_episode_schedule_v5_1,
    run_oc_episode_schedule_v5_2,
    PhaseAwareSchedulerConfig,
)
from test_oc_episode_schedule_v5 import (
    DIMENSION,
    _config,
    _flat_checkpoint,
    _flat_problem,
    _checkpoint,
    _problem,
)


def _flip_problem(flip_after: int) -> OptimizationProblem:
    """Flat 1.0 until ``flip_after`` objective calls, then 0.5 forever.

    The grant window containing the flip is the only material segment.
    With ``flip_after`` inside AOR's horizon reservation, the discovery
    strands in the challenger lane unless the promotion fires -- the
    Gate 51c R2 mechanism in miniature.
    """

    state = {"calls": 0}

    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        state["calls"] += batch.shape[0]
        level = 1.0 if state["calls"] <= flip_after else 0.5
        out = np.full(batch.shape[0], level)
        return float(out[0]) if rows.ndim == 1 else out

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def test_v5_2_earns_lock_from_a_material_maturity_ticket() -> None:
    result = run_oc_episode_schedule_v5_2(
        _problem(), _checkpoint(), action_seed=20260845, config=_config()
    )

    locks = [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "adaptive_lock"
    ]
    assert result.scheduler_version == DEFAULT_SCHEDULER_VERSION_V5_2
    assert result.scheduler_policy == SCHEDULER_POLICY_V5_2
    assert result.schema_version == OC_EPISODE_SCHEMA_V5_2
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


def test_v5_2_protected_runway_is_bounded_after_adaptive_lock() -> None:
    result = run_oc_episode_schedule_v5_2(
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


def test_v5_2_plateau_release_is_visible_in_receipt() -> None:
    result = run_oc_episode_schedule_v5_2(
        _flat_problem(), _flat_checkpoint(), action_seed=20260845, config=_config()
    )
    releases = [receipt for receipt in result.receipts if receipt.plateau_release]
    assert releases
    assert all(receipt.released for receipt in releases)


def test_v5_2_flat_landscape_releases_without_earned_lock() -> None:
    result = run_oc_episode_schedule_v5_2(
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


def test_retired_v5_entry_raises_instead_of_mislabelling() -> None:
    with pytest.raises(RuntimeError, match="run_oc_episode_schedule_v5 is retired"):
        run_oc_episode_schedule_v5(
            _problem(), _checkpoint(), action_seed=20260845, config=_config()
        )


def test_retired_v5_1_entry_raises_instead_of_mislabelling() -> None:
    with pytest.raises(RuntimeError, match="run_oc_episode_schedule_v5_1 is retired"):
        run_oc_episode_schedule_v5_1(
            _problem(), _checkpoint(), action_seed=20260845, config=_config()
        )


def test_v5_features_require_the_v5_2_version_label() -> None:
    with pytest.raises(ValueError, match="require scheduler_version"):
        run_oc_episode_schedule_v4(
            _flat_problem(),
            _flat_checkpoint(),
            action_seed=20260845,
            config=_config(horizon_protected=True),
        )


def test_legacy_version_labels_are_not_producible() -> None:
    with pytest.raises(ValueError, match="cannot be produced by this tree"):
        run_oc_episode_schedule_v4(
            _flat_problem(),
            _flat_checkpoint(),
            action_seed=20260845,
            config=_config(scheduler_version="v5.1"),
        )


def test_v5_2_material_horizon_earns_one_bounded_promotion() -> None:
    result = run_oc_episode_schedule_v5_2(
        _flip_problem(18_200), _flat_checkpoint(), action_seed=20260845, config=_config()
    )
    material_horizons = [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "horizon" and receipt.material
    ]
    assert material_horizons
    promotions = [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "horizon_promotion"
    ]
    assert len(promotions) == 1
    promotion = promotions[0]
    assert promotion.grant_kind == "exploit"
    assert promotion.episode == material_horizons[0].episode
    assert promotion.segment_index == material_horizons[0].segment_index + 1
    assert promotion.window_fes <= 2 * _config().maturity_window_fes
    # The post-flip landscape is flat at the new level: the verification
    # window plateaus and releases immediately -- bounded loss.
    assert promotion.plateau_release and promotion.released
    assert all(result.audit.values()), result.audit


def test_v5_2_no_promotion_without_a_material_horizon() -> None:
    # One batch later the flip lands in an smp EXPLOIT window instead;
    # exploit-lane materiality must not mint a horizon promotion.
    result = run_oc_episode_schedule_v5_2(
        _flip_problem(18_800), _flat_checkpoint(), action_seed=20260845, config=_config()
    )
    assert not [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "horizon_promotion"
    ]
    assert any(
        receipt.grant_kind == "exploit" and receipt.material
        for receipt in result.receipts
    )
    assert all(result.audit.values()), result.audit
