"""Targeted contracts for the v5.3 geometric verification ladder."""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks import OptimizationProblem
from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V5_3,
    OC_EPISODE_SCHEMA_V5_3,
    SCHEDULER_POLICY_V5_3,
    run_oc_episode_schedule_v4,
    run_oc_episode_schedule_v5,
    run_oc_episode_schedule_v5_1,
    run_oc_episode_schedule_v5_2,
    run_oc_episode_schedule_v5_3,
    PhaseAwareSchedulerConfig,
)
from arac.runtime.contracts import PhaseCheckpoint
from test_oc_episode_schedule_v5 import (
    DIMENSION,
    _config,
    _flat_checkpoint,
    _flat_problem,
    _checkpoint,
    _problem,
)

TOTAL_FES = 30_000


def _flip_problem(flip_after: int) -> OptimizationProblem:
    """Flat 1.0 until ``flip_after`` objective calls, then 0.5 forever.

    The grant window containing the flip is the only material segment;
    every later adoption arms a grace whose first flat window must not
    release its episode (the Gate 51c v5.2 S5 warm-up in miniature).
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


def _flip_checkpoint() -> PhaseCheckpoint:
    incumbent = tuple(0.5 for _ in range(DIMENSION))
    return PhaseCheckpoint(
        protocol="v5_3-flip-unit",
        run_seed=3,
        total_budget_fes=TOTAL_FES,
        phase1_fes=500,
        incumbent=incumbent,
        incumbent_error=1.0,
        feature_names=("log10_center_error",),
        feature_values=(1.0,),
        blocks=tuple(
            tuple(range(start, start + 6)) for start in range(0, DIMENSION, 6)
        ),
        relations=(),
    )


def test_v5_3_policy_surface_and_lock_ladder() -> None:
    result = run_oc_episode_schedule_v5_3(
        _problem(), _checkpoint(), action_seed=20260845, config=_config()
    )
    assert result.scheduler_version == DEFAULT_SCHEDULER_VERSION_V5_3
    assert result.scheduler_policy == SCHEDULER_POLICY_V5_3
    assert result.schema_version == OC_EPISODE_SCHEMA_V5_3
    assert all(result.audit.values()), result.audit
    locks = [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "adaptive_lock"
    ]
    assert locks
    lock = locks[0]
    # F2 fix: the lock verification starts at the bottom rung (w1), never
    # a full segment.
    assert lock.window_fes <= _config().maturity_window_fes
    assert lock.verification_rung == 0
    assert lock.material
    runways = [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "protected_runway"
        and receipt.episode == lock.episode
    ]
    assert runways
    first = runways[0]
    # A material lock promotes the episode to rung 1: the next window is
    # w(1) = min(2*w1, segment_fes), not v5.2's flat w1.
    assert first.verification_rung == 1
    assert first.window_fes == min(
        2 * _config().maturity_window_fes, _config().segment_fes
    )


def test_v5_3_flat_lock_releases_at_the_bottom_rung() -> None:
    # The flip lands inside gcb's material ticket; its lock window then
    # sees no residual value and must release immediately at rung 0
    # (bounded w1 exposure -- the pre-registered lock@rung-0 decision).
    result = run_oc_episode_schedule_v5_3(
        _flip_problem(17_000), _flip_checkpoint(), action_seed=20260845, config=_config()
    )
    assert all(result.audit.values()), result.audit
    assert result.final_error == pytest.approx(0.5)
    locks = [
        receipt
        for receipt in result.receipts
        if receipt.reservation_kind == "adaptive_lock"
    ]
    assert locks
    lock = locks[0]
    assert not lock.material
    assert lock.window_fes <= _config().maturity_window_fes
    assert lock.released and not lock.grace_consumed
    assert not lock.would_release_v5_2


def test_v5_3_adoption_grace_absorbs_the_warmup_flat_window() -> None:
    result = run_oc_episode_schedule_v5_3(
        _flip_problem(17_900), _flip_checkpoint(), action_seed=20260845, config=_config()
    )
    assert all(result.audit.values()), result.audit
    graces = [receipt for receipt in result.receipts if receipt.grace_consumed]
    assert graces
    adoptions = [h for h in result.handoffs if h.adopted]
    assert len(adoptions) >= len(graces)
    for receipt in graces:
        # The grace absorbed a genuinely flat window without releasing.
        assert not receipt.material
        assert not receipt.released
        # The v5.2 counterfactual fires exactly here: single-window
        # semantics would have released this episode.
        assert receipt.would_release_v5_2


def test_v5_3_counterfactual_flag_marks_only_v5_2_divergences() -> None:
    for flip in (17_000, 17_900, 25_000):
        result = run_oc_episode_schedule_v5_3(
            _flip_problem(flip), _flip_checkpoint(), action_seed=20260845, config=_config()
        )
        assert all(result.audit.values()), result.audit
        for receipt in result.receipts:
            if receipt.grant_kind != "exploit":
                assert not receipt.would_release_v5_2
                continue
            diverged = not receipt.material and (
                receipt.grace_consumed
                or (
                    receipt.reservation_kind == "protected_runway"
                    and receipt.window_fes > _config().maturity_window_fes
                )
            )
            assert receipt.would_release_v5_2 == diverged


def test_v5_3_material_horizon_promotion_stays_bounded() -> None:
    # flip=25,000: aor's horizon reservation is the material window and
    # its promotion verification (rung-1 sizing) stays flat and bounded.
    result = run_oc_episode_schedule_v5_3(
        _flip_problem(25_000), _flip_checkpoint(), action_seed=20260845, config=_config()
    )
    assert all(result.audit.values()), result.audit
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
    assert promotion.window_fes <= min(
        2 * _config().maturity_window_fes, _config().segment_fes
    )
    # A flat promotion at v5.2-identical sizing releases identically:
    # no counterfactual divergence.
    assert not promotion.material
    assert promotion.released and not promotion.would_release_v5_2


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


def test_retired_v5_entries_raise_instead_of_mislabelling() -> None:
    for retired in (
        run_oc_episode_schedule_v5,
        run_oc_episode_schedule_v5_1,
        run_oc_episode_schedule_v5_2,
    ):
        with pytest.raises(RuntimeError, match="is retired"):
            retired(
                _problem(), _checkpoint(), action_seed=20260845, config=_config()
            )


def test_v5_features_require_the_v5_3_version_label() -> None:
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
            config=_config(scheduler_version="v5.2"),
        )
 
