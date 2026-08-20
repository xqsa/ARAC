"""Targeted contract tests for the opt-in HPR-GCB scheduler."""

from __future__ import annotations

import numpy as np

from arac.benchmarks import OptimizationProblem
from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V5_3,
    OC_EPISODE_SCHEMA_V5_3,
    SCHEDULER_POLICY_V5_3,
    run_oc_episode_schedule_v4,
    PhaseAwareSchedulerConfig,
)
from arac.runtime.contracts import PhaseCheckpoint


DIMENSION = 24
BLOCKS = tuple(tuple(range(start, start + 6)) for start in range(0, DIMENSION, 6))


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        for block in BLOCKS:
            inner = batch[:, list(block)]
            result += 0.25 * np.sum(inner**2, axis=1) ** 2 / len(block)
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _flat_problem() -> OptimizationProblem:
    """A deterministic zero-gain landscape for release/reservation tests."""

    def objective(values):
        rows = np.asarray(values, dtype=float)
        return 1.0 if rows.ndim == 1 else np.ones(rows.shape[0])

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _checkpoint(total: int = 20_000, phase1: int = 500) -> PhaseCheckpoint:
    problem = _problem()
    incumbent = tuple(0.5 for _ in range(DIMENSION))
    return PhaseCheckpoint(
        protocol="hpr-gcb-unit",
        run_seed=3,
        total_budget_fes=total,
        phase1_fes=phase1,
        incumbent=incumbent,
        incumbent_error=float(problem.objective(np.asarray(incumbent))),
        feature_names=("log10_center_error", "line_high_frequency_fraction_median"),
        feature_values=(1.0, 0.4),
        blocks=BLOCKS,
        relations=(),
    )


def _flat_checkpoint(total: int = 20_000, phase1: int = 500) -> PhaseCheckpoint:
    incumbent = tuple(0.5 for _ in range(DIMENSION))
    return PhaseCheckpoint(
        protocol="hpr-flat-unit",
        run_seed=3,
        total_budget_fes=total,
        phase1_fes=phase1,
        incumbent=incumbent,
        incumbent_error=1.0,
        feature_names=("log10_center_error", "line_high_frequency_fraction_median"),
        feature_values=(1.0, 0.4),
        blocks=BLOCKS,
        relations=(),
    )


def _config(**overrides) -> PhaseAwareSchedulerConfig:
    values = dict(
        maturity_window_fes=800,
        revelation_horizon_fes=3_000,
        exploration_and_development_cap=0.80,
        exploitation_reserve_ratio=0.05,
        cold_start_probe_cap=0.25,
        probe_min_fes=200,
        segment_fes=1_500,
        calibration_ref="hpr-unit",
    )
    values.update(overrides)
    return PhaseAwareSchedulerConfig(**values)


def test_v5_policy_and_receipt_surface() -> None:
    # HPR-only v5.2 (adaptive off): the retired v5.0 entry's machinery,
    # driven through the audited v4 entry with an explicit v5.2 label.
    result = run_oc_episode_schedule_v4(
        _problem(),
        _checkpoint(),
        action_seed=20260845,
        config=_config(
            scheduler_version=DEFAULT_SCHEDULER_VERSION_V5_3,
            horizon_protected=True,
        ),
    )
    assert result.scheduler_version == DEFAULT_SCHEDULER_VERSION_V5_3
    assert result.scheduler_policy == SCHEDULER_POLICY_V5_3
    assert result.schema_version == OC_EPISODE_SCHEMA_V5_3
    assert result.terminal_fes == 20_000
    assert all(result.audit.values()), result.audit
    assert all(hasattr(receipt, "reservation_kind") for receipt in result.receipts)
    assert all(receipt.handoff_penalty >= 0 for receipt in result.receipts)


def test_v5_emits_horizon_reservation_after_a_non_material_exploit() -> None:
    result = run_oc_episode_schedule_v4(
        _flat_problem(),
        _flat_checkpoint(),
        action_seed=20260845,
        config=_config(
            scheduler_version=DEFAULT_SCHEDULER_VERSION_V5_3,
            horizon_protected=True,
            revelation_horizon_fes=2_500,
        ),
    )
    reservations = [r for r in result.receipts if r.reservation_kind == "horizon"]
    assert reservations, [r.__dict__ for r in result.receipts]
    assert all(r.grant_kind == "challenger" for r in reservations)
    assert all(r.window_fes > 0 for r in reservations)


def test_v5_marks_plateau_release_on_zero_gain_exploit() -> None:
    result = run_oc_episode_schedule_v4(
        _flat_problem(),
        _flat_checkpoint(),
        action_seed=20260845,
        config=_config(
            scheduler_version=DEFAULT_SCHEDULER_VERSION_V5_3,
            horizon_protected=True,
        ),
    )
    releases = [r for r in result.receipts if r.plateau_release]
    assert releases, [r.__dict__ for r in result.receipts]
    assert all(r.grant_kind == "exploit" and not r.material for r in releases)
