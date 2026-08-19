"""Contracts for the fresh-seed paired runner's anytime metrics."""

from __future__ import annotations

import math

from experiments.oc_phase_aware_gate51c import (
    ANYTIME_CHECKPOINTS,
    _anytime_points,
    _log_error_auc,
    _sample_anytime,
)


def test_anytime_points_use_phase1_and_preserve_strict_best() -> None:
    result = {
        "segments": [
            {"cumulative_phase2_fes": 100, "error_after": 80.0},
            {"cumulative_phase2_fes": 500, "error_after": 90.0},
            {"cumulative_phase2_fes": 900, "error_after": 20.0},
        ]
    }
    points = _anytime_points(result, 100.0)
    assert points[0] == {"total_fes": 180_000.0, "error": 100.0}
    assert [point["error"] for point in points] == [100.0, 80.0, 80.0, 20.0]


def test_anytime_sampling_has_fixed_protocol_grid() -> None:
    points = _anytime_points(
        {"segments": [{"cumulative_phase2_fes": 2_820_000, "error_after": 2.0}]},
        100.0,
    )
    sampled = _sample_anytime(points)
    assert tuple(int(key) for key in sampled) == ANYTIME_CHECKPOINTS
    assert all(value == 100.0 for key, value in sampled.items() if int(key) < 3_000_000)


def test_log_error_auc_is_finite_and_decreases_for_better_curve() -> None:
    worse = _anytime_points(
        {"segments": [{"cumulative_phase2_fes": 2_820_000, "error_after": 10.0}]},
        100.0,
    )
    better = _anytime_points(
        {"segments": [{"cumulative_phase2_fes": 2_820_000, "error_after": 1.0}]},
        100.0,
    )
    worse_auc = _log_error_auc(worse)
    better_auc = _log_error_auc(better)
    assert math.isfinite(worse_auc) and math.isfinite(better_auc)
    assert better_auc < worse_auc
