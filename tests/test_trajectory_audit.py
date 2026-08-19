from __future__ import annotations

import pytest

from arac.analysis.trajectory_audit import audit_trajectories


def _run():
    return {
        "ctp": (10.0, 8.0, 7.0, 6.5),
        "smp": (10.0, 9.0, 8.5, 8.0),
        "gcb": (10.0, 9.5, 7.5, 5.0),
        "aor": (10.0, 9.5, 9.0, 8.5),
    }


def test_trajectory_audit_exposes_a_late_blooming_winner() -> None:
    audit = audit_trajectories(_run(), horizon_index=1)

    assert audit.run_count == 1
    assert audit.crossover_rate == 1.0
    assert audit.horizon_rank_correlation == pytest.approx(0.4)
    assert 0.0 <= audit.marginal_gain_stability <= 1.0


def test_trajectory_audit_aggregates_runs_and_rejects_drift() -> None:
    audit = audit_trajectories([_run(), _run()], horizon_index=1)
    assert audit.run_count == 2

    broken = _run()
    broken["aor"] = (1.0, 2.0)
    with pytest.raises(ValueError, match="same length"):
        audit_trajectories(broken, horizon_index=1)
