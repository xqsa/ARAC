from __future__ import annotations

from arac.analysis.delayed_commit import decide_delayed_commit


def _trajectories():
    return {
        "ctp": (10.0, 9.9, 9.8, 9.7, 9.6),
        "smp": (10.0, 9.8, 9.6, 9.4, 9.2),
        "gcb": (10.0, 9.7, 9.4, 9.1, 8.8),
        "aor": (10.0, 9.5, 9.0, 8.5, 8.0),
    }


def test_delayed_commit_protects_the_exploration_floor() -> None:
    decision = decide_delayed_commit(
        _trajectories(),
        horizon_index=2,
        observed_fes=8,
        exploration_floor_fes=12,
    )
    assert decision.action_name is None
    assert decision.reason == "protected_exploration_floor"


def test_delayed_commit_returns_a_clear_stable_leader() -> None:
    decision = decide_delayed_commit(
        _trajectories(),
        horizon_index=2,
        observed_fes=12,
        exploration_floor_fes=12,
        min_relative_margin=0.03,
    )
    assert decision.action_name == "aor"
    assert decision.reason == "stable_margin"
    assert decision.relative_margin > 0.0


def test_delayed_commit_abstains_when_margin_is_small() -> None:
    trajectories = _trajectories()
    trajectories["aor"] = (10.0, 9.7, 9.41, 9.2, 9.0)
    decision = decide_delayed_commit(
        trajectories,
        horizon_index=2,
        observed_fes=12,
        exploration_floor_fes=12,
        min_relative_margin=0.05,
    )
    assert decision.action_name is None
    assert decision.reason == "insufficient_margin"
