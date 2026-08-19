from __future__ import annotations

import numpy as np
import pytest

from arac.runtime.branches import CommonAnchorProbe
from arac.runtime.contracts import ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import Phase2StateError
from arac.benchmarks.aob import OptimizationProblem


def _context() -> ActionContext:
    dimension = 6
    problem = OptimizationProblem(
        objective=lambda values: np.sum(np.asarray(values, dtype=float) ** 2, axis=-1),
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )
    checkpoint = PhaseCheckpoint(
        protocol="common-anchor-test-v1",
        run_seed=7,
        total_budget_fes=34,
        phase1_fes=4,
        incumbent=(1.0,) * dimension,
        incumbent_error=float(dimension),
        feature_names=("line_high_frequency_fraction_median",),
        feature_values=(0.4,),
        blocks=((0, 1), (2, 3), (4, 5)),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    return ActionContext("aor", checkpoint, problem, ledger, action_seed=23)


def test_common_anchor_probe_isolates_ledgers_and_restores_prefixes() -> None:
    probe = CommonAnchorProbe(_context(), branch_budget_fes=6)
    probe.advance(3)
    probe.advance(3)
    snapshot = probe.snapshot()

    uninterrupted = CommonAnchorProbe(_context(), branch_budget_fes=6)
    uninterrupted.advance(6)

    restored = CommonAnchorProbe.restore(_context(), snapshot)
    assert snapshot.snapshot_hash == restored.snapshot().snapshot_hash
    assert probe.best_action() == uninterrupted.best_action()
    for action in probe.states:
        assert probe.states[action].context.ledger.count == 10
        assert (
            probe.states[action].context.ledger.best_error
            == uninterrupted.states[action].context.ledger.best_error
        )
        assert (
            restored.states[action].context.ledger.best_x.tolist()
            == probe.states[action].context.ledger.best_x.tolist()
        )


def test_common_anchor_exposes_identity_blind_best_so_far_trajectories() -> None:
    probe = CommonAnchorProbe(_context(), branch_budget_fes=8)
    probe.advance(8)

    trajectories = probe.trajectories

    assert tuple(trajectories) == ("ctp", "smp", "gcb", "aor")
    assert all(len(trace) == 9 for trace in trajectories.values())
    assert all(trace[0] == 6.0 for trace in trajectories.values())
    assert all(
        state.consumed_fes == 8
        for state in probe.states.values()
    )


def test_common_anchor_trajectory_readout_requires_a_shared_position() -> None:
    probe = CommonAnchorProbe(_context(), branch_budget_fes=8)
    probe.states["aor"].step(1)

    with pytest.raises(Phase2StateError, match="shared position"):
        _ = probe.trajectories


def test_common_anchor_commit_exposes_only_the_selected_branch() -> None:
    probe = CommonAnchorProbe(_context(), branch_budget_fes=5)
    probe.advance(5)
    selected = probe.commit("smp")
    selected.step(25)

    assert selected.complete
    assert selected.result().action_name == "smp"
    assert probe.committed_action == "smp"
    with pytest.raises(Phase2StateError, match="already committed"):
        probe.advance(1)


def test_common_anchor_requires_equal_complete_probe_before_commit() -> None:
    probe = CommonAnchorProbe(_context(), branch_budget_fes=5)
    with pytest.raises(Phase2StateError, match="finish"):
        probe.commit("aor")
    with pytest.raises(ValueError, match="unsupported"):
        probe.commit("unknown")


def test_common_anchor_can_extend_a_shared_probe_without_replaying_fe() -> None:
    probe = CommonAnchorProbe(_context(), branch_budget_fes=4)
    probe.advance(2)
    snapshot = probe.snapshot()

    restored = CommonAnchorProbe.restore(_context(), snapshot)
    restored.extend(7)
    restored.advance(5)

    assert restored.complete
    assert all(state.context.ledger.count == 11 for state in restored.states.values())
