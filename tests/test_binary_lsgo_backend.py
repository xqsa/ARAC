from dataclasses import asdict

import pytest

from arac.backends.binary_lsgo import (
    BinaryLsgoExecutionRequest,
    BinaryLsgoGroupStats,
    BinaryLsgoSnapshot,
    build_binary_lsgo_evidence_profile,
)
from arac.benchmarks.binary_lsgo import BinaryLsgoSpec, generate_binary_lsgo
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS


def small_problem():
    return generate_binary_lsgo(
        BinaryLsgoSpec("small", 40, 8, 2, 5, True, 0.5, 0.5, 0.5, 0.5, 11)
    )


def test_request_rejects_invalid_budget_or_seed():
    problem = small_problem()
    with pytest.raises(ValueError, match="total_fes"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=1, total_fes=1)
    with pytest.raises(ValueError, match="phase_one_fraction"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=1, phase_one_fraction=1.0)
    with pytest.raises(ValueError, match="optimizer_seed"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=-1)


def test_snapshot_converts_to_runtime_legal_evidence():
    problem = small_problem()
    stats = tuple(
        BinaryLsgoGroupStats(group_index=index, proposed=2, accepted=index % 2, gain=float(index))
        for index in range(len(problem.topology.groups))
    )
    snapshot = BinaryLsgoSnapshot(
        run_id="test",
        lane_id="arac_policy",
        problem_id=problem.spec.problem_id,
        optimizer_seed=9,
        consumed_fes=8,
        total_fes=40,
        group_stats=stats,
        shared_proposals=4,
        rejected_shared_proposals=1,
        conflicting_shared_variables=2,
        rank_stability=0.75,
        topology=problem.topology,
    )
    evidence = build_binary_lsgo_evidence_profile(snapshot)
    assert evidence.problem_id == "small"
    assert evidence.budget_remaining_ratio == pytest.approx(0.8)
    assert evidence.harmful_coord_score == pytest.approx(0.25)
    assert set(asdict(evidence)).isdisjoint(FORBIDDEN_RUNTIME_FIELDS)
