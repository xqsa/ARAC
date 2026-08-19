from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence import Phase1OverlapAdapter, discover_overlap_sparse
from arac.runtime.contracts import PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger


GROUPS = ((0, 1, 2), (2, 3, 4))


def _problem(dimension: int = 5) -> OptimizationProblem:
    groups = GROUPS if dimension == 5 else ()

    def objective(values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        single = converted.ndim == 1
        batch = converted[np.newaxis, :] if single else converted
        result = np.sum(batch**2, axis=1)
        for group in groups:
            local = batch[:, group]
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    result += local[:, left] * local[:, right]
        return float(result[0]) if single else result

    return OptimizationProblem(
        objective=objective,
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )


def _anchors(dimension: int) -> tuple[tuple[float, ...], ...]:
    rng = np.random.default_rng(71)
    return tuple(tuple(row) for row in rng.uniform(-2.0, 2.0, size=(2, dimension)))


def _checkpoint(dimension: int) -> PhaseCheckpoint:
    return PhaseCheckpoint(
        protocol="sparse-overlap-discovery-test-v1",
        run_seed=3,
        total_budget_fes=10000,
        phase1_fes=1,
        incumbent=(0.0,) * dimension,
        incumbent_error=0.0,
        feature_names=("probe",),
        feature_values=(1.0,),
        blocks=tuple((variable,) for variable in range(dimension)),
    )


def test_sparse_probe_recovers_known_overlap_with_exact_fe() -> None:
    problem = _problem()
    rounds = 12
    bucket_size = 2
    bucket_count = 3
    screen = 2 * rounds * (1 + bucket_count + bucket_count * (bucket_count - 1) // 2)
    candidate_pairs = 6
    expected = screen + 2 * (1 + problem.dimension + candidate_pairs)
    ledger = EvaluationLedger(problem, expected)

    result = discover_overlap_sparse(
        problem,
        ledger,
        anchors=_anchors(5),
        step=0.25,
        run_seed=11,
        rounds=rounds,
        bucket_size=bucket_size,
        max_candidate_pairs=32,
    )

    assert result.complete
    assert result.complete_reason == "complete"
    assert result.evidence.groups == GROUPS
    assert result.evidence.memberships == ((0,), (0,), (0, 1), (1,), (1,))
    assert result.separated_pair_fraction == 1.0
    assert result.consumed_fes == ledger.count == result.expected_fes == expected
    assert Phase1OverlapAdapter().adapt(_checkpoint(5), result.evidence).ready


def test_sparse_probe_is_deterministic() -> None:
    problem = _problem()
    kwargs = dict(
        anchors=_anchors(5),
        step=0.25,
        run_seed=11,
        rounds=12,
        bucket_size=2,
        max_candidate_pairs=32,
    )
    first = discover_overlap_sparse(problem, EvaluationLedger(problem, 1000), **kwargs)
    replay = discover_overlap_sparse(problem, EvaluationLedger(problem, 1000), **kwargs)

    assert first == replay


def test_sparse_probe_budget_shortfall_is_incomplete_and_spends_nothing() -> None:
    problem = _problem()
    ledger = EvaluationLedger(problem, 1)

    result = discover_overlap_sparse(
        problem,
        ledger,
        anchors=_anchors(5),
        step=0.25,
        run_seed=11,
        rounds=12,
        bucket_size=2,
    )

    assert not result.complete
    assert result.complete_reason == "budget_insufficient_for_screening"
    assert ledger.count == 0
    assert not Phase1OverlapAdapter().adapt(_checkpoint(5), result.evidence).ready


def test_sparse_probe_reports_incomplete_when_pair_separation_is_not_covered() -> None:
    problem = _problem()
    result = discover_overlap_sparse(
        problem,
        EvaluationLedger(problem, 1000),
        anchors=_anchors(5),
        step=0.25,
        run_seed=11,
        rounds=1,
        bucket_size=2,
    )

    assert not result.complete
    assert result.complete_reason == "separation_coverage_incomplete"
    assert result.separated_pair_fraction < 1.0


def test_sparse_probe_candidate_cap_is_fail_closed() -> None:
    problem = _problem()
    result = discover_overlap_sparse(
        problem,
        EvaluationLedger(problem, 1000),
        anchors=_anchors(5),
        step=0.25,
        run_seed=11,
        rounds=12,
        bucket_size=2,
        max_candidate_pairs=1,
    )

    assert not result.complete
    assert result.complete_reason == "candidate_pair_cap_exceeded"
    assert result.candidate_pair_count > 1


@pytest.mark.parametrize("dimension", [64, 128])
def test_sparse_probe_screening_cost_is_bucketed_for_large_dimension(dimension: int) -> None:
    problem = _problem(dimension)
    anchors = _anchors(dimension)[:1]
    rounds = 4
    bucket_size = 8
    bucket_count = (dimension + bucket_size - 1) // bucket_size
    expected_screen = rounds * (1 + bucket_count + bucket_count * (bucket_count - 1) // 2)
    ledger = EvaluationLedger(problem, expected_screen)

    result = discover_overlap_sparse(
        problem,
        ledger,
        anchors=anchors,
        step=0.1,
        run_seed=19,
        rounds=rounds,
        bucket_size=bucket_size,
        max_candidate_pairs=1,
    )

    assert not result.complete
    assert result.complete_reason == "budget_insufficient_for_refinement"
    assert result.consumed_fes == expected_screen
    assert expected_screen < dimension * (dimension - 1) // 2
