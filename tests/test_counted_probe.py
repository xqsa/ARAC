"""Unit tests for the counted two-sided conflict probe."""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import LocalProposal, OverlapStructure, counted_probe
from arac.runtime.ledger import EvaluationLedger


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        result += 0.5 * batch[:, 0] ** 2 * batch[:, 1] ** 2
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=3,
        lower_bounds=(-5.0,) * 3,
        upper_bounds=(5.0,) * 3,
    )


def _structure() -> OverlapStructure:
    return OverlapStructure(dimension=3, groups=((0, 1), (1, 2)))


def _ledger(problem: OptimizationProblem, incumbent: tuple[float, ...]) -> EvaluationLedger:
    return EvaluationLedger.from_checkpoint(
        problem,
        total_budget=200,
        phase1_fes=4,
        incumbent=incumbent,
        incumbent_error=float(
            np.sum(np.asarray(incumbent) ** 2)
            + 0.5 * incumbent[0] ** 2 * incumbent[1] ** 2
        ),
    )


def test_probe_consumes_exactly_two_fes_per_variable_and_reuses_incumbent() -> None:
    problem = _problem()
    ledger = _ledger(problem, (1.0, 2.0, 1.0))
    before = ledger.count

    results = counted_probe(_structure(), ledger, (1,), proposals=())

    assert ledger.count - before == 2
    assert len(results) == 1
    assert results[0].variable == 1
    assert results[0].f_plus != results[0].f_minus


def test_probe_is_deterministic() -> None:
    problem = _problem()
    first_ledger = _ledger(problem, (1.0, 2.0, 1.0))
    second_ledger = _ledger(problem, (1.0, 2.0, 1.0))

    first = counted_probe(_structure(), first_ledger, (1,), proposals=())
    second = counted_probe(_structure(), second_ledger, (1,), proposals=())

    assert first == second


def test_symmetric_flat_sides_give_zero_conflict() -> None:
    def flat(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        return float(result[0]) if rows.ndim == 1 else result

    flat_problem = OptimizationProblem(
        objective=flat,
        dimension=3,
        lower_bounds=(-5.0,) * 3,
        upper_bounds=(5.0,) * 3,
    )
    ledger = EvaluationLedger.from_checkpoint(
        flat_problem,
        total_budget=200,
        phase1_fes=4,
        incumbent=(0.0, 0.0, 0.0),
        incumbent_error=0.0,
    )
    # At the symmetric optimum of a quadratic, both sides respond equally.
    results = counted_probe(_structure(), ledger, (1,), proposals=())

    assert results[0].bias == pytest.approx(0.0, abs=1e-12)
    assert results[0].conflict_score == pytest.approx(0.0, abs=1e-12)


def test_asymmetric_response_gives_signed_bias_in_bounds() -> None:
    problem = _problem()
    # Off-center incumbent on the coupled term gives an asymmetric response.
    ledger = _ledger(problem, (2.0, 2.0, 0.5))

    results = counted_probe(_structure(), ledger, (1,), proposals=())

    assert -1.0 <= results[0].bias <= 1.0
    assert 0.0 <= results[0].conflict_score <= 1.0


def test_proposal_disagreement_sets_the_probe_scale() -> None:
    problem = _problem()
    narrow = _ledger(problem, (1.0, 1.0, 1.0))
    wide = _ledger(problem, (1.0, 1.0, 1.0))
    small = LocalProposal(
        group=0,
        values=((0, 1.0), (1, 1.05)),
        improvement=0.0,
        uncertainty=((0, 0.01), (1, 0.01)),
    )
    large = LocalProposal(
        group=0,
        values=((0, 1.0), (1, 1.0)),
        improvement=0.0,
        uncertainty=((0, 0.01), (1, 0.01)),
    )
    other_small = LocalProposal(
        group=1,
        values=((1, 1.10), (2, 1.0)),
        improvement=0.0,
        uncertainty=((1, 0.01), (2, 0.01)),
    )
    other_large = LocalProposal(
        group=1,
        values=((1, 2.5), (2, 1.0)),
        improvement=0.0,
        uncertainty=((1, 0.01), (2, 0.01)),
    )

    narrow_step = counted_probe(
        _structure(), narrow, (1,), proposals=(small, other_small)
    )[0].step
    wide_step = counted_probe(
        _structure(), wide, (1,), proposals=(large, other_large)
    )[0].step

    assert wide_step > narrow_step


def test_probe_fails_closed_on_non_shared_variable_and_budget() -> None:
    problem = _problem()
    ledger = _ledger(problem, (1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="shared"):
        counted_probe(_structure(), ledger, (0,), proposals=())
    tiny = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=5,
        phase1_fes=4,
        incumbent=(1.0, 1.0, 1.0),
        incumbent_error=3.0,
    )
    with pytest.raises(ValueError, match="budget"):
        counted_probe(_structure(), tiny, (1,), proposals=())
