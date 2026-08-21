from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.consistency import owner_preference_calibration
from arac.coordination.overlap import OverlapStructure
from arac.runtime.ledger import EvaluationLedger


def _setup(groups, incumbent_error=10.0):
    structure = OverlapStructure(6, tuple(tuple(g) for g in groups))
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=6,
        lower_bounds=(-5.0,) * 6,
        upper_bounds=(5.0,) * 6,
        optimum=0.0,
    )
    ledger = EvaluationLedger(problem, 500, initial_incumbent=(0.1,) * 6, initial_error=incumbent_error)
    return structure, ledger


def test_owner_calibration_exact_fe_and_scope_validation() -> None:
    structure, ledger = _setup([(0, 1, 2), (2, 3, 4), (4, 5)])
    labels = owner_preference_calibration(
        structure, ledger, structure.shared_variables, rounds=2, population=3, seed=1
    )
    scope = sorted(structure.shared_variables)
    assert [item.variable for item in labels] == scope
    # owner groups: 0-1-2, 2-3-4, 4-5 -> owners of shared vars 2 and 4
    expected_fes = 2 * 3 * 3  # rounds * population * distinct owner groups
    assert all(item.group_fes == 6 for item in labels)
    assert ledger.count == expected_fes


def test_owner_calibration_rejects_non_shared_scope() -> None:
    structure, ledger = _setup([(0, 1, 2), (2, 3, 4), (4, 5)])
    with pytest.raises(ValueError, match="shared"):
        owner_preference_calibration(structure, ledger, (0,))


def test_owner_calibration_budget_guard() -> None:
    structure, _ledger = _setup([(0, 1, 2), (2, 3, 4), (4, 5)])
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=6,
        lower_bounds=(-5.0,) * 6,
        upper_bounds=(5.0,) * 6,
    )
    tiny = EvaluationLedger(problem, 5, initial_incumbent=(0.1,) * 6, initial_error=10.0)
    with pytest.raises(ValueError, match="budget"):
        owner_preference_calibration(structure, tiny, structure.shared_variables)


def test_conforming_quadratic_labels_via_owner_endpoints() -> None:
    # Single shared quadratic well shared by both owners: block-local search
    # from a shifted incumbent drives both owners' endpoints toward zero, so
    # disagreement collapses and the label is conforming.
    structure = OverlapStructure(4, ((0, 1, 2), (2, 3)))
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    ledger = EvaluationLedger(problem, 400, initial_incumbent=(1.0, 1.0, 1.5, 1.0), initial_error=5.5)
    labels = owner_preference_calibration(
        structure, ledger, structure.shared_variables, rounds=6, population=6, seed=3
    )
    assert len(labels) == 1
    assert labels[0].variable == 2
    assert labels[0].label == "conforming"
    assert labels[0].disagreement < 0.05
