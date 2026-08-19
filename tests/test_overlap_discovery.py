from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence import Phase1OverlapAdapter, discover_overlap
from arac.runtime.contracts import PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger


GROUPS = ((0, 1, 2), (2, 3, 4))


def _coupled_problem(*, conflicting: bool = False) -> OptimizationProblem:
    targets = ((-1.0, 0.5, 1.5), (2.0 if conflicting else 1.5, -0.5, 1.0))

    def objective(values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        single = converted.ndim == 1
        batch = converted[np.newaxis, :] if single else converted
        result = np.zeros(len(batch))
        for group, target in zip(GROUPS, targets, strict=True):
            local = batch[:, group] - np.asarray(target)
            result += np.sum(local**2, axis=1)
            for left in range(local.shape[1]):
                for right in range(left + 1, local.shape[1]):
                    result += local[:, left] * local[:, right]
        return float(result[0]) if single else result

    return OptimizationProblem(
        objective=objective,
        dimension=5,
        lower_bounds=(-5.0,) * 5,
        upper_bounds=(5.0,) * 5,
    )


def _anchors() -> tuple[tuple[float, ...], ...]:
    return (
        (-1.5, -0.5, 0.5, 1.0, 1.5),
        (1.5, 0.5, -0.5, -1.0, -1.5),
    )


def _checkpoint() -> PhaseCheckpoint:
    return PhaseCheckpoint(
        protocol="overlap-discovery-test-v1",
        run_seed=7,
        total_budget_fes=100,
        phase1_fes=1,
        incumbent=(0.0,) * 5,
        incumbent_error=0.0,
        feature_names=("complete",),
        feature_values=(1.0,),
        blocks=((0,), (1,), (2,), (3,), (4,)),
    )


@pytest.mark.parametrize("conflicting", [False, True])
def test_discovery_recovers_shared_variable_and_exact_groups(conflicting: bool) -> None:
    problem = _coupled_problem(conflicting=conflicting)
    ledger = EvaluationLedger(problem, total_budget=32)

    result = discover_overlap(
        problem,
        ledger,
        anchors=_anchors(),
        step=0.25,
        edge_threshold=1.0e-8,
    )

    assert result.evidence.groups == GROUPS
    assert result.evidence.memberships == ((0,), (0,), (0, 1), (1,), (1,))
    assert result.consumed_fes == ledger.count == 32
    assert result.identifiable
    assert result.edges == ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4))
    adapted = Phase1OverlapAdapter().adapt(_checkpoint(), result.evidence)
    assert adapted.ready
    assert adapted.structure is not None
    assert adapted.structure.shared_variables == (2,)


def test_discovery_is_deterministic_for_the_same_probe_design() -> None:
    problem = _coupled_problem()

    first = discover_overlap(
        problem,
        EvaluationLedger(problem, 32),
        anchors=_anchors(),
        step=0.25,
    )
    replay = discover_overlap(
        problem,
        EvaluationLedger(problem, 32),
        anchors=_anchors(),
        step=0.25,
    )

    assert first == replay


def test_separable_control_does_not_invent_shared_variables() -> None:
    problem = OptimizationProblem(
        objective=lambda values: np.sum(np.asarray(values, dtype=float) ** 2, axis=-1),
        dimension=5,
        lower_bounds=(-5.0,) * 5,
        upper_bounds=(5.0,) * 5,
    )

    result = discover_overlap(
        problem,
        EvaluationLedger(problem, 32),
        anchors=_anchors(),
        step=0.25,
    )

    assert not result.identifiable
    assert result.evidence.groups == ((0,), (1,), (2,), (3,), (4,))
    assert all(len(owners) == 1 for owners in result.evidence.memberships)


def test_probe_rejects_an_out_of_bounds_design_before_spending_fes() -> None:
    problem = _coupled_problem()
    ledger = EvaluationLedger(problem, 32)

    with pytest.raises(ValueError, match="escaped"):
        discover_overlap(
            problem,
            ledger,
            anchors=((4.9,) * 5,),
            step=0.25,
        )

    assert ledger.count == 0
