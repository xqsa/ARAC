from __future__ import annotations

import numpy as np
import pytest

from arac.baselines import (
    PYPOP_METHODS,
    GroupingResult,
    hcc_global_phase_fes,
    overlap_degree,
    run_hcc_es,
    run_pypop_baseline,
)


@pytest.mark.parametrize("method", tuple(PYPOP_METHODS))
def test_pypop_nda_adapters_are_deterministic_bounded_and_exactly_budgeted(
    method: str,
) -> None:
    calls: list[np.ndarray] = []

    def objective(candidate: np.ndarray) -> float:
        values = np.asarray(candidate, dtype=float)
        assert values.shape == (8,)
        assert np.all((0.0 <= values) & (values <= 1.0))
        calls.append(values.copy())
        return float(np.sum(np.square(values - 0.2)))

    first = run_pypop_baseline(
        objective,
        method,
        8,
        max_function_evaluations=31,
        seed=20260723,
        sigma=3.0,
    )
    first_call_count = len(calls)
    second = run_pypop_baseline(
        objective,
        method,
        8,
        max_function_evaluations=31,
        seed=20260723,
        sigma=3.0,
    )

    assert first.result_hash == second.result_hash
    assert first.method == method
    assert first.backend.startswith("PyPop7.")
    assert first.optimization_fes == 31
    assert first.decomposition_fes == 0
    assert first_call_count == 31
    assert len(calls) == 62
    assert first.phase_fes == (("initial_context", 1), ("optimizer", 30))
    assert first.repaired_candidate_count > 0
    assert len(first.best_so_far_trace) == first.optimization_fes
    assert np.all((0.0 <= np.asarray(first.best_x)) & (np.asarray(first.best_x) <= 1.0))


def test_pypop_adapter_rejects_unknown_method_before_evaluation() -> None:
    calls = 0

    def objective(_candidate: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    with pytest.raises(ValueError, match="unsupported PyPop7 baseline"):
        run_pypop_baseline(
            objective,
            "PyPop7-HCC",
            4,
            max_function_evaluations=10,
            seed=1,
        )
    assert calls == 0


def test_pypop_adapter_supports_a_one_fe_initial_context_smoke() -> None:
    result = run_pypop_baseline(
        lambda x: float(np.sum(x)),
        "Sep-CMAES",
        4,
        max_function_evaluations=1,
        seed=3,
    )

    assert result.optimization_fes == 1
    assert result.phase_fes == (("initial_context", 1),)
    assert result.best_x == (0.5, 0.5, 0.5, 0.5)


def test_hcc_budget_formula_uses_supplied_topology_overlap_degree() -> None:
    disjoint = GroupingResult(
        method="fixture",
        dimension=4,
        groups=((0, 1), (2, 3)),
        decomposition_fes=0,
        allows_overlap=False,
        origin="test",
    )
    overlapping = GroupingResult(
        method="fixture",
        dimension=4,
        groups=((0, 1), (1, 2), (3,)),
        decomposition_fes=0,
        allows_overlap=True,
        origin="test",
    )

    assert overlap_degree(disjoint) == pytest.approx(0.0)
    assert overlap_degree(overlapping) == pytest.approx(0.25)
    assert hcc_global_phase_fes(100, disjoint) == 20
    assert hcc_global_phase_fes(100, overlapping) == 40


def test_generic_hcc_es_is_deterministic_reconciles_overlap_and_counts_exact_fes() -> None:
    grouping = GroupingResult(
        method="RDDSM",
        dimension=4,
        groups=((0, 1), (1, 2), (3,)),
        decomposition_fes=11,
        allows_overlap=True,
        origin="test",
    )

    def objective(candidate: np.ndarray) -> float:
        values = np.asarray(candidate, dtype=float)
        assert values.shape == (4,)
        assert np.all((0.0 <= values) & (values <= 1.0))
        return float(
            np.sum(np.square(values - 0.2))
            + 2.0 * (values[0] - values[1]) ** 2
            + 2.0 * (values[1] - values[2]) ** 2
        )

    first = run_hcc_es(
        objective,
        grouping,
        max_function_evaluations=61,
        seed=20260723,
        sigma=0.8,
        group_block_fes=7,
    )
    second = run_hcc_es(
        objective,
        grouping,
        max_function_evaluations=61,
        seed=20260723,
        sigma=0.8,
        group_block_fes=7,
    )

    assert first.result_hash == second.result_hash
    assert first.method == "HCC-ES"
    assert first.grouping_hash == grouping.grouping_hash
    assert first.decomposition_fes == 11
    assert first.optimization_fes == 61
    assert sum(count for _, count in first.phase_fes) == 61
    assert dict(first.phase_fes)["initial_context"] == 1
    assert dict(first.phase_fes)["global_mmes"] == 23
    assert dict(first.phase_fes)["overlap_reconciliation"] > 0
    assert first.repaired_candidate_count > 0
