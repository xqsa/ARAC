"""Gate 43: billability and signal-quality tests for variable signatures."""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.variable_signature import compute_variable_signatures
from arac.runtime.ledger import EvaluationLedger

DIMENSION = 80
GROUP_SIZE = 5
GROUP_COUNT = 8
GROUPS = tuple(
    tuple(range(index * GROUP_SIZE, (index + 1) * GROUP_SIZE)) for index in range(GROUP_COUNT)
)
DUMMY = tuple(range(GROUP_COUNT * GROUP_SIZE, DIMENSION))


def _permuted_group_problem(perm: np.ndarray) -> tuple[OptimizationProblem, dict[int, tuple[int, ...]]]:
    """Nonseparable groups in original space, observed through a permutation."""

    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        original = batch[:, perm]
        result = np.sum(original**2, axis=1)
        for group in GROUPS:
            block = original[:, list(group)]
            result += 0.5 * np.sum(block**2, axis=1) ** 2 / len(group)
        return float(result[0]) if rows.ndim == 1 else result

    problem = OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )
    observed_groups = {int(perm[variable]): group for group in GROUPS for variable in group}
    return problem, observed_groups


def _anchor(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-2.0, 2.0, size=DIMENSION)


def test_gate43_billability_exact_fe_and_determinism() -> None:
    rng = np.random.default_rng(20260817)
    perm = rng.permutation(DIMENSION)
    problem, _ = _permuted_group_problem(perm)
    ledger = EvaluationLedger(problem, 200_000)
    anchor = _anchor(rng)
    expected = 1 + DIMENSION + 12 + DIMENSION * 12 + 12 * 5

    first = compute_variable_signatures(
        problem, ledger, anchor=anchor, probe_count=12, probe_size=5, seed=7
    )
    second = compute_variable_signatures(
        problem,
        EvaluationLedger(problem, 200_000),
        anchor=anchor,
        probe_count=12,
        probe_size=5,
        seed=7,
    )

    assert first.consumed_fes == expected == ledger.count
    assert first.expected_fes == expected
    assert np.array_equal(first.signatures, second.signatures)


def test_gate43_fails_closed_on_insufficient_budget() -> None:
    rng = np.random.default_rng(1)
    perm = rng.permutation(DIMENSION)
    problem, _ = _permuted_group_problem(perm)
    tiny = EvaluationLedger(problem, 100)

    with pytest.raises(ValueError, match="budget"):
        compute_variable_signatures(problem, tiny, anchor=_anchor(rng), seed=3)


def _hit_rate(result, observed_groups) -> float:
    hits = []
    total = 0
    for variable, group in observed_groups.items():
        k = len(group) - 1
        neighbors = result.top_neighbors(variable, k)
        same = sum(1 for n in neighbors if n in observed_groups and observed_groups[n] is group)
        hits.append(same)
        total += k
    return sum(hits) / total


@pytest.mark.parametrize("seed", [20260818, 20260819, 20260820])
def test_gate43_signal_quality_and_permutation_invariance(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(3):
        perm = rng.permutation(DIMENSION)
        problem, observed_groups = _permuted_group_problem(perm)
        ledger = EvaluationLedger(problem, 400_000)
        result = compute_variable_signatures(
            problem,
            ledger,
            anchor=_anchor(rng),
            step=0.5,
            probe_count=16,
            probe_size=5,
            seed=int(rng.integers(0, 2**31)),
        )
        rates.append(_hit_rate(result, observed_groups))
    baseline = (GROUP_SIZE - 1) / (DIMENSION - 1)
    mean_rate = sum(rates) / len(rates)
    assert mean_rate >= 2.0 * baseline, f"hit rate {mean_rate:.3f} vs baseline {baseline:.3f}"


def test_gate43_dummy_variables_have_zero_interaction_signatures() -> None:
    rng = np.random.default_rng(20260821)
    perm = rng.permutation(DIMENSION)
    problem, _ = _permuted_group_problem(perm)
    ledger = EvaluationLedger(problem, 400_000)
    result = compute_variable_signatures(
        problem,
        ledger,
        anchor=_anchor(rng),
        step=0.5,
        probe_count=16,
        probe_size=5,
        seed=11,
    )
    observed_dummies = [int(perm[variable]) for variable in DUMMY]
    # Dummy coordinates are fully separable: every mixed difference vanishes.
    for variable in observed_dummies:
        assert np.allclose(result.signatures[variable], 0.0, atol=1e-12)
    # Group members must show at least one non-zero interaction component.
    for group in GROUPS:
        observed = [int(perm[v]) for v in group]
        assert float(np.abs(result.signatures[observed]).max()) > 1e-9
