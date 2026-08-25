"""Regression tests for the soft-RDDSM shared-variable evidence dataflow.

Historical bug (2026-08-15 audit): blocks were mutually exclusive while
shared candidacy required a variable to appear in two blocks' INTERACT-OV
result sets -- a condition that no run could ever satisfy, so recall was
structurally zero.  These tests pin the corrected semantics: bridging
evidence is an ordered pair (variable, source block, target block), and a
shared candidate must survive block-level separability with the variable
removed.
"""

from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.soft_rddsm import SoftDsmConfig, _rdg_interact, discover_hierarchical_soft
from arac.evidence.soft_rddsm_adapter import soft_evidence_to_overlap_evidence
from arac.runtime.ledger import EvaluationLedger

DIMENSION = 7
GROUP_A = (0, 1, 2, 3)
GROUP_B = (3, 4, 5, 6)
SHARED = 3
TEST_SEED = 20260851


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        result += np.sum(batch[:, list(GROUP_A)], axis=1) ** 2
        result += np.sum(batch[:, list(GROUP_B)], axis=1) ** 2
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _discover(budget: int = 20_000):
    problem = _problem()
    ledger = EvaluationLedger(problem, budget)
    result = discover_hierarchical_soft(
        problem,
        ledger,
        run_seed=TEST_SEED,
        config=SoftDsmConfig(dsm_budget=10_000),
        signature_probe_count=6,
        signature_probe_size=4,
    )
    return problem, ledger, result


def test_two_blocks_one_shared_variable_is_recovered() -> None:
    _problem_obj, ledger, result = _discover()

    # RDG grouping reconstructs the two overlapping groups exactly.
    assert result.blocks == (GROUP_A, GROUP_B[1:])

    # The shared variable is the single confirmed candidate ...
    assert result.shared_candidates == (SHARED,)
    assert len(result.evidence.resolved_hyperedges) == 1

    # ... with a hyperedge anchored at its home leaf spanning both blocks.
    hyperedge = result.evidence.resolved_hyperedges[0]
    assert hyperedge.variable == SHARED
    assert len(hyperedge.regions) == 2
    assert len(set(hyperedge.regions)) == 2
    tree = result.evidence.region_tree
    assert hyperedge.regions[0] == tree.leaf_of(SHARED).node_id

    # Non-shared members of the smeared side are rejected by the
    # separability confirmation even though they carry bridging evidence.
    bridging = {interaction.variable for interaction in result.evidence.variable_region_interactions}
    assert bridging == {SHARED, 4, 5, 6}
    assert set(result.shared_candidates) <= bridging

    sidecar = soft_evidence_to_overlap_evidence(result.evidence)
    assert sidecar.complete is True
    assert sidecar.memberships[SHARED] == (0, 1)
    assert tuple(
        variable
        for variable, owners in enumerate(sidecar.memberships)
        if len(owners) > 1
    ) == (SHARED,)


def test_level_budgets_reconcile_exactly_with_ledger() -> None:
    _problem_obj, ledger, result = _discover()
    by_stage = dict(result.level_budgets)
    assert set(by_stage) == {"signature", "dsm", "rdg"}
    assert sum(by_stage.values()) == ledger.count
    assert ledger.count > 0


def test_rdg_interact_consumes_exactly_three_evaluations() -> None:
    problem = _problem()
    ledger = EvaluationLedger(problem, 100)
    base_point = (problem.lower_array + problem.upper_array) / 2.0
    base_value = float(np.asarray(ledger.evaluate(base_point[np.newaxis, :])).reshape(-1)[0])
    before = ledger.count
    _rdg_interact(
        problem,
        ledger,
        set_a=[0],
        set_b=[4],
        base_point=base_point,
        base_value=base_value,
        threshold=1e-13,
    )
    assert ledger.count - before == 3
