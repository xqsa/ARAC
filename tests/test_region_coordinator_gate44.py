"""Gate 44: HIERARCHICAL interface tests for region-level coordination.

Pass criteria (docs/arac-phase1-v10-design.md §6):
- RegionProposal, region conflict probe and region patch run end-to-end on
  RegionStructure evidence with exact per-stage FE accounting;
- strict-best holds throughout (ledger archive never degrades);
- the whole path constructs no OverlapStructure.
"""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.overlap import OverlapStructure
from arac.coordination.region import (
    RegionCoordinator,
    produce_region_proposal,
    region_conflict_probe,
)
from arac.evidence.hierarchical import (
    Phase1Evidence,
    RegionNode,
    RegionRelation,
    RegionStructure,
    RegionTree,
    VariableRegionInteraction,
)
from arac.runtime.ledger import EvaluationLedger

DIMENSION = 12


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        result += 3.0 * batch[:, 2] * batch[:, 8]
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _evidence() -> Phase1Evidence:
    tree = RegionTree(
        dimension=DIMENSION,
        nodes=(
            RegionNode(0, None, 0, tuple(range(DIMENSION))),
            RegionNode(1, 0, 1, (0, 1, 2, 3, 4, 5)),
            RegionNode(2, 0, 1, (6, 7, 8, 9, 10, 11)),
        ),
    )
    interaction_a = VariableRegionInteraction(
        variable=2, source_region=1, target_region=2, q_lb=0.8, support=6, sign_stability=1.0
    )
    interaction_b = VariableRegionInteraction(
        variable=8, source_region=2, target_region=1, q_lb=0.8, support=6, sign_stability=1.0
    )
    return Phase1Evidence(
        dimension=DIMENSION,
        region_tree=tree,
        region_relations=(RegionRelation(left=1, right=2, score=3.0, stability=1.0, depth=1),),
        variable_region_interactions=(interaction_a, interaction_b),
        variable_status=tuple(
            (variable, "member_candidate" if variable in (2, 8) else "observed_separable")
            for variable in range(DIMENSION)
        ),
        per_component_mode=(((1, 2), "HIERARCHICAL"),),
        level_budgets=(("probe", 4),),
    )


def _ledger(problem: OptimizationProblem) -> EvaluationLedger:
    incumbent = np.ones(DIMENSION)
    error = float(np.sum(incumbent**2) + 3.0)
    return EvaluationLedger.from_checkpoint(
        problem,
        total_budget=2_000,
        phase1_fes=4,
        incumbent=tuple(incumbent),
        incumbent_error=error,
    )


def test_gate44_cycle_exact_fe_strict_best_and_receipt() -> None:
    problem = _problem()
    ledger = _ledger(problem)
    coordinator = RegionCoordinator(RegionStructure(evidence=_evidence()), ledger)
    before = ledger.best_error

    receipt = coordinator.run_cycle(
        proposal_budget_fes=16,
        patch_budget_fes=8,
        probe_step=0.5,
        seed=5,
    )

    total = receipt.proposal_fes + receipt.probe_fes + receipt.patch_fes
    assert receipt.proposal_fes == 16
    assert receipt.probe_fes == 4  # two candidates, 2 FE each
    assert receipt.patch_fes % 2 == 0 and receipt.patch_fes <= 8
    assert ledger.best_error <= before
    assert receipt.best_error_after == float(ledger.best_error)
    assert receipt.component == (1, 2)
    assert total > 0


def test_gate44_path_never_constructs_overlap_structure(monkeypatch) -> None:
    problem = _problem()
    ledger = _ledger(problem)

    def forbidden(self, *args, **kwargs):
        raise AssertionError("OverlapStructure constructed inside the region path")

    monkeypatch.setattr(OverlapStructure, "__init__", forbidden)
    coordinator = RegionCoordinator(RegionStructure(evidence=_evidence()), ledger)
    receipt = coordinator.run_cycle(proposal_budget_fes=16, patch_budget_fes=8, seed=5)
    assert receipt.component == (1, 2)


def test_gate44_probe_is_billed_and_deterministic() -> None:
    problem = _problem()
    evidence = _evidence()
    candidates = evidence.variable_region_interactions

    first_ledger = _ledger(problem)
    first = region_conflict_probe(candidates, problem=problem, ledger=first_ledger, step=0.5)
    second_ledger = _ledger(problem)
    second = region_conflict_probe(candidates, problem=problem, ledger=second_ledger, step=0.5)

    assert first_ledger.count == 4 + 4  # phase1_fes + probe
    assert first == second
    assert all(item.consumed_fes == 2 for item in first)
    assert all(0.0 <= item.conflict_score <= 1.0 for item in first)
    coupled = {item.variable for item in first if item.conflict_score > 0.1}
    assert coupled == {2, 8}


def test_gate44_proposal_budget_is_exact() -> None:
    problem = _problem()
    ledger = _ledger(problem)
    structure = RegionStructure(evidence=_evidence())

    proposal = produce_region_proposal(
        structure,
        1,
        problem=problem,
        global_ledger=ledger,
        anchor=ledger.best_x,
        anchor_error=float(ledger.best_error),
        budget_fes=24,
        seed=9,
    )

    assert proposal.consumed_fes == 24
    assert proposal.leaf_id == 1
    assert {variable for variable, _ in proposal.values} == set(structure.region_variables(1))
    assert proposal.improvement >= 0.0


def test_gate44_coordinator_rejects_empty_evidence() -> None:
    problem = _problem()
    ledger = _ledger(problem)
    evidence = Phase1Evidence(
        dimension=DIMENSION,
        region_tree=_evidence().region_tree,
        region_relations=(),
        variable_region_interactions=(),
        variable_status=tuple((v, "observed_separable") for v in range(DIMENSION)),
    )
    coordinator = RegionCoordinator(RegionStructure(evidence=evidence), ledger)

    with pytest.raises(ValueError, match="at least one relation"):
        coordinator.run_cycle(proposal_budget_fes=16, patch_budget_fes=8)
