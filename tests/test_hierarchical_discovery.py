"""Pipeline tests for the five-stage hierarchical discovery."""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.hierarchical import Phase1Evidence, RegionStructure, to_overlap_structure
from arac.evidence.hierarchical_discovery import (
    HierarchicalDiscoveryConfig,
    complete_incumbent,
    discover_hierarchical,
)
from arac.runtime.ledger import EvaluationLedger

DIMENSION = 48
BLOCK_A = tuple(range(0, 20))
BLOCK_B = tuple(range(20, 40))
SHARED_A = 15
SHARED_B = 30
DUMMY = tuple(range(40, 48))
TEST_SEED = 20260840


def _problem(perm: np.ndarray) -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        original = batch[:, perm]
        result = np.sum(original**2, axis=1)
        for block in (BLOCK_A, BLOCK_B):
            inner = original[:, list(block)]
            result += 0.5 * np.sum(inner**2, axis=1) ** 2 / len(block)
        result += 3.0 * original[:, SHARED_A] * original[:, SHARED_B]
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _config() -> HierarchicalDiscoveryConfig:
    return HierarchicalDiscoveryConfig(
        anchor_count=2,
        coarse_rounds=2,
        coarse_regions=8,
        signature_probe_count=6,
        signature_probe_size=4,
        step=0.5,
        min_region_size=6,
        max_depth=4,
        s_max=60,
        c_max=60,
        per_split_candidates=4,
        k_dir=2,
        a_cond=2,
        incumbent_min=0,
        edge_threshold=1e-3,
    )


def _discover(perm: np.ndarray, budget: int = 60_000):
    problem = _problem(perm)
    ledger = EvaluationLedger(problem, budget)
    result = discover_hierarchical(
        problem,
        ledger,
        run_seed=TEST_SEED,
        config=_config(),
    )
    return problem, ledger, result


def test_pipeline_stage_budgets_reconcile_exactly() -> None:
    rng = np.random.default_rng(TEST_SEED)
    problem, ledger, result = _discover(rng.permutation(DIMENSION))
    by_stage = dict(result.level_budgets)
    assert sum(by_stage.values()) == ledger.count
    assert by_stage["coarse"] == 2 * 2 * (1 + 8 + 28)  # anchors*rounds*(1 + R + C(R,2))
    signature_expected = 1 + DIMENSION + 6 + DIMENSION * 6 + 6 * 4
    assert by_stage["signature"] == signature_expected
    assert by_stage["splits"] % 4 == 0
    assert by_stage["conditional"] >= 0


def test_pipeline_produces_valid_coherent_evidence() -> None:
    rng = np.random.default_rng(TEST_SEED + 1)
    perm = rng.permutation(DIMENSION)
    problem, ledger, result = _discover(perm)
    evidence: Phase1Evidence = result.evidence

    # Strong internal coupling must yield at least one significant relation.
    assert evidence.region_relations, "expected region relations from coupled blocks"
    for relation in evidence.region_relations:
        assert relation.score > _config().edge_threshold

    # Every interaction references recorded evidence and valid leaves.
    leaves = {leaf.node_id for leaf in evidence.region_tree.leaves}
    for interaction in evidence.variable_region_interactions:
        assert interaction.source_region in leaves
        assert interaction.target_region in leaves
        assert interaction.source_region != interaction.target_region

    # Statuses, modes and budgets stay coherent.
    assert set(mode for _, mode in evidence.per_component_mode) <= {
        "SPARSE",
        "HIERARCHICAL",
        "EVIDENCE_DENSE",
    }
    covered = {leaf for component, _ in evidence.per_component_mode for leaf in component}
    assert covered <= leaves
    structure = RegionStructure(evidence=evidence)
    assert structure.components()
    if evidence.resolved_hyperedges:
        shared = to_overlap_structure(evidence).shared_variables
        assert set(shared) == {h.variable for h in evidence.resolved_hyperedges}


def test_pipeline_incumbent_floor_is_enforced() -> None:
    rng = np.random.default_rng(TEST_SEED + 2)
    problem = _problem(rng.permutation(DIMENSION))
    ledger = EvaluationLedger(problem, 60_000)
    result = discover_hierarchical(
        problem, ledger, run_seed=TEST_SEED, config=_config()
    )
    consumed = dict(result.level_budgets)
    assert ledger.remaining >= _config().incumbent_min
    terminal = complete_incumbent(
        problem, ledger, run_seed=TEST_SEED, incumbent_min=0
    )
    assert terminal == 60_000
    assert consumed

    tight = EvaluationLedger(problem, sum(dict(result.level_budgets).values()) + 5)
    discover_hierarchical(problem, tight, run_seed=TEST_SEED, config=_config())
    with pytest.raises(RuntimeError, match="incumbent floor"):
        complete_incumbent(
            problem, tight, run_seed=TEST_SEED, incumbent_min=6
        )


def test_pipeline_determinism() -> None:
    rng = np.random.default_rng(TEST_SEED + 3)
    perm = rng.permutation(DIMENSION)
    _, _, first = _discover(perm)
    _, _, second = _discover(perm)
    assert first.order == second.order
    assert first.level_budgets == second.level_budgets
    assert first.evidence == second.evidence
