"""Tests for the hyperedge_ctp_v1 upgrade candidate (T0 / H0)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from arac.actions.recovered_registry import RecoveredActionRegistry


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
T0_SUMMARY = REPOSITORY_ROOT / "artifacts/upgrade_hyperedge_ctp_v1_t0_v1/summary.json"
H0_SUMMARY = REPOSITORY_ROOT / "artifacts/upgrade_hyperedge_ctp_v1_h0_v1/summary.json"


def _toy_problem(dim: int = 80) -> OptimizationProblem:
    def objective(vector):
        values = np.asarray(vector, dtype=float)
        flat = values.reshape(-1, dim)
        out = np.sum(flat**2, axis=1)
        return float(out[0]) if values.ndim == 1 else out

    return OptimizationProblem(
        objective=objective,
        dimension=dim,
        lower_bounds=(-5.0,) * dim,
        upper_bounds=(5.0,) * dim,
        optimum=0.0,
    )


def _checkpoint(dim: int, relations) -> PhaseCheckpoint:
    block = dim // 4
    blocks = tuple(tuple(range(index * block, (index + 1) * block)) for index in range(4))
    incumbent = tuple(0.5 for _ in range(dim))
    return PhaseCheckpoint(
        protocol="arac-identity-blind-evidence-v9",
        run_seed=20270111,
        total_budget_fes=40_000,
        phase1_fes=1_000,
        incumbent=incumbent,
        incumbent_error=float(np.sum(np.asarray(incumbent) ** 2)),
        feature_names=("a",),
        feature_values=(1.0,),
        blocks=blocks,
        relations=relations,
    )


def _run(problem, checkpoint, *, registry) -> tuple:
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=registry.allow_out_of_bounds,
    )
    result = execute_phase2_action("ctp", checkpoint, problem, ledger, action_seed=20270111, registry=registry)
    assert ledger.count == checkpoint.total_budget_fes
    return result, ledger


def test_t0_variant_registry_covers_frozen_action_set() -> None:
    from experiments.upgrade.hyperedge_ctp_v1.t0_tail_causal import T0VariantRegistry

    registry = T0VariantRegistry()
    assert registry.action_names == RecoveredActionRegistry().action_names
    assert registry.allow_out_of_bounds is True


def test_t0_variant_is_bitwise_frozen_without_relations() -> None:
    from experiments.upgrade.hyperedge_ctp_v1.t0_tail_causal import T0VariantRegistry

    problem = _toy_problem()
    checkpoint = _checkpoint(problem.dimension, ())
    frozen, _ = _run(problem, checkpoint, registry=RecoveredActionRegistry())
    variant, _ = _run(problem, checkpoint, registry=T0VariantRegistry())
    assert frozen.final_error == variant.final_error
    assert frozen.route == variant.route
    assert frozen.result_hash == variant.result_hash


def test_t0_variant_removes_only_the_positive_relation_reserve() -> None:
    from experiments.upgrade.hyperedge_ctp_v1.t0_tail_causal import T0VariantRegistry

    problem = _toy_problem()
    checkpoint = _checkpoint(problem.dimension, (RelationEvidence(0, 1, 0.5, 0.1),))
    frozen, frozen_ledger = _run(problem, checkpoint, registry=RecoveredActionRegistry())
    variant, variant_ledger = _run(problem, checkpoint, registry=T0VariantRegistry())
    assert frozen_ledger.count == variant_ledger.count == checkpoint.total_budget_fes
    assert "then_mmes_tail_" in frozen.route and "then_mmes_tail_" in variant.route
    frozen_tail = int(frozen.route.rsplit("then_mmes_tail_", 1)[1])
    variant_tail = int(variant.route.rsplit("then_mmes_tail_", 1)[1])
    assert frozen_tail > variant_tail
    assert frozen.final_error != variant.final_error


def test_t0_protocol_loads_without_drift() -> None:
    from experiments.upgrade.hyperedge_ctp_v1.t0_tail_causal import DEFAULT_PROTOCOL, load_protocol

    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol["seeds"] == [20270111, 20270112, 20270113, 20270114, 20270115]
    assert protocol["cases"] == ["S3", "S6"]


def test_h0_protocol_pins_plan_configuration() -> None:
    from experiments.upgrade.hyperedge_ctp_v1.h0_sidecar import DEFAULT_PROTOCOL, load_protocol

    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol["activation_cells"] == ["chain4-strong", "pairs3-strong"]
    assert protocol["discovery_seeds"] == [20270101, 20270102, 20270103]
    assert protocol["soft_config"]["dsm_budget"] == 55_000
    assert protocol["soft_config"]["tau_block"] == 0.6
    assert protocol["generator_freeze"]["linkage_lambda"] == 2.0


def test_hyperedge_audit_counts_true_positives_against_owner_pairs() -> None:
    from experiments.upgrade.hyperedge_ctp_v1.h0_sidecar import _hyperedge_audit

    class _Hyperedge:
        def __init__(self, variable, regions):
            self.variable = variable
            self.regions = regions

    class _Evidence:
        def __init__(self, hyperedges):
            self.resolved_hyperedges = tuple(hyperedges)

    class _Truth:
        shared_variables = (5, 105)
        shared_owner_pairs = ((5, 0, 1), (105, 1, 2))

    leaf_variables = {1: tuple(range(0, 10)), 2: tuple(range(100, 110)), 3: tuple(range(200, 210))}
    evidence = _Evidence(
        (
            _Hyperedge(5, (1, 2)),   # correct: owners {0,1}
            _Hyperedge(105, (2, 3)),  # correct: owners {1,2}
            _Hyperedge(7, (1, 3)),   # not planted shared -> false positive
            _Hyperedge(5, (1, 2, 3)),  # three-region, never certified
        )
    )
    audit = _hyperedge_audit(evidence, leaf_variables, _Truth())
    assert audit["certified_hyperedge_count"] == 3
    assert audit["hyperedges_with_more_than_two_regions"] == 1
    assert audit["true_positive_count"] == 2
    assert audit["precision"] == pytest.approx(2 / 3)
    assert audit["recall"] == 1.0
    # the three 2-region hyperedges close a triangle 1-2-3: a cycle
    assert audit["region_graph_forest"] is False
    assert audit["region_graph_max_degree"] == 2


@pytest.mark.skipif(not T0_SUMMARY.is_file(), reason="T0 summary artifact not present")
def test_t0_summary_contract() -> None:
    summary = json.loads(T0_SUMMARY.read_text(encoding="utf-8"))
    claimed = summary.pop("result_hash")
    assert claimed == canonical_sha256(summary)
    assert summary["gate_passed"] in (True, False)
    assert summary["line_b_unaffected"] is True
    for row in summary["case_rows"]:
        assert row["pair_count"] == 5
        if summary["gate_passed"]:
            assert row["causal_signal_confirmed"] is True


@pytest.mark.skipif(not H0_SUMMARY.is_file(), reason="H0 summary artifact not present")
def test_h0_summary_gate_contract() -> None:
    summary = json.loads(H0_SUMMARY.read_text(encoding="utf-8"))
    claimed = summary.pop("result_hash")
    assert claimed == canonical_sha256(summary)
    assert summary["h1_authorized"] is summary["gate_passed"]
    assert len(summary["cell_rows"]) == 6
    if not summary["gate_passed"]:
        # a failed H0 must never activate an uncertified cell
        assert all(not row["activated"] for row in summary["cell_rows"])
