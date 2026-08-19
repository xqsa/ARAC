from __future__ import annotations

import math

import numpy as np

from arac.actions._execution import BLOCK_POPULATION_SIZE
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.ctp_tail_ablation import (
    CASES,
    PHASE1_FES,
    SEEDS,
    TOTAL_BUDGET_FES,
    VARIANTS,
    _execute_no_reserved_tail,
    _summarize,
    load_protocol,
)


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )


def test_protocol_freezes_a_paired_gate_without_reference_or_selector() -> None:
    protocol = load_protocol()

    assert tuple(protocol["cases"]) == CASES
    assert tuple(protocol["seeds"]) == SEEDS
    assert tuple(protocol["variants"]) == VARIANTS
    assert protocol["reference_thresholds_used_for_decision"] is False
    assert protocol["selector_execution_allowed"] is False
    assert protocol["production_hcc_runtime_imports_allowed"] is False


def test_no_reserved_tail_reproduces_only_the_alignment_residue() -> None:
    problem = _problem()
    checkpoint = PhaseCheckpoint(
        protocol="test-ctp-ablation-v1",
        run_seed=31_001,
        total_budget_fes=3_000,
        phase1_fes=180,
        incumbent=(0.0,) * 40,
        incumbent_error=0.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=(tuple(range(20)), tuple(range(20, 40))),
        relations=(RelationEvidence(0, 1, strength=0.4, disagreement=0.2),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )

    result, components = _execute_no_reserved_tail(
        ActionContext("ctp", checkpoint, problem, ledger, action_seed=31_001)
    )

    relation_cover_sweep = 3 * BLOCK_POPULATION_SIZE
    assert result.terminal_fes == 3_000
    assert sum(components.values()) == 3_000 - 180
    assert 0 <= components["tail_fes"] < relation_cover_sweep


def _row(
    case_id: str,
    seed: int,
    variant: str,
    error: float,
) -> dict[str, object]:
    candidate = variant == "ctp_mmes_tail_20pct"
    return {
        "case_id": case_id,
        "run_seed": seed,
        "variant": variant,
        "checkpoint_hash": f"{case_id}-{seed}",
        "phase1_relation_count": 1,
        "terminal_fes": TOTAL_BUDGET_FES,
        "final_error": error,
        "tail_fes": int((TOTAL_BUDGET_FES - PHASE1_FES) * 0.20) if candidate else 12,
        "runtime_warnings": [],
    }


def test_summary_uses_paired_improvement_not_historical_thresholds() -> None:
    protocol = load_protocol()
    rows = []
    for case_id in CASES:
        for offset, seed in enumerate(SEEDS):
            baseline = 100.0 + offset
            rows.append(_row(case_id, seed, "ctp_no_reserved_tail", baseline))
            rows.append(_row(case_id, seed, "ctp_mmes_tail_20pct", baseline * 0.8))

    summary = _summarize(rows, protocol)

    assert summary["paired_development_gate_passed"] is True
    assert summary["candidate_win_or_tie_count"] == 6
    assert summary["final_25_seed_recovery_evaluated"] is False
    assert summary["selector_or_routing_authorized"] is False
    assert all(
        math.isclose(
            case["candidate_to_baseline_geometric_mean_ratio"],
            0.8,
        )
        for case in summary["case_summaries"]
    )
