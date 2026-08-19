from __future__ import annotations

import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.gcb_schedule_ablation import (
    CASES,
    PHASE1_FES,
    SEEDS,
    TOTAL_BUDGET_FES,
    VARIANTS,
    _execute_variant,
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


def _context() -> ActionContext:
    problem = _problem()
    checkpoint = PhaseCheckpoint(
        protocol="test-gcb-schedule-v1",
        run_seed=31_001,
        total_budget_fes=20_000,
        phase1_fes=180,
        incumbent=(1.0,) * 40,
        incumbent_error=40.0,
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
    return ActionContext("gcb", checkpoint, problem, ledger, action_seed=31_001)


def test_protocol_freezes_fresh_paired_gate_without_reference_or_selector() -> None:
    protocol = load_protocol()

    assert tuple(protocol["cases"]) == CASES
    assert tuple(protocol["seeds"]) == SEEDS
    assert tuple(protocol["variants"]) == VARIANTS
    assert protocol["max_workers"] == 18
    assert protocol["reference_thresholds_used_for_decision"] is False
    assert protocol["selector_execution_allowed"] is False
    assert protocol["production_hcc_runtime_imports_allowed"] is False


def test_candidate_trace_is_deterministic_and_restores_no_optimizer_state() -> None:
    first_result, first_trace, first_summary = _execute_variant(
        "gcb_three_source_burst_native",
        _context(),
    )
    second_result, second_trace, second_summary = _execute_variant(
        "gcb_three_source_burst_native",
        _context(),
    )

    assert first_result.final_error == second_result.final_error
    assert first_trace == second_trace
    assert first_summary == second_summary
    assert first_summary is not None and first_summary["valid"] is True
    assert first_summary["trigger"] == "relation_dispatch"
    assert first_summary["source_actual_fes"] == first_summary["coordination_actual_fes"]
    assert first_summary["source_sweep_group_counts"] == {"0": 2, "1": 2, "2": 2}
    assert first_summary["native_sweep_group_counts"] == {"3": 2, "4": 2, "5": 2}
    assert first_summary["state_restore_count"] == 0
    assert first_result.terminal_fes == 20_000


def _row(
    case_id: str,
    seed: int,
    variant: str,
    error: float,
) -> dict[str, object]:
    candidate = variant == "gcb_three_source_burst_native"
    return {
        "case_id": case_id,
        "run_seed": seed,
        "variant": variant,
        "checkpoint_hash": f"{case_id}-{seed}",
        "terminal_fes": TOTAL_BUDGET_FES,
        "final_error": error,
        "schedule_trace_summary": {"valid": True} if candidate else None,
        "runtime_warnings": [],
    }


def test_summary_uses_paired_improvement_without_historical_thresholds() -> None:
    protocol = load_protocol()
    rows = []
    for case_id in CASES:
        for offset, seed in enumerate(SEEDS):
            baseline = 100.0 + offset
            rows.append(_row(case_id, seed, "gcb_frozen_current", baseline))
            rows.append(
                _row(
                    case_id,
                    seed,
                    "gcb_three_source_burst_native",
                    baseline * 0.8,
                )
            )

    summary = _summarize(rows, protocol)

    assert summary["phase1_fes"] == PHASE1_FES
    assert summary["paired_development_gate_passed"] is True
    assert summary["candidate_win_or_tie_count"] == 9
    assert summary["candidate_schedule_trace_valid"] is True
    assert summary["final_25_seed_recovery_evaluated"] is False
    assert summary["selector_or_routing_authorized"] is False
    assert all(
        math.isclose(case["candidate_to_baseline_geometric_mean_ratio"], 0.8)
        for case in summary["case_summaries"]
    )
