from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.gcb_trigger_diagnostic import (
    CASES,
    SEEDS,
    TOTAL_BUDGET_FES,
    VARIANTS,
    _execute_variant,
    _summarize,
    load_protocol,
)
import experiments.historical_recovery.gcb_mechanism_diagnostic as parent


def _context(*, with_relation: bool) -> ActionContext:
    problem = OptimizationProblem(
        objective=lambda x: np.ones(np.asarray(x).shape[:-1]),
        dimension=60,
        lower_bounds=(-5.0,) * 60,
        upper_bounds=(5.0,) * 60,
    )
    relations = (RelationEvidence(0, 1, strength=0.4, disagreement=0.2),) if with_relation else ()
    checkpoint = PhaseCheckpoint(
        protocol="test-gcb-trigger-diagnostic-v1",
        run_seed=31_001,
        total_budget_fes=30_000,
        phase1_fes=180,
        incumbent=(0.0,) * 60,
        incumbent_error=1.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=(tuple(range(20)), tuple(range(20, 40)), tuple(range(40, 60))),
        relations=relations,
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    return ActionContext("gcb", checkpoint, problem, ledger, action_seed=31_001)


def test_protocol_freezes_trigger_only_development_variants() -> None:
    protocol = load_protocol()

    assert tuple(protocol["variants"]) == VARIANTS
    assert protocol["baseline_source_root"] == "artifacts/gcb_mechanism_diagnostic_v1"
    assert protocol["reference_thresholds_used_for_decision"] is False
    assert protocol["selector_execution_allowed"] is False


def test_relation_dispatch_occurs_after_both_owners_inside_first_native_sweep() -> None:
    result, _, summary = _execute_variant("native_order_relation_dispatch", _context(with_relation=True))

    assert result.terminal_fes == 30_000
    assert summary["valid"] is True
    assert summary["trigger"] == "relation_dispatch"
    assert summary["selected_relation"] == [0, 1]
    assert summary["mixed_sweep_pre_dispatch_groups"] == [0, 1]
    assert summary["mixed_sweep_post_dispatch_groups"] == [2]
    assert summary["native_sweep_group_counts"] == {"3": 3, "4": 3, "5": 3}


def test_zero_relation_keeps_phase_boundary_path_identical_across_orders() -> None:
    candidate = _execute_variant(VARIANTS[0], _context(with_relation=False))
    baseline = parent._execute_variant(
        "native_order_historical_seeds",
        _context(with_relation=False),
    )

    assert candidate[2]["valid"] is True
    assert candidate[2]["trigger"] == "phase_boundary"
    assert candidate[0].final_error == baseline[0].final_error
    assert candidate[1] == baseline[1]


def test_summary_requires_r1_equivalence_and_both_positive_cases_to_improve() -> None:
    protocol = load_protocol()
    rows = []
    for variant in VARIANTS:
        for case_id in CASES:
            factor = 1.0 if case_id == "R1" else 0.8
            for seed in SEEDS:
                rows.append(
                    {
                        "variant": variant,
                        "case_id": case_id,
                        "run_seed": seed,
                        "checkpoint_hash": f"{case_id}-{seed}",
                        "matched_baseline_final_error": 100.0,
                        "frozen_current_final_error": 100.0,
                        "final_error": 100.0 * factor,
                        "terminal_fes": TOTAL_BUDGET_FES,
                        "schedule_trace_summary": {"valid": True},
                        "runtime_warnings": [],
                    }
                )

    summary = _summarize(rows, protocol)

    assert summary["integrity_gate_passed"] is True
    assert summary["passing_variants"] == list(VARIANTS)
    assert summary["diagnostic_gate_passed"] is True
    assert summary["production_gcb_integration_authorized"] is False
    assert summary["final_25_seed_recovery_evaluated"] is False
