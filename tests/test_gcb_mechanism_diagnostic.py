from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.gcb_mechanism_diagnostic import (
    CASES,
    SEEDS,
    TOTAL_BUDGET_FES,
    VARIANTS,
    _execute_variant,
    _summarize,
    load_protocol,
)


def _context() -> ActionContext:
    problem = OptimizationProblem(
        objective=lambda x: np.ones(np.asarray(x).shape[:-1]),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-gcb-mechanism-diagnostic-v1",
        run_seed=31_001,
        total_budget_fes=20_000,
        phase1_fes=180,
        incumbent=(0.0,) * 40,
        incumbent_error=1.0,
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


def test_protocol_freezes_only_development_mechanism_variants() -> None:
    protocol = load_protocol()

    assert tuple(protocol["variants"]) == VARIANTS
    assert protocol["source_root"] == "artifacts/gcb_schedule_ablation_v2"
    assert protocol["reference_thresholds_used_for_decision"] is False
    assert protocol["selector_execution_allowed"] is False


def test_each_variant_has_three_source_and_native_windows_without_restore() -> None:
    for variant in VARIANTS:
        result, _, trace = _execute_variant(variant, _context())

        assert result.terminal_fes == 20_000
        assert trace["valid"] is True
        assert trace["source_sweep_group_counts"] == {"0": 2, "1": 2, "2": 2}
        assert trace["native_sweep_group_counts"] == {"3": 2, "4": 2, "5": 2}
        assert trace["state_restore_count"] == 0


def test_summary_selects_only_variants_that_pass_all_case_gates() -> None:
    protocol = load_protocol()
    rows = []
    for variant in VARIANTS:
        factor = 0.8 if variant == "native_order_historical_seeds" else 1.2
        for case_id in CASES:
            for seed in SEEDS:
                rows.append(
                    {
                        "variant": variant,
                        "case_id": case_id,
                        "run_seed": seed,
                        "checkpoint_hash": f"{case_id}-{seed}",
                        "baseline_final_error": 100.0,
                        "final_error": 100.0 * factor,
                        "terminal_fes": TOTAL_BUDGET_FES,
                        "schedule_trace_summary": {"valid": True},
                        "runtime_warnings": [],
                    }
                )

    summary = _summarize(rows, protocol)

    assert summary["integrity_gate_passed"] is True
    assert summary["passing_variants"] == ["native_order_historical_seeds"]
    assert summary["diagnostic_gate_passed"] is True
    assert summary["final_25_seed_recovery_evaluated"] is False
