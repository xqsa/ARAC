from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.gcb_native_contract_diagnostic import (
    CASES,
    SEEDS,
    TOTAL_BUDGET_FES,
    VARIANTS,
    _execute_variant,
    _summarize,
    _variant_settings,
    load_protocol,
)


def _context(*, allow_out_of_bounds: bool) -> ActionContext:
    problem = OptimizationProblem(
        objective=lambda x: np.ones(np.asarray(x).shape[:-1]),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-gcb-native-contract-v1",
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
        allow_out_of_bounds=allow_out_of_bounds,
    )
    return ActionContext("gcb", checkpoint, problem, ledger, action_seed=31_001)


def test_protocol_freezes_three_native_contract_variants() -> None:
    protocol = load_protocol()

    assert tuple(protocol["variants"]) == VARIANTS
    assert protocol["matched_baseline_variant"] == "native_order_historical_seeds"
    assert protocol["reference_thresholds_used_for_decision"] is False
    assert protocol["selector_execution_allowed"] is False


def test_variant_flags_form_the_precheck_and_clipping_factorial() -> None:
    assert _variant_settings("native_order_no_clip") == (False, False)
    assert _variant_settings("native_order_precheck") == (True, True)
    assert _variant_settings("native_order_no_clip_precheck") == (False, True)


def test_each_variant_records_complete_cold_windows_and_prechecks() -> None:
    for variant in VARIANTS:
        clip, precheck = _variant_settings(variant)
        result, _, summary = _execute_variant(
            variant,
            _context(allow_out_of_bounds=not clip),
        )

        assert result.terminal_fes == 20_000
        assert summary["valid"] is True
        assert summary["clip_offspring"] is clip
        assert summary["precheck_incumbent"] is precheck
        assert summary["source_sweep_group_counts"] == {"0": 2, "1": 2, "2": 2}
        assert summary["native_sweep_group_counts"] == {"3": 2, "4": 2, "5": 2}
        assert (summary["precheck_fes"] > 0) is precheck
        assert summary["state_restore_count"] == 0


def test_summary_selects_only_variants_that_beat_parent_and_frozen_current() -> None:
    protocol = load_protocol()
    rows = []
    for variant in VARIANTS:
        factor = 0.8 if variant == "native_order_no_clip_precheck" else 1.2
        for case_id in CASES:
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
    assert summary["passing_variants"] == ["native_order_no_clip_precheck"]
    assert summary["diagnostic_gate_passed"] is True
    assert summary["production_gcb_integration_authorized"] is False
    assert summary["final_25_seed_recovery_evaluated"] is False
