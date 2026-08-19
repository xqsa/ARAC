from __future__ import annotations

import numpy as np

import arac.actions._execution as execution
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import PhaseCheckpoint
from experiments.historical_recovery.smp_semantic_parity_diagnostic import (
    VARIANTS,
    _context,
    _native_restart_option,
    _stateful_arm,
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


def _checkpoint() -> PhaseCheckpoint:
    return PhaseCheckpoint(
        protocol="smp-semantic-diagnostic-test",
        run_seed=117,
        total_budget_fes=4_900,
        phase1_fes=100,
        incumbent=(0.0,) * 40,
        incumbent_error=0.0,
        feature_names=("dummy",),
        feature_values=(0.0,),
        blocks=tuple(tuple(range(start, start + 10)) for start in range(0, 40, 10)),
    )


def test_protocol_freezes_short_prefix_without_selector_or_hcc() -> None:
    protocol = load_protocol()

    assert tuple(protocol["variants"]) == VARIANTS
    assert protocol["screen_step_fes"] == 120_000
    assert protocol["production_hcc_dependency_allowed"] is False
    assert protocol["selector_execution_allowed"] is False


def test_stateful_arm_accounts_exact_prefix_budget() -> None:
    context = _context(_checkpoint(), _problem(), 4_800)

    result, events = _stateful_arm(context, requested_fes=4_800, rescue=False)

    assert result.consumed_fes == 4_800
    assert result.terminal_fes == 4_900
    assert result.final_error == 0.0
    assert events["stateful_fes"] + events["noop_fes"] == 4_800
    assert events["visit_count"] > 0


def test_native_restart_option_is_local_and_restored() -> None:
    original = execution.CMAES

    with _native_restart_option(True):
        assert execution.CMAES is not original

    assert execution.CMAES is original


def _receipt(variant: str, final_error: float, result_hash: str) -> dict[str, object]:
    return {
        "variant": variant,
        "source_checkpoint_hash": "source",
        "screen_checkpoint_hash": "screen",
        "source_phase1_fes": 100,
        "screen_step_fes": 200,
        "runtime_warnings": [],
        "native_thread_limit_verified": True,
        "receipt_hash": variant,
        "result": {
            "consumed_fes": 200,
            "terminal_fes": 300,
            "final_error": final_error,
            "result_hash": result_hash,
            "route": variant,
        },
    }


def test_summary_separates_rescue_and_restart_sensitivity() -> None:
    receipts = [
        _receipt("schedule_only", 10.0, "schedule"),
        _receipt("stateful_only", 8.0, "same"),
        _receipt("stateful_prefix_no_rescue", 7.0, "no-rescue"),
        _receipt("stateful_plus_rescue", 4.0, "rescue"),
        _receipt("current_complete_smp", 4.0, "complete"),
        _receipt("stateful_native_restart_on", 8.0, "same"),
    ]

    summary = _summarize(receipts)

    assert summary["all_exact_screen_fes"] is True
    assert summary["rescue_final_error_delta_no_rescue_minus_rescue"] == 3.0
    assert summary["native_restart_on_off_identical"] is True
    assert summary["terminal_parity_evaluated"] is False
    assert summary["selector_evaluation_authorized"] is False
