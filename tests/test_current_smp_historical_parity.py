from __future__ import annotations

import json

from experiments.historical_recovery.current_smp_historical_parity import (
    HISTORICAL_TARGET,
    build_boundary_audit,
    run_first_sweep_dual_track,
    run_historical_contract_prefix,
    run_lockstep,
    write_boundary_audit,
)


def test_exact_historical_reproduction_is_the_numerical_gate() -> None:
    audit = build_boundary_audit()

    assert audit["historical"]["final_error"] == HISTORICAL_TARGET
    assert audit["historical"]["fitness_evaluations"] == 3_000_000
    assert audit["historical"]["restore_count"] == 122
    assert audit["historical"]["reset_count"] == 36
    assert audit["historical"]["abstain_count"] == 0


def test_first_material_divergence_is_the_action_entry_boundary() -> None:
    audit = build_boundary_audit()

    assert audit["historical"]["phase1_fes"] == 0
    assert audit["historical"]["available_action_fes"] == 3_000_000
    assert audit["current"]["phase1_fes"] == 180_000
    assert audit["current"]["available_action_fes"] == 2_820_000
    assert audit["decision"]["current_checkpoint_is_exact_exp052_boundary"] is False


def test_group_partition_is_exact_but_order_is_not() -> None:
    audit = build_boundary_audit()
    comparison = audit["group_comparison"]

    assert comparison["same_unordered_partition"] is True
    assert comparison["same_outer_order"] is False
    assert comparison["same_internal_order"] is False
    assert comparison["historical_to_current_indices"] == [
        16,
        8,
        7,
        6,
        3,
        10,
        13,
        11,
        1,
        15,
        9,
        17,
        5,
        18,
        14,
        2,
        4,
        0,
        12,
        19,
    ]


def test_budget_and_lifecycle_differences_are_explicit() -> None:
    audit = build_boundary_audit()
    historical = audit["historical"]
    mismatch = set(audit["material_mismatches"])

    assert historical["visit_count"] == 165
    assert historical["cold_start_count"] == 43
    assert historical["budget"] == {
        "budget_accounting": "strict",
        "max_fes": 3_000_000,
        "global_phase_fe": 0,
        "cc_phase_fe": 2_999_831,
        "rescue_fe": 0,
        "overhead_fe": 169,
        "budget_aligned_fe": 3_000_000,
        "same_budget_violation": 0,
    }
    assert audit["budget_delta"]["historical_cc_minus_current_stateful_fes"] == 743_840
    assert {
        "optimizer_snapshot_restore_lifecycle",
        "per_visit_seed_and_rng_lifecycle",
        "offspring_boundary_handling",
        "stateful_vs_rescue_budget_ownership",
        "terminal_budget_fill",
    } <= mismatch
    assert "native_cma_restart_disabled" in audit["confirmed_non_mismatches"]


def test_audit_is_machine_readable_and_keeps_expensive_gates_closed(tmp_path) -> None:
    output = tmp_path / "audit.json"
    expected = write_boundary_audit(output)

    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert expected["decision"] == {
        "current_checkpoint_is_exact_exp052_boundary": False,
        "production_change_authorized": False,
        "terminal_3m_run_authorized": False,
        "next_gate": "two-visit-first-group-lockstep",
    }


def test_two_visit_lockstep_matches_recovered_cmaes_without_production_hcc() -> None:
    result = run_lockstep()

    assert result["passed"] is True
    assert result["first_visit"] == {
        "candidate_matrix_bitwise_equal": True,
        "fitness_bitwise_equal": True,
        "state_bitwise_equal": True,
    }
    assert result["second_visit"] == result["first_visit"]
    assert result["ledger_consumed_fes"] == 32
    assert result["production_hcc_import_allowed"] is False
    assert result["terminal_3m_run_authorized"] is False


def test_first_five_groups_keep_independent_dual_tracks_in_exact_lockstep() -> None:
    result = run_first_sweep_dual_track(group_count=5, generations=2)

    assert result["passed"] is True
    assert result["group_count"] == 5
    assert result["population_aligned_visit_budgets"] == [32, 32, 32, 32, 38]
    assert result["historical_total_fes"] == 171
    assert result["current_total_fes"] == 171
    assert result["independent_problem_instances"] is True
    assert result["independent_incumbents"] is True
    assert result["first_divergence"] is None
    assert all(row["generation_trace"]["passed"] for row in result["rows"])
    assert all(row["state_bitwise_equal"] for row in result["rows"])
    assert result["terminal_3m_run_authorized"] is False


def test_short_historical_contract_gate_closes_noop_tail_exactly() -> None:
    result = run_historical_contract_prefix(3_200)

    assert result["passed"] is True
    assert result["zero_start"] is True
    assert result["historical_order"] is True
    assert result["clip_offspring"] is False
    assert result["precheck_fes"] == result["visit_count"]
    assert result["noop_fes"] == 3
    assert result["ledger_count"] == result["requested_fes"]
    assert result["terminal_3m_run_authorized"] is False


def test_incumbent_precheck_can_refresh_the_strict_archive_value() -> None:
    import numpy as np

    from arac.benchmarks.aob import OptimizationProblem
    from arac.runtime.ledger import EvaluationLedger

    values = iter((11.0, 10.0))

    def objective(candidate: np.ndarray) -> float:
        del candidate
        return next(values)

    problem = OptimizationProblem(
        objective=objective,
        dimension=1,
        lower_bounds=(-1.0,),
        upper_bounds=(1.0,),
    )
    ledger = EvaluationLedger(
        problem,
        total_budget=2,
        initial_incumbent=(0.0,),
        initial_error=10.0,
    )

    assert ledger.evaluate_incumbent(refresh_error=True) == 11.0
    assert ledger.best_error == 11.0
    ledger.evaluate(np.asarray([0.5]))
    assert ledger.best_error == 10.0
    assert np.array_equal(ledger.best_x, np.asarray([0.5]))
