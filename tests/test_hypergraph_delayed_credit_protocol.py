from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "hypergraph_delayed_credit_v1.json"
SPEC_PATH = ROOT / "docs" / "design" / "hypergraph-delayed-credit-v1.md"
ALL_CASES = [
    *(f"E{index}" for index in range(1, 7)),
    *(f"S{index}" for index in range(1, 7)),
    *(f"R{index}" for index in range(1, 7)),
    *(f"A{index}" for index in range(1, 7)),
]


def _load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _validate_frozen_config(config: dict[str, object]) -> None:
    assert config["protocol_version"] == "hypergraph-delayed-credit-v1"
    assert config["config_schema_version"] == 2
    assert config["status"] == "frozen_before_new_optimizer_fe"

    baseline = config["baseline"]
    assert baseline["profile"] == "arac_evidence_action_controller_v37"
    assert {
        "cma_kernel",
        "sigma",
        "csa",
        "population",
        "requested_group_budgets",
        "optimizer_seeds",
        "native_group_writeback",
    }.issubset(baseline["immutable"])

    topology = config["topology"]
    assert topology["hyperedges"] == "raw_overlapping_groups"
    assert topology["preserve_multiple_membership"] is True
    assert topology["transitive_closure_allowed"] is False
    assert topology["union_find_allowed"] is False
    assert topology["neighbor_of_neighbor_expansion_allowed"] is False

    numeric = config["numeric"]
    assert numeric["history_complete_sweeps"] == 3
    assert numeric["ewma_alpha"] == 0.5
    assert numeric["owner_weight_cap"] == 0.65
    assert numeric["bootstrap_count"] == 2000
    assert numeric["bootstrap_lcb_quantile"] == 0.05

    integrity = config["integrity"]
    assert integrity["formal_trace_requires_clean_tracked_tree"] is True
    assert integrity["full_requires_prior_screen"] is True
    assert integrity["source_bundle_files"] == [
        "src/arac/policy/overlap_hypergraph.py",
        "src/arac/backends/hcc_hypergraph_trace.py",
        "src/arac/backends/hcc.py",
        "scripts/hcc_smoke_runner.py",
        "experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py",
        "scripts/audit_hypergraph_trace.py",
    ]
    assert integrity["per_run_artifact_binding"].endswith(
        "exact_ordered_reconstruction_of_all_four_aggregate_raw_csvs"
    )
    assert integrity["derived_numeric_binding"].startswith(
        "raw_recomputed_values_must_equal_runtime_17e"
    )
    assert integrity["terminal_integrity"] == (
        "hcc_status_completed_and_target_minus_tolerance_le_actual_fe_le_target"
    )
    assert integrity["policy_ast_allowlist"].startswith(
        "frozen_state_and_score_call_graph"
    )

    state = config["state"]
    assert state["difficulty_formula"] == "1-success_ratio_3"
    assert state["stagnation_formula"] == "min(consecutive_u_le_0,3)/3"
    assert state["actual_fe_scope"].startswith("full_native_group_interval")
    assert state["proposal_capture_watermark"] == (
        "after_group_local_rescue_recovery_before_relation_writeback"
    )
    assert state["missing_required_state"] == "inapplicable_fail_closed"
    assert state["required_derived_features"] == [
        "current_unit_fe_contribution",
        "ewma_unit_fe_contribution_3",
        "zero_gain_difficulty",
        "stagnation_ratio_3",
        "direct_owner_proposal_disagreement",
        "prior_next_sweep_overwrite",
    ]
    assert state["audit_only_complements"] == [
        "success_ratio_3",
        "prior_next_sweep_survival",
    ]
    assert state["duplicate_complements_in_policy_state_allowed"] is False

    scores = config["scores"]
    assert scores["owner_reliability"] == (
        "mean(rank_current_u,rank_ewma_u,rank_one_minus_prior_overwrite)"
    )

    action = config["coordination_action"]
    assert action["protocol_version"] == "one-hop-shared-commit-v1"
    assert action["max_decisions_per_trajectory"] == 1
    assert action["max_actions_per_trajectory"] == 1
    assert action["invalid_candidate"] == (
        "permanent_abstain_consume_decision_lock"
    )
    assert action["evaluation_order"] == ["anchor", "candidate"]
    assert action["evaluation_fe"] == 2
    assert action["consume_optimizer_rng"] is False
    assert action["update_cma_kernel"] is False
    assert action["commit_rule"] == (
        "candidate_error_strictly_less_than_anchor_error"
    )

    delayed = config["delayed_credit"]
    assert delayed["minimum_complete_normal_sweeps"] == 1
    assert delayed["entry_time_closure_allowed"] is False
    assert delayed["can_enter_current_commit"] is False

    prediction = config["prediction"]
    assert prediction["decision_cohort"] == (
        "first_history_complete_snapshot_once_per_trajectory"
    )
    assert prediction["maximum_decision_snapshots_per_trajectory"] == 1
    assert prediction["within_snapshot_rows"] == "all_eligible_raw_hyperedges"
    assert prediction["later_sweep_use"] == (
        "resolve_first_snapshot_labels_only_no_new_decision"
    )
    assert prediction["bootstrap_unit"] == (
        "case_by_seed_trajectory_hyperedge_rows_not_independent"
    )
    assert prediction["decision_eligibility"].endswith("pre_label_only")
    assert prediction["terminal_censoring"] == (
        "valid_integrity_evidence_but_label_incomplete_and_stage_no_go"
    )
    assert prediction["required_state_missing_fraction"] == {
        "numerator": (
            "first_locked_overlap_opportunity_trajectories_missing_any_required_"
            "six_feature_state"
        ),
        "denominator": (
            "all_first_locked_overlap_opportunity_trajectories_before_state_"
            "completeness_or_label_filtering"
        ),
        "zero_denominator": "undefined_fail_closed",
    }
    assert prediction["auditor_recomputation"]["trusted_derived_rows"] is False
    assert prediction["auditor_recomputation"]["first_opportunity"].startswith(
        "derive_earliest_lock"
    )
    assert prediction["auditor_recomputation"]["proposal_backfill"].startswith(
        "t_next_sweep_value"
    )
    assert prediction["auditor_recomputation"]["labels"].endswith(
        "require_exact_runtime_17e_equality"
    )
    assert prediction["auditor_recomputation"]["derived_hash"].startswith(
        "format_raw_recomputed_state"
    )
    assert prediction["auditor_recomputation"][
        "nonapplicable_manifest"
    ].startswith("rebuild_pending")
    assert prediction["auditor_recomputation"]["mismatch"] == (
        "integrity_failure"
    )
    assert prediction["trajectory_equal_weight"] is True
    assert prediction["trajectory_metrics"] == {
        "priority_spearman": (
            "within_trajectory_spearman_focal_priority_vs_next_gain_"
            "constant_or_undefined_is_zero"
        ),
        "focal_rank_advantage": (
            "next_gain_midrank_percentile_of_unique_focal_minus_0_5"
        ),
        "owner_survival_spearman": (
            "within_trajectory_spearman_owner_reliability_vs_next_survival_"
            "constant_or_undefined_is_zero"
        ),
        "diagnostic_focal_gain_delta": (
            "next_gain_focal_minus_mean_next_gain_nonfocal_not_a_cross_case_gate"
        ),
    }
    assert prediction["support"] == {
        "status": "diagnostic_deferred_no_runtime_model",
        "used_for_filtering": False,
        "used_for_gate": False,
        "cannot_rescue_all_state_result": True,
    }
    assert prediction["bootstrap"]["method"] == (
        "case_seed_two_way_pigeonhole_on_trajectory_scalars"
    )
    overwrite = prediction["overwrite_prediction"]
    assert overwrite["weighted_median"].startswith("lower_weighted_quantile")
    assert overwrite["single_class_route"] == "undefined_fail_closed"
    assert overwrite["single_class_bootstrap_replicate"] == (
        "retain_replicate_with_balanced_accuracy_zero"
    )

    matrices = config["matrices"]
    expected_counts = {
        "observer_cli": 8,
        "observer_trace": 4,
        "trace_screen": 40,
        "trace_full": 192,
        "action_cli": 12,
        "action_trace": 3,
        "action_screen": 120,
        "action_confirm": 576,
    }
    assert {name: matrix["run_count"] for name, matrix in matrices.items()} == (
        expected_counts
    )
    assert matrices["trace_full"]["cases"] == ALL_CASES
    assert matrices["action_confirm"]["cases"] == ALL_CASES
    assert matrices["trace_full"]["seeds"] == list(range(96, 104))
    assert matrices["trace_full"]["requires_prior_screen_pass"] is True
    assert matrices["action_confirm"]["seeds"] == list(range(109, 117))

    trace_full = config["trace_gates"]["full"]
    assert trace_full["minimum_complete_next_sweep_label_fraction"] == 1.0
    assert trace_full["minimum_applicable_trajectories"] == 120
    assert trace_full["requires_recomputed_screen_pass_same_source_bundle"] is True
    assert trace_full["overwrite_balanced_accuracy_lcb_strictly_above"] == 0.5
    assert trace_full["metric_scope"] == (
        "all_decision_eligible_trajectories_equal_weight_no_support_filter"
    )
    assert trace_full["trajectory_metric_lcb_strictly_positive"] == [
        "trajectory_priority_spearman",
        "trajectory_focal_rank_advantage",
        "trajectory_owner_survival_spearman",
    ]
    assert trace_full["primary_direction_metric"] == (
        "trajectory_focal_rank_advantage"
    )
    assert trace_full["case_advantage_share_formula"] == (
        "max_abs_case_mean_divided_by_sum_abs_case_means_zero_denominator_fail_"
        "closed"
    )

    trace_screen = config["trace_gates"]["screen"]
    assert trace_screen["minimum_complete_next_sweep_label_fraction"] == 1.0
    assert trace_screen[
        "minimum_mean_trajectory_priority_spearman_strictly_above"
    ] == 0.0
    assert trace_screen[
        "minimum_mean_trajectory_focal_rank_advantage_strictly_above"
    ] == 0.0
    assert trace_screen["metric_scope"] == (
        "equal_weight_decision_eligible_trajectories_after_required_100_percent_"
        "label_closure"
    )
    assert trace_screen["overwrite_balanced_accuracy_scope"] == (
        "both_lco_and_lso_cross_fitted_trajectory_weighted_rows"
    )
    assert trace_screen["support_filter_applied"] is False

    action_screen = config["action_gates"]["screen"]
    action_confirm = config["action_gates"]["confirm"]
    assert action_screen["minimum_commit_eligible_pairs"] == 10
    assert action_confirm["minimum_commit_eligible_pairs"] == 59
    assert action_confirm["minimum_tau_A_case_mean_wins"] == 13
    assert action_confirm["minimum_tau_C_case_mean_wins"] == 12
    assert action_confirm["minimum_worst_ten_percent_cvar_tau_A"] == 0.0
    assert action_confirm["minimum_worst_ten_percent_cvar_tau_C"] == 0.0
    assert action_confirm["maximum_catastrophic_events"] == 0
    assert action_confirm["catastrophic_cp_ucb_max"] == 0.05

    authorization = config["authorization"]
    assert authorization["action_runtime_default"] is False
    assert authorization["action_requires_passing_identifiability_gate"] is True
    assert authorization["resource_reallocation"] is False
    assert authorization["scheduler"] is False
    assert config["artifacts"]["runtime_model_bundle_allowed"] is False


def test_canonical_hypergraph_protocol_config_is_complete() -> None:
    config = _load_config()

    _validate_frozen_config(config)
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "permanently retired" in spec
    assert "Union-find" in spec
    assert "next complete normal sweep" in spec
    assert "Failure of either trace gate" in spec
    assert "standalone full matrix cannot authorize" in spec
    assert "numeric tolerance is not accepted" in spec


def test_retired_precision_route_cannot_authorize_new_runtime() -> None:
    config = _load_config()
    retired = config["retired_precision_routes"]

    assert retired["status"] == "permanently_retired_for_new_optimizer_runs"
    assert retired["sigma_ratio"] == 0.5
    assert retired["allowed_use"] == "explicit_offline_frozen_replay_only"
    assert {"sigma", "threshold", "opportunity", "scheduler"} == set(
        retired["forbidden_changes"]
    )


def test_policy_whitelist_excludes_identity_objective_and_future_outcomes() -> None:
    config = _load_config()
    allowed = set(config["state"]["required_derived_features"])
    forbidden = set(config["forbidden_policy_inputs"])

    assert allowed.isdisjoint(forbidden)
    assert {
        "case",
        "seed",
        "function_family",
        "group_index",
        "variable_index",
        "graph_fingerprint",
        "raw_objective",
        "paper_best",
        "terminal_outcome",
        "future_survival",
        "future_overwrite",
        "catastrophic_label",
    }.issubset(forbidden)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("baseline", "profile"), "arac_evidence_action_controller_v38"),
        (("topology", "transitive_closure_allowed"), True),
        (("topology", "preserve_multiple_membership"), False),
        (("coordination_action", "evaluation_fe"), 0),
        (("delayed_credit", "entry_time_closure_allowed"), True),
        (("authorization", "scheduler"), True),
        (("integrity", "formal_trace_requires_clean_tracked_tree"), False),
        (("integrity", "full_requires_prior_screen"), False),
        (("matrices", "trace_full", "requires_prior_screen_pass"), False),
        (("action_gates", "confirm", "minimum_commit_eligible_pairs"), 58),
    ],
)
def test_mutated_preregistration_fails_validation(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(_load_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(AssertionError):
        _validate_frozen_config(config)
