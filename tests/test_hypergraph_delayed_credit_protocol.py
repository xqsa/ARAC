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
    assert config["config_schema_version"] == 1
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

    state = config["state"]
    assert state["difficulty_formula"] == "1-success_ratio_3"
    assert state["stagnation_formula"] == "min(consecutive_u_le_0,3)/3"
    assert state["actual_fe_scope"].startswith("full_native_group_interval")
    assert state["proposal_capture_watermark"] == (
        "after_group_local_rescue_recovery_before_relation_writeback"
    )
    assert state["missing_required_state"] == "inapplicable_fail_closed"
    assert len(state["required_derived_features"]) == 8

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
    assert matrices["action_confirm"]["seeds"] == list(range(109, 117))

    trace_full = config["trace_gates"]["full"]
    assert trace_full["minimum_applicable_trajectories"] == 120
    assert trace_full["minimum_lco_in_support_fraction"] == 0.6
    assert trace_full["minimum_lso_in_support_fraction"] == 0.6
    assert trace_full["overwrite_balanced_accuracy_lcb_strictly_above"] == 0.5

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
