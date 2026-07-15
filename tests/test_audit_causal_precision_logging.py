from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_causal_precision_logging.py"
SPEC = importlib.util.spec_from_file_location("audit_causal_precision_logging", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _digest(label: str) -> str:
    return MODULE.hashlib.sha256(label.encode("utf-8")).hexdigest()


def _refresh_logging_manifest(root: Path) -> None:
    path = root / "causal_logging_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["raw_artifact_sha256"] = {
        name: MODULE._file_sha256(root / name)
        for name in (*MODULE.RAW_FILENAMES, "feature_manifest.json")
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_csv_cell(
    root: Path,
    filename: str,
    *,
    row_index: int,
    field: str,
    value: str,
) -> None:
    header, rows = MODULE._read_csv(root / filename)
    rows[row_index][field] = value
    _write_csv(root / filename, rows, header)
    _refresh_logging_manifest(root)


def _raw_dataset(
    root: Path,
    *,
    cases: tuple[str, ...] = ("A1", "A2", "A3"),
    seeds: tuple[int, ...] = (1, 2, 3),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    features: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    branches: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    randomized: list[dict[str, object]] = []
    for case_index, problem_id in enumerate(cases):
        for seed_index, seed in enumerate(seeds):
            pair_id = MODULE.expected_pair_id(problem_id, seed)
            feature_values = tuple(
                0.2 + 0.01 * feature_index + 0.003 * case_index + 0.001 * seed_index
                for feature_index in range(len(MODULE.UTILITY_FEATURE_NAMES))
            )
            state = MODULE._canonical_feature_state(feature_values)
            logged_arm = MODULE.expected_logged_arm(problem_id, seed)
            decision_fe = 100
            intervention_end_fe = 110
            checkpoint = 100.0
            tau = 0.04 if (case_index + seed_index) % 3 else -0.02
            baseline_error = 80.0
            action_error = baseline_error * math.exp(-tau)
            y0 = MODULE._log_progress(checkpoint, baseline_error)
            y1 = MODULE._log_progress(checkpoint, action_error)
            catastrophic = int(action_error >= 1.2 * baseline_error)
            prefix_hash = _digest(f"prefix:{pair_id}")
            candidate_hash = _digest(f"candidate:{pair_id}")
            controller_hash = _digest(f"controller:{pair_id}")
            random_hash = _digest(f"random:{problem_id}:{seed}")
            decision_id = MODULE.expected_decision_id(
                prefix_record_sha256=prefix_hash,
                feature_sha256=state.feature_sha256,
                not_applicable_reason="",
            )
            features.append(
                {
                    "decision_id": decision_id,
                    **dict(zip(MODULE.UTILITY_FEATURE_NAMES, feature_values, strict=True)),
                }
            )
            audits.append(
                {
                    "protocol_version": MODULE.PROTOCOL_VERSION,
                    "pair_id": pair_id,
                    "decision_id": decision_id,
                    "problem_id": problem_id,
                    "seed": seed,
                    "decision_status": "applicable",
                    "not_applicable_reason": "",
                    "logged_arm": logged_arm,
                    "propensity": "0.5",
                    "decision_fe": decision_fe,
                    "checkpoint_fitness": checkpoint,
                    "remaining_fe": 900,
                    "component_id": "component_0",
                    "component_group_count": 2,
                    "component_shared_var_count": 3,
                    "component_unlocked": 1,
                    "scheduler_revisit_reachable": 1,
                    "scheduler_revisit_cap_fe": 10,
                    "source_phase_i_end_fe": 90,
                    "source_cc_history_end_fe": 91,
                    "source_disagreement_history_end_fe": 92,
                    "source_cma_history_end_fe": 93,
                    "source_end_fe": 93,
                    "prefix_record_sha256": prefix_hash,
                    "checkpoint_candidate_sha256": candidate_hash,
                    "controller_state_sha256": controller_hash,
                    "random_descriptor_sha256": random_hash,
                    "feature_sha256": state.feature_sha256,
                    "feature_schema_sha256": MODULE.FEATURE_SCHEMA_SHA256,
                    "decision_status_match": 1,
                    "decision_id_match": 1,
                    "feature_match": 1,
                    "prefix_match": 1,
                    "controller_state_match": 1,
                    "checkpoint_candidate_match": 1,
                    "random_descriptor_match": 1,
                    "intervention_end_fe_match": 1,
                    "not_applicable_reason_match": 1,
                    "pair_integrity": 1,
                }
            )
            for arm in ("baseline", "action"):
                error = baseline_error if arm == "baseline" else action_error
                branches.append(
                    {
                        "pair_id": pair_id,
                        "decision_id": decision_id,
                        "problem_id": problem_id,
                        "seed": seed,
                        "arm": arm,
                        "lane_id": f"precision_causal_{arm}",
                        "fresh_optimizer_execution": 1,
                        "status": "completed",
                        "result_source": "fresh_hcc_optimizer_execution",
                        "output_root": f"run/{arm}/{problem_id}/{seed}",
                        "decision_status": "applicable",
                        "not_applicable_reason": "",
                        "action_applied": int(arm == "action"),
                        "decision_fe": decision_fe,
                        "intervention_end_fe": intervention_end_fe,
                        "checkpoint_fitness": checkpoint,
                        "normal_sigma": 0.1,
                        "candidate_sigma": 0.01,
                        "applied_sigma": 0.1 if arm == "baseline" else 0.01,
                        "requested_fe": 10,
                        "actual_fe": 10,
                        "configured_max_fes": 1000,
                        "terminal_target_fe": 1000,
                        "terminal_observed_fe": 1000,
                        "terminal_status": "complete",
                        "prefix_record_sha256": prefix_hash,
                        "checkpoint_candidate_sha256": candidate_hash,
                        "controller_state_sha256": controller_hash,
                        "feature_sha256": state.feature_sha256,
                        "random_descriptor_sha256": random_hash,
                        "terminal_error": error,
                        "terminal_record_sha256": _digest(f"terminal:{pair_id}:{arm}"),
                        "optimizer_fe_used": 1000,
                        "same_budget_violation": 0,
                    }
                )
            outcomes.append(
                {
                    "pair_id": pair_id,
                    "decision_id": decision_id,
                    "problem_id": problem_id,
                    "seed": seed,
                    "decision_status": "applicable",
                    "checkpoint_error": checkpoint,
                    "baseline_terminal_error": baseline_error,
                    "action_terminal_error": action_error,
                    "baseline_log_progress": y0,
                    "action_log_progress": y1,
                    "paired_tau": tau,
                    "catastrophic": catastrophic,
                    "equal_checkpoint": 1,
                    "equal_terminal_target_fe": 1,
                    "equal_terminal_observed_fe": 1,
                    "outcome_valid": 1,
                }
            )
            observed_action = int(logged_arm == "action")
            randomized.append(
                {
                    "pair_id": pair_id,
                    "decision_id": decision_id,
                    "problem_id": problem_id,
                    "seed": seed,
                    "logged_arm": logged_arm,
                    "propensity": 0.5,
                    "observed_treatment": observed_action,
                    "observed_terminal_error": action_error if observed_action else baseline_error,
                    "observed_log_progress": y1 if observed_action else y0,
                    "terminal_target_fe": 1000,
                    "terminal_observed_fe": 1000,
                    "outcome_valid": 1,
                }
            )

    _write_csv(root / "causal_decision_features.csv", features, MODULE.FEATURE_COLUMNS)
    _write_csv(root / "causal_decision_audit.csv", audits, MODULE.AUDIT_REQUIRED_COLUMNS)
    _write_csv(root / "causal_branch_manifest.csv", branches, MODULE.BRANCH_REQUIRED_COLUMNS)
    _write_csv(root / "causal_outcomes.csv", outcomes, MODULE.OUTCOME_REQUIRED_COLUMNS)
    _write_csv(root / "randomized_log.csv", randomized, MODULE.RANDOMIZED_REQUIRED_COLUMNS)
    schedule = {
        "protocol_version": MODULE.PROTOCOL_VERSION,
        "status": "scheduled_before_subprocess",
        "randomization_salt": MODULE.RANDOMIZATION_SALT,
        "randomization_algorithm": MODULE.RANDOMIZATION_ALGORITHM,
        "coin_material": "{salt}|{problem_id.upper()}|{int(seed)}",
        "arm_mapping": {"0": "baseline", "1": "action"},
        "preregistration": {
            "path": MODULE.PREREGISTRATION_PATH,
            "sha256": MODULE.PREREGISTRATION_SHA256,
            "commit": MODULE.PREREGISTRATION_COMMIT,
        },
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "problem_id": row["problem_id"],
                "seed": row["seed"],
                "logged_arm": row["logged_arm"],
                "propensity": 0.5,
            }
            for row in audits
        ],
    }
    (root / "causal_randomization_schedule.json").write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    feature_manifest = {
        "schema_version": MODULE.PRE_ACTION_UTILITY_SCHEMA_VERSION,
        "feature_names": list(MODULE.UTILITY_FEATURE_NAMES),
        "feature_schema_sha256": MODULE.FEATURE_SCHEMA_SHA256,
        "features": [
            {"name": name, "formula": f"synthetic:{name}", "source_timing": "strictly_pre_action"}
            for name in MODULE.UTILITY_FEATURE_NAMES
        ],
        "identity_fields": ["problem_id", "seed", "component_id"],
        "identity_fields_location": "causal_decision_audit.csv_only",
        "forbidden_model_fields": ["problem_id", "seed", "final_error"],
        "immutable_snapshot": True,
    }
    (root / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logging_manifest = {
        "protocol_version": MODULE.PROTOCOL_VERSION,
        "offline_only": True,
        "runtime_scheduler_authorized": False,
        "lane_profile": MODULE.PRECISION_LANE_PROFILE,
        "preregistration": {
            "path": MODULE.PREREGISTRATION_PATH,
            "sha256": MODULE.PREREGISTRATION_SHA256,
            "commit": MODULE.PREREGISTRATION_COMMIT,
        },
        "feature_schema": {
            "schema_version": MODULE.PRE_ACTION_UTILITY_SCHEMA_VERSION,
            "feature_schema_sha256": MODULE.FEATURE_SCHEMA_SHA256,
            "feature_names": list(MODULE.UTILITY_FEATURE_NAMES),
        },
        "randomization": {
            "randomization_salt": MODULE.RANDOMIZATION_SALT,
            "randomization_algorithm": MODULE.RANDOMIZATION_ALGORITHM,
            "coin_material": "{salt}|{problem_id.upper()}|{int(seed)}",
            "arm_mapping": {"0": "baseline", "1": "action"},
            "propensity": 0.5,
        },
        "integrity": {
            "status": "pass",
            "failures": [],
            "applicable_pairs": len(audits),
            "total_pairs": len(audits),
        },
        "matrix": {
            "problem_ids": list(cases),
            "seeds": list(seeds),
            "arms": ["baseline", "action"],
            "max_fes": 1000,
            "jobs": 1,
        },
        "raw_artifact_sha256": {},
    }
    (root / "causal_logging_manifest.json").write_text(
        json.dumps(logging_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_logging_manifest(root)
    return root


def test_cli_help_runs_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=SCRIPT_PATH.parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_raw_pair_loader_recomputes_assignment_hashes_and_terminal_tau(tmp_path: Path) -> None:
    root = _raw_dataset(tmp_path / "raw")

    pairs, stats, blockers = MODULE.load_decision_pairs(root)

    assert blockers == []
    assert len(pairs) == 9
    assert stats["critical_feature_missing_rate"] == 0.0
    assert pairs[0].pair_id == MODULE.expected_pair_id(pairs[0].problem_id, pairs[0].seed)
    assert pairs[0].logged_arm == MODULE.expected_logged_arm(pairs[0].problem_id, pairs[0].seed)
    assert pairs[0].tau == pytest.approx(
        math.log(pairs[0].baseline_error / pairs[0].action_error)
    )


def test_loader_fails_closed_on_assignment_watermark_and_branch_outcome_drift(tmp_path: Path) -> None:
    root = _raw_dataset(tmp_path / "raw")
    _rewrite_csv_cell(root, "causal_decision_audit.csv", row_index=0, field="logged_arm", value="baseline")
    _rewrite_csv_cell(root, "causal_decision_audit.csv", row_index=1, field="source_end_fe", value="100")
    _rewrite_csv_cell(root, "causal_branch_manifest.csv", row_index=0, field="terminal_error", value="999")

    _pairs, _stats, blockers = MODULE.load_decision_pairs(root)

    assert any(item.startswith("logged_arm_assignment_drift:") for item in blockers)
    assert any(item.startswith("pre_action_watermark_violation:") for item in blockers)
    assert any(item.startswith("branch_outcome_terminal_error_mismatch:") for item in blockers)


def test_feature_table_rejects_any_non_allowlisted_column(tmp_path: Path) -> None:
    root = _raw_dataset(tmp_path / "raw")
    header, rows = MODULE._read_csv(root / "causal_decision_features.csv")
    rows[0]["problem_id"] = "A1"
    _write_csv(root / "causal_decision_features.csv", rows, (*header, "problem_id"))
    _refresh_logging_manifest(root)

    with pytest.raises(ValueError, match="exactly decision_id plus the 16-feature allowlist"):
        MODULE.load_decision_pairs(root)


def test_material_effect_uses_action_relative_to_baseline() -> None:
    assert MODULE._material_one_percent(-math.log(1.01))
    assert MODULE._material_one_percent(-math.log(0.99))
    assert not MODULE._material_one_percent(-math.log(1.009))


def test_robust_support_combines_raw_range_and_fifth_neighbor() -> None:
    rows = [tuple(float(index + feature) for feature in range(16)) for index in range(8)]
    support = MODULE.fit_robust_support(rows)

    in_support, reasons = support.evaluate(rows[3])
    outside, outside_reasons = support.evaluate(tuple(-100.0 for _ in range(16)))

    assert in_support is True
    assert reasons == ()
    assert outside is False
    assert "feature_out_of_range:remaining_fe_ratio" in outside_reasons
    assert "knn_distance_exceeded" in outside_reasons


def test_exact_risk_bound_requires_59_zero_event_releases() -> None:
    assert MODULE.clopper_pearson_upper(0, 58) > 0.05
    assert MODULE.clopper_pearson_upper(0, 59) <= 0.05


def test_blocked_pilot_writes_both_policy_summaries_and_no_model(tmp_path: Path) -> None:
    root = _raw_dataset(
        tmp_path / "raw",
        cases=("A1", "A2", "A3", "A4", "A5", "A6"),
        seeds=(1, 2, 3, 4, 5),
    )
    output = tmp_path / "audit"

    MODULE.write_reports(
        input_root=root,
        output_root=output,
        stage="pilot",
        tree_count=20,
        policy_bootstrap_count=40,
    )

    gate = json.loads((output / "causal_identifiability_gate.json").read_text(encoding="utf-8"))
    _header, policies = MODULE._read_csv(output / "policy_value_summary.csv")
    assert {row["policy_kind"] for row in policies} == {
        "utility_candidate_policy",
        "safe_release_policy",
    }
    assert gate["runtime_scheduler_authorized"] is False
    assert "pilot-only" in gate["policy_semantics"]["utility_candidate_policy"]
    assert not (output / "causal_risk_precision_model.json").exists()


def test_pilot_gate_uses_both_schemes_for_support_and_catastrophe() -> None:
    pairs = []
    for problem_id in MODULE.PILOT_CASES:
        for seed in MODULE.PILOT_SEEDS:
            values = tuple(0.1 + index * 0.01 for index in range(16))
            state = MODULE._canonical_feature_state(values)
            pairs.append(
                MODULE.DecisionPair(
                    pair_id=MODULE.expected_pair_id(problem_id, seed),
                    decision_id=f"d:{problem_id}:{seed}",
                    problem_id=problem_id,
                    seed=seed,
                    features=values,
                    feature_sha256=state.feature_sha256,
                    logged_arm=MODULE.expected_logged_arm(problem_id, seed),
                    observed_treatment=int(MODULE.expected_logged_arm(problem_id, seed) == "action"),
                    observed_y=0.03,
                    checkpoint_error=1.0,
                    baseline_error=1.0,
                    action_error=math.exp(-0.03),
                    y0=0.0,
                    y1=0.03,
                    tau=0.03 if seed % 2 else -0.02,
                    catastrophic=0,
                )
            )

    def summary(scheme: str, policy: str, *, support: float, catastrophes: int) -> dict[str, object]:
        return {
            "validation_scheme": scheme,
            "policy_kind": policy,
            "scope": "overall",
            "fold_id": "",
            "pair_count": 40,
            "case_count": 8,
            "seed_count": 5,
            "in_support_rate": support,
            "selected_catastrophic_count": catastrophes,
            "selected_catastrophic_cp_ucb": 0.01,
            "dr_policy_value": 0.02,
            "dr_policy_value_lcb_95": 0.01,
            "exact_pair_policy_value": 0.02,
            "sign_balanced_accuracy": 0.8,
            "sign_balanced_accuracy_lcb_95": 0.6,
        }

    policy_rows = [
        summary("LCO", "utility_candidate_policy", support=0.8, catastrophes=0),
        summary("LSO", "utility_candidate_policy", support=0.49, catastrophes=1),
        summary("LCO", "safe_release_policy", support=0.8, catastrophes=0),
        summary("LSO", "safe_release_policy", support=0.8, catastrophes=0),
    ]
    raw_stats = {
        "applicable_pair_count": 40,
        "critical_feature_missing_rate": 0.0,
        "source_matrix": {
            "problem_ids": list(MODULE.PILOT_CASES),
            "seeds": list(MODULE.PILOT_SEEDS),
            "arms": list(MODULE.CAUSAL_ARMS),
            "max_fes": MODULE.STRICT_MAX_FES,
            "jobs": 24,
            "budget_accounting": "strict",
        },
        "source_lane_profile": MODULE.PRECISION_LANE_PROFILE,
    }
    assert MODULE._stage_matrix_matches(raw_stats, "pilot")
    wrong_jobs = {**raw_stats, "source_matrix": {**raw_stats["source_matrix"], "jobs": 23}}
    wrong_budget = {
        **raw_stats,
        "source_matrix": {**raw_stats["source_matrix"], "budget_accounting": "soft"},
    }
    assert not MODULE._stage_matrix_matches(wrong_jobs, "pilot")
    assert not MODULE._stage_matrix_matches(wrong_budget, "pilot")
    gate = MODULE.build_identifiability_gate(
        stage="pilot",
        pairs=pairs,
        raw_stats=raw_stats,
        integrity_blockers=[],
        estimator_blockers=[],
        predictions=[],
        policy_rows=policy_rows,
        tree_count=1000,
        policy_bootstrap_count=2000,
        random_seed=MODULE.DEFAULT_RANDOM_SEED,
    )

    assert gate["pilot_checks"]["both_schemes_in_support_ge_0_50"] is False
    assert gate["pilot_checks"]["candidate_selected_catastrophic_zero_both_schemes"] is False
    assert gate["pilot_criteria_pass"] is False
    wrong_seed_gate = MODULE.build_identifiability_gate(
        stage="pilot",
        pairs=pairs,
        raw_stats=raw_stats,
        integrity_blockers=[],
        estimator_blockers=[],
        predictions=[],
        policy_rows=policy_rows,
        tree_count=1000,
        policy_bootstrap_count=2000,
        random_seed=1,
    )
    assert wrong_seed_gate["pilot_checks"]["frozen_random_seed_20260715"] is False


def test_heldout_counterfactual_mutation_does_not_change_lco_prediction(tmp_path: Path) -> None:
    root = _raw_dataset(
        tmp_path / "raw",
        cases=("A1", "A2", "A3", "A4"),
        seeds=(1, 2, 3, 4),
    )
    pairs, _stats, blockers = MODULE.load_decision_pairs(root)
    assert blockers == []
    mutated = [
        replace(pair, tau=9.0, catastrophic=1, y0=-8.0, y1=1.0)
        if pair.problem_id == "A1"
        else pair
        for pair in pairs
    ]

    original_rows, original_blockers = MODULE.build_crossfit_predictions(
        pairs, scheme="LCO", tree_count=30, random_seed=123
    )
    mutated_rows, mutated_blockers = MODULE.build_crossfit_predictions(
        mutated, scheme="LCO", tree_count=30, random_seed=123
    )
    assert original_blockers == mutated_blockers == []
    original_a1 = {row["pair_id"]: row for row in original_rows if row["fold_id"] == "A1"}
    mutated_a1 = {row["pair_id"]: row for row in mutated_rows if row["fold_id"] == "A1"}
    predictive_fields = (
        "tau_hat",
        "tau_lcb",
        "catastrophic_risk_ucb",
        "mu_baseline",
        "mu_action",
        "dr_effect_score",
    )
    for pair_id in original_a1:
        assert {field: original_a1[pair_id][field] for field in predictive_fields} == {
            field: mutated_a1[pair_id][field] for field in predictive_fields
        }
        assert original_a1[pair_id]["exact_tau"] != mutated_a1[pair_id]["exact_tau"]


def test_runtime_model_export_is_hash_valid_and_has_exact_tree_count() -> None:
    pairs = []
    for case_index, problem_id in enumerate(("A1", "A2", "A3", "A4", "A5", "A6")):
        for seed in range(1, 7):
            values = tuple(
                0.1 + 0.01 * feature + 0.002 * case_index + 0.001 * seed
                for feature in range(16)
            )
            state = MODULE._canonical_feature_state(values)
            tau = 0.03 if seed % 2 else -0.015
            pairs.append(
                MODULE.DecisionPair(
                    pair_id=MODULE.expected_pair_id(problem_id, seed),
                    decision_id=f"d_{problem_id}_{seed}",
                    problem_id=problem_id,
                    seed=seed,
                    features=values,
                    feature_sha256=state.feature_sha256,
                    logged_arm=MODULE.expected_logged_arm(problem_id, seed),
                    observed_treatment=int(MODULE.expected_logged_arm(problem_id, seed) == "action"),
                    observed_y=tau if MODULE.expected_logged_arm(problem_id, seed) == "action" else 0.0,
                    checkpoint_error=1.0,
                    baseline_error=1.0,
                    action_error=math.exp(-tau),
                    y0=0.0,
                    y1=tau,
                    tau=tau,
                    catastrophic=0,
                )
            )

    payload = MODULE.build_model_payload(
        pairs,
        tree_count=MODULE.BOOTSTRAP_TREE_COUNT,
        random_seed=123,
    )

    bundle = MODULE.CausalRiskModelBundle.from_mapping(payload)
    assert bundle.model_sha256 == payload["model_sha256"]
    assert len(payload["utility"]["bootstrap_trees"]) == 1000
    assert len(payload["catastrophic_risk"]["bootstrap_trees"]) == 1000
    state = MODULE._canonical_feature_state(pairs[0].features)
    ood = payload["ood"]
    scaled = tuple(
        (value - median) / (iqr if iqr > 0.0 else 1.0)
        for value, median, iqr in zip(
            state.feature_values, ood["median"], ood["iqr"], strict=True
        )
    )
    utility_predictions = [
        MODULE._predict_tree(tree, scaled)
        for tree in payload["utility"]["bootstrap_trees"]
    ]
    risk_predictions = [
        MODULE._predict_tree(tree, scaled)
        for tree in payload["catastrophic_risk"]["bootstrap_trees"]
    ]
    estimate = bundle.estimate(state)
    assert estimate.tau_hat == pytest.approx(sum(utility_predictions) / 1000)
    assert estimate.tau_lcb == pytest.approx(
        MODULE._quantile(utility_predictions, 0.05)
        - payload["utility"]["conformal_margin"]
    )
    assert estimate.catastrophic_risk_ucb == pytest.approx(
        max(
            MODULE._quantile(risk_predictions, 0.95),
            MODULE._predict_tree(
                payload["catastrophic_risk"]["clopper_pearson_tree"], scaled
            ),
        )
    )
    support = MODULE.RobustSupport(
        median=tuple(ood["median"]),
        iqr=tuple(ood["iqr"]),
        minimum=tuple(ood["minimum"]),
        maximum=tuple(ood["maximum"]),
        reference_scaled=tuple(tuple(row) for row in ood["reference_scaled"]),
        knn_distance_threshold=ood["knn_distance_threshold"],
    )
    expected_in_distribution, expected_reasons = support.evaluate(state.feature_values)
    assert estimate.in_distribution is expected_in_distribution
    assert estimate.ood_reasons == expected_reasons
