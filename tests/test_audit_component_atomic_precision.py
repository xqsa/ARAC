from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_component_atomic_precision.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_component_atomic_precision_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset(
    root: Path,
    *,
    stage: str,
    applicable_count: int | None = None,
    effect: float = 0.02,
    excluded_applicable_seed: int | None = None,
) -> Path:
    config = AUDIT._load_config()
    stage_config = config[stage]
    cases = tuple(stage_config["cases"])
    seeds = tuple(stage_config["seeds"])
    registered = [(case, seed) for case in cases for seed in seeds]
    if applicable_count is None:
        applicable_count = len(registered)
    config_hash = AUDIT._file_sha256(AUDIT.CONFIG_PATH)
    spec_hash = AUDIT._file_sha256(AUDIT.SPEC_PATH)
    digest_a = "a" * 64
    digest_b = "b" * 64
    digest_c = "c" * 64
    digest_d = "d" * 64
    commit = "1" * 40
    branch_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    survival_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []

    for index, (problem_id, seed) in enumerate(registered):
        applicable = index < applicable_count and seed != excluded_applicable_seed
        pair_id = AUDIT.component_action_pair_id(problem_id, seed)
        tau_h = effect if applicable else 0.0
        tau_t = effect if applicable else 0.0
        a0_h = 100.0
        a1_h = a0_h * math.exp(-tau_h)
        a0_t = 100.0
        a1_t = a0_t * math.exp(-tau_t)
        public_a0 = digest_c
        public_a1 = digest_d if applicable else public_a0
        plan = digest_b if applicable else ""
        reason = "" if applicable else "no_safe_component_opportunity"
        for arm in AUDIT.ARMS:
            action_arm = arm == "a1_precision_component_once"
            branch_rows.append(
                {
                    "protocol_version": AUDIT.PROTOCOL_VERSION,
                    "stage": stage,
                    "pair_id": pair_id,
                    "problem_id": problem_id,
                    "seed": seed,
                    "arm": arm,
                    "fresh_optimizer_execution": 1,
                    "status": "complete",
                    "result_source": "fresh_subprocess",
                    "action_applied": int(applicable and action_arm),
                    "decision_status": "applicable" if applicable else "not_applicable",
                    "not_applicable_reason": reason,
                    "decision_fe": 1000 if applicable else 0,
                    "decision_outer_iter": 2 if applicable else 0,
                    "component_id": "component_x" if applicable else "",
                    "component_group_indices": "0;1;2;3" if applicable else "",
                    "component_group_count": 4 if applicable else 0,
                    "component_shared_var_count": 3 if applicable else 0,
                    "component_horizon_requested_fe": 128 if applicable else 0,
                    "component_horizon_actual_fe": (
                        (64 if action_arm else 128) if applicable else 0
                    ),
                    "component_horizon_interval_fe": (
                        (80 if action_arm else 144) if applicable else 0
                    ),
                    "component_end_fe": (
                        (1080 if action_arm else 1144) if applicable else 0
                    ),
                    "terminal_target_fe": 2_999_984,
                    "terminal_completion_tolerance_fe": 16,
                    "terminal_observed_fe": 2_999_984,
                    "horizon_error": a1_h if action_arm else a0_h,
                    "terminal_error": a1_t if action_arm else a0_t,
                    "prefix_record_sha256": digest_a,
                    "checkpoint_candidate_sha256": digest_b,
                    "crn_descriptor_sha256": digest_c if applicable else "",
                    "component_plan_sha256": plan,
                    "normal_sigma": 0.5 if applicable else "",
                    "precision_sigma": 0.25 if applicable else "",
                    "public_trace_sha256": public_a1 if action_arm else public_a0,
                    "terminal_record_sha256": digest_d if action_arm else digest_c,
                    "optimizer_fe_used": 2_999_984,
                    "configured_max_fes": 3_000_000,
                    "same_budget_violation": 0,
                    "component_plan_frozen": int(applicable),
                    "mid_horizon_redispatch_count": 0,
                    "unique_h_endpoint": int(applicable),
                    "component_horizon_complete": int(applicable),
                    "delayed_review_fe": (
                        (1080 if action_arm else 1144) if applicable else 0
                    ),
                    "delayed_review_outer_iter": 3 if applicable else 0,
                    "delayed_review_group_index": 0 if applicable else "",
                    "config_sha256": config_hash,
                    "preregistration_sha256": spec_hash,
                    "preregistration_git_commit": (
                        AUDIT.PREREGISTRATION_GIT_COMMIT
                    ),
                    "source_git_commit": commit,
                }
            )
            budget_rows.append(
                {
                    "protocol_version": AUDIT.PROTOCOL_VERSION,
                    "stage": stage,
                    "pair_id": pair_id,
                    "problem_id": problem_id,
                    "seed": seed,
                    "arm": arm,
                    "fresh_optimizer_execution": 1,
                    "group_indices": "0;1;2;3" if applicable else "",
                    "population_sizes": "16;16;16;16" if applicable else "",
                    "requested_group_fes": "32;32;32;32" if applicable else "",
                    "actual_group_fes": (
                        ("16;16;16;16" if action_arm else "32;32;32;32")
                        if applicable
                        else ""
                    ),
                    "interval_group_fes": (
                        ("20;20;20;20" if action_arm else "36;36;36;36")
                        if applicable
                        else ""
                    ),
                    "auxiliary_group_fes": (
                        "4;4;4;4" if applicable else ""
                    ),
                    "applied_group_sigmas": (
                        ("0.25;0.25;0.25;0.25" if action_arm else "0.5;0.5;0.5;0.5")
                        if applicable
                        else ""
                    ),
                    "normal_sigma": 0.5 if applicable else "",
                    "precision_sigma": 0.25 if applicable else "",
                    "component_horizon_actual_fe": (
                        (64 if action_arm else 128) if applicable else 0
                    ),
                    "component_interval_actual_fe": (
                        (80 if action_arm else 144) if applicable else 0
                    ),
                    "component_auxiliary_fe": 16 if applicable else 0,
                    "component_precision_fe": 64 if applicable and action_arm else 0,
                    "optimizer_fe_used": 2_999_984,
                    "configured_max_fes": 3_000_000,
                    "same_budget_violation": 0,
                    "strict_terminal_reached": 1,
                    "aob_unchanged": 1,
                    "anti_leakage_pass": 1,
                    "component_endpoint_closed": int(applicable),
                    "delayed_endpoint_closed": int(applicable),
                }
            )
        component_rows.append(
            {
                "protocol_version": AUDIT.PROTOCOL_VERSION,
                "stage": stage,
                "pair_id": pair_id,
                "problem_id": problem_id,
                "seed": seed,
                "applicable": int(applicable),
                "component_closed": int(applicable),
                "endpoint_sequence_match": 1,
                "a0_horizon_error": a0_h,
                "a1_horizon_error": a1_h,
                "tau_H": tau_h,
                "component_catastrophic": 0,
            }
        )
        survival_rows.append(
            {
                "protocol_version": AUDIT.PROTOCOL_VERSION,
                "stage": stage,
                "pair_id": pair_id,
                "problem_id": problem_id,
                "seed": seed,
                "applicable": int(applicable),
                "component_closed": int(applicable),
                "delayed_closed": int(applicable),
                "a0_shared_path_l1": 1.0 if applicable else "",
                "a1_shared_path_l1": 1.0 if applicable else "",
                "a0_shared_net_l1": 0.4 if applicable else "",
                "a1_shared_net_l1": 0.6 if applicable else "",
                "a0_delayed_drift_l1": 0.28 if applicable else "",
                "a1_delayed_drift_l1": 0.3 if applicable else "",
                "a0_s_h": 0.4 if applicable else "",
                "a1_s_h": 0.6 if applicable else "",
                "delta_s_h": 0.2 if applicable else "",
                "a0_strict_survival": 1 if applicable else "",
                "a1_strict_survival": 1 if applicable else "",
                "a0_s_d": 0.3 if applicable else "",
                "a1_s_d": 0.5 if applicable else "",
                "delta_s_d": 0.2 if applicable else "",
            }
        )
        pair_rows.append(
            {
                "protocol_version": AUDIT.PROTOCOL_VERSION,
                "stage": stage,
                "pair_id": pair_id,
                "problem_id": problem_id,
                "seed": seed,
                "pair_integrity": 1,
                "applicable": int(applicable),
                "not_applicable_reason": reason,
                "prefix_match": 1,
                "checkpoint_match": 1,
                "plan_match": 1,
                "action_applied": int(applicable),
                "abstain_parity": int(not applicable),
                "a0_terminal_error": a0_t,
                "a1_terminal_error": a1_t,
                "tau_T": tau_t,
                "terminal_catastrophic": 0,
            }
        )

    names = AUDIT._artifact_names(config)
    _write_csv(root / names["branches"], branch_rows, AUDIT.BRANCH_COLUMNS)
    _write_csv(
        root / names["component_outcomes"], component_rows, AUDIT.COMPONENT_COLUMNS
    )
    _write_csv(root / names["survival"], survival_rows, AUDIT.SURVIVAL_COLUMNS)
    _write_csv(root / names["pairs"], pair_rows, AUDIT.PAIR_COLUMNS)
    _write_csv(root / names["budget"], budget_rows, AUDIT.BUDGET_COLUMNS)
    return root


def _rewrite_cell(path: Path, row_index: int, field: str, value: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[row_index][field] = value
    _write_csv(path, rows, fields)


def _screen_gate(path: Path) -> Path:
    source_root = _dataset(path.parent / f"{path.stem}_source", stage="screen")
    payload = AUDIT.audit_component_atomic_precision(
        source_root,
        stage="screen",
        resamples=2000,
    )
    assert payload["status"] == "screen_pass"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_freezes_two_arms_and_one_component_wide_dose() -> None:
    config = AUDIT._load_config()

    assert AUDIT._file_sha256(AUDIT.CONFIG_PATH) == AUDIT.CONFIG_SHA256
    assert AUDIT._file_sha256(AUDIT.SPEC_PATH) == AUDIT.SPEC_SHA256
    assert tuple(config["arms"]) == (
        "a0_v37",
        "a1_precision_component_once",
    )
    assert config["action"]["phase_rescue_must_be_inactive"] is True
    assert (
        config["action"]["active_auxiliary_fe_route"]
        == "abstain_without_consuming_once_lock"
    )
    assert config["action"]["dose"] == "all_component_groups_once"
    assert (
        config["action"]["execution"]
        == "trajectory_branch_local_sequential_component_horizon"
    )
    assert not {"probe", "response_gate", "group_mask", "response_arms"}.intersection(
        config["action"]
    )


def test_two_way_bootstrap_and_exact_risk_are_deterministic() -> None:
    rows = [
        {"problem_id": case, "seed": seed, "tau": 0.1}
        for case in ("A4", "A5", "E2", "S2")
        for seed in (65, 66, 67)
    ]

    first = AUDIT._effect_summary(rows, "tau", resamples=200, seed=7)
    second = AUDIT._effect_summary(rows, "tau", resamples=200, seed=7)

    assert first == second
    assert first["lcb_95"] == pytest.approx(0.1)
    assert AUDIT.clopper_pearson_upper(0, 58) > 0.05
    assert AUDIT.clopper_pearson_upper(0, 59) <= 0.05


def test_screen_gate_allows_generation_complete_treatment_early_stop(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "screen", stage="screen", applicable_count=35)

    gate = AUDIT.audit_component_atomic_precision(root, stage="screen")

    assert gate["status"] == "screen_pass"
    assert gate["population"]["itt_count"] == 40
    assert gate["population"]["att_count"] == 35
    assert gate["effects"]["itt"]["tau_T"]["mean"] > 0.0
    assert gate["runtime_scheduler_authorized"] is False
    assert gate["full_24_authorized"] is False
    assert len(gate["input_artifact_sha256"]) == 5


def test_screen_requires_applicable_coverage_of_every_registered_seed(
    tmp_path: Path,
) -> None:
    root = _dataset(
        tmp_path / "missing_att_seed",
        stage="screen",
        excluded_applicable_seed=69,
    )

    gate = AUDIT.audit_component_atomic_precision(root, stage="screen")

    assert gate["status"] == "screen_no_go"
    assert gate["checks"]["all_applicable_seeds_present"] is False


def test_confirm_gate_passes_complete_positive_two_arm_matrix(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "confirm", stage="confirm")
    screen_gate = _screen_gate(tmp_path / "screen_gate.json")

    gate = AUDIT.audit_component_atomic_precision(
        root,
        stage="confirm",
        screen_gate_path=screen_gate,
    )

    assert gate["status"] == "confirm_pass"
    assert gate["population"]["itt_count"] == 192
    assert gate["population"]["att_count"] == 192
    assert gate["checks"]["catastrophic_cp_ucb_within_0_05"] is True
    assert gate["checks"]["non_screen_16_mean_direction_positive"] is True
    assert gate["action_validity_supported"] is True
    assert gate["runtime_scheduler_authorized"] is False


def test_confirm_requires_an_explicit_passing_screen_gate(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "confirm_blocked", stage="confirm")

    missing = AUDIT.audit_component_atomic_precision(root, stage="confirm")
    assert missing["status"] == "confirm_no_go"
    assert "explicit screen gate" in missing["integrity"]["blockers"][0]

    failed_path = _screen_gate(tmp_path / "failed_screen_gate.json")
    payload = json.loads(failed_path.read_text(encoding="utf-8"))
    payload["status"] = "screen_no_go"
    failed_path.write_text(json.dumps(payload), encoding="utf-8")
    failed = AUDIT.audit_component_atomic_precision(
        root,
        stage="confirm",
        screen_gate_path=failed_path,
    )
    assert failed["status"] == "confirm_no_go"
    assert "did not pass" in failed["integrity"]["blockers"][0]

    different_source_path = _screen_gate(tmp_path / "different_source_gate.json")
    different_source = json.loads(
        different_source_path.read_text(encoding="utf-8")
    )
    different_source["source_git_commit"] = "2" * 40
    different_source_path.write_text(
        json.dumps(different_source), encoding="utf-8"
    )
    mismatched = AUDIT.audit_component_atomic_precision(
        root,
        stage="confirm",
        screen_gate_path=different_source_path,
    )
    assert mismatched["status"] == "confirm_no_go"
    assert "does not match audited screen artifacts" in mismatched[
        "integrity"
    ]["blockers"][0]


def test_confirm_revalidates_screen_checks_and_artifact_hashes(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "confirm_revalidation", stage="confirm")
    string_false_path = _screen_gate(tmp_path / "string_false_gate.json")
    payload = json.loads(string_false_path.read_text(encoding="utf-8"))
    payload["checks"] = {"all_hard_gates": "false"}
    string_false_path.write_text(json.dumps(payload), encoding="utf-8")

    string_false = AUDIT.audit_component_atomic_precision(
        root,
        stage="confirm",
        screen_gate_path=string_false_path,
    )
    assert string_false["status"] == "confirm_no_go"
    assert "did not pass" in string_false["integrity"]["blockers"][0]

    wrong_keys_path = _screen_gate(tmp_path / "wrong_keys_gate.json")
    wrong_keys_payload = json.loads(wrong_keys_path.read_text(encoding="utf-8"))
    removed_name, removed_hash = wrong_keys_payload[
        "input_artifact_sha256"
    ].popitem()
    wrong_keys_payload["input_artifact_sha256"][f"forged_{removed_name}"] = (
        removed_hash
    )
    wrong_keys_path.write_text(json.dumps(wrong_keys_payload), encoding="utf-8")
    wrong_keys = AUDIT.audit_component_atomic_precision(
        root,
        stage="confirm",
        screen_gate_path=wrong_keys_path,
    )
    assert wrong_keys["status"] == "confirm_no_go"
    assert "did not pass" in wrong_keys["integrity"]["blockers"][0]

    changed_path = _screen_gate(tmp_path / "changed_artifact_gate.json")
    changed_payload = json.loads(changed_path.read_text(encoding="utf-8"))
    artifact_name = next(iter(changed_payload["input_artifact_sha256"]))
    (Path(changed_payload["source_root"]) / artifact_name).write_text(
        "changed\n", encoding="utf-8"
    )
    changed = AUDIT.audit_component_atomic_precision(
        root,
        stage="confirm",
        screen_gate_path=changed_path,
    )
    assert changed["status"] == "confirm_no_go"
    assert "screen artifacts changed" in changed["integrity"]["blockers"][0]


def test_outcome_or_survival_drift_fails_closed(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "drift", stage="screen", applicable_count=35)
    config = AUDIT._load_config()
    names = AUDIT._artifact_names(config)
    _rewrite_cell(root / names["pairs"], 0, "tau_T", "9")
    _rewrite_cell(root / names["survival"], 1, "delta_s_d", "-0.5")
    _rewrite_cell(root / names["survival"], 1, "a0_shared_net_l1", "0.9")

    gate = AUDIT.audit_component_atomic_precision(root, stage="screen")

    assert gate["status"] == "screen_no_go"
    assert gate["integrity"]["status"] == "blocked"
    blockers = "\n".join(gate["integrity"]["blockers"])
    assert "pair_recompute_mismatch:tau_T" in blockers
    assert "survival_recompute_mismatch:delta_s_d" in blockers
    assert "survival_path_recompute_mismatch:a0" in blockers


def test_pair_identity_and_component_interval_accounting_fail_closed(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "identity_interval", stage="screen")
    names = AUDIT._artifact_names(AUDIT._load_config())
    _rewrite_cell(root / names["pairs"], 0, "pair_id", "forged")
    _rewrite_cell(root / names["budget"], 2, "component_auxiliary_fe", "15")
    _rewrite_cell(root / names["branches"], 2, "delayed_review_group_index", "3")

    gate = AUDIT.audit_component_atomic_precision(root, stage="screen")

    assert gate["status"] == "screen_no_go"
    blockers = "\n".join(gate["integrity"]["blockers"])
    assert "pair_id_mismatch" in blockers
    assert "component_interval_total_mismatch" in blockers
    assert "delayed_review_watermark_invalid" in blockers


def test_non_applicable_pair_must_be_bit_equivalent(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "parity", stage="screen", applicable_count=30)
    config = AUDIT._load_config()
    names = AUDIT._artifact_names(config)
    # Pair 31 is the first registered no-op pair; its A1 branch is row 61.
    _rewrite_cell(root / names["branches"], 61, "terminal_error", "101")

    gate = AUDIT.audit_component_atomic_precision(root, stage="screen")

    assert gate["status"] == "screen_no_go"
    assert any(
        "non_applicable_not_bit_equivalent" in blocker
        for blocker in gate["integrity"]["blockers"]
    )


def test_all_applicable_component_groups_must_receive_the_frozen_dose(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dose", stage="screen", applicable_count=35)
    config = AUDIT._load_config()
    names = AUDIT._artifact_names(config)
    # Pair 1 A1 is budget row 1. One normal-sigma group is an illegal mask.
    _rewrite_cell(
        root / names["budget"],
        1,
        "applied_group_sigmas",
        "0.25;0.5;0.25;0.25",
    )

    gate = AUDIT.audit_component_atomic_precision(root, stage="screen")

    assert gate["status"] == "screen_no_go"
    assert any(
        "budget_component_dose_mismatch" in blocker
        for blocker in gate["integrity"]["blockers"]
    )


def test_natural_early_stop_must_end_on_a_complete_generation(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "generation", stage="screen", applicable_count=35)
    config = AUDIT._load_config()
    names = AUDIT._artifact_names(config)
    _rewrite_cell(
        root / names["budget"],
        1,
        "actual_group_fes",
        "17;16;16;16",
    )
    _rewrite_cell(root / names["budget"], 1, "component_horizon_actual_fe", "65")
    _rewrite_cell(
        root / names["branches"], 1, "component_horizon_actual_fe", "65"
    )

    gate = AUDIT.audit_component_atomic_precision(root, stage="screen")

    assert gate["status"] == "screen_no_go"
    assert any(
        "budget_actual_fe_not_generation_complete" in blocker
        for blocker in gate["integrity"]["blockers"]
    )


def test_missing_input_fails_closed_and_cli_never_authorizes_runtime(
    tmp_path: Path,
) -> None:
    gate = AUDIT.audit_component_atomic_precision(tmp_path, stage="screen")

    assert gate["status"] == "screen_no_go"
    assert gate["runtime_scheduler_authorized"] is False
    assert gate["full_24_authorized"] is False
    assert gate["integrity"]["status"] == "blocked"
