from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from arac.actions import ActionFamily
from arac.backends.hcc import HccAobExecutionRequest, HccAobExecutionResult, build_hcc_aob_smoke_command
from experiments.pilots.exp_003_hcc_runtime_consumer_smoke.run import (
    LaneConfig,
    PRECISION_RESPONSE_ARMS,
    PRECISION_RESPONSE_BRANCH_FIELDS,
    PRECISION_RESPONSE_CONFIG_PATH,
    PRECISION_RESPONSE_CONFIG_SHA256,
    PRECISION_RESPONSE_PREREGISTRATION_PATH,
    PRECISION_RESPONSE_PREREGISTRATION_SHA256,
    _precision_response_raw_rows,
    _sha256_file,
    lanes_for_profile,
    parse_args,
)
import experiments.pilots.exp_003_hcc_runtime_consumer_smoke.run as exp003
from scripts.assemble_precision_response_pilot import (
    assemble_precision_response_pilot,
)
from scripts.audit_precision_response_loop import audit_precision_response


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0] if rows else ["empty"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def test_precision_response_profile_is_three_fresh_v37_arms() -> None:
    lanes = lanes_for_profile("precision_response_logging")

    assert tuple(lane.precision_response_arm for lane in lanes) == PRECISION_RESPONSE_ARMS
    assert {lane.runner_action_name for lane in lanes} == {
        "arac_evidence_action_controller_v37"
    }
    assert all(lane.relation_dispatch_enabled for lane in lanes)
    parsed = parse_args(
        [
            "--lane-profile",
            "precision_response_logging",
            "--response-arms",
            "a0_v37",
            "a2_probe_gated",
        ]
    )
    assert parsed.response_arms == ["a0_v37", "a2_probe_gated"]


def test_precision_response_preregistration_and_config_hashes_are_frozen() -> None:
    assert _sha256_file(exp003.ARAC_REPO_ROOT / PRECISION_RESPONSE_CONFIG_PATH) == (
        PRECISION_RESPONSE_CONFIG_SHA256
    )
    assert _sha256_file(
        exp003.ARAC_REPO_ROOT / PRECISION_RESPONSE_PREREGISTRATION_PATH
    ) == PRECISION_RESPONSE_PREREGISTRATION_SHA256


def test_backend_emits_independent_response_flag_and_blocks_v41() -> None:
    request = HccAobExecutionRequest(
        problem_id="E2",
        seed=1,
        max_fes=5_000,
        output_dir=Path("results/precision-response-command"),
        arac_action="arac_evidence_action_controller_v37",
        enable_relation_dispatch=True,
        relation_policy_mode="controller_v31",
        precision_response_arm="a2_probe_gated",
    )

    command = build_hcc_aob_smoke_command(request)

    index = command.argv.index("--precision-response-arm")
    assert command.argv[index + 1] == "a2_probe_gated"
    assert "--precision-causal-arm" not in command.argv
    with pytest.raises(ValueError, match="v41 is frozen"):
        build_hcc_aob_smoke_command(
            HccAobExecutionRequest(
                problem_id="E2",
                seed=1,
                max_fes=5_000,
                output_dir=Path("results/frozen-v41"),
                arac_action="arac_evidence_action_controller_v41",
            )
        )
    replay = build_hcc_aob_smoke_command(
        HccAobExecutionRequest(
            problem_id="E2",
            seed=1,
            max_fes=5_000,
            output_dir=Path("results/frozen-v41-replay"),
            arac_action="arac_evidence_action_controller_v41",
            offline_frozen_replay=True,
        )
    )
    assert "--offline-frozen-replay" in replay.argv


def _response_records(tmp_path: Path, *, released: bool):
    records = []
    terminal_errors = {
        "a0_v37": 10.0,
        "a1_probe_only": 8.0,
        "a2_probe_gated": 5.0 if released else 8.0,
    }
    gate_features = {
        "decision_id": "decision-same",
        "valid": "True",
        "invalid_reason": "",
        "pair_count": "16",
        "direction_hash_match": "True",
        "paired_win_count": "13",
        "paired_win_rate": "0.8125",
        "paired_win_lcb": "0.6",
        "median_relative_advantage": "0.1",
        "large_loss_count": "0",
        "large_loss_rate": "0",
        "large_loss_ucb": "0.15",
        "standardized_diversity_ratio": "1",
        "normal_boundary_hit_count": "0",
        "precision_boundary_hit_count": "0",
        "normal_boundary_hit_rate": "0",
        "precision_boundary_hit_rate": "0",
        "precision_best_relative_gain": "0.1",
    }
    for arm in PRECISION_RESPONSE_ARMS:
        output_root = tmp_path / arm
        trace = {
            "protocol_version": "precision-response-loop-v1",
            "problem_id": "E2",
            "seed": "60",
            "response_arm": arm,
            "decision_status": "applicable",
            "not_applicable_reason": "",
            "decision_id": "decision-same",
            "decision_fe": "100",
            "prefix_record_sha256": "p" * 64,
            "checkpoint_candidate_sha256": "c" * 64,
            "probe_seed": "123",
            "probe_executed": "0" if arm == "a0_v37" else "1",
            "probe_fe": "0" if arm == "a0_v37" else "32",
            "gate_state_sha256": "" if arm == "a0_v37" else "g" * 64,
            "gate_would_release": "0" if arm == "a0_v37" else "1",
            "lease_applied": str(int(arm == "a2_probe_gated" and released)),
            "gate_reason": "test",
            "main_requested_fe": "100",
            "main_actual_fe": "100",
            "intervention_end_fe": "232",
            "delayed_credit_status": "resolved" if released and arm == "a2_probe_gated" else "not_released",
            "terminal_target_fe": "4990",
            "terminal_observed_fe": "4990",
            "terminal_error": str(terminal_errors[arm]),
            "terminal_status": "complete",
        }
        _write_rows(output_root / "E2_precision_response_trace.csv", [trace])
        _write_rows(output_root / "E2_precision_probe_audit.csv", [], ["empty"])
        _write_rows(output_root / "E2_precision_lease_credit.csv", [], ["empty"])
        _write_rows(
            output_root / "E2_precision_probe_gate_features.csv",
            [] if arm == "a0_v37" else [gate_features],
            list(gate_features),
        )
        lane = LaneConfig(
            lane_id=f"precision_response_{arm}",
            action_family=ActionFamily.TRAJECTORY,
            selected_action_name="arac_evidence_action_controller_v37",
            runner_action_name="arac_evidence_action_controller_v37",
            dispatch_scope="precision_response_test",
            precision_response_arm=arm,
        )
        result = HccAobExecutionResult(
            problem_id="E2",
            seed=60,
            max_fes=5_000,
            final_error=terminal_errors[arm],
            fe_used=4_995,
            time_seconds=0.0,
            output_root=output_root,
            fresh_optimizer_execution=True,
            status="completed",
            result_source="test",
            optimizer_final_fe_used=4_995,
        )
        records.append({"lane": lane, "result": result})
    return records


@pytest.mark.parametrize("released", [False, True])
def test_triplet_audit_enforces_probe_and_abstain_parity(
    tmp_path: Path,
    released: bool,
) -> None:
    branches, _, features, _, triplets, failures = _precision_response_raw_rows(
        _response_records(tmp_path, released=released)
    )

    assert failures == []
    assert len(branches) == 3
    assert len(features) == 1
    assert triplets[0]["triplet_integrity"] == "1"
    assert triplets[0]["a2_released"] == str(int(released))
    expected_tau = math.log(8.0 / (5.0 if released else 8.0))
    assert float(triplets[0]["tau_lease"]) == pytest.approx(expected_tau)


def test_treatment_only_aggregate_preserves_gate_features(tmp_path: Path) -> None:
    records = [
        record
        for record in _response_records(tmp_path, released=False)
        if record["lane"].precision_response_arm != "a0_v37"
    ]

    branches, _, features, _, triplets, failures = _precision_response_raw_rows(
        records
    )

    assert failures == []
    assert len(branches) == 2
    assert len(features) == 1
    assert triplets == []


def test_phased_response_assembly_builds_complete_triplet(tmp_path: Path) -> None:
    records = _response_records(tmp_path / "raw", released=True)
    coverage_records = [
        record
        for record in records
        if record["lane"].precision_response_arm == "a0_v37"
    ]
    treatment_records = [
        record
        for record in records
        if record["lane"].precision_response_arm != "a0_v37"
    ]
    coverage_branches, _, _, _, _, coverage_failures = (
        _precision_response_raw_rows(coverage_records)
    )
    treatment_branches, _, treatment_features, _, _, treatment_failures = (
        _precision_response_raw_rows(treatment_records)
    )
    assert coverage_failures == []
    assert treatment_failures == []

    coverage = tmp_path / "coverage"
    treatment = tmp_path / "treatment"
    coverage.mkdir()
    treatment.mkdir()
    common_manifest = {
        "protocol_version": "precision-response-loop-v1",
        "status": "pass",
        "integrity_failures": [],
        "config": {"path": "config", "sha256": "c" * 64},
        "preregistration": {"path": "spec", "sha256": "p" * 64},
        "forbidden_outputs": ["causal_risk_precision_model.json"],
    }
    for root, arms, branches in (
        (coverage, ["a0_v37"], coverage_branches),
        (treatment, ["a1_probe_only", "a2_probe_gated"], treatment_branches),
    ):
        (root / "precision_response_manifest.json").write_text(
            json.dumps(
                {**common_manifest, "arms": arms, "run_count": len(branches)}
            ),
            encoding="utf-8",
        )
        (root / "runtime_environment.json").write_text(
            json.dumps({"status": "pass"}),
            encoding="utf-8",
        )
        _write_rows(
            root / "precision_response_branch_manifest.csv",
            branches,
            PRECISION_RESPONSE_BRANCH_FIELDS,
        )
        _write_rows(
            root / "same_budget_ledger.csv",
            [{"same_budget_violation": "0"}],
        )
        _write_rows(root / "aob_input_manifest.csv", [{"unchanged": "1"}])
        _write_rows(
            root / "anti_leakage_audit.csv",
            [{"audit_status": "pass"}],
        )
        _write_rows(root / "precision_probe_audit.csv", [], ["empty"])
        _write_rows(
            root / "precision_lease_credit.csv",
            [],
            ["component_credit_status"],
        )

    feature_fields = list(treatment_features[0])
    _write_rows(
        coverage / "precision_probe_gate_features.csv",
        [],
        feature_fields,
    )
    _write_rows(
        treatment / "precision_probe_gate_features.csv",
        treatment_features,
        feature_fields,
    )
    output = tmp_path / "assembled"

    manifest = assemble_precision_response_pilot(coverage, treatment, output)

    with (output / "precision_response_triplets.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        triplets = list(csv.DictReader(handle))
    gate = audit_precision_response(output, resamples=10)
    assert manifest["run_count"] == 3
    assert len(triplets) == 1
    assert triplets[0]["triplet_integrity"] == "1"
    assert gate["integrity"]["status"] == "pass"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assemble_precision_response_pilot(coverage, treatment, output)


def test_manifest_does_not_authorize_runtime_model() -> None:
    payload = json.loads(
        (exp003.ARAC_REPO_ROOT / PRECISION_RESPONSE_CONFIG_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert payload["lease"]["renewal_enabled"] is False
