from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from arac.backends.hcc import (
    HccAobExecutionRequest,
    HccAobExecutionResult,
    build_hcc_aob_smoke_command,
)
from experiments.pilots.exp_003_hcc_runtime_consumer_smoke import run as exp003


AUDIT_SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_component_atomic_precision.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "component_atomic_precision_exp_e2e", AUDIT_SCRIPT
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
COMPONENT_AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = COMPONENT_AUDIT
AUDIT_SPEC.loader.exec_module(COMPONENT_AUDIT)


def _request(**overrides: object) -> HccAobExecutionRequest:
    values: dict[str, object] = {
        "problem_id": "E2",
        "seed": 1,
        "max_fes": 5_000,
        "output_dir": Path("results/component-precision-command"),
        "arac_action": "arac_evidence_action_controller_v37",
        "enable_relation_dispatch": True,
        "relation_policy_mode": "controller_v31",
    }
    values.update(overrides)
    return HccAobExecutionRequest(**values)


def test_component_precision_profile_is_two_fresh_v37_arms() -> None:
    lanes = exp003.lanes_for_profile("component_precision_action_validity")

    assert tuple(lane.component_precision_arm for lane in lanes) == (
        "a0_v37",
        "a1_precision_component_once",
    )
    assert tuple(lane.lane_id for lane in lanes) == (
        "component_precision_a0_v37",
        "component_precision_a1_precision_component_once",
    )
    assert {lane.runner_action_name for lane in lanes} == {
        "arac_evidence_action_controller_v37"
    }
    assert all(lane.relation_dispatch_enabled for lane in lanes)


def test_backend_emits_component_precision_flag_only_when_enabled() -> None:
    enabled = build_hcc_aob_smoke_command(
        _request(component_precision_arm="a1_precision_component_once")
    )
    disabled = build_hcc_aob_smoke_command(_request())

    index = enabled.argv.index("--component-precision-arm")
    assert enabled.argv[index + 1] == "a1_precision_component_once"
    assert "--component-precision-arm" not in disabled.argv
    assert "--precision-causal-arm" not in enabled.argv
    assert "--precision-response-arm" not in enabled.argv


def test_component_precision_backend_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported component precision arm"):
        build_hcc_aob_smoke_command(_request(component_precision_arm="unknown"))
    with pytest.raises(ValueError, match="requires the frozen v37 action"):
        build_hcc_aob_smoke_command(
            _request(
                arac_action="arac_evidence_action_controller_v38",
                component_precision_arm="a0_v37",
            )
        )


@pytest.mark.parametrize(
    "arms",
    [
        {
            "precision_causal_arm": "action",
            "component_precision_arm": "a0_v37",
        },
        {
            "precision_response_arm": "a1_probe_only",
            "component_precision_arm": "a0_v37",
        },
        {
            "precision_causal_arm": "baseline",
            "precision_response_arm": "a0_v37",
        },
    ],
)
def test_precision_experiment_arms_are_pairwise_exclusive(
    arms: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_hcc_aob_smoke_command(_request(**arms))


def test_component_precision_cli_parses_profile_and_arm_filter() -> None:
    parsed = exp003.parse_args(
        [
            "--lane-profile",
            "component_precision_action_validity",
            "--component-precision-arms",
            "a1_precision_component_once",
            "--component-precision-stage",
            "smoke",
        ]
    )

    assert parsed.lane_profile == "component_precision_action_validity"
    assert parsed.component_precision_arms == ["a1_precision_component_once"]
    assert parsed.component_precision_stage == "smoke"


def test_component_precision_lane_is_forwarded_to_execution_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[HccAobExecutionRequest] = []
    lane = exp003.lanes_for_profile("component_precision_action_validity")[1]

    def fake_execution(request: HccAobExecutionRequest) -> HccAobExecutionResult:
        observed.append(request)
        return HccAobExecutionResult(
            problem_id=request.problem_id,
            seed=request.seed,
            max_fes=request.max_fes,
            final_error=1.0,
            fe_used=request.max_fes,
            time_seconds=0.0,
            output_root=request.output_dir,
            fresh_optimizer_execution=True,
            status="completed",
            result_source="fake",
            optimizer_final_fe_used=request.max_fes,
        )

    monkeypatch.setattr(exp003, "_existing_completed_result", lambda _request: None)
    exp003._records(
        output_dir=tmp_path,
        execution_runner=fake_execution,
        hcc_root=exp003.HCC_VENDOR_ROOT,
        aob_data_root=exp003.DEFAULT_AOB_DATA_ROOT,
        python_executable="python",
        seeds=(1,),
        problem_ids=("E2",),
        max_fes=5_000,
        lanes=(lane,),
    )

    assert len(observed) == 1
    assert observed[0].component_precision_arm == "a1_precision_component_once"
    assert observed[0].precision_causal_arm == "off"
    assert observed[0].precision_response_arm == "off"


def test_component_precision_arm_filter_is_profile_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def stop_after_filter(lanes: tuple[exp003.LaneConfig, ...], _problems: object) -> None:
        captured.extend(lanes)
        raise RuntimeError("stop after component arm filter")

    monkeypatch.setattr(exp003, "_require_hcc_action_preflight", stop_after_filter)
    monkeypatch.setattr(
        exp003, "_require_component_precision_source_binding", lambda: None
    )
    with pytest.raises(RuntimeError, match="stop after component arm filter"):
        exp003.run_hcc_runtime_consumer_smoke(
            lane_profile="component_precision_action_validity",
            component_precision_arms=("a1_precision_component_once",),
            component_precision_stage="smoke",
            seeds=(1,),
            problem_ids=("E2",),
        )

    assert [lane.component_precision_arm for lane in captured] == [
        "a1_precision_component_once"
    ]
    with pytest.raises(
        ValueError,
        match="component_precision_arms requires component_precision_action_validity",
    ):
        exp003.run_hcc_runtime_consumer_smoke(
            lane_profile="runtime_smoke",
            component_precision_arms=("a0_v37",),
            seeds=(1,),
            problem_ids=("E2",),
        )


def test_component_source_binding_requires_clean_entire_tracked_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    commands: list[tuple[str, ...]] = []

    def clean_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(tuple(command))
        return SimpleNamespace(
            stdout=f"{commit}\n" if command[:2] == ["git", "rev-parse"] else "",
            returncode=0,
        )

    monkeypatch.setattr(exp003, "_git_commit", lambda: commit)
    monkeypatch.setattr(exp003.subprocess, "run", clean_run)
    exp003._require_component_precision_source_binding()

    assert ("git", "diff", "--quiet") in commands
    assert ("git", "diff", "--cached", "--quiet") in commands
    assert all("--" not in command for command in commands if "diff" in command)

    def dirty_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            stdout=f"{commit}\n" if command[:2] == ["git", "rev-parse"] else "",
            returncode=1 if command == ["git", "diff", "--quiet"] else 0,
        )

    monkeypatch.setattr(exp003.subprocess, "run", dirty_run)
    with pytest.raises(RuntimeError, match="clean tracked tree"):
        exp003._require_component_precision_source_binding()


def test_component_precision_stage_and_formal_matrix_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="component_precision_stage must be explicit"):
        exp003.run_hcc_runtime_consumer_smoke(
            lane_profile="component_precision_action_validity",
            seeds=(65,),
            problem_ids=("E2",),
        )

    monkeypatch.setattr(
        exp003, "_require_component_precision_source_binding", lambda: None
    )
    with pytest.raises(ValueError, match="screen matrix mismatch"):
        exp003.run_hcc_runtime_consumer_smoke(
            lane_profile="component_precision_action_validity",
            component_precision_stage="screen",
            seeds=(65,),
            problem_ids=("E2",),
            jobs=24,
            max_fes=3_000_000,
        )


def test_component_confirm_preflight_rejects_tampered_screen_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(
        (exp003.ARAC_REPO_ROOT / exp003.COMPONENT_PRECISION_CONFIG_PATH).read_text(
            encoding="utf-8"
        )
    )
    screen_source = tmp_path / "screen_source"
    screen_source.mkdir()
    artifact_names = tuple(
        config["artifacts"][key]
        for key in (
            "branches",
            "component_outcomes",
            "survival",
            "pairs",
            "budget",
        )
    )
    for name in artifact_names:
        (screen_source / name).write_text(f"{name}\n", encoding="utf-8")
    gate_path = tmp_path / "screen_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "protocol_version": exp003.COMPONENT_PRECISION_PROTOCOL_VERSION,
                "stage": "screen",
                "status": "screen_pass",
                "source_root": str(screen_source),
                "source_git_commit": "b" * 40,
                "checks": {"all_hard_gates": True},
                "integrity": {"status": "pass"},
                "input_artifact_sha256": {
                    name: exp003._sha256_file(screen_source / name)
                    for name in artifact_names
                },
                "bootstrap": {"resamples": 2000},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exp003, "_require_component_precision_source_binding", lambda: None
    )
    monkeypatch.setattr(exp003, "_git_commit", lambda: "a" * 40)

    def run_confirm() -> None:
        exp003.run_hcc_runtime_consumer_smoke(
            lane_profile="component_precision_action_validity",
            component_precision_stage="confirm",
            component_precision_screen_gate=gate_path,
            seeds=tuple(config["confirm"]["seeds"]),
            problem_ids=tuple(config["confirm"]["cases"]),
            jobs=24,
            max_fes=3_000_000,
        )

    with pytest.raises(ValueError, match="screen gate is not an audited pass"):
        run_confirm()

    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload["source_git_commit"] = "a" * 40
    payload["checks"] = {"all_hard_gates": "false"}
    gate_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="screen gate is not an audited pass"):
        run_confirm()

    payload["checks"] = {"all_hard_gates": True}
    removed_name, removed_hash = payload["input_artifact_sha256"].popitem()
    payload["input_artifact_sha256"][f"forged_{removed_name}"] = removed_hash
    gate_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="screen gate is not an audited pass"):
        run_confirm()

    payload["input_artifact_sha256"] = {
        name: exp003._sha256_file(screen_source / name)
        for name in artifact_names
    }
    gate_path.write_text(json.dumps(payload), encoding="utf-8")
    (screen_source / artifact_names[0]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="screen artifacts changed"):
        run_confirm()


def test_component_precision_rejects_nonempty_output_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "old.csv").write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(
        exp003, "_require_component_precision_source_binding", lambda: None
    )
    monkeypatch.setattr(exp003, "_require_hcc_action_preflight", lambda *_: None)
    monkeypatch.setattr(
        exp003,
        "require_pinned_hcc_runtime_environment",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ValueError, match="must be absent or empty"):
        exp003.run_hcc_runtime_consumer_smoke(
            output_dir=output,
            lane_profile="component_precision_action_validity",
            component_precision_stage="smoke",
            seeds=(1,),
            problem_ids=("E2",),
        )


def _write_artifact(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _component_record(
    tmp_path: Path,
    arm: str,
    *,
    applicable: bool = True,
    branch_overrides: dict[str, str] | None = None,
    budget_overrides: dict[int, dict[str, str]] | None = None,
    survival_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    lane = next(
        lane
        for lane in exp003.lanes_for_profile("component_precision_action_validity")
        if lane.component_precision_arm == arm
    )
    output_root = tmp_path / arm
    actual_values = (32, 32) if arm == "a0_v37" else (16, 32)
    auxiliary_values = (4, 4)
    interval_values = tuple(
        actual + auxiliary
        for actual, auxiliary in zip(actual_values, auxiliary_values, strict=True)
    )
    endpoint_error = 90.0 if arm == "a0_v37" else 80.0
    terminal_error = 80.0 if arm == "a0_v37" else 70.0
    sigma = 0.2 if arm == "a0_v37" else 0.1
    branch = {
        "protocol_version": exp003.COMPONENT_PRECISION_PROTOCOL_VERSION,
        "schema_version": "component-atomic-precision-v1",
        "fresh_optimizer_execution": "1",
        "problem_id": "E2",
        "seed": "65",
        "component_precision_arm": arm,
        "decision_id": "decision-1" if applicable else "",
        "decision_status": "applicable" if applicable else "not_applicable",
        "not_applicable_reason": "" if applicable else "no_safe_component_horizon_opportunity",
        "decision_fe": "1000" if applicable else "",
        "outer_iter": "1" if applicable else "",
        "component_id": "component-1" if applicable else "",
        "component_group_indices": "2;3" if applicable else "",
        "component_group_count": "2" if applicable else "0",
        "component_shared_var_count": "4" if applicable else "0",
        "prefix_record_sha256": "a" * 64 if applicable else "",
        "checkpoint_candidate_sha256": "b" * 64 if applicable else "",
        "crn_descriptor_sha256": "c" * 64 if applicable else "",
        "component_plan_sha256": "d" * 64 if applicable else "",
        "normal_sigma": "2.00000000000000011e-01" if applicable else "",
        "precision_sigma": "1.00000000000000006e-01" if applicable else "",
        "action_applied": str(int(applicable and arm != "a0_v37")),
        "component_plan_frozen": str(int(applicable)),
        "mid_horizon_redispatch_count": "0",
        "atomic_closed": str(int(applicable)),
        "unique_h_endpoint": str(int(applicable)),
        "component_horizon_requested_fe": "64" if applicable else "",
        "component_horizon_actual_fe": str(sum(actual_values)) if applicable else "",
        "component_horizon_interval_fe": (
            str(sum(interval_values)) if applicable else ""
        ),
        "component_end_fe": str(1000 + sum(interval_values)) if applicable else "",
        "h_endpoint_count": "1" if applicable else "0",
        "plan_integrity_valid": "1" if applicable else "0",
        "delayed_review_fe": "2000" if applicable else "",
        "delayed_review_outer_iter": "2" if applicable else "",
        "delayed_review_group_index": "2" if applicable else "",
        "delayed_status": "resolved_next_component_entry" if applicable else "not_applicable",
        "terminal_target_fe": "2999984",
        "terminal_completion_tolerance_fe": "16",
        "terminal_observed_fe": "2999984",
        "terminal_error": f"{terminal_error:.17e}" if applicable else "8.00000000000000000e+01",
        "terminal_record_sha256": (
            "e" * 64
            if not applicable or arm == "a0_v37"
            else "f" * 64
        ),
        "terminal_status": "complete",
    }
    if branch_overrides:
        branch.update(branch_overrides)
    endpoint_rows = []
    survival_rows = []
    budget_rows = []
    if applicable:
        group_start_fe = 1000
        for position, (group_index, actual, interval, auxiliary) in enumerate(
            zip(
                (2, 3),
                actual_values,
                interval_values,
                auxiliary_values,
                strict=True,
            )
        ):
            row = {
                "decision_id": "decision-1",
                "component_precision_arm": arm,
                "group_position": str(position),
                "group_index": str(group_index),
                "population_size": "16",
                "requested_fe": "32",
                "actual_fe": str(actual),
                "interval_actual_fe": str(interval),
                "auxiliary_actual_fe": str(auxiliary),
                "sigma": f"{sigma:.17e}",
                "group_start_fe": str(group_start_fe),
                "group_end_fe": str(group_start_fe + interval),
                "group_endpoint_error": f"{endpoint_error + 5 - position * 5:.17e}",
            }
            if budget_overrides and position in budget_overrides:
                row.update(budget_overrides[position])
            budget_rows.append(row)
            group_start_fe += interval
        requested_sum = sum(int(row["requested_fe"]) for row in budget_rows)
        actual_sum = sum(int(row["actual_fe"]) for row in budget_rows)
        branch["component_horizon_requested_fe"] = str(requested_sum)
        branch["component_horizon_actual_fe"] = str(actual_sum)
        endpoint_rows = [
            {
                "decision_id": "decision-1",
                "component_precision_arm": arm,
                "checkpoint_error": "1.00000000000000000e+02",
                "endpoint_error": f"{endpoint_error:.17e}",
                "component_log_gain": f"{math.log(100.0 / endpoint_error):.17e}",
                "material": "1",
                "component_start_fe": "1000",
                "component_end_fe": branch["component_end_fe"],
                "component_requested_fe": str(requested_sum),
                "component_actual_fe": str(actual_sum),
                "component_interval_fe": str(
                    sum(int(row["interval_actual_fe"]) for row in budget_rows)
                ),
                "group_endpoint_errors": ";".join(
                    row["group_endpoint_error"] for row in budget_rows
                ),
            }
        ]
        survival = {
            "decision_id": "decision-1",
            "component_precision_arm": arm,
            "shared_path_l1": "1.0",
            "shared_net_l1": "0.7" if arm == "a0_v37" else "0.8",
            "s_h": "0.7" if arm == "a0_v37" else "0.8",
            "delayed_drift_l1": "0.28" if arm == "a0_v37" else "0.2",
            "s_d": "0.6" if arm == "a0_v37" else "0.75",
            "strict_survival": "1",
            "delayed_status": "resolved_next_component_entry",
        }
        if survival_overrides:
            survival.update(survival_overrides)
        survival_rows = [survival]
    _write_artifact(
        output_root / "component_action_branch_manifest.csv",
        exp003.COMPONENT_BRANCH_RAW_FIELDS,
        [branch],
    )
    _write_artifact(
        output_root / "component_endpoint_outcomes.csv",
        exp003.COMPONENT_ENDPOINT_RAW_FIELDS,
        endpoint_rows,
    )
    _write_artifact(
        output_root / "component_shared_survival.csv",
        exp003.COMPONENT_SURVIVAL_RAW_FIELDS,
        survival_rows,
    )
    _write_artifact(
        output_root / "component_budget_ledger.csv",
        exp003.COMPONENT_BUDGET_RAW_FIELDS,
        budget_rows,
    )
    action_trace_path = output_root / "action_trace.csv"
    action_trace_path.write_text(
        "public_trace\n"
        + ((arm + "\n") if applicable else "no_action\n"),
        encoding="utf-8",
    )
    result = HccAobExecutionResult(
        problem_id="E2",
        seed=65,
        max_fes=3_000_000,
        final_error=terminal_error if applicable else 80.0,
        fe_used=2_999_984,
        time_seconds=0.0,
        output_root=output_root,
        fresh_optimizer_execution=True,
        status="completed",
        result_source="fake",
        optimizer_final_fe_used=2_999_984,
        action_trace_path=action_trace_path,
    )
    return {"lane": lane, "lane_id": lane.lane_id, "result": result}


def _aggregate_component_records(
    records: list[dict[str, object]],
    *,
    stage: str = "screen",
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[str],
]:
    aob_rows = [
        {
            "problem_id": record["result"].problem_id,
            "seed": str(record["result"].seed),
            "lane_id": record["lane_id"],
            "file": "aob.txt",
            "unchanged": "1",
        }
        for record in records
    ]
    return exp003._component_precision_raw_rows(
        records,
        stage=stage,
        aob_input_rows=aob_rows,
        anti_leakage_rows=[{"audit_status": "pass"}],
    )


def test_component_pair_aggregation_allows_natural_actual_fe_difference(
    tmp_path: Path,
) -> None:
    records = [
        _component_record(tmp_path, "a0_v37"),
        _component_record(tmp_path, "a1_precision_component_once"),
    ]

    branches, endpoints, survival, budget, pairs, failures = (
        _aggregate_component_records(records)
    )

    assert failures == []
    assert len(branches) == 2
    assert len(endpoints) == 1
    assert len(survival) == 1
    assert len(budget) == 2
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_integrity"] == "1"
    assert float(pair["tau_T"]) == pytest.approx(math.log(80.0 / 70.0))
    assert float(endpoints[0]["tau_H"]) == pytest.approx(math.log(90.0 / 80.0))
    assert float(survival[0]["delta_s_h"]) == pytest.approx(0.1)
    assert float(survival[0]["delta_s_d"]) == pytest.approx(0.15)
    assert float(survival[0]["a0_shared_net_l1"]) == pytest.approx(0.7)
    assert float(survival[0]["a1_shared_net_l1"]) == pytest.approx(0.8)
    assert budget[1]["actual_group_fes"] == "16;32"
    assert budget[1]["interval_group_fes"] == "20;36"
    assert budget[1]["auxiliary_group_fes"] == "4;4"


def test_component_root_schemas_match_the_frozen_auditor() -> None:
    assert tuple(exp003.COMPONENT_BRANCH_ROOT_FIELDS) == COMPONENT_AUDIT.BRANCH_COLUMNS
    assert tuple(exp003.COMPONENT_ENDPOINT_ROOT_FIELDS) == COMPONENT_AUDIT.COMPONENT_COLUMNS
    assert tuple(exp003.COMPONENT_SURVIVAL_ROOT_FIELDS) == COMPONENT_AUDIT.SURVIVAL_COLUMNS
    assert tuple(exp003.COMPONENT_ACTION_PAIR_FIELDS) == COMPONENT_AUDIT.PAIR_COLUMNS
    assert tuple(exp003.COMPONENT_BUDGET_ROOT_FIELDS) == COMPONENT_AUDIT.BUDGET_COLUMNS


def test_component_root_artifacts_are_consumed_by_the_frozen_auditor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _component_record(tmp_path / "lanes", arm)
        for arm in exp003.COMPONENT_PRECISION_ARMS
    ]
    branches, endpoints, survival, budget, pairs, failures = (
        _aggregate_component_records(records)
    )
    assert failures == []
    for name, fields, rows in (
        (
            "component_action_branch_manifest.csv",
            exp003.COMPONENT_BRANCH_ROOT_FIELDS,
            branches,
        ),
        (
            "component_endpoint_outcomes.csv",
            exp003.COMPONENT_ENDPOINT_ROOT_FIELDS,
            endpoints,
        ),
        (
            "component_shared_survival.csv",
            exp003.COMPONENT_SURVIVAL_ROOT_FIELDS,
            survival,
        ),
        (
            "component_action_pairs.csv",
            exp003.COMPONENT_ACTION_PAIR_FIELDS,
            pairs,
        ),
        (
            "component_budget_ledger.csv",
            exp003.COMPONENT_BUDGET_ROOT_FIELDS,
            budget,
        ),
    ):
        _write_artifact(tmp_path / name, fields, rows)

    config = json.loads(json.dumps(COMPONENT_AUDIT._load_config()))
    config["screen"].update(
        {
            "cases": ["E2"],
            "seeds": [65],
            "minimum_applicable": 1,
            "minimum_applicable_cases": 1,
            "minimum_positive_seed_means": 1,
            "minimum_material_pairs": 1,
        }
    )
    monkeypatch.setattr(COMPONENT_AUDIT, "_load_config", lambda: config)

    gate = COMPONENT_AUDIT.audit_component_atomic_precision(
        tmp_path,
        stage="screen",
        resamples=2_000,
    )

    assert gate["status"] == "screen_pass", gate
    assert gate["integrity"] == {"status": "pass", "blockers": []}


@pytest.mark.parametrize(
    ("arm", "branch_overrides", "budget_overrides", "survival_overrides", "failure"),
    [
        (
            "a1_precision_component_once",
            {"prefix_record_sha256": "different"},
            None,
            None,
            "prefix_match",
        ),
        (
            "a1_precision_component_once",
            None,
            {0: {"requested_fe": "48"}},
            None,
            "requested_budgets_match",
        ),
        (
            "a0_v37",
            {"action_applied": "1"},
            None,
            None,
            "a0_no_action",
        ),
        (
            "a1_precision_component_once",
            None,
            {0: {"sigma": "2.00000000000000011e-01"}},
            None,
            "a1_half_sigma_valid",
        ),
        (
            "a1_precision_component_once",
            None,
            {0: {"actual_fe": "15"}},
            None,
            "complete_populations",
        ),
        (
            "a1_precision_component_once",
            {"component_group_count": "3"},
            None,
            None,
            "group_order_match",
        ),
        (
            "a1_precision_component_once",
            None,
            {0: {"decision_id": "other-decision"}},
            None,
            "group_order_match",
        ),
        (
            "a1_precision_component_once",
            {"delayed_status": "pending_next_component_entry"},
            None,
            {"delayed_status": "pending_next_component_entry", "s_d": ""},
            "delayed_closure",
        ),
        (
            "a1_precision_component_once",
            {"delayed_review_group_index": "3"},
            None,
            None,
            "delayed_closure",
        ),
        (
            "a1_precision_component_once",
            {"delayed_review_outer_iter": "4"},
            None,
            None,
            "delayed_closure",
        ),
    ],
)
def test_component_pair_integrity_fails_closed_per_contract(
    tmp_path: Path,
    arm: str,
    branch_overrides: dict[str, str] | None,
    budget_overrides: dict[int, dict[str, str]] | None,
    survival_overrides: dict[str, str] | None,
    failure: str,
) -> None:
    records = [
        _component_record(
            tmp_path,
            candidate_arm,
            branch_overrides=branch_overrides if candidate_arm == arm else None,
            budget_overrides=budget_overrides if candidate_arm == arm else None,
            survival_overrides=survival_overrides if candidate_arm == arm else None,
        )
        for candidate_arm in exp003.COMPONENT_PRECISION_ARMS
    ]

    *_, pairs, failures = _aggregate_component_records(records)

    assert pairs[0]["pair_integrity"] == "0"
    assert any(item.endswith(f":{failure}") for item in failures)


def test_component_pair_not_applicable_requires_no_op_parity(tmp_path: Path) -> None:
    records = [
        _component_record(tmp_path, arm, applicable=False)
        for arm in exp003.COMPONENT_PRECISION_ARMS
    ]

    branches, endpoints, survival, budget, pairs, failures = (
        _aggregate_component_records(records)
    )

    assert failures == []
    assert len(branches) == 2
    assert len(endpoints) == 1
    assert len(survival) == 1
    assert len(budget) == 2
    assert pairs[0]["pair_integrity"] == "1"
    assert pairs[0]["applicable"] == "0"
    assert float(endpoints[0]["tau_H"]) == 0.0
    assert float(pairs[0]["tau_T"]) == 0.0
    assert all(row["component_horizon_actual_fe"] == "0" for row in budget)

    mismatched = [
        _component_record(tmp_path / "mismatch", "a0_v37", applicable=False),
        _component_record(
            tmp_path / "mismatch",
            "a1_precision_component_once",
            applicable=False,
            branch_overrides={
                "terminal_error": "9.00000000000000000e+01",
                "terminal_record_sha256": "different-terminal",
            },
        ),
    ]
    *_, mismatched_pairs, mismatched_failures = (
        _aggregate_component_records(mismatched)
    )
    assert mismatched_pairs[0]["pair_integrity"] == "0"
    assert any(item.endswith(":no_op_parity") for item in mismatched_failures)


def test_component_manifest_is_response_independent_and_does_not_run_gate(
    tmp_path: Path,
) -> None:
    records = [
        _component_record(tmp_path, arm)
        for arm in exp003.COMPONENT_PRECISION_ARMS
    ]
    branches, endpoints, survival, budget, pairs, failures = (
        _aggregate_component_records(records)
    )
    artifacts = (
        (
            "component_action_branch_manifest.csv",
            exp003.COMPONENT_BRANCH_ROOT_FIELDS,
            branches,
        ),
        (
            "component_endpoint_outcomes.csv",
            exp003.COMPONENT_ENDPOINT_ROOT_FIELDS,
            endpoints,
        ),
        (
            "component_shared_survival.csv",
            exp003.COMPONENT_SURVIVAL_ROOT_FIELDS,
            survival,
        ),
        (
            "component_action_pairs.csv",
            exp003.COMPONENT_ACTION_PAIR_FIELDS,
            pairs,
        ),
        (
            "component_budget_ledger.csv",
            exp003.COMPONENT_BUDGET_ROOT_FIELDS,
            budget,
        ),
    )
    for name, fields, rows in artifacts:
        _write_artifact(tmp_path / name, fields, rows)

    manifest = exp003._component_precision_manifest(
        output_root=tmp_path,
        stage="screen",
        lanes=exp003.lanes_for_profile("component_precision_action_validity"),
        branch_rows=branches,
        pair_rows=pairs,
        integrity_failures=failures,
    )

    assert manifest["status"] == "pass"
    assert manifest["precision_response_dependency"] is False
    assert manifest["precision_response_artifacts_consumed"] == []
    assert manifest["gate_evaluated"] is False
    assert manifest["portfolio_evaluated"] is False
    assert "component_action_gate.json" in manifest["forbidden_outputs"]
    assert set(manifest["input_artifact_sha256"]) == {
        name for name, _, _ in artifacts
    }
    assert all(
        len(value) == 64
        for value in manifest["input_artifact_sha256"].values()
    )
