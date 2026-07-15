from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from arac.actions import ActionFamily
from arac.backends.hcc import (
    HccAobExecutionRequest,
    HccAobExecutionResult,
    build_hcc_aob_smoke_command,
)
from arac.policy.causal_risk_scheduler import (
    FEATURE_SCHEMA_SHA256,
    UTILITY_FEATURE_NAMES,
)
from arac.policy.component_delayed_credit import ComponentDelayedCreditTrace
from experiments.pilots.exp_003_hcc_runtime_consumer_smoke.run import (
    LaneConfig,
    _precision_causal_raw_rows,
    lanes_for_profile,
    precision_causal_logged_arm,
    precision_causal_pair_id,
)
import experiments.pilots.exp_003_hcc_runtime_consumer_smoke.run as exp003


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "vendor" / "hcc"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    runner_path = repo_root / "scripts" / "hcc_smoke_runner.py"
    spec = importlib.util.spec_from_file_location(
        "hcc_smoke_runner_for_causal_logging_test",
        runner_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_precision_causal_profile_is_two_fresh_v37_arms() -> None:
    lanes = lanes_for_profile("precision_causal_logging")

    assert [lane.lane_id for lane in lanes] == [
        "precision_baseline",
        "precision_action",
    ]
    assert [lane.precision_causal_arm for lane in lanes] == [
        "baseline",
        "action",
    ]
    assert {lane.runner_action_name for lane in lanes} == {
        "arac_evidence_action_controller_v37"
    }
    assert all(lane.relation_dispatch_enabled for lane in lanes)


def test_precision_causal_backend_flag_is_independent_from_car() -> None:
    request = HccAobExecutionRequest(
        problem_id="E2",
        seed=7,
        max_fes=5_000,
        output_dir=Path("results/precision-causal-command"),
        arac_action="arac_evidence_action_controller_v37",
        enable_relation_dispatch=True,
        relation_policy_mode="controller_v31",
        precision_causal_arm="action",
    )

    command = build_hcc_aob_smoke_command(request)

    assert "--precision-causal-arm" in command.argv
    flag_index = command.argv.index("--precision-causal-arm")
    assert command.argv[flag_index + 1] == "action"
    assert "--car-actionability-arm" not in command.argv


def test_randomized_assignment_is_materialized_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "scheduled"
    observed_schedule = []

    def fake_execution(request: HccAobExecutionRequest) -> HccAobExecutionResult:
        schedule_path = output / "causal_randomization_schedule.json"
        assert schedule_path.exists()
        observed_schedule.append(schedule_path.read_text(encoding="utf-8"))
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

    monkeypatch.setattr(exp003, "_existing_completed_result", lambda request: None)
    monkeypatch.setattr(exp003, "_prepare_precision_causal_provenance", lambda request: None)
    monkeypatch.setattr(
        exp003,
        "_complete_precision_causal_provenance",
        lambda request, result: None,
    )

    exp003._records(
        output_dir=output,
        execution_runner=fake_execution,
        hcc_root=exp003.HCC_VENDOR_ROOT,
        aob_data_root=exp003.DEFAULT_AOB_DATA_ROOT,
        python_executable=sys.executable,
        seeds=(1,),
        problem_ids=("E2",),
        max_fes=5_000,
        lanes=lanes_for_profile("precision_causal_logging"),
    )

    assert len(observed_schedule) == 2
    assert observed_schedule[0] == observed_schedule[1]
    assert precision_causal_pair_id("E2", 1) in observed_schedule[0]
    assert precision_causal_logged_arm("E2", 1) in observed_schedule[0]
    schedule = json.loads(observed_schedule[0])
    assert schedule["status"] == "scheduled_before_subprocess"
    assert schedule["preregistration"] == {
        "path": exp003.PRECISION_CAUSAL_PREREGISTRATION_PATH,
        "sha256": exp003.PRECISION_CAUSAL_PREREGISTRATION_SHA256,
        "commit": exp003.PRECISION_CAUSAL_PREREGISTRATION_COMMIT,
    }
    assert (
        exp003._sha256_file(
            exp003.ARAC_REPO_ROOT / exp003.PRECISION_CAUSAL_PREREGISTRATION_PATH
        )
        == exp003.PRECISION_CAUSAL_PREREGISTRATION_SHA256
    )


def test_precision_causal_runner_cli_requires_v37() -> None:
    runner = _load_runner_module()
    common = [
        "--functions",
        "elliptic",
        "--ids",
        "2",
        "--output-root",
        "out",
        "--seed",
        "1",
        "--max-fes",
        "5000",
        "--precision-causal-arm",
        "baseline",
    ]

    parsed = runner.parse_args(
        [*common, "--arac-action", "arac_evidence_action_controller_v37"]
    )
    assert parsed.precision_causal_arm == "baseline"

    with pytest.raises(SystemExit):
        runner.parse_args([*common, "--arac-action", "conservative_no_action"])


def test_pre_action_snapshot_uses_only_completed_history() -> None:
    runner = _load_runner_module()
    groups = [[0, 1], [1, 2]]
    tracker = ComponentDelayedCreditTrace(groups, lower=-5.0, upper=5.0)
    component = tracker.topology.for_group(0)
    cap = SimpleNamespace(
        reachable=True,
        cap_fe=100,
        reason="scheduler_revisit_cap_available",
    )
    cma_history = [
        runner.CMATraceOnlyDiagnostic(end, ratio, success, diversity)
        for end, ratio, success, diversity in (
            (40, 0.9, 0.5, 0.2),
            (55, 0.8, 0.4, 0.15),
            (70, 0.7, 0.3, 0.1),
        )
    ]

    snapshot = runner.build_precision_causal_snapshot(
        checkpoint_fitness=10.0,
        decision_fe=80,
        max_fes=1_000,
        phase_i_tail_progress_rate=0.01,
        phase_i_source_end_fe=20,
        cc_progress_history=[0.4, 0.3, 0.2, 0.1],
        cc_source_end_fes=[30, 40, 50, 60],
        component_disagreement_history=[(50, 0.2), (65, 0.4)],
        cma_history=cma_history,
        grouping_result=groups,
        dimension=3,
        component=component,
        scheduler_revisit_cap=cap,
        controller_state_sha256="c" * 64,
        prefix_record_sha256="p" * 64,
        checkpoint_candidate_sha256="i" * 64,
        random_descriptor_sha256="r" * 64,
        normal_sigma=0.25,
        candidate_sigma=0.125,
    )

    assert snapshot.decision_status == "applicable"
    assert snapshot.source_end_fe == 70
    assert snapshot.source_end_fe < snapshot.decision_fe
    assert snapshot.state is not None
    assert snapshot.state.candidate_dose_ratio == pytest.approx(0.5)
    assert snapshot.state.proposal_disagreement_mean_2 == pytest.approx(0.3)
    assert snapshot.state.feature_sha256

    missing = runner.build_precision_causal_snapshot(
        checkpoint_fitness=10.0,
        decision_fe=80,
        max_fes=1_000,
        phase_i_tail_progress_rate=0.01,
        phase_i_source_end_fe=20,
        cc_progress_history=[0.1],
        cc_source_end_fes=[30],
        component_disagreement_history=[],
        cma_history=[],
        grouping_result=groups,
        dimension=3,
        component=component,
        scheduler_revisit_cap=cap,
        controller_state_sha256="c" * 64,
        prefix_record_sha256="p" * 64,
        checkpoint_candidate_sha256="i" * 64,
        random_descriptor_sha256="r" * 64,
        normal_sigma=0.25,
        candidate_sigma=0.125,
    )
    assert missing.state is None
    assert missing.decision_status == "not_applicable"
    assert "missing_pre_action_history" in missing.not_applicable_reason

    selected, action_active = runner.select_first_complete_precision_snapshot(
        None,
        missing,
        audit_arm="action",
    )
    assert selected is None
    assert action_active is False

    selected, action_active = runner.select_first_complete_precision_snapshot(
        selected,
        snapshot,
        audit_arm="action",
    )
    assert selected is snapshot
    assert action_active is True

    later, later_action_active = runner.select_first_complete_precision_snapshot(
        selected,
        snapshot,
        audit_arm="action",
    )
    assert later is snapshot
    assert later_action_active is False


def test_cma_diagnostic_is_computed_from_existing_batches() -> None:
    runner = _load_runner_module()

    diagnostic = runner.summarize_cma_trace_only_diagnostic(
        objective_values=[10.0, 9.0, 11.0, 12.0, 9.5, 8.0, 9.0, 10.0],
        batch_diversities=[0.2, 0.1],
        population_size=4,
        pre_block_fitness=10.0,
        initial_sigma=0.25,
        terminal_sigma=0.125,
        source_end_fe=100,
    )

    assert diagnostic is not None
    assert diagnostic.terminal_sigma_ratio == pytest.approx(0.5)
    assert diagnostic.success_generation_ratio == pytest.approx(1.0)
    assert diagnostic.offspring_diversity_ratio == pytest.approx(0.15)
    assert diagnostic.source_end_fe == 100


def test_trace_only_baseline_is_bit_equivalent_to_v37_at_5k(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    off_dir = tmp_path / "off"
    baseline_dir = tmp_path / "baseline"
    off_dir.mkdir()
    baseline_dir.mkdir()
    common = {
        "max_fes": 5_000,
        "seed": 1,
        "arac_action": "arac_evidence_action_controller_v37",
        "enable_relation_dispatch": True,
        "relation_policy_mode": "controller_v31",
        "budget_accounting": "strict",
        "skip_plots": True,
        "aob_data_root": runner.DATA_DIR,
    }
    before = runner.snapshot_aob_inputs(2, runner.DATA_DIR)

    off_record, _, off_trace = runner.run_problem(
        "elliptic",
        2,
        off_dir,
        runner.SmokeConfig(**common),
    )
    baseline_record, _, baseline_trace = runner.run_problem(
        "elliptic",
        2,
        baseline_dir,
        runner.SmokeConfig(**common, precision_causal_arm="baseline"),
    )
    after = runner.snapshot_aob_inputs(2, runner.DATA_DIR)

    assert baseline_record == off_record
    assert len(baseline_record) == len(off_record)
    assert len(baseline_trace) == len(off_trace)
    for off_row, baseline_row in zip(off_trace, baseline_trace, strict=True):
        assert {
            key: value
            for key, value in baseline_row.items()
            if key not in runner.COMPONENT_CREDIT_TRACE_FIELDS
        } == {
            key: value
            for key, value in off_row.items()
            if key not in runner.COMPONENT_CREDIT_TRACE_FIELDS
        }
    assert after == before


def _write_trace(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _paired_records(tmp_path: Path, *, mismatched_end: bool = False):
    pair_rows = {}
    feature_values = {name: "5.00000000000000000e-01" for name in UTILITY_FEATURE_NAMES}
    for arm, terminal_error in (("baseline", 10.0), ("action", 5.0)):
        intervention_end = 201 if arm == "action" and mismatched_end else 200
        row = {
            "protocol_version": "precision-causal-logging-v1",
            "fresh_optimizer_execution": "1",
            "problem_id": "E2",
            "seed": "1",
            "audit_arm": arm,
            "decision_id": "precision_same",
            "decision_status": "applicable",
            "not_applicable_reason": "",
            **feature_values,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "feature_sha256": "f" * 64,
            "decision_fe": "100",
            "checkpoint_fitness": "20",
            "remaining_fe": "4900",
            "component_id": "audit-only-component",
            "component_group_count": "2",
            "component_shared_var_count": "1",
            "component_unlocked": "1",
            "scheduler_revisit_reachable": "1",
            "scheduler_revisit_cap_fe": "100",
            "scheduler_revisit_reason": "scheduler_revisit_cap_available",
            "source_phase_i_end_fe": "10",
            "source_cc_history_end_fe": "90",
            "source_disagreement_history_end_fe": "90",
            "source_cma_history_end_fe": "90",
            "source_end_fe": "90",
            "prefix_record_sha256": "p" * 64,
            "checkpoint_candidate_sha256": "i" * 64,
            "controller_state_sha256": "c" * 64,
            "random_descriptor_sha256": "r" * 64,
            "action_applied": "1" if arm == "action" else "0",
            "normal_sigma": "0.25",
            "candidate_sigma": "0.125",
            "applied_sigma": "0.125" if arm == "action" else "0.25",
            "requested_fe": "100",
            "actual_fe": "100",
            "intervention_end_fe": str(intervention_end),
            "configured_max_fes": "5000",
            "terminal_target_fe": "4990",
            "terminal_observed_fe": "4990",
            "terminal_error": str(terminal_error),
            "terminal_status": "complete",
            "terminal_record_sha256": arm[0] * 64,
        }
        lane_id = f"precision_{arm}"
        output_root = tmp_path / lane_id
        _write_trace(output_root / "E2_precision_causal_trace.csv", row)
        lane = LaneConfig(
            lane_id=lane_id,
            action_family=ActionFamily.TRAJECTORY,
            selected_action_name="arac_evidence_action_controller_v37",
            runner_action_name="arac_evidence_action_controller_v37",
            dispatch_scope="offline",
            precision_causal_arm=arm,
        )
        result = HccAobExecutionResult(
            problem_id="E2",
            seed=1,
            max_fes=5_000,
            final_error=terminal_error,
            fe_used=4_995,
            time_seconds=1.0,
            output_root=output_root,
            fresh_optimizer_execution=True,
            status="completed",
            result_source="fresh",
            optimizer_final_fe_used=4_995,
        )
        pair_rows[arm] = {"lane": lane, "lane_id": lane_id, "result": result}
    return [pair_rows["baseline"], pair_rows["action"]]


def test_raw_pair_rows_bind_features_crn_fe_and_terminal_outcomes(
    tmp_path: Path,
) -> None:
    features, audit, branches, outcomes, randomized, failures = (
        _precision_causal_raw_rows(_paired_records(tmp_path))
    )

    assert failures == []
    assert list(features[0]) == ["decision_id", *UTILITY_FEATURE_NAMES]
    assert not {"problem_id", "seed", "component_id"}.intersection(features[0])
    assert audit[0]["pair_id"] == precision_causal_pair_id("E2", 1)
    assert audit[0]["logged_arm"] == precision_causal_logged_arm("E2", 1)
    assert audit[0]["pair_integrity"] == 1
    assert audit[0]["random_descriptor_match"] == 1
    assert audit[0]["intervention_end_fe_match"] == 1
    assert {row["arm"] for row in branches} == {"baseline", "action"}
    assert {row["terminal_error"] for row in branches} == {"10.0", "5.0"}
    assert float(outcomes[0]["paired_tau"]) == pytest.approx(math.log(2.0))
    assert outcomes[0]["outcome_valid"] == 1
    assert randomized[0]["propensity"] == "0.5"


def test_pair_rows_fail_closed_on_intervention_fe_mismatch(tmp_path: Path) -> None:
    _, audit, _, outcomes, _, failures = _precision_causal_raw_rows(
        _paired_records(tmp_path, mismatched_end=True)
    )

    assert failures == [
        f"{precision_causal_pair_id('E2', 1)}:preaction_pair_mismatch",
        f"{precision_causal_pair_id('E2', 1)}:invalid_paired_outcome",
    ]
    assert audit[0]["intervention_end_fe_match"] == 0
    assert audit[0]["pair_integrity"] == 0
    assert outcomes[0]["outcome_valid"] == 0
