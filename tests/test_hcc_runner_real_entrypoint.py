from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from arac.actions.group_optimizer_type import (
    DIAGONAL_COVARIANCE_MODE,
    FULL_CMAES_MODE,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
)
from scripts import hcc_smoke_runner


@pytest.mark.parametrize(
    "group_optimizer_mode",
    [FULL_CMAES_MODE, DIAGONAL_COVARIANCE_MODE],
)
def test_real_e1_run_problem_entrypoint_completes(
    group_optimizer_mode: str,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / group_optimizer_mode
    output_path.mkdir()
    config = hcc_smoke_runner.SmokeConfig(
        max_fes=200,
        seed=117,
        run_id=f"entrypoint-{group_optimizer_mode}",
        verbose=0,
        arac_action=hcc_smoke_runner.NATIVE_EQ8_ACTION,
        enable_relation_dispatch=False,
        relation_policy_mode="controller_v31",
        budget_accounting="strict",
        skip_plots=True,
        aob_data_root=hcc_smoke_runner.DATA_DIR.resolve(),
        evidence_overlay_mode="off",
        runtime_probe_repair_mode="hard_repair",
        group_optimizer_mode=group_optimizer_mode,
    )

    fitness_record, elapsed, trace_rows = hcc_smoke_runner.run_problem(
        "elliptic",
        1,
        output_path,
        config,
    )

    assert fitness_record
    assert all(math.isfinite(float(value)) for value in fitness_record)
    assert elapsed >= 0.0
    assert isinstance(trace_rows, list)


def test_terminal_comparison_ignores_branch_specific_tail() -> None:
    comparison_fe, comparison_error = hcc_smoke_runner.terminal_comparison_metrics(
        (10.0, 8.0, 7.0, 1.0),
        configured_max_fes=5,
        population_sizes=(2, 1),
    )

    assert comparison_fe == 3
    assert comparison_error == 7.0


def test_real_e1_action_ceiling_skips_empty_overlap_relations(
    tmp_path: Path,
) -> None:
    config = hcc_smoke_runner.SmokeConfig(
        max_fes=10_000,
        seed=117,
        run_id="entrypoint-e1-action-ceiling",
        verbose=0,
        arac_action=hcc_smoke_runner.EVIDENCE_ACTION_CONTROLLER_V37,
        enable_relation_dispatch=True,
        relation_policy_mode=hcc_smoke_runner.ACTION_CEILING_POLICY,
        budget_accounting="strict",
        skip_plots=True,
        aob_data_root=hcc_smoke_runner.DATA_DIR.resolve(),
        evidence_overlay_mode="paired_owner",
        runtime_probe_repair_mode="hard_repair",
        action_ceiling_capture=True,
        action_ceiling_cohort="real_aob",
        group_optimizer_mode=FULL_CMAES_MODE,
    )

    fitness_record, _, trace_rows = hcc_smoke_runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        config,
    )

    assert fitness_record
    assert trace_rows == []
    context_lines = (tmp_path / "E1_action_ceiling_contexts.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    arm_lines = (tmp_path / "E1_action_ceiling_arm_results.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert context_lines == [",".join(ACTION_CEILING_CONTEXT_FIELDS)]
    assert arm_lines == [",".join(ACTION_CEILING_ARM_RESULT_FIELDS)]
    with (tmp_path / "E1_budget_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        budget = next(csv.DictReader(handle))
    assert int(budget["global_phase_fe"]) == 0
    assert int(budget["cc_phase_fe"]) > 0
    assert int(budget["same_budget_violation"]) == 0
    assert int(budget["fitness_record_fe"]) <= config.max_fes


def test_real_r4_gcb_resumes_native_hcc_to_exact_fe(
    tmp_path: Path,
) -> None:
    config = hcc_smoke_runner.SmokeConfig(
        max_fes=25_000,
        seed=117,
        run_id="entrypoint-r4-persistent-phase2",
        verbose=0,
        arac_action=hcc_smoke_runner.EVIDENCE_ACTION_CONTROLLER_V37,
        enable_relation_dispatch=True,
        relation_policy_mode=hcc_smoke_runner.PERSISTENT_PHASE2_POLICY,
        budget_accounting="strict",
        skip_plots=True,
        aob_data_root=hcc_smoke_runner.DATA_DIR.resolve(),
        evidence_overlay_mode="paired_owner",
        runtime_probe_repair_mode="hard_repair",
        group_optimizer_mode=FULL_CMAES_MODE,
        persistent_phase2_action=hcc_smoke_runner.GCB_ACTION,
    )

    fitness_record, _, _ = hcc_smoke_runner.run_problem(
        "rastrigin",
        4,
        tmp_path,
        config,
    )

    action = json.loads(
        (tmp_path / "gcb_action.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tmp_path / "R4_evidence_overlay_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(fitness_record) == config.max_fes
    assert manifest["complete_sweep_count"] >= (
        hcc_smoke_runner.EVIDENCE_OVERLAY_REQUIRED_SWEEPS + 3
    )
    assert manifest["delayed_outcomes_required"] == 0
    assert manifest["delayed_label_expected"] == 0
    assert manifest["observer_integrity"] == 1
    assert action["schema_version"] == "gcb-relation-action-v1"
    assert action["trigger_scope"] == "relation_dispatch"
    assert action["action"]["trigger_scope"] == "relation_dispatch"
    assert action["execution_mode"] == "one_native_sweep_burst_then_native"
    assert action["selection_fe"] == manifest["probe_end_fe"]
    assert action["checkpoint_fe"] > action["selection_fe"]
    assert action["action_start_fe"] == action["checkpoint_fe"] + 1
    assert action["action_actual_fes"] == action["action_budget_fes"]
    assert action["action_budget_fes"] == action["budget_source_actual_fes"]
    assert action["action_completed_fe"] == (
        action["checkpoint_fe"] + action["action_budget_fes"]
    )
    assert action["action_completed_fe"] < config.max_fes
    assert action["native_resumed"] is True
    assert action["native_resume_start_fe"] == action["action_completed_fe"] + 1
    assert action["post_action_native_fes"] == (
        config.max_fes - action["action_completed_fe"]
    )
    assert action["native_resume_sweeps_planned"] == 3
    assert action["native_resume_sweeps_completed"] == 3
    assert action["lifecycle"]["completed_fe"] == action["action_completed_fe"]
    assert action["action"]["target_sweep"] == action["action"]["issued_sweep"] + 1
    assert action["budget_source_sweep"] == action["action"]["issued_sweep"]
    assert action["start_sweep"] == action["action"]["target_sweep"]


def test_real_r1_gcb_dispatches_at_phase_boundary(
    tmp_path: Path,
) -> None:
    config = hcc_smoke_runner.SmokeConfig(
        max_fes=25_000,
        seed=117,
        run_id="entrypoint-r1-global-phase2",
        verbose=0,
        arac_action=hcc_smoke_runner.NATIVE_EQ8_ACTION,
        enable_relation_dispatch=False,
        relation_policy_mode=hcc_smoke_runner.GLOBAL_PHASE2_POLICY,
        budget_accounting="strict",
        skip_plots=True,
        aob_data_root=hcc_smoke_runner.DATA_DIR.resolve(),
        evidence_overlay_mode="off",
        runtime_probe_repair_mode="hard_repair",
        group_optimizer_mode=FULL_CMAES_MODE,
        persistent_phase2_action=hcc_smoke_runner.GCB_ACTION,
    )

    fitness_record, _, _ = hcc_smoke_runner.run_problem(
        "rastrigin",
        1,
        tmp_path,
        config,
    )

    action = json.loads(
        (tmp_path / "gcb_action.json").read_text(encoding="utf-8")
    )

    assert len(fitness_record) == config.max_fes
    assert action["schema_version"] == "gcb-phase-boundary-action-v1"
    assert action["trigger_scope"] == "phase_boundary"
    assert action["relation"] is None
    assert action["runtime_consumed"] is True
    assert action["action"]["trigger_scope"] == "phase_boundary"
    assert action["action_actual_fes"] == action["action_budget_fes"]
    assert action["native_resume_sweeps_completed"] == 3


def test_real_s5_persistent_budget_early_stopping_closes_exact_fe(
    tmp_path: Path,
) -> None:
    config = hcc_smoke_runner.SmokeConfig(
        max_fes=10_000,
        seed=117,
        run_id="entrypoint-s5-persistent-budget",
        verbose=0,
        arac_action=hcc_smoke_runner.EVIDENCE_ACTION_CONTROLLER_V37,
        enable_relation_dispatch=True,
        relation_policy_mode=hcc_smoke_runner.PERSISTENT_PHASE2_POLICY,
        budget_accounting="strict",
        skip_plots=True,
        aob_data_root=hcc_smoke_runner.DATA_DIR.resolve(),
        evidence_overlay_mode="paired_owner",
        runtime_probe_repair_mode="hard_repair",
        group_optimizer_mode=FULL_CMAES_MODE,
        persistent_phase2_action=(
            hcc_smoke_runner.PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION
        ),
        early_stopping_evaluations=1,
    )

    fitness_record, _, _ = hcc_smoke_runner.run_problem(
        "schwefel",
        5,
        tmp_path,
        config,
    )

    artifact = json.loads(
        (tmp_path / "persistent_budget_action.json").read_text(encoding="utf-8")
    )
    lifecycle = artifact["lifecycle"]
    applications = lifecycle["details"]["applications"]

    assert len(fitness_record) == config.max_fes
    assert artifact["terminal_fe"] == config.max_fes
    assert artifact["action"]["end_absolute_fe"] == config.max_fes
    assert artifact["status"] == "completed"
    assert lifecycle["status"] == "completed"
    assert lifecycle["completed_fe"] == config.max_fes
    assert lifecycle["details"]["completed_fe"] == config.max_fes
    assert artifact["application_count"] == len(applications)
    assert len(applications) > 1

    application_interval_fes = [
        sum(application["group_interval_fes"]) for application in applications
    ]
    assert applications[0]["application_fe"] == artifact["checkpoint_fe"] + 1
    assert all(
        current["application_fe"]
        == previous["application_fe"] + previous_interval_fes
        for previous, previous_interval_fes, current in zip(
            applications,
            application_interval_fes,
            applications[1:],
            strict=False,
        )
    )
    assert any(
        actual_fes < applied_budget
        for application in applications
        for actual_fes, applied_budget in zip(
            application["actual_optimizer_fes"],
            application["applied_group_budgets"],
            strict=True,
        )
    )
    assert artifact["action_actual_fes"] == sum(application_interval_fes)
    assert artifact["action_actual_fes"] == (
        artifact["action"]["end_absolute_fe"] - artifact["checkpoint_fe"]
    )
