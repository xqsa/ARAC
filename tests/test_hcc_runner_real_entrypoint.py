from __future__ import annotations

import csv
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
