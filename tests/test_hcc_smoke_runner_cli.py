from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_runner_module():
    hcc_src = Path(__file__).resolve().parents[1] / "HCC_SRC"
    sys.path.insert(0, str(hcc_src))
    runner_path = hcc_src / "arac_hcc_smoke_runner.py"
    spec = importlib.util.spec_from_file_location("arac_hcc_smoke_runner_for_test", runner_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hcc_smoke_runner_parses_arac_action_argument() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "2",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--arac-action",
            "repair_shared_variable_binding",
        ]
    )

    assert args.arac_action == "repair_shared_variable_binding"
    assert args.enable_relation_dispatch is False


def test_hcc_smoke_runner_parses_relation_dispatch_flag() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "2",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--enable-relation-dispatch",
        ]
    )

    assert args.enable_relation_dispatch is True


def test_repair_shared_variable_binding_selects_owner_by_delta() -> None:
    runner = _load_runner_module()
    previous_values = np.array([1.0, 2.0])
    current_values = np.array([10.0, 20.0])

    repaired = runner.apply_arac_overlap_action(
        action_name="repair_shared_variable_binding",
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=3.0,
    )

    np.testing.assert_allclose(repaired, current_values)


def test_repair_shared_variable_binding_keeps_previous_when_previous_delta_wins() -> None:
    runner = _load_runner_module()
    previous_values = np.array([1.0, 2.0])
    current_values = np.array([10.0, 20.0])

    repaired = runner.apply_arac_overlap_action(
        action_name="repair_shared_variable_binding",
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=5.0,
        current_delta=3.0,
    )

    np.testing.assert_allclose(repaired, previous_values)


def test_conservative_no_action_uses_native_overlap_blend() -> None:
    runner = _load_runner_module()
    previous_values = np.array([1.0, 2.0])
    current_values = np.array([10.0, 20.0])

    native = runner.blend_overlap_values(
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=3.0,
    )
    action_result = runner.apply_arac_overlap_action(
        action_name="conservative_no_action",
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=3.0,
    )

    np.testing.assert_allclose(action_result, native)


def test_degree_of_overlap_accepts_scalar_overlap_groups() -> None:
    runner = _load_runner_module()

    degree = runner.calculate_degree_of_overlap([1, [2, 3]], problem_dimension=10)

    assert degree == 0.3


def test_build_action_trace_row_marks_runtime_consumed_repair() -> None:
    runner = _load_runner_module()

    row = runner.build_action_trace_row(
        problem_id="E2",
        seed=1,
        outer_iter=0,
        group_index=1,
        selected_action_name="repair_shared_variable_binding",
        overlap_size=3,
        previous_delta=1.0,
        current_delta=3.0,
    )

    assert row["problem_id"] == "E2"
    assert row["owner_selected"] == "current"
    assert row["semantic_surface"] == "shared_variable_owner_rebinding"
    assert row["optimizer_consumed"] == "1"


def test_build_overlap_relation_trace_exposes_adjacent_relations(tmp_path: Path) -> None:
    runner = _load_runner_module()

    relations = runner.build_overlap_relation_trace(
        problem_id="E2",
        outer_iter=1,
        grouping_result=[[0, 1, 2], [2, 3], [3, 4]],
        overlapping_elements=[[2], [3]],
        fitness_delta_list=[3.0, 1.0, 1.0],
        budget_remaining_ratio=0.4,
    )

    assert [relation.relation_id for relation in relations] == ["O1_0_1", "O1_1_2"]
    assert relations[0].shared_vars == (2,)
    assert relations[0].delta_signal == 2.0
    assert relations[0].budget_remaining_ratio == 0.4

    output_path = tmp_path / "overlap_relations.csv"
    runner._write_overlap_relation_trace(output_path, relations)

    written = output_path.read_text(encoding="utf-8")
    assert "relation_id,problem_id,outer_iter" in written
    assert "O1_0_1,E2,1,0,1,2,1.000000,2.000000,0.000000,0.400000" in written


def test_apply_action_to_relation_reuses_reassign_repair_logic() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=2.0,
        rank_signal=0.5,
        budget_remaining_ratio=0.8,
    )
    action = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="reassign_repair",
        action_family="reassign_repair",
        confidence=1.0,
        trigger_reason="test",
    )
    previous_values = np.array([1.0, 2.0])
    current_values = np.array([10.0, 20.0])

    repaired = runner.apply_action_to_relation(
        relation=relation,
        action=action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=3.0,
    )

    np.testing.assert_allclose(repaired, current_values)


def test_append_action_decision_log_preserves_existing_rows(tmp_path: Path) -> None:
    runner = _load_runner_module()
    first_relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=0.8,
    )
    second_relation = runner.OverlapRelation(
        relation_id="O1_1_2",
        problem_id="E2",
        outer_iter=1,
        group_left=1,
        group_right=2,
        shared_vars=(3, 4),
        overlap_strength=2.0,
        delta_signal=2.5,
        rank_signal=0.2,
        budget_remaining_ratio=0.7,
    )
    first_action = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="coordinate",
        action_family="coordinate",
        confidence=0.95,
        trigger_reason="stable",
    )
    second_action = runner.RelationActionDecision(
        relation_id="O1_1_2",
        action_name="isolate_conflicting_relation",
        action_family="isolate",
        confidence=1.0,
        trigger_reason="conflict",
    )
    output_path = tmp_path / "action_decision.csv"

    runner._append_action_decision_log(output_path, "run-001", [first_relation], [first_action])
    runner._append_action_decision_log(output_path, "run-001", [second_relation], [second_action])

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "run_id",
        "problem_id",
        "relation_id",
        "group_left",
        "group_right",
        "shared_vars_count",
        "overlap_strength",
        "delta_signal",
        "rank_signal",
        "action_name",
        "action_family",
        "confidence",
        "trigger_reason",
    ]
    assert len(rows) == 2
    assert rows[0]["relation_id"] == "O1_0_1"
    assert rows[0]["shared_vars_count"] == "1"
    assert rows[0]["action_name"] == "coordinate"
    assert rows[1]["relation_id"] == "O1_1_2"
    assert rows[1]["shared_vars_count"] == "2"
    assert rows[1]["action_name"] == "isolate_conflicting_relation"


@pytest.mark.integration
def test_conservative_fallback_matches_default_hcc_smoke_behavior(tmp_path: Path) -> None:
    if os.environ.get("ARAC_RUN_HCC_SMOKE") != "1":
        pytest.skip("set ARAC_RUN_HCC_SMOKE=1 to run the HCC subprocess smoke")

    from arac.backends.hcc import HccAobExecutionRequest, run_hcc_aob_smoke_execution

    python_executable = (
        r"C:\Users\83718\.cache\codex-runtimes\codex-primary-runtime\dependencies"
        r"\python\python.exe"
    )
    shared = {
        "problem_id": "E2",
        "seed": 1,
        "max_fes": 2_000,
        "hcc_root": Path("E:/HCC-main"),
        "python_executable": python_executable,
    }
    default_result = run_hcc_aob_smoke_execution(
        HccAobExecutionRequest(
            **shared,
            output_dir=(tmp_path / "default").resolve(),
            timestamp="fallback-equivalence-default",
        )
    )
    fallback_result = run_hcc_aob_smoke_execution(
        HccAobExecutionRequest(
            **shared,
            output_dir=(tmp_path / "fallback").resolve(),
            timestamp="fallback-equivalence-explicit",
            arac_action="conservative_no_action",
        )
    )

    assert default_result.status == "completed"
    assert fallback_result.status == "completed"
    assert fallback_result.final_error == pytest.approx(default_result.final_error)
    assert fallback_result.fe_used == default_result.fe_used
