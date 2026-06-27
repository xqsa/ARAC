from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
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


def test_hcc_smoke_runner_help_works_without_pythonpath() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "HCC_SRC" / "arac_hcc_smoke_runner.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(runner_path), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--enable-relation-dispatch" in completed.stdout


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


def test_hcc_smoke_runner_parses_relation_policy_options() -> None:
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
            "--relation-policy",
            "rule",
        ]
    )

    assert args.relation_policy == "rule"

    shuffled = runner.parse_args(
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
            "--relation-policy",
            "shuffled",
        ]
    )

    assert shuffled.relation_policy == "shuffled"


def test_shuffled_relation_policy_rotates_rule_action_deterministically() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O0_0_1",
        problem_id="E2",
        outer_iter=0,
        group_left=0,
        group_right=1,
        shared_vars=(7,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=1.0,
    )
    rule_action = runner.RelationActionDecision(
        relation_id=relation.relation_id,
        action_name="coordinate",
        action_family="coordinate",
        confidence=0.8,
        trigger_reason="rule",
    )

    shuffled = runner.select_relation_action_for_policy(
        relation=relation,
        action=rule_action,
        relation_policy_mode="shuffled",
    )

    assert shuffled.relation_action_name == "reassign_repair"
    assert shuffled.canonical_action_name == "repair_shared_variable_binding"
    assert shuffled.action_family == "reassign_repair"
    assert shuffled.trigger_reason.startswith("deterministic_shuffled_negative_control")


def test_shuffled_relation_policy_keeps_empty_overlap_fallback() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O0_0_1",
        problem_id="E1",
        outer_iter=0,
        group_left=0,
        group_right=1,
        shared_vars=(),
        overlap_strength=0.0,
        delta_signal=0.9,
        rank_signal=0.2,
        budget_remaining_ratio=0.8,
    )
    action = runner.RelationActionDecision(
        relation_id=relation.relation_id,
        action_name="fallback",
        action_family="fallback",
        confidence=0.0,
        trigger_reason="no_shared_overlap_support",
    )

    shuffled = runner.select_relation_action_for_policy(
        relation=relation,
        action=action,
        relation_policy_mode="shuffled",
    )

    assert shuffled.action_name == "fallback"
    assert shuffled.canonical_action_name == "conservative_no_action"


def test_hcc_smoke_runner_rejects_unsupported_action_file() -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner.parse_args(
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
                "--arac-action-file",
                "actions.csv",
            ]
        )


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


def test_allow_beneficial_coordination_uses_clipped_consensus_blend() -> None:
    runner = _load_runner_module()
    previous_values = np.array([0.0])
    current_values = np.array([100.0])

    coordinated = runner.apply_arac_overlap_action(
        action_name="allow_beneficial_coordination",
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=99.0,
    )

    np.testing.assert_allclose(coordinated, np.array([65.0]))


def test_isolate_conflicting_relation_keeps_stronger_side() -> None:
    runner = _load_runner_module()
    previous_values = np.array([1.0, 2.0])
    current_values = np.array([10.0, 20.0])

    isolated = runner.apply_arac_overlap_action(
        action_name="isolate_conflicting_relation",
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=5.0,
        current_delta=3.0,
    )

    np.testing.assert_allclose(isolated, previous_values)


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
    assert row["relation_id"] == ""
    assert row["canonical_action_name"] == "repair_shared_variable_binding"
    assert row["action_family"] == "reassign_repair"
    assert row["relation_policy_source"] == ""
    assert row["owner_selected"] == "current"
    assert row["semantic_surface"] == "shared_variable_owner_rebinding"
    assert row["state_mutated"] == "1"
    assert row["action_value_delta_norm"] == "0.000000e+00"
    assert row["downstream_consumed"] == "1"
    assert row["downstream_consumption_scope"] == "same_outer_iteration"
    assert row["optimizer_consumed"] == "1"


def test_build_action_trace_row_includes_relation_join_fields() -> None:
    runner = _load_runner_module()

    row = runner.build_action_trace_row(
        problem_id="E2",
        seed=1,
        outer_iter=0,
        group_index=1,
        selected_action_name="allow_beneficial_coordination",
        overlap_size=2,
        previous_delta=1.0,
        current_delta=3.0,
        relation_id="O0_0_1",
        group_left=0,
        group_right=1,
        shared_vars=(2, 3),
        action_family="coordinate",
        canonical_action_name="allow_beneficial_coordination",
        relation_policy_source="rule_based_relation_policy",
        action_value_delta_norm=1.25,
    )

    assert row["relation_id"] == "O0_0_1"
    assert row["group_left"] == "0"
    assert row["group_right"] == "1"
    assert row["shared_vars_hash"]
    assert row["action_family"] == "coordinate"
    assert row["canonical_action_name"] == "allow_beneficial_coordination"
    assert row["relation_policy_source"] == "rule_based_relation_policy"
    assert row["action_value_delta_norm"] == "1.250000e+00"


def test_build_action_trace_row_marks_isolate_as_value_selection() -> None:
    runner = _load_runner_module()

    row = runner.build_action_trace_row(
        problem_id="E2",
        seed=1,
        outer_iter=0,
        group_index=1,
        selected_action_name="isolate_conflicting_relation",
        overlap_size=1,
        previous_delta=5.0,
        current_delta=3.0,
    )

    assert row["owner_selected"] == "previous"
    assert row["semantic_surface"] == "overlap_value_selection"
    assert row["optimizer_consumed"] == "1"


def test_build_action_trace_row_marks_terminal_action_not_downstream_consumed() -> None:
    runner = _load_runner_module()

    row = runner.build_action_trace_row(
        problem_id="E2",
        seed=1,
        outer_iter=0,
        group_index=19,
        selected_action_name="repair_shared_variable_binding",
        overlap_size=1,
        previous_delta=1.0,
        current_delta=3.0,
        downstream_consumed=False,
    )

    assert row["state_mutated"] == "1"
    assert row["downstream_consumed"] == "0"
    assert row["downstream_consumption_scope"] == "same_outer_iteration"
    assert row["optimizer_consumed"] == "0"


def test_case_artifact_path_disambiguates_problem_outputs(tmp_path: Path) -> None:
    runner = _load_runner_module()

    assert runner.case_artifact_path(tmp_path, "E2", "action_trace.csv") == (
        tmp_path / "E2_action_trace.csv"
    )
    assert runner.case_artifact_path(tmp_path, "S6", "overlap_relations.csv") == (
        tmp_path / "S6_overlap_relations.csv"
    )


def test_budget_remaining_ratio_uses_iteration_start_fes() -> None:
    runner = _load_runner_module()

    assert runner.iteration_start_budget_remaining_ratio(max_fes=2_000, sum_fes=500) == 0.75
    assert runner.iteration_start_budget_remaining_ratio(max_fes=2_000, sum_fes=2_128) == 0.0


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
    assert relations[0].previous_delta == 3.0
    assert relations[0].current_delta == 1.0
    assert relations[0].delta_signed_gap == -2.0
    assert relations[0].shared_var_count == 1
    assert relations[0].budget_remaining_ratio == 0.4
    assert relations[0].feature_coverage == 1.0
    assert relations[0].rank_signal > 0.0

    output_path = tmp_path / "overlap_relations.csv"
    runner._write_overlap_relation_trace(output_path, relations)

    written = output_path.read_text(encoding="utf-8")
    assert "relation_id,problem_id,outer_iter" in written
    assert "previous_delta,current_delta,delta_abs_gap,delta_signed_gap" in written
    assert "O1_0_1,E2,1,0,1,2,1.000000,2.000000" in written


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


def test_apply_action_to_relation_uses_canonical_fallback_and_coordinate_semantics() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=98.0,
        rank_signal=0.5,
        budget_remaining_ratio=0.8,
    )
    previous_values = np.array([0.0])
    current_values = np.array([100.0])
    fallback_action = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="fallback",
        action_family="fallback",
        confidence=0.0,
        trigger_reason="test",
    )
    coordinate_action = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="coordinate",
        action_family="coordinate",
        confidence=1.0,
        trigger_reason="test",
    )

    fallback = runner.apply_action_to_relation(
        relation=relation,
        action=fallback_action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=99.0,
    )
    coordinate = runner.apply_action_to_relation(
        relation=relation,
        action=coordinate_action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=99.0,
    )

    np.testing.assert_allclose(fallback, np.array([99.0]))
    np.testing.assert_allclose(coordinate, np.array([65.0]))


def test_write_action_decision_log_overwrites_previous_rows(tmp_path: Path) -> None:
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

    runner._write_action_decision_log(output_path, "run-001", [first_relation], [first_action])
    runner._write_action_decision_log(output_path, "run-001", [second_relation], [second_action])

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
        "relation_action_name",
        "canonical_action_name",
        "action_family",
        "confidence",
        "trigger_reason",
    ]
    assert len(rows) == 1
    assert rows[0]["relation_id"] == "O1_1_2"
    assert rows[0]["shared_vars_count"] == "2"
    assert rows[0]["relation_action_name"] == "isolate_conflicting_relation"
    assert rows[0]["canonical_action_name"] == "isolate_conflicting_relation"


def test_relation_dispatch_is_applied_before_next_group_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    bases_seen_by_combine: list[np.ndarray] = []
    optimize_calls = {"count": 0}

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 4, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            optimize_calls["count"] += 1
            if optimize_calls["count"] == 1:
                return {
                    "n_function_evaluations": 1,
                    "best_so_far_y": 900.0,
                    "best_so_far_x": np.array([0.0, 10.0]),
                }
            if optimize_calls["count"] == 2:
                return {
                    "n_function_evaluations": 1,
                    "best_so_far_y": 700.0,
                    "best_so_far_x": np.array([100.0, 20.0]),
                }
            self.problem["fitness_function"](np.array([30.0, 40.0]))
            return {
                "n_function_evaluations": 1,
                "best_so_far_y": 600.0,
                "best_so_far_x": np.array([30.0, 40.0]),
            }

    def fake_combine(x_batch, base, dims):
        bases_seen_by_combine.append(base.copy())
        combined = base.copy()
        combined[dims] = x_batch
        return combined

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "combine", fake_combine)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id: [[0, 1], [1, 2], [2, 3]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2]], [[1], [2]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id: {"dimension": 4, "overlap_degree": 1, "subgroups": [2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(
        runner,
        "decide_actions_for_relations",
        lambda relations: [
            runner.RelationActionDecision(
                relation_id=relations[0].relation_id,
                action_name="reassign_repair",
                action_family="reassign_repair",
                confidence=1.0,
                trigger_reason="test_forced_repair",
            )
        ],
    )

    runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=30,
            seed=1,
            enable_relation_dispatch=True,
            verbose=0,
        ),
    )

    assert bases_seen_by_combine
    assert bases_seen_by_combine[0][1] == 100.0


def test_run_problem_caps_aob_fitness_record_at_max_fes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 4, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            population_size = self.options["n_individuals"]
            x_batch = np.zeros((population_size, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": population_size,
                "best_so_far_y": 1000.0,
                "best_so_far_x": x_batch[0],
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id: [[0, 1], [1, 2], [2, 3]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2]], [[1], [2]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id: {"dimension": 4, "overlap_degree": 1, "subgroups": [2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    record, _elapsed, _trace_rows = runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(max_fes=20, seed=1, verbose=0),
    )

    assert len(record) == 20


def test_main_preserves_case_level_action_traces_for_multiple_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    def fake_run_problem(fun_name, fun_id, output_path, config):
        problem_id = runner._problem_id(fun_name, fun_id)
        relation = runner.OverlapRelation(
            relation_id=f"O0_{fun_id - 1}_{fun_id}",
            problem_id=problem_id,
            outer_iter=0,
            group_left=fun_id - 1,
            group_right=fun_id,
            shared_vars=(fun_id,),
            overlap_strength=1.0,
            delta_signal=0.1,
            rank_signal=0.9,
            budget_remaining_ratio=0.8,
        )
        action = runner.RelationActionDecision(
            relation_id=relation.relation_id,
            action_name="fallback",
            action_family="fallback",
            confidence=0.0,
            trigger_reason="test",
        )
        runner._write_action_decision_log(
            runner.case_artifact_path(output_path, problem_id, "action_decision.csv"),
            config.run_id,
            [relation],
            [action],
        )
        rows = [
            runner.build_action_trace_row(
                problem_id=problem_id,
                seed=config.seed,
                outer_iter=0,
                group_index=1,
                selected_action_name="conservative_no_action",
                overlap_size=1,
                previous_delta=1.0,
                current_delta=1.0,
            )
        ]
        return [1.0], 0.0, rows

    monkeypatch.setattr(runner, "run_problem", fake_run_problem)
    monkeypatch.setattr(runner, "evaluation_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "plot_evaluation_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "plot_evaluation_curve_best_so_far",
        lambda *args, **kwargs: None,
    )

    runner.main(
        [
            "--functions",
            "elliptic",
            "--ids",
            "1",
            "2",
            "--output-root",
            str(tmp_path),
            "--timestamp",
            "multi-case",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--enable-relation-dispatch",
        ]
    )

    output_path = tmp_path / "multi-case" / "elliptic"
    assert (output_path / "E1_action_trace.csv").exists()
    assert (output_path / "E2_action_trace.csv").exists()
    with (output_path / "action_trace.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["problem_id"] for row in rows] == ["E1", "E2"]
    with (output_path / "action_decision.csv").open(newline="", encoding="utf-8") as handle:
        decision_rows = list(csv.DictReader(handle))

    assert [row["problem_id"] for row in decision_rows] == ["E1", "E2"]
    assert (output_path / "E1_action_decision.csv").exists()
    assert (output_path / "E2_action_decision.csv").exists()


def test_terminal_relation_trace_is_not_downstream_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    def fake_run_problem(fun_name, fun_id, output_path, config):
        rows = [
            runner.build_action_trace_row(
                problem_id="E2",
                seed=config.seed,
                outer_iter=0,
                group_index=1,
                selected_action_name="repair_shared_variable_binding",
                overlap_size=1,
                previous_delta=1.0,
                current_delta=2.0,
                downstream_consumed=True,
            ),
            runner.build_action_trace_row(
                problem_id="E2",
                seed=config.seed,
                outer_iter=0,
                group_index=2,
                selected_action_name="repair_shared_variable_binding",
                overlap_size=1,
                previous_delta=2.0,
                current_delta=3.0,
                downstream_consumed=False,
            ),
        ]
        return [1.0], 0.0, rows

    monkeypatch.setattr(runner, "run_problem", fake_run_problem)
    monkeypatch.setattr(runner, "evaluation_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "plot_evaluation_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "plot_evaluation_curve_best_so_far",
        lambda *args, **kwargs: None,
    )

    runner.main(
        [
            "--functions",
            "elliptic",
            "--ids",
            "2",
            "--output-root",
            str(tmp_path),
            "--timestamp",
            "terminal-consumption",
            "--seed",
            "1",
            "--max-fes",
            "2000",
        ]
    )

    with (tmp_path / "terminal-consumption" / "elliptic" / "action_trace.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows[-1]["state_mutated"] == "1"
    assert rows[-1]["downstream_consumed"] == "0"
    assert rows[-1]["optimizer_consumed"] == "0"


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
