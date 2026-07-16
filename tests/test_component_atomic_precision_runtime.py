from __future__ import annotations

import ast
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_runner_module():
    root = Path(__file__).resolve().parents[1]
    vendor_root = root / "vendor" / "hcc"
    sys.path.insert(0, str(vendor_root))
    path = root / "scripts" / "hcc_smoke_runner.py"
    spec = importlib.util.spec_from_file_location(
        "hcc_smoke_runner_component_atomic_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _install_fake_runtime(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    *,
    auxiliary_fe_route_active: bool = False,
):
    options_seen: list[dict[str, object]] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            array = np.asarray(vector, dtype=float)
            count = 1 if array.ndim == 1 else len(array)
            values = [1000.0] * count
            self.fitness_record.extend(values)
            return values

    class FakeBenchmark:
        def __init__(self, _output_dir: str, data_dir=None) -> None:
            self.data_dir = data_dir

        def get_function(self, _fun_name: str, _fun_id: int):
            return FakeFunction()

        def get_info(self, _fun_name: str, _fun_id: int):
            return {"dimension": 20, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            options_seen.append(dict(self.options))
            population = int(self.options["n_individuals"])
            batch = np.zeros((population, self.problem["ndim_problem"]))
            self.problem["fitness_function"](batch)
            return {
                "n_function_evaluations": population,
                "best_so_far_y": 900.0,
                "best_so_far_x": np.ones(self.problem["ndim_problem"]),
                "mean": np.ones(self.problem["ndim_problem"]),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda _fun_id, data_root=None: [[0, 1], [1, 2]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1]], [[1]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda _fun_id, data_root=None: {
            "dimension": 20,
            "overlap_degree": 1,
            "subgroups": [2, 2],
        },
    )
    monkeypatch.setattr(
        runner,
        "calculate_global_fes",
        lambda _max_fes, _degree: 0,
    )
    monkeypatch.setattr(
        runner,
        "calculate_cmaes_population_size",
        lambda _dimension: 4,
    )
    original_state_builder = runner.build_evidence_action_controller_v31_run_state

    def build_state(*args, **kwargs):
        state = original_state_builder(*args, **kwargs)
        state.phase_rescue_retired = not auxiliary_fe_route_active
        state.phase_rescue_productive_mature = auxiliary_fe_route_active
        return state

    monkeypatch.setattr(
        runner,
        "build_evidence_action_controller_v31_run_state",
        build_state,
    )
    monkeypatch.setattr(
        runner,
        "_plan_component_group_budgets",
        lambda **kwargs: tuple(
            int(kwargs["current_optimizer_budget"])
            for _ in kwargs["group_indices"]
        ),
    )
    return options_seen


def _run_arm(runner, root: Path, arm: str):
    root.mkdir()
    return runner.run_problem(
        "elliptic",
        2,
        root,
        runner.SmokeConfig(
            max_fes=100,
            seed=7,
            verbose=0,
            early_stopping_evaluations=1000,
            arac_action=runner.EVIDENCE_ACTION_CONTROLLER_V37,
            enable_relation_dispatch=False,
            budget_accounting="strict",
            component_precision_arm=arm,
        ),
    )


def test_component_arm_cli_is_v37_only_and_exclusive() -> None:
    runner = _load_runner_module()
    common = [
        "--functions",
        "elliptic",
        "--ids",
        "2",
        "--output-root",
        "out",
        "--max-fes",
        "5000",
    ]
    args = runner.parse_args(
        [
            *common,
            "--arac-action",
            runner.EVIDENCE_ACTION_CONTROLLER_V37,
            "--component-precision-arm",
            "a1_precision_component_once",
        ]
    )
    assert args.component_precision_arm == "a1_precision_component_once"

    with pytest.raises(SystemExit):
        runner.parse_args(
            [*common, "--component-precision-arm", "a0_v37"]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                *common,
                "--arac-action",
                runner.EVIDENCE_ACTION_CONTROLLER_V37,
                "--component-precision-arm",
                "a0_v37",
                "--precision-causal-arm",
                "baseline",
            ]
        )


def test_component_budget_plan_requires_contiguous_complete_populations() -> None:
    runner = _load_runner_module()
    assert runner._plan_component_group_budgets(
        group_indices=(1, 2),
        current_group_index=1,
        current_optimizer_budget=16,
        current_group_budget_fe=16,
        population_sizes=(4, 4, 4),
        decision_fe=20,
        cc_budget_limit_fe=80,
        terminal_target_fe=100,
    ) == (16, 16)
    assert runner._plan_component_group_budgets(
        group_indices=(0, 2),
        current_group_index=0,
        current_optimizer_budget=16,
        current_group_budget_fe=16,
        population_sizes=(4, 4, 4),
        decision_fe=20,
        cc_budget_limit_fe=80,
        terminal_target_fe=100,
    ) == ()


def test_runtime_policy_call_has_only_pre_action_contract_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts" / "hcc_smoke_runner.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "plan_component_atomic_precision"
    ]
    assert len(calls) == 1
    assert {keyword.arg for keyword in calls[0].keywords} == {
        "candidate_feasible",
        "component_unlocked",
        "horizon_reachable",
        "once_lock_consumed",
        "group_indices",
        "group_budgets",
        "population_sizes",
        "normal_sigma",
        "precision_sigma",
    }
    direct_names = {
        node.id
        for keyword in calls[0].keywords
        for node in ast.walk(keyword.value)
        if isinstance(node, ast.Name)
    }
    assert direct_names.isdisjoint(
        {
            "problem_id",
            "fun_id",
            "fun_name",
            "seed",
            "paper_best",
            "terminal_error",
            "endpoint_error",
        }
    )


def test_a0_is_publicly_bit_equivalent_and_a1_locks_one_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    options = _install_fake_runtime(runner, monkeypatch)

    off_record, _, off_trace = _run_arm(runner, tmp_path / "off", "off")
    off_options = list(options)
    options.clear()
    a0_record, _, a0_trace = _run_arm(runner, tmp_path / "a0", "a0_v37")
    a0_options = list(options)
    options.clear()
    a1_record, _, _a1_trace = _run_arm(
        runner,
        tmp_path / "a1",
        "a1_precision_component_once",
    )
    a1_options = list(options)

    assert a0_record == off_record
    assert a0_trace == off_trace
    assert [
        (
            row["max_function_evaluations"],
            row["sigma"],
            row.get("seed_rng"),
        )
        for row in a0_options
    ] == [
        (
            row["max_function_evaluations"],
            row["sigma"],
            row.get("seed_rng"),
        )
        for row in off_options
    ]
    assert len(a1_record) == len(a0_record)

    a0_branch = _read_rows(
        tmp_path / "a0" / "E2_component_action_branch_manifest.csv"
    )[0]
    a1_branch = _read_rows(
        tmp_path / "a1" / "E2_component_action_branch_manifest.csv"
    )[0]
    assert a0_branch["decision_id"] == a1_branch["decision_id"]
    assert a0_branch["component_plan_sha256"] == a1_branch["component_plan_sha256"]
    assert a0_branch["crn_descriptor_sha256"] == a1_branch["crn_descriptor_sha256"]
    assert a0_branch["action_applied"] == "0"
    assert a1_branch["action_applied"] == "1"
    assert a0_branch["atomic_closed"] == a1_branch["atomic_closed"] == "1"
    assert a0_branch["h_endpoint_count"] == a1_branch["h_endpoint_count"] == "1"
    assert (
        a0_branch["plan_integrity_valid"]
        == a1_branch["plan_integrity_valid"]
        == "1"
    )
    assert (
        a0_branch["delayed_status"]
        == a1_branch["delayed_status"]
        == "resolved_next_component_entry"
    )
    assert int(a0_branch["delayed_review_outer_iter"]) == int(a0_branch["outer_iter"]) + 1
    assert int(a1_branch["delayed_review_outer_iter"]) == int(a1_branch["outer_iter"]) + 1
    assert (
        a0_branch["delayed_review_group_index"]
        == a1_branch["delayed_review_group_index"]
        == a0_branch["component_group_indices"].split(";", maxsplit=1)[0]
    )

    a0_budget = _read_rows(tmp_path / "a0" / "E2_component_budget_ledger.csv")
    a1_budget = _read_rows(tmp_path / "a1" / "E2_component_budget_ledger.csv")
    assert len(a0_budget) == len(a1_budget) == 2
    assert {float(row["sigma"]) for row in a0_budget} == {0.25}
    assert {float(row["sigma"]) for row in a1_budget} == {0.125}
    assert [row["requested_fe"] for row in a0_budget] == [
        row["requested_fe"] for row in a1_budget
    ]
    assert [row["auxiliary_actual_fe"] for row in a0_budget] == ["0", "1"]
    assert [row["auxiliary_actual_fe"] for row in a1_budget] == ["0", "1"]
    assert all(
        int(row["interval_actual_fe"])
        == int(row["actual_fe"]) + int(row["auxiliary_actual_fe"])
        for row in a0_budget + a1_budget
    )
    assert a0_budget[1]["group_start_fe"] == a0_budget[0]["group_end_fe"]
    assert a1_budget[1]["group_start_fe"] == a1_budget[0]["group_end_fe"]
    assert any(float(row["sigma"]) == 0.25 for row in a1_options)
    assert sum(float(row["sigma"]) == 0.125 for row in a1_options) == 2


def test_active_auxiliary_fe_route_abstains_without_changing_v37(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    _install_fake_runtime(
        runner,
        monkeypatch,
        auxiliary_fe_route_active=True,
    )

    off_record, _, off_trace = _run_arm(runner, tmp_path / "off", "off")
    a0_record, _, a0_trace = _run_arm(runner, tmp_path / "a0", "a0_v37")
    a1_record, _, a1_trace = _run_arm(
        runner,
        tmp_path / "a1",
        "a1_precision_component_once",
    )

    assert a0_record == off_record == a1_record
    assert a0_trace == off_trace == a1_trace
    for arm, root in (
        ("a0_v37", tmp_path / "a0"),
        ("a1_precision_component_once", tmp_path / "a1"),
    ):
        branch = _read_rows(root / "E2_component_action_branch_manifest.csv")[0]
        assert branch["component_precision_arm"] == arm
        assert branch["decision_status"] == "not_applicable"
        assert branch["not_applicable_reason"] == "active_auxiliary_fe_route"
        assert branch["action_applied"] == "0"
        assert branch["component_plan_frozen"] == "0"


def test_no_overlap_is_bit_equivalent_across_off_a0_and_a1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    _install_fake_runtime(runner, monkeypatch)
    grouping = [[0, 1], [2, 3]]
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda _fun_id, data_root=None: grouping,
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda _grouping: (grouping, [], [[]]),
    )

    off_record, _, off_trace = _run_arm(runner, tmp_path / "off", "off")
    a0_record, _, a0_trace = _run_arm(runner, tmp_path / "a0", "a0_v37")
    a1_record, _, a1_trace = _run_arm(
        runner,
        tmp_path / "a1",
        "a1_precision_component_once",
    )

    assert off_record == a0_record == a1_record
    assert off_trace == a0_trace == a1_trace
    a0_branch = _read_rows(
        tmp_path / "a0" / "E2_component_action_branch_manifest.csv"
    )[0]
    a1_branch = _read_rows(
        tmp_path / "a1" / "E2_component_action_branch_manifest.csv"
    )[0]
    assert a0_branch["decision_status"] == a1_branch["decision_status"] == "not_applicable"
    assert (
        a0_branch["not_applicable_reason"]
        == a1_branch["not_applicable_reason"]
        == "no_overlap_component_candidate"
    )
    assert a0_branch["terminal_record_sha256"] == a1_branch["terminal_record_sha256"]


def test_legacy_revisit_cap_does_not_control_component_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    options = _install_fake_runtime(runner, monkeypatch)

    def fail_if_called(**_kwargs):
        raise AssertionError("legacy revisit cap must not gate component action")

    monkeypatch.setattr(
        runner,
        "calculate_scheduler_revisit_cap",
        fail_if_called,
    )

    _run_arm(runner, tmp_path / "a1", "a1_precision_component_once")

    branch = _read_rows(
        tmp_path / "a1" / "E2_component_action_branch_manifest.csv"
    )[0]
    assert branch["decision_status"] == "applicable"
    assert branch["not_applicable_reason"] == ""
    assert branch["action_applied"] == "1"
    assert any(float(row["sigma"]) == 0.125 for row in options)


def test_delayed_review_only_closes_at_the_immediate_canonical_entry() -> None:
    runner = _load_runner_module()

    assert runner._is_next_component_canonical_entry(
        decision_outer_iter=3,
        current_outer_iter=4,
        canonical_group_index=2,
        current_group_index=2,
    )
    assert not runner._is_next_component_canonical_entry(
        decision_outer_iter=3,
        current_outer_iter=3,
        canonical_group_index=2,
        current_group_index=2,
    )
    assert not runner._is_next_component_canonical_entry(
        decision_outer_iter=3,
        current_outer_iter=5,
        canonical_group_index=2,
        current_group_index=2,
    )
    assert not runner._is_next_component_canonical_entry(
        decision_outer_iter=3,
        current_outer_iter=4,
        canonical_group_index=2,
        current_group_index=3,
    )
