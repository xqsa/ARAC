from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


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
