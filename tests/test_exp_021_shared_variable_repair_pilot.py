from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.pilots.exp_020_beneficial_coordination_pilot.run import (
    build_command,
    build_run_matrix,
    load_config,
)
from experiments.pilots.exp_021_shared_variable_repair_pilot.run import (
    DEFAULT_CONFIG_PATH,
)
from experiments.pilots.exp_021_shared_variable_repair_pilot import run as repair_run
from scripts import hcc_smoke_runner


def _runner_args(action: str, *extra: str) -> list[str]:
    return [
        "--functions",
        "elliptic",
        "--ids",
        "3",
        "--output-root",
        "results/exp021-test",
        "--max-fes",
        "100000",
        "--seed",
        "1",
        "--arac-action",
        action,
        "--relation-policy",
        "controller_v31",
        *extra,
    ]


def test_repair_config_changes_only_target_action_profile() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    execution = config["execution"]
    lanes = {lane["lane_id"]: lane["arac_action"] for lane in config["lanes"]}

    assert execution["cases"] == ["E3", "A4", "S5"]
    assert execution["seeds"] == [1]
    assert execution["max_fes"] == 100_000
    assert execution["enable_relation_dispatch"] is False
    assert execution["evidence_overlay_mode"] == "off"
    assert lanes == {
        "hcc_baseline": "conservative_no_action",
        "shared_variable_repair": "repair_shared_variable_binding",
    }


def test_repair_matrix_builds_six_non_dispatch_commands(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    specs = build_run_matrix(config, tmp_path)
    repair = next(spec for spec in specs if spec.lane_id == "shared_variable_repair")
    command = build_command(repair, config, python_executable="python-test")

    assert len(specs) == 6
    assert repair.trajectory_id.startswith("exp_021_shared_variable_repair_pilot-")
    assert command[command.index("--arac-action") + 1] == "repair_shared_variable_binding"
    assert "--enable-relation-dispatch" not in command


def test_repair_entrypoint_forwards_process_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(repair_run.sys, "argv", ["run.py", "--jobs", "1"])
    monkeypatch.setattr(
        repair_run,
        "paired_action_main",
        lambda argv, **_kwargs: observed.append(argv) or 0,
    )

    assert repair_run.main() == 0
    assert observed[0][-2:] == ["--jobs", "1"]


def test_runner_accepts_repair_only_without_dispatch_or_overlay() -> None:
    parsed = hcc_smoke_runner.parse_args(_runner_args("repair_shared_variable_binding"))
    assert parsed.enable_relation_dispatch is False

    with pytest.raises(SystemExit):
        hcc_smoke_runner.parse_args(
            _runner_args(
                "repair_shared_variable_binding",
                "--enable-relation-dispatch",
                "--evidence-overlay-mode",
                "paired_owner",
            )
        )


def test_existing_repair_action_selects_larger_improvement_owner() -> None:
    previous = np.array([1.0, 2.0])
    current = np.array([3.0, 4.0])

    assert np.array_equal(
        hcc_smoke_runner.apply_arac_overlap_action(
            "repair_shared_variable_binding", previous, current, 1.0, 2.0
        ),
        current,
    )
    assert np.array_equal(
        hcc_smoke_runner.apply_arac_overlap_action(
            "repair_shared_variable_binding", previous, current, 3.0, 2.0
        ),
        previous,
    )
