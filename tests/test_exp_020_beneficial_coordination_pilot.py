from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments.pilots.exp_020_beneficial_coordination_pilot.run import (
    DEFAULT_CONFIG_PATH,
    RunResult,
    build_command,
    build_decision,
    build_paired_comparison,
    build_run_matrix,
    load_config,
)
from scripts import hcc_smoke_runner


def _runner_args(action: str, *extra: str) -> list[str]:
    return [
        "--functions",
        "elliptic",
        "--ids",
        "3",
        "--output-root",
        "results/exp020-test",
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


def _result(case: str, lane: str, error: float, *, fe: int = 100_000) -> RunResult:
    action = (
        "conservative_no_action"
        if lane == "hcc_baseline"
        else "allow_beneficial_coordination"
    )
    return RunResult(
        trajectory_id=f"{case}-{lane}",
        case=case,
        seed=1,
        lane_id=lane,
        arac_action=action,
        enable_relation_dispatch=False,
        evidence_overlay_mode="off",
        status="completed",
        final_error=error,
        fitness_record_fe=fe,
        max_fes=100_000,
        same_budget_violation=0,
        action_trace_rows=3,
        selected_actions=action,
        elapsed_seconds=1.0,
        returncode=0,
        output_root=Path("results/test"),
        error_detail="",
    )


def test_config_freezes_requested_action_and_disables_dispatch() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    execution = config["execution"]
    lanes = {lane["lane_id"]: lane["arac_action"] for lane in config["lanes"]}

    assert execution["cases"] == ["E3", "A4", "S5"]
    assert execution["enable_relation_dispatch"] is False
    assert execution["evidence_overlay_mode"] == "off"
    assert lanes == {
        "hcc_baseline": "conservative_no_action",
        "beneficial_coordination": "allow_beneficial_coordination",
    }


def test_matrix_contains_six_paired_fresh_trajectories(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    specs = build_run_matrix(config, tmp_path)

    assert len(specs) == 6
    assert {(spec.case, spec.lane_id) for spec in specs} == {
        (case, lane)
        for case in ("E3", "A4", "S5")
        for lane in ("hcc_baseline", "beneficial_coordination")
    }
    command = build_command(specs[-1], config, python_executable="python-test")
    assert command[0] == "python-test"
    assert command[command.index("--arac-action") + 1] == "allow_beneficial_coordination"
    assert command[command.index("--evidence-overlay-mode") + 1] == "off"
    assert "--enable-relation-dispatch" not in command


@pytest.mark.parametrize(
    "action",
    ("conservative_no_action", "allow_beneficial_coordination"),
)
def test_shared_runner_accepts_only_requested_non_dispatch_actions(action: str) -> None:
    parsed = hcc_smoke_runner.parse_args(_runner_args(action))

    assert parsed.arac_action == action
    assert parsed.enable_relation_dispatch is False


def test_shared_runner_keeps_overlay_frozen_to_v37() -> None:
    with pytest.raises(SystemExit):
        hcc_smoke_runner.parse_args(
            _runner_args(
                "allow_beneficial_coordination",
                "--enable-relation-dispatch",
                "--evidence-overlay-mode",
                "paired_owner",
            )
        )


def test_positive_decision_requires_pair_majority_and_no_catastrophe() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    results = [
        _result("E3", "hcc_baseline", 100.0),
        _result("E3", "beneficial_coordination", 80.0),
        _result("A4", "hcc_baseline", 10.0),
        _result("A4", "beneficial_coordination", 9.0),
        _result("S5", "hcc_baseline", 20.0),
        _result("S5", "beneficial_coordination", 21.0),
    ]
    paired = build_paired_comparison(results, config)
    decision = build_decision(results, paired, config)

    assert decision["status"] == "pilot_positive_effect"
    assert decision["positive_pairs"] == 2
    assert decision["catastrophic_pairs"] == 0
    assert math.isclose(float(paired[0]["log_gain"]), math.log(1.25))


def test_small_terminal_batch_difference_keeps_same_budget_gate_open() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    results = [
        _result(case, lane, 10.0, fe=99_999 if case == "A4" and lane != "hcc_baseline" else 100_000)
        for case in ("E3", "A4", "S5")
        for lane in ("hcc_baseline", "beneficial_coordination")
    ]
    paired = build_paired_comparison(results, config)
    decision = build_decision(results, paired, config)

    assert decision["status"] == "pilot_no_positive_effect"
    assert paired[1]["status"] == "completed"
    assert paired[1]["equal_fe"] == 0
    assert paired[1]["fe_difference"] == -1
    assert paired[1]["same_budget_gate"] == 1


def test_terminal_shortfall_beyond_preregistered_tolerance_blocks_pilot() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    results = [
        _result(case, lane, 10.0, fe=99_799 if case == "A4" and lane != "hcc_baseline" else 100_000)
        for case in ("E3", "A4", "S5")
        for lane in ("hcc_baseline", "beneficial_coordination")
    ]
    paired = build_paired_comparison(results, config)
    decision = build_decision(results, paired, config)

    assert decision["status"] == "pilot_blocked"
    assert paired[1]["same_budget_gate"] == 0
