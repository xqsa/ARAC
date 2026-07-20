from __future__ import annotations

from pathlib import Path

import pytest

from experiments.pilots.exp_024_boundary_gated_blend_pilot.run import (
    DEFAULT_CONFIG_PATH as EXP024_CONFIG_PATH,
)
from experiments.pilots.exp_024_boundary_gated_blend_pilot.run import (
    build_command as build_exp024_command,
)
from experiments.pilots.exp_024_boundary_gated_blend_pilot.run import (
    load_config as load_exp024_config,
)
from experiments.pilots.exp_025_repair_withheld_control_pilot.run import (
    DEFAULT_CONFIG_PATH as EXP025_CONFIG_PATH,
)
from experiments.pilots.exp_025_repair_withheld_control_pilot.run import (
    build_command as build_exp025_command,
)
from experiments.pilots.exp_025_repair_withheld_control_pilot.run import (
    load_config as load_exp025_config,
)
from scripts import hcc_smoke_runner as runner


@pytest.mark.parametrize(
    ("load_config", "config_path", "build_command", "expected_mode"),
    (
        (
            load_exp024_config,
            EXP024_CONFIG_PATH,
            build_exp024_command,
            "boundary_gated_exact",
        ),
        (
            load_exp025_config,
            EXP025_CONFIG_PATH,
            build_exp025_command,
            "always_withhold_repair",
        ),
    ),
)
def test_g0_abstention_pilot_commands_pass_runner_contract(
    load_config,
    config_path: Path,
    build_command,
    expected_mode: str,
    tmp_path: Path,
) -> None:
    config = load_config(config_path)
    command = build_command("E3", 117, config, tmp_path, "python-test")

    parsed = runner.parse_args(list(command[2:]))

    assert config["status"] == "authorized_by_user_for_fresh_runtime_pilot"
    assert config["protocol_version"] == "runtime-probe-g0-abstention-pilot-v1"
    assert parsed.runtime_probe_repair_mode == expected_mode


def test_boundary_gate_only_withholds_low_utility_repairs() -> None:
    threshold = runner.SHADOW_GAIN_THRESHOLD

    assert runner.runtime_probe_repair_abstain_reason(
        canonical_action="repair_shared_variable_binding",
        utility=1.5 * threshold,
        mode="boundary_gated_exact",
    ) == "boundary_utility_gate"
    assert runner.runtime_probe_repair_abstain_reason(
        canonical_action="repair_shared_variable_binding",
        utility=2.1 * threshold,
        mode="boundary_gated_exact",
    ) == ""
    assert runner.runtime_probe_repair_abstain_reason(
        canonical_action="allow_beneficial_coordination",
        utility=1.5 * threshold,
        mode="boundary_gated_exact",
    ) == ""


def test_withheld_control_only_withholds_repairs() -> None:
    assert runner.runtime_probe_repair_abstain_reason(
        canonical_action="repair_shared_variable_binding",
        utility=1.0,
        mode="always_withhold_repair",
    ) == "repair_writeback_withheld"
    assert runner.runtime_probe_repair_abstain_reason(
        canonical_action="allow_beneficial_coordination",
        utility=1.0,
        mode="always_withhold_repair",
    ) == ""
