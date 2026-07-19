from __future__ import annotations

import numpy as np

from experiments.pilots.exp_023_conflict_conditioned_blend_pilot.run import (
    DEFAULT_CONFIG_PATH,
    build_command,
    load_config,
)
from scripts import hcc_smoke_runner as runner


def test_exp_023_explicitly_selects_blend_mode(tmp_path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    execution = config["execution"]
    command = build_command("E3", 117, config, tmp_path, "python-test")

    assert config["status"] == "authorized_by_user_for_fresh_runtime_pilot"
    assert execution["cases"] == ["E3", "S5", "R4"]
    assert execution["seeds"] == [117, 119, 120, 121, 122]
    assert execution["max_fes"] == 3_000_000
    assert execution["runtime_probe_repair_mode"] == "conflict_conditioned_blend"
    assert command[command.index("--runtime-probe-repair-mode") + 1] == (
        "conflict_conditioned_blend"
    )


def test_conflict_blend_equals_clipped_hcc_blend_at_threshold() -> None:
    previous = np.array([0.0, 10.0])
    current = np.array([10.0, 0.0])

    blended = runner.conflict_conditioned_context_blend(
        previous,
        current,
        previous_delta=1.0,
        current_delta=3.0,
        probe_utility=runner.SHADOW_GAIN_THRESHOLD,
    )

    assert np.allclose(blended, np.array([6.5, 3.5]))


def test_conflict_blend_sharpens_toward_each_winner() -> None:
    previous = np.array([0.0])
    current = np.array([10.0])
    utility = runner.SHADOW_GAIN_THRESHOLD * 100.0

    current_wins = runner.conflict_conditioned_context_blend(
        previous,
        current,
        previous_delta=1.0,
        current_delta=3.0,
        probe_utility=utility,
    )
    previous_wins = runner.conflict_conditioned_context_blend(
        previous,
        current,
        previous_delta=3.0,
        current_delta=1.0,
        probe_utility=utility,
    )

    assert current_wins[0] > 9.99
    assert previous_wins[0] < 0.01


def test_conflict_blend_uses_midpoint_when_deltas_cancel() -> None:
    blended = runner.conflict_conditioned_context_blend(
        np.array([2.0, 4.0]),
        np.array([6.0, 8.0]),
        previous_delta=-1.0,
        current_delta=1.0,
        probe_utility=1.0,
    )

    assert np.array_equal(blended, np.array([4.0, 6.0]))
