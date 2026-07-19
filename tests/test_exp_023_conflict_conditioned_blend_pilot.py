from __future__ import annotations

import csv

import numpy as np

from experiments.pilots.exp_023_conflict_conditioned_blend_pilot.analyze_relation_impacts import (
    RepairImpactEvent,
    analyze,
    load_repair_events,
)
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


def test_repair_trace_carries_probe_and_local_impact_telemetry(tmp_path) -> None:
    trace_path = tmp_path / "E3_action_trace.csv"
    row = runner.build_action_trace_row(
        problem_id="E3",
        seed=117,
        outer_iter=3,
        group_index=4,
        selected_action_name="repair_shared_variable_binding",
        overlap_size=3,
        previous_delta=3.0,
        current_delta=1.0,
        relation_id="O3_3_4",
        group_left=3,
        group_right=4,
        shared_vars=(1, 2, 3),
        probe_utility=runner.SHADOW_GAIN_THRESHOLD * 1.5,
        probe_utility_threshold=runner.SHADOW_GAIN_THRESHOLD,
    )
    row["local_pre_writeback_fitness"] = "1.00000000000000000e+02"
    row["local_post_writeback_fitness"] = "9.00000000000000000e+01"
    row["local_objective_credit"] = "1.00000000000000000e-01"
    incomplete = {**row, "outer_iter": "4", "local_objective_credit": ""}
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=runner.ACTION_TRACE_FIELDS)
        writer.writeheader()
        writer.writerows((row, incomplete))

    events, repair_rows, incomplete_rows = load_repair_events(tmp_path)

    assert repair_rows == 2
    assert incomplete_rows == 1
    assert len(events) == 1
    assert events[0].utility_ratio == 1.5
    assert events[0].delta_gap_ratio == 2.0 / 3.0
    assert events[0].local_objective_credit == 0.1


def test_relation_impact_analysis_separates_boundary_variance() -> None:
    events = []
    for relation_index, (ratio, credits) in enumerate(
        (
            (1.2, (-0.30, 0.30)),
            (1.5, (-0.20, 0.20)),
            (3.0, (0.04, 0.06)),
            (4.0, (0.05, 0.07)),
        )
    ):
        for outer_iter, credit in enumerate(credits):
            events.append(
                RepairImpactEvent(
                    problem_id="E3" if relation_index % 2 == 0 else "S5",
                    seed=117 + relation_index,
                    outer_iter=outer_iter,
                    group_left=relation_index,
                    group_right=relation_index + 1,
                    shared_vars_hash=f"hash-{relation_index}",
                    probe_utility=runner.SHADOW_GAIN_THRESHOLD * ratio,
                    probe_utility_threshold=runner.SHADOW_GAIN_THRESHOLD,
                    previous_delta=3.0,
                    current_delta=1.0,
                    local_pre_writeback_fitness=100.0,
                    local_post_writeback_fitness=100.0 * (1.0 - credit),
                    local_objective_credit=credit,
                )
            )

    report = analyze(
        events,
        boundary_ratio_max=2.0,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert report["boundary_repairs"]["event_count"] == 4
    assert report["nonboundary_repairs"]["event_count"] == 4
    assert report["boundary_vs_nonboundary"]["variance_ratio"] > 10.0
    assert report["relation_cluster_bootstrap"]["samples_usable"] > 0
