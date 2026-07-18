from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import pytest

from experiments.pilots.exp_019_conflict_resolution_pilot import _diagnostic_worker
from experiments.pilots.exp_019_conflict_resolution_pilot.benchmark import (
    ConflictBenchmarkFactory,
)
from experiments.pilots.exp_019_conflict_resolution_pilot.diagnostic import (
    CONFIG_PATH,
    LARGE_LOSS_THRESHOLD,
    MATERIAL_THRESHOLD,
    ORACLE_MAX_FES,
    ORACLE_SEEDS,
    TrajectorySpec,
    _forbidden_runtime_hits,
    _side_checks,
    analyze_trajectory,
    build_specs,
    load_config,
    select_baseline_owner,
    summarize_side,
    wilson_bounds,
)


def _analysis_bundle(deltas: list[float] | None = None) -> dict[str, object]:
    relation_deltas = deltas or [0.02, 0.03, 0.04, 0.05]
    plan_rows = []
    probe_rows = []
    for index, delta in enumerate(relation_deltas):
        relation_id = f"g{index}-{index + 1}:v{index}"
        plan_rows.append(
            {
                "relation_id": relation_id,
                "selected": "1",
                "left_owner_reliability": "0.8",
                "right_owner_reliability": "0.2",
            }
        )
        bridge_fitness = 100.0
        left_fitness = bridge_fitness * math.exp(delta)
        candidates = {
            "x0": 120.0,
            "left_owner": left_fitness,
            "right_owner": left_fitness + 20.0,
            "bridge": bridge_fitness,
        }
        for candidate, fitness in candidates.items():
            probe_rows.append(
                {
                    "relation_id": relation_id,
                    "candidate": candidate,
                    "fitness": str(fitness),
                }
            )
    return {
        "spec": TrajectorySpec("oracle", "E3", 117, ORACLE_MAX_FES),
        "plan_rows": plan_rows,
        "probe_rows": probe_rows,
    }


def _side_rows(delta: float) -> list[dict[str, object]]:
    return [
        {
            "problem_id": problem_id,
            "seed": seed,
            "trajectory_delta": delta,
            "best_owner_trajectory_delta": delta - 0.01,
        }
        for problem_id in ("E3", "A4", "S5")
        for seed in ORACLE_SEEDS
    ]


def test_frozen_config_and_run_matrices() -> None:
    config = load_config()

    assert config["observer_only"] is True
    assert build_specs("smoke") == (
        TrajectorySpec("smoke", "A4", 1, 100_000),
    )
    oracle = build_specs("oracle")
    assert len(oracle) == 15
    assert {(spec.problem_id, spec.seed) for spec in oracle} == {
        (problem_id, seed)
        for problem_id in ("E3", "A4", "S5")
        for seed in ORACLE_SEEDS
    }


def test_config_matrix_drift_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    drifted = copy.deepcopy(config)
    drifted["oracle"]["seeds"] = [117]
    path = tmp_path / "diagnostic_config.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")

    with pytest.raises(ValueError, match="oracle matrix"):
        load_config(path)


def test_baseline_uses_higher_reliability_and_ties_to_left() -> None:
    assert select_baseline_owner(0.7, 0.2) == "left_owner"
    assert select_baseline_owner(0.2, 0.7) == "right_owner"
    assert select_baseline_owner(0.5, 0.5) == "left_owner"


def test_trajectory_delta_is_median_of_four_relations() -> None:
    deltas = [-0.2, 0.01, 0.03, 0.5]
    relations, trajectory = analyze_trajectory(_analysis_bundle(deltas), "conflict")

    assert len(relations) == 4
    assert trajectory["trajectory_delta"] == pytest.approx(0.02)
    assert trajectory["material_win"] == 1
    assert trajectory["large_loss"] == 0
    assert all(row["baseline_owner"] == "left_owner" for row in relations)


def test_missing_relation_candidate_fails_closed() -> None:
    bundle = _analysis_bundle()
    bundle["probe_rows"] = bundle["probe_rows"][:-1]

    with pytest.raises(ValueError, match="missing relation candidate"):
        analyze_trajectory(bundle, "conflict")


def test_wilson_one_sided_bounds_cover_extremes() -> None:
    all_win_lcb, all_win_ucb = wilson_bounds(15, 15)
    no_win_lcb, no_win_ucb = wilson_bounds(0, 15)

    assert all_win_lcb > 0.5
    assert all_win_ucb == pytest.approx(1.0)
    assert no_win_lcb == pytest.approx(0.0)
    assert no_win_ucb < 0.5
    assert all_win_lcb == pytest.approx(1.0 - no_win_ucb)


def test_side_summary_material_win_and_large_loss_boundaries() -> None:
    positive = summarize_side(_side_rows(MATERIAL_THRESHOLD + 1e-6), "conflict")
    negative = summarize_side(_side_rows(LARGE_LOSS_THRESHOLD), "conflict")

    assert positive["material_win_count"] == 15
    assert positive["paired_win_lcb"] > 0.5
    assert positive["large_loss_count"] == 0
    assert negative["material_win_count"] == 0
    assert negative["large_loss_count"] == 15
    assert negative["large_loss_ucb"] == pytest.approx(1.0)


def test_wrong_case_seed_pairing_fails_closed() -> None:
    rows = _side_rows(0.0)
    rows[-1] = {**rows[-1], "seed": 117}

    with pytest.raises(ValueError, match="case-seed pairing"):
        summarize_side(rows, "conform")


def test_gate_uses_strict_conflict_and_bounded_conform_conditions() -> None:
    conflict = summarize_side(_side_rows(MATERIAL_THRESHOLD + 1e-6), "conflict")
    conform = summarize_side(_side_rows(0.0), "conform")

    assert all(_side_checks(conflict, conform).values())


def test_runtime_forbidden_fields_are_detected_by_fragment() -> None:
    forbidden = ("oracle", "final_error", "relative_gain")

    assert _forbidden_runtime_hits(
        ["remaining_fe", "oracle_owner", "prior_final_error"],
        forbidden,
    ) == ["oracle_owner", "prior_final_error"]
    assert _forbidden_runtime_hits(["remaining_fe", "normal_sweep_fe"], forbidden) == []
    with pytest.raises(ValueError, match="list of strings"):
        _forbidden_runtime_hits("oracle", forbidden)


def test_conform_trajectory_id_is_not_labeled_conflict() -> None:
    spec = TrajectorySpec("conform_control", "A4", 117, ORACLE_MAX_FES)

    assert spec.trajectory_id.startswith("conform:conform_control:A4:")


def test_worker_builds_frozen_paired_owner_request() -> None:
    args = argparse.Namespace(
        case="A4",
        seed=1,
        max_fes=100_000,
        output_root="results/test",
        timestamp="fixed-smoke",
    )
    request = _diagnostic_worker.build_runner_args(args)

    assert request[request.index("--functions") + 1] == "ackley"
    assert request[request.index("--ids") + 1] == "4"
    assert request[request.index("--evidence-overlay-mode") + 1] == "paired_owner"
    assert "--enable-relation-dispatch" in request
    assert "--skip-plots" in request


def test_worker_replaces_factory_only_in_child_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import hcc_smoke_runner

    observed: list[list[str]] = []
    monkeypatch.setattr(hcc_smoke_runner, "main", lambda args: observed.append(args))
    monkeypatch.setattr(hcc_smoke_runner, "Benchmark", object())

    result = _diagnostic_worker.main(
        [
            "--case",
            "A4",
            "--seed",
            "1",
            "--max-fes",
            "100000",
            "--output-root",
            "results/test",
            "--timestamp",
            "fixed-smoke",
        ]
    )

    assert result == 0
    assert hcc_smoke_runner.Benchmark is ConflictBenchmarkFactory
    assert len(observed) == 1
