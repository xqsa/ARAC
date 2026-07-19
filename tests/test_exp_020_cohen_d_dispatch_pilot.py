from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from experiments.pilots.exp_020_cohen_d_dispatch_pilot.run import (
    DEFAULT_CONFIG_PATH,
    REPAIR_ACTION,
    RunResult,
    build_cohen_d_summary,
    build_command,
    build_decision,
    build_run_matrix,
    load_config,
    read_budget_audit,
)
from scripts import hcc_smoke_runner as runner
from src.arac.evidence.overlap_relation_builder import OverlapRelation
from src.arac.policy.evidence_overlay import cohen_d_from_moments


def _relation(*, cohen_d: float = 0.0) -> OverlapRelation:
    return OverlapRelation(
        relation_id="O0_0_1",
        problem_id="E3",
        outer_iter=0,
        group_left=0,
        group_right=1,
        shared_vars=(20,),
        overlap_strength=1.0,
        delta_signal=0.0,
        rank_signal=0.0,
        budget_remaining_ratio=0.5,
        cohen_d=cohen_d,
    )


def _runner_args() -> list[str]:
    return [
        "--functions",
        "rastrigin",
        "--ids",
        "4",
        "--output-root",
        "results/exp020-test",
        "--max-fes",
        "100000",
        "--seed",
        "1",
        "--arac-action",
        runner.EVIDENCE_ACTION_CONTROLLER_V37,
        "--relation-policy",
        runner.COHEN_D_RELATION_POLICY,
        "--enable-relation-dispatch",
    ]


def test_config_freezes_four_cases_five_seeds_and_dispatch() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    execution = config["execution"]

    assert execution["cases"] == ["E3", "A4", "S5", "R4"]
    assert execution["seeds"] == [1, 2, 3, 4, 5]
    assert execution["max_fes"] == 100_000
    assert execution["enable_relation_dispatch"] is True
    assert execution["relation_policy"] == "cohen_d_repair"
    assert execution["cohen_d_threshold"] == 0.8
    assert config["controls"]["conforming_overlap_cases"] == ["R4"]


def test_matrix_builds_twenty_v37_dispatch_trajectories(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    specs = build_run_matrix(config, tmp_path)
    command = build_command(specs[-1], config, python_executable="python-test")

    assert len(specs) == 20
    assert {(spec.case, spec.seed) for spec in specs} == {
        (case, seed) for case in ("E3", "A4", "S5", "R4") for seed in range(1, 6)
    }
    assert command[command.index("--functions") + 1] == "rastrigin"
    assert command[command.index("--arac-action") + 1] == runner.EVIDENCE_ACTION_CONTROLLER_V37
    assert command[command.index("--relation-policy") + 1] == "cohen_d_repair"
    assert "--enable-relation-dispatch" in command


def test_r4_cli_accepts_cohen_d_dispatch() -> None:
    parsed = runner.parse_args(_runner_args())

    assert parsed.functions == ["rastrigin"]
    assert parsed.ids == [4]
    assert parsed.enable_relation_dispatch is True
    assert parsed.relation_policy == "cohen_d_repair"


def test_r4_vendor_binding_evaluates_native_dimension(tmp_path: Path) -> None:
    benchmark = runner.Benchmark(
        str(tmp_path),
        data_dir=runner.DATA_DIR,
    )
    function = benchmark.get_function("rastrigin", 4)
    info = function.info()

    assert info["dimension"] == 1000
    assert np.isfinite(function(np.zeros(info["dimension"]))[0])


def test_budget_audit_reads_runner_csv_fields(tmp_path: Path) -> None:
    spec = build_run_matrix(load_config(DEFAULT_CONFIG_PATH), tmp_path)[0]
    artifact = spec.output_root / "nested" / "E3_budget_summary.csv"
    artifact.parent.mkdir(parents=True)
    with artifact.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("fitness_record_fe", "same_budget_violation"),
        )
        writer.writeheader()
        writer.writerow({"fitness_record_fe": "99997", "same_budget_violation": "0"})

    assert read_budget_audit(spec) == (99_997, 0)


def test_cohen_d_threshold_is_strict() -> None:
    assert (
        runner.overlap_action_name_for_lane(
            runner.EVIDENCE_ACTION_CONTROLLER_V37,
            _relation(cohen_d=0.8),
        )
        == "conservative_no_action"
    )
    assert (
        runner.overlap_action_name_for_lane(
            runner.EVIDENCE_ACTION_CONTROLLER_V37,
            _relation(cohen_d=0.8000001),
        )
        == REPAIR_ACTION
    )
    assert (
        runner.decide_cohen_d_relation_action(_relation(cohen_d=0.8000001)).canonical_action_name
        == REPAIR_ACTION
    )


def test_cohen_d_uses_pooled_top_k_standard_deviation() -> None:
    assert math.isclose(
        cohen_d_from_moments((1.0,), (5.0,), (1.0,), (1.0,)),
        4.0,
    )
    assert cohen_d_from_moments((1.0,), (5.0,), (0.0,), (0.0,)) == 0.0


def test_top_k_local_coordinates_map_to_global_shared_variable() -> None:
    relation = runner.with_relation_population_evidence(
        _relation(),
        [[10, 20, 30], [40, 20, 50]],
        {
            0: ((10.0, 0.0, 30.0), (11.0, 2.0, 31.0)),
            1: ((40.0, 4.0, 50.0), (41.0, 6.0, 51.0)),
        },
    )

    assert relation.left_distribution_centers == (1.0,)
    assert relation.right_distribution_centers == (5.0,)
    assert relation.left_distribution_standard_deviations == (1.0,)
    assert relation.right_distribution_standard_deviations == (1.0,)
    assert relation.left_top_k_count == relation.right_top_k_count == 2
    assert math.isclose(relation.cohen_d, 4.0)
    assert relation.owner_dominance_direction == -1
    assert relation.population_spread_asymmetry == 0.0


def test_summary_and_decision_require_exact_trigger_consistency() -> None:
    audit_rows = [
        {
            "trajectory_id": f"exp-{case}-{seed}",
            "case": case,
            "seed": seed,
            "cohen_d": 1.0 if case != "R4" else 0.4,
            "above_threshold": int(case != "R4"),
            "selected_action_name": (REPAIR_ACTION if case != "R4" else "conservative_no_action"),
            "trigger_consistent": 1,
        }
        for case in ("E3", "A4", "S5", "R4")
        for seed in range(1, 6)
    ]
    results = [
        RunResult(
            trajectory_id=f"exp-{case}-{seed}",
            case=case,
            seed=seed,
            status="completed",
            final_error=1.0,
            fitness_record_fe=100_000,
            max_fes=100_000,
            same_budget_violation=0,
            relation_count=1,
            repair_count=int(case != "R4"),
            conservative_count=int(case == "R4"),
            trigger_mismatch_count=0,
            selected_relation_actions="",
            elapsed_seconds=1.0,
            returncode=0,
            output_root=Path("results/test"),
            error_detail="",
        )
        for case in ("E3", "A4", "S5", "R4")
        for seed in range(1, 6)
    ]
    config = load_config(DEFAULT_CONFIG_PATH)
    summaries = build_cohen_d_summary(audit_rows)
    decision = build_decision(results, audit_rows, summaries, config)

    assert decision["status"] == "mechanism_verified"
    assert decision["repair_count"] == 15
    assert decision["conservative_count"] == 5
    assert decision["r4_control_median"] == 0.4
