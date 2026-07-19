from __future__ import annotations

from experiments.pilots.exp_022_runtime_probe_dispatch_pilot.run import (
    DEFAULT_CONFIG_PATH,
    build_command,
    load_config,
)
from scripts import hcc_smoke_runner as runner
from src.arac.evidence.overlap_relation_builder import OverlapRelation


def _relation() -> OverlapRelation:
    return OverlapRelation(
        relation_id="O3_1_2",
        problem_id="E3",
        outer_iter=3,
        group_left=1,
        group_right=2,
        shared_vars=(10, 11),
        overlap_strength=2.0,
        delta_signal=0.0,
        rank_signal=0.0,
        budget_remaining_ratio=0.5,
    )


def test_exp_022_config_freezes_three_cases_and_five_seeds(tmp_path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    execution = config["execution"]

    assert execution["cases"] == ["E3", "S5", "R4"]
    assert execution["seeds"] == [117, 119, 120, 121, 122]
    assert execution["max_fes"] == 3_000_000
    command = build_command("R4", 122, config, tmp_path, "python-test")
    assert command[command.index("--functions") + 1] == "rastrigin"
    assert command[command.index("--relation-policy") + 1] == "runtime_probe"
    assert "--enable-relation-dispatch" in command


def test_runtime_probe_relation_key_is_structural_not_iteration_scoped() -> None:
    relation = _relation()

    assert runner.runtime_probe_relation_key(
        relation.group_left,
        relation.group_right,
        relation.shared_vars,
    ) == ((1, 2), (10, 11))


def test_runtime_probe_preserves_repair_coordinate_and_fallback_actions() -> None:
    relation = _relation()

    repair = runner.decide_runtime_probe_relation_action(
        relation,
        "repair_shared_variable_binding",
    )
    coordinate = runner.decide_runtime_probe_relation_action(
        relation,
        "allow_beneficial_coordination",
    )
    fallback = runner.decide_runtime_probe_relation_action(
        relation,
        "conservative_no_action",
    )

    assert repair.canonical_action_name == "repair_shared_variable_binding"
    assert coordinate.canonical_action_name == "allow_beneficial_coordination"
    assert fallback.canonical_action_name == "conservative_no_action"
    assert repair.trigger_reason == runner.RUNTIME_PROBE_REPAIR_TRIGGER
    assert coordinate.trigger_reason == runner.RUNTIME_PROBE_COORDINATE_TRIGGER
    assert fallback.trigger_reason == runner.RUNTIME_PROBE_FALLBACK_TRIGGER
