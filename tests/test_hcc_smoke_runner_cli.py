from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


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
    assert args.enable_relation_dispatch is False


def test_hcc_smoke_runner_parses_explicit_aob_data_root(tmp_path: Path) -> None:
    runner = _load_runner_module()
    data_root = tmp_path / "canonical-aob-data"

    args = runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "6",
            "--output-root",
            "out",
            "--seed",
            "3",
            "--max-fes",
            "3000000",
            "--aob-data-root",
            str(data_root),
        ]
    )

    assert args.aob_data_root == data_root.resolve()


def test_runner_aob_loaders_ignore_cwd_when_data_root_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    data_root = Path(__file__).resolve().parents[1] / "HCC_SRC" / "AOB" / "AOBG" / "datafile"
    monkeypatch.chdir(tmp_path)

    metadata = runner.load_aob_metadata(6, data_root)
    permutation = runner.load_permutation_vector(6, data_root)

    assert int(metadata["dimension"]) == 1000
    assert len(permutation) == 1000


def test_aob_benchmark_forwards_explicit_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_runner_module()
    import AOB.AOB as aob_module

    data_root = (tmp_path / "aob-data").resolve()
    captured: dict[str, object] = {}

    def fake_elliptic(function_id, output_path, data_dir=None):
        captured.update(
            function_id=function_id,
            output_path=output_path,
            data_dir=data_dir,
        )
        return object()

    monkeypatch.setattr(aob_module, "elliptic", fake_elliptic)

    benchmark = aob_module.Benchmark(None, data_dir=data_root)
    benchmark.get_function("elliptic", 6)

    assert captured["data_dir"] == data_root


def _write_minimal_aob_data_root(root: Path, function_id: int = 6) -> None:
    root.mkdir(parents=True)
    prefix = f"F{function_id}"
    (root / f"{prefix}-info.txt").write_text(
        "dimension: 2\n"
        "dimension_real: 2\n"
        "sub_num: 1\n"
        "subgroups: [2]\n"
        "subgroups_type: [2]\n"
        "overlap_degree: 0\n",
        encoding="utf-8",
    )
    for suffix in ("design", "p", "s", "w", "xopt", "R2"):
        (root / f"{prefix}-{suffix}.txt").write_text(
            f"{prefix}-{suffix}\n",
            encoding="utf-8",
        )


def test_aob_input_snapshot_is_deterministic_and_complete(tmp_path: Path) -> None:
    runner = _load_runner_module()
    data_root = tmp_path / "aob-data"
    _write_minimal_aob_data_root(data_root)

    first = runner.snapshot_aob_inputs(6, data_root)
    second = runner.snapshot_aob_inputs(6, data_root)

    assert first == second
    assert list(first) == sorted(first)
    assert set(first) == {
        "F6-R2.txt",
        "F6-design.txt",
        "F6-info.txt",
        "F6-p.txt",
        "F6-s.txt",
        "F6-w.txt",
        "F6-xopt.txt",
    }


def test_aob_input_audit_rejects_file_change(tmp_path: Path) -> None:
    runner = _load_runner_module()
    data_root = tmp_path / "aob-data"
    _write_minimal_aob_data_root(data_root)
    before = runner.snapshot_aob_inputs(6, data_root)
    (data_root / "F6-p.txt").write_text("changed\n", encoding="utf-8")
    after = runner.snapshot_aob_inputs(6, data_root)

    rows = runner.build_aob_input_audit_rows("E6", before, after)

    changed = [row for row in rows if row["file"] == "F6-p.txt"]
    assert changed[0]["unchanged"] == "0"
    with pytest.raises(RuntimeError, match="AOB input changed during E6"):
        runner.require_unchanged_aob_inputs("E6", rows)


def test_hcc_smoke_runner_parses_budget_shift_mean_blend_action() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "4",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--arac-action",
            "budget_shift_mean_blend",
        ]
    )

    assert args.arac_action == "budget_shift_mean_blend"


def test_hcc_smoke_runner_parses_bipop_search_state_action() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "ackley",
            "--ids",
            "4",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--arac-action",
            "bipop_search_state_restart",
        ]
    )

    assert args.arac_action == "bipop_search_state_restart"


def test_hcc_smoke_runner_parses_cc_harm_guarded_sep_refresh_action() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "rastrigin",
            "--ids",
            "4",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "3000000",
            "--arac-action",
            "cc_harm_guarded_sep_refresh",
        ]
    )

    assert args.arac_action == "cc_harm_guarded_sep_refresh"


def test_hcc_smoke_runner_parses_separable_cmaes_dispatch_action() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "rastrigin",
            "--ids",
            "5",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "3000000",
            "--arac-action",
            "separable_cmaes_dispatch_action",
        ]
    )

    assert args.arac_action == "separable_cmaes_dispatch_action"


def test_hcc_smoke_runner_parses_repair_bipop_search_state_action() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "ackley",
            "--ids",
            "5",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--arac-action",
            "repair_bipop_search_state_restart",
        ]
    )

    assert args.arac_action == "repair_bipop_search_state_restart"


def test_hcc_smoke_runner_parses_repair_protect_refine_action() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "ackley",
            "--ids",
            "5",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--arac-action",
            "repair_protect_refine",
        ]
    )

    assert args.arac_action == "repair_protect_refine"


def test_hcc_smoke_runner_parses_repair_protect_deep_refine_action() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "ackley",
            "--ids",
            "3",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--arac-action",
            "repair_protect_deep_refine",
        ]
    )

    assert args.arac_action == "repair_protect_deep_refine"


def test_hcc_smoke_runner_parses_phase_rescue_multistart_action() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "rastrigin",
            "--ids",
            "4",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--arac-action",
            "phase_rescue_multistart",
        ]
    )

    assert args.arac_action == "phase_rescue_multistart"


def test_hcc_smoke_runner_parses_evidence_action_controller() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "rastrigin",
            "--ids",
            "4",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "3000000",
            "--arac-action",
            "arac_evidence_action_controller_v1",
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v26",
        ]
    )

    assert args.arac_action == "arac_evidence_action_controller_v1"
    assert args.enable_relation_dispatch is True
    assert args.relation_policy == "adaptive_v26"


def test_hcc_smoke_runner_parses_evidence_action_controller_v2() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "schwefel",
            "--ids",
            "6",
            "--output-root",
            "out",
            "--seed",
            "2",
            "--max-fes",
            "3000000",
            "--arac-action",
            "arac_evidence_action_controller_v2",
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v24",
        ]
    )

    assert args.arac_action == "arac_evidence_action_controller_v2"
    assert args.enable_relation_dispatch is True
    assert args.relation_policy == "adaptive_v24"


def test_hcc_smoke_runner_parses_evidence_action_controller_v3() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "schwefel",
            "--ids",
            "6",
            "--output-root",
            "out",
            "--seed",
            "2",
            "--max-fes",
            "3000000",
            "--arac-action",
            "arac_evidence_action_controller_v3",
            "--enable-relation-dispatch",
            "--relation-policy",
            "controller_v3",
        ]
    )

    assert args.arac_action == "arac_evidence_action_controller_v3"
    assert args.enable_relation_dispatch is True
    assert args.relation_policy == "controller_v3"


def test_hcc_smoke_runner_parses_evidence_action_controller_v31() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "schwefel",
            "--ids",
            "6",
            "--output-root",
            "out",
            "--seed",
            "2",
            "--max-fes",
            "3000000",
            "--arac-action",
            "arac_evidence_action_controller_v31",
            "--enable-relation-dispatch",
            "--relation-policy",
            "controller_v31",
        ]
    )

    assert args.arac_action == "arac_evidence_action_controller_v31"
    assert args.enable_relation_dispatch is True
    assert args.relation_policy == "controller_v31"


@pytest.mark.parametrize("action_name", ["budget_shift_only", "mean_blend_only"])
def test_hcc_smoke_runner_parses_trajectory_diagnostic_actions(action_name: str) -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "4",
            "--output-root",
            "out",
            "--max-fes",
            "2000",
            "--arac-action",
            action_name,
        ]
    )

    assert args.arac_action == action_name


def test_trajectory_budget_shift_redistributes_same_total() -> None:
    runner = _load_runner_module()

    budgets = runner.allocate_trajectory_group_budgets(
        total_budget=90,
        population_sizes=[4, 4, 4],
        overlap_support=[0.5, 1.0, 0.5],
    )

    assert sum(budgets) == 90
    assert all(budget >= 4 for budget in budgets)
    assert budgets[1] > budgets[0]
    assert budgets[1] > budgets[2]


def test_trajectory_budget_shift_requires_runtime_contribution_signal() -> None:
    runner = _load_runner_module()

    budgets = runner.allocate_trajectory_group_budgets(
        total_budget=90,
        population_sizes=[4, 4, 4],
        overlap_support=[0.5, 1.0, 0.5],
        contribution_credit=[0.0, 0.0, 0.0],
    )

    assert sum(budgets) == 90
    assert max(budgets) - min(budgets) <= 1


def test_trajectory_budget_shift_follows_runtime_contribution_credit() -> None:
    runner = _load_runner_module()

    budgets = runner.allocate_trajectory_group_budgets(
        total_budget=90,
        population_sizes=[4, 4, 4],
        overlap_support=[0.5, 1.0, 0.5],
        contribution_credit=[0.0, 10.0, 0.0],
    )

    assert sum(budgets) == 90
    assert budgets[1] > budgets[0]
    assert budgets[1] > budgets[2]


def test_trajectory_credit_requires_multiple_positive_groups() -> None:
    runner = _load_runner_module()

    assert not runner.has_sufficient_trajectory_credit([])
    assert not runner.has_sufficient_trajectory_credit([0.0, 10.0, 0.0])
    assert runner.has_sufficient_trajectory_credit([1.0, 10.0, 0.0])


def test_trajectory_mean_blend_uses_cached_optimizer_mean() -> None:
    runner = _load_runner_module()

    blended, applied_count, delta_norm = runner.blend_trajectory_mean(
        base_mean=np.array([0.0, 10.0]),
        dims=[1, 2],
        variable_mean_cache={2: 20.0},
        lower=-100.0,
        upper=100.0,
        strength=0.25,
    )

    np.testing.assert_allclose(blended, np.array([0.0, 12.5]))
    assert applied_count == 1
    assert delta_norm == pytest.approx(2.5)


def test_trajectory_action_preserves_native_overlap_blend() -> None:
    runner = _load_runner_module()

    adjusted = runner.apply_arac_overlap_action(
        action_name="budget_shift_mean_blend",
        previous_values=np.array([100.0]),
        current_values=np.array([0.0]),
        previous_delta=10.0,
        current_delta=1.0,
    )

    np.testing.assert_allclose(adjusted, np.array([1000.0 / 11.0]))


def test_bipop_restart_plan_alternates_large_and_small_restart_modes() -> None:
    runner = _load_runner_module()

    first = runner.build_bipop_restart_plan(
        group_index=0,
        restart_count=0,
        base_population_size=8,
        base_sigma=0.5,
        base_budget=100,
        remaining_fes=100,
        rng=np.random.default_rng(1),
    )
    second = runner.build_bipop_restart_plan(
        group_index=0,
        restart_count=1,
        base_population_size=8,
        base_sigma=0.5,
        base_budget=100,
        remaining_fes=100,
        rng=np.random.default_rng(1),
    )

    assert first.restart_mode == "large_ipop"
    assert first.population_size == 16
    assert first.sigma == pytest.approx(1.0)
    assert first.escape_budget % first.population_size == 0
    assert second.restart_mode == "small_bipop"
    assert 2 <= second.population_size <= 8
    assert second.sigma <= 1.0
    assert second.escape_budget % second.population_size == 0


def test_bipop_restart_plan_respects_remaining_budget() -> None:
    runner = _load_runner_module()

    plan = runner.build_bipop_restart_plan(
        group_index=0,
        restart_count=0,
        base_population_size=8,
        base_sigma=0.5,
        base_budget=100,
        remaining_fes=15,
        rng=np.random.default_rng(1),
    )

    assert plan.population_size == 16
    assert plan.escape_budget == 0


def test_bipop_restart_plan_keeps_small_population_safe_for_hcc_cmaes() -> None:
    runner = _load_runner_module()

    plan = runner.build_bipop_restart_plan(
        group_index=0,
        restart_count=1,
        base_population_size=8,
        base_sigma=0.5,
        base_budget=20,
        remaining_fes=60,
        rng=np.random.default_rng(1),
    )

    assert plan.population_size >= 4
    assert int(plan.population_size / 2) > 1


def test_bipop_search_state_lane_uses_native_overlap_action() -> None:
    runner = _load_runner_module()

    assert runner.overlap_action_name_for_lane("bipop_search_state_restart") == "conservative_no_action"


def test_phase_rescue_multistart_uses_native_overlap_and_search_state() -> None:
    runner = _load_runner_module()

    assert runner.overlap_action_name_for_lane("phase_rescue_multistart") == "conservative_no_action"
    assert runner.is_phase_rescue_multistart_action("phase_rescue_multistart")
    assert runner.is_search_state_action("phase_rescue_multistart")


def test_repair_bipop_lane_uses_repair_overlap_action_and_bipop_search_state() -> None:
    runner = _load_runner_module()

    assert runner.overlap_action_name_for_lane("repair_bipop_search_state_restart") == "repair_shared_variable_binding"
    assert runner.is_bipop_search_state_action("repair_bipop_search_state_restart")


def test_repair_protect_refine_uses_repair_overlap_and_refine_sigma() -> None:
    runner = _load_runner_module()

    assert runner.overlap_action_name_for_lane("repair_protect_refine") == "repair_shared_variable_binding"
    assert runner.refine_sigma_for_action("repair_protect_refine", 0.5) == pytest.approx(0.25)
    assert runner.overlap_action_name_for_lane("repair_protect_deep_refine") == "repair_shared_variable_binding"
    assert runner.refine_sigma_for_action("repair_protect_deep_refine", 0.5) == pytest.approx(0.125)
    assert runner.refine_sigma_for_action("repair_shared_variable_binding", 0.5) == pytest.approx(0.5)


def test_bipop_restart_requires_consecutive_stagnation_and_cooldown() -> None:
    runner = _load_runner_module()

    assert not runner.should_trigger_bipop_restart(
        stagnation_count=1,
        cooldown_remaining=0,
        escape_budget=8,
    )
    assert runner.should_trigger_bipop_restart(
        stagnation_count=2,
        cooldown_remaining=0,
        escape_budget=8,
    )
    assert not runner.should_trigger_bipop_restart(
        stagnation_count=3,
        cooldown_remaining=1,
        escape_budget=8,
    )
    assert not runner.should_trigger_bipop_restart(
        stagnation_count=3,
        cooldown_remaining=0,
        escape_budget=0,
    )


def test_bipop_restart_acceptance_requires_material_relative_improvement() -> None:
    runner = _load_runner_module()

    assert not runner.should_accept_bipop_restart(
        candidate_best=999.95,
        incumbent_fitness=1000.0,
    )
    assert runner.should_accept_bipop_restart(
        candidate_best=999.80,
        incumbent_fitness=1000.0,
    )
    assert not runner.should_accept_bipop_restart(
        candidate_best=1000.0,
        incumbent_fitness=1000.0,
    )


def test_bipop_rejected_restart_uses_sweep_level_backoff() -> None:
    runner = _load_runner_module()

    assert runner.bipop_cooldown_after_restart(
        restart_accepted=True,
        sub_num=20,
        rejected_restart_streak=5,
    ) == runner.BIPOP_RESTART_COOLDOWN
    assert runner.bipop_cooldown_after_restart(
        restart_accepted=False,
        sub_num=20,
        rejected_restart_streak=1,
    ) == 20
    assert runner.bipop_cooldown_after_restart(
        restart_accepted=False,
        sub_num=20,
        rejected_restart_streak=3,
    ) == 60
    assert runner.bipop_cooldown_after_restart(
        restart_accepted=False,
        sub_num=20,
        rejected_restart_streak=10,
    ) == 60


def test_hcc_smoke_runner_help_works_without_pythonpath() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "HCC_SRC" / "arac_hcc_smoke_runner.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(runner_path), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--enable-relation-dispatch" in completed.stdout


def test_hcc_smoke_runner_loads_ragged_local_design_matrix() -> None:
    runner = _load_runner_module()

    design = runner.load_design_matrix(5)

    assert design.shape == (1000, 1000)


def test_hcc_smoke_runner_seed_derivation_matches_hcc_es_cycle_stage_shape() -> None:
    runner = _load_runner_module()

    assert runner.derive_optimizer_seed("1", "schwefel", 2, 0, 3) == 6494672570720988326


def test_hcc_smoke_runner_parses_relation_dispatch_flag() -> None:
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
            "--enable-relation-dispatch",
        ]
    )

    assert args.enable_relation_dispatch is True


def test_hcc_smoke_runner_parses_optimizer_restart_modes() -> None:
    runner = _load_runner_module()

    base_args = [
        "--functions",
        "schwefel",
        "--ids",
        "1",
        "--output-root",
        "out",
        "--seed",
        "1",
        "--max-fes",
        "2000",
    ]

    assert runner.parse_args(base_args).cmaes_restart is True
    assert runner.parse_args(base_args).mmes_restart is True
    assert runner.parse_args(base_args + ["--no-cmaes-restart"]).cmaes_restart is False
    assert runner.parse_args(base_args + ["--no-mmes-restart"]).mmes_restart is False


def test_hcc_smoke_runner_parses_budget_accounting_mode() -> None:
    runner = _load_runner_module()

    base_args = [
        "--functions",
        "schwefel",
        "--ids",
        "1",
        "--output-root",
        "out",
        "--seed",
        "1",
        "--max-fes",
        "2000",
    ]

    assert runner.parse_args(base_args).budget_accounting == "strict"
    assert runner.parse_args(base_args + ["--budget-accounting", "source"]).budget_accounting == "source"


def test_hcc_smoke_runner_parses_focused_compare_lane_profile() -> None:
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
            "--lane-profile",
            "focused_compare",
        ]
    )

    assert args.lane_profile == "focused_compare"


def test_hcc_smoke_runner_parses_focused_core_lane_profile() -> None:
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
            "--lane-profile",
            "focused_core",
        ]
    )

    assert args.lane_profile == "focused_core"


def test_hcc_smoke_runner_parses_landscape_escape_lane_profile() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "ackley",
            "--ids",
            "4",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--lane-profile",
            "landscape_escape",
        ]
    )

    assert args.lane_profile == "landscape_escape"


def test_hcc_smoke_runner_parses_skip_plots() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "schwefel",
            "--ids",
            "1",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--skip-plots",
        ]
    )

    assert args.skip_plots is True


def test_hcc_smoke_runner_parses_relation_policy_options() -> None:
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "rule",
        ]
    )

    assert args.relation_policy == "rule"

    shuffled = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "shuffled",
        ]
    )

    assert shuffled.relation_policy == "shuffled"

    lagged = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "lagged",
        ]
    )

    assert lagged.relation_policy == "lagged"

    adaptive_v2 = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v2",
        ]
    )

    assert adaptive_v2.relation_policy == "adaptive_v2"

    adaptive_v21 = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v21",
        ]
    )

    assert adaptive_v21.relation_policy == "adaptive_v21"

    adaptive_v22 = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v22",
        ]
    )

    assert adaptive_v22.relation_policy == "adaptive_v22"

    adaptive_v23 = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v23",
        ]
    )

    assert adaptive_v23.relation_policy == "adaptive_v23"

    adaptive_v24 = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v24",
        ]
    )

    assert adaptive_v24.relation_policy == "adaptive_v24"

    adaptive_v25 = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v25",
        ]
    )

    assert adaptive_v25.relation_policy == "adaptive_v25"

    adaptive_v26 = runner.parse_args(
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
            "--enable-relation-dispatch",
            "--relation-policy",
            "adaptive_v26",
        ]
    )

    assert adaptive_v26.relation_policy == "adaptive_v26"


def test_shuffled_relation_policy_uses_relation_local_action_shuffle() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O0_1_2",
        problem_id="E2",
        outer_iter=0,
        group_left=1,
        group_right=2,
        shared_vars=(7,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=1.0,
    )
    rule_action = runner.RelationActionDecision(
        relation_id=relation.relation_id,
        action_name="coordinate",
        action_family="coordinate",
        confidence=0.8,
        trigger_reason="rule",
    )
    previous_rule_action = runner.RelationActionDecision(
        relation_id="O0_0_1",
        action_name="fallback",
        action_family="fallback",
        confidence=0.7,
        trigger_reason="previous_rule",
    )

    shuffled = runner.select_relation_action_for_policy(
        relation=relation,
        action=rule_action,
        relation_policy_mode="shuffled",
        shuffled_source_action=previous_rule_action,
    )

    assert shuffled.relation_id == relation.relation_id
    assert shuffled.relation_action_name == "reassign_repair"
    assert shuffled.canonical_action_name == "repair_shared_variable_binding"
    assert shuffled.action_family == "reassign_repair"
    assert shuffled.trigger_reason == "deterministic_shuffled_negative_control_from:coordinate"


def test_lagged_relation_policy_uses_previous_rule_action_without_relabeling() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O0_1_2",
        problem_id="E2",
        outer_iter=0,
        group_left=1,
        group_right=2,
        shared_vars=(7,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=1.0,
    )
    rule_action = runner.RelationActionDecision(
        relation_id=relation.relation_id,
        action_name="fallback",
        action_family="fallback",
        confidence=0.8,
        trigger_reason="rule",
    )
    previous_rule_action = runner.RelationActionDecision(
        relation_id="O0_0_1",
        action_name="coordinate",
        action_family="coordinate",
        confidence=0.7,
        trigger_reason="previous_rule",
    )

    lagged = runner.select_relation_action_for_policy(
        relation=relation,
        action=rule_action,
        relation_policy_mode="lagged",
        shuffled_source_action=previous_rule_action,
    )

    assert lagged.relation_id == relation.relation_id
    assert lagged.relation_action_name == "coordinate"
    assert lagged.canonical_action_name == "allow_beneficial_coordination"
    assert lagged.action_family == "coordinate"
    assert lagged.trigger_reason.startswith("deterministic_lagged_relation_policy")

    first_shuffled = runner.select_relation_action_for_policy(
        relation=relation,
        action=rule_action,
        relation_policy_mode="lagged",
        shuffled_source_action=None,
    )

    assert first_shuffled.relation_action_name == "fallback"
    assert first_shuffled.canonical_action_name == "conservative_no_action"


def test_controller_v31_effective_policy_is_accepted_by_relation_selector() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O0_1_2",
        problem_id="S6",
        outer_iter=0,
        group_left=1,
        group_right=2,
        shared_vars=(7,),
        overlap_strength=1.0,
        delta_signal=0.0,
        rank_signal=0.2,
        budget_remaining_ratio=1.0,
        previous_delta=0.0,
        current_delta=0.0,
        shared_var_support_ratio=0.04,
    )
    rule_action = runner.RelationActionDecision(
        relation_id=relation.relation_id,
        action_name="fallback",
        action_family="fallback",
        confidence=0.0,
        trigger_reason="rule",
    )
    effective_mode = runner.effective_relation_policy_mode(
        "controller_v31",
        [relation, relation, relation],
    )

    selected = runner.select_relation_action_for_policy(
        relation=relation,
        action=rule_action,
        relation_policy_mode=effective_mode,
    )

    assert effective_mode == "adaptive_v26"
    assert selected.relation_action_name == "fallback"


def test_controller_v31_dense_run_collects_with_v24_then_locks_once() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.18)
    e6_like_relations = [
        runner.OverlapRelation(
            relation_id=f"O0_{index}_{index + 1}",
            problem_id="runtime_case",
            outer_iter=0,
            group_left=index,
            group_right=index + 1,
            shared_vars=tuple(range(10)),
            overlap_strength=1.0,
            delta_signal=1.0,
            rank_signal=0.50,
            budget_remaining_ratio=0.8,
            previous_delta=1.0,
            current_delta=1.0,
            both_positive=True,
            delta_ratio_gap=0.10 if index < 2 else 0.58,
            rank_stability=0.0 if index < 2 else 0.67,
            shared_var_count=10,
            shared_var_support_ratio=0.10,
            fallback_margin_proxy=0.90 if index < 2 else 0.83,
        )
        for index in range(3)
    ]

    assert state.effective_policy_mode == "adaptive_v24"
    assert state.phase_rescue_enabled is False

    state.lock_from_runtime_prefix(e6_like_relations[:2])

    assert state.effective_policy_mode == "adaptive_v24"

    state.lock_from_runtime_prefix(e6_like_relations)

    assert state.effective_policy_mode == "adaptive_v26"
    state.lock_from_runtime_prefix([])
    assert state.effective_policy_mode == "adaptive_v26"


def test_controller_v31_dense_run_keeps_v24_for_weaker_positive_prefix() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.20)
    relations = [
        runner.OverlapRelation(
            relation_id=f"O0_{index}_{index + 1}",
            problem_id="runtime_case",
            outer_iter=0,
            group_left=index,
            group_right=index + 1,
            shared_vars=tuple(range(10)),
            overlap_strength=1.0,
            delta_signal=1.0,
            rank_signal=0.50,
            budget_remaining_ratio=0.8,
            previous_delta=1.0,
            current_delta=1.0,
            both_positive=True,
            delta_ratio_gap=0.10 if index < 2 else 0.75,
            rank_stability=0.0 if index < 2 else 0.33,
            shared_var_count=10,
            shared_var_support_ratio=0.10,
            fallback_margin_proxy=0.90 if index < 2 else 0.79,
        )
        for index in range(3)
    ]

    state.lock_from_runtime_prefix(relations)

    assert state.effective_policy_mode == "adaptive_v24"
    assert state.phase_rescue_enabled is False
    assert runner.refine_sigma_for_action(
        runner.EVIDENCE_ACTION_CONTROLLER_V31,
        0.5,
        controller_v31_run_state=state,
    ) == pytest.approx(0.5)


def test_controller_v31_non_dense_run_selects_v26_precision_and_rescue() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)

    assert state.effective_policy_mode == "adaptive_v26"
    assert state.phase_rescue_enabled is True
    assert runner.refine_sigma_for_action(
        runner.EVIDENCE_ACTION_CONTROLLER_V31,
        0.5,
        controller_v31_run_state=state,
    ) == pytest.approx(0.25)


def _bounded_refresh_relation(
    runner,
    *,
    index: int,
    shared_var_count: int = 3,
    budget_remaining_ratio: float = 0.20,
):
    return runner.OverlapRelation(
        relation_id=f"O4_{index}_{index + 1}",
        problem_id="runtime_case",
        outer_iter=4,
        group_left=index,
        group_right=index + 1,
        shared_vars=tuple(range(shared_var_count)),
        overlap_strength=float(shared_var_count),
        delta_signal=0.0,
        rank_signal=0.5,
        budget_remaining_ratio=budget_remaining_ratio,
        previous_delta=0.0,
        current_delta=0.0,
        both_positive=False,
        one_side_zero=False,
        delta_ratio_gap=0.0,
        rank_stability=1.0,
        shared_var_count=shared_var_count,
        shared_var_support_ratio=0.1,
        feature_coverage=1.0,
        fallback_margin_proxy=1.0,
    )


def test_controller_v31_plans_bounded_late_refresh_from_runtime_evidence() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)
    relations = [_bounded_refresh_relation(runner, index=index) for index in range(2)]

    plan = runner.plan_bounded_late_nda_refresh(
        controller_v31_run_state=state,
        current_outer_relations=relations,
        fitness_deltas=[0.0, 0.0, 0.0],
        overlap_writeback_norms=[0.0, 0.0],
        reference_fitness=1_000_000.0,
        remaining_fes=600_000,
        max_fes=3_000_000,
        population_size=40,
        expected_group_count=3,
    )

    assert plan is not None
    assert plan.refresh_budget == 450_000
    assert plan.continuation_reserve == 150_000
    assert plan.remaining_budget_ratio == pytest.approx(0.20)
    assert plan.shared_var_count == 3
    assert plan.trigger_reason == "low_cc_gain+severe_group_stagnation"


def test_controller_v31_plans_refresh_for_group_sparse_conflict_even_with_positive_gain() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)
    relations = [_bounded_refresh_relation(runner, index=index) for index in range(5)]
    fitness_deltas = [0.0, 10.0, 0.0, 20.0, 0.0, 30.0]

    plan = runner.plan_bounded_late_nda_refresh(
        controller_v31_run_state=state,
        current_outer_relations=relations,
        fitness_deltas=fitness_deltas,
        overlap_writeback_norms=[1.0, 1.0],
        reference_fitness=1_000_000.0,
        remaining_fes=600_000,
        max_fes=3_000_000,
        population_size=40,
        expected_group_count=6,
    )

    assert plan is not None
    assert plan.trigger_reason == (
        "group_sparse_stagnation+high_relation_conflict"
    )


def test_controller_v31_rejects_partial_outer_sweep_for_bounded_refresh() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)
    relations = [_bounded_refresh_relation(runner, index=index) for index in range(2)]

    plan = runner.plan_bounded_late_nda_refresh(
        controller_v31_run_state=state,
        current_outer_relations=relations,
        fitness_deltas=[0.0, 10.0, 0.0],
        overlap_writeback_norms=[1.0, 1.0],
        reference_fitness=1_000_000.0,
        remaining_fes=600_000,
        max_fes=3_000_000,
        population_size=40,
        expected_group_count=6,
    )

    assert plan is None


@pytest.mark.parametrize(
    ("state_overlap", "shared_count", "remaining_fes", "repair_locked", "deltas"),
    [
        (0.18, 3, 600_000, False, [0.0, 0.0, 0.0]),
        (0.10, 5, 600_000, False, [0.0, 0.0, 0.0]),
        (0.10, 1, 600_000, False, [0.0, 0.0, 0.0]),
        (0.10, 3, 1_050_000, False, [0.0, 0.0, 0.0]),
        (0.10, 3, 150_000, False, [0.0, 0.0, 0.0]),
        (0.10, 3, 600_000, True, [0.0, 0.0, 0.0]),
        (0.10, 3, 600_000, False, [100.0, 100.0, 100.0]),
    ],
)
def test_controller_v31_rejects_nonmatching_bounded_refresh_evidence(
    state_overlap: float,
    shared_count: int,
    remaining_fes: int,
    repair_locked: bool,
    deltas: list[float],
) -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(state_overlap)
    state.non_dense_repair_locked = repair_locked
    relations = [
        _bounded_refresh_relation(runner, index=index, shared_var_count=shared_count)
        for index in range(2)
    ]

    plan = runner.plan_bounded_late_nda_refresh(
        controller_v31_run_state=state,
        current_outer_relations=relations,
        fitness_deltas=deltas,
        overlap_writeback_norms=[0.0, 0.0],
        reference_fitness=1_000_000.0,
        remaining_fes=remaining_fes,
        max_fes=3_000_000,
        population_size=40,
        expected_group_count=len(deltas),
    )

    assert plan is None


def test_controller_v31_non_dense_guarded_prefix_locks_subsequent_repair() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)
    relations = [
        runner.OverlapRelation(
            relation_id=f"O0_{index}_{index + 1}",
            problem_id="runtime_case",
            outer_iter=0,
            group_left=index,
            group_right=index + 1,
            shared_vars=(10 + index, 20 + index, 30 + index),
            overlap_strength=3.0,
            delta_signal=4.0,
            rank_signal=0.5,
            budget_remaining_ratio=0.8,
            shared_var_count=3,
        )
        for index in range(4)
    ]
    prefix_actions = [
        runner.RelationActionDecision(
            relation_id=relations[0].relation_id,
            action_name="fallback",
            action_family="fallback",
            confidence=0.0,
            trigger_reason="no_deterministic_relation_rule_triggered",
        ),
        *[
            runner.RelationActionDecision(
                relation_id=relation.relation_id,
                action_name="reassign_repair",
                action_family="reassign_repair",
                confidence=0.8,
                trigger_reason="runtime_repair_candidate",
            )
            for relation in relations[1:3]
        ],
    ]

    guarded_prefix = []
    for relation, action in zip(relations[:3], prefix_actions, strict=True):
        guarded_action, _adjusted, _norm = runner.apply_relation_action_with_controller_v31(
            relation=relation,
            action=action,
            previous_values=np.zeros(3),
            current_values=np.full(3, 4.0),
            previous_delta=5.0,
            current_delta=1.0,
            controller_v31_run_state=state,
        )
        guarded_prefix.append(guarded_action)

    assert [action.relation_action_name for action in guarded_prefix] == [
        "fallback",
        "fallback",
        "reassign_repair",
    ]
    assert guarded_prefix[1].trigger_reason == "action_value_delta_guard_exceeded"
    assert guarded_prefix[2].trigger_reason == "controller_v31_non_dense_prefix_repair_lock"
    assert state.non_dense_repair_locked is True
    assert state.phase_rescue_enabled is False

    forced_action, adjusted, action_value_delta_norm = (
        runner.apply_relation_action_with_controller_v31(
            relation=relations[3],
            action=runner.RelationActionDecision(
                relation_id=relations[3].relation_id,
                action_name="fallback",
                action_family="fallback",
                confidence=0.0,
                trigger_reason="high_fallback_margin_keeps_native_overlap_blend",
            ),
            previous_values=np.zeros(3),
            current_values=np.full(3, 4.0),
            previous_delta=5.0,
            current_delta=1.0,
            controller_v31_run_state=state,
        )
    )

    assert forced_action.relation_action_name == "reassign_repair"
    assert forced_action.trigger_reason == "controller_v31_non_dense_prefix_repair_lock"
    np.testing.assert_allclose(adjusted, np.zeros(3))
    assert action_value_delta_norm > runner.ACTION_VALUE_DELTA_GUARD_THRESHOLD
    assert runner.relation_policy_source_name(
        "controller_v31",
        "adaptive_v26",
        action=forced_action,
    ).endswith(":non_dense_prefix_repair_lock")


def test_controller_v31_non_dense_large_unstable_fallback_locks_repair_immediately() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)
    relation = runner.OverlapRelation(
        relation_id="O0_0_1",
        problem_id="runtime_case",
        outer_iter=0,
        group_left=0,
        group_right=1,
        shared_vars=(10, 20, 30),
        overlap_strength=3.0,
        delta_signal=0.5,
        rank_signal=0.0,
        budget_remaining_ratio=0.8,
        previous_delta=5.0,
        current_delta=5.5,
        delta_ratio_gap=0.09,
        both_positive=True,
        shared_var_count=3,
    )

    action, adjusted, action_value_delta_norm = (
        runner.apply_relation_action_with_controller_v31(
            relation=relation,
            action=runner.RelationActionDecision(
                relation_id=relation.relation_id,
                action_name="fallback",
                action_family="fallback",
                confidence=0.0,
                trigger_reason="no_deterministic_relation_rule_triggered",
            ),
            previous_values=np.zeros(3),
            current_values=np.full(3, 40.0),
            previous_delta=5.0,
            current_delta=5.5,
            controller_v31_run_state=state,
        )
    )

    assert action.relation_action_name == "reassign_repair"
    assert action.trigger_reason == "controller_v31_non_dense_large_fallback_repair_lock"
    np.testing.assert_allclose(adjusted, np.full(3, 40.0))
    assert action_value_delta_norm == pytest.approx(0.0)
    assert state.non_dense_repair_locked is True
    assert state.phase_rescue_enabled is False
    assert runner.relation_policy_source_name(
        "controller_v31",
        "adaptive_v26",
        action=action,
    ).endswith(":non_dense_large_fallback_repair_lock")


@pytest.mark.parametrize(
    ("delta_ratio_gap", "current_value"),
    [
        (0.60, 40.0),
        (0.09, 1.0),
    ],
)
def test_controller_v31_non_dense_large_fallback_requires_low_delta_gap_and_large_shift(
    delta_ratio_gap: float,
    current_value: float,
) -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)
    relation = runner.OverlapRelation(
        relation_id="O0_0_1",
        problem_id="runtime_case",
        outer_iter=0,
        group_left=0,
        group_right=1,
        shared_vars=(10, 20, 30),
        overlap_strength=3.0,
        delta_signal=4.0,
        rank_signal=0.5,
        budget_remaining_ratio=0.8,
        previous_delta=5.0,
        current_delta=5.5,
        delta_ratio_gap=delta_ratio_gap,
        both_positive=True,
        shared_var_count=3,
    )

    action, _adjusted, _norm = runner.apply_relation_action_with_controller_v31(
        relation=relation,
        action=runner.RelationActionDecision(
            relation_id=relation.relation_id,
            action_name="fallback",
            action_family="fallback",
            confidence=0.0,
            trigger_reason="no_deterministic_relation_rule_triggered",
        ),
        previous_values=np.zeros(3),
        current_values=np.full(3, current_value),
        previous_delta=5.0,
        current_delta=5.5,
        controller_v31_run_state=state,
    )

    assert action.relation_action_name == "fallback"
    assert state.non_dense_repair_locked is False


@pytest.mark.parametrize(
    "prefix_actions",
    [
        [
            ("fallback", "no_deterministic_relation_rule_triggered"),
            ("fallback", "no_deterministic_relation_rule_triggered"),
            ("fallback", "high_fallback_margin_keeps_native_overlap_blend"),
        ],
        [
            ("fallback", "no_deterministic_relation_rule_triggered"),
            ("coordinate", "adaptive_v2_conflict_coordinate_evidence"),
            ("fallback", "action_value_delta_guard_exceeded"),
        ],
    ],
)
def test_controller_v31_non_dense_similar_three_relation_prefix_does_not_lock_repair(
    prefix_actions: list[tuple[str, str]],
) -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)

    for index, (action_name, trigger_reason) in enumerate(prefix_actions):
        relation = runner.OverlapRelation(
            relation_id=f"O0_{index}_{index + 1}",
            problem_id="runtime_case",
            outer_iter=0,
            group_left=index,
            group_right=index + 1,
            shared_vars=(10 + index, 20 + index, 30 + index),
            overlap_strength=3.0,
            delta_signal=0.0,
            rank_signal=0.5,
            budget_remaining_ratio=0.8,
            shared_var_count=3,
        )
        state.observe_guarded_relation_action(
            relation,
            runner.RelationActionDecision(
                relation_id=relation.relation_id,
                action_name=action_name,
                action_family=("coordinate" if action_name == "coordinate" else "fallback"),
                confidence=0.0,
                trigger_reason=trigger_reason,
            ),
        )

    assert state.non_dense_repair_locked is False


def test_controller_v31_never_enables_cc_harm_full_budget_takeover() -> None:
    runner = _load_runner_module()

    assert runner.uses_cc_harm_guard_during_run(
        runner.EVIDENCE_ACTION_CONTROLLER_V31,
        evidence_controller_search_state_enabled=True,
    ) is False
    assert runner.uses_cc_harm_guard_during_run(
        runner.EVIDENCE_ACTION_CONTROLLER_V3,
        evidence_controller_search_state_enabled=True,
    ) is True


def test_budget_summary_records_runtime_stage_breakdown(tmp_path: Path) -> None:
    runner = _load_runner_module()
    summary_path = tmp_path / "budget_summary.csv"

    runner._write_budget_summary(
        summary_path,
        problem_id="runtime_case",
        budget_accounting="strict",
        max_fes=100,
        optimizer_reported_fe=90,
        fitness_record_fe=100,
        global_phase_fe=20,
        cc_phase_fe=50,
        rescue_fe=20,
        refresh_fe=0,
        separable_continuation_fe=0,
    )

    with summary_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["global_phase_fe"] == "20"
    assert row["cc_phase_fe"] == "50"
    assert row["rescue_fe"] == "20"
    assert row["refresh_fe"] == "0"
    assert row["separable_continuation_fe"] == "0"
    assert row["overhead_fe"] == "10"


def test_shuffled_relation_policy_keeps_empty_overlap_fallback() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O0_0_1",
        problem_id="E1",
        outer_iter=0,
        group_left=0,
        group_right=1,
        shared_vars=(),
        overlap_strength=0.0,
        delta_signal=0.9,
        rank_signal=0.2,
        budget_remaining_ratio=0.8,
    )
    action = runner.RelationActionDecision(
        relation_id=relation.relation_id,
        action_name="fallback",
        action_family="fallback",
        confidence=0.0,
        trigger_reason="no_shared_overlap_support",
    )

    shuffled = runner.select_relation_action_for_policy(
        relation=relation,
        action=action,
        relation_policy_mode="shuffled",
    )

    assert shuffled.action_name == "fallback"
    assert shuffled.canonical_action_name == "conservative_no_action"


def test_hcc_smoke_runner_rejects_unsupported_action_file() -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner.parse_args(
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
                "--arac-action-file",
                "actions.csv",
            ]
        )


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
    expected = 0.25 * previous_values + 0.75 * current_values

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

    np.testing.assert_allclose(native, expected)
    np.testing.assert_allclose(action_result, native)


def test_allow_beneficial_coordination_uses_clipped_consensus_blend() -> None:
    runner = _load_runner_module()
    previous_values = np.array([0.0])
    current_values = np.array([100.0])

    coordinated = runner.apply_arac_overlap_action(
        action_name="allow_beneficial_coordination",
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=99.0,
    )

    np.testing.assert_allclose(coordinated, np.array([65.0]))


def test_isolate_conflicting_relation_keeps_stronger_side() -> None:
    runner = _load_runner_module()
    previous_values = np.array([1.0, 2.0])
    current_values = np.array([10.0, 20.0])

    isolated = runner.apply_arac_overlap_action(
        action_name="isolate_conflicting_relation",
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=5.0,
        current_delta=3.0,
    )

    np.testing.assert_allclose(isolated, previous_values)


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
    assert row["relation_id"] == ""
    assert row["canonical_action_name"] == "repair_shared_variable_binding"
    assert row["action_family"] == "reassign_repair"
    assert row["relation_policy_source"] == ""
    assert row["owner_selected"] == "current"
    assert row["semantic_surface"] == "shared_variable_owner_rebinding"
    assert row["state_mutated"] == "1"
    assert row["action_value_delta_norm"] == "0.000000e+00"
    assert row["downstream_consumed"] == "1"
    assert row["downstream_consumption_scope"] == "same_outer_iteration"
    assert row["optimizer_consumed"] == "1"


def test_build_action_trace_row_audits_bounded_refresh_runtime_state() -> None:
    runner = _load_runner_module()

    row = runner.build_action_trace_row(
        problem_id="R3",
        seed=3,
        outer_iter=4,
        group_index=2,
        selected_action_name=runner.BOUNDED_LATE_NDA_REFRESH_ACTION,
        overlap_size=3,
        previous_delta=0.0,
        current_delta=1.0,
        downstream_consumption_scope="subsequent_outer_iterations",
        trace_event="start",
        remaining_budget_ratio=0.20,
        shared_var_count=3,
        repair_lock_active=False,
        refresh_budget=450_000,
        continuation_reserve=150_000,
        optimizer_seed=12345,
    )

    assert row["trace_event"] == "start"
    assert row["remaining_budget_ratio"] == "2.000000e-01"
    assert row["shared_var_count"] == "3"
    assert row["repair_lock_active"] == "0"
    assert row["refresh_budget"] == "450000"
    assert row["continuation_reserve"] == "150000"
    assert row["optimizer_seed"] == "12345"
    assert row["downstream_consumption_scope"] == "subsequent_outer_iterations"


def test_build_action_trace_row_includes_relation_join_fields() -> None:
    runner = _load_runner_module()

    row = runner.build_action_trace_row(
        problem_id="E2",
        seed=1,
        outer_iter=0,
        group_index=1,
        selected_action_name="allow_beneficial_coordination",
        overlap_size=2,
        previous_delta=1.0,
        current_delta=3.0,
        relation_id="O0_0_1",
        group_left=0,
        group_right=1,
        shared_vars=(2, 3),
        action_family="coordinate",
        canonical_action_name="allow_beneficial_coordination",
        relation_policy_source="rule_based_relation_policy",
        action_value_delta_norm=1.25,
    )

    assert row["relation_id"] == "O0_0_1"
    assert row["group_left"] == "0"
    assert row["group_right"] == "1"
    assert row["shared_vars_hash"]
    assert row["action_family"] == "coordinate"
    assert row["canonical_action_name"] == "allow_beneficial_coordination"
    assert row["relation_policy_source"] == "rule_based_relation_policy"
    assert row["action_value_delta_norm"] == "1.250000e+00"


def test_build_action_trace_row_marks_isolate_as_value_selection() -> None:
    runner = _load_runner_module()

    row = runner.build_action_trace_row(
        problem_id="E2",
        seed=1,
        outer_iter=0,
        group_index=1,
        selected_action_name="isolate_conflicting_relation",
        overlap_size=1,
        previous_delta=5.0,
        current_delta=3.0,
    )

    assert row["owner_selected"] == "previous"
    assert row["semantic_surface"] == "overlap_value_selection"
    assert row["optimizer_consumed"] == "1"


def test_build_action_trace_row_marks_terminal_action_not_downstream_consumed() -> None:
    runner = _load_runner_module()

    row = runner.build_action_trace_row(
        problem_id="E2",
        seed=1,
        outer_iter=0,
        group_index=19,
        selected_action_name="repair_shared_variable_binding",
        overlap_size=1,
        previous_delta=1.0,
        current_delta=3.0,
        downstream_consumed=False,
    )

    assert row["state_mutated"] == "1"
    assert row["downstream_consumed"] == "0"
    assert row["downstream_consumption_scope"] == "same_outer_iteration"
    assert row["optimizer_consumed"] == "0"


def test_case_artifact_path_disambiguates_problem_outputs(tmp_path: Path) -> None:
    runner = _load_runner_module()

    assert runner.case_artifact_path(tmp_path, "E2", "action_trace.csv") == (
        tmp_path / "E2_action_trace.csv"
    )
    assert runner.case_artifact_path(tmp_path, "S6", "overlap_relations.csv") == (
        tmp_path / "S6_overlap_relations.csv"
    )


def test_budget_remaining_ratio_uses_iteration_start_fes() -> None:
    runner = _load_runner_module()

    assert runner.iteration_start_budget_remaining_ratio(max_fes=2_000, sum_fes=500) == 0.75
    assert runner.iteration_start_budget_remaining_ratio(max_fes=2_000, sum_fes=2_128) == 0.0


def test_build_overlap_relation_trace_exposes_adjacent_relations(tmp_path: Path) -> None:
    runner = _load_runner_module()

    relations = runner.build_overlap_relation_trace(
        problem_id="E2",
        outer_iter=1,
        grouping_result=[[0, 1, 2], [2, 3], [3, 4]],
        overlapping_elements=[[2], [3]],
        fitness_delta_list=[3.0, 1.0, 1.0],
        budget_remaining_ratio=0.4,
    )

    assert [relation.relation_id for relation in relations] == ["O1_0_1", "O1_1_2"]
    assert relations[0].shared_vars == (2,)
    assert relations[0].delta_signal == 2.0
    assert relations[0].previous_delta == 3.0
    assert relations[0].current_delta == 1.0
    assert relations[0].delta_signed_gap == -2.0
    assert relations[0].shared_var_count == 1
    assert relations[0].budget_remaining_ratio == 0.4
    assert relations[0].feature_coverage == 1.0
    assert relations[0].rank_signal > 0.0

    output_path = tmp_path / "overlap_relations.csv"
    runner._write_overlap_relation_trace(output_path, relations)

    written = output_path.read_text(encoding="utf-8")
    assert "relation_id,problem_id,outer_iter" in written
    assert "previous_delta,current_delta,delta_abs_gap,delta_signed_gap" in written
    assert "O1_0_1,E2,1,0,1,2,1.000000,2.000000" in written


def test_apply_action_to_relation_reuses_reassign_repair_logic() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=2.0,
        rank_signal=0.5,
        budget_remaining_ratio=0.8,
    )
    action = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="reassign_repair",
        action_family="reassign_repair",
        confidence=1.0,
        trigger_reason="test",
    )
    previous_values = np.array([1.0, 2.0])
    current_values = np.array([10.0, 20.0])

    repaired = runner.apply_action_to_relation(
        relation=relation,
        action=action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=3.0,
    )

    np.testing.assert_allclose(repaired, current_values)


def test_apply_action_to_relation_uses_canonical_fallback_and_coordinate_semantics() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=98.0,
        rank_signal=0.5,
        budget_remaining_ratio=0.8,
    )
    previous_values = np.array([0.0])
    current_values = np.array([100.0])
    fallback_action = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="fallback",
        action_family="fallback",
        confidence=0.0,
        trigger_reason="test",
    )
    coordinate_action = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="coordinate",
        action_family="coordinate",
        confidence=1.0,
        trigger_reason="test",
    )

    fallback = runner.apply_action_to_relation(
        relation=relation,
        action=fallback_action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=99.0,
    )
    coordinate = runner.apply_action_to_relation(
        relation=relation,
        action=coordinate_action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=1.0,
        current_delta=99.0,
    )

    np.testing.assert_allclose(fallback, np.array([99.0]))
    np.testing.assert_allclose(coordinate, np.array([65.0]))


def test_write_action_decision_log_overwrites_previous_rows(tmp_path: Path) -> None:
    runner = _load_runner_module()
    first_relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=0.8,
    )
    second_relation = runner.OverlapRelation(
        relation_id="O1_1_2",
        problem_id="E2",
        outer_iter=1,
        group_left=1,
        group_right=2,
        shared_vars=(3, 4),
        overlap_strength=2.0,
        delta_signal=2.5,
        rank_signal=0.2,
        budget_remaining_ratio=0.7,
    )
    first_action = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="coordinate",
        action_family="coordinate",
        confidence=0.95,
        trigger_reason="stable",
    )
    second_action = runner.RelationActionDecision(
        relation_id="O1_1_2",
        action_name="isolate_conflicting_relation",
        action_family="isolate",
        confidence=1.0,
        trigger_reason="conflict",
    )
    output_path = tmp_path / "action_decision.csv"

    runner._write_action_decision_log(output_path, "run-001", [first_relation], [first_action])
    runner._write_action_decision_log(output_path, "run-001", [second_relation], [second_action])

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "run_id",
        "problem_id",
        "relation_id",
        "group_left",
        "group_right",
        "shared_vars_count",
        "overlap_strength",
        "delta_signal",
        "rank_signal",
        "relation_action_name",
        "canonical_action_name",
        "action_family",
        "confidence",
        "trigger_reason",
    ]
    assert len(rows) == 1
    assert rows[0]["relation_id"] == "O1_1_2"
    assert rows[0]["shared_vars_count"] == "2"
    assert rows[0]["relation_action_name"] == "isolate_conflicting_relation"
    assert rows[0]["canonical_action_name"] == "isolate_conflicting_relation"


def test_write_action_mismatch_audit_log_overwrites_previous_rows(tmp_path: Path) -> None:
    runner = _load_runner_module()
    first_relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=0.8,
        previous_delta=1.0,
        current_delta=1.1,
        delta_abs_gap=0.1,
        delta_signed_gap=0.1,
        delta_ratio_gap=0.090909,
        both_positive=True,
        rank_stability=0.9,
        shared_var_count=1,
        shared_var_support_ratio=0.1,
        feature_coverage=1.0,
        fallback_margin_proxy=0.9,
    )
    second_relation = runner.OverlapRelation(
        relation_id="O1_1_2",
        problem_id="E2",
        outer_iter=1,
        group_left=1,
        group_right=2,
        shared_vars=(3,),
        overlap_strength=1.0,
        delta_signal=0.05,
        rank_signal=0.78,
        budget_remaining_ratio=0.8,
        previous_delta=1.0,
        current_delta=1.05,
        delta_abs_gap=0.05,
        delta_signed_gap=0.05,
        delta_ratio_gap=0.047619,
        both_positive=True,
        rank_stability=0.78,
        shared_var_count=1,
        shared_var_support_ratio=0.1,
        feature_coverage=1.0,
        fallback_margin_proxy=1.0,
    )
    output_path = tmp_path / "action_mismatch_audit.csv"

    runner._write_action_mismatch_audit_log(output_path, "run-001", [first_relation])
    runner._write_action_mismatch_audit_log(output_path, "run-001", [second_relation])

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == runner.ACTION_MISMATCH_AUDIT_FIELDS
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-001"
    assert rows[0]["relation_id"] == "O1_1_2"
    assert rows[0]["final_action_name"] == "fallback"
    assert rows[0]["second_best_action_name"] == "fallback"
    assert rows[0]["abstain_reason"] == "candidate_margin_below_threshold"
    assert "coordinate=" in rows[0]["candidate_scores"]


def test_action_mismatch_audit_scores_reset_by_outer_iteration(tmp_path: Path) -> None:
    runner = _load_runner_module()
    first_outer_relation = runner.OverlapRelation(
        relation_id="O0_0_1",
        problem_id="E2",
        outer_iter=0,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=1.0,
        rank_signal=0.75,
        budget_remaining_ratio=0.8,
        previous_delta=2.0,
        current_delta=1.0,
        delta_abs_gap=1.0,
        delta_signed_gap=-1.0,
        delta_ratio_gap=0.8,
        both_positive=True,
        rank_stability=0.75,
        shared_var_count=1,
        shared_var_support_ratio=0.166667,
        feature_coverage=1.0,
        fallback_margin_proxy=0.86,
    )
    next_outer_relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=1.0,
        rank_signal=0.60,
        budget_remaining_ratio=0.8,
        previous_delta=2.0,
        current_delta=1.0,
        delta_abs_gap=1.0,
        delta_signed_gap=-1.0,
        delta_ratio_gap=0.5,
        both_positive=True,
        rank_stability=0.60,
        shared_var_count=1,
        shared_var_support_ratio=0.10,
        feature_coverage=1.0,
        fallback_margin_proxy=0.9,
    )
    actions = [
        runner.RelationActionDecision(
            relation_id="O0_0_1",
            action_name="coordinate",
            action_family="coordinate",
            confidence=0.86,
            trigger_reason="balanced_mid_support_coordinate_mode",
        ),
        runner.RelationActionDecision(
            relation_id="O1_0_1",
            action_name="fallback",
            action_family="fallback",
            confidence=0.0,
            trigger_reason="no_deterministic_relation_rule_triggered",
        ),
    ]
    output_path = tmp_path / "action_mismatch_audit.csv"

    runner._write_action_mismatch_audit_log(
        output_path,
        "run-001",
        [first_outer_relation, next_outer_relation],
        actions,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[1]["relation_id"] == "O1_0_1"
    assert rows[1]["best_action_name"] == "fallback"
    assert rows[1]["final_action_name"] == "fallback"


def test_relation_action_value_delta_guard_allows_coordinate_blend_adjustment() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=0.8,
    )
    coordinate = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="coordinate",
        action_family="coordinate",
        confidence=0.95,
        trigger_reason="stable",
    )

    kept = runner.guard_relation_action_by_value_delta(
        relation,
        coordinate,
        runner.ACTION_VALUE_DELTA_GUARD_THRESHOLD,
    )
    moderate_coordinate = runner.guard_relation_action_by_value_delta(
        relation,
        coordinate,
        2.0,
    )
    guarded = runner.guard_relation_action_by_value_delta(
        relation,
        coordinate,
        2.501,
    )

    assert kept.relation_action_name == "coordinate"
    assert moderate_coordinate.relation_action_name == "coordinate"
    assert guarded.relation_action_name == "fallback"
    assert guarded.canonical_action_name == "conservative_no_action"
    assert guarded.trigger_reason == "action_value_delta_guard_exceeded"


def test_relation_action_value_delta_guard_keeps_strict_non_coordinate_limit() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=0.8,
    )
    isolate = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="isolate_conflicting_relation",
        action_family="isolate",
        confidence=0.95,
        trigger_reason="conflict",
    )

    guarded = runner.guard_relation_action_by_value_delta(
        relation,
        isolate,
        runner.ACTION_VALUE_DELTA_GUARD_THRESHOLD + 0.001,
    )

    assert guarded.relation_action_name == "fallback"
    assert guarded.canonical_action_name == "conservative_no_action"


def test_apply_and_guard_relation_action_recomputes_guarded_fallback_delta() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=0.8,
    )
    coordinate = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="coordinate",
        action_family="coordinate",
        confidence=0.95,
        trigger_reason="stable",
    )

    action, adjusted_values, action_value_delta_norm = (
        runner.apply_and_guard_action_to_relation(
            relation=relation,
            action=coordinate,
            previous_values=np.array([10.0]),
            current_values=np.array([0.0]),
            previous_delta=1.0,
            current_delta=0.0,
        )
    )

    assert action.relation_action_name == "fallback"
    assert adjusted_values is not None
    assert adjusted_values.tolist() == [10.0]
    assert action_value_delta_norm == 10.0


def test_apply_and_guard_falls_back_when_guard_blocks_isolate() -> None:
    runner = _load_runner_module()
    relation = runner.OverlapRelation(
        relation_id="O1_0_1",
        problem_id="E2",
        outer_iter=1,
        group_left=0,
        group_right=1,
        shared_vars=(2,),
        overlap_strength=1.0,
        delta_signal=0.1,
        rank_signal=0.9,
        budget_remaining_ratio=0.8,
    )
    isolate = runner.RelationActionDecision(
        relation_id="O1_0_1",
        action_name="isolate_conflicting_relation",
        action_family="isolate",
        confidence=0.95,
        trigger_reason="conflict",
    )

    action, adjusted_values, action_value_delta_norm = (
        runner.apply_and_guard_action_to_relation(
            relation=relation,
            action=isolate,
            previous_values=np.array([2.0]),
            current_values=np.array([0.0]),
            previous_delta=1.0,
            current_delta=0.0,
        )
    )

    assert action.relation_action_name == "fallback"
    assert action.canonical_action_name == "conservative_no_action"
    assert action.trigger_reason == "action_value_delta_guard_exceeded"
    assert adjusted_values is not None
    assert adjusted_values.tolist() == pytest.approx([2.0])
    assert action_value_delta_norm == pytest.approx(2.0)


def test_relation_dispatch_is_applied_before_next_group_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    bases_seen_by_combine: list[np.ndarray] = []
    policy_batch_sizes: list[int] = []
    optimize_calls = {"count": 0}

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 4, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            optimize_calls["count"] += 1
            if optimize_calls["count"] == 1:
                return {
                    "n_function_evaluations": 1,
                    "best_so_far_y": 900.0,
                    "best_so_far_x": np.array([0.0, 10.0]),
                }
            if optimize_calls["count"] == 2:
                return {
                    "n_function_evaluations": 1,
                    "best_so_far_y": 700.0,
                    "best_so_far_x": np.array([100.0, 20.0]),
                }
            self.problem["fitness_function"](np.array([30.0, 40.0]))
            return {
                "n_function_evaluations": 1,
                "best_so_far_y": 600.0,
                "best_so_far_x": np.array([30.0, 40.0]),
            }

    def fake_combine(x_batch, base, dims):
        bases_seen_by_combine.append(base.copy())
        combined = base.copy()
        combined[dims] = x_batch
        return combined

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "combine", fake_combine)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2]], [[1], [2]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 4, "overlap_degree": 1, "subgroups": [2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)

    def fake_decide_actions_for_relations(relations):
        policy_batch_sizes.append(len(relations))
        return [
            runner.RelationActionDecision(
                relation_id=relation.relation_id,
                action_name="reassign_repair",
                action_family="reassign_repair",
                confidence=1.0,
                trigger_reason="test_forced_repair",
            )
            for relation in relations
        ]

    monkeypatch.setattr(
        runner,
        "decide_actions_for_relations",
        fake_decide_actions_for_relations,
    )

    runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=30,
            seed=1,
            enable_relation_dispatch=True,
            verbose=0,
        ),
    )

    assert bases_seen_by_combine
    assert bases_seen_by_combine[0][1] == 100.0
    assert policy_batch_sizes[:2] == [1, 2]


def test_run_problem_caps_aob_fitness_record_at_max_fes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 4, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            population_size = self.options["n_individuals"]
            x_batch = np.zeros((population_size, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": population_size,
                "best_so_far_y": 1000.0,
                "best_so_far_x": x_batch[0],
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2]], [[1], [2]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 4, "overlap_degree": 1, "subgroups": [2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    record, _elapsed, _trace_rows = runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(max_fes=20, seed=1, verbose=0),
    )

    assert len(record) == 20


def test_run_problem_trajectory_action_passes_shifted_budget_and_blended_mean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    budgets_seen: list[int] = []
    means_seen: list[np.ndarray] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 4, "lower": -100.0, "upper": 100.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            budget = self.options["max_function_evaluations"]
            budgets_seen.append(budget)
            means_seen.append(np.asarray(self.options["mean"][0], dtype=float).copy())
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            result_mean = np.zeros(self.problem["ndim_problem"])
            best_so_far_x = np.zeros(self.problem["ndim_problem"])
            if len(budgets_seen) == 1:
                result_mean = np.array([0.0, 20.0])
                best_so_far_x = np.array([0.0, 8.0])
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 900.0,
                "best_so_far_x": best_so_far_x,
                "mean": result_mean,
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2]], [[1], [2]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 4, "overlap_degree": 1, "subgroups": [2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    record, _elapsed, _trace_rows = runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=90,
            seed=1,
            verbose=0,
            arac_action="budget_shift_mean_blend",
        ),
    )

    assert len(record) <= 90
    assert max(budgets_seen[:3]) - min(budgets_seen[:3]) <= 1
    assert means_seen[1][0] == pytest.approx(8.0)


def test_run_problem_bipop_search_state_restart_accepts_only_improving_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    options_seen: list[dict] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            options_seen.append(self.options)
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            is_escape = self.options.get("arac_search_state_action") == "bipop_search_state_restart"
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 900.0 if is_escape else 1000.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 3.0),
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 5, "overlap_degree": 1, "subgroups": [2, 2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    record, _elapsed, trace_rows = runner.run_problem(
        "ackley",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=500,
            seed=1,
            verbose=0,
            arac_action="bipop_search_state_restart",
        ),
    )

    escape_options = [
        options for options in options_seen
        if options.get("arac_search_state_action") == "bipop_search_state_restart"
    ]
    assert escape_options
    assert escape_options[0]["n_individuals"] == 8
    assert escape_options[0]["sigma"] == pytest.approx(1.0)
    assert len(record) <= 500

    bipop_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "bipop_search_state_restart"
    ]
    assert bipop_rows
    assert bipop_rows[0]["search_state_action_type"] == "bipop_restart"
    assert bipop_rows[0]["bipop_restart_mode"] == "large_ipop"
    assert bipop_rows[0]["restart_triggered"] == "1"
    assert bipop_rows[0]["restart_accepted"] == "1"
    assert bipop_rows[0]["restart_candidate_best"] == "9.000000e+02"
    assert bipop_rows[0]["restart_relative_improvement"] == "1.000000e-01"
    assert bipop_rows[0]["restart_acceptance_threshold"] == "1.000000e-04"
    assert bipop_rows[0]["population_after"] == "8"
    assert bipop_rows[0]["sigma_after"] == "1.000000e+00"
    assert bipop_rows[0]["optimizer_consumed"] == "1"
    overlap_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "conservative_no_action"
    ]
    assert overlap_rows
    assert all(row["search_state_action_type"] == "" for row in overlap_rows)


def test_run_problem_bipop_search_state_restart_rejects_tiny_escape_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            is_escape = self.options.get("arac_search_state_action") == "bipop_search_state_restart"
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 999.95 if is_escape else 1000.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 3.0),
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 5, "overlap_degree": 1, "subgroups": [2, 2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    _record, _elapsed, trace_rows = runner.run_problem(
        "ackley",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=500,
            seed=1,
            verbose=0,
            arac_action="bipop_search_state_restart",
        ),
    )

    bipop_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "bipop_search_state_restart"
    ]
    assert bipop_rows
    assert bipop_rows[0]["restart_triggered"] == "1"
    assert bipop_rows[0]["restart_accepted"] == "0"
    assert bipop_rows[0]["state_mutated"] == "0"
    assert bipop_rows[0]["restart_candidate_best"] == "9.999500e+02"
    assert bipop_rows[0]["restart_relative_improvement"] == "5.000000e-05"
    assert bipop_rows[0]["restart_acceptance_threshold"] == "1.000000e-04"
    assert bipop_rows[0]["best_before"] == "1.000000e+03"
    assert bipop_rows[0]["best_after"] == "1.000000e+03"


def test_run_problem_phase_rescue_multistart_accepts_best_improving_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    options_seen: list[dict] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            options_seen.append(self.options)
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            candidate_index = self.options.get("arac_phase_rescue_candidate")
            candidate_scores = {0: 980.0, 1: 870.0, 2: 920.0}
            best_y = candidate_scores.get(candidate_index, 1000.0)
            best_x = np.full(self.problem["ndim_problem"], float(candidate_index or 0) + 3.0)
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": best_y,
                "best_so_far_x": best_x,
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 5, "overlap_degree": 1, "subgroups": [2, 2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    _record, _elapsed, trace_rows = runner.run_problem(
        "rastrigin",
        4,
        tmp_path,
        runner.SmokeConfig(
            max_fes=500,
            seed=1,
            verbose=0,
            arac_action="phase_rescue_multistart",
        ),
    )

    rescue_options = [
        options
        for options in options_seen
        if options.get("arac_search_state_action") == "phase_rescue_multistart"
    ]
    assert len(rescue_options) >= 3
    assert {options["arac_phase_rescue_candidate"] for options in rescue_options[:3]} == {0, 1, 2}

    rescue_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "phase_rescue_multistart"
    ]
    assert rescue_rows
    assert rescue_rows[0]["search_state_action_type"] == "phase_rescue_multistart"
    assert rescue_rows[0]["bipop_restart_mode"] == "phase_rescue_3_start"
    assert rescue_rows[0]["restart_triggered"] == "1"
    assert rescue_rows[0]["restart_accepted"] == "1"
    assert rescue_rows[0]["restart_candidate_best"] == "8.700000e+02"
    assert rescue_rows[0]["restart_relative_improvement"] == "1.300000e-01"
    assert rescue_rows[0]["restart_acceptance_threshold"] == "0.000000e+00"
    assert rescue_rows[0]["best_before"] == "1.000000e+03"
    assert rescue_rows[0]["best_after"] == "8.700000e+02"


def test_run_problem_repair_phase_rescue_combines_repair_overlap_with_multistart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    options_seen: list[dict] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            options_seen.append(self.options)
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            candidate_index = self.options.get("arac_phase_rescue_candidate")
            candidate_scores = {0: 980.0, 1: 860.0, 2: 920.0}
            best_y = candidate_scores.get(candidate_index, 1000.0)
            best_x = np.full(self.problem["ndim_problem"], float(candidate_index or 0) + 3.0)
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": best_y,
                "best_so_far_x": best_x,
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 5, "overlap_degree": 1, "subgroups": [2, 2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    _record, _elapsed, trace_rows = runner.run_problem(
        "rastrigin",
        3,
        tmp_path,
        runner.SmokeConfig(
            max_fes=500,
            seed=1,
            verbose=0,
            arac_action="repair_phase_rescue_multistart",
        ),
    )

    rescue_options = [
        options
        for options in options_seen
        if options.get("arac_search_state_action") == "phase_rescue_multistart"
    ]
    assert len(rescue_options) >= 3

    rescue_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "repair_phase_rescue_multistart"
    ]
    assert rescue_rows
    assert rescue_rows[0]["search_state_action_type"] == "phase_rescue_multistart"
    assert rescue_rows[0]["restart_accepted"] == "1"
    assert rescue_rows[0]["restart_candidate_best"] == "8.600000e+02"

    repair_overlap_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "repair_shared_variable_binding"
    ]
    assert repair_overlap_rows
    assert all(row["search_state_action_type"] == "" for row in repair_overlap_rows)


def test_run_problem_cc_harm_guarded_sep_refresh_protects_phase_i_and_runs_nda_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    mmes_options_seen: list[dict] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeMMES:
        call_count = 0

        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            FakeMMES.call_count += 1
            mmes_options_seen.append(dict(self.options))
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            is_guarded_refresh = (
                self.options.get("arac_search_state_action")
                == "cc_harm_guarded_sep_refresh"
            )
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 700.0 if is_guarded_refresh else 800.0,
                "best_so_far_x": np.full(
                    self.problem["ndim_problem"],
                    4.0 if is_guarded_refresh else 1.0,
                ),
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 1000.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 3.0),
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "MMES", FakeMMES)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 5, "overlap_degree": 1, "subgroups": [2, 2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 16)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    _record, _elapsed, trace_rows = runner.run_problem(
        "rastrigin",
        4,
        tmp_path,
        runner.SmokeConfig(
            max_fes=160,
            seed=1,
            verbose=0,
            arac_action="cc_harm_guarded_sep_refresh",
        ),
    )

    assert len(mmes_options_seen) == 2
    assert mmes_options_seen[-1]["arac_search_state_action"] == "cc_harm_guarded_sep_refresh"
    assert np.allclose(mmes_options_seen[-1]["mean"][0], np.ones(5))

    guarded_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "cc_harm_guarded_sep_refresh"
    ]
    assert guarded_rows
    guarded_row = guarded_rows[0]
    assert guarded_row["search_state_action_type"] == "cc_harm_guarded_sep_refresh"
    assert guarded_row["restart_triggered"] == "1"
    assert guarded_row["restart_accepted"] == "1"
    assert guarded_row["best_before"] == "8.000000e+02"
    assert guarded_row["restart_candidate_best"] == "7.000000e+02"


def test_controller_v31_runs_one_bounded_refresh_then_resumes_cc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    call_order: list[str] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if np.asarray(vector).ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, _fun_name: str, _fun_id: int):
            return FakeFunction()

        def get_info(self, _fun_name: str, _fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            call_order.append("refresh")
            budget = self.options["max_function_evaluations"]
            self.problem["fitness_function"](
                np.zeros((budget, self.problem["ndim_problem"]))
            )
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 600.0,
                "best_so_far_x": np.ones(self.problem["ndim_problem"]),
            }

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            call_order.append("cc")
            budget = self.options["max_function_evaluations"]
            self.problem["fitness_function"](
                np.zeros((budget, self.problem["ndim_problem"]))
            )
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 650.0,
                "best_so_far_x": np.zeros(self.problem["ndim_problem"]),
                "mean": np.zeros(self.problem["ndim_problem"]),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "MMES", FakeMMES)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda _fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda _fun_id, data_root=None: {
            "dimension": 5,
            "overlap_degree": 1,
            "subgroups": [2, 2, 2, 2],
        },
    )
    monkeypatch.setattr(
        runner, "calculate_global_fes", lambda _max_fes, _degree: 0
    )
    monkeypatch.setattr(
        runner, "calculate_cmaes_population_size", lambda _dimension: 4
    )
    planner_calls = 0

    def fake_plan(**kwargs):
        nonlocal planner_calls
        planner_calls += 1
        state = kwargs["controller_v31_run_state"]
        if state.bounded_late_nda_refresh_consumed:
            return None
        return runner.BoundedLateNdaRefreshPlan(
            refresh_budget=20,
            continuation_reserve=8,
            remaining_budget_ratio=0.20,
            shared_var_count=3,
            trigger_reason="low_cc_gain+high_relation_conflict",
        )

    monkeypatch.setattr(runner, "plan_bounded_late_nda_refresh", fake_plan)

    _record, _elapsed, trace_rows = runner.run_problem(
        "rastrigin",
        3,
        tmp_path,
        runner.SmokeConfig(
            max_fes=160,
            seed=3,
            verbose=0,
            arac_action=runner.EVIDENCE_ACTION_CONTROLLER_V31,
            enable_relation_dispatch=True,
            relation_policy_mode="controller_v31",
        ),
    )

    refresh_index = call_order.index("refresh")
    assert "cc" in call_order[refresh_index + 1 :]
    bounded_rows = [
        row
        for row in trace_rows
        if row["selected_action_name"] == runner.BOUNDED_LATE_NDA_REFRESH_ACTION
    ]
    assert len(bounded_rows) == 2
    assert bounded_rows[0]["bipop_restart_mode"].startswith(
        "bounded_late_nda_refresh:start"
    )
    assert bounded_rows[1]["bipop_restart_mode"] == (
        "bounded_late_nda_refresh:completion"
    )
    assert bounded_rows[0]["restart_accepted"] == "1"
    assert planner_calls >= 1


def test_cc_harm_guarded_nda_continuation_rejects_worse_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    options_seen: list[dict] = []

    class FakeMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            options_seen.append(dict(self.options))
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget + 1, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": budget + 1,
                "best_so_far_y": 900.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 9.0),
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([900.0] * batch_size)
            return [900.0] * batch_size

    monkeypatch.setattr(runner, "MMES", FakeMMES)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    guard = np.full(3, 1.0)
    accepted, candidate, guarded_best, refresh_fes, candidate_best = (
        runner.run_guarded_nda_continuation(
            fun=FakeFunction(),
            info={"dimension": 3, "lower": -5.0, "upper": 5.0},
            config=runner.SmokeConfig(
                max_fes=100,
                seed=1,
                verbose=0,
                arac_action="cc_harm_guarded_sep_refresh",
            ),
            fun_name="rastrigin",
            fun_id=4,
            outer_iter=0,
            guard_individual=guard,
            guard_fitness=800.0,
            remaining_fes=20,
        )
    )

    assert accepted is False
    assert np.allclose(candidate, guard)
    assert guarded_best == 800.0
    assert candidate_best == 900.0
    assert refresh_fes == 17
    assert options_seen[0]["arac_search_state_action"] == "cc_harm_guarded_sep_refresh"


def test_guarded_nda_continuation_honors_bounded_budget_and_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    options_seen: list[dict] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if np.asarray(vector).ndim == 1 else len(vector)
            self.fitness_record.extend([700.0] * batch_size)
            return [700.0] * batch_size

    class FakeMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            options_seen.append(dict(self.options))
            budget = self.options["max_function_evaluations"]
            self.problem["fitness_function"](
                np.zeros((budget + 1, self.problem["ndim_problem"]))
            )
            return {
                "n_function_evaluations": budget + 1,
                "best_so_far_y": 700.0,
                "best_so_far_x": np.zeros(self.problem["ndim_problem"]),
            }

    monkeypatch.setattr(runner, "MMES", FakeMMES)
    monkeypatch.setattr(
        runner, "calculate_cmaes_population_size", lambda _dimension: 4
    )

    accepted, _candidate, best, used, candidate_best = (
        runner.run_guarded_nda_continuation(
            fun=FakeFunction(),
            info={"dimension": 3, "lower": -5.0, "upper": 5.0},
            config=runner.SmokeConfig(max_fes=100, seed=7, verbose=0),
            fun_name="rastrigin",
            fun_id=3,
            outer_iter=4,
            guard_individual=np.ones(3),
            guard_fitness=800.0,
            remaining_fes=40,
            requested_fes=20,
            search_state_action=runner.BOUNDED_LATE_NDA_REFRESH_ACTION,
        )
    )

    assert accepted is True
    assert best == 700.0
    assert candidate_best == 700.0
    assert used == 17
    assert options_seen[0]["max_function_evaluations"] == 16
    assert options_seen[0]["arac_search_state_action"] == (
        "bounded_late_nda_refresh"
    )
    assert options_seen[0]["seed_rng"] == runner.derive_optimizer_seed(
        7, "rastrigin", 3, 5, 23011
    )


def test_guarded_nda_continuation_rejects_nonfinite_optimizer_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class InvalidMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            return {
                "n_function_evaluations": 4,
                "best_so_far_y": float("nan"),
                "best_so_far_x": np.zeros(3),
            }

    monkeypatch.setattr(runner, "MMES", InvalidMMES)
    monkeypatch.setattr(
        runner, "calculate_cmaes_population_size", lambda _dimension: 4
    )

    with pytest.raises(RuntimeError, match="guarded NDA returned non-finite fitness"):
        runner.run_guarded_nda_continuation(
            fun=lambda _vector: [1.0],
            info={"dimension": 3, "lower": -5.0, "upper": 5.0},
            config=runner.SmokeConfig(max_fes=100, seed=7, verbose=0),
            fun_name="rastrigin",
            fun_id=3,
            outer_iter=4,
            guard_individual=np.ones(3),
            guard_fitness=800.0,
            remaining_fes=40,
            requested_fes=20,
            search_state_action=runner.BOUNDED_LATE_NDA_REFRESH_ACTION,
        )


def test_guarded_nda_continuation_uses_objective_fe_when_backend_reports_one_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if np.asarray(vector).ndim == 1 else len(vector)
            self.fitness_record.extend([700.0] * batch_size)
            return [700.0] * batch_size

    class OffByOneMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            budget = self.options["max_function_evaluations"]
            self.problem["fitness_function"](
                np.zeros((budget + 1, self.problem["ndim_problem"]))
            )
            return {
                "n_function_evaluations": budget + 1,
                "best_so_far_y": 700.0,
                "best_so_far_x": np.zeros(self.problem["ndim_problem"]),
            }

    monkeypatch.setattr(runner, "MMES", OffByOneMMES)
    monkeypatch.setattr(
        runner, "calculate_cmaes_population_size", lambda _dimension: 4
    )

    accepted, _candidate, best, used, candidate_best = (
        runner.run_guarded_nda_continuation(
            fun=FakeFunction(),
            info={"dimension": 3, "lower": -5.0, "upper": 5.0},
            config=runner.SmokeConfig(max_fes=100, seed=7, verbose=0),
            fun_name="rastrigin",
            fun_id=3,
            outer_iter=4,
            guard_individual=np.ones(3),
            guard_fitness=800.0,
            remaining_fes=40,
            requested_fes=20,
            search_state_action=runner.BOUNDED_LATE_NDA_REFRESH_ACTION,
        )
    )

    assert accepted is True
    assert best == 700.0
    assert candidate_best == 700.0
    assert used == 17


def test_direct_separable_cmaes_dispatch_keeps_incumbent_when_candidates_are_worse() -> None:
    runner = _load_runner_module()

    class WorseFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            array = np.asarray(vector, dtype=float)
            batch_size = 1 if array.ndim == 1 else len(array)
            values = [10.0] * batch_size
            self.fitness_record.extend(values)
            return values

    initial_mean = np.array([1.0, -2.0, 3.0])
    result = runner.run_direct_separable_cmaes_dispatch(
        fun=WorseFunction(),
        info={"dimension": 3, "lower": -5.0, "upper": 5.0},
        config=runner.SmokeConfig(
            max_fes=8,
            seed=1,
            verbose=0,
            arac_action="separable_cmaes_dispatch_action",
        ),
        fun_name="rastrigin",
        fun_id=5,
        initial_mean=initial_mean,
        incumbent_fitness=1.0,
        max_function_evaluations=8,
    )

    assert result["best_so_far_y"] == 1.0
    np.testing.assert_allclose(result["best_so_far_x"], initial_mean)
    assert result["n_function_evaluations"] == 8


def test_run_problem_separable_cmaes_dispatch_uses_full_space_diagonal_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    mmes_options_seen = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            array = np.asarray(vector, dtype=float)
            if array.ndim == 1:
                values = np.array([float(np.sum(array * array))])
            else:
                values = np.sum(array * array, axis=1).astype(float)
            self.fitness_record.extend(values.tolist())
            return values.tolist()

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 3, "lower": -5.0, "upper": 5.0}

    class FakeMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            budget = int(self.options["max_function_evaluations"])
            mmes_options_seen.append(dict(self.options))
            x_batch = np.ones((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 100.0,
                "best_so_far_x": np.ones(self.problem["ndim_problem"]),
            }

    def fail_cmaes(*args, **kwargs):
        raise AssertionError("separable dispatch should bypass Phase-II CMAES")

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "MMES", FakeMMES)
    monkeypatch.setattr(runner, "CMAES", fail_cmaes)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id, data_root=None: [[0, 1], [1, 2]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1]], [[1]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 3, "overlap_degree": 1, "subgroups": [2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 4)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    record, _elapsed, trace_rows = runner.run_problem(
        "rastrigin",
        5,
        tmp_path,
        runner.SmokeConfig(
            max_fes=20,
            seed=1,
            verbose=0,
            arac_action="separable_cmaes_dispatch_action",
        ),
    )

    assert len(record) == 20
    assert len(mmes_options_seen) == 1
    assert mmes_options_seen[0]["max_function_evaluations"] == 4
    assert mmes_options_seen[0]["arac_search_state_action"] == "separable_cmaes_dispatch_action"
    assert len(trace_rows) == 1
    row = trace_rows[0]
    assert row["selected_action_name"] == "separable_cmaes_dispatch_action"
    assert row["canonical_action_name"] == "separable_cmaes_dispatch_action"
    assert row["semantic_surface"] == "full_space_diagonal_separable_search_takeover"
    assert row["optimizer_consumed"] == "1"
    assert row["search_state_action_type"] == "separable_cmaes_dispatch_action"
    assert row["escape_budget"] == "16"
    assert row["population_after"] == "4"
    assert row["best_before"] == "1.000000e+02"
    assert float(row["best_after"]) <= 100.0
    assert row["bipop_restart_mode"] == "phase_i_warm_started_direct_full_space_diagonal_separable_cmaes"
    budget_summary = tmp_path / "R5_budget_summary.csv"
    assert budget_summary.exists()
    with budget_summary.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["optimizer_reported_fe"] == "20"
    assert summary_rows[0]["fitness_record_fe"] == "20"
    assert summary_rows[0]["same_budget_violation"] == "0"


def test_run_problem_repair_bipop_combines_repair_overlap_with_guarded_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            is_escape = self.options.get("arac_search_state_action") == "bipop_search_state_restart"
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 900.0 if is_escape else 1000.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 3.0),
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 5, "overlap_degree": 1, "subgroups": [2, 2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    _record, _elapsed, trace_rows = runner.run_problem(
        "ackley",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=500,
            seed=1,
            verbose=0,
            arac_action="repair_bipop_search_state_restart",
        ),
    )

    restart_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "repair_bipop_search_state_restart"
    ]
    assert restart_rows
    assert restart_rows[0]["search_state_action_type"] == "bipop_restart"
    assert restart_rows[0]["restart_accepted"] == "1"
    repair_overlap_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "repair_shared_variable_binding"
    ]
    assert repair_overlap_rows
    assert all(row["search_state_action_type"] == "" for row in repair_overlap_rows)


def test_run_problem_repair_protect_refine_uses_repair_overlap_and_smaller_sigma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    sigmas_seen: list[float] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            sigmas_seen.append(float(self.options["sigma"]))
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 900.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 3.0),
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 5, "overlap_degree": 1, "subgroups": [2, 2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    _record, _elapsed, trace_rows = runner.run_problem(
        "ackley",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=64,
            seed=1,
            sigma=0.5,
            verbose=0,
            arac_action="repair_protect_refine",
        ),
    )

    assert sigmas_seen
    assert all(sigma == pytest.approx(0.25) for sigma in sigmas_seen)
    repair_overlap_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "repair_shared_variable_binding"
    ]
    assert repair_overlap_rows


def test_run_problem_repair_protect_deep_refine_uses_deeper_sigma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    sigmas_seen: list[float] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            sigmas_seen.append(float(self.options["sigma"]))
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 900.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 3.0),
                "mean": np.full(self.problem["ndim_problem"], 2.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 5, "overlap_degree": 1, "subgroups": [2, 2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    _record, _elapsed, trace_rows = runner.run_problem(
        "ackley",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=64,
            seed=1,
            sigma=0.5,
            verbose=0,
            arac_action="repair_protect_deep_refine",
        ),
    )

    assert sigmas_seen
    assert all(sigma == pytest.approx(0.125) for sigma in sigmas_seen)
    repair_overlap_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == "repair_shared_variable_binding"
    ]
    assert repair_overlap_rows


def test_run_problem_trajectory_mean_cache_only_accepts_improving_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    means_seen: list[np.ndarray] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 3, "lower": -100.0, "upper": 100.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            means_seen.append(np.asarray(self.options["mean"][0], dtype=float).copy())
            budget = self.options["max_function_evaluations"]
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 1100.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 7.0),
                "mean": np.full(self.problem["ndim_problem"], 20.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id, data_root=None: [[0, 1], [1, 2]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1]], [[1]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 3, "overlap_degree": 1, "subgroups": [2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=16,
            seed=1,
            verbose=0,
            arac_action="budget_shift_mean_blend",
        ),
    )

    np.testing.assert_allclose(means_seen[1], np.array([0.0, 0.0]))


def test_run_problem_mean_blend_only_keeps_uniform_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    budgets_seen: list[int] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 4, "lower": -100.0, "upper": 100.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            budget = self.options["max_function_evaluations"]
            budgets_seen.append(budget)
            self.problem["fitness_function"](np.zeros((budget, self.problem["ndim_problem"])))
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 900.0,
                "best_so_far_x": np.zeros(self.problem["ndim_problem"]),
                "mean": np.zeros(self.problem["ndim_problem"]),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2]], [[1], [2]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 4, "overlap_degree": 1, "subgroups": [2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=90,
            seed=1,
            verbose=0,
            arac_action="mean_blend_only",
        ),
    )

    assert max(budgets_seen[:3]) - min(budgets_seen[:3]) <= 1


def test_run_problem_budget_shift_only_does_not_blend_mean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    means_seen: list[np.ndarray] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 4, "lower": -100.0, "upper": 100.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            means_seen.append(np.asarray(self.options["mean"][0], dtype=float).copy())
            budget = self.options["max_function_evaluations"]
            self.problem["fitness_function"](np.zeros((budget, self.problem["ndim_problem"])))
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 900.0,
                "best_so_far_x": np.full(self.problem["ndim_problem"], 8.0),
                "mean": np.full(self.problem["ndim_problem"], 20.0),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2]], [[1], [2]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 4, "overlap_degree": 1, "subgroups": [2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=90,
            seed=1,
            verbose=0,
            arac_action="budget_shift_only",
        ),
    )

    np.testing.assert_allclose(means_seen[1], np.array([8.0, 0.0]))


def test_run_problem_source_budget_accounting_matches_hcc_reported_fes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    budgets_seen: list[int] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if vector.ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, fun_name: str, fun_id: int):
            return FakeFunction()

        def get_info(self, fun_name: str, fun_id: int):
            return {"dimension": 4, "lower": -5.0, "upper": 5.0}

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            budget = self.options["max_function_evaluations"]
            budgets_seen.append(budget)
            x_batch = np.zeros((budget, self.problem["ndim_problem"]))
            self.problem["fitness_function"](x_batch)
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 1000.0,
                "best_so_far_x": x_batch[0],
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(runner, "decompose_problem", lambda fun_id, data_root=None: [[0, 1], [1, 2], [2, 3]])
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2]], [[1], [2]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda fun_id, data_root=None: {"dimension": 4, "overlap_degree": 1, "subgroups": [2, 2, 2]},
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda max_fes, degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda dimension: 4)

    record, _elapsed, _trace_rows = runner.run_problem(
        "elliptic",
        1,
        tmp_path,
        runner.SmokeConfig(
            max_fes=20,
            seed=1,
            verbose=0,
            budget_accounting="source",
        ),
    )

    assert budgets_seen == [7, 7, 7]
    assert len(record) == 24
    summary_path = tmp_path / "E1_budget_summary.csv"
    assert summary_path.exists()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert summary_rows == [
        {
            "problem_id": "E1",
            "budget_accounting": "source",
            "max_fes": "20",
            "optimizer_reported_fe": "21",
            "fitness_record_fe": "24",
            "budget_aligned_fe": "20",
            "same_budget_violation": "1",
            "global_phase_fe": "0",
            "cc_phase_fe": "21",
            "rescue_fe": "0",
            "refresh_fe": "0",
            "separable_continuation_fe": "0",
            "overhead_fe": "3",
        }
    ]


def test_main_preserves_case_level_action_traces_for_multiple_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    def fake_run_problem(fun_name, fun_id, output_path, config):
        problem_id = runner._problem_id(fun_name, fun_id)
        relation = runner.OverlapRelation(
            relation_id=f"O0_{fun_id - 1}_{fun_id}",
            problem_id=problem_id,
            outer_iter=0,
            group_left=fun_id - 1,
            group_right=fun_id,
            shared_vars=(fun_id,),
            overlap_strength=1.0,
            delta_signal=0.1,
            rank_signal=0.9,
            budget_remaining_ratio=0.8,
        )
        action = runner.RelationActionDecision(
            relation_id=relation.relation_id,
            action_name="fallback",
            action_family="fallback",
            confidence=0.0,
            trigger_reason="test",
        )
        runner._write_action_decision_log(
            runner.case_artifact_path(output_path, problem_id, "action_decision.csv"),
            config.run_id,
            [relation],
            [action],
        )
        rows = [
            runner.build_action_trace_row(
                problem_id=problem_id,
                seed=config.seed,
                outer_iter=0,
                group_index=1,
                selected_action_name="conservative_no_action",
                overlap_size=1,
                previous_delta=1.0,
                current_delta=1.0,
            )
        ]
        return [1.0], 0.0, rows

    monkeypatch.setattr(runner, "run_problem", fake_run_problem)
    monkeypatch.setattr(runner, "evaluation_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "plot_evaluation_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "plot_evaluation_curve_best_so_far",
        lambda *args, **kwargs: None,
    )

    runner.main(
        [
            "--functions",
            "elliptic",
            "--ids",
            "1",
            "2",
            "--output-root",
            str(tmp_path),
            "--timestamp",
            "multi-case",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--enable-relation-dispatch",
        ]
    )

    output_path = tmp_path / "multi-case" / "elliptic"
    assert (output_path / "E1_action_trace.csv").exists()
    assert (output_path / "E2_action_trace.csv").exists()
    with (output_path / "action_trace.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["problem_id"] for row in rows] == ["E1", "E2"]
    with (output_path / "action_decision.csv").open(newline="", encoding="utf-8") as handle:
        decision_rows = list(csv.DictReader(handle))

    assert [row["problem_id"] for row in decision_rows] == ["E1", "E2"]
    assert (output_path / "E1_action_decision.csv").exists()
    assert (output_path / "E2_action_decision.csv").exists()


def test_terminal_relation_trace_is_not_downstream_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    def fake_run_problem(fun_name, fun_id, output_path, config):
        rows = [
            runner.build_action_trace_row(
                problem_id="E2",
                seed=config.seed,
                outer_iter=0,
                group_index=1,
                selected_action_name="repair_shared_variable_binding",
                overlap_size=1,
                previous_delta=1.0,
                current_delta=2.0,
                downstream_consumed=True,
            ),
            runner.build_action_trace_row(
                problem_id="E2",
                seed=config.seed,
                outer_iter=0,
                group_index=2,
                selected_action_name="repair_shared_variable_binding",
                overlap_size=1,
                previous_delta=2.0,
                current_delta=3.0,
                downstream_consumed=False,
            ),
        ]
        return [1.0], 0.0, rows

    monkeypatch.setattr(runner, "run_problem", fake_run_problem)
    monkeypatch.setattr(runner, "evaluation_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "plot_evaluation_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "plot_evaluation_curve_best_so_far",
        lambda *args, **kwargs: None,
    )

    runner.main(
        [
            "--functions",
            "elliptic",
            "--ids",
            "2",
            "--output-root",
            str(tmp_path),
            "--timestamp",
            "terminal-consumption",
            "--seed",
            "1",
            "--max-fes",
            "2000",
        ]
    )

    with (tmp_path / "terminal-consumption" / "elliptic" / "action_trace.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows[-1]["state_mutated"] == "1"
    assert rows[-1]["downstream_consumed"] == "0"
    assert rows[-1]["optimizer_consumed"] == "0"


@pytest.mark.integration
def test_conservative_fallback_matches_default_hcc_smoke_behavior(tmp_path: Path) -> None:
    if os.environ.get("ARAC_RUN_HCC_SMOKE") != "1":
        pytest.skip("set ARAC_RUN_HCC_SMOKE=1 to run the HCC subprocess smoke")

    from arac.backends.hcc import HccAobExecutionRequest, run_hcc_aob_smoke_execution

    python_executable = (
        r"C:\Users\83718\.cache\codex-runtimes\codex-primary-runtime\dependencies"
        r"\python\python.exe"
    )
    shared = {
        "problem_id": "E2",
        "seed": 1,
        "max_fes": 2_000,
        "hcc_root": Path("E:/HCC-main"),
        "python_executable": python_executable,
    }
    default_result = run_hcc_aob_smoke_execution(
        HccAobExecutionRequest(
            **shared,
            output_dir=(tmp_path / "default").resolve(),
            timestamp="fallback-equivalence-default",
        )
    )
    fallback_result = run_hcc_aob_smoke_execution(
        HccAobExecutionRequest(
            **shared,
            output_dir=(tmp_path / "fallback").resolve(),
            timestamp="fallback-equivalence-explicit",
            arac_action="conservative_no_action",
        )
    )

    assert default_result.status == "completed"
    assert fallback_result.status == "completed"
    assert fallback_result.final_error == pytest.approx(default_result.final_error)
    assert fallback_result.fe_used == default_result.fe_used
