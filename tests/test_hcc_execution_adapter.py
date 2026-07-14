from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from arac.backends import hcc as hcc_backend
from arac.backends.hcc import (
    HccAobExecutionRequest,
    HccAobExecutionResult,
    build_hcc_aob_smoke_command,
)
from arac.evidence import validate_runtime_payload


def test_hcc_aob_smoke_command_targets_canonical_vendor_subprocess(tmp_path: Path) -> None:
    request = HccAobExecutionRequest(
        problem_id="E1",
        seed=1,
        max_fes=2_000,
        output_dir=tmp_path / "hcc-smoke",
    )

    command = build_hcc_aob_smoke_command(request)

    assert command.cwd == hcc_backend.HCC_VENDOR_ROOT
    assert command.argv[0] == "python"
    assert Path(command.argv[1]) == hcc_backend.ARAC_HCC_SMOKE_RUNNER
    assert Path(command.argv[1]).name == "hcc_smoke_runner.py"
    assert Path(command.argv[1]).is_absolute()
    assert command.cwd.is_absolute()
    assert "--functions" in command.argv
    assert "elliptic" in command.argv
    assert "--ids" in command.argv
    assert "1" in command.argv
    assert "--max-fes" in command.argv
    assert "2000" in command.argv
    assert "--seed" in command.argv
    assert "--output-root" in command.argv
    assert str(tmp_path / "hcc-smoke") in command.argv


def test_hcc_aob_smoke_command_passes_explicit_aob_data_root(tmp_path: Path) -> None:
    data_root = hcc_backend.HCC_VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
    request = HccAobExecutionRequest(
        problem_id="E6",
        seed=3,
        max_fes=3_000_000,
        output_dir=tmp_path / "canonical-e6",
        aob_data_root=data_root,
    )

    command = build_hcc_aob_smoke_command(request)

    data_root_index = command.argv.index("--aob-data-root")
    assert command.argv[data_root_index + 1] == str(data_root.resolve())


def test_hcc_aob_smoke_command_is_independent_of_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = HccAobExecutionRequest(
        problem_id="E1",
        seed=1,
        max_fes=2_000,
        output_dir=tmp_path / "hcc-smoke",
    )

    monkeypatch.chdir(hcc_backend.ARAC_REPO_ROOT)
    from_repo_root = build_hcc_aob_smoke_command(request)
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    from_unrelated_cwd = build_hcc_aob_smoke_command(request)

    assert from_unrelated_cwd == from_repo_root
    assert from_repo_root.cwd == hcc_backend.HCC_VENDOR_ROOT


def test_hcc_execution_rejects_incomplete_aob_data_root_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess_called = False

    def fake_run(*_args, **_kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        raise AssertionError("subprocess must not run for an invalid AOB data root")

    monkeypatch.setattr(hcc_backend.subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError, match="AOB data root"):
        hcc_backend.run_hcc_aob_smoke_execution(
            HccAobExecutionRequest(
                problem_id="S6",
                seed=2,
                max_fes=3_000_000,
                output_dir=tmp_path / "invalid-data-root",
                aob_data_root=tmp_path / "missing-data",
            )
        )

    assert subprocess_called is False


def test_hcc_aob_smoke_command_passes_arac_action(tmp_path: Path) -> None:
    request = HccAobExecutionRequest(
        problem_id="E2",
        seed=1,
        max_fes=2_000,
        output_dir=tmp_path / "hcc-smoke",
        arac_action="repair_shared_variable_binding",
    )

    command = build_hcc_aob_smoke_command(request)

    action_arg_index = command.argv.index("--arac-action")
    assert command.argv[action_arg_index + 1] == "repair_shared_variable_binding"


def test_hcc_aob_smoke_command_passes_diagonal_search_state_backend(
    tmp_path: Path,
) -> None:
    command = build_hcc_aob_smoke_command(
        HccAobExecutionRequest(
            problem_id="R3",
            seed=3,
            max_fes=3_000_000,
            output_dir=tmp_path,
            search_state_backend="diagonal_cma",
        )
    )

    option_index = command.argv.index("--search-state-backend")
    assert command.argv[option_index + 1] == "diagonal_cma"


@pytest.mark.parametrize("mode", ["shuffled_graph", "paired_fallback"])
def test_hcc_aob_smoke_command_passes_car_candidate_control(
    tmp_path: Path,
    mode: str,
) -> None:
    command = build_hcc_aob_smoke_command(
        HccAobExecutionRequest(
            problem_id="E2",
            seed=3,
            max_fes=3_000_000,
            output_dir=tmp_path,
            car_candidate_mode=mode,
        )
    )

    option_index = command.argv.index("--car-candidate-mode")
    assert command.argv[option_index + 1] == mode


def test_hcc_aob_smoke_command_rejects_unknown_search_state_backend(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="search_state_backend"):
        build_hcc_aob_smoke_command(
            HccAobExecutionRequest(
                problem_id="R3",
                seed=3,
                max_fes=3_000_000,
                output_dir=tmp_path,
                search_state_backend="oracle_backend",
            )
        )


def test_hcc_aob_smoke_command_passes_relation_dispatch_options(tmp_path: Path) -> None:
    request = HccAobExecutionRequest(
        problem_id="E2",
        seed=1,
        max_fes=2_000,
        output_dir=tmp_path / "hcc-smoke",
        enable_relation_dispatch=True,
        relation_policy_mode="rule",
    )

    command = build_hcc_aob_smoke_command(request)

    assert "--enable-relation-dispatch" in command.argv
    policy_arg_index = command.argv.index("--relation-policy")
    assert command.argv[policy_arg_index + 1] == "rule"

    shuffled = build_hcc_aob_smoke_command(
        HccAobExecutionRequest(
            problem_id="E2",
            seed=1,
            max_fes=2_000,
            output_dir=tmp_path / "hcc-shuffled-smoke",
            enable_relation_dispatch=True,
            relation_policy_mode="shuffled",
        )
    )

    shuffled_policy_arg_index = shuffled.argv.index("--relation-policy")
    assert shuffled.argv[shuffled_policy_arg_index + 1] == "shuffled"


def test_hcc_aob_smoke_command_passes_budget_accounting_mode(tmp_path: Path) -> None:
    request = HccAobExecutionRequest(
        problem_id="S1",
        seed=1,
        max_fes=3_000_000,
        output_dir=tmp_path / "hcc-source-budget-smoke",
        budget_accounting="source",
    )

    command = build_hcc_aob_smoke_command(request)

    budget_arg_index = command.argv.index("--budget-accounting")
    assert command.argv[budget_arg_index + 1] == "source"


def test_hcc_aob_smoke_command_passes_restart_modes(tmp_path: Path) -> None:
    request = HccAobExecutionRequest(
        problem_id="S4",
        seed=1,
        max_fes=3_000_000,
        output_dir=tmp_path / "hcc-paper-fidelity-smoke",
        cmaes_restart=False,
        mmes_restart=False,
    )

    command = build_hcc_aob_smoke_command(request)

    assert "--no-cmaes-restart" in command.argv
    assert "--no-mmes-restart" in command.argv


def test_hcc_aob_smoke_command_passes_skip_plots(tmp_path: Path) -> None:
    request = HccAobExecutionRequest(
        problem_id="S4",
        seed=1,
        max_fes=3_000_000,
        output_dir=tmp_path / "hcc-fast-smoke",
        skip_plots=True,
    )

    command = build_hcc_aob_smoke_command(request)

    assert "--skip-plots" in command.argv


def test_hcc_execution_runner_passes_skip_plots_to_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[str, ...]] = {}

    def fake_run(argv, **_kwargs):
        captured["argv"] = tuple(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(hcc_backend.subprocess, "run", fake_run)
    monkeypatch.setattr(
        hcc_backend,
        "_parse_hcc_evaluation_record_with_optimizer_final_fe",
        lambda _output_dir, budget_limit: (0.0, budget_limit, budget_limit),
    )
    monkeypatch.setattr(hcc_backend, "_find_hcc_action_trace", lambda _output_dir: (None, 0))

    result = hcc_backend.run_hcc_aob_smoke_execution(
        HccAobExecutionRequest(
            problem_id="S4",
            seed=1,
            max_fes=3_000_000,
            output_dir=tmp_path / "hcc-fast-smoke",
            skip_plots=True,
        )
    )

    assert result.status == "completed"
    assert "--skip-plots" in captured["argv"]


def test_hcc_execution_normalizes_relative_output_once_for_all_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_output = Path("results") / "relative-hcc-smoke"
    expected_output = (hcc_backend.ARAC_REPO_ROOT / relative_output).resolve()
    observed: dict[str, Path] = {}

    def fake_run(argv, **_kwargs):
        output_index = argv.index("--output-root")
        observed["argv"] = Path(argv[output_index + 1])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def fake_parse(output_dir: Path, budget_limit: int):
        observed["evaluation"] = output_dir
        return 0.0, budget_limit, budget_limit

    def fake_trace(output_dir: Path):
        observed["trace"] = output_dir
        return output_dir / "action_trace.csv", 1

    def fake_budget(output_dir: Path):
        observed["budget"] = output_dir
        return {}

    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.setattr(hcc_backend.subprocess, "run", fake_run)
    monkeypatch.setattr(
        hcc_backend,
        "_parse_hcc_evaluation_record_with_optimizer_final_fe",
        fake_parse,
    )
    monkeypatch.setattr(hcc_backend, "_find_hcc_action_trace", fake_trace)
    monkeypatch.setattr(hcc_backend, "_parse_hcc_budget_summary", fake_budget)

    result = hcc_backend.run_hcc_aob_smoke_execution(
        HccAobExecutionRequest(
            problem_id="E1",
            seed=1,
            max_fes=2_000,
            output_dir=relative_output,
        )
    )

    assert observed == {
        "argv": expected_output,
        "evaluation": expected_output,
        "trace": expected_output,
        "budget": expected_output,
    }
    assert result.output_root == expected_output
    assert result.action_trace_path == expected_output / "action_trace.csv"


def test_hcc_aob_smoke_command_rejects_unsupported_action_file(tmp_path: Path) -> None:
    request = HccAobExecutionRequest(
        problem_id="E2",
        seed=1,
        max_fes=2_000,
        output_dir=tmp_path / "hcc-smoke",
        arac_action_file=tmp_path / "actions.csv",
    )

    with pytest.raises(ValueError, match="arac_action_file"):
        build_hcc_aob_smoke_command(request)


def test_hcc_budget_parser_reads_search_state_fe_and_legacy_defaults_to_zero(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "R3_budget_summary.csv"
    summary.write_text(
        "problem_id,budget_accounting,max_fes,optimizer_reported_fe,"
        "fitness_record_fe,budget_aligned_fe,same_budget_violation,global_phase_fe,"
        "cc_phase_fe,rescue_fe,refresh_fe,search_state_fe,separable_continuation_fe,"
        "overhead_fe\n"
        "R3,strict,3000000,3000000,3000000,3000000,0,1200000,1500000,0,0,30000,0,270000\n",
        encoding="utf-8",
    )

    parsed = hcc_backend._parse_hcc_budget_summary(tmp_path)

    assert parsed["search_state_fe"] == 30000

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "R3_budget_summary.csv").write_text(
        "problem_id,fitness_record_fe\nR3,100\n",
        encoding="utf-8",
    )
    legacy = hcc_backend._parse_hcc_budget_summary(legacy_dir)
    assert legacy["search_state_fe"] == 0


def test_hcc_execution_result_fields_are_offline_only() -> None:
    result = HccAobExecutionResult(
        problem_id="E1",
        seed=1,
        max_fes=2_000,
        final_error=123.456,
        fe_used=2_000,
        time_seconds=0.5,
        output_root=Path("results/hcc-smoke"),
        fresh_optimizer_execution=True,
        status="completed",
        result_source="hcc_subprocess_smoke_execution",
        action_trace_path=Path("results/hcc-smoke/action_trace.csv"),
        action_trace_rows=3,
    )

    runtime_payload = {
        "problem_id": result.problem_id,
        "seed": result.seed,
        "budget_limit": result.max_fes,
        "used_for_runtime": 1,
    }
    validate_runtime_payload(runtime_payload)

    offline_row = result.to_offline_row()
    assert offline_row["final_error"] == "1.234560e+02"
    assert offline_row["runtime_dispatch_allowed"] == "0"
    assert offline_row["fresh_optimizer_execution"] == "1"
    assert offline_row["action_trace_path"] == "results\\hcc-smoke\\action_trace.csv"
    assert offline_row["action_trace_rows"] == "3"
    assert offline_row["same_budget_violation"] == "0"
    assert offline_row["performance_claim_allowed"] == "0"


def test_hcc_execution_result_marks_over_budget_not_performance_claimable() -> None:
    result = HccAobExecutionResult(
        problem_id="E2",
        seed=1,
        max_fes=2_000,
        final_error=1.0,
        fe_used=2_128,
        time_seconds=0.5,
        output_root=Path("results/hcc-smoke"),
        fresh_optimizer_execution=True,
        status="completed",
        result_source="hcc_subprocess_smoke_execution",
    )

    offline_row = result.to_offline_row()

    assert offline_row["same_budget_violation"] == "1"
    assert offline_row["performance_claim_allowed"] == "0"


def test_hcc_execution_result_marks_optimizer_final_overrun() -> None:
    result = HccAobExecutionResult(
        problem_id="E2",
        seed=1,
        max_fes=2_000,
        final_error=1.0,
        fe_used=2_000,
        optimizer_final_fe_used=2_128,
        time_seconds=0.5,
        output_root=Path("results/hcc-smoke"),
        fresh_optimizer_execution=True,
        status="completed",
        result_source="hcc_subprocess_smoke_execution",
    )

    offline_row = result.to_offline_row()

    assert offline_row["fe_used"] == "2000"
    assert offline_row["optimizer_final_fe_used"] == "2128"
    assert offline_row["same_budget_violation"] == "1"
