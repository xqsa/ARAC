from __future__ import annotations

from pathlib import Path

from arac.backends.hcc import (
    HccAobExecutionRequest,
    HccAobExecutionResult,
    build_hcc_aob_smoke_command,
)
from arac.evidence import validate_runtime_payload


def test_hcc_aob_smoke_command_targets_hcc_main_subprocess(tmp_path: Path) -> None:
    request = HccAobExecutionRequest(
        problem_id="E1",
        seed=1,
        max_fes=2_000,
        output_dir=tmp_path / "hcc-smoke",
    )

    command = build_hcc_aob_smoke_command(request)

    assert command.cwd == Path("E:/HCC-main")
    assert command.argv[:2] == ("python", "HCC_SRC/arac_hcc_smoke_runner.py")
    assert "--functions" in command.argv
    assert "elliptic" in command.argv
    assert "--ids" in command.argv
    assert "1" in command.argv
    assert "--max-fes" in command.argv
    assert "2000" in command.argv
    assert "--seed" in command.argv
    assert "--output-root" in command.argv
    assert str(tmp_path / "hcc-smoke") in command.argv


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
