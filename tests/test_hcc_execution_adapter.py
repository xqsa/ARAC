from __future__ import annotations

import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

from arac.backends import hcc as hcc_backend
from arac.backends.hcc import HccAobExecutionRequest, build_hcc_aob_smoke_command


def _request(tmp_path: Path, **overrides: object) -> HccAobExecutionRequest:
    values: dict[str, object] = {
        "problem_id": "E3",
        "seed": 117,
        "max_fes": 100_000,
        "output_dir": tmp_path / "hcc-run",
        "python_executable": "python-test",
        "timestamp": "exp018-test",
    }
    values.update(overrides)
    return HccAobExecutionRequest(**values)


def test_execution_request_exposes_only_exp_018_inputs() -> None:
    assert {field.name for field in fields(HccAobExecutionRequest)} == {
        "problem_id",
        "seed",
        "max_fes",
        "output_dir",
        "hcc_root",
        "aob_data_root",
        "python_executable",
        "timestamp",
        "evidence_overlay_mode",
    }


def test_hcc_command_freezes_v37_controller_and_runtime_profile(tmp_path: Path) -> None:
    command = build_hcc_aob_smoke_command(
        _request(tmp_path, evidence_overlay_mode="paired_owner")
    )

    assert command.cwd == hcc_backend.HCC_VENDOR_ROOT
    assert command.argv[0] == "python-test"
    assert Path(command.argv[1]) == hcc_backend.ARAC_HCC_SMOKE_RUNNER
    assert command.argv[command.argv.index("--functions") + 1] == "elliptic"
    assert command.argv[command.argv.index("--ids") + 1] == "3"
    assert command.argv[command.argv.index("--arac-action") + 1] == (
        "arac_evidence_action_controller_v37"
    )
    assert command.argv[command.argv.index("--relation-policy") + 1] == "controller_v31"
    assert command.argv[command.argv.index("--budget-accounting") + 1] == "strict"
    assert command.argv[command.argv.index("--search-state-backend") + 1] == (
        "phase_i_mmes"
    )
    assert command.argv[command.argv.index("--evidence-overlay-mode") + 1] == (
        "paired_owner"
    )
    assert "--enable-relation-dispatch" in command.argv
    assert "--skip-plots" in command.argv
    assert "--no-cmaes-restart" not in command.argv
    assert "--no-mmes-restart" not in command.argv


def test_off_mode_omits_only_overlay_option(tmp_path: Path) -> None:
    command = build_hcc_aob_smoke_command(_request(tmp_path))

    assert "--evidence-overlay-mode" not in command.argv
    assert "--enable-relation-dispatch" in command.argv
    assert "--relation-policy" in command.argv


def test_hcc_command_passes_explicit_aob_data_root(tmp_path: Path) -> None:
    command = build_hcc_aob_smoke_command(
        _request(tmp_path, aob_data_root=hcc_backend.DEFAULT_AOB_DATA_ROOT)
    )

    index = command.argv.index("--aob-data-root")
    assert command.argv[index + 1] == str(hcc_backend.DEFAULT_AOB_DATA_ROOT)


def test_hcc_command_is_independent_of_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    expected = build_hcc_aob_smoke_command(request)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    assert build_hcc_aob_smoke_command(request) == expected


def test_execution_rejects_incomplete_aob_data_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(hcc_backend.subprocess, "run", fake_run)
    with pytest.raises(FileNotFoundError, match="AOB data root"):
        hcc_backend.run_hcc_aob_smoke_execution(
            _request(tmp_path, problem_id="S5", aob_data_root=tmp_path / "missing")
        )
    assert called is False


@pytest.mark.parametrize(
    ("overrides", "error"),
    (
        ({"problem_id": "R3"}, "exp_018"),
        ({"problem_id": "E2"}, "exp_018"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"max_fes": 0}, "max_fes"),
        ({"evidence_overlay_mode": "unknown"}, "evidence_overlay_mode"),
    ),
)
def test_hcc_command_rejects_inputs_outside_frozen_protocol(
    tmp_path: Path,
    overrides: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        build_hcc_aob_smoke_command(_request(tmp_path, **overrides))


def test_execution_normalizes_output_once_for_command_and_parsers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_output = Path("results") / "relative-exp018"
    expected_output = (hcc_backend.ARAC_REPO_ROOT / relative_output).resolve()
    observed: dict[str, Path] = {}

    def fake_run(argv, **_kwargs):
        observed["command"] = Path(argv[argv.index("--output-root") + 1])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def fake_evaluation(output_dir: Path, budget_limit: int):
        observed["evaluation"] = output_dir
        return 1.0, budget_limit, budget_limit

    def fake_budget(output_dir: Path):
        observed["budget"] = output_dir
        return {
            "fitness_record_fe": 100_000,
            "global_phase_fe": 20_000,
            "cc_phase_fe": 79_984,
            "evidence_overlay_fe": 16,
            "overhead_fe": 0,
        }

    monkeypatch.setattr(hcc_backend.subprocess, "run", fake_run)
    monkeypatch.setattr(
        hcc_backend,
        "_parse_hcc_evaluation_record_with_optimizer_final_fe",
        fake_evaluation,
    )
    monkeypatch.setattr(hcc_backend, "_parse_hcc_budget_summary", fake_budget)

    result = hcc_backend.run_hcc_aob_smoke_execution(
        _request(tmp_path, output_dir=relative_output)
    )

    assert observed == {
        "command": expected_output,
        "evaluation": expected_output,
        "budget": expected_output,
    }
    assert result.output_root == expected_output
    assert result.status == "completed"
    assert result.optimizer_final_fe_used == 100_000


def test_budget_parser_keeps_exp_018_ledger_columns(tmp_path: Path) -> None:
    (tmp_path / "E3_budget_summary.csv").write_text(
        "fitness_record_fe,optimizer_reported_fe,global_phase_fe,cc_phase_fe,"
        "rescue_fe,refresh_fe,search_state_fe,precision_probe_fe,"
        "evidence_overlay_fe,separable_continuation_fe,overhead_fe\n"
        "100000,100000,20000,79984,0,0,0,0,16,0,0\n",
        encoding="utf-8",
    )

    parsed = hcc_backend._parse_hcc_budget_summary(tmp_path)

    assert parsed["fitness_record_fe"] == 100_000
    assert parsed["global_phase_fe"] == 20_000
    assert parsed["cc_phase_fe"] == 79_984
    assert parsed["evidence_overlay_fe"] == 16
    assert parsed["overhead_fe"] == 0
