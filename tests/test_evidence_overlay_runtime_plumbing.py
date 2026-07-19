from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from arac.backends import hcc as hcc_backend
from arac.backends.hcc import HccAobExecutionRequest, build_hcc_aob_smoke_command
from scripts import hcc_smoke_runner as runner


def _request(**overrides: object) -> HccAobExecutionRequest:
    values: dict[str, object] = {
        "problem_id": "E3",
        "seed": 117,
        "max_fes": 5_000,
        "output_dir": Path("results/evidence-overlay-command"),
    }
    values.update(overrides)
    return HccAobExecutionRequest(**values)


def _runner_args(*extra: str) -> list[str]:
    return [
        "--functions",
        "elliptic",
        "--ids",
        "3",
        "--output-root",
        "results/evidence-overlay-cli",
        "--max-fes",
        "5000",
        "--seed",
        "117",
        "--arac-action",
        "arac_evidence_action_controller_v37",
        "--enable-relation-dispatch",
        "--relation-policy",
        "controller_v31",
        *extra,
    ]


@pytest.mark.parametrize(
    ("total_fes", "degree_of_overlap", "expected"),
    (
        (3_000_000, 0.0, 600_000),
        (1_000, 0.4, 520),
        (1_000, 1e-12, 200),
    ),
)
def test_global_phase_reserves_at_least_twenty_percent(
    total_fes: int,
    degree_of_overlap: float,
    expected: int,
) -> None:
    assert runner.calculate_global_fes(total_fes, degree_of_overlap) == expected


@pytest.mark.parametrize(
    "mode",
    ("native_audit", "paired_owner", "shuffled_owner"),
)
def test_backend_forwards_only_explicit_overlay_modes(mode: str) -> None:
    enabled = build_hcc_aob_smoke_command(
        _request(evidence_overlay_mode=mode)
    )
    disabled = build_hcc_aob_smoke_command(_request())

    index = enabled.argv.index("--evidence-overlay-mode")
    assert enabled.argv[index + 1] == mode
    assert "--evidence-overlay-mode" not in disabled.argv


def test_backend_rejects_unknown_overlay_mode() -> None:
    with pytest.raises(ValueError, match="evidence_overlay_mode"):
        build_hcc_aob_smoke_command(
            _request(evidence_overlay_mode="unknown")
        )


@pytest.mark.parametrize(
    "mode",
    ("native_audit", "paired_owner", "shuffled_owner"),
)
def test_runner_cli_accepts_only_frozen_overlay_profile(mode: str) -> None:
    parsed = runner.parse_args(
        _runner_args("--evidence-overlay-mode", mode)
    )

    assert parsed.evidence_overlay_mode == mode
    assert runner.parse_args(_runner_args()).evidence_overlay_mode == "off"


def test_runner_cli_requires_explicit_seed_for_overlay() -> None:
    args = _runner_args("--evidence-overlay-mode", "native_audit")
    seed_index = args.index("--seed")
    del args[seed_index : seed_index + 2]

    with pytest.raises(SystemExit):
        runner.parse_args(args)


@pytest.mark.parametrize(
    ("mode", "sweeps", "closed", "attempted", "complete", "expected"),
    (
        ("off", 3, True, False, True, False),
        ("native_audit", 2, True, False, True, False),
        ("native_audit", 3, False, False, True, False),
        ("native_audit", 3, True, True, True, False),
        ("native_audit", 3, True, False, False, False),
        ("native_audit", 3, True, False, True, True),
        ("paired_owner", 4, True, False, True, True),
        ("shuffled_owner", 3, True, False, True, True),
    ),
)
def test_overlay_barrier_requires_three_closed_complete_sweeps(
    mode: str,
    sweeps: int,
    closed: bool,
    attempted: bool,
    complete: bool,
    expected: bool,
) -> None:
    assert runner.evidence_overlay_sweep_barrier_ready(
        mode=mode,
        complete_sweep_count=sweeps,
        previous_survival_closed=closed,
        barrier_attempted=attempted,
        all_raw_groups_completed=complete,
    ) is expected


def test_overlay_barrier_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="unsupported evidence overlay mode"):
        runner.evidence_overlay_sweep_barrier_ready(
            mode="unknown",
            complete_sweep_count=3,
            previous_survival_closed=True,
            barrier_attempted=False,
            all_raw_groups_completed=True,
        )
    with pytest.raises(ValueError, match="complete_sweep_count"):
        runner.evidence_overlay_sweep_barrier_ready(
            mode="native_audit",
            complete_sweep_count=-1,
            previous_survival_closed=True,
            barrier_attempted=False,
            all_raw_groups_completed=True,
        )


def test_budget_summary_keeps_off_schema_and_records_active_zero_fe(
    tmp_path: Path,
) -> None:
    off_path = tmp_path / "off.csv"
    audit_path = tmp_path / "audit.csv"
    paired_path = tmp_path / "paired.csv"
    shared = {
        "problem_id": "E3",
        "budget_accounting": "strict",
        "max_fes": 100,
        "optimizer_reported_fe": 100,
        "fitness_record_fe": 100,
        "global_phase_fe": 20,
        "cc_phase_fe": 64,
    }
    runner._write_budget_summary(off_path, **shared)
    runner._write_budget_summary(audit_path, **shared, evidence_overlay_fe=0)
    runner._write_budget_summary(paired_path, **shared, evidence_overlay_fe=16)

    with off_path.open(newline="", encoding="utf-8") as handle:
        off = list(csv.DictReader(handle))[0]
    with audit_path.open(newline="", encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))[0]
    with paired_path.open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))[0]

    assert "evidence_overlay_fe" not in off
    assert audit["evidence_overlay_fe"] == "0"
    assert audit["overhead_fe"] == "16"
    assert paired["evidence_overlay_fe"] == "16"
    assert paired["overhead_fe"] == "0"


def test_budget_parser_reads_overlay_fe_and_defaults_legacy_to_zero(
    tmp_path: Path,
) -> None:
    (tmp_path / "E3_budget_summary.csv").write_text(
        "problem_id,fitness_record_fe,evidence_overlay_fe\nE3,100,32\n",
        encoding="utf-8",
    )
    parsed = hcc_backend._parse_hcc_budget_summary(tmp_path)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "E3_budget_summary.csv").write_text(
        "problem_id,fitness_record_fe\nE3,100\n",
        encoding="utf-8",
    )

    assert parsed["evidence_overlay_fe"] == 32
    assert hcc_backend._parse_hcc_budget_summary(legacy)["evidence_overlay_fe"] == 0


def test_execution_result_parses_overlay_manifest_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "active-overlay"
    manifest_path = output / "nested" / "E3_evidence_overlay_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "barrier_status": "probed",
                "native_terminal_error": 2.5,
                "all_evaluation_best_error": 1.25,
            }
        ),
        encoding="utf-8",
    )
    subprocess_kwargs: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        subprocess_kwargs.update(kwargs)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(hcc_backend.subprocess, "run", fake_run)
    monkeypatch.setattr(
        hcc_backend,
        "_parse_hcc_evaluation_record_with_optimizer_final_fe",
        lambda _output_dir, budget_limit: (1.25, budget_limit, budget_limit),
    )
    monkeypatch.setattr(
        hcc_backend,
        "_parse_hcc_budget_summary",
        lambda _output_dir: {"evidence_overlay_fe": 16},
    )

    result = hcc_backend.run_hcc_aob_smoke_execution(
        _request(
            output_dir=output,
            evidence_overlay_mode="paired_owner",
        )
    )

    assert result.status == "completed"
    assert result.native_terminal_error == 2.5
    assert result.all_evaluation_best_error == 1.25
    assert result.evidence_overlay_manifest_path == manifest_path
    environment = subprocess_kwargs["env"]
    assert isinstance(environment, dict)
    assert {
        key: environment[key]
        for key in hcc_backend.EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT
    } == hcc_backend.EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT


def test_off_execution_does_not_override_subprocess_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(hcc_backend.subprocess, "run", fake_run)
    hcc_backend.run_hcc_aob_smoke_execution(
        _request(output_dir=tmp_path / "off")
    )

    assert observed["env"] is None


def test_failed_overlay_subprocess_preserves_written_fe_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed-overlay"
    artifact_dir = output / "nested"
    artifact_dir.mkdir(parents=True)
    manifest_path = artifact_dir / "E3_evidence_overlay_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "barrier_status": "failed",
                "native_terminal_error": 2.5,
                "all_evaluation_best_error": 1.25,
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "E3_budget_summary.csv").write_text(
        "fitness_record_fe,optimizer_reported_fe,global_phase_fe,cc_phase_fe,"
        "rescue_fe,refresh_fe,search_state_fe,precision_probe_fe,"
        "evidence_overlay_fe,separable_continuation_fe,overhead_fe\n"
        "105,105,20,64,0,0,0,0,5,0,16\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hcc_backend.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            7,
            stdout="",
            stderr="synthetic fail-closed trajectory",
        ),
    )

    result = hcc_backend.run_hcc_aob_smoke_execution(
        _request(
            output_dir=output,
            evidence_overlay_mode="paired_owner",
        )
    )

    assert result.status == "failed_returncode_7"
    assert result.fresh_optimizer_execution
    assert result.fe_used == 105
    assert result.optimizer_final_fe_used == 105
    assert result.evidence_overlay_fe == 5
    assert result.overhead_fe == 16
    assert result.native_terminal_error == 2.5
    assert result.all_evaluation_best_error == 1.25
    assert result.final_error == 1.25
    assert result.evidence_overlay_manifest_path == manifest_path
