from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.pilots.exp_026_arac_vs_hcc_paired import run as exp026
from scripts import hcc_smoke_runner


def _config() -> dict[str, object]:
    return exp026.load_config(exp026.DEFAULT_CONFIG_PATH)


def _spec(case: str = "R4", seed: int = 117, output_root: Path = Path("unused")) -> exp026.RunSpec:
    return exp026.RunSpec("exp_026_arac_vs_hcc_paired", case, seed, exp026._expected_action(case), output_root)


def _write_valid_artifacts(
    spec: exp026.RunSpec,
    *,
    schema: str = exp026.PERSISTENT_ACTION_ARTIFACT_SCHEMA,
    action_hash: str = "a" * 64,
) -> None:
    spec.result_directory.mkdir(parents=True, exist_ok=True)
    (spec.result_directory / "run_summary.json").write_text(json.dumps({
        "protocol_version": exp026.RUN_SUMMARY_PROTOCOL_VERSION,
        "problem_id": spec.case,
        "seed": spec.seed,
        "configured_max_fes": exp026.EXACT_MAX_FES,
        "fitness_evaluations": exp026.EXACT_MAX_FES,
        "final_error": 12.5,
    }), encoding="utf-8")

    checkpoint_fe = 100_000
    action_start_fe = checkpoint_fe + 1
    common = {
        "schema_version": schema,
        "problem_id": spec.case,
        "run_seed": spec.seed,
        "configured_max_fes": exp026.EXACT_MAX_FES,
        "terminal_fe": exp026.EXACT_MAX_FES,
        "selected_action": spec.action,
        "selection_count": 1,
        "runtime_authorized": True,
        "runtime_consumed": True,
        "status": "completed",
        "action_set_hash": "1" * 64,
        "checkpoint_fe": checkpoint_fe,
        "checkpoint_hash": "2" * 64,
        "action_start_fe": action_start_fe,
        "action_hash": action_hash,
        "parameter_hash": "3" * 64,
        "lifecycle_state_hash": "4" * 64,
    }
    if spec.action == exp026.R_ACTION:
        action_budget_fes = 20_000
        action_completed_fe = checkpoint_fe + action_budget_fes
        artifact = {
            **common,
            "execution_mode": "one_native_sweep_burst_then_native",
            "action_completed_fe": action_completed_fe,
            "action_actual_fes": action_budget_fes,
            "action_budget_fes": action_budget_fes,
            "budget_source": "previous_complete_native_sweep_actual_fes",
            "budget_source_sweep": 3,
            "budget_source_actual_fes": action_budget_fes,
            "action_accepted": True,
            "candidate_fitness": 99.0,
            "native_resumed": True,
            "native_resume_start_fe": action_completed_fe + 1,
            "post_action_native_fes": exp026.EXACT_MAX_FES - action_completed_fe,
            "native_resume_sweeps_planned": 3,
            "native_resume_sweeps_completed": 3,
            "start_sweep": 4,
            "candidate_hash": "5" * 64,
            "post_incumbent_hash": "6" * 64,
            "action": {
                "action": spec.action,
                "problem_id": spec.case,
                "run_seed": spec.seed,
                "checkpoint_fe": checkpoint_fe,
                "budget_fes": action_budget_fes,
                "seed_namespace": exp026.R_ACTION,
                "issued_sweep": 3,
                "target_sweep": 4,
            },
            "lifecycle": {
                "action_hash": action_hash,
                "status": "completed",
                "consumed_fes": action_budget_fes,
                "started_fe": action_start_fe,
                "completed_fe": action_completed_fe,
                "state_hash": "4" * 64,
                "details": {
                    "action": exp026.R_ACTION,
                "action_hash": action_hash,
                "status": "completed",
                "consumed_fes": action_budget_fes,
                "started_fe": checkpoint_fe,
                "completed_fe": action_completed_fe,
            },
            },
        }
    else:
        action_actual_fes = exp026.EXACT_MAX_FES - checkpoint_fe
        artifact = {
            **common,
            "execution_mode": "persistent_budget_caps_until_terminal",
            "action_completed_fe": exp026.EXACT_MAX_FES,
            "action_actual_fes": action_actual_fes,
            "action_budget_fes": action_actual_fes,
            "budget_source": "phase1_frozen_efficiency_ewma",
            "native_resumed": False,
            "native_resume_start_fe": None,
            "post_action_native_fes": 0,
            "application_count": 1,
            "action": {
                "action": spec.action,
                "problem_id": spec.case,
                "run_seed": spec.seed,
                "checkpoint_fe": checkpoint_fe,
                "end_absolute_fe": exp026.EXACT_MAX_FES,
            },
            "lifecycle": {
                "action_hash": action_hash,
                "status": "completed",
                "consumed_fes": action_actual_fes,
                "started_fe": action_start_fe,
                "completed_fe": exp026.EXACT_MAX_FES,
                "state_hash": "4" * 64,
                "details": {
                    "action": exp026.S_ACTION,
                    "action_hash": action_hash,
                    "status": "completed",
                    "applications": [{"sweep_index": 4}],
                    "completed_fe": exp026.EXACT_MAX_FES,
                },
            },
        }
    (spec.result_directory / "persistent_phase2_action.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )


def _completed_results() -> list[dict[str, object]]:
    config = _config()
    results = []
    for index, spec in enumerate(exp026.build_run_matrix(config, Path("unused"))):
        results.append({"trajectory_id": spec.trajectory_id, "case": spec.case, "seed": spec.seed, "action": spec.action, "final_error": 100.0 + index, "ok": True, "status": "completed"})
    return results


def test_config_freezes_exact_persistent_phase2_cohort() -> None:
    config = _config()
    execution = config["execution"]
    assert isinstance(execution, dict)
    assert (
        exp026.PERSISTENT_ACTION_ARTIFACT_SCHEMA
        == hcc_smoke_runner.PERSISTENT_PHASE2_ARTIFACT_SCHEMA
    )
    assert (
        exp026.NATIVE_RESUME_SWEEPS
        == hcc_smoke_runner.PERSISTENT_SEP_CMA_NATIVE_RESUME_SWEEPS
    )
    assert config["protocol_version"] == "rs-phase2-action-validation-v2"
    assert config["config_schema_version"] == 6
    assert config["stage"] == "phase2_action_validation"
    assert tuple(execution["cases"]) == exp026.SUPPORTED_CASES
    assert tuple(execution["seeds"]) == exp026.VALIDATION_SEEDS
    assert execution["max_fes"] == 3_000_000
    assert execution["native_resume_sweeps"] == exp026.NATIVE_RESUME_SWEEPS
    assert execution["jobs"] == 20
    assert execution["runner_contract"]["evidence_overlay_mode"] == "paired_owner"
    assert "arm_a" not in execution and "arm_b" not in execution


@pytest.mark.parametrize("case", exp026.SUPPORTED_CASES)
def test_case_command_passes_the_real_runner_parser(case: str, tmp_path: Path) -> None:
    config = _config()
    spec = _spec(case, output_root=tmp_path)
    command = exp026.build_command(spec, config, "python")
    parsed = hcc_smoke_runner.parse_args(list(command[2:]))

    assert parsed.max_fes == exp026.EXACT_MAX_FES
    assert parsed.relation_policy == "persistent_phase2"
    assert parsed.persistent_phase2_action == spec.action
    assert parsed.evidence_overlay_mode == "paired_owner"
    assert parsed.enable_relation_dispatch is True


def test_config_rejects_old_or_relaxed_protocol(tmp_path: Path) -> None:
    payload = deepcopy(_config())
    payload["execution"]["max_fes"] = 300_000
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exact 3M"):
        exp026.load_config(path)


@pytest.mark.parametrize("case", ("R4", "S5"))
def test_artifact_gate_accepts_case_specific_v2_lifecycle(
    tmp_path: Path,
    case: str,
) -> None:
    spec = _spec(case, output_root=tmp_path)
    _write_valid_artifacts(spec)

    audited = exp026.read_trajectory_artifacts(spec)

    assert audited["fitness_evaluations"] == exp026.EXACT_MAX_FES
    assert audited["action"] == exp026._expected_action(case)
    assert audited["action_hash"] == "a" * 64


@pytest.mark.parametrize(
    "schema",
    ("persistent-phase2-action-v1", "phase2-action-v1"),
)
def test_artifact_gate_rejects_old_schema(tmp_path: Path, schema: str) -> None:
    spec = _spec(output_root=tmp_path)
    _write_valid_artifacts(spec, schema=schema)

    with pytest.raises(ValueError, match="schema_version"):
        exp026.read_trajectory_artifacts(spec)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("execution_mode", "persistent_budget_caps_until_terminal", "execution_mode"),
        ("native_resumed", False, "resume native"),
        ("candidate_hash", "invalid", "candidate_hash"),
        ("native_resume_sweeps_completed", 2, "must be 3"),
    ],
)
def test_r_artifact_gate_rejects_burst_semantic_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    spec = _spec("R4", output_root=tmp_path)
    _write_valid_artifacts(spec)
    path = spec.result_directory / "persistent_phase2_action.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact[field] = value
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        exp026.read_trajectory_artifacts(spec)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("execution_mode", "one_native_sweep_burst_then_native", "execution_mode"),
        ("action_completed_fe", exp026.EXACT_MAX_FES - 1, "terminal FE"),
        ("action_actual_fes", 1000, "actual FE accounting"),
        ("native_resumed", True, "must not use a native resume"),
    ],
)
def test_s_artifact_gate_rejects_persistent_semantic_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    spec = _spec("S5", output_root=tmp_path)
    _write_valid_artifacts(spec)
    path = spec.result_directory / "persistent_phase2_action.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact[field] = value
    if field == "action_completed_fe":
        artifact["lifecycle"]["completed_fe"] = value
    elif field == "action_actual_fes":
        artifact["lifecycle"]["consumed_fes"] = value
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        exp026.read_trajectory_artifacts(spec)


def test_artifact_gate_rejects_duplicate_or_hash_mismatched_action(tmp_path: Path) -> None:
    spec = _spec(output_root=tmp_path)
    _write_valid_artifacts(spec)
    path = spec.result_directory / "persistent_phase2_action.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection_count"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selection_count"):
        exp026.read_trajectory_artifacts(spec)

    payload["selection_count"] = 1
    payload["lifecycle"]["action_hash"] = "b" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lifecycle hash"):
        exp026.read_trajectory_artifacts(spec)


def test_resume_reuses_trajectory_only_after_strict_artifact_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(output_root=tmp_path)
    _write_valid_artifacts(spec)

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("valid resume artifact must not launch the runner")

    monkeypatch.setattr(exp026.subprocess, "run", forbidden_subprocess)

    result = exp026._run_one_resumable(spec, _config(), "python")

    assert result["ok"] is True
    assert result["execution_source"] == "reused_valid_artifact"


def test_resume_reruns_artifact_that_fails_strict_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(output_root=tmp_path)
    _write_valid_artifacts(spec, schema="persistent-phase2-action-v1")
    calls = []

    def fake_subprocess(command: object, **_kwargs: object) -> object:
        calls.append(command)
        _write_valid_artifacts(spec)
        return exp026.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(exp026.subprocess, "run", fake_subprocess)

    result = exp026._run_one_resumable(spec, _config(), "python")

    assert len(calls) == 1
    assert result["ok"] is True
    assert result["execution_source"] == "rerun_after_artifact_gate_failure"
    assert "schema_version" in result["resume_gate_error"]


def test_resume_reruns_missing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(output_root=tmp_path)
    calls = []

    def fake_subprocess(command: object, **_kwargs: object) -> object:
        calls.append(command)
        _write_valid_artifacts(spec)
        return exp026.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(exp026.subprocess, "run", fake_subprocess)

    result = exp026._run_one_resumable(spec, _config(), "python")

    assert len(calls) == 1
    assert result["ok"] is True
    assert result["execution_source"] == "rerun_after_artifact_gate_failure"
    assert "missing at exact path" in result["resume_gate_error"]


def test_runner_output_streams_to_bounded_merged_log_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(output_root=tmp_path)
    output = ("stdout\n" + "x" * 2500 + "\nstderr\n").encode()

    def fake_subprocess(command: object, **kwargs: object) -> object:
        assert "capture_output" not in kwargs
        assert kwargs["stderr"] is exp026.subprocess.STDOUT
        log = kwargs["stdout"]
        log.write(output)  # type: ignore[union-attr]
        return exp026.subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(exp026.subprocess, "run", fake_subprocess)

    result = exp026.run_one(spec, _config(), "python")

    log_path = spec.run_directory / "runner.log"
    assert log_path.read_bytes() == output
    assert result["runner_log_path"] == str(log_path)
    assert len(result["stderr_tail"]) == 2000
    assert result["stderr_tail"].endswith("stderr\n")


def test_reuse_existing_is_offline_validation_without_filesystem_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(output_root=tmp_path)

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("offline validation must not launch the runner")

    monkeypatch.setattr(exp026.subprocess, "run", forbidden_subprocess)

    result = exp026.run_one(spec, _config(), "python", run_subprocess=False)

    assert result["ok"] is False
    assert result["execution_source"] == "offline_validation"
    assert not spec.run_directory.exists()


def test_run_experiment_reports_each_future_as_it_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [_spec("R4", 117, tmp_path), _spec("S5", 118, tmp_path)]
    observed: list[tuple[str, int]] = []

    monkeypatch.setattr(exp026, "build_run_matrix", lambda _config, _root: specs)

    def fake_run_one(
        spec: exp026.RunSpec,
        _config: object,
        _python: str,
        *,
        run_subprocess: bool,
    ) -> dict[str, object]:
        assert run_subprocess is True
        return {
            "case": spec.case,
            "seed": spec.seed,
            "ok": False,
            "status": "synthetic_test_result",
            "execution_source": "fresh_execution",
        }

    monkeypatch.setattr(exp026, "run_one", fake_run_one)

    exp026.run_experiment(
        output_root=tmp_path,
        jobs=2,
        progress_callback=lambda row: observed.append((str(row["case"]), int(row["seed"]))),
    )

    assert set(observed) == {("R4", 117), ("S5", 118)}


def test_resume_and_reuse_existing_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        exp026.run_experiment(reuse_existing=True, resume=True)


def test_case_summary_is_five_seed_descriptive_with_paper_ratios() -> None:
    summaries = exp026.build_case_summaries(_completed_results(), _config())
    r2 = summaries[0]

    assert len(summaries) == 10
    assert r2["seed_count"] == 5
    assert r2["sample_std_error"] > 0.0
    assert len(r2["bootstrap_mean_95_ci"]) == 2
    assert r2["observed_to_paper_bold_mean_ratio"] == pytest.approx(102.0 / 248000.0)
    assert "descriptive" in r2["comparison_note"].lower()


def test_summary_fails_closed_on_missing_seed() -> None:
    results = _completed_results()
    results.pop()
    with pytest.raises(ValueError, match="exactly 50"):
        exp026.build_case_summaries(results, _config())
