from __future__ import annotations

import argparse
import csv
from pathlib import Path
from random import Random

import pytest

from arac.actions.budget_reallocation import (
    FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
)
from arac.actions.shrunk_budget_pulse import (
    SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
    CATASTROPHIC_DELTA,
    S_FAMILY_BUDGET_PULSE_ARMS,
)
from experiments.pilots.exp_019_conflict_resolution_pilot import _diagnostic_worker
from experiments.pilots.exp_031_s_family_budget_pulse_diagnostic import run as exp031


def _arm_rows(delta_factory) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in exp031.SEEDS:
        for context_index in range(exp031.EXPECTED_CONTEXTS_PER_SEED):
            context_id = f"S5-seed{seed}-context{context_index}"
            deltas = delta_factory(seed, context_index)
            assert set(deltas) == set(S_FAMILY_BUDGET_PULSE_ARMS)
            for arm in S_FAMILY_BUDGET_PULSE_ARMS:
                rows.append(
                    {
                        "seed": str(seed),
                        "context_id": context_id,
                        "arm": arm,
                        "horizon": "sweep_1",
                        "delta": f"{deltas[arm]:.17e}",
                    }
                )
    return rows


def _deltas(
    *,
    native: float = 0.0,
    true_no_writeback: float = -0.01,
    raw: float = -0.01,
    shrunk: float = -0.01,
) -> dict[str, float]:
    return {
        "native_eq8": native,
        "true_no_writeback": true_no_writeback,
        FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION: raw,
        SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION: shrunk,
    }


def test_config_specs_and_worker_command_freeze_five_seed_matrix(tmp_path: Path) -> None:
    config = exp031.load_config()
    specs = exp031.build_specs(tmp_path)

    assert config["protocol_version"] == exp031.PROTOCOL_VERSION
    assert tuple(config["execution"]["seeds"]) == exp031.SEEDS
    assert len(specs) == 5
    assert exp031.EXPECTED_TOTAL_CONTEXTS == 20
    assert exp031.EXPECTED_TOTAL_ARM_ROWS == 240
    command = exp031.build_worker_command(specs[1], "python")
    assert command[command.index("--seed") + 1] == "118"
    assert command[command.index("--max-fes") + 1] == "300000"
    assert command[-2:] == ("--profile", "s_family_budget_pulse")


def test_conditional_pulse_headroom_advances_only_to_real_aob_pilot() -> None:
    rows = _arm_rows(
        lambda _seed, context: _deltas(
            raw=0.03 if context < 2 else -0.04,
            shrunk=-0.04 if context < 2 else 0.03,
        )
    )

    summary = exp031.summarize_diagnostic(rows, bootstrap_replicates=100)

    assert summary["runtime_eligible_budget_pulse_vbs"]["delta_lcb"] > 0.0
    assert summary["runtime_eligible_budget_sbs"]["arm"] == "native_eq8"
    assert summary["primary_recommendation"] == (
        "advance_budget_pulse_portfolio_to_real_aob_pilot"
    )
    assert summary["s5_action_diagnostic_passed"] == 1
    assert summary["real_aob_pilot_authorized"] == 1
    assert summary["action_gate_authorized"] == 0
    assert summary["evidence_separability_authorized"] == 0
    assert summary["selector_authorized"] == 0
    assert summary["runtime_authorized"] == 0


def test_positive_fixed_budget_sbs_prioritizes_fixed_real_aob_pilot() -> None:
    rows = _arm_rows(lambda _seed, _context: _deltas(raw=0.02, shrunk=-0.01))

    summary = exp031.summarize_diagnostic(rows, bootstrap_replicates=100)

    sbs = summary["runtime_eligible_budget_sbs"]
    assert sbs["arm"] == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    assert sbs["delta_lcb"] > 0.0
    assert summary["primary_recommendation"] == (
        "advance_fixed_budget_pulse_to_real_aob_pilot"
    )
    assert summary["fixed_action_validation_target"] == (
        FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    )
    assert summary["fixed_action_real_aob_pilot_authorized"] == 1
    assert summary["fixed_action_validation_authorized"] == 0
    assert summary["evidence_separability_authorized"] == 0


def test_significant_lower_mean_fixed_arm_preempts_portfolio_research() -> None:
    per_seed_raw = {
        117: -0.10,
        118: -0.10,
        119: 0.20,
        120: 0.20,
        121: 0.20,
    }
    rows = _arm_rows(
        lambda seed, _context: _deltas(raw=per_seed_raw[seed], shrunk=0.02)
    )

    summary = exp031.summarize_diagnostic(rows, bootstrap_replicates=2_000)

    assert summary["runtime_eligible_budget_sbs"]["arm"] == (
        FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    )
    assert summary["runtime_eligible_budget_sbs"]["delta_lcb"] <= 0.0
    assert summary["arm_statistics"][SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION][
        "delta_lcb"
    ] > 0.0
    assert summary["positive_fixed_budget_arms"] == [
        SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
    ]
    assert summary["fixed_action_validation_target"] == (
        SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
    )
    assert summary["primary_recommendation"] == (
        "advance_fixed_budget_pulse_to_real_aob_pilot"
    )


def test_true_no_writeback_headroom_does_not_validate_budget_pulse() -> None:
    rows = _arm_rows(
        lambda _seed, _context: _deltas(
            true_no_writeback=0.03,
            raw=-0.01,
            shrunk=-0.02,
        )
    )

    summary = exp031.summarize_diagnostic(rows, bootstrap_replicates=100)

    assert summary["registered_vbs"]["mean_delta"] == pytest.approx(0.03)
    assert summary["registered_sbs"]["arm"] == "true_no_writeback"
    assert summary["runtime_eligible_budget_pulse_vbs"]["mean_delta"] == 0.0
    assert summary["control_headroom_only"] is True
    assert summary["primary_recommendation"] == "redesign_budget_pulse_actions"
    assert summary["action_gate_authorized"] == 0


def test_uncertain_s5_headroom_collects_more_contexts_without_opening_gates() -> None:
    first_seed = exp031.SEEDS[0]
    rows = _arm_rows(
        lambda seed, _context: _deltas(raw=0.03 if seed == first_seed else -0.01)
    )

    summary = exp031.summarize_diagnostic(rows, bootstrap_replicates=2_000)

    assert summary["runtime_eligible_budget_pulse_vbs"]["mean_delta"] > 0.0
    assert summary["runtime_eligible_budget_pulse_vbs"]["delta_lcb"] == 0.0
    assert summary["primary_recommendation"] == "collect_more_s5_ceiling_contexts"
    assert summary["real_aob_pilot_authorized"] == 0
    assert summary["action_gate_authorized"] == 0
    assert summary["evidence_separability_authorized"] == 0


def test_sparse_s5_headroom_preempts_positive_fixed_arm() -> None:
    first_seed = exp031.SEEDS[0]
    rows = _arm_rows(
        lambda seed, context: _deltas(
            raw=0.02 if (seed, context) == (first_seed, 0) else 0.005
        )
    )

    summary = exp031.summarize_diagnostic(rows, bootstrap_replicates=2_000)

    vbs = summary["runtime_eligible_budget_pulse_vbs"]
    assert vbs["delta_lcb"] > 0.0
    assert vbs["material_positive_ucb"] < exp031.SPARSE_POSITIVE_THRESHOLD
    assert summary["positive_fixed_budget_arms"] == [
        FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    ]
    assert summary["fixed_action_validation_target"] is None
    assert summary["primary_recommendation"] == "force_abstain_sparse_s5_headroom"
    assert summary["real_aob_pilot_authorized"] == 0
    assert summary["action_gate_authorized"] == 0


def test_catastrophic_budget_arm_is_rejected_without_hiding_safe_arm() -> None:
    first_seed = exp031.SEEDS[0]

    def values(seed: int, context: int) -> dict[str, float]:
        raw = CATASTROPHIC_DELTA - 0.01 if (seed, context) == (first_seed, 0) else 0.05
        return _deltas(raw=raw, shrunk=0.02)

    summary = exp031.summarize_diagnostic(
        _arm_rows(values),
        bootstrap_replicates=100,
    )

    assert summary["rejected_catastrophic_budget_arms"] == [
        FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    ]
    assert summary["runtime_eligible_budget_arms"] == [
        SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
    ]
    assert summary["runtime_eligible_budget_sbs"]["arm"] == (
        SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
    )
    assert summary["primary_recommendation"] == (
        "advance_fixed_budget_pulse_to_real_aob_pilot"
    )


def test_summary_rejects_incomplete_seed_cluster() -> None:
    rows = _arm_rows(lambda _seed, _context: _deltas(raw=0.02))
    rows = [row for row in rows if row["seed"] != str(exp031.SEEDS[-1])]

    with pytest.raises(ValueError, match="primary context count"):
        exp031.summarize_diagnostic(rows, bootstrap_replicates=20)


@pytest.mark.parametrize("native_delta", [5e-16, -5e-16])
def test_summary_rejects_noncanonical_native_delta(native_delta: float) -> None:
    rows = _arm_rows(
        lambda _seed, _context: _deltas(native=native_delta, raw=-0.01)
    )

    with pytest.raises(ValueError, match="zero-delta reference"):
        exp031.summarize_diagnostic(rows, bootstrap_replicates=20)


def test_native_first_tolerance_tie_keeps_vbs_exactly_zero() -> None:
    rows = _arm_rows(lambda _seed, _context: _deltas(raw=5e-16, shrunk=-0.01))

    summary = exp031.summarize_diagnostic(rows, bootstrap_replicates=20)

    assert summary["registered_budget_pulse_vbs"]["mean_delta"] == 0.0
    assert summary["runtime_eligible_budget_pulse_vbs"]["mean_delta"] == 0.0
    assert summary["primary_recommendation"] == "redesign_budget_pulse_actions"


def test_arm_confidence_interval_resamples_whole_seed_clusters() -> None:
    seed_values = {
        117: -0.08,
        118: -0.02,
        119: 0.01,
        120: 0.06,
        121: 0.13,
    }
    replicates = 137
    bootstrap_seed = 917
    rows = _arm_rows(
        lambda seed, _context: _deltas(raw=seed_values[seed], shrunk=-0.01)
    )

    summary = exp031.summarize_diagnostic(
        rows,
        bootstrap_replicates=replicates,
        bootstrap_seed=bootstrap_seed,
    )
    rng = Random(bootstrap_seed)
    expected_means = []
    for _ in range(replicates):
        sampled = [
            exp031.SEEDS[rng.randrange(len(exp031.SEEDS))]
            for _seed in exp031.SEEDS
        ]
        expected_means.append(sum(seed_values[seed] for seed in sampled) / len(sampled))

    raw_stats = summary["arm_statistics"][
        FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    ]
    assert raw_stats["delta_lcb"] == pytest.approx(
        exp031._quantile(expected_means, 0.025)
    )
    assert raw_stats["delta_ucb"] == pytest.approx(
        exp031._quantile(expected_means, 0.975)
    )


def test_resume_validates_before_launching_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = exp031.RunSpec(117, tmp_path)
    sentinel = object()
    monkeypatch.setattr(exp031, "_validate_spec", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(
        exp031,
        "_run_worker",
        lambda *_args, **_kwargs: pytest.fail("resume launched an already-valid worker"),
    )

    result = exp031.run_one(
        spec,
        python_executable="fixture-python",
        config_path=exp031.DEFAULT_CONFIG_PATH,
        execution_identity={},
        resume=True,
        reuse_existing=False,
    )

    assert result is sentinel


def test_resume_reruns_only_after_artifact_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = exp031.RunSpec(117, tmp_path)
    calls: list[str] = []
    sentinel = object()

    def validate(*_args, **_kwargs):
        calls.append("validate")
        if calls.count("validate") == 1:
            raise ValueError("invalid artifact")
        return sentinel

    monkeypatch.setattr(exp031, "_validate_spec", validate)
    monkeypatch.setattr(
        exp031,
        "_validated_artifact_snapshot",
        lambda *_args, **_kwargs: (object(), {}),
    )
    monkeypatch.setattr(exp031, "_execution_identity", lambda *_args, **_kwargs: {"id": 1})
    monkeypatch.setattr(exp031, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        exp031,
        "_run_worker",
        lambda *_args, **_kwargs: calls.append("worker"),
    )

    result = exp031.run_one(
        spec,
        python_executable="fixture-python",
        config_path=exp031.DEFAULT_CONFIG_PATH,
        execution_identity={"id": 1},
        resume=True,
        reuse_existing=False,
    )

    assert result is sentinel
    assert calls == ["validate", "worker", "validate"]


def test_resume_reruns_when_validated_artifacts_have_no_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = exp031.RunSpec(117, tmp_path)
    calls: list[str] = []
    identity = {"identity": "fixture"}
    monkeypatch.setattr(
        exp031,
        "_validated_artifact_snapshot",
        lambda *_args, **_kwargs: (object(), {}),
    )
    monkeypatch.setattr(exp031, "_execution_identity", lambda *_args: identity)
    monkeypatch.setattr(
        exp031,
        "_run_worker",
        lambda *_args, **_kwargs: calls.append("worker"),
    )

    result = exp031.run_one(
        spec,
        python_executable="fixture-python",
        config_path=exp031.DEFAULT_CONFIG_PATH,
        execution_identity=identity,
        resume=True,
        reuse_existing=False,
    )

    assert calls == ["worker"]
    assert result.execution_source == "rerun_after_artifact_gate_failure"
    assert result.generation_source == "rerun_after_artifact_gate_failure"
    assert "execution receipt is missing or unreadable" in result.resume_gate_error


def test_run_experiment_stops_before_later_seed_after_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (exp031.RunSpec(117, tmp_path), exp031.RunSpec(118, tmp_path))
    calls: list[int] = []
    monkeypatch.setattr(
        exp031,
        "load_config",
        lambda *_args: {"execution": {"jobs": 1}},
    )
    monkeypatch.setattr(exp031, "build_specs", lambda *_args: specs)
    monkeypatch.setattr(exp031, "_execution_identity", lambda *_args: {})

    def fail_first_seed(spec: exp031.RunSpec, **_kwargs) -> None:
        calls.append(spec.seed)
        raise RuntimeError("fixture worker failure")

    monkeypatch.setattr(exp031, "run_one", fail_first_seed)

    with pytest.raises(RuntimeError, match="fixture worker failure"):
        exp031.run_experiment(output_root=tmp_path)

    assert calls == [117]


def test_execution_receipt_rejects_source_drift(tmp_path: Path) -> None:
    spec = exp031.RunSpec(117, tmp_path)
    identity = {
        "protocol_version": exp031.PROTOCOL_VERSION,
        "config_sha256": "a" * 64,
        "source_sha256": {"src/arac/example.py": "b" * 64},
    }
    receipt = exp031._execution_receipt(
        spec,
        identity,
        artifact_sha256={"seed/artifact.csv": "d" * 64},
        generation_source="fresh_execution",
        generation_resume_gate_error="",
    )
    receipt_path = spec.artifact_dir / exp031.EXECUTION_RECEIPT_NAME
    exp031._write_json(receipt_path, receipt)

    artifact_sha256 = {"seed/artifact.csv": "d" * 64}
    exp031._validate_execution_receipt(
        receipt_path,
        spec,
        identity,
        artifact_sha256,
    )
    with pytest.raises(ValueError, match="source/config/protocol mismatch"):
        exp031._validate_execution_receipt(
            receipt_path,
            spec,
            identity,
            {"seed/artifact.csv": "e" * 64},
        )
    receipt["source_sha256"] = {"src/arac/example.py": "c" * 64}
    exp031._write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="source/config/protocol mismatch"):
        exp031._validate_execution_receipt(
            receipt_path,
            spec,
            identity,
            artifact_sha256,
        )


def test_execution_identity_binds_current_aob_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    aob_path = tmp_path / "F5-info.txt"
    aob_path.write_text("first\n", encoding="utf-8")
    monkeypatch.setattr(exp031, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(exp031, "SOURCE_FILES", ())
    monkeypatch.setattr(
        exp031,
        "required_aob_data_files",
        lambda *_args: (aob_path,),
    )

    before = exp031._execution_identity(config_path)
    aob_path.write_text("second\n", encoding="utf-8")
    after = exp031._execution_identity(config_path)

    assert before["aob_input_sha256"] != after["aob_input_sha256"]


def test_s_profile_worker_rejects_header_only_artifacts(tmp_path: Path) -> None:
    args = argparse.Namespace(
        cohort="real_aob",
        case="S5",
        seed=117,
        max_fes=300_000,
        output_root=str(tmp_path),
        timestamp="s-diagnostic",
        profile=exp031.S_FAMILY_BUDGET_PULSE_PROFILE,
    )
    base = tmp_path / args.timestamp / "schwefel"
    base.mkdir(parents=True)
    for path, fields in (
        (base / "S5_action_ceiling_contexts.csv", ACTION_CEILING_CONTEXT_FIELDS),
        (base / "S5_action_ceiling_arm_results.csv", ACTION_CEILING_ARM_RESULT_FIELDS),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    with pytest.raises(RuntimeError, match="incomplete action-ceiling artifacts"):
        _diagnostic_worker.require_s_budget_pulse_artifacts(args)
