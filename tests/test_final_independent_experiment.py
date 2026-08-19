from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import experiments.final.run as final_run
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ACTION_NAMES
from experiments.final.run import (
    CONFIG_SCHEMA,
    ArmContext,
    CheckpointContext,
    execute_method,
    load_config,
    run_experiment,
    run_outcome_campaign,
)


class _FixedSelector:
    def __init__(self, action: str) -> None:
        self.action = action
        self.received_names = None

    def select(self, feature_names, feature_values):
        self.received_names = tuple(feature_names)
        assert len(feature_values) == len(feature_names)
        return self.action


@pytest.mark.parametrize("action", ACTION_NAMES)
def test_complete_method_selects_once_and_finishes_exact_budget(action: str) -> None:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )
    selector = _FixedSelector(action)

    execution = execute_method(problem, run_seed=21, max_fes=500, selector=selector)

    assert execution.selected_action == action
    assert execution.result.action_name == action
    assert execution.result.terminal_fes == 500
    assert "case_id" not in selector.received_names
    assert "family" not in selector.received_names


def test_config_rejects_family_action_or_adapter_keys(tmp_path) -> None:
    config = {
        "schema_version": CONFIG_SCHEMA,
        "cases": [f"{family}{index}" for family in "AERS" for index in range(1, 7)],
        "calibration_seeds": [1],
        "holdout_seeds": [2],
        "evaluation_seeds": [3],
        "max_fes": 500,
        "max_workers": 1,
        "selector_directory": "artifacts/selectors/test",
        "selector_files": {},
        "canonical_action_by_family": {"A": "aor"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="keys drifted"):
        load_config(path)


def test_outcome_campaign_prepares_one_checkpoint_per_case_seed(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    def record_contexts(contexts, *args, **kwargs):
        calls.append(tuple(contexts))
        return []

    monkeypatch.setattr("experiments.final.run._run_parallel", record_contexts)

    records = run_outcome_campaign(
        ("A1",),
        (11, 13),
        max_fes=500,
        max_workers=2,
        output_root=tmp_path,
        resume=False,
    )

    assert records == ()
    assert len(calls) == 2
    assert len(calls[0]) == 2
    assert all(isinstance(context, CheckpointContext) for context in calls[0])
    assert len(calls[1]) == 2 * len(ACTION_NAMES)
    assert all(isinstance(context, ArmContext) for context in calls[1])


def test_runtime_warning_provenance_classifies_known_overflow() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn_explicit(
            "overflow encountered in multiply",
            RuntimeWarning,
            str(final_run.REPOSITORY_ROOT / "src" / "arac" / "actions" / "_execution.py"),
            151,
        )

    entries = final_run._serialize_runtime_warnings(caught)
    summary = final_run._runtime_warning_summary(
        ({"action_name": "smp", "runtime_warnings": entries},)
    )

    assert entries == [
        {
            "category": "RuntimeWarning",
            "message": "overflow encountered in multiply",
            "source": "src/arac/actions/_execution.py",
            "line": 151,
            "count": 1,
            "known": True,
        }
    ]
    assert summary["runtime_warning_count"] == 1
    assert summary["runtime_warning_counts_by_action"] == {"smp": 1}
    assert summary["all_runtime_warnings_known"] is True
    final_run._require_known_runtime_warnings(summary, stage="test campaign")


def test_runtime_warning_provenance_flags_unexpected_warning() -> None:
    entries = [
        {
            "category": "RuntimeWarning",
            "message": "unexpected numerical warning",
            "source": "optimizer.py",
            "line": 7,
            "count": 2,
            "known": False,
        }
    ]

    summary = final_run._runtime_warning_summary(
        ({"action_name": "ctp", "runtime_warnings": entries},)
    )

    assert summary["unexpected_runtime_warning_count"] == 2
    assert summary["all_runtime_warnings_known"] is False

    with pytest.raises(RuntimeError, match="downstream stages are blocked"):
        final_run._require_known_runtime_warnings(summary, stage="test campaign")


def test_outcome_campaign_unknown_warning_stops_after_writing_summary(
    monkeypatch,
    tmp_path,
) -> None:
    unexpected = [
        {
            "category": "RuntimeWarning",
            "message": "unexpected numerical warning",
            "source": "optimizer.py",
            "line": 7,
            "count": 1,
            "known": False,
        }
    ]

    def completed_campaign(contexts, *args, **kwargs):
        if contexts and isinstance(contexts[0], CheckpointContext):
            return []
        return [{"action_name": "ctp", "runtime_warnings": unexpected}]

    monkeypatch.setattr(final_run, "_run_parallel", completed_campaign)
    monkeypatch.setattr(final_run, "_records_from_arms", lambda rows: ())
    output_root = tmp_path / "campaign"

    with pytest.raises(RuntimeError, match="unknown runtime warning"):
        run_outcome_campaign(
            ("A1",),
            (11,),
            max_fes=500,
            max_workers=1,
            output_root=output_root,
            resume=False,
        )

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["unexpected_runtime_warning_count"] == 1
    assert summary["all_runtime_warnings_known"] is False


def test_calibration_training_gate_runs_before_holdout(monkeypatch, tmp_path) -> None:
    config = {
        "schema_version": CONFIG_SCHEMA,
        "cases": [f"{family}{index}" for family in "AERS" for index in range(1, 7)],
        "calibration_seeds": [11, 13],
        "holdout_seeds": [17],
        "evaluation_seeds": [19],
        "max_fes": 500,
        "max_workers": 1,
        "selector_directory": "artifacts/selectors/test",
        "selector_files": {},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls = []

    def empty_campaign(*args, **kwargs):
        calls.append(kwargs["output_root"])
        return ()

    monkeypatch.setattr(final_run, "run_outcome_campaign", empty_campaign)
    monkeypatch.setattr(
        final_run,
        "evaluate_training_selector",
        lambda records: {
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "terminal_regret": {
                "mean_log10_regret": 0.06,
                "worst_log10_regret": 0.3,
            },
            "passed": False,
        },
    )

    output_root = tmp_path / "calibration"
    with pytest.raises(RuntimeError, match="holdout campaign was not started"):
        final_run.calibrate_selector(
            config_path=config_path,
            output_root=output_root,
        )

    assert calls == [output_root / "train"]
    assert (output_root / "training_preflight.json").is_file()
    assert not (output_root / "holdout").exists()


def test_outcome_campaign_rejects_nonempty_fresh_root(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "campaign"
    output_root.mkdir()
    (output_root / "unrelated.txt").write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(
        "experiments.final.run._run_parallel",
        lambda *args, **kwargs: pytest.fail("workers must not start"),
    )

    with pytest.raises(ValueError, match="not empty"):
        run_outcome_campaign(
            ("A1",),
            (11,),
            max_fes=500,
            max_workers=24,
            output_root=output_root,
            resume=False,
        )


def test_outcome_campaign_resume_requires_immutable_manifest(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"version": 1}\n', encoding="utf-8")
    output_root = tmp_path / "campaign"
    calls = []

    def empty_campaign(contexts, *args, **kwargs):
        calls.append(tuple(contexts))
        return []

    monkeypatch.setattr(final_run, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(final_run, "_run_parallel", empty_campaign)

    run_outcome_campaign(
        ("A1",),
        (11,),
        max_fes=500,
        max_workers=24,
        output_root=output_root,
        resume=False,
        config_path=config,
    )
    manifest = json.loads(
        (output_root / final_run.CAMPAIGN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["cases"] == ["A1"]
    assert manifest["seeds"] == [11]
    assert manifest["max_fes"] == 500
    assert manifest["actions"] == list(ACTION_NAMES)
    assert manifest["config_sha256"] == final_run.file_sha256(config)
    assert manifest["source_hashes"] == {"test_source": final_run.file_sha256(source)}
    assert manifest["vendor_trees"]["vendor/aob"]["file_count"] == 67
    assert len(manifest["vendor_trees"]["vendor/aob"]["tree_sha256"]) == 64

    run_outcome_campaign(
        ("A1",),
        (11,),
        max_fes=500,
        max_workers=24,
        output_root=output_root,
        resume=True,
        config_path=config,
    )
    assert len(calls) == 4

    for cases, seeds, max_fes in (
        (("A2",), (11,), 500),
        (("A1",), (13,), 500),
        (("A1",), (11,), 501),
    ):
        with pytest.raises(ValueError, match="campaign manifest contract drifted"):
            run_outcome_campaign(
                cases,
                seeds,
                max_fes=max_fes,
                max_workers=24,
                output_root=output_root,
                resume=True,
                config_path=config,
            )

    config.write_text('{"version": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="campaign manifest contract drifted"):
        run_outcome_campaign(
            ("A1",),
            (11,),
            max_fes=500,
            max_workers=24,
            output_root=output_root,
            resume=True,
            config_path=config,
        )

    config.write_text('{"version": 1}\n', encoding="utf-8")
    source.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="campaign manifest contract drifted"):
        run_outcome_campaign(
            ("A1",),
            (11,),
            max_fes=500,
            max_workers=24,
            output_root=output_root,
            resume=True,
            config_path=config,
        )


def test_outcome_campaign_resume_requires_existing_manifest(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="resume campaign root is missing"):
        run_outcome_campaign(
            ("A1",),
            (11,),
            max_fes=500,
            max_workers=24,
            output_root=tmp_path / "missing",
            resume=True,
        )

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="campaign manifest is missing"):
        run_outcome_campaign(
            ("A1",),
            (11,),
            max_fes=500,
            max_workers=24,
            output_root=empty_root,
            resume=True,
        )


def test_directory_tree_hash_ignores_python_cache_artifacts(tmp_path) -> None:
    vendor_root = tmp_path / "vendor"
    vendor_root.mkdir()
    stable_source = vendor_root / "benchmark.py"
    stable_source.write_text("VALUE = 1\n", encoding="utf-8")
    cache_root = vendor_root / "__pycache__"
    cache_root.mkdir()
    cached_bytecode = cache_root / "benchmark.cpython-312.pyc"
    cached_bytecode.write_bytes(b"cache-v1")
    (cache_root / "nested.txt").write_text("generated", encoding="utf-8")
    optimized_bytecode = vendor_root / "benchmark.pyo"
    optimized_bytecode.write_bytes(b"optimized-v1")

    file_count, initial_hash = final_run._directory_tree_sha256(vendor_root)
    assert file_count == 1

    cached_bytecode.write_bytes(b"cache-v2")
    optimized_bytecode.write_bytes(b"optimized-v2")
    assert final_run._directory_tree_sha256(vendor_root) == (file_count, initial_hash)

    stable_source.write_text("VALUE = 2\n", encoding="utf-8")
    assert final_run._directory_tree_sha256(vendor_root)[1] != initial_hash


def test_freeze_protocol_copies_config_once_and_only_validates_on_resume(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"version": 1}\n', encoding="utf-8")
    selector_directory = tmp_path / "selector"
    selector_directory.mkdir()
    selector_hashes = {}
    for name in (final_run.MODEL_FILENAME, final_run.METADATA_FILENAME, final_run.EVALUATION_FILENAME):
        path = selector_directory / name
        path.write_text(name, encoding="utf-8")
        selector_hashes[name] = final_run.file_sha256(path)
    output_root = tmp_path / "output"
    monkeypatch.setattr(final_run, "SOURCE_PATHS", {"test_source": source})
    source_hashes = final_run._source_hashes()

    final_run._freeze_protocol(
        output_root,
        config_path=config,
        selector_directory=selector_directory,
        selector_hashes=selector_hashes,
        source_hashes=source_hashes,
        phase1_fes=125,
    )
    frozen_root = output_root / "frozen_protocol"
    assert (frozen_root / "config.json").read_bytes() == config.read_bytes()

    def reject_copy(*args, **kwargs):
        pytest.fail("an existing frozen snapshot must never be overwritten")

    monkeypatch.setattr(final_run.shutil, "copy2", reject_copy)
    final_run._freeze_protocol(
        output_root,
        config_path=config,
        selector_directory=selector_directory,
        selector_hashes=selector_hashes,
        source_hashes=source_hashes,
        phase1_fes=125,
    )

    (frozen_root / "config.json").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen config hash drifted"):
        final_run._freeze_protocol(
            output_root,
            config_path=config,
            selector_directory=selector_directory,
            selector_hashes=selector_hashes,
            source_hashes=source_hashes,
            phase1_fes=125,
        )


def test_e2e_loads_selector_gate_before_creating_campaign(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "output"
    selector_hashes = {
        final_run.MODEL_FILENAME: "1" * 64,
        final_run.METADATA_FILENAME: "2" * 64,
        final_run.EVALUATION_FILENAME: "3" * 64,
    }
    config = {
        "cases": ["A1"],
        "calibration_seeds": [11],
        "holdout_seeds": [13],
        "evaluation_seeds": [17],
        "max_fes": 500,
        "max_workers": 24,
        "selector_directory": "artifacts/selectors/test",
        "selector_files": selector_hashes,
    }
    monkeypatch.setattr(final_run, "load_config", lambda path: config)
    monkeypatch.setattr(final_run, "_validate_selector_files", lambda *args: None)

    def reject_selector(directory):
        raise ValueError("selector holdout gate failed")

    monkeypatch.setattr(final_run.OutcomeSelector, "load", staticmethod(reject_selector))

    with pytest.raises(ValueError, match="holdout gate failed"):
        run_experiment(config_path=config_path, output_root=output_root)
    assert not output_root.exists()


def test_e2e_manifest_allows_worker_change_but_rejects_budget_drift(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    selector_hashes = {
        final_run.MODEL_FILENAME: "1" * 64,
        final_run.METADATA_FILENAME: "2" * 64,
        final_run.EVALUATION_FILENAME: "3" * 64,
    }
    config = {
        "cases": ["A1"],
        "calibration_seeds": [11],
        "holdout_seeds": [13],
        "evaluation_seeds": [17],
        "max_fes": 500,
        "max_workers": 24,
        "selector_directory": "artifacts/selectors/test",
        "selector_files": selector_hashes,
    }
    row = {
        "case_id": "A1",
        "run_seed": 17,
        "selected_action": ACTION_NAMES[0],
        "phase1_fes": final_run.phase1_budget(500),
        "terminal_fes": 500,
        "max_fes": 500,
        "final_error": 1.0,
        "elapsed_seconds": 0.1,
        "checkpoint_hash": "a" * 64,
        "action_result_hash": "b" * 64,
        "receipt_hash": "c" * 64,
        "selected_action_only": True,
    }
    worker_counts = []
    monkeypatch.setattr(final_run, "load_config", lambda path: config)
    monkeypatch.setattr(final_run, "_validate_selector_files", lambda *args: None)
    monkeypatch.setattr(final_run.OutcomeSelector, "load", staticmethod(lambda directory: object()))
    monkeypatch.setattr(final_run, "_freeze_protocol", lambda *args, **kwargs: None)

    def completed_campaign(contexts, *args, **kwargs):
        worker_counts.append(kwargs["max_workers"])
        return [row]

    monkeypatch.setattr(final_run, "_run_parallel", completed_campaign)
    output_root = tmp_path / "output"

    run_experiment(
        config_path=config_path,
        output_root=output_root,
        max_workers=24,
        resume=False,
    )
    manifest = json.loads(
        (output_root / final_run.CAMPAIGN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["campaign_kind"] == "end_to_end"
    assert manifest["selector_hashes"] == selector_hashes

    run_experiment(
        config_path=config_path,
        output_root=output_root,
        max_workers=12,
        resume=True,
    )
    assert worker_counts == [24, 12]

    with pytest.raises(ValueError, match="campaign manifest contract drifted"):
        run_experiment(
            config_path=config_path,
            output_root=output_root,
            max_fes=501,
            max_workers=24,
            resume=True,
        )
