from __future__ import annotations

from concurrent.futures import Future
import json
from pathlib import Path

import pytest

import experiments.phase2_v2_aob_restored_pilot as pilot
from experiments.phase2_v2_aob_restored_pilot import (
    METHODS,
    load_config,
    matched_total_fes,
)


CONFIG = Path("experiments/phase2_v2_aob_restored_pilot_config.json")


def test_restored_pilot_config_uses_original_phase1_boundary() -> None:
    config = load_config(CONFIG)
    assert config["global_max_fes"] == 3_000_000
    assert config["phase1_fes"] == 180_000
    assert config["max_workers"] == 8
    assert tuple(config["methods"]) == METHODS
    assert set(config["run_seeds"]).isdisjoint({117, 129, 141, 142, 20260753})


def test_matched_control_reserves_all_four_probe_branches() -> None:
    config = load_config(CONFIG)
    assert matched_total_fes(config) == 2_868_928
    assert config["global_max_fes"] - matched_total_fes(config) == 4 * config["branch_probe_fes"]


def test_config_has_no_unregistered_keys() -> None:
    values = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert set(values) == {
        "schema_version",
        "global_max_fes",
        "phase1_fes",
        "branch_probe_fes",
        "decision_horizon_fes",
        "exploration_floor_fes",
        "min_relative_margin",
        "min_leader_stability",
        "max_workers",
        "methods",
        "aob_cases",
        "run_seeds",
    }


class _InlineProcessPool:
    submitted = 0

    def __init__(self, **_: object) -> None:
        pass

    def __enter__(self) -> _InlineProcessPool:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def submit(self, function, *args: object) -> Future:
        type(self).submitted += 1
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:  # pragma: no cover - exercised through Future.result
            future.set_exception(exc)
        return future


def _fake_run_one(
    context: dict[str, object],
    method: str,
    config: dict[str, object],
) -> tuple[dict[str, object], int]:
    total = int(config["global_max_fes"])
    branch = int(config["branch_probe_fes"])
    objective_fes = total - 4 * branch if method == "mechanism_score_matched_v1" else total
    checkpoint = f"{context['case']}-{context['run_seed']}"
    result = {
        "method": method,
        "phase1": {
            "checkpoint_sha256": checkpoint,
            "phase1_fes": int(config["phase1_fes"]),
            "structural_inference_complete": 1.0,
        },
        "selected_action": "aor",
        "final_error": float(METHODS.index(method) + 1),
        "objective_fes": objective_fes,
        "global_total_fes": total,
        "reserved_probe_tax_fes": 0 if method == "mechanism_score_full_v1" else 4 * branch,
        "terminal_complete": method != "mechanism_score_matched_v1",
    }
    return result, objective_fes


def test_parallel_pilot_resumes_only_missing_method(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "pilot"
    _InlineProcessPool.submitted = 0
    monkeypatch.setattr(pilot, "ProcessPoolExecutor", _InlineProcessPool)
    monkeypatch.setattr(pilot, "_run_one", _fake_run_one)

    summary = pilot.run_pilot(
        config_path=CONFIG,
        output_root=output_root,
        max_workers=8,
    )

    assert summary["completed"] == 24
    assert summary["failed"] == 0
    assert _InlineProcessPool.submitted == 8
    frozen_config = (output_root / "config.json").read_bytes()
    missing = output_root / "receipts" / "002_aob_A1_s20261211_mechanism_score_matched_v1.json"
    missing.unlink()
    resumed_calls = []

    def tracking_run_one(context, method, config):
        resumed_calls.append((context["case"], context["run_seed"], method))
        return _fake_run_one(context, method, config)

    monkeypatch.setattr(pilot, "_run_one", tracking_run_one)
    resumed = pilot.run_pilot(
        config_path=CONFIG,
        output_root=output_root,
        max_workers=8,
        resume=True,
    )

    assert resumed["completed"] == 24
    assert resumed_calls == [("A1", 20261211, "mechanism_score_matched_v1")]
    assert (output_root / "config.json").read_bytes() == frozen_config


def test_resume_rejects_manifest_and_frozen_config_drift(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pilot, "ProcessPoolExecutor", _InlineProcessPool)
    monkeypatch.setattr(pilot, "_run_one", _fake_run_one)
    output_root = tmp_path / "pilot"
    pilot.run_pilot(config_path=CONFIG, output_root=output_root, max_workers=8)

    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["max_workers"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash drifted"):
        pilot.run_pilot(
            config_path=CONFIG,
            output_root=output_root,
            max_workers=8,
            resume=True,
        )

    pilot._write_json_atomic(manifest_path, pilot._manifest(CONFIG.resolve(), load_config(CONFIG), max_workers=8))
    (output_root / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen config drifted"):
        pilot.run_pilot(
            config_path=CONFIG,
            output_root=output_root,
            max_workers=8,
            resume=True,
        )


def test_fresh_run_rejects_nonempty_root(tmp_path) -> None:
    output_root = tmp_path / "pilot"
    output_root.mkdir()
    (output_root / "unexpected.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="output root is not empty"):
        pilot.run_pilot(config_path=CONFIG, output_root=output_root, max_workers=8)


def test_safe_print_tolerates_closed_stdout(monkeypatch) -> None:
    def closed_stdout(*_: object, **__: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", closed_stdout)
    pilot._safe_print("progress")
