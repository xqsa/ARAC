from __future__ import annotations

import json

import pytest

from experiments.phase2_v2_validation import (
    DEFAULT_CONFIG,
    METHODS,
    _comparison_rows,
    _contexts,
    load_config,
    run_validation,
)


def test_validation_config_freezes_contexts_methods_and_budget() -> None:
    config = load_config(DEFAULT_CONFIG)
    contexts = _contexts(config)

    assert tuple(config["methods"]) == METHODS
    assert config["global_max_fes"] == 40_000
    assert config["branch_probe_fes"] == 512
    assert len(contexts) == 32
    assert sum(row["suite"] == "aob" for row in contexts) == 8
    assert sum(row["suite"] == "ioh_bbob" for row in contexts) == 24
    assert set(config["run_seeds"]).isdisjoint({117, 129, 141, 142, 20260753})


def test_validation_comparison_requires_equal_checkpoints_and_reports_wins() -> None:
    def receipt(method: str, error: float, checkpoint: str = "a" * 64):
        return {
            "context_id": "ioh_f01_i1_d40_s1",
            "method": method,
            "checkpoint_sha256": checkpoint,
            "benchmark": {"suite": "ioh_bbob"},
            "result": {
                "final_error": error,
                "selected_action": "aor",
                "selection_reason": "test",
            },
        }

    rows, summary = _comparison_rows(
        [
            receipt("probe_commit_v2", 1.0),
            receipt("mechanism_score_v1", 2.0),
        ]
    )
    assert rows[0]["winner"] == "probe_commit_v2"
    assert rows[0]["shifted_log10_probe_over_mechanism"] == pytest.approx(
        -0.17609125905568127
    )
    assert summary["ioh_bbob"]["probe_wins"] == 1

    with pytest.raises(ValueError, match="checkpoint pair drifted"):
        _comparison_rows(
            [
                receipt("probe_commit_v2", 1.0),
                receipt("mechanism_score_v1", 2.0, "b" * 64),
            ]
        )


def test_validation_rejects_nonempty_output_root(tmp_path) -> None:
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "keep.json").write_text(json.dumps({"keep": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_validation(config_path=DEFAULT_CONFIG, output_root=root)
