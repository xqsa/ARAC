from __future__ import annotations

import json

import pytest

from experiments.phase2_v2_pilot import (
    DEFAULT_CONFIG,
    CONFIG_SCHEMA,
    SOURCE_PATHS,
    load_config,
    run_pilot,
)


def test_phase2_v2_pilot_config_is_frozen_and_globally_budgeted() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config["schema_version"] == CONFIG_SCHEMA
    assert config["ioh_version"] == "0.3.22"
    assert 4 * config["branch_probe_fes"] < config["global_max_fes"]
    assert {run["suite"] for run in config["runs"]} == {"aob", "ioh_bbob"}
    assert len({run["run_seed"] for run in config["runs"]}) == len(config["runs"])
    assert "experiments/phase2_v2_pilot.py" in SOURCE_PATHS


def test_phase2_v2_pilot_rejects_a_nonempty_output_root(tmp_path) -> None:
    output_root = tmp_path / "occupied"
    output_root.mkdir()
    (output_root / "unrelated.json").write_text(json.dumps({"keep": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_pilot(config_path=DEFAULT_CONFIG, output_root=output_root)
