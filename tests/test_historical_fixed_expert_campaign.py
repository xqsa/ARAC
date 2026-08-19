from __future__ import annotations

import copy
import json

import pytest

from experiments.historical_recovery.fixed_expert_campaign import (
    DEFAULT_CONFIG,
    EXPECTED_CASES,
    EXPECTED_MAPPING,
    EXPECTED_SEEDS,
    build_contexts,
    load_config,
)


def test_fixed_expert_config_binds_historical_scope_and_parallelism() -> None:
    config = load_config()

    assert tuple(config["cases"]) == EXPECTED_CASES
    assert tuple(config["seeds"]) == EXPECTED_SEEDS
    assert config["expert_mapping"] == EXPECTED_MAPPING
    assert config["max_fes"] == 3_000_000
    assert config["max_workers"] == 24
    assert len(config["cases"]) * len(config["seeds"]) == 600


def test_fixed_expert_contexts_execute_only_the_family_mapped_arm(tmp_path) -> None:
    checkpoints, arms = build_contexts(
        ("A1", "E1", "R1", "S1"),
        (117, 118),
        EXPECTED_MAPPING,
        max_fes=3_000_000,
        output_root=tmp_path,
    )

    assert len(checkpoints) == len(arms) == 8
    assert [(row.case_id, row.action_name) for row in arms[:4]] == [
        ("A1", "aor"),
        ("E1", "smp"),
        ("R1", "gcb"),
        ("S1", "ctp"),
    ]
    assert all(row.max_fes == 3_000_000 for row in arms)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seeds", list(range(118, 143)), "seeds 117..141"),
        ("max_workers", 25, "workers must be in 1..24"),
        ("expert_mapping", {**EXPECTED_MAPPING, "E": "ctp"}, "mapping drifted"),
    ],
)
def test_fixed_expert_config_rejects_protocol_drift(
    tmp_path,
    field,
    value,
    message,
) -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(path)
