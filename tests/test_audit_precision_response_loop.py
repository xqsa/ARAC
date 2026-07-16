from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_precision_response_loop.py"
SPEC = importlib.util.spec_from_file_location("audit_precision_response_loop_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_two_way_bootstrap_is_deterministic_and_positive() -> None:
    rows = [
        {"problem_id": case, "seed": str(seed), "tau": "0.1"}
        for case in ("A4", "A5", "E2", "E3")
        for seed in (60, 61, 62)
    ]

    first = AUDIT._two_way_cluster_summary(rows, "tau", resamples=200, seed=7)
    second = AUDIT._two_way_cluster_summary(rows, "tau", resamples=200, seed=7)

    assert first == second
    assert first["mean"] == pytest.approx(0.1)
    assert first["lcb_95"] == pytest.approx(0.1)


def test_two_way_bootstrap_fails_closed_without_cluster_support() -> None:
    summary = AUDIT._two_way_cluster_summary(
        [{"problem_id": "A4", "seed": "60", "tau": "0.1"}],
        "tau",
        resamples=20,
        seed=1,
    )

    assert summary["lcb_95"] is None
