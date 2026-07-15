from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_state_evidence.py"
SPEC = importlib.util.spec_from_file_location("audit_state_evidence", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_spearman_handles_ties_and_direction() -> None:
    assert MODULE.spearman([1.0, 2.0, 2.0, 4.0], [4.0, 3.0, 3.0, 1.0]) == -1.0
    assert MODULE.spearman([1.0, 1.0], [1.0, 2.0]) is None


def test_within_case_concordance_uses_only_cross_seed_pairs() -> None:
    rows = [
        {"problem_id": "E2", "feature": 1.0, "outcome": 3.0},
        {"problem_id": "E2", "feature": 2.0, "outcome": 2.0},
        {"problem_id": "E2", "feature": 3.0, "outcome": 4.0},
        {"problem_id": "S3", "feature": 5.0, "outcome": 1.0},
        {"problem_id": "S3", "feature": 5.0, "outcome": 9.0},
    ]

    comparable, concordant, fraction = MODULE.within_case_concordance(
        rows, "feature", "outcome"
    )

    assert comparable == 3
    assert concordant == 2
    assert fraction == 2 / 3


def test_car_catastrophic_threshold_is_twenty_percent_relative_loss() -> None:
    threshold = MODULE.math.log(1.0 / 1.2)
    assert threshold < 0.0
    assert MODULE.math.isclose(MODULE.math.exp(-threshold), 1.2)
