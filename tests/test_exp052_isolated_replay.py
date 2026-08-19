from __future__ import annotations

from pathlib import Path

import pytest

from experiments.historical_recovery.run_exp052_isolated_replay import (
    OUTPUT_ROOT,
    build_preflight,
    verify_reproduction,
)


def test_exp052_replay_preflight_is_scoped_to_one_historical_trajectory() -> None:
    preflight = build_preflight()

    assert preflight["authorization_scope"] == "one_trajectory_only"
    assert preflight["claim_boundary"] == (
        "version_level_session_provenance_not_bitwise_receipt_bound"
    )
    assert preflight["target"] == {
        "case": "E1",
        "condition": "candidate_smp",
        "seed": 117,
        "max_fes": 3_000_000,
    }
    assert preflight["exact_override_count"] == 41
    assert preflight["source_hashes_valid"] is True
    assert preflight["session_observed_environment_binding"] is True
    assert preflight["receipt_environment_binding"] is False


@pytest.mark.skipif(
    not (Path(OUTPUT_ROOT) / "reproduction_summary.json").is_file(),
    reason="isolated reproduction has not been executed",
)
def test_completed_exp052_reproduction_verifies_exactly() -> None:
    verification = verify_reproduction()

    assert verification["verification_passed"] is True
    assert verification["checks"]["exact_historical_value_match"] is True
