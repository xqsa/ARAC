from __future__ import annotations

from experiments.oracle_sparse_overlap_discovery_gate6_budget import run_pilot


def test_d1000_pilot_fits_the_180k_phase1_budget() -> None:
    payload = run_pilot()

    assert payload["budget_passed"]
    assert payload["complete"]
    assert payload["complete_reason"] == "complete"
    assert payload["separated_pair_fraction"] == 1.0
    assert payload["expected_shared"] == payload["inferred_shared"] == (2, 102)
    assert payload["consumed_fes"] == payload["expected_fes"] == 126085
    assert payload["remaining_fes"] == 53915
    assert payload["candidate_pair_count"] == 12
