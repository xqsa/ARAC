from __future__ import annotations

from experiments.phase1_overlap_integration_gate7 import run_gate


def test_gate7_phase1_overlap_integration_is_complete() -> None:
    payload = run_gate()

    assert all(payload["gate_checks"].values())
    assert payload["phase1_consumed_fes"] == 180_000
    assert payload["phase1_remaining_to_total"] == 2_820_000
    assert payload["inferred_shared"] == payload["expected_shared"] == (2, 102)
