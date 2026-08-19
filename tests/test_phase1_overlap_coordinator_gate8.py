from __future__ import annotations

from experiments.phase1_overlap_coordinator_gate8 import run_gate


def test_gate8_coordinator_consumes_phase1_shared_memberships() -> None:
    payload = run_gate()

    assert payload["gate_passed"]
    assert all(payload["gate_checks"].values())
    assert payload["shared_variables"] == (2, 102)
    assert payload["phase1_fes"] == 180_000
    assert payload["phase2_budget"] == 2_820_000
