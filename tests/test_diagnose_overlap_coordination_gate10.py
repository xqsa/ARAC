from __future__ import annotations

from experiments.diagnose_overlap_coordination_gate10 import run_diagnosis


def test_gate10_diagnosis_is_read_only_and_reports_both_modes() -> None:
    payload = run_diagnosis()

    assert payload["context_count"] == 60
    assert set(payload["mode_summary"]) == {"conforming", "conflicting"}
    assert payload["trigger_gain"]["conflicting_trigger_count"] >= 1
    assert "likely_mechanism" in payload["interpretation"]
    assert "next_interface_to_test" in payload["interpretation"]
