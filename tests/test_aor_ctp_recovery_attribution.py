from __future__ import annotations

from experiments.historical_recovery.aor_ctp_recovery_attribution import load_protocol, verify


def test_protocol_freezes_aor_ctp_attribution_inputs() -> None:
    protocol = load_protocol()
    assert tuple(protocol["aor_cases"]) == ("A4", "A6")
    assert tuple(protocol["ctp_cases"]) == ("S6",)
    assert protocol["patch_enabled"] is False
    assert protocol["selector_enabled"] is False


def test_existing_attribution_report_verifies_across_generation_time() -> None:
    report = verify()
    assert "generated_at_utc" in report
    assert report["aor_source_identity_exact"] is True
    assert report["ctp_case_summary"]["matched_tail_ablation_same_checkpoint"] is True
