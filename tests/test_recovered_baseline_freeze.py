from __future__ import annotations

from experiments.historical_recovery.verify_recovered_baseline_freeze import verify


def test_recovered_baseline_manifest_is_intact() -> None:
    result = verify()
    assert result["freeze_id"] == "arac-recovered-baseline-20260823-v1"
    assert result["checked_file_count"] == 21
    assert result["screen_contract_green"] is True
    assert result["smp_smoke_green"] is True
    assert result["e1_preservation_green"] is True
    assert result["patch_enabled"] is False
    assert result["soft_routing_enabled"] is False
    assert result["selector_enabled"] is False
