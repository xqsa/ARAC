"""Regression checks for receipt-only mechanism diagnostics."""

from __future__ import annotations

from pathlib import Path

from experiments.oc_mechanism_diagnostics import diagnose_r2, diagnose_s5


ROOT = "artifacts/oc_phase_aware_gate51c_v5_1"


def test_s5_diagnostic_exposes_plateau_release_and_matched_budget() -> None:
    row = diagnose_s5(root=Path(ROOT), seed=20260901)
    assert row["protected_runway_fes"] > 0
    assert row["plateau_release_count"] >= 1
    assert row["ctp_matched_budget"]
    assert row["handoff_count"] > 0
    assert row["release_receipts"][0]["released"] is True
    assert row["release_receipts"][0]["next_episode"] != "ctp"


def test_r2_diagnostic_reports_aor_horizon_or_explicit_absence() -> None:
    row = diagnose_r2(root=Path(ROOT), seed=20260901)
    assert row["horizon_fes"] == 450_000
    assert row["aor_receipts"]
    assert row["horizon_crossing"] is not None
    # This seed is a useful negative horizon result: AOR reaches the
    # calibrated 450k runtime without a material global archive gain.
    assert row["first_material_global_receipt"] is None or row["first_material_global_receipt"]["runtime_fes"] >= 450_000
