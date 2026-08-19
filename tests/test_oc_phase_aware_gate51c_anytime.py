"""Checks for post-processing legacy Gate 51c cells into anytime/AUC."""

from pathlib import Path

from experiments.oc_phase_aware_gate51c_anytime import build


def test_existing_gate51c_cells_have_fixed_grid_anytime_auc() -> None:
    payload = build(Path("artifacts/oc_phase_aware_gate51c_v5_1"))
    assert payload["grid_total_fes"] == [600_000, 1_000_000, 2_000_000, 3_000_000]
    assert set(payload["per_case"]) == {"A3", "R2", "R6", "S5"}
    for rows in payload["per_case"].values():
        assert len(rows) == 3
        for row in rows:
            assert set(row["on_anytime"]) == {"600000", "1000000", "2000000", "3000000"}
            assert row["on_log_error_auc"] > 0.0
            assert row["best_log_error_auc"] > 0.0
