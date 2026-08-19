from __future__ import annotations

from experiments.historical_recovery.independent_terminal_parity import (
    _baseline_rows,
    load_protocol,
    nearest_rank_p90,
)


def test_nearest_rank_p90_is_frozen() -> None:
    assert nearest_rank_p90([float(value) for value in range(1, 11)]) == 9.0
    assert nearest_rank_p90([5.0, 1.0, 3.0, 2.0, 4.0]) == 5.0


def test_protocol_runs_only_current_historical_level_failures() -> None:
    protocol = load_protocol()
    rows = _baseline_rows(protocol)

    assert protocol["required_candidate_actions"] == ["aor", "smp"]
    assert {row["action"] for row in rows if row["current_historical_level_passed"]} == {
        "ctp",
        "gcb",
    }
    assert all(row["historical"]["count"] == 25 for row in rows if row["action"] != "smp")
    assert next(row for row in rows if row["action"] == "smp")["historical"]["count"] == 5

