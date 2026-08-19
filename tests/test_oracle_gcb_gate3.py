from __future__ import annotations

from experiments.oracle_gcb_gate3 import TOPOLOGIES, run_diagnostic


def test_gcb_confirmation_smoke_is_budget_exact_and_nonworsening() -> None:
    result = run_diagnostic((17, 23), workers=1)

    assert set(result["summary"]) >= {item.name for item in TOPOLOGIES}
    assert all(
        item["exact_budget"] and item["archive_nonworsening"]
        for name, item in result["summary"].items()
        if name in {topology.name for topology in TOPOLOGIES}
    )
