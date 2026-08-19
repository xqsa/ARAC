"""Protocol-level reachability checks for Gate47-R's fresh route witness."""

from __future__ import annotations

from experiments.oc_residual_topology_gate47r import _route_witness


def test_gate47r_witness_covers_all_dispatch_branches() -> None:
    result = _route_witness(20260836)
    assert result["all_expected_routes"] is True
    assert {case["action"] for case in result["cases"]} == {
        "arbitration_only",
        "ctp_restricted",
        "ctp_shared_core",
        "smp",
        "aor",
    }
