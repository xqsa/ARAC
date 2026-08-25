from __future__ import annotations

from experiments.historical_recovery.gcb_recovery_attribution import (
    EXPECTED_CASES,
    EXPECTED_SEEDS,
    load_protocol,
    parse_route,
    summarize,
)


def test_protocol_freezes_existing_gcb_pairs_without_reexecution() -> None:
    protocol = load_protocol()
    assert tuple(protocol["cases"]) == EXPECTED_CASES
    assert tuple(protocol["seeds"]) == EXPECTED_SEEDS
    assert protocol["patch_enabled"] is False
    assert protocol["reference_thresholds_used_for_decision"] is False


def test_route_parser_normalizes_current_and_historical_schedule_namespaces() -> None:
    current = parse_route(
        "positive_relation_graph_source_471276_sweeps_3_coordination_133032_cold_native_2215688_windows_15_tail_4",
        "current",
    )
    historical = parse_route(
        "positive_relation_graph_cold_warmup_609235_sweeps_3_coordination_209099_cold_continuation_2001659_sweeps_14_tail_7",
        "historical_compatible",
    )
    assert current["warmup"] == 471276
    assert current["continuation"] == 2215688
    assert historical["warmup"] == 609235
    assert historical["continuation"] == 2001659


def test_summary_never_authorizes_uniform_gcb_rollback() -> None:
    rows = []
    for case_id in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            current_route = "positive_relation_graph_source_100_sweeps_3_coordination_100_cold_native_100_windows_3_tail_0"
            historical_route = "positive_relation_graph_cold_warmup_100_sweeps_3_coordination_100_cold_continuation_100_sweeps_3_tail_0"
            for variant, route, error in (
                ("current", current_route, 10.0),
                ("historical_compatible", historical_route, 9.0),
            ):
                rows.append({
                    "case_id": case_id,
                    "run_seed": seed,
                    "variant": variant,
                    "checkpoint_hash": f"{case_id}-{seed}",
                    "action_result": {"checkpoint_hash": f"{case_id}-{seed}"},
                    "terminal_fes": 3_000_000,
                    "final_error": error,
                    "route": route,
                    "_schedule": parse_route(route, variant),
                })
    report = summarize(rows, load_protocol())
    assert report["pair_count"] == 30
    assert report["decision"]["uniform_historical_rollback_supported"] is False
    assert report["decision"]["production_gcb_change_authorized"] is False
