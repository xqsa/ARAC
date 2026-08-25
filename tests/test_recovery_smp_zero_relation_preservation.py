from __future__ import annotations

from experiments.historical_recovery.recovery_smp_zero_relation_preservation import (
    EXPECTED_SEEDS,
    load_protocol,
    summarize,
)


def _row(seed: int, *, exact: bool = True) -> dict[str, object]:
    return {
        "run_seed": seed,
        "terminal_fes": 3_000_000,
        "checkpoint_hash": "a" * 64,
        "action_result": {"checkpoint_hash": "a" * 64},
        "baseline_final_error": 1.0,
        "final_error": 1.0 if exact else 2.0,
        "exact_baseline_match": exact,
    }


def test_protocol_freezes_e1_zero_relation_preservation() -> None:
    protocol = load_protocol()
    assert protocol["case_id"] == "E1"
    assert tuple(protocol["seeds"]) == EXPECTED_SEEDS
    assert protocol["max_workers"] == 5
    assert protocol["patch_enabled"] is False


def test_summary_requires_exact_per_seed_baseline_match() -> None:
    rows = [_row(seed) for seed in EXPECTED_SEEDS]
    summary = summarize(rows, load_protocol())
    assert summary["context_count"] == 5
    assert summary["all_exact_baseline_matches"] is True
    assert summary["preservation_gate_passed"] is True

    rows[-1] = _row(EXPECTED_SEEDS[-1], exact=False)
    assert summarize(rows, load_protocol())["preservation_gate_passed"] is False
