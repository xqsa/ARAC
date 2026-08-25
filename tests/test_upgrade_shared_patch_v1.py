"""Tests for the shared_patch_v1 upgrade candidate (U0/U1/S1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.upgrade.shared_patch_v1 import s1_leverage_sweep as s1
from experiments.upgrade.shared_patch_v1.conflicting_generator import CELL_IDS, build_problem, relation_leverage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
U0_SUMMARY = REPOSITORY_ROOT / "artifacts/upgrade_u0_baseline_guard_v1/summary.json"
U1_SUMMARY = REPOSITORY_ROOT / "artifacts/upgrade_u1_host_reachability_v1/summary.json"
S1_SUMMARY = REPOSITORY_ROOT / "artifacts/upgrade_s1_leverage_sweep_v1/summary.json"


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------- U0


def test_u0_protocol_loads_without_drift() -> None:
    from experiments.upgrade.shared_patch_v1.u0_baseline_guard import DEFAULT_PROTOCOL, load_protocol

    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol["freeze_anchor"] == "arac-recovered-baseline-20260823-v1"
    assert protocol["expected_verifier"]["patch_enabled"] is False


@pytest.mark.skipif(not U0_SUMMARY.is_file(), reason="U0 summary artifact not present")
def test_u0_summary_gate_passed_and_hash_valid() -> None:
    summary = _load(U0_SUMMARY)
    assert summary["gate_passed"] is True
    assert summary["upgrade_authorized_to"] == "u1"
    claimed = summary.pop("result_hash")
    assert claimed == canonical_sha256(summary)
    assert summary["verifier_report"]["status"] == "frozen"


# ---------------------------------------------------------------- generator


def test_generator_cells_are_the_six_preregistered_cells() -> None:
    assert CELL_IDS == (
        "chain-mild",
        "chain-strong",
        "hub-mild",
        "hub-strong",
        "pairs-mild",
        "pairs-strong",
    )


def test_generator_problem_is_deterministic_and_identity_blind() -> None:
    problem_a, truth_a = build_problem("chain-strong", 117)
    problem_b, truth_b = build_problem("chain-strong", 117)
    assert truth_a.ground_truth_hash == truth_b.ground_truth_hash
    probe = np.zeros(problem_a.dimension)
    assert problem_a.objective(probe) == problem_b.objective(probe)
    assert problem_a.dimension == 1000
    assert len(truth_a.shared_variables) == 9 * 8
    assert truth_a.conflict_distance == 0.25 * 10.0 / 2.0


def test_generator_conflict_actually_splits_owner_optima() -> None:
    problem, truth = build_problem("hub-strong", 117)
    shared = truth.shared_variables[0]
    vector = np.zeros(problem.dimension)
    vector[shared] = truth.conflict_distance
    better = problem.objective(vector)
    vector[shared] = 2.0 * truth.conflict_distance
    worse = problem.objective(vector)
    assert better < worse


def test_relation_leverage_counts_incidence_only() -> None:
    from arac.runtime.contracts import RelationEvidence

    blocks = ((0, 1), (1, 2), (3,))
    relations = (
        RelationEvidence(left_block=0, right_block=1, strength=0.5, disagreement=0.0),
        RelationEvidence(left_block=0, right_block=1, strength=0.9, disagreement=0.5),
    )
    assert relation_leverage(blocks, relations) == (2, 2, 0)


# ---------------------------------------------------------------- S1 helpers


def test_head_slot_count_uses_ceiling_of_twenty_percent() -> None:
    assert s1.head_slot_count(8) == 2
    assert s1.head_slot_count(20) == 4
    assert s1.head_slot_count(1) == 1


def test_s1_order_puts_top_leverage_scopes_into_head_slots() -> None:
    leverage = (0, 3, 0, 1)
    order = s1.s1_order(4, leverage, (0, 1, 2, 3))
    assert order == (1, 0, 2, 3)
    assert order[0] == 1
    assert sorted(order) == [0, 1, 2, 3]


def test_s1_order_breaks_ties_by_original_rank() -> None:
    leverage = (2, 2, 0)
    order = s1.s1_order(3, leverage, (0, 1, 2))
    assert order == (0, 1, 2)


def test_s1_order_is_identity_when_leverage_is_zero() -> None:
    baseline = (2, 0, 3, 1)
    assert s1.s1_order(4, (0, 0, 0, 0), baseline) == tuple(baseline)


def test_s1_order_preserves_baseline_tail_order() -> None:
    baseline = (3, 0, 2, 1, 4)
    leverage = (0, 1, 0, 0, 0)
    order = s1.s1_order(5, leverage, baseline)
    assert order[0] == 1
    assert order[1:] == (3, 0, 2, 4)


def test_s1_order_rejects_incomplete_baseline() -> None:
    with pytest.raises(ValueError):
        s1.s1_order(3, (0, 0, 0), (0, 1))


def _toy_problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda vector: float(np.sum(np.asarray(vector, dtype=float) ** 2)),
        dimension=3,
        lower_bounds=(-5.0, -5.0, -5.0),
        upper_bounds=(5.0, 5.0, 5.0),
    )


def test_milestone_sampler_records_crossings_without_changing_values() -> None:
    ledger = EvaluationLedger(
        _toy_problem(),
        10,
        initial_incumbent=(1.0, 1.0, 1.0),
        initial_error=3.0,
    )
    sampler = s1.MilestoneSampler(ledger, milestones=(4, 8))
    for _ in range(4):
        ledger.evaluate(np.zeros(3) + 0.5)
    ledger.evaluate(np.zeros(3))
    for _ in range(4):
        ledger.evaluate(np.zeros(3) + 0.25)
    samples = sampler.payload()
    assert [sample["fes"] for sample in samples] == [0, 4, 8]
    assert samples[0]["best_error"] == 3.0
    assert samples[1]["best_error"] == 0.75
    assert samples[2]["best_error"] == 0.0
    assert ledger.count == 9


# ---------------------------------------------------------------- S2 handoff state


def test_handoff_prefers_remaining_neighbor_after_acceptance() -> None:
    from experiments.upgrade.shared_patch_v1.s2_propagation_handoff import HandoffState

    edges = {(0, 1), (1, 2)}
    state = HandoffState(lambda left, right: (min(left, right), max(left, right)) in edges, block_count=4)
    first, record = state.select_next({0, 1, 2, 3}, (0, 1, 2, 3))
    assert first == 0 and record["handoff_reason"] == "no_acceptance_event"
    state.report(0, improved=True)
    second, record = state.select_next({1, 2, 3}, (0, 1, 2, 3))
    assert second == 1 and record["handoff_reason"] == "neighbor_of_improved_scope"
    assert record["handoff_source_scope"] == 0 and record["shared_neighbor_count"] == 1
    state.report(1, improved=False)
    third, record = state.select_next({2, 3}, (0, 1, 2, 3))
    assert third == 2 and record["handoff_reason"] == "no_acceptance_event"


def test_handoff_falls_back_to_static_order_without_remaining_neighbors() -> None:
    from experiments.upgrade.shared_patch_v1.s2_propagation_handoff import HandoffState

    edges = {(0, 1)}
    state = HandoffState(lambda left, right: (min(left, right), max(left, right)) in edges, block_count=3)
    state.report(2, improved=True)
    selected, record = state.select_next({0, 1}, (0, 1, 2))
    assert selected == 0 and record["handoff_reason"] == "no_remaining_neighbor"
    assert record["handoff_source_scope"] == 2


def test_handoff_chain_propagates_through_improving_scopes() -> None:
    from experiments.upgrade.shared_patch_v1.s2_propagation_handoff import HandoffState

    edges = {(0, 1), (1, 2), (2, 0)}
    state = HandoffState(lambda left, right: (min(left, right), max(left, right)) in edges, block_count=3)
    state.report(0, improved=True)
    selected, _ = state.select_next({1, 2}, (0, 1, 2))
    assert selected == 1
    state.report(1, improved=True)
    selected, _ = state.select_next({2}, (0, 1, 2))
    assert selected == 2
    payload = state.payload()
    assert payload["handoff_trace_nonempty"] is True
    assert payload["handoff_selection_count"] == 2


# ---------------------------------------------------------------- instrumentation


def test_sweep_recorder_summary_uses_string_keys_and_counts_leverage() -> None:
    from experiments.upgrade.shared_patch_v1.host_instrumentation import SweepRecorder

    recorder = SweepRecorder()
    recorder.segments = [
        {
            "kind": "cold_start_sweeps",
            "per_block_visits": {"0": 2, "10": 1, "2": 3},
        },
        {
            "kind": "sequential_polish",
            "per_block_visits": {"1": 1},
        },
    ]
    summary = recorder.summary((1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1))
    assert summary["total_block_visits"] == 7
    assert summary["leverage_positive_block_visits"] == 2 + 3 + 1
    assert set(summary["per_block_visits"]) == {"0", "1", "2", "10"}


def test_recorder_receipt_hash_survives_json_round_trip() -> None:
    import json as json_module

    from experiments.upgrade.shared_patch_v1.host_instrumentation import SweepRecorder

    recorder = SweepRecorder()
    recorder.segments = [
        {
            "kind": "persistent_coverage",
            "per_block_visits": {str(index): 1 for index in range(20)},
            "block_order": list(range(20)),
        }
    ]
    body = {"segments": recorder.segments, "note": "round-trip"}
    receipt = {**body, "receipt_hash": canonical_sha256(body)}
    parsed = json_module.loads(json_module.dumps(receipt, sort_keys=True))
    rebuilt = {key: value for key, value in parsed.items() if key != "receipt_hash"}
    assert canonical_sha256(rebuilt) == receipt["receipt_hash"]


# ---------------------------------------------------------------- U1 / S1 artifacts


@pytest.mark.skipif(not U1_SUMMARY.is_file(), reason="U1 summary artifact not present")
def test_u1_reachability_table_covers_all_rows_and_gate_passed() -> None:
    summary = _load(U1_SUMMARY)
    claimed = summary.pop("result_hash")
    assert claimed == canonical_sha256(summary)
    assert summary["gate_passed"] is True
    rows = summary["reachability_table"]
    aob = [row for row in rows if row["lane"] == "aob"]
    generator = [row for row in rows if row["lane"] == "generator"]
    assert len(aob) == 24 and len(generator) == 12
    assert all(row["host_status"] == "mount_absent_by_contract" for row in aob if row["host"] is None)
    assert all(row["identity_with_frozen_screen"] is True for row in aob if row["host"] is not None)
    assert all(row["reachable"] is True for row in generator)
    assert summary["performance_comparison_authorized"] is False


@pytest.mark.skipif(not S1_SUMMARY.is_file(), reason="S1 summary artifact not present")
def test_s1_screen_summary_gate_contract() -> None:
    summary = _load(S1_SUMMARY)
    claimed = summary.pop("result_hash")
    assert claimed == canonical_sha256(summary)
    assert summary["gate_passed"] in (True, False)
    assert summary["performance_claim_authorized"] is False
    if summary["gate_passed"]:
        assert summary["s2_screen_authorized"] is True
        assert summary["checks"]["ov0_exact_no_tax"] is True
        assert summary["checks"]["routes_unchanged_all_pairs"] is True
