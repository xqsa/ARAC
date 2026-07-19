from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts import hcc_smoke_runner as runner

from arac.backends.hcc_evidence_overlay import (
    CHECKPOINT_FIELDS,
    DELAYED_OUTCOME_FIELDS,
    EVIDENCE_OVERLAY_PROTOCOL_VERSION,
    EVIDENCE_OVERLAY_SOURCE_MODE,
    PLAN_FIELDS,
    PROBE_EVIDENCE_FIELDS,
    RUNTIME_ACTION_FIELDS,
    RUNTIME_INPUT_FIELDS,
    SHADOW_DECISION_FIELDS,
    TERMINAL_TOLERANCE_RULE,
    EvidenceOverlayArtifactPaths,
    EvidenceOverlayRuntimeError,
    HccEvidenceOverlayObserver,
    RuntimeProbeActionLedger,
    RuntimeProbeConsumption,
)
from arac.policy.evidence_overlay import (
    BridgeWeights,
    LOCAL_OPTIMUM_TOP_K,
    PROPOSAL_DISAGREEMENT_METRIC,
    RelationKey,
    RuntimeProbeAction,
    runtime_probe_anchor_hash,
    runtime_probe_shared_values_hash,
)


GROUPS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
)
DIMENSION = 7
RELATION_COUNT = 5
CONFIGURED_MAX_FES = 1_000


def _observer(
    mode: str = "paired_owner",
    *,
    terminal_tolerance_fe: int = 0,
) -> HccEvidenceOverlayObserver:
    return HccEvidenceOverlayObserver(
        mode=mode,
        grouping_result=GROUPS,
        problem_id="A4",
        seed=117,
        run_id=f"runtime-{mode}",
        configured_max_fes=CONFIGURED_MAX_FES,
        terminal_tolerance_fe=terminal_tolerance_fe,
        lower_bound=-10.0,
        upper_bound=10.0,
    )


def _sweep_endpoint() -> tuple[float, ...]:
    endpoint = [0.0] * DIMENSION
    for variable in range(1, RELATION_COUNT + 1):
        endpoint[variable] = variable / 2.0
    return tuple(endpoint)


def _record_complete_sweep(
    observer: HccEvidenceOverlayObserver,
    sweep: int,
) -> tuple[float, ...]:
    for group in range(len(GROUPS)):
        start = sweep * 60 + group * 10
        before = [0.0] * DIMENSION
        proposal = before.copy()
        if group < RELATION_COUNT:
            proposal[group + 1] = float(group + 1)
        observer.record_group(
            sweep_index=sweep,
            group_index=group,
            pre_error=200.0,
            best_error=199.0 - group,
            primary_requested_fe=8,
            primary_actual_fe=8,
            full_interval_actual_fe=10,
            full_interval_start_fe=start,
            full_interval_end_fe=start + 10,
            pre_block_candidate=before,
            final_owner_candidate=proposal,
            local_top_candidates=(
                tuple(proposal[variable] for variable in GROUPS[group]),
            ),
        )

    sweep_end_fe = (sweep + 1) * 60
    endpoint = _sweep_endpoint()
    assert observer.complete_sweep(
        sweep_index=sweep,
        sweep_end_fe=sweep_end_fe,
        sweep_end_candidate=endpoint,
        sweep_end_error=180.0 - sweep,
        fitness_record=[200.0 - index / 1_000.0 for index in range(sweep_end_fe)],
        all_raw_groups_completed=True,
        native_sweep_end_completed=True,
    )
    return endpoint


def _prepare(observer: HccEvidenceOverlayObserver) -> tuple[float, ...]:
    endpoint: tuple[float, ...] = ()
    for sweep in range(3):
        endpoint = _record_complete_sweep(observer, sweep)
    assert observer.plan_ready
    return endpoint


def _objective(candidate: tuple[float, ...]) -> float:
    return 1.0 + sum(value * value for value in candidate)


def test_local_top_candidate_archive_keeps_best_five_across_batches() -> None:
    archive: list[tuple[float, tuple[float, ...]]] = []

    runner._update_local_top_candidates(
        archive,
        ((6.0, 60.0), (2.0, 20.0), (4.0, 40.0)),
        (6.0, 2.0, 4.0),
    )
    runner._update_local_top_candidates(
        archive,
        ((1.0, 10.0), (7.0, 70.0), (3.0, 30.0), (5.0, 50.0)),
        (1.0, 7.0, 3.0, 5.0),
    )

    assert [score for score, _ in archive] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [candidate[0] for _, candidate in archive] == [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(RuntimeError, match="aligned finite"):
        runner._update_local_top_candidates(archive, ((1.0, 2.0),), (1.0, 2.0))


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _write_complete_artifacts(
    observer: HccEvidenceOverlayObserver,
    root: Path,
    ledger: RuntimeProbeActionLedger | None = None,
) -> tuple[EvidenceOverlayArtifactPaths, dict[str, object]]:
    paths = EvidenceOverlayArtifactPaths(
        manifest=root / "A4_evidence_overlay_manifest.json",
        checkpoint=root / "A4_evidence_overlay_checkpoint.csv",
        plan=root / "A4_evidence_overlay_plan.csv",
        probe_evidence=root / "A4_evidence_overlay_probe_evidence.csv",
        delayed_outcomes=root / "A4_evidence_overlay_delayed_outcomes.csv",
        shadow_decisions=root / "A4_evidence_overlay_shadow_decisions.csv",
        runtime_actions=root / "A4_evidence_overlay_runtime_actions.csv",
    )
    manifest = observer.write_artifacts(
        paths=paths,
        native_terminal_error=70.0,
        all_evaluation_best_error=60.0,
        runtime_action_ledger=ledger,
    )
    return paths, manifest


def _runtime_action(
    *,
    winner: str = "left_owner",
    shared_values: tuple[float, ...] = (1.25,),
    anchor_values: tuple[float, ...] = (0.5,),
) -> RuntimeProbeAction:
    relation = RelationKey((0, 1), (1,))
    return RuntimeProbeAction(
        relation=relation,
        winner=winner,
        shared_values=shared_values,
        shared_values_hash=runtime_probe_shared_values_hash(
            relation,
            shared_values,
        ),
        candidate_hash="a" * 64,
        bridge_weights=BridgeWeights(0.4, 0.6),
        utility=0.2,
        anchor_hash=runtime_probe_anchor_hash(relation, anchor_values),
        checkpoint_fe=180,
        checkpoint_hash="b" * 64,
        issued_sweep=2,
        ttl_sweeps=1,
        expires_sweep=3,
    )


def test_runtime_probe_action_ledger_is_local_next_sweep_and_one_shot() -> None:
    action = _runtime_action()
    ledger = RuntimeProbeActionLedger()
    ledger.issue((action,))
    writes: list[tuple[float, ...]] = []

    early = ledger.consume(
        action=ledger.action_for(action.relation),
        relation=action.relation,
        anchor_hash=action.anchor_hash,
        checkpoint_hash=action.checkpoint_hash,
        current_sweep=2,
        current_fe=181,
        write_shared_values=writes.append,
    )
    assert not early.consumed
    assert early.reason == "not_next_sweep"
    assert writes == []

    consumed = ledger.consume(
        action=ledger.action_for(action.relation),
        relation=action.relation,
        anchor_hash=action.anchor_hash,
        checkpoint_hash=action.checkpoint_hash,
        current_sweep=3,
        current_fe=200,
        write_shared_values=writes.append,
    )
    assert consumed == RuntimeProbeConsumption(action, True, "consumed")
    assert writes == [action.shared_values]

    repeated = ledger.consume(
        action=ledger.action_for(action.relation),
        relation=action.relation,
        anchor_hash=action.anchor_hash,
        checkpoint_hash=action.checkpoint_hash,
        current_sweep=3,
        current_fe=201,
        write_shared_values=writes.append,
    )
    assert not repeated.consumed
    assert repeated.reason == "already_consumed"
    assert writes == [action.shared_values]


def test_runtime_probe_action_ttl_is_frozen_to_one_sweep() -> None:
    with pytest.raises(ValueError, match="ttl_sweeps=1"):
        replace(_runtime_action(), ttl_sweeps=2, expires_sweep=4)


def test_runtime_probe_anchor_ignores_unrelated_values_and_invalidates_shared_change() -> None:
    action = _runtime_action()
    original = np.array([9.0, 0.5, 8.0])
    unrelated_change = np.array([-9.0, 0.5, -8.0])
    shared_change = np.array([9.0, 0.6, 8.0])

    assert runtime_probe_anchor_hash(action.relation, original[[1]]) == (
        runtime_probe_anchor_hash(action.relation, unrelated_change[[1]])
    )
    assert runtime_probe_anchor_hash(action.relation, shared_change[[1]]) != (
        action.anchor_hash
    )

    ledger = RuntimeProbeActionLedger()
    ledger.issue((action,))
    mismatch = ledger.consume(
        action=ledger.action_for(action.relation),
        relation=action.relation,
        anchor_hash=runtime_probe_anchor_hash(action.relation, shared_change[[1]]),
        checkpoint_hash=action.checkpoint_hash,
        current_sweep=3,
        current_fe=200,
        write_shared_values=lambda _: pytest.fail("anchor mismatch wrote values"),
    )
    assert not mismatch.consumed
    assert mismatch.reason == "anchor_mismatch"
    assert ledger.records[0].status == "abstained"


@pytest.mark.parametrize(
    ("checkpoint_hash", "current_sweep", "reason"),
    [("c" * 64, 3, "checkpoint_mismatch"), ("b" * 64, 4, "ttl_expired")],
)
def test_runtime_probe_action_invalidates_checkpoint_or_ttl(
    checkpoint_hash: str,
    current_sweep: int,
    reason: str,
) -> None:
    action = _runtime_action()
    ledger = RuntimeProbeActionLedger()
    ledger.issue((action,))

    result = ledger.consume(
        action=ledger.action_for(action.relation),
        relation=action.relation,
        anchor_hash=action.anchor_hash,
        checkpoint_hash=checkpoint_hash,
        current_sweep=current_sweep,
        current_fe=200,
        write_shared_values=lambda _: pytest.fail(f"{reason} wrote values"),
    )
    assert not result.consumed
    assert result.reason == reason


def test_runtime_probe_uses_saved_bridge_not_current_delta_blend() -> None:
    action = _runtime_action(winner="bridge", shared_values=(0.75,))
    ledger = RuntimeProbeActionLedger()
    ledger.issue((action,))
    writes: list[tuple[float, ...]] = []

    consumed = ledger.consume(
        action=ledger.action_for(action.relation),
        relation=action.relation,
        anchor_hash=action.anchor_hash,
        checkpoint_hash=action.checkpoint_hash,
        current_sweep=3,
        current_fe=200,
        write_shared_values=writes.append,
    )
    assert consumed.consumed
    assert writes == [(0.75,)]


def test_runtime_probe_uses_saved_owner_when_phase2_delta_prefers_other_owner() -> None:
    action = _runtime_action(winner="left_owner", shared_values=(1.25,))
    phase2_previous_delta = -1.0
    phase2_current_delta = 10.0
    assert phase2_current_delta > phase2_previous_delta
    ledger = RuntimeProbeActionLedger()
    ledger.issue((action,))
    writes: list[tuple[float, ...]] = []

    consumed = ledger.consume(
        action=ledger.action_for(action.relation),
        relation=action.relation,
        anchor_hash=action.anchor_hash,
        checkpoint_hash=action.checkpoint_hash,
        current_sweep=3,
        current_fe=200,
        write_shared_values=writes.append,
    )
    assert consumed.consumed
    assert writes == [(1.25,)]


def test_runtime_probe_relation_mismatch_and_write_failure_abstain() -> None:
    action = _runtime_action()
    mismatch_relation = RelationKey((0, 2), (1,))
    ledger = RuntimeProbeActionLedger()
    ledger.issue((action,))

    mismatch = ledger.consume(
        action=action,
        relation=mismatch_relation,
        anchor_hash=action.anchor_hash,
        checkpoint_hash=action.checkpoint_hash,
        current_sweep=3,
        current_fe=200,
        write_shared_values=lambda _: pytest.fail("relation mismatch wrote values"),
    )
    assert not mismatch.consumed
    assert mismatch.reason == "relation_mismatch"
    assert ledger.records[0].invalidation_reason == "relation_mismatch"

    ledger.issue((action,))

    def fail_write(_: tuple[float, ...]) -> None:
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        ledger.consume(
            action=action,
            relation=action.relation,
            anchor_hash=action.anchor_hash,
            checkpoint_hash=action.checkpoint_hash,
            current_sweep=3,
            current_fe=200,
            write_shared_values=fail_write,
        )
    assert not ledger.records[0].runtime_consumed
    assert ledger.records[0].status == "abstained"
    assert ledger.records[0].invalidation_reason == "write_failed"


def test_plan_requires_three_complete_sweeps_with_prior_credit_closed(
    tmp_path: Path,
) -> None:
    observer = _observer("native_audit")

    _record_complete_sweep(observer, 0)
    _record_complete_sweep(observer, 1)
    assert not observer.plan_ready

    _record_complete_sweep(observer, 2)
    assert observer.plan_ready
    paths, _ = _write_complete_artifacts(observer, tmp_path)
    _, checkpoint_rows = _read_csv(paths.checkpoint)

    assert len(checkpoint_rows) == 1
    assert checkpoint_rows[0]["checkpoint_fe"] == "180"
    assert checkpoint_rows[0]["phase_boundary_fe"] == "180"
    assert checkpoint_rows[0]["history_sweeps"] == "0;1;2"
    assert checkpoint_rows[0]["previous_survival_closed"] == "1"

    gapped = _observer("native_audit")
    _record_complete_sweep(gapped, 0)
    assert gapped.consecutive_complete_sweep_count == 1
    assert not gapped.complete_sweep(
        sweep_index=1,
        sweep_end_fe=61,
        sweep_end_candidate=_sweep_endpoint(),
        sweep_end_error=179.0,
        fitness_record=[200.0] * 61,
        all_raw_groups_completed=False,
        native_sweep_end_completed=False,
    )
    assert gapped.consecutive_complete_sweep_count == 0
    _record_complete_sweep(gapped, 2)
    assert gapped.consecutive_complete_sweep_count == 1
    _record_complete_sweep(gapped, 3)
    assert gapped.consecutive_complete_sweep_count == 2
    assert not gapped.plan_ready
    _record_complete_sweep(gapped, 4)
    assert gapped.consecutive_complete_sweep_count == 3
    assert gapped.plan_ready


def test_native_audit_and_insufficient_budget_make_zero_objective_calls() -> None:
    calls: list[tuple[float, ...]] = []

    def counted(candidate: tuple[float, ...]) -> float:
        calls.append(candidate)
        return _objective(candidate)

    audit = _observer("native_audit")
    audit_anchor = _prepare(audit)
    audit_result = audit.execute_barrier(
        counted,
        audit_anchor,
        remaining_fe=100,
        normal_sweep_fe=60,
        tolerance_fe=0,
    )
    assert audit_result.status == "abstained"
    assert audit_result.reason == "native_audit_zero_probe_fe"
    assert audit_result.actual_fe == 0
    assert calls == []

    paired = _observer("paired_owner")
    paired_anchor = _prepare(paired)
    budget_result = paired.execute_barrier(
        counted,
        paired_anchor,
        remaining_fe=75,
        normal_sweep_fe=60,
        tolerance_fe=0,
    )
    assert budget_result.status == "abstained"
    assert budget_result.reason == "insufficient_budget_for_probe_and_native_sweep"
    assert budget_result.requested_fe == 16
    assert budget_result.actual_fe == 0
    assert calls == []


def test_paired_and_shuffled_use_exact_fe_and_select_different_top4(
    tmp_path: Path,
) -> None:
    selected_by_mode: dict[str, set[str]] = {}
    checkpoint_by_mode: dict[str, dict[str, str]] = {}

    for mode in ("paired_owner", "shuffled_owner"):
        observer = _observer(mode)
        anchor = list(_prepare(observer))
        frozen_anchor = tuple(anchor)
        result = observer.execute_barrier(
            _objective,
            anchor,
            remaining_fe=100,
            normal_sweep_fe=60,
            tolerance_fe=0,
        )

        assert result.status == "probed"
        assert result.requested_fe == result.actual_fe == 16
        assert result.selected_relation_count == 4
        assert observer.evidence_overlay_fe == 16
        assert tuple(anchor) == frozen_anchor
        assert result.anchor_unchanged
        observer.record_runtime_audit(
            fingerprints_before={"runtime": "a" * 64},
            fingerprints_after={"runtime": "a" * 64},
            probe_start_fe=180,
            probe_end_fe=196,
        )

        _record_complete_sweep(observer, 3)
        paths, manifest = _write_complete_artifacts(observer, tmp_path / mode)
        _, plan_rows = _read_csv(paths.plan)
        _, checkpoint_rows = _read_csv(paths.checkpoint)
        selected_by_mode[mode] = {
            row["relation_id"] for row in plan_rows if row["selected"] == "1"
        }
        checkpoint_by_mode[mode] = checkpoint_rows[0]
        assert manifest["applicable"] == 1
        assert manifest["abstain_reason"] == ""

    assert len(selected_by_mode["paired_owner"]) == 4
    assert len(selected_by_mode["shuffled_owner"]) == 4
    assert selected_by_mode["paired_owner"] != selected_by_mode["shuffled_owner"]
    for field in (
        "checkpoint_fe",
        "fitness_prefix_hash",
        "incumbent_hash",
        "rddsm_topology_hash",
        "rddsm_order_hash",
        "phase_boundary_fe",
    ):
        assert checkpoint_by_mode["paired_owner"][field] == (
            checkpoint_by_mode["shuffled_owner"][field]
        )


def test_objective_failure_is_one_shot_and_fail_closed(tmp_path: Path) -> None:
    observer = _observer()
    anchor = _prepare(observer)
    calls = 0

    def failing(candidate: tuple[float, ...]) -> float:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("synthetic probe failure")
        return _objective(candidate)

    with pytest.raises(EvidenceOverlayRuntimeError, match="failed after probe admission"):
        observer.execute_barrier(
            failing,
            anchor,
            remaining_fe=100,
            normal_sweep_fe=60,
            tolerance_fe=0,
        )

    assert calls == observer.evidence_overlay_fe == 5
    assert observer.barrier_result is not None
    assert observer.barrier_result.status == "failed"
    assert observer.barrier_result.actual_fe == 5
    with pytest.raises(ValueError, match="only once"):
        observer.execute_barrier(
            _objective,
            anchor,
            remaining_fe=100,
            normal_sweep_fe=60,
            tolerance_fe=0,
        )

    _, manifest = _write_complete_artifacts(observer, tmp_path)
    assert manifest["applicable"] == 0
    assert manifest["abstain_reason"] == "four_point_probe_objective_failure"
    assert manifest["observer_integrity"] == 0
    assert manifest["failure"] == {
        "stage": "four_point_probe",
        "error_type": "RuntimeError",
        "error_message": "synthetic probe failure",
        "objective_calls": "5",
    }


def test_anchor_invariant_failure_is_also_one_shot() -> None:
    observer = _observer()
    anchor = list(_prepare(observer))
    calls = 0

    def mutating(candidate: tuple[float, ...]) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            anchor[0] = 99.0
        return _objective(candidate)

    with pytest.raises(EvidenceOverlayRuntimeError, match="mutated the caller anchor"):
        observer.execute_barrier(
            mutating,
            anchor,
            remaining_fe=100,
            normal_sweep_fe=60,
            tolerance_fe=0,
        )

    assert calls == 16
    assert observer.barrier_result is not None
    assert observer.barrier_result.status == "failed"
    assert observer.barrier_result.reason == "four_point_probe_integrity_failure"
    assert not observer.barrier_result.anchor_unchanged
    with pytest.raises(ValueError, match="only once"):
        observer.execute_barrier(
            _objective,
            _sweep_endpoint(),
            remaining_fe=100,
            normal_sweep_fe=60,
            tolerance_fe=0,
        )


def test_repeated_x0_fitness_must_be_exactly_deterministic() -> None:
    observer = _observer()
    anchor = _prepare(observer)
    calls = 0

    def stateful(candidate: tuple[float, ...]) -> float:
        nonlocal calls
        calls += 1
        value = _objective(candidate)
        return value + calls * 1e-6 if candidate == anchor else value

    with pytest.raises(EvidenceOverlayRuntimeError, match="x0 fitness"):
        observer.execute_barrier(
            stateful,
            anchor,
            remaining_fe=100,
            normal_sweep_fe=60,
            tolerance_fe=0,
        )

    assert calls == 16
    assert observer.barrier_result is not None
    assert observer.barrier_result.status == "failed"
    assert observer.barrier_result.reason == "four_point_probe_integrity_failure"
    assert observer.barrier_result.anchor_unchanged


def test_incomplete_next_sweep_does_not_close_delayed_labels(tmp_path: Path) -> None:
    observer = _observer()
    anchor = _prepare(observer)
    observer.execute_barrier(
        _objective,
        anchor,
        remaining_fe=100,
        normal_sweep_fe=60,
        tolerance_fe=0,
    )
    observer.record_runtime_audit(
        fingerprints_before={"runtime": "a" * 64},
        fingerprints_after={"runtime": "a" * 64},
        probe_start_fe=180,
        probe_end_fe=196,
    )

    assert not observer.complete_sweep(
        sweep_index=3,
        sweep_end_fe=181,
        sweep_end_candidate=anchor,
        sweep_end_error=177.0,
        fitness_record=[100.0] * 181,
        all_raw_groups_completed=False,
        native_sweep_end_completed=False,
    )
    paths, manifest = _write_complete_artifacts(observer, tmp_path)
    _, delayed_rows = _read_csv(paths.delayed_outcomes)

    assert len(delayed_rows) == 8
    assert all(row["label_closed"] == "0" for row in delayed_rows)
    assert all(
        row["label_status"] == "incomplete_next_native_sweep"
        for row in delayed_rows
    )
    assert all(row["survival_label"] == "" for row in delayed_rows)
    assert manifest["delayed_label_closed"] == 0
    assert manifest["observer_integrity"] == 0
    assert manifest["applicable"] == 0


def test_artifact_schemas_hashes_and_reference_blind_manifest_are_frozen(
    tmp_path: Path,
) -> None:
    observer = _observer(terminal_tolerance_fe=7)
    anchor = _prepare(observer)
    observer.execute_barrier(
        _objective,
        anchor,
        remaining_fe=100,
        normal_sweep_fe=60,
        tolerance_fe=7,
    )
    observer.record_runtime_audit(
        fingerprints_before={"best_individual": "a" * 64},
        fingerprints_after={"best_individual": "a" * 64},
        probe_start_fe=180,
        probe_end_fe=196,
    )
    _record_complete_sweep(observer, 3)
    paths, manifest = _write_complete_artifacts(observer, tmp_path)

    checkpoint_fields, checkpoint_rows = _read_csv(paths.checkpoint)
    plan_fields, plan_rows = _read_csv(paths.plan)
    probe_fields, probe_rows = _read_csv(paths.probe_evidence)
    delayed_fields, delayed_rows = _read_csv(paths.delayed_outcomes)
    shadow_fields, shadow_rows = _read_csv(paths.shadow_decisions)
    runtime_fields, runtime_rows = _read_csv(paths.runtime_actions)

    assert checkpoint_fields == CHECKPOINT_FIELDS
    assert plan_fields == PLAN_FIELDS
    assert probe_fields == PROBE_EVIDENCE_FIELDS
    assert delayed_fields == DELAYED_OUTCOME_FIELDS
    assert shadow_fields == SHADOW_DECISION_FIELDS
    assert runtime_fields == RUNTIME_ACTION_FIELDS
    assert runtime_rows == []
    assert len(checkpoint_rows) == 1
    assert len(plan_rows) == 5
    assert sum(row["selected"] == "1" for row in plan_rows) == 4
    assert all(
        row["disagreement_metric"] == PROPOSAL_DISAGREEMENT_METRIC
        for row in plan_rows
    )
    assert all(row["left_top_k_count"] == "1" for row in plan_rows)
    assert all(row["right_top_k_count"] == "1" for row in plan_rows)
    assert len(probe_rows) == 16
    assert Counter(row["candidate"] for row in probe_rows) == {
        "x0": 4,
        "left_owner": 4,
        "right_owner": 4,
        "bridge": 4,
    }
    assert len(delayed_rows) == 8
    assert Counter(row["owner"] for row in delayed_rows) == {"left": 4, "right": 4}
    assert all(row["label_closed"] == "1" for row in delayed_rows)
    assert len(shadow_rows) == 4
    for rows in (
        checkpoint_rows,
        plan_rows,
        probe_rows,
        delayed_rows,
        shadow_rows,
    ):
        assert all(row["runtime_authorized"] == "0" for row in rows)

    checkpoint = checkpoint_rows[0]
    for field in (
        "fitness_prefix_hash",
        "incumbent_hash",
        "rddsm_topology_hash",
        "rddsm_order_hash",
    ):
        assert len(checkpoint[field]) == 64
        int(checkpoint[field], 16)

    required_manifest_fields = {
        "protocol_version",
        "problem_id",
        "seed",
        "evidence_overlay_mode",
        "configured_max_fes",
        "source_mode",
        "terminal_tolerance_rule",
        "terminal_tolerance_fe",
        "native_terminal_error",
        "all_evaluation_best_error",
        "runtime_authorized",
        "runtime_input_fields",
        "proposal_disagreement_metric",
        "local_optimum_top_k",
        "artifacts",
        "artifact_sha256",
        "applicable",
        "abstain_reason",
    }
    assert required_manifest_fields <= set(manifest)
    assert manifest["protocol_version"] == EVIDENCE_OVERLAY_PROTOCOL_VERSION
    assert manifest["source_mode"] == EVIDENCE_OVERLAY_SOURCE_MODE
    assert manifest["terminal_tolerance_rule"] == TERMINAL_TOLERANCE_RULE
    assert manifest["terminal_tolerance_fe"] == 7
    assert manifest["evidence_overlay_mode"] == "paired_owner"
    assert manifest["configured_max_fes"] == CONFIGURED_MAX_FES
    assert manifest["runtime_authorized"] == 0
    assert manifest["aob_truth_runtime_used"] == 0
    assert len(manifest["runtime_fingerprint_before"]) == 64
    assert manifest["runtime_fingerprint_before"] == manifest[
        "runtime_fingerprint_after"
    ]
    assert manifest["state_fingerprints"] == {
        "best_individual": {"before": "a" * 64, "after": "a" * 64}
    }
    assert manifest["native_state_unchanged"] == 1
    assert manifest["probe_start_fe"] == 180
    assert manifest["probe_end_fe"] == 196
    assert manifest["runtime_input_fields"] == list(RUNTIME_INPUT_FIELDS)
    assert manifest["proposal_disagreement_metric"] == PROPOSAL_DISAGREEMENT_METRIC
    assert manifest["local_optimum_top_k"] == LOCAL_OPTIMUM_TOP_K
    assert manifest["native_terminal_error"] == 70.0
    assert manifest["all_evaluation_best_error"] == 60.0
    assert manifest["delayed_label_expected"] == 8
    assert manifest["delayed_label_closed"] == 8
    assert manifest["observer_integrity"] == 1
    assert not {
        "native_terminal_error",
        "all_evaluation_best_error",
    }.intersection(manifest["runtime_input_fields"])

    artifact_paths = {
        path.name: path
        for path in (
            paths.checkpoint,
            paths.plan,
            paths.probe_evidence,
            paths.delayed_outcomes,
            paths.shadow_decisions,
            paths.runtime_actions,
        )
    }
    assert manifest["artifacts"] == {
        "checkpoint": paths.checkpoint.name,
        "plan": paths.plan.name,
        "probe_evidence": paths.probe_evidence.name,
        "delayed_outcomes": paths.delayed_outcomes.name,
        "shadow_decisions": paths.shadow_decisions.name,
        "runtime_actions": paths.runtime_actions.name,
    }
    assert set(manifest["artifact_sha256"]) == set(artifact_paths)
    assert manifest["artifact_sha256"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in artifact_paths.items()
    }
    assert json.loads(paths.manifest.read_text(encoding="utf-8")) == manifest


def test_runtime_action_artifact_reports_authorization_and_consumption_truth(
    tmp_path: Path,
) -> None:
    observer = _observer()
    anchor = _prepare(observer)
    observer.execute_barrier(
        _objective,
        anchor,
        remaining_fe=100,
        normal_sweep_fe=60,
        tolerance_fe=0,
    )
    observer.record_runtime_audit(
        fingerprints_before={"runtime": "a" * 64},
        fingerprints_after={"runtime": "a" * 64},
        probe_start_fe=180,
        probe_end_fe=196,
    )
    actions = observer.runtime_probe_actions
    assert actions
    ledger = RuntimeProbeActionLedger()
    ledger.issue(actions)
    first = actions[0]
    result = ledger.consume(
        action=ledger.action_for(first.relation),
        relation=first.relation,
        anchor_hash=first.anchor_hash,
        checkpoint_hash=first.checkpoint_hash,
        current_sweep=first.expires_sweep,
        current_fe=196,
        write_shared_values=lambda _: None,
    )
    assert result.consumed

    _record_complete_sweep(observer, 3)
    paths, manifest = _write_complete_artifacts(observer, tmp_path, ledger)
    fields, rows = _read_csv(paths.runtime_actions)
    assert fields == RUNTIME_ACTION_FIELDS
    assert len(rows) == len(actions)
    expected_relation_id = "g{}:v{}".format(
        "-".join(str(value) for value in first.relation.owner_group_indices),
        "-".join(str(value) for value in first.relation.shared_variable_indices),
    )
    consumed = next(
        row for row in rows if row["relation_id"] == expected_relation_id
    )
    assert consumed["runtime_authorized"] == "1"
    assert consumed["runtime_consumed"] == "1"
    assert consumed["status"] == "consumed"
    assert consumed["consumed_sweep"] == str(first.expires_sweep)
    abstained = [row for row in rows if row["runtime_consumed"] == "0"]
    assert all(row["status"] == "abstained" for row in abstained)
    assert all(row["invalidation_reason"] == "not_dispatched" for row in abstained)
    assert manifest["runtime_authorized"] == 1
    assert manifest["runtime_consumed"] == 1
    assert manifest["runtime_actions_issued"] == len(actions)
    assert manifest["runtime_actions_consumed"] == 1
    assert manifest["runtime_actions_abstained"] == len(actions) - 1


def test_terminal_tolerance_is_frozen_even_without_sweeps_or_barrier(
    tmp_path: Path,
) -> None:
    observer = HccEvidenceOverlayObserver(
        mode="native_audit",
        grouping_result=((0,), (1,)),
        problem_id="E1",
        seed=117,
        run_id="runtime-native-audit-e1",
        configured_max_fes=CONFIGURED_MAX_FES,
        terminal_tolerance_fe=23,
        lower_bound=-10.0,
        upper_bound=10.0,
    )
    manifest = observer.write_artifacts(
        paths=EvidenceOverlayArtifactPaths.under(tmp_path),
        native_terminal_error=70.0,
        all_evaluation_best_error=70.0,
    )

    assert manifest["terminal_tolerance_fe"] == 23
    assert manifest["barrier_status"] == "not_executed"
    assert manifest["applicable"] == 0
    assert manifest["abstain_reason"] == "barrier_not_executed"

    prepared = _observer(terminal_tolerance_fe=7)
    anchor = _prepare(prepared)
    with pytest.raises(ValueError, match="frozen terminal tolerance"):
        prepared.execute_barrier(
            _objective,
            anchor,
            remaining_fe=100,
            normal_sweep_fe=60,
            tolerance_fe=0,
        )
    assert prepared.barrier_result is None
    assert prepared.evidence_overlay_fe == 0


def test_checkpoint_history_stays_frozen_after_delayed_and_later_sweeps(
    tmp_path: Path,
) -> None:
    observer = _observer(terminal_tolerance_fe=7)
    anchor = _prepare(observer)
    observer.execute_barrier(
        _objective,
        anchor,
        remaining_fe=100,
        normal_sweep_fe=60,
        tolerance_fe=7,
    )
    assert observer.delayed_outcomes_pending
    _record_complete_sweep(observer, 3)
    assert not observer.delayed_outcomes_pending
    _record_complete_sweep(observer, 4)

    paths = EvidenceOverlayArtifactPaths.under(tmp_path)
    observer.write_artifacts(
        paths=paths,
        native_terminal_error=70.0,
        all_evaluation_best_error=60.0,
    )
    rows = list(csv.DictReader(paths.checkpoint.open(encoding="utf-8")))

    assert rows[0]["history_sweeps"] == "0;1;2"


def test_observer_rejects_nonfresh_runs_and_invalid_configured_budget() -> None:
    with pytest.raises(ValueError, match="fresh optimizer execution"):
        HccEvidenceOverlayObserver(
            mode="paired_owner",
            grouping_result=GROUPS,
            problem_id="A4",
            seed=117,
            run_id="stale",
            configured_max_fes=CONFIGURED_MAX_FES,
            terminal_tolerance_fe=0,
            lower_bound=-10.0,
            upper_bound=10.0,
            fresh_optimizer_execution=False,
        )
    with pytest.raises(ValueError, match="configured_max_fes"):
        HccEvidenceOverlayObserver(
            mode="paired_owner",
            grouping_result=GROUPS,
            problem_id="A4",
            seed=117,
            run_id="zero-budget",
            configured_max_fes=0,
            terminal_tolerance_fe=0,
            lower_bound=-10.0,
            upper_bound=10.0,
        )


def test_active_grouping_is_reference_blind_and_off_keeps_legacy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design_matrix = np.eye(4, dtype=float)

    class FakeDecomposition:
        def __init__(self, matrix: np.ndarray) -> None:
            assert np.array_equal(matrix, design_matrix)

        def decomposition(self) -> list[list[int]]:
            return [[2, 3], [1, 2], [0, 1]]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("active grouping touched an oracle/AOB path")

    monkeypatch.setattr(runner, "Decomposition", FakeDecomposition)
    monkeypatch.setattr(
        runner,
        "load_reference_blind_design_matrix",
        lambda _fun_id, _data_root: design_matrix,
    )
    for name in (
        "decompose_problem",
        "load_design_matrix",
        "build_aob_topology_groups",
        "load_aob_metadata",
        "load_permutation_vector",
        "order_grouping_by_aob_topology",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    active = runner.load_runtime_grouping(
        2,
        Path("unused"),
        evidence_overlay_mode="paired_owner",
    )
    assert active == [[0, 1], [1, 2], [2, 3]]

    legacy = [[9, 10]]
    monkeypatch.setattr(runner, "decompose_problem", lambda *_args: legacy)
    assert runner.load_runtime_grouping(
        2,
        Path("unused"),
        evidence_overlay_mode="off",
    ) is legacy


@pytest.mark.parametrize("fun_id", (1, 3, 4, 5))
def test_real_overlay_grouping_reads_only_the_ragged_design_matrix(
    fun_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    path_open = Path.open

    def audited_open(path: Path, *args: object, **kwargs: object):
        opened.append(path.name)
        if path.name.endswith(("-info.txt", "-s.txt", "-p.txt")):
            raise AssertionError("active overlay grouping opened AOB truth metadata")
        return path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    groups = runner.load_runtime_grouping(
        fun_id,
        runner.DATA_DIR,
        evidence_overlay_mode="paired_owner",
    )

    assert [name for name in opened if name.startswith(f"F{fun_id}-")] == [
        f"F{fun_id}-design.txt"
    ]
    assert set().union(*(set(group) for group in groups)) == set(range(1000))
    assert len(groups) == 20


def test_reference_blind_design_reconstruction_fails_without_a_unique_paradigm(
    tmp_path: Path,
) -> None:
    design = tmp_path / "F1-design.txt"
    design.write_text(
        "1,1,1,0\n"
        "1,1,1,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple paradigms"):
        runner.load_reference_blind_design_matrix(1, tmp_path)


def test_overlay_schedule_freezes_slots_and_accounts_for_prechecks() -> None:
    overlap_kwargs = {
        "mode": "paired_owner",
        "has_overlap": True,
        "cc_budget_limit_fe": 1_000,
        "current_fe": 100,
        "terminal_tolerance_fe": 20,
        "sub_num": 2,
        "population_sizes": (4, 6),
    }
    frozen = runner.evidence_overlay_scheduled_sub_fes(
        complete_sweep_count=0,
        frozen_sub_fes=None,
        **overlap_kwargs,
    )
    assert frozen == 66
    available = 1_000 - 100 - 16 - 20
    assert 4 * runner.evidence_overlay_normal_sweep_reserve(
        (4, 6),
        sub_fes=frozen,
    ) <= available
    assert 4 * runner.evidence_overlay_normal_sweep_reserve(
        (4, 6),
        sub_fes=frozen + 1,
    ) > available

    for complete_sweep_count in (1, 2, 3):
        assert runner.evidence_overlay_scheduled_sub_fes(
            complete_sweep_count=complete_sweep_count,
            frozen_sub_fes=frozen,
            **overlap_kwargs,
        ) == frozen
    assert runner.evidence_overlay_scheduled_sub_fes(
        complete_sweep_count=4,
        frozen_sub_fes=frozen,
        **overlap_kwargs,
    ) is None

    normal_sweep_fe = runner.evidence_overlay_normal_sweep_reserve(
        (4, 6),
        sub_fes=frozen,
    )
    assert runner.evidence_overlay_scheduled_sub_fes(
        complete_sweep_count=3,
        frozen_sub_fes=frozen,
        current_fe=100 + 3 * normal_sweep_fe + 16,
        probe_pending=False,
        **{
            key: value
            for key, value in overlap_kwargs.items()
            if key != "current_fe"
        },
    ) == frozen
    terminal_remaining = 1_000 - (
        100 + 4 * normal_sweep_fe + 16
    )
    assert terminal_remaining >= 20

    no_overlap = dict(overlap_kwargs)
    no_overlap.update({"mode": "native_audit", "has_overlap": False})
    e1_frozen = runner.evidence_overlay_scheduled_sub_fes(
        complete_sweep_count=0,
        frozen_sub_fes=None,
        **no_overlap,
    )
    assert e1_frozen == 90
    e1_available = 1_000 - 100 - 20
    assert 3 * runner.evidence_overlay_normal_sweep_reserve(
        (4, 6),
        sub_fes=e1_frozen,
    ) <= e1_available
    assert 3 * runner.evidence_overlay_normal_sweep_reserve(
        (4, 6),
        sub_fes=e1_frozen + 1,
    ) > e1_available
    assert runner.evidence_overlay_scheduled_sub_fes(
        complete_sweep_count=3,
        frozen_sub_fes=e1_frozen,
        **no_overlap,
    ) is None
    assert runner.evidence_overlay_normal_sweep_reserve((4, 6)) == 12
    assert runner.evidence_overlay_group_interval_reserve(10, 10) == 41
    assert runner.evidence_overlay_group_interval_reserve(4, 100) == 161


def test_overlay_schedule_fails_closed_when_population_cannot_fit() -> None:
    with pytest.raises(RuntimeError, match="insufficient budget"):
        runner.evidence_overlay_scheduled_sub_fes(
            mode="paired_owner",
            has_overlap=True,
            complete_sweep_count=0,
            cc_budget_limit_fe=200,
            current_fe=100,
            terminal_tolerance_fe=0,
            sub_num=2,
            population_sizes=(64, 64),
            frozen_sub_fes=None,
        )
    with pytest.raises(RuntimeError, match="sub-population"):
        runner.evidence_overlay_scheduled_sub_fes(
            mode="paired_owner",
            has_overlap=True,
            complete_sweep_count=0,
            cc_budget_limit_fe=1_000,
            current_fe=100,
            terminal_tolerance_fe=0,
            sub_num=2,
            population_sizes=(4, 6),
            frozen_sub_fes=5,
        )

    with pytest.raises(RuntimeError, match="remaining reserve"):
        runner.evidence_overlay_scheduled_sub_fes(
            mode="paired_owner",
            has_overlap=True,
            complete_sweep_count=1,
            cc_budget_limit_fe=1_000,
            current_fe=800,
            terminal_tolerance_fe=20,
            sub_num=2,
            population_sizes=(4, 6),
            frozen_sub_fes=66,
        )


def test_overlay_schedule_requires_boolean_probe_pending() -> None:
    with pytest.raises(ValueError, match="probe_pending"):
        runner.evidence_overlay_scheduled_sub_fes(
            mode="paired_owner",
            has_overlap=True,
            complete_sweep_count=0,
            cc_budget_limit_fe=1_000,
            current_fe=100,
            terminal_tolerance_fe=20,
            sub_num=2,
            population_sizes=(4, 6),
            frozen_sub_fes=None,
            probe_pending=1,
        )


def test_overlay_schedule_does_not_exit_before_observer_plan_is_ready() -> None:
    assert runner.evidence_overlay_scheduled_sub_fes(
        mode="paired_owner",
        has_overlap=True,
        complete_sweep_count=4,
        cc_budget_limit_fe=1_000,
        current_fe=700,
        terminal_tolerance_fe=20,
        sub_num=2,
        population_sizes=(4, 6),
        frozen_sub_fes=20,
        plan_ready=False,
    ) == 20


def test_overlay_schedule_stops_after_delayed_closure_even_if_streak_resets() -> None:
    assert runner.evidence_overlay_scheduled_sub_fes(
        mode="paired_owner",
        has_overlap=True,
        complete_sweep_count=0,
        cc_budget_limit_fe=1_000,
        current_fe=990,
        terminal_tolerance_fe=0,
        sub_num=2,
        population_sizes=(4, 6),
        frozen_sub_fes=66,
        plan_ready=True,
        probe_pending=False,
        barrier_attempted=True,
        delayed_outcomes_pending=False,
    ) is None


def test_overlay_runtime_fingerprints_are_stable_and_detect_mutations() -> None:
    controller = runner.EvidenceActionControllerV31RunState(dense_overlap=False)
    best = np.zeros(4, dtype=float)
    incumbent = np.ones(4, dtype=float)
    kwargs = {
        "best_individual": best,
        "guarded_incumbent": incumbent,
        "guarded_incumbent_fitness": 3.0,
        "grouping_result": [[0, 1], [1, 2]],
        "controller": controller,
        "trajectory_mean_cache": {0: 0.5},
        "previous_group_contribution_credit": [0.1, 0.2],
    }
    baseline = runner.evidence_overlay_runtime_fingerprints(**kwargs)
    assert baseline == runner.evidence_overlay_runtime_fingerprints(**kwargs)

    incumbent[0] = 2.0
    changed_incumbent = runner.evidence_overlay_runtime_fingerprints(**kwargs)
    assert changed_incumbent["guarded_incumbent"] != baseline["guarded_incumbent"]

    controller.phase_rescue_retired = True
    changed_controller = runner.evidence_overlay_runtime_fingerprints(**kwargs)
    assert changed_controller["controller"] != baseline["controller"]

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.random()
        np.random.random()
        changed_rng = runner.evidence_overlay_runtime_fingerprints(**kwargs)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    assert changed_rng["rng"] != baseline["rng"]


def test_runtime_fingerprint_mismatch_writes_inapplicable_manifest(
    tmp_path: Path,
) -> None:
    observer = _observer()
    anchor = _prepare(observer)
    observer.execute_barrier(
        _objective,
        anchor,
        remaining_fe=100,
        normal_sweep_fe=60,
        tolerance_fe=0,
    )
    with pytest.raises(EvidenceOverlayRuntimeError, match="mutated frozen"):
        observer.record_runtime_audit(
            fingerprints_before={"controller": "a" * 64},
            fingerprints_after={"controller": "b" * 64},
            probe_start_fe=180,
            probe_end_fe=196,
        )
    _record_complete_sweep(observer, 3)
    _, manifest = _write_complete_artifacts(observer, tmp_path)

    assert manifest["native_state_unchanged"] == 0
    assert manifest["observer_integrity"] == 0
    assert manifest["applicable"] == 0
