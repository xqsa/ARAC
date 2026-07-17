from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import pytest

from arac.backends.hcc_hypergraph_trace import (
    HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
    HypergraphTraceArtifactPaths,
    HypergraphTraceObserver,
)
from arac.policy.overlap_hypergraph import build_overlap_hypergraph
from scripts import audit_hypergraph_trace as audit


ROOT = Path(__file__).parents[1]


def _observer(
    grouping: list[list[int]] | None = None,
) -> HypergraphTraceObserver:
    return HypergraphTraceObserver(
        topology=build_overlap_hypergraph(
            [[0, 1], [1, 2]] if grouping is None else grouping
        ),
        problem_id="E2",
        seed=91,
        run_id="audit-hypergraph-test",
        fresh_optimizer_execution=True,
        lower_bound=-5.0,
        upper_bound=5.0,
        rng_descriptor_sha256="a" * 64,
        protocol_config_path=ROOT / "configs" / "hypergraph_delayed_credit_v1.json",
        protocol_spec_path=ROOT / "docs" / "design" / "hypergraph-delayed-credit-v1.md",
        runner_source_path=ROOT / "scripts" / "hcc_smoke_runner.py",
        terminal_target_fe=80,
        terminal_completion_tolerance_fe=80,
    )


def _record_complete_sweep(
    observer: HypergraphTraceObserver,
    sweep: int,
) -> list[float]:
    for group in range(2):
        start = sweep * 20 + group * 10
        pre_error = 100.0 - sweep * 5.0 - group
        before = (0.0, float(sweep + group), 0.0)
        proposal = (
            0.0,
            float(sweep + group) + (0.5 if group == 0 else -0.25),
            0.0,
        )
        observer.record_group(
            sweep_index=sweep,
            group_index=group,
            pre_error=pre_error,
            best_error=pre_error - 1.0,
            primary_requested_fe=8,
            primary_actual_fe=8,
            full_interval_start_fe=start,
            full_interval_end_fe=start + 10,
            pre_block_candidate=before,
            final_owner_candidate=proposal,
        )
    decision_fe = (sweep + 1) * 20
    record = [100.0 - index * 0.01 for index in range(decision_fe)]
    assert observer.complete_sweep(
        sweep_index=sweep,
        optimized_group_count=2,
        all_raw_groups_completed=True,
        native_sweep_end_completed=True,
        native_sweep_end_stage=HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
        sweep_end_fe=decision_fe,
        sweep_end_candidate=(0.0, float(sweep) + 0.25, 0.0),
        fitness_record=record,
    )
    return record


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _artifacts(
    tmp_path: Path,
    *,
    complete_sweeps: int,
    grouping: list[list[int]] | None = None,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, object],
]:
    observer = _observer(grouping)
    record: list[float] = []
    for sweep in range(complete_sweeps):
        record = _record_complete_sweep(observer, sweep)
    paths = HypergraphTraceArtifactPaths(
        manifest=tmp_path / "hypergraph_manifest.json",
        features=tmp_path / "hyperedge_cycle_features.csv",
        audit=tmp_path / "hyperedge_cycle_audit.csv",
        proposals=tmp_path / "shared_proposal_audit.csv",
        outcomes=tmp_path / "hyperedge_cycle_outcomes.csv",
    )
    manifest = observer.write_artifacts(
        paths=paths,
        final_fitness_record=record,
    )
    return (
        _read_csv(paths.features),
        _read_csv(paths.audit),
        _read_csv(paths.proposals),
        _read_csv(paths.outcomes),
        manifest,
    )


def _join(
    bundle: tuple[
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        dict[str, object],
    ],
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    features, audits, proposals, outcomes, manifest = bundle
    return audit._join_and_validate(
        features,
        audits,
        proposals,
        outcomes,
        source_manifests=[manifest],
    )


def _feature_hash(row: dict[str, str]) -> str:
    payload = {field: row[field] for field in audit.FEATURE_FIELDS[1:]}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_index_fixture(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, list[dict[str, str]]],
    Path,
]:
    features, audits, proposals, outcomes, source_manifest = _artifacts(
        tmp_path,
        complete_sweeps=4,
    )
    manifest_path = tmp_path / "hypergraph_manifest.json"
    entry = {
        **source_manifest,
        "lane_id": "hypergraph_v37_observer",
        "path": "hypergraph_manifest.json",
        "sha256": audit._sha256(manifest_path),
        "hcc_result_status": "completed",
        "hcc_result_max_fes": 80,
        "hcc_result_actual_fe_used": 80,
    }
    aggregate = {"source_manifest_count": 1, "source_manifests": [entry]}
    config = {
        "matrices": {
            "trace_screen": {
                "cases": ["E2"],
                "seeds": [91],
                "terminal_fe": 80,
            },
        }
    }
    aggregate_rows = {
        "hyperedge_cycle_features.csv": features,
        "hyperedge_cycle_audit.csv": audits,
        "shared_proposal_audit.csv": proposals,
        "hyperedge_cycle_outcomes.csv": outcomes,
    }
    return aggregate, config, aggregate_rows, manifest_path


def test_schema_uses_six_independent_pre_action_features() -> None:
    assert audit.STATE_FIELDS == (
        "current_unit_fe_contribution",
        "ewma_unit_fe_contribution_3",
        "zero_gain_difficulty",
        "stagnation_ratio_3",
        "direct_owner_proposal_disagreement",
        "prior_next_sweep_overwrite",
    )
    assert "success_ratio_3" not in audit.FEATURE_FIELDS
    assert "prior_next_sweep_survival" not in audit.FEATURE_FIELDS
    assert audit._sha256(audit.CONFIG_PATH) == audit.CONFIG_SHA256
    assert audit._sha256(audit.SPEC_PATH) == audit.SPEC_SHA256


def test_aggregate_manifest_uses_canonical_trace_only_boundary() -> None:
    matrix = {"cases": ["E2"], "seeds": [91], "terminal_fe": 3_000_000}
    manifest = {
        "protocol_version": audit.PROTOCOL_VERSION,
        "stage": "screen",
        "status": "pass",
        "hypergraph_trace_mode": "observer",
        "runtime_profile_authorized": False,
        "runtime_model_bundle_allowed": False,
        "diagnostic_model_used": False,
        "source_bundle": audit.hypergraph_source_bundle(),
        "static_ast_audit": audit.hypergraph_static_ast_audit(),
        "source_git_commit": audit._git_commit(),
        "observer_calls": {"objective": 0, "rng": 0, "optimizer": 0, "fe": 0},
        "problem_ids": ["E2"],
        "seeds": [91],
        "max_fes": 3_000_000,
        "config": {"sha256": audit.CONFIG_SHA256},
        "spec": {"sha256": audit.SPEC_SHA256},
    }

    assert audit._aggregate_manifest_blockers(
        manifest, stage="screen", matrix=matrix
    ) == []
    legacy = {**manifest, "mode": "observer"}
    legacy.pop("hypergraph_trace_mode")
    assert "manifest_trace_mode_mismatch" in audit._aggregate_manifest_blockers(
        legacy, stage="screen", matrix=matrix
    )
    unsafe = copy.deepcopy(manifest)
    unsafe["observer_calls"]["rng"] = 1  # type: ignore[index]
    assert "manifest_observer_calls_nonzero" in audit._aggregate_manifest_blockers(
        unsafe, stage="screen", matrix=matrix
    )
    malformed = {**manifest, "config": "not-an-object", "max_fes": None}
    malformed_blockers = audit._aggregate_manifest_blockers(
        malformed, stage="screen", matrix=matrix
    )
    assert "manifest_config_hash_mismatch" in malformed_blockers
    assert "manifest_terminal_fe_mismatch" in malformed_blockers


def test_source_manifest_index_binds_embedded_payload_to_file(tmp_path: Path) -> None:
    aggregate, config, aggregate_rows, _ = _source_index_fixture(tmp_path)

    indexed, blockers = audit._source_manifest_index(
        aggregate,
        source_root=tmp_path,
        stage="screen",
        config=config,
        aggregate_rows=aggregate_rows,
    )

    assert blockers == []
    assert set(indexed) == {("E2", "91")}
    tampered = copy.deepcopy(aggregate)
    tampered["source_manifests"][0]["lower_bound"] = -4.0  # type: ignore[index]
    _, blockers = audit._source_manifest_index(
        tampered,
        source_root=tmp_path,
        stage="screen",
        config=config,
        aggregate_rows=aggregate_rows,
    )
    assert any("embedded payload mismatch" in blocker for blocker in blockers)


def test_source_manifest_paths_are_relative_canonical_and_contained(
    tmp_path: Path,
) -> None:
    aggregate, config, aggregate_rows, manifest_path = _source_index_fixture(tmp_path)
    for invalid_path in (str(manifest_path.resolve()), "../hypergraph_manifest.json"):
        tampered = copy.deepcopy(aggregate)
        tampered["source_manifests"][0]["path"] = invalid_path  # type: ignore[index]
        _, blockers = audit._source_manifest_index(
            tampered,
            source_root=tmp_path,
            stage="screen",
            config=config,
            aggregate_rows=aggregate_rows,
        )
        assert any("source_manifest_provenance_failed" in item for item in blockers)


def test_aggregate_rows_must_rebuild_exactly_from_per_run_artifacts(
    tmp_path: Path,
) -> None:
    aggregate, config, aggregate_rows, _ = _source_index_fixture(tmp_path)
    tampered_rows = copy.deepcopy(aggregate_rows)
    tampered_rows["hyperedge_cycle_outcomes.csv"][0][
        "next_sweep_unit_fe_contribution"
    ] = "9.00000000000000000e+00"

    _, blockers = audit._source_manifest_index(
        aggregate,
        source_root=tmp_path,
        stage="screen",
        config=config,
        aggregate_rows=tampered_rows,
    )

    assert "aggregate_source_rebuild_mismatch:hyperedge_cycle_outcomes.csv" in blockers


def test_source_manifest_rejects_zero_terminal_tolerance(tmp_path: Path) -> None:
    aggregate, config, aggregate_rows, manifest_path = _source_index_fixture(tmp_path)
    source_manifest = audit._read_json(manifest_path)
    source_manifest["terminal_completion_tolerance_fe"] = 0
    audit._write_json(manifest_path, source_manifest)
    entry = {
        **source_manifest,
        "lane_id": "hypergraph_v37_observer",
        "path": "hypergraph_manifest.json",
        "sha256": audit._sha256(manifest_path),
        "hcc_result_status": "completed",
        "hcc_result_max_fes": 80,
        "hcc_result_actual_fe_used": 80,
    }
    aggregate["source_manifests"] = [entry]

    _, blockers = audit._source_manifest_index(
        aggregate,
        source_root=tmp_path,
        stage="screen",
        config=config,
        aggregate_rows=aggregate_rows,
    )

    assert any("source_manifest_terminal_failed" in item for item in blockers)


def test_observer_ledger_is_a_unique_terminal_fe_cross_binding() -> None:
    source_entry = {
        "hcc_result_max_fes": 80,
        "hcc_result_actual_fe_used": 80,
        "terminal_target_fe": 80,
        "terminal_observed_fe": 80,
        "terminal_completion_tolerance_fe": 20,
    }
    source_index = {("E2", "91"): source_entry}
    matrix = {"cases": ["E2"], "seeds": [91]}
    valid = {
        "lane_id": "hypergraph_v37_observer",
        "problem_id": "E2",
        "seed": "91",
        "fresh_execution": "1",
        "same_budget_violation": "0",
        "actual_fe_used": "80",
        "total_fe": "80",
        "budget_limit": "80",
        "configured_budget_limit": "80",
    }
    assert audit._observer_ledger_blockers(
        [valid],
        source_index=source_index,
        matrix=matrix,
    ) == []

    duplicate = audit._observer_ledger_blockers(
        [valid, dict(valid)],
        source_index=source_index,
        matrix=matrix,
    )
    assert any("duplicate_observer_ledger_route" in item for item in duplicate)
    assert "observer_ledger_matrix_mismatch" in audit._observer_ledger_blockers(
        [],
        source_index=source_index,
        matrix=matrix,
    )
    early = dict(valid, actual_fe_used="59", total_fe="59")
    early_source = {
        **source_entry,
        "hcc_result_actual_fe_used": 59,
        "terminal_observed_fe": 59,
    }
    blockers = audit._observer_ledger_blockers(
        [early],
        source_index={("E2", "91"): early_source},
        matrix=matrix,
    )
    assert any("outside the completion interval" in item for item in blockers)


def test_source_bundle_and_policy_ast_are_fail_closed() -> None:
    matrix = {"cases": ["E2"], "seeds": [91], "terminal_fe": 80}
    manifest = {
        "protocol_version": audit.PROTOCOL_VERSION,
        "stage": "screen",
        "status": "pass",
        "hypergraph_trace_mode": "observer",
        "runtime_profile_authorized": False,
        "runtime_model_bundle_allowed": False,
        "diagnostic_model_used": False,
        "source_bundle": audit.hypergraph_source_bundle(),
        "static_ast_audit": audit.hypergraph_static_ast_audit(),
        "source_git_commit": audit._git_commit(),
        "observer_calls": {"objective": 0, "rng": 0, "optimizer": 0, "fe": 0},
        "problem_ids": ["E2"],
        "seeds": [91],
        "max_fes": 80,
        "config": {"sha256": audit.CONFIG_SHA256},
        "spec": {"sha256": audit.SPEC_SHA256},
    }
    tampered = copy.deepcopy(manifest)
    tampered["source_bundle"]["bundle_sha256"] = "0" * 64  # type: ignore[index]
    assert "manifest_source_bundle_mismatch" in audit._aggregate_manifest_blockers(
        tampered,
        stage="screen",
        matrix=matrix,
    )

    policy_source = (ROOT / "src" / "arac" / "policy" / "overlap_hypergraph.py").read_text(
        encoding="utf-8"
    )
    unfrozen = policy_source.replace(
        "@dataclass(frozen=True)\nclass HyperedgeCycleState:",
        "@dataclass(frozen=False)\nclass HyperedgeCycleState:",
        1,
    )
    assert audit._policy_ast_audit_from_source(unfrozen)["status"] == "fail"
    seed_injected = policy_source.replace(
        "def score_hyperedge_states(\n    states:",
        "def score_hyperedge_states(\n    seed: int,\n    states:",
        1,
    )
    injected_audit = audit._policy_ast_audit_from_source(seed_injected)
    assert injected_audit["status"] == "fail"
    assert any("seed" in item for item in injected_audit["blockers"])

    helper_injected = policy_source.replace(
        "def midrank_percentiles(values: Sequence[float])",
        "def midrank_percentiles(values: Sequence[float], seed: int = 0)",
        1,
    )
    helper_audit = audit._policy_ast_audit_from_source(helper_injected)
    assert helper_audit["status"] == "fail"
    assert "midrank_percentiles" in helper_audit["reachable_definitions"]
    assert any("seed" in item for item in helper_audit["blockers"])
    for forbidden in sorted(audit._CONFIG_FORBIDDEN_POLICY_INPUTS):
        forbidden_injected = policy_source.replace(
            "def midrank_percentiles(values: Sequence[float])",
            (
                "def midrank_percentiles(values: Sequence[float], "
                f"{forbidden}: int = 0)"
            ),
            1,
        )
        forbidden_audit = audit._policy_ast_audit_from_source(forbidden_injected)
        assert forbidden_audit["status"] == "fail", forbidden
        assert any(forbidden in item for item in forbidden_audit["blockers"])
    method_injected = policy_source.replace(
        "def shared_for_group(self, group_index: int)",
        "def shared_for_group(self, group_index: int, seed: int = 0)",
        1,
    )
    method_audit = audit._policy_ast_audit_from_source(method_injected)
    assert method_audit["status"] == "fail"
    assert any("seed" in item for item in method_audit["blockers"])


def test_source_bundle_hash_is_order_invariant_and_uses_sorted_mapping() -> None:
    forward = audit._source_bundle_from_paths(audit.SOURCE_BUNDLE_PATHS)
    reversed_bundle = audit._source_bundle_from_paths(
        tuple(reversed(audit.SOURCE_BUNDLE_PATHS))
    )

    assert forward == reversed_bundle
    expected_paths = sorted(audit.SOURCE_BUNDLE_PATHS)
    assert list(forward["file_sha256"]) == expected_paths
    assert [row["path"] for row in forward["files"]] == expected_paths
    assert forward["bundle_sha256"] == audit._canonical_sha256(
        forward["file_sha256"]
    )


def test_unknown_git_commit_cannot_satisfy_manifest_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "_git_commit", lambda: "unknown")
    manifest = {
        "protocol_version": audit.PROTOCOL_VERSION,
        "stage": "screen",
        "status": "pass",
        "hypergraph_trace_mode": "observer",
        "runtime_profile_authorized": False,
        "runtime_model_bundle_allowed": False,
        "diagnostic_model_used": False,
        "source_bundle": audit.hypergraph_source_bundle(),
        "source_git_commit": "unknown",
        "static_ast_audit": audit.hypergraph_static_ast_audit(),
        "observer_calls": {"objective": 0, "rng": 0, "optimizer": 0, "fe": 0},
        "problem_ids": ["E2"],
        "seeds": [91],
        "max_fes": 3_000_000,
        "config": {"sha256": audit.CONFIG_SHA256},
        "spec": {"sha256": audit.SPEC_SHA256},
    }

    blockers = audit._aggregate_manifest_blockers(
        manifest,
        stage="screen",
        matrix={"cases": ["E2"], "seeds": [91], "terminal_fe": 3_000_000},
    )

    assert "manifest_source_git_commit_mismatch" in blockers


def test_raw_evidence_reconstructs_first_closed_snapshot(tmp_path: Path) -> None:
    joined, coverage, blockers = _join(_artifacts(tmp_path, complete_sweeps=4))

    assert blockers == []
    assert len(joined) == 2
    assert {row["sweep_index"] for row in joined} == {2}
    assert coverage["runtime_applicable_trajectories"] == 1
    assert coverage["applicable_trajectories"] == 1
    assert coverage["complete_next_sweep_label_fraction"] == 1.0
    assert coverage["required_state_missing_fraction"] == 0.0


def test_manifest_and_cohort_cannot_shift_the_first_lock_later(tmp_path: Path) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    audits = copy.deepcopy(bundle[1])
    manifest = copy.deepcopy(bundle[4])
    for row in audits:
        if row["sweep_index"] == "2":
            row["cohort_locked"] = "0"
        elif row["sweep_index"] == "3":
            row["cohort_locked"] = "1"
    manifest["decision_snapshot_sweep"] = 3
    manifest["cohort_locked_sweep"] = 3
    bundle[1] = audits
    bundle[4] = manifest

    _, _, blockers = _join(tuple(bundle))  # type: ignore[arg-type]

    assert any("decision_snapshot_not_earliest" in blocker for blocker in blockers)


def test_complete_sweep_requires_one_decision_fe_and_fitness_hash(
    tmp_path: Path,
) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))

    fe_tamper = copy.deepcopy(bundle)
    fe_row = next(
        row
        for row in fe_tamper[1]
        if row["sweep_index"] == "0" and row["group_index"] == "1"
    )
    fe_row["source_end_fe"] = "21"
    fe_row["decision_fe"] = "21"
    _, _, blockers = _join(tuple(fe_tamper))  # type: ignore[arg-type]
    assert any("complete_sweep_decision_fe_mismatch" in blocker for blocker in blockers)

    hash_tamper = copy.deepcopy(bundle)
    hash_row = next(
        row
        for row in hash_tamper[1]
        if row["sweep_index"] == "0" and row["group_index"] == "1"
    )
    hash_row["fitness_record_sha256"] = "b" * 64
    _, _, blockers = _join(tuple(hash_tamper))  # type: ignore[arg-type]
    assert any("complete_sweep_fitness_hash_mismatch" in blocker for blocker in blockers)


def test_complete_sweep_requires_canonical_order_and_disjoint_fe(
    tmp_path: Path,
) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))

    order_tamper = copy.deepcopy(bundle)
    first = next(
        index
        for index, row in enumerate(order_tamper[1])
        if row["sweep_index"] == "0" and row["group_index"] == "0"
    )
    second = next(
        index
        for index, row in enumerate(order_tamper[1])
        if row["sweep_index"] == "0" and row["group_index"] == "1"
    )
    order_tamper[1][first], order_tamper[1][second] = (
        order_tamper[1][second],
        order_tamper[1][first],
    )
    _, _, blockers = _join(tuple(order_tamper))  # type: ignore[arg-type]
    assert any("noncanonical_raw_group_order" in blocker for blocker in blockers)

    interval_tamper = copy.deepcopy(bundle)
    interval_row = next(
        row
        for row in interval_tamper[1]
        if row["sweep_index"] == "0" and row["group_index"] == "1"
    )
    interval_row["full_interval_start_fe"] = "5"
    interval_row["full_interval_actual_fe"] = "15"
    interval_row["unit_fe_contribution"] = audit._format_float(
        audit.unit_fe_contribution(
            pre_error=float(interval_row["pre_error"]),
            best_error=float(interval_row["best_error"]),
            actual_fe=15,
        )
    )
    _, _, blockers = _join(tuple(interval_tamper))  # type: ignore[arg-type]
    assert any("complete_sweep_fe_interval_overlap" in blocker for blocker in blockers)


def test_rehashed_feature_tamper_is_rejected_by_raw_recompute(
    tmp_path: Path,
) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    features = copy.deepcopy(bundle[0])
    audits = copy.deepcopy(bundle[1])
    features[0]["current_unit_fe_contribution"] = "9.00000000000000000e+00"
    audit_by_id = {row["decision_id"]: row for row in audits}
    audit_by_id[features[0]["decision_id"]]["feature_sha256"] = _feature_hash(
        features[0]
    )
    bundle[0] = features
    bundle[1] = audits

    _, _, blockers = _join(tuple(bundle))  # type: ignore[arg-type]

    assert any("raw_feature_mismatch" in blocker for blocker in blockers)


def test_raw_feature_hash_rejects_tolerance_sized_rehash(tmp_path: Path) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    features = copy.deepcopy(bundle[0])
    audits = copy.deepcopy(bundle[1])
    original = float(features[0]["current_unit_fe_contribution"])
    features[0]["current_unit_fe_contribution"] = audit._format_float(
        original + 1e-14
    )
    audit_by_id = {row["decision_id"]: row for row in audits}
    audit_by_id[features[0]["decision_id"]]["feature_sha256"] = _feature_hash(
        features[0]
    )
    bundle[0] = features
    bundle[1] = audits

    _, _, blockers = _join(tuple(bundle))  # type: ignore[arg-type]

    assert any("raw_feature_hash_mismatch" in blocker for blocker in blockers)


def test_prior_and_future_credit_are_recomputed_from_raw_proposals(
    tmp_path: Path,
) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    proposals = copy.deepcopy(bundle[2])
    outcomes = copy.deepcopy(bundle[3])
    prior = next(
        row
        for row in proposals
        if row["sweep_index"] == "1" and row["group_index"] == "0"
    )
    prior["next_sweep_value"] = "1.00000000000000000e+00"
    outcomes[0]["next_sweep_survival"] = "9.00000000000000000e-01"
    outcomes[0]["next_sweep_overwrite"] = "1.00000000000000000e-01"
    bundle[2] = proposals
    bundle[3] = outcomes

    _, _, blockers = _join(tuple(bundle))  # type: ignore[arg-type]

    assert any(
        "proposal_next_sweep_endpoint_mismatch" in blocker for blocker in blockers
    )
    assert any("raw outcome encoding mismatch" in blocker for blocker in blockers)


def test_tolerance_sized_outcome_labels_cannot_enter_statistics(tmp_path: Path) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    next_sweep_rows = [
        row
        for row in bundle[1]
        if row["sweep_index"] == "3" and row["group_index"] in {"0", "1"}
    ]
    for row in next_sweep_rows:
        row["best_error"] = row["pre_error"]
        row["successful"] = "0"
        row["unit_fe_contribution"] = audit._format_float(0.0)
    exact = copy.deepcopy(bundle)
    for row in exact[3]:
        row["next_sweep_unit_fe_contribution"] = audit._format_float(0.0)
    joined, _, blockers = _join(tuple(exact))  # type: ignore[arg-type]
    assert blockers == []
    summaries, summary_blockers = audit._trajectory_summaries(joined)
    assert summary_blockers == []
    assert audit._metric(summaries, "trajectory_priority_spearman") == 0.0

    tampered = copy.deepcopy(exact)
    for index, row in enumerate(tampered[3], start=1):
        row["next_sweep_unit_fe_contribution"] = audit._format_float(index * 1e-16)
    joined, _, blockers = _join(tuple(tampered))  # type: ignore[arg-type]
    assert joined == []
    assert any("raw outcome encoding mismatch" in item for item in blockers)


def test_tolerance_sized_raw_unit_fe_cannot_be_synchronized_into_labels(
    tmp_path: Path,
) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    forged = (1e-16, 2e-16)
    for index, row in enumerate(
        row for row in bundle[1] if row["sweep_index"] == "3"
    ):
        row["best_error"] = row["pre_error"]
        row["successful"] = "0"
        row["unit_fe_contribution"] = audit._format_float(forged[index])
    for index, row in enumerate(bundle[3]):
        row["next_sweep_unit_fe_contribution"] = audit._format_float(
            forged[index]
        )

    joined, _, blockers = _join(tuple(bundle))  # type: ignore[arg-type]

    assert joined == []
    assert any("unit-FE contribution encoding mismatch" in item for item in blockers)


def test_tolerance_sized_proposal_backfill_cannot_enter_survival(
    tmp_path: Path,
) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    outcomes = {row["decision_id"]: row for row in bundle[3]}
    for row in bundle[2]:
        if row["sweep_index"] != "2":
            continue
        forged_endpoint = math.nextafter(
            float(row["next_sweep_value"]),
            math.inf,
        )
        row["next_sweep_value"] = audit._format_float(forged_endpoint)
        credit = audit.directional_survival(
            anchor_values=(float(row["anchor_value"]),),
            candidate_values=(float(row["proposed_value"]),),
            next_sweep_values=(forged_endpoint,),
        )
        outcome = outcomes[row["decision_id"]]
        outcome["next_sweep_survival"] = audit._format_float(credit.survival)
        outcome["next_sweep_overwrite"] = audit._format_float(credit.overwrite)

    _, _, blockers = _join(tuple(bundle))  # type: ignore[arg-type]

    assert any("proposal_next_sweep_endpoint_mismatch" in item for item in blockers)


def test_proposal_backfill_must_match_raw_next_sweep_endpoint(
    tmp_path: Path,
) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    proposals = copy.deepcopy(bundle[2])
    outcomes = copy.deepcopy(bundle[3])
    outcomes_by_id = {row["decision_id"]: row for row in outcomes}
    for row in proposals:
        if row["sweep_index"] != "2":
            continue
        forged_endpoint = 4.0
        row["next_sweep_value"] = audit._format_float(forged_endpoint)
        credit = audit.directional_survival(
            anchor_values=(float(row["anchor_value"]),),
            candidate_values=(float(row["proposed_value"]),),
            next_sweep_values=(forged_endpoint,),
        )
        outcome = outcomes_by_id[row["decision_id"]]
        outcome["next_sweep_survival"] = audit._format_float(credit.survival)
        outcome["next_sweep_overwrite"] = audit._format_float(credit.overwrite)
    bundle[2] = proposals
    bundle[3] = outcomes

    _, _, blockers = _join(tuple(bundle))  # type: ignore[arg-type]

    assert any("proposal_next_sweep_endpoint_mismatch" in blocker for blocker in blockers)


def test_terminal_censor_is_valid_evidence_but_fails_label_closure(
    tmp_path: Path,
) -> None:
    joined, coverage, blockers = _join(_artifacts(tmp_path, complete_sweeps=3))

    assert blockers == []
    assert joined == []
    assert coverage["runtime_applicable_trajectories"] == 1
    assert coverage["applicable_trajectories"] == 1
    assert coverage["labeled_applicable_trajectories"] == 0
    assert coverage["complete_next_sweep_label_fraction"] == 0.0
    assert coverage["terminal_censored_trajectories"] == 1


def test_no_overlap_status_and_label_closure_are_recomputed(tmp_path: Path) -> None:
    bundle = _artifacts(
        tmp_path,
        complete_sweeps=3,
        grouping=[[0], [1]],
    )
    joined, coverage, blockers = _join(bundle)

    assert blockers == []
    assert joined == []
    assert coverage["applicable_trajectories"] == 0

    tampered = list(copy.deepcopy(bundle))
    tampered[4]["decision_status"] = "applicable"
    tampered[4]["label_closure"] = "closed"
    _, _, blockers = _join(tuple(tampered))  # type: ignore[arg-type]
    assert any("decision_status_mismatch" in blocker for blocker in blockers)
    assert any("label_closure_mismatch" in blocker for blocker in blockers)


def test_second_locked_snapshot_is_rejected(tmp_path: Path) -> None:
    bundle = list(_artifacts(tmp_path, complete_sweeps=4))
    audits = copy.deepcopy(bundle[1])
    next(row for row in audits if row["sweep_index"] == "3")["cohort_locked"] = "1"
    bundle[1] = audits

    _, _, blockers = _join(tuple(bundle))  # type: ignore[arg-type]

    assert any("invalid_one_shot_cohort" in blocker for blocker in blockers)


def test_incomplete_first_opportunity_placeholders_are_not_corruption(
    tmp_path: Path,
) -> None:
    observer = _observer()
    record: list[float] = []
    for sweep in range(2):
        record = _record_complete_sweep(observer, sweep)
    observer.record_group(
        sweep_index=2,
        group_index=0,
        pre_error=90.0,
        best_error=89.0,
        primary_requested_fe=8,
        primary_actual_fe=8,
        full_interval_start_fe=40,
        full_interval_end_fe=50,
        pre_block_candidate=(0.0, 2.0, 0.0),
        final_owner_candidate=(0.0, 2.5, 0.0),
    )
    assert not observer.complete_sweep(
        sweep_index=2,
        optimized_group_count=1,
        all_raw_groups_completed=False,
        native_sweep_end_completed=False,
        native_sweep_end_stage=HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
        sweep_end_fe=50,
        sweep_end_candidate=(0.0, 2.25, 0.0),
        fitness_record=[100.0] * 50,
    )
    paths = HypergraphTraceArtifactPaths(
        manifest=tmp_path / "manifest.json",
        features=tmp_path / "features.csv",
        audit=tmp_path / "audit.csv",
        proposals=tmp_path / "proposals.csv",
        outcomes=tmp_path / "outcomes.csv",
    )
    manifest = observer.write_artifacts(
        paths=paths,
        final_fitness_record=[100.0] * 50,
    )
    bundle = (
        _read_csv(paths.features),
        _read_csv(paths.audit),
        _read_csv(paths.proposals),
        _read_csv(paths.outcomes),
        manifest,
    )

    joined, coverage, blockers = _join(bundle)

    assert blockers == []
    assert joined == []
    assert coverage["applicable_trajectories"] == 0

    tampered = list(copy.deepcopy(bundle))
    tampered[4]["decision_status"] = "applicable"
    tampered[4]["label_closure"] = "closed"
    _, _, blockers = _join(tuple(tampered))  # type: ignore[arg-type]
    assert any("decision_status_mismatch" in blocker for blocker in blockers)
    assert any("label_closure_mismatch" in blocker for blocker in blockers)


def test_cli_fail_closed_still_writes_machine_readable_gate(tmp_path: Path) -> None:
    gate = audit.audit_hypergraph_trace(tmp_path, stage="screen")

    assert gate["status"] == "screen_no_go"
    assert gate["action_implementation_authorized"] is False
    assert gate["scheduler_authorized"] is False
    assert gate["support_filter_applied"] is False
    assert (tmp_path / "hypergraph_identifiability_gate.json").is_file()


def test_screen_rejects_prior_gate_and_full_requires_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not accept"):
        audit.audit_hypergraph_trace(
            tmp_path,
            stage="screen",
            screen_gate=tmp_path / "screen.json",
        )

    full_root = tmp_path / "full"
    full_root.mkdir()
    gate = audit.audit_hypergraph_trace(full_root, stage="full")
    assert gate["status"] == "full_no_go"
    assert "prior_screen_gate_required" in gate["blockers"]
    assert gate["checks"]["prior_screen_gate_binding"] is False
    assert gate["action_implementation_authorized"] is False


def test_prior_screen_gate_is_recomputed_and_bound_to_full_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_path = tmp_path / "hypergraph_identifiability_gate.json"
    current_bundle = audit.hypergraph_source_bundle()
    current_ast = audit.hypergraph_static_ast_audit()
    current_commit = audit._git_commit()
    payload = {
        "protocol_version": audit.PROTOCOL_VERSION,
        "stage": "screen",
        "status": "screen_pass",
        "checks": {"integrity_fraction": True},
        "blockers": [],
        "config_sha256": audit.CONFIG_SHA256,
        "spec_sha256": audit.SPEC_SHA256,
        "source_git_commit": current_commit,
        "source_bundle": current_bundle,
        "static_ast_audit": current_ast,
        "source_root": str(tmp_path.resolve()),
    }
    audit._write_json(gate_path, payload)
    monkeypatch.setattr(
        audit,
        "_recompute_screen_gate",
        lambda source_root: copy.deepcopy(payload),
    )
    gate_hash = audit._sha256(gate_path)
    prior_binding = {
        "path": str(gate_path.resolve()),
        "sha256": gate_hash,
        "source_root": str(tmp_path.resolve()),
        "status": "screen_pass",
        "source_git_commit": current_commit,
        "source_bundle": current_bundle,
        "config_sha256": audit.CONFIG_SHA256,
        "spec_sha256": audit.SPEC_SHA256,
    }
    full_manifest = {
        "source_bundle": current_bundle,
        "static_ast_audit": current_ast,
        "source_git_commit": current_commit,
        "config": {"sha256": audit.CONFIG_SHA256},
        "spec": {"sha256": audit.SPEC_SHA256},
        "prior_screen_gate": prior_binding,
    }

    gate_hash, binding, blockers = audit._prior_screen_gate_binding(
        gate_path,
        full_manifest=full_manifest,
    )

    assert gate_hash == audit._sha256(gate_path)
    assert binding["status"] == "pass"
    assert blockers == []

    tampered = copy.deepcopy(payload)
    tampered["status"] = "screen_no_go"
    audit._write_json(gate_path, tampered)
    _, binding, blockers = audit._prior_screen_gate_binding(
        gate_path,
        full_manifest=full_manifest,
    )
    assert binding["status"] == "fail"
    assert any("prior_screen_gate_binding_failed" in item for item in blockers)


def test_crossfit_artifacts_use_trajectory_weights_without_fake_support() -> None:
    joined = [
        {
            "decision_id": f"{case}-{group}",
            "problem_id": case,
            "seed": 1,
            "owner_reliability": reliability,
            "next_sweep_overwrite": overwrite,
        }
        for case, group_count, reliability, overwrite in (
            ("E2", 2, 0.2, 0.8),
            ("E3", 4, 0.8, 0.2),
        )
        for group in range(group_count)
    ]

    assignments, predictions, blockers = audit._crossfit(joined, fold_type="lco")

    assert blockers == []
    assert "in_support" not in audit.FOLD_FIELDS
    assert "in_support" not in audit.PREDICTION_FIELDS
    assert all(row["support_reason"] == "no_support_filter" for row in assignments)
    assert {
        (row["problem_id"], row["trajectory_weight"]) for row in assignments
    } == {("E2", 0.5), ("E3", 0.25)}
    assert all("trajectory_weight" in row for row in predictions)


def test_trajectory_metrics_are_equal_weight_and_constant_maps_to_zero() -> None:
    joined = [
        {
            "problem_id": "E2",
            "seed": 1,
            "sweep_index": 2,
            "selected_focal": 0,
            "focal_priority": 0.2,
            "next_sweep_unit_fe_contribution": 0.0,
            "owner_reliability": 0.2,
            "next_sweep_survival": 0.5,
        },
        {
            "problem_id": "E2",
            "seed": 1,
            "sweep_index": 2,
            "selected_focal": 1,
            "focal_priority": 0.8,
            "next_sweep_unit_fe_contribution": 1.0,
            "owner_reliability": 0.8,
            "next_sweep_survival": 0.5,
        },
        {
            "problem_id": "E3",
            "seed": 1,
            "sweep_index": 2,
            "selected_focal": 0,
            "focal_priority": 0.1,
            "next_sweep_unit_fe_contribution": 4.0,
            "owner_reliability": 0.1,
            "next_sweep_survival": 0.9,
        },
        {
            "problem_id": "E3",
            "seed": 1,
            "sweep_index": 2,
            "selected_focal": 0,
            "focal_priority": 0.4,
            "next_sweep_unit_fe_contribution": 3.0,
            "owner_reliability": 0.4,
            "next_sweep_survival": 0.6,
        },
        {
            "problem_id": "E3",
            "seed": 1,
            "sweep_index": 2,
            "selected_focal": 0,
            "focal_priority": 0.6,
            "next_sweep_unit_fe_contribution": 2.0,
            "owner_reliability": 0.6,
            "next_sweep_survival": 0.4,
        },
        {
            "problem_id": "E3",
            "seed": 1,
            "sweep_index": 2,
            "selected_focal": 1,
            "focal_priority": 0.9,
            "next_sweep_unit_fe_contribution": 1.0,
            "owner_reliability": 0.9,
            "next_sweep_survival": 0.1,
        },
    ]

    summaries, blockers = audit._trajectory_summaries(joined)

    assert blockers == []
    assert len(summaries) == 2
    assert audit._metric(summaries, "trajectory_priority_spearman") == 0.0
    assert summaries[0]["trajectory_owner_survival_spearman"] == 0.0
    assert summaries[0]["trajectory_focal_rank_advantage"] > 0.0
    assert summaries[1]["trajectory_focal_rank_advantage"] < 0.0


def test_weighted_median_and_balanced_accuracy_equalize_trajectories() -> None:
    values = [0.9, 0.8, *([0.1] * 10)]
    weights = [1.0, 1.0, *([0.1] * 10)]

    assert audit._weighted_median(values, weights) == 0.8
    assert audit._balanced_accuracy(
        [0, 1, 1],
        [0, 1, 0],
        [1.0, 0.5, 0.5],
    ) == 0.75


def test_pigeonhole_resample_keeps_all_rows_of_a_trajectory_clustered() -> None:
    rows = [
        {"problem_id": case, "seed": seed, "decision_id": f"{case}-{seed}-{group}"}
        for case in ("E2", "E3")
        for seed in (1, 2)
        for group in (0, 1)
    ]
    sampled = audit._pigeonhole_sample(rows, random.Random(17))
    counts = Counter(str(row["decision_id"]) for row in sampled)

    for case in ("E2", "E3"):
        for seed in (1, 2):
            assert counts[f"{case}-{seed}-0"] == counts[f"{case}-{seed}-1"]


def test_single_class_bootstrap_replicate_contributes_zero_not_deletion() -> None:
    rows = [
        {
            "problem_id": "E2",
            "seed": 1,
            "next_sweep_overwrite": 0.1,
            "overwrite_prediction": 0,
            "trajectory_weight": 0.5,
        },
        {
            "problem_id": "E2",
            "seed": 1,
            "next_sweep_overwrite": 0.2,
            "overwrite_prediction": 0,
            "trajectory_weight": 0.5,
        },
    ]

    assert audit._bootstrap_lcb(
        rows,
        "overwrite_balanced_accuracy",
        resamples=20,
        seed=7,
        quantile=0.05,
    ) == 0.0
