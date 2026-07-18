from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from arac.backends.hcc import (
    DEFAULT_AOB_DATA_ROOT,
    HccAobExecutionResult,
    required_aob_data_files,
)
from arac.backends.hcc_evidence_overlay import (
    EvidenceOverlayArtifactPaths,
    HccEvidenceOverlayObserver,
)
from experiments.pilots.exp_018_rddsm_evidence_overlay_pilot.protocol import (
    AGGREGATE_ARTIFACTS,
    GateInputs,
    LANE_MODE_PAIRS,
    SOURCE_MODE,
    _is_catastrophic,
    _phase_boundary_consistent,
    _probe_bundle_valid,
    _raw_runtime_unauthorized,
    _relation_bundle_ids_match,
    build_execution_request,
    build_promotion_gate,
    build_run_matrix,
    load_config,
)
from experiments.pilots.exp_018_rddsm_evidence_overlay_pilot.run import (
    ExecutionRecord,
    _aob_data_bundle,
    _source_bundle,
    _validate_prior_smoke_gate,
    collect_artifacts,
    config_sha256,
    parse_args,
    run_pilot,
)
from arac.backends.hcc import (
    DEFAULT_AOB_DATA_ROOT,
    EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT,
    required_aob_data_files,
)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_aob_rows(problem_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in required_aob_data_files(DEFAULT_AOB_DATA_ROOT, int(problem_id[1:])):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "problem_id": problem_id,
                "file": path.name,
                "path": str(path.resolve()),
                "sha256_before": digest,
                "sha256_after": digest,
                "unchanged": "1",
            }
        )
    return rows


def _result(request, *, overlay_fe: int = 0) -> HccAobExecutionResult:
    return HccAobExecutionResult(
        problem_id=request.problem_id,
        seed=request.seed,
        max_fes=request.max_fes,
        final_error=1.0,
        fe_used=request.max_fes,
        time_seconds=0.01,
        output_root=request.output_dir,
        fresh_optimizer_execution=True,
        status="completed",
        result_source="hcc_subprocess_smoke_execution",
        optimizer_final_fe_used=request.max_fes,
        global_phase_fe=request.max_fes - overlay_fe,
        cc_phase_fe=0,
        rescue_fe=0,
        refresh_fe=0,
        search_state_fe=0,
        precision_probe_fe=0,
        evidence_overlay_fe=overlay_fe,
        separable_continuation_fe=0,
        overhead_fe=0,
    )


def test_frozen_config_builds_exact_smoke_and_mechanism_matrices() -> None:
    config = load_config()
    smoke = build_run_matrix(config, "smoke")
    mechanism = build_run_matrix(config, "mechanism")

    assert len(smoke) == 15
    assert len(mechanism) == 60
    assert tuple(
        (spec.lane.lane_id, spec.lane.evidence_overlay_mode)
        for spec in smoke[:3]
    ) == LANE_MODE_PAIRS
    assert {spec.problem_id for spec in mechanism} == {"E1", "E3", "A4", "S5"}
    assert {spec.seed for spec in mechanism} == {117, 118, 119, 120, 121}
    assert {spec.max_fes for spec in mechanism} == {3_000_000}
    assert sum(spec.problem_id == "A4" and spec.max_fes == 3_000_000 for spec in smoke) == 3
    assert config["promotion_gate"]["negative_control"]["bootstrap_count"] == 2000
    assert config["promotion_gate"]["negative_control"]["bootstrap_seed"] == 2026071701


def test_requests_use_shared_hcc_overlay_modes_without_hypergraph_proxy(tmp_path: Path) -> None:
    config = load_config()
    specs = build_run_matrix(config, "smoke")[:3]
    requests = [
        build_execution_request(
            spec,
            tmp_path / spec.lane.lane_id,
            config=config,
            python_executable="python",
        )
        for spec in specs
    ]

    assert [request.evidence_overlay_mode for request in requests] == [
        "native_audit",
        "paired_owner",
        "shuffled_owner",
    ]
    assert all(request.python_executable == "python" for request in requests)
    assert all(request.problem_id == "E1" for request in requests)
    assert all(request.seed == 1 for request in requests)
    assert all(request.max_fes == 100_000 for request in requests)


def test_nonfresh_source_mode_fails_before_matrix_execution(tmp_path: Path) -> None:
    payload = load_config()
    payload["execution"]["source_mode"] = "reused_artifact"
    path = tmp_path / "nonfresh.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="execution section is frozen"):
        load_config(path)
    with pytest.raises(ValueError, match="source_mode"):
        run_pilot(tmp_path / "results", stage="smoke", source_mode="reused_artifact")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("execution", "skip_plots", False),
        ("overlay", "owner_weight_cap", 0.66),
        (
            "promotion_gate",
            "negative_control.bootstrap_count",
            1999,
        ),
    ),
)
def test_runtime_and_gate_config_sections_are_exactly_frozen(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    payload = json.loads(json.dumps(load_config()))
    target = payload[section]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    path = tmp_path / f"mutated-{section}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{section} section is frozen"):
        load_config(path)


def test_relation_bundle_rejects_id_mismatch_and_nonunit_probe_fe() -> None:
    plans = [{"relation_id": f"r{index}"} for index in range(4)]
    probes = [
        {
            "relation_id": f"r{relation}",
            "candidate": candidate,
            "utility": "0",
            "actual_fe": "1",
        }
        for relation in range(4)
        for candidate in ("x0", "left_owner", "right_owner", "bridge")
    ]
    delayed = [
        {"relation_id": f"r{relation}", "owner": owner}
        for relation in range(4)
        for owner in ("left", "right")
    ]
    shadow = [{"relation_id": f"r{index}"} for index in range(4)]

    assert _probe_bundle_valid(probes, relation_count=4, expected_fe=16)
    assert _relation_bundle_ids_match(plans, probes, delayed, shadow)
    probes[0]["actual_fe"] = "2"
    probes[1]["actual_fe"] = "0"
    assert not _probe_bundle_valid(probes, relation_count=4, expected_fe=16)
    delayed[-1]["relation_id"] = "wrong"
    assert not _relation_bundle_ids_match(plans, probes, delayed, shadow)


def test_phase_boundary_and_every_raw_authorization_bit_are_enforced() -> None:
    checkpoint = {"phase_boundary_fe": "180", "runtime_authorized": "0"}
    plan = [{"phase_boundary_fe": "180", "runtime_authorized": "0"}]
    probe = [{"phase_boundary_fe": "180", "runtime_authorized": "0"}]
    delayed = [{"runtime_authorized": "0"}]
    shadow = [{"runtime_authorized": "0"}]
    inputs = GateInputs(
        run_results=[],
        ledger_rows=[],
        checkpoint_rows={"trajectory": checkpoint},
        plan_rows=plan,
        probe_rows=probe,
        delayed_rows=delayed,
        shadow_rows=shadow,
        aob_rows=[],
        anti_leakage_rows=[],
    )

    assert _phase_boundary_consistent(checkpoint, plan, probe)
    assert _raw_runtime_unauthorized(inputs)
    probe[0]["phase_boundary_fe"] = "181"
    probe[0]["runtime_authorized"] = "1"
    assert not _phase_boundary_consistent(checkpoint, plan, probe)
    assert not _raw_runtime_unauthorized(inputs)


def test_catastrophic_boundary_and_zero_comparator_are_explicit() -> None:
    assert not _is_catastrophic(1.199999, 1.0, 1.2)
    assert _is_catastrophic(1.2, 1.0, 1.2)
    assert not _is_catastrophic(0.0, 0.0, 1.2)
    assert _is_catastrophic(1e-300, 0.0, 1.2)


def _synthetic_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _mechanism_gate_fixture() -> tuple[dict[str, object], tuple, GateInputs]:
    config = load_config()
    specs = build_run_matrix(config, "mechanism")
    run_results: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    checkpoint_rows: dict[str, dict[str, object]] = {}
    plan_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    delayed_rows: list[dict[str, object]] = []
    shadow_rows: list[dict[str, object]] = []
    aob_rows: list[dict[str, object]] = []
    anti_leakage_rows: list[dict[str, object]] = []
    phase_boundary_fe = 180

    for spec in specs:
        lane_id = spec.lane.lane_id
        active = spec.problem_id != "E1" and lane_id != "a_rddsm_original_order"
        overlay_fe = 16 if active else 0
        run_results.append(
            {
                "trajectory_id": spec.trajectory_id,
                "problem_id": spec.problem_id,
                "seed": spec.seed,
                "lane_id": lane_id,
                "status": "completed",
                "fresh_optimizer_execution": 1,
                "source_mode": SOURCE_MODE,
                "evidence_overlay_fe": overlay_fe,
                "applicable": int(active),
                "native_terminal_error": 1.0,
            }
        )
        ledger_rows.append(
            {
                "trajectory_id": spec.trajectory_id,
                "same_budget_violation": 0,
                "ledger_closed": 1,
                "terminal_tolerance_rule": "maximum_native_group_population",
                "terminal_tolerance_fe": 64,
                "actual_fe_used": spec.max_fes,
                "budget_limit": spec.max_fes,
            }
        )
        triplet_digest = _synthetic_digest(spec.triplet_id)
        checkpoint_rows[spec.trajectory_id] = {
            "problem_id": spec.problem_id,
            "seed": spec.seed,
            "mode": spec.lane.evidence_overlay_mode,
            "checkpoint_fe": phase_boundary_fe,
            "fitness_prefix_hash": triplet_digest,
            "incumbent_hash": triplet_digest,
            "rddsm_topology_hash": triplet_digest,
            "rddsm_order_hash": triplet_digest,
            "phase_boundary_fe": phase_boundary_fe,
            "history_sweeps": "0;1;2",
            "previous_survival_closed": 1,
            "plan_status": "selected" if active else "abstained",
            "plan_reason": "top_four_unique" if active else "no_active_probe",
            "runtime_authorized": 0,
        }
        for aob_path in required_aob_data_files(
            DEFAULT_AOB_DATA_ROOT,
            int(spec.problem_id[1:]),
        ):
            aob_digest = hashlib.sha256(aob_path.read_bytes()).hexdigest()
            aob_rows.append(
                {
                    "trajectory_id": spec.trajectory_id,
                    "problem_id": spec.problem_id,
                    "lane_id": lane_id,
                    "file": aob_path.name,
                    "path": str(aob_path.resolve()),
                    "sha256_before": aob_digest,
                    "sha256_after": aob_digest,
                    "unchanged": "1",
                }
            )
        anti_leakage_rows.append(
            {
                "trajectory_id": spec.trajectory_id,
                "problem_id": spec.problem_id,
                "seed": spec.seed,
                "lane_id": lane_id,
                "audit_status": "pass",
            }
        )
        if not active:
            continue

        lane_prefix = "b" if lane_id == "b_rddsm_evidence_overlay" else "c"
        for relation_index in range(4):
            relation_id = f"{lane_prefix}_r{relation_index}"
            base = {
                "trajectory_id": spec.trajectory_id,
                "problem_id": spec.problem_id,
                "seed": spec.seed,
                "lane_id": lane_id,
                "mode": spec.lane.evidence_overlay_mode,
                "relation_id": relation_id,
                "runtime_authorized": 0,
            }
            plan_rows.append(
                {
                    **base,
                    "owner_groups": f"{relation_index};{relation_index + 1}",
                    "shared_variables": str(relation_index + 1),
                    "selected": 1,
                    "voi": float(relation_index + 1),
                    "native_voi": float(relation_index + 1),
                    "proposal_disagreement": float(relation_index + 1),
                    "owner_priority": 1.0,
                    "left_owner_reliability": 0.5,
                    "right_owner_reliability": 0.5,
                    "score_source_relation_id": relation_id,
                    "phase_boundary_fe": phase_boundary_fe,
                }
            )
            if lane_prefix == "b":
                utilities = {
                    "x0": 0.0,
                    "left_owner": 0.10 + 0.01 * relation_index,
                    "right_owner": 1.0 + 0.1 * relation_index,
                    "bridge": 0.50 + 0.01 * relation_index,
                }
                shadow = ("repair", "right_owner", utilities["right_owner"])
            else:
                utilities = {
                    "x0": 0.0,
                    "left_owner": 0.010 + 0.001 * relation_index,
                    "right_owner": 0.020 + 0.001 * relation_index,
                    "bridge": 0.030 + 0.001 * relation_index,
                }
                shadow = ("coordinate", "bridge", utilities["bridge"])
            for candidate, utility in utilities.items():
                probe_rows.append(
                    {
                        **base,
                        "candidate": candidate,
                        "fitness": 100.0 - utility,
                        "utility": utility,
                        "owner_reliability": 0.5,
                        "candidate_hash": _synthetic_digest(
                            f"{spec.trajectory_id}:{relation_id}:{candidate}"
                        ),
                        "phase_boundary_fe": phase_boundary_fe,
                        "actual_fe": 1,
                    }
                )
            for owner in ("left", "right"):
                right_owner = owner == "right"
                delayed_rows.append(
                    {
                        **base,
                        "owner": owner,
                        "action_sweep_index": 2,
                        "resolution_sweep_index": 3,
                        "survival_label": int(right_owner),
                        "overwrite_label": int(not right_owner),
                        "next_sweep_log_improvement": (
                            float(relation_index + 1) if right_owner else 0.0
                        ),
                        "overwrite_penalized_credit": (
                            float(relation_index + 1) if right_owner else 0.0
                        ),
                        "label_closed": 1,
                        "label_status": "closed_next_complete_sweep",
                        "resolution_fe": 240,
                    }
                )
            shadow_rows.append(
                {
                    **base,
                    "action": shadow[0],
                    "winner": shadow[1],
                    "utility": shadow[2],
                    "reason": "unique_probe_winner_above_one_percent",
                }
            )

    return config, specs, GateInputs(
        run_results=run_results,
        ledger_rows=ledger_rows,
        checkpoint_rows=checkpoint_rows,
        plan_rows=plan_rows,
        probe_rows=probe_rows,
        delayed_rows=delayed_rows,
        shadow_rows=shadow_rows,
        aob_rows=aob_rows,
        anti_leakage_rows=anti_leakage_rows,
    )


def test_synthetic_mechanism_fixture_exercises_every_promotion_gate() -> None:
    config, specs, inputs = _mechanism_gate_fixture()

    gate = build_promotion_gate("mechanism", config, specs, inputs)

    assert gate["status"] == "pilot_go", gate["blockers"]
    assert all(gate["checks"].values()), gate["checks"]
    assert gate["coverage"]["applicable_triplet_count"] == 15
    assert gate["coverage"]["delayed_closure_fraction"] == 1.0
    owner = gate["metrics"]["owner_identifiability"]
    assert owner["owner_rows"] == 120
    assert owner["lco_enhanced_balanced_accuracy"] == 1.0
    assert owner["lso_enhanced_balanced_accuracy"] == 1.0
    assert owner["lco_baseline_balanced_accuracy"] == 0.5
    assert owner["lso_baseline_balanced_accuracy"] == 0.5
    assert owner["lco_improvement_lcb_95"] > 0.0
    assert owner["lso_improvement_lcb_95"] > 0.0
    delayed = gate["metrics"]["delayed_alignment"]
    assert delayed["direction_pairs"] == 60
    assert delayed["direction_agreement"] == 1.0
    assert delayed["trajectory_spearman_count"] == 15
    assert delayed["mean_trajectory_voi_credit_spearman"] == 1.0
    negative = gate["metrics"]["negative_control"]
    assert negative["paired_triplets"] == 15
    assert negative["median_b_minus_c_probe_value_per_fe"] > 0.0
    assert negative["lcb_95_b_minus_c_probe_value_per_fe"] > 0.0
    assert negative["bootstrap_count"] == 2000
    assert gate["metrics"]["shadow"] == {
        "non_fallback_count": 60,
        "case_count": 3,
        "seed_count": 5,
    }
    assert gate["metrics"]["catastrophic"]["events"] == []
    assert gate["action_v2_design_authorized"] is True


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("unknown_shadow_action", "paired_overlay_integrity"),
        ("shadow_winner_not_probe_best", "paired_overlay_integrity"),
        ("owner_signal_reversed", "lco_enhanced_owner_balanced_accuracy"),
        ("owner_improvement_zero", "lco_improvement_bootstrap_lcb"),
        ("direction_reversed", "owner_preference_direction_agreement"),
        ("voi_credit_reversed", "positive_voi_delayed_credit_spearman"),
        ("shuffle_outperforms", "probe_value_vs_shuffle_lcb"),
        ("shadow_all_fallback", "shadow_non_fallback_count"),
        ("catastrophic_at_boundary", "zero_catastrophic_trajectories"),
        ("aob_cross_lane_hash_drift", "aob_input_hash_consistency"),
        ("aob_manifest_incomplete", "aob_input_hash_consistency"),
        ("nonconsecutive_checkpoint", "checkpoint_triplet_parity"),
        ("checkpoint_history_mismatch", "checkpoint_triplet_parity"),
        ("checkpoint_after_budget", "checkpoint_triplet_parity"),
    ),
)
def test_synthetic_mechanism_fixture_fails_closed_on_targeted_mutations(
    mutation: str,
    failed_check: str,
) -> None:
    config, specs, inputs = _mechanism_gate_fixture()
    if mutation == "unknown_shadow_action":
        row = next(
            row
            for row in inputs.shadow_rows
            if row["lane_id"] == "b_rddsm_evidence_overlay"
        )
        row["action"] = "unknown_action"
    elif mutation == "shadow_winner_not_probe_best":
        row = next(
            row
            for row in inputs.shadow_rows
            if row["lane_id"] == "b_rddsm_evidence_overlay"
        )
        row["winner"] = "left_owner"
    elif mutation == "owner_signal_reversed":
        for row in inputs.probe_rows:
            if row["lane_id"] == "b_rddsm_evidence_overlay":
                if row["candidate"] == "left_owner":
                    row["utility"] = 1.0
                elif row["candidate"] == "right_owner":
                    row["utility"] = 0.1
        for row in inputs.shadow_rows:
            if row["lane_id"] == "b_rddsm_evidence_overlay":
                row["winner"] = "left_owner"
                row["utility"] = 1.0
    elif mutation == "owner_improvement_zero":
        for row in inputs.plan_rows:
            if row["lane_id"] == "b_rddsm_evidence_overlay":
                row["left_owner_reliability"] = 0.1
                row["right_owner_reliability"] = 0.9
    elif mutation == "direction_reversed":
        for row in inputs.delayed_rows:
            if row["lane_id"] == "b_rddsm_evidence_overlay":
                row["survival_label"] = 1 if row["owner"] == "left" else 0
    elif mutation == "voi_credit_reversed":
        for row in inputs.delayed_rows:
            if row["lane_id"] == "b_rddsm_evidence_overlay":
                relation_index = int(str(row["relation_id"]).split("r")[-1])
                row["overwrite_penalized_credit"] = 4 - relation_index
    elif mutation == "shuffle_outperforms":
        for row in inputs.probe_rows:
            if row["lane_id"] == "c_rddsm_shuffled_overlay":
                row["utility"] = {
                    "left_owner": 1.9,
                    "right_owner": 2.0,
                    "bridge": 2.1,
                }.get(row["candidate"], row["utility"])
        for row in inputs.shadow_rows:
            if row["lane_id"] == "c_rddsm_shuffled_overlay":
                row["utility"] = 2.1
    elif mutation == "shadow_all_fallback":
        for row in inputs.probe_rows:
            if row["lane_id"] == "b_rddsm_evidence_overlay":
                row["utility"] = {
                    "left_owner": 0.001,
                    "right_owner": 0.005,
                    "bridge": 0.003,
                }.get(row["candidate"], row["utility"])
            elif row["lane_id"] == "c_rddsm_shuffled_overlay":
                row["utility"] = {
                    "left_owner": 0.0001,
                    "right_owner": 0.0002,
                    "bridge": 0.0003,
                }.get(row["candidate"], row["utility"])
        for row in inputs.shadow_rows:
            row["action"] = "fallback"
            row["winner"] = "none"
            row["reason"] = "probe_gain_below_one_percent"
            row["utility"] = 0.005 if row["lane_id"] == "b_rddsm_evidence_overlay" else 0.0003
    elif mutation == "catastrophic_at_boundary":
        row = next(
            row
            for row in inputs.run_results
            if row["problem_id"] == "E3"
            and row["seed"] == 117
            and row["lane_id"] == "b_rddsm_evidence_overlay"
        )
        row["native_terminal_error"] = 1.2
    elif mutation == "aob_cross_lane_hash_drift":
        row = next(
            row
            for row in inputs.aob_rows
            if row["problem_id"] == "A4"
            and row["lane_id"] == "b_rddsm_evidence_overlay"
        )
        row["sha256_before"] = row["sha256_after"] = _synthetic_digest("drift")
    elif mutation == "aob_manifest_incomplete":
        trajectory_id = specs[0].trajectory_id
        inputs.aob_rows[:] = [
            row
            for row in inputs.aob_rows
            if row["trajectory_id"] != trajectory_id
            or row["file"] == f"F{specs[0].problem_id[1:]}-design.txt"
        ]
    elif mutation == "nonconsecutive_checkpoint":
        trajectory_id = next(
            spec.trajectory_id
            for spec in specs
            if spec.problem_id == "S5"
            and spec.seed == 121
            and spec.lane.lane_id == "c_rddsm_shuffled_overlay"
        )
        inputs.checkpoint_rows[trajectory_id]["history_sweeps"] = "0;2;3"
    elif mutation == "checkpoint_history_mismatch":
        trajectory_id = next(
            spec.trajectory_id
            for spec in specs
            if spec.problem_id == "S5"
            and spec.seed == 121
            and spec.lane.lane_id == "c_rddsm_shuffled_overlay"
        )
        inputs.checkpoint_rows[trajectory_id]["history_sweeps"] = "1;2;3"
    else:
        triplet_id = next(
            spec.triplet_id
            for spec in specs
            if spec.problem_id == "A4" and spec.seed == 120
        )
        for spec in specs:
            if spec.triplet_id == triplet_id:
                row = inputs.checkpoint_rows[spec.trajectory_id]
                row["checkpoint_fe"] = spec.max_fes + 1
                row["phase_boundary_fe"] = spec.max_fes + 1
        for rows in (inputs.plan_rows, inputs.probe_rows):
            for row in rows:
                if row.get("trajectory_id") in {
                    spec.trajectory_id for spec in specs if spec.triplet_id == triplet_id
                }:
                    row["phase_boundary_fe"] = 3_000_001

    gate = build_promotion_gate("mechanism", config, specs, inputs)

    assert gate["status"] == "pilot_no_go"
    assert gate["checks"][failed_check] is False
    assert f"gate_failed:{failed_check}" in gate["blockers"]
    assert gate["action_v2_design_authorized"] is False


def _record_overlay_sweep(
    observer: HccEvidenceOverlayObserver,
    sweep: int,
) -> tuple[float, ...]:
    groups = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6))
    for group in range(len(groups)):
        start = sweep * 60 + group * 10
        before = [0.0] * 7
        proposal = before.copy()
        if group < 5:
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
        )
    endpoint = [0.0] * 7
    for variable in range(1, 6):
        endpoint[variable] = variable / 2.0
    sweep_end_fe = (sweep + 1) * 60
    assert observer.complete_sweep(
        sweep_index=sweep,
        sweep_end_fe=sweep_end_fe,
        sweep_end_candidate=endpoint,
        sweep_end_error=180.0 - sweep,
        fitness_record=[200.0 - index / 1000.0 for index in range(sweep_end_fe)],
        all_raw_groups_completed=True,
        native_sweep_end_completed=True,
    )
    return tuple(endpoint)


def _record_no_overlap_sweep(
    observer: HccEvidenceOverlayObserver,
    sweep: int,
) -> tuple[float, ...]:
    endpoint = (0.0, 0.0, 0.0, 0.0)
    for group in range(2):
        start = sweep * 20 + group * 10
        observer.record_group(
            sweep_index=sweep,
            group_index=group,
            pre_error=100.0,
            best_error=99.0,
            primary_requested_fe=8,
            primary_actual_fe=8,
            full_interval_actual_fe=10,
            full_interval_start_fe=start,
            full_interval_end_fe=start + 10,
            pre_block_candidate=endpoint,
            final_owner_candidate=endpoint,
        )
    sweep_end_fe = (sweep + 1) * 20
    assert observer.complete_sweep(
        sweep_index=sweep,
        sweep_end_fe=sweep_end_fe,
        sweep_end_candidate=endpoint,
        sweep_end_error=90.0 - sweep,
        fitness_record=[100.0 - index / 1000.0 for index in range(sweep_end_fe)],
        all_raw_groups_completed=True,
        native_sweep_end_completed=True,
    )
    return endpoint


def test_e1_no_overlap_still_writes_paired_checkpoint_for_all_lanes(
    tmp_path: Path,
) -> None:
    config = load_config()
    specs = build_run_matrix(config, "smoke")[:3]
    records: list[ExecutionRecord] = []
    digest = "a" * 64
    for spec in specs:
        run_output = tmp_path / spec.lane.lane_id
        request = build_execution_request(
            spec,
            run_output,
            config=config,
            python_executable="python",
        )
        raw = run_output / "nested"
        observer = HccEvidenceOverlayObserver(
            mode=spec.lane.evidence_overlay_mode,
            grouping_result=((0, 1), (2, 3)),
            problem_id="E1",
            seed=1,
            run_id=spec.trajectory_id,
            configured_max_fes=100_000,
            terminal_tolerance_fe=7,
            lower_bound=-10.0,
            upper_bound=10.0,
        )
        anchor: tuple[float, ...] = ()
        for sweep in range(3):
            anchor = _record_no_overlap_sweep(observer, sweep)

        def forbidden_objective(_candidate):
            raise AssertionError("E1/no-overlap must not evaluate a probe")

        barrier = observer.execute_barrier(
            forbidden_objective,
            anchor,
            remaining_fe=100,
            normal_sweep_fe=20,
            tolerance_fe=7,
        )
        assert barrier.actual_fe == 0
        observer.record_runtime_audit(
            fingerprints_before={"runtime": digest},
            fingerprints_after={"runtime": digest},
            probe_start_fe=60,
            probe_end_fe=60,
        )
        paths = EvidenceOverlayArtifactPaths(
            manifest=raw / "E1_evidence_overlay_manifest.json",
            checkpoint=raw / "E1_evidence_overlay_checkpoint.csv",
            plan=raw / "E1_evidence_overlay_plan.csv",
            probe_evidence=raw / "E1_evidence_overlay_probe_evidence.csv",
            delayed_outcomes=raw / "E1_evidence_overlay_delayed_outcomes.csv",
            shadow_decisions=raw / "E1_evidence_overlay_shadow_decisions.csv",
        )
        manifest = observer.write_artifacts(
            paths=paths,
            native_terminal_error=1.0,
            all_evaluation_best_error=1.0,
        )
        assert manifest["evidence_overlay_fe"] == 0
        _write_csv(
            raw / "E1_aob_input_manifest.csv",
            ("problem_id", "file", "path", "sha256_before", "sha256_after", "unchanged"),
            _canonical_aob_rows("E1"),
        )
        records.append(ExecutionRecord(spec, request, _result(request)))

    collected = collect_artifacts(records, config)
    assert collected.blockers == [], collected.per_run_manifests
    assert len(collected.checkpoints) == 3
    signatures = {
        tuple(
            row[field]
            for field in (
                "checkpoint_fe",
                "fitness_prefix_hash",
                "incumbent_hash",
                "rddsm_topology_hash",
                "rddsm_order_hash",
                "phase_boundary_fe",
            )
        )
        for row in collected.checkpoints.values()
    }
    assert len(signatures) == 1
    gate = build_promotion_gate(
        "smoke",
        config,
        specs,
        GateInputs(
            run_results=collected.run_results,
            ledger_rows=collected.ledger_rows,
            checkpoint_rows=collected.checkpoints,
            plan_rows=collected.plan_rows,
            probe_rows=collected.probe_rows,
            delayed_rows=collected.delayed_rows,
            shadow_rows=collected.shadow_rows,
            aob_rows=collected.aob_rows,
            anti_leakage_rows=collected.anti_leakage_rows,
        ),
    )
    assert gate["checks"]["checkpoint_triplet_parity"] is True
    assert gate["checks"]["e1_zero_probe"] is True


def test_real_observer_writer_round_trips_through_collector(tmp_path: Path) -> None:
    config = load_config()
    spec = next(
        spec
        for spec in build_run_matrix(config, "smoke")
        if spec.problem_id == "A4"
        and spec.max_fes == 100_000
        and spec.lane.evidence_overlay_mode == "paired_owner"
    )
    run_output = tmp_path / "trajectory"
    request = build_execution_request(
        spec,
        run_output,
        config=config,
        python_executable="python",
    )
    raw = run_output / "nested"
    raw.mkdir(parents=True)
    digest = "a" * 64
    observer = HccEvidenceOverlayObserver(
        mode="paired_owner",
        grouping_result=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)),
        problem_id=spec.problem_id,
        seed=spec.seed,
        run_id=spec.trajectory_id,
        configured_max_fes=spec.max_fes,
        terminal_tolerance_fe=7,
        lower_bound=-10.0,
        upper_bound=10.0,
    )
    anchor: tuple[float, ...] = ()
    for sweep in range(3):
        anchor = _record_overlay_sweep(observer, sweep)
    objective = lambda candidate: 1.0 + sum(value * value for value in candidate)
    barrier = observer.execute_barrier(
        objective,
        anchor,
        remaining_fe=100,
        normal_sweep_fe=60,
        tolerance_fe=7,
    )
    assert barrier.actual_fe == 16
    observer.record_runtime_audit(
        fingerprints_before={"runtime": digest},
        fingerprints_after={"runtime": digest},
        probe_start_fe=180,
        probe_end_fe=196,
    )
    _record_overlay_sweep(observer, 3)
    paths = EvidenceOverlayArtifactPaths(
        manifest=raw / f"{spec.problem_id}_evidence_overlay_manifest.json",
        checkpoint=raw / f"{spec.problem_id}_evidence_overlay_checkpoint.csv",
        plan=raw / f"{spec.problem_id}_evidence_overlay_plan.csv",
        probe_evidence=raw / f"{spec.problem_id}_evidence_overlay_probe_evidence.csv",
        delayed_outcomes=raw / f"{spec.problem_id}_evidence_overlay_delayed_outcomes.csv",
        shadow_decisions=raw / f"{spec.problem_id}_evidence_overlay_shadow_decisions.csv",
    )
    manifest = observer.write_artifacts(
        paths=paths,
        native_terminal_error=1.0,
        all_evaluation_best_error=0.9,
    )
    assert manifest["artifacts"]["probe_evidence"] == paths.probe_evidence.name
    _write_csv(
        raw / f"{spec.problem_id}_aob_input_manifest.csv",
        ("problem_id", "file", "path", "sha256_before", "sha256_after", "unchanged"),
        _canonical_aob_rows(spec.problem_id),
    )

    result = _result(request, overlay_fe=16)
    collected = collect_artifacts([ExecutionRecord(spec, request, result)], config)

    assert collected.blockers == [], manifest
    assert len(collected.checkpoints[spec.trajectory_id]["rddsm_topology_hash"]) == 64
    assert collected.run_results[0]["source_mode"] == SOURCE_MODE
    assert collected.run_results[0]["native_terminal_error"] == 1.0
    assert len(collected.plan_rows) == 5
    assert sum(row["selected"] == "1" for row in collected.plan_rows) == 4
    assert len(collected.probe_rows) == 16
    assert {row["candidate"] for row in collected.probe_rows} == {
        "x0",
        "left_owner",
        "right_owner",
        "bridge",
    }
    assert len(collected.delayed_rows) == 8
    assert {row["owner"] for row in collected.delayed_rows} == {"left", "right"}
    assert len(collected.shadow_rows) == 4
    assert collected.aob_rows[0]["unchanged"] == "1"
    assert collected.anti_leakage_rows[0]["audit_status"] == "pass"


def test_missing_runtime_artifacts_materialize_all_outputs_but_fail_closed(tmp_path: Path) -> None:
    observed_modes: list[str] = []

    def fake_runner(request):
        observed_modes.append(request.evidence_overlay_mode)
        overlay_fe = 0 if request.evidence_overlay_mode == "native_audit" else 16
        return _result(request, overlay_fe=overlay_fe)

    output = run_pilot(
        tmp_path / "exp018",
        stage="smoke",
        execution_runner=fake_runner,
        jobs=3,
    )

    assert set(observed_modes) == {"native_audit", "paired_owner", "shuffled_owner"}
    assert all((output / name).is_file() for name in AGGREGATE_ARTIFACTS)
    gate = json.loads((output / "promotion_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pilot_no_go"
    assert gate["runtime_profile_authorized"] is False
    assert any("manifest_invalid" in blocker for blocker in gate["blockers"])
    offline = gate["metrics"]["offline_reference_topology"]
    assert offline["used_for_runtime"] is False
    assert offline["used_for_gate"] is False
    assert {row["problem_id"] for row in offline["by_problem"]} == {
        "E1",
        "E3",
        "A4",
        "S5",
    }
    aggregate_manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        "src/arac/policy/overlap_hypergraph.py"
        in aggregate_manifest["source_bundle"]["files"]
    )
    assert aggregate_manifest["offline_reference_topology_evaluation"][
        "used_for_promotion"
    ] is False
    rows = list(csv.DictReader((output / "run_results.csv").open(encoding="utf-8")))
    assert len(rows) == 15
    with pytest.raises(FileExistsError, match="must be empty"):
        run_pilot(tmp_path / "exp018", stage="smoke", execution_runner=fake_runner, jobs=1)


def test_cli_defaults_to_fresh_runtime_probe() -> None:
    args = parse_args(["--stage", "mechanism"])
    assert args.stage == "mechanism"
    assert args.source_mode == SOURCE_MODE
    assert args.jobs is None


def test_prior_smoke_gate_requires_object_and_complete_current_bindings(
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "promotion_gate.json"
    gate_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="JSON object"):
        _validate_prior_smoke_gate(gate_path, Path("configs/rddsm_evidence_overlay_pilot_v1.json"))

    source = _source_bundle()
    data = _aob_data_bundle(DEFAULT_AOB_DATA_ROOT)
    config_path = Path("configs/rddsm_evidence_overlay_pilot_v1.json").resolve()
    gate = {
        "protocol_version": load_config()["protocol_version"],
        "stage": "smoke",
        "status": "smoke_pass",
        "config_sha256": config_sha256(config_path),
        "source_bundle": source,
        "source_integrity": {
            "source_bundle_unchanged": True,
            "config_unchanged": True,
            "end_source_bundle_sha256": source["sha256"],
            "end_config_sha256": config_sha256(config_path),
        },
        "runtime_binding": {
            "hcc_root": str(Path("vendor/hcc").resolve()),
            "aob_data_root": str(Path(DEFAULT_AOB_DATA_ROOT).resolve()),
            "canonical_roots_required": True,
            "subprocess_environment": EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT,
        },
        "aob_data_bundle": data,
        "aob_data_integrity": {
            "unchanged": True,
            "end_sha256": data["sha256"],
        },
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    _validate_prior_smoke_gate(gate_path, config_path)

    gate["source_integrity"]["end_source_bundle_sha256"] = "0" * 64
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding failed"):
        _validate_prior_smoke_gate(gate_path, config_path)


def test_mechanism_rejects_noncanonical_roots_and_nonfrozen_jobs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen at 24"):
        run_pilot(tmp_path / "jobs", stage="mechanism", jobs=23)
    with pytest.raises(ValueError, match="canonical HCC"):
        run_pilot(tmp_path / "hcc", stage="smoke", hcc_root=tmp_path)
    with pytest.raises(ValueError, match="canonical AOB"):
        run_pilot(tmp_path / "aob", stage="smoke", aob_data_root=tmp_path)
