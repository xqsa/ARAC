from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from arac.backends.hcc import HccAobExecutionRequest, build_hcc_aob_smoke_command
from arac.backends.hcc_hypergraph_trace import (
    HYPERGRAPH_AUDIT_FIELDS,
    HYPERGRAPH_FEATURE_FIELDS,
    HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
    HYPERGRAPH_OUTCOME_FIELDS,
    HYPERGRAPH_PROPOSAL_FIELDS,
    HypergraphTraceArtifactPaths,
    HypergraphTraceObserver,
    write_hypergraph_initialization_failure_manifest,
)
from arac.policy.overlap_hypergraph import build_overlap_hypergraph
from scripts import hcc_smoke_runner as runner


ROOT = Path(__file__).parents[1]


def _request(**overrides: object) -> HccAobExecutionRequest:
    values: dict[str, object] = {
        "problem_id": "E2",
        "seed": 1,
        "max_fes": 5_000,
        "output_dir": Path("results/hypergraph-trace-command"),
        "arac_action": "arac_evidence_action_controller_v37",
        "enable_relation_dispatch": True,
        "relation_policy_mode": "controller_v31",
    }
    values.update(overrides)
    return HccAobExecutionRequest(**values)


def _observer(
    grouping: list[list[int]] | None = None,
    *,
    terminal_target_fe: int = 100,
    terminal_completion_tolerance_fe: int = 100,
) -> HypergraphTraceObserver:
    return HypergraphTraceObserver(
        topology=build_overlap_hypergraph(grouping or [[0, 1], [1, 2]]),
        problem_id="E2",
        seed=91,
        run_id="hypergraph-runtime-test",
        fresh_optimizer_execution=True,
        lower_bound=-5.0,
        upper_bound=5.0,
        rng_descriptor_sha256="a" * 64,
        protocol_config_path=ROOT / "configs" / "hypergraph_delayed_credit_v1.json",
        protocol_spec_path=ROOT
        / "docs"
        / "design"
        / "hypergraph-delayed-credit-v1.md",
        runner_source_path=ROOT / "scripts" / "hcc_smoke_runner.py",
        terminal_target_fe=terminal_target_fe,
        terminal_completion_tolerance_fe=terminal_completion_tolerance_fe,
    )


def _record_complete_sweep(observer: HypergraphTraceObserver, sweep: int) -> list[float]:
    for group in range(2):
        start = sweep * 20 + group * 10
        pre = 100.0 - sweep * 5.0 - group
        before = (0.0, float(sweep + group), 0.0)
        proposal = (0.0, float(sweep + group) + (0.5 if group == 0 else -0.25), 0.0)
        observer.record_group(
            sweep_index=sweep,
            group_index=group,
            pre_error=pre,
            best_error=pre - 1.0,
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


@pytest.mark.parametrize(
    ("terminal_target_fe", "terminal_completion_tolerance_fe", "error"),
    (
        (0, 1, "terminal_target_fe must be positive"),
        (100, 0, "terminal_completion_tolerance_fe must be between"),
        (100, 101, "terminal_completion_tolerance_fe must be between"),
    ),
)
def test_observer_rejects_invalid_terminal_completion_window(
    terminal_target_fe: int,
    terminal_completion_tolerance_fe: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _observer(
            terminal_target_fe=terminal_target_fe,
            terminal_completion_tolerance_fe=terminal_completion_tolerance_fe,
        )


@pytest.mark.parametrize(
    ("terminal_observed_fe", "expected_integrity"),
    ((90, 1), (89, 0), (100, 1), (101, 0)),
)
def test_terminal_completion_window_is_closed_and_fail_closed(
    tmp_path: Path,
    terminal_observed_fe: int,
    expected_integrity: int,
) -> None:
    observer = _observer(
        terminal_target_fe=100,
        terminal_completion_tolerance_fe=10,
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
        final_fitness_record=[100.0] * terminal_observed_fe,
    )

    assert manifest["terminal_target_fe"] == 100
    assert manifest["terminal_observed_fe"] == terminal_observed_fe
    assert manifest["terminal_completion_tolerance_fe"] == 10
    assert manifest["observer_integrity"] == expected_integrity


def test_backend_forwards_observer_only_when_explicitly_enabled() -> None:
    enabled = build_hcc_aob_smoke_command(
        _request(hypergraph_trace_mode="observer")
    )
    disabled = build_hcc_aob_smoke_command(_request())

    index = enabled.argv.index("--hypergraph-trace-mode")
    assert enabled.argv[index + 1] == "observer"
    assert "--hypergraph-trace-mode" not in disabled.argv
    assert "--component-precision-arm" not in enabled.argv
    assert "--precision-causal-arm" not in enabled.argv
    assert "--precision-response-arm" not in enabled.argv


def test_backend_and_runner_fail_closed_for_invalid_observer_profiles() -> None:
    with pytest.raises(ValueError, match="hypergraph_trace_mode"):
        build_hcc_aob_smoke_command(_request(hypergraph_trace_mode="unknown"))
    with pytest.raises(ValueError, match="requires the frozen v37"):
        build_hcc_aob_smoke_command(
            _request(
                arac_action="arac_evidence_action_controller_v38",
                hypergraph_trace_mode="observer",
            )
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_hcc_aob_smoke_command(
            _request(
                hypergraph_trace_mode="observer",
                precision_causal_arm="baseline",
            )
        )

    parsed = runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "2",
            "--output-root",
            "results/hypergraph-cli",
            "--max-fes",
            "5000",
            "--arac-action",
            "arac_evidence_action_controller_v37",
            "--hypergraph-trace-mode",
            "observer",
        ]
    )
    assert parsed.hypergraph_trace_mode == "observer"
    assert runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "2",
            "--output-root",
            "results/hypergraph-cli",
            "--max-fes",
            "5000",
        ]
    ).hypergraph_trace_mode == "off"

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--functions",
                "elliptic",
                "--ids",
                "2",
                "--output-root",
                "results/hypergraph-cli",
                "--max-fes",
                "5000",
                "--arac-action",
                "arac_evidence_action_controller_v38",
                "--hypergraph-trace-mode",
                "observer",
            ]
        )


@pytest.mark.parametrize(
    ("request_field", "arm", "runner_flag"),
    [
        ("precision_causal_arm", "baseline", "--precision-causal-arm"),
        ("precision_response_arm", "a1_probe_only", "--precision-response-arm"),
        ("component_precision_arm", "a0_v37", "--component-precision-arm"),
    ],
)
def test_observer_is_mutually_exclusive_with_every_frozen_precision_route(
    request_field: str,
    arm: str,
    runner_flag: str,
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_hcc_aob_smoke_command(
            _request(hypergraph_trace_mode="observer", **{request_field: arm})
        )

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--functions",
                "elliptic",
                "--ids",
                "2",
                "--output-root",
                "results/hypergraph-cli",
                "--max-fes",
                "5000",
                "--arac-action",
                "arac_evidence_action_controller_v37",
                "--hypergraph-trace-mode",
                "observer",
                runner_flag,
                arm,
            ]
        )


def test_observer_closes_next_sweep_labels_and_writes_independent_artifacts(
    tmp_path: Path,
) -> None:
    observer = _observer()
    record: list[float] = []
    for sweep in range(4):
        record = _record_complete_sweep(observer, sweep)

    observer.record_group(
        sweep_index=4,
        group_index=0,
        pre_error=80.0,
        best_error=79.0,
        primary_requested_fe=8,
        primary_actual_fe=8,
        full_interval_start_fe=80,
        full_interval_end_fe=90,
        pre_block_candidate=(0.0, 4.0, 0.0),
        final_owner_candidate=(0.0, 4.5, 0.0),
    )
    assert not observer.complete_sweep(
        sweep_index=4,
        optimized_group_count=1,
        all_raw_groups_completed=False,
        native_sweep_end_completed=False,
        native_sweep_end_stage=HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
        sweep_end_fe=90,
        sweep_end_candidate=(0.0, 4.25, 0.0),
        fitness_record=[100.0 - index * 0.01 for index in range(90)],
    )
    record = [100.0 - index * 0.01 for index in range(90)]

    paths = HypergraphTraceArtifactPaths(
        manifest=tmp_path / "E2_hypergraph_manifest.json",
        features=tmp_path / "E2_hyperedge_cycle_features.csv",
        audit=tmp_path / "E2_hyperedge_cycle_audit.csv",
        proposals=tmp_path / "E2_shared_proposal_audit.csv",
        outcomes=tmp_path / "E2_hyperedge_cycle_outcomes.csv",
    )
    manifest = observer.write_artifacts(paths=paths, final_fitness_record=record)
    features = _read_csv(paths.features)
    audit = _read_csv(paths.audit)
    proposals = _read_csv(paths.proposals)
    outcomes = _read_csv(paths.outcomes)

    assert tuple(features[0]) == HYPERGRAPH_FEATURE_FIELDS
    assert tuple(audit[0]) == HYPERGRAPH_AUDIT_FIELDS
    assert tuple(proposals[0]) == HYPERGRAPH_PROPOSAL_FIELDS
    assert tuple(outcomes[0]) == HYPERGRAPH_OUTCOME_FIELDS
    assert len(features) == 2
    assert len(audit) == 10
    assert len(proposals) == 8
    assert len(outcomes) == 2
    assert sum(row["outcome_complete"] == "1" for row in outcomes) == 2
    assert sum(row["terminal_censored"] == "1" for row in outcomes) == 0
    assert sum(row["applicable"] == "1" for row in audit) == 2
    assert audit[-1]["not_applicable_reason"] == "incomplete_native_sweep"
    assert all(row["capture_watermark"].endswith("before_relation_writeback") for row in proposals)
    assert all(row["next_sweep_value"] for row in proposals[:6])
    assert all(not row["next_sweep_value"] for row in proposals[6:])

    numeric_payload = {
        field: features[0][field] for field in HYPERGRAPH_FEATURE_FIELDS[1:]
    }
    expected_feature_sha = hashlib.sha256(
        json.dumps(
            numeric_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    joined_audit = {
        row["decision_id"]: row for row in audit if row["applicable"] == "1"
    }
    assert joined_audit[features[0]["decision_id"]]["feature_sha256"] == (
        expected_feature_sha
    )
    assert manifest["topology_source"] == (
        "raw_grouping_result_direct_no_transitive_closure"
    )
    assert manifest["transitive_closure_used"] == 0
    assert manifest["observer_fe"] == 0
    assert manifest["observer_objective_calls"] == 0
    assert manifest["observer_rng_calls"] == 0
    assert manifest["observer_optimizer_calls"] == 0
    assert manifest["complete_sweep_count"] == 4
    assert manifest["incomplete_sweep_count"] == 1
    assert manifest["terminal_censored_outcome_count"] == 0
    assert manifest["decision_lock_consumed"] == 1
    assert manifest["decision_snapshot_sweep"] == 2
    assert manifest["decision_feature_row_count"] == 2
    assert manifest["terminal_target_fe"] == 100
    assert manifest["terminal_observed_fe"] == 90
    assert manifest["terminal_completion_tolerance_fe"] == 100
    assert manifest["observer_integrity"] == 1
    assert set(manifest["artifact_sha256"]) == {
        paths.features.name,
        paths.audit.name,
        paths.proposals.name,
        paths.outcomes.name,
    }


def test_one_shot_terminal_snapshot_is_explicitly_censored(tmp_path: Path) -> None:
    observer = _observer()
    record: list[float] = []
    for sweep in range(3):
        record = _record_complete_sweep(observer, sweep)
    paths = HypergraphTraceArtifactPaths(
        manifest=tmp_path / "manifest.json",
        features=tmp_path / "features.csv",
        audit=tmp_path / "audit.csv",
        proposals=tmp_path / "proposals.csv",
        outcomes=tmp_path / "outcomes.csv",
    )
    manifest = observer.write_artifacts(paths=paths, final_fitness_record=record)
    outcomes = _read_csv(paths.outcomes)

    assert len(_read_csv(paths.features)) == 2
    assert len(outcomes) == 2
    assert all(row["outcome_complete"] == "0" for row in outcomes)
    assert all(row["terminal_censored"] == "1" for row in outcomes)
    assert all(
        row["resolution_sweep_index"] == ""
        and row["next_sweep_unit_fe_contribution"] == ""
        and row["next_sweep_survival"] == ""
        and row["next_sweep_overwrite"] == ""
        for row in outcomes
    )
    assert manifest["terminal_censored_outcome_count"] == 2


def test_closed_labels_do_not_overwrite_inapplicable_tie_status(
    tmp_path: Path,
) -> None:
    observer = _observer()
    record: list[float] = []
    for sweep in range(4):
        for group in range(2):
            start = sweep * 20 + group * 10
            observer.record_group(
                sweep_index=sweep,
                group_index=group,
                pre_error=100.0,
                best_error=99.0,
                primary_requested_fe=8,
                primary_actual_fe=8,
                full_interval_start_fe=start,
                full_interval_end_fe=start + 10,
                pre_block_candidate=(0.0, 0.0, 0.0),
                final_owner_candidate=(0.0, 0.5, 0.0),
            )
        decision_fe = (sweep + 1) * 20
        record = [100.0] * decision_fe
        assert observer.complete_sweep(
            sweep_index=sweep,
            optimized_group_count=2,
            all_raw_groups_completed=True,
            native_sweep_end_completed=True,
            native_sweep_end_stage=HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
            sweep_end_fe=decision_fe,
            sweep_end_candidate=(0.0, 0.25, 0.0),
            fitness_record=record,
        )

    paths = HypergraphTraceArtifactPaths(
        manifest=tmp_path / "manifest.json",
        features=tmp_path / "features.csv",
        audit=tmp_path / "audit.csv",
        proposals=tmp_path / "proposals.csv",
        outcomes=tmp_path / "outcomes.csv",
    )
    manifest = observer.write_artifacts(paths=paths, final_fitness_record=record)
    cohort = [row for row in _read_csv(paths.audit) if row["cohort_locked"] == "1"]

    assert manifest["decision_status"] == "inapplicable"
    assert manifest["decision_reason"] == "focal_priority_tie"
    assert manifest["label_closure"] == "closed"
    assert len(cohort) == 2
    assert all(row["state_complete"] == "1" for row in cohort)
    assert all(row["unique_focal"] == "0" for row in cohort)
    assert all(row["applicable"] == "0" for row in cohort)
    assert all(row["not_applicable_reason"] == "focal_priority_tie" for row in cohort)


def test_cohort_lock_marks_noneligible_raw_groups_as_complete(tmp_path: Path) -> None:
    observer = _observer([[0, 1], [1, 2], [3]])
    record: list[float] = []
    for sweep in range(3):
        for group in range(3):
            start = sweep * 30 + group * 10
            candidate = [0.0, float(sweep + group), 0.0, 0.0]
            proposal = candidate.copy()
            if group < 2:
                proposal[1] += 0.5 if group == 0 else -0.25
            observer.record_group(
                sweep_index=sweep,
                group_index=group,
                pre_error=100.0 - group,
                best_error=99.0 - group,
                primary_requested_fe=8,
                primary_actual_fe=8,
                full_interval_start_fe=start,
                full_interval_end_fe=start + 10,
                pre_block_candidate=candidate,
                final_owner_candidate=proposal,
            )
        decision_fe = (sweep + 1) * 30
        record = [100.0 - index * 0.01 for index in range(decision_fe)]
        assert observer.complete_sweep(
            sweep_index=sweep,
            optimized_group_count=3,
            all_raw_groups_completed=True,
            native_sweep_end_completed=True,
            native_sweep_end_stage=HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
            sweep_end_fe=decision_fe,
            sweep_end_candidate=(0.0, float(sweep) + 0.25, 0.0, 0.0),
            fitness_record=record,
        )

    paths = HypergraphTraceArtifactPaths(
        manifest=tmp_path / "manifest.json",
        features=tmp_path / "features.csv",
        audit=tmp_path / "audit.csv",
        proposals=tmp_path / "proposals.csv",
        outcomes=tmp_path / "outcomes.csv",
    )
    observer.write_artifacts(paths=paths, final_fitness_record=record)
    cohort = [row for row in _read_csv(paths.audit) if row["cohort_locked"] == "1"]
    by_group = {row["group_index"]: row for row in cohort}

    assert set(by_group) == {"0", "1", "2"}
    assert all(row["state_complete"] == "1" for row in cohort)
    assert all(row["unique_focal"] == "1" for row in cohort)
    assert by_group["0"]["applicable"] == "1"
    assert by_group["1"]["applicable"] == "1"
    assert by_group["2"]["applicable"] == "0"
    assert by_group["2"]["not_applicable_reason"] == "no_shared_variables"


def test_no_overlap_and_incomplete_first_opportunities_consume_one_shot_lock(
    tmp_path: Path,
) -> None:
    no_overlap = _observer([[0], [1]])
    no_overlap_record: list[float] = []
    for sweep in range(3):
        no_overlap_record = _record_complete_sweep(no_overlap, sweep)
    no_overlap_paths = HypergraphTraceArtifactPaths(
        manifest=tmp_path / "no-overlap-manifest.json",
        features=tmp_path / "no-overlap-features.csv",
        audit=tmp_path / "no-overlap-audit.csv",
        proposals=tmp_path / "no-overlap-proposals.csv",
        outcomes=tmp_path / "no-overlap-outcomes.csv",
    )
    no_overlap_manifest = no_overlap.write_artifacts(
        paths=no_overlap_paths,
        final_fitness_record=no_overlap_record,
    )
    assert no_overlap_manifest["decision_lock_consumed"] == 1
    assert no_overlap_manifest["decision_status"] == "inapplicable"
    assert no_overlap_manifest["decision_reason"] == "no_shared_hyperedge"
    assert no_overlap_manifest["decision_feature_row_count"] == 0

    incomplete = _observer()
    for sweep in range(2):
        _record_complete_sweep(incomplete, sweep)
    incomplete.record_group(
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
    assert not incomplete.complete_sweep(
        sweep_index=2,
        optimized_group_count=1,
        all_raw_groups_completed=False,
        native_sweep_end_completed=False,
        native_sweep_end_stage=HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
        sweep_end_fe=50,
        sweep_end_candidate=(0.0, 2.25, 0.0),
        fitness_record=[100.0] * 50,
    )
    incomplete_paths = HypergraphTraceArtifactPaths(
        manifest=tmp_path / "incomplete-manifest.json",
        features=tmp_path / "incomplete-features.csv",
        audit=tmp_path / "incomplete-audit.csv",
        proposals=tmp_path / "incomplete-proposals.csv",
        outcomes=tmp_path / "incomplete-outcomes.csv",
    )
    incomplete_manifest = incomplete.write_artifacts(
        paths=incomplete_paths,
        final_fitness_record=[100.0] * 50,
    )
    incomplete_audit = _read_csv(incomplete_paths.audit)
    locked_rows = [row for row in incomplete_audit if row["cohort_locked"] == "1"]
    locked_by_group = {row["group_index"]: row for row in locked_rows}

    assert incomplete_manifest["decision_lock_consumed"] == 1
    assert incomplete_manifest["decision_status"] == "inapplicable"
    assert incomplete_manifest["decision_reason"] == "incomplete_native_sweep"
    assert incomplete_manifest["decision_feature_row_count"] == 0
    assert incomplete_manifest["observer_integrity"] == 1
    assert set(locked_by_group) == {"0", "1"}
    assert locked_by_group["0"]["full_interval_actual_fe"] == "10"
    assert locked_by_group["0"]["watermark_valid"] == "1"
    assert locked_by_group["1"]["state_complete"] == "0"
    assert locked_by_group["1"]["not_applicable_reason"] == (
        "incomplete_native_sweep"
    )
    assert locked_by_group["1"]["source_end_fe"] == ""
    assert locked_by_group["1"]["full_interval_actual_fe"] == ""
    assert locked_by_group["1"]["pre_error"] == ""
    assert locked_by_group["1"]["proposal_capture_watermark"] == ""
    assert locked_by_group["1"]["watermark_valid"] == "0"
    assert locked_by_group["1"]["observer_integrity"] == "1"


def test_runner_observer_watermarks_follow_native_v37_lifecycle() -> None:
    source = inspect.getsource(runner.run_problem)
    observer_source = inspect.getsource(HypergraphTraceObserver)

    group_start = source.index("group_interval_start_fe =")
    existing_precheck = source.index("original_fitness = float(fun(best_individual)[0])")
    proposal_capture = source.index("hypergraph_observer.record_group(")
    contribution_append = source.index("fitness_delta_list.append(current_delta)")
    native_credit_close = source.index("component_credit_trace.complete_sweep(")
    outcome_close = source.index("hypergraph_observer.complete_sweep(")
    next_sweep = source.index("outer_iter += 1")

    assert group_start < existing_precheck < proposal_capture < contribution_append
    assert native_credit_close < outcome_close < next_sweep
    assert "topology=build_overlap_hypergraph(grouping_result)" in source
    assert "terminal_target_fe=config.max_fes" in source
    assert (
        "terminal_completion_tolerance_fe=(\n"
        "                    terminal_completion_tolerance_fe\n"
        "                )"
    ) in source
    assert source.index("terminal_completion_tolerance_fe =") < source.index(
        "hypergraph_observer = HypergraphTraceObserver("
    )
    assert "final_owner_candidate=best_individual.copy()" in source
    assert "primary_evaluations_before != group_interval_start_fe + 1" in source
    assert "primary_evaluations_before + primary_cc_fe" in source
    assert "build_overlap_components(grouping_result) if hypergraph" not in source
    assert "if len(eligible) < 2:" in observer_source
    recovery = source.index("reconcile_trajectory_recovery_context(")
    relation_dispatch = source.index("if index > 0:", proposal_capture)
    assert recovery < proposal_capture < relation_dispatch
    assert 'stage="group_capture"' in source
    assert 'stage="sweep_closure"' in source
    assert 'stage="artifact_write"' in source
    assert "hypergraph_observer_active = False" in source


def test_observer_call_graph_has_no_objective_rng_or_optimizer_handle() -> None:
    source = inspect.getsource(HypergraphTraceObserver)
    tree = ast.parse(source)
    parameter_names = {
        argument.arg
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (*node.args.args, *node.args.kwonlyargs)
    }
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert not {"fun", "objective", "optimizer", "rng"} & parameter_names
    assert not {"CMAES", "MMES", "default_rng", "derive_optimizer_seed"} & call_names
    action_trace_fields = set(
        runner.ACTION_TRACE_FIELDS
        + runner.V33_ACTION_TRACE_FIELDS
        + runner.V34_ACTION_TRACE_FIELDS
        + runner.V36_ACTION_TRACE_FIELDS
        + runner.V37_ACTION_TRACE_FIELDS
    )
    assert not any(field.startswith("hypergraph") for field in action_trace_fields)


def test_observer_failure_is_explicit_and_native_closure_requires_real_bool(
    tmp_path: Path,
) -> None:
    observer = _observer()
    for group in range(2):
        observer.record_group(
            sweep_index=0,
            group_index=group,
            pre_error=100.0,
            best_error=99.0,
            primary_requested_fe=8,
            primary_actual_fe=8,
            full_interval_start_fe=group * 10,
            full_interval_end_fe=(group + 1) * 10,
            pre_block_candidate=(0.0, 0.0, 0.0),
            final_owner_candidate=(0.0, 0.5, 0.0),
        )
    with pytest.raises(TypeError, match="must be boolean"):
        observer.complete_sweep(
            sweep_index=0,
            optimized_group_count=2,
            all_raw_groups_completed=True,
            native_sweep_end_completed="0",  # type: ignore[arg-type]
            native_sweep_end_stage=HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
            sweep_end_fe=20,
            sweep_end_candidate=(0.0, 0.25, 0.0),
            fitness_record=[100.0] * 20,
        )
    with pytest.raises(ValueError, match="completion stage"):
        observer.complete_sweep(
            sweep_index=0,
            optimized_group_count=2,
            all_raw_groups_completed=True,
            native_sweep_end_completed=True,
            native_sweep_end_stage="before_search_state_handler",
            sweep_end_fe=20,
            sweep_end_candidate=(0.0, 0.25, 0.0),
            fitness_record=[100.0] * 20,
        )

    observer.record_failure(
        stage="sweep_closure",
        error=ValueError("synthetic trace failure"),
        source_fe=20,
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
        final_fitness_record=[100.0] * 20,
    )

    assert manifest["fresh_optimizer_execution"] == 1
    assert manifest["observer_status"] == "failed"
    assert manifest["observer_error_stage"] == "sweep_closure"
    assert manifest["observer_error_type"] == "ValueError"
    assert manifest["terminal_target_fe"] == 100
    assert manifest["terminal_observed_fe"] == 20
    assert manifest["terminal_completion_tolerance_fe"] == 100
    assert manifest["observer_integrity"] == 0


def test_initialization_failure_manifest_is_fail_closed(tmp_path: Path) -> None:
    manifest = write_hypergraph_initialization_failure_manifest(
        path=tmp_path / "E2_hypergraph_manifest.json",
        problem_id="E2",
        seed=91,
        run_id="init-failure",
        fresh_optimizer_execution=True,
        terminal_target_fe=100,
        terminal_completion_tolerance_fe=10,
        error=FileNotFoundError("missing protocol binding"),
        source_fe=0,
    )

    assert manifest["observer_status"] == "failed"
    assert manifest["observer_error_stage"] == "initialization"
    assert manifest["observer_error_type"] == "FileNotFoundError"
    assert manifest["terminal_target_fe"] == 100
    assert manifest["terminal_observed_fe"] == 0
    assert manifest["terminal_completion_tolerance_fe"] == 10
    assert manifest["observer_integrity"] == 0
