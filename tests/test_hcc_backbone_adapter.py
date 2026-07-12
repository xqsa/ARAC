from __future__ import annotations

from pathlib import Path

import pytest

from arac.actions import ActionFamily
from arac.backends import hcc as hcc_backend
from arac.backends.hcc import (
    HccBackboneSnapshot,
    HccGroupSignal,
    build_hcc_action_execution_plan,
    build_hcc_evidence_profile,
    hcc_backend_semantics_for,
    load_hcc_aob_topology,
)
from arac.evidence import validate_runtime_payload
from arac.actions import ActionDecision


def test_explicit_vendor_root_resolves_hcc_source_boundary(tmp_path: Path) -> None:
    vendor_root = tmp_path / "repo" / "vendor" / "hcc"
    (vendor_root / "AOB").mkdir(parents=True)
    (vendor_root / "HCC").mkdir()
    runner = tmp_path / "repo" / "scripts" / "hcc_smoke_runner.py"
    runner.parent.mkdir()
    runner.write_text("# test runner\n", encoding="utf-8")

    paths = hcc_backend.resolve_hcc_vendor_paths(
        vendor_root,
        repo_root=tmp_path / "repo",
    )

    assert paths.vendor_root == vendor_root.resolve()
    assert paths.aob_root == vendor_root.resolve() / "AOB"
    assert paths.hcc_root == vendor_root.resolve() / "HCC"
    assert paths.aob_data_root == vendor_root.resolve() / "AOB" / "AOBG" / "datafile"
    assert paths.runner == runner.resolve()


def test_vendor_root_requires_aob_hcc_and_explicit_runner_context(tmp_path: Path) -> None:
    vendor_root = tmp_path / "vendor-copy"
    vendor_root.mkdir()

    with pytest.raises(FileNotFoundError, match="valid HCC vendor root.*AOB"):
        hcc_backend.resolve_hcc_vendor_paths(
            vendor_root,
            runner_path=tmp_path / "runner.py",
        )

    (vendor_root / "AOB").mkdir()
    with pytest.raises(FileNotFoundError, match="valid HCC vendor root.*HCC"):
        hcc_backend.resolve_hcc_vendor_paths(
            vendor_root,
            runner_path=tmp_path / "runner.py",
        )

    (vendor_root / "HCC").mkdir()
    with pytest.raises(ValueError, match="repo_root or runner_path"):
        hcc_backend.resolve_hcc_vendor_paths(vendor_root)


def test_external_hcc_main_root_is_rejected_as_offline_only(tmp_path: Path) -> None:
    external_root = tmp_path / "HCC-main"
    (external_root / "AOB").mkdir(parents=True)
    (external_root / "HCC").mkdir()

    with pytest.raises(ValueError, match="offline-only.*vendor root"):
        hcc_backend.resolve_hcc_vendor_paths(
            external_root,
            runner_path=tmp_path / "hcc_smoke_runner.py",
        )


def test_shallow_invalid_vendor_root_never_leaks_index_error(tmp_path: Path) -> None:
    shallow_root = tmp_path.anchor

    with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
        hcc_backend.resolve_hcc_vendor_paths(shallow_root)

    assert not isinstance(exc_info.value, IndexError)


def test_hcc_snapshot_builds_reference_blind_evidence_profile() -> None:
    snapshot = HccBackboneSnapshot(
        run_id="hcc-smoke",
        problem_id="S4",
        seed=7,
        dimension=1000,
        group_count=4,
        overlap_group_count=2,
        overlapping_element_count=50,
        budget_remaining_ratio=0.75,
        groups=(
            HccGroupSignal(group_id="g0", fitness_delta=12.0, rank=1, shared_variable_count=20),
            HccGroupSignal(group_id="g1", fitness_delta=4.0, rank=2, shared_variable_count=15),
            HccGroupSignal(group_id="g2", fitness_delta=1.0, rank=3, shared_variable_count=0),
            HccGroupSignal(group_id="g3", fitness_delta=0.5, rank=4, shared_variable_count=0),
        ),
    )

    evidence = build_hcc_evidence_profile(snapshot)

    assert evidence.run_id == "hcc-smoke"
    assert evidence.problem_id == "S4"
    assert evidence.unit_type == "problem"
    assert evidence.overlap_degree == pytest.approx(0.5)
    assert evidence.shared_var_support_ratio == pytest.approx(0.05)
    assert evidence.group_gain_asymmetry > 0
    assert evidence.priority_spread == pytest.approx(0.75)
    assert evidence.budget_remaining_ratio == pytest.approx(0.75)
    validate_runtime_payload(evidence.__dict__)


def test_hcc_backend_semantics_maps_action_families_to_hcc_effects() -> None:
    decisions = {
        "isolate": ActionDecision(
            ActionFamily.ISOLATE,
            "isolate_conflicting_relation",
            "allow",
            "test",
            0.4,
        ),
        "protect": ActionDecision(
            ActionFamily.PROTECT,
            "protect_high_margin_group",
            "allow",
            "test",
            0.4,
        ),
        "repair": ActionDecision(
            ActionFamily.REASSIGN_REPAIR,
            "repair_shared_variable_binding",
            "allow",
            "test",
            0.4,
        ),
        "coordinate": ActionDecision(
            ActionFamily.COORDINATE,
            "allow_beneficial_coordination",
            "allow",
            "test",
            0.4,
        ),
        "bipop": ActionDecision(
            ActionFamily.TRAJECTORY,
            "bipop_search_state_restart",
            "allow",
            "test",
            0.4,
        ),
        "phase_rescue": ActionDecision(
            ActionFamily.TRAJECTORY,
            "phase_rescue_multistart",
            "allow",
            "test",
            0.4,
        ),
        "repair_phase_rescue": ActionDecision(
            ActionFamily.TRAJECTORY,
            "repair_phase_rescue_multistart",
            "allow",
            "test",
            0.4,
        ),
        "repair_bipop": ActionDecision(
            ActionFamily.TRAJECTORY,
            "repair_bipop_search_state_restart",
            "allow",
            "test",
            0.4,
        ),
        "repair_refine": ActionDecision(
            ActionFamily.TRAJECTORY,
            "repair_protect_refine",
            "allow",
            "test",
            0.4,
        ),
        "deep_refine": ActionDecision(
            ActionFamily.TRAJECTORY,
            "repair_protect_deep_refine",
            "allow",
            "test",
            0.4,
        ),
        "cc_harm_guarded_sep_refresh": ActionDecision(
            ActionFamily.TRAJECTORY,
            "cc_harm_guarded_sep_refresh",
            "allow",
            "test",
            0.4,
        ),
        "separable_cmaes_dispatch_action": ActionDecision(
            ActionFamily.TRAJECTORY,
            "separable_cmaes_dispatch_action",
            "allow",
            "test",
            0.4,
        ),
    }

    isolate_diff = hcc_backend_semantics_for(decisions["isolate"], optimizer_consumed=True)
    assert isolate_diff.relation_handling_changed
    assert not isolate_diff.variable_owner_changed
    assert hcc_backend_semantics_for(
        decisions["protect"], optimizer_consumed=True
    ).budget_allocation_changed
    assert hcc_backend_semantics_for(
        decisions["repair"], optimizer_consumed=True
    ).variable_owner_changed
    assert hcc_backend_semantics_for(
        decisions["coordinate"], optimizer_consumed=True
    ).coordination_mode_changed
    bipop_diff = hcc_backend_semantics_for(decisions["bipop"], optimizer_consumed=True)
    assert bipop_diff.budget_allocation_changed
    assert bipop_diff.update_order_changed
    assert bipop_diff.acceptance_rule_changed
    phase_rescue_diff = hcc_backend_semantics_for(decisions["phase_rescue"], optimizer_consumed=True)
    assert phase_rescue_diff.budget_allocation_changed
    assert phase_rescue_diff.update_order_changed
    assert phase_rescue_diff.acceptance_rule_changed
    assert not phase_rescue_diff.variable_owner_changed
    repair_phase_rescue_diff = hcc_backend_semantics_for(
        decisions["repair_phase_rescue"],
        optimizer_consumed=True,
    )
    assert repair_phase_rescue_diff.variable_owner_changed
    assert repair_phase_rescue_diff.budget_allocation_changed
    assert repair_phase_rescue_diff.update_order_changed
    assert repair_phase_rescue_diff.acceptance_rule_changed
    repair_bipop_diff = hcc_backend_semantics_for(
        decisions["repair_bipop"],
        optimizer_consumed=True,
    )
    assert repair_bipop_diff.variable_owner_changed
    assert repair_bipop_diff.budget_allocation_changed
    assert repair_bipop_diff.update_order_changed
    assert repair_bipop_diff.acceptance_rule_changed
    repair_refine_diff = hcc_backend_semantics_for(
        decisions["repair_refine"],
        optimizer_consumed=True,
    )
    assert repair_refine_diff.variable_owner_changed
    assert repair_refine_diff.budget_allocation_changed
    assert repair_refine_diff.acceptance_rule_changed
    assert not repair_refine_diff.update_order_changed
    deep_refine_diff = hcc_backend_semantics_for(
        decisions["deep_refine"],
        optimizer_consumed=True,
    )
    assert deep_refine_diff.variable_owner_changed
    assert deep_refine_diff.budget_allocation_changed
    assert deep_refine_diff.acceptance_rule_changed
    assert not deep_refine_diff.update_order_changed
    cc_harm_diff = hcc_backend_semantics_for(
        decisions["cc_harm_guarded_sep_refresh"],
        optimizer_consumed=True,
    )
    assert cc_harm_diff.budget_allocation_changed
    assert cc_harm_diff.update_order_changed
    assert cc_harm_diff.acceptance_rule_changed
    assert not cc_harm_diff.variable_owner_changed
    separable_diff = hcc_backend_semantics_for(
        decisions["separable_cmaes_dispatch_action"],
        optimizer_consumed=True,
    )
    assert separable_diff.budget_allocation_changed
    assert separable_diff.update_order_changed
    assert separable_diff.acceptance_rule_changed
    assert not separable_diff.variable_owner_changed


def test_hcc_backend_semantics_stay_empty_without_optimizer_consumption() -> None:
    decision = ActionDecision(
        ActionFamily.REASSIGN_REPAIR,
        "repair_shared_variable_binding",
        "allow",
        "test",
        0.4,
    )

    diff = hcc_backend_semantics_for(decision, optimizer_consumed=False)

    assert not diff.changed


def test_hcc_backend_binds_resumed_phase_i_state_action() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "resume_phase_i_search_state",
        "allow",
        "runtime_state_evidence",
        0.9,
    )

    plan = build_hcc_action_execution_plan("R3", decision)

    assert plan.selected_action_family == "trajectory"
    assert plan.backend_effect_kind == "resumable_mmes_state_block"
    assert plan.optimizer_consumed is True
    assert plan.execution_mode == "hcc_stateful_search_action"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_snapshot_rejects_forbidden_outcome_fields() -> None:
    snapshot = HccBackboneSnapshot(
        run_id="bad",
        problem_id="S4",
        seed=7,
        dimension=1000,
        group_count=1,
        overlap_group_count=0,
        overlapping_element_count=0,
        budget_remaining_ratio=0.5,
        groups=(HccGroupSignal(group_id="g0", fitness_delta=1.0, rank=1),),
        runtime_payload_extra={"final_error": 1.23},
    )

    with pytest.raises(ValueError, match="final_error"):
        build_hcc_evidence_profile(snapshot)


def test_load_hcc_aob_topology_reads_source_metadata_without_optimizer_run() -> None:
    topology = load_hcc_aob_topology("S6")

    assert topology.problem_id == "S6"
    assert topology.function_name == "schwefel"
    assert topology.function_id == 6
    assert topology.dimension == 1000
    assert topology.dimension_real == 1190
    assert topology.overlap_gamma == 10
    assert topology.group_count == 20
    assert topology.overlap_group_count == 19
    assert topology.overlapping_element_count == 190
    assert topology.degree_of_overlap == pytest.approx(0.19)
    assert topology.global_fes == 1_056_000
    assert topology.source_level == "hcc_source_topology"
    assert topology.fresh_optimizer_execution is False
    assert topology.groups[0].shared_variable_count == 10


def test_hcc_aob_topology_preserves_aob_overlap_gradient() -> None:
    topologies = [load_hcc_aob_topology(f"E{idx}") for idx in range(1, 7)]

    assert [topology.overlap_gamma for topology in topologies] == [0, 1, 3, 5, 7, 10]
    assert [topology.dimension for topology in topologies] == [1000] * 6
    assert [topology.dimension_real for topology in topologies] == [1000, 1019, 1057, 1095, 1133, 1190]
