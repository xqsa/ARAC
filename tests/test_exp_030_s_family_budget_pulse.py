from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arac.actions.budget_reallocation import BudgetAllocationExecutionState
from arac.actions.shrunk_budget_pulse import ShrunkBudgetPulseExecutionState
from arac.backends.hcc import required_aob_data_files
from arac.backends.hcc_action_ceiling import (
    freeze_efficiency_budget_action,
    freeze_shrunk_efficiency_budget_pulse_action,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
    ACTION_CEILING_HORIZONS,
    S_FAMILY_BUDGET_PULSE_ARMS,
    actionability_delta,
)
from experiments.pilots.exp_030_s_family_budget_pulse import run as exp030


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_aob_root(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "F5-info.txt").write_text(
        "dimension: 1000\nsub_num: 20\nsubgroups_type: []\n",
        encoding="utf-8",
    )
    for suffix in ("design", "p", "s", "w", "xopt"):
        (root / f"F5-{suffix}.txt").write_text(f"fixture-{suffix}\n", encoding="utf-8")
    return root


def _schedule(
    *,
    horizon: int,
    action_budgets: tuple[int, ...] | None,
    start_group: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    sweeps: list[int] = []
    groups: list[int] = []
    budgets: list[int] = []
    starts: list[int] = []
    sweep = 1
    group = start_group
    relative_fe = 1
    while relative_fe <= horizon:
        budget = action_budgets[group] if action_budgets is not None and sweep == 2 else 10
        sweeps.append(sweep)
        groups.append(group)
        budgets.append(budget)
        starts.append(relative_fe)
        relative_fe += 1 + budget
        group += 1
        if group == exp030.EXPECTED_GROUPS:
            sweep += 1
            group = 0
    return sweeps, groups, budgets, starts


def _lifecycle(action: object, dispatch_fe: int, relative_application_fe: int) -> dict[str, object]:
    if action.audit_payload()["action"] == exp030.SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION:
        state = ShrunkBudgetPulseExecutionState.for_action(action)
    else:
        state = BudgetAllocationExecutionState.for_action(action)
    state.consume(
        action,
        current_sweep=action.target_sweep,
        application_fe=dispatch_fe + relative_application_fe,
        dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
        anchor_hash=action.anchor_hash,
    )
    payload = {
        "action": action.audit_payload()["action"],
        "instance": action.audit_payload(),
        "instance_hash": action.action_hash,
        "execution": state.audit_payload(action),
        "execution_hash": state.state_hash(action),
    }
    return payload


def _overlay_identity(seed: int) -> dict[str, str]:
    identity = {
        "fitness_prefix_hash": _hash(f"overlay-fitness-prefix-{seed}"),
        "incumbent_hash": _hash(f"overlay-incumbent-{seed}"),
        "rddsm_topology_hash": _hash("overlay-topology"),
        "rddsm_order_hash": _hash("overlay-order"),
    }
    identity["checkpoint_hash"] = _canonical_hash(
        {
            "problem_id": exp030.CASE,
            "seed": seed,
            "checkpoint_fe": 1000,
            **{key: value for key, value in identity.items()},
        }
    )
    return identity


def _fixture_rows(
    *,
    seed: int = exp030.SEED,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    contexts: list[dict[str, str]] = []
    arm_rows: list[dict[str, str]] = []
    overlay_identity = _overlay_identity(seed)
    populations = (2,) * exp030.EXPECTED_GROUPS
    uniform = (10,) * exp030.EXPECTED_GROUPS
    efficiency = tuple(float(exp030.EXPECTED_GROUPS - index) for index in range(20))
    horizon_fe = sum(budget + 1 for budget in uniform)
    for context_index in range(exp030.EXPECTED_CONTEXTS):
        dispatch_fe = 10_000 + context_index * 50
        dispatch_hash = _hash(f"dispatch-{context_index}")
        context = {field: "" for field in ACTION_CEILING_CONTEXT_FIELDS}
        relation_id = f"g{context_index}-{context_index + 1}:v{context_index}"
        right_values = (float(context_index),)
        dispatch_anchor_hash = _hash(f"dispatch-current-values-{context_index}")
        context_id = (
            f"real_aob:S5:seed{seed}:s1:g{context_index}-{context_index + 1}:"
            f"{dispatch_hash[:12]}"
        )
        context.update(
            {
                "protocol_version": exp030.PROTOCOL_VERSION,
                "cohort": exp030.COHORT,
                "problem_id": exp030.CASE,
                "seed": str(seed),
                "context_id": context_id,
                "relation_id": relation_id,
                "action_set_hash": _hash(f"action-set-{context_index}"),
                "checkpoint_hash": overlay_identity["checkpoint_hash"],
                "dispatch_checkpoint_hash": dispatch_hash,
                "dispatch_anchor_hash": dispatch_anchor_hash,
                "phase_boundary_fe": "1000",
                "dispatch_fe": str(dispatch_fe),
                "issued_sweep": "0",
                "target_sweep": "1",
                "group_index": str(context_index + 1),
                "efficiency_ewma": json.dumps(efficiency),
                "completed_efficiency_sweeps": "1",
                "stagnation_streaks": json.dumps([0] * 20),
                "population_sizes": json.dumps(populations),
                "uniform_group_budgets": json.dumps(uniform),
                "horizon_fe": str(horizon_fe),
                "selector_arm": "native_eq8",
                "selector_reason": "fixture_abstain",
                "anchor_values": "[0.0]",
                "left_values": "[0.0]",
                "right_values": json.dumps(right_values),
                "bridge_values": "[0.0]",
                "bridge_weights": '{"left_owner": 0.5, "right_owner": 0.5}',
                "native_parity": "1",
                "runtime_authorized": "0",
                "status": "complete",
            }
        )
        contexts.append(context)
        raw = freeze_efficiency_budget_action(
            problem_id=exp030.CASE,
            run_seed=seed,
            checkpoint_fe=dispatch_fe,
            dispatch_checkpoint_hash=dispatch_hash,
            source_efficiency_ewma=efficiency,
            population_sizes=populations,
            uniform_group_budgets=uniform,
            issued_sweep=1,
            target_sweep=2,
        )
        shrunk = freeze_shrunk_efficiency_budget_pulse_action(
            problem_id=exp030.CASE,
            run_seed=seed,
            checkpoint_fe=dispatch_fe,
            dispatch_checkpoint_hash=dispatch_hash,
            raw_group_budgets=raw.group_budgets,
            population_sizes=populations,
            uniform_group_budgets=uniform,
            issued_sweep=1,
            target_sweep=2,
        )
        action_by_arm = {
            exp030.FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION: raw,
            exp030.SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION: shrunk,
        }
        start_group = context_index + 2
        full_raw_trace = _schedule(
            horizon=3 * horizon_fe,
            action_budgets=raw.group_budgets,
            start_group=start_group,
        )
        raw_target_start = min(
            start
            for sweep, start in zip(full_raw_trace[0], full_raw_trace[3], strict=True)
            if sweep == raw.target_sweep
        )
        lifecycle_by_arm = {
            arm: _lifecycle(action, dispatch_fe, raw_target_start)
            for arm, action in action_by_arm.items()
        }
        native_post_hash = _hash(f"native-post-{context_index}")
        native_errors = {"immediate": 100.0, "sweep_1": 90.0, "sweep_3": 80.0}
        error_shift = {
            "native_eq8": 0.0,
            "true_no_writeback": 1.0,
            exp030.FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION: -1.0,
            exp030.SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION: -2.0,
        }
        target_relative = {"immediate": 1, "sweep_1": horizon_fe, "sweep_3": 3 * horizon_fe}
        for arm in S_FAMILY_BUDGET_PULSE_ARMS:
            action = action_by_arm.get(arm)
            action_budgets = None if action is None else action.group_budgets
            for horizon in ACTION_CEILING_HORIZONS:
                relative_fe = target_relative[horizon]
                sweeps, groups, budgets, starts = _schedule(
                    horizon=relative_fe,
                    action_budgets=action_budgets,
                    start_group=start_group,
                )
                target_visible = action is not None and action.target_sweep in sweeps
                native_error = native_errors[horizon]
                arm_error = native_error + error_shift[arm]
                row = {field: "" for field in ACTION_CEILING_ARM_RESULT_FIELDS}
                row.update(
                    {
                        "protocol_version": exp030.PROTOCOL_VERSION,
                        "cohort": exp030.COHORT,
                        "problem_id": exp030.CASE,
                        "seed": str(seed),
                        "context_id": context["context_id"],
                        "arm": arm,
                        "horizon": horizon,
                        "target_fe": str(dispatch_fe + relative_fe),
                        "natural_endpoint_fe": str(dispatch_fe + 3 * horizon_fe),
                        "native_error": f"{native_error:.17e}",
                        "arm_error": f"{arm_error:.17e}",
                        "delta": f"{actionability_delta(native_error, arm_error):.17e}",
                        "action_budget_fes": "0",
                        "action_actual_fes": "0",
                        "action_accepted": "1" if action is not None else "0",
                        "action_post_incumbent_hash": (
                            native_post_hash if arm != "true_no_writeback" else ""
                        ),
                        "optimizer_scope": (
                            "decomposed_groups" if action is not None else "relation_writeback"
                        ),
                        "optimizer_population_size": "0",
                        "optimizer_generation_count": "0",
                        "counterfactual_applied": ("0" if arm == "true_no_writeback" else "1"),
                        "mutation_norm": "0.0" if arm == "true_no_writeback" else "0.1",
                        "optimizer_mean_mutation_norm": "0.0",
                        "continuation_policy_applied": str(int(target_visible)),
                        "execution_sweep_trace": json.dumps(sweeps),
                        "execution_order_trace": json.dumps(groups),
                        "group_budget_trace": json.dumps(budgets),
                        "execution_start_fe_trace": json.dumps(starts),
                        "warm_start_trigger_count": "0",
                        "warm_start_mean_shift_norm": "0.0",
                        "selected_candidate": "current" if arm == "true_no_writeback" else arm,
                        "runtime_authorized": "0",
                        "status": "complete",
                    }
                )
                if action is not None:
                    lifecycle = lifecycle_by_arm[arm]
                    row.update(
                        {
                            "action_instance_hash": action.action_hash,
                            "action_lifecycle_payload": json.dumps(
                                lifecycle,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "action_lifecycle_hash": _canonical_hash(lifecycle),
                            "action_candidate_hash": _canonical_hash(
                                {"group_budgets": action.group_budgets}
                            ),
                            "optimizer_parameter_hash": action.parameter_hash,
                        }
                    )
                arm_rows.append(row)
    return contexts, arm_rows


def _write_overlay_fixture(
    artifact_dir: Path,
    contexts: list[dict[str, str]],
    *,
    seed: int,
    run_id: str,
) -> None:
    identity = _overlay_identity(seed)
    relations = [row["relation_id"] for row in contexts]
    resolution_fe = max(int(row["dispatch_fe"]) for row in contexts)
    common = {
        "problem_id": exp030.CASE,
        "seed": str(seed),
        "mode": "paired_owner",
        "runtime_authorized": "0",
    }
    checkpoint = {field: "" for field in exp030.CHECKPOINT_FIELDS}
    checkpoint.update(
        {
            **common,
            "checkpoint_fe": "1000",
            "fitness_prefix_hash": identity["fitness_prefix_hash"],
            "incumbent_hash": identity["incumbent_hash"],
            "rddsm_topology_hash": identity["rddsm_topology_hash"],
            "rddsm_order_hash": identity["rddsm_order_hash"],
            "phase_boundary_fe": "1000",
            "history_sweeps": "0;1;2",
            "previous_survival_closed": "1",
            "plan_status": "selected",
            "plan_reason": "top_relation_set_selected",
        }
    )
    plan_rows = []
    probe_rows = []
    delayed_rows = []
    shadow_rows = []
    candidate_utilities = {
        "x0": 0.0,
        "left_owner": 0.01,
        "right_owner": 0.03,
        "bridge": 0.02,
    }
    for relation_index, relation_id in enumerate(relations):
        plan = {field: "" for field in exp030.PLAN_FIELDS}
        plan.update(
            {
                **common,
                "relation_id": relation_id,
                "owner_groups": f"{relation_index};{relation_index + 1}",
                "shared_variables": str(relation_index),
                "selected": "1",
                "score_source_relation_id": relation_id,
                "phase_boundary_fe": "1000",
            }
        )
        plan_rows.append(plan)
        for candidate, utility in candidate_utilities.items():
            probe = {field: "" for field in exp030.PROBE_EVIDENCE_FIELDS}
            probe.update(
                {
                    **common,
                    "relation_id": relation_id,
                    "candidate": candidate,
                    "fitness": f"{100.0 / math.exp(utility):.17e}",
                    "utility": f"{utility:.17e}",
                    "owner_reliability": (
                        "" if candidate == "x0" else f"{0.5:.17e}"
                    ),
                    "candidate_hash": (
                        identity["incumbent_hash"]
                        if candidate == "x0"
                        else _hash(f"{seed}-{relation_id}-{candidate}")
                    ),
                    "phase_boundary_fe": "1000",
                    "actual_fe": "1",
                }
            )
            probe_rows.append(probe)
        for owner, survival in (("left", 0.4), ("right", 0.6)):
            delayed = {field: "" for field in exp030.DELAYED_OUTCOME_FIELDS}
            delayed.update(
                {
                    **common,
                    "relation_id": relation_id,
                    "owner": owner,
                    "action_sweep_index": "0",
                    "resolution_sweep_index": "1",
                    "survival_label": f"{survival:.17e}",
                    "overwrite_label": f"{1.0 - survival:.17e}",
                    "next_sweep_log_improvement": f"{0.01:.17e}",
                    "overwrite_penalized_credit": f"{0.005:.17e}",
                    "label_closed": "1",
                    "label_status": "closed_next_complete_sweep",
                    "resolution_fe": str(resolution_fe),
                }
            )
            delayed_rows.append(delayed)
        shadow = {field: "" for field in exp030.SHADOW_DECISION_FIELDS}
        shadow.update(
            {
                **common,
                "relation_id": relation_id,
                "action": "repair",
                "winner": "right_owner",
                "utility": f"{candidate_utilities['right_owner']:.17e}",
                "reason": "unique_probe_winner_above_one_percent",
            }
        )
        shadow_rows.append(shadow)

    artifact_rows = {
        "checkpoint": (exp030.CHECKPOINT_FIELDS, [checkpoint]),
        "delayed_outcomes": (exp030.DELAYED_OUTCOME_FIELDS, delayed_rows),
        "plan": (exp030.PLAN_FIELDS, plan_rows),
        "probe_evidence": (exp030.PROBE_EVIDENCE_FIELDS, probe_rows),
        "runtime_actions": (exp030.RUNTIME_ACTION_FIELDS, []),
        "shadow_decisions": (exp030.SHADOW_DECISION_FIELDS, shadow_rows),
    }
    artifact_names = {
        key: f"S5_evidence_overlay_{key}.csv" for key in artifact_rows
    }
    for key, (fields, rows) in artifact_rows.items():
        _write_csv(artifact_dir / artifact_names[key], fields, rows)
    artifact_hashes = {
        name: hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()
        for name in artifact_names.values()
    }
    state_fingerprints = {
        component: {
            "before": _hash(f"state-{component}"),
            "after": _hash(f"state-{component}"),
        }
        for component in exp030.OVERLAY_STATE_FINGERPRINT_COMPONENTS
    }
    runtime_fingerprint = _canonical_hash(
        {
            component: state_fingerprints[component]["before"]
            for component in sorted(state_fingerprints)
        }
    )
    manifest = {
        "protocol_version": exp030.EVIDENCE_OVERLAY_PROTOCOL_VERSION,
        "schema_version": 2,
        "source_mode": exp030.EVIDENCE_OVERLAY_SOURCE_MODE,
        "problem_id": exp030.CASE,
        "seed": seed,
        "run_id": run_id,
        "configured_max_fes": exp030.CONFIGURED_MAX_FES,
        "evidence_overlay_mode": "paired_owner",
        "terminal_tolerance_rule": exp030.TERMINAL_TOLERANCE_RULE,
        "terminal_tolerance_fe": 2,
        "runtime_input_fields": list(exp030.RUNTIME_INPUT_FIELDS),
        "phase_boundary_fe": 1000,
        "rddsm_topology_hash": identity["rddsm_topology_hash"],
        "rddsm_order_hash": identity["rddsm_order_hash"],
        "probe_start_fe": 1000,
        "probe_end_fe": 1016,
        "objective_calls": 16,
        "evidence_overlay_fe": 16,
        "optimizer_calls": 0,
        "rng_calls": 0,
        "failure": None,
        "applicable": 1,
        "abstain_reason": "",
        "barrier_status": "probed",
        "barrier_reason": "four_point_probe_complete",
        "selected_relation_count": 4,
        "delayed_outcomes_required": 1,
        "delayed_label_expected": 8,
        "delayed_label_closed": 8,
        "fresh_optimizer_execution": 1,
        "observer_integrity": 1,
        "native_state_unchanged": 1,
        "aob_truth_runtime_used": 0,
        "runtime_authorized": 0,
        "runtime_consumed": 0,
        "runtime_actions_authorized": 0,
        "runtime_actions_issued": 0,
        "runtime_actions_consumed": 0,
        "runtime_actions_abstained": 0,
        "runtime_fingerprint_before": runtime_fingerprint,
        "runtime_fingerprint_after": runtime_fingerprint,
        "native_terminal_error": 1.0,
        "all_evaluation_best_error": 1.0,
        "state_fingerprints": state_fingerprints,
        "artifacts": artifact_names,
        "artifact_sha256": artifact_hashes,
    }
    (artifact_dir / "S5_evidence_overlay_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_fixture(
    artifact_dir: Path,
    aob_root: Path,
    *,
    seed: int = exp030.SEED,
    run_id: str | None = None,
) -> tuple[Path, Path, list[dict[str, str]], list[dict[str, str]]]:
    contexts, arm_rows = _fixture_rows(seed=seed)
    context_path = artifact_dir / "S5_action_ceiling_contexts.csv"
    arm_path = artifact_dir / "S5_action_ceiling_arm_results.csv"
    _write_csv(context_path, ACTION_CEILING_CONTEXT_FIELDS, contexts)
    _write_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS, arm_rows)
    aob_rows = []
    for path in required_aob_data_files(aob_root, 5):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aob_rows.append(
            {
                "problem_id": "S5",
                "file": path.name,
                "path": str(path.resolve()),
                "sha256_before": digest,
                "sha256_after": digest,
                "unchanged": "1",
            }
        )
    _write_csv(
        artifact_dir / "S5_aob_input_manifest.csv",
        exp030.AOB_INPUT_MANIFEST_FIELDS,
        aob_rows,
    )
    (artifact_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "protocol_version": "hcc-run-summary-v2",
                "problem_id": "S5",
                "seed": seed,
                "configured_max_fes": 300_000,
                "fitness_evaluations": 299_999,
                "final_error": 1.0,
                "comparison_fe": 299_998,
                "comparison_error": 1.5,
                "group_optimizer_mode": "full_cmaes",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        artifact_dir / "S5_budget_summary.csv",
        exp030.BUDGET_SUMMARY_FIELDS,
        [
            {
                "problem_id": "S5",
                "budget_accounting": "strict",
                "max_fes": "300000",
                "optimizer_reported_fe": "299998",
                "fitness_record_fe": "299999",
                "budget_aligned_fe": "299999",
                "same_budget_violation": "0",
                "global_phase_fe": "1000",
                "cc_phase_fe": "298982",
                "rescue_fe": "0",
                "refresh_fe": "0",
                "search_state_fe": "0",
                "separable_continuation_fe": "0",
                "overhead_fe": "1",
                "evidence_overlay_fe": "16",
            }
        ],
    )
    resolved_run_id = run_id or f"{exp030.EXPERIMENT_ID}-s5-seed{seed}"
    _write_overlay_fixture(
        artifact_dir,
        contexts,
        seed=seed,
        run_id=resolved_run_id,
    )
    (artifact_dir / "evaluation_record.txt").write_text(
        "fixture evaluation record\n",
        encoding="utf-8",
    )
    _write_csv(
        artifact_dir / "S5_action_trace.csv",
        ("problem_id", "seed"),
        [{"problem_id": "S5", "seed": str(seed)}],
    )
    for name in ("S5_action_decision.csv", "S5_action_mismatch_audit.csv"):
        _write_csv(
            artifact_dir / name,
            ("run_id", "problem_id"),
            [{"run_id": resolved_run_id, "problem_id": "S5"}],
        )
    _write_csv(
        artifact_dir / "S5_overlap_relations.csv",
        ("problem_id",),
        [{"problem_id": "S5"}],
    )
    for alias, canonical in (
        ("aob_input_manifest.csv", "S5_aob_input_manifest.csv"),
        ("action_trace.csv", "S5_action_trace.csv"),
        ("action_decision.csv", "S5_action_decision.csv"),
        ("action_mismatch_audit.csv", "S5_action_mismatch_audit.csv"),
    ):
        (artifact_dir / alias).write_bytes((artifact_dir / canonical).read_bytes())
    return context_path, arm_path, contexts, arm_rows


def test_config_and_worker_command_freeze_single_s5_smoke(tmp_path: Path) -> None:
    config = exp030.load_config()
    assert config["protocol_version"] == exp030.PROTOCOL_VERSION
    assert exp030.PROTOCOL_VERSION == "exp030-s-family-budget-pulse-action-validation-v1"
    assert config["execution"]["terminal_fe_policy"] == "native_population_aligned"
    assert exp030.EXPECTED_ARM_ROWS == 48
    command = exp030.build_worker_command(tmp_path, "python")
    assert command[-2:] == ("--profile", "s_family_budget_pulse")
    assert command[command.index("--max-fes") + 1] == "300000"
    assert command[command.index("--seed") + 1] == "117"


def test_module_entrypoint_loads_without_pytest_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "experiments.pilots.exp_030_s_family_budget_pulse.run",
            "--help",
        ),
        cwd=exp030.REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--reuse-existing" in completed.stdout


def test_validator_accepts_complete_s5_artifact(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)

    validated = exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)

    assert len(validated.context_rows) == 4
    assert len(validated.arm_rows) == 48
    assert validated.run_summary["fitness_evaluations"] == 299_999


def test_validator_accepts_an_explicit_non_smoke_seed(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root, seed=118)

    validated = exp030.validate_artifacts(
        artifact_dir,
        aob_data_root=aob_root,
        expected_seed=118,
        expected_run_id=f"{exp030.EXPERIMENT_ID}-s5-seed118",
    )

    assert {row["seed"] for row in validated.context_rows} == {"118"}
    with pytest.raises(ValueError, match="truth contract"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_old_exp029_s_protocol(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    context_path, _arm_path, contexts, _arm_rows = _write_fixture(artifact_dir, aob_root)
    contexts[0]["protocol_version"] = "exp029-s-family-budget-pulse-validation-v1"
    _write_csv(context_path, ACTION_CEILING_CONTEXT_FIELDS, contexts)

    with pytest.raises(ValueError, match="truth contract"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_duplicate_opaque_dispatch_anchor(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    context_path, _arm_path, contexts, _arm_rows = _write_fixture(artifact_dir, aob_root)
    contexts[0]["dispatch_anchor_hash"] = contexts[1]["dispatch_anchor_hash"]
    _write_csv(context_path, ACTION_CEILING_CONTEXT_FIELDS, contexts)

    with pytest.raises(ValueError, match="duplicate dispatch anchor hash"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_relative_budget_application_fe(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _context_path, arm_path, _contexts, arm_rows = _write_fixture(artifact_dir, aob_root)
    for row in arm_rows:
        if row["arm"] != exp030.SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION:
            continue
        payload = json.loads(row["action_lifecycle_payload"])
        payload["execution"]["application_fe"] -= payload["instance"]["checkpoint_fe"]
        execution = payload["execution"]
        payload["execution_hash"] = _canonical_hash(execution)
        row["action_lifecycle_payload"] = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        row["action_lifecycle_hash"] = _canonical_hash(payload)
    _write_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS, arm_rows)

    with pytest.raises(ValueError, match="application_fe is not absolute"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_nested_lifecycle_hash_drift(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _context_path, arm_path, _contexts, arm_rows = _write_fixture(artifact_dir, aob_root)
    row = next(
        candidate
        for candidate in arm_rows
        if candidate["arm"] == exp030.FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
        and candidate["horizon"] == "sweep_1"
    )
    payload = json.loads(row["action_lifecycle_payload"])
    payload["execution_hash"] = "0" * 64
    row["action_lifecycle_payload"] = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    row["action_lifecycle_hash"] = _canonical_hash(payload)
    _write_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS, arm_rows)

    with pytest.raises(ValueError, match="execution hash changed"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_shrunk_source_not_matching_raw_action(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _context_path, arm_path, _contexts, arm_rows = _write_fixture(artifact_dir, aob_root)
    row = next(
        candidate
        for candidate in arm_rows
        if candidate["arm"] == exp030.SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
    )
    payload = json.loads(row["action_lifecycle_payload"])
    payload["instance"]["raw_group_budgets"][0] += 1
    row["action_lifecycle_payload"] = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    row["action_lifecycle_hash"] = _canonical_hash(payload)
    _write_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS, arm_rows)

    with pytest.raises(ValueError, match="raw budgets must preserve|recorded budget action"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_budget_trace_without_uniform_restore(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _context_path, arm_path, contexts, arm_rows = _write_fixture(artifact_dir, aob_root)
    context = contexts[0]
    row = next(
        candidate
        for candidate in arm_rows
        if candidate["context_id"] == context["context_id"]
        and candidate["arm"] == exp030.SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
        and candidate["horizon"] == "sweep_3"
    )
    sweeps = json.loads(row["execution_sweep_trace"])
    groups = json.loads(row["execution_order_trace"])
    budgets = json.loads(row["group_budget_trace"])
    restored_sweep = int(context["target_sweep"]) + 2
    position = next(index for index, sweep in enumerate(sweeps) if sweep == restored_sweep)
    budgets[position] += 1
    row["group_budget_trace"] = json.dumps(budgets)
    assert groups[position] >= 0
    _write_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS, arm_rows)

    with pytest.raises(ValueError, match="frozen allocation|uniform sweep"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_non_prefix_horizon_trace(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _context_path, arm_path, _contexts, arm_rows = _write_fixture(artifact_dir, aob_root)
    row = next(
        candidate
        for candidate in arm_rows
        if candidate["arm"] == "native_eq8" and candidate["horizon"] == "sweep_1"
    )
    starts = json.loads(row["execution_start_fe_trace"])
    starts[-1] += 1
    row["execution_start_fe_trace"] = json.dumps(starts)
    _write_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS, arm_rows)

    with pytest.raises(ValueError, match="exact prefixes|start FE interval"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


@pytest.mark.parametrize("terminal_fe", (299_997, 300_001))
def test_validator_rejects_terminal_fe_outside_population_aligned_window(
    tmp_path: Path,
    terminal_fe: int,
) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    summary_path = artifact_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["fitness_evaluations"] = terminal_fe
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="population-aligned budget window"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


@pytest.mark.parametrize("terminal_fe", (299_998, 300_000))
def test_validator_accepts_population_aligned_terminal_boundaries(
    tmp_path: Path,
    terminal_fe: int,
) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    summary_path = artifact_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["fitness_evaluations"] = terminal_fe
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    budget_path = artifact_dir / "S5_budget_summary.csv"
    with budget_path.open("r", encoding="utf-8", newline="") as handle:
        budget_rows = list(csv.DictReader(handle))
    budget_rows[0]["fitness_record_fe"] = str(terminal_fe)
    budget_rows[0]["budget_aligned_fe"] = str(terminal_fe)
    budget_rows[0]["overhead_fe"] = str(terminal_fe - 299_998)
    _write_csv(budget_path, exp030.BUDGET_SUMMARY_FIELDS, budget_rows)

    validated = exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)

    assert validated.run_summary["fitness_evaluations"] == terminal_fe


def test_validator_rejects_comparison_fe_not_bound_to_population_ceiling(
    tmp_path: Path,
) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    summary_path = artifact_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["comparison_fe"] = 299_997
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen population ceiling"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_noncanonical_group_optimizer_mode(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    summary_path = artifact_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["group_optimizer_mode"] = "cmaes"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="configured FE budget"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_arm_natural_endpoint_above_budget(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _context_path, arm_path, _contexts, arm_rows = _write_fixture(
        artifact_dir,
        aob_root,
    )
    arm_rows[0]["natural_endpoint_fe"] = "300001"
    _write_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS, arm_rows)

    with pytest.raises(ValueError, match="horizon FE contract"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_strict_budget_accounting_drift(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    budget_path = artifact_dir / "S5_budget_summary.csv"
    with budget_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["same_budget_violation"] = "1"
    _write_csv(budget_path, exp030.BUDGET_SUMMARY_FIELDS, rows)

    with pytest.raises(ValueError, match="strict budget accounting"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_nonzero_separable_continuation_budget(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    budget_path = artifact_dir / "S5_budget_summary.csv"
    with budget_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["cc_phase_fe"] = str(int(rows[0]["cc_phase_fe"]) - 1)
    rows[0]["separable_continuation_fe"] = "1"
    _write_csv(budget_path, exp030.BUDGET_SUMMARY_FIELDS, rows)

    with pytest.raises(ValueError, match="strict budget accounting"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_overlay_child_hash_drift(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    probe_path = artifact_dir / "S5_evidence_overlay_probe_evidence.csv"
    probe_path.write_text(
        probe_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlay artifact hash mismatch"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_self_consistent_cross_relation_x0_fitness_drift(
    tmp_path: Path,
) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    probe_path = artifact_dir / "S5_evidence_overlay_probe_evidence.csv"
    with probe_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target_relation = rows[0]["relation_id"]
    for row in rows:
        if row["relation_id"] == target_relation:
            row["fitness"] = f"{2.0 * float(row['fitness']):.17e}"
    _write_csv(probe_path, exp030.PROBE_EVIDENCE_FIELDS, rows)
    manifest_path = artifact_dir / "S5_evidence_overlay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][probe_path.name] = hashlib.sha256(
        probe_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="repeated x0 fitness"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_runtime_audit_copy_drift(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    alias_path = artifact_dir / "action_trace.csv"
    alias_path.write_text(
        alias_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime audit artifact copies"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_self_consistent_overlay_identity_forgery(
    tmp_path: Path,
) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    plan_path = artifact_dir / "S5_evidence_overlay_plan.csv"
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["seed"] = "999"
    _write_csv(plan_path, exp030.PLAN_FIELDS, rows)
    manifest_path = artifact_dir / "S5_evidence_overlay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][plan_path.name] = hashlib.sha256(
        plan_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overlay relation plan is invalid"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_incomplete_overlay_state_fingerprint_set(
    tmp_path: Path,
) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    manifest_path = artifact_dir / "S5_evidence_overlay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state_fingerprints"].pop("rng")
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="state fingerprints"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_final_error_worse_than_fixed_comparison_error(
    tmp_path: Path,
) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    summary_path = artifact_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_error"] = 2.0
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run summary metrics"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_non_monotone_best_so_far_error_with_valid_delta(
    tmp_path: Path,
) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _context_path, arm_path, _contexts, arm_rows = _write_fixture(artifact_dir, aob_root)
    row = next(
        candidate
        for candidate in arm_rows
        if candidate["arm"] == exp030.SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
        and candidate["horizon"] == "sweep_3"
    )
    native_error = float(row["native_error"])
    row["arm_error"] = f"{100.0:.17e}"
    row["delta"] = f"{actionability_delta(native_error, 100.0):.17e}"
    _write_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS, arm_rows)

    with pytest.raises(ValueError, match="best-so-far monotonicity"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_validator_rejects_aob_file_changed_after_worker(tmp_path: Path) -> None:
    aob_root = _make_aob_root(tmp_path / "aob")
    artifact_dir = tmp_path / "artifacts"
    _write_fixture(artifact_dir, aob_root)
    (aob_root / "F5-p.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="current immutable input"):
        exp030.validate_artifacts(artifact_dir, aob_data_root=aob_root)


def test_fresh_and_reuse_paths_emit_closed_gate_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "results"
    artifact_dir = exp030.trajectory_artifact_dir(output_root)
    _write_fixture(artifact_dir, exp030.AOB_DATA_ROOT)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        exp030,
        "run_worker",
        lambda root, executable: calls.append((root, executable)),
    )

    fresh = exp030.run_experiment(
        output_root=output_root,
        python_executable="fixture-python",
    )
    existing_bytes = {
        path.name: path.read_bytes()
        for path in (
            output_root / "action_ceiling_contexts.csv",
            output_root / "action_ceiling_arm_results.csv",
            output_root / "manifest.json",
        )
    }
    monkeypatch.setattr(
        exp030,
        "_copy_atomic",
        lambda *_args, **_kwargs: pytest.fail("reuse attempted to write aggregate CSV"),
    )
    monkeypatch.setattr(
        exp030,
        "_write_json",
        lambda *_args, **_kwargs: pytest.fail("reuse attempted to write manifest"),
    )
    reused = exp030.run_experiment(output_root=output_root, reuse_existing=True)

    assert calls == [(output_root.resolve(), "fixture-python")]
    assert fresh["status"] == reused["status"] == "mechanical_smoke_pass"
    assert fresh["context_count"] == 4
    assert fresh["arm_row_count"] == 48
    assert fresh["runtime_authorized"] == 0
    assert fresh["selector_authorized"] == 0
    assert fresh["inference_authorized"] == 0
    assert fresh["action_gate_authorized"] == 0
    assert fresh["primary_recommendation"] == "mechanical_smoke_only"
    assert fresh["integrity_checks"]["absolute_application_fe"] == 1
    assert fresh["fe_summary"] == {
        "terminal_fe_policy": "native_population_aligned",
        "configured_max_fes": 300_000,
        "observed_fitness_evaluations": 299_999,
        "comparison_fe": 299_998,
        "terminal_shortfall_fes": 1,
        "population_alignment_window_fes": 2,
    }
    assert fresh["fe_budget_gate_passed"] == 1
    assert "src/arac/actions/shrunk_budget_pulse.py" in fresh["source_sha256"]
    assert (output_root / "action_ceiling_contexts.csv").is_file()
    assert (output_root / "action_ceiling_arm_results.csv").is_file()
    assert (output_root / "manifest.json").is_file()
    assert existing_bytes == {
        path.name: path.read_bytes()
        for path in (
            output_root / "action_ceiling_contexts.csv",
            output_root / "action_ceiling_arm_results.csv",
            output_root / "manifest.json",
        )
    }


def test_reuse_rejects_authorization_or_aggregate_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "results"
    _write_fixture(exp030.trajectory_artifact_dir(output_root), exp030.AOB_DATA_ROOT)
    monkeypatch.setattr(exp030, "run_worker", lambda *_args: None)
    exp030.run_experiment(output_root=output_root)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_authorized"] = 1
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest contract"):
        exp030.run_experiment(output_root=output_root, reuse_existing=True)

    manifest["runtime_authorized"] = 0
    manifest["selector_gate"] = {"authorized": 1}
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest contract"):
        exp030.run_experiment(output_root=output_root, reuse_existing=True)

    del manifest["selector_gate"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    aggregate_path = output_root / "action_ceiling_arm_results.csv"
    aggregate_path.write_bytes(aggregate_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="aggregate CSVs differ"):
        exp030.run_experiment(output_root=output_root, reuse_existing=True)
