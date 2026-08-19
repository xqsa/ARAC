"""Isolate GCB relation-dispatch timing on frozen development checkpoints."""

# Thread caps must be set before NumPy, PyPop7, or ARAC imports.
# ruff: noqa: E402

from __future__ import annotations

import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path

from threadpoolctl import threadpool_limits

import experiments.final.run as final_run
import experiments.historical_recovery.gcb_mechanism_diagnostic as base
from arac.actions._execution import (
    _PersistentBlockSession,
    _aligned_visit_budget,
    _block_population_size,
    _run_block_visit,
    derived_seed,
    run_cold_start_block_sweeps,
    run_full_space,
    terminal_result,
)
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, ActionResult, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger


DEFAULT_PROTOCOL = Path(__file__).with_name("gcb_trigger_diagnostic_protocol.json")
CASES = base.CASES
SEEDS = base.SEEDS
VARIANTS = (
    "native_order_relation_dispatch",
)
BASELINE_VARIANTS = {
    "native_order_relation_dispatch": "native_order_historical_seeds",
}
TOTAL_BUDGET_FES = base.TOTAL_BUDGET_FES
PHASE1_FES = base.PHASE1_FES
SOURCE_SWEEP_COUNT = 3
NATIVE_WINDOW_COUNT = 3
RECEIPT_SCHEMA = "arac-gcb-trigger-diagnostic-receipt-v1"
SUMMARY_SCHEMA = "arac-gcb-trigger-diagnostic-summary-v1"
MANIFEST_SCHEMA = "arac-gcb-trigger-diagnostic-manifest-v1"
SOURCE_PATHS = (
    "experiments/historical_recovery/gcb_trigger_diagnostic.py",
    "experiments/historical_recovery/gcb_trigger_diagnostic_protocol.json",
    *base.SOURCE_PATHS,
)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    protocol = base._load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-gcb-trigger-diagnostic-protocol-v1",
        "status": "frozen_paired_development_diagnostic",
        "cases": list(CASES),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "matched_baseline_variants": BASELINE_VARIANTS,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "expected_phase1_fes": PHASE1_FES,
        "native_threads": 1,
        "max_workers": 9,
        "checkpoint_source_root": "artifacts/gcb_schedule_ablation_v2",
        "baseline_source_root": "artifacts/gcb_mechanism_diagnostic_v1",
        "output_root": "artifacts/gcb_trigger_diagnostic_v1",
        "production_hcc_runtime_imports_allowed": False,
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    if set(protocol) != set(expected) | {"purpose", "acceptance_gates"}:
        raise ValueError("GCB trigger-diagnostic protocol keys drifted")
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"GCB trigger-diagnostic protocol drifted: {key}")
    gates = {
        "same_checkpoint_as_matched_baseline": True,
        "exact_terminal_fes": True,
        "all_final_errors_finite": True,
        "candidate_schedule_trace_valid": True,
        "zero_relation_bitwise_equivalent": True,
        "positive_relation_per_case_geometric_mean_ratio_lt": 1.0,
        "positive_relation_vs_frozen_current_per_case_geometric_mean_ratio_lt": 1.0,
        "positive_relation_win_count_gte": 4,
        "maximum_pair_ratio_lte": 10.0,
        "all_runtime_warnings_known": True,
    }
    if protocol.get("acceptance_gates") != gates:
        raise ValueError("GCB trigger-diagnostic acceptance gates drifted")
    return protocol


def _source_hashes() -> dict[str, str]:
    return {relative: base._sha256(base._resolved(relative)) for relative in SOURCE_PATHS}


def preflight(path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = base._resolved(str(protocol["output_root"]))
    if output_root.exists() and not resume:
        raise ValueError(f"fresh GCB trigger output already exists: {output_root}")
    missing = [relative for relative in SOURCE_PATHS if not base._resolved(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"GCB trigger inputs are missing: {missing}")
    imports = base._production_hcc_imports()
    if imports:
        raise ValueError(f"production HCC imports remain: {imports}")
    baseline_summary = base.verify()
    return {
        "schema_version": "arac-gcb-trigger-diagnostic-preflight-v1",
        "protocol_sha256": base._sha256(protocol_path),
        "output_root": str(output_root),
        "source_sha256": _source_hashes(),
        "baseline_summary_sha256": baseline_summary["summary_sha256"],
        "vendor_trees": final_run._vendor_tree_hashes(),
        "arm_count": len(CASES) * len(SEEDS) * len(VARIANTS),
        "production_hcc_runtime_imports": imports,
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
        "passed": True,
    }


def _manifest(protocol_path: Path, gate: Mapping[str, object]) -> dict[str, object]:
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": base._sha256(protocol_path),
        "source_sha256": gate["source_sha256"],
        "baseline_summary_sha256": gate["baseline_summary_sha256"],
        "vendor_trees": gate["vendor_trees"],
        "production_hcc_runtime_imports": [],
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    return {**body, "manifest_sha256": base._canonical(body)}


def _load_manifest(output_root: Path, protocol_path: Path) -> dict[str, object]:
    manifest = base._load_json(output_root / "manifest.json")
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != base._canonical(manifest):
        raise ValueError("GCB trigger manifest hash drifted")
    if manifest.get("protocol_sha256") != base._sha256(protocol_path):
        raise ValueError("GCB trigger protocol hash drifted")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("GCB trigger source hashes drifted")
    if manifest.get("baseline_summary_sha256") != base.verify()["summary_sha256"]:
        raise ValueError("GCB trigger baseline summary drifted")
    if manifest.get("vendor_trees") != final_run._vendor_tree_hashes():
        raise ValueError("GCB trigger AOB vendor tree drifted")
    manifest["manifest_sha256"] = claimed
    return manifest


@dataclass(frozen=True)
class ArmContext:
    case_id: str
    run_seed: int
    variant: str
    checkpoint_source_root: Path
    baseline_source_root: Path
    output_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}:{self.variant}"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.case_id / f"seed_{self.run_seed}" / f"{self.variant}.json"


def _matched_baseline(context: ArmContext) -> dict[str, object]:
    name = BASELINE_VARIANTS[context.variant]
    path = context.baseline_source_root / "arms" / context.case_id / f"seed_{context.run_seed}" / f"{name}.json"
    receipt = base._load_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != base._canonical(receipt):
        raise ValueError(f"matched baseline receipt drifted: {context.key}")
    if (
        receipt.get("case_id") != context.case_id
        or receipt.get("run_seed") != context.run_seed
        or receipt.get("variant") != name
    ):
        raise ValueError(f"matched baseline binding drifted: {context.key}")
    receipt["receipt_sha256"] = claimed
    return receipt


def _top_relation(checkpoint: PhaseCheckpoint) -> RelationEvidence | None:
    if not checkpoint.relations:
        return None
    return min(
        checkpoint.relations,
        key=lambda item: (
            -item.strength * (1.0 + item.disagreement),
            item.left_block,
            item.right_block,
        ),
    )


def _variant_settings(
    variant: str,
    context: ActionContext,
) -> tuple[tuple[int, ...] | None, object]:
    if variant == "native_order_relation_dispatch":
        block_order = None
    else:
        raise ValueError(f"unsupported GCB trigger variant: {variant}")
    namespace = f"gcb-native-{context.checkpoint.checkpoint_hash}"

    def seed_factory(stage_index: int) -> int:
        return derived_seed(context, namespace, stage_index)

    return block_order, seed_factory


def _cold_visit(
    context: ActionContext,
    *,
    index: int,
    requested_fes: int,
    namespace: str,
    sweep_index: int,
    seed_factory: object,
    trace: list[dict[str, object]],
) -> int:
    population = _block_population_size(len(context.checkpoint.blocks[index]))
    visit_budget = _aligned_visit_budget(requested_fes, context.ledger.remaining, population)
    if visit_budget == 0:
        return 0
    stage_index = sweep_index * len(context.checkpoint.blocks) + index + 1
    start_fes = context.ledger.count
    session = _PersistentBlockSession(
        context,
        context.checkpoint.blocks[index],
        index,
        visit_budget,
        population_size=population,
        seed_namespace=f"{namespace}-sweep-{sweep_index}",
        seed_factory=seed_factory,  # type: ignore[arg-type]
        stage_index=stage_index,
    )
    consumed = _run_block_visit(session, visit_budget)
    trace.append(
        {
            "event": "cold_group_visit",
            "namespace": namespace,
            "sweep_index": sweep_index,
            "group_index": index,
            "stage_index": stage_index,
            "seed": seed_factory(stage_index),  # type: ignore[operator]
            "start_fes": start_fes,
            "requested_fes": visit_budget,
            "actual_fes": consumed,
            "end_fes": context.ledger.count,
            "cold_start": True,
            "state_restored": False,
        }
    )
    return consumed


def _coordination(
    context: ActionContext,
    *,
    budget_fes: int,
    trigger: str,
    trace: list[dict[str, object]],
) -> None:
    if context.ledger.remaining < budget_fes:
        raise RuntimeError("GCB relation dispatch lacks its frozen burst budget")
    start_fes = context.ledger.count
    run_full_space(
        context,
        algorithm="sepcmaes",
        budget_fes=budget_fes,
        namespace=f"gcb-global-coordination-{context.checkpoint.checkpoint_hash}",
    )
    trace.append(
        {
            "event": "full_space_coordination",
            "trigger": trigger,
            "start_fes": start_fes,
            "requested_fes": budget_fes,
            "actual_fes": context.ledger.count - start_fes,
            "end_fes": context.ledger.count,
        }
    )


def _trace_summary(
    trace: Sequence[Mapping[str, object]],
    checkpoint: PhaseCheckpoint,
) -> dict[str, object]:
    groups = [event for event in trace if event.get("event") == "cold_group_visit"]
    source = [event for event in groups if event.get("namespace") == "gcb-warmup"]
    native = [event for event in groups if event.get("namespace") == "gcb-continuation"]
    coordination = [event for event in trace if event.get("event") == "full_space_coordination"]
    block_count = len(checkpoint.blocks)
    source_counts = {
        str(index): sum(int(event["sweep_index"]) == index for event in source)
        for index in range(SOURCE_SWEEP_COUNT)
    }
    native_counts = {
        str(index): sum(int(event["sweep_index"]) == index for event in native)
        for index in range(SOURCE_SWEEP_COUNT, SOURCE_SWEEP_COUNT + NATIVE_WINDOW_COUNT)
    }
    source_actual = sum(
        int(event["actual_fes"])
        for event in source
        if int(event["sweep_index"]) == SOURCE_SWEEP_COUNT - 1
    )
    relation = _top_relation(checkpoint)
    expected_trigger = "phase_boundary" if relation is None else "relation_dispatch"
    coordination_index = next(
        (index for index, event in enumerate(trace) if event.get("event") == "full_space_coordination"),
        None,
    )
    before_dispatch = (
        [event for event in trace[:coordination_index] if event.get("namespace") == "gcb-continuation" and int(event["sweep_index"]) == 3]
        if coordination_index is not None
        else []
    )
    after_dispatch = (
        [event for event in trace[coordination_index + 1 :] if event.get("namespace") == "gcb-continuation" and int(event["sweep_index"]) == 3]
        if coordination_index is not None
        else []
    )
    owner_indices = () if relation is None else (relation.left_block, relation.right_block)
    observed_before = {int(event["group_index"]) for event in before_dispatch}
    dispatch_valid = (
        not before_dispatch
        if relation is None
        else set(owner_indices).issubset(observed_before)
        and not set(owner_indices).intersection(int(event["group_index"]) for event in after_dispatch)
    )
    complete = all(
        sorted(
            int(event["group_index"])
            for event in groups
            if int(event["sweep_index"]) == sweep_index
        )
        == list(range(block_count))
        for sweep_index in range(SOURCE_SWEEP_COUNT + NATIVE_WINDOW_COUNT)
    )
    ordered = all(
        int(right["start_fes"]) >= int(left["end_fes"])
        for left, right in zip(trace, trace[1:], strict=False)
    )
    trigger_matches = (
        coordination[0].get("trigger") in (None, "phase_boundary")
        if relation is None and len(coordination) == 1
        else len(coordination) == 1 and coordination[0].get("trigger") == expected_trigger
    )
    valid = (
        len(coordination) == 1
        and trigger_matches
        and int(coordination[0]["actual_fes"]) == source_actual
        and all(value == block_count for value in source_counts.values())
        and all(value == block_count for value in native_counts.values())
        and complete
        and dispatch_valid
        and all(event.get("cold_start") is True for event in groups)
        and all(event.get("state_restored") is False for event in groups)
        and len({int(event["stage_index"]) for event in groups}) == len(groups)
        and ordered
    )
    return {
        "valid": valid,
        "trigger": expected_trigger,
        "selected_relation": list(owner_indices) if owner_indices else None,
        "source_actual_fes": source_actual,
        "coordination_actual_fes": int(coordination[0]["actual_fes"]) if len(coordination) == 1 else None,
        "block_count": block_count,
        "source_sweep_group_counts": source_counts,
        "native_sweep_group_counts": native_counts,
        "mixed_sweep_pre_dispatch_groups": [int(event["group_index"]) for event in before_dispatch],
        "mixed_sweep_post_dispatch_groups": [int(event["group_index"]) for event in after_dispatch],
        "state_restore_count": sum(event.get("state_restored") is True for event in groups),
        "events_ordered": ordered,
    }


def _execute_variant(
    variant: str,
    context: ActionContext,
) -> tuple[ActionResult, list[dict[str, object]], dict[str, object]]:
    block_order, seed_factory = _variant_settings(variant, context)
    order = tuple(range(len(context.checkpoint.blocks))) if block_order is None else block_order
    relation = _top_relation(context.checkpoint)
    if relation is None:
        result, trace, _ = base._execute_variant(
            "native_order_historical_seeds",
            context,
        )
        return result, trace, _trace_summary(trace, context.checkpoint)

    trace: list[dict[str, object]] = []
    source_fes, source_sweeps = run_cold_start_block_sweeps(
        context,
        requested_fes=context.ledger.remaining,
        sweep_limit=SOURCE_SWEEP_COUNT,
        block_order=block_order,
        namespace="gcb-warmup",
        seed_factory=seed_factory,  # type: ignore[arg-type]
        start_sweep_index=0,
        event_trace=trace,
    )
    if len(source_sweeps) != SOURCE_SWEEP_COUNT:
        raise RuntimeError("GCB trigger source schedule did not complete three sweeps")
    coordination_budget = min(context.ledger.remaining, source_sweeps[-1])
    owners = {relation.left_block, relation.right_block}
    observed: set[int] = set()
    block_window_budget = context.ledger.remaining - coordination_budget
    if block_window_budget <= 0:
        raise RuntimeError("GCB mixed sweep lacks a post-burst block budget")
    requested_per_block = math.ceil(block_window_budget / len(order))
    dispatched = False
    for index in order:
        consumed = _cold_visit(
            context,
            index=index,
            requested_fes=max(
                requested_per_block,
                _block_population_size(len(context.checkpoint.blocks[index])),
            ),
            namespace="gcb-continuation",
            sweep_index=SOURCE_SWEEP_COUNT,
            seed_factory=seed_factory,
            trace=trace,
        )
        if consumed == 0:
            raise RuntimeError("GCB mixed native sweep did not cover every block")
        observed.add(index)
        if not dispatched and owners.issubset(observed):
            _coordination(
                context,
                budget_fes=coordination_budget,
                trigger="relation_dispatch",
                trace=trace,
            )
            dispatched = True
    if not dispatched:
        raise RuntimeError("GCB relation was not dispatched in the mixed sweep")
    native_start = SOURCE_SWEEP_COUNT + 1
    native_fes, native_sweeps = run_cold_start_block_sweeps(
        context,
        requested_fes=context.ledger.remaining,
        block_order=block_order,
        namespace="gcb-continuation",
        seed_factory=seed_factory,  # type: ignore[arg-type]
        start_sweep_index=native_start,
        event_trace=trace,
    )
    required_complete = NATIVE_WINDOW_COUNT - 1
    if len(native_sweeps) < required_complete:
        raise RuntimeError("GCB trigger native schedule did not complete three windows")
    tail_fes = context.ledger.remaining
    if tail_fes:
        run_full_space(
            context,
            algorithm="sepcmaes",
            namespace=f"gcb-terminal-alignment-{context.checkpoint.checkpoint_hash}",
        )
    route = (
        f"{variant}_source_{source_fes}_sweeps_{len(source_sweeps)}_"
        f"coordination_{coordination_budget}_native_{native_fes}_"
        f"windows_{len(native_sweeps) + 1}_tail_{tail_fes}"
    )
    return terminal_result(context, route=route), trace, _trace_summary(trace, context.checkpoint)


def _checkpoint(context: ArmContext) -> PhaseCheckpoint:
    checkpoint_context = final_run.CheckpointContext(
        context.case_id,
        context.run_seed,
        TOTAL_BUDGET_FES,
        context.checkpoint_source_root,
    )
    receipt = final_run._validate_checkpoint(checkpoint_context.receipt_path, checkpoint_context)
    return final_run._checkpoint_from_payload(receipt["checkpoint"])


def _run_arm(context: ArmContext) -> dict[str, object]:
    checkpoint = _checkpoint(context)
    matched = _matched_baseline(context)
    if matched["checkpoint_hash"] != checkpoint.checkpoint_hash:
        raise ValueError(f"{context.key} matched checkpoint drifted")
    problem = AobBenchmark().load(
        context.case_id,
        output_directory=context.receipt_path.parent / f"benchmark_{context.variant}",
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    action_context = ActionContext("gcb", checkpoint, problem, ledger, action_seed=context.run_seed)
    with threadpool_limits(limits=1):
        pools = base._threadpools()
        if not pools or any(item["num_threads"] != 1 for item in pools):
            raise RuntimeError(f"{context.key} native thread limit is not one")
        (result, trace, trace_summary), runtime_warnings = final_run._call_with_warning_capture(
            _execute_variant,
            context.variant,
            action_context,
        )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "variant": context.variant,
        "matched_baseline_variant": BASELINE_VARIANTS[context.variant],
        "matched_baseline_receipt_sha256": matched["receipt_sha256"],
        "matched_baseline_final_error": matched["final_error"],
        "frozen_current_final_error": matched["baseline_final_error"],
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "action_checkpoint_hash": result.checkpoint_hash,
        "phase1_fes": checkpoint.phase1_fes,
        "phase1_relation_count": checkpoint.overlap_relation_count,
        "phase2_consumed_fes": result.consumed_fes,
        "terminal_fes": result.terminal_fes,
        "final_error": result.final_error,
        "terminal_state_finite": math.isfinite(result.final_error),
        "route": result.route,
        "schedule_trace": trace,
        "schedule_trace_summary": trace_summary,
        "runtime_warnings": runtime_warnings,
        "native_thread_limit_verified": True,
        "threadpools": pools,
        "production_hcc_runtime_imports": base._production_hcc_imports(),
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    receipt = {**body, "receipt_sha256": base._canonical(body)}
    base._write_json(context.receipt_path, receipt)
    return receipt


def _validate_arm(path: Path, context: ArmContext) -> dict[str, object]:
    receipt = base._load_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != base._canonical(receipt):
        raise ValueError(f"{context.key} receipt hash drifted")
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "variant": context.variant,
        "matched_baseline_variant": BASELINE_VARIANTS[context.variant],
        "phase1_fes": PHASE1_FES,
        "phase2_consumed_fes": TOTAL_BUDGET_FES - PHASE1_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "terminal_state_finite": True,
        "native_thread_limit_verified": True,
        "production_hcc_runtime_imports": [],
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{context.key} receipt field drifted: {key}")
    matched = _matched_baseline(context)
    if receipt.get("matched_baseline_receipt_sha256") != matched["receipt_sha256"]:
        raise ValueError(f"{context.key} matched baseline hash drifted")
    if receipt.get("checkpoint_hash") != receipt.get("action_checkpoint_hash"):
        raise ValueError(f"{context.key} action checkpoint drifted")
    if receipt.get("checkpoint_hash") != matched["checkpoint_hash"]:
        raise ValueError(f"{context.key} matched checkpoint drifted")
    trace = receipt.get("schedule_trace")
    trace_summary = receipt.get("schedule_trace_summary")
    if not isinstance(trace, list) or not isinstance(trace_summary, Mapping):
        raise ValueError(f"{context.key} trigger trace is missing")
    checkpoint = _checkpoint(context)
    if trace_summary != _trace_summary(trace, checkpoint) or trace_summary.get("valid") is not True:
        raise ValueError(f"{context.key} trigger trace is invalid")
    if context.case_id == "R1" and (
        float(receipt["final_error"]) != float(matched["final_error"])
        or trace != matched["schedule_trace"]
    ):
        raise ValueError(f"{context.key} zero-relation parent path drifted")
    final_run._validate_runtime_warnings(receipt.get("runtime_warnings"), context.key)
    receipt["receipt_sha256"] = claimed
    return receipt


def _summarize(
    rows: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    gates = protocol["acceptance_gates"]
    summaries = []
    for variant in VARIANTS:
        active = [row for row in rows if row["variant"] == variant]
        pairs = []
        for row in active:
            candidate = float(row["final_error"])
            baseline_error = float(row["matched_baseline_final_error"])
            pairs.append(
                {
                    "case_id": row["case_id"],
                    "run_seed": row["run_seed"],
                    "checkpoint_hash": row["checkpoint_hash"],
                    "baseline_final_error": baseline_error,
                    "frozen_current_final_error": row["frozen_current_final_error"],
                    "candidate_final_error": candidate,
                    "candidate_to_baseline_ratio": base._ratio(candidate, baseline_error),
                    "candidate_to_frozen_current_ratio": base._ratio(
                        candidate,
                        float(row["frozen_current_final_error"]),
                    ),
                    "candidate_won_or_tied": candidate <= baseline_error,
                }
            )
        zero_pairs = [pair for pair in pairs if pair["case_id"] == "R1"]
        zero_equivalent = all(
            float(pair["candidate_final_error"]) == float(pair["baseline_final_error"])
            for pair in zero_pairs
        )
        case_summaries = []
        for case_id in ("R3", "R6"):
            case_pairs = [pair for pair in pairs if pair["case_id"] == case_id]
            ratios = [float(pair["candidate_to_baseline_ratio"]) for pair in case_pairs]
            geometric = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            frozen_ratios = [
                float(pair["candidate_to_frozen_current_ratio"]) for pair in case_pairs
            ]
            frozen_geometric = math.exp(
                sum(math.log(value) for value in frozen_ratios) / len(frozen_ratios)
            )
            case_summaries.append(
                {
                    "case_id": case_id,
                    "parent_geometric_mean_ratio": geometric,
                    "frozen_current_geometric_mean_ratio": frozen_geometric,
                    "win_or_tie_count": sum(bool(pair["candidate_won_or_tied"]) for pair in case_pairs),
                    "passed": (
                        geometric
                        < float(gates["positive_relation_per_case_geometric_mean_ratio_lt"])  # type: ignore[index]
                        and frozen_geometric
                        < float(  # type: ignore[index]
                            gates[
                                "positive_relation_vs_frozen_current_per_case_geometric_mean_ratio_lt"
                            ]
                        )
                    ),
                }
            )
        positive_pairs = [pair for pair in pairs if pair["case_id"] in ("R3", "R6")]
        win_count = sum(bool(pair["candidate_won_or_tied"]) for pair in positive_pairs)
        maximum_ratio = max(float(pair["candidate_to_baseline_ratio"]) for pair in positive_pairs)
        passed = (
            zero_equivalent
            and all(bool(item["passed"]) for item in case_summaries)
            and win_count >= int(gates["positive_relation_win_count_gte"])  # type: ignore[index]
            and maximum_ratio <= float(gates["maximum_pair_ratio_lte"])  # type: ignore[index]
        )
        summaries.append(
            {
                "variant": variant,
                "zero_relation_bitwise_equivalent": zero_equivalent,
                "positive_relation_win_or_tie_count": win_count,
                "maximum_positive_relation_ratio": maximum_ratio,
                "positive_relation_case_summaries": case_summaries,
                "pairs": pairs,
                "performance_gate_passed": passed,
            }
        )
    warning_summary = final_run._runtime_warning_summary(rows)
    integrity = (
        all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows)
        and all(math.isfinite(float(row["final_error"])) for row in rows)
        and all(row["schedule_trace_summary"]["valid"] is True for row in rows)  # type: ignore[index]
        and warning_summary["all_runtime_warnings_known"] is True
    )
    passing = [item["variant"] for item in summaries if item["performance_gate_passed"]]
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "arm_count": len(rows),
        "integrity_gate_passed": integrity,
        "variant_summaries": summaries,
        "passing_variants": passing,
        "diagnostic_gate_passed": integrity and bool(passing),
        **warning_summary,
        "production_gcb_integration_authorized": False,
        "final_25_seed_recovery_evaluated": False,
        "selector_or_routing_authorized": False,
    }
    return {**body, "summary_sha256": base._canonical(body)}


def _contexts(protocol: Mapping[str, object], manifest_sha256: str) -> tuple[ArmContext, ...]:
    checkpoint_root = base._resolved(str(protocol["checkpoint_source_root"]))
    baseline_root = base._resolved(str(protocol["baseline_source_root"]))
    output_root = base._resolved(str(protocol["output_root"]))
    return tuple(
        ArmContext(case_id, seed, variant, checkpoint_root, baseline_root, output_root, manifest_sha256)
        for seed in SEEDS
        for case_id in CASES
        for variant in VARIANTS
    )


def run(
    path: Path = DEFAULT_PROTOCOL,
    *,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    gate = preflight(protocol_path, resume=resume)
    output_root = base._resolved(str(protocol["output_root"]))
    max_workers = int(protocol["max_workers"] if workers is None else workers)
    if not 1 <= max_workers <= len(CASES) * len(SEEDS) * len(VARIANTS):
        raise ValueError("workers is outside the GCB trigger arm count")
    if resume:
        manifest = _load_manifest(output_root, protocol_path)
    else:
        output_root.mkdir(parents=True)
        base._write_json(output_root / "protocol.json", protocol)
        base._write_json(output_root / "preflight.json", gate)
        manifest = _manifest(protocol_path, gate)
        base._write_json(output_root / "manifest.json", manifest)
    contexts = _contexts(protocol, str(manifest["manifest_sha256"]))
    rows = final_run._run_parallel(
        contexts,
        _run_arm,
        max_workers=max_workers,
        progress_path=output_root / "parallel_progress.json",
        receipt_path=lambda context: context.receipt_path,
        validator=_validate_arm,
        resume=resume,
    )
    summary = _summarize(rows, protocol)
    base._write_json(output_root / "summary.json", summary)
    final_run._require_known_runtime_warnings(summary, stage="GCB trigger diagnostic")
    return summary


def verify(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = base._resolved(str(protocol["output_root"]))
    manifest = _load_manifest(output_root, protocol_path)
    contexts = _contexts(protocol, str(manifest["manifest_sha256"]))
    rows = [_validate_arm(context.receipt_path, context) for context in contexts]
    expected = _summarize(rows, protocol)
    stored = base._load_json(output_root / "summary.json")
    claimed = stored.pop("summary_sha256", None)
    if claimed != base._canonical(stored):
        raise ValueError("GCB trigger summary hash drifted")
    expected.pop("summary_sha256", None)
    expected.pop("generated_at_utc", None)
    stored.pop("generated_at_utc", None)
    if stored != expected:
        raise ValueError("GCB trigger summary content drifted")
    stored["summary_sha256"] = claimed
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.protocol, resume=args.resume)
    elif args.command == "run":
        result = run(args.protocol, resume=args.resume, workers=args.workers)
    else:
        result = verify(args.protocol)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
