"""Run the fresh paired GCB schedule-recovery development gate."""

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
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

from threadpoolctl import threadpool_info, threadpool_limits

import experiments.final.run as final_run
from arac.actions.gcb import GcbExecutor
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, ActionResult, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("gcb_schedule_ablation_protocol.json")
CASES = ("R1", "R3", "R6")
SEEDS = (31_001, 31_002, 31_003)
VARIANTS = ("gcb_frozen_current", "gcb_three_source_burst_native")
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
SOURCE_WINDOW_FRACTION = 0.08
SOURCE_SWEEP_COUNT = 3
NATIVE_WINDOW_COUNT = 3
RECEIPT_SCHEMA = "arac-gcb-schedule-ablation-receipt-v2"
SUMMARY_SCHEMA = "arac-gcb-schedule-ablation-summary-v2"
MANIFEST_SCHEMA = "arac-gcb-schedule-ablation-manifest-v2"
SOURCE_PATHS = (
    "experiments/historical_recovery/gcb_schedule_ablation.py",
    "experiments/historical_recovery/gcb_schedule_ablation_protocol.json",
    "experiments/final/run.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/evidence/phase1.py",
    "src/arac/evidence/structural.py",
    "src/arac/evidence/mechanism_features.py",
    "src/arac/runtime/contracts.py",
    "src/arac/runtime/ledger.py",
    "src/arac/runtime/optimizers.py",
    "src/arac/actions/_execution.py",
    "src/arac/actions/gcb.py",
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    final_run._atomic_json(path, payload)


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def _production_hcc_imports() -> list[str]:
    matches = []
    for path in (REPOSITORY_ROOT / "src" / "arac").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        if any(token in source for token in ("from hcc", "import hcc", "vendor.hcc", "vendor/hcc")):
            matches.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    return matches


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = _load_json(protocol_path)
    expected_keys = {
        "schema_version",
        "status",
        "purpose",
        "cases",
        "seeds",
        "variants",
        "total_budget_fes",
        "expected_phase1_fes",
        "source_window_fraction",
        "source_sweep_count",
        "native_window_count",
        "native_threads",
        "max_workers",
        "output_root",
        "legacy_source",
        "legacy_source_sha256",
        "production_hcc_runtime_imports_allowed",
        "selector_execution_allowed",
        "reference_thresholds_used_for_decision",
        "acceptance_gates",
    }
    if set(protocol) != expected_keys:
        raise ValueError("GCB schedule-ablation protocol keys drifted")
    expected = {
        "schema_version": "arac-gcb-schedule-ablation-protocol-v2",
        "status": "frozen_paired_development_gate",
        "cases": list(CASES),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "total_budget_fes": TOTAL_BUDGET_FES,
        "expected_phase1_fes": PHASE1_FES,
        "source_window_fraction": SOURCE_WINDOW_FRACTION,
        "source_sweep_count": SOURCE_SWEEP_COUNT,
        "native_window_count": NATIVE_WINDOW_COUNT,
        "native_threads": 1,
        "max_workers": 18,
        "production_hcc_runtime_imports_allowed": False,
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"GCB schedule-ablation protocol drifted: {key}")
    expected_gates = {
        "same_checkpoint_per_pair": True,
        "exact_terminal_fes": True,
        "all_final_errors_finite": True,
        "candidate_schedule_trace_valid": True,
        "per_case_geometric_mean_ratio_lt": 1.0,
        "overall_candidate_win_count_gte": 6,
        "maximum_pair_ratio_lte": 10.0,
        "all_runtime_warnings_known": True,
    }
    if protocol.get("acceptance_gates") != expected_gates:
        raise ValueError("GCB schedule-ablation acceptance gates drifted")
    legacy_path = _resolved(str(protocol["legacy_source"]))
    if _sha256(legacy_path) != protocol["legacy_source_sha256"]:
        raise ValueError("frozen current GCB source drifted")
    return protocol


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(_resolved(relative)) for relative in SOURCE_PATHS}


def _preflight(path: Path, *, resume: bool) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    if output_root.exists() and not resume:
        raise ValueError(f"fresh GCB ablation output already exists: {output_root}")
    missing = [relative for relative in SOURCE_PATHS if not _resolved(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"GCB schedule-ablation inputs are missing: {missing}")
    imports = _production_hcc_imports()
    if imports:
        raise ValueError(f"production HCC imports remain: {imports}")
    return {
        "schema_version": "arac-gcb-schedule-ablation-preflight-v2",
        "protocol_sha256": _sha256(protocol_path),
        "output_root": str(output_root),
        "source_sha256": _source_hashes(),
        "legacy_source_sha256": protocol["legacy_source_sha256"],
        "vendor_trees": final_run._vendor_tree_hashes(),
        "checkpoint_count": len(CASES) * len(SEEDS),
        "arm_count": len(CASES) * len(SEEDS) * len(VARIANTS),
        "production_hcc_runtime_imports": imports,
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
        "passed": True,
    }


def preflight(path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, object]:
    return _preflight(Path(path).resolve(), resume=resume)


def _manifest(protocol_path: Path, gate: Mapping[str, object]) -> dict[str, object]:
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha256(protocol_path),
        "source_sha256": gate["source_sha256"],
        "legacy_source_sha256": gate["legacy_source_sha256"],
        "vendor_trees": gate["vendor_trees"],
        "production_hcc_runtime_imports": [],
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    return {**body, "manifest_sha256": _canonical(body)}


def _load_manifest(output_root: Path, protocol_path: Path) -> dict[str, object]:
    manifest = _load_json(output_root / "manifest.json")
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _canonical(manifest):
        raise ValueError("GCB schedule-ablation manifest hash drifted")
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("GCB schedule-ablation protocol hash drifted")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("GCB schedule-ablation source hashes drifted")
    if manifest.get("vendor_trees") != final_run._vendor_tree_hashes():
        raise ValueError("GCB schedule-ablation AOB vendor tree drifted")
    manifest["manifest_sha256"] = claimed
    return manifest


@dataclass(frozen=True)
class ArmContext:
    case_id: str
    run_seed: int
    variant: str
    output_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}:{self.variant}"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.case_id / f"seed_{self.run_seed}" / f"{self.variant}.json"


_LEGACY_MODULE: ModuleType | None = None


def _legacy_module() -> ModuleType:
    global _LEGACY_MODULE
    if _LEGACY_MODULE is None:
        path = _resolved(str(load_protocol()["legacy_source"]))
        spec = importlib.util.spec_from_file_location("arac_frozen_current_gcb", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the frozen current GCB source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LEGACY_MODULE = module
    return _LEGACY_MODULE


def _trace_summary(
    trace: Sequence[Mapping[str, object]],
    checkpoint: PhaseCheckpoint,
) -> dict[str, object]:
    groups = [event for event in trace if event.get("event") == "cold_group_visit"]
    source_groups = [event for event in groups if event.get("namespace") == "gcb-source"]
    native_groups = [event for event in groups if event.get("namespace") == "gcb-native"]
    coordination = [
        event for event in trace if event.get("event") == "full_space_coordination"
    ]
    block_count = len(checkpoint.blocks)
    expected_trigger = "phase_boundary" if checkpoint.overlap_relation_count == 0 else "relation_dispatch"
    source_sweep_counts = {
        str(index): sum(int(event["sweep_index"]) == index for event in source_groups)
        for index in range(SOURCE_SWEEP_COUNT)
    }
    native_sweep_counts = {
        str(index): sum(int(event["sweep_index"]) == index for event in native_groups)
        for index in range(SOURCE_SWEEP_COUNT, SOURCE_SWEEP_COUNT + NATIVE_WINDOW_COUNT)
    }
    source_actual = sum(
        int(event["actual_fes"])
        for event in source_groups
        if int(event["sweep_index"]) == SOURCE_SWEEP_COUNT - 1
    )
    ordered = all(
        int(right["start_fes"]) >= int(left["end_fes"])
        for left, right in zip(trace, trace[1:], strict=False)
    )
    complete_groups = all(
        sorted(
            int(event["group_index"])
            for event in source_groups + native_groups
            if int(event["sweep_index"]) == index
        )
        == list(range(block_count))
        for index in range(SOURCE_SWEEP_COUNT + NATIVE_WINDOW_COUNT)
    )
    valid = (
        len(coordination) == 1
        and coordination[0].get("trigger") == expected_trigger
        and int(coordination[0]["actual_fes"]) == source_actual
        and complete_groups
        and all(value == block_count for value in source_sweep_counts.values())
        and all(value == block_count for value in native_sweep_counts.values())
        and all(event.get("cold_start") is True for event in groups)
        and all(event.get("state_restored") is False for event in groups)
        and len({int(event["stage_index"]) for event in groups}) == len(groups)
        and len({int(event["seed"]) for event in groups}) == len(groups)
        and ordered
    )
    return {
        "valid": valid,
        "trigger": expected_trigger,
        "source_actual_fes": source_actual,
        "coordination_actual_fes": (
            int(coordination[0]["actual_fes"]) if len(coordination) == 1 else None
        ),
        "block_count": block_count,
        "cold_group_visit_count": len(groups),
        "source_sweep_group_counts": source_sweep_counts,
        "native_sweep_group_counts": native_sweep_counts,
        "unique_stage_seed_count": len({int(event["seed"]) for event in groups}),
        "state_restore_count": sum(event.get("state_restored") is True for event in groups),
        "events_ordered": ordered,
    }


def _execute_variant(
    variant: str,
    context: ActionContext,
) -> tuple[ActionResult, list[dict[str, object]], dict[str, object] | None]:
    if variant == "gcb_frozen_current":
        result = _legacy_module().GcbExecutor().execute(context)
        return result, [], None
    if variant == "gcb_three_source_burst_native":
        trace: list[dict[str, object]] = []
        result = GcbExecutor().execute_schedule(context, event_trace=trace)
        return result, trace, _trace_summary(trace, context.checkpoint)
    raise ValueError(f"unsupported GCB schedule-ablation variant: {variant}")


def _threadpools() -> list[dict[str, object]]:
    return [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
            "user_api": item.get("user_api"),
        }
        for item in threadpool_info()
    ]


def _run_arm(context: ArmContext) -> dict[str, object]:
    checkpoint_context = final_run.CheckpointContext(
        context.case_id,
        context.run_seed,
        TOTAL_BUDGET_FES,
        context.output_root,
    )
    checkpoint_receipt = final_run._validate_checkpoint(
        checkpoint_context.receipt_path,
        checkpoint_context,
    )
    checkpoint_payload = checkpoint_receipt.get("checkpoint")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError(f"{context.key} shared checkpoint is invalid")
    checkpoint = final_run._checkpoint_from_payload(checkpoint_payload)
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
        pools = _threadpools()
        if not pools or any(item["num_threads"] != 1 for item in pools):
            raise RuntimeError(f"{context.key} native thread limit is not one: {pools}")
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
        "action_seed": context.run_seed,
        "phase1_fes": checkpoint.phase1_fes,
        "phase1_relation_count": checkpoint.overlap_relation_count,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "action_checkpoint_hash": result.checkpoint_hash,
        "phase2_consumed_fes": result.consumed_fes,
        "terminal_fes": result.terminal_fes,
        "final_error": result.final_error,
        "terminal_state_finite": math.isfinite(result.final_error),
        "route": result.route,
        "schedule_trace": trace,
        "schedule_trace_summary": trace_summary,
        "action_result_hash": result.result_hash,
        "runtime_warnings": runtime_warnings,
        "native_thread_limit_verified": True,
        "threadpools": pools,
        "production_hcc_runtime_imports": _production_hcc_imports(),
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    receipt = {**body, "receipt_sha256": _canonical(body)}
    _write_json(context.receipt_path, receipt)
    return receipt


def _validate_arm(path: Path, context: ArmContext) -> dict[str, object]:
    receipt = _load_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != _canonical(receipt):
        raise ValueError(f"{context.key} receipt hash drifted")
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "variant": context.variant,
        "action_seed": context.run_seed,
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
    if receipt.get("checkpoint_hash") != receipt.get("action_checkpoint_hash"):
        raise ValueError(f"{context.key} action did not use its shared checkpoint")
    if not math.isfinite(float(receipt.get("final_error", math.nan))):
        raise ValueError(f"{context.key} final error is not finite")
    trace = receipt.get("schedule_trace")
    trace_summary = receipt.get("schedule_trace_summary")
    if context.variant == "gcb_three_source_burst_native":
        if not isinstance(trace, list) or not isinstance(trace_summary, Mapping):
            raise ValueError(f"{context.key} candidate trace is missing")
        checkpoint_context = final_run.CheckpointContext(
            context.case_id,
            context.run_seed,
            TOTAL_BUDGET_FES,
            context.output_root,
        )
        checkpoint_receipt = final_run._validate_checkpoint(
            checkpoint_context.receipt_path,
            checkpoint_context,
        )
        checkpoint = final_run._checkpoint_from_payload(checkpoint_receipt["checkpoint"])
        if trace_summary != _trace_summary(trace, checkpoint) or trace_summary.get("valid") is not True:
            raise ValueError(f"{context.key} candidate trace is invalid")
    elif trace != [] or trace_summary is not None:
        raise ValueError(f"{context.key} baseline unexpectedly carries a candidate trace")
    final_run._validate_runtime_warnings(receipt.get("runtime_warnings"), context.key)
    receipt["receipt_sha256"] = claimed
    return receipt


def _paired_ratio(candidate: float, baseline: float) -> float | None:
    if baseline > 0.0:
        return candidate / baseline
    return 1.0 if candidate == 0.0 else None


def _summarize(
    rows: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    by_key = {
        (str(row["case_id"]), int(row["run_seed"]), str(row["variant"])): row
        for row in rows
    }
    pairs = []
    for case_id in CASES:
        for seed in SEEDS:
            baseline = by_key[(case_id, seed, "gcb_frozen_current")]
            candidate = by_key[(case_id, seed, "gcb_three_source_burst_native")]
            baseline_error = float(baseline["final_error"])
            candidate_error = float(candidate["final_error"])
            ratio = _paired_ratio(candidate_error, baseline_error)
            pairs.append(
                {
                    "case_id": case_id,
                    "run_seed": seed,
                    "checkpoint_hash": candidate["checkpoint_hash"],
                    "same_checkpoint": candidate["checkpoint_hash"] == baseline["checkpoint_hash"],
                    "baseline_final_error": baseline_error,
                    "candidate_final_error": candidate_error,
                    "candidate_to_baseline_ratio": ratio,
                    "candidate_won_or_tied": candidate_error <= baseline_error,
                    "candidate_schedule_trace_valid": bool(
                        candidate.get("schedule_trace_summary", {}).get("valid", False)  # type: ignore[union-attr]
                    ),
                }
            )
    case_summaries = []
    for case_id in CASES:
        active = [pair for pair in pairs if pair["case_id"] == case_id]
        ratios = [pair["candidate_to_baseline_ratio"] for pair in active]
        geometric_mean = (
            math.exp(sum(math.log(float(value)) for value in ratios) / len(ratios))
            if all(value is not None and float(value) > 0.0 for value in ratios)
            else 0.0 if all(value == 0.0 for value in ratios) else None
        )
        case_summaries.append(
            {
                "case_id": case_id,
                "pair_count": len(active),
                "candidate_win_or_tie_count": sum(
                    bool(pair["candidate_won_or_tied"]) for pair in active
                ),
                "candidate_to_baseline_geometric_mean_ratio": geometric_mean,
                "paired_improvement_passed": (
                    geometric_mean is not None
                    and geometric_mean
                    < float(
                        protocol["acceptance_gates"][  # type: ignore[index]
                            "per_case_geometric_mean_ratio_lt"
                        ]
                    )
                ),
            }
        )
    warning_summary = final_run._runtime_warning_summary(rows)
    ratios = [pair["candidate_to_baseline_ratio"] for pair in pairs]
    exact_terminal = all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows)
    same_checkpoints = all(bool(pair["same_checkpoint"]) for pair in pairs)
    finite = all(math.isfinite(float(row["final_error"])) for row in rows)
    valid_traces = all(bool(pair["candidate_schedule_trace_valid"]) for pair in pairs)
    win_count = sum(bool(pair["candidate_won_or_tied"]) for pair in pairs)
    maximum_ratio = max(float(value) for value in ratios if value is not None)
    gates = protocol["acceptance_gates"]
    gate_passed = (
        same_checkpoints
        and exact_terminal
        and finite
        and valid_traces
        and all(bool(row["paired_improvement_passed"]) for row in case_summaries)
        and win_count >= int(gates["overall_candidate_win_count_gte"])  # type: ignore[index]
        and all(value is not None for value in ratios)
        and maximum_ratio <= float(gates["maximum_pair_ratio_lte"])  # type: ignore[index]
        and warning_summary["all_runtime_warnings_known"] is True
    )
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_count": len(CASES) * len(SEEDS),
        "arm_count": len(rows),
        "pair_count": len(pairs),
        "phase1_fes": PHASE1_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "same_checkpoint_per_pair": same_checkpoints,
        "exact_terminal_fes": exact_terminal,
        "all_final_errors_finite": finite,
        "candidate_schedule_trace_valid": valid_traces,
        "candidate_win_or_tie_count": win_count,
        "maximum_candidate_to_baseline_ratio": maximum_ratio,
        "case_summaries": case_summaries,
        "pairs": pairs,
        **warning_summary,
        "paired_development_gate_passed": gate_passed,
        "final_25_seed_recovery_evaluated": False,
        "selector_or_routing_authorized": False,
    }
    return {**body, "summary_sha256": _canonical(body)}


def _contexts(output_root: Path, manifest_sha256: str) -> tuple[ArmContext, ...]:
    return tuple(
        ArmContext(case_id, seed, variant, output_root, manifest_sha256)
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
    gate = _preflight(protocol_path, resume=resume)
    output_root = _resolved(str(protocol["output_root"]))
    max_workers = int(protocol["max_workers"] if workers is None else workers)
    arm_count = len(CASES) * len(SEEDS) * len(VARIANTS)
    if not 1 <= max_workers <= arm_count:
        raise ValueError(f"workers must be in 1..{arm_count}")
    if resume:
        manifest = _load_manifest(output_root, protocol_path)
    else:
        output_root.mkdir(parents=True)
        _write_json(output_root / "protocol.json", protocol)
        _write_json(output_root / "preflight.json", gate)
        manifest = _manifest(protocol_path, gate)
        _write_json(output_root / "manifest.json", manifest)
    checkpoint_contexts = tuple(
        final_run.CheckpointContext(case_id, seed, TOTAL_BUDGET_FES, output_root)
        for seed in SEEDS
        for case_id in CASES
    )
    final_run._run_parallel(
        checkpoint_contexts,
        final_run._run_checkpoint,
        max_workers=min(max_workers, len(checkpoint_contexts)),
        progress_path=output_root / "checkpoint_progress.json",
        receipt_path=lambda context: context.receipt_path,
        validator=final_run._validate_checkpoint,
        resume=resume,
    )
    contexts = _contexts(output_root, str(manifest["manifest_sha256"]))
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
    _write_json(output_root / "summary.json", summary)
    final_run._require_known_runtime_warnings(summary, stage="GCB paired development gate")
    return summary


def verify(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    manifest = _load_manifest(output_root, protocol_path)
    checkpoint_contexts = tuple(
        final_run.CheckpointContext(case_id, seed, TOTAL_BUDGET_FES, output_root)
        for seed in SEEDS
        for case_id in CASES
    )
    for context in checkpoint_contexts:
        final_run._validate_checkpoint(context.receipt_path, context)
    contexts = _contexts(output_root, str(manifest["manifest_sha256"]))
    rows = [_validate_arm(context.receipt_path, context) for context in contexts]
    expected = _summarize(rows, protocol)
    stored = _load_json(output_root / "summary.json")
    stored_claimed = stored.pop("summary_sha256", None)
    if stored_claimed != _canonical(stored):
        raise ValueError("GCB schedule-ablation summary hash drifted")
    expected.pop("summary_sha256", None)
    expected.pop("generated_at_utc", None)
    stored.pop("generated_at_utc", None)
    if stored != expected:
        raise ValueError("GCB schedule-ablation summary content drifted")
    stored["summary_sha256"] = stored_claimed
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
