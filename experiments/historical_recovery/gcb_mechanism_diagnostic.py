"""Isolate GCB order and seed-lifecycle deltas on frozen development checkpoints."""

# Thread caps must be set before NumPy, PyPop7, or ARAC imports.
# ruff: noqa: E402

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path

from threadpoolctl import threadpool_info, threadpool_limits

import experiments.final.run as final_run
import experiments.historical_recovery.gcb_schedule_ablation as source_gate
from arac.actions._execution import (
    derived_seed,
    run_cold_start_block_sweeps,
    run_full_space,
    terminal_result,
)
from arac.actions.gcb import GcbExecutor
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, ActionResult, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("gcb_mechanism_diagnostic_protocol.json")
CASES = ("R1", "R3", "R6")
SEEDS = (31_001, 31_002, 31_003)
VARIANTS = (
    "graph_order_historical_seeds",
    "native_order_legacy_seeds",
    "native_order_historical_seeds",
)
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
RECEIPT_SCHEMA = "arac-gcb-mechanism-diagnostic-receipt-v1"
SUMMARY_SCHEMA = "arac-gcb-mechanism-diagnostic-summary-v1"
SOURCE_PATHS = (
    "experiments/historical_recovery/gcb_mechanism_diagnostic.py",
    "experiments/historical_recovery/gcb_mechanism_diagnostic_protocol.json",
    "experiments/historical_recovery/gcb_schedule_ablation.py",
    "experiments/final/run.py",
    "src/arac/actions/_execution.py",
    "src/arac/actions/gcb.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/evidence/phase1.py",
    "src/arac/runtime/contracts.py",
    "src/arac/runtime/ledger.py",
    "src/arac/runtime/optimizers.py",
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
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def _write_json(path: Path, payload: object) -> None:
    final_run._atomic_json(path, payload)


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(_resolved(relative)) for relative in SOURCE_PATHS}


def _production_hcc_imports() -> list[str]:
    matches = []
    for path in (REPOSITORY_ROOT / "src" / "arac").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        if any(token in source for token in ("from hcc", "import hcc", "vendor.hcc", "vendor/hcc")):
            matches.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    return matches


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-gcb-mechanism-diagnostic-protocol-v1",
        "status": "frozen_paired_development_diagnostic",
        "cases": list(CASES),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "total_budget_fes": TOTAL_BUDGET_FES,
        "expected_phase1_fes": PHASE1_FES,
        "native_threads": 1,
        "max_workers": 18,
        "source_root": "artifacts/gcb_schedule_ablation_v2",
        "output_root": "artifacts/gcb_mechanism_diagnostic_v1",
        "production_hcc_runtime_imports_allowed": False,
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"GCB mechanism diagnostic protocol drifted: {key}")
    if set(protocol) != set(expected) | {"purpose", "acceptance_gates"}:
        raise ValueError("GCB mechanism diagnostic protocol keys drifted")
    gates = {
        "same_checkpoint_as_frozen_baseline": True,
        "exact_terminal_fes": True,
        "all_final_errors_finite": True,
        "candidate_schedule_trace_valid": True,
        "per_case_geometric_mean_ratio_lt": 1.0,
        "overall_candidate_win_count_gte": 6,
        "maximum_pair_ratio_lte": 10.0,
        "all_runtime_warnings_known": True,
    }
    if protocol.get("acceptance_gates") != gates:
        raise ValueError("GCB mechanism diagnostic acceptance gates drifted")
    return protocol


def _baseline_receipt(source_root: Path, case_id: str, seed: int) -> dict[str, object]:
    path = source_root / "arms" / case_id / f"seed_{seed}" / "gcb_frozen_current.json"
    receipt = _load_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != _canonical(receipt):
        raise ValueError(f"frozen baseline receipt drifted: {case_id}:{seed}")
    receipt["receipt_sha256"] = claimed
    return receipt


def preflight(path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    source_summary = source_gate.verify()
    source_root = _resolved(str(protocol["source_root"]))
    output_root = _resolved(str(protocol["output_root"]))
    if output_root.exists() and not resume:
        raise ValueError(f"fresh GCB diagnostic output already exists: {output_root}")
    imports = _production_hcc_imports()
    if imports:
        raise ValueError(f"production HCC imports remain: {imports}")
    baseline_hashes = {
        f"{case_id}:{seed}": str(_baseline_receipt(source_root, case_id, seed)["receipt_sha256"])
        for case_id in CASES
        for seed in SEEDS
    }
    return {
        "schema_version": "arac-gcb-mechanism-diagnostic-preflight-v1",
        "protocol_sha256": _sha256(protocol_path),
        "source_summary_sha256": source_summary["summary_sha256"],
        "source_sha256": _source_hashes(),
        "baseline_receipt_sha256": baseline_hashes,
        "arm_count": len(CASES) * len(SEEDS) * len(VARIANTS),
        "production_hcc_runtime_imports": imports,
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
        "passed": True,
    }


def _manifest(protocol_path: Path, gate: Mapping[str, object]) -> dict[str, object]:
    body = {
        "schema_version": "arac-gcb-mechanism-diagnostic-manifest-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha256(protocol_path),
        "source_summary_sha256": gate["source_summary_sha256"],
        "source_sha256": gate["source_sha256"],
        "baseline_receipt_sha256": gate["baseline_receipt_sha256"],
        "production_hcc_runtime_imports": [],
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    return {**body, "manifest_sha256": _canonical(body)}


def _load_manifest(output_root: Path, protocol_path: Path) -> dict[str, object]:
    manifest = _load_json(output_root / "manifest.json")
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _canonical(manifest):
        raise ValueError("GCB diagnostic manifest hash drifted")
    gate = preflight(protocol_path, resume=True)
    for key in ("protocol_sha256", "source_summary_sha256", "source_sha256", "baseline_receipt_sha256"):
        if manifest.get(key) != gate.get(key):
            raise ValueError(f"GCB diagnostic manifest drifted: {key}")
    manifest["manifest_sha256"] = claimed
    return manifest


@dataclass(frozen=True)
class ArmContext:
    case_id: str
    run_seed: int
    variant: str
    source_root: Path
    output_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}:{self.variant}"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.case_id / f"seed_{self.run_seed}" / f"{self.variant}.json"


def _variant_settings(
    variant: str,
    context: ActionContext,
) -> tuple[tuple[int, ...] | None, object | None, bool]:
    graph_order = GcbExecutor._block_order(context)
    if variant == "native_order_legacy_seeds":
        return None, None, False

    namespace = f"gcb-native-{context.checkpoint.checkpoint_hash}"

    def seed_factory(stage_index: int) -> int:
        return derived_seed(context, namespace, stage_index)

    if variant == "graph_order_historical_seeds":
        return graph_order, seed_factory, True
    if variant == "native_order_historical_seeds":
        return None, seed_factory, True
    raise ValueError(f"unsupported GCB diagnostic variant: {variant}")


def _trace_summary(
    trace: Sequence[Mapping[str, object]],
    checkpoint: PhaseCheckpoint,
    *,
    source_namespace: str,
    native_namespace: str,
) -> dict[str, object]:
    groups = [event for event in trace if event.get("event") == "cold_group_visit"]
    source = [event for event in groups if event.get("namespace") == source_namespace]
    native = [event for event in groups if event.get("namespace") == native_namespace]
    coordination = [event for event in trace if event.get("event") == "full_space_coordination"]
    block_count = len(checkpoint.blocks)
    source_counts = {
        str(index): sum(int(event["sweep_index"]) == index for event in source)
        for index in range(3)
    }
    native_counts = {
        str(index): sum(int(event["sweep_index"]) == index for event in native)
        for index in range(3, 6)
    }
    source_actual = sum(
        int(event["actual_fes"]) for event in source if int(event["sweep_index"]) == 2
    )
    ordered = all(
        int(right["start_fes"]) >= int(left["end_fes"])
        for left, right in zip(trace, trace[1:], strict=False)
    )
    valid = (
        len(coordination) == 1
        and int(coordination[0]["actual_fes"]) == source_actual
        and all(value == block_count for value in source_counts.values())
        and all(value == block_count for value in native_counts.values())
        and all(event.get("cold_start") is True for event in groups)
        and all(event.get("state_restored") is False for event in groups)
        and ordered
    )
    return {
        "valid": valid,
        "source_actual_fes": source_actual,
        "coordination_actual_fes": int(coordination[0]["actual_fes"]) if len(coordination) == 1 else None,
        "block_count": block_count,
        "cold_group_visit_count": len(groups),
        "source_sweep_group_counts": source_counts,
        "native_sweep_group_counts": native_counts,
        "state_restore_count": sum(event.get("state_restored") is True for event in groups),
        "events_ordered": ordered,
    }


def _execute_variant(
    variant: str,
    context: ActionContext,
) -> tuple[ActionResult, list[dict[str, object]], dict[str, object]]:
    block_order, seed_factory, checkpoint_burst = _variant_settings(variant, context)
    source_namespace = "gcb-warmup"
    native_namespace = "gcb-continuation"
    trace: list[dict[str, object]] = []
    source_fes, source_sweeps = run_cold_start_block_sweeps(
        context,
        requested_fes=context.ledger.remaining,
        sweep_limit=3,
        block_order=block_order,
        namespace=source_namespace,
        seed_factory=seed_factory,  # type: ignore[arg-type]
        start_sweep_index=0,
        event_trace=trace,
    )
    if len(source_sweeps) != 3:
        raise RuntimeError("GCB diagnostic source schedule did not complete three sweeps")
    coordination_budget = min(context.ledger.remaining, source_sweeps[-1])
    coordination_start = context.ledger.count
    burst_namespace = "gcb-global-coordination"
    if checkpoint_burst:
        burst_namespace += f"-{context.checkpoint.checkpoint_hash}"
    run_full_space(
        context,
        algorithm="sepcmaes",
        budget_fes=coordination_budget,
        namespace=burst_namespace,
    )
    trace.append(
        {
            "event": "full_space_coordination",
            "start_fes": coordination_start,
            "requested_fes": coordination_budget,
            "actual_fes": context.ledger.count - coordination_start,
            "end_fes": context.ledger.count,
        }
    )
    native_fes, native_sweeps = run_cold_start_block_sweeps(
        context,
        requested_fes=context.ledger.remaining,
        block_order=block_order,
        namespace=native_namespace,
        seed_factory=seed_factory,  # type: ignore[arg-type]
        start_sweep_index=3,
        event_trace=trace,
    )
    if len(native_sweeps) < 3:
        raise RuntimeError("GCB diagnostic native schedule did not complete three windows")
    tail_fes = context.ledger.remaining
    if tail_fes:
        run_full_space(context, algorithm="sepcmaes", namespace="gcb-terminal-alignment")
    route = (
        f"{variant}_source_{source_fes}_sweeps_{len(source_sweeps)}_"
        f"coordination_{coordination_budget}_native_{native_fes}_"
        f"windows_{len(native_sweeps)}_tail_{tail_fes}"
    )
    return (
        terminal_result(context, route=route),
        trace,
        _trace_summary(
            trace,
            context.checkpoint,
            source_namespace=source_namespace,
            native_namespace=native_namespace,
        ),
    )


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


def _checkpoint(context: ArmContext) -> PhaseCheckpoint:
    checkpoint_context = final_run.CheckpointContext(
        context.case_id,
        context.run_seed,
        TOTAL_BUDGET_FES,
        context.source_root,
    )
    receipt = final_run._validate_checkpoint(checkpoint_context.receipt_path, checkpoint_context)
    return final_run._checkpoint_from_payload(receipt["checkpoint"])


def _run_arm(context: ArmContext) -> dict[str, object]:
    checkpoint = _checkpoint(context)
    baseline = _baseline_receipt(context.source_root, context.case_id, context.run_seed)
    if baseline["checkpoint_hash"] != checkpoint.checkpoint_hash:
        raise ValueError(f"{context.key} baseline checkpoint drifted")
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
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "baseline_receipt_sha256": baseline["receipt_sha256"],
        "baseline_final_error": baseline["final_error"],
        "action_checkpoint_hash": result.checkpoint_hash,
        "phase1_fes": checkpoint.phase1_fes,
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
    checkpoint = _checkpoint(context)
    baseline = _baseline_receipt(context.source_root, context.case_id, context.run_seed)
    if (
        receipt.get("checkpoint_hash") != checkpoint.checkpoint_hash
        or receipt.get("action_checkpoint_hash") != checkpoint.checkpoint_hash
        or receipt.get("baseline_receipt_sha256") != baseline["receipt_sha256"]
        or receipt.get("baseline_final_error") != baseline["final_error"]
    ):
        raise ValueError(f"{context.key} source binding drifted")
    trace = receipt.get("schedule_trace")
    trace_summary = receipt.get("schedule_trace_summary")
    if not isinstance(trace, list) or not isinstance(trace_summary, Mapping):
        raise ValueError(f"{context.key} trace is missing")
    source_namespace = "gcb-warmup"
    native_namespace = "gcb-continuation"
    if trace_summary != _trace_summary(
        trace,
        checkpoint,
        source_namespace=source_namespace,
        native_namespace=native_namespace,
    ) or trace_summary.get("valid") is not True:
        raise ValueError(f"{context.key} trace is invalid")
    final_run._validate_runtime_warnings(receipt.get("runtime_warnings"), context.key)
    receipt["receipt_sha256"] = claimed
    return receipt


def _ratio(candidate: float, baseline: float) -> float | None:
    if baseline > 0.0:
        return candidate / baseline
    return 1.0 if candidate == 0.0 else None


def _summarize(
    rows: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    gates = protocol["acceptance_gates"]
    variant_summaries = []
    for variant in VARIANTS:
        active = [row for row in rows if row["variant"] == variant]
        pairs = []
        for row in active:
            ratio = _ratio(float(row["final_error"]), float(row["baseline_final_error"]))
            pairs.append(
                {
                    "case_id": row["case_id"],
                    "run_seed": row["run_seed"],
                    "checkpoint_hash": row["checkpoint_hash"],
                    "baseline_final_error": row["baseline_final_error"],
                    "candidate_final_error": row["final_error"],
                    "candidate_to_baseline_ratio": ratio,
                    "candidate_won_or_tied": float(row["final_error"]) <= float(row["baseline_final_error"]),
                }
            )
        case_summaries = []
        for case_id in CASES:
            case_pairs = [pair for pair in pairs if pair["case_id"] == case_id]
            ratios = [float(pair["candidate_to_baseline_ratio"]) for pair in case_pairs]
            geometric = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            case_summaries.append(
                {
                    "case_id": case_id,
                    "geometric_mean_ratio": geometric,
                    "win_or_tie_count": sum(bool(pair["candidate_won_or_tied"]) for pair in case_pairs),
                    "passed": geometric < float(gates["per_case_geometric_mean_ratio_lt"]),  # type: ignore[index]
                }
            )
        win_count = sum(bool(pair["candidate_won_or_tied"]) for pair in pairs)
        maximum_ratio = max(float(pair["candidate_to_baseline_ratio"]) for pair in pairs)
        passed = (
            all(bool(item["passed"]) for item in case_summaries)
            and win_count >= int(gates["overall_candidate_win_count_gte"])  # type: ignore[index]
            and maximum_ratio <= float(gates["maximum_pair_ratio_lte"])  # type: ignore[index]
        )
        variant_summaries.append(
            {
                "variant": variant,
                "candidate_win_or_tie_count": win_count,
                "maximum_candidate_to_baseline_ratio": maximum_ratio,
                "case_summaries": case_summaries,
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
    passing = [item["variant"] for item in variant_summaries if item["performance_gate_passed"]]
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "arm_count": len(rows),
        "integrity_gate_passed": integrity,
        "variant_summaries": variant_summaries,
        "passing_variants": passing,
        "diagnostic_gate_passed": integrity and bool(passing),
        **warning_summary,
        "final_25_seed_recovery_evaluated": False,
        "selector_or_routing_authorized": False,
    }
    return {**body, "summary_sha256": _canonical(body)}


def _contexts(protocol: Mapping[str, object], manifest_sha256: str) -> tuple[ArmContext, ...]:
    source_root = _resolved(str(protocol["source_root"]))
    output_root = _resolved(str(protocol["output_root"]))
    return tuple(
        ArmContext(case_id, seed, variant, source_root, output_root, manifest_sha256)
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
    output_root = _resolved(str(protocol["output_root"]))
    max_workers = int(protocol["max_workers"] if workers is None else workers)
    if not 1 <= max_workers <= len(CASES) * len(SEEDS) * len(VARIANTS):
        raise ValueError("workers is outside the diagnostic arm count")
    if resume:
        manifest = _load_manifest(output_root, protocol_path)
    else:
        output_root.mkdir(parents=True)
        _write_json(output_root / "protocol.json", protocol)
        _write_json(output_root / "preflight.json", gate)
        manifest = _manifest(protocol_path, gate)
        _write_json(output_root / "manifest.json", manifest)
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
    _write_json(output_root / "summary.json", summary)
    final_run._require_known_runtime_warnings(summary, stage="GCB mechanism diagnostic")
    return summary


def verify(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    manifest = _load_manifest(output_root, protocol_path)
    contexts = _contexts(protocol, str(manifest["manifest_sha256"]))
    rows = [_validate_arm(context.receipt_path, context) for context in contexts]
    expected = _summarize(rows, protocol)
    stored = _load_json(output_root / "summary.json")
    claimed = stored.pop("summary_sha256", None)
    if claimed != _canonical(stored):
        raise ValueError("GCB diagnostic summary hash drifted")
    expected.pop("summary_sha256", None)
    expected.pop("generated_at_utc", None)
    stored.pop("generated_at_utc", None)
    if stored != expected:
        raise ValueError("GCB diagnostic summary content drifted")
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
