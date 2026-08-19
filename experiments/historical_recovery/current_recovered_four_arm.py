"""Build the recovered four-action outcome matrix from shared Phase-I checkpoints."""

# Thread caps must be set before NumPy and optimizer imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any
import warnings

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

from threadpoolctl import threadpool_info, threadpool_limits

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.analysis.outcome_selector import ActionOutcome, OutcomeRecord
from arac.benchmarks.aob import AobBenchmark
from arac.evidence.phase1 import PHASE1_FEATURE_NAMES, PHASE1_PROTOCOL
from arac.runtime.contracts import ACTION_NAMES, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery.replay import _checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("current_recovered_four_arm_protocol.json")
PROTOCOL_SCHEMA = "arac-current-recovered-four-arm-protocol-v1"
MANIFEST_SCHEMA = "arac-current-recovered-four-arm-manifest-v1"
ARM_SCHEMA = "arac-current-recovered-four-arm-receipt-v1"
SUMMARY_SCHEMA = "arac-current-recovered-four-arm-summary-v1"
CHECKPOINT_RECEIPT_SCHEMA = "arac-independent-phase1-checkpoint-v1"
CURRENT_E2E_RECEIPT_SCHEMA = "arac-current-arac-aob24-recovery-receipt-v1"

SOURCE_PATHS = (
    "experiments/historical_recovery/current_recovered_four_arm.py",
    "src/arac/actions/_execution.py",
    "src/arac/actions/recovered.py",
    "src/arac/actions/recovered_registry.py",
    "src/arac/actions/ctp.py",
    "src/arac/actions/gcb.py",
    "src/arac/runtime/contracts.py",
    "src/arac/runtime/ledger.py",
    "src/arac/runtime/phase2.py",
    "src/arac/runtime/optimizers.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/analysis/outcome_selector.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path, pattern: str) -> tuple[int, str]:
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return len(files), digest.hexdigest()


def _verified_hash(payload: dict[str, Any], field: str, label: str) -> dict[str, Any]:
    claimed = payload.pop(field, None)
    if claimed != canonical_sha256(payload):
        raise ValueError(f"{label} hash drifted")
    payload[field] = claimed
    return payload


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": PROTOCOL_SCHEMA,
        "actions": list(ACTION_NAMES),
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "registry": "RecoveredActionRegistry",
        "allow_out_of_bounds": True,
        "boundary_profile": "official_hcc_no_offspring_clipping_reproduction",
        "native_threads": 1,
        "selector_execution_allowed": False,
        "probe_execution_allowed": False,
        "racing_execution_allowed": False,
    }
    if any(protocol.get(name) != value for name, value in expected.items()):
        raise ValueError("recovered four-arm protocol anchor drifted")
    cases = tuple(str(value) for value in protocol.get("cases", ()))
    seeds = tuple(int(value) for value in protocol.get("seeds", ()))
    if len(cases) != 24 or len(set(cases)) != 24 or len(seeds) != 25 or len(set(seeds)) != 25:
        raise ValueError("recovered four-arm matrix dimensions drifted")
    expected_cases = {f"{family}{index}" for family in "AERS" for index in range(1, 7)}
    if set(cases) != expected_cases:
        raise ValueError("recovered four-arm cases do not cover AOB-24")
    if int(protocol.get("max_workers", 0)) <= 0:
        raise ValueError("max_workers must be positive")
    for name in ("checkpoint_root", "current_e2e_receipt_root"):
        if not (REPOSITORY_ROOT / str(protocol[name])).is_dir():
            raise ValueError(f"protocol source root is missing: {name}")
    return protocol


@dataclass(frozen=True)
class ArmContext:
    case_id: str
    run_seed: int
    action_name: str
    checkpoint_root: Path
    current_receipt_root: Path
    output_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}:{self.action_name}"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_root / self.case_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def current_receipt_path(self) -> Path:
        return self.current_receipt_root / self.case_id / f"seed_{self.run_seed}" / "receipt.json"

    @property
    def receipt_path(self) -> Path:
        return (
            self.output_root
            / "arms"
            / self.case_id
            / f"seed_{self.run_seed}"
            / f"{self.action_name}.json"
        )

    @property
    def failure_path(self) -> Path:
        return (
            self.output_root
            / "failures"
            / self.case_id
            / f"seed_{self.run_seed}"
            / f"{self.action_name}.json"
        )


def _load_verified_checkpoint(context: ArmContext):
    wrapper = _verified_hash(
        _load_json(context.checkpoint_path),
        "receipt_hash",
        f"{context.key} retained checkpoint receipt",
    )
    if (
        wrapper.get("schema_version") != CHECKPOINT_RECEIPT_SCHEMA
        or wrapper.get("case_id") != context.case_id
        or wrapper.get("run_seed") != context.run_seed
        or wrapper.get("max_fes") != 3_000_000
    ):
        raise ValueError(f"{context.key} retained checkpoint identity drifted")
    checkpoint_payload = wrapper.get("checkpoint")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError(f"{context.key} checkpoint payload is missing")
    checkpoint = _checkpoint(checkpoint_payload)
    if (
        checkpoint.checkpoint_hash != wrapper.get("checkpoint_hash")
        or checkpoint.protocol != PHASE1_PROTOCOL
        or checkpoint.run_seed != context.run_seed
        or checkpoint.phase1_fes != 180_000
        or checkpoint.total_budget_fes != 3_000_000
        or checkpoint.feature_names != PHASE1_FEATURE_NAMES
    ):
        raise ValueError(f"{context.key} retained checkpoint contract drifted")

    current = _verified_hash(
        _load_json(context.current_receipt_path),
        "receipt_sha256",
        f"{context.key} current E2E receipt",
    )
    if (
        current.get("schema_version") != CURRENT_E2E_RECEIPT_SCHEMA
        or current.get("case_id") != context.case_id
        or current.get("run_seed") != context.run_seed
        or current.get("phase1_fes") != 180_000
        or current.get("phase1_checkpoint_hash") != checkpoint.checkpoint_hash
        or current.get("action_checkpoint_hash") != checkpoint.checkpoint_hash
    ):
        raise ValueError(f"{context.key} checkpoint does not match current E2E")
    return checkpoint


def _threadpools() -> list[dict[str, Any]]:
    return [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
        }
        for item in threadpool_info()
    ]


def _warning_rows(caught: Sequence[warnings.WarningMessage]) -> list[dict[str, Any]]:
    counts = Counter((item.category.__name__, str(item.message)) for item in caught)
    return [
        {"category": category, "message": message, "count": count}
        for (category, message), count in sorted(counts.items())
    ]


def _run_arm(context: ArmContext) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            pools = _threadpools()
            if any(pool["num_threads"] != 1 for pool in pools):
                raise RuntimeError(f"native thread limit is not one: {pools}")
            checkpoint = _load_verified_checkpoint(context)
            problem = AobBenchmark().load(context.case_id)
            registry = RecoveredActionRegistry()
            ledger = EvaluationLedger.from_checkpoint(
                problem,
                total_budget=checkpoint.total_budget_fes,
                phase1_fes=checkpoint.phase1_fes,
                incumbent=checkpoint.incumbent,
                incumbent_error=checkpoint.incumbent_error,
                allow_out_of_bounds=registry.allow_out_of_bounds,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = execute_phase2_action(
                    context.action_name,
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=context.run_seed,
                    registry=registry,
                )
            if (
                result.checkpoint_hash != checkpoint.checkpoint_hash
                or result.action_name != context.action_name
                or result.action_seed != context.run_seed
                or result.consumed_fes != 2_820_000
                or result.terminal_fes != 3_000_000
                or ledger.count != 3_000_000
                or not math.isfinite(result.final_error)
                or result.final_error != ledger.best_error
            ):
                raise RuntimeError(f"{context.key} terminal action contract failed")
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": context.manifest_sha256,
                "case_id": context.case_id,
                "run_seed": context.run_seed,
                "action_name": context.action_name,
                "action_seed": result.action_seed,
                "phase1_protocol": checkpoint.protocol,
                "phase1_fes": checkpoint.phase1_fes,
                "phase2_fes": result.consumed_fes,
                "terminal_fes": result.terminal_fes,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "checkpoint_file_sha256": _sha256(context.checkpoint_path),
                "feature_names": list(checkpoint.feature_names),
                "feature_values": list(checkpoint.feature_values),
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "action_result": result.payload(),
                "action_result_hash": result.result_hash,
                "route": result.route,
                "allow_out_of_bounds": ledger.allow_out_of_bounds,
                "boundary_profile": "official_hcc_no_offspring_clipping_reproduction",
                "runtime_warnings": _warning_rows(caught),
                "threadpools": pools,
                "native_thread_limit_verified": True,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            receipt = {**body, "receipt_hash": canonical_sha256(body)}
            _write_json(context.receipt_path, receipt)
            return receipt
    except BaseException as exc:
        failure = {
            "schema_version": "arac-current-recovered-four-arm-failure-v1",
            "manifest_sha256": context.manifest_sha256,
            "key": context.key,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(context.failure_path, failure)
        raise


def _validate_arm(path: Path, context: ArmContext) -> dict[str, Any]:
    receipt = _verified_hash(_load_json(path), "receipt_hash", f"{context.key} arm receipt")
    expected = {
        "schema_version": ARM_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "action_name": context.action_name,
        "action_seed": context.run_seed,
        "phase1_protocol": PHASE1_PROTOCOL,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "terminal_fes": 3_000_000,
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "allow_out_of_bounds": True,
        "boundary_profile": "official_hcc_no_offspring_clipping_reproduction",
        "native_thread_limit_verified": True,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"{context.key} receipt field drifted: {name}")
    checkpoint = _load_verified_checkpoint(context)
    if (
        receipt.get("checkpoint_hash") != checkpoint.checkpoint_hash
        or receipt.get("checkpoint_file_sha256") != _sha256(context.checkpoint_path)
        or receipt.get("action_result_hash")
        != canonical_sha256(receipt.get("action_result"))
        or not math.isfinite(float(receipt.get("final_error", math.nan)))
    ):
        raise ValueError(f"{context.key} receipt provenance drifted")
    result = receipt.get("action_result")
    if not isinstance(result, Mapping) or (
        result.get("checkpoint_hash") != checkpoint.checkpoint_hash
        or result.get("action_name") != context.action_name
        or result.get("action_seed") != context.run_seed
        or result.get("consumed_fes") != 2_820_000
        or result.get("terminal_fes") != 3_000_000
        or result.get("final_error") != receipt.get("final_error")
    ):
        raise ValueError(f"{context.key} action result drifted")
    return receipt


def _manifest(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    *,
    mode: str,
    cases: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    checkpoint_root = REPOSITORY_ROOT / str(protocol["checkpoint_root"])
    current_root = REPOSITORY_ROOT / str(protocol["current_e2e_receipt_root"])
    checkpoint_count, checkpoint_tree = _tree_sha256(checkpoint_root, "checkpoint.json")
    current_count, current_tree = _tree_sha256(current_root, "receipt.json")
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "mode": mode,
        "protocol_sha256": _sha256(protocol_path),
        "cases": list(cases),
        "seeds": list(seeds),
        "actions": list(ACTION_NAMES),
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "registry": "RecoveredActionRegistry",
        "boundary_profile": protocol["boundary_profile"],
        "source_sha256": {
            relative: _sha256(REPOSITORY_ROOT / relative) for relative in SOURCE_PATHS
        },
        "checkpoint_source": {
            "root": protocol["checkpoint_root"],
            "file_count": checkpoint_count,
            "tree_sha256": checkpoint_tree,
        },
        "current_e2e_source": {
            "root": protocol["current_e2e_receipt_root"],
            "file_count": current_count,
            "tree_sha256": current_tree,
        },
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _prepare_output(output_root: Path, manifest: Mapping[str, Any], *, resume: bool) -> None:
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"four-arm output already exists: {output_root}")
        if _load_json(manifest_path) != manifest:
            raise ValueError("resume manifest does not match the frozen campaign")
        return
    output_root.mkdir(parents=True)
    _write_json(manifest_path, manifest)


def _records_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[OutcomeRecord, ...]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), int(row["run_seed"]))].append(row)
    records = []
    for (case_id, run_seed), arms in sorted(grouped.items()):
        by_action = {str(row["action_name"]): row for row in arms}
        if set(by_action) != set(ACTION_NAMES):
            raise ValueError(f"{case_id}:seed-{run_seed} lacks all four recovered actions")
        hashes = {str(row["checkpoint_hash"]) for row in arms}
        features = {tuple(float(value) for value in row["feature_values"]) for row in arms}
        if len(hashes) != 1 or len(features) != 1:
            raise ValueError(f"{case_id}:seed-{run_seed} arms do not share one checkpoint")
        records.append(
            OutcomeRecord(
                case_id=case_id,
                run_seed=run_seed,
                checkpoint_hash=hashes.pop(),
                feature_names=PHASE1_FEATURE_NAMES,
                feature_values=features.pop(),
                outcomes=tuple(
                    ActionOutcome(
                        action_name=action,
                        final_error=float(by_action[action]["final_error"]),
                        result_hash=str(by_action[action]["action_result_hash"]),
                    )
                    for action in ACTION_NAMES
                ),
            )
        )
    return tuple(records)


def _write_records(path: Path, records: Sequence[OutcomeRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record.payload(), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _summary(rows: Sequence[Mapping[str, Any]], records: Sequence[OutcomeRecord]) -> dict[str, Any]:
    warning_counts = Counter()
    for row in rows:
        for warning in row.get("runtime_warnings", []):
            warning_counts[(warning["category"], warning["message"])] += int(warning["count"])
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "context_count": len(records),
        "arm_count": len(rows),
        "all_terminal_fes_exact": all(row["terminal_fes"] == 3_000_000 for row in rows),
        "all_checkpoint_bindings_exact": all(
            row["checkpoint_hash"] == row["action_result"]["checkpoint_hash"] for row in rows
        ),
        "all_final_errors_finite": all(math.isfinite(float(row["final_error"])) for row in rows),
        "label_counts": dict(sorted(Counter(record.action_label for record in records).items())),
        "runtime_warning_counts": [
            {"category": key[0], "message": key[1], "count": count}
            for key, count in sorted(warning_counts.items())
        ],
        "boundary_profile": "official_hcc_no_offspring_clipping_reproduction",
    }
    return {**body, "summary_hash": canonical_sha256(body)}


def _contexts(
    protocol: Mapping[str, Any],
    output_root: Path,
    manifest_sha256: str,
    cases: Sequence[str],
    seeds: Sequence[int],
) -> tuple[ArmContext, ...]:
    checkpoint_root = (REPOSITORY_ROOT / str(protocol["checkpoint_root"])).resolve()
    current_root = (REPOSITORY_ROOT / str(protocol["current_e2e_receipt_root"])).resolve()
    return tuple(
        ArmContext(
            case_id=case_id,
            run_seed=int(seed),
            action_name=action,
            checkpoint_root=checkpoint_root,
            current_receipt_root=current_root,
            output_root=output_root,
            manifest_sha256=manifest_sha256,
        )
        for seed in seeds
        for case_id in cases
        for action in ACTION_NAMES
    )


def run_campaign(
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    mode: str,
    resume: bool,
    max_workers: int | None = None,
) -> dict[str, Any]:
    resolved_protocol = Path(protocol_path).resolve()
    protocol = load_protocol(resolved_protocol)
    if mode == "preflight":
        cases = tuple(str(value) for value in protocol["preflight_cases"])
        seeds = tuple(int(value) for value in protocol["preflight_seeds"])
        output_root = (REPOSITORY_ROOT / str(protocol["preflight_output_root"])).resolve()
    elif mode == "full":
        cases = tuple(str(value) for value in protocol["cases"])
        seeds = tuple(int(value) for value in protocol["seeds"])
        output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    else:
        raise ValueError("mode must be preflight or full")
    manifest = _manifest(resolved_protocol, protocol, mode=mode, cases=cases, seeds=seeds)
    _prepare_output(output_root, manifest, resume=resume)
    contexts = _contexts(protocol, output_root, str(manifest["manifest_sha256"]), cases, seeds)
    rows: list[dict[str, Any]] = []
    pending = []
    for context in contexts:
        if resume and context.receipt_path.is_file():
            rows.append(_validate_arm(context.receipt_path, context))
        else:
            pending.append(context)
    progress = {
        "schema_version": "arac-current-recovered-four-arm-progress-v1",
        "mode": mode,
        "total": len(contexts),
        "completed": len(rows),
        "failed": 0,
        "pending": len(pending),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(output_root / "progress.json", progress)
    failures = []
    workers = int(protocol["max_workers"] if max_workers is None else max_workers)
    if workers <= 0:
        raise ValueError("max_workers must be positive")
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_arm, context): context for context in pending}
            for future in as_completed(futures):
                context = futures[future]
                try:
                    rows.append(_validate_arm(context.receipt_path, context))
                    future.result()
                except BaseException as exc:
                    failures.append({"key": context.key, "error": f"{type(exc).__name__}: {exc}"})
                progress.update(
                    {
                        "completed": len(rows),
                        "failed": len(failures),
                        "pending": len(contexts) - len(rows) - len(failures),
                        "updated_at_utc": datetime.now(UTC).isoformat(),
                    }
                )
                _write_json(output_root / "progress.json", progress)
    if failures:
        _write_json(
            output_root / "failure_summary.json",
            {"failure_count": len(failures), "failures": failures},
        )
        raise RuntimeError(f"recovered four-arm campaign has {len(failures)} failed arms")
    records = _records_from_rows(rows)
    expected_records = len(cases) * len(seeds)
    if len(rows) != expected_records * len(ACTION_NAMES) or len(records) != expected_records:
        raise RuntimeError("recovered four-arm campaign is incomplete")
    _write_records(output_root / "outcomes.jsonl", records)
    summary = _summary(rows, records)
    _write_json(output_root / "summary.json", summary)
    return summary


def verify_campaign(protocol_path: Path = DEFAULT_PROTOCOL, *, mode: str) -> dict[str, Any]:
    resolved_protocol = Path(protocol_path).resolve()
    protocol = load_protocol(resolved_protocol)
    if mode == "preflight":
        cases = tuple(str(value) for value in protocol["preflight_cases"])
        seeds = tuple(int(value) for value in protocol["preflight_seeds"])
        output_root = (REPOSITORY_ROOT / str(protocol["preflight_output_root"])).resolve()
    elif mode == "full":
        cases = tuple(str(value) for value in protocol["cases"])
        seeds = tuple(int(value) for value in protocol["seeds"])
        output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    else:
        raise ValueError("mode must be preflight or full")
    expected_manifest = _manifest(resolved_protocol, protocol, mode=mode, cases=cases, seeds=seeds)
    if _load_json(output_root / "manifest.json") != expected_manifest:
        raise ValueError("four-arm campaign manifest drifted")
    contexts = _contexts(
        protocol,
        output_root,
        str(expected_manifest["manifest_sha256"]),
        cases,
        seeds,
    )
    rows = [_validate_arm(context.receipt_path, context) for context in contexts]
    records = _records_from_rows(rows)
    expected_summary = _summary(rows, records)
    if _load_json(output_root / "summary.json") != expected_summary:
        raise ValueError("four-arm campaign summary drifted")
    expected_lines = [json.dumps(record.payload(), sort_keys=True, allow_nan=False) for record in records]
    observed_lines = (output_root / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    if observed_lines != expected_lines:
        raise ValueError("four-arm outcome records drifted")
    return expected_summary


def status(protocol_path: Path = DEFAULT_PROTOCOL, *, mode: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    name = "preflight_output_root" if mode == "preflight" else "output_root"
    output_root = (REPOSITORY_ROOT / str(protocol[name])).resolve()
    progress_path = output_root / "progress.json"
    if not progress_path.is_file():
        return {"mode": mode, "state": "not_started", "output_root": str(output_root)}
    return _load_json(progress_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("run-preflight", "verify-preflight", "run", "verify", "status-preflight", "status"),
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run-preflight":
        result = run_campaign(args.protocol, mode="preflight", resume=args.resume, max_workers=args.workers)
    elif args.command == "verify-preflight":
        result = verify_campaign(args.protocol, mode="preflight")
    elif args.command == "run":
        result = run_campaign(args.protocol, mode="full", resume=args.resume, max_workers=args.workers)
    elif args.command == "verify":
        result = verify_campaign(args.protocol, mode="full")
    elif args.command == "status-preflight":
        result = status(args.protocol, mode="preflight")
    else:
        result = status(args.protocol, mode="full")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArmContext",
    "DEFAULT_PROTOCOL",
    "load_protocol",
    "run_campaign",
    "status",
    "verify_campaign",
]
