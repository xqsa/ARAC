"""Run the 24 x 5 historical mapped-action recovery screen (120 arms)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

from arac.runtime.contracts import ACTION_NAMES, canonical_sha256
from experiments.audit_historical_recovery import parse_target
from experiments.historical_recovery import current_recovered_four_arm as fixed


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("recovery_first_screen_protocol_v1.json")
EXPECTED_CASES = tuple(f"{family}{index}" for family in "AERS" for index in range(1, 7))
EXPECTED_SEEDS = (117, 123, 129, 135, 141)
EXPECTED_MAPPING = {
    **{f"A{index}": "aor" for index in range(1, 7)},
    **{f"E{index}": "smp" for index in range(1, 7)},
    **{f"R{index}": "gcb" for index in range(1, 7)},
    **{f"S{index}": "ctp" for index in range(1, 7)},
}
MAX_WORKERS = 24


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> tuple[int, str]:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            entries.append((path.relative_to(root).as_posix(), _sha256(path)))
    return len(entries), canonical_sha256(entries)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-recovery-first-screen-protocol-v1",
        "cases": list(EXPECTED_CASES),
        "screen_seeds": list(EXPECTED_SEEDS),
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "terminal_fes": 3_000_000,
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
        "aggregate_precision": ".2E",
        "max_workers": MAX_WORKERS,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"screen protocol drifted: {key}")
    mapping = {str(key): str(value) for key, value in protocol.get("historical_action_mapping", {}).items()}
    if mapping != EXPECTED_MAPPING or set(mapping.values()) != set(ACTION_NAMES):
        raise ValueError("screen action mapping drifted")
    for key in ("checkpoint_root", "current_e2e_receipt_root", "historical_table"):
        if not (REPOSITORY_ROOT / str(protocol[key])).exists():
            raise ValueError(f"screen source is missing: {key}")
    return protocol


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_root = REPOSITORY_ROOT / str(protocol["checkpoint_root"])
    current_root = REPOSITORY_ROOT / str(protocol["current_e2e_receipt_root"])
    checkpoint_count, checkpoint_tree = _tree_sha256(checkpoint_root)
    current_count, current_tree = _tree_sha256(current_root)
    source_paths = {
        "screen_protocol": protocol_path,
        "screen_campaign": Path(__file__).resolve(),
        "fixed_campaign": Path(fixed.__file__).resolve(),
    }
    body = {
        "schema_version": "arac-recovery-first-screen-manifest-v1",
        "protocol_sha256": _sha256(protocol_path),
        "cases": list(protocol["cases"]),
        "screen_seeds": list(protocol["screen_seeds"]),
        "mapping": dict(protocol["historical_action_mapping"]),
        "source_sha256": {name: _sha256(path) for name, path in sorted(source_paths.items())},
        "checkpoint_tree": {"root": str(protocol["checkpoint_root"]), "file_count": checkpoint_count, "sha256": checkpoint_tree},
        "current_receipt_tree": {"root": str(protocol["current_e2e_receipt_root"]), "file_count": current_count, "sha256": current_tree},
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _contexts(protocol: Mapping[str, Any], output_root: Path, manifest_sha256: str) -> tuple[fixed.ArmContext, ...]:
    checkpoint_root = (REPOSITORY_ROOT / str(protocol["checkpoint_root"])).resolve()
    current_root = (REPOSITORY_ROOT / str(protocol["current_e2e_receipt_root"])).resolve()
    return tuple(
        fixed.ArmContext(
            case_id=case_id,
            run_seed=int(seed),
            action_name=str(protocol["historical_action_mapping"][case_id]),
            checkpoint_root=checkpoint_root,
            current_receipt_root=current_root,
            output_root=output_root,
            manifest_sha256=manifest_sha256,
        )
        for seed in protocol["screen_seeds"]
        for case_id in protocol["cases"]
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _prepare_output(output_root: Path, manifest: Mapping[str, Any], *, resume: bool) -> None:
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"screen output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path) != manifest:
            existing_receipts = tuple(output_root.glob("arms/*/seed_*/*.json"))
            if existing_receipts:
                raise ValueError("screen manifest does not match frozen protocol")
            _write_json(manifest_path, manifest)
        return
    output_root.mkdir(parents=True)
    _write_json(manifest_path, manifest)


def _validate_arm(path: Path, context: fixed.ArmContext) -> dict[str, Any]:
    return fixed._validate_arm(path, context)


def _historical_targets(protocol: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    targets = {}
    path = REPOSITORY_ROOT / str(protocol["historical_table"])
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in __import__("csv").DictReader(handle):
            case_id = str(row["case"]).strip()
            if case_id in EXPECTED_CASES:
                targets[case_id] = parse_target(row[str(protocol["historical_target_column"])])
    if set(targets) != set(EXPECTED_CASES):
        raise ValueError("historical target table does not cover AOB-24")
    return targets


def summarize(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    targets = _historical_targets(protocol)
    case_summaries = []
    for case_id in protocol["cases"]:
        expected_action = protocol["historical_action_mapping"][case_id]
        case_rows = [row for row in rows if row["case_id"] == case_id and row["action_name"] == expected_action]
        values = [float(row["final_error"]) for row in case_rows]
        if len(values) != len(protocol["screen_seeds"]):
            raise ValueError(f"screen coverage incomplete: {case_id}")
        mean = fmean(values)
        sample_std = stdev(values)
        historical_mean, historical_std = targets[case_id]
        precision = str(protocol["aggregate_precision"])
        case_summaries.append(
            {
                "case_id": case_id,
                "action": expected_action,
                "seed_count": len(values),
                "mean": mean,
                "sample_std": sample_std,
                "historical_mean": historical_mean,
                "historical_sample_std": historical_std,
                "formatted_mean": format(mean, precision).upper(),
                "formatted_historical_mean": format(historical_mean, precision).upper(),
                "displayed_mean_not_higher": float(format(mean, precision)) <= float(format(historical_mean, precision)),
            }
        )
    body = {
        "schema_version": "arac-recovery-first-screen-summary-v1",
        "context_count": len(rows),
        "expected_context_count": len(protocol["cases"]) * len(protocol["screen_seeds"]),
        "all_terminal_fes_exact": all(row["terminal_fes"] == protocol["terminal_fes"] for row in rows),
        "all_checkpoint_bindings_exact": all(row["checkpoint_hash"] == row["action_result"]["checkpoint_hash"] for row in rows),
        "all_receipt_hashes_valid": all(_validate_arm(Path(row["_receipt_path"]), row["_context"]) is not None for row in rows),
        "case_summaries": case_summaries,
        "screen_rule": protocol["screen_rule"],
    }
    body["screen_gate_passed"] = (
        body["context_count"] == body["expected_context_count"]
        and body["all_terminal_fes_exact"]
        and body["all_checkpoint_bindings_exact"]
        and body["all_receipt_hashes_valid"]
        and all(row["displayed_mean_not_higher"] for row in case_summaries)
    )
    body["final_recovery_claim_authorized"] = False
    body["result_hash"] = canonical_sha256(body)
    return body


def run_campaign(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False, workers: int | None = None) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = _manifest(resolved, protocol)
    _prepare_output(output_root, manifest, resume=resume)
    contexts = _contexts(protocol, output_root, str(manifest["manifest_sha256"]))
    rows: list[dict[str, Any]] = []
    pending = []
    for context in contexts:
        if resume and context.receipt_path.is_file():
            rows.append(_validate_arm(context.receipt_path, context))
        else:
            pending.append(context)
    workers_value = int(protocol["max_workers"] if workers is None else workers)
    if workers_value <= 0:
        raise ValueError("workers must be positive")
    _write_json(output_root / "progress.json", {"total": len(contexts), "completed": len(rows), "pending": len(pending), "updated_at_utc": datetime.now(UTC).isoformat()})
    failures = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers_value) as executor:
            futures = {executor.submit(fixed._run_arm, context): context for context in pending}
            for future in as_completed(futures):
                context = futures[future]
                try:
                    future.result()
                    rows.append(_validate_arm(context.receipt_path, context))
                except BaseException as exc:
                    failures.append({"key": context.key, "error": f"{type(exc).__name__}: {exc}"})
                _write_json(output_root / "progress.json", {"total": len(contexts), "completed": len(rows), "failed": len(failures), "pending": len(contexts) - len(rows) - len(failures), "updated_at_utc": datetime.now(UTC).isoformat()})
    if failures:
        _write_json(output_root / "failure_summary.json", {"failures": failures})
        raise RuntimeError(f"screen campaign has {len(failures)} failed arms")
    # Private audit fields are used only to re-check receipt hashes, then removed.
    contexts_by_key = {(context.case_id, context.run_seed): context for context in contexts}
    audited_rows = []
    for row in rows:
        context = contexts_by_key[(str(row["case_id"]), int(row["run_seed"]))]
        audited_rows.append({**row, "_receipt_path": str(context.receipt_path), "_context": context})
    summary = summarize(audited_rows, protocol)
    _write_json(output_root / "summary.json", summary)
    return summary


def recompute_summary(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    """Recompute only aggregate comparisons from completed, frozen receipts."""

    protocol = load_protocol(protocol_path)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = _load_json(output_root / "manifest.json")
    contexts = _contexts(protocol, output_root, str(manifest["manifest_sha256"]))
    if any(not context.receipt_path.is_file() for context in contexts):
        raise RuntimeError("cannot recompute screen summary before all 120 receipts exist")
    rows = []
    for context in contexts:
        row = _validate_arm(context.receipt_path, context)
        rows.append({**row, "_receipt_path": str(context.receipt_path), "_context": context})
    summary = summarize(rows, protocol)
    summary["summary_recomputed_after_code_only_fix"] = True
    summary["result_hash"] = canonical_sha256({key: value for key, value in summary.items() if key != "result_hash"})
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recompute-summary", action="store_true")
    args = parser.parse_args(argv)
    result = recompute_summary(args.protocol) if args.recompute_summary else run_campaign(args.protocol, resume=args.resume, workers=args.workers)
    print(json.dumps({"context_count": result["context_count"], "screen_gate_passed": result["screen_gate_passed"], "final_recovery_claim_authorized": result["final_recovery_claim_authorized"]}, indent=2, sort_keys=True))
    return 0 if result["screen_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_PROTOCOL", "EXPECTED_CASES", "EXPECTED_MAPPING", "EXPECTED_SEEDS", "load_protocol", "recompute_summary", "run_campaign", "summarize"]
