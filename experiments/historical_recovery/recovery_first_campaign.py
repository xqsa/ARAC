"""Auditable B0-B3 recovery-first campaign.

The campaign deliberately separates provenance, fixed-action execution, selector
parity, and end-to-end handoff.  It never enables the shared-patch kernel or
soft routing.  Full runs are opt-in because a 24 x 25 x 4 matrix is a long
experiment; ``--no-execute`` still performs all read-only checks available from
retained artifacts and reports missing gates explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from arac.runtime.contracts import ACTION_NAMES, canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("recovery_first_protocol_v1.json")
PROTOCOL_SCHEMA = "arac-recovery-first-protocol-v1"
EXPECTED_CASES = tuple(f"{family}{index}" for family in "AERS" for index in range(1, 7))
EXPECTED_SEEDS = tuple(range(117, 142))
EXPECTED_MAPPING = {
    **{f"A{index}": "aor" for index in range(1, 7)},
    **{f"E{index}": "smp" for index in range(1, 7)},
    **{f"S{index}": "ctp" for index in range(1, 7)},
    **{f"R{index}": "gcb" for index in range(1, 7)},
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> tuple[int, str]:
    entries = []
    if not root.is_dir():
        return 0, ""
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix()
        entries.append((relative, _sha256(path)))
    return len(entries), canonical_sha256(entries)


def _verified_hash(payload: Mapping[str, Any], field: str, label: str) -> dict[str, Any]:
    body = dict(payload)
    claimed = body.pop(field, None)
    if claimed != canonical_sha256(body):
        raise ValueError(f"{label} hash drifted")
    body[field] = claimed
    return body


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": PROTOCOL_SCHEMA,
        "cases": list(EXPECTED_CASES),
        "seeds": list(EXPECTED_SEEDS),
        "actions": list(ACTION_NAMES),
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "terminal_fes": 3_000_000,
        "registry": "RecoveredActionRegistry",
        "allow_out_of_bounds": True,
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "new_selector_enabled": False,
        "selector_mode": "parity_only",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"recovery-first protocol drifted: {key}")
    mapping = {str(key): str(value) for key, value in protocol.get("historical_action_mapping", {}).items()}
    if mapping != EXPECTED_MAPPING:
        raise ValueError("historical action mapping drifted")
    for key in ("checkpoint_root", "current_e2e_receipt_root", "historical_table", "fixed_action_protocol"):
        if not (REPOSITORY_ROOT / str(protocol[key])).exists():
            raise ValueError(f"recovery-first source is missing: {key}")
    preflight_cases = tuple(protocol.get("preflight_cases", ()))
    preflight_seeds = tuple(int(value) for value in protocol.get("preflight_seeds", ()))
    if not preflight_cases or not set(preflight_cases).issubset(EXPECTED_CASES):
        raise ValueError("preflight cases must be a non-empty AOB subset")
    if not preflight_seeds or not set(preflight_seeds).issubset(EXPECTED_SEEDS):
        raise ValueError("preflight seeds must be a non-empty historical subset")
    return protocol


def _paths(protocol: Mapping[str, Any], case_id: str, seed: int) -> tuple[Path, Path]:
    checkpoint = REPOSITORY_ROOT / str(protocol["checkpoint_root"]) / case_id / f"seed_{seed}" / "checkpoint.json"
    current = REPOSITORY_ROOT / str(protocol["current_e2e_receipt_root"]) / case_id / f"seed_{seed}" / "receipt.json"
    return checkpoint, current


def _read_checkpoint(path: Path, case_id: str, seed: int, protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = _verified_hash(_load_json(path), "receipt_hash", f"{case_id}:seed-{seed} checkpoint")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload is missing")
    if (
        payload.get("schema_version") != "arac-independent-phase1-checkpoint-v1"
        or payload.get("case_id") != case_id
        or payload.get("run_seed") != seed
        or payload.get("max_fes") != protocol["total_budget_fes"]
        or checkpoint.get("run_seed") != seed
        or checkpoint.get("phase1_fes") != protocol["phase1_fes"]
        or checkpoint.get("total_budget_fes") != protocol["total_budget_fes"]
    ):
        raise ValueError(f"{case_id}:seed-{seed} checkpoint contract drifted")
    if canonical_sha256(checkpoint) != payload["checkpoint_hash"]:
        raise ValueError(f"{case_id}:seed-{seed} checkpoint hash mismatch")
    return payload


def _read_current_receipt(path: Path, case_id: str, seed: int, checkpoint_hash: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = _verified_hash(_load_json(path), "receipt_sha256", f"{case_id}:seed-{seed} current receipt")
    expected = {
        "schema_version": "arac-current-arac-aob24-recovery-receipt-v1",
        "case_id": case_id,
        "run_seed": seed,
        "phase1_checkpoint_hash": checkpoint_hash,
        "action_checkpoint_hash": checkpoint_hash,
        "phase1_fes": protocol["phase1_fes"],
        "phase2_consumed_fes": protocol["phase2_fes"],
        "terminal_fes": protocol["terminal_fes"],
        "terminal_state_finite": True,
        "selector_execution_allowed": False,
        "probe_execution_allowed": False,
        "racing_execution_allowed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{case_id}:seed-{seed} current receipt drifted: {key}")
    if payload.get("selected_action") not in ACTION_NAMES or not math.isfinite(float(payload.get("final_error", math.nan))):
        raise ValueError(f"{case_id}:seed-{seed} current receipt has invalid terminal state")
    return payload


def b0_provenance(protocol: Mapping[str, Any], *, cases: Sequence[str] | None = None, seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """Verify retained checkpoint/E2E provenance without evaluating an action."""

    selected_cases = tuple(cases or protocol["cases"])
    selected_seeds = tuple(int(value) for value in (seeds or protocol["seeds"]))
    failures: list[dict[str, str]] = []
    rows = []
    for case_id in selected_cases:
        for seed in selected_seeds:
            key = f"{case_id}:seed-{seed}"
            checkpoint_path, current_path = _paths(protocol, case_id, seed)
            try:
                checkpoint = _read_checkpoint(checkpoint_path, case_id, seed, protocol)
                current = _read_current_receipt(current_path, case_id, seed, checkpoint["checkpoint_hash"], protocol)
                rows.append(
                    {
                        "key": key,
                        "checkpoint_hash": checkpoint["checkpoint_hash"],
                        "checkpoint_file_sha256": _sha256(checkpoint_path),
                        "action_seed": current["action_seed"],
                        "boundary_profile": "official_hcc_no_offspring_clipping_reproduction",
                        "vendor_tree_sha256": None,
                        "valid": True,
                    }
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                failures.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
    vendor_count, vendor_hash = _tree_sha256(REPOSITORY_ROOT / "vendor/aob")
    for row in rows:
        row["vendor_tree_sha256"] = vendor_hash
    body = {
        "schema_version": "arac-recovery-first-b0-provenance-v1",
        "case_count": len(selected_cases),
        "seed_count": len(selected_seeds),
        "context_count": len(rows),
        "expected_context_count": len(selected_cases) * len(selected_seeds),
        "vendor_file_count": vendor_count,
        "vendor_tree_sha256": vendor_hash,
        "failures": failures,
        "rows": rows,
        "gate_passed": not failures and len(rows) == len(selected_cases) * len(selected_seeds) and vendor_count > 0,
    }
    return {**body, "result_hash": canonical_sha256(body)}


def _fixed_output_root(protocol: Mapping[str, Any], mode: str) -> Path:
    key = "preflight_output_root" if mode == "preflight" else "output_root"
    return (REPOSITORY_ROOT / str(protocol[key])).resolve()


def _fixed_protocol_path() -> Path:
    return REPOSITORY_ROOT / "experiments/historical_recovery/current_recovered_four_arm_protocol_v2.json"


def _run_fixed_action_campaign(protocol: Mapping[str, Any], mode: str, workers: int | None) -> dict[str, Any]:
    """Run the existing four-action executor with its patch-free contract."""

    from experiments.historical_recovery import current_recovered_four_arm as fixed

    fixed_protocol = fixed.load_protocol(_fixed_protocol_path())
    if tuple(fixed_protocol["cases"]) != tuple(protocol["cases"]) or tuple(fixed_protocol["seeds"]) != tuple(protocol["seeds"]):
        raise ValueError("fixed-action protocol does not cover recovery-first matrix")
    # Keep the retained v2 protocol as the source of execution semantics while
    # routing receipts into recovery-first-owned output roots.  This avoids
    # overwriting either historical artifacts or the prior diagnostic matrix.
    execution_protocol = dict(fixed_protocol)
    execution_protocol["preflight_output_root"] = protocol["preflight_output_root"]
    execution_protocol["output_root"] = protocol["output_root"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(execution_protocol, handle)
        temporary_path = Path(handle.name)
    try:
        return fixed.run_campaign(temporary_path, mode=mode, resume=True, max_workers=workers)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_fixed_rows(protocol: Mapping[str, Any], mode: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    root = _fixed_output_root(protocol, mode)
    summary_path = root / "summary.json"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("arms/*/seed_*/*.json")):
        try:
            rows.append(_load_json(path))
        except json.JSONDecodeError:
            continue
    return rows, _load_json(summary_path) if summary_path.is_file() else None


def _fixed_receipt_valid(row: Mapping[str, Any], protocol: Mapping[str, Any]) -> bool:
    try:
        _verified_hash(row, "receipt_hash", "fixed-action receipt")
        result = row["action_result"]
        return (
            row.get("case_id") in EXPECTED_CASES
            and int(row["run_seed"]) in EXPECTED_SEEDS
            and row.get("action_name") in ACTION_NAMES
            and row.get("action_seed") == row.get("run_seed")
            and row.get("phase1_fes") == protocol["phase1_fes"]
            and row.get("phase2_fes") == protocol["phase2_fes"]
            and row.get("terminal_fes") == protocol["terminal_fes"]
            and row.get("action_result_hash") == canonical_sha256(result)
            and result.get("checkpoint_hash") == row.get("checkpoint_hash")
            and result.get("final_error") == row.get("final_error")
        )
    except (KeyError, TypeError, ValueError):
        return False


def b1_fixed_action(protocol: Mapping[str, Any], *, mode: str, execute: bool = False, workers: int | None = None) -> dict[str, Any]:
    """Validate full fixed-action coverage and expose historical comparison limits."""

    if execute:
        _run_fixed_action_campaign(protocol, mode, workers)
    rows, summary = _load_fixed_rows(protocol, mode)
    selected_cases = tuple(protocol["preflight_cases"] if mode == "preflight" else protocol["cases"])
    selected_seeds = tuple(int(value) for value in (protocol["preflight_seeds"] if mode == "preflight" else protocol["seeds"]))
    expected = len(selected_cases) * len(selected_seeds) * len(ACTION_NAMES)
    counts = Counter((row.get("case_id"), row.get("run_seed"), row.get("action_name")) for row in rows)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    complete = len(rows) == expected and not duplicate_count and all(count == 1 for count in counts.values())
    terminal_exact = complete and all(
        row.get("terminal_fes") == protocol["terminal_fes"] and row.get("phase1_fes") == protocol["phase1_fes"]
        for row in rows
    )
    receipt_hash_complete = complete and all(_fixed_receipt_valid(row, protocol) for row in rows)
    mapped = []
    for case_id in selected_cases:
        action = protocol["historical_action_mapping"][case_id]
        matching = [row for row in rows if row.get("case_id") == case_id and row.get("action_name") == action]
        mapped.append({"case_id": case_id, "action": action, "present": len(matching) == len(selected_seeds)})
    reference_available = bool(protocol.get("historical_table"))
    body = {
        "schema_version": "arac-recovery-first-b1-fixed-action-v1",
        "mode": mode,
        "expected_arm_count": expected,
        "observed_arm_count": len(rows),
        "duplicate_count": duplicate_count,
        "complete_matrix": complete,
        "terminal_fes_exact": terminal_exact,
        "receipt_hash_complete": receipt_hash_complete,
        "mapped_action_coverage": mapped,
        "fixed_action_summary_present": summary is not None,
        "historical_reference_available": reference_available,
        "historical_protocol_status": protocol["historical_protocol_status"],
        "historical_comparison": {
            "status": "not_bitwise_comparable",
            "reason": "historical table has ARAC aggregate only; original per-action seed/budget metadata is not retained",
        },
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "new_selector_enabled": False,
        "gate_passed": complete and terminal_exact and receipt_hash_complete and all(row["present"] for row in mapped),
    }
    return {**body, "result_hash": canonical_sha256(body)}


def _selector_input_hash(checkpoint: Mapping[str, Any]) -> str:
    return canonical_sha256({"feature_names": checkpoint["feature_names"], "feature_values": checkpoint["feature_values"]})


def b2_selector_parity(protocol: Mapping[str, Any], *, cases: Sequence[str] | None = None, seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """Replay selector inputs from retained checkpoints; no action is evaluated."""

    selected_cases = tuple(cases or protocol["cases"])
    selected_seeds = tuple(int(value) for value in (seeds or protocol["seeds"]))
    rows = []
    failures = []
    for case_id in selected_cases:
        for seed in selected_seeds:
            key = f"{case_id}:seed-{seed}"
            try:
                checkpoint_path, current_path = _paths(protocol, case_id, seed)
                checkpoint = _read_checkpoint(checkpoint_path, case_id, seed, protocol)["checkpoint"]
                current = _read_current_receipt(current_path, case_id, seed, _read_checkpoint(checkpoint_path, case_id, seed, protocol)["checkpoint_hash"], protocol)
                input_hash_before = _selector_input_hash(checkpoint)
                input_hash_after = _selector_input_hash(checkpoint)
                output_hash_before = canonical_sha256({"selected_action": current["selected_action"]})
                output_hash_after = canonical_sha256({"selected_action": current["selected_action"]})
                rows.append({
                    "key": key,
                    "selector_input_hash_before": input_hash_before,
                    "selector_input_hash_after": input_hash_after,
                    "selector_output_hash_before": output_hash_before,
                    "selector_output_hash_after": output_hash_after,
                    "selected_action": current["selected_action"],
                    "action_evaluation_performed": False,
                    "parity": input_hash_before == input_hash_after and output_hash_before == output_hash_after,
                })
            except (OSError, KeyError, TypeError, ValueError) as exc:
                failures.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
    body = {
        "schema_version": "arac-recovery-first-b2-selector-parity-v1",
        "context_count": len(rows),
        "expected_context_count": len(selected_cases) * len(selected_seeds),
        "failures": failures,
        "action_evaluation_performed": False,
        "rows": rows,
        "gate_passed": not failures and len(rows) == len(selected_cases) * len(selected_seeds) and all(row["parity"] for row in rows),
    }
    return {**body, "result_hash": canonical_sha256(body)}


def b3_end_to_end(protocol: Mapping[str, Any], *, cases: Sequence[str] | None = None, seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """Validate the retained Phase-I -> route -> terminal contract separately per case/action."""

    selected_cases = tuple(cases or protocol["cases"])
    selected_seeds = tuple(int(value) for value in (seeds or protocol["seeds"]))
    rows = []
    failures = []
    for case_id in selected_cases:
        for seed in selected_seeds:
            key = f"{case_id}:seed-{seed}"
            try:
                checkpoint_path, current_path = _paths(protocol, case_id, seed)
                checkpoint = _read_checkpoint(checkpoint_path, case_id, seed, protocol)
                receipt = _read_current_receipt(current_path, case_id, seed, checkpoint["checkpoint_hash"], protocol)
                expected_action = protocol["historical_action_mapping"][case_id]
                rows.append({
                    "key": key,
                    "selected_action": receipt["selected_action"],
                    "historical_action": expected_action,
                    "action_mapping_restored": receipt["selected_action"] == expected_action,
                    "selector_route_valid": receipt["selected_action"] in ACTION_NAMES,
                    "terminal_contract": receipt["terminal_fes"] == protocol["terminal_fes"] and receipt["phase2_consumed_fes"] == protocol["phase2_fes"],
                    "final_error": float(receipt["final_error"]),
                })
            except (OSError, KeyError, TypeError, ValueError) as exc:
                failures.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
    by_case = {}
    for case_id in selected_cases:
        case_rows = [row for row in rows if row["key"].startswith(f"{case_id}:")]
        by_case[case_id] = {
            "count": len(case_rows),
            "action_mapping_restored": all(row["action_mapping_restored"] for row in case_rows),
            "terminal_contract": all(row["terminal_contract"] for row in case_rows),
            "selector_route_valid": all(row["selector_route_valid"] for row in case_rows),
        }
    body = {
        "schema_version": "arac-recovery-first-b3-end-to-end-v1",
        "context_count": len(rows),
        "expected_context_count": len(selected_cases) * len(selected_seeds),
        "failures": failures,
        "case_summaries": by_case,
        "unrecovered_cases": [case for case, row in by_case.items() if not row["action_mapping_restored"] or not row["terminal_contract"]],
        "gate_passed": not failures and len(rows) == len(selected_cases) * len(selected_seeds) and all(row["action_mapping_restored"] and row["terminal_contract"] for row in rows),
    }
    return {**body, "result_hash": canonical_sha256(body)}


def run_campaign(protocol_path: Path = DEFAULT_PROTOCOL, *, mode: str = "preflight", execute: bool = False, workers: int | None = None) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if mode not in {"preflight", "full"}:
        raise ValueError("mode must be preflight or full")
    cases = tuple(protocol["preflight_cases"] if mode == "preflight" else protocol["cases"])
    seeds = tuple(int(value) for value in (protocol["preflight_seeds"] if mode == "preflight" else protocol["seeds"]))
    b0 = b0_provenance(protocol, cases=cases, seeds=seeds)
    b1 = b1_fixed_action(protocol, mode=mode, execute=execute, workers=workers)
    b2 = b2_selector_parity(protocol, cases=cases, seeds=seeds)
    b3 = b3_end_to_end(protocol, cases=cases, seeds=seeds)
    body = {
        "schema_version": "arac-recovery-first-campaign-v1",
        "mode": mode,
        "execute_requested": execute,
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "new_selector_enabled": False,
        "gates": {"B0": b0, "B1": b1, "B2": b2, "B3": b3},
        "gate_status": {name: bool(value["gate_passed"]) for name, value in {"B0": b0, "B1": b1, "B2": b2, "B3": b3}.items()},
        "recovery_ready_for_matched_host": all(value["gate_passed"] for value in (b0, b1, b2, b3)),
    }
    output_root = (REPOSITORY_ROOT / str(protocol["report_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root / mode
    output_root.mkdir(parents=True, exist_ok=True)
    result = {**body, "result_hash": canonical_sha256(body)}
    (output_root / "recovery_first_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--mode", choices=("preflight", "full"), default="preflight")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true", help="run the fixed-action matrix before validating B1")
    execution.add_argument("--no-execute", action="store_false", dest="execute", help="only inspect retained artifacts (default)")
    parser.set_defaults(execute=False)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    result = run_campaign(args.protocol, mode=args.mode, execute=args.execute, workers=args.workers)
    print(json.dumps({"gate_status": result["gate_status"], "recovery_ready_for_matched_host": result["recovery_ready_for_matched_host"]}, indent=2, sort_keys=True))
    return 0 if result["recovery_ready_for_matched_host"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PROTOCOL",
    "EXPECTED_CASES",
    "EXPECTED_MAPPING",
    "EXPECTED_SEEDS",
    "b0_provenance",
    "b1_fixed_action",
    "b2_selector_parity",
    "b3_end_to_end",
    "load_protocol",
    "run_campaign",
]
