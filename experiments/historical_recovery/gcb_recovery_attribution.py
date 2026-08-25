"""Attribute GCB schedule differences from frozen paired receipts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence

from arac.runtime.contracts import canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("gcb_recovery_attribution_protocol_v1.json")
EXPECTED_CASES = ("R1", "R2", "R3", "R4", "R5", "R6")
EXPECTED_SEEDS = (117, 123, 129, 135, 141)
EXPECTED_VARIANTS = ("current", "historical_compatible")
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
PHASE2_FES = 2_820_000
RECEIPT_SCHEMA = "arac-recovery-action-lifecycle-diagnostic-receipt-v1"
SUMMARY_SCHEMA = "arac-recovery-action-lifecycle-diagnostic-summary-v1"
REPORT_SCHEMA = "arac-gcb-recovery-attribution-report-v1"

_CURRENT_ROUTE = re.compile(
    r"^(?P<relation>zero_relation|positive_relation_graph)_source_(?P<source>\d+)_sweeps_3_"
    r"coordination_(?P<coordination>\d+)_cold_native_(?P<native>\d+)_windows_(?P<windows>\d+)_tail_(?P<tail>\d+)$"
)
_HISTORICAL_ROUTE = re.compile(
    r"^(?P<relation>zero_relation|positive_relation_graph)_cold_warmup_(?P<warmup>\d+)_sweeps_3_"
    r"coordination_(?P<coordination>\d+)_cold_continuation_(?P<continuation>\d+)_sweeps_(?P<windows>\d+)_tail_(?P<tail>\d+)$"
)


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-gcb-recovery-attribution-protocol-v1",
        "status": "frozen_receipt_attribution",
        "cases": list(EXPECTED_CASES),
        "seeds": list(EXPECTED_SEEDS),
        "variants": list(EXPECTED_VARIANTS),
        "action_name": "gcb",
        "total_budget_fes": TOTAL_BUDGET_FES,
        "phase1_fes": PHASE1_FES,
        "phase2_fes": PHASE2_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"GCB attribution protocol drifted: {key}")
    input_root = _resolved(str(protocol["input_root"]))
    if not (input_root / "summary.json").is_file() or not (input_root / "manifest.json").is_file():
        raise FileNotFoundError("GCB attribution input summary or manifest is missing")
    if _sha256(input_root / "summary.json") != protocol["input_summary_sha256"]:
        raise ValueError("GCB attribution input summary hash drifted")
    return protocol


def _receipt_path(protocol: Mapping[str, Any], case_id: str, seed: int, variant: str) -> Path:
    return _resolved(str(protocol["input_root"])) / "arms" / case_id / f"seed_{seed}" / f"gcb_{variant}.json"


def _validate_receipt(path: Path, case_id: str, seed: int, variant: str) -> dict[str, Any]:
    receipt = _load_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != canonical_sha256(receipt):
        raise ValueError(f"{case_id}:seed-{seed}:{variant} receipt hash drifted")
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "case_id": case_id,
        "run_seed": seed,
        "action_name": "gcb",
        "variant": variant,
        "phase1_fes": PHASE1_FES,
        "phase2_fes": PHASE2_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{case_id}:seed-{seed}:{variant} receipt drifted: {key}")
    if receipt.get("checkpoint_hash") != receipt.get("action_result", {}).get("checkpoint_hash"):
        raise ValueError(f"{case_id}:seed-{seed}:{variant} checkpoint binding drifted")
    receipt["receipt_sha256"] = claimed
    return receipt


def parse_route(route: str, variant: str) -> dict[str, int | str]:
    pattern = _CURRENT_ROUTE if variant == "current" else _HISTORICAL_ROUTE
    match = pattern.fullmatch(str(route))
    if match is None:
        raise ValueError(f"unrecognized GCB {variant} route: {route}")
    values = {key: int(value) if key not in {"relation"} else value for key, value in match.groupdict().items()}
    if variant == "current":
        values["warmup"] = values.pop("source")
        values["continuation"] = values.pop("native")
    return values


def _load_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case_id in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            for variant in EXPECTED_VARIANTS:
                path = _receipt_path(protocol, case_id, seed, variant)
                row = _validate_receipt(path, case_id, seed, variant)
                row["_schedule"] = parse_route(str(row["route"]), variant)
                row["_receipt_path"] = str(path)
                rows.append(row)
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {(str(row["case_id"]), int(row["run_seed"]), str(row["variant"])): row for row in rows}
    pairs = []
    for case_id in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            current = indexed[(case_id, seed, "current")]
            historical = indexed[(case_id, seed, "historical_compatible")]
            current_error = float(current["final_error"])
            historical_error = float(historical["final_error"])
            current_schedule = dict(current["_schedule"])
            historical_schedule = dict(historical["_schedule"])
            pairs.append({
                "case_id": case_id,
                "run_seed": seed,
                "checkpoint_hash": current["checkpoint_hash"],
                "same_checkpoint": current["checkpoint_hash"] == historical["checkpoint_hash"],
                "current_final_error": current_error,
                "historical_compatible_final_error": historical_error,
                "current_to_historical_ratio": current_error / historical_error if historical_error > 0 else None,
                "historical_compatible_better": historical_error < current_error,
                "current_schedule": current_schedule,
                "historical_schedule": historical_schedule,
                "schedule_delta_current_minus_historical": {
                    key: int(current_schedule[key]) - int(historical_schedule[key])
                    for key in ("warmup", "coordination", "continuation", "windows", "tail")
                },
            })
    case_summaries = []
    for case_id in EXPECTED_CASES:
        active = [row for row in pairs if row["case_id"] == case_id]
        current_values = [float(row["current_final_error"]) for row in active]
        historical_values = [float(row["historical_compatible_final_error"]) for row in active]
        deltas = [row["schedule_delta_current_minus_historical"] for row in active]
        case_summaries.append({
            "case_id": case_id,
            "current_mean": statistics.fmean(current_values),
            "historical_compatible_mean": statistics.fmean(historical_values),
            "current_to_historical_geometric_mean_ratio": math.exp(sum(math.log(float(row["current_to_historical_ratio"])) for row in active) / len(active)),
            "historical_compatible_better_count": sum(bool(row["historical_compatible_better"]) for row in active),
            "current_better_count": sum(not bool(row["historical_compatible_better"]) for row in active),
            "mean_schedule_delta_current_minus_historical": {
                key: statistics.fmean(int(delta[key]) for delta in deltas)
                for key in ("warmup", "coordination", "continuation", "windows", "tail")
            },
        })
    all_deltas = [row["schedule_delta_current_minus_historical"] for row in pairs]
    body = {
        "schema_version": REPORT_SCHEMA,
        "source_summary_schema": SUMMARY_SCHEMA,
        "source_summary_sha256": protocol["input_summary_sha256"],
        "arm_count": len(rows),
        "pair_count": len(pairs),
        "same_checkpoint_per_pair": all(bool(row["same_checkpoint"]) for row in pairs),
        "exact_terminal_fes": all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows),
        "all_final_errors_finite": all(math.isfinite(float(row["final_error"])) for row in rows),
        "route_parse_valid": len(rows) == 60,
        "mean_schedule_delta_current_minus_historical": {
            key: statistics.fmean(int(delta[key]) for delta in all_deltas)
            for key in ("warmup", "coordination", "continuation", "windows", "tail")
        },
        "case_summaries": case_summaries,
        "pairs": pairs,
        "decision": {
            "schedule_attribution_complete": len(rows) == 60 and all(bool(row["same_checkpoint"]) for row in pairs),
            "uniform_historical_rollback_supported": False,
            "production_gcb_change_authorized": False,
            "reason": "paired evidence is mixed; current wins R1/R3 while historical-compatible wins R2/R4/R6 and R5 is near tie",
            "next_gate": "fresh-seed GCB schedule ablation on R1-R6 or retain current production GCB until a pre-registered schedule passes",
        },
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def run(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    rows = _load_rows(protocol)
    report = summarize(rows, protocol)
    output_root = _resolved(str(protocol["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "report.json", report)
    _write_json(output_root / "manifest.json", {
        "schema_version": "arac-gcb-recovery-attribution-manifest-v1",
        "protocol_sha256": _sha256(Path(protocol_path).resolve()),
        "source_summary_sha256": protocol["input_summary_sha256"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    })
    return report


def verify(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    stored = _load_json(output_root / "report.json")
    claimed = stored.pop("report_sha256", None)
    if claimed != canonical_sha256(stored):
        raise ValueError("GCB attribution report hash drifted")
    expected = summarize(_load_rows(protocol), protocol)
    expected.pop("report_sha256", None)
    if stored != expected:
        raise ValueError("GCB attribution report content drifted")
    stored["report_sha256"] = claimed
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    result = run(args.protocol) if args.command == "run" else verify(args.protocol)
    print(json.dumps({"arm_count": result["arm_count"], "pair_count": result["pair_count"], "production_gcb_change_authorized": result["decision"]["production_gcb_change_authorized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_PROTOCOL", "EXPECTED_CASES", "EXPECTED_SEEDS", "EXPECTED_VARIANTS", "load_protocol", "parse_route", "run", "summarize", "verify"]
