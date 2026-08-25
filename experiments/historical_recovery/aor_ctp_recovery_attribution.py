"""Attribute AOR A4/A6 and CTP S6 from frozen recovery evidence."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from arac.runtime.contracts import canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("aor_ctp_recovery_attribution_protocol_v1.json")
SCREEN_CASES = ("A4", "A6", "S6")
AOR_CASES = ("A4", "A6")
SCREEN_SEEDS = (117, 123, 129, 135, 141)
CTP_ABLATION_SEEDS = (31_001, 31_002, 31_003)
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
PHASE2_FES = 2_820_000


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
        "schema_version": "arac-aor-ctp-recovery-attribution-protocol-v1",
        "status": "frozen_receipt_and_ablation_attribution",
        "aor_cases": list(AOR_CASES),
        "ctp_cases": ["S6"],
        "screen_seeds": list(SCREEN_SEEDS),
        "ctp_ablation_seeds": list(CTP_ABLATION_SEEDS),
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
            raise ValueError(f"AOR/CTP attribution protocol drifted: {key}")
    required_paths = (
        "screen_root", "ctp_ablation_root", "historical_table",
        "aor_current_source", "aor_historical_source", "ctp_current_source", "ctp_historical_source",
    )
    for key in required_paths:
        if not _resolved(str(protocol[key])).exists():
            raise FileNotFoundError(f"AOR/CTP attribution source is missing: {key}")
    return protocol


def _screen_receipt(protocol: Mapping[str, Any], case_id: str, seed: int) -> dict[str, Any]:
    root = _resolved(str(protocol["screen_root"]))
    path = root / "arms" / case_id / f"seed_{seed}" / ("aor.json" if case_id in AOR_CASES else "ctp.json")
    receipt = _load_json(path)
    claimed = receipt.pop("receipt_hash", None)
    if claimed != canonical_sha256(receipt):
        raise ValueError(f"screen receipt hash drifted: {case_id}:{seed}")
    receipt["receipt_hash"] = claimed
    if receipt.get("case_id") != case_id or receipt.get("run_seed") != seed or receipt.get("terminal_fes") != TOTAL_BUDGET_FES:
        raise ValueError(f"screen receipt identity drifted: {case_id}:{seed}")
    return receipt


def _ctp_ablation_summary(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolved(str(protocol["ctp_ablation_root"])) / "summary.json"
    summary = _load_json(path)
    claimed = summary.get("summary_sha256")
    body = dict(summary)
    body.pop("summary_sha256", None)
    if claimed != canonical_sha256(body):
        raise ValueError("CTP tail ablation summary hash drifted")
    if summary.get("candidate_reserved_tail_executed") is not True or summary.get("same_checkpoint_per_pair") is not True:
        raise ValueError("CTP tail ablation lacks matched-checkpoint evidence")
    return summary


def _table_targets(protocol: Mapping[str, Any]) -> dict[str, float]:
    import csv

    path = _resolved(str(protocol["historical_table"]))
    targets: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row["case"]).strip()
            if case_id in SCREEN_CASES:
                raw = str(row[str(protocol["historical_target_column"])])
                targets[case_id] = float(raw.split("+/-")[0].replace(",", "").strip())
    if set(targets) != set(SCREEN_CASES):
        raise ValueError("historical table does not cover A4/A6/S6")
    return targets


def summarize(protocol: Mapping[str, Any]) -> dict[str, Any]:
    source_hashes = {
        key: _sha256(_resolved(str(protocol[key])))
        for key in ("aor_current_source", "aor_historical_source", "ctp_current_source", "ctp_historical_source")
    }
    aor_rows = []
    for case_id in AOR_CASES:
        rows = [_screen_receipt(protocol, case_id, seed) for seed in SCREEN_SEEDS]
        values = [float(row["final_error"]) for row in rows]
        target = _table_targets(protocol)[case_id]
        aor_rows.append({
            "case_id": case_id,
            "seed_count": len(values),
            "screen_mean": statistics.fmean(values),
            "screen_sample_std": statistics.stdev(values),
            "historical_target_mean": target,
            "screen_to_historical_target_ratio": statistics.fmean(values) / target,
            "screen_displayed_mean_not_higher": float(format(statistics.fmean(values), ".2E")) <= float(format(target, ".2E")),
            "route_set": sorted({str(row["route"]) for row in rows}),
            "action_result_hashes_valid": all(row["action_result_hash"] == canonical_sha256(row["action_result"]) for row in rows),
        })
    ctp_screen = [_screen_receipt(protocol, "S6", seed) for seed in SCREEN_SEEDS]
    ctp_target = _table_targets(protocol)["S6"]
    tail_summary = _ctp_ablation_summary(protocol)
    tail_case = next(item for item in tail_summary["case_summaries"] if item["case_id"] == "S6")
    ctp_rows = {
        "case_id": "S6",
        "seed_count": len(ctp_screen),
        "screen_mean": statistics.fmean(float(row["final_error"]) for row in ctp_screen),
        "screen_sample_std": statistics.stdev(float(row["final_error"]) for row in ctp_screen),
        "historical_target_mean": ctp_target,
        "screen_to_historical_target_ratio": statistics.fmean(float(row["final_error"]) for row in ctp_screen) / ctp_target,
        "screen_displayed_mean_not_higher": float(format(statistics.fmean(float(row["final_error"]) for row in ctp_screen), ".2E")) <= float(format(ctp_target, ".2E")),
        "screen_route_set": sorted({str(row["route"]) for row in ctp_screen}),
        "screen_route_has_reserved_tail": all("mmes_tail_" in str(row["route"]) for row in ctp_screen),
        "matched_tail_ablation_seed_count": int(tail_case["pair_count"]),
        "matched_tail_ablation_geometric_ratio": float(tail_case["candidate_to_baseline_geometric_mean_ratio"]),
        "matched_tail_ablation_candidate_win_count": int(tail_case["candidate_win_or_tie_count"]),
        "matched_tail_ablation_same_checkpoint": bool(tail_summary["same_checkpoint_per_pair"]),
        "matched_tail_ablation_candidate_tail_fes": [int(pair["candidate_tail_fes"]) for pair in tail_summary["pairs"] if pair["case_id"] == "S6"],
        "matched_tail_ablation_baseline_tail_fes": [int(pair["baseline_tail_fes"]) for pair in tail_summary["pairs"] if pair["case_id"] == "S6"],
        "action_result_hashes_valid": all(row["action_result_hash"] == canonical_sha256(row["action_result"]) for row in ctp_screen),
    }
    body = {
        "schema_version": "arac-aor-ctp-recovery-attribution-report-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes,
        "aor_source_identity_exact": source_hashes["aor_current_source"] == source_hashes["aor_historical_source"],
        "ctp_source_differs": source_hashes["ctp_current_source"] != source_hashes["ctp_historical_source"],
        "aor_case_summaries": aor_rows,
        "ctp_case_summary": ctp_rows,
        "decision": {
            "aor_lifecycle_attribution_complete": True,
            "aor_code_change_authorized": False,
            "aor_interpretation": "current and historical AOR sources are byte-identical; A4/A6 residual is not an AOR lifecycle delta",
            "ctp_tail_mechanism_matched_evidence": ctp_rows["matched_tail_ablation_same_checkpoint"] and ctp_rows["matched_tail_ablation_geometric_ratio"] < 1.0,
            "ctp_production_change_authorized": False,
            "ctp_interpretation": "S6 tail ablation is positive on matched 31001-31003 checkpoints, but it is not paired to screen seeds 117-141; do not claim screen recovery",
            "next_gate": "fresh matched S6 screen-seed tail attribution or retain current CTP while AOR/GCB remain unresolved",
        },
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
        "reference_thresholds_used_for_decision": False,
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def run(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    report = summarize(protocol)
    output_root = _resolved(str(protocol["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "report.json", report)
    return report


def verify(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    stored = _load_json(output_root / "report.json")
    claimed = stored.pop("report_sha256", None)
    if claimed != canonical_sha256(stored):
        raise ValueError("AOR/CTP attribution report hash drifted")
    expected = summarize(protocol)
    expected.pop("report_sha256", None)

    # The timestamp documents when the report was materialized, but it is not
    # experimental content. Exclude it from the semantic comparison so a
    # deterministic re-summary can verify an existing report on later runs.
    stored_content = dict(stored)
    expected_content = dict(expected)
    stored_content.pop("generated_at_utc", None)
    expected_content.pop("generated_at_utc", None)
    if stored_content != expected_content:
        raise ValueError("AOR/CTP attribution report content drifted")
    stored["report_sha256"] = claimed
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    result = run(args.protocol) if args.command == "run" else verify(args.protocol)
    print(json.dumps({"aor_source_identity_exact": result["aor_source_identity_exact"], "ctp_matched_tail_ratio": result["ctp_case_summary"]["matched_tail_ablation_geometric_ratio"], "aor_code_change_authorized": result["decision"]["aor_code_change_authorized"], "ctp_production_change_authorized": result["decision"]["ctp_production_change_authorized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_PROTOCOL", "load_protocol", "run", "summarize", "verify"]
