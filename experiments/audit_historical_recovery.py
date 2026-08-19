"""Audit fixed expert actions against the historical ARAC result table."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import fmean, stdev
from typing import Any

from arac.runtime.contracts import canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "config.json"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "experiments" / "historical_recovery" / "recovery_audit.json"
)
DEFAULT_REPORT = (
    REPOSITORY_ROOT / "experiments" / "historical_recovery" / "recovery_audit.md"
)
TARGET_PATTERN = re.compile(
    r"^\s*(?P<mean>[0-9.]+E[+-][0-9]+)\s+\+/-\s+"
    r"(?P<std>[0-9.]+E[+-][0-9]+)\s*$",
    re.IGNORECASE,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def parse_target(value: str) -> tuple[float, float]:
    """Parse one historical `mean +/- sample std` table cell."""

    match = TARGET_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid historical target: {value!r}")
    return float(match.group("mean")), float(match.group("std"))


def _reference_rows(root: Path, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    reference = config["historical_reference"]
    path = root / reference["table_csv"]
    target_column = reference["target_column"]
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            case_id = row["case"].strip()
            if not re.fullmatch(r"[AESR][1-6]", case_id):
                continue
            mean, sample_std = parse_target(row[target_column])
            rows[case_id] = {"mean": mean, "sample_std": sample_std}
    if len(rows) != config["gate"]["required_case_count"]:
        raise ValueError("historical table does not contain all required cases")
    return rows


def _result_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"case", "seed", "final_error", "fitness_evaluations"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid fixed-expert result CSV: {path}")
    return rows


def _format(value: float, precision: str) -> str:
    return format(value, precision).upper()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_action(
    *,
    root: Path,
    action: str,
    evidence: dict[str, Any],
    targets: dict[str, dict[str, float]],
    seeds: set[int],
    max_fes: int,
    precision: str,
) -> list[dict[str, Any]]:
    expected_cases = list(evidence["cases"])
    result_path = evidence.get("results_csv")
    if result_path is None:
        return [
            {
                "case": case_id,
                "action": action,
                "status": "missing",
                "reason": "complete_25_seed_result_absent",
                "target_mean": targets[case_id]["mean"],
                "target_sample_std": targets[case_id]["sample_std"],
            }
            for case_id in expected_cases
        ]

    rows = _result_rows(root / result_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["case"].strip()].append(row)

    audited: list[dict[str, Any]] = []
    for case_id in expected_cases:
        case_rows = grouped.get(case_id, [])
        observed_seeds = {int(row["seed"]) for row in case_rows}
        exact_fes = all(int(row["fitness_evaluations"]) == max_fes for row in case_rows)
        if observed_seeds != seeds or not exact_fes:
            audited.append(
                {
                    "case": case_id,
                    "action": action,
                    "status": "missing",
                    "reason": "incomplete_seed_or_fe_coverage",
                    "observed_seed_count": len(observed_seeds),
                    "exact_terminal_fes": exact_fes,
                    "target_mean": targets[case_id]["mean"],
                    "target_sample_std": targets[case_id]["sample_std"],
                }
            )
            continue

        values = [float(row["final_error"]) for row in case_rows]
        mean = fmean(values)
        sample_std = stdev(values)
        target = targets[case_id]
        mean_matches = _format(mean, precision) == _format(target["mean"], precision)
        std_matches = _format(sample_std, precision) == _format(
            target["sample_std"], precision
        )
        audited.append(
            {
                "case": case_id,
                "action": action,
                "status": "recovered" if mean_matches and std_matches else "failed",
                "reason": "rounded_aggregate_match"
                if mean_matches and std_matches
                else "historical_aggregate_mismatch",
                "seed_count": len(observed_seeds),
                "exact_terminal_fes": exact_fes,
                "mean": mean,
                "sample_std": sample_std,
                "target_mean": target["mean"],
                "target_sample_std": target["sample_std"],
                "formatted_mean": _format(mean, precision),
                "formatted_sample_std": _format(sample_std, precision),
                "formatted_target_mean": _format(target["mean"], precision),
                "formatted_target_sample_std": _format(target["sample_std"], precision),
            }
        )
    return audited


def _campaign_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "case_id",
        "run_seed",
        "action_name",
        "terminal_fes",
        "final_error",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid fixed-expert campaign results CSV: {path}")
    return rows


def _audit_current_fixed_expert(
    *,
    root: Path,
    config: dict[str, Any],
    targets: dict[str, dict[str, float]],
    precision: str,
) -> dict[str, Any]:
    campaign = config.get("current_fixed_expert")
    if not campaign:
        return {
            "status": "not_configured",
            "gate_passed": False,
            "context_count": 0,
            "case_count": 0,
            "counts": {"recovered": 0, "failed": 0, "missing": 0},
            "cases": [],
        }

    results_path = root / campaign["results_csv"]
    summary_path = root / campaign["summary"]
    manifest_path = root / campaign["campaign_manifest"]
    if not results_path.is_file() or not summary_path.is_file() or not manifest_path.is_file():
        return {
            "status": "not_run",
            "gate_passed": False,
            "context_count": 0,
            "case_count": 0,
            "counts": {"recovered": 0, "failed": 0, "missing": len(targets)},
            "results_csv": campaign["results_csv"],
            "summary": campaign["summary"],
            "campaign_manifest": campaign["campaign_manifest"],
            "cases": [],
        }

    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)
    rows = _campaign_result_rows(results_path)
    protocol = config["candidate_protocol"]
    expected_cases = set(targets)
    expected_seeds = {int(seed) for seed in protocol["seeds"]}
    mapping = config["expert_mapping"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"].strip()].append(row)

    audited: list[dict[str, Any]] = []
    for case_id in sorted(expected_cases):
        case_rows = grouped.get(case_id, [])
        observed_seeds = {int(row["run_seed"]) for row in case_rows}
        exact_fes = all(
            int(row["terminal_fes"]) == int(protocol["max_fes"]) for row in case_rows
        )
        mapped_action = mapping[case_id[0]]
        action_matches = all(row["action_name"].strip() == mapped_action for row in case_rows)
        coverage_complete = (
            len(case_rows) == len(expected_seeds)
            and observed_seeds == expected_seeds
            and exact_fes
            and action_matches
        )
        if not coverage_complete:
            reasons = []
            if len(case_rows) != len(expected_seeds) or observed_seeds != expected_seeds:
                reasons.append("incomplete_seed_coverage")
            if not exact_fes:
                reasons.append("terminal_fe_mismatch")
            if not action_matches:
                reasons.append("mapped_action_mismatch")
            audited.append(
                {
                    "case": case_id,
                    "action": mapped_action,
                    "status": "missing" if not case_rows else "failed",
                    "reason": ";".join(reasons),
                    "seed_count": len(observed_seeds),
                    "exact_terminal_fes": exact_fes,
                    "mapped_action": mapped_action,
                    "observed_actions": sorted({row["action_name"] for row in case_rows}),
                    "target_mean": targets[case_id]["mean"],
                    "target_sample_std": targets[case_id]["sample_std"],
                }
            )
            continue

        values = [float(row["final_error"]) for row in case_rows]
        mean = fmean(values)
        sample_std = stdev(values)
        target = targets[case_id]
        mean_matches = _format(mean, precision) == _format(target["mean"], precision)
        std_matches = _format(sample_std, precision) == _format(
            target["sample_std"], precision
        )
        audited.append(
            {
                "case": case_id,
                "action": mapped_action,
                "status": "recovered" if mean_matches and std_matches else "failed",
                "reason": "rounded_aggregate_match"
                if mean_matches and std_matches
                else "historical_aggregate_mismatch",
                "seed_count": len(observed_seeds),
                "exact_terminal_fes": exact_fes,
                "mapped_action": mapped_action,
                "mean": mean,
                "sample_std": sample_std,
                "target_mean": target["mean"],
                "target_sample_std": target["sample_std"],
                "formatted_mean": _format(mean, precision),
                "formatted_sample_std": _format(sample_std, precision),
                "formatted_target_mean": _format(target["mean"], precision),
                "formatted_target_sample_std": _format(target["sample_std"], precision),
                "mean_match": mean_matches,
                "sample_std_match": std_matches,
            }
        )

    counts = {
        status: sum(row["status"] == status for row in audited)
        for status in ("recovered", "failed", "missing")
    }
    context_count = len(rows)
    expected_context_count = len(expected_cases) * len(expected_seeds)
    summary_consistent = (
        summary.get("context_count") == context_count
        and summary.get("case_count") == len(expected_cases)
        and summary.get("seed_count_per_case") == len(expected_seeds)
    )
    manifest_consistent = (
        manifest.get("campaign_kind") == "historical_fixed_expert"
        and int(manifest.get("max_fes", -1)) == int(protocol["max_fes"])
        and {str(case) for case in manifest.get("cases", [])} == expected_cases
        and {int(seed) for seed in manifest.get("seeds", [])} == expected_seeds
    )
    all_terminal_fes_exact = all(row.get("exact_terminal_fes", False) for row in audited)
    return {
        "status": "passed" if counts == {"recovered": 24, "failed": 0, "missing": 0} else "failed",
        "gate_passed": counts == {"recovered": 24, "failed": 0, "missing": 0}
        and context_count == expected_context_count
        and all_terminal_fes_exact
        and summary_consistent
        and manifest_consistent,
        "context_count": context_count,
        "expected_context_count": expected_context_count,
        "case_count": len(audited),
        "seed_count_per_case": len(expected_seeds),
        "all_terminal_fes_exact": all_terminal_fes_exact,
        "summary_consistent": summary_consistent,
        "manifest_consistent": manifest_consistent,
        "counts": counts,
        "mean_match_count": sum(row.get("mean_match", False) for row in audited),
        "sample_std_match_count": sum(row.get("sample_std_match", False) for row in audited),
        "recovered_case_count": counts["recovered"],
        "max_fes": int(protocol["max_fes"]),
        "max_workers": manifest.get("max_workers"),
        "results_csv": campaign["results_csv"],
        "summary": campaign["summary"],
        "campaign_manifest": campaign["campaign_manifest"],
        "summary_gate_passed": summary.get("gate_passed"),
        "runtime_warning_count": summary.get("runtime_warning_count"),
        "unexpected_runtime_warning_count": summary.get("unexpected_runtime_warning_count"),
        "cases": audited,
    }


def _audit_frozen_matrix(
    *,
    root: Path,
    config: dict[str, Any],
    targets: dict[str, dict[str, float]],
    precision: str,
) -> dict[str, Any]:
    matrix = config["frozen_independent_matrix"]
    manifest = _load_json(root / matrix["manifest"])
    expert_mapping = config["expert_mapping"]
    expected_seeds = {int(seed) for seed in manifest["seeds"]}
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with (root / matrix["outcomes"]).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            case_id = record["case_id"]
            action = expert_mapping[case_id[0]]
            outcome = next(
                item for item in record["outcomes"] if item["action_name"] == action
            )
            grouped[case_id].append((int(record["run_seed"]), float(outcome["final_error"])))

    cases = []
    for case_id, target in sorted(targets.items()):
        records = grouped.get(case_id, [])
        observed_seeds = {seed for seed, _ in records}
        if observed_seeds != expected_seeds:
            raise ValueError(f"incomplete frozen matrix expert lane: {case_id}")
        values = [value for _, value in records]
        mean = fmean(values)
        sample_std = stdev(values)
        formatted_mean = _format(mean, precision)
        formatted_target = _format(target["mean"], precision)
        mean_met = float(formatted_mean) <= float(formatted_target)
        cases.append(
            {
                "case": case_id,
                "action": expert_mapping[case_id[0]],
                "status": "historical_mean_met" if mean_met else "historical_mean_not_met",
                "seed_count": len(observed_seeds),
                "mean": mean,
                "sample_std": sample_std,
                "target_mean": target["mean"],
                "target_sample_std": target["sample_std"],
                "mean_ratio": mean / target["mean"],
                "formatted_mean": formatted_mean,
                "formatted_sample_std": _format(sample_std, precision),
                "formatted_target_mean": formatted_target,
                "formatted_target_sample_std": _format(target["sample_std"], precision),
            }
        )

    source_checks = []
    for component, relative_path in sorted(matrix["source_files"].items()):
        expected = manifest["source_hashes"][component]
        actual = _sha256(root / relative_path)
        source_checks.append(
            {
                "component": component,
                "matches_frozen_matrix": actual == expected,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    mean_met_count = sum(row["status"] == "historical_mean_met" for row in cases)
    source_match_count = sum(row["matches_frozen_matrix"] for row in source_checks)
    return {
        "phase1_protocol": manifest["phase1_protocol"],
        "max_fes": manifest["max_fes"],
        "seed_count": len(expected_seeds),
        "case_count": len(cases),
        "rounded_historical_mean_met_count": mean_met_count,
        "rounded_historical_mean_not_met_count": len(cases) - mean_met_count,
        "source_hash_match_count": source_match_count,
        "source_hash_total": len(source_checks),
        "current_source_compatible": source_match_count == len(source_checks),
        "cases": cases,
        "source_checks": source_checks,
    }


def _audit_current_replay(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    replay = config["current_replay"]
    summary_path = root / replay["summary"]
    if not summary_path.is_file():
        return {"status": "not_run", "context_count": 0, "passed_count": 0, "cases": []}
    summary = _load_json(summary_path)
    rows = []
    for path in sorted((root / replay["receipts"]).glob("*.json")):
        receipt = _load_json(path)
        claimed = receipt.pop("receipt_hash", None)
        if claimed != canonical_sha256(receipt):
            raise ValueError(f"current replay receipt hash drifted: {path}")
        receipt["receipt_hash"] = claimed
        rows.append(
            {
                "case": receipt["case"],
                "action": receipt["action"],
                "terminal_fes_match": receipt["terminal_fes_match"],
                "final_error_match": receipt["final_error_match"],
                "exact_result_match": receipt["exact_result_match"],
                "replay_passed": receipt["replay_passed"],
                "final_error": receipt["final_error"],
                "expected_final_error": receipt["expected_final_error"],
                "final_error_ratio": receipt["final_error"]
                / receipt["expected_final_error"],
                "runtime_warning_count": len(receipt["runtime_warnings"]),
            }
        )
    if len(rows) != summary["context_count"]:
        raise ValueError("current replay summary and receipt count disagree")
    return {
        "status": "passed" if summary["all_replays_passed"] else "failed",
        "context_count": len(rows),
        "passed_count": sum(row["replay_passed"] for row in rows),
        "failed_count": sum(not row["replay_passed"] for row in rows),
        "cases": rows,
    }


def _validated_replay_receipts(
    root: Path,
    relative_directory: str,
    *,
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    receipts: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((root / relative_directory).glob("*.json")):
        receipt = _load_json(path)
        claimed = receipt.pop("receipt_hash", None)
        if claimed != canonical_sha256(receipt):
            raise ValueError(f"{label} receipt hash drifted: {path}")
        receipt["receipt_hash"] = claimed
        key = (str(receipt["case"]), str(receipt["action"]))
        if key in receipts:
            raise ValueError(f"duplicate {label} receipt: {key}")
        receipts[key] = receipt
    return receipts


def _audit_frozen_source_control(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    control = config["frozen_source_control"]
    current_receipts = _validated_replay_receipts(
        root,
        config["current_replay"]["receipts"],
        label="current replay",
    )
    frozen_receipts = _validated_replay_receipts(
        root,
        control["receipts"],
        label="frozen-source control",
    )
    if not frozen_receipts:
        return {
            "status": "not_run",
            "context_count": 0,
            "matched_count": 0,
            "source_hash_match_count": 0,
            "source_hash_total": 0,
            "cases": [],
        }
    if set(current_receipts) != set(frozen_receipts):
        raise ValueError("current and frozen-source control contexts disagree")

    manifest_path = root / control["source_manifest"]
    manifest = _load_json(manifest_path)
    source_root = root / control["source_root"]
    source_checks = []
    for relative, expected in sorted(manifest["source_hashes"].items()):
        actual = _sha256(source_root / relative)
        source_checks.append(
            {
                "path": relative,
                "matches_manifest": actual == expected,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    source_match_count = sum(row["matches_manifest"] for row in source_checks)

    cases = []
    for key in sorted(current_receipts):
        current = current_receipts[key]
        frozen = frozen_receipts[key]
        comparison = {
            "terminal_fes": current["terminal_fes"] == frozen["terminal_fes"],
            "final_error": current["final_error"] == frozen["final_error"],
            "result_hash": current["result_hash"] == frozen["result_hash"],
            "receipt_hash": current["receipt_hash"] == frozen["receipt_hash"],
        }
        cases.append(
            {
                "case": key[0],
                "action": key[1],
                "runtime_match": all(comparison.values()),
                "comparison": comparison,
                "current_final_error": current["final_error"],
                "frozen_source_final_error": frozen["final_error"],
                "stored_matrix_final_error": current["expected_final_error"],
                "stored_matrix_exact_match": current["replay_passed"],
            }
        )
    matched_count = sum(row["runtime_match"] for row in cases)
    all_sources_match = source_match_count == len(source_checks)
    return {
        "status": "matched"
        if matched_count == len(cases) and all_sources_match
        else "drifted",
        "execution_method": control["execution_method"],
        "source_manifest": control["source_manifest"],
        "source_manifest_sha256": _sha256(manifest_path),
        "context_count": len(cases),
        "matched_count": matched_count,
        "source_hash_match_count": source_match_count,
        "source_hash_total": len(source_checks),
        "cases": cases,
        "source_checks": source_checks,
    }


def run_audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    root = REPOSITORY_ROOT
    config = _load_json(config_path)
    targets = _reference_rows(root, config)
    protocol = config["candidate_protocol"]
    seeds = {int(seed) for seed in protocol["seeds"]}
    cases: list[dict[str, Any]] = []
    for action, evidence in config["evidence"].items():
        cases.extend(
            _audit_action(
                root=root,
                action=action,
                evidence=evidence,
                targets=targets,
                seeds=seeds,
                max_fes=int(protocol["max_fes"]),
                precision=config["historical_reference"]["aggregate_precision"],
            )
        )
    cases.sort(key=lambda row: row["case"])
    counts = {
        status: sum(row["status"] == status for row in cases)
        for status in ("recovered", "failed", "missing")
    }
    frozen_matrix = _audit_frozen_matrix(
        root=root,
        config=config,
        targets=targets,
        precision=config["historical_reference"]["aggregate_precision"],
    )
    current_fixed_expert = _audit_current_fixed_expert(
        root=root,
        config=config,
        targets=targets,
        precision=config["historical_reference"]["aggregate_precision"],
    )
    current_replay = _audit_current_replay(root, config)
    frozen_source_control = _audit_frozen_source_control(root, config)
    return {
        "schema_version": config["schema_version"],
        "gate_passed": current_fixed_expert["gate_passed"]
        and counts == {"recovered": 24, "failed": 0, "missing": 0}
        and frozen_matrix["rounded_historical_mean_not_met_count"] == 0
        and frozen_matrix["current_source_compatible"]
        and current_replay["status"] == "passed",
        "counts": counts,
        "protocol": protocol,
        "source_table": config["historical_reference"]["table_csv"],
        "cases": cases,
        "current_fixed_expert": current_fixed_expert,
        "frozen_independent_matrix": frozen_matrix,
        "current_replay": current_replay,
        "frozen_source_control": frozen_source_control,
        "partial_evidence": {
            "E1": {
                "status": "insufficient_and_mean_gate_failed",
                "source": config["evidence"]["smp"]["partial_e1_summary"],
                "seed_count": 5,
                "observed_mean": 1282914.970176869,
                "historical_target_mean": targets["E1"]["mean"],
            },
            "E3": {
                "status": "insufficient_five_seed_pilot_only",
                "source": config["evidence"]["smp"]["partial_e1_e3_summary"],
                "seed_count": 5,
            },
        },
    }


def render_report(audit: dict[str, Any]) -> str:
    counts = audit["counts"]
    current_fixed_expert = audit["current_fixed_expert"]
    matrix = audit["frozen_independent_matrix"]
    replay = audit["current_replay"]
    control = audit["frozen_source_control"]
    lines = [
        "# ARAC historical recovery audit",
        "",
        f"- Gate passed: **{str(audit['gate_passed']).lower()}**",
        f"- Recovered: **{counts['recovered']}/24**",
        f"- Failed: **{counts['failed']}/24**",
        f"- Missing: **{counts['missing']}/24**",
            f"- Source: `{audit['source_table']}`",
            f"- Current/frozen-source runtime parity: "
            f"**{control['matched_count']}/{control['context_count']}**",
        "",
        "| Case | Expert | Status | Mean | Historical mean | Sample std | Historical std |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in audit["cases"]:
        lines.append(
            "| {case} | {action} | {status} | {mean} | {target_mean:.2E} | "
            "{sample_std} | {target_std:.2E} |".format(
                case=row["case"],
                action=row["action"].upper(),
                status=row["status"],
                mean=row.get("formatted_mean", "NA"),
                target_mean=row["target_mean"],
                sample_std=row.get("formatted_sample_std", "NA"),
                target_std=row["target_sample_std"],
            )
        )
    lines.extend(
        [
            "",
            "## Current fixed-expert campaign",
            "",
            f"- Status: **{current_fixed_expert['status']}**",
            f"- Campaign gate passed: **{str(current_fixed_expert['gate_passed']).lower()}**",
            f"- Complete arms: **{current_fixed_expert['context_count']}/"
            f"{current_fixed_expert.get('expected_context_count', 600)}**",
            f"- Exact terminal FE: **{str(current_fixed_expert.get('all_terminal_fes_exact', False)).lower()}**",
            f"- Mean matches at displayed precision: **{current_fixed_expert.get('mean_match_count', 0)}/24**",
            f"- Sample-std matches at displayed precision: **{current_fixed_expert.get('sample_std_match_count', 0)}/24**",
            f"- Recovered cases: **{current_fixed_expert.get('recovered_case_count', 0)}/24**",
            "",
            "| Case | Expert | Status | Mean | Historical mean | Sample std | Historical std |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in current_fixed_expert["cases"]:
        lines.append(
            "| {case} | {action} | {status} | {mean} | {target_mean:.2E} | "
            "{sample_std} | {target_std:.2E} |".format(
                case=row["case"],
                action=row["action"].upper(),
                status=row["status"],
                mean=row.get("formatted_mean", "NA"),
                target_mean=row["target_mean"],
                sample_std=row.get("formatted_sample_std", "NA"),
                target_std=row["target_sample_std"],
            )
        )
    lines.extend(
        [
            "",
            "## Frozen independent-action matrix",
            "",
            f"- Historical mean met at displayed precision: "
            f"**{matrix['rounded_historical_mean_met_count']}/24**",
            f"- Historical mean not met: "
            f"**{matrix['rounded_historical_mean_not_met_count']}/24**",
            f"- Current source hashes matching the frozen matrix: "
            f"**{matrix['source_hash_match_count']}/{matrix['source_hash_total']}**",
            "",
            "| Case | Expert | Frozen mean | Historical mean | Ratio | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in matrix["cases"]:
        lines.append(
            "| {case} | {action} | {mean} | {target} | {ratio:.3f} | {status} |".format(
                case=row["case"],
                action=row["action"].upper(),
                mean=row["formatted_mean"],
                target=row["formatted_target_mean"],
                ratio=row["mean_ratio"],
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "## Current-code checkpoint replay",
            "",
            f"- Replay status: **{replay['status']}**",
            f"- Exact replay passed: **{replay['passed_count']}/{replay['context_count']}**",
            "",
            "| Case | Expert | Current error | Frozen error | Ratio | FE | Error | Hash |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in replay["cases"]:
        lines.append(
            "| {case} | {action} | {current:.6E} | {expected:.6E} | {ratio:.3f} | "
            "{fe} | {error} | {hash_match} |".format(
                case=row["case"],
                action=row["action"].upper(),
                current=row["final_error"],
                expected=row["expected_final_error"],
                ratio=row["final_error_ratio"],
                fe=row["terminal_fes_match"],
                error=row["final_error_match"],
                hash_match=row["exact_result_match"],
            )
        )
    lines.extend(
        [
            "",
            "## Frozen-source runtime control",
            "",
            f"- Control status: **{control['status']}**",
            f"- Current results matching the manifest-bound frozen source: "
            f"**{control['matched_count']}/{control['context_count']}**",
            f"- Frozen source files matching their manifest: "
            f"**{control['source_hash_match_count']}/{control['source_hash_total']}**",
            "",
            "| Case | Expert | Current/frozen source | Stored v5 arm |",
            "|---|---|---|---|",
        ]
    )
    for row in control["cases"]:
        lines.append(
            "| {case} | {action} | {runtime} | {stored} |".format(
                case=row["case"],
                action=row["action"].upper(),
                runtime=row["runtime_match"],
                stored=row["stored_matrix_exact_match"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The current fixed-expert campaign completed all 600 mapped arms at the exact "
            "3,000,000-FE budget, but its aggregate gate remains closed: only the six A-series "
            "means match the displayed historical precision, and no case matches both mean "
            "and sample standard deviation. The historical artifact layer still has an "
            "incomplete E/SMP lane, while the frozen independent matrix misses historical "
            "means on multiple cases. The current legacy path matches its manifest-bound "
            "frozen source on all four representative contexts, but three stored v5 "
            "block-action arms do not reproduce from that same source. Selector correctness "
            "and ARAC-Core end-to-end claims must remain deferred.",
            "",
        ]
    )
    return "\n".join(lines)


def _expected_outputs(audit: dict[str, Any]) -> tuple[str, str]:
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    return payload, render_report(audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args(argv)

    audit = run_audit(args.config.resolve())
    payload, report = _expected_outputs(audit)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != payload:
            raise ValueError(f"stale or missing audit output: {args.output}")
        if not args.report.is_file() or args.report.read_text(encoding="utf-8") != report:
            raise ValueError(f"stale or missing audit report: {args.report}")
    else:
        args.output.write_text(payload, encoding="utf-8")
        args.report.write_text(report, encoding="utf-8")

    print(json.dumps({"gate_passed": audit["gate_passed"], **audit["counts"]}))
    return int(args.require_passed and not audit["gate_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
