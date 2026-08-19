"""Audit a restored Phase-I AOB pilot without rerunning failed contexts."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any

from arac.runtime.contracts import canonical_sha256
import experiments.phase2_v2_aob_restored_pilot as pilot
from experiments.phase2_v2_pilot import _file_sha256, _tree_sha256, _write_json_atomic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "artifacts" / "phase2_v2_aob_restored_pilot_v2"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "artifacts" / "phase2_v2_aob_restored_pilot_v2_analysis_v1"
)
ANALYSIS_MANIFEST_SCHEMA = "arac-phase2-v2-aob-restored-analysis-manifest-v1"
ANALYSIS_SUMMARY_SCHEMA = "arac-phase2-v2-aob-restored-analysis-summary-v1"


def _validate_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = pilot._read_json(root / "manifest.json", "input manifest")
    body = dict(manifest)
    claimed_hash = body.pop("manifest_sha256", None)
    if claimed_hash != canonical_sha256(body):
        raise ValueError("restored analysis input manifest hash drifted")
    if manifest.get("schema_version") != pilot.MANIFEST_SCHEMA:
        raise ValueError("restored analysis input manifest schema drifted")
    config_path = root / "config.json"
    config = pilot.load_config(config_path)
    if _file_sha256(config_path) != manifest.get("config_sha256"):
        raise ValueError("restored analysis frozen config hash drifted")
    contexts = pilot._contexts(config)
    if (
        manifest.get("context_count") != len(contexts)
        or manifest.get("method_count") != len(pilot.METHODS)
        or manifest.get("matched_total_fes") != pilot.matched_total_fes(config)
        or manifest.get("max_workers") != config["max_workers"]
        or manifest.get("parallel_unit") != "context_triplet"
    ):
        raise ValueError("restored analysis input manifest contract drifted")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(pilot.SOURCE_PATHS):
        raise ValueError("restored analysis source manifest drifted")
    for relative, expected in source_hashes.items():
        if _file_sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"restored analysis source hash drifted: {relative}")
    vendor_count, vendor_hash = _tree_sha256(REPOSITORY_ROOT / "vendor" / "aob")
    if manifest.get("aob_vendor_tree") != {
        "file_count": vendor_count,
        "tree_sha256": vendor_hash,
    }:
        raise ValueError("restored analysis AOB vendor tree drifted")
    return manifest, config


def _validate_failed_receipt(
    path: Path,
    receipt: dict[str, Any],
    *,
    context_index: int,
    context: dict[str, Any],
    method: str,
    manifest_sha256: str,
) -> None:
    body = dict(receipt)
    claimed_hash = body.pop("receipt_sha256", None)
    if claimed_hash != canonical_sha256(body):
        raise ValueError(f"restored analysis receipt hash drifted: {path.name}")
    expected = {
        "schema_version": pilot.RECEIPT_SCHEMA,
        "status": "failed",
        "run_index": pilot._run_index(context_index, method),
        "context_id": pilot._context_id(context),
        "benchmark": context,
        "method": method,
        "manifest_sha256": manifest_sha256,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"restored analysis failed receipt {name} drifted: {path.name}")
    required = {
        *expected,
        "error_type",
        "error",
        "traceback",
        "receipt_sha256",
    }
    if set(receipt) != required or any(
        not isinstance(receipt.get(name), str) or not receipt[name]
        for name in ("error_type", "error", "traceback")
    ):
        raise ValueError(f"restored analysis failed receipt payload drifted: {path.name}")


def _load_receipts(
    root: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    receipt_root = root / "receipts"
    contexts = pilot._contexts(config)
    planned = [
        (context_index, context, method)
        for context_index, context in enumerate(contexts)
        for method in pilot.METHODS
    ]
    expected_names = {
        pilot._receipt_path(receipt_root, context_index, context, method).name
        for context_index, context, method in planned
    }
    actual_names = {path.name for path in receipt_root.glob("*.json")}
    if actual_names != expected_names:
        raise ValueError(
            "restored analysis receipt set drifted: "
            f"missing={sorted(expected_names - actual_names)} "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    receipts = []
    for context_index, context, method in planned:
        path = pilot._receipt_path(receipt_root, context_index, context, method)
        receipt = pilot._read_json(path, "input receipt")
        if receipt.get("status") == "completed":
            pilot._validate_receipt(
                path,
                context_index=context_index,
                context=context,
                method=method,
                config=config,
                manifest_sha256=manifest["manifest_sha256"],
            )
        elif receipt.get("status") == "failed":
            _validate_failed_receipt(
                path,
                receipt,
                context_index=context_index,
                context=context,
                method=method,
                manifest_sha256=manifest["manifest_sha256"],
            )
        else:
            raise ValueError(f"restored analysis receipt status drifted: {path.name}")
        receipts.append(receipt)
    return receipts


def _complete_triplets(
    receipts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    context_ids = sorted({receipt["context_id"] for receipt in receipts})
    complete = []
    incomplete = []
    for context_id in context_ids:
        context_receipts = [
            receipt for receipt in receipts if receipt["context_id"] == context_id
        ]
        if (
            len(context_receipts) == len(pilot.METHODS)
            and {receipt["method"] for receipt in context_receipts}
            == set(pilot.METHODS)
            and all(receipt["status"] == "completed" for receipt in context_receipts)
        ):
            complete.extend(context_receipts)
        else:
            incomplete.append(context_id)
    return complete, incomplete


def _comparison_counts(rows: list[dict[str, Any]], baseline: str) -> dict[str, int]:
    baseline_key = f"{baseline}_error"
    wins = sum(row["probe_error"] < row[baseline_key] for row in rows)
    losses = sum(row["probe_error"] > row[baseline_key] for row in rows)
    return {"wins": wins, "losses": losses, "ties": len(rows) - wins - losses}


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def run_analysis(
    *,
    input_root: Path = DEFAULT_INPUT,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = Path(input_root).resolve()
    target = Path(output_root).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"restored analysis output root is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    manifest, config = _validate_manifest(source)
    receipts = _load_receipts(source, manifest, config)
    completed = [receipt for receipt in receipts if receipt["status"] == "completed"]
    failed = [receipt for receipt in receipts if receipt["status"] == "failed"]
    complete_receipts, incomplete_contexts = _complete_triplets(receipts)
    comparison = pilot._comparison_rows(complete_receipts) if complete_receipts else []
    action_counts = {
        method: dict(
            Counter(
                receipt["result"]["selected_action"]
                for receipt in complete_receipts
                if receipt["method"] == method
            )
        )
        for method in pilot.METHODS
    }
    shifted_log_means = {
        name: statistics.fmean(row[name] for row in comparison) if comparison else None
        for name in (
            "probe_vs_matched_shifted_log10",
            "matched_vs_full_shifted_log10",
            "probe_vs_full_shifted_log10",
        )
    }
    promotion_gate_passed = not failed and len(comparison) == len(pilot._contexts(config))
    failure_rows = [
        {
            "run_index": receipt["run_index"],
            "context_id": receipt["context_id"],
            "method": receipt["method"],
            "error_type": receipt["error_type"],
            "error": receipt["error"],
            "receipt_sha256": receipt["receipt_sha256"],
        }
        for receipt in failed
    ]
    summary = {
        "schema_version": ANALYSIS_SUMMARY_SCHEMA,
        "input_manifest_sha256": manifest["manifest_sha256"],
        "planned_receipts": len(receipts),
        "completed_receipts": len(completed),
        "failed_receipts": len(failed),
        "complete_context_triplets": len(comparison),
        "incomplete_contexts": incomplete_contexts,
        "all_completed_receipt_budgets_valid": True,
        "promotion_gate_passed": promotion_gate_passed,
        "downstream_blocked": not promotion_gate_passed,
        "failure_evidence": failure_rows,
        "partial_comparison": {
            "context_count": len(comparison),
            "probe_vs_matched": _comparison_counts(comparison, "matched"),
            "probe_vs_full": _comparison_counts(comparison, "full"),
            "shifted_log10_means": shifted_log_means,
            "method_action_counts": action_counts,
            "rows": comparison,
        },
        "quality_claim": (
            "limited_restored_phase1_pilot_not_generalization"
            if promotion_gate_passed
            else "failed_campaign_partial_context_diagnostic_not_promotion"
        ),
    }
    analysis_manifest_body = {
        "schema_version": ANALYSIS_MANIFEST_SCHEMA,
        "input_manifest_sha256": manifest["manifest_sha256"],
        "input_receipt_sha256": [
            receipt["receipt_sha256"]
            for receipt in sorted(receipts, key=lambda item: item["run_index"])
        ],
        "analyzer_sha256": _file_sha256(Path(__file__).resolve()),
    }
    analysis_manifest = {
        **analysis_manifest_body,
        "analysis_manifest_sha256": canonical_sha256(analysis_manifest_body),
    }
    _write_json_atomic(target / "manifest.json", analysis_manifest)
    _write_json_atomic(target / "summary.json", summary)
    _write_json_atomic(target / "failures.json", failure_rows)
    _write_csv_atomic(target / "comparison_complete_contexts.csv", comparison)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    pilot._safe_print(
        json.dumps(
            run_analysis(input_root=args.input_root, output_root=args.output_root),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
