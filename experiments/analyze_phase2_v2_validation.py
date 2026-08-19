"""Re-audit frozen validation receipts with the shifted log-error metric."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from arac.runtime.contracts import canonical_sha256
from experiments.phase2_v2_pilot import _file_sha256, _write_json_atomic
from experiments.phase2_v2_validation import _comparison_rows, _write_csv_atomic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "artifacts" / "phase2_v2_validation_ioh_v2"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "artifacts" / "phase2_v2_validation_ioh_v2_analysis_v1"
)
ANALYSIS_MANIFEST_SCHEMA = "arac-phase2-v2-validation-analysis-manifest-v1"
ANALYSIS_SUMMARY_SCHEMA = "arac-phase2-v2-validation-analysis-summary-v1"


def _read_frozen_receipts(root: Path) -> tuple[str, list[dict[str, object]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest_sha256 = manifest.pop("manifest_sha256")
    if canonical_sha256(manifest) != manifest_sha256:
        raise ValueError("input validation manifest hash drifted")
    receipts: list[dict[str, object]] = []
    for path in sorted((root / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt_sha256 = receipt.pop("receipt_sha256")
        if canonical_sha256(receipt) != receipt_sha256:
            raise ValueError(f"input validation receipt hash drifted: {path.name}")
        if receipt.get("status") != "completed":
            raise ValueError(f"input validation contains a failed receipt: {path.name}")
        if receipt.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"input validation receipt manifest drifted: {path.name}")
        receipt["receipt_sha256"] = receipt_sha256
        receipts.append(receipt)
    if not receipts:
        raise ValueError("input validation has no receipts")
    return str(manifest_sha256), receipts


def analyze_validation(*, input_root: Path, output_root: Path) -> dict[str, object]:
    source = Path(input_root).resolve()
    target = Path(output_root).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"analysis output root is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    input_manifest_sha256, receipts = _read_frozen_receipts(source)
    rows, comparison = _comparison_rows(receipts)
    _write_csv_atomic(target / "results_shifted.csv", rows)
    receipt_set_sha256 = canonical_sha256(
        [receipt["receipt_sha256"] for receipt in receipts]
    )
    manifest_payload = {
        "schema_version": ANALYSIS_MANIFEST_SCHEMA,
        "input_manifest_sha256": input_manifest_sha256,
        "input_summary_sha256": _file_sha256(source / "summary.json"),
        "receipt_set_sha256": receipt_set_sha256,
        "analysis_source_sha256": _file_sha256(Path(__file__).resolve()),
        "comparison_metric": "log10((probe_error + 1) / (mechanism_error + 1))",
    }
    analysis_manifest = {
        **manifest_payload,
        "analysis_manifest_sha256": canonical_sha256(manifest_payload),
    }
    _write_json_atomic(target / "manifest.json", analysis_manifest)

    probe_receipts = [
        receipt for receipt in receipts if receipt["method"] == "probe_commit_v2"
    ]
    mechanism_receipts = [
        receipt for receipt in receipts if receipt["method"] == "mechanism_score_v1"
    ]
    probe_reasons = Counter(
        receipt["result"]["selection_reason"] for receipt in probe_receipts
    )
    mechanism_reasons = Counter(
        receipt["result"]["selection_reason"] for receipt in mechanism_receipts
    )
    numerical_repairs = {
        method: {
            "total": sum(
                int(receipt["result"]["numerical_repair_count"])
                for receipt in receipts
                if receipt["method"] == method
            ),
            "runs": sum(
                int(receipt["result"]["numerical_repair_count"]) > 0
                for receipt in receipts
                if receipt["method"] == method
            ),
        }
        for method in ("probe_commit_v2", "mechanism_score_v1")
    }
    fallback_count = sum(
        reason.startswith("probe_cap_")
        for reason in (
            receipt["result"]["selection_reason"] for receipt in probe_receipts
        )
    )
    summary = {
        "schema_version": ANALYSIS_SUMMARY_SCHEMA,
        "analysis_manifest_sha256": analysis_manifest["analysis_manifest_sha256"],
        "input_manifest_sha256": input_manifest_sha256,
        "receipt_count": len(receipts),
        "context_count": len(rows),
        "comparison": comparison,
        "probe_reason_counts": dict(sorted(probe_reasons.items())),
        "mechanism_reason_counts": dict(sorted(mechanism_reasons.items())),
        "probe_cap_fallback_count": fallback_count,
        "probe_cap_fallback_rate": fallback_count / len(probe_receipts),
        "numerical_sigma_floor_repairs": numerical_repairs,
        "quality_claim": "limited_pre_registered_validation_not_generalization",
    }
    _write_json_atomic(target / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    summary = analyze_validation(
        input_root=arguments.input_root,
        output_root=arguments.output_root,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
