"""Read-only decomposition of the Phase-II v2 AOB preservation failure."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

from arac.runtime.contracts import ACTION_NAMES, canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_ROOT = REPOSITORY_ROOT / "artifacts" / "phase2_v2_validation_ioh_v2"
DEFAULT_V3_ROOT = REPOSITORY_ROOT / "artifacts" / "final_24x25_v3_bounded"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "artifacts" / "phase2_v2_aob_preservation_audit_v1"
)
SCHEMA = "arac-phase2-v2-aob-preservation-audit-v1"
CASES = {"A1", "E1", "R1", "S1"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _v2_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "receipts").glob("*.json")):
        receipt = _load(path)
        benchmark = receipt.get("benchmark", {})
        if receipt.get("status") != "completed" or benchmark.get("suite") != "aob":
            continue
        result = receipt["result"]
        phase1 = result["phase1"]
        rows.append(
            {
                "context_id": receipt["context_id"],
                "case_id": benchmark["case"],
                "run_seed": int(benchmark["run_seed"]),
                "method": receipt["method"],
                "global_max_fes": int(result["global_total_fes"]),
                "phase1_fes": int(phase1["phase1_fes"]),
                "structural_inference_complete": float(
                    phase1["structural_inference_complete"]
                ),
                "selected_action": result["selected_action"],
                "final_error": float(result["final_error"]),
                "selected_ledger_fes": int(
                    result.get("selected_ledger_fes", result["global_total_fes"])
                ),
                "selected_action_fes": int(
                    result.get("selected_action_fes", result["global_total_fes"])
                ),
                "selection_reason": result["selection_reason"],
                "branch_probe_fes": int(result.get("branch_probe_fes", 0)),
            }
        )
    if not rows:
        raise ValueError(f"no completed AOB v2 receipts found under {root}")
    return rows


def _v3_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("runs/*/seed_*/receipt.json")):
        receipt = _load(path)
        case_id = receipt.get("case_id")
        if case_id not in CASES:
            continue
        names = receipt.get("feature_names", [])
        values = receipt.get("feature_values", [])
        feature_map = dict(zip(names, values, strict=True))
        rows.append(
            {
                "context_id": f"aob_{case_id}_s{receipt['run_seed']}",
                "case_id": case_id,
                "run_seed": int(receipt["run_seed"]),
                "global_max_fes": int(receipt["max_fes"]),
                "phase1_fes": int(receipt["phase1_fes"]),
                "structural_inference_complete": float(
                    feature_map["structural_inference_complete"]
                ),
                "selected_action": receipt["selected_action"],
                "final_error": float(receipt["final_error"]),
            }
        )
    if not rows:
        raise ValueError(f"no reference AOB v3 receipts found under {root}")
    return rows


def summarize_v2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["context_id"]][row["method"]] = row
    pairs = []
    for context_id, methods in sorted(grouped.items()):
        if set(methods) != {"probe_commit_v2", "mechanism_score_v1"}:
            raise ValueError(f"incomplete v2 method pair: {context_id}")
        probe = methods["probe_commit_v2"]
        mechanism = methods["mechanism_score_v1"]
        pairs.append(
            {
                "context_id": context_id,
                "case_id": probe["case_id"],
                "run_seed": probe["run_seed"],
                "phase1_fes": probe["phase1_fes"],
                "structural_inference_complete": probe[
                    "structural_inference_complete"
                ],
                "probe_action": probe["selected_action"],
                "mechanism_action": mechanism["selected_action"],
                "same_action": probe["selected_action"] == mechanism["selected_action"],
                "probe_error": probe["final_error"],
                "mechanism_error": mechanism["final_error"],
                "probe_selected_ledger_fes": probe["selected_ledger_fes"],
                "mechanism_selected_ledger_fes": mechanism["selected_ledger_fes"],
                "probe_tax_fes": mechanism["selected_ledger_fes"]
                - probe["selected_ledger_fes"],
                "probe_reason": probe["selection_reason"],
                "mechanism_reason": mechanism["selection_reason"],
            }
        )
    return {
        "context_count": len(pairs),
        "phase1_fes": sorted({row["phase1_fes"] for row in pairs}),
        "structural_complete_counts": dict(
            Counter(str(row["structural_inference_complete"]) for row in pairs)
        ),
        "action_pair_counts": dict(
            Counter(
                f"{row['probe_action']}->{row['mechanism_action']}"
                for row in pairs
            )
        ),
        "same_action_count": sum(row["same_action"] for row in pairs),
        "probe_tax_fes": sorted({row["probe_tax_fes"] for row in pairs}),
        "probe_action_counts": {
            action: sum(row["probe_action"] == action for row in pairs)
            for action in ACTION_NAMES
        },
        "mechanism_action_counts": {
            action: sum(row["mechanism_action"] == action for row in pairs)
            for action in ACTION_NAMES
        },
        "pairs": pairs,
    }


def summarize_v3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "context_count": len(rows),
        "global_max_fes": sorted({row["global_max_fes"] for row in rows}),
        "phase1_fes": sorted({row["phase1_fes"] for row in rows}),
        "structural_complete_counts": dict(
            Counter(str(row["structural_inference_complete"]) for row in rows)
        ),
        "action_counts": {
            action: sum(row["selected_action"] == action for row in rows)
            for action in ACTION_NAMES
        },
        "cases": sorted({row["case_id"] for row in rows}),
    }


def run_analysis(*, v2_root: Path, v3_root: Path, output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"analysis output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    v2 = summarize_v2(_v2_rows(v2_root.resolve()))
    v3 = summarize_v3(_v3_rows(v3_root.resolve()))
    manifest_payload = {
        "schema_version": SCHEMA,
        "v2_root": str(v2_root.resolve()),
        "v3_root": str(v3_root.resolve()),
        "v2_manifest_sha256": _load(v2_root / "summary.json")["manifest_sha256"],
        "v3_config_sha256": _load(
            next(v3_root.glob("runs/*/seed_*/receipt.json"))
        )["config_sha256"],
    }
    manifest = {
        **manifest_payload,
        "manifest_sha256": canonical_sha256(manifest_payload),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "manifest_sha256": manifest["manifest_sha256"],
                "quality_claim": "budget_and_selection_diagnostic_only",
                "v2": {key: value for key, value in v2.items() if key != "pairs"},
                "v3_reference": v3,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (root / "v2_aob_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        pairs = v2["pairs"]
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    return {"manifest_sha256": manifest["manifest_sha256"], "v2": v2, "v3_reference": v3}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run_analysis(
        v2_root=args.v2_root,
        v3_root=args.v3_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
