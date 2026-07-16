"""Assemble phased precision-response coverage and treatment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ARAC_REPO_ROOT = Path(__file__).resolve().parents[1]
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"
for import_root in (ARAC_REPO_ROOT, ARAC_SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.pilots.exp_003_hcc_runtime_consumer_smoke.run import (
    PRECISION_RESPONSE_ARMS,
    PRECISION_RESPONSE_BRANCH_FIELDS,
    PRECISION_RESPONSE_TRIPLET_FIELDS,
    _precision_response_triplet_rows,
)


PROTOCOL_VERSION = "precision-response-loop-v1"
MERGED_CSV_ARTIFACTS = (
    "precision_probe_audit.csv",
    "precision_probe_gate_features.csv",
    "precision_lease_credit.csv",
    "same_budget_ledger.csv",
    "aob_input_manifest.csv",
    "anti_leakage_audit.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(root: Path, expected_arms: set[str]) -> dict[str, object]:
    path = root / "precision_response_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{root}: protocol version mismatch")
    if manifest.get("status") != "pass" or manifest.get("integrity_failures"):
        raise ValueError(f"{root}: source response manifest is blocked")
    if set(manifest.get("arms", [])) != expected_arms:
        raise ValueError(f"{root}: unexpected response arms")
    if (root / "causal_risk_precision_model.json").exists():
        raise ValueError(f"{root}: forbidden runtime model is present")
    return manifest


def _merge_csv_artifact(
    roots: tuple[Path, Path],
    name: str,
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] | None = None
    merged: list[dict[str, str]] = []
    for root in roots:
        source_fields, rows = _read_csv(root / name)
        if fieldnames is None:
            fieldnames = source_fields
        elif source_fields != fieldnames:
            raise ValueError(f"{name}: source schemas differ")
        merged.extend(rows)
    if not fieldnames:
        raise ValueError(f"{name}: empty schema")
    return fieldnames, merged


def assemble_precision_response_pilot(
    coverage_dir: Path,
    treatment_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    coverage = coverage_dir.resolve()
    treatment = treatment_dir.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if coverage == treatment or output in {coverage, treatment}:
        raise ValueError("coverage, treatment, and output directories must differ")

    coverage_manifest = _read_manifest(coverage, {"a0_v37"})
    treatment_manifest = _read_manifest(
        treatment,
        {"a1_probe_only", "a2_probe_gated"},
    )
    for field in ("config", "preregistration"):
        if coverage_manifest.get(field) != treatment_manifest.get(field):
            raise ValueError(f"source {field} manifests differ")

    coverage_fields, coverage_branches = _read_csv(
        coverage / "precision_response_branch_manifest.csv"
    )
    treatment_fields, treatment_branches = _read_csv(
        treatment / "precision_response_branch_manifest.csv"
    )
    if coverage_fields != PRECISION_RESPONSE_BRANCH_FIELDS:
        raise ValueError("coverage branch schema mismatch")
    if treatment_fields != PRECISION_RESPONSE_BRANCH_FIELDS:
        raise ValueError("treatment branch schema mismatch")
    branches = coverage_branches + treatment_branches
    branch_keys = [
        (row["problem_id"], row["seed"], row["response_arm"])
        for row in branches
    ]
    if len(branch_keys) != len(set(branch_keys)):
        raise ValueError("duplicate problem/seed/arm branch")
    if {row["response_arm"] for row in branches} != set(PRECISION_RESPONSE_ARMS):
        raise ValueError("assembled branches do not cover all response arms")

    triplets, triplet_failures = _precision_response_triplet_rows(branches)
    expected_triplets = len({(row["problem_id"], row["seed"]) for row in branches})
    if triplet_failures or len(triplets) != expected_triplets:
        details = ",".join(triplet_failures) or "missing_complete_triplet"
        raise ValueError(f"assembled triplet integrity failed: {details}")

    runtime_environment = json.loads(
        (coverage / "runtime_environment.json").read_text(encoding="utf-8")
    )
    treatment_environment = json.loads(
        (treatment / "runtime_environment.json").read_text(encoding="utf-8")
    )
    if runtime_environment != treatment_environment:
        raise ValueError("source runtime environments differ")

    merged_artifacts = {
        name: _merge_csv_artifact((coverage, treatment), name)
        for name in MERGED_CSV_ARTIFACTS
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "pass",
        "runtime_scheduler_authorized": False,
        "arms": list(PRECISION_RESPONSE_ARMS),
        "run_count": len(branches),
        "applicable_count": sum(
            row.get("decision_status") == "applicable" for row in branches
        ),
        "release_count": sum(
            row.get("response_arm") == "a2_probe_gated"
            and row.get("lease_applied") == "1"
            for row in branches
        ),
        "integrity_failures": [],
        "config": coverage_manifest["config"],
        "preregistration": coverage_manifest["preregistration"],
        "forbidden_outputs": coverage_manifest.get("forbidden_outputs", []),
        "assembly": {
            "mode": "phased_a0_then_a1_a2_no_overwrite",
            "coverage_dir": str(coverage),
            "coverage_manifest_sha256": _sha256(
                coverage / "precision_response_manifest.json"
            ),
            "treatment_dir": str(treatment),
            "treatment_manifest_sha256": _sha256(
                treatment / "precision_response_manifest.json"
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output / "precision_response_branch_manifest.csv",
        PRECISION_RESPONSE_BRANCH_FIELDS,
        branches,
    )
    _write_csv(
        output / "precision_response_triplets.csv",
        PRECISION_RESPONSE_TRIPLET_FIELDS,
        triplets,
    )
    for name, (fieldnames, rows) in merged_artifacts.items():
        _write_csv(output / name, fieldnames, rows)
    (output / "runtime_environment.json").write_text(
        json.dumps(runtime_environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "precision_response_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_dir", type=Path)
    parser.add_argument("treatment_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = assemble_precision_response_pilot(
        args.coverage_dir,
        args.treatment_dir,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
