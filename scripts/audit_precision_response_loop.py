"""Audit the frozen precision-response-loop-v1 coverage and pilot gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


PROTOCOL_VERSION = "precision-response-loop-v1"
PILOT_CASES = ("A4", "A5", "E1", "E2", "E3", "E4", "S2", "S5")
PILOT_SEEDS = (60, 61, 62, 63, 64)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _two_way_cluster_summary(
    rows: list[dict[str, str]],
    value_field: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int | None]:
    values = [
        (row["problem_id"], int(row["seed"]), float(row[value_field]))
        for row in rows
        if row.get(value_field, "")
    ]
    if not values:
        return {"n": 0, "mean": None, "lcb_95": None}
    point = sum(value for _, _, value in values) / len(values)
    cases = sorted({case for case, _, _ in values})
    seeds = sorted({seed_value for _, seed_value, _ in values})
    if len(cases) < 2 or len(seeds) < 2:
        return {"n": len(values), "mean": point, "lcb_95": None}
    rng = random.Random(seed)
    bootstrap_values: list[float] = []
    for _ in range(int(resamples)):
        case_weights = Counter(rng.choices(cases, k=len(cases)))
        seed_weights = Counter(rng.choices(seeds, k=len(seeds)))
        weighted_sum = 0.0
        total_weight = 0
        for case, seed_value, value in values:
            weight = case_weights[case] * seed_weights[seed_value]
            weighted_sum += weight * value
            total_weight += weight
        if total_weight:
            bootstrap_values.append(weighted_sum / total_weight)
    if not bootstrap_values:
        return {"n": len(values), "mean": point, "lcb_95": None}
    bootstrap_values.sort()
    lcb_index = max(0, math.ceil(0.05 * len(bootstrap_values)) - 1)
    return {
        "n": len(values),
        "mean": point,
        "lcb_95": bootstrap_values[lcb_index],
    }


def _integrity_gate(root: Path) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    manifest_path = root / "precision_response_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
        failures.append("missing_precision_response_manifest")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        failures.append("protocol_version_mismatch")
    if manifest.get("status") != "pass":
        failures.append("response_manifest_blocked")
    if (root / "causal_risk_precision_model.json").exists():
        failures.append("forbidden_model_bundle_present")

    branches = _read_csv(root / "precision_response_branch_manifest.csv")
    ledger = _read_csv(root / "same_budget_ledger.csv")
    aob = _read_csv(root / "aob_input_manifest.csv")
    leakage = _read_csv(root / "anti_leakage_audit.csv")
    if not branches or any(row.get("fresh_optimizer_execution") != "1" for row in branches):
        failures.append("not_all_runs_fresh")
    if not ledger or any(row.get("same_budget_violation") != "0" for row in ledger):
        failures.append("same_budget_violation")
    if not aob or any(row.get("unchanged") != "1" for row in aob):
        failures.append("aob_input_changed_or_missing")
    if not leakage or any(row.get("audit_status") != "pass" for row in leakage):
        failures.append("anti_leakage_failed_or_missing")
    return manifest, failures


def audit_precision_response(root: Path, *, resamples: int = 2000) -> dict[str, object]:
    root = root.resolve()
    _, integrity_failures = _integrity_gate(root)
    branches = _read_csv(root / "precision_response_branch_manifest.csv")
    triplets = _read_csv(root / "precision_response_triplets.csv")
    lease_credit = _read_csv(root / "precision_lease_credit.csv")

    a0_rows = [row for row in branches if row.get("response_arm") == "a0_v37"]
    applicable_a0 = [row for row in a0_rows if row.get("decision_status") == "applicable"]
    coverage_cases = {row["problem_id"] for row in applicable_a0}
    coverage_seeds = {int(row["seed"]) for row in applicable_a0}
    coverage_failures = []
    if len(applicable_a0) < 30:
        coverage_failures.append("applicable_below_30")
    if len(coverage_cases) < 6:
        coverage_failures.append("applicable_cases_below_6")
    if coverage_seeds != set(PILOT_SEEDS):
        coverage_failures.append("coverage_missing_registered_seed")
    coverage_pass = not integrity_failures and not coverage_failures

    complete_triplets = [
        row
        for row in triplets
        if row.get("triplet_integrity") == "1" and row.get("applicable") == "1"
    ]
    released = [row for row in complete_triplets if row.get("a2_released") == "1"]
    total_summary = _two_way_cluster_summary(
        complete_triplets,
        "tau_total",
        resamples=resamples,
        seed=2026071601,
    )
    lease_summary = _two_way_cluster_summary(
        released,
        "tau_lease",
        resamples=resamples,
        seed=2026071602,
    )
    pilot_failures: list[str] = []
    release_cases = {row["problem_id"] for row in released}
    release_seeds = {int(row["seed"]) for row in released}
    if len(released) < 10:
        pilot_failures.append("releases_below_10")
    if len(release_cases) < 4:
        pilot_failures.append("release_cases_below_4")
    if len(release_seeds) < 3:
        pilot_failures.append("release_seeds_below_3")
    if total_summary["lcb_95"] is None or float(total_summary["lcb_95"]) <= 0.0:
        pilot_failures.append("total_advantage_lcb_not_positive")
    if lease_summary["lcb_95"] is None or float(lease_summary["lcb_95"]) <= 0.0:
        pilot_failures.append("lease_advantage_lcb_not_positive")
    if not complete_triplets or statistics.median(
        float(row["tau_total"]) for row in complete_triplets
    ) < 0.0:
        pilot_failures.append("total_median_negative")
    if not released or statistics.median(float(row["tau_lease"]) for row in released) < 0.0:
        pilot_failures.append("lease_median_negative")

    positive_seed_count = 0
    for seed_value in PILOT_SEEDS:
        total_values = [
            float(row["tau_total"])
            for row in complete_triplets
            if int(row["seed"]) == seed_value
        ]
        lease_values = [
            float(row["tau_lease"])
            for row in released
            if int(row["seed"]) == seed_value
        ]
        if (
            total_values
            and lease_values
            and statistics.fmean(total_values) > 0.0
            and statistics.fmean(lease_values) > 0.0
        ):
            positive_seed_count += 1
    if positive_seed_count < 3:
        pilot_failures.append("positive_seed_strata_below_3")

    lease_wins = sum(float(row["tau_lease"]) > 0.0 for row in released)
    lease_losses = sum(float(row["tau_lease"]) < 0.0 for row in released)
    if lease_wins <= lease_losses:
        pilot_failures.append("released_wins_not_above_losses")
    material_positive = sum(
        row.get("lease_material_positive_1pct") == "1" for row in released
    )
    if material_positive < 5:
        pilot_failures.append("material_positive_lease_effects_below_5")
    catastrophic = sum(
        row.get(field) == "1"
        for row in complete_triplets
        for field in (
            "probe_catastrophic",
            "lease_catastrophic",
            "total_catastrophic",
        )
    )
    if catastrophic:
        pilot_failures.append("catastrophic_loss_observed")
    resolved_credit = sum(
        row.get("component_credit_status") == "resolved" for row in lease_credit
    )
    if len(released) != resolved_credit:
        pilot_failures.append("released_delayed_credit_not_fully_closed")

    treatment_present = {
        row.get("response_arm") for row in branches
    }.issuperset({"a1_probe_only", "a2_probe_gated"})
    if not treatment_present:
        status = "coverage_pass" if coverage_pass else "coverage_no_go"
    else:
        status = (
            "pilot_pass"
            if coverage_pass and not pilot_failures
            else "pilot_no_go"
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "runtime_scheduler_authorized": False,
        "full_24_authorized": False,
        "integrity": {
            "status": "pass" if not integrity_failures else "blocked",
            "failures": integrity_failures,
        },
        "coverage": {
            "status": "pass" if coverage_pass else "blocked",
            "applicable": len(applicable_a0),
            "cases": sorted(coverage_cases),
            "seeds": sorted(coverage_seeds),
            "failures": coverage_failures,
        },
        "pilot": {
            "status": "pass" if treatment_present and not pilot_failures else "blocked",
            "treatment_present": treatment_present,
            "releases": len(released),
            "release_cases": sorted(release_cases),
            "release_seeds": sorted(release_seeds),
            "total_advantage": total_summary,
            "lease_advantage": lease_summary,
            "positive_seed_strata": positive_seed_count,
            "lease_wins": lease_wins,
            "lease_losses": lease_losses,
            "material_positive_lease_effects": material_positive,
            "catastrophic_losses": catastrophic,
            "resolved_delayed_credit": resolved_credit,
            "failures": pilot_failures,
        },
        "hard_stop": "pilot_complete_no_runtime_or_full24_authorization",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = audit_precision_response(
        args.output_dir,
        resamples=args.bootstrap_resamples,
    )
    output_path = args.output_dir.resolve() / "precision_response_pilot_gate.json"
    output_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["status"] in {"coverage_pass", "pilot_pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
