"""AOB-24 applicability audit for the sparse-overlap coordinator path.

Records, for every AOB case, whether Phase-I sparse overlap discovery can
produce complete variable-membership evidence inside the 180k-FE Phase-I
budget.  This documents the applicability boundary of the overlap
coordinator with receipts instead of silently forcing it onto functions
whose interaction density violates the sparse-overlap hypothesis.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from arac.benchmarks.aob import AobBenchmark

FAMILIES = ("A", "E", "R", "S")
AUDIT_SEED = 20260834
PHASE1_KWARGS = {
    "anchor_count": 5,
    "step": 0.25,
    "rounds": 12,
    "bucket_size": 16,
    "max_candidate_pairs": 128,
}
TOTAL_BUDGET_FES = 3_000_000
OUTPUT_SCHEMA = "arac-aob24-overlap-applicability-audit-v1"


def audit_case(case_id: str) -> dict[str, object]:
    from arac.evidence import run_phase1_overlap_pilot

    problem = AobBenchmark().load(case_id)
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=AUDIT_SEED,
        **PHASE1_KWARGS,
    )
    discovery = pilot.discovery
    structure = pilot.adaptation.structure
    return {
        "case_id": case_id,
        "dimension": problem.dimension,
        "phase1_fes": pilot.consumed_fes,
        "adapter_ready": bool(pilot.adaptation.ready),
        "adapter_reason": pilot.adaptation.reason,
        "discovery_complete_reason": discovery.complete_reason,
        "candidate_pair_count": discovery.candidate_pair_count,
        "separated_pair_fraction": discovery.separated_pair_fraction,
        "discovery_consumed_fes": discovery.consumed_fes,
        "inferred_groups": None if structure is None else len(structure.groups),
        "inferred_shared": None if structure is None else len(structure.shared_variables),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/aob24_overlap_applicability_audit/audit.json"),
    )
    args = parser.parse_args()
    cases = [f"{family}{index}" for family in FAMILIES for index in range(1, 7)]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_case, case): case for case in cases}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['case_id']}: ready={row['adapter_ready']} "
                f"reason={row['discovery_complete_reason']} "
                f"candidate_pairs={row['candidate_pair_count']}",
                flush=True,
            )
    rows.sort(key=lambda row: row["case_id"])
    ready_count = sum(1 for row in rows if row["adapter_ready"])
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "audit_seed": AUDIT_SEED,
            "phase1_kwargs": PHASE1_KWARGS,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "candidate_pair_cap": PHASE1_KWARGS["max_candidate_pairs"],
        },
        "summary": {
            "case_count": len(rows),
            "adapter_ready_count": ready_count,
            "fail_closed_count": len(rows) - ready_count,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
