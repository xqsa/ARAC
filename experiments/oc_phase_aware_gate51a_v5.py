"""Gate 51a-v5: four-case mechanism screening for HPR-GCB."""

from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from arac.benchmarks.aob import AobBenchmark
from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V5,
    run_oc_episode_schedule_v5,
)
from arac.runtime.manifest import implementation_manifest_hash
from experiments.oc_phase_aware_gate51a import (
    ACTION_SEED,
    CALIBRATION_PATH,
    _load_cached_phase_one,
    load_config as load_v4_config,
)

CASES = ("R2", "A3", "S5", "R6")
ARMS = ("on", "off")
TOTAL_FES = 600_000
OUTPUT_ROOT = Path("artifacts/oc_phase_aware_gate51a_v5")
CELL_SCHEMA = "arac-oc-gate51a-v5-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate51a-v5"
MANIFEST_FILES = [
    Path("src/arac/coordination/episodes.py"),
    Path("src/arac/actions/phase2_v2.py"),
    Path("src/arac/runtime/phase2.py"),
    Path("experiments/oc_phase_aware_gate51a_v5.py"),
]
# The first 600k run completed before the reporting-only fixes below.  Its
# scheduler dependency tree is unchanged, so the cells remain valid evidence;
# retain the exact hash as an explicit migration allowance rather than
# silently accepting arbitrary stale cells.
LEGACY_CELL_MANIFESTS = {
    "sha256:4c7316ac064527d735d82477a6a6084cd81ffe94bcab093ed79744af943c220c",
}


def _manifest() -> str:
    root = Path(__file__).resolve().parents[1]
    return implementation_manifest_hash(root, MANIFEST_FILES)


def load_config():
    config, calibration_hash = load_v4_config()
    return dataclasses.replace(
        config,
        scheduler_version=DEFAULT_SCHEDULER_VERSION_V5,
        horizon_protected=True,
    ), calibration_hash


def run_cell(case_id: str, arm: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint = _load_cached_phase_one(case_id)
    if checkpoint is None:
        raise RuntimeError(f"missing cached Phase-I checkpoint for {case_id}")
    checkpoint = dataclasses.replace(checkpoint, total_budget_fes=TOTAL_FES)
    config, calibration_hash = load_config()
    config = dataclasses.replace(config, handoff_enabled=(arm == "on"))
    result = run_oc_episode_schedule_v5(
        problem, checkpoint, action_seed=ACTION_SEED, config=config
    )
    reservations = [
        r for r in result.receipts if r.reservation_kind == "horizon"
    ]
    return {
        "case_id": case_id,
        "arm": arm,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "manifest_hash": _manifest(),
        "calibration_hash": calibration_hash,
        "result": {
            "schema_version": result.schema_version,
            "scheduler_policy": result.scheduler_policy,
            "scheduler_version": result.scheduler_version,
            "calibration_ref": result.calibration_ref,
            "final_error": result.final_error,
            "terminal_fes": result.terminal_fes,
            "schedule_hash": result.schedule_hash,
            "audit": result.audit,
            "funded_fes": result.funded_fes,
            "cold_start_probe_tax_fes": result.cold_start_probe_tax_fes,
            "development_fes": result.development_fes,
            "exploitation_fes": result.exploitation_fes,
            "switches": result.switches,
            "probe_order": list(result.probe_order),
            "tickets": [t.__dict__ for t in result.tickets],
            "probes": [p.__dict__ for p in result.probes],
            "handoffs": [h.__dict__ for h in result.handoffs],
            "receipts": [r.__dict__ for r in result.receipts],
            "horizon_reservation_count": len(reservations),
            "plateau_release_count": sum(
                1 for r in result.receipts if r.plateau_release
            ),
            "handoff_penalty_max": max(
                (r.handoff_penalty for r in result.receipts), default=0
            ),
            "magnitude_repairs": result.magnitude_repairs,
        },
    }


def _valid(path: Path, case_id: str, arm: str, manifest: str, calibration_hash: str) -> bool:
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    result = row.get("result", {})
    return (
        row.get("schema_version") == CELL_SCHEMA
        and result.get("case_id") == case_id
        and result.get("arm") == arm
        and result.get("manifest_hash") in {manifest, *LEGACY_CELL_MANIFESTS}
        and result.get("calibration_hash") == calibration_hash
        and result.get("result", {}).get("schema_version") == "arac-oc-episode-schedule-v5"
        and result.get("result", {}).get("terminal_fes") == TOTAL_FES
        and all(result.get("result", {}).get("audit", {}).values())
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config, calibration_hash = load_config()
    manifest = _manifest()
    cells_dir = OUTPUT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(case, arm) for case in CASES for arm in ARMS]
    rows: list[dict[str, object]] = []
    pending = []
    for case_id, arm in jobs:
        path = cells_dir / f"{case_id}_{arm}.json"
        if _valid(path, case_id, arm, manifest, calibration_hash):
            rows.append(json.loads(path.read_text(encoding="utf-8"))["result"])
        else:
            pending.append((case_id, arm))
    failures: list[dict[str, str]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_cell, *job): job for job in pending}
            for future in as_completed(futures):
                case_id, arm = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    failures.append({
                        "case": case_id,
                        "arm": arm,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"{case_id}/{arm}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    continue
                (cells_dir / f"{case_id}_{arm}.json").write_text(
                    json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{case_id}/{arm}: final={row['result']['final_error']:.6g} "
                    f"reservations={row['result']['horizon_reservation_count']} "
                    f"plateaus={row['result']['plateau_release_count']}",
                    flush=True,
                )

    by_case = {case: {} for case in CASES}
    for row in rows:
        by_case[row["case_id"]][row["arm"]] = row["result"]
    checks = {
        f"{case}_{arm}_complete": (
            arm in by_case[case]
            and by_case[case][arm]["terminal_fes"] == TOTAL_FES
            and all(by_case[case][arm]["audit"].values())
        )
        for case in CASES
        for arm in ARMS
    }
    summary = {
        case: {
            arm: by_case[case][arm]["final_error"]
            for arm in ARMS
            if arm in by_case[case]
        }
        for case in CASES
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "scheduler_version": DEFAULT_SCHEDULER_VERSION_V5,
        "calibration_ref": config.calibration_ref,
        "calibration_path": str(CALIBRATION_PATH),
        "manifest_hash": manifest,
        "protocol": {"cases": list(CASES), "arms": list(ARMS), "total_fes": TOTAL_FES},
        "checks": checks,
        "summary": summary,
        "failures": failures,
        "passed": not failures and len(rows) == len(jobs) and all(checks.values()),
    }
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
