"""Mechanism ablation screening: v4 (all-off) vs v5.3 (full-on) (G4c screening).

Protocol note (registered in .codex-tasks/arac-oc-evidence-closure/EPIC.md):
the requested seven-way off-by-one elimination would require patching the
frozen v5 scheduler internals.  The tree already provides the clean
coarse ladder via version-isolated entry points:

    v4_all_off      run_oc_episode_schedule_v4, frozen v4 config
                    (no horizon protection, no adaptive exploration,
                    no verification ladder, no adoption grace)
    v5_3_full_on    run_oc_episode_schedule_v5_3 (all mechanisms)

Only these two strata are runnable without touching production scheduler
code (the v5/v5.2 entry points are retired and raise).  Per the EPIC
interaction rule, a finer off-by-one campaign is registered ONLY if this
screening shows a stratum whose paired difference CI excludes zero.

Pre-registered falsifiable predictions (from the G5 diagnostic):

1. S5: the v4 arm funds CTP dominantly and recovers toward the off-level
   final error (budget/trigger failure confirmed at the stratum level);
2. A3: the v4 arm completes its schedule audit where v5.3-on failed
   (`ladder_flat_exposure_bounded`) -- if v4 also fails, the audit
   contract problem is not caused by the v5 stratum;
3. R2/R6: if the v4 arm keeps the v5.3 gains, the adaptive stratum is not
   what carries them; if the gains vanish, they belong to the v5 stratum.

Judgment: per config x case, median and worst final-error ratio vs the
best frozen standalone arm, on-vs-off (frozen v5.3-off cells as the common
no-steering reference), funded_fes per episode (CTP starvation check),
dev/exp split, and audit validity.  v5.3-on cells are reused from the
gate51c v5.3 artifact when valid; missing/audit-failed cells are
protocol_invalid and excluded from means with their count reported.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from arac.benchmarks.aob import AobBenchmark
from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V4,
    run_oc_episode_schedule_v4,
    run_oc_episode_schedule_v5_3,
)
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence
from arac.runtime.manifest import implementation_manifest_hash
from experiments.oc_phase_aware_gate51a import load_config as load_v4_config
from experiments.oc_phase_aware_gate51c import (
    ANYTIME_CHECKPOINTS,
    CASES,
    FRESH_SEEDS,
    NONINFERIOR_TOL,
    STANDALONE_ARMS,
    TOTAL_FES,
    WORST_SEED_TOL,
    _anytime_points,
    _log_error_auc,
    _sample_anytime,
)
from experiments.oc_phase_aware_gate51c_v5_3 import _parse_phase_one

OUTPUT_SCHEMA = "arac-oc-mechanism-ablation-screening-v1"
OUTPUT_ROOT = Path("artifacts/oc_mechanism_ablation_screening")
V53_ROOT = Path("artifacts/oc_phase_aware_gate51c_v5_3")
MANIFEST_FILES = [
    Path("src/arac/coordination/episodes.py"),
    Path("experiments/oc_mechanism_ablation_screening.py"),
]
CONFIGS = ("v4_all_off", "v5_3_full_on")


def _manifest() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return implementation_manifest_hash(repo_root, MANIFEST_FILES)


def _config_for(config_name: str):
    base = load_v4_config()[0]
    if config_name == "v4_all_off":
        return dataclasses.replace(
            base,
            scheduler_version=DEFAULT_SCHEDULER_VERSION_V4,
            horizon_protected=False,
            adaptive_exploration=False,
            handoff_enabled=True,
        )
    if config_name == "v5_3_full_on":
        return dataclasses.replace(
            base,
            scheduler_version="v5.3",
            horizon_protected=True,
            adaptive_exploration=True,
            handoff_enabled=True,
        )
    raise ValueError(f"unknown config: {config_name}")


def _phase_one(case_id: str, seed: int) -> PhaseCheckpoint:
    source = V53_ROOT / "phase1" / f"{case_id}_{seed}.json"
    if not source.exists():
        raise RuntimeError(f"Phase-I cache missing: {source}")
    return _parse_phase_one(json.loads(source.read_text(encoding="utf-8")))


def run_cell(config_name: str, case_id: str, seed: int, manifest: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint = _phase_one(case_id, seed)
    config = _config_for(config_name)
    if config_name == "v4_all_off":
        schedule = run_oc_episode_schedule_v4(
            problem, checkpoint, action_seed=seed, config=config
        )
    else:
        schedule = run_oc_episode_schedule_v5_3(
            problem, checkpoint, action_seed=seed, config=config
        )
    result: dict[str, object] = {
        "final_error": schedule.final_error,
        "terminal_fes": schedule.terminal_fes,
        "audit": schedule.audit,
        "funded_fes": schedule.funded_fes,
        "development_fes": schedule.development_fes,
        "exploitation_fes": schedule.exploitation_fes,
        "scheduler_version": schedule.scheduler_version,
    }
    points = _anytime_points(result, float(checkpoint.incumbent_error))
    result["anytime_points"] = points
    result["anytime"] = _sample_anytime(points)
    result["log_error_auc"] = _log_error_auc(points)
    return {
        "config": config_name,
        "case_id": case_id,
        "seed": seed,
        "checkpoint_error": float(checkpoint.incumbent_error),
        "manifest_hash": manifest,
        "result": result,
    }


def _cell_path(config_name: str, case_id: str, seed: int) -> Path:
    return OUTPUT_ROOT / "cells" / f"{config_name}_{case_id}_{seed}_on.json"


def _cell_reusable(config_name: str, case_id: str, seed: int, manifest: str) -> dict[str, object] | None:
    """Reuse valid v5.3-on cells from the frozen gate artifact."""

    if config_name != "v5_3_full_on":
        return None
    path = V53_ROOT / "cells" / f"{case_id}_{seed}_on.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))["result"]
    result = row["result"]
    if result.get("terminal_fes") != TOTAL_FES:
        return None
    if not all(result.get("audit", {}).values()):
        return None
    return {
        "config": config_name,
        "case_id": case_id,
        "seed": seed,
        "checkpoint_error": float(row.get("checkpoint_error", 0.0)) or None,
        "manifest_hash": "reused:oc_phase_aware_gate51c_v5_3",
        "result": result,
    }


def _frozen_reference(case_id: str, seed: int) -> dict[str, float] | None:
    """Common references: best standalone final error and the v5.3-off cell."""

    best: float | None = None
    for arm in STANDALONE_ARMS:
        path = V53_ROOT / "cells" / f"{case_id}_{seed}_{arm}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))["result"]["result"]
        value = float(row.get("final_error", math.inf))
        best = value if best is None else min(best, value)
    off_path = V53_ROOT / "cells" / f"{case_id}_{seed}_off.json"
    off = None
    if off_path.exists():
        off = float(
            json.loads(off_path.read_text(encoding="utf-8"))["result"]["result"]["final_error"]
        )
    if best is None or off is None:
        return None
    return {"best_standalone": best, "off": off}


def run_gate(workers: int = 8) -> dict[str, object]:
    manifest = _manifest()
    (OUTPUT_ROOT / "cells").mkdir(parents=True, exist_ok=True)
    jobs = [
        (config, case, seed)
        for config in CONFIGS
        for seed in FRESH_SEEDS
        for case in CASES
    ]
    rows: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    pending = []
    for config, case, seed in jobs:
        reused = _cell_reusable(config, case, seed, manifest)
        if reused is not None:
            rows.append(reused)
            continue
        own = _cell_path(config, case, seed)
        if own.exists():
            row = json.loads(own.read_text(encoding="utf-8"))["result"]
            if row["manifest_hash"] == manifest and row["result"]["terminal_fes"] == TOTAL_FES:
                rows.append(row)
                continue
        pending.append((config, case, seed))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_cell, config, case, seed, manifest): (config, case, seed)
                for config, case, seed in pending
            }
            for future in as_completed(futures):
                config, case, seed = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"{config}/{case}/{seed}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    invalid.append(
                        {
                            "config": config,
                            "case": case,
                            "seed": seed,
                            "classification": "protocol_invalid",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                _cell_path(config, case, seed).write_text(
                    json.dumps({"schema_version": OUTPUT_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{config}/{case}/{seed}: final={row['result']['final_error']:.6g}",
                    flush=True,
                )
    per_config: dict[str, dict[str, object]] = {}
    for config in CONFIGS:
        per_case: dict[str, dict[str, object]] = {}
        for case in CASES:
            ratios: list[float] = []
            on_off: list[float] = []
            ctp_share_on: list[float] = []
            ctp_share_off: list[float] = []
            audits_ok = 0
            total = 0
            for seed in FRESH_SEEDS:
                cell = next(
                    (
                        r
                        for r in rows
                        if r["config"] == config and r["case_id"] == case and r["seed"] == seed
                    ),
                    None,
                )
                reference = _frozen_reference(case, seed)
                if cell is None or reference is None:
                    continue
                total += 1
                result = cell["result"]
                audit_ok = all(result.get("audit", {}).values())
                audits_ok += int(audit_ok)
                final = float(result["final_error"])
                ratios.append(final / reference["best_standalone"])
                on_off.append(final / reference["off"])
                funded = result.get("funded_fes", {})
                funded_total = sum(funded.values()) or 1
                ctp_share_on.append(float(funded.get("ctp", 0)) / funded_total)
                off_cell = V53_ROOT / "cells" / f"{case}_{seed}_off.json"
                if off_cell.exists():
                    off_funded = (
                        json.loads(off_cell.read_text(encoding="utf-8"))["result"]["result"]
                        .get("funded_fes", {})
                    )
                    off_total = sum(off_funded.values()) or 1
                    ctp_share_off.append(float(off_funded.get("ctp", 0)) / off_total)
            if not ratios:
                per_case[case] = {"status": "no_valid_cells"}
                continue
            ordered = sorted(ratios)
            median = ordered[len(ordered) // 2] if len(ordered) % 2 else (
                (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
            )
            per_case[case] = {
                "status": "ok",
                "valid_cells": total,
                "audit_ok_cells": audits_ok,
                "median_ratio_vs_best_standalone": median,
                "worst_ratio_vs_best_standalone": max(ratios),
                "noninferior": median <= NONINFERIOR_TOL + 1e-9,
                "worst_seed_ok": max(ratios) <= WORST_SEED_TOL + 1e-9,
                "median_on_off_ratio": sorted(on_off)[len(on_off) // 2],
                "ctp_funded_share_on": sorted(ctp_share_on)[len(ctp_share_on) // 2] if ctp_share_on else None,
                "ctp_funded_share_off_reference": sorted(ctp_share_off)[len(ctp_share_off) // 2] if ctp_share_off else None,
            }
        per_config[config] = per_case
    checks = {
        "all_jobs_resolved": len(rows) + len(invalid) == len(jobs),
        "at_least_one_valid_cell_per_config": all(
            any(
                entry.get("status") == "ok"
                for entry in per_config[config].values()
            )
            for config in CONFIGS
        ),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "configs": list(CONFIGS),
            "cases": list(CASES),
            "seeds": list(FRESH_SEEDS),
            "total_fes": TOTAL_FES,
            "references": "frozen gate51c v5.3 standalone + off cells",
            "invalid_cell_policy": "audit failure or missing cell = protocol_invalid, excluded from means",
            "predictions_registered": [
                "S5: v4 funds CTP dominantly and recovers toward off-level error",
                "A3: v4 completes audit where v5.3-on failed",
                "R2/R6: gains persist under v4 iff they do not belong to the v5 stratum",
            ],
        },
        "implementation_manifest_hash": manifest,
        "per_config": per_config,
        "invalid_cells": invalid,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"per_config": payload["per_config"], "invalid_cells": payload["invalid_cells"], "gate_passed": payload["gate_passed"]}, indent=1, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
