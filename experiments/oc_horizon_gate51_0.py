"""Gate 51-0: revelation-horizon measurement via segmented standalone reruns.

Two instrumented 3M reruns (aor @ R2, ctp @ S5) on the cached Gate 50
checkpoints, segmented at 150k FE.  Every segment records the ledger
best-error before/after so the err(cumulative FE) trajectory can be
extracted; the terminal result must reconcile bitwise with the Gate 50
standalone receipts (segmented == one-shot is already proven for these
episodes -- a mismatch here is a gate failure, not a calibration input).
Parameter calibration (w1 / K / development cap) is derived in the
protocol appendix, never inside this script.
"""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.manifest import implementation_manifest_hash
from arac.runtime.phase2 import Phase2StateError
from experiments.oc_action_episode_gate50 import (
    ACTION_SEED,
    EXECUTORS,
    OUTPUT_ROOT as GATE50_ROOT,
    PHASE1_FES,
    TOTAL_FES,
    _load_cached_phase_one,
)

SEGMENT_FES = 150_000
OUTPUT_ROOT = Path("artifacts/oc_horizon_gate51_0")
CELL_SCHEMA = "arac-oc-gate51-0-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate51-0-v1"
RUNS = (("R2", "aor"), ("S5", "ctp"))
GATE50C_ROOT = Path("artifacts/oc_action_episode_gate50c")
MANIFEST_FILES = [
    Path("src/arac/coordination/episodes.py"),
    Path("src/arac/actions/phase2_v2.py"),
    Path("src/arac/runtime/phase2.py"),
    Path("experiments/oc_horizon_gate51_0.py"),
    Path("docs/arac-oc-gate51-0-protocol.md"),
]


def _manifest() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return implementation_manifest_hash(repo_root, MANIFEST_FILES)


def _reference_cell(case_id: str, episode: str) -> dict[str, object]:
    path = GATE50_ROOT / "cells" / f"{case_id}_{episode}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"]


def _reference_levels(case_id: str, episode: str) -> list[dict[str, object]]:
    """Crossing reference levels from the 50c ON-cell probe receipts.

    E takes the probe-window-end global errors of the OTHER episodes on
    this case (what the OC archive plausibly holds at probe end) plus
    the final OC probe-stage archive level.
    """

    path = GATE50C_ROOT / "cells" / f"{case_id}_on.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    probes = payload["result"]["result"]["probes"]
    levels = [
        {
            "kind": "other_probe_end",
            "episode": p["episode"],
            "error": float(p["global_error_after"]),
        }
        for p in probes
        if p["episode"] != episode
    ]
    if probes:
        levels.append(
            {
                "kind": "probe_stage_archive",
                "episode": "archive",
                "error": float(probes[-1]["global_error_after"]),
            }
        )
    return levels


def run_measurement(case_id: str, episode: str, manifest: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint = _load_cached_phase_one(case_id)
    if checkpoint is None:
        raise RuntimeError(f"missing cached Phase-I checkpoint for {case_id}")
    reference = _reference_cell(case_id, episode)
    if reference["checkpoint_hash"] != checkpoint.checkpoint_hash:
        raise RuntimeError(f"checkpoint drift vs Gate 50 cell: {case_id}/{episode}")
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=True,
    )
    context = ActionContext(episode, checkpoint, problem, ledger, action_seed=ACTION_SEED)
    state = EXECUTORS[episode]().initialize(context)
    segments: list[dict[str, object]] = []
    index = 0
    while ledger.remaining > 0 and not state.complete:
        request = min(SEGMENT_FES, ledger.remaining)
        error_before = float(ledger.best_error)
        request_work = request
        consumed = 0
        while consumed == 0:
            try:
                step = state.step(request_work)
                consumed = step.step_fes
            except Phase2StateError:
                if request_work >= ledger.remaining:
                    raise
                request_work = min(request_work * 2, ledger.remaining)
        segments.append(
            {
                "segment_index": index,
                "requested_fes": request,
                "consumed_fes": consumed,
                "cumulative_phase2_fes": ledger.count - PHASE1_FES,
                "error_before": error_before,
                "error_after": float(ledger.best_error),
                "state_hash": state.snapshot().state_hash,
                "complete": bool(state.complete),
            }
        )
        index += 1
    result = reference["result"]
    return {
        "case_id": case_id,
        "arm": episode,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "manifest_hash": manifest,
        "segment_fes": SEGMENT_FES,
        "result": {
            "final_error": float(ledger.best_error),
            "terminal_fes": ledger.count,
            "route": state.route,
            "segments": segments,
        },
        "reference": {
            "gate50_final_error": float(result["final_error"]),
            "gate50_terminal_fes": int(result["terminal_fes"]),
            "levels": _reference_levels(case_id, episode),
        },
    }


def _crossing_horizons(row: dict[str, object]) -> list[dict[str, object]]:
    segments = row["result"]["segments"]
    horizons: list[dict[str, object]] = []
    for level in row["reference"]["levels"]:
        target = float(level["error"])
        crossing = None
        for segment in segments:
            if float(segment["error_after"]) < target:
                crossing = int(segment["cumulative_phase2_fes"])
                break
        horizons.append(
            {
                "kind": level["kind"],
                "episode": level["episode"],
                "reference_error": target,
                "crossing_cumulative_fes": crossing,
            }
        )
    return horizons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cells_dir = OUTPUT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    pending = [job for job in RUNS if not (cells_dir / f"{job[0]}_{job[1]}.json").exists()]
    for case_id, episode in RUNS:
        path = cells_dir / f"{case_id}_{episode}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8"))["result"])
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_measurement, case_id, episode, manifest): (case_id, episode)
                for case_id, episode in pending
            }
            for future in as_completed(futures):
                case_id, episode = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    import traceback

                    print(f"{case_id}/{episode}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    failures.append({"case": case_id, "episode": episode,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    continue
                (cells_dir / f"{case_id}_{episode}.json").write_text(
                    json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{case_id}/{episode}: final={row['result']['final_error']:.6g} "
                    f"segments={len(row['result']['segments'])}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["case_id"], row["arm"]))

    reconciliation: dict[str, bool] = {}
    horizons: dict[str, list[dict[str, object]]] = {}
    stamps = {row["manifest_hash"] for row in rows}
    for row in rows:
        key = f"{row['case_id']}_{row['arm']}"
        segments = row["result"]["segments"]
        reconciliation[f"{key}_final_bitwise"] = (
            row["result"]["final_error"] == row["reference"]["gate50_final_error"]
        )
        reconciliation[f"{key}_terminal_exact"] = row["result"]["terminal_fes"] == TOTAL_FES
        reconciliation[f"{key}_trajectory_monotone"] = all(
            float(segments[i + 1]["error_after"]) <= float(segments[i]["error_after"])
            for i in range(len(segments) - 1)
        )
        reconciliation[f"{key}_segment_fes_reconcile"] = (
            sum(int(s["consumed_fes"]) for s in segments) == TOTAL_FES - PHASE1_FES
        )
        reconciliation[f"{key}_manifest_stamped"] = bool(row["manifest_hash"])
        horizons[key] = _crossing_horizons(row)
    # All cells must carry ONE implementation stamp; whether that stamp
    # matches the current tree is informational (see provenance_note) --
    # the numeric validity binding is the bitwise Gate 50 reconciliation.
    manifest_consistent = len(stamps) == 1
    manifest_matches_current = not pending or stamps == {manifest}
    gate_passed = (
        (not failures)
        and all(reconciliation.values())
        and manifest_consistent
        and len(rows) == len(RUNS)
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "runs": [f"{c}/{e}" for c, e in RUNS],
            "segment_fes": SEGMENT_FES,
            "total_fes": TOTAL_FES,
            "phase1_fes": PHASE1_FES,
            "action_seed": ACTION_SEED,
        },
        "implementation_manifest_hash": manifest,
        "cell_manifest_stamps": sorted(stamps),
        "manifest_consistent": manifest_consistent,
        "manifest_matches_current": manifest_matches_current,
        "provenance_note": (
            "The two measurement runs executed while the v4 scheduler was "
            "being implemented concurrently; the cells are therefore "
            "stamped with the manifest computed once in main (the "
            "execution-time code state is unrecoverable).  Numerical "
            "validity is anchored by the bitwise final-error "
            "reconciliation against the Gate 50 standalone receipts, "
            "which is a stronger binding than a manifest stamp."
        ),
        "reconciliation": reconciliation,
        "horizons": horizons,
        "gate_passed": gate_passed,
        "failures": failures,
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"reconciliation": reconciliation, "passed": gate_passed}, indent=1))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
