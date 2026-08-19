"""Isolate recovered SMP semantics at the retained Phase-I checkpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from arac.actions._execution import run_stateful_block_visits_with_sessions
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.current_smp_historical_parity import (
    _historical_stage_seed,
)
from experiments.historical_recovery.replay import _checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "historical_recovery_fixed_expert_v1"
    / "checkpoints"
    / "E1"
    / "seed_117"
    / "checkpoint.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "smp_checkpoint_handoff_ablation_v1"
SHORT_PHASE2_FES = 120_000
HISTORICAL_P90 = 1.8255606813339802

VARIANTS: dict[str, dict[str, bool]] = {
    "recovered": {
        "historical_seed": True,
        "clip_offspring": False,
        "precheck_incumbent": True,
        "strict_material_gain": True,
    },
    "clip": {
        "historical_seed": True,
        "clip_offspring": True,
        "precheck_incumbent": True,
        "strict_material_gain": True,
    },
    "no_precheck": {
        "historical_seed": True,
        "clip_offspring": False,
        "precheck_incumbent": False,
        "strict_material_gain": True,
    },
    "derived_seed": {
        "historical_seed": False,
        "clip_offspring": False,
        "precheck_incumbent": True,
        "strict_material_gain": True,
    },
    "identity_blind_bounded": {
        "historical_seed": False,
        "clip_offspring": True,
        "precheck_incumbent": True,
        "strict_material_gain": True,
    },
}


def _load_checkpoint(*, phase2_fes: int):
    wrapper = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    checkpoint = _checkpoint(wrapper["checkpoint"])
    if wrapper.get("checkpoint_hash") != checkpoint.checkpoint_hash:
        raise ValueError("retained checkpoint hash drifted")
    if checkpoint.phase1_fes != 180_000 or checkpoint.total_budget_fes != 3_000_000:
        raise ValueError("retained checkpoint budget drifted")
    return replace(
        checkpoint,
        total_budget_fes=checkpoint.phase1_fes + int(phase2_fes),
    )


def run_variant(name: str, phase2_fes: int = SHORT_PHASE2_FES) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"unknown SMP handoff variant: {name}")
    settings = VARIANTS[name]
    checkpoint = _load_checkpoint(phase2_fes=phase2_fes)
    problem = AobBenchmark().load("E1")
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=not settings["clip_offspring"],
    )
    context = ActionContext("smp", checkpoint, problem, ledger, action_seed=117)
    events: list[dict[str, object]] = []
    consumed, visits, resets, sessions = run_stateful_block_visits_with_sessions(
        context,
        requested_fes=phase2_fes,
        seed_factory=_historical_stage_seed if settings["historical_seed"] else None,
        clip_offspring=settings["clip_offspring"],
        precheck_incumbent=settings["precheck_incumbent"],
        strict_material_gain=settings["strict_material_gain"],
        event_trace=events,
    )
    noop_fes = 0
    while context.ledger.remaining:
        context.ledger.evaluate(context.ledger.best_x)
        noop_fes += 1
    body = {
        "schema_version": "arac-smp-checkpoint-handoff-ablation-receipt-v1",
        "variant": name,
        "settings": settings,
        "source_checkpoint_hash": json.loads(
            CHECKPOINT_PATH.read_text(encoding="utf-8")
        )["checkpoint_hash"],
        "screen_checkpoint_hash": checkpoint.checkpoint_hash,
        "phase1_fes": checkpoint.phase1_fes,
        "phase2_fes": phase2_fes,
        "terminal_fes": ledger.count,
        "initial_error": checkpoint.incumbent_error,
        "final_error": ledger.best_error,
        "consumed_fes": consumed,
        "noop_fes": noop_fes,
        "visit_count": visits,
        "reset_count": resets,
        "cold_start_count": sum(event["route"] == "cold_start" for event in events),
        "restore_count": sum(event["route"] == "restore" for event in events),
        "terminal_state_finite": all(session.optimizer.sigma > 0.0 for session in sessions),
        "historical_p90": HISTORICAL_P90,
    }
    return {**body, "receipt_hash": canonical_sha256(body)}


def run_screen(
    output_root: Path = DEFAULT_OUTPUT,
    *,
    variants: tuple[str, ...] = tuple(VARIANTS),
    phase2_fes: int = SHORT_PHASE2_FES,
    max_workers: int = 4,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")
    if not variants or any(name not in VARIANTS for name in variants):
        raise ValueError("screen variants must be known and non-empty")
    if phase2_fes <= 0 or phase2_fes > 2_820_000:
        raise ValueError("screen Phase-II FE must be in (0, 2,820,000]")
    receipts = output_root / "receipts"
    receipts.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(variants))) as executor:
        futures = {
            executor.submit(run_variant, name, phase2_fes): name for name in variants
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            path = receipts / f"{row['variant']}.json"
            path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows.sort(key=lambda row: row["final_error"])
    summary_body = {
        "schema_version": "arac-smp-checkpoint-handoff-ablation-summary-v1",
        "phase1_fes": 180_000,
        "phase2_fes": phase2_fes,
        "rows": [
            {
                "variant": row["variant"],
                "final_error": row["final_error"],
                "visit_count": row["visit_count"],
                "reset_count": row["reset_count"],
                "receipt_hash": row["receipt_hash"],
            }
            for row in rows
        ],
        "best_variant": rows[0]["variant"],
    }
    summary = {**summary_body, "summary_hash": canonical_sha256(summary_body)}
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("screen", "variant"))
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS))
    parser.add_argument("--phase2-fes", type=int, default=SHORT_PHASE2_FES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.command == "screen":
        result = run_screen(
            args.output_root,
            variants=tuple(args.variants or VARIANTS),
            phase2_fes=args.phase2_fes,
            max_workers=args.max_workers,
        )
    else:
        if args.variant is None:
            parser.error("--variant is required for variant")
        result = run_variant(args.variant, args.phase2_fes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
