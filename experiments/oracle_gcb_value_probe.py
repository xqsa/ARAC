"""Paired development gate for objective-aware GCB component dispatch."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler, OverlapCoordinator, OverlapStructure
from experiments.oracle_gcb_gate3 import (
    DISPATCH_BUDGET_FES,
    TOPOLOGIES,
    _build,
    _ledger,
    _problem,
    _repair_seed,
)


DEVELOPMENT_SEEDS = tuple(range(2026081801, 2026081826))


@dataclass(frozen=True)
class ValueProbeTrial:
    topology: str
    seed: int
    selected_component: tuple[int, ...]
    control_component: tuple[int, ...]
    selected_probe_gain: float
    control_probe_gain: float
    value_probe_error: float
    control_error: float
    value_probe_gain: float
    value_probe_ledger_fes: int
    control_ledger_fes: int
    value_probe_ctp_fes: int
    control_ctp_fes: int
    probes_identical: bool
    archive_nonworsening: bool


def _run(spec, proposals, seed: int, *, control: bool):
    problem = _problem(max(max(group) for group in spec.groups) + 1)
    coordinator = OverlapCoordinator(
        OverlapStructure(problem.dimension, spec.groups),
        _ledger(problem, problem.dimension, budget=100),
    )
    scheduler = GraphCoordinationScheduler(coordinator)
    scheduler.prime(proposals)
    forced = min(scheduler.overlap_components) if control else None
    result = scheduler.dispatch_value_probe(
        proposals,
        total_ctp_budget_fes=DISPATCH_BUDGET_FES,
        forced_component=forced,
        seed=_repair_seed(seed),
    )
    selected = result.events[0].component
    gains = {item.component: item.estimated_gain for item in result.value_probes}
    return selected, gains, float(coordinator.ledger.best_error), result


def run_trial(spec, seed: int) -> ValueProbeTrial:
    _, proposals = _build(spec, seed)
    selected, value_gains, value_error, value_result = _run(
        spec,
        proposals,
        seed,
        control=False,
    )
    control, control_gains, control_error, control_result = _run(
        spec,
        proposals,
        seed,
        control=True,
    )
    probes_identical = value_result.value_probes == control_result.value_probes
    archive_nonworsening = all(
        event.best_error_after <= event.best_error_before
        for event in (*value_result.events, *control_result.events)
    )
    return ValueProbeTrial(
        topology=spec.name,
        seed=seed,
        selected_component=selected,
        control_component=control,
        selected_probe_gain=value_gains[selected],
        control_probe_gain=control_gains[control],
        value_probe_error=value_error,
        control_error=control_error,
        value_probe_gain=control_error - value_error,
        value_probe_ledger_fes=value_result.ledger_fes,
        control_ledger_fes=control_result.ledger_fes,
        value_probe_ctp_fes=value_result.consumed_ctp_fes,
        control_ctp_fes=control_result.consumed_ctp_fes,
        probes_identical=probes_identical,
        archive_nonworsening=archive_nonworsening,
    )


def _task(task: tuple[str, int]) -> ValueProbeTrial:
    topology, seed = task
    spec = next(item for item in TOPOLOGIES if item.name == topology)
    return run_trial(spec, seed)


def _summary(trials: list[ValueProbeTrial]) -> dict[str, object]:
    result: dict[str, object] = {}
    for topology in (item.name for item in TOPOLOGIES):
        values = [trial for trial in trials if trial.topology == topology]
        gains = np.asarray([trial.value_probe_gain for trial in values], dtype=float)
        exact = all(
            trial.value_probe_ledger_fes == trial.control_ledger_fes == 48
            and trial.value_probe_ctp_fes == trial.control_ctp_fes == DISPATCH_BUDGET_FES
            for trial in values
        )
        result[topology] = {
            "runs": len(values),
            "wins": int(np.sum(gains > 1e-12)),
            "ties": int(np.sum(np.abs(gains) <= 1e-12)),
            "losses": int(np.sum(gains < -1e-12)),
            "median_gain": float(np.median(gains)),
            "exact_equal_budget": exact,
            "probes_identical": all(trial.probes_identical for trial in values),
            "archive_nonworsening": all(trial.archive_nonworsening for trial in values),
        }
    checks = {
        "all_topologies_have_90pct_wins_or_ties": all(
            (item["wins"] + item["ties"]) / item["runs"] >= 0.90
            for item in result.values()
        ),
        "all_budgets_exact_and_equal": all(item["exact_equal_budget"] for item in result.values()),
        "all_probes_identical": all(item["probes_identical"] for item in result.values()),
        "all_archives_nonworsening": all(item["archive_nonworsening"] for item in result.values()),
    }
    result["gate_checks"] = checks
    result["gate_passed"] = all(checks.values())
    return result


def run_diagnostic(seeds: tuple[int, ...] = DEVELOPMENT_SEEDS, *, workers: int = 1):
    tasks = [(topology.name, seed) for seed in seeds for topology in TOPOLOGIES]
    if workers == 1:
        trials = [_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trials = list(executor.map(_task, tasks))
    return {
        "schema_version": "arac-oracle-gcb-value-probe-v1",
        "seeds": list(seeds),
        "probe_fes_per_component": 2,
        "ctp_budget_fes": DISPATCH_BUDGET_FES,
        "trials": [asdict(trial) for trial in trials],
        "summary": _summary(trials),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=DEVELOPMENT_SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(DEVELOPMENT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("artifacts/oracle_gcb_value_probe/development.json"))
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    payload = run_diagnostic(seeds, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
