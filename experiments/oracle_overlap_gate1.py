"""Paired oracle diagnostic for the minimal overlap coordinator.

This experiment isolates Phase-II coordination on two groups sharing one
variable.  It does not invoke Phase-I or any of the four production actions.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import LocalProposal, OverlapCoordinator, OverlapStructure
from arac.runtime.ledger import EvaluationLedger


DEFAULT_SEEDS = tuple(range(2026081301, 2026081326))
ANCHOR = np.asarray((2.0, 0.0, 2.0), dtype=float)
GROUPS = ((0, 1), (1, 2))


class _CountingObjective:
    def __init__(self, conflicting: bool) -> None:
        self.conflicting = conflicting
        self.count = 0

    def __call__(self, candidate: np.ndarray) -> float | np.ndarray:
        values = np.asarray(candidate, dtype=float)
        batch = values[np.newaxis, :] if values.ndim == 1 else values
        self.count += int(batch.shape[0])
        x0, shared, x2 = batch.T
        second_coupling = shared + x2 if self.conflicting else shared - x2
        results = (
            (x0 - 1.0) ** 2
            + (shared - x0) ** 2
            + (x2 - 1.0) ** 2
            + second_coupling**2
        )
        return float(results[0]) if values.ndim == 1 else results


def _problem(objective: Callable[[np.ndarray], float | np.ndarray]) -> OptimizationProblem:
    return OptimizationProblem(
        objective=objective,
        dimension=3,
        lower_bounds=(-3.0, -3.0, -3.0),
        upper_bounds=(3.0, 3.0, 3.0),
    )


def _local_proposal(
    problem: OptimizationProblem,
    group: int,
    *,
    seed: int,
    sample_fes: int,
    anchor_error: float,
) -> tuple[LocalProposal, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    variables = GROUPS[group]
    samples = np.repeat(ANCHOR[np.newaxis, :], sample_fes, axis=0)
    samples[:, variables] = rng.uniform(-3.0, 3.0, size=(sample_fes, len(variables)))
    ledger = EvaluationLedger(
        problem,
        total_budget=sample_fes,
        initial_incumbent=tuple(float(value) for value in ANCHOR),
        initial_error=anchor_error,
    )
    errors = np.asarray(ledger.evaluate(samples), dtype=float)
    elite_count = max(8, sample_fes // 16)
    elite = samples[np.argsort(errors)[:elite_count]][:, variables]
    uncertainty = np.maximum(np.std(elite, axis=0, ddof=1), 1e-9)
    best = ledger.best_x
    proposal = LocalProposal(
        group=group,
        values=tuple((variable, float(best[variable])) for variable in variables),
        improvement=max(0.0, anchor_error - float(ledger.best_error)),
        uncertainty=tuple(
            (variable, float(sigma))
            for variable, sigma in zip(variables, uncertainty, strict=True)
        ),
    )
    return proposal, best, float(ledger.best_error)


@dataclass(frozen=True)
class TrialResult:
    case: str
    seed: int
    proposal_fes: int
    arbitration_fes: int
    total_fes: int
    conflict_score: float
    conflict_level: str
    proposal_values: tuple[float, float]
    proposal_uncertainties: tuple[float, float]
    owner_error: float
    uncoordinated_error: float
    coordinated_error: float
    accepted_candidate: str | None

    @property
    def coordination_gain(self) -> float:
        return self.uncoordinated_error - self.coordinated_error


def run_trial(
    case: str,
    seed: int,
    *,
    sample_fes: int = 512,
    medium_threshold: float = 1.0,
    high_threshold: float = 2.0,
) -> TrialResult:
    if case not in {"conforming", "conflicting"}:
        raise ValueError("case must be conforming or conflicting")
    if sample_fes < 32:
        raise ValueError("sample_fes must be at least 32")
    objective = _CountingObjective(conflicting=case == "conflicting")
    problem = _problem(objective)
    anchor_error = float(problem.objective(ANCHOR))
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = [int(item.generate_state(1)[0]) for item in seed_sequence.spawn(2)]
    generated = [
        _local_proposal(
            problem,
            group,
            seed=child_seeds[group],
            sample_fes=sample_fes,
            anchor_error=anchor_error,
        )
        for group in range(2)
    ]
    proposals = tuple(item[0] for item in generated)
    branch_archive = min(
        [(ANCHOR.copy(), anchor_error), *((item[1], item[2]) for item in generated)],
        key=lambda item: item[1],
    )
    proposal_fes = 1 + len(GROUPS) * sample_fes
    arbitration_ledger = EvaluationLedger(
        problem,
        total_budget=4,
        initial_incumbent=tuple(float(value) for value in branch_archive[0]),
        initial_error=float(branch_archive[1]),
    )
    coordinator = OverlapCoordinator(
        OverlapStructure(dimension=3, groups=GROUPS),
        arbitration_ledger,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    result = coordinator.coordinate((0, 1), proposals)
    errors = dict(result.candidate_errors)
    owner_error = float(errors["owner"])
    uncoordinated_error = min(float(branch_archive[1]), owner_error)
    residual = result.residuals[0]
    aggregate_fes = proposal_fes + arbitration_ledger.count
    if objective.count != aggregate_fes:
        raise RuntimeError("objective and aggregate FE ledgers disagree")
    return TrialResult(
        case=case,
        seed=seed,
        proposal_fes=proposal_fes,
        arbitration_fes=len(result.candidates),
        total_fes=aggregate_fes,
        conflict_score=residual.conflict_score,
        conflict_level=result.conflict_level.value,
        proposal_values=(proposals[0].value(1), proposals[1].value(1)),
        proposal_uncertainties=(proposals[0].sigma(1), proposals[1].sigma(1)),
        owner_error=owner_error,
        uncoordinated_error=uncoordinated_error,
        coordinated_error=result.best_error_after,
        accepted_candidate=result.accepted_candidate,
    )


def _summarize(trials: list[TrialResult]) -> dict[str, object]:
    summary: dict[str, object] = {}
    by_case = {
        case: [trial for trial in trials if trial.case == case]
        for case in ("conforming", "conflicting")
    }
    for case, values in by_case.items():
        gains = np.asarray([trial.coordination_gain for trial in values], dtype=float)
        scores = np.asarray([trial.conflict_score for trial in values], dtype=float)
        summary[case] = {
            "runs": len(values),
            "low_rate": sum(trial.conflict_level == "low" for trial in values) / len(values),
            "high_rate": sum(trial.conflict_level == "high" for trial in values) / len(values),
            "median_conflict_score": float(np.median(scores)),
            "coordination_wins": int(np.sum(gains > 1e-12)),
            "coordination_losses": int(np.sum(gains < -1e-12)),
            "coordination_ties": int(np.sum(np.abs(gains) <= 1e-12)),
            "median_coordination_gain": float(np.median(gains)),
            "strict_best_monotone": bool(np.all(gains >= -1e-12)),
        }
    conforming = summary["conforming"]
    conflicting = summary["conflicting"]
    gate_checks = {
        "conforming_low_rate_at_least_0_8": conforming["low_rate"] >= 0.8,
        "conflicting_high_rate_at_least_0_8": conflicting["high_rate"] >= 0.8,
        "conflicting_coordination_win_rate_at_least_0_6": (
            conflicting["coordination_wins"] / conflicting["runs"] >= 0.6
        ),
        "strict_best_monotone_in_all_runs": (
            conforming["strict_best_monotone"] and conflicting["strict_best_monotone"]
        ),
    }
    summary["gate_checks"] = gate_checks
    summary["gate_passed"] = all(gate_checks.values())
    return summary


def run_diagnostic(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    sample_fes: int = 512,
    workers: int = 1,
    medium_threshold: float = 1.0,
    high_threshold: float = 2.0,
) -> dict[str, object]:
    tasks = [
        (case, seed, sample_fes, medium_threshold, high_threshold)
        for seed in seeds
        for case in ("conforming", "conflicting")
    ]
    if workers == 1:
        trials = [
            run_trial(
                case,
                seed,
                sample_fes=budget,
                medium_threshold=medium,
                high_threshold=high,
            )
            for case, seed, budget, medium, high in tasks
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trials = list(executor.map(_run_trial_task, tasks))
    return {
        "schema_version": "arac-oracle-overlap-gate1-v1",
        "seeds": list(seeds),
        "sample_fes_per_group": sample_fes,
        "medium_threshold": medium_threshold,
        "high_threshold": high_threshold,
        "trials": [asdict(trial) | {"coordination_gain": trial.coordination_gain} for trial in trials],
        "summary": _summarize(trials),
    }


def _run_trial_task(task: tuple[str, int, int, float, float]) -> TrialResult:
    case, seed, sample_fes, medium_threshold, high_threshold = task
    return run_trial(
        case,
        seed,
        sample_fes=sample_fes,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-fes", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--medium-threshold", type=float, default=1.0)
    parser.add_argument("--high-threshold", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oracle_overlap_gate1/result.json"),
    )
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    payload = run_diagnostic(
        seeds,
        sample_fes=args.sample_fes,
        workers=args.workers,
        medium_threshold=args.medium_threshold,
        high_threshold=args.high_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
