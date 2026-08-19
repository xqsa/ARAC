"""Paired oracle diagnostic for persistent-conflict CTP core repair."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import LocalProposal, OverlapCoordinator, OverlapStructure
from arac.runtime.ledger import EvaluationLedger
from experiments.oracle_overlap_gate1 import (
    ANCHOR,
    GROUPS,
    _CountingObjective,
    _local_proposal,
    _problem,
)


DEFAULT_SEEDS = tuple(range(2026081501, 2026081526))
CTP_BUDGET_FES = 32


@dataclass(frozen=True)
class CtpTrialResult:
    case: str
    seed: int
    proposal_fes: int
    baseline_fes: int
    ctp_fes: int
    baseline_error: float
    ctp_error: float
    ctp_gain: float
    budget_matched: bool
    ctp_error_before_repair: float | None
    ctp_archive_nonworsening: bool
    first_streak: int
    second_streak: int
    ctp_triggered: bool
    ctp_consumed_fes: int
    conflict_score: float


def _build_proposals(case: str, seed: int, sample_fes: int):
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
    archive = min(
        [(ANCHOR.copy(), anchor_error), *((item[1], item[2]) for item in generated)],
        key=lambda item: item[1],
    )
    return problem, tuple(item[0] for item in generated), archive


def _ledger(problem: OptimizationProblem, archive: tuple[np.ndarray, float], budget: int):
    return EvaluationLedger(
        problem,
        total_budget=budget,
        initial_incumbent=tuple(float(value) for value in archive[0]),
        initial_error=float(archive[1]),
    )


def _independent_owner_continuation(
    ledger: EvaluationLedger,
    proposals: tuple[LocalProposal, ...],
    *,
    budget_fes: int,
    seed: int,
) -> int:
    """Spend an equal budget by continuing one owner at a time.

    This is the budget-matched control: each evaluation updates one local
    owner proposal while the other coordinates stay at the global incumbent.
    It cannot jointly sample the shared core.
    """

    if budget_fes <= 0:
        return 0
    rng = np.random.default_rng(seed)
    groups = tuple(sorted(proposal.group for proposal in proposals))
    by_group = {proposal.group: proposal for proposal in proposals}
    batch = np.repeat(ledger.best_x[np.newaxis, :], budget_fes, axis=0)
    shared_variables = tuple(sorted({1}))
    for index in range(budget_fes):
        group = groups[index % len(groups)]
        proposal = by_group[group]
        for variable in shared_variables:
            value = proposal.value(variable)
            batch[index, variable] = value + rng.normal(0.0, proposal.sigma(variable))
    batch = np.clip(batch, ledger.problem.lower_array, ledger.problem.upper_array)
    ledger.evaluate(batch)
    return budget_fes


def run_trial(case: str, seed: int, *, sample_fes: int = 512) -> CtpTrialResult:
    if case not in {"conforming", "conflicting"}:
        raise ValueError("case must be conforming or conflicting")
    proposal_fes = 1 + len(GROUPS) * sample_fes

    baseline_problem, baseline_proposals, baseline_archive = _build_proposals(case, seed, sample_fes)
    baseline = OverlapCoordinator(
            OverlapStructure(dimension=3, groups=GROUPS),
            _ledger(baseline_problem, baseline_archive, budget=8 + CTP_BUDGET_FES),
    )
    baseline_first = baseline.coordinate((0, 1), baseline_proposals)
    baseline_second = baseline.coordinate((0, 1), baseline_proposals)
    ctp_problem, ctp_proposals, ctp_archive = _build_proposals(case, seed, sample_fes)
    ctp_ledger = _ledger(ctp_problem, ctp_archive, budget=8 + CTP_BUDGET_FES)
    ctp = OverlapCoordinator(
        OverlapStructure(dimension=3, groups=GROUPS),
        ctp_ledger,
    )
    ctp_first = ctp.coordinate((0, 1), ctp_proposals, ctp_budget_fes=CTP_BUDGET_FES, ctp_seed=seed)
    ctp_second = ctp.coordinate((0, 1), ctp_proposals, ctp_budget_fes=CTP_BUDGET_FES, ctp_seed=seed)
    baseline_extra = 0
    if ctp_second.ctp_triggered:
        baseline_extra = _independent_owner_continuation(
            baseline.ledger,
            baseline_proposals,
            budget_fes=CTP_BUDGET_FES,
            seed=seed,
        )
    if baseline_extra != ctp_second.ctp_consumed_fes:
        raise RuntimeError("equal-FE owner control did not consume its declared budget")
    if baseline.ledger.count != ctp.ledger.count:
        raise RuntimeError("CTP and owner control consumed different total FE")
    if case == "conflicting" and (
        not ctp_second.ctp_triggered
        or ctp_second.ctp_consumed_fes != CTP_BUDGET_FES
    ):
        raise RuntimeError("persistent high conflict did not consume the declared CTP budget")
    if case == "conforming" and ctp_second.ctp_triggered:
        raise RuntimeError("conforming proposals unexpectedly triggered CTP")
    if baseline_first.conflict_level != ctp_first.conflict_level:
        raise RuntimeError("paired baseline and CTP did not observe the same first conflict")
    if baseline_second.conflict_level != ctp_second.conflict_level:
        raise RuntimeError("paired baseline and CTP did not observe the same second conflict")
    return CtpTrialResult(
        case=case,
        seed=seed,
        proposal_fes=proposal_fes,
        baseline_fes=baseline.ledger.count,
        ctp_fes=ctp.ledger.count,
        baseline_error=float(baseline.ledger.best_error),
        ctp_error=float(ctp.ledger.best_error),
        ctp_gain=float(baseline.ledger.best_error - ctp.ledger.best_error),
        budget_matched=baseline.ledger.count == ctp.ledger.count,
        ctp_error_before_repair=ctp_second.ctp_best_error_before,
        ctp_archive_nonworsening=(
            ctp_second.ctp_best_error_before is None
            or ctp_second.best_error_after <= ctp_second.ctp_best_error_before
        ),
        first_streak=ctp_first.conflict_streak,
        second_streak=ctp_second.conflict_streak,
        ctp_triggered=ctp_second.ctp_triggered,
        ctp_consumed_fes=ctp_second.ctp_consumed_fes,
        conflict_score=ctp_second.residuals[0].conflict_score,
    )


def _task(task: tuple[str, int, int]) -> CtpTrialResult:
    case, seed, sample_fes = task
    return run_trial(case, seed, sample_fes=sample_fes)


def _summary(trials: list[CtpTrialResult]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for case in ("conforming", "conflicting"):
        values = [trial for trial in trials if trial.case == case]
        gains = np.asarray([trial.ctp_gain for trial in values], dtype=float)
        summary[case] = {
            "runs": len(values),
            "ctp_trigger_rate": sum(trial.ctp_triggered for trial in values) / len(values),
            "budget_exact_rate": sum(
                trial.ctp_consumed_fes == (CTP_BUDGET_FES if trial.ctp_triggered else 0)
                for trial in values
            )
            / len(values),
            "median_conflict_score": float(np.median([trial.conflict_score for trial in values])),
            "ctp_wins": int(np.sum(gains > 1e-12)),
            "ctp_losses": int(np.sum(gains < -1e-12)),
            "ctp_ties": int(np.sum(np.abs(gains) <= 1e-12)),
            "median_ctp_gain": float(np.median(gains)),
            "ctp_archive_nonworsening": all(
                trial.ctp_archive_nonworsening for trial in values
            ),
            "budget_matched_rate": sum(trial.budget_matched for trial in values) / len(values),
        }
    conforming = summary["conforming"]
    conflicting = summary["conflicting"]
    checks = {
        "conforming_no_ctp_trigger": conforming["ctp_trigger_rate"] == 0.0,
        "conflicting_ctp_trigger_at_least_0_8": conflicting["ctp_trigger_rate"] >= 0.8,
        "conflicting_ctp_win_rate_at_least_0_6": conflicting["ctp_wins"] / conflicting["runs"] >= 0.6,
        "conflicting_budget_exact": conflicting["budget_exact_rate"] == 1.0,
        "strict_archive_nonworsening": (
            conforming["ctp_archive_nonworsening"]
            and conflicting["ctp_archive_nonworsening"]
        ),
    }
    summary["gate_checks"] = checks
    summary["gate_passed"] = all(checks.values())
    return summary


def run_diagnostic(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    sample_fes: int = 512,
    workers: int = 1,
) -> dict[str, object]:
    tasks = [(case, seed, sample_fes) for seed in seeds for case in ("conforming", "conflicting")]
    if workers == 1:
        trials = [_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trials = list(executor.map(_task, tasks))
    return {
        "schema_version": "arac-oracle-ctp-gate2-v1",
        "seeds": list(seeds),
        "sample_fes_per_group": sample_fes,
        "ctp_budget_fes": CTP_BUDGET_FES,
        "trials": [asdict(trial) for trial in trials],
        "summary": _summary(trials),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-fes", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("artifacts/oracle_ctp_gate2/result.json"))
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    payload = run_diagnostic(seeds, sample_fes=args.sample_fes, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
