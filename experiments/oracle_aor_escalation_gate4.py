"""Oracle Gate 4 for escalation from shared-core CTP to full-space AOR."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import LocalProposal, OverlapCoordinator, OverlapStructure, compute_proposal_residuals
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import ResumableOptimizerSession
from experiments.oracle_overlap_gate1 import ANCHOR, GROUPS, _CountingObjective, _problem


DEVELOPMENT_SEEDS = tuple(range(2026082001, 2026082026))
FRESH_SEEDS = tuple(range(2026082101, 2026082126))
CTP_BUDGET_FES = 32
REFRESH_FES_PER_GROUP = 32
ESCALATION_BUDGET_FES = 64
HIGH_THRESHOLD = 2.0


@dataclass(frozen=True)
class TrialResult:
    case: str
    seed: int
    first_conflict_level: str
    post_ctp_conflict_score: float
    post_ctp_conflict_level: str
    escalation_triggered: bool
    ctp_error_before: float
    ctp_error_after: float
    aor_error: float
    control_error: float
    aor_gain: float
    ctp_consumed_fes: int
    refresh_consumed_fes: int
    aor_consumed_fes: int
    control_consumed_fes: int
    aor_total_ledger_fes: int
    control_total_ledger_fes: int
    aor_archive_nonworsening: bool
    control_archive_nonworsening: bool


def _initial_proposals(problem: OptimizationProblem, seed: int) -> tuple[LocalProposal, ...]:
    """Use the frozen Gate 1 proposal sampler without changing the gate boundary."""

    objective = problem.objective
    anchor_error = float(objective(ANCHOR))
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = [int(item.generate_state(1)[0]) for item in seed_sequence.spawn(2)]
    proposals: list[LocalProposal] = []
    for group, variables in enumerate(GROUPS):
        rng = np.random.default_rng(child_seeds[group])
        values = []
        for variable in variables:
            if variable == 1:
                value = (-1.25 if group == 0 else 1.25) + 0.10 * rng.random()
            else:
                value = 0.0
            values.append((variable, float(value)))
        proposals.append(
            LocalProposal(
                group=group,
                values=tuple(values),
                improvement=max(0.0, anchor_error),
                uncertainty=tuple((variable, 0.08) for variable in variables),
            )
        )
    return tuple(proposals)


def _refresh_proposals(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    seed: int,
    sample_fes_per_group: int = REFRESH_FES_PER_GROUP,
) -> tuple[LocalProposal, ...]:
    """Generate new owner-local proposals and count every refresh FE."""

    if sample_fes_per_group <= 0:
        raise ValueError("sample_fes_per_group must be positive")
    base = ledger.best_x
    base_error = ledger.best_error
    rngs = np.random.SeedSequence(seed).spawn(len(GROUPS))
    batches = []
    for group, variables in enumerate(GROUPS):
        rng = np.random.default_rng(int(rngs[group].generate_state(1)[0]))
        batch = np.repeat(base[np.newaxis, :], sample_fes_per_group, axis=0)
        batch[:, variables] += rng.normal(0.0, 0.55, size=(sample_fes_per_group, len(variables)))
        batches.append(np.clip(batch, problem.lower_array, problem.upper_array))
    combined = np.concatenate(batches, axis=0)
    errors = np.asarray(ledger.evaluate(combined), dtype=float)
    proposals = []
    for group, variables in enumerate(GROUPS):
        start = group * sample_fes_per_group
        stop = start + sample_fes_per_group
        local_errors = errors[start:stop]
        local_batch = combined[start:stop]
        order = np.argsort(local_errors)
        elite_count = max(4, sample_fes_per_group // 8)
        elite = local_batch[order[:elite_count]][:, variables]
        best = local_batch[order[0]]
        proposals.append(
            LocalProposal(
                group=group,
                values=tuple((variable, float(best[variable])) for variable in variables),
                improvement=max(0.0, base_error - float(local_errors[order[0]])),
                uncertainty=tuple(
                    (variable, float(max(np.std(elite[:, index], ddof=1), 1e-9)))
                    for index, variable in enumerate(variables)
                ),
            )
        )
    return tuple(proposals)


def _owner_continuation(
    ledger: EvaluationLedger,
    proposals: tuple[LocalProposal, ...],
    *,
    budget_fes: int,
    seed: int,
) -> int:
    rng = np.random.default_rng(seed)
    batch = np.repeat(ledger.best_x[np.newaxis, :], budget_fes, axis=0)
    for index in range(budget_fes):
        proposal = proposals[index % len(proposals)]
        for variable, value in proposal.values:
            batch[index, variable] = value + rng.normal(0.0, proposal.sigma(variable))
    batch = np.clip(batch, ledger.problem.lower_array, ledger.problem.upper_array)
    ledger.evaluate(batch)
    return budget_fes


def _aor_continuation(ledger: EvaluationLedger, *, budget_fes: int, seed: int) -> int:
    current = ledger.count
    session = ResumableOptimizerSession(
        "sepcmaes",
        problem=ledger.problem,
        ledger=ledger,
        initial_mean=ledger.best_x,
        sigma=0.5,
        seed=seed,
        budget_fes=current + budget_fes,
        population_size=16,
        initial_consumed=current,
        anchor=ledger.best_x,
    )
    session.step(budget_fes)
    return budget_fes


def run_trial(case: str, seed: int) -> TrialResult:
    if case not in {"conforming", "conflicting"}:
        raise ValueError("case must be conforming or conflicting")
    objective = _CountingObjective(conflicting=case == "conflicting")
    problem = _problem(objective)
    anchor_error = float(problem.objective(ANCHOR))
    initial = _initial_proposals(problem, seed)
    structure = OverlapStructure(dimension=3, groups=GROUPS)
    total_budget = 2048
    aor_ledger = EvaluationLedger(
        problem,
        total_budget,
        initial_incumbent=tuple(float(value) for value in ANCHOR),
        initial_error=anchor_error,
    )
    control_ledger = EvaluationLedger(
        problem,
        total_budget,
        initial_incumbent=tuple(float(value) for value in ANCHOR),
        initial_error=anchor_error,
    )
    aor_coord = OverlapCoordinator(structure, aor_ledger, high_threshold=HIGH_THRESHOLD)
    control_coord = OverlapCoordinator(structure, control_ledger, high_threshold=HIGH_THRESHOLD)
    aor_first = aor_coord.coordinate((0, 1), initial)
    control_first = control_coord.coordinate((0, 1), initial)
    if control_first.conflict_level != aor_first.conflict_level:
        raise RuntimeError("paired first conflict level drifted")
    aor_second = aor_coord.coordinate((0, 1), initial, ctp_budget_fes=CTP_BUDGET_FES, ctp_seed=seed + 41)
    control_second = control_coord.coordinate((0, 1), initial, ctp_budget_fes=CTP_BUDGET_FES, ctp_seed=seed + 41)
    if aor_second.ctp_consumed_fes != control_second.ctp_consumed_fes:
        raise RuntimeError("paired CTP FE drifted")
    ctp_error_before = aor_second.ctp_best_error_before
    if ctp_error_before is None:
        raise RuntimeError("persistent CTP did not expose a repair boundary")
    refreshed_aor = _refresh_proposals(problem, aor_ledger, seed=seed + 1000)
    refreshed_control = _refresh_proposals(problem, control_ledger, seed=seed + 1000)
    if refreshed_aor != refreshed_control:
        raise RuntimeError("post-CTP refresh proposals drifted between paired arms")
    residuals = compute_proposal_residuals(structure, refreshed_aor)
    score = max(item.conflict_score for item in residuals.values())
    level = "high" if score >= HIGH_THRESHOLD else "low"
    triggered = level == "high"
    aor_before = aor_ledger.best_error
    control_before = control_ledger.best_error
    if triggered:
        aor_fes = _aor_continuation(aor_ledger, budget_fes=ESCALATION_BUDGET_FES, seed=seed + 2000)
        control_fes = _owner_continuation(
            control_ledger,
            refreshed_control,
            budget_fes=ESCALATION_BUDGET_FES,
            seed=seed + 2000,
        )
    else:
        aor_fes = control_fes = 0
    return TrialResult(
        case=case,
        seed=seed,
        first_conflict_level=aor_first.conflict_level.value,
        post_ctp_conflict_score=float(score),
        post_ctp_conflict_level=level,
        escalation_triggered=triggered,
        ctp_error_before=float(ctp_error_before),
        ctp_error_after=float(aor_second.best_error_after),
        aor_error=float(aor_ledger.best_error),
        control_error=float(control_ledger.best_error),
        aor_gain=float(control_ledger.best_error - aor_ledger.best_error),
        ctp_consumed_fes=aor_second.ctp_consumed_fes,
        refresh_consumed_fes=2 * REFRESH_FES_PER_GROUP,
        aor_consumed_fes=aor_fes,
        control_consumed_fes=control_fes,
        aor_total_ledger_fes=aor_ledger.count,
        control_total_ledger_fes=control_ledger.count,
        aor_archive_nonworsening=aor_ledger.best_error <= aor_before,
        control_archive_nonworsening=control_ledger.best_error <= control_before,
    )


def _task(task: tuple[str, int]) -> TrialResult:
    case, seed = task
    return run_trial(case, seed)


def _summary(trials: list[TrialResult]) -> dict[str, object]:
    result = {}
    for case in ("conforming", "conflicting"):
        values = [item for item in trials if item.case == case]
        triggered = [item for item in values if item.escalation_triggered]
        gains = np.asarray([item.aor_gain for item in triggered], dtype=float)
        result[case] = {
            "runs": len(values),
            "escalation_trigger_rate": sum(item.escalation_triggered for item in values) / len(values),
            "aor_wins": int(np.sum(gains > 1e-12)),
            "aor_ties": int(np.sum(np.abs(gains) <= 1e-12)),
            "aor_losses": int(np.sum(gains < -1e-12)),
            "median_aor_gain": float(np.median(gains)) if len(gains) else 0.0,
            "exact_escalation_budget": all(
                item.aor_consumed_fes == item.control_consumed_fes == (ESCALATION_BUDGET_FES if item.escalation_triggered else 0)
                and item.aor_total_ledger_fes == item.control_total_ledger_fes
                for item in values
            ),
            "archive_nonworsening": all(
                item.aor_archive_nonworsening and item.control_archive_nonworsening for item in values
            ),
            "triggered_high_consistent": all(
                not item.escalation_triggered or item.post_ctp_conflict_level == "high"
                for item in values
            ),
        }
    conflicting_triggered = [
        item for item in trials
        if item.case == "conflicting" and item.escalation_triggered
    ]
    conflicting_triggered_gains = np.asarray(
        [item.aor_gain for item in conflicting_triggered],
        dtype=float,
    )
    checks = {
        "conforming_no_escalation_at_least_0_8": result["conforming"]["escalation_trigger_rate"] <= 0.2,
        "all_triggered_runs_are_high": all(
            item["triggered_high_consistent"] for item in result.values()
        ),
        "conflicting_aor_wins_or_ties_at_least_0_6": (
            (int(np.sum(conflicting_triggered_gains >= -1e-12)))
            / max(1, len(conflicting_triggered))
            >= 0.6
        ),
        "exact_budgets": all(item["exact_escalation_budget"] for item in result.values()),
        "archives_nonworsening": all(item["archive_nonworsening"] for item in result.values()),
    }
    result["gate_checks"] = checks
    result["gate_passed"] = all(checks.values())
    return result


def run_diagnostic(seeds: tuple[int, ...], *, workers: int = 1) -> dict[str, object]:
    tasks = [(case, seed) for seed in seeds for case in ("conforming", "conflicting")]
    if workers == 1:
        trials = [_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trials = list(executor.map(_task, tasks))
    return {
        "schema_version": "arac-oracle-aor-escalation-gate4-v1",
        "seeds": list(seeds),
        "ctp_budget_fes": CTP_BUDGET_FES,
        "refresh_fes_per_group": REFRESH_FES_PER_GROUP,
        "escalation_budget_fes": ESCALATION_BUDGET_FES,
        "trials": [asdict(item) for item in trials],
        "summary": _summary(trials),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=DEVELOPMENT_SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(DEVELOPMENT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("artifacts/oracle_aor_escalation_gate4/development.json"))
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    payload = run_diagnostic(seeds, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
