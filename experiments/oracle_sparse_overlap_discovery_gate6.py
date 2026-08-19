"""Fresh oracle-vs-inferred Gate 6 for sparse overlap discovery."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.evidence import DEFAULT_EDGE_THRESHOLD, Phase1OverlapAdapter, discover_overlap_sparse
from arac.runtime.ledger import EvaluationLedger
if __package__:
    from experiments.oracle_overlap_discovery_gate5 import (
        BASE_FUNCTIONS,
        MODES,
        TOPOLOGIES,
        _checkpoint,
        _problem,
    )
else:
    from oracle_overlap_discovery_gate5 import (  # type: ignore[no-redef]
        BASE_FUNCTIONS,
        MODES,
        TOPOLOGIES,
        _checkpoint,
        _problem,
    )


DEFAULT_SEEDS = tuple(range(2026081301, 2026081306))
ANCHOR_COUNT = 5
PROBE_STEP = 10.0
ROUNDS = 12
BUCKET_SIZE = 4
MAX_CANDIDATE_PAIRS = 128
PILOT_DIMENSION = 64
PILOT_ACTIVE_GROUPS = ((0, 1, 2), (2, 3, 4), (10, 11, 12), (12, 13, 14))
PILOT_GROUPS = PILOT_ACTIVE_GROUPS + tuple(
    (variable,)
    for variable in range(PILOT_DIMENSION)
    if not any(variable in group for group in PILOT_ACTIVE_GROUPS)
)


@dataclass(frozen=True)
class DiscoveryTrial:
    seed: int
    topology: str
    base_function: str
    conflict_mode: str
    dimension: int
    expected_groups: tuple[tuple[int, ...], ...]
    inferred_groups: tuple[tuple[int, ...], ...]
    expected_shared: tuple[int, ...]
    inferred_shared: tuple[int, ...]
    group_exact: bool
    shared_precision: float
    shared_recall: float
    candidate_pair_count: int
    consumed_fes: int
    expected_fes: int
    full_pair_fes: int
    separated_pair_fraction: float
    adapter_ready: bool
    deterministic: bool


def _pilot_trial(seed: int) -> DiscoveryTrial:
    """Run the declared d=64 sparse-complexity pilot."""

    topology = "d64_sparse_pilot"
    dimension = PILOT_DIMENSION
    groups = PILOT_GROUPS
    problem = _problem(
        dimension,
        groups,
        base_function="ackley",
        conflict_mode="conforming",
        seed=seed,
    )
    anchor_rng = np.random.default_rng(seed ^ 0x5A17)
    anchors = tuple(
        tuple(float(value) for value in row)
        for row in anchor_rng.uniform(-2.0, 2.0, size=(3, dimension))
    )
    full_pair_fes = len(anchors) * (1 + dimension + dimension * (dimension - 1) // 2)
    budget = full_pair_fes * 2
    result = discover_overlap_sparse(
        problem,
        EvaluationLedger(problem, budget),
        anchors=anchors,
        step=0.25,
        run_seed=seed,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=MAX_CANDIDATE_PAIRS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
    )
    replay = discover_overlap_sparse(
        problem,
        EvaluationLedger(problem, budget),
        anchors=anchors,
        step=0.25,
        run_seed=seed,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=MAX_CANDIDATE_PAIRS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
    )
    expected_shared = (2, 12)
    inferred_shared = tuple(
        variable
        for variable, owners in enumerate(result.evidence.memberships)
        if len(owners) > 1
    )
    true_shared = set(expected_shared)
    found_shared = set(inferred_shared)
    return DiscoveryTrial(
        seed=seed,
        topology=topology,
        base_function="ackley",
        conflict_mode="conforming",
        dimension=dimension,
        expected_groups=groups,
        inferred_groups=result.evidence.groups,
        expected_shared=expected_shared,
        inferred_shared=inferred_shared,
        group_exact=set(result.evidence.groups) == set(groups),
        shared_precision=len(true_shared & found_shared) / max(len(found_shared), 1),
        shared_recall=len(true_shared & found_shared) / max(len(true_shared), 1),
        candidate_pair_count=result.candidate_pair_count,
        consumed_fes=result.consumed_fes,
        expected_fes=result.expected_fes,
        full_pair_fes=full_pair_fes,
        separated_pair_fraction=result.separated_pair_fraction,
        adapter_ready=Phase1OverlapAdapter().adapt(_checkpoint(dimension), result.evidence).ready,
        deterministic=result == replay,
    )


def _trial(task: tuple[int, str, str, str]) -> DiscoveryTrial:
    seed, topology, base_function, conflict_mode = task
    dimension, groups = TOPOLOGIES[topology]
    problem = _problem(
        dimension,
        groups,
        base_function=base_function,
        conflict_mode=conflict_mode,
        seed=seed,
    )
    anchor_rng = np.random.default_rng(seed ^ 0x5A17)
    anchors = tuple(
        tuple(float(value) for value in row)
        for row in anchor_rng.uniform(-50.0, 50.0, size=(ANCHOR_COUNT, dimension))
    )
    full_pair_fes = ANCHOR_COUNT * (1 + dimension + dimension * (dimension - 1) // 2)
    budget = max(full_pair_fes * 2, 10_000)
    result = discover_overlap_sparse(
        problem,
        EvaluationLedger(problem, budget),
        anchors=anchors,
        step=PROBE_STEP,
        run_seed=seed,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=MAX_CANDIDATE_PAIRS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
    )
    replay = discover_overlap_sparse(
        problem,
        EvaluationLedger(problem, budget),
        anchors=anchors,
        step=PROBE_STEP,
        run_seed=seed,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=MAX_CANDIDATE_PAIRS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
    )
    expected_shared = tuple(
        variable
        for variable in range(dimension)
        if sum(variable in group for group in groups) > 1
    )
    inferred_shared = tuple(
        variable
        for variable, owners in enumerate(result.evidence.memberships)
        if len(owners) > 1
    )
    true_shared = set(expected_shared)
    found_shared = set(inferred_shared)
    precision = len(true_shared & found_shared) / max(len(found_shared), 1)
    recall = len(true_shared & found_shared) / max(len(true_shared), 1)
    adapted = Phase1OverlapAdapter().adapt(_checkpoint(dimension), result.evidence)
    return DiscoveryTrial(
        seed=seed,
        topology=topology,
        base_function=base_function,
        conflict_mode=conflict_mode,
        dimension=dimension,
        expected_groups=groups,
        inferred_groups=result.evidence.groups,
        expected_shared=expected_shared,
        inferred_shared=inferred_shared,
        group_exact=set(result.evidence.groups) == set(groups),
        shared_precision=precision,
        shared_recall=recall,
        candidate_pair_count=result.candidate_pair_count,
        consumed_fes=result.consumed_fes,
        expected_fes=result.expected_fes,
        full_pair_fes=full_pair_fes,
        separated_pair_fraction=result.separated_pair_fraction,
        adapter_ready=adapted.ready,
        deterministic=result == replay,
    )


def _summary(trials: list[DiscoveryTrial], pilots: list[DiscoveryTrial] | None = None) -> dict[str, object]:
    pilots = [] if pilots is None else pilots
    identifiable = [trial for trial in trials if trial.topology != "disjoint"]
    disjoint = [trial for trial in trials if trial.topology == "disjoint"]
    checks = {
        "all_complete": all(trial.consumed_fes == trial.expected_fes for trial in trials),
        "identifiable_groups_exact": all(trial.group_exact for trial in identifiable),
        "identifiable_shared_precision_one": all(trial.shared_precision == 1.0 for trial in identifiable),
        "identifiable_shared_recall_one": all(trial.shared_recall == 1.0 for trial in identifiable),
        "disjoint_no_shared_false_positive": all(not trial.inferred_shared for trial in disjoint),
        "coverage_complete": all(trial.separated_pair_fraction == 1.0 for trial in trials),
        "candidate_cap_respected": all(trial.candidate_pair_count <= MAX_CANDIDATE_PAIRS for trial in trials + pilots),
        "adapter_ready": all(trial.adapter_ready for trial in trials),
        "deterministic": all(trial.deterministic for trial in trials),
        "large_dimension_pilot_exact": all(trial.group_exact and trial.shared_precision == trial.shared_recall == 1.0 for trial in pilots),
        "large_dimension_pilot_sparse": all(trial.consumed_fes < trial.full_pair_fes for trial in pilots),
        "large_dimension_pilot_coverage": all(trial.separated_pair_fraction == 1.0 for trial in pilots),
        "large_dimension_pilot_deterministic": all(trial.deterministic for trial in pilots),
    }
    return {
        "runs": len(trials),
        "identifiable_runs": len(identifiable),
        "disjoint_runs": len(disjoint),
        "pilot_runs": len(pilots),
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def run_diagnostic(*, seeds: tuple[int, ...] = DEFAULT_SEEDS, workers: int = 1) -> dict[str, object]:
    tasks = [
        (seed, topology, base_function, mode)
        for seed in seeds
        for topology in TOPOLOGIES
        for base_function in BASE_FUNCTIONS
        for mode in MODES
    ]
    if workers == 1:
        trials = [_trial(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trials = list(executor.map(_trial, tasks))
    pilots = [_pilot_trial(seed) for seed in seeds]
    return {
        "schema_version": "arac-oracle-sparse-overlap-discovery-gate6-v1",
        "seeds": list(seeds),
        "anchor_count": ANCHOR_COUNT,
        "probe_step": PROBE_STEP,
        "rounds": ROUNDS,
        "bucket_size": BUCKET_SIZE,
        "max_candidate_pairs": MAX_CANDIDATE_PAIRS,
        "trials": [asdict(trial) for trial in trials],
        "pilots": [asdict(trial) for trial in pilots],
        "summary": _summary(trials, pilots),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oracle_sparse_overlap_discovery_gate6/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    payload = run_diagnostic(
        seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
