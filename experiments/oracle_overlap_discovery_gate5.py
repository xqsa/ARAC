"""Fresh oracle-vs-inferred Gate 5 for variable-level overlap discovery."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.benchmarks.overlap_groups import OverlapGrouping, OverlapStructure
from arac.benchmarks.overlap_objective import OverlapObjective, OverlapObjectiveConfig
from arac.evidence import DEFAULT_EDGE_THRESHOLD, Phase1OverlapAdapter, discover_overlap
from arac.runtime.ledger import EvaluationLedger


DEFAULT_SEEDS = tuple(range(2026081301, 2026081306))
ANCHOR_COUNT = 5
PROBE_STEP = 10.0
TOPOLOGIES = {
    "pair": (5, ((0, 1, 2), (2, 3, 4))),
    "chain": (7, ((0, 1, 2), (2, 3, 4), (4, 5, 6))),
    "hub": (7, ((0, 1, 2), (2, 3, 4), (2, 5, 6))),
    "double_overlap": (6, ((0, 1, 2, 3), (2, 3, 4, 5))),
    "disjoint": (6, ((0, 1, 2), (3, 4, 5))),
}
BASE_FUNCTIONS = ("sphere", "ackley", "elliptic", "rastrigin", "schwefel")
MODES = ("conforming", "conflicting")


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
    edge_count: int
    consumed_fes: int
    expected_fes: int
    adapter_ready: bool
    deterministic: bool


def _structure(dimension: int, groups: tuple[tuple[int, ...], ...], seed: int) -> OverlapStructure:
    memberships = tuple(
        tuple(group for group, variables in enumerate(groups) if variable in variables)
        for variable in range(dimension)
    )
    overlap_budget = sum(len(group) for group in groups) - dimension
    grouping = OverlapGrouping(
        dimension=dimension,
        overlap_budget=overlap_budget,
        min_group_size=1,
        max_group_size=dimension,
        contiguous=False,
        seed=seed,
        num_groups=len(groups),
    )
    return OverlapStructure(
        grouping=grouping,
        groups=groups,
        group_sizes=tuple(len(group) for group in groups),
        overlap_shares=(),
        membership=memberships,
        shared_variables=tuple(variable for variable, owners in enumerate(memberships) if len(owners) > 1),
        occurrence={variable: len(owners) for variable, owners in enumerate(memberships)},
    )


def _problem(
    dimension: int,
    groups: tuple[tuple[int, ...], ...],
    *,
    base_function: str,
    conflict_mode: str,
    seed: int,
) -> OptimizationProblem:
    objective = OverlapObjective(
        _structure(dimension, groups, seed),
        OverlapObjectiveConfig(
            base_function=base_function,
            conflict_mode=conflict_mode,
            bounds=100.0,
            rotation=True,
            transforms=True,
            seed=seed,
        ),
    )

    def evaluate(values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        if converted.ndim == 1:
            return objective.evaluate(converted)
        return np.asarray([objective.evaluate(row) for row in converted], dtype=float)

    return OptimizationProblem(
        objective=evaluate,
        dimension=dimension,
        lower_bounds=(-100.0,) * dimension,
        upper_bounds=(100.0,) * dimension,
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
    expected_fes = ANCHOR_COUNT * (1 + dimension + dimension * (dimension - 1) // 2)
    result = discover_overlap(
        problem,
        EvaluationLedger(problem, expected_fes),
        anchors=anchors,
        step=PROBE_STEP,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
    )
    replay = discover_overlap(
        problem,
        EvaluationLedger(problem, expected_fes),
        anchors=anchors,
        step=PROBE_STEP,
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
    adapted = Phase1OverlapAdapter().adapt(
        _checkpoint(dimension),
        result.evidence,
    )
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
        edge_count=len(result.edges),
        consumed_fes=result.consumed_fes,
        expected_fes=expected_fes,
        adapter_ready=adapted.ready,
        deterministic=result == replay,
    )


def _checkpoint(dimension: int):
    from arac.runtime.contracts import PhaseCheckpoint

    return PhaseCheckpoint(
        protocol="oracle-overlap-discovery-gate5-v1",
        run_seed=0,
        total_budget_fes=1000,
        phase1_fes=1,
        incumbent=(0.0,) * dimension,
        incumbent_error=0.0,
        feature_names=("probe",),
        feature_values=(1.0,),
        blocks=tuple((variable,) for variable in range(dimension)),
    )


def _summary(trials: list[DiscoveryTrial]) -> dict[str, object]:
    identifiable = [trial for trial in trials if trial.topology != "disjoint"]
    disjoint = [trial for trial in trials if trial.topology == "disjoint"]
    checks = {
        "identifiable_groups_exact": all(trial.group_exact for trial in identifiable),
        "identifiable_shared_precision_one": all(trial.shared_precision == 1.0 for trial in identifiable),
        "identifiable_shared_recall_one": all(trial.shared_recall == 1.0 for trial in identifiable),
        "disjoint_no_shared_false_positive": all(not trial.inferred_shared for trial in disjoint),
        "exact_fe": all(trial.consumed_fes == trial.expected_fes for trial in trials),
        "adapter_ready": all(trial.adapter_ready for trial in trials),
        "deterministic": all(trial.deterministic for trial in trials),
    }
    return {
        "runs": len(trials),
        "identifiable_runs": len(identifiable),
        "disjoint_runs": len(disjoint),
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
    return {
        "schema_version": "arac-oracle-overlap-discovery-gate5-v1",
        "seeds": list(seeds),
        "anchor_count": ANCHOR_COUNT,
        "probe_step": PROBE_STEP,
        "edge_threshold": DEFAULT_EDGE_THRESHOLD,
        "topologies": TOPOLOGIES,
        "base_functions": BASE_FUNCTIONS,
        "conflict_modes": MODES,
        "trials": [asdict(trial) for trial in trials],
        "summary": _summary(trials),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oracle_overlap_discovery_gate5/confirmation_fresh.json"),
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
