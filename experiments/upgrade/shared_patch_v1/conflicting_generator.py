"""Preregistered conflicting-overlap generator for the shared_patch_v1 ladder.

Each cell builds an identity-blind 1000-dim problem whose planted structure
contains genuinely contested variables: a shared coordinate contributes to
two block subfunctions whose per-block optima for that coordinate disagree
by a fixed conflict distance.  The executors only ever see the
OptimizationProblem facade and a Phase-I checkpoint produced by the frozen
discovery protocol, exactly like the AOB lane.

Cells (frozen before any U1 run):

    topology in {chain, hub, pairs}  x  conflict in {mild, strong}

    chain: 10 planted blocks, adjacent pairs (i, i+1) share w=8 vars
    hub:   10 planted blocks, block 0 shares w=8 vars with blocks 1..4
    pairs: 10 planted blocks, pairs (2i, 2i+1) share w=8 vars
    mild:  conflict distance 0.10 * range ; strong: 0.25 * range

Planted blocks are 100 vars each on [-5, 5]^1000 with per-block elliptic
conditioning (1e6).  All construction randomness is derived from
(cell_id, run_seed) so a cell/seed pair is byte-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.phase1 import PHASE1_FES, run_phase1
from arac.runtime.contracts import PhaseCheckpoint, canonical_sha256
from arac.runtime.ledger import EvaluationLedger


DIMENSION = 1000
PLANTED_BLOCK_COUNT = 10
PLANTED_BLOCK_SIZE = 100
SHARED_WIDTH = 8
LOWER_BOUND = -5.0
UPPER_BOUND = 5.0
RANGE = UPPER_BOUND - LOWER_BOUND
CONDITION = 1e6
CONFLICT_FRACTIONS = {"mild": 0.10, "strong": 0.25}
TOPOLOGY_LINKS = {
    "chain": tuple((index, index + 1) for index in range(PLANTED_BLOCK_COUNT - 1)),
    "hub": tuple((0, index) for index in range(1, 5)),
    "pairs": tuple((2 * index, 2 * index + 1) for index in range(5)),
}
CELL_IDS = tuple(
    f"{topology}-{conflict}"
    for topology in ("chain", "hub", "pairs")
    for conflict in ("mild", "strong")
)
TOTAL_BUDGET_FES = 3_000_000
GENERATOR_PROTOCOL = "arac-upgrade-conflicting-generator-v1"


@dataclass(frozen=True)
class GeneratorGroundTruth:
    cell_id: str
    run_seed: int
    planted_blocks: tuple[tuple[int, ...], ...]
    links: tuple[tuple[int, int], ...]
    shared_variables: tuple[int, ...]
    conflict_distance: float
    optimum: float
    ground_truth_hash: str


def _cell_rng(cell_id: str, run_seed: int) -> np.random.Generator:
    digest = int(canonical_sha256({"cell_id": cell_id, "run_seed": int(run_seed)})[:16], 16)
    return np.random.default_rng(digest)


def _planted_blocks() -> tuple[tuple[int, ...], ...]:
    order = np.arange(DIMENSION)
    return tuple(
        tuple(int(value) for value in order[index * PLANTED_BLOCK_SIZE : (index + 1) * PLANTED_BLOCK_SIZE])
        for index in range(PLANTED_BLOCK_COUNT)
    )


def _build(cell_id: str, run_seed: int) -> tuple[GeneratorGroundTruth, "object"]:
    if cell_id not in CELL_IDS:
        raise ValueError(f"unknown generator cell: {cell_id}")
    topology, conflict = cell_id.split("-")
    rng = _cell_rng(cell_id, run_seed)
    blocks = _planted_blocks()
    links = TOPOLOGY_LINKS[topology]
    conflict_distance = CONFLICT_FRACTIONS[conflict] * RANGE / 2.0
    offsets = rng.uniform(-2.0, 2.0, DIMENSION)
    owner_sign: dict[int, list[float]] = {}
    shared: list[int] = []
    for left, right in links:
        candidates = [variable for variable in blocks[left] if variable not in owner_sign]
        chosen = sorted(int(value) for value in rng.choice(np.asarray(candidates), size=SHARED_WIDTH, replace=False))
        for variable in chosen:
            owner_sign[variable] = [conflict_distance, -conflict_distance]
            offsets[variable] = 0.0
            shared.append(variable)
    weights = np.empty(DIMENSION)
    for block in blocks:
        local = np.asarray(block, dtype=int)
        exponents = 6.0 * np.arange(len(local)) / max(1, len(local) - 1)
        weights[local] = np.power(10.0, exponents)
    minimizer = offsets.copy()
    for variable in shared:
        minimizer[variable] = 0.0

    def objective(vector: np.ndarray) -> float | np.ndarray:
        values = np.asarray(vector, dtype=float)
        flat = values.reshape(-1, DIMENSION)
        errors = (flat - offsets[None, :]) ** 2
        for variable in shared:
            distances = owner_sign[variable]
            errors[:, variable] = (flat[:, variable] - distances[0]) ** 2 + (
                flat[:, variable] - distances[1]
            ) ** 2
        total = (errors * weights[None, :]).sum(axis=1)
        return float(total[0]) if values.ndim == 1 else total

    optimum = float(objective(minimizer))
    truth = GeneratorGroundTruth(
        cell_id=cell_id,
        run_seed=int(run_seed),
        planted_blocks=blocks,
        links=links,
        shared_variables=tuple(sorted(shared)),
        conflict_distance=conflict_distance,
        optimum=optimum,
        ground_truth_hash=canonical_sha256(
            {
                "cell_id": cell_id,
                "run_seed": int(run_seed),
                "links": [list(link) for link in links],
                "shared_variables": sorted(shared),
                "conflict_distance": conflict_distance,
                "optimum": optimum,
                "offsets": [float(value) for value in offsets],
            }
        ),
    )
    return truth, objective


def build_ground_truth(cell_id: str, run_seed: int) -> GeneratorGroundTruth:
    return _build(cell_id, run_seed)[0]


def build_problem(cell_id: str, run_seed: int) -> tuple[OptimizationProblem, GeneratorGroundTruth]:
    truth, objective = _build(cell_id, run_seed)
    problem = OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(LOWER_BOUND,) * DIMENSION,
        upper_bounds=(UPPER_BOUND,) * DIMENSION,
        optimum=truth.optimum,
    )
    return problem, truth


def run_generator_phase1(cell_id: str, run_seed: int) -> tuple[PhaseCheckpoint, dict[str, object]]:
    """Run the frozen Phase-I discovery protocol on one generator cell."""

    problem, truth = build_problem(cell_id, run_seed)
    ledger = EvaluationLedger(problem, total_budget=TOTAL_BUDGET_FES)
    probe = run_phase1(problem, ledger, run_seed=int(run_seed))
    checkpoint = probe.checkpoint
    if checkpoint.phase1_fes != PHASE1_FES or ledger.count != PHASE1_FES:
        raise RuntimeError("generator Phase-I FE boundary drifted")
    stats = {
        "cell_id": cell_id,
        "run_seed": int(run_seed),
        "generator_protocol": GENERATOR_PROTOCOL,
        "ground_truth_hash": truth.ground_truth_hash,
        "planted_link_count": len(truth.links),
        "planted_shared_variable_count": len(truth.shared_variables),
        "discovered_block_count": len(checkpoint.blocks),
        "discovered_relation_count": len(checkpoint.relations),
        "discovered_relation_strength_max": max(
            (relation.strength for relation in checkpoint.relations), default=0.0
        ),
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "checkpoint": _checkpoint_payload(checkpoint),
    }
    return checkpoint, stats


def _checkpoint_payload(checkpoint: PhaseCheckpoint) -> dict[str, object]:
    return checkpoint.payload()


def relation_leverage(blocks: Sequence[Sequence[int]], relations) -> tuple[int, ...]:
    """Preregistered executor-lane leverage: incident relation count per block."""

    leverage = [0] * len(blocks)
    for relation in relations:
        leverage[relation.left_block] += 1
        leverage[relation.right_block] += 1
    return tuple(leverage)


def relation_block_order(blocks: Sequence[Sequence[int]], relations) -> tuple[int, ...]:
    """The frozen gcb baseline order (strength-weighted, index tiebreak)."""

    scores = [0.0] * len(blocks)
    for relation in relations:
        score = relation.strength * (1.0 + relation.disagreement)
        scores[relation.left_block] += score
        scores[relation.right_block] += score
    return tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index)))


__all__ = [
    "CELL_IDS",
    "DIMENSION",
    "GENERATOR_PROTOCOL",
    "TOTAL_BUDGET_FES",
    "build_ground_truth",
    "build_problem",
    "relation_block_order",
    "relation_leverage",
    "run_generator_phase1",
]
