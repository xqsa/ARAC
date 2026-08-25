"""Preregistered conflicting-overlap generator v3 (sparse topologies).

v2.2 plan section 5 requires sparse relation topologies after the v2
generator degenerated to complete-graph discovery (S1 evidence: leverage
identically distributed, order mechanism inert).  v3 keeps the
identity-blind facade and the frozen Phase-I discovery of v2 but changes
the planted structure:

- topologies (10 planted blocks, degree <= 3, no complete graph):
    chain4: links (0,1), (1,2), (2, 3)          - one chain of four blocks
    pairs3: links (0,1), (4,5), (8,9)           - three disjoint pairs
    hub3:   links (0,1), (0,2), (0,3)           - one hub of degree three
- conflict levels: mild = 0.10 * range, strong = 0.25 * range (as v2)
- shared width w and per-block conditioning are preflight-selected from the
  frozen candidate list below (first configuration whose six cells pass the
  preflight hard criteria on every screen seed is frozen).

Preflight hard criteria per cell and seed (v2.2 section 5):

1. discovered relation count > 0;
2. variance of the discovered incident-relation leverage across blocks > 0;
3. at least one multi-block component exists (core scope non-empty) and
   every core chunk after the K_core split is <= K_core = 50.

The construction truth (links, shared variables, optima, per-group
contribution) is derivable and recorded in every receipt exactly as in v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.phase1 import run_phase1
from arac.runtime.contracts import PhaseCheckpoint, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.upgrade.shared_patch_v1.conflicting_generator import (
    DIMENSION,
    LOWER_BOUND,
    RANGE,
    TOTAL_BUDGET_FES,
    UPPER_BOUND,
    CONFLICT_FRACTIONS,
    _planted_blocks,
    relation_leverage,
)


PLANTED_BLOCK_COUNT = 10
K_CORE = 50
SHARED_WIDTH_CANDIDATES = (4, 8)
CONDITIONING_CANDIDATES = ("linked-sphere", "linked-elliptic")
LINKAGE_LAMBDA = 1.0
TOPOLOGY_LINKS = {
    "chain4": ((0, 1), (1, 2), (2, 3)),
    "pairs3": ((0, 1), (4, 5), (8, 9)),
    "hub3": ((0, 1), (0, 2), (0, 3)),
}
TOPOLOGIES = ("chain4", "pairs3", "hub3")
CONFLICTS = ("mild", "strong")
V3_CELL_IDS = tuple(
    f"{topology}-{conflict}"
    for topology in TOPOLOGIES
    for conflict in CONFLICTS
)
GENERATOR_V3_PROTOCOL = "arac-upgrade-conflicting-generator-v3"
SELECTION_RULE = (
    "iterate conditioning in (linked-sphere, linked-elliptic) x width in (4, 8) "
    "in the listed order; freeze the first configuration where all six cells "
    "pass the preflight hard criteria on every screen seed (117,123,129,135,141); "
    "probe runs live under tmp/ and are deleted after selection.  Rationale: "
    "fully separable quadratics saturate the structural discovery into "
    "complete-graph artifacts (v2/v3 probe evidence); per-block linkage terms "
    "give the discovery real within-block structure so that only the planted "
    "shared-variable couplings survive as between-block relations."
)


@dataclass(frozen=True)
class V3GroundTruth:
    cell_id: str
    run_seed: int
    conditioning: str
    shared_width: int
    planted_blocks: tuple[tuple[int, ...], ...]
    links: tuple[tuple[int, int], ...]
    shared_variables: tuple[int, ...]
    shared_owner_pairs: tuple[tuple[int, int, int], ...]
    conflict_distance: float
    optimum: float
    ground_truth_hash: str


def build_v3_ground_truth(cell_id: str, run_seed: int, *, conditioning: str, shared_width: int, linkage_lambda: float = LINKAGE_LAMBDA) -> tuple[V3GroundTruth, object]:
    if cell_id not in V3_CELL_IDS:
        raise ValueError(f"unknown v3 cell: {cell_id}")
    topology, conflict = cell_id.split("-")
    family = conditioning.split("-", 1)[1]
    digest = int(canonical_sha256({"generator": "v3", "cell_id": cell_id, "run_seed": int(run_seed), "conditioning": conditioning, "shared_width": int(shared_width), "linkage_lambda": float(linkage_lambda)})[:16], 16)
    rng = np.random.default_rng(digest)
    blocks = _planted_blocks()
    links = TOPOLOGY_LINKS[topology]
    conflict_distance = CONFLICT_FRACTIONS[conflict] * RANGE / 2.0
    offsets = rng.uniform(-2.0, 2.0, DIMENSION)
    weights = np.empty(DIMENSION)
    for block in blocks:
        local = np.asarray(block, dtype=int)
        if family == "sphere":
            weights[local] = 1.0
        elif family == "elliptic":
            exponents = 6.0 * np.arange(len(local)) / max(1, len(local) - 1)
            weights[local] = np.power(10.0, exponents)
        else:
            raise ValueError(f"unknown conditioning family: {conditioning}")
    # per-block linkage membership: private variables plus, for every link,
    # the shared variables with their per-owner conflict offsets; a shared
    # variable enters BOTH owners' linkage sums, which is what makes the
    # planted overlap visible to interaction-based discovery
    linkage_members: list[dict[int, float]] = [{variable: float(offsets[variable]) for variable in block} for block in blocks]
    shared: list[int] = []
    shared_owner_pairs: list[tuple[int, int, int]] = []
    for left, right in links:
        candidates = [variable for variable in blocks[left] if variable not in shared]
        chosen = sorted(int(value) for value in rng.choice(np.asarray(candidates), size=shared_width, replace=False))
        for variable in chosen:
            linkage_members[left][variable] = conflict_distance
            linkage_members[right][variable] = -conflict_distance
            offsets[variable] = 0.0
            shared.append(variable)
            shared_owner_pairs.append((variable, left, right))
    shared_set = set(shared)
    linkage_arrays = [
        (np.asarray(sorted(members), dtype=int), np.asarray([members[variable] for variable in sorted(members)], dtype=float))
        for members in linkage_members
    ]

    def objective(vector: np.ndarray):
        values = np.asarray(vector, dtype=float)
        flat = values.reshape(-1, DIMENSION)
        quadratic = (flat - offsets[None, :]) ** 2 * weights[None, :]
        for variable in shared_set:
            distance = conflict_distance
            quadratic[:, variable] = weights[variable] * (
                (flat[:, variable] - distance) ** 2 + (flat[:, variable] + distance) ** 2
            )
        linkage = np.zeros(flat.shape[0])
        for variables, member_offsets in linkage_arrays:
            sums = (flat[:, variables] - member_offsets[None, :]).sum(axis=1)
            linkage += linkage_lambda * sums**2
        total = quadratic.sum(axis=1) + linkage
        return float(total[0]) if values.ndim == 1 else total

    minimizer = _quadratic_minimizer(weights, offsets, shared_set, conflict_distance, linkage_arrays, linkage_lambda=linkage_lambda)
    raw_optimum = float(objective(minimizer))

    # AOB convention: the objective's global minimum value must be zero.
    # Phase-I is calibrated for zero-optimum problems (its structural pass
    # reconstructs the raw base value as best_error + problem.optimum, which
    # equals the raw value only when the optimum is zero); an unshifted
    # objective injects its minimum as a constant offset into every
    # interaction difference and saturates relation discovery into a
    # complete-graph artifact (the v2/v3 degeneracy root cause).
    def shifted_objective(vector: np.ndarray):
        values = objective(vector)
        return values - raw_optimum

    optimum = 0.0
    truth = V3GroundTruth(
        cell_id=cell_id,
        run_seed=int(run_seed),
        conditioning=conditioning,
        shared_width=int(shared_width),
        planted_blocks=blocks,
        links=links,
        shared_variables=tuple(sorted(shared)),
        shared_owner_pairs=tuple(sorted(shared_owner_pairs)),
        conflict_distance=conflict_distance,
        optimum=optimum,
        ground_truth_hash=canonical_sha256(
            {
                "generator": "v3",
                "cell_id": cell_id,
                "run_seed": int(run_seed),
                "conditioning": conditioning,
                "shared_width": int(shared_width),
                "linkage_lambda": float(linkage_lambda),
                "links": [list(link) for link in links],
                "shared_variables": sorted(shared),
                "shared_owner_pairs": [list(item) for item in sorted(shared_owner_pairs)],
                "conflict_distance": conflict_distance,
                "raw_optimum": raw_optimum,
                "optimum": optimum,
                "offsets": [float(value) for value in offsets],
            }
        ),
    )
    return truth, shifted_objective


def _quadratic_minimizer(
    weights: np.ndarray,
    offsets: np.ndarray,
    shared_set: set[int],
    conflict_distance: float,
    linkage_arrays,
    linkage_lambda: float = LINKAGE_LAMBDA,
) -> np.ndarray:
    """Solve the positive-definite quadratic exactly for the true optimum."""

    diagonal = np.zeros(DIMENSION)
    linear = np.zeros(DIMENSION)
    for variable in range(DIMENSION):
        if variable in shared_set:
            diagonal[variable] = 2.0 * weights[variable]
        else:
            diagonal[variable] = weights[variable]
            linear[variable] = weights[variable] * offsets[variable]
    matrix = np.diag(diagonal)
    for variables, member_offsets in linkage_arrays:
        target = float(member_offsets.sum())
        selector = np.zeros(DIMENSION)
        selector[variables] = 1.0
        matrix += linkage_lambda * np.outer(selector, selector)
        linear += linkage_lambda * target * selector
    return np.linalg.solve(matrix, linear)


def build_v3_problem(cell_id: str, run_seed: int, *, conditioning: str, shared_width: int, linkage_lambda: float = LINKAGE_LAMBDA) -> tuple[OptimizationProblem, V3GroundTruth]:
    truth, objective = build_v3_ground_truth(cell_id, run_seed, conditioning=conditioning, shared_width=shared_width, linkage_lambda=linkage_lambda)
    problem = OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(LOWER_BOUND,) * DIMENSION,
        upper_bounds=(UPPER_BOUND,) * DIMENSION,
        optimum=truth.optimum,
    )
    return problem, truth


def relation_components(blocks: Sequence[Sequence[int]], relations) -> tuple[tuple[int, ...], ...]:
    """Connected components of the checkpoint relation graph over blocks."""

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(blocks))}
    for relation in relations:
        adjacency[relation.left_block].add(relation.right_block)
        adjacency[relation.right_block].add(relation.left_block)
    seen: set[int] = set()
    components = []
    for start in range(len(blocks)):
        if start in seen:
            continue
        stack = [start]
        members = []
        seen.add(start)
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(members)))
    return tuple(components)


def split_core_scope(core_variables: Sequence[int], *, k_core: int = K_CORE) -> tuple[tuple[int, ...], ...]:
    """Split an oversized core into contiguous chunks of at most k_core."""

    ordered = tuple(sorted(int(variable) for variable in core_variables))
    if len(ordered) <= k_core:
        return (ordered,)
    return tuple(ordered[index : index + k_core] for index in range(0, len(ordered), k_core))


def s3a_scopes(blocks: Sequence[Sequence[int]], relations, *, k_core: int = K_CORE) -> tuple[tuple[int, ...], ...]:
    """Preregistered executor-lane S3a scope list (v2.2 section 3 union semantics).

    Components come from the relation graph; every multi-block component is
    emitted as one core scope (union of member variables, split into chunks
    of at most k_core) at its minimum member block position; non-minimum
    members are skipped; singleton blocks pass through unchanged.  With no
    relations the list equals the original blocks exactly.
    """

    components = relation_components(blocks, relations)
    min_member = {members[0]: members for members in components if len(members) > 1}
    skip = {member for members in min_member.values() for member in members[1:]}
    scopes: list[tuple[int, ...]] = []
    for index in range(len(blocks)):
        if index in skip:
            continue
        if index in min_member:
            union: set[int] = set()
            for member in min_member[index]:
                union.update(blocks[member])
            scopes.extend(split_core_scope(sorted(union), k_core=k_core))
        else:
            scopes.append(tuple(int(variable) for variable in blocks[index]))
    flattened = tuple(variable for scope in scopes for variable in scope)
    if sorted(flattened) != sorted(variable for block in blocks for variable in block):
        raise RuntimeError("S3a restructuring lost variable coverage")
    if len(set(flattened)) != len(flattened):
        raise RuntimeError("S3a restructuring duplicated a variable")
    return tuple(scopes)


def preflight_check(checkpoint: PhaseCheckpoint) -> dict[str, object]:
    leverage = relation_leverage(checkpoint.blocks, checkpoint.relations)
    leverage_values = np.asarray(leverage, dtype=float)
    components = relation_components(checkpoint.blocks, checkpoint.relations)
    multi = [members for members in components if len(members) > 1]
    core_sizes = [
        len(chunk)
        for members in multi
        for chunk in split_core_scope(sorted(set().union(*(set(checkpoint.blocks[member]) for member in members))))
    ]
    return {
        "relation_count": len(checkpoint.relations),
        "leverage_variance": float(np.var(leverage_values)),
        "multi_block_component_count": len(multi),
        "core_scope_sizes": core_sizes,
        "max_core_chunk": max(core_sizes, default=0),
        "passed": bool(
            len(checkpoint.relations) > 0
            and float(np.var(leverage_values)) > 0.0
            and len(multi) > 0
            and all(size <= K_CORE for size in core_sizes)
        ),
    }


def run_v3_phase1(cell_id: str, run_seed: int, *, conditioning: str, shared_width: int, linkage_lambda: float = LINKAGE_LAMBDA) -> tuple[PhaseCheckpoint, dict[str, object]]:
    problem, truth = build_v3_problem(cell_id, run_seed, conditioning=conditioning, shared_width=shared_width, linkage_lambda=linkage_lambda)
    ledger = EvaluationLedger(problem, total_budget=TOTAL_BUDGET_FES)
    probe = run_phase1(problem, ledger, run_seed=int(run_seed))
    checkpoint = probe.checkpoint
    stats = {
        "cell_id": cell_id,
        "run_seed": int(run_seed),
        "generator_protocol": GENERATOR_V3_PROTOCOL,
        "conditioning": conditioning,
        "shared_width": int(shared_width),
        "linkage_lambda": float(linkage_lambda),
        "ground_truth_hash": truth.ground_truth_hash,
        "planted_link_count": len(truth.links),
        "planted_shared_variable_count": len(truth.shared_variables),
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "preflight": preflight_check(checkpoint),
        "checkpoint": checkpoint.payload(),
    }
    return checkpoint, stats


__all__ = [
    "CONDITIONING_CANDIDATES",
    "GENERATOR_V3_PROTOCOL",
    "K_CORE",
    "SELECTION_RULE",
    "SHARED_WIDTH_CANDIDATES",
    "V3_CELL_IDS",
    "build_v3_ground_truth",
    "build_v3_problem",
    "preflight_check",
    "relation_components",
    "run_v3_phase1",
    "s3a_scopes",
    "split_core_scope",
]
