"""Soft DSM construction and noise-robust RDDSM extraction (v10.4 branch).

Replaces the Fiedler-order + dyadic-bisection + boundary-probe path
(v10.3) with the row-based decomposition borrowed from the SOTA's RDDSM,
adapted to estimated evidence:

```text
variable signatures (Gate 43, shared probe basis)
  -> mutual-kNN candidate edges (no d^2 matrix)
  -> two-tier billed conditional pair probes (screen then confirm)
  -> soft design-structure matrix (weighted, with support)
  -> RDG coarse grouping (complete candidate pool, exclusive blocks)
  -> small-region recursive refinement (soft group cover)
  -> complete-intersection separability and two-sided confirmation
     -> ResolvedOverlapHyperedge (Gate 42 semantics)
```

Forensic motivation (2026-08-16): the fixed-threshold binary gate on R2
decided inside the noise band.  The two-tier probes here estimate
per-edge support instead of deciding on single thresholds, and block
alignment removes the order bias that starved the t-selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.hierarchical import (
    Phase1Evidence,
    RegionNode,
    RegionRelation,
    RegionTree,
    ResolvedOverlapHyperedge,
    VariableRegionInteraction,
)
from arac.evidence.variable_signature import (
    VariableSignatureResult,
    compute_variable_signatures,
)
from arac.runtime.ledger import EvaluationLedger


@dataclass(frozen=True)
class SoftDsmConfig:
    """Pre-registered integers for the soft-RDDSM branch."""

    k_mutual: int = 6
    screen_anchors: int = 2
    confirm_extra: int = 2
    edge_threshold: float = 1e-13
    confirm_threshold: float = 1e-10
    # High-threshold edges separate groups: shared-to-other-group (~1e-6)
    # and nonshared-to-own-group (~1e-6) survive; shared-to-own-group
    # (~1e-11) is filtered, pushing shared vars into their other group.
    block_edge_threshold: float = 1e-8
    tau_block: float = 0.6
    tau_connect: float = 0.3
    max_block_size: int = 60
    block_separability_probes: int = 5
    rdg_region_size: int = 16
    min_residual_size: int = 3
    dsm_budget: int = 55_000

    def __post_init__(self) -> None:
        for name in (
            "k_mutual",
            "screen_anchors",
            "confirm_extra",
            "block_separability_probes",
            "rdg_region_size",
            "min_residual_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.dsm_budget, bool) or not isinstance(self.dsm_budget, int) or self.dsm_budget < 1:
            raise ValueError("dsm_budget must be a positive integer")
        for name in ("tau_block", "tau_connect"):
            value = getattr(self, name)
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not 0.0 < float(self.edge_threshold) < 1.0:
            raise ValueError("edge_threshold must lie in (0, 1)")
        if not float(self.edge_threshold) <= float(self.confirm_threshold) < 1.0:
            raise ValueError("confirm_threshold must be >= edge_threshold and < 1")


@dataclass(frozen=True)
class SoftEdge:
    left: int
    right: int
    score: float
    support: int
    probes: int
    consumed_fes: int


@dataclass(frozen=True)
class SoftDiscoveryResult:
    """Complete receipt of the soft-RDDSM discovery branch."""

    evidence: Phase1Evidence
    signature_result: VariableSignatureResult
    edges: tuple[SoftEdge, ...]
    blocks: tuple[tuple[int, ...], ...]
    level_budgets: tuple[tuple[str, int], ...]
    shared_candidates: tuple[int, ...]
    discovery_start_fes: int = 0
    discovery_consumed_fes: int = 0
    discovery_end_fes: int = 0


def _probe_score(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    anchor: np.ndarray,
    left: int,
    right: int,
    step: float,
    sign: float,
) -> float:
    delta_l = np.zeros(problem.dimension)
    delta_l[left] = step * sign
    delta_r = np.zeros(problem.dimension)
    delta_r[right] = step * sign
    rows = np.asarray(
        (
            anchor,
            np.clip(anchor + delta_l, problem.lower_array, problem.upper_array),
            np.clip(anchor + delta_r, problem.lower_array, problem.upper_array),
            np.clip(anchor + delta_l + delta_r, problem.lower_array, problem.upper_array),
        ),
        dtype=float,
    )
    values = np.asarray(ledger.evaluate(rows), dtype=float)
    base, single_l, single_r, joint = (float(value) for value in values)
    scale = abs(base) + abs(single_l) + abs(single_r) + abs(joint) + 1.0
    return abs(joint - single_l - single_r + base) / scale


def _mutual_knn_candidates(signatures: np.ndarray, k: int) -> set[tuple[int, int]]:
    dimension = signatures.shape[0]
    norms = np.linalg.norm(signatures, axis=1)
    neighbours: dict[int, set[int]] = {}
    for variable in range(dimension):
        if norms[variable] <= 1e-12:
            continue
        similarities = np.asarray(
            [
                (
                    float(np.dot(signatures[variable], signatures[other]))
                    / (norms[variable] * norms[other])
                    if norms[other] > 1e-12 and other != variable
                    else -np.inf
                )
                for other in range(dimension)
            ]
        )
        order = np.argsort(-similarities, kind="stable")
        neighbours[variable] = {int(other) for other in order[:k]}
    candidates = set()
    for variable, near in neighbours.items():
        for other in near:
            candidates.add((min(variable, other), max(variable, other)))
    return candidates


def build_soft_dsm(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    signatures: np.ndarray,
    *,
    config: SoftDsmConfig | None = None,
    step: float = 0.25,
    seed: int = 0,
    budget_fes: int | None = None,
) -> tuple[dict[tuple[int, int], SoftEdge], int, int]:
    """Probe candidate edges with a two-tier screen/confirm scheme."""

    config = SoftDsmConfig() if config is None else config
    if budget_fes is not None and (
        isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes < 0
    ):
        raise ValueError("budget_fes must be a non-negative integer")
    if budget_fes is not None and budget_fes > ledger.remaining:
        raise ValueError("soft DSM budget reservation exceeds ledger headroom")
    rng = np.random.default_rng(seed ^ 0x50F7)
    dimension = problem.dimension
    center = (problem.lower_array + problem.upper_array) / 2.0
    span = problem.upper_array - problem.lower_array
    anchor_rng = np.random.default_rng(seed ^ 0xA4C)
    anchors = [
        center + anchor_rng.uniform(-0.2, 0.2, size=dimension) * span
        for _ in range(config.screen_anchors)
    ]

    candidates = sorted(_mutual_knn_candidates(signatures, config.k_mutual))
    start = ledger.count
    stage_budget = config.dsm_budget if budget_fes is None else min(config.dsm_budget, budget_fes)
    edges: dict[tuple[int, int], SoftEdge] = {}
    for left, right in candidates:
        screen_cost = 4 * len(anchors)
        if ledger.count - start + screen_cost > stage_budget:
            break
        screen_scores = [
            _probe_score(problem, ledger, anchor, left, right, step, float(sign))
            for anchor, sign in zip(
                anchors,
                rng.choice((-1.0, 1.0), size=len(anchors)),
                strict=True,
            )
        ]
        if max(screen_scores) <= config.edge_threshold:
            continue
        scores = list(screen_scores)
        for _ in range(config.confirm_extra):
            if ledger.count - start + 4 > stage_budget:
                break
            anchor = anchors[int(rng.integers(0, len(anchors)))]
            scores.append(
                _probe_score(
                    problem,
                    ledger,
                    anchor,
                    left,
                    right,
                    step,
                    float(rng.choice((-1.0, 1.0))),
                )
            )
        support = sum(1 for score in scores if score > config.confirm_threshold)
        edges[(left, right)] = SoftEdge(
            left=left,
            right=right,
            score=float(np.mean(scores)),
            support=support,
            probes=len(scores),
            consumed_fes=4 * len(scores),
        )
    return edges, len(candidates), ledger.count - start


def soft_rddsm_blocks(
    edges: dict[tuple[int, int], SoftEdge],
    signatures: np.ndarray,
    *,
    tau_block: float,
    tau_connect: float,
    max_block_size: int = 60,
    block_edge_threshold: float = 1e-8,
) -> tuple[tuple[int, ...], ...]:
    """Size-capped components on high-threshold edges.

    Uses a higher edge threshold than the DSM confirm threshold: shared
    variables' cross-group interactions (~1e-6) and non-shared variables'
    within-group interactions (~1e-6) survive, while shared variables'
    within-group interactions (~1e-11) are filtered.  This pushes shared
    variables into their "other" group's component, creating blocks that
    align with true groups minus their shared boundary members.
    """

    adjacency: dict[int, set[int]] = {}
    for (left, right), edge in edges.items():
        if edge.score < block_edge_threshold:
            continue  # weak edges (shared-to-own-group) don't join blocks
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    visited: set[int] = set()
    blocks: list[tuple[int, ...]] = []
    for seed in sorted(adjacency, key=lambda v: (-len(adjacency[v]), v)):
        if seed in visited:
            continue
        component: set[int] = {seed}
        visited.add(seed)
        stack = [seed]
        while stack and len(component) < max_block_size:
            node = stack.pop()
            for neighbour in sorted(adjacency[node]):
                if neighbour not in visited and len(component) < max_block_size:
                    visited.add(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        if len(component) >= 2:
            blocks.append(tuple(sorted(component)))
    soft_rddsm_blocks.articulation = set()
    return tuple(blocks)


def _rdg_interact(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    set_a: list[int],
    set_b: list[int],
    base_point: np.ndarray,
    base_value: float,
    threshold: float,
) -> bool:
    """Moderate-range set-to-set interaction test (exactly 3 FE).

    Full-range RDG (lower→upper) is numerically meaningless on AOB where
    function values span 36 orders of magnitude.  This variant perturbs
    set_a by ±range/4 and set_b by ±range/8 from ``base_point``, giving a
    strong but bounded interaction signal.  The base value is supplied by
    the caller so the mixed difference costs three evaluations, matching
    every budget guard in this module.
    """

    lower = problem.lower_array
    upper = problem.upper_array
    span = upper - lower

    # x_plus: set_a moved +span/4 from the base point, set_b unchanged.
    x_plus = np.clip(base_point.copy(), lower, upper)
    x_plus[set_a] = np.clip(base_point[set_a] + span[set_a] * 0.25, lower[set_a], upper[set_a])
    y_plus = float(np.asarray(ledger.evaluate(x_plus[np.newaxis, :])).reshape(-1)[0])

    # x_mid: set_a unchanged, set_b moved +span/8.
    x_mid = np.clip(base_point.copy(), lower, upper)
    x_mid[set_b] = np.clip(base_point[set_b] + span[set_b] * 0.125, lower[set_b], upper[set_b])
    y_mid = float(np.asarray(ledger.evaluate(x_mid[np.newaxis, :])).reshape(-1)[0])

    # x_both: set_a at +span/4, set_b at +span/8.
    x_both = x_plus.copy()
    x_both[set_b] = np.clip(base_point[set_b] + span[set_b] * 0.125, lower[set_b], upper[set_b])
    y_both = float(np.asarray(ledger.evaluate(x_both[np.newaxis, :])).reshape(-1)[0])

    base = float(base_value)
    delta_1 = base - y_plus
    delta_2 = y_mid - y_both
    lam = abs(delta_1 - delta_2)
    eps = threshold * (abs(base) + abs(y_plus) + abs(y_mid) + abs(y_both) + 1.0)
    return lam > eps


def _rdg_refine_interactors(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    seed: int,
    candidates: list[int],
    region_size: int,
    base_point: np.ndarray,
    base_value: float,
    threshold: float,
    budget_start: int,
    budget: int,
) -> set[int]:
    """Find a seed's interactors without testing a numerically huge set.

    AOB's mixed difference is reliable on small complements but can be
    dominated or cancelled when a 1000-variable complement is evaluated in
    one shot.  Test fixed-size regions first, then recursively isolate only
    positive regions.  ``region_size`` is a probe-stability control, not a
    claim about the hidden component size.
    """

    if isinstance(region_size, bool) or not isinstance(region_size, int) or region_size < 1:
        raise ValueError("region_size must be a positive integer")

    def available() -> bool:
        return ledger.count - budget_start + 3 <= budget

    def isolate(items: list[int], *, known_positive: bool = False) -> set[int]:
        if not items or not available():
            return set()
        if not known_positive and not _rdg_interact(
            problem,
            ledger,
            set_a=[seed],
            set_b=items,
            base_point=base_point,
            base_value=base_value,
            threshold=threshold,
        ):
            return set()
        if len(items) == 1:
            return set(items)
        middle = len(items) // 2
        return isolate(items[:middle]) | isolate(items[middle:])

    interactors: set[int] = set()
    for begin in range(0, len(candidates), region_size):
        if not available():
            break
        region = candidates[begin : begin + region_size]
        if _rdg_interact(
            problem,
            ledger,
            set_a=[seed],
            set_b=region,
            base_point=base_point,
            base_value=base_value,
            threshold=threshold,
        ):
            # The region has already been tested, so avoid paying for the
            # same set test again in ``isolate``.
            interactors.update(isolate(region, known_positive=True))
    return interactors


def _block_separable(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    anchor: np.ndarray,
    members_a: list[int],
    members_b: list[int],
    *,
    step: float,
    threshold: float,
    probes: int,
    seed: int,
    budget_start: int | None = None,
    budget: int | None = None,
) -> bool:
    """True when two member sets show no joint interaction across probes.

    Each probe perturbs the two sets coherently -- one shared sign per
    set, drawn independently per probe -- because independent per-member
    signs let multiple bridge terms cancel in the mixed difference and
    admit interacting pairs as separable.
    """

    rng = np.random.default_rng(seed)
    for _probe in range(probes):
        if budget_start is not None and budget is not None:
            if ledger.count - budget_start + 4 > budget:
                return False
        elif ledger.remaining < 4:
            return False
        delta_a = np.zeros(problem.dimension)
        delta_a[members_a] = step * float(rng.choice((-1.0, 1.0)))
        delta_b = np.zeros(problem.dimension)
        delta_b[members_b] = step * float(rng.choice((-1.0, 1.0)))
        rows = np.asarray(
            (
                anchor,
                np.clip(anchor + delta_a, problem.lower_array, problem.upper_array),
                np.clip(anchor + delta_b, problem.lower_array, problem.upper_array),
                np.clip(anchor + delta_a + delta_b, problem.lower_array, problem.upper_array),
            ),
            dtype=float,
        )
        values = np.asarray(ledger.evaluate(rows), dtype=float)
        base, single_a, single_b, joint = (float(value) for value in values)
        scale = abs(base) + abs(single_a) + abs(single_b) + abs(joint) + 1.0
        if abs(joint - single_a - single_b + base) / scale > threshold:
            return False
    return True


def discover_hierarchical_soft(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    run_seed: int,
    config: SoftDsmConfig | None = None,
    signature_probe_count: int = 12,
    signature_probe_size: int = 16,
    step: float = 0.25,
    budget_fes: int | None = None,
) -> SoftDiscoveryResult:
    """Full soft-RDDSM branch: signature -> soft DSM -> blocks -> evidence.

    The region tree assigns every variable to ONE primary block (or a
    singleton), preserving the Gate 42 partition invariant. A separate
    refined soft cover establishes multi-membership evidence; confirmed
    shared variables are materialized only in resolved hyperedges backed by
    residual-group probes.
    """

    if not isinstance(problem, OptimizationProblem):
        raise TypeError("problem must be OptimizationProblem")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be EvaluationLedger")
    if ledger.problem is not problem:
        raise ValueError("discovery requires the ledger for the same problem")
    config = SoftDsmConfig() if config is None else config
    discovery_start_fes = ledger.count
    if budget_fes is None:
        budget_fes = ledger.remaining
    if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes < 0:
        raise ValueError("budget_fes must be a non-negative integer")
    if budget_fes > ledger.remaining:
        raise ValueError("soft-RDDSM budget reservation exceeds ledger headroom")

    def discovery_consumed() -> int:
        return ledger.count - discovery_start_fes

    def discovery_remaining() -> int:
        return budget_fes - discovery_consumed()

    dimension = problem.dimension
    center = (problem.lower_array + problem.upper_array) / 2.0
    span = problem.upper_array - problem.lower_array
    anchor_rng = np.random.default_rng(run_seed ^ 0x5A17)
    signature_anchor = center + anchor_rng.uniform(-0.2, 0.2, size=dimension) * span

    signature_expected = (
        1
        + dimension
        + signature_probe_count
        + dimension * signature_probe_count
        + signature_probe_count * signature_probe_size
    )
    # The RDG stage always needs one centre evaluation after signatures and
    # DSM.  Reserve that anchor before any objective call; otherwise a DSM
    # screen can legally consume the last FE and make the later guard fail
    # only after a partial discovery receipt has been written.
    minimum_discovery_fes = signature_expected + 1
    if discovery_remaining() < minimum_discovery_fes:
        raise ValueError(
            "soft-RDDSM budget reservation cannot pay the variable-signature stage: "
            f"need {minimum_discovery_fes}, have {discovery_remaining()}"
        )
    signature_result = compute_variable_signatures(
        problem,
        ledger,
        anchor=signature_anchor,
        step=step,
        probe_count=signature_probe_count,
        probe_size=signature_probe_size,
        seed=run_seed ^ 0x516,
    )
    edges, _candidates, dsm_fes = build_soft_dsm(
        problem,
        ledger,
        signature_result.signatures,
        config=config,
        step=step,
        seed=run_seed ^ 0x50F7,
        # Keep one FE reserved for the RDG centre anchor.  All DSM screens
        # and confirmations still draw from the same discovery reservation.
        budget_fes=discovery_remaining() - 1,
    )
    # Note: blocks are now formed by RDG recursive detection below,
    # replacing the kNN-edge connected components that absorbed overlap.

    # Use moderate-range RDG to find interactors of the seed (like OEDG's
    # INTERACT function).  This naturally includes shared vars (they interact
    # with the seed through the shared group) but excludes the other group's
    # non-shared members.  A size cap prevents unbounded growth.
    centre_point = (problem.lower_array + problem.upper_array) / 2.0
    rdg_start = ledger.count
    rdg_budget = discovery_remaining()
    if rdg_budget < 1:
        raise ValueError("soft-RDDSM budget reservation cannot pay the RDG anchor evaluation")
    centre_value = float(
        np.asarray(ledger.evaluate(centre_point[np.newaxis, :])).reshape(-1)[0]
    )

    def _find_interactors(seed: int, candidates: list[int], budget_left: int) -> set[int]:
        """Recursive bisection: find candidates that interact with seed."""
        if (
            not candidates
            or budget_left < 3
            or ledger.count - rdg_start + 3 > rdg_budget
        ):
            return set()
        if _rdg_interact(
            problem, ledger,
            set_a=[seed], set_b=candidates,
            base_point=centre_point, base_value=centre_value,
            threshold=config.edge_threshold,
        ):
            if len(candidates) == 1:
                return set(candidates)
            mid = len(candidates) // 2
            return (
                _find_interactors(seed, candidates[:mid], budget_left - 3)
                | _find_interactors(seed, candidates[mid:], budget_left - 3)
            )
        return set()

    # Stage 1: build a disjoint coarse cover.  The candidate pool is complete
    # (never a random subset), so a positive recursive branch cannot lose a
    # member before the next seed is selected.
    grouped: set[int] = set()
    rdg_blocks: list[tuple[int, ...]] = []
    for seed in range(dimension):
        if seed in grouped:
            continue
        remaining_budget = rdg_budget - (ledger.count - rdg_start)
        if remaining_budget < 3:
            break
        # The recursive test is itself the budgeted search.  Sampling this
        # set before recursion is unsafe: a sampled subset can omit members
        # of the seed's true component, after which ``grouped.update`` makes
        # those omitted variables permanently unavailable to that component.
        # On AOB this turns every true group into several fragments and leaves
        # no reliable cross-group evidence.  Keep the complete candidate set;
        # the positive-branch recursion visits only regions containing an
        # interaction and remains well below the registered RDG budget.
        candidates = [v for v in range(dimension) if v not in grouped and v != seed]
        interactors = _find_interactors(seed, candidates, remaining_budget)
        group = tuple(sorted({seed} | interactors))
        if len(group) >= 2:
            rdg_blocks.append(group)
        grouped.update(group)

    # Stage 2: refine one representative from each coarse block against
    # small regions of the full variable set.  This recovers the same shared
    # variable in multiple soft groups without making the region tree overlap.
    refined_groups: list[tuple[int, ...]] = []
    refinement_complete = True
    for block in rdg_blocks:
        if not block:
            continue
        if ledger.count - rdg_start + 3 > rdg_budget:
            refinement_complete = False
            break
        seed = block[0]
        candidates = [variable for variable in range(dimension) if variable != seed]
        interactors = _rdg_refine_interactors(
            problem,
            ledger,
            seed=seed,
            candidates=candidates,
            region_size=config.rdg_region_size,
            base_point=centre_point,
            base_value=centre_value,
            threshold=config.edge_threshold,
            budget_start=rdg_start,
            budget=rdg_budget,
        )
        group = tuple(sorted({seed} | interactors))
        if len(group) >= 2:
            refined_groups.append(group)
    if len(refined_groups) != len(rdg_blocks):
        refinement_complete = False

    blocks = tuple(rdg_blocks)

    # Build the evidence tree from RDG blocks
    primary: dict[int, int] = {}
    for block_index, block in enumerate(blocks):
        for variable in block:
            primary.setdefault(variable, block_index)
    primary_members: list[list[int]] = [[] for _ in blocks]
    for variable, block_index in primary.items():
        primary_members[block_index].append(variable)
    leaf_variables: list[tuple[int, ...]] = []
    for members in primary_members:
        if members:
            leaf_variables.append(tuple(sorted(members)))
    for variable in range(dimension):
        if variable not in primary:
            leaf_variables.append((variable,))
    nodes = [RegionNode(0, None, 0, tuple(range(dimension)))]
    for index, members in enumerate(leaf_variables, start=1):
        nodes.append(RegionNode(index, 0, 1, members))
    tree = RegionTree(dimension=dimension, nodes=tuple(nodes))
    leaf_of_variable = {
        variable: leaf.node_id for leaf in tree.leaves for variable in leaf.variables
    }

    relations: list[RegionRelation] = []
    cross_edges: dict[int, list[tuple[int, SoftEdge]]] = {}
    for (left, right), edge in sorted(edges.items()):
        left_leaf = leaf_of_variable.get(left)
        right_leaf = leaf_of_variable.get(right)
        if left_leaf is None or right_leaf is None or left_leaf == right_leaf:
            continue
        relations.append(
            RegionRelation(
                left=left_leaf,
                right=right_leaf,
                score=edge.score,
                stability=edge.support / edge.probes,
                depth=1,
            )
        )
        cross_edges.setdefault(left, []).append((right, edge))
        cross_edges.setdefault(right, []).append((left, edge))

    # Shared candidates come from validated intersections of the refined
    # groups.  Removing the *complete* intersection is essential for AOB:
    # adjacent groups may share ten variables, so removing one variable can
    # never make the residual groups separable.
    interactions_by_key: dict[tuple[int, int, int], VariableRegionInteraction] = {}
    shared_targets: dict[int, set[int]] = {}
    candidate_shared_variables: set[int] = set()
    unresolved_shared_variables: set[int] = set()
    separability_rng = np.random.default_rng(run_seed ^ 0x5EED)
    separability_anchor = center + separability_rng.uniform(-0.2, 0.2, size=dimension) * span
    candidate_groups = tuple(refined_groups) if refinement_complete else ()
    for left_index, left_group in enumerate(candidate_groups):
        left_set = set(left_group)
        for right_index, right_group in enumerate(
            candidate_groups[left_index + 1 :], start=left_index + 1
        ):
            right_set = set(right_group)
            common = left_set & right_set
            left_residual = sorted(left_set - common)
            right_residual = sorted(right_set - common)
            if not common or not left_residual or not right_residual:
                continue
            # An intersection is a shared-variable candidate, not a proof.
            # Keep it separate from confirmed hyperedges so an unfinished or
            # rejected confirmation cannot be mistaken for separability.
            candidate_shared_variables.update(common)
            confirmed_common: list[int] = []
            tested_common: set[int] = set()
            for variable in sorted(common):
                if ledger.count - rdg_start + 6 > rdg_budget:
                    unresolved_shared_variables.update(set(common) - tested_common)
                    break
                tested_common.add(variable)
                interacts_left = _rdg_interact(
                    problem,
                    ledger,
                    set_a=[variable],
                    set_b=left_residual,
                    base_point=centre_point,
                    base_value=centre_value,
                    threshold=config.edge_threshold,
                )
                interacts_right = _rdg_interact(
                    problem,
                    ledger,
                    set_a=[variable],
                    set_b=right_residual,
                    base_point=centre_point,
                    base_value=centre_value,
                    threshold=config.edge_threshold,
                )
                if interacts_left and interacts_right:
                    confirmed_common.append(variable)
            common = set(confirmed_common)
            left_residual = sorted(left_set - common)
            right_residual = sorted(right_set - common)
            # A one- or two-variable residual cannot support a reliable
            # overlap claim under a mixed-difference probe; those are exactly
            # the tiny fragments produced by a noisy recursive split.
            if (
                not common
                or len(left_residual) < config.min_residual_size
                or len(right_residual) < config.min_residual_size
            ):
                if common and (
                    len(left_residual) < config.min_residual_size
                    or len(right_residual) < config.min_residual_size
                ):
                    unresolved_shared_variables.update(common)
                continue
            if not _block_separable(
                problem,
                ledger,
                separability_anchor,
                left_residual,
                right_residual,
                step=step,
                threshold=config.edge_threshold,
                probes=config.block_separability_probes,
                seed=run_seed ^ (0x7EA5 * (left_index + 1) ^ right_index),
                budget_start=rdg_start,
                budget=rdg_budget,
            ):
                unresolved_shared_variables.update(common)
                continue
            # The sign-randomized set test above is conservative but can
            # cancel on a tiny residual.  A deterministic mixed-difference
            # check provides a second direction and rejects fragments of the
            # same true component that happened to share one variable.
            if ledger.count - rdg_start + 3 > rdg_budget:
                unresolved_shared_variables.update(common)
                continue
            if _rdg_interact(
                problem,
                ledger,
                set_a=left_residual,
                set_b=right_residual,
                base_point=centre_point,
                base_value=centre_value,
                threshold=config.edge_threshold,
            ):
                unresolved_shared_variables.update(common)
                continue

            for variable in sorted(left_set | right_set):
                if variable in left_set - common:
                    continue
                source_leaf = leaf_of_variable[variable]
                opposite_members: set[int] = set()
                if variable in common:
                    # A primary leaf may belong to either side of the soft
                    # pair.  Keep both residual sides for a shared variable
                    # so the evidence is orientation-invariant.
                    opposite_members.update((left_set | right_set) - {variable})
                elif variable in left_set:
                    # Keep the shared boundary on the source side.  This
                    # mirrors the disjoint leaf contract and avoids claiming
                    # every left-only member as a cross-region bridge.
                    opposite_members.update(right_set - common)
                elif variable in right_set:
                    opposite_members.update(left_set)
                target_leaves = {
                    leaf_of_variable[target]
                    for target in opposite_members
                    if target != variable and leaf_of_variable[target] != source_leaf
                }
                for target_leaf in sorted(target_leaves):
                    key = (variable, source_leaf, target_leaf)
                    interactions_by_key.setdefault(
                        key,
                        VariableRegionInteraction(
                            variable=variable,
                            source_region=source_leaf,
                            target_region=target_leaf,
                            q_lb=1.0,
                            support=1,
                            sign_stability=1.0,
                        ),
                    )
                if variable in common:
                    shared_targets.setdefault(variable, set()).update(target_leaves)

    interactions = [interactions_by_key[key] for key in sorted(interactions_by_key)]
    hyperedges: list[ResolvedOverlapHyperedge] = []
    shared_candidates: list[int] = []
    for variable, targets in sorted(shared_targets.items()):
        source_leaf = leaf_of_variable[variable]
        targets.discard(source_leaf)
        if not targets:
            continue
        evidence_for_variable = tuple(
            interaction
            for interaction in interactions
            if interaction.variable == variable
            and interaction.source_region == source_leaf
            and interaction.target_region in targets
        )
        covered = {interaction.target_region for interaction in evidence_for_variable}
        if covered != targets:
            continue
        shared_candidates.append(variable)
        hyperedges.append(
            ResolvedOverlapHyperedge(
                variable=variable,
                regions=(source_leaf, *sorted(targets)),
                evidence=evidence_for_variable,
            )
        )

    adjacency: dict[int, set[int]] = {node.node_id: set() for node in tree.leaves}
    for relation in relations:
        adjacency[relation.left].add(relation.right)
        adjacency[relation.right].add(relation.left)
    unseen = set(adjacency)
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        stack_c, component = [root], set()
        while stack_c:
            node = stack_c.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack_c.extend(adjacency[node] - component)
        components.append(tuple(sorted(component)))

    confirmed_variables = set(shared_candidates)
    leaf_components = {
        leaf: component
        for component in components
        for leaf in component
    }
    unresolved_variables = (
        set(unresolved_shared_variables)
        | (candidate_shared_variables - confirmed_variables)
    )
    variable_status = tuple(
        (
            variable,
            "member_candidate"
            if variable in confirmed_variables
            else "not_yet_resolved"
            if variable in unresolved_variables or not refinement_complete
            else "observed_separable",
        )
        for variable in range(dimension)
    )
    per_component_mode = []
    for component in components:
        component_set = set(component)
        component_hyperedges = {
            hyperedge.variable
            for hyperedge in hyperedges
            if set(hyperedge.regions).issubset(component_set)
        }
        component_unresolved = {
            variable
            for variable in unresolved_variables
            if leaf_of_variable[variable] in component_set
        }
        if len(component) < 2:
            per_component_mode.append((component, "SPARSE"))
        elif not refinement_complete or component_unresolved or not component_hyperedges:
            # A local candidate without a confirming hyperedge is unresolved;
            # evidence from another component cannot close this component.
            per_component_mode.append((component, "HIERARCHICAL"))
        else:
            per_component_mode.append((component, "SPARSE"))

    evidence = Phase1Evidence(
        dimension=dimension,
        region_tree=tree,
        region_relations=tuple(sorted(relations, key=lambda item: (item.left, item.right))),
        variable_region_interactions=tuple(
            sorted(interactions, key=lambda item: item.variable)
        ),
        resolved_hyperedges=tuple(hyperedges),
        variable_status=variable_status,
        per_component_mode=tuple(per_component_mode),
        level_budgets=(
            ("signature", signature_result.consumed_fes),
            ("dsm", dsm_fes),
            ("rdg", ledger.count - rdg_start),
        ),
    )
    return SoftDiscoveryResult(
        evidence=evidence,
        signature_result=signature_result,
        edges=tuple(sorted(edges.values(), key=lambda edge: (edge.left, edge.right))),
        blocks=blocks,
        level_budgets=evidence.level_budgets,
        shared_candidates=tuple(shared_candidates),
        discovery_start_fes=discovery_start_fes,
        discovery_consumed_fes=discovery_consumed(),
        discovery_end_fes=ledger.count,
    )


__all__ = [
    "SoftDiscoveryResult",
    "SoftDsmConfig",
    "SoftEdge",
    "build_soft_dsm",
    "discover_hierarchical_soft",
    "soft_rddsm_blocks",
]
