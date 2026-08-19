"""Five-stage hierarchical overlap discovery (Phase-I v10.2, §3).

Stage A  coarse screening over a fixed random hash partition
Stage A' billable variable signatures on a shared probe basis (Gate 43)
Stage B  spectral (Fiedler) reordering of the signature similarity
Stage C  recursive bisection of the reordered order with counted split tests
Stage D  conditional variable-region probes around significant boundaries
Stage E  incumbent completion under a hard floor

Every stage bills the shared ledger and reports its exact consumption; the
whole run stops at the frozen Phase-I boundary.  The output is a
``Phase1Evidence`` whose semantics are fixed by Gate 42.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.hierarchical import (
    EVIDENCE_MODES,
    Phase1Evidence,
    RegionNode,
    RegionRelation,
    RegionTree,
    ResolvedOverlapHyperedge,
    VariableRegionInteraction,
)
from arac.evidence.variable_signature import compute_variable_signatures
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort


@dataclass(frozen=True)
class HierarchicalDiscoveryConfig:
    """Pre-registered integers; only offline calibration may change them."""

    anchor_count: int = 5
    coarse_rounds: int = 6
    coarse_regions: int = 32
    signature_probe_count: int = 12
    signature_probe_size: int = 16
    step: float = 0.25
    min_region_size: int = 8
    max_depth: int = 7
    s_max: int = 900
    c_max: int = 1000
    per_split_candidates: int = 12
    neighbour_scan: int = 8
    two_sided_budget: int = 30_000
    k_dir: int = 3
    a_cond: int = 2
    incumbent_min: int = 50_000
    edge_threshold: float = 1e-10
    dense_region_density: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "anchor_count",
            "coarse_rounds",
            "coarse_regions",
            "min_region_size",
            "max_depth",
            "s_max",
            "c_max",
            "per_split_candidates",
            "k_dir",
            "a_cond",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.incumbent_min, bool)
            or not isinstance(self.incumbent_min, int)
            or self.incumbent_min < 0
        ):
            raise ValueError("incumbent_min must be a non-negative integer")
        if not 0.0 < float(self.edge_threshold) < 1.0:
            raise ValueError("edge_threshold must lie in (0, 1)")
        if not 0.0 < float(self.dense_region_density) <= 1.0:
            raise ValueError("dense_region_density must lie in (0, 1]")
        if self.min_region_size < 2:
            raise ValueError("min_region_size must be at least two")


@dataclass(frozen=True)
class GateAuditRecord:
    """Forensic record of one two-sided gate decision (pass or reject)."""

    variable: int
    target_region: int
    boundary: int | None
    neighbour: int | None
    passed: bool
    coupling_scores: tuple[float, ...]
    separability_scores: tuple[float, ...]


@dataclass(frozen=True)
class HierarchicalDiscoveryResult:
    evidence: Phase1Evidence
    order: tuple[int, ...]
    level_budgets: tuple[tuple[str, int], ...]
    split_scores: tuple[tuple[int, float], ...]
    conditional_probes: int
    incumbent_error: float
    gate_audit: tuple[GateAuditRecord, ...] = ()


def _anchors(problem: OptimizationProblem, count: int, seed: int) -> np.ndarray:
    center = (problem.lower_array + problem.upper_array) / 2.0
    span = problem.upper_array - problem.lower_array
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-0.2, 0.2, size=(count, problem.dimension)) * span


def _normalized_score(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    anchor: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    lower = problem.lower_array
    upper = problem.upper_array
    rows = np.asarray(
        (
            anchor,
            np.clip(anchor + (left != 0) * left, lower, upper),
            np.clip(anchor + (right != 0) * right, lower, upper),
            np.clip(anchor + left + right, lower, upper),
        ),
        dtype=float,
    )
    values = np.asarray(ledger.evaluate(rows), dtype=float)
    base, single_l, single_r, joint = (float(value) for value in values)
    scale = abs(base) + abs(single_l) + abs(single_r) + abs(joint) + 1.0
    return abs(joint - single_l - single_r + base) / scale


def _coarse_screen(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    anchors: np.ndarray,
    config: HierarchicalDiscoveryConfig,
    seed: int,
) -> tuple[dict[tuple[int, int], float], int]:
    rng = np.random.default_rng(seed ^ 0xC0A05E)
    dimension = problem.dimension
    assignment = np.concatenate([rng.permutation(dimension)]).reshape(
        config.coarse_regions, -1
    ) if dimension % config.coarse_regions == 0 else None
    if assignment is None:
        indices = rng.permutation(dimension)
        sizes = np.full(config.coarse_regions, dimension // config.coarse_regions)
        sizes[: dimension % config.coarse_regions] += 1
        assignment = np.split(indices, np.cumsum(sizes)[:-1])
    start = ledger.count
    scores: dict[tuple[int, int], list[float]] = {}
    pair_index = {
        (left, right): index
        for index, (left, right) in enumerate(
            (left, right)
            for left in range(config.coarse_regions)
            for right in range(left + 1, config.coarse_regions)
        )
    }
    lower = problem.lower_array
    upper = problem.upper_array
    for anchor in anchors:
        for _round in range(config.coarse_rounds):
            deltas = [
                config.step * rng.choice((-1.0, 1.0), size=len(region))
                for region in assignment
            ]
            rows = [anchor]
            for delta, region in zip(deltas, assignment, strict=True):
                row = anchor.copy()
                row[region] += delta
                rows.append(row)
            for left in range(config.coarse_regions):
                for right in range(left + 1, config.coarse_regions):
                    row = anchor.copy()
                    row[assignment[left]] += deltas[left]
                    row[assignment[right]] += deltas[right]
                    rows.append(row)
            batch = np.clip(np.asarray(rows, dtype=float), lower, upper)
            values = np.asarray(ledger.evaluate(batch), dtype=float)
            base = float(values[0])
            singles = values[1 : 1 + config.coarse_regions]
            joints = values[1 + config.coarse_regions :]
            for (left, right), index in pair_index.items():
                joint = float(joints[index])
                single_l = float(singles[left])
                single_r = float(singles[right])
                scale = abs(base) + abs(single_l) + abs(single_r) + abs(joint) + 1.0
                scores.setdefault((left, right), []).append(
                    abs(joint - single_l - single_r + base) / scale
                )
    stability = {
        pair: float(np.mean(np.asarray(values) > config.edge_threshold))
        for pair, values in scores.items()
    }
    return stability, ledger.count - start


def _fiedler_order(signatures: np.ndarray, seed: int) -> tuple[int, ...]:
    dimension = signatures.shape[0]
    norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    normalized = signatures / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 0.0)
    degree = similarity.sum(axis=1)
    laplacian = np.diag(degree) - similarity
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, int(np.argsort(eigenvalues)[1])]
    if float(np.dot(fiedler, np.arange(dimension))) < 0.0:
        fiedler = -fiedler
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.0, 1e-9, size=dimension)
    # Variables without any interaction evidence (zero signatures) park at the
    # tail deterministically: they are observed-separable either way.
    inert = (norms[:, 0] <= 1e-12).astype(int)
    keys = np.lexsort((fiedler + noise, inert))
    return tuple(int(index) for index in keys)


def discover_hierarchical(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    run_seed: int,
    config: HierarchicalDiscoveryConfig | None = None,
) -> HierarchicalDiscoveryResult:
    """Run the five-stage discovery inside the ledger's remaining budget."""

    if not isinstance(problem, OptimizationProblem):
        raise TypeError("problem must be OptimizationProblem")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be EvaluationLedger")
    if ledger.problem is not problem:
        raise ValueError("discovery requires the ledger for the same problem")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed < 0:
        raise ValueError("run_seed must be a non-negative integer")
    config = HierarchicalDiscoveryConfig() if config is None else config
    if not isinstance(config, HierarchicalDiscoveryConfig):
        raise TypeError("config must be HierarchicalDiscoveryConfig")
    dimension = problem.dimension

    anchors = _anchors(problem, config.anchor_count, run_seed)
    stability, coarse_fes = _coarse_screen(
        problem, ledger, anchors, config, run_seed
    )
    signature_result = compute_variable_signatures(
        problem,
        ledger,
        anchor=anchors[0],
        step=config.step,
        probe_count=config.signature_probe_count,
        probe_size=config.signature_probe_size,
        seed=run_seed ^ 0x516,
    )
    order = _fiedler_order(signature_result.signatures, run_seed ^ 0xF1ED)

    # Stage C: recursive bisection on the reordered variables.
    rng = np.random.default_rng(run_seed ^ 0x5B1C)
    nodes: list[RegionNode] = [RegionNode(0, None, 0, tuple(order))]
    split_score: dict[int, float] = {}
    split_children: dict[int, tuple[int, int]] = {}
    stack = [(0, tuple(order), 0)]
    split_fes = 0
    while stack:
        node_id, variables, depth = stack.pop()
        if (
            len(variables) <= config.min_region_size
            or depth >= config.max_depth
            or len(nodes) - 1 >= config.s_max
        ):
            continue
        # Always refine to the minimum size: the signature order concentrates
        # interacting variables at the head, and dyadic split points only
        # reach them after several levels, so an insignificant split must not
        # stop refinement of the region that contains the structure.
        mid = len(variables) // 2
        left_vars, right_vars = variables[:mid], variables[mid:]
        score = 0.0
        for anchor in anchors[: config.a_cond]:
            delta_l = np.zeros(dimension)
            delta_l[list(left_vars)] = config.step * rng.choice(
                (-1.0, 1.0), size=len(left_vars)
            )
            delta_r = np.zeros(dimension)
            delta_r[list(right_vars)] = config.step * rng.choice(
                (-1.0, 1.0), size=len(right_vars)
            )
            score = max(
                score,
                _normalized_score(problem, ledger, anchor, delta_l, delta_r),
            )
            split_fes += 4
        split_score[node_id] = score
        left_id, right_id = len(nodes), len(nodes) + 1
        nodes.append(RegionNode(left_id, node_id, depth + 1, left_vars))
        nodes.append(RegionNode(right_id, node_id, depth + 1, right_vars))
        split_children[node_id] = (left_id, right_id)
        stack.append((left_id, left_vars, depth + 1))
        stack.append((right_id, right_vars, depth + 1))

    tree = RegionTree(dimension=dimension, nodes=tuple(nodes))
    leaf_of_variable = {
        variable: leaf.node_id for leaf in tree.leaves for variable in leaf.variables
    }

    def _lca_split(a: int, b: int) -> int | None:
        ancestors = {a}
        cursor = tree.nodes[a]
        while cursor.parent is not None:
            ancestors.add(cursor.parent)
            cursor = tree.nodes[cursor.parent]
        node = tree.nodes[b]
        while node.node_id not in ancestors:
            if node.parent is None:
                return None
            node = tree.nodes[node.parent]
        return node.node_id

    relations: list[RegionRelation] = []
    leaves = [leaf.node_id for leaf in tree.leaves]
    # A significant split carries ONE leaf-level relation: between the leaves
    # containing the two boundary variables.  Global-order imprecision must
    # not inflate one significant split into thousands of leaf pairs.
    for node_id, score in sorted(split_score.items()):
        if score <= config.edge_threshold or node_id not in split_children:
            continue
        left_vars = tree.nodes[split_children[node_id][0]].variables
        right_vars = tree.nodes[split_children[node_id][1]].variables
        if not left_vars or not right_vars:
            continue
        left_leaf = leaf_of_variable[left_vars[-1]]
        right_leaf = leaf_of_variable[right_vars[0]]
        if left_leaf == right_leaf:
            continue
        relations.append(
            RegionRelation(
                left=left_leaf,
                right=right_leaf,
                score=float(score),
                stability=1.0,
                depth=tree.nodes[node_id].depth,
            )
        )
    deduplicated: dict[tuple[int, int], RegionRelation] = {}
    for relation in relations:
        key = (min(relation.left, relation.right), max(relation.left, relation.right))
        if key not in deduplicated or relation.score > deduplicated[key].score:
            deduplicated[key] = relation
    relations = list(deduplicated.values())

    # Stage D: conditional probes around significant split boundaries.
    interactions: list[VariableRegionInteraction] = []
    probed_pairs = 0
    probe_budget_start = ledger.count
    significant_splits = [
        node_id
        for node_id, score in split_score.items()
        if score > config.edge_threshold and node_id in split_children
    ]
    candidate_targets: dict[int, int] = {}
    for node_id in significant_splits:
        left_id, right_id = split_children[node_id]
        left_vars = tree.nodes[left_id].variables
        right_vars = tree.nodes[right_id].variables
        if not left_vars or not right_vars:
            continue
        # Boundary-adjacent variables probe toward the leaf that actually
        # contains the opposite side of the split boundary.
        left_target = leaf_of_variable[right_vars[0]]
        right_target = leaf_of_variable[left_vars[-1]]
        for variable in left_vars[-config.per_split_candidates :]:
            candidate_targets[variable] = left_target
        for variable in right_vars[: config.per_split_candidates]:
            candidate_targets[variable] = right_target
    for variable, target_leaf in candidate_targets.items():
        if probed_pairs >= config.c_max:
            break
        home_leaf = leaf_of_variable[variable]
        if home_leaf == target_leaf:
            continue
        stable = 0
        total = 0
        for anchor in anchors[: config.a_cond]:
            for direction in range(config.k_dir):
                delta_j = np.zeros(dimension)
                delta_j[variable] = config.step * (1.0 if direction % 2 == 0 else -1.0)
                delta_g = np.zeros(dimension)
                members = tree.nodes[target_leaf].variables
                delta_g[list(members)] = config.step * rng.choice(
                    (-1.0, 1.0), size=len(members)
                )
                score = _normalized_score(problem, ledger, anchor, delta_j, delta_g)
                total += 1
                stable += int(score > config.edge_threshold)
        probed_pairs += 1
        if total == 0:
            continue
        fraction = stable / total
        support = total
        lower = max(0.0, (fraction * support + 0.5) / (support + 1.0) - 1.96 * math.sqrt(
            max(fraction * (1.0 - fraction), 1e-6) / support
        ))
        if fraction >= 5.0 / 6.0:
            interactions.append(
                VariableRegionInteraction(
                    variable=variable,
                    source_region=home_leaf,
                    target_region=target_leaf,
                    q_lb=float(min(1.0, lower)),
                    support=support,
                    sign_stability=float(fraction),
                )
            )

    # Two-sided evidence gate (triple form): a true shared variable j couples
    # its own side (neighbour h) AND the target side (boundary variable t),
    # while those two territories are mutually separable.  A fragmentation
    # artifact's h and t sit inside the same group and interact.  Single-
    # variable pairs survive leaf contamination far better than leaf-level
    # sets.
    order_position = {variable: position for position, variable in enumerate(order)}
    two_sided_passed: list[VariableRegionInteraction] = []
    gate_audit: list[GateAuditRecord] = []
    gate_start = ledger.count
    for interaction in interactions:
        variable = interaction.variable
        position = order_position[variable]
        if position == 0:
            continue
        neighbour = order[position - 1]
        target_members = tree.require_leaf(interaction.target_region).variables
        if not target_members:
            continue
        # t must be a member of T that INDIVIDUALLY couples with j; an
        # aggregate-only trigger is leaf contamination, not j's partner.
        boundary = None
        for member in sorted(
            target_members,
            key=lambda item: (abs(order_position.get(item, 10**9) - position), item),
        ):
            fires = False
            for anchor in anchors[: config.a_cond]:
                delta_j = np.zeros(dimension)
                delta_j[variable] = config.step
                delta_m = np.zeros(dimension)
                delta_m[member] = config.step * rng.choice((-1.0, 1.0))
                if _normalized_score(problem, ledger, anchor, delta_j, delta_m) > config.edge_threshold:
                    fires = True
                    break
            if fires:
                boundary = member
                break
        if boundary is None:
            gate_audit.append(
                GateAuditRecord(
                    variable=variable,
                    target_region=interaction.target_region,
                    boundary=None,
                    neighbour=None,
                    passed=False,
                    coupling_scores=(),
                    separability_scores=(),
                )
            )
            continue
        # h is SEARCHED in j's order neighbourhood (radius
        # neighbour_scan, both directions): the first variable that couples
        # with j and is separable from t completes the triple.  Fixed
        # neighbours fail because shared variables often sit inside their
        # group territory, not at cluster interfaces.
        passed = False
        for radius in range(1, config.neighbour_scan + 1):
            for offset in (-radius, radius):
                npos = position + offset
                if not 0 <= npos < dimension:
                    continue
                neighbour = order[npos]
                if neighbour == boundary or neighbour == variable:
                    continue
                if ledger.count - gate_start + 8 * config.a_cond * 4 > config.two_sided_budget:
                    break
                # Coupling must fire on at least one anchor; separability
                # must hold on every anchor (two-anchor trade-off variant:
                # R6 recall 0.12 at precision 0.92, near-zero F1 false
                # positives; the per-anchor direction replication priced the
                # gate out of its budget and is deferred).
                delta_j = np.zeros(dimension)
                delta_j[variable] = config.step
                delta_h = np.zeros(dimension)
                delta_h[neighbour] = config.step * rng.choice((-1.0, 1.0))
                coupling_raw = [
                    _normalized_score(problem, ledger, anchor, delta_j, delta_h)
                    for anchor in anchors[: config.a_cond]
                ]
                couples = any(score > config.edge_threshold for score in coupling_raw)
                if not couples:
                    continue
                delta_t = np.zeros(dimension)
                delta_t[boundary] = config.step * rng.choice((-1.0, 1.0))
                separability_raw = [
                    _normalized_score(problem, ledger, anchor, delta_h, delta_t)
                    for anchor in anchors[: config.a_cond]
                ]
                separable = all(score <= config.edge_threshold for score in separability_raw)
                gate_audit.append(
                    GateAuditRecord(
                        variable=variable,
                        target_region=interaction.target_region,
                        boundary=boundary,
                        neighbour=neighbour,
                        passed=bool(separable),
                        coupling_scores=tuple(coupling_raw),
                        separability_scores=tuple(separability_raw),
                    )
                )
                if not separable:
                    continue
                passed = True
                break
            if passed:
                break
        if ledger.count - gate_start >= config.two_sided_budget:
            break
        if passed:
            two_sided_passed.append(interaction)
    conditional_fes = ledger.count - probe_budget_start

    hyperedges = [
        ResolvedOverlapHyperedge(
            variable=interaction.variable,
            regions=(interaction.source_region, interaction.target_region),
            evidence=(interaction,),
        )
        for interaction in two_sided_passed
    ]
    member_variables = {interaction.variable for interaction in interactions}
    variable_status = tuple(
        (
            variable,
            "member_candidate" if variable in member_variables else "observed_separable",
        )
        for variable in range(dimension)
    )
    adjacency: dict[int, set[int]] = {leaf: set() for leaf in leaves}
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
    coarse_density = (
        sum(1 for value in stability.values() if value > 0.5) / len(stability)
        if stability
        else 0.0
    )
    per_component_mode = []
    for component in components:
        if len(component) < 2:
            # An isolated leaf is locally homogeneous: its variables are
            # observed separable, which is the SPARSE outcome.
            per_component_mode.append((component, "SPARSE"))
            continue
        if hyperedges:
            mode = "SPARSE"
        elif coarse_density >= config.dense_region_density and interactions:
            mode = "EVIDENCE_DENSE"
        elif interactions:
            mode = "HIERARCHICAL"
        else:
            mode = "SPARSE"
        per_component_mode.append((component, mode))

    evidence = Phase1Evidence(
        dimension=dimension,
        region_tree=tree,
        region_relations=tuple(
            sorted(relations, key=lambda item: (item.left, item.right))
        ),
        variable_region_interactions=tuple(
            sorted(interactions, key=lambda item: item.variable)
        ),
        resolved_hyperedges=tuple(hyperedges),
        variable_status=variable_status,
        per_component_mode=tuple(per_component_mode),
        level_budgets=(
            ("coarse", coarse_fes),
            ("signature", signature_result.consumed_fes),
            ("splits", split_fes),
            ("conditional", conditional_fes),
        ),
    )
    return HierarchicalDiscoveryResult(
        evidence=evidence,
        order=order,
        level_budgets=evidence.level_budgets,
        split_scores=tuple(sorted(split_score.items())),
        conditional_probes=probed_pairs,
        incumbent_error=float(ledger.best_error),
        gate_audit=tuple(gate_audit),
    )


def complete_incumbent(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    run_seed: int,
    incumbent_min: int,
) -> int:
    """Stage E: spend the remaining Phase-I budget on incumbent quality."""

    if ledger.remaining < incumbent_min:
        raise RuntimeError("hierarchical discovery violated the incumbent floor")
    if ledger.remaining:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=int(run_seed) ^ 0xE71D_3A26,
            budget_fes=ledger.remaining,
            population_size=24,
            restart=False,
        )
    return ledger.count


__all__ = [
    "EVIDENCE_MODES",
    "HierarchicalDiscoveryConfig",
    "HierarchicalDiscoveryResult",
    "complete_incumbent",
    "discover_hierarchical",
]
