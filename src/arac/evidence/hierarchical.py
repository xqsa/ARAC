"""Three-layer hierarchical overlap evidence (Phase-I v10.2, Gate 42).

Layer 1 ``RegionRelation``: two disjoint region leaves interact. Nothing more.
Layer 2 ``VariableRegionInteraction``: a variable in its home leaf conditionally
interacts with a target leaf. This is NOT membership — leaves are disjoint and
no variable can belong to two of them.
Layer 3 ``ResolvedOverlapHyperedge``: dual-evidence-confirmed shared-variable
assignments. The ONLY legal input for building an ``OverlapStructure``.

Schema hard constraints (design doc ``docs/arac-phase1-v10-design.md`` §1):
- No field path expresses "variable belongs to two disjoint leaves";
- ``to_overlap_structure`` fails closed unless resolved hyperedges exist;
- Every hyperedge carries its supporting interactions as an audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

EVIDENCE_MODES = ("SPARSE", "HIERARCHICAL", "EVIDENCE_DENSE")
VARIABLE_STATUSES = ("observed_separable", "member_candidate", "not_yet_resolved")
MIN_SIGN_STABILITY = 5.0 / 6.0


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _leaf_ids(value, name: str) -> tuple[int, ...]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RegionNode:
    """One node of the immutable region bisection tree."""

    node_id: int
    parent: int | None
    depth: int
    variables: tuple[int, ...]

    def __post_init__(self) -> None:
        _leaf_ids(self.node_id, "node_id")
        if self.parent is not None:
            _leaf_ids(self.parent, "parent")
        _leaf_ids(self.depth, "depth")
        if isinstance(self.variables, bool) or not isinstance(self.variables, tuple):
            raise ValueError("variables must be a tuple")
        if not self.variables:
            raise ValueError("region nodes must contain at least one variable")
        seen = set()
        for variable in self.variables:
            _leaf_ids(variable, "variable index")
            if variable in seen:
                raise ValueError("region variables must be unique")
            seen.add(variable)


@dataclass(frozen=True)
class RegionTree:
    """An immutable bisection tree whose leaves partition all variables."""

    dimension: int
    nodes: tuple[RegionNode, ...]

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not self.nodes:
            raise ValueError("region tree requires at least one node")
        by_id = {}
        for node in self.nodes:
            if not isinstance(node, RegionNode):
                raise TypeError("region tree nodes must be RegionNode instances")
            if node.node_id in by_id:
                raise ValueError("region node ids must be unique")
            by_id[node.node_id] = node
        for node in self.nodes:
            if node.parent is None:
                if node.depth != 0:
                    raise ValueError("the root must have depth zero")
            else:
                if node.parent not in by_id:
                    raise ValueError("region parent references an unknown node")
                if by_id[node.parent].depth != node.depth - 1:
                    raise ValueError("region depths must increase by one per level")
            for variable in node.variables:
                if variable >= self.dimension:
                    raise ValueError("region variables must stay below the dimension")
        covered = sorted(variable for leaf in self.leaves for variable in leaf.variables)
        if covered != list(range(self.dimension)):
            raise ValueError("region leaves must partition every variable exactly once")

    @property
    def leaves(self) -> tuple[RegionNode, ...]:
        children_exist = {node.parent for node in self.nodes if node.parent is not None}
        return tuple(node for node in self.nodes if node.node_id not in children_exist)

    def leaf_of(self, variable: int) -> RegionNode:
        for leaf in self.leaves:
            if variable in leaf.variables:
                return leaf
        raise ValueError(f"variable {variable} is not covered by any leaf")

    def require_leaf(self, node_id: int) -> RegionNode:
        for leaf in self.leaves:
            if leaf.node_id == _leaf_ids(node_id, "node_id"):
                return leaf
        raise ValueError(f"node {node_id} is not a region leaf")


@dataclass(frozen=True)
class RegionRelation:
    """Layer 1: region leaf r and region leaf g interact. No membership claim."""

    left: int
    right: int
    score: float
    stability: float
    depth: int

    def __post_init__(self) -> None:
        left = _leaf_ids(self.left, "left")
        right = _leaf_ids(self.right, "right")
        if left == right:
            raise ValueError("a region relation must connect two distinct leaves")
        _finite(self.score, "score")
        _unit(self.stability, "stability")
        _leaf_ids(self.depth, "depth")


@dataclass(frozen=True)
class VariableRegionInteraction:
    """Layer 2: variable j (home leaf = source) conditionally interacts with target.

    This is interaction evidence, NOT membership: leaves are disjoint and no
    schema field may express that j belongs to the target leaf.
    """

    variable: int
    source_region: int
    target_region: int
    q_lb: float
    support: int
    sign_stability: float

    def __post_init__(self) -> None:
        _leaf_ids(self.variable, "variable")
        source = _leaf_ids(self.source_region, "source_region")
        target = _leaf_ids(self.target_region, "target_region")
        if source == target:
            raise ValueError("a variable cannot interact with its own home leaf")
        _unit(self.q_lb, "q_lb")
        if isinstance(self.support, bool) or not isinstance(self.support, int) or self.support < 0:
            raise ValueError("support must be a non-negative integer")
        _unit(self.sign_stability, "sign_stability")


@dataclass(frozen=True)
class ResolvedOverlapHyperedge:
    """Layer 3: one shared-variable assignment confirmed by dual evidence.

    ``regions[0]`` is the variable's home leaf; every further region must be
    backed by an embedded interaction whose sign stability clears the
    candidate threshold.
    """

    variable: int
    regions: tuple[int, ...]
    evidence: tuple[VariableRegionInteraction, ...]

    def __post_init__(self) -> None:
        _leaf_ids(self.variable, "variable")
        if not isinstance(self.regions, tuple) or len(self.regions) < 2:
            raise ValueError("a resolved hyperedge must span at least two regions")
        if len(set(self.regions)) != len(self.regions):
            raise ValueError("hyperedge regions must be unique")
        if not isinstance(self.evidence, tuple):
            raise ValueError("hyperedge evidence must be a tuple")
        targets = set(self.regions[1:])
        covered = set()
        for interaction in self.evidence:
            if not isinstance(interaction, VariableRegionInteraction):
                raise TypeError("hyperedge evidence must be VariableRegionInteraction instances")
            if interaction.variable != self.variable:
                raise ValueError("hyperedge evidence must reference the same variable")
            if interaction.source_region != self.regions[0]:
                raise ValueError("hyperedge evidence must originate at the home region")
            if interaction.target_region not in targets:
                raise ValueError("hyperedge evidence must target a spanned region")
            if interaction.sign_stability < MIN_SIGN_STABILITY:
                raise ValueError("hyperedge evidence must clear the sign-stability threshold")
            covered.add(interaction.target_region)
        if covered != targets:
            raise ValueError("every spanned region needs confirming evidence")


@dataclass(frozen=True)
class Phase1Evidence:
    """Immutable hierarchical Phase-I evidence (v10.2 schema)."""

    dimension: int
    region_tree: RegionTree
    region_relations: tuple[RegionRelation, ...]
    variable_region_interactions: tuple[VariableRegionInteraction, ...]
    resolved_hyperedges: tuple[ResolvedOverlapHyperedge, ...] = ()
    variable_status: tuple[tuple[int, str], ...] = ()
    per_component_mode: tuple[tuple[tuple[int, ...], str], ...] = ()
    level_budgets: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(self.region_tree, RegionTree):
            raise TypeError("region_tree must be a RegionTree")
        if self.region_tree.dimension != self.dimension:
            raise ValueError("region tree and evidence dimensions disagree")
        for relation in self.region_relations:
            if not isinstance(relation, RegionRelation):
                raise TypeError("region_relations must be RegionRelation instances")
            self.region_tree.require_leaf(relation.left)
            self.region_tree.require_leaf(relation.right)
        known_interactions = set()
        for interaction in self.variable_region_interactions:
            if not isinstance(interaction, VariableRegionInteraction):
                raise TypeError("variable_region_interactions must be VariableRegionInteraction instances")
            home = self.region_tree.leaf_of(interaction.variable)
            if home.node_id != interaction.source_region:
                raise ValueError("interaction source must be the variable's home leaf")
            self.region_tree.require_leaf(interaction.target_region)
            known_interactions.add(interaction)
        for hyperedge in self.resolved_hyperedges:
            if not isinstance(hyperedge, ResolvedOverlapHyperedge):
                raise TypeError("resolved_hyperedges must be ResolvedOverlapHyperedge instances")
            home = self.region_tree.leaf_of(hyperedge.variable)
            if home.node_id != hyperedge.regions[0]:
                raise ValueError("hyperedge home region must contain the variable")
            for region in hyperedge.regions[1:]:
                self.region_tree.require_leaf(region)
            for interaction in hyperedge.evidence:
                if interaction not in known_interactions:
                    raise ValueError("hyperedge evidence must be part of the recorded interactions")
        status_map = dict(self.variable_status)
        if sorted(status_map) != list(range(self.dimension)):
            raise ValueError("variable_status must cover every variable exactly once")
        for status in status_map.values():
            if status not in VARIABLE_STATUSES:
                raise ValueError(f"unknown variable status: {status}")
        covered_leaves = set()
        for component, mode in self.per_component_mode:
            if mode not in EVIDENCE_MODES:
                raise ValueError(f"unknown evidence mode: {mode}")
            if not isinstance(component, tuple) or not component:
                raise ValueError("mode components must be non-empty leaf tuples")
            for leaf in component:
                self.region_tree.require_leaf(leaf)
                if leaf in covered_leaves:
                    raise ValueError("mode components must not share leaves")
                covered_leaves.add(leaf)
        if covered_leaves and covered_leaves != {leaf.node_id for leaf in self.region_tree.leaves}:
            raise ValueError("mode components must partition the leaves")
        for name, fes in self.level_budgets:
            if not isinstance(name, str) or not name:
                raise ValueError("budget names must be non-empty strings")
            if isinstance(fes, bool) or not isinstance(fes, int) or fes < 0:
                raise ValueError("budget entries must be non-negative integers")


def to_overlap_structure(evidence: Phase1Evidence):
    """Build the variable-level structure from confirmed hyperedges only.

    Fail-closed: without resolved hyperedges, region evidence cannot masquerade
    as variable overlap, so this raises instead of returning an empty structure.
    """

    # Deferred import: importing the coordinator-side structure at module
    # level would create a package cycle (evidence <-> coordination).
    from arac.coordination.overlap import OverlapStructure

    if not isinstance(evidence, Phase1Evidence):
        raise TypeError("to_overlap_structure requires Phase1Evidence")
    if not evidence.resolved_hyperedges:
        raise ValueError(
            "no resolved overlap hyperedges: region-level evidence cannot be "
            "converted into variable-level overlap structure"
        )
    leaves = evidence.region_tree.leaves
    leaf_index = {leaf.node_id: index for index, leaf in enumerate(leaves)}
    groups = [list(leaf.variables) for leaf in leaves]
    for hyperedge in evidence.resolved_hyperedges:
        for region in hyperedge.regions:
            if hyperedge.variable not in groups[leaf_index[region]]:
                groups[leaf_index[region]].append(hyperedge.variable)
    return OverlapStructure(
        dimension=evidence.dimension,
        groups=tuple(tuple(sorted(group)) for group in groups),
    )


@dataclass(frozen=True)
class RegionStructure:
    """Coordinator-facing view of region-level evidence.

    Deliberately provides no conversion to variable-level structures; the only
    legal bridge is ``to_overlap_structure`` on ``Phase1Evidence``, gated by
    resolved hyperedges.
    """

    evidence: Phase1Evidence

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, Phase1Evidence):
            raise TypeError("RegionStructure requires Phase1Evidence")

    @property
    def tree(self) -> RegionTree:
        return self.evidence.region_tree

    @property
    def relations(self) -> tuple[RegionRelation, ...]:
        return self.evidence.region_relations

    def region_variables(self, leaf_id: int) -> tuple[int, ...]:
        return self.tree.require_leaf(leaf_id).variables

    def partners(self, leaf_id: int) -> tuple[int, ...]:
        self.tree.require_leaf(leaf_id)
        found = set()
        for relation in self.evidence.region_relations:
            if relation.left == leaf_id:
                found.add(relation.right)
            elif relation.right == leaf_id:
                found.add(relation.left)
        return tuple(sorted(found))

    def components(self) -> tuple[tuple[int, ...], ...]:
        adjacency: dict[int, set[int]] = {
            leaf.node_id: set() for leaf in self.tree.leaves
        }
        for relation in self.evidence.region_relations:
            adjacency[relation.left].add(relation.right)
            adjacency[relation.right].add(relation.left)
        unseen = set(adjacency)
        components = []
        while unseen:
            root = min(unseen)
            stack, component = [root], set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                unseen.discard(node)
                stack.extend(adjacency[node] - component)
            components.append(tuple(sorted(component)))
        return tuple(components)


def mode_of_component(evidence: Phase1Evidence, component: tuple[int, ...]) -> str:
    """Return the declared evidence mode of one region component."""

    component = tuple(component)
    for declared, mode in evidence.per_component_mode:
        if declared == component:
            return mode
    raise ValueError(f"component {component} has no declared evidence mode")


__all__ = [
    "EVIDENCE_MODES",
    "MIN_SIGN_STABILITY",
    "Phase1Evidence",
    "RegionNode",
    "RegionRelation",
    "RegionStructure",
    "RegionTree",
    "ResolvedOverlapHyperedge",
    "VARIABLE_STATUSES",
    "VariableRegionInteraction",
    "mode_of_component",
    "to_overlap_structure",
]
