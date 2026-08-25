"""Benchmark-independent action routing from a structural evidence certificate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from arac.evidence.overlap_adapter import Phase1OverlapEvidence


ACTION_AOR = "aor"
ACTION_CTP = "ctp"
ACTION_SMP = "smp"
ACTION_GCB = "gcb"


def _components(
    group_count: int,
    memberships: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    adjacency = {group: set() for group in range(group_count)}
    for owners in memberships:
        if len(owners) < 2:
            continue
        for left in owners:
            adjacency[left].update(right for right in owners if right != left)
    unseen = set(adjacency)
    result: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        queue = deque([root])
        component: set[int] = set()
        while queue:
            group = queue.popleft()
            if group in component:
                continue
            component.add(group)
            unseen.discard(group)
            queue.extend(sorted(adjacency[group] - component))
        result.append(tuple(sorted(component)))
    return tuple(result)


@dataclass(frozen=True)
class StructuralRouteDecision:
    """A deterministic primary route and static evidence summaries.

    ``action_name`` is the primary coordination route for this candidate; it
    is not an admissibility claim that excludes the other actions.  In
    particular, complete shared-variable evidence keeps SMP structurally
    compatible and selects its overlap-aware lifecycle mode even when CTP or
    GCB is the primary route.
    """

    action_name: str
    reason: str
    evidence_complete: bool
    smp_compatible: bool
    smp_mode: str
    shared_variables: tuple[int, ...]
    components: tuple[tuple[int, ...], ...]
    largest_component_fraction: float
    mean_shared_degree: float
    maximum_shared_degree: int
    component_weights: tuple[tuple[tuple[int, ...], float], ...]

    def __post_init__(self) -> None:
        if self.action_name not in {ACTION_AOR, ACTION_CTP, ACTION_SMP, ACTION_GCB}:
            raise ValueError("structural router selected an unsupported action")
        if not self.reason:
            raise ValueError("structural route reason must be non-empty")
        if not isinstance(self.evidence_complete, bool):
            raise ValueError("evidence completeness must be boolean")
        if not isinstance(self.smp_compatible, bool):
            raise ValueError("SMP compatibility must be boolean")
        if self.smp_mode not in {"unavailable", "zero_relation", "overlap_aware"}:
            raise ValueError("unsupported SMP structural mode")
        if self.smp_compatible != self.evidence_complete:
            raise ValueError("SMP compatibility must follow complete evidence")
        expected_smp_mode = (
            "unavailable"
            if not self.evidence_complete
            else "overlap_aware"
            if self.shared_variables
            else "zero_relation"
        )
        if self.smp_mode != expected_smp_mode:
            raise ValueError("SMP structural mode does not match overlap evidence")
        for value, name in (
            (self.largest_component_fraction, "largest_component_fraction"),
            (self.mean_shared_degree, "mean_shared_degree"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(self.maximum_shared_degree, bool) or self.maximum_shared_degree < 0:
            raise ValueError("maximum_shared_degree must be non-negative")
        weights = [weight for _, weight in self.component_weights]
        if any(not math.isfinite(float(weight)) or weight < 0.0 for weight in weights):
            raise ValueError("component weights must be finite and non-negative")
        if weights and abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("component weights must sum to one")


def _component_weights(
    evidence: Phase1OverlapEvidence,
    components: tuple[tuple[int, ...], ...],
    shared_variables: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], float], ...]:
    confidence = {
        (variable, group): float(value)
        for variable, group, value in evidence.membership_confidences
    }
    raw: list[float] = []
    for component in components:
        score = 0.0
        component_set = set(component)
        for variable in shared_variables:
            owners = set(evidence.memberships[variable])
            if not owners.intersection(component_set):
                continue
            degree_mass = max(0, len(owners) - 1)
            score += degree_mass * sum(confidence[(variable, group)] for group in owners)
        raw.append(score)
    total = sum(raw)
    if total <= 0.0:
        uniform = 1.0 / len(components) if components else 0.0
        return tuple((component, uniform) for component in components)
    return tuple((component, score / total) for component, score in zip(components, raw, strict=True))


def route_from_overlap_evidence(
    evidence: Phase1OverlapEvidence,
) -> StructuralRouteDecision:
    """Route by evidence sufficiency and overlap topology, never by outcomes.

    The route is intentionally structural: no objective scale, benchmark name,
    fitted selector, or online gain signal enters the decision.
    """

    if not isinstance(evidence, Phase1OverlapEvidence):
        raise TypeError("structural routing requires Phase1OverlapEvidence")
    shared_variables = tuple(
        variable
        for variable, owners in enumerate(evidence.memberships)
        if len(owners) > 1
    )
    components = _components(len(evidence.groups), evidence.memberships)
    largest = max((len(component) for component in components), default=0)
    largest_fraction = largest / len(evidence.groups) if evidence.groups else 0.0
    degrees = [len(evidence.memberships[variable]) for variable in shared_variables]
    mean_degree = sum(degrees) / len(degrees) if degrees else 0.0
    maximum_degree = max(degrees, default=0)
    weights = _component_weights(evidence, components, shared_variables)

    if not evidence.complete:
        action, reason = ACTION_AOR, "overlap_evidence_incomplete"
    elif not shared_variables:
        action, reason = ACTION_SMP, "complete_disjoint_structure"
    elif len(components) > 1:
        action, reason = ACTION_CTP, "complete_disconnected_overlap_components"
    else:
        action, reason = ACTION_GCB, "complete_connected_overlap_graph"

    smp_compatible = bool(evidence.complete)
    smp_mode = (
        "unavailable"
        if not evidence.complete
        else "overlap_aware"
        if shared_variables
        else "zero_relation"
    )

    return StructuralRouteDecision(
        action_name=action,
        reason=reason,
        evidence_complete=evidence.complete,
        smp_compatible=smp_compatible,
        smp_mode=smp_mode,
        shared_variables=shared_variables,
        components=components,
        largest_component_fraction=float(largest_fraction),
        mean_shared_degree=float(mean_degree),
        maximum_shared_degree=int(maximum_degree),
        component_weights=weights,
    )


__all__ = [
    "ACTION_AOR",
    "ACTION_CTP",
    "ACTION_GCB",
    "ACTION_SMP",
    "StructuralRouteDecision",
    "route_from_overlap_evidence",
]
