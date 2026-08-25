"""Convert hierarchical soft-RDDSM evidence into an actionable sidecar.

The hierarchical Phase-I schema deliberately keeps region leaves disjoint.
Phase-II actions, however, need an explicit variable-to-owner view.  This
module is the narrow, auditable bridge: leaves remain the partition used by
the legacy checkpoint while confirmed hyperedges add a variable to each
confirmed owner group in a separate :class:`Phase1OverlapEvidence` object.
"""

from __future__ import annotations

from typing import Any

from arac.evidence.hierarchical import Phase1Evidence
from arac.evidence.overlap_adapter import Phase1OverlapEvidence
from arac.runtime.contracts import canonical_sha256


def soft_evidence_to_overlap_evidence(
    evidence: Phase1Evidence,
) -> Phase1OverlapEvidence:
    """Build a variable-level overlap sidecar from confirmed hyperedges.

    Only ``ResolvedOverlapHyperedge`` records can add an owner.  Region-level
    interactions without a resolved hyperedge make the certificate
    incomplete, rather than silently becoming overlap memberships.
    """

    if not isinstance(evidence, Phase1Evidence):
        raise TypeError("soft-RDDSM evidence must be a Phase1Evidence instance")

    leaves = evidence.region_tree.leaves
    leaf_index = {leaf.node_id: index for index, leaf in enumerate(leaves)}
    groups: list[set[int]] = [set(leaf.variables) for leaf in leaves]
    confidence_by_pair: dict[tuple[int, int], float] = {
        (variable, group): 1.0
        for group, leaf in enumerate(leaves)
        for variable in leaf.variables
    }

    for hyperedge in evidence.resolved_hyperedges:
        quality_by_region = {
            interaction.target_region: min(
                float(interaction.q_lb),
                float(interaction.sign_stability),
            )
            for interaction in hyperedge.evidence
        }
        home_region = hyperedge.regions[0]
        for region in hyperedge.regions:
            group = leaf_index[region]
            groups[group].add(hyperedge.variable)
            quality = (
                min(quality_by_region.values(), default=1.0)
                if region == home_region
                else quality_by_region[region]
            )
            key = (hyperedge.variable, group)
            confidence_by_pair[key] = min(confidence_by_pair.get(key, 1.0), quality)

    memberships = tuple(
        tuple(group for group, variables in enumerate(groups) if variable in variables)
        for variable in range(evidence.dimension)
    )
    if any(not owners for owners in memberships):
        raise ValueError("soft-RDDSM leaves do not cover every variable")

    status_by_variable = dict(evidence.variable_status)
    confirmed_variables = {
        hyperedge.variable for hyperedge in evidence.resolved_hyperedges
    }
    unresolved = {
        variable
        for variable, status in status_by_variable.items()
        if status == "not_yet_resolved"
        or (status == "member_candidate" and variable not in confirmed_variables)
    }
    # Component completion is local.  A hyperedge in component A cannot close
    # an unresolved component B, and a non-SPARSE mode is never a proof of
    # disjointness.
    # An empty mode receipt means no component was explicitly closed.  It is
    # not equivalent to an all-SPARSE certificate, even if all variables
    # happen to carry an ``observed_separable`` status.
    unresolved_mode = (
        not evidence.per_component_mode
        or any(mode != "SPARSE" for _, mode in evidence.per_component_mode)
    )
    complete = not unresolved and not unresolved_mode

    confidences = tuple(
        (variable, group, float(confidence_by_pair[(variable, group)]))
        for variable, owners in enumerate(memberships)
        for group in owners
    )
    return Phase1OverlapEvidence(
        dimension=evidence.dimension,
        groups=tuple(tuple(sorted(variables)) for variables in groups),
        memberships=memberships,
        membership_confidences=confidences,
        complete=complete,
    )


def overlap_evidence_payload(evidence: Phase1OverlapEvidence) -> dict[str, Any]:
    """Return a deterministic, JSON-ready sidecar payload."""

    if not isinstance(evidence, Phase1OverlapEvidence):
        raise TypeError("overlap evidence payload requires Phase1OverlapEvidence")
    return {
        "schema_version": "arac-soft-rddsm-overlap-evidence-v1",
        "dimension": evidence.dimension,
        "groups": [list(group) for group in evidence.groups],
        "memberships": [list(owners) for owners in evidence.memberships],
        "membership_confidences": [
            [variable, group, confidence]
            for variable, group, confidence in evidence.membership_confidences
        ],
        "complete": evidence.complete,
    }


def overlap_evidence_hash(evidence: Phase1OverlapEvidence) -> str:
    """Hash the exact sidecar that is handed to Phase-II."""

    return canonical_sha256(overlap_evidence_payload(evidence))


__all__ = [
    "overlap_evidence_hash",
    "overlap_evidence_payload",
    "soft_evidence_to_overlap_evidence",
]
