"""Fail-closed boundary from Phase-I evidence to overlap coordination."""

from __future__ import annotations

from dataclasses import dataclass
import math

from arac.coordination.overlap import OverlapStructure
from arac.runtime.contracts import PhaseCheckpoint


INFERENCE_READY = "ready"
INFERENCE_INCOMPLETE = "inference_incomplete"
PARTITION_ONLY_REASON = "checkpoint_contains_partition_only"


@dataclass(frozen=True)
class Phase1OverlapEvidence:
    """Explicit variable-level evidence required by the coordination chain."""

    dimension: int
    groups: tuple[tuple[int, ...], ...]
    memberships: tuple[tuple[int, ...], ...]
    membership_confidences: tuple[tuple[int, int, float], ...]
    complete: bool

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError("evidence dimension must be a positive integer")
        if not isinstance(self.complete, bool):
            raise ValueError("evidence completeness must be boolean")
        for group in self.groups:
            if any(isinstance(variable, bool) or not isinstance(variable, int) for variable in group):
                raise ValueError("evidence group variables must be integers")
            if len(set(group)) != len(group) or any(
                variable < 0 or variable >= self.dimension for variable in group
            ):
                raise ValueError("evidence group variables must be unique and in bounds")
        for owners in self.memberships:
            if any(isinstance(owner, bool) or not isinstance(owner, int) for owner in owners):
                raise ValueError("evidence membership owners must be integers")
            if len(set(owners)) != len(owners) or any(
                owner < 0 or owner >= len(self.groups) for owner in owners
            ):
                raise ValueError("evidence membership owners must be unique and in bounds")
        confidence_keys = []
        for variable, group, confidence in self.membership_confidences:
            if (
                isinstance(variable, bool)
                or not isinstance(variable, int)
                or isinstance(group, bool)
                or not isinstance(group, int)
                or variable < 0
                or variable >= self.dimension
                or group < 0
                or group >= len(self.groups)
            ):
                raise ValueError("membership confidence references an unknown member")
            numeric = float(confidence)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError("membership confidence must be finite and in [0, 1]")
            confidence_keys.append((variable, group))
        if len(set(confidence_keys)) != len(confidence_keys):
            raise ValueError("membership confidences must be unique per variable and group")


@dataclass(frozen=True)
class Phase1OverlapAdaptation:
    """Result of attempting to make Phase-I evidence actionable."""

    status: str
    reason: str
    checkpoint_hash: str
    structure: OverlapStructure | None = None

    @property
    def ready(self) -> bool:
        return self.status == INFERENCE_READY


class Phase1OverlapAdapter:
    """Construct overlap structure only from explicit variable memberships."""

    def adapt(
        self,
        checkpoint: PhaseCheckpoint,
        evidence: Phase1OverlapEvidence | None = None,
    ) -> Phase1OverlapAdaptation:
        if not isinstance(checkpoint, PhaseCheckpoint):
            raise TypeError("overlap adaptation requires PhaseCheckpoint")
        if evidence is None:
            return self._incomplete(checkpoint, PARTITION_ONLY_REASON)
        if not isinstance(evidence, Phase1OverlapEvidence):
            raise TypeError("overlap evidence must be Phase1OverlapEvidence")
        if evidence.dimension != len(checkpoint.incumbent):
            raise ValueError("overlap evidence and checkpoint dimensions disagree")
        if not evidence.complete:
            return self._incomplete(checkpoint, "variable_membership_evidence_incomplete")
        if not evidence.groups or any(not group for group in evidence.groups):
            return self._incomplete(checkpoint, "overlap_groups_incomplete")
        if len(evidence.memberships) != evidence.dimension or any(
            not owners for owners in evidence.memberships
        ):
            return self._incomplete(checkpoint, "variable_memberships_incomplete")

        derived_memberships = self._derive_memberships(evidence)
        if evidence.memberships != derived_memberships:
            raise ValueError("variable memberships disagree with overlap groups")
        expected_confidences = {
            (variable, group)
            for variable, owners in enumerate(evidence.memberships)
            for group in owners
        }
        observed_confidences = {
            (variable, group)
            for variable, group, _ in evidence.membership_confidences
        }
        if observed_confidences != expected_confidences:
            return self._incomplete(checkpoint, "membership_confidence_incomplete")

        structure = OverlapStructure(
            dimension=evidence.dimension,
            groups=evidence.groups,
            member_confidences=evidence.membership_confidences,
        )
        return Phase1OverlapAdaptation(
            status=INFERENCE_READY,
            reason="variable_membership_evidence_complete",
            checkpoint_hash=checkpoint.checkpoint_hash,
            structure=structure,
        )

    @staticmethod
    def _derive_memberships(
        evidence: Phase1OverlapEvidence,
    ) -> tuple[tuple[int, ...], ...]:
        owners: list[list[int]] = [[] for _ in range(evidence.dimension)]
        for group, variables in enumerate(evidence.groups):
            for variable in variables:
                owners[variable].append(group)
        return tuple(tuple(items) for items in owners)

    @staticmethod
    def _incomplete(
        checkpoint: PhaseCheckpoint,
        reason: str,
    ) -> Phase1OverlapAdaptation:
        return Phase1OverlapAdaptation(
            status=INFERENCE_INCOMPLETE,
            reason=reason,
            checkpoint_hash=checkpoint.checkpoint_hash,
        )


__all__ = [
    "INFERENCE_INCOMPLETE",
    "INFERENCE_READY",
    "PARTITION_ONLY_REASON",
    "Phase1OverlapAdaptation",
    "Phase1OverlapAdapter",
    "Phase1OverlapEvidence",
]
