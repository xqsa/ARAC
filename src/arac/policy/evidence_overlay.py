"""Pure, reference-blind policy primitives for the RDDSM evidence overlay."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace

from .overlap_hypergraph import midrank_percentiles


EVIDENCE_OVERLAY_SCHEMA_VERSION = "rddsm-evidence-overlay-v1"
SHUFFLE_SALT = "arac-evidence-overlay-shuffled-v1"
TOP_RELATION_COUNT = 4
LOCAL_OPTIMUM_TOP_K = 5
PROPOSAL_DISAGREEMENT_METRIC = "mean_normalized_gaussian_w2"
SHADOW_GAIN_THRESHOLD = math.log(1.01)
UTILITY_EPSILON = 1e-300
FORBIDDEN_RUNTIME_FIELD_FRAGMENTS = (
    "aob",
    "case",
    "problem",
    "family",
    "identity",
    "paper",
    "oracle",
    "historical",
    "outcome",
    "terminal",
    "final",
    "catastrophic",
)


def _integer(value: int, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    converted = int(value)
    if converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return converted


def _finite(value: float, *, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _unit_interval(value: float, *, name: str) -> float:
    converted = _finite(value, name=name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return converted


def _canonical_groups(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    converted: list[tuple[int, ...]] = []
    for group_index, raw_group in enumerate(groups):
        group = tuple(
            _integer(variable, name=f"groups[{group_index}] variable")
            for variable in raw_group
        )
        if not group:
            raise ValueError("RDDSM groups must be non-empty")
        if len(set(group)) != len(group):
            raise ValueError("RDDSM groups must not contain duplicate variables")
        converted.append(tuple(sorted(group)))
    if not converted:
        raise ValueError("RDDSM grouping must contain at least one group")
    return tuple(converted)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def topology_sha256(groups: Sequence[Sequence[int]]) -> str:
    """Hash the unordered RDDSM membership topology, preserving multiplicity."""

    canonical = _canonical_groups(groups)
    return _canonical_sha256(
        {
            "schema_version": EVIDENCE_OVERLAY_SCHEMA_VERSION,
            "groups": sorted(canonical),
        }
    )


def ordering_sha256(
    groups: Sequence[Sequence[int]],
    group_order: Sequence[int],
) -> str:
    """Hash the structural topology together with its ordered memberships."""

    canonical = _canonical_groups(groups)
    order = tuple(
        _integer(value, name="group_order entry") for value in group_order
    )
    if tuple(sorted(order)) != tuple(range(len(canonical))):
        raise ValueError("group_order must be a permutation of every group index")
    return _canonical_sha256(
        {
            "schema_version": EVIDENCE_OVERLAY_SCHEMA_VERSION,
            "topology_sha256": topology_sha256(canonical),
            "ordered_groups": [canonical[index] for index in order],
        }
    )


@dataclass(frozen=True)
class ReferenceBlindOrdering:
    """A structural group order derived without benchmark reference metadata."""

    groups: tuple[tuple[int, ...], ...]
    group_order: tuple[int, ...]
    has_overlap: bool
    topology_sha256: str
    ordering_sha256: str

    @property
    def ordered_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self.groups[index] for index in self.group_order)


def build_reference_blind_ordering(
    groups: Sequence[Sequence[int]],
) -> ReferenceBlindOrdering:
    """Order disjoint groups lexicographically or traverse a simple overlap path.

    Any overlapping topology that is disconnected, cyclic, or branched fails
    closed because it has no unique path traversal under this v1 protocol.
    """

    canonical = _canonical_groups(groups)
    group_count = len(canonical)
    adjacency = [set() for _ in canonical]
    edge_count = 0
    for left in range(group_count):
        left_variables = set(canonical[left])
        for right in range(left + 1, group_count):
            if left_variables.intersection(canonical[right]):
                adjacency[left].add(right)
                adjacency[right].add(left)
                edge_count += 1

    if edge_count == 0:
        order = tuple(sorted(range(group_count), key=lambda index: (canonical[index], index)))
        has_overlap = False
    else:
        degrees = tuple(len(neighbors) for neighbors in adjacency)
        endpoints = tuple(index for index, degree in enumerate(degrees) if degree == 1)
        if (
            edge_count != group_count - 1
            or len(endpoints) != 2
            or any(degree not in (1, 2) for degree in degrees)
        ):
            raise ValueError("overlap intersection graph must be one simple path")

        start = min(endpoints, key=lambda index: (canonical[index], index))
        traversal: list[int] = []
        previous: int | None = None
        current = start
        while True:
            traversal.append(current)
            forward = adjacency[current] - ({previous} if previous is not None else set())
            if not forward:
                break
            if len(forward) != 1:
                raise ValueError("overlap intersection graph must be one simple path")
            previous, current = current, next(iter(forward))
        if len(traversal) != group_count:
            raise ValueError("overlap intersection graph must be one simple path")
        order = tuple(traversal)
        has_overlap = True

    return ReferenceBlindOrdering(
        groups=canonical,
        group_order=order,
        has_overlap=has_overlap,
        topology_sha256=topology_sha256(canonical),
        ordering_sha256=ordering_sha256(canonical, order),
    )


@dataclass(frozen=True, order=True)
class RelationKey:
    owner_group_indices: tuple[int, int]
    shared_variable_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.owner_group_indices) != 2:
            raise ValueError("relation key requires exactly two direct owners")
        owners = tuple(
            _integer(owner, name="owner_group_indices")
            for owner in self.owner_group_indices
        )
        if owners != tuple(sorted(set(owners))):
            raise ValueError("relation owners must be sorted and distinct")
        if not self.shared_variable_indices:
            raise ValueError("relation key requires shared variables")
        shared = tuple(
            _integer(variable, name="shared_variable_indices")
            for variable in self.shared_variable_indices
        )
        if shared != tuple(sorted(set(shared))):
            raise ValueError("shared relation variables must be sorted and unique")


@dataclass(frozen=True)
class RelationCandidate:
    """One owner pair with point proposals and top-k distribution evidence."""

    key: RelationKey
    owner_proposals: tuple[tuple[float, ...], tuple[float, ...]]
    owner_reliabilities: tuple[float, float]
    proposal_disagreement: float
    owner_priority: float
    owner_population_centers: tuple[tuple[float, ...], tuple[float, ...]]
    owner_population_standard_deviations: tuple[
        tuple[float, ...], tuple[float, ...]
    ]
    owner_population_sizes: tuple[int, int]

    def __post_init__(self) -> None:
        owner_vectors = (
            self.owner_proposals,
            self.owner_population_centers,
            self.owner_population_standard_deviations,
        )
        if (
            any(len(vectors) != 2 for vectors in owner_vectors)
            or len(self.owner_reliabilities) != 2
            or len(self.owner_population_sizes) != 2
        ):
            raise ValueError("relation evidence must align with exactly two owners")
        proposals = tuple(
            tuple(_finite(value, name="owner_proposals") for value in vector)
            for vector in self.owner_proposals
        )
        shared_count = len(self.key.shared_variable_indices)
        if any(len(vector) != shared_count for vector in proposals):
            raise ValueError("owner proposal vectors must align with shared variables")
        for name, vectors in (
            ("owner_population_centers", self.owner_population_centers),
            (
                "owner_population_standard_deviations",
                self.owner_population_standard_deviations,
            ),
        ):
            converted = tuple(
                tuple(_finite(value, name=name) for value in vector)
                for vector in vectors
            )
            if any(len(vector) != shared_count for vector in converted):
                raise ValueError(f"{name} must align with shared variables")
        if any(
            value < 0.0
            for vector in self.owner_population_standard_deviations
            for value in vector
        ):
            raise ValueError("owner population standard deviations must be non-negative")
        for size in self.owner_population_sizes:
            _integer(size, name="owner_population_sizes", minimum=1)
        for value in self.owner_reliabilities:
            _unit_interval(value, name="owner_reliabilities")
        disagreement = _finite(
            self.proposal_disagreement,
            name="proposal_disagreement",
        )
        if disagreement < 0.0:
            raise ValueError("proposal_disagreement must be non-negative")
        _unit_interval(self.owner_priority, name="owner_priority")


def cohen_d_from_moments(
    left_centers: Sequence[float],
    right_centers: Sequence[float],
    left_standard_deviations: Sequence[float],
    right_standard_deviations: Sequence[float],
) -> float:
    """Return the mean per-variable absolute Cohen's d for two owners."""

    vectors = tuple(
        tuple(_finite(value, name=name) for value in values)
        for name, values in (
            ("left_centers", left_centers),
            ("right_centers", right_centers),
            ("left_standard_deviations", left_standard_deviations),
            ("right_standard_deviations", right_standard_deviations),
        )
    )
    if not vectors[0]:
        return 0.0
    if any(len(vector) != len(vectors[0]) for vector in vectors[1:]):
        raise ValueError("Cohen's d moment vectors must have equal non-zero length")
    if any(value < 0.0 for vector in vectors[2:] for value in vector):
        raise ValueError("standard deviations must be non-negative")

    left_mu, right_mu, left_std, right_std = vectors
    effects = []
    for mu_left, mu_right, std_left, std_right in zip(
        left_mu,
        right_mu,
        left_std,
        right_std,
        strict=True,
    ):
        pooled_variance = (std_left**2 + std_right**2) / 2.0
        effects.append(
            abs(mu_left - mu_right) / math.sqrt(pooled_variance)
            if pooled_variance > 0.0
            else 0.0
        )
    return math.fsum(effects) / len(effects)


def relation_cohen_d(relation: RelationCandidate) -> float:
    """Calculate Cohen's d from a relation candidate's top-k moments."""

    if not isinstance(relation, RelationCandidate):
        raise TypeError("relation must be a RelationCandidate")
    return cohen_d_from_moments(
        relation.owner_population_centers[0],
        relation.owner_population_centers[1],
        relation.owner_population_standard_deviations[0],
        relation.owner_population_standard_deviations[1],
    )


def build_relation_candidates(
    groups: Sequence[Sequence[int]],
    owner_proposals: Mapping[tuple[int, int], float],
    owner_population_samples: Mapping[tuple[int, int], Sequence[float]],
    group_priorities: Sequence[float],
    owner_reliabilities: Sequence[float],
    *,
    lower_bound: float,
    upper_bound: float,
) -> tuple[RelationCandidate, ...]:
    """Build group-pair candidates from variables with exactly two direct owners."""

    canonical = _canonical_groups(groups)
    lower = _finite(lower_bound, name="lower_bound")
    upper = _finite(upper_bound, name="upper_bound")
    if upper <= lower:
        raise ValueError("upper_bound must exceed lower_bound")
    domain_width = upper - lower
    priorities = tuple(
        _unit_interval(value, name="group_priorities") for value in group_priorities
    )
    reliabilities = tuple(
        _unit_interval(value, name="owner_reliabilities")
        for value in owner_reliabilities
    )
    if len(priorities) != len(canonical) or len(reliabilities) != len(canonical):
        raise ValueError("group priorities and reliabilities must align with groups")

    proposals: dict[tuple[int, int], float] = {}
    for raw_key, raw_value in owner_proposals.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise ValueError("owner proposal keys must be (group, variable) tuples")
        group = _integer(raw_key[0], name="proposal group")
        variable = _integer(raw_key[1], name="proposal variable")
        if group >= len(canonical) or variable not in canonical[group]:
            raise ValueError("owner proposal must belong to its structural group")
        proposals[(group, variable)] = _finite(
            raw_value,
            name="owner proposal value",
        )

    population_samples: dict[tuple[int, int], tuple[float, ...]] = {}
    for raw_key, raw_samples in owner_population_samples.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise ValueError("population sample keys must be (group, variable) tuples")
        group = _integer(raw_key[0], name="population sample group")
        variable = _integer(raw_key[1], name="population sample variable")
        if group >= len(canonical) or variable not in canonical[group]:
            raise ValueError("population samples must belong to their structural group")
        samples = tuple(
            _finite(value, name="owner population sample") for value in raw_samples
        )
        if not samples:
            raise ValueError("owner population samples must be non-empty")
        population_samples[(group, variable)] = samples

    owners_by_variable: dict[int, list[int]] = {}
    for group_index, group in enumerate(canonical):
        for variable in group:
            owners_by_variable.setdefault(variable, []).append(group_index)

    variables_by_owner_pair: dict[tuple[int, int], list[int]] = {}
    for variable, raw_owners in sorted(owners_by_variable.items()):
        if len(raw_owners) != 2:
            continue
        owners = (raw_owners[0], raw_owners[1])
        variables_by_owner_pair.setdefault(owners, []).append(variable)

    candidates: list[RelationCandidate] = []
    for owners, raw_variables in sorted(variables_by_owner_pair.items()):
        shared_variables = tuple(sorted(raw_variables))
        try:
            left_values = tuple(
                proposals[(owners[0], variable)] for variable in shared_variables
            )
            right_values = tuple(
                proposals[(owners[1], variable)] for variable in shared_variables
            )
            left_samples = tuple(
                population_samples[(owners[0], variable)]
                for variable in shared_variables
            )
            right_samples = tuple(
                population_samples[(owners[1], variable)]
                for variable in shared_variables
            )
        except KeyError as exc:
            raise ValueError(
                "missing proposal or population samples from a direct relation owner"
            ) from exc
        population_sizes = (len(left_samples[0]), len(right_samples[0]))
        if any(len(samples) != population_sizes[0] for samples in left_samples) or any(
            len(samples) != population_sizes[1] for samples in right_samples
        ):
            raise ValueError("owner population sample counts must align across variables")

        def moments(samples_by_variable: Sequence[Sequence[float]]) -> tuple[
            tuple[float, ...], tuple[float, ...]
        ]:
            centers = tuple(
                math.fsum(samples) / len(samples) for samples in samples_by_variable
            )
            standard_deviations = tuple(
                math.sqrt(
                    math.fsum((value - center) ** 2 for value in samples)
                    / len(samples)
                )
                for samples, center in zip(samples_by_variable, centers)
            )
            return centers, standard_deviations

        left_centers, left_standard_deviations = moments(left_samples)
        right_centers, right_standard_deviations = moments(right_samples)
        candidates.append(
            RelationCandidate(
                key=RelationKey(owners, shared_variables),
                owner_proposals=(left_values, right_values),
                owner_reliabilities=(
                    reliabilities[owners[0]],
                    reliabilities[owners[1]],
                ),
                proposal_disagreement=math.fsum(
                    math.hypot(left_center - right_center, left_std - right_std)
                    / domain_width
                    for left_center, right_center, left_std, right_std in zip(
                        left_centers,
                        right_centers,
                        left_standard_deviations,
                        right_standard_deviations,
                    )
                )
                / len(shared_variables),
                owner_priority=max(priorities[owners[0]], priorities[owners[1]]),
                owner_population_centers=(left_centers, right_centers),
                owner_population_standard_deviations=(
                    left_standard_deviations,
                    right_standard_deviations,
                ),
                owner_population_sizes=population_sizes,
            )
        )
    return tuple(candidates)


@dataclass(frozen=True)
class ScoredRelation:
    relation: RelationCandidate
    disagreement_rank: float
    priority_rank: float
    voi_score: float
    score_source: RelationKey

    def __post_init__(self) -> None:
        _unit_interval(self.disagreement_rank, name="disagreement_rank")
        _unit_interval(self.priority_rank, name="priority_rank")
        _unit_interval(self.voi_score, name="voi_score")


def _harmonic_mean(left: float, right: float) -> float:
    return 0.0 if left + right == 0.0 else 2.0 * left * right / (left + right)


def score_relations(
    candidates: Sequence[RelationCandidate],
) -> tuple[ScoredRelation, ...]:
    """Score D/P with within-checkpoint midranks and their harmonic mean."""

    converted = tuple(candidates)
    if not converted:
        return ()
    if not all(isinstance(candidate, RelationCandidate) for candidate in converted):
        raise TypeError("candidates must contain RelationCandidate values")
    keys = tuple(candidate.key for candidate in converted)
    if len(set(keys)) != len(keys):
        raise ValueError("relation candidates must have unique structural keys")

    ordered = tuple(sorted(converted, key=lambda candidate: candidate.key))
    disagreement_ranks = midrank_percentiles(
        [candidate.proposal_disagreement for candidate in ordered]
    )
    priority_ranks = midrank_percentiles(
        [candidate.owner_priority for candidate in ordered]
    )
    return tuple(
        ScoredRelation(
            relation=candidate,
            disagreement_rank=disagreement_ranks[index],
            priority_rank=priority_ranks[index],
            voi_score=_harmonic_mean(
                disagreement_ranks[index],
                priority_ranks[index],
            ),
            score_source=candidate.key,
        )
        for index, candidate in enumerate(ordered)
    )


@dataclass(frozen=True)
class RelationSelection:
    selected: tuple[ScoredRelation, ...]
    abstained: bool
    reason: str


def select_top_relations(
    scored_relations: Sequence[ScoredRelation],
    *,
    count: int | None = TOP_RELATION_COUNT,
) -> RelationSelection:
    """Select a unique top-k set, abstaining on an ambiguous cutoff.

    Pass count=None to select all eligible relations (no VoI cutoff).
    """

    requested = (
        len(scored_relations)
        if count is None
        else _integer(count, name="count", minimum=1)
    )
    scored = tuple(scored_relations)
    if len({item.relation.key for item in scored}) != len(scored):
        raise ValueError("scored relations must have unique structural keys")
    ranked = tuple(
        sorted(scored, key=lambda item: (-item.voi_score, item.relation.key))
    )
    if len(ranked) < requested:
        return RelationSelection((), True, "insufficient_eligible_relations")
    if len(ranked) > requested and math.isclose(
        ranked[requested - 1].voi_score,
        ranked[requested].voi_score,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        return RelationSelection((), True, "non_unique_top_relation_cutoff")
    return RelationSelection(ranked[:requested], False, "top_relation_set_selected")


def _shifted_scores(
    native: tuple[ScoredRelation, ...],
    shift: int,
) -> tuple[ScoredRelation, ...]:
    return tuple(
        replace(
            target,
            voi_score=native[(index + shift) % len(native)].voi_score,
            score_source=native[(index + shift) % len(native)].relation.key,
        )
        for index, target in enumerate(native)
    )


def _top_key_set(
    scored: Sequence[ScoredRelation],
    count: int,
) -> frozenset[RelationKey]:
    return frozenset(
        item.relation.key
        for item in sorted(
            scored,
            key=lambda item: (-item.voi_score, item.relation.key),
        )[:count]
    )


def shuffle_relation_scores(
    scored_relations: Sequence[ScoredRelation],
    *,
    seed: int,
    count: int = TOP_RELATION_COUNT,
) -> tuple[ScoredRelation, ...]:
    """Assign scores by a deterministic, no-fixed-point cyclic permutation."""

    base_seed = _integer(seed, name="seed")
    requested = _integer(count, name="count", minimum=1)
    native = tuple(sorted(scored_relations, key=lambda item: item.relation.key))
    if len(native) < 2:
        raise ValueError("score derangement requires at least two relations")
    if len({item.relation.key for item in native}) != len(native):
        raise ValueError("scored relations must have unique structural keys")
    if any(item.score_source != item.relation.key for item in native):
        raise ValueError("only native relation scores may be shuffled")

    descriptor = {
        "salt": SHUFFLE_SALT,
        "seed": base_seed,
        "relations": [
            (
                item.relation.key.owner_group_indices,
                item.relation.key.shared_variable_indices,
            )
            for item in native
        ],
    }
    initial_shift = int(_canonical_sha256(descriptor), 16) % (len(native) - 1) + 1
    shifts = tuple(
        ((initial_shift - 1 + offset) % (len(native) - 1)) + 1
        for offset in range(len(native) - 1)
    )
    native_top = _top_key_set(native, min(requested, len(native)))
    initial = _shifted_scores(native, shifts[0])
    for shift in shifts:
        shuffled = _shifted_scores(native, shift)
        if _top_key_set(shuffled, min(requested, len(native))) != native_top:
            return shuffled
    return initial


@dataclass(frozen=True)
class BridgeWeights:
    left_owner: float
    right_owner: float

    def __post_init__(self) -> None:
        left = _unit_interval(self.left_owner, name="left_owner weight")
        right = _unit_interval(self.right_owner, name="right_owner weight")
        if max(left, right) > 0.65 + 1e-15:
            raise ValueError("a bridge owner weight cannot exceed 0.65")
        if not math.isclose(left + right, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("bridge owner weights must sum to one")


def bridge_weights(left_reliability: float, right_reliability: float) -> BridgeWeights:
    """Weight owners in proportion to 1 + reliability, capped at 0.65."""

    left = _unit_interval(left_reliability, name="left_reliability")
    right = _unit_interval(right_reliability, name="right_reliability")
    raw_left = (1.0 + left) / (2.0 + left + right)
    capped_left = min(0.65, max(0.35, raw_left))
    return BridgeWeights(capped_left, 1.0 - capped_left)


@dataclass(frozen=True)
class FourPointProbe:
    relation: RelationKey
    weights: BridgeWeights
    x0: tuple[float, ...]
    x_left: tuple[float, ...]
    x_right: tuple[float, ...]
    x_bridge: tuple[float, ...]


def build_four_point_probe(
    anchor: Sequence[float],
    relation: RelationCandidate,
) -> FourPointProbe:
    """Construct x0/xL/xR/xB from one immutable anchor."""

    x0 = tuple(_finite(value, name="anchor value") for value in anchor)
    if not x0:
        raise ValueError("anchor must be non-empty")
    shared_variables = relation.key.shared_variable_indices
    if any(variable >= len(x0) for variable in shared_variables):
        raise ValueError("relation variable is outside the anchor")
    left_values, right_values = relation.owner_proposals
    weights = bridge_weights(*relation.owner_reliabilities)
    bridge_values = tuple(
        weights.left_owner * left + weights.right_owner * right
        for left, right in zip(left_values, right_values)
    )

    x_left = list(x0)
    x_right = list(x0)
    x_bridge = list(x0)
    for index, variable in enumerate(shared_variables):
        x_left[variable] = left_values[index]
        x_right[variable] = right_values[index]
        x_bridge[variable] = bridge_values[index]
    return FourPointProbe(
        relation=relation.key,
        weights=weights,
        x0=x0,
        x_left=tuple(x_left),
        x_right=tuple(x_right),
        x_bridge=tuple(x_bridge),
    )


@dataclass(frozen=True)
class ProbeUtilities:
    left_owner: float
    right_owner: float
    bridge: float

    def __post_init__(self) -> None:
        _finite(self.left_owner, name="left_owner utility")
        _finite(self.right_owner, name="right_owner utility")
        _finite(self.bridge, name="bridge utility")


def summarize_probe_utilities(
    *,
    anchor_fitness: float,
    left_fitness: float,
    right_fitness: float,
    bridge_fitness: float,
    epsilon: float = UTILITY_EPSILON,
) -> ProbeUtilities:
    """Summarize minimization utilities as log((f0 + eps) / (fa + eps))."""

    fitness_values = {
        "anchor_fitness": anchor_fitness,
        "left_fitness": left_fitness,
        "right_fitness": right_fitness,
        "bridge_fitness": bridge_fitness,
    }
    converted = {
        name: _finite(value, name=name) for name, value in fitness_values.items()
    }
    if any(value < 0.0 for value in converted.values()):
        raise ValueError("probe fitness values must be non-negative")
    eps = _finite(epsilon, name="epsilon")
    if eps <= 0.0:
        raise ValueError("epsilon must be strictly positive")
    f0 = converted["anchor_fitness"]

    def utility(value: float) -> float:
        return math.log((f0 + eps) / (value + eps))

    return ProbeUtilities(
        left_owner=utility(converted["left_fitness"]),
        right_owner=utility(converted["right_fitness"]),
        bridge=utility(converted["bridge_fitness"]),
    )


@dataclass(frozen=True)
class ShadowDecision:
    shadow_action: str
    winner: str
    utility: float
    reason: str
    runtime_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.shadow_action not in {"repair", "coordinate", "fallback"}:
            raise ValueError("unsupported shadow action")
        if self.winner not in {"left_owner", "right_owner", "bridge", "none"}:
            raise ValueError("unsupported shadow winner")
        _finite(self.utility, name="shadow utility")


def decide_shadow_action(utilities: ProbeUtilities) -> ShadowDecision:
    """Record an observer-only action; no decision can authorize runtime use."""

    values = (
        ("left_owner", utilities.left_owner),
        ("right_owner", utilities.right_owner),
        ("bridge", utilities.bridge),
    )
    best_utility = max(value for _, value in values)
    winners = tuple(name for name, value in values if value == best_utility)
    if len(winners) != 1:
        return ShadowDecision(
            shadow_action="fallback",
            winner="none",
            utility=best_utility,
            reason="non_unique_best_probe_utility",
        )
    winner = winners[0]
    if best_utility < SHADOW_GAIN_THRESHOLD:
        return ShadowDecision(
            shadow_action="fallback",
            winner="none",
            utility=best_utility,
            reason="probe_gain_below_one_percent",
        )
    return ShadowDecision(
        shadow_action="coordinate" if winner == "bridge" else "repair",
        winner=winner,
        utility=best_utility,
        reason="unique_probe_winner_above_one_percent",
    )


def assert_runtime_schema_is_reference_blind() -> None:
    """Guard the pure runtime records against AOB identity/outcome leakage."""

    record_types = (
        ReferenceBlindOrdering,
        RelationKey,
        RelationCandidate,
        ScoredRelation,
        RelationSelection,
        BridgeWeights,
        FourPointProbe,
        ProbeUtilities,
        ShadowDecision,
    )
    names = {item.name.lower() for record in record_types for item in fields(record)}
    leaked = sorted(
        name
        for name in names
        if any(fragment in name for fragment in FORBIDDEN_RUNTIME_FIELD_FRAGMENTS)
    )
    if leaked:
        raise RuntimeError(f"forbidden evidence overlay fields: {','.join(leaked)}")


assert_runtime_schema_is_reference_blind()
