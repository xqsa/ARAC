"""Pure overlap-hypergraph evidence and delayed-credit contracts.

Raw decomposition groups remain hyperedges.  In particular, this module never
forms connected components: a focal scope contains only the groups that directly
own one of the focal hyperedge's shared variables.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


HYPERGRAPH_SCHEMA_VERSION = "hypergraph-delayed-credit-v1"
ERROR_FLOOR = 1e-300
HISTORY_SWEEPS = 3
EWMA_ALPHA = 0.5
MAX_OWNER_WEIGHT = 0.65
DELAYED_OVERWRITE_PENALTY = math.log(1.01)
FINAL_OWNER_PROPOSAL_WATERMARK = (
    "after_group_local_rescue_recovery_before_relation_writeback"
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


def _finite_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    vector = tuple(_finite(value, name=name) for value in values)
    if not vector:
        raise ValueError(f"{name} must be non-empty")
    return vector


def _value_pairs(
    values: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    name: str,
) -> tuple[tuple[int, float], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    converted: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_index, raw_value in items:
        index = _integer(raw_index, name=f"{name} variable")
        if index in seen:
            raise ValueError(f"{name} contains duplicate variable {index}")
        seen.add(index)
        converted.append((index, _finite(raw_value, name=f"{name}[{index}]")))
    return tuple(sorted(converted))


@dataclass(frozen=True)
class SharedVariableStar:
    """The direct group owners of one variable."""

    variable_index: int
    owner_group_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.variable_index, name="variable_index")
        if not isinstance(self.owner_group_indices, tuple) or not self.owner_group_indices:
            raise ValueError("owner_group_indices must be a non-empty tuple")
        owners = tuple(
            _integer(value, name="owner_group_indices")
            for value in self.owner_group_indices
        )
        if owners != tuple(sorted(set(owners))):
            raise ValueError("owner_group_indices must be sorted and unique")

    @property
    def shared(self) -> bool:
        return len(self.owner_group_indices) >= 2


@dataclass(frozen=True)
class HyperedgeFocalScope:
    """A focal hyperedge and only its one-hop direct owners."""

    focal_group_index: int
    shared_variables: tuple[int, ...]
    direct_owner_group_indices: tuple[int, ...]
    neighbor_group_indices: tuple[int, ...]


@dataclass(frozen=True)
class OverlapHypergraphTopology:
    """Immutable raw hyperedges and variable-owner stars."""

    hyperedges: tuple[tuple[int, ...], ...]
    stars: tuple[SharedVariableStar, ...]
    group_shared_variables: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hyperedges, tuple) or not self.hyperedges:
            raise ValueError("hyperedges must be a non-empty tuple")
        for group in self.hyperedges:
            if not isinstance(group, tuple) or not group:
                raise ValueError("every hyperedge must be a non-empty tuple")
            if group != tuple(sorted(set(group))) or any(
                isinstance(value, bool) or int(value) != value or value < 0
                for value in group
            ):
                raise ValueError("hyperedge variables must be sorted unique integers")
        if len(self.group_shared_variables) != len(self.hyperedges):
            raise ValueError("group_shared_variables must align with hyperedges")
        star_variables = tuple(star.variable_index for star in self.stars)
        if star_variables != tuple(sorted(set(star_variables))):
            raise ValueError("stars must be sorted by unique variable index")

    def star_for_variable(self, variable_index: int) -> SharedVariableStar:
        variable = _integer(variable_index, name="variable_index")
        for star in self.stars:
            if star.variable_index == variable:
                return star
        raise KeyError(variable)

    def shared_for_group(self, group_index: int) -> tuple[int, ...]:
        group = _integer(group_index, name="group_index")
        try:
            return self.group_shared_variables[group]
        except IndexError as exc:
            raise IndexError("group_index is outside the hypergraph") from exc

    def focal_scope(self, group_index: int) -> HyperedgeFocalScope:
        group = _integer(group_index, name="group_index")
        if group >= len(self.hyperedges):
            raise IndexError("group_index is outside the hypergraph")
        shared = self.group_shared_variables[group]
        direct_owners = tuple(
            sorted(
                {
                    owner
                    for variable in shared
                    for owner in self.star_for_variable(variable).owner_group_indices
                }
            )
        )
        neighbors = tuple(owner for owner in direct_owners if owner != group)
        return HyperedgeFocalScope(
            focal_group_index=group,
            shared_variables=shared,
            direct_owner_group_indices=direct_owners,
            neighbor_group_indices=neighbors,
        )

    @property
    def eligible_group_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, shared in enumerate(self.group_shared_variables)
            if shared
        )

    @property
    def variable_owner_groups(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Typed immutable equivalent of the legacy ``var_to_groups`` map."""

        return tuple(
            (star.variable_index, star.owner_group_indices) for star in self.stars
        )

    @property
    def overlap_variables(self) -> tuple[int, ...]:
        return tuple(star.variable_index for star in self.stars if star.shared)

    @property
    def membership_histogram(self) -> tuple[tuple[int, int], ...]:
        counts: dict[int, int] = {}
        for star in self.stars:
            membership = len(star.owner_group_indices)
            counts[membership] = counts.get(membership, 0) + 1
        return tuple(sorted(counts.items()))


def build_overlap_hypergraph(
    grouping_result: Sequence[Sequence[int]],
) -> OverlapHypergraphTopology:
    """Build direct ownership without taking a transitive closure."""

    hyperedges: list[tuple[int, ...]] = []
    owners: dict[int, list[int]] = {}
    for group_index, raw_group in enumerate(grouping_result):
        converted = tuple(
            _integer(value, name=f"grouping_result[{group_index}]")
            for value in raw_group
        )
        if not converted:
            raise ValueError("grouping_result must contain only non-empty groups")
        if len(set(converted)) != len(converted):
            raise ValueError("a hyperedge cannot contain duplicate variables")
        hyperedge = tuple(sorted(converted))
        hyperedges.append(hyperedge)
        for variable in hyperedge:
            owners.setdefault(variable, []).append(group_index)
    if not hyperedges:
        raise ValueError("grouping_result must contain at least one group")

    stars = tuple(
        SharedVariableStar(variable, tuple(group_owners))
        for variable, group_owners in sorted(owners.items())
    )
    shared_variables = {
        star.variable_index for star in stars if star.shared
    }
    group_shared_variables = tuple(
        tuple(variable for variable in group if variable in shared_variables)
        for group in hyperedges
    )
    return OverlapHypergraphTopology(
        hyperedges=tuple(hyperedges),
        stars=stars,
        group_shared_variables=group_shared_variables,
    )


@dataclass(frozen=True)
class SharedProposal:
    """Shared-coordinate proposal extracted from one native v37 group visit."""

    group_index: int
    anchor_values: tuple[tuple[int, float], ...]
    proposed_values: tuple[tuple[int, float], ...]
    capture_watermark: str = FINAL_OWNER_PROPOSAL_WATERMARK

    def __post_init__(self) -> None:
        _integer(self.group_index, name="group_index")
        anchors = _value_pairs(self.anchor_values, name="anchor_values")
        proposals = _value_pairs(self.proposed_values, name="proposed_values")
        if anchors != self.anchor_values or proposals != self.proposed_values:
            raise ValueError("shared proposal values must be sorted and canonical")
        if tuple(index for index, _ in anchors) != tuple(
            index for index, _ in proposals
        ):
            raise ValueError("anchor and proposed shared variables must align")
        if self.capture_watermark != FINAL_OWNER_PROPOSAL_WATERMARK:
            raise ValueError("shared proposal must use the final owner watermark")

    @property
    def variables(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self.proposed_values)

    def proposed_value(self, variable_index: int) -> float:
        variable = _integer(variable_index, name="variable_index")
        for index, value in self.proposed_values:
            if index == variable:
                return value
        raise KeyError(variable)


@dataclass(frozen=True)
class GroupCycleObservation:
    """A trace-only native group observation with raw errors discarded."""

    sweep_index: int
    group_index: int
    primary_requested_fe: int
    primary_actual_fe: int
    full_interval_actual_fe: int
    full_interval_start_fe: int
    full_interval_end_fe: int
    unit_fe_contribution: float
    successful: bool
    shared_proposal: SharedProposal

    def __post_init__(self) -> None:
        _integer(self.sweep_index, name="sweep_index")
        _integer(self.group_index, name="group_index")
        requested = _integer(
            self.primary_requested_fe,
            name="primary_requested_fe",
            minimum=1,
        )
        primary_actual = _integer(
            self.primary_actual_fe,
            name="primary_actual_fe",
        )
        full_actual = _integer(
            self.full_interval_actual_fe,
            name="full_interval_actual_fe",
            minimum=1,
        )
        start = _integer(self.full_interval_start_fe, name="full_interval_start_fe")
        end = _integer(self.full_interval_end_fe, name="full_interval_end_fe")
        contribution = _finite(
            self.unit_fe_contribution,
            name="unit_fe_contribution",
        )
        if primary_actual > requested:
            raise ValueError("primary_actual_fe must not exceed primary_requested_fe")
        if full_actual < primary_actual:
            raise ValueError("full_interval_actual_fe must cover primary_actual_fe")
        if end - start != full_actual:
            raise ValueError(
                "full_interval_end_fe - full_interval_start_fe must equal "
                "full_interval_actual_fe"
            )
        if contribution < 0.0:
            raise ValueError("unit_fe_contribution must be non-negative")
        if not isinstance(self.successful, bool):
            raise ValueError("successful must be boolean")
        if self.shared_proposal.group_index != self.group_index:
            raise ValueError("shared proposal group must match observation group")


def unit_fe_contribution(
    *,
    pre_error: float,
    best_error: float,
    actual_fe: int,
) -> float:
    """Return the preregistered improvement per 1,000 objective evaluations."""

    pre = _finite(pre_error, name="pre_error")
    best = _finite(best_error, name="best_error")
    evaluations = _integer(actual_fe, name="actual_fe", minimum=1)
    if pre < 0.0 or best < 0.0:
        raise ValueError("error values must be non-negative")
    endpoint = min(pre, best)
    gain = math.log(pre + ERROR_FLOOR) - math.log(endpoint + ERROR_FLOOR)
    return float(1000.0 * gain / evaluations)


def build_group_cycle_observation(
    topology: OverlapHypergraphTopology,
    *,
    sweep_index: int,
    group_index: int,
    pre_error: float,
    best_error: float,
    primary_requested_fe: int,
    primary_actual_fe: int,
    full_interval_actual_fe: int,
    full_interval_start_fe: int,
    full_interval_end_fe: int,
    pre_block_candidate: Sequence[float],
    final_owner_candidate: Sequence[float],
) -> GroupCycleObservation:
    """Extract state after local recovery and before relation writeback.

    Unit-FE contribution uses the complete native interval, including existing
    precheck, rescue, and recovery evaluations.  The primary CMA request and
    actual use are retained only as separate audit facts.
    """

    if not isinstance(topology, OverlapHypergraphTopology):
        raise TypeError("topology must be OverlapHypergraphTopology")
    group = _integer(group_index, name="group_index")
    if group >= len(topology.hyperedges):
        raise IndexError("group_index is outside the hypergraph")
    pre = _finite(pre_error, name="pre_error")
    best = _finite(best_error, name="best_error")
    if pre < 0.0 or best < 0.0:
        raise ValueError("error values must be non-negative")
    before = _finite_vector(pre_block_candidate, name="pre_block_candidate")
    proposal = _finite_vector(final_owner_candidate, name="final_owner_candidate")
    if len(before) != len(proposal):
        raise ValueError("pre-block and final owner candidates must align")
    shared = topology.shared_for_group(group)
    if shared and max(shared) >= len(before):
        raise ValueError("candidate width does not cover shared variables")
    anchor_values = tuple((variable, before[variable]) for variable in shared)
    proposed_values = tuple((variable, proposal[variable]) for variable in shared)
    return GroupCycleObservation(
        sweep_index=_integer(sweep_index, name="sweep_index"),
        group_index=group,
        primary_requested_fe=_integer(
            primary_requested_fe,
            name="primary_requested_fe",
            minimum=1,
        ),
        primary_actual_fe=_integer(
            primary_actual_fe,
            name="primary_actual_fe",
        ),
        full_interval_actual_fe=_integer(
            full_interval_actual_fe,
            name="full_interval_actual_fe",
            minimum=1,
        ),
        full_interval_start_fe=_integer(
            full_interval_start_fe,
            name="full_interval_start_fe",
        ),
        full_interval_end_fe=_integer(
            full_interval_end_fe,
            name="full_interval_end_fe",
        ),
        unit_fe_contribution=unit_fe_contribution(
            pre_error=pre,
            best_error=best,
            actual_fe=full_interval_actual_fe,
        ),
        successful=best < pre,
        shared_proposal=SharedProposal(
            group_index=group,
            anchor_values=anchor_values,
            proposed_values=proposed_values,
            capture_watermark=FINAL_OWNER_PROPOSAL_WATERMARK,
        ),
    )


def midrank_percentiles(values: Sequence[float]) -> tuple[float, ...]:
    """Return ascending midrank percentiles; a singleton receives 0.5."""

    converted = tuple(_finite(value, name="rank value") for value in values)
    if not converted:
        raise ValueError("midrank_percentiles requires at least one value")
    positions: dict[float, list[int]] = {}
    for position, (_, original_index) in enumerate(
        sorted((value, index) for index, value in enumerate(converted)),
        start=1,
    ):
        positions.setdefault(converted[original_index], []).append(position)
    n = len(converted)
    rank_by_value = {
        value: (math.fsum(value_positions) / len(value_positions) - 0.5) / n
        for value, value_positions in positions.items()
    }
    return tuple(rank_by_value[value] for value in converted)


def _domain_width(lower_bound: float, upper_bound: float) -> float:
    lower = _finite(lower_bound, name="lower_bound")
    upper = _finite(upper_bound, name="upper_bound")
    if upper <= lower:
        raise ValueError("upper_bound must exceed lower_bound")
    return upper - lower


def normalized_direct_owner_proposal_disagreement(
    topology: OverlapHypergraphTopology,
    *,
    focal_group_index: int,
    current_observations: Mapping[int, GroupCycleObservation],
    lower_bound: float,
    upper_bound: float,
) -> float:
    """Average normalized owner range over the focal shared variables."""

    width = _domain_width(lower_bound, upper_bound)
    scope = topology.focal_scope(focal_group_index)
    if not scope.shared_variables:
        return 0.0
    ranges: list[float] = []
    for variable in scope.shared_variables:
        owners = topology.star_for_variable(variable).owner_group_indices
        values: list[float] = []
        for owner in owners:
            try:
                observation = current_observations[owner]
            except KeyError as exc:
                raise ValueError("missing direct-owner observation") from exc
            values.append(observation.shared_proposal.proposed_value(variable))
        ranges.append(min(1.0, max(0.0, (max(values) - min(values)) / width)))
    return float(math.fsum(ranges) / len(ranges))


@dataclass(frozen=True)
class HyperedgeCycleState:
    """Three-sweep pre-action state for one eligible raw hyperedge."""

    current_unit_fe_contribution: float
    ewma_unit_fe_contribution_3: float
    success_ratio_3: float
    zero_gain_difficulty: float
    stagnation_ratio_3: float
    direct_owner_proposal_disagreement: float
    prior_next_sweep_survival: float
    prior_next_sweep_overwrite: float

    def __post_init__(self) -> None:
        for name in (
            "current_unit_fe_contribution",
            "ewma_unit_fe_contribution_3",
        ):
            if _finite(getattr(self, name), name=name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "success_ratio_3",
            "zero_gain_difficulty",
            "stagnation_ratio_3",
            "direct_owner_proposal_disagreement",
            "prior_next_sweep_survival",
            "prior_next_sweep_overwrite",
        ):
            _unit_interval(getattr(self, name), name=name)


@dataclass(frozen=True)
class HyperedgeScore:
    """Within-sweep fixed ranks and scores for one eligible hyperedge."""

    contribution_score: float
    need_score: float
    focal_priority: float
    owner_reliability: float

    def __post_init__(self) -> None:
        for name in (
            "contribution_score",
            "need_score",
            "focal_priority",
            "owner_reliability",
        ):
            _unit_interval(getattr(self, name), name=name)


def build_hyperedge_cycle_states(
    topology: OverlapHypergraphTopology,
    observations: Sequence[GroupCycleObservation],
    *,
    prior_next_sweep_survival_by_group: Mapping[int, float],
    prior_next_sweep_overwrite_by_group: Mapping[int, float],
    lower_bound: float,
    upper_bound: float,
) -> tuple[HyperedgeCycleState, ...]:
    """Build complete three-sweep states, failing closed on missing history."""

    if not isinstance(topology, OverlapHypergraphTopology):
        raise TypeError("topology must be OverlapHypergraphTopology")
    eligible = topology.eligible_group_indices
    if not eligible:
        return ()
    by_key: dict[tuple[int, int], GroupCycleObservation] = {}
    for observation in observations:
        if not isinstance(observation, GroupCycleObservation):
            raise TypeError("observations must contain GroupCycleObservation values")
        key = (observation.sweep_index, observation.group_index)
        if key in by_key:
            raise ValueError("duplicate group observation in a sweep")
        by_key[key] = observation
    sweep_sets = {
        group: tuple(
            sorted(sweep for sweep, observed_group in by_key if observed_group == group)
        )
        for group in eligible
    }
    reference_sweeps = sweep_sets[eligible[0]]
    if len(reference_sweeps) != HISTORY_SWEEPS or reference_sweeps != tuple(
        range(reference_sweeps[-1] - HISTORY_SWEEPS + 1, reference_sweeps[-1] + 1)
    ):
        raise ValueError("eligible hyperedges require three consecutive complete sweeps")
    if any(sweeps != reference_sweeps for sweeps in sweep_sets.values()):
        raise ValueError("eligible hyperedge histories must cover the same sweeps")
    current_by_group = {
        group: by_key[(reference_sweeps[-1], group)] for group in eligible
    }

    states: list[HyperedgeCycleState] = []
    for group in eligible:
        history = tuple(by_key[(sweep, group)] for sweep in reference_sweeps)
        ewma = history[0].unit_fe_contribution
        for observation in history[1:]:
            ewma = EWMA_ALPHA * observation.unit_fe_contribution + (1.0 - EWMA_ALPHA) * ewma
        successes = math.fsum(float(observation.successful) for observation in history)
        success_ratio = successes / HISTORY_SWEEPS
        trailing_zero = 0
        for observation in reversed(history):
            if observation.unit_fe_contribution > 0.0:
                break
            trailing_zero += 1
        try:
            prior_survival = prior_next_sweep_survival_by_group[group]
            prior_overwrite = prior_next_sweep_overwrite_by_group[group]
        except KeyError as exc:
            raise ValueError("missing closed prior survival/overwrite") from exc
        if not math.isclose(
            float(prior_survival) + float(prior_overwrite),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("prior survival and overwrite must sum to one")
        states.append(
            HyperedgeCycleState(
                current_unit_fe_contribution=history[-1].unit_fe_contribution,
                ewma_unit_fe_contribution_3=ewma,
                success_ratio_3=success_ratio,
                zero_gain_difficulty=1.0 - success_ratio,
                stagnation_ratio_3=(
                    min(trailing_zero, HISTORY_SWEEPS) / HISTORY_SWEEPS
                ),
                direct_owner_proposal_disagreement=(
                    normalized_direct_owner_proposal_disagreement(
                    topology,
                    focal_group_index=group,
                    current_observations=current_by_group,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    )
                ),
                prior_next_sweep_survival=prior_survival,
                prior_next_sweep_overwrite=prior_overwrite,
            )
        )
    return tuple(states)


def score_hyperedge_states(
    states: Sequence[HyperedgeCycleState],
) -> tuple[HyperedgeScore, ...]:
    """Apply the fixed within-sweep midrank score without identity tie-breaks."""

    converted = tuple(states)
    if not converted:
        return ()
    if not all(isinstance(state, HyperedgeCycleState) for state in converted):
        raise TypeError("states must contain HyperedgeCycleState values")

    current_rank = midrank_percentiles(
        [state.current_unit_fe_contribution for state in converted]
    )
    ewma_rank = midrank_percentiles(
        [state.ewma_unit_fe_contribution_3 for state in converted]
    )
    difficulty_rank = midrank_percentiles(
        [state.zero_gain_difficulty for state in converted]
    )
    stagnation_rank = midrank_percentiles(
        [state.stagnation_ratio_3 for state in converted]
    )
    disagreement_rank = midrank_percentiles(
        [state.direct_owner_proposal_disagreement for state in converted]
    )
    survival_rank = midrank_percentiles(
        [state.prior_next_sweep_survival for state in converted]
    )
    non_overwrite_rank = midrank_percentiles(
        [1.0 - state.prior_next_sweep_overwrite for state in converted]
    )

    scored: list[HyperedgeScore] = []
    for index, state in enumerate(converted):
        contribution = (current_rank[index] + ewma_rank[index]) / 2.0
        need = (
            difficulty_rank[index]
            + stagnation_rank[index]
            + disagreement_rank[index]
        ) / 3.0
        priority = (
            0.0
            if contribution + need == 0.0
            else 2.0 * contribution * need / (contribution + need)
        )
        reliability = (
            current_rank[index]
            + ewma_rank[index]
            + survival_rank[index]
            + non_overwrite_rank[index]
        ) / 4.0
        scored.append(
            HyperedgeScore(
                contribution_score=contribution,
                need_score=need,
                focal_priority=priority,
                owner_reliability=reliability,
            )
        )
    return tuple(scored)


def project_owner_weights(
    reliability: Sequence[float],
    *,
    maximum_weight: float = MAX_OWNER_WEIGHT,
) -> tuple[float, ...]:
    """Euclidean projection of proportional owner weights onto a capped simplex."""

    values = tuple(_unit_interval(value, name="owner reliability") for value in reliability)
    if not values:
        raise ValueError("at least one owner reliability is required")
    cap = _unit_interval(maximum_weight, name="maximum_weight")
    if cap <= 0.0 or len(values) * cap < 1.0:
        raise ValueError("owner weight cap makes the simplex infeasible")
    denominator = math.fsum(1.0 + item for item in values)
    raw = tuple((1.0 + value) / denominator for value in values)
    active = set(range(len(raw)))
    fixed: dict[int, float] = {}
    while active:
        shift = (
            math.fsum(raw[index] for index in active)
            + math.fsum(fixed.values())
            - 1.0
        ) / len(active)
        high = {index for index in active if raw[index] - shift > cap}
        low = {index for index in active if raw[index] - shift < 0.0}
        if not high and not low:
            for index in active:
                fixed[index] = raw[index] - shift
            active.clear()
            break
        for index in high:
            fixed[index] = cap
        for index in low:
            fixed[index] = 0.0
        active.difference_update(high | low)
    projected = tuple(fixed[index] for index in range(len(raw)))
    if not math.isclose(math.fsum(projected), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("capped simplex projection did not conserve weight")
    return projected


@dataclass(frozen=True)
class SweepCoordinationPlan:
    """Pure one-hop candidate plan; it cannot evaluate or commit the candidate."""

    selected: bool
    reason: str
    focal_group_index: int | None
    focal_priority: float | None
    shared_variables: tuple[int, ...]
    direct_owner_group_indices: tuple[int, ...]
    owner_weights: tuple[tuple[int, int, float], ...]
    structural_risk: float | None
    step_scale: float | None
    proposal_range_norm: float | None
    target_displacement_norm: float | None
    candidate: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("coordination plan reason is required")
        if self.focal_group_index is not None:
            _integer(self.focal_group_index, name="focal_group_index")
        for name in ("focal_priority", "structural_risk", "step_scale"):
            value = getattr(self, name)
            if value is not None:
                _unit_interval(value, name=name)
        for name in ("proposal_range_norm", "target_displacement_norm"):
            value = getattr(self, name)
            if value is not None and _finite(value, name=name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.selected and (
            self.focal_group_index is None
            or not self.shared_variables
            or not self.direct_owner_group_indices
            or not self.candidate
        ):
            raise ValueError("selected coordination plan is incomplete")


def _abstain_plan(
    reason: str,
    *,
    focal_group_index: int | None = None,
    focal_priority: float | None = None,
    shared_variables: tuple[int, ...] = (),
    direct_owner_group_indices: tuple[int, ...] = (),
    owner_weights: tuple[tuple[int, int, float], ...] = (),
    structural_risk: float | None = None,
    step_scale: float | None = None,
    proposal_range_norm: float | None = None,
    target_displacement_norm: float | None = None,
) -> SweepCoordinationPlan:
    return SweepCoordinationPlan(
        selected=False,
        reason=reason,
        focal_group_index=focal_group_index,
        focal_priority=focal_priority,
        shared_variables=shared_variables,
        direct_owner_group_indices=direct_owner_group_indices,
        owner_weights=owner_weights,
        structural_risk=structural_risk,
        step_scale=step_scale,
        proposal_range_norm=proposal_range_norm,
        target_displacement_norm=target_displacement_norm,
        candidate=(),
    )


def plan_sweep_coordination(
    topology: OverlapHypergraphTopology,
    *,
    scores: Sequence[HyperedgeScore],
    current_observations: Mapping[int, GroupCycleObservation],
    sweep_end_anchor: Sequence[float],
    lower_bound: float,
    upper_bound: float,
) -> SweepCoordinationPlan:
    """Select the unique focal hyperedge and build the fixed one-hop candidate."""

    width = _domain_width(lower_bound, upper_bound)
    if not isinstance(topology, OverlapHypergraphTopology):
        raise TypeError("topology must be OverlapHypergraphTopology")
    scored = tuple(scores)
    if not scored:
        return _abstain_plan("abstain_no_shared_hyperedge")
    if not all(isinstance(score, HyperedgeScore) for score in scored):
        raise TypeError("scores must contain HyperedgeScore values")
    eligible = topology.eligible_group_indices
    if len(scored) != len(eligible):
        raise ValueError("scores must cover every eligible raw hyperedge")
    score_by_group = dict(zip(eligible, scored, strict=True))
    try:
        eligible_observations = tuple(current_observations[group] for group in eligible)
    except KeyError as exc:
        raise ValueError("missing eligible current-sweep observation") from exc
    if any(
        observation.group_index != group
        for group, observation in zip(eligible, eligible_observations, strict=True)
    ):
        raise ValueError("current observation route does not match its group")
    proposal_sweeps = {
        observation.sweep_index for observation in eligible_observations
    }
    if len(proposal_sweeps) != 1:
        raise ValueError("candidate proposals must come from one complete sweep")
    highest = max(score.focal_priority for score in scored)
    winners = tuple(
        (group, score)
        for group, score in zip(eligible, scored, strict=True)
        if score.focal_priority == highest
    )
    if len(winners) != 1:
        return _abstain_plan("abstain_focal_priority_tie")

    focal_group, focal = winners[0]
    scope = topology.focal_scope(focal_group)
    anchor = _finite_vector(sweep_end_anchor, name="sweep_end_anchor")
    if max(topology.stars, key=lambda star: star.variable_index).variable_index >= len(anchor):
        raise ValueError("sweep_end_anchor does not cover hypergraph variables")
    risk = max(
        len(scope.shared_variables) / len(topology.hyperedges[focal_group]),
        len(scope.neighbor_group_indices) / (len(scope.neighbor_group_indices) + 1),
    )

    owner_weight_rows: list[tuple[int, int, float]] = []
    targets: list[float] = []
    normalized_ranges: list[float] = []
    normalized_displacements: list[float] = []
    for variable in scope.shared_variables:
        owners = topology.star_for_variable(variable).owner_group_indices
        reliability = tuple(score_by_group[owner].owner_reliability for owner in owners)
        weights = project_owner_weights(reliability)
        try:
            proposals = tuple(
                current_observations[owner].shared_proposal.proposed_value(variable)
                for owner in owners
            )
        except KeyError as exc:
            raise ValueError("missing direct-owner proposal for candidate") from exc
        target = math.fsum(
            weight * proposal
            for weight, proposal in zip(weights, proposals, strict=True)
        )
        targets.append(target)
        normalized_ranges.append((max(proposals) - min(proposals)) / width)
        normalized_displacements.append((target - anchor[variable]) / width)
        owner_weight_rows.extend(
            (variable, owner, weight)
            for owner, weight in zip(owners, weights, strict=True)
        )

    proposal_range = math.sqrt(math.fsum(value * value for value in normalized_ranges))
    target_displacement = math.sqrt(
        math.fsum(value * value for value in normalized_displacements)
    )
    step_scale = min(
        1.0,
        (1.0 - risk) * proposal_range / (target_displacement + ERROR_FLOOR),
    )
    candidate = list(anchor)
    for variable, target in zip(scope.shared_variables, targets, strict=True):
        candidate[variable] = anchor[variable] + step_scale * (target - anchor[variable])
    candidate_tuple = tuple(candidate)
    common = {
        "focal_group_index": focal_group,
        "focal_priority": focal.focal_priority,
        "shared_variables": scope.shared_variables,
        "direct_owner_group_indices": scope.direct_owner_group_indices,
        "owner_weights": tuple(owner_weight_rows),
        "structural_risk": risk,
        "step_scale": step_scale,
        "proposal_range_norm": proposal_range,
        "target_displacement_norm": target_displacement,
    }
    if not all(math.isfinite(value) for value in candidate_tuple):
        return _abstain_plan("abstain_candidate_nonfinite", **common)
    lower = float(lower_bound)
    upper = float(upper_bound)
    if any(value < lower or value > upper for value in candidate_tuple):
        return _abstain_plan("abstain_candidate_out_of_domain", **common)
    if not any(
        candidate_tuple[variable] != anchor[variable]
        for variable in scope.shared_variables
    ):
        return _abstain_plan("abstain_zero_coordination_displacement", **common)
    return SweepCoordinationPlan(
        selected=True,
        reason="coordination_candidate_ready",
        candidate=candidate_tuple,
        **common,
    )


@dataclass(frozen=True)
class DirectionalSurvival:
    """Displacement-weighted survival and overwrite of a shared proposal."""

    survival: float
    overwrite: float
    changed_variable_count: int

    def __post_init__(self) -> None:
        _unit_interval(self.survival, name="survival")
        _unit_interval(self.overwrite, name="overwrite")
        _integer(self.changed_variable_count, name="changed_variable_count")
        if not math.isclose(
            self.survival + self.overwrite,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("survival and overwrite must sum to one")


def directional_survival(
    *,
    anchor_values: Sequence[float],
    candidate_values: Sequence[float],
    next_sweep_values: Sequence[float],
) -> DirectionalSurvival:
    """Project next-sweep state onto the candidate displacement per variable."""

    anchor = _finite_vector(anchor_values, name="anchor_values")
    candidate = _finite_vector(candidate_values, name="candidate_values")
    next_values = _finite_vector(next_sweep_values, name="next_sweep_values")
    if len(anchor) != len(candidate) or len(anchor) != len(next_values):
        raise ValueError("directional survival vectors must align")
    weighted_projection = 0.0
    total_displacement = 0.0
    changed = 0
    for before, committed, after in zip(anchor, candidate, next_values, strict=True):
        displacement = committed - before
        weight = abs(displacement)
        if weight == 0.0:
            continue
        changed += 1
        projection = max(0.0, min(1.0, (after - before) / displacement))
        weighted_projection += weight * projection
        total_displacement += weight
    if changed == 0 or total_displacement == 0.0:
        return DirectionalSurvival(
            survival=0.5,
            overwrite=0.5,
            changed_variable_count=0,
        )
    survival = weighted_projection / total_displacement
    return DirectionalSurvival(
        survival=survival,
        overwrite=1.0 - survival,
        changed_variable_count=changed,
    )


@dataclass(frozen=True)
class DelayedHyperedgeCredit:
    """Outcome label closed only at the next complete native sweep end."""

    action_sweep_index: int
    resolution_sweep_index: int
    survival: float
    overwrite: float
    next_sweep_log_improvement: float
    penalized_credit: float

    def __post_init__(self) -> None:
        action = _integer(self.action_sweep_index, name="action_sweep_index")
        resolution = _integer(self.resolution_sweep_index, name="resolution_sweep_index")
        if resolution != action + 1:
            raise ValueError("delayed credit must close at the next complete sweep")
        _unit_interval(self.survival, name="survival")
        _unit_interval(self.overwrite, name="overwrite")
        _finite(self.next_sweep_log_improvement, name="next_sweep_log_improvement")
        _finite(self.penalized_credit, name="penalized_credit")


def build_delayed_hyperedge_credit(
    *,
    action_sweep_index: int,
    resolution_sweep_index: int,
    all_groups_completed: bool,
    native_sweep_end_completed: bool,
    anchor_error: float,
    next_sweep_error: float,
    anchor_shared_values: Sequence[float],
    candidate_shared_values: Sequence[float],
    next_sweep_shared_values: Sequence[float],
) -> DelayedHyperedgeCredit:
    """Close overwrite-penalized credit after all next-sweep handlers finish."""

    action = _integer(action_sweep_index, name="action_sweep_index")
    resolution = _integer(resolution_sweep_index, name="resolution_sweep_index")
    if resolution != action + 1:
        raise ValueError("delayed credit must wait for the next sweep")
    if not all_groups_completed or not native_sweep_end_completed:
        raise ValueError("delayed credit requires the complete native sweep end")
    anchor = _finite(anchor_error, name="anchor_error")
    endpoint = _finite(next_sweep_error, name="next_sweep_error")
    if anchor < 0.0 or endpoint < 0.0:
        raise ValueError("delayed-credit errors must be non-negative")
    retained = directional_survival(
        anchor_values=anchor_shared_values,
        candidate_values=candidate_shared_values,
        next_sweep_values=next_sweep_shared_values,
    )
    improvement = math.log(anchor + ERROR_FLOOR) - math.log(endpoint + ERROR_FLOOR)
    return DelayedHyperedgeCredit(
        action_sweep_index=action,
        resolution_sweep_index=resolution,
        survival=retained.survival,
        overwrite=retained.overwrite,
        next_sweep_log_improvement=improvement,
        penalized_credit=improvement - DELAYED_OVERWRITE_PENALTY * retained.overwrite,
    )
