"""Reference-blind policy for bounded, resumable search-state actions."""

from __future__ import annotations

import math
from dataclasses import dataclass


CONTINUE_CANONICAL_CC = "continue_canonical_cc"
RESUME_PHASE_I_SEARCH_STATE = "resume_phase_i_search_state"
CONTINUE_DIAGONAL_SEARCH_STATE = "continue_diagonal_search_state"
SUPPORTED_TRAJECTORY_ACTIONS = frozenset(
    {RESUME_PHASE_I_SEARCH_STATE, CONTINUE_DIAGONAL_SEARCH_STATE}
)
SEARCH_STATE_PROBE = "probe"
SEARCH_STATE_AWAITING_CONFIRMATION_CC = "awaiting_confirmation_cc"
SEARCH_STATE_CONFIRMATION = "confirmation"
SEARCH_STATE_EXPANSION = "expansion"
SEARCH_STATE_BLOCKED = "blocked"
SEARCH_STATE_INITIAL_PROBE = "initial_probe"

FIRST_PROBE_FRACTION = 0.01
CUMULATIVE_INTERVENTION_FRACTION = 0.15
CC_RESERVE_FRACTION = 0.10
BASE_UTILITY_RATIO = 1.50
ACCELERATING_CC_UTILITY_RATIO = 2.00
MINIMUM_CONFLICT_FRACTION = 0.50


@dataclass(frozen=True)
class SearchStateEvidence:
    complete_sweep: bool
    overlap_degree: float
    phase_rescue_enabled: bool
    repair_lock_active: bool
    phase_i_tail_utility: float
    non_coordinate_fraction: float
    conflict_fraction: float
    writeback_unstable: bool
    recent_cc_utilities: tuple[float, ...]
    remaining_fes: int
    max_fes: int
    population_size: int
    active_intervention_fraction: float = 0.0
    relative_writeback_max: float = 0.0
    relative_writeback_unstable: bool = False


@dataclass(frozen=True)
class SearchStateSchedulerState:
    phase: str = SEARCH_STATE_INITIAL_PROBE
    probe_utilities: tuple[float, ...] = ()
    intervention_fe: int = 0


@dataclass(frozen=True)
class SearchStateActionPlan:
    action_name: str
    stage: str
    requested_fes: int
    cc_reserve_fes: int
    required_utility_ratio: float
    trigger_reason: str


@dataclass(frozen=True)
class PreHoldEvidence:
    phase_i_tail_utility: float
    group_count: int
    mean_group_size: float
    overlap_edge_count: int
    overlap_edge_fraction: float
    shared_variable_count: int
    shared_variable_ratio: float
    mean_overlap_width: float
    remaining_fes: int
    remaining_ratio: float
    scheduled_hold_fes: int
    projected_unheld_group_fes: int
    projected_held_group_fes: int
    budget_retention_ratio: float


def build_pre_hold_evidence(
    *,
    phase_i_tail_utility: float,
    group_sizes: tuple[int, ...],
    overlapping_elements: tuple[tuple[int, ...], ...],
    dimension: int,
    remaining_fes: int,
    max_fes: int,
    scheduled_hold_fes: int,
) -> PreHoldEvidence:
    """Build a reference-blind snapshot before CC budget is withheld."""

    group_count = len(group_sizes)
    overlap_widths = tuple(len(shared) for shared in overlapping_elements)
    nonempty_widths = tuple(width for width in overlap_widths if width > 0)
    shared_variables = {
        int(variable)
        for shared in overlapping_elements
        for variable in shared
    }
    remaining_fes = max(0, int(remaining_fes))
    max_fes = max(0, int(max_fes))
    scheduled_hold_fes = min(remaining_fes, max(0, int(scheduled_hold_fes)))
    projected_unheld = (
        int(math.ceil(remaining_fes / group_count)) if group_count else 0
    )
    projected_held = (
        int(math.ceil((remaining_fes - scheduled_hold_fes) / group_count))
        if group_count
        else 0
    )
    tail_utility = float(phase_i_tail_utility)
    if not math.isfinite(tail_utility) or tail_utility < 0.0:
        tail_utility = 0.0
    return PreHoldEvidence(
        phase_i_tail_utility=tail_utility,
        group_count=group_count,
        mean_group_size=(sum(max(0, int(size)) for size in group_sizes) / group_count)
        if group_count
        else 0.0,
        overlap_edge_count=len(nonempty_widths),
        overlap_edge_fraction=(len(nonempty_widths) / len(overlap_widths))
        if overlap_widths
        else 0.0,
        shared_variable_count=len(shared_variables),
        shared_variable_ratio=(len(shared_variables) / max(1, int(dimension))),
        mean_overlap_width=(sum(nonempty_widths) / len(nonempty_widths))
        if nonempty_widths
        else 0.0,
        remaining_fes=remaining_fes,
        remaining_ratio=(remaining_fes / max_fes) if max_fes else 0.0,
        scheduled_hold_fes=scheduled_hold_fes,
        projected_unheld_group_fes=projected_unheld,
        projected_held_group_fes=projected_held,
        budget_retention_ratio=(projected_held / projected_unheld)
        if projected_unheld
        else 0.0,
    )


def normalized_gain_utility(
    incumbent_before: float,
    incumbent_after: float,
    actual_fes: int,
) -> float:
    """Return non-negative improvement per objective evaluation."""

    if not all(math.isfinite(float(value)) for value in (incumbent_before, incumbent_after)):
        return 0.0
    improvement = max(0.0, float(incumbent_before) - float(incumbent_after))
    return improvement / (
        max(abs(float(incumbent_before)), 1.0) * max(int(actual_fes), 1)
    )


def population_rounded_budget(budget: int, population_size: int) -> int:
    """Round a budget down so a block contains only complete populations."""

    budget = max(0, int(budget))
    population_size = int(population_size)
    if population_size <= 0:
        return 0
    return (budget // population_size) * population_size


def _required_utility_ratio(evidence: SearchStateEvidence) -> float:
    recent = tuple(float(value) for value in evidence.recent_cc_utilities[-2:])
    if len(recent) == 2 and recent[0] < recent[1]:
        return ACCELERATING_CC_UTILITY_RATIO
    return BASE_UTILITY_RATIO


def _reserve_fes(max_fes: int, *, terminal_probe: bool = False) -> int:
    if terminal_probe:
        return 0
    return int(math.ceil(max(0, int(max_fes)) * CC_RESERVE_FRACTION))


def _abstain(
    evidence: SearchStateEvidence,
    state: SearchStateSchedulerState,
    reason: str,
    *,
    terminal_probe: bool = False,
) -> SearchStateActionPlan:
    return SearchStateActionPlan(
        action_name=CONTINUE_CANONICAL_CC,
        stage=state.phase,
        requested_fes=0,
        cc_reserve_fes=_reserve_fes(
            evidence.max_fes,
            terminal_probe=terminal_probe,
        ),
        required_utility_ratio=_required_utility_ratio(evidence),
        trigger_reason=reason,
    )


def _eligible(
    evidence: SearchStateEvidence,
    trajectory_action_name: str,
) -> bool:
    structural_support = (
        evidence.non_coordinate_fraction >= MINIMUM_CONFLICT_FRACTION
        or evidence.conflict_fraction >= MINIMUM_CONFLICT_FRACTION
        or evidence.writeback_unstable
    )
    return (
        evidence.complete_sweep
        and evidence.overlap_degree > 0.0
        and evidence.phase_rescue_enabled
        and (
            not evidence.repair_lock_active
            or trajectory_action_name == CONTINUE_DIAGONAL_SEARCH_STATE
        )
        and evidence.phase_i_tail_utility > 0.0
        and structural_support
    )


def plan_search_state_action(
    evidence: SearchStateEvidence,
    state: SearchStateSchedulerState,
    *,
    new_complete_cc_sweep: bool = False,
    trajectory_action_name: str = RESUME_PHASE_I_SEARCH_STATE,
    terminal_probe: bool = False,
) -> SearchStateActionPlan:
    """Plan at most one bounded state action from current-run evidence."""

    if trajectory_action_name not in SUPPORTED_TRAJECTORY_ACTIONS:
        raise ValueError(
            f"unsupported trajectory action: {trajectory_action_name}"
        )
    if terminal_probe and state.intervention_fe > 0:
        return _abstain(
            evidence,
            state,
            "terminal_probe_already_consumed",
            terminal_probe=True,
        )
    if state.phase == SEARCH_STATE_BLOCKED:
        return _abstain(
            evidence,
            state,
            "state_action_permanently_blocked",
            terminal_probe=terminal_probe,
        )
    if (
        state.phase == SEARCH_STATE_AWAITING_CONFIRMATION_CC
        and not new_complete_cc_sweep
    ):
        return _abstain(
            evidence,
            state,
            "awaiting_new_complete_cc_sweep",
            terminal_probe=terminal_probe,
        )
    if state.phase not in {
        SEARCH_STATE_INITIAL_PROBE,
        SEARCH_STATE_AWAITING_CONFIRMATION_CC,
        SEARCH_STATE_EXPANSION,
    }:
        return _abstain(
            evidence,
            state,
            "state_action_phase_not_ready",
            terminal_probe=terminal_probe,
        )
    if not _eligible(evidence, trajectory_action_name):
        return _abstain(
            evidence,
            state,
            "runtime_evidence_ineligible",
            terminal_probe=terminal_probe,
        )

    max_fes = max(0, int(evidence.max_fes))
    reserve_fes = _reserve_fes(max_fes, terminal_probe=terminal_probe)
    cumulative_cap = int(math.floor(max_fes * CUMULATIVE_INTERVENTION_FRACTION))
    block_cap = int(math.floor(max_fes * FIRST_PROBE_FRACTION))
    remaining_action_fes = min(
        max(0, int(evidence.remaining_fes) - reserve_fes),
        max(0, cumulative_cap - int(state.intervention_fe)),
        block_cap,
    )
    requested_fes = population_rounded_budget(
        remaining_action_fes,
        evidence.population_size,
    )
    if requested_fes <= 0:
        return _abstain(
            evidence,
            state,
            "state_action_budget_unfunded",
            terminal_probe=terminal_probe,
        )

    stage = (
        SEARCH_STATE_CONFIRMATION
        if state.phase == SEARCH_STATE_AWAITING_CONFIRMATION_CC
        else SEARCH_STATE_EXPANSION
        if state.phase == SEARCH_STATE_EXPANSION
        else SEARCH_STATE_PROBE
    )
    reason = {
        SEARCH_STATE_PROBE: "initial_stateful_probe_from_runtime_evidence",
        SEARCH_STATE_CONFIRMATION: "confirmation_after_new_complete_cc_sweep",
        SEARCH_STATE_EXPANSION: "expansion_after_two_qualified_state_blocks",
    }[stage]
    if terminal_probe:
        reason = "terminal_probe_from_runtime_evidence"
    return SearchStateActionPlan(
        action_name=trajectory_action_name,
        stage=stage,
        requested_fes=requested_fes,
        cc_reserve_fes=reserve_fes,
        required_utility_ratio=_required_utility_ratio(evidence),
        trigger_reason=reason,
    )


def _utility_gate_passes(
    *,
    utility: float,
    required_utility_ratio: float,
    cc_utility: float,
) -> bool:
    if not math.isfinite(float(utility)) or float(utility) <= 0.0:
        return False
    latest_cc_utility = max(0.0, float(cc_utility))
    if latest_cc_utility == 0.0:
        return True
    return float(utility) >= float(required_utility_ratio) * latest_cc_utility


def record_search_state_outcome(
    state: SearchStateSchedulerState,
    *,
    stage: str,
    accepted: bool,
    utility: float,
    required_utility_ratio: float,
    cc_utility: float,
    used_fes: int,
) -> SearchStateSchedulerState:
    """Advance the state machine after a strict incumbent/utility audit."""

    used_fes = max(0, int(used_fes))
    intervention_fe = int(state.intervention_fe) + used_fes
    gate_passed = (
        used_fes > 0
        and bool(accepted)
        and _utility_gate_passes(
            utility=utility,
            required_utility_ratio=required_utility_ratio,
            cc_utility=cc_utility,
        )
    )
    if state.phase == SEARCH_STATE_BLOCKED or not gate_passed:
        return SearchStateSchedulerState(
            phase=SEARCH_STATE_BLOCKED,
            probe_utilities=state.probe_utilities,
            intervention_fe=intervention_fe,
        )

    if stage == SEARCH_STATE_PROBE and state.phase == SEARCH_STATE_INITIAL_PROBE:
        phase = SEARCH_STATE_AWAITING_CONFIRMATION_CC
    elif (
        stage == SEARCH_STATE_CONFIRMATION
        and state.phase == SEARCH_STATE_AWAITING_CONFIRMATION_CC
    ):
        phase = SEARCH_STATE_EXPANSION
    elif stage == SEARCH_STATE_EXPANSION and state.phase == SEARCH_STATE_EXPANSION:
        phase = SEARCH_STATE_EXPANSION
    else:
        return SearchStateSchedulerState(
            phase=SEARCH_STATE_BLOCKED,
            probe_utilities=state.probe_utilities,
            intervention_fe=intervention_fe,
        )

    return SearchStateSchedulerState(
        phase=phase,
        probe_utilities=state.probe_utilities + (float(utility),),
        intervention_fe=intervention_fe,
    )
