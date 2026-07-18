from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

ARAC_REPO_ROOT = Path(__file__).resolve().parents[1]
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"
HCC_VENDOR_ROOT = ARAC_REPO_ROOT / "vendor" / "hcc"
for import_root in (ARAC_REPO_ROOT, ARAC_SRC_ROOT, HCC_VENDOR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.arac.evidence.overlap_relation_builder import (
    OverlapRelation,
    build_overlap_relations,
)
from src.arac.policy.relation_policy import (
    ActionDecision as RelationActionDecision,
    RELATION_ACTION_ALIASES,
    action_mismatch_audit_row,
    score_actions_for_relations,
    score_actions_for_relations_v2,
    score_actions_for_relations_v21,
    score_actions_for_relations_v22,
    score_actions_for_relations_v23,
    score_actions_for_relations_v24,
    score_actions_for_relations_v25,
    score_actions_for_relations_v26,
    relation_policy_mode_for_evidence_action_controller_v3,
    relation_policy_mode_for_evidence_action_controller_v31,
    is_evidence_action_controller_v31_dense_overlap,
    select_evidence_action_controller_v31_dense_lock_mode,
)
from src.arac.policy.relation_policy import (
    decide_actions_for_relations_v24,
    decide_actions_for_relations_v26,
)
from src.arac.policy.action_trust_policy import (
    ActionTrustDecision,
    ActionTrustPolicy,
    make_action_key,
    normalized_objective_credit,
    robust_damped_writeback,
)
from src.arac.actions.controller_profiles import (
    controller_has_capability,
    controller_profile_by_version,
)
from arac.backends.hcc_evidence_overlay import (
    EvidenceOverlayArtifactPaths,
    HccEvidenceOverlayObserver,
)
from arac.policy.evidence_overlay import build_reference_blind_ordering
from src.arac.backends.hcc import (
    EVIDENCE_OVERLAY_MODES,
    required_aob_data_files,
    validate_aob_data_root,
)
from HCC.NDAs.MMES.state import MMESState

from AOB.utils import (
    combine,
    evaluation_record,
    load_design_matrix as load_aob_design_matrix,
    remove_overlapping_groups,
)


def Benchmark(*args, **kwargs):
    from AOB.AOB import Benchmark as _Benchmark

    return _Benchmark(*args, **kwargs)


def MMES(*args, **kwargs):
    from HCC.NDAs.MMES.mmes import MMES as _MMES

    return _MMES(*args, **kwargs)


def CMAES(*args, **kwargs):
    from HCC.OPT.CMAES.cmaes import CMAES as _CMAES

    return _CMAES(*args, **kwargs)


def Decomposition(*args, **kwargs):
    from HCC.RDDSM import Decomposition as _Decomposition

    return _Decomposition(*args, **kwargs)


def plot_evaluation_curve(*args, **kwargs):
    from AOB.utils import plot_evaluation_curve as _plot_evaluation_curve

    return _plot_evaluation_curve(*args, **kwargs)


def plot_evaluation_curve_best_so_far(*args, **kwargs):
    from AOB.utils import plot_evaluation_curve_best_so_far as _plot_evaluation_curve_best_so_far

    return _plot_evaluation_curve_best_so_far(*args, **kwargs)


DATA_DIR = HCC_VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
FUNCTION_NAMES = ("elliptic", "schwefel", "ackley")
PROBLEM_IDS = (1, 3, 4, 5)
ACTIVE_FUNCTION_ID_PAIRS = frozenset(
    {
        ("elliptic", 1),
        ("elliptic", 3),
        ("ackley", 4),
        ("schwefel", 5),
    }
)
ACTION_TRACE_FIELDS = [
    "problem_id",
    "seed",
    "outer_iter",
    "group_index",
    "selected_action_name",
    "relation_id",
    "group_left",
    "group_right",
    "shared_vars_hash",
    "action_family",
    "canonical_action_name",
    "relation_policy_source",
    "overlap_size",
    "previous_delta",
    "current_delta",
    "owner_selected",
    "semantic_surface",
    "state_mutated",
    "action_value_delta_norm",
    "downstream_consumed",
    "downstream_consumption_scope",
    "optimizer_consumed",
    "search_state_action_type",
    "search_state_backend",
    "candidate_protected",
    "cc_context_replaced",
    "stagnation_window",
    "delta_mean",
    "sigma_before",
    "sigma_after",
    "population_before",
    "population_after",
    "escape_budget",
    "bipop_restart_mode",
    "restart_triggered",
    "restart_accepted",
    "best_before",
    "restart_candidate_best",
    "restart_relative_improvement",
    "restart_acceptance_threshold",
    "best_after",
    "trace_event",
    "remaining_budget_ratio",
    "shared_var_count",
    "repair_lock_active",
    "refresh_budget",
    "continuation_reserve",
    "optimizer_seed",
    "scheduler_phase",
    "decision_point",
    "cc_block_fe",
    "cc_utility",
    "trust_key",
    "trust_phase",
    "trust_reason",
    "trust_score",
    "trust_exposure",
    "trust_cooldown",
    "trust_credit",
    "trust_unstable",
    "trust_pre_writeback_fitness",
    "trust_post_writeback_fitness",
    "fallback_route",
    "active_maturity_route",
    "sweep_evidence_relation_count",
    "sweep_evidence_active_count",
    "sweep_evidence_active_fraction",
    "sweep_evidence_support",
    "sweep_evidence_reason",
    "phase_rescue_resource_route",
    "phase_rescue_rejected_before_maturity",
    "phase_rescue_productive_mature",
    "phase_rescue_retired",
]
OVERLAP_RELATION_FIELDS = [
    "relation_id",
    "problem_id",
    "outer_iter",
    "group_left",
    "group_right",
    "shared_vars",
    "overlap_strength",
    "delta_signal",
    "rank_signal",
    "budget_remaining_ratio",
    "previous_delta",
    "current_delta",
    "delta_abs_gap",
    "delta_signed_gap",
    "delta_ratio_gap",
    "both_positive",
    "one_side_zero",
    "rank_gap",
    "rank_stability",
    "shared_var_count",
    "shared_var_support_ratio",
    "feature_coverage",
    "fallback_margin_proxy",
]
ACTION_DECISION_FIELDS = [
    "run_id",
    "problem_id",
    "relation_id",
    "group_left",
    "group_right",
    "shared_vars_count",
    "overlap_strength",
    "delta_signal",
    "rank_signal",
    "relation_action_name",
    "canonical_action_name",
    "action_family",
    "confidence",
    "trigger_reason",
]
ACTION_MISMATCH_AUDIT_FIELDS = [
    "run_id",
    "problem_id",
    "relation_id",
    "group_left",
    "group_right",
    "candidate_scores",
    "coordinate_score",
    "isolate_conflicting_relation_score",
    "reassign_repair_score",
    "fallback_score",
    "best_action_name",
    "best_score",
    "second_best_action_name",
    "second_best_score",
    "margin",
    "final_action_name",
    "final_canonical_action_name",
    "confidence",
    "trigger_reason",
    "abstain_reason",
]
BUDGET_SUMMARY_FIELDS = [
    "problem_id",
    "budget_accounting",
    "max_fes",
    "optimizer_reported_fe",
    "fitness_record_fe",
    "budget_aligned_fe",
    "same_budget_violation",
    "global_phase_fe",
    "cc_phase_fe",
    "rescue_fe",
    "refresh_fe",
    "search_state_fe",
    "separable_continuation_fe",
    "overhead_fe",
]
AOB_INPUT_MANIFEST_FIELDS = [
    "problem_id",
    "file",
    "path",
    "sha256_before",
    "sha256_after",
    "unchanged",
]
EVIDENCE_OVERLAY_REQUIRED_SWEEPS = 3
EVIDENCE_OVERLAY_PROBE_FE = 16
ACTION_VALUE_DELTA_GUARD_THRESHOLD = 0.5
COORDINATE_ACTION_VALUE_DELTA_GUARD_THRESHOLD = 2.5
ACTION_TRUST_MIN_WRITEBACK_NORM = 1e-12
V36_FIRST_SWEEP_OUTER_ITER = 0
V36_MIN_ACTIVE_COUNT = 4
V36_MIN_ACTIVE_FRACTION = 0.20
V36_MAX_ACTIVE_FRACTION = 0.30
V36_MIN_CONFIDENCE_RANK_SUPPORT = 0.50
V31_NON_DENSE_PREFIX_RELATION_COUNT = 3
V31_NON_DENSE_PREFIX_SHARED_VAR_COUNT = 3
V31_NON_DENSE_PREFIX_REPAIR_TRIGGER = "controller_v31_non_dense_prefix_repair_lock"
V31_NON_DENSE_LARGE_FALLBACK_DELTA_RATIO_MAX = 0.15
V31_NON_DENSE_LARGE_FALLBACK_NORM_MIN = 10.0
V31_NON_DENSE_LARGE_FALLBACK_REPAIR_TRIGGER = (
    "controller_v31_non_dense_large_fallback_repair_lock"
)
PHASE_RESCUE_MULTISTART_ACTION = "phase_rescue_multistart"
EVIDENCE_ACTION_CONTROLLER_V37 = controller_profile_by_version(37).action_name
REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER = 0.5
BIPOP_STAGNATION_EPSILON = 1e-8
BIPOP_RESTART_COOLDOWN = 1
BIPOP_REJECT_BACKOFF_SWEEP_CAP = 3
PHASE_RESCUE_START_COUNT = 3
PHASE_RESCUE_SIGMA_MULTIPLIER = 1.5
PHASE_RESCUE_ESCAPE_BUDGET_FRACTION = 0.60
PHASE_RESCUE_STAGNATION_WINDOW = 1
RELATIVE_WRITEBACK_UNSTABLE_THRESHOLD = 0.10
REPAIR_ACTION_NAMES = {"repair_shared_variable_binding"}
RELATION_ACTION_FAMILIES = {
    "coordinate": "coordinate",
    "isolate_conflicting_relation": "isolate",
    "reassign_repair": "reassign_repair",
    "fallback": "fallback",
}
SHUFFLED_NEGATIVE_CONTROL_ACTIONS = {
    "coordinate": "reassign_repair",
    "reassign_repair": "coordinate",
    "isolate_conflicting_relation": "coordinate",
    "fallback": "fallback",
}


def evidence_overlay_sweep_barrier_ready(
    *,
    mode: str,
    complete_sweep_count: int,
    previous_survival_closed: bool,
    barrier_attempted: bool,
    all_raw_groups_completed: bool,
) -> bool:
    """Return whether the one-shot overlay barrier is ready to execute."""

    if mode not in EVIDENCE_OVERLAY_MODES:
        raise ValueError("unsupported evidence overlay mode")
    if isinstance(complete_sweep_count, bool) or complete_sweep_count < 0:
        raise ValueError("complete_sweep_count must be non-negative")
    return bool(
        mode != "off"
        and not barrier_attempted
        and all_raw_groups_completed
        and complete_sweep_count >= EVIDENCE_OVERLAY_REQUIRED_SWEEPS
        and previous_survival_closed
    )


def evidence_overlay_scheduled_sub_fes(
    *,
    mode: str,
    has_overlap: bool,
    complete_sweep_count: int,
    cc_budget_limit_fe: int,
    current_fe: int,
    terminal_tolerance_fe: int,
    sub_num: int,
    population_sizes: list[int] | tuple[int, ...],
    frozen_sub_fes: int | None,
    plan_ready: bool = True,
    probe_pending: bool = True,
    barrier_attempted: bool = False,
    delayed_outcomes_pending: bool = False,
) -> int | None:
    """Plan equal evidence slots with a worst-case v37 rescue reserve."""

    if mode not in EVIDENCE_OVERLAY_MODES:
        raise ValueError("unsupported evidence overlay mode")
    if not isinstance(plan_ready, bool):
        raise ValueError("plan_ready must be boolean")
    if not isinstance(probe_pending, bool):
        raise ValueError("probe_pending must be boolean")
    if not isinstance(barrier_attempted, bool):
        raise ValueError("barrier_attempted must be boolean")
    if not isinstance(delayed_outcomes_pending, bool):
        raise ValueError("delayed_outcomes_pending must be boolean")
    integer_values = {
        "complete_sweep_count": complete_sweep_count,
        "cc_budget_limit_fe": cc_budget_limit_fe,
        "current_fe": current_fe,
        "terminal_tolerance_fe": terminal_tolerance_fe,
        "sub_num": sub_num,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values.values()
    ):
        raise ValueError("evidence overlay schedule values must be integers")
    if complete_sweep_count < 0 or current_fe < 0 or terminal_tolerance_fe < 0:
        raise ValueError("evidence overlay schedule values must be non-negative")
    if cc_budget_limit_fe < current_fe or sub_num <= 0:
        raise ValueError("evidence overlay schedule budget is invalid")
    if mode == "off":
        return None
    if barrier_attempted and not delayed_outcomes_pending:
        return None
    target_sweeps = EVIDENCE_OVERLAY_REQUIRED_SWEEPS + int(has_overlap)
    if complete_sweep_count >= target_sweeps and plan_ready:
        return None
    effective_complete_sweeps = min(complete_sweep_count, target_sweeps - 1)
    populations = tuple(int(value) for value in population_sizes)
    if len(populations) != sub_num or any(value <= 0 for value in populations):
        raise ValueError("population sizes must align with overlay groups")
    probe_reserve = (
        EVIDENCE_OVERLAY_PROBE_FE
        if has_overlap and probe_pending
        else 0
    )
    available = max(
        0,
        cc_budget_limit_fe
        - current_fe
        - probe_reserve
        - terminal_tolerance_fe,
    )
    remaining_slots = target_sweeps - effective_complete_sweeps
    minimum_sub_fes = max(populations)

    def fits(candidate_sub_fes: int) -> bool:
        return (
            remaining_slots
            * evidence_overlay_normal_sweep_reserve(
                populations,
                sub_fes=candidate_sub_fes,
            )
            <= available
        )

    if frozen_sub_fes is not None:
        if frozen_sub_fes < minimum_sub_fes:
            raise RuntimeError("frozen evidence-overlay group budget is sub-population")
        if not fits(frozen_sub_fes):
            raise RuntimeError(
                "frozen evidence-overlay group budget exceeds remaining reserve"
            )
        return frozen_sub_fes

    upper_sub_fes = available // max(1, remaining_slots * sub_num)
    if upper_sub_fes < minimum_sub_fes:
        raise RuntimeError(
            "insufficient budget for frozen evidence-overlay sweep populations"
        )
    low = minimum_sub_fes
    high = upper_sub_fes
    best = minimum_sub_fes - 1
    while low <= high:
        candidate = (low + high) // 2
        if fits(candidate):
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    if best < minimum_sub_fes:
        raise RuntimeError(
            "insufficient budget for frozen evidence-overlay sweep populations"
        )
    return best


def evidence_overlay_normal_sweep_reserve(
    population_sizes: list[int] | tuple[int, ...],
    *,
    sub_fes: int | None = None,
) -> int:
    converted = tuple(int(value) for value in population_sizes)
    if not converted or any(value <= 0 for value in converted):
        raise ValueError("population sizes must be positive")
    if sub_fes is None:
        return sum(value + 1 for value in converted)
    if isinstance(sub_fes, bool) or not isinstance(sub_fes, int) or sub_fes < 0:
        raise ValueError("sub_fes must be a non-negative integer")
    return sum(
        evidence_overlay_group_interval_reserve(population, sub_fes)
        for population in converted
    )


def evidence_overlay_group_interval_reserve(
    population_size: int,
    sub_fes: int,
) -> int:
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size <= 0
    ):
        raise ValueError("population_size must be a positive integer")
    if isinstance(sub_fes, bool) or not isinstance(sub_fes, int) or sub_fes < 0:
        raise ValueError("sub_fes must be a non-negative integer")
    return (
        1
        + sub_fes
        + max(
            PHASE_RESCUE_START_COUNT * population_size,
            math.ceil(sub_fes * PHASE_RESCUE_ESCAPE_BUDGET_FRACTION),
        )
    )


@dataclass(frozen=True)
class SmokeConfig:
    max_fes: int
    seed: int | None
    run_id: str = "arac-hcc-smoke"
    sigma: float = 0.5
    verbose: int = 1000
    early_stopping_evaluations: int = 1000
    mmes_restart: bool = True
    cmaes_restart: bool = True
    arac_action: str = "conservative_no_action"
    enable_relation_dispatch: bool = False
    relation_policy_mode: str = "rule"
    arac_action_file: Path | None = None
    budget_accounting: str = "strict"
    skip_plots: bool = False
    aob_data_root: Path = DATA_DIR
    search_state_backend: str = "phase_i_mmes"
    evidence_overlay_mode: str = "off"


@dataclass(frozen=True)
class RelationExecutionContext:
    overlap_indices: list[int]
    previous_values: np.ndarray
    current_values: np.ndarray
    previous_delta: float
    current_delta: float


@dataclass
class PendingActionTrustObservation:
    decision: ActionTrustDecision
    pre_writeback_fitness: float
    unstable: bool
    trace_row: dict[str, str]


@dataclass
class EvidenceActionControllerV31RunState:
    dense_overlap: bool
    action_trust_policy: ActionTrustPolicy | None = field(default=None, repr=False)
    pending_action_trust: PendingActionTrustObservation | None = field(
        default=None,
        repr=False,
    )
    locked_policy_mode: str | None = None
    non_dense_repair_locked: bool = False
    non_dense_repair_lock_trigger: str = ""
    phase_i_optimizer: object | None = field(default=None, repr=False)
    phase_i_state: MMESState | None = field(default=None, repr=False)
    phase_i_runtime_tail_utility: float = 0.0
    v36_enabled: bool = False
    v37_enabled: bool = False
    sweep_evidence_outer_iter: int | None = None
    sweep_evidence_relation_count: int = 0
    sweep_evidence_active_count: int = 0
    sweep_evidence_active_families: set[str] = field(default_factory=set)
    sweep_evidence_support_sum: float = 0.0
    sweep_evidence_valid: bool = True
    sweep_evidence_finalized: bool = False
    coordinate_maturity_latched: bool = False
    sweep_evidence_reason: str = ""
    phase_rescue_rejected_before_maturity: int = 0
    phase_rescue_productive_mature: bool = False
    phase_rescue_retired: bool = False
    phase_rescue_resource_reason: str = ""
    _non_dense_guarded_prefix: list[tuple[int, int, str, str]] = field(
        default_factory=list,
        repr=False,
    )

    @property
    def effective_policy_mode(self) -> str:
        if not self.dense_overlap:
            return "adaptive_v26"
        return self.locked_policy_mode or "adaptive_v24"

    @property
    def phase_rescue_enabled(self) -> bool:
        return (
            not self.dense_overlap
            and not self.non_dense_repair_locked
            and not self.phase_rescue_retired
        )

    @property
    def sweep_evidence_active_fraction(self) -> float:
        if self.sweep_evidence_relation_count == 0:
            return 0.0
        return (
            self.sweep_evidence_active_count
            / self.sweep_evidence_relation_count
        )

    @property
    def sweep_evidence_support(self) -> float:
        if self.sweep_evidence_active_count == 0:
            return 0.0
        return self.sweep_evidence_support_sum / self.sweep_evidence_active_count

    def prepare_v36_outer_iter(self, outer_iter: int) -> None:
        if not self.v36_enabled or self.sweep_evidence_finalized:
            return
        current_outer_iter = int(outer_iter)
        if self.sweep_evidence_outer_iter is None:
            self.sweep_evidence_outer_iter = current_outer_iter
            return
        if (
            self.sweep_evidence_outer_iter == V36_FIRST_SWEEP_OUTER_ITER
            and current_outer_iter != V36_FIRST_SWEEP_OUTER_ITER
        ):
            self._finalize_v36_first_sweep()

    def observe_v36_relation(
        self,
        relation: OverlapRelation,
        action: RelationActionDecision,
    ) -> None:
        if (
            not self.v36_enabled
            or self.sweep_evidence_finalized
            or relation.outer_iter != V36_FIRST_SWEEP_OUTER_ITER
        ):
            return
        self.sweep_evidence_relation_count += 1
        if action.action_family == "fallback":
            return
        self.sweep_evidence_active_count += 1
        self.sweep_evidence_active_families.add(action.action_family)
        confidence = float(action.confidence)
        rank_signal = float(relation.rank_signal)
        if (
            not math.isfinite(confidence)
            or not math.isfinite(rank_signal)
            or not 0.0 <= confidence <= 1.0
            or not 0.0 <= rank_signal <= 1.0
        ):
            self.sweep_evidence_valid = False
            return
        self.sweep_evidence_support_sum += confidence * rank_signal

    def _finalize_v36_first_sweep(self) -> None:
        active_fraction = self.sweep_evidence_active_fraction
        support = self.sweep_evidence_support
        self.coordinate_maturity_latched = (
            self.sweep_evidence_valid
            and self.sweep_evidence_active_count >= V36_MIN_ACTIVE_COUNT
            and V36_MIN_ACTIVE_FRACTION
            <= active_fraction
            <= V36_MAX_ACTIVE_FRACTION
            and self.sweep_evidence_active_families == {"coordinate"}
            and support >= V36_MIN_CONFIDENCE_RANK_SUPPORT
            and not self.non_dense_repair_locked
        )
        self.sweep_evidence_reason = (
            "first_sweep_sparse_coordinate_mature"
            if self.coordinate_maturity_latched
            else "first_sweep_evidence_not_mature"
        )
        self.sweep_evidence_finalized = True

    def observe_v37_phase_rescue(self, *, accepted: bool) -> str:
        if not self.v37_enabled or self.phase_rescue_retired:
            return ""
        if accepted:
            if self.phase_rescue_productive_mature:
                return ""
            self.phase_rescue_productive_mature = True
            self.phase_rescue_resource_reason = "productive_phase_rescue_mature"
            return self.phase_rescue_resource_reason
        if self.phase_rescue_productive_mature:
            return ""
        self.phase_rescue_rejected_before_maturity += 1
        if self.phase_rescue_rejected_before_maturity < PHASE_RESCUE_START_COUNT:
            return ""
        self.phase_rescue_retired = True
        self.phase_rescue_resource_reason = "zero_yield_phase_rescue_retired"
        return self.phase_rescue_resource_reason

    def lock_from_runtime_prefix(self, relations: list[OverlapRelation]) -> None:
        if not self.dense_overlap or self.locked_policy_mode is not None:
            return
        selected_mode = select_evidence_action_controller_v31_dense_lock_mode(
            relations
        )
        if selected_mode is not None:
            self.locked_policy_mode = selected_mode

    def observe_guarded_relation_action(
        self,
        relation: OverlapRelation,
        action: RelationActionDecision,
    ) -> None:
        if (
            self.dense_overlap
            or self.non_dense_repair_locked
            or len(self._non_dense_guarded_prefix) >= V31_NON_DENSE_PREFIX_RELATION_COUNT
        ):
            return
        self._non_dense_guarded_prefix.append(
            (
                relation.outer_iter,
                len(relation.shared_vars),
                action.relation_action_name,
                action.trigger_reason,
            )
        )
        if len(self._non_dense_guarded_prefix) < V31_NON_DENSE_PREFIX_RELATION_COUNT:
            return
        outer_iterations = {row[0] for row in self._non_dense_guarded_prefix}
        shared_var_counts = [row[1] for row in self._non_dense_guarded_prefix]
        action_names = [row[2] for row in self._non_dense_guarded_prefix]
        trigger_reasons = [row[3] for row in self._non_dense_guarded_prefix]
        should_lock = (
            len(outer_iterations) == 1
            and all(
                count == V31_NON_DENSE_PREFIX_SHARED_VAR_COUNT
                for count in shared_var_counts
            )
            and all(action_name == "fallback" for action_name in action_names)
            and trigger_reasons[-2:]
            == [
                "action_value_delta_guard_exceeded",
                "action_value_delta_guard_exceeded",
            ]
        )
        if should_lock:
            self.non_dense_repair_locked = True
            self.non_dense_repair_lock_trigger = V31_NON_DENSE_PREFIX_REPAIR_TRIGGER

    def lock_from_large_fallback_writeback(
        self,
        relation: OverlapRelation,
        action: RelationActionDecision,
        action_value_delta_norm: float,
    ) -> None:
        if (
            self.dense_overlap
            or self.non_dense_repair_locked
            or self._non_dense_guarded_prefix
            or len(relation.shared_vars) != V31_NON_DENSE_PREFIX_SHARED_VAR_COUNT
            or action.relation_action_name != "fallback"
            or action.trigger_reason != "no_deterministic_relation_rule_triggered"
            or not relation.both_positive
            or relation.delta_ratio_gap > V31_NON_DENSE_LARGE_FALLBACK_DELTA_RATIO_MAX
            or action_value_delta_norm < V31_NON_DENSE_LARGE_FALLBACK_NORM_MIN
        ):
            return
        self.non_dense_repair_locked = True
        self.non_dense_repair_lock_trigger = V31_NON_DENSE_LARGE_FALLBACK_REPAIR_TRIGGER

    def forced_relation_action(
        self,
        relation: OverlapRelation,
    ) -> RelationActionDecision | None:
        if (
            self.dense_overlap
            or not self.non_dense_repair_locked
            or not relation.shared_vars
        ):
            return None
        return RelationActionDecision(
            relation_id=relation.relation_id,
            action_name="reassign_repair",
            action_family="reassign_repair",
            confidence=1.0,
            trigger_reason=self.non_dense_repair_lock_trigger,
        )

    def register_pending_action_trust(
        self,
        *,
        decision: ActionTrustDecision | None,
        pre_writeback_fitness: float,
        unstable: bool,
        trace_row: dict[str, str],
    ) -> None:
        if (
            self.action_trust_policy is None
            or decision is None
            or not decision.allow_intervention
        ):
            self.pending_action_trust = None
            return
        self.pending_action_trust = PendingActionTrustObservation(
            decision=decision,
            pre_writeback_fitness=float(pre_writeback_fitness),
            unstable=bool(unstable),
            trace_row=trace_row,
        )
        trace_row["trust_pre_writeback_fitness"] = (
            f"{float(pre_writeback_fitness):.17e}"
        )

    def observe_pending_action_trust(
        self,
        *,
        post_writeback_fitness: float,
    ) -> float | None:
        pending = self.pending_action_trust
        if self.action_trust_policy is None or pending is None:
            return None
        self.pending_action_trust = None
        credit = normalized_objective_credit(
            pending.pre_writeback_fitness,
            post_writeback_fitness,
        )
        self.action_trust_policy.observe(
            pending.decision.key,
            credit=credit,
            unstable=pending.unstable,
        )
        pending.trace_row["trust_credit"] = f"{credit:.6e}"
        pending.trace_row["trust_unstable"] = str(int(pending.unstable))
        pending.trace_row["trust_post_writeback_fitness"] = (
            f"{float(post_writeback_fitness):.17e}"
        )
        pending.trace_row["downstream_consumed"] = "1"
        pending.trace_row["downstream_consumption_scope"] = (
            "next_group_original_fitness"
        )
        pending.trace_row["optimizer_consumed"] = "1"
        return credit

    def invalidate_pending_action_trust(self, reason: str) -> None:
        pending = self.pending_action_trust
        if pending is None:
            return
        if not reason:
            raise ValueError("pending action invalidation reason must not be empty")
        pending.trace_row["trust_reason"] = reason
        pending.trace_row["trust_credit"] = ""
        pending.trace_row["trust_post_writeback_fitness"] = ""
        self.pending_action_trust = None

def build_evidence_action_controller_v31_run_state(
    degree_of_overlap: float,
    *,
    action_name: str | None = None,
) -> EvidenceActionControllerV31RunState:
    return EvidenceActionControllerV31RunState(
        dense_overlap=is_evidence_action_controller_v31_dense_overlap(degree_of_overlap),
        action_trust_policy=(
            ActionTrustPolicy()
            if action_name is not None
            and controller_has_capability(action_name, "risk_aware_trust")
            else None
        ),
        v36_enabled=(
            action_name is not None
            and controller_has_capability(action_name, "maturity")
        ),
        v37_enabled=(
            action_name is not None
            and controller_has_capability(action_name, "rescue_retirement")
        ),
    )


def _resolved_aob_data_root(data_root: Path | str | None = None) -> Path:
    return Path(DATA_DIR if data_root is None else data_root).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_aob_inputs(
    fun_id: int,
    data_root: Path | str | None = None,
) -> dict[str, dict[str, str]]:
    root = validate_aob_data_root(_resolved_aob_data_root(data_root), fun_id)
    snapshot: dict[str, dict[str, str]] = {}
    for path in sorted(required_aob_data_files(root, fun_id), key=lambda item: item.name):
        snapshot[path.name] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
    return snapshot


def build_aob_input_audit_rows(
    problem_id: str,
    before: dict[str, dict[str, str]],
    after: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for filename in sorted(set(before) | set(after)):
        before_row = before.get(filename, {})
        after_row = after.get(filename, {})
        before_hash = before_row.get("sha256", "missing")
        after_hash = after_row.get("sha256", "missing")
        rows.append(
            {
                "problem_id": problem_id,
                "file": filename,
                "path": before_row.get("path", after_row.get("path", "")),
                "sha256_before": before_hash,
                "sha256_after": after_hash,
                "unchanged": str(int(before_hash == after_hash and before_hash != "missing")),
            }
        )
    return rows


def require_unchanged_aob_inputs(
    problem_id: str,
    rows: list[dict[str, str]],
) -> None:
    changed = [row["file"] for row in rows if row["unchanged"] != "1"]
    if changed:
        raise RuntimeError(
            f"AOB input changed during {problem_id}: {','.join(changed)}"
        )


def _write_aob_input_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AOB_INPUT_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_aob_metadata(fun_id: int, data_root: Path | str | None = None) -> dict:
    root = _resolved_aob_data_root(data_root)
    with (root / f"F{fun_id}-info.txt").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_design_matrix(fun_id: int, data_root: Path | str | None = None) -> np.ndarray:
    root = _resolved_aob_data_root(data_root)
    return load_aob_design_matrix(root / f"F{fun_id}-design.txt")


def load_reference_blind_design_matrix(
    fun_id: int,
    data_root: Path | str | None = None,
) -> np.ndarray:
    """Load a possibly truncated RDDSM matrix without consulting AOB truth.

    The bundled pilot matrices contain a contiguous prefix of complete rows and
    an optional truncated tail.  A missing row is recoverable only when its
    observed symmetric prefix maps to one unique, already observed full-row
    paradigm.  Any ambiguity or structural inconsistency fails closed instead
    of falling back to ``F*-info/s/p`` metadata.
    """

    root = _resolved_aob_data_root(data_root)
    path = root / f"F{fun_id}-design.txt"
    rows: list[np.ndarray] = []
    dimension: int | None = None
    incomplete_seen = False
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, raw_row in enumerate(csv.reader(handle), start=1):
            if not raw_row:
                raise ValueError(
                    f"reference-blind design matrix has an empty row at line {line_number}"
                )
            try:
                values = np.asarray([float(value) for value in raw_row], dtype=float)
            except ValueError as error:
                raise ValueError(
                    f"reference-blind design matrix has a non-numeric value at line {line_number}"
                ) from error
            if not np.all(np.isfinite(values)) or not np.all(
                np.logical_or(values == 0.0, values == 1.0)
            ):
                raise ValueError(
                    f"reference-blind design matrix must be finite and binary at line {line_number}"
                )
            if dimension is None:
                dimension = int(values.size)
                if dimension <= 0:
                    raise ValueError("reference-blind design matrix must be non-empty")
            if values.size > dimension:
                raise ValueError(
                    "reference-blind design matrix row exceeds the first-row dimension"
                )
            if values.size < dimension:
                incomplete_seen = True
            elif incomplete_seen:
                raise ValueError(
                    "reference-blind design matrix complete rows must form one prefix"
                )
            rows.append(values.astype(np.uint8, copy=False))

    if dimension is None or not rows:
        raise ValueError("reference-blind design matrix must be non-empty")
    if len(rows) > dimension:
        raise ValueError("reference-blind design matrix has more rows than columns")

    complete_count = 0
    for row in rows:
        if row.size != dimension:
            break
        complete_count += 1
    if complete_count == 0:
        raise ValueError(
            "reference-blind design matrix requires a complete row prefix"
        )

    complete_rows = np.stack(rows[:complete_count])
    prefix_templates: dict[bytes, np.ndarray] = {}
    ambiguous_prefixes: set[bytes] = set()
    for row in complete_rows:
        key = row[:complete_count].tobytes()
        existing = prefix_templates.get(key)
        if existing is None:
            prefix_templates[key] = row
        elif not np.array_equal(existing, row):
            ambiguous_prefixes.add(key)

    matrix = np.empty((dimension, dimension), dtype=np.uint8)
    matrix[:complete_count] = complete_rows
    for row_index in range(complete_count, dimension):
        key = complete_rows[:, row_index].tobytes()
        if key in ambiguous_prefixes:
            raise ValueError(
                "reference-blind design row prefix maps to multiple paradigms: "
                f"row={row_index}"
            )
        template = prefix_templates.get(key)
        if template is None:
            raise ValueError(
                "reference-blind design row prefix has no observed paradigm: "
                f"row={row_index}"
            )
        if row_index < len(rows):
            observed = rows[row_index]
            if not np.array_equal(template[: observed.size], observed):
                raise ValueError(
                    "reference-blind design row conflicts with its observed cells: "
                    f"row={row_index}"
                )
        matrix[row_index] = template

    if not np.array_equal(matrix, matrix.T):
        raise ValueError("reference-blind reconstructed design matrix is not symmetric")
    if not np.all(np.diag(matrix) == 1):
        raise ValueError(
            "reference-blind reconstructed design matrix has a missing diagonal"
        )
    return matrix


def load_permutation_vector(fun_id: int, data_root: Path | str | None = None) -> list[int]:
    root = _resolved_aob_data_root(data_root)
    return (np.loadtxt(root / f"F{fun_id}-p.txt", delimiter=",").reshape(-1).astype(int) - 1).tolist()


def build_aob_topology_groups(
    fun_id: int,
    data_root: Path | str | None = None,
) -> list[list[int]]:
    metadata = load_aob_metadata(fun_id, data_root)
    permutation = load_permutation_vector(fun_id, data_root)
    overlap = int(metadata["overlap_degree"])
    groups: list[list[int]] = []
    begin_index = 0
    for index, subgroup_size in enumerate(metadata["subgroups"]):
        end_index = begin_index + int(subgroup_size)
        groups.append(permutation[begin_index:end_index])
        if index != len(metadata["subgroups"]) - 1:
            begin_index = end_index - overlap
    return groups


def order_grouping_by_aob_topology(
    grouping_result: list[list[int]],
    fun_id: int,
    data_root: Path | str | None = None,
) -> list[list[int]]:
    topology_groups = build_aob_topology_groups(fun_id, data_root)
    grouping_by_members = {
        frozenset(int(variable) for variable in group): [int(variable) for variable in group]
        for group in grouping_result
    }
    ordered_groups = []
    missing_groups = []
    for topology_group in topology_groups:
        key = frozenset(topology_group)
        if key not in grouping_by_members:
            missing_groups.append(sorted(key))
            continue
        ordered_groups.append([int(variable) for variable in topology_group])

    topology_keys = {frozenset(group) for group in topology_groups}
    extra_groups = [sorted(key) for key in grouping_by_members if key not in topology_keys]
    if missing_groups or extra_groups:
        raise ValueError(
            "RDDSM grouping does not match AOB topology: "
            f"missing={len(missing_groups)}, extra={len(extra_groups)}"
        )
    return ordered_groups


def decompose_problem(
    fun_id: int,
    data_root: Path | str | None = None,
) -> list[list[int]]:
    grouping_result = Decomposition(load_design_matrix(fun_id, data_root)).decomposition()
    return order_grouping_by_aob_topology(grouping_result, fun_id, data_root)


def load_runtime_grouping(
    fun_id: int,
    data_root: Path | str | None,
    *,
    evidence_overlay_mode: str,
) -> list[list[int]]:
    if evidence_overlay_mode not in EVIDENCE_OVERLAY_MODES:
        raise ValueError("unsupported evidence overlay mode")
    if evidence_overlay_mode == "off":
        return decompose_problem(fun_id, data_root)
    raw_groups = Decomposition(
        load_reference_blind_design_matrix(fun_id, data_root)
    ).decomposition()
    ordering = build_reference_blind_ordering(raw_groups)
    return [list(group) for group in ordering.ordered_groups]


def calculate_runtime_overlap_degree(
    overlap_groups: list[list[int]],
    *,
    problem_dimension: int,
    fun_id: int,
    data_root: Path | str | None,
    evidence_overlay_mode: str,
) -> float:
    if evidence_overlay_mode not in EVIDENCE_OVERLAY_MODES:
        raise ValueError("unsupported evidence overlay mode")
    dimension = int(problem_dimension)
    if dimension <= 0:
        raise ValueError("problem_dimension must be positive")
    if evidence_overlay_mode != "off":
        return calculate_degree_of_overlap(overlap_groups, dimension)
    metadata = load_aob_metadata(fun_id, data_root)
    return calculate_degree_of_overlap(overlap_groups, int(metadata["dimension"]))


def calculate_degree_of_overlap(overlap_groups: list[list[int]], problem_dimension: int) -> float:
    overlapping_variables = set()
    for group in overlap_groups:
        if isinstance(group, np.integer):
            overlapping_variables.add(int(group))
        elif isinstance(group, int):
            overlapping_variables.add(group)
        else:
            overlapping_variables.update(group)
    return len(overlapping_variables) / problem_dimension


def calculate_global_fes(total_fes: int, degree_of_overlap: float) -> int:
    if degree_of_overlap == 0:
        return 0
    return int((0.2 + (4 / 5) * degree_of_overlap) * total_fes)


def calculate_cmaes_population_size(subspace_dimension: int) -> int:
    return 4 + 3 * math.ceil(math.log(subspace_dimension))


def current_fitness_evaluations(fun) -> int:
    return len(getattr(fun, "fitness_record", []))


def observed_optimizer_fe(
    fun,
    *,
    evaluations_before: int,
    optimizer_reported_fe: int,
) -> int:
    """Prefer objective-observed FE so partial final batches are not overcounted."""

    reported = max(0, int(optimizer_reported_fe))
    if not hasattr(fun, "fitness_record"):
        return reported
    observed = current_fitness_evaluations(fun) - max(0, int(evaluations_before))
    if observed < 0:
        raise RuntimeError("objective FE counter moved backwards")
    return observed


def normalized_gain_utility(
    incumbent_before: float,
    incumbent_after: float,
    actual_fes: int,
) -> float:
    if not all(
        math.isfinite(float(value))
        for value in (incumbent_before, incumbent_after)
    ):
        return 0.0
    improvement = max(0.0, float(incumbent_before) - float(incumbent_after))
    return improvement / (
        max(abs(float(incumbent_before)), 1.0) * max(int(actual_fes), 1)
    )


def bounded_population_budget(
    requested_fes: int,
    remaining_fes: int,
    population_size: int,
) -> int:
    usable_fes = min(requested_fes, remaining_fes)
    if usable_fes <= 0 or population_size <= 0:
        return 0
    return (usable_fes // population_size) * population_size


def scale_free_writeback_norm(
    *,
    delta_norm: float,
    shared_count: int,
    lower: float,
    upper: float,
) -> float:
    """Normalize a shared-variable writeback by its bounded subspace span."""

    delta_norm = abs(float(delta_norm))
    shared_count = max(1, int(shared_count))
    span = abs(float(upper) - float(lower))
    if not all(math.isfinite(value) for value in (delta_norm, span)) or span <= 0.0:
        return 0.0
    return delta_norm / (math.sqrt(shared_count) * span)


def is_evidence_action_controller_v37(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V37


def uses_v33_trust_trace_schema(action_name: str) -> bool:
    return controller_has_capability(action_name, "trust_trace")


def relation_downstream_consumption_scope(
    *,
    action_name: str,
    writeback_active: bool,
) -> str:
    if not uses_v33_trust_trace_schema(action_name):
        return "same_outer_iteration"
    return "same_outer_iteration" if writeback_active else "no_state_change"


def controller_v33_fallback_route(
    *,
    canonical_action_name: str,
    controller_run_state: EvidenceActionControllerV31RunState | None,
) -> str:
    if canonical_action_name in {
        "allow_beneficial_coordination",
        "repair_shared_variable_binding",
        "isolate_conflicting_relation",
    }:
        return ""
    if controller_run_state is None:
        return ""
    if controller_run_state.dense_overlap:
        return "dense_preserve_v31"
    return "non_dense_bounded_0_5"


def uses_phase_rescue_during_run(
    action_name: str,
    *,
    phase_rescue_enabled: bool,
) -> bool:
    return bool(
        is_evidence_action_controller_v37(action_name)
        and phase_rescue_enabled
    )


def overlap_action_name_for_lane(action_name: str) -> str:
    if is_evidence_action_controller_v37(action_name):
        return "conservative_no_action"
    return action_name


def refine_sigma_for_action(
    action_name: str,
    base_sigma: float,
    *,
    controller_v31_run_state: EvidenceActionControllerV31RunState | None = None,
) -> float:
    if (
        is_evidence_action_controller_v37(action_name)
        and controller_v31_run_state is not None
        and not controller_v31_run_state.dense_overlap
    ):
        return float(base_sigma) * REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER
    return float(base_sigma)


def bipop_relative_improvement(candidate_best: float, incumbent_fitness: float) -> float:
    denominator = max(abs(float(incumbent_fitness)), 1e-12)
    return max(0.0, (float(incumbent_fitness) - float(candidate_best)) / denominator)


def bipop_cooldown_after_restart(
    *,
    restart_accepted: bool,
    sub_num: int,
    rejected_restart_streak: int,
) -> int:
    if bool(restart_accepted):
        return BIPOP_RESTART_COOLDOWN
    sweep_size = max(1, int(sub_num))
    backoff_sweeps = min(
        BIPOP_REJECT_BACKOFF_SWEEP_CAP,
        max(1, int(rejected_restart_streak)),
    )
    return sweep_size * backoff_sweeps


def perturb_bipop_restart_mean(
    base_mean: np.ndarray,
    lower: float,
    upper: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.asarray(base_mean, dtype=float).reshape(-1)
    span = max(float(upper) - float(lower), 1e-12)
    perturbation = rng.normal(0.0, min(float(sigma), span), size=mean.shape)
    return np.clip(mean + perturbation, float(lower), float(upper))


def group_delta_stagnated(delta: float, reference_fitness: float) -> bool:
    threshold = max(BIPOP_STAGNATION_EPSILON, abs(float(reference_fitness)) * 1e-10)
    return abs(float(delta)) <= threshold


def runtime_tail_utility(
    fitness_record: list[float],
    start_index: int,
    population_size: int,
) -> float:
    values = np.asarray(fitness_record[int(start_index):], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    best_so_far = np.minimum.accumulate(values)
    window_size = min(values.size, max(2, 3 * int(population_size)))
    tail = best_so_far[-window_size:]
    return normalized_gain_utility(
        float(tail[0]),
        float(tail[-1]),
        int(window_size - 1),
    )


def iteration_start_budget_remaining_ratio(max_fes: int, sum_fes: int) -> float:
    if max_fes <= 0:
        return 0.0
    return max(0.0, (max_fes - sum_fes) / max_fes)


def derive_optimizer_seed(
    base_seed: int,
    fun_name: str,
    fun_id: int,
    cycle_index: int,
    stage_index: int,
) -> int:
    payload = f"{base_seed}:{fun_name}:{fun_id}:{cycle_index}:{stage_index}".encode(
        "utf-8"
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def blend_overlap_values(
    previous_values: np.ndarray,
    current_values: np.ndarray,
    previous_delta: float,
    current_delta: float,
) -> np.ndarray:
    denominator = previous_delta + current_delta
    if denominator == 0:
        return (previous_values + current_values) / 2
    return (previous_delta / denominator) * previous_values + (
        current_delta / denominator
    ) * current_values


def clipped_consensus_blend(
    previous_values: np.ndarray,
    current_values: np.ndarray,
    previous_delta: float,
    current_delta: float,
) -> np.ndarray:
    denominator = previous_delta + current_delta
    if denominator == 0:
        return (previous_values + current_values) / 2
    current_weight = float(np.clip(current_delta / denominator, 0.35, 0.65))
    previous_weight = 1.0 - current_weight
    return (previous_weight * previous_values) + (current_weight * current_values)


def apply_arac_overlap_action(
    action_name: str,
    previous_values: np.ndarray,
    current_values: np.ndarray,
    previous_delta: float,
    current_delta: float,
) -> np.ndarray:
    if action_name == "repair_shared_variable_binding":
        if current_delta >= previous_delta:
            return current_values
        return previous_values
    if action_name == "isolate_conflicting_relation":
        if previous_delta >= current_delta:
            return previous_values
        return current_values
    if action_name == "allow_beneficial_coordination":
        return clipped_consensus_blend(
            previous_values=previous_values,
            current_values=current_values,
            previous_delta=previous_delta,
            current_delta=current_delta,
        )
    return blend_overlap_values(
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=previous_delta,
        current_delta=current_delta,
    )


def _problem_id(fun_name: str, fun_id: int) -> str:
    return f"{fun_name[0].upper()}{fun_id}"


def _owner_selected(action_name: str, previous_delta: float, current_delta: float) -> str:
    if action_name == PHASE_RESCUE_MULTISTART_ACTION:
        return "trajectory_budget_mean_blend"
    if action_name in REPAIR_ACTION_NAMES:
        if current_delta >= previous_delta:
            return "current"
        return "previous"
    if action_name == "isolate_conflicting_relation":
        if previous_delta >= current_delta:
            return "previous"
        return "current"
    if action_name == "allow_beneficial_coordination":
        return "clipped_consensus_blend"
    if action_name == "conservative_no_action":
        return "weighted_blend"
    return "weighted_blend"


def _semantic_surface(action_name: str) -> str:
    if action_name == PHASE_RESCUE_MULTISTART_ACTION:
        return "optimizer_budget_and_mean_trajectory"
    if action_name in REPAIR_ACTION_NAMES:
        return "shared_variable_owner_rebinding"
    if action_name == "isolate_conflicting_relation":
        return "overlap_value_selection"
    if action_name == "allow_beneficial_coordination":
        return "coordination_clipped_consensus_blend"
    if action_name == "conservative_no_action":
        return "native_overlap_blend"
    return "native_overlap_blend"


def _state_mutated(action_name: str) -> str:
    if action_name == PHASE_RESCUE_MULTISTART_ACTION:
        return "1"
    if action_name in {
        "repair_shared_variable_binding",
        "isolate_conflicting_relation",
        "allow_beneficial_coordination",
        "conservative_no_action",
    }:
        return "1"
    return "0"


def _optimizer_consumed(action_name: str, downstream_consumed: bool = True) -> str:
    if action_name == PHASE_RESCUE_MULTISTART_ACTION:
        return "1"
    if not downstream_consumed:
        return "0"
    if action_name in {
        "repair_shared_variable_binding",
        "isolate_conflicting_relation",
        "allow_beneficial_coordination",
        "conservative_no_action",
    }:
        return "1"
    return "0"


def _action_family_for_canonical(action_name: str) -> str:
    if action_name == PHASE_RESCUE_MULTISTART_ACTION:
        return "trajectory"
    if action_name == "repair_shared_variable_binding":
        return "reassign_repair"
    if action_name == "isolate_conflicting_relation":
        return "isolate"
    if action_name == "allow_beneficial_coordination":
        return "coordinate"
    if action_name == "conservative_no_action":
        return "fallback"
    if action_name == "protect_high_margin_group":
        return "protect"
    return ""


def select_relation_action_for_policy(
    relation: OverlapRelation,
    action: RelationActionDecision,
    relation_policy_mode: str,
) -> RelationActionDecision:
    if relation_policy_mode not in {"adaptive_v24", "adaptive_v26"}:
        raise ValueError(f"unsupported relation policy mode: {relation_policy_mode}")
    return action


def _canonical_relation_action_name(action: RelationActionDecision) -> str:
    if getattr(action, "canonical_action_name", ""):
        return action.canonical_action_name
    return RELATION_ACTION_ALIASES.get(action.action_name, action.action_name)


def guard_relation_action_by_value_delta(
    relation: OverlapRelation,
    action: RelationActionDecision,
    action_value_delta_norm: float,
) -> RelationActionDecision:
    canonical_action_name = _canonical_relation_action_name(action)
    guard_threshold = (
        COORDINATE_ACTION_VALUE_DELTA_GUARD_THRESHOLD
        if canonical_action_name == "allow_beneficial_coordination"
        else ACTION_VALUE_DELTA_GUARD_THRESHOLD
    )
    if (
        canonical_action_name == "conservative_no_action"
        or action_value_delta_norm <= guard_threshold
    ):
        return action
    return RelationActionDecision(
        relation_id=relation.relation_id,
        action_name="fallback",
        action_family="fallback",
        confidence=0.0,
        trigger_reason="action_value_delta_guard_exceeded",
    )


def _shared_vars_hash(shared_vars: tuple[int, ...]) -> str:
    if not shared_vars:
        return ""
    payload = ";".join(str(variable) for variable in shared_vars).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def case_artifact_path(output_path: Path, problem_id: str, artifact_name: str) -> Path:
    artifact = Path(artifact_name)
    return output_path / f"{problem_id}_{artifact.stem}{artifact.suffix}"


def build_action_trace_row(
    problem_id: str,
    seed: int | None,
    outer_iter: int,
    group_index: int,
    selected_action_name: str,
    overlap_size: int,
    previous_delta: float,
    current_delta: float,
    *,
    relation_id: str = "",
    group_left: int | None = None,
    group_right: int | None = None,
    shared_vars: tuple[int, ...] = (),
    action_family: str = "",
    canonical_action_name: str = "",
    relation_policy_source: str = "",
    state_mutated: bool | None = None,
    action_value_delta_norm: float = 0.0,
    downstream_consumed: bool = True,
    downstream_consumption_scope: str = "same_outer_iteration",
    search_state_action_type: str = "",
    search_state_backend: str = "",
    candidate_protected: bool | None = None,
    cc_context_replaced: bool | None = None,
    stagnation_window: int | None = None,
    delta_mean: float | None = None,
    sigma_before: float | None = None,
    sigma_after: float | None = None,
    population_before: int | None = None,
    population_after: int | None = None,
    escape_budget: int | None = None,
    bipop_restart_mode: str = "",
    restart_triggered: bool | None = None,
    restart_accepted: bool | None = None,
    best_before: float | None = None,
    restart_candidate_best: float | None = None,
    restart_relative_improvement: float | None = None,
    restart_acceptance_threshold: float | None = None,
    best_after: float | None = None,
    trace_event: str = "",
    remaining_budget_ratio: float | None = None,
    shared_var_count: int | None = None,
    repair_lock_active: bool | None = None,
    refresh_budget: int | None = None,
    continuation_reserve: int | None = None,
    optimizer_seed: int | None = None,
    scheduler_phase: str = "",
    decision_point: str = "",
    cc_block_fe: int | None = None,
    cc_utility: float | None = None,
    trust_decision: ActionTrustDecision | None = None,
    trust_credit: float | None = None,
    trust_unstable: bool | None = None,
    fallback_route: str = "",
    active_maturity_route: str = "",
    sweep_evidence_relation_count: int | None = None,
    sweep_evidence_active_count: int | None = None,
    sweep_evidence_active_fraction: float | None = None,
    sweep_evidence_support: float | None = None,
    sweep_evidence_reason: str = "",
    phase_rescue_resource_route: str = "",
    phase_rescue_rejected_before_maturity: int | None = None,
    phase_rescue_productive_mature: bool | None = None,
    phase_rescue_retired: bool | None = None,
) -> dict[str, str]:
    canonical_action_name = canonical_action_name or selected_action_name
    action_family = action_family or _action_family_for_canonical(canonical_action_name)
    state_mutated_value = (
        _state_mutated(selected_action_name)
        if state_mutated is None
        else str(int(state_mutated))
    )
    row = {
        "problem_id": problem_id,
        "seed": "" if seed is None else str(seed),
        "outer_iter": str(outer_iter),
        "group_index": str(group_index),
        "selected_action_name": selected_action_name,
        "relation_id": relation_id,
        "group_left": "" if group_left is None else str(group_left),
        "group_right": "" if group_right is None else str(group_right),
        "shared_vars_hash": _shared_vars_hash(shared_vars),
        "action_family": action_family,
        "canonical_action_name": canonical_action_name,
        "relation_policy_source": relation_policy_source,
        "overlap_size": str(overlap_size),
        "previous_delta": f"{previous_delta:.6e}",
        "current_delta": f"{current_delta:.6e}",
        "owner_selected": _owner_selected(
            selected_action_name,
            previous_delta,
            current_delta,
        ),
        "semantic_surface": _semantic_surface(selected_action_name),
        "state_mutated": state_mutated_value,
        "action_value_delta_norm": f"{action_value_delta_norm:.6e}",
        "downstream_consumed": str(int(downstream_consumed)),
        "downstream_consumption_scope": downstream_consumption_scope,
        "optimizer_consumed": _optimizer_consumed(selected_action_name, downstream_consumed),
        "search_state_action_type": search_state_action_type,
        "search_state_backend": search_state_backend,
        "candidate_protected": ""
        if candidate_protected is None
        else str(int(candidate_protected)),
        "cc_context_replaced": ""
        if cc_context_replaced is None
        else str(int(cc_context_replaced)),
        "stagnation_window": "" if stagnation_window is None else str(stagnation_window),
        "delta_mean": "" if delta_mean is None else f"{delta_mean:.6e}",
        "sigma_before": "" if sigma_before is None else f"{sigma_before:.6e}",
        "sigma_after": "" if sigma_after is None else f"{sigma_after:.6e}",
        "population_before": "" if population_before is None else str(population_before),
        "population_after": "" if population_after is None else str(population_after),
        "escape_budget": "" if escape_budget is None else str(escape_budget),
        "bipop_restart_mode": bipop_restart_mode,
        "restart_triggered": "" if restart_triggered is None else str(int(restart_triggered)),
        "restart_accepted": "" if restart_accepted is None else str(int(restart_accepted)),
        "best_before": "" if best_before is None else f"{best_before:.6e}",
        "restart_candidate_best": "" if restart_candidate_best is None else f"{restart_candidate_best:.6e}",
        "restart_relative_improvement": ""
        if restart_relative_improvement is None
        else f"{restart_relative_improvement:.6e}",
        "restart_acceptance_threshold": ""
        if restart_acceptance_threshold is None
        else f"{restart_acceptance_threshold:.6e}",
        "best_after": "" if best_after is None else f"{best_after:.6e}",
        "trace_event": trace_event,
        "remaining_budget_ratio": ""
        if remaining_budget_ratio is None
        else f"{remaining_budget_ratio:.6e}",
        "shared_var_count": ""
        if shared_var_count is None
        else str(shared_var_count),
        "repair_lock_active": ""
        if repair_lock_active is None
        else str(int(repair_lock_active)),
        "refresh_budget": "" if refresh_budget is None else str(refresh_budget),
        "continuation_reserve": ""
        if continuation_reserve is None
        else str(continuation_reserve),
        "optimizer_seed": "" if optimizer_seed is None else str(optimizer_seed),
        "scheduler_phase": scheduler_phase,
        "decision_point": decision_point,
        "cc_block_fe": "" if cc_block_fe is None else str(cc_block_fe),
        "cc_utility": "" if cc_utility is None else f"{cc_utility:.6e}",
    }
    row.update(
        {
            "trust_key": "" if trust_decision is None else trust_decision.key,
            "trust_phase": "" if trust_decision is None else trust_decision.phase,
            "trust_reason": "" if trust_decision is None else trust_decision.reason,
            "trust_score": ""
            if trust_decision is None
            else f"{trust_decision.trust_score:.6e}",
            "trust_exposure": ""
            if trust_decision is None
            else f"{trust_decision.exposure:.6e}",
            "trust_cooldown": ""
            if trust_decision is None
            else str(trust_decision.cooldown_remaining),
            "trust_credit": "" if trust_credit is None else f"{trust_credit:.6e}",
            "trust_unstable": ""
            if trust_unstable is None
            else str(int(trust_unstable)),
            "trust_pre_writeback_fitness": "",
            "trust_post_writeback_fitness": "",
            "fallback_route": fallback_route,
            "active_maturity_route": active_maturity_route,
            "sweep_evidence_relation_count": ""
            if sweep_evidence_relation_count is None
            else str(sweep_evidence_relation_count),
            "sweep_evidence_active_count": ""
            if sweep_evidence_active_count is None
            else str(sweep_evidence_active_count),
            "sweep_evidence_active_fraction": ""
            if sweep_evidence_active_fraction is None
            else f"{sweep_evidence_active_fraction:.6e}",
            "sweep_evidence_support": ""
            if sweep_evidence_support is None
            else f"{sweep_evidence_support:.6e}",
            "sweep_evidence_reason": sweep_evidence_reason,
            "phase_rescue_resource_route": phase_rescue_resource_route,
            "phase_rescue_rejected_before_maturity": ""
            if phase_rescue_rejected_before_maturity is None
            else str(phase_rescue_rejected_before_maturity),
            "phase_rescue_productive_mature": ""
            if phase_rescue_productive_mature is None
            else str(int(phase_rescue_productive_mature)),
            "phase_rescue_retired": ""
            if phase_rescue_retired is None
            else str(int(phase_rescue_retired)),
        }
    )
    return row


def _write_action_trace(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ACTION_TRACE_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_overlap_relation_trace(
    problem_id: str,
    outer_iter: int,
    grouping_result: list[list[int]],
    overlapping_elements: list[list[int]],
    fitness_delta_list: list[float] | None = None,
    budget_remaining_ratio: float = 1.0,
) -> list[OverlapRelation]:
    hcc_trace = {
        "outer_iter": outer_iter,
        "groups": grouping_result,
        "overlapping_elements": overlapping_elements,
        "fitness_deltas": [] if fitness_delta_list is None else fitness_delta_list,
        "group_ranks": []
        if fitness_delta_list is None
        else dense_rank_descending(fitness_delta_list),
        "budget_remaining_ratio": budget_remaining_ratio,
    }
    return build_overlap_relations(hcc_trace, problem_id)


def dense_rank_descending(values: list[float]) -> list[int]:
    rank_by_value = {
        value: rank
        for rank, value in enumerate(sorted(set(values), reverse=True), start=1)
    }
    return [rank_by_value[value] for value in values]


def apply_action_to_relation(
    relation: OverlapRelation,
    action: RelationActionDecision,
    previous_values: np.ndarray | None = None,
    current_values: np.ndarray | None = None,
    previous_delta: float = 0.0,
    current_delta: float = 0.0,
) -> np.ndarray | None:
    if previous_values is None or current_values is None:
        return None
    canonical_action_name = _canonical_relation_action_name(action)
    if canonical_action_name not in {
        "conservative_no_action",
        "allow_beneficial_coordination",
        "isolate_conflicting_relation",
        "repair_shared_variable_binding",
    }:
        raise ValueError(
            f"unknown relation action for {relation.relation_id}: {action.action_name}"
        )
    return apply_arac_overlap_action(
        action_name=canonical_action_name,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=previous_delta,
        current_delta=current_delta,
    )


def apply_and_guard_action_to_relation(
    relation: OverlapRelation,
    action: RelationActionDecision,
    previous_values: np.ndarray | None = None,
    current_values: np.ndarray | None = None,
    previous_delta: float = 0.0,
    current_delta: float = 0.0,
) -> tuple[RelationActionDecision, np.ndarray | None, float]:
    adjusted_values = apply_action_to_relation(
        relation=relation,
        action=action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=previous_delta,
        current_delta=current_delta,
    )
    action_value_delta_norm = (
        0.0
        if adjusted_values is None or current_values is None
        else float(np.linalg.norm(adjusted_values - current_values))
    )
    guarded_action = guard_relation_action_by_value_delta(
        relation,
        action,
        action_value_delta_norm,
    )
    if guarded_action is action:
        return action, adjusted_values, action_value_delta_norm
    adjusted_values = apply_action_to_relation(
        relation=relation,
        action=guarded_action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=previous_delta,
        current_delta=current_delta,
    )
    action_value_delta_norm = (
        0.0
        if adjusted_values is None or current_values is None
        else float(np.linalg.norm(adjusted_values - current_values))
    )
    return guarded_action, adjusted_values, action_value_delta_norm


def apply_relation_action_with_controller_v31(
    relation: OverlapRelation,
    action: RelationActionDecision,
    previous_values: np.ndarray | None = None,
    current_values: np.ndarray | None = None,
    previous_delta: float = 0.0,
    current_delta: float = 0.0,
    controller_v31_run_state: EvidenceActionControllerV31RunState | None = None,
) -> tuple[RelationActionDecision, np.ndarray | None, float]:
    forced_action = (
        None
        if controller_v31_run_state is None
        else controller_v31_run_state.forced_relation_action(relation)
    )
    if forced_action is None:
        executed_action, adjusted_values, action_value_delta_norm = (
            apply_and_guard_action_to_relation(
                relation=relation,
                action=action,
                previous_values=previous_values,
                current_values=current_values,
                previous_delta=previous_delta,
                current_delta=current_delta,
            )
        )
    else:
        executed_action = forced_action
        adjusted_values = apply_action_to_relation(
            relation=relation,
            action=executed_action,
            previous_values=previous_values,
            current_values=current_values,
            previous_delta=previous_delta,
            current_delta=current_delta,
        )
        action_value_delta_norm = (
            0.0
            if adjusted_values is None or current_values is None
            else float(np.linalg.norm(adjusted_values - current_values))
        )
    if controller_v31_run_state is not None:
        controller_v31_run_state.lock_from_large_fallback_writeback(
            relation,
            executed_action,
            action_value_delta_norm,
        )
        controller_v31_run_state.observe_guarded_relation_action(
            relation,
            executed_action,
        )
        newly_forced_action = controller_v31_run_state.forced_relation_action(relation)
        if forced_action is None and newly_forced_action is not None:
            executed_action = newly_forced_action
            adjusted_values = apply_action_to_relation(
                relation=relation,
                action=executed_action,
                previous_values=previous_values,
                current_values=current_values,
                previous_delta=previous_delta,
                current_delta=current_delta,
            )
            action_value_delta_norm = (
                0.0
                if adjusted_values is None or current_values is None
                else float(np.linalg.norm(adjusted_values - current_values))
            )
    return executed_action, adjusted_values, action_value_delta_norm


def apply_v33_guard_to_executed_relation(
    *,
    relation: OverlapRelation,
    executed_action: RelationActionDecision,
    adjusted_values: np.ndarray | None,
    action_value_delta_norm: float,
    current_values: np.ndarray | None,
    controller_run_state: EvidenceActionControllerV31RunState | None,
) -> tuple[
    RelationActionDecision,
    np.ndarray | None,
    float,
    ActionTrustDecision | None,
    str,
]:
    policy = (
        None
        if controller_run_state is None
        else controller_run_state.action_trust_policy
    )
    if policy is None or current_values is None or adjusted_values is None:
        return executed_action, adjusted_values, action_value_delta_norm, None, ""

    canonical_action_name = _canonical_relation_action_name(executed_action)
    adjusted_values, action_value_delta_norm, fallback_route = (
        apply_topology_scoped_fallback_guard(
            executed_action=executed_action,
            adjusted_values=adjusted_values,
            action_value_delta_norm=action_value_delta_norm,
            current_values=current_values,
            controller_run_state=controller_run_state,
        )
    )
    if fallback_route:
        return (
            executed_action,
            adjusted_values,
            action_value_delta_norm,
            None,
            fallback_route,
        )
    if action_value_delta_norm <= ACTION_TRUST_MIN_WRITEBACK_NORM:
        return (
            executed_action,
            np.asarray(current_values, dtype=float).copy(),
            0.0,
            None,
            "",
        )

    trust_key = make_action_key(
        group_left=relation.group_left,
        group_right=relation.group_right,
        shared_vars=relation.shared_vars,
        canonical_action_name=canonical_action_name,
    )
    trust_decision = policy.decide(trust_key)
    if not trust_decision.allow_intervention:
        executed_action = RelationActionDecision(
            relation_id=relation.relation_id,
            action_name="fallback",
            action_family="fallback",
            confidence=0.0,
            trigger_reason=f"controller_v33_{trust_decision.reason}",
        )
        adjusted_values = np.asarray(current_values, dtype=float).copy()
        return executed_action, adjusted_values, 0.0, trust_decision, ""

    guard_threshold = (
        COORDINATE_ACTION_VALUE_DELTA_GUARD_THRESHOLD
        if canonical_action_name == "allow_beneficial_coordination"
        else ACTION_VALUE_DELTA_GUARD_THRESHOLD
    )
    adjusted_values = robust_damped_writeback(
        current_values=np.asarray(current_values, dtype=float),
        proposed_values=np.asarray(adjusted_values, dtype=float),
        blend_strength=trust_decision.blend_strength,
        max_delta_norm=guard_threshold,
    )
    action_value_delta_norm = float(
        np.linalg.norm(adjusted_values - np.asarray(current_values, dtype=float))
    )
    if action_value_delta_norm <= ACTION_TRUST_MIN_WRITEBACK_NORM:
        policy.rollback_decision(trust_decision)
        return (
            executed_action,
            np.asarray(current_values, dtype=float).copy(),
            0.0,
            None,
            "",
        )
    return (
        executed_action,
        adjusted_values,
        action_value_delta_norm,
        trust_decision,
        "",
    )


def apply_topology_scoped_fallback_guard(
    *,
    executed_action: RelationActionDecision,
    adjusted_values: np.ndarray | None,
    action_value_delta_norm: float,
    current_values: np.ndarray | None,
    controller_run_state: EvidenceActionControllerV31RunState | None,
) -> tuple[np.ndarray | None, float, str]:
    fallback_route = controller_v33_fallback_route(
        canonical_action_name=_canonical_relation_action_name(executed_action),
        controller_run_state=controller_run_state,
    )
    if not fallback_route or current_values is None or adjusted_values is None:
        return adjusted_values, float(action_value_delta_norm), ""

    current = np.asarray(current_values, dtype=float)
    if action_value_delta_norm <= ACTION_TRUST_MIN_WRITEBACK_NORM:
        return current.copy(), 0.0, ""
    if fallback_route == "dense_preserve_v31":
        return adjusted_values, float(action_value_delta_norm), fallback_route

    bounded = robust_damped_writeback(
        current_values=current,
        proposed_values=np.asarray(adjusted_values, dtype=float),
        blend_strength=1.0,
        max_delta_norm=ACTION_VALUE_DELTA_GUARD_THRESHOLD,
    )
    return (
        bounded,
        float(np.linalg.norm(bounded - current)),
        fallback_route,
    )


def apply_relation_action_with_controller_v36(
    relation: OverlapRelation,
    action: RelationActionDecision,
    previous_values: np.ndarray | None = None,
    current_values: np.ndarray | None = None,
    previous_delta: float = 0.0,
    current_delta: float = 0.0,
    controller_run_state: EvidenceActionControllerV31RunState | None = None,
) -> tuple[
    RelationActionDecision,
    np.ndarray | None,
    float,
    ActionTrustDecision | None,
    str,
    str,
]:
    """Apply v36 maturity routes, otherwise preserve the v33 guard."""

    if controller_run_state is not None:
        controller_run_state.prepare_v36_outer_iter(relation.outer_iter)
    executed_action, adjusted_values, action_value_delta_norm = (
        apply_relation_action_with_controller_v31(
            relation=relation,
            action=action,
            previous_values=previous_values,
            current_values=current_values,
            previous_delta=previous_delta,
            current_delta=current_delta,
            controller_v31_run_state=controller_run_state,
        )
    )
    if controller_run_state is not None:
        controller_run_state.observe_v36_relation(relation, executed_action)

    canonical_action_name = _canonical_relation_action_name(executed_action)
    if (
        controller_run_state is not None
        and controller_run_state.non_dense_repair_locked
        and canonical_action_name == "repair_shared_variable_binding"
    ):
        return (
            executed_action,
            adjusted_values,
            action_value_delta_norm,
            None,
            "",
            "repair_lock_transparent",
        )
    if (
        controller_run_state is not None
        and controller_run_state.coordinate_maturity_latched
        and canonical_action_name == "allow_beneficial_coordination"
    ):
        return (
            executed_action,
            adjusted_values,
            action_value_delta_norm,
            None,
            "",
            "first_sweep_sparse_coordinate_mature",
        )

    protected = apply_v33_guard_to_executed_relation(
        relation=relation,
        executed_action=executed_action,
        adjusted_values=adjusted_values,
        action_value_delta_norm=action_value_delta_norm,
        current_values=current_values,
        controller_run_state=controller_run_state,
    )
    return (*protected, "")


def _format_shared_vars(shared_vars: tuple[int, ...]) -> str:
    return ";".join(str(variable) for variable in shared_vars)


def _overlap_relation_row(relation: OverlapRelation) -> dict[str, str]:
    raw = asdict(relation)
    row = {
        field_name: str(raw.get(field_name, ""))
        for field_name in OVERLAP_RELATION_FIELDS
    }
    row["shared_vars"] = _format_shared_vars(relation.shared_vars)
    for field_name in (
        "overlap_strength",
        "delta_signal",
        "rank_signal",
        "budget_remaining_ratio",
        "previous_delta",
        "current_delta",
        "delta_abs_gap",
        "delta_signed_gap",
        "delta_ratio_gap",
        "rank_gap",
        "rank_stability",
        "shared_var_support_ratio",
        "feature_coverage",
        "fallback_margin_proxy",
    ):
        row[field_name] = f"{float(raw.get(field_name, 0.0)):.6f}"
    row["both_positive"] = str(int(bool(relation.both_positive)))
    row["one_side_zero"] = str(int(bool(relation.one_side_zero)))
    row["shared_var_count"] = str(relation.shared_var_count)
    return row


def _write_overlap_relation_trace(path: Path, relations: list[OverlapRelation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERLAP_RELATION_FIELDS)
        writer.writeheader()
        writer.writerows(_overlap_relation_row(relation) for relation in relations)


def _action_decision_row(
    run_id: str,
    relation: OverlapRelation,
    action: RelationActionDecision,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "problem_id": relation.problem_id,
        "relation_id": relation.relation_id,
        "group_left": str(relation.group_left),
        "group_right": str(relation.group_right),
        "shared_vars_count": str(len(relation.shared_vars)),
        "overlap_strength": f"{relation.overlap_strength:.6f}",
        "delta_signal": f"{relation.delta_signal:.6f}",
        "rank_signal": f"{relation.rank_signal:.6f}",
        "relation_action_name": action.relation_action_name,
        "canonical_action_name": _canonical_relation_action_name(action),
        "action_family": action.action_family,
        "confidence": f"{action.confidence:.6f}",
        "trigger_reason": action.trigger_reason,
    }


def _write_action_decision_log(
    path: Path,
    run_id: str,
    relations: list[OverlapRelation],
    actions: list[RelationActionDecision],
) -> None:
    if len(relations) != len(actions):
        raise ValueError("relations and actions must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTION_DECISION_FIELDS)
        writer.writeheader()
        for relation, action in zip(relations, actions, strict=True):
            writer.writerow(_action_decision_row(run_id, relation, action))


def _score_relations_with_runtime_prefix_context(
    relations: list[OverlapRelation],
    relation_policy_mode: str = "rule",
):
    scored_actions = []
    context_key: tuple[str, int] | None = None
    context_relations: list[OverlapRelation] = []
    for relation in relations:
        relation_key = (relation.problem_id, relation.outer_iter)
        if relation_key != context_key:
            context_key = relation_key
            context_relations = []
        context_relations.append(relation)
        effective_mode = effective_relation_policy_mode(
            relation_policy_mode,
            context_relations,
        )
        scored_actions.append(_relation_policy_scorer(effective_mode)(context_relations)[-1])
    return scored_actions


def effective_relation_policy_mode(
    relation_policy_mode: str,
    relation_policy_context: list[OverlapRelation],
) -> str:
    if relation_policy_mode == "controller_v3":
        return relation_policy_mode_for_evidence_action_controller_v3(
            relation_policy_context
        )
    if relation_policy_mode == "controller_v31":
        return relation_policy_mode_for_evidence_action_controller_v31(
            relation_policy_context
        )
    return relation_policy_mode


def _relation_policy_scorer(relation_policy_mode: str):
    return (
        score_actions_for_relations_v26
        if relation_policy_mode == "adaptive_v26"
        else score_actions_for_relations_v25
        if relation_policy_mode == "adaptive_v25"
        else score_actions_for_relations_v24
        if relation_policy_mode == "adaptive_v24"
        else score_actions_for_relations_v23
        if relation_policy_mode == "adaptive_v23"
        else score_actions_for_relations_v22
        if relation_policy_mode == "adaptive_v22"
        else score_actions_for_relations_v21
        if relation_policy_mode == "adaptive_v21"
        else score_actions_for_relations_v2
        if relation_policy_mode == "adaptive_v2"
        else score_actions_for_relations
    )


def relation_policy_source_name(
    relation_policy_mode: str,
    effective_mode: str,
    *,
    action: RelationActionDecision | None = None,
) -> str:
    if relation_policy_mode == "controller_v3":
        controller_mode = (
            "relation_first"
            if effective_mode == "adaptive_v24"
            else "search_state_assisted"
        )
        return f"controller_v3:{controller_mode}:{effective_mode}_relation_policy"
    if relation_policy_mode == "controller_v31":
        controller_mode = (
            "relation_first"
            if effective_mode == "adaptive_v24"
            else "search_state_assisted"
        )
        source = f"controller_v31:{controller_mode}:{effective_mode}_relation_policy"
        if action is not None:
            if action.trigger_reason == V31_NON_DENSE_PREFIX_REPAIR_TRIGGER:
                return f"{source}:non_dense_prefix_repair_lock"
            if action.trigger_reason == V31_NON_DENSE_LARGE_FALLBACK_REPAIR_TRIGGER:
                return f"{source}:non_dense_large_fallback_repair_lock"
        return source
    if relation_policy_mode == "shuffled":
        return "deterministic_shuffled_negative_control"
    if relation_policy_mode == "lagged":
        return "deterministic_lagged_relation_policy"
    if relation_policy_mode.startswith("adaptive_"):
        return f"{relation_policy_mode}_relation_policy"
    return "rule_based_relation_policy"


def _write_action_mismatch_audit_log(
    path: Path,
    run_id: str,
    relations: list[OverlapRelation],
    actions: list[RelationActionDecision] | None = None,
    relation_policy_mode: str = "rule",
) -> None:
    if actions is not None and len(relations) != len(actions):
        raise ValueError("relations and actions must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTION_MISMATCH_AUDIT_FIELDS)
        writer.writeheader()
        scored_actions = _score_relations_with_runtime_prefix_context(
            relations,
            relation_policy_mode=relation_policy_mode,
        )
        if actions is None:
            for relation, scored in zip(
                relations,
                scored_actions,
                strict=True,
            ):
                row = action_mismatch_audit_row(relation, scored)
                row["run_id"] = run_id
                writer.writerow(row)
            return
        for relation, scored, action in zip(
            relations,
            scored_actions,
            actions,
            strict=True,
        ):
            row = action_mismatch_audit_row(relation, scored, final_action=action)
            row["run_id"] = run_id
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_raw_action_decision_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTION_DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_raw_action_mismatch_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTION_MISMATCH_AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_budget_summary(
    path: Path,
    *,
    problem_id: str,
    budget_accounting: str,
    max_fes: int,
    optimizer_reported_fe: int,
    fitness_record_fe: int,
    global_phase_fe: int = 0,
    cc_phase_fe: int = 0,
    rescue_fe: int = 0,
    refresh_fe: int = 0,
    search_state_fe: int = 0,
    precision_probe_fe: int = 0,
    evidence_overlay_fe: int | None = None,
    separable_continuation_fe: int = 0,
) -> None:
    budget_aligned_fe = min(max_fes, fitness_record_fe)
    stage_fe = (
        global_phase_fe
        + cc_phase_fe
        + rescue_fe
        + refresh_fe
        + search_state_fe
        + precision_probe_fe
        + (0 if evidence_overlay_fe is None else evidence_overlay_fe)
        + separable_continuation_fe
    )
    overhead_fe = max(0, fitness_record_fe - stage_fe)
    row = {
        "problem_id": problem_id,
        "budget_accounting": budget_accounting,
        "max_fes": str(max_fes),
        "optimizer_reported_fe": str(optimizer_reported_fe),
        "fitness_record_fe": str(fitness_record_fe),
        "budget_aligned_fe": str(budget_aligned_fe),
        "same_budget_violation": str(int(fitness_record_fe > max_fes)),
        "global_phase_fe": str(global_phase_fe),
        "cc_phase_fe": str(cc_phase_fe),
        "rescue_fe": str(rescue_fe),
        "refresh_fe": str(refresh_fe),
        "search_state_fe": str(search_state_fe),
        "separable_continuation_fe": str(separable_continuation_fe),
        "overhead_fe": str(overhead_fe),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(BUDGET_SUMMARY_FIELDS)
        if precision_probe_fe > 0:
            fieldnames.append("precision_probe_fe")
            row["precision_probe_fe"] = str(precision_probe_fe)
        if evidence_overlay_fe is not None:
            fieldnames.append("evidence_overlay_fe")
            row["evidence_overlay_fe"] = str(evidence_overlay_fe)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def build_overlap_relation_for_pair(
    problem_id: str,
    outer_iter: int,
    grouping_result: list[list[int]],
    overlapping_elements: list[list[int]],
    fitness_delta_list: list[float],
    group_right: int,
    budget_remaining_ratio: float,
) -> OverlapRelation:
    relations = build_overlap_relation_trace(
        problem_id=problem_id,
        outer_iter=outer_iter,
        grouping_result=grouping_result,
        overlapping_elements=overlapping_elements,
        fitness_delta_list=fitness_delta_list,
        budget_remaining_ratio=budget_remaining_ratio,
    )
    group_left = group_right - 1
    for relation in relations:
        if relation.group_left == group_left and relation.group_right == group_right:
            return relation
    raise ValueError(f"missing overlap relation for groups {group_left}-{group_right}")


def _canonical_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_overlay_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return {
            "callable": (
                f"{getattr(value, '__module__', '')}."
                f"{getattr(value, '__qualname__', type(value).__name__)}"
            )
        }
    if hasattr(type(value), "__dataclass_fields__"):
        return _evidence_overlay_jsonable(asdict(value))
    if isinstance(value, dict):
        return {
            repr(key): _evidence_overlay_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_evidence_overlay_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            _evidence_overlay_jsonable(item)
            for item in sorted(value, key=repr)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"unsupported evidence-overlay fingerprint value: {type(value).__name__}"
    )


def _evidence_overlay_optimizer_payload(optimizer: object | None) -> object:
    if optimizer is None:
        return None
    volatile = {
        "rng",
        "rng_initialization",
        "rng_optimization",
        "start_time",
        "runtime",
        "time_function_evaluations",
        "fitness_function",
        "Terminations",
    }
    payload: dict[str, object] = {}
    for name, value in sorted(vars(optimizer).items()):
        if name in volatile:
            continue
        payload[name] = _evidence_overlay_jsonable(value)
    payload["rng_states"] = {
        name: _evidence_overlay_jsonable(
            getattr(optimizer, name).bit_generator.state
        )
        for name in ("rng", "rng_initialization", "rng_optimization")
        if hasattr(getattr(optimizer, name, None), "bit_generator")
    }
    return payload


def _evidence_overlay_controller_payload(
    controller: EvidenceActionControllerV31RunState | None,
) -> object:
    if controller is None:
        return None
    policy = controller.action_trust_policy
    policy_payload = None
    if policy is not None:
        policy_payload = {
            "config": asdict(policy.config),
            "states": {
                key: asdict(state)
                for key, state in sorted(policy._states.items())
            },
        }

    payload: dict[str, object] = {}
    for name in controller.__dataclass_fields__:
        value = getattr(controller, name)
        if name == "action_trust_policy":
            payload[name] = policy_payload
        elif name == "phase_i_optimizer":
            payload[name] = _evidence_overlay_optimizer_payload(value)
        elif name == "phase_i_state":
            payload[name] = None if value is None else value.fingerprint()
        else:
            payload[name] = _evidence_overlay_jsonable(value)
    payload.update(
        {
            "trajectory_guard_enabled": False,
            "pending_trajectory_recovery": None,
            "search_state_scheduler_state": {
                "'phase'": "initial_probe",
                "'probe_utilities'": [],
                "'intervention_fe'": 0,
            },
            "diagonal_cma_state": None,
            "cc_utility_history": [],
            "v38_enabled": False,
            "v39_enabled": False,
            "_v39_cma_sigma_factors": {},
        }
    )
    return payload


def _evidence_overlay_phase_i_payload(
    controller: EvidenceActionControllerV31RunState | None,
) -> object:
    if controller is None:
        return None
    optimizer = controller.phase_i_optimizer
    state = controller.phase_i_state
    return {
        "optimizer": _evidence_overlay_optimizer_payload(optimizer),
        "state_fingerprint": None if state is None else state.fingerprint(),
    }


def evidence_overlay_runtime_fingerprints(
    *,
    best_individual: np.ndarray,
    guarded_incumbent: np.ndarray,
    guarded_incumbent_fitness: float,
    grouping_result: list[list[int]],
    controller: EvidenceActionControllerV31RunState | None,
    trajectory_mean_cache: dict[int, float],
    previous_group_contribution_credit: list[float],
) -> dict[str, str]:
    phase_i_payload = _evidence_overlay_phase_i_payload(controller)
    optimizer = None if controller is None else controller.phase_i_optimizer
    components = {
        "best_individual": _evidence_overlay_jsonable(best_individual),
        "guarded_incumbent": _evidence_overlay_jsonable(guarded_incumbent),
        "guarded_incumbent_fitness": _evidence_overlay_jsonable(
            float(guarded_incumbent_fitness)
        ),
        "grouping": _evidence_overlay_jsonable(grouping_result),
        "phase_i": phase_i_payload,
        "controller": {
            "state": _evidence_overlay_controller_payload(controller),
            "trajectory_mean_cache": _evidence_overlay_jsonable(
                trajectory_mean_cache
            ),
            "previous_group_contribution_credit": _evidence_overlay_jsonable(
                previous_group_contribution_credit
            ),
        },
        "rng": {
            "python": _evidence_overlay_jsonable(random.getstate()),
            "numpy": _evidence_overlay_jsonable(np.random.get_state()),
            "phase_i_optimizer": (
                None
                if optimizer is None
                else _evidence_overlay_optimizer_payload(optimizer)["rng_states"]
            ),
        },
    }
    return {
        name: _canonical_payload_sha256(_evidence_overlay_jsonable(payload))
        for name, payload in components.items()
    }




def run_problem(fun_name: str, fun_id: int, output_path: Path, config: SmokeConfig) -> tuple[list[float], float, list[dict[str, str]]]:
    if config.budget_accounting not in {"strict", "source"}:
        raise ValueError(f"unsupported budget accounting mode: {config.budget_accounting}")
    if config.evidence_overlay_mode not in EVIDENCE_OVERLAY_MODES:
        raise ValueError("unsupported evidence overlay mode")
    if not is_evidence_action_controller_v37(config.arac_action):
        raise ValueError("exp_018 runner requires frozen v37")
    evidence_overlay_enabled = config.evidence_overlay_mode != "off"
    if evidence_overlay_enabled:
        if not is_evidence_action_controller_v37(config.arac_action):
            raise ValueError("evidence overlay requires frozen v37")
        if not config.enable_relation_dispatch:
            raise ValueError("evidence overlay requires relation dispatch")
        if config.relation_policy_mode != "controller_v31":
            raise ValueError("evidence overlay requires controller_v31")
        if config.seed is None:
            raise ValueError("evidence overlay requires an explicit seed")
        if config.budget_accounting != "strict":
            raise ValueError("evidence overlay requires strict FE accounting")
        if not config.cmaes_restart or not config.mmes_restart:
            raise ValueError("evidence overlay requires frozen restart settings")
        if config.search_state_backend != "phase_i_mmes":
            raise ValueError("evidence overlay requires phase_i_mmes")
    time_start = time.time()
    bench = Benchmark(str(output_path) + "/", data_dir=config.aob_data_root)
    fun = bench.get_function(fun_name, fun_id)
    info = bench.get_info(fun_name, fun_id)
    problem_id = _problem_id(fun_name, fun_id)
    grouping_result = load_runtime_grouping(
        fun_id,
        config.aob_data_root,
        evidence_overlay_mode=config.evidence_overlay_mode,
    )
    terminal_completion_tolerance_fe = max(
        1,
        max(
            calculate_cmaes_population_size(len(group))
            for group in grouping_result
        ),
    )
    evidence_overlay_observer = (
        HccEvidenceOverlayObserver(
            mode=config.evidence_overlay_mode,
            grouping_result=grouping_result,
            problem_id=problem_id,
            seed=int(config.seed),
            run_id=config.run_id,
            configured_max_fes=config.max_fes,
            terminal_tolerance_fe=terminal_completion_tolerance_fe,
            lower_bound=float(info["lower"]),
            upper_bound=float(info["upper"]),
            fresh_optimizer_execution=True,
        )
        if evidence_overlay_enabled
        else None
    )
    _, overlap_groups, overlapping_elements = remove_overlapping_groups(grouping_result)
    degree = calculate_runtime_overlap_degree(
        overlap_groups,
        problem_dimension=int(info["dimension"]),
        fun_id=fun_id,
        data_root=config.aob_data_root,
        evidence_overlay_mode=config.evidence_overlay_mode,
    )
    global_fes = calculate_global_fes(config.max_fes, degree)
    controller_v31_run_state = build_evidence_action_controller_v31_run_state(
        degree,
        action_name=config.arac_action,
    )
    best_individual = np.zeros(info["dimension"])
    trajectory_mean_cache: dict[int, float] = {}
    sum_fes = 0
    global_phase_fe = 0
    cc_phase_fe = 0
    rescue_fe = 0
    refresh_fe = 0
    search_state_fe = 0
    action_trace_rows: list[dict[str, str]] = []
    relations: list[OverlapRelation] = []
    action_decisions: list[RelationActionDecision] = []
    previous_group_contribution_credit: list[float] = []
    evidence_overlay_fe = 0
    evidence_overlay_barrier_attempted = False
    evidence_overlay_frozen_sub_fes: int | None = None
    evidence_overlay_probe_slice: tuple[int, int] | None = None
    evidence_overlay_runtime_failure: BaseException | None = None
    evidence_overlay_has_overlap = bool(any(overlapping_elements))
    group_stagnation_counts = [0 for _ in grouping_result]
    bipop_global_cooldown = 0
    bipop_restart_count = 0
    bipop_rejected_restart_streak = 0
    guarded_incumbent = best_individual.copy()
    guarded_incumbent_fitness = math.inf
    phase_rescue_enabled = (
        controller_v31_run_state.phase_rescue_enabled
        if controller_v31_run_state is not None
        else False
    )
    if global_fes != 0:
        problem = {
            "fitness_function": fun,
            "ndim_problem": info["dimension"],
            "lower_boundary": info["lower"] * np.ones((info["dimension"],)),
            "upper_boundary": info["upper"] * np.ones((info["dimension"],)),
        }
        options = {
            "max_function_evaluations": global_fes,
            "mean": (best_individual,),
            "sigma": config.sigma,
            "is_restart": config.mmes_restart,
            "verbose": config.verbose,
        }
        if config.seed is not None:
            options["seed_rng"] = derive_optimizer_seed(config.seed, fun_name, fun_id, 0, 0)
        phase_i_evaluations_before = current_fitness_evaluations(fun)
        results = MMES(problem, options).optimize()
        best_individual = results["best_so_far_x"].copy()
        guarded_incumbent = best_individual.copy()
        guarded_incumbent_fitness = float(results["best_so_far_y"])
        global_phase_fe = observed_optimizer_fe(
            fun,
            evaluations_before=phase_i_evaluations_before,
            optimizer_reported_fe=results["n_function_evaluations"],
        )
        sum_fes += global_phase_fe
        if controller_v31_run_state is not None:
            controller_v31_run_state.phase_i_runtime_tail_utility = (
                runtime_tail_utility(
                    fun.fitness_record,
                    phase_i_evaluations_before,
                    calculate_cmaes_population_size(int(info["dimension"])),
                )
            )
    outer_iter = 0
    while (
        sum_fes if config.budget_accounting == "source" else current_fitness_evaluations(fun)
    ) < config.max_fes:
        current_fes = (
            sum_fes if config.budget_accounting == "source" else current_fitness_evaluations(fun)
        )
        iteration_budget_remaining_ratio = iteration_start_budget_remaining_ratio(
            max_fes=config.max_fes,
            sum_fes=current_fes,
        )
        sub_num = len(grouping_result)
        cc_budget_limit_fes = config.max_fes
        population_sizes = [
            calculate_cmaes_population_size(len(dims)) for dims in grouping_result
        ]
        try:
            overlay_sub_fes = evidence_overlay_scheduled_sub_fes(
                mode=config.evidence_overlay_mode,
                has_overlap=evidence_overlay_has_overlap,
                complete_sweep_count=(
                    0
                    if evidence_overlay_observer is None
                    else evidence_overlay_observer.consecutive_complete_sweep_count
                ),
                cc_budget_limit_fe=cc_budget_limit_fes,
                current_fe=current_fes,
                terminal_tolerance_fe=terminal_completion_tolerance_fe,
                sub_num=sub_num,
                population_sizes=population_sizes,
                frozen_sub_fes=evidence_overlay_frozen_sub_fes,
                plan_ready=(
                    evidence_overlay_observer is None
                    or evidence_overlay_observer.plan_ready
                ),
                probe_pending=not evidence_overlay_barrier_attempted,
                barrier_attempted=evidence_overlay_barrier_attempted,
                delayed_outcomes_pending=(
                    False
                    if evidence_overlay_observer is None
                    else evidence_overlay_observer.delayed_outcomes_pending
                ),
            )
        except BaseException as error:
            if evidence_overlay_observer is None:
                raise
            evidence_overlay_runtime_failure = error
            break
        if overlay_sub_fes is not None:
            if evidence_overlay_frozen_sub_fes is None:
                evidence_overlay_frozen_sub_fes = overlay_sub_fes
            sub_fes = overlay_sub_fes
        else:
            sub_fes = math.ceil(max(0, cc_budget_limit_fes - current_fes) / sub_num)
        fitness_delta_list: list[float] = []
        current_outer_relations: list[OverlapRelation] = []
        optimized_any_group = False
        outer_stagnation_streak = 0
        for index, dims in enumerate(grouping_result):
            population_size = population_sizes[index]
            if (
                config.budget_accounting == "strict"
                and cc_budget_limit_fes - current_fitness_evaluations(fun)
                <= population_size
            ):
                break
            group_interval_start_fe = current_fitness_evaluations(fun)
            evidence_overlay_pre_block_candidate = (
                best_individual.copy()
                if evidence_overlay_observer is not None
                else None
            )
            original_best = best_individual.copy()
            original_fitness = float(fun(best_individual)[0])
            evidence_overlay_pre_error = original_fitness
            if controller_v31_run_state is not None:
                controller_v31_run_state.observe_pending_action_trust(
                    post_writeback_fitness=original_fitness,
                )
            if config.budget_accounting == "source":
                optimizer_budget = sub_fes
            else:
                requested_fes = max(sub_fes, population_size)
                optimizer_budget = bounded_population_budget(
                    requested_fes=requested_fes,
                    remaining_fes=(
                        cc_budget_limit_fes - current_fitness_evaluations(fun)
                    ),
                    population_size=population_size,
                )
            if optimizer_budget <= 0:
                break
            cc_mean = np.asarray(best_individual[dims], dtype=float).copy()
            primary_evaluations_before = current_fitness_evaluations(fun)
            cc_sigma = refine_sigma_for_action(
                config.arac_action,
                config.sigma,
                controller_v31_run_state=controller_v31_run_state,
            )
            def objective_function(x_batch, dims=dims):
                return fun(combine(x_batch, best_individual, dims))

            problem_cc = {
                "fitness_function": objective_function,
                "ndim_problem": len(dims),
                "lower_boundary": info["lower"] * np.ones((len(dims),)),
                "upper_boundary": info["upper"] * np.ones((len(dims),)),
            }
            options_cc = {
                "max_function_evaluations": optimizer_budget,
                "mean": (cc_mean,),
                "sigma": cc_sigma,
                "n_individuals": population_size,
                "is_restart": config.cmaes_restart,
                "verbose": config.verbose,
                "early_stopping_evaluations": config.early_stopping_evaluations,
            }
            if config.seed is not None:
                stage_index = outer_iter * sub_num + index + 1
                options_cc["seed_rng"] = derive_optimizer_seed(
                    config.seed,
                    fun_name,
                    fun_id,
                    0,
                    stage_index,
                )
            cc_optimizer = CMAES(problem_cc, options_cc)
            results_cc = cc_optimizer.optimize()
            optimized_any_group = True
            primary_cc_fe = observed_optimizer_fe(
                fun,
                evaluations_before=primary_evaluations_before,
                optimizer_reported_fe=results_cc["n_function_evaluations"],
            )
            cc_phase_fe += primary_cc_fe
            sum_fes += primary_cc_fe
            new_best_y = float(results_cc["best_so_far_y"])
            if new_best_y < original_fitness:
                best_individual[dims] = results_cc["best_so_far_x"].copy()
                current_delta = original_fitness - new_best_y
                if (
                    controller_v31_run_state is not None
                    and new_best_y < guarded_incumbent_fitness
                ):
                    guarded_incumbent = best_individual.copy()
                    guarded_incumbent_fitness = new_best_y
            else:
                current_delta = 0.0
            if uses_phase_rescue_during_run(
                config.arac_action,
                phase_rescue_enabled=phase_rescue_enabled,
            ):
                if bipop_global_cooldown > 0:
                    bipop_global_cooldown -= 1
                if group_delta_stagnated(current_delta, original_fitness):
                    group_stagnation_counts[index] += 1
                    outer_stagnation_streak += 1
                else:
                    group_stagnation_counts[index] = 0
                    outer_stagnation_streak = 0
                rescue_budget_limit = (
                    cc_budget_limit_fes
                    if evidence_overlay_observer is not None
                    else config.max_fes
                )
                remaining_fes = max(
                    0,
                    rescue_budget_limit - current_fitness_evaluations(fun),
                )
                rescue_sigma = float(config.sigma) * PHASE_RESCUE_SIGMA_MULTIPLIER
                rescue_population_size = population_size
                requested_escape_budget = int(
                    max(
                        rescue_population_size * PHASE_RESCUE_START_COUNT,
                        math.ceil(optimizer_budget * PHASE_RESCUE_ESCAPE_BUDGET_FRACTION),
                    )
                )
                total_escape_budget = bounded_population_budget(
                    requested_fes=requested_escape_budget,
                    remaining_fes=remaining_fes,
                    population_size=rescue_population_size,
                )
                candidate_budget = bounded_population_budget(
                    requested_fes=max(
                        rescue_population_size,
                        total_escape_budget // PHASE_RESCUE_START_COUNT,
                    ),
                    remaining_fes=total_escape_budget,
                    population_size=rescue_population_size,
                )
                start_count = (
                    0
                    if candidate_budget <= 0
                    else min(
                        PHASE_RESCUE_START_COUNT,
                        total_escape_budget // candidate_budget,
                    )
                )
                if (
                    max(group_stagnation_counts[index], outer_stagnation_streak)
                    >= PHASE_RESCUE_STAGNATION_WINDOW
                    and bipop_global_cooldown <= 0
                    and start_count > 0
                    and (
                        controller_v31_run_state is None
                        or controller_v31_run_state.phase_rescue_enabled
                    )
                ):
                    stagnation_window_for_trace = max(
                        group_stagnation_counts[index],
                        outer_stagnation_streak,
                    )
                    primary_delta_for_trace = current_delta
                    post_primary_fitness = original_fitness - current_delta
                    rescue_rng = np.random.default_rng(
                        derive_optimizer_seed(
                            config.seed if config.seed is not None else 0,
                            fun_name,
                            fun_id,
                            outer_iter + 1,
                            (index + 1) * 12011 + bipop_restart_count,
                        )
                    )
                    best_candidate_y = math.inf
                    best_candidate_x: np.ndarray | None = None
                    total_rescue_fes = 0
                    mean_shift_norms: list[float] = []
                    for candidate_index in range(start_count):
                        rescue_mean = perturb_bipop_restart_mean(
                            base_mean=np.asarray(best_individual[dims], dtype=float),
                            lower=info["lower"],
                            upper=info["upper"],
                            sigma=rescue_sigma,
                            rng=rescue_rng,
                        )
                        mean_shift_norms.append(float(np.linalg.norm(rescue_mean - cc_mean)))
                        rescue_options = {
                            "max_function_evaluations": candidate_budget,
                            "mean": (rescue_mean,),
                            "sigma": rescue_sigma,
                            "n_individuals": rescue_population_size,
                            "is_restart": config.cmaes_restart,
                            "verbose": config.verbose,
                            "early_stopping_evaluations": config.early_stopping_evaluations,
                            "arac_search_state_action": PHASE_RESCUE_MULTISTART_ACTION,
                            "arac_phase_rescue_candidate": candidate_index,
                            "arac_bipop_restart_mode": f"phase_rescue_{start_count}_start",
                        }
                        if config.seed is not None:
                            rescue_options["seed_rng"] = derive_optimizer_seed(
                                config.seed,
                                fun_name,
                                fun_id,
                                outer_iter + 1,
                                (index + 1) * 17011 + candidate_index,
                            )
                        rescue_evaluations_before = current_fitness_evaluations(fun)
                        rescue_optimizer = CMAES(problem_cc, rescue_options)
                        rescue_results = rescue_optimizer.optimize()
                        total_rescue_fes += observed_optimizer_fe(
                            fun,
                            evaluations_before=rescue_evaluations_before,
                            optimizer_reported_fe=rescue_results[
                                "n_function_evaluations"
                            ],
                        )
                        rescue_best = float(rescue_results["best_so_far_y"])
                        if rescue_best < best_candidate_y:
                            best_candidate_y = rescue_best
                            best_candidate_x = np.asarray(
                                rescue_results["best_so_far_x"],
                                dtype=float,
                            ).copy()
                    bipop_restart_count += start_count
                    rescue_fe += total_rescue_fes
                    sum_fes += total_rescue_fes
                    rescue_relative_improvement = bipop_relative_improvement(
                        candidate_best=best_candidate_y,
                        incumbent_fitness=post_primary_fitness,
                    )
                    rescue_accepted = (
                        best_candidate_x is not None
                        and best_candidate_y < post_primary_fitness
                    )
                    phase_rescue_resource_route = (
                        controller_v31_run_state.observe_v37_phase_rescue(
                            accepted=rescue_accepted
                        )
                        if controller_v31_run_state is not None
                        else ""
                    )
                    if rescue_accepted:
                        best_individual[dims] = best_candidate_x.copy()
                        current_delta = original_fitness - best_candidate_y
                        group_stagnation_counts[index] = 0
                        outer_stagnation_streak = 0
                        bipop_rejected_restart_streak = 0
                    else:
                        bipop_rejected_restart_streak += 1
                    bipop_global_cooldown = bipop_cooldown_after_restart(
                        restart_accepted=rescue_accepted,
                        sub_num=sub_num,
                        rejected_restart_streak=bipop_rejected_restart_streak,
                    )
                    action_trace_rows.append(
                        build_action_trace_row(
                            problem_id=_problem_id(fun_name, fun_id),
                            seed=config.seed,
                            outer_iter=outer_iter,
                            group_index=index,
                            selected_action_name=PHASE_RESCUE_MULTISTART_ACTION,
                            overlap_size=0,
                            previous_delta=primary_delta_for_trace,
                            current_delta=0.0 if not rescue_accepted else current_delta,
                            state_mutated=rescue_accepted,
                            action_value_delta_norm=max(mean_shift_norms) if mean_shift_norms else 0.0,
                            downstream_consumed=index < sub_num - 1,
                            search_state_action_type=PHASE_RESCUE_MULTISTART_ACTION,
                            stagnation_window=stagnation_window_for_trace,
                            delta_mean=max(mean_shift_norms) if mean_shift_norms else 0.0,
                            sigma_before=config.sigma,
                            sigma_after=rescue_sigma,
                            population_before=population_size,
                            population_after=rescue_population_size,
                            escape_budget=total_rescue_fes,
                            bipop_restart_mode=f"phase_rescue_{start_count}_start",
                            restart_triggered=True,
                            restart_accepted=rescue_accepted,
                            best_before=post_primary_fitness,
                            restart_candidate_best=best_candidate_y,
                            restart_relative_improvement=rescue_relative_improvement,
                            restart_acceptance_threshold=0.0,
                            best_after=best_candidate_y if rescue_accepted else post_primary_fitness,
                            phase_rescue_resource_route=phase_rescue_resource_route,
                            phase_rescue_rejected_before_maturity=(
                                controller_v31_run_state.phase_rescue_rejected_before_maturity
                                if controller_v31_run_state is not None
                                and controller_v31_run_state.v37_enabled
                                else None
                            ),
                            phase_rescue_productive_mature=(
                                controller_v31_run_state.phase_rescue_productive_mature
                                if controller_v31_run_state is not None
                                and controller_v31_run_state.v37_enabled
                                else None
                            ),
                            phase_rescue_retired=(
                                controller_v31_run_state.phase_rescue_retired
                                if controller_v31_run_state is not None
                                and controller_v31_run_state.v37_enabled
                                else None
                            ),
                        )
                    )
            if evidence_overlay_observer is not None:
                group_interval_end_fe = current_fitness_evaluations(fun)
                if (
                    evidence_overlay_pre_block_candidate is None
                    or group_interval_end_fe <= group_interval_start_fe
                ):
                    raise RuntimeError(
                        "evidence overlay group interval is incomplete"
                    )
                if primary_evaluations_before != group_interval_start_fe + 1:
                    raise RuntimeError(
                        "evidence overlay must reuse the native precheck FE"
                    )
                interval_reserve = evidence_overlay_group_interval_reserve(
                    population_size,
                    optimizer_budget,
                )
                if group_interval_end_fe - group_interval_start_fe > interval_reserve:
                    raise RuntimeError(
                        "evidence overlay native group interval exceeded its frozen reserve"
                    )
                group_best_error = min(
                    float(fun.fitness_record[fe_index])
                    for fe_index in range(
                        group_interval_start_fe,
                        group_interval_end_fe,
                    )
                )
                evidence_overlay_observer.record_group(
                    sweep_index=outer_iter,
                    group_index=index,
                    pre_error=evidence_overlay_pre_error,
                    best_error=min(evidence_overlay_pre_error, group_best_error),
                    primary_requested_fe=optimizer_budget,
                    primary_actual_fe=primary_cc_fe,
                    full_interval_actual_fe=(
                        group_interval_end_fe - group_interval_start_fe
                    ),
                    full_interval_start_fe=group_interval_start_fe,
                    full_interval_end_fe=group_interval_end_fe,
                    pre_block_candidate=evidence_overlay_pre_block_candidate,
                    final_owner_candidate=best_individual.copy(),
                )
            fitness_delta_list.append(current_delta)
            if index > 0:
                overlap_indices = overlapping_elements[index - 1]
                if config.enable_relation_dispatch:
                    if config.relation_policy_mode != "controller_v31":
                        raise ValueError("v37 requires controller_v31")
                    context = RelationExecutionContext(
                        overlap_indices=list(overlap_indices),
                        previous_values=original_best[overlap_indices].copy(),
                        current_values=best_individual[overlap_indices].copy(),
                        previous_delta=fitness_delta_list[index - 1],
                        current_delta=current_delta,
                    )
                    relation = build_overlap_relation_for_pair(
                        problem_id=_problem_id(fun_name, fun_id),
                        outer_iter=outer_iter,
                        grouping_result=grouping_result,
                        overlapping_elements=overlapping_elements,
                        fitness_delta_list=fitness_delta_list,
                        group_right=index,
                        budget_remaining_ratio=iteration_budget_remaining_ratio,
                    )
                    relation_policy_context = current_outer_relations + [relation]
                    controller_v31_run_state.lock_from_runtime_prefix(
                        relation_policy_context
                    )
                    effective_policy_mode = (
                        controller_v31_run_state.effective_policy_mode
                    )
                    phase_rescue_enabled = (
                        controller_v31_run_state.phase_rescue_enabled
                    )
                    if effective_policy_mode == "adaptive_v26":
                        rule_action = decide_actions_for_relations_v26(
                            relation_policy_context
                        )[-1]
                    elif effective_policy_mode == "adaptive_v24":
                        rule_action = decide_actions_for_relations_v24(
                            relation_policy_context
                        )[-1]
                    else:
                        raise RuntimeError(
                            f"unsupported v37 policy mode: {effective_policy_mode}"
                        )
                    action = select_relation_action_for_policy(
                        relation=relation,
                        action=rule_action,
                        relation_policy_mode=effective_policy_mode,
                    )
                    trust_decision: ActionTrustDecision | None = None
                    fallback_route = ""
                    active_maturity_route = ""
                    (
                        action,
                        adjusted_values,
                        action_value_delta_norm,
                        trust_decision,
                        fallback_route,
                        active_maturity_route,
                    ) = apply_relation_action_with_controller_v36(
                        relation=relation,
                        action=action,
                        previous_values=context.previous_values,
                        current_values=context.current_values,
                        previous_delta=context.previous_delta,
                        current_delta=context.current_delta,
                        controller_run_state=controller_v31_run_state,
                    )
                    if adjusted_values is not None:
                        best_individual[context.overlap_indices] = adjusted_values
                    relative_writeback_norm = scale_free_writeback_norm(
                        delta_norm=action_value_delta_norm,
                        shared_count=len(context.overlap_indices),
                        lower=float(info["lower"]),
                        upper=float(info["upper"]),
                    )
                    trust_unstable = (
                        relative_writeback_norm
                        >= RELATIVE_WRITEBACK_UNSTABLE_THRESHOLD
                    )
                    canonical_action_name = _canonical_relation_action_name(action)
                    current_outer_relations.append(relation)
                    relations.append(relation)
                    action_decisions.append(action)
                    trust_writeback_active = (
                        action_value_delta_norm > ACTION_TRUST_MIN_WRITEBACK_NORM
                    )
                    action_trace_row = build_action_trace_row(
                            problem_id=_problem_id(fun_name, fun_id),
                            seed=config.seed,
                            outer_iter=outer_iter,
                            group_index=relation.group_right,
                            selected_action_name=canonical_action_name,
                            overlap_size=len(relation.shared_vars),
                            previous_delta=context.previous_delta,
                            current_delta=context.current_delta,
                            relation_id=relation.relation_id,
                            group_left=relation.group_left,
                            group_right=relation.group_right,
                            shared_vars=relation.shared_vars,
                            action_family=action.action_family,
                            canonical_action_name=canonical_action_name,
                            relation_policy_source=relation_policy_source_name(
                                config.relation_policy_mode,
                                effective_policy_mode,
                                action=action,
                            ),
                            state_mutated=(
                                adjusted_values is not None
                                and trust_writeback_active
                            )
                            if uses_v33_trust_trace_schema(config.arac_action)
                            else adjusted_values is not None,
                            action_value_delta_norm=action_value_delta_norm,
                            downstream_consumed=(
                                index < sub_num - 1 and trust_writeback_active
                                if uses_v33_trust_trace_schema(config.arac_action)
                                else index < sub_num - 1
                            ),
                            downstream_consumption_scope=(
                                relation_downstream_consumption_scope(
                                    action_name=config.arac_action,
                                    writeback_active=trust_writeback_active,
                                )
                            ),
                            trust_decision=trust_decision,
                            trust_unstable=(
                                trust_unstable
                                if trust_decision is not None
                                else None
                            ),
                            fallback_route=fallback_route,
                            active_maturity_route=active_maturity_route,
                            sweep_evidence_relation_count=(
                                controller_v31_run_state.sweep_evidence_relation_count
                                if controller_v31_run_state is not None
                                and controller_v31_run_state.v36_enabled
                                else None
                            ),
                            sweep_evidence_active_count=(
                                controller_v31_run_state.sweep_evidence_active_count
                                if controller_v31_run_state is not None
                                and controller_v31_run_state.v36_enabled
                                else None
                            ),
                            sweep_evidence_active_fraction=(
                                controller_v31_run_state.sweep_evidence_active_fraction
                                if controller_v31_run_state is not None
                                and controller_v31_run_state.v36_enabled
                                else None
                            ),
                            sweep_evidence_support=(
                                controller_v31_run_state.sweep_evidence_support
                                if controller_v31_run_state is not None
                                and controller_v31_run_state.v36_enabled
                                else None
                            ),
                            sweep_evidence_reason=(
                                controller_v31_run_state.sweep_evidence_reason
                                if controller_v31_run_state is not None
                                and controller_v31_run_state.v36_enabled
                                else ""
                            ),
                    )
                    action_trace_rows.append(action_trace_row)
                    if controller_v31_run_state is not None:
                        controller_v31_run_state.register_pending_action_trust(
                            decision=trust_decision,
                            pre_writeback_fitness=original_fitness - current_delta,
                            unstable=trust_unstable,
                            trace_row=action_trace_row,
                        )
                else:
                    overlap_action_name = overlap_action_name_for_lane(config.arac_action)
                    current_overlap_values = best_individual[overlap_indices].copy()
                    adjusted_values = apply_arac_overlap_action(
                        action_name=overlap_action_name,
                        previous_values=original_best[overlap_indices],
                        current_values=current_overlap_values,
                        previous_delta=fitness_delta_list[index - 1],
                        current_delta=current_delta,
                    )
                    best_individual[overlap_indices] = adjusted_values
                    overlap_writeback_norm = float(
                        np.linalg.norm(adjusted_values - current_overlap_values)
                    )
                    action_trace_rows.append(
                        build_action_trace_row(
                            problem_id=_problem_id(fun_name, fun_id),
                            seed=config.seed,
                            outer_iter=outer_iter,
                            group_index=index,
                            selected_action_name=overlap_action_name,
                            overlap_size=len(overlap_indices),
                            previous_delta=fitness_delta_list[index - 1],
                            current_delta=current_delta,
                            state_mutated=True,
                            action_value_delta_norm=overlap_writeback_norm,
                            downstream_consumed=index < sub_num - 1,
                        )
                    )
        if not optimized_any_group:
            break
        if not config.enable_relation_dispatch:
            iteration_relations = build_overlap_relation_trace(
                problem_id=_problem_id(fun_name, fun_id),
                outer_iter=outer_iter,
                grouping_result=grouping_result,
                overlapping_elements=overlapping_elements,
                fitness_delta_list=fitness_delta_list,
                budget_remaining_ratio=iteration_budget_remaining_ratio,
            )
            relations.extend(iteration_relations)

        if evidence_overlay_observer is not None:
            all_groups_completed = len(fitness_delta_list) == sub_num
            delayed_outcomes_pending = (
                evidence_overlay_observer.delayed_outcomes_pending
            )
            completed = evidence_overlay_observer.complete_sweep(
                sweep_index=outer_iter,
                sweep_end_fe=current_fitness_evaluations(fun),
                sweep_end_candidate=best_individual.copy(),
                sweep_end_error=float(guarded_incumbent_fitness),
                fitness_record=fun.fitness_record,
                all_raw_groups_completed=all_groups_completed,
                native_sweep_end_completed=all_groups_completed,
            )
            if not completed:
                evidence_overlay_frozen_sub_fes = None
                if (
                    evidence_overlay_barrier_attempted
                    and delayed_outcomes_pending
                ):
                    evidence_overlay_runtime_failure = RuntimeError(
                        "evidence-overlay delayed sweep was incomplete"
                    )
            if evidence_overlay_sweep_barrier_ready(
                mode=config.evidence_overlay_mode,
                complete_sweep_count=(
                    evidence_overlay_observer.consecutive_complete_sweep_count
                ),
                previous_survival_closed=evidence_overlay_observer.plan_ready,
                barrier_attempted=evidence_overlay_barrier_attempted,
                all_raw_groups_completed=all_groups_completed,
            ):
                evidence_overlay_barrier_attempted = True
                probe_start_fe = current_fitness_evaluations(fun)
                fingerprints_before = evidence_overlay_runtime_fingerprints(
                    best_individual=best_individual,
                    guarded_incumbent=guarded_incumbent,
                    guarded_incumbent_fitness=guarded_incumbent_fitness,
                    grouping_result=grouping_result,
                    controller=controller_v31_run_state,
                    trajectory_mean_cache=trajectory_mean_cache,
                    previous_group_contribution_credit=(
                        previous_group_contribution_credit
                    ),
                )
                if evidence_overlay_frozen_sub_fes is None:
                    raise RuntimeError(
                        "evidence overlay barrier has no frozen group budget"
                    )
                normal_sweep_fe = evidence_overlay_normal_sweep_reserve(
                    population_sizes,
                    sub_fes=evidence_overlay_frozen_sub_fes,
                )
                barrier_error: BaseException | None = None
                try:
                    evidence_overlay_observer.execute_barrier(
                        lambda candidate: fun(
                            np.asarray(candidate, dtype=float)
                        ),
                        best_individual.copy(),
                        remaining_fe=max(
                            0,
                            config.max_fes
                            - probe_start_fe,
                        ),
                        normal_sweep_fe=normal_sweep_fe,
                        tolerance_fe=terminal_completion_tolerance_fe,
                    )
                except BaseException as error:
                    barrier_error = error
                probe_end_fe = current_fitness_evaluations(fun)
                fingerprints_after = evidence_overlay_runtime_fingerprints(
                    best_individual=best_individual,
                    guarded_incumbent=guarded_incumbent,
                    guarded_incumbent_fitness=guarded_incumbent_fitness,
                    grouping_result=grouping_result,
                    controller=controller_v31_run_state,
                    trajectory_mean_cache=trajectory_mean_cache,
                    previous_group_contribution_credit=(
                        previous_group_contribution_credit
                    ),
                )
                if evidence_overlay_observer.barrier_result is not None:
                    try:
                        evidence_overlay_observer.record_runtime_audit(
                            fingerprints_before=fingerprints_before,
                            fingerprints_after=fingerprints_after,
                            probe_start_fe=probe_start_fe,
                            probe_end_fe=probe_end_fe,
                        )
                    except BaseException as error:
                        if barrier_error is None:
                            barrier_error = error
                actual_overlay_fe = (
                    probe_end_fe - probe_start_fe
                )
                evidence_overlay_fe += actual_overlay_fe
                sum_fes += actual_overlay_fe
                if actual_overlay_fe > 0:
                    evidence_overlay_probe_slice = (
                        probe_start_fe,
                        probe_end_fe,
                    )
                if barrier_error is not None:
                    evidence_overlay_runtime_failure = barrier_error
        if evidence_overlay_runtime_failure is not None:
            break
        previous_group_contribution_credit = fitness_delta_list
        outer_iter += 1
    problem_id = _problem_id(fun_name, fun_id)
    if evidence_overlay_observer is not None:
        evidence_overlay_paths = EvidenceOverlayArtifactPaths(
            manifest=case_artifact_path(
                output_path,
                problem_id,
                "evidence_overlay_manifest.json",
            ),
            checkpoint=case_artifact_path(
                output_path,
                problem_id,
                "evidence_overlay_checkpoint.csv",
            ),
            plan=case_artifact_path(
                output_path,
                problem_id,
                "evidence_overlay_plan.csv",
            ),
            probe_evidence=case_artifact_path(
                output_path,
                problem_id,
                "evidence_overlay_probe_evidence.csv",
            ),
            delayed_outcomes=case_artifact_path(
                output_path,
                problem_id,
                "evidence_overlay_delayed_outcomes.csv",
            ),
            shadow_decisions=case_artifact_path(
                output_path,
                problem_id,
                "evidence_overlay_shadow_decisions.csv",
            ),
        )
        terminal_record = tuple(float(value) for value in fun.fitness_record)
        native_record = terminal_record
        if evidence_overlay_probe_slice is not None:
            probe_start_fe, probe_end_fe = evidence_overlay_probe_slice
            native_record = (
                terminal_record[:probe_start_fe]
                + terminal_record[probe_end_fe:]
            )
        if not terminal_record or not native_record:
            raise RuntimeError("evidence overlay terminal fitness record is empty")
        try:
            evidence_overlay_observer.write_artifacts(
                paths=evidence_overlay_paths,
                native_terminal_error=min(native_record),
                all_evaluation_best_error=min(terminal_record),
            )
        except Exception as error:
            if evidence_overlay_runtime_failure is None:
                evidence_overlay_runtime_failure = error
    _write_overlap_relation_trace(
        case_artifact_path(output_path, problem_id, "overlap_relations.csv"),
        relations,
    )
    if config.enable_relation_dispatch:
        _write_action_decision_log(
            case_artifact_path(output_path, problem_id, "action_decision.csv"),
            config.run_id,
            relations,
            action_decisions,
        )
        _write_action_decision_log(
            output_path / "action_decision.csv",
            config.run_id,
            relations,
            action_decisions,
        )
        _write_action_mismatch_audit_log(
            case_artifact_path(output_path, problem_id, "action_mismatch_audit.csv"),
            config.run_id,
            relations,
            action_decisions,
            relation_policy_mode=config.relation_policy_mode,
        )
        _write_action_mismatch_audit_log(
            output_path / "action_mismatch_audit.csv",
            config.run_id,
            relations,
            action_decisions,
            relation_policy_mode=config.relation_policy_mode,
        )
    _write_budget_summary(
        case_artifact_path(output_path, problem_id, "budget_summary.csv"),
        problem_id=problem_id,
        budget_accounting=config.budget_accounting,
        max_fes=config.max_fes,
        optimizer_reported_fe=sum_fes,
        fitness_record_fe=current_fitness_evaluations(fun),
        global_phase_fe=global_phase_fe,
        cc_phase_fe=cc_phase_fe,
        rescue_fe=rescue_fe,
        refresh_fe=refresh_fe,
        search_state_fe=search_state_fe,
        evidence_overlay_fe=(
            evidence_overlay_fe if evidence_overlay_enabled else None
        ),
    )
    if evidence_overlay_runtime_failure is not None:
        raise RuntimeError(
            "evidence overlay trajectory failed closed after writing artifacts"
        ) from evidence_overlay_runtime_failure
    print(f"{problem_id} overlap relations extracted: {len(relations)}")
    return fun.fitness_record, time.time() - time_start, action_trace_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen exp_018 HCC evidence-overlay runner."
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        choices=FUNCTION_NAMES,
        required=True,
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        choices=PROBLEM_IDS,
        required=True,
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--aob-data-root",
        type=lambda value: Path(value).resolve(),
        default=DATA_DIR.resolve(),
    )
    parser.add_argument("--timestamp", default="arac-hcc-smoke")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-fes", type=int, required=True)
    parser.add_argument(
        "--arac-action",
        choices=[EVIDENCE_ACTION_CONTROLLER_V37],
        required=True,
    )
    parser.add_argument(
        "--budget-accounting",
        choices=["strict"],
        default="strict",
    )
    parser.add_argument(
        "--search-state-backend",
        choices=["phase_i_mmes"],
        default="phase_i_mmes",
    )
    parser.add_argument("--enable-relation-dispatch", action="store_true")
    parser.add_argument(
        "--relation-policy",
        choices=["controller_v31"],
        required=True,
    )
    parser.add_argument(
        "--evidence-overlay-mode",
        choices=sorted(EVIDENCE_OVERLAY_MODES),
        default="off",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.set_defaults(
        verbose=1000,
        early_stopping_evaluations=1000,
        mmes_restart=True,
        cmaes_restart=True,
        arac_action_file=None,
    )
    args = parser.parse_args(argv)
    if args.max_fes <= 0:
        parser.error("--max-fes must be positive")
    if len(args.functions) != 1 or len(args.ids) != 1:
        parser.error("exp_018 runner accepts exactly one function/id pair")
    if (args.functions[0], args.ids[0]) not in ACTIVE_FUNCTION_ID_PAIRS:
        parser.error("function/id pair is outside the frozen exp_018 cases")
    if args.evidence_overlay_mode != "off" and not args.enable_relation_dispatch:
        parser.error("--evidence-overlay-mode requires relation dispatch")
    return args


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    for fun_id in args.ids:
        validate_aob_data_root(args.aob_data_root, fun_id)
    config = SmokeConfig(
        run_id=args.timestamp,
        max_fes=args.max_fes,
        seed=args.seed,
        verbose=args.verbose,
        early_stopping_evaluations=args.early_stopping_evaluations,
        mmes_restart=args.mmes_restart,
        cmaes_restart=args.cmaes_restart,
        arac_action=args.arac_action,
        enable_relation_dispatch=args.enable_relation_dispatch,
        relation_policy_mode=args.relation_policy,
        arac_action_file=args.arac_action_file,
        budget_accounting=args.budget_accounting,
        skip_plots=args.skip_plots,
        aob_data_root=args.aob_data_root,
        search_state_backend=args.search_state_backend,
        evidence_overlay_mode=args.evidence_overlay_mode,
    )
    output_paths = []
    for fun_name in args.functions:
        output_path = Path(args.output_root) / args.timestamp / fun_name
        output_path.mkdir(parents=True, exist_ok=True)
        output_data = {}
        function_trace_rows: list[dict[str, str]] = []
        function_action_decision_rows: list[dict[str, str]] = []
        function_action_mismatch_rows: list[dict[str, str]] = []
        function_aob_input_rows: list[dict[str, str]] = []
        _remove_if_exists(output_path / "action_decision.csv")
        _remove_if_exists(output_path / "action_mismatch_audit.csv")
        for fun_id in args.ids:
            algorithm = f"{fun_name}_{fun_id}"
            output_data[algorithm] = []
            output_data[f"{algorithm}_time"] = []
            aob_inputs_before = snapshot_aob_inputs(fun_id, config.aob_data_root)
            record, elapsed, trace_rows = run_problem(fun_name, fun_id, output_path, config)
            aob_inputs_after = snapshot_aob_inputs(fun_id, config.aob_data_root)
            output_data[algorithm].append(record)
            output_data[f"{algorithm}_time"].append(elapsed)
            problem_id = _problem_id(fun_name, fun_id)
            aob_input_rows = build_aob_input_audit_rows(
                problem_id,
                aob_inputs_before,
                aob_inputs_after,
            )
            _write_aob_input_manifest(
                case_artifact_path(output_path, problem_id, "aob_input_manifest.csv"),
                aob_input_rows,
            )
            require_unchanged_aob_inputs(problem_id, aob_input_rows)
            function_aob_input_rows.extend(aob_input_rows)
            _write_action_trace(
                case_artifact_path(output_path, problem_id, "action_trace.csv"),
                trace_rows,
            )
            function_trace_rows.extend(trace_rows)
            if config.enable_relation_dispatch:
                function_action_decision_rows.extend(
                    _read_csv_rows(
                        case_artifact_path(output_path, problem_id, "action_decision.csv")
                    )
                )
                function_action_mismatch_rows.extend(
                    _read_csv_rows(
                        case_artifact_path(
                            output_path,
                            problem_id,
                            "action_mismatch_audit.csv",
                        )
                    )
                )
            print(f"{algorithm} average time: {elapsed}")
        _write_action_trace(
            output_path / "action_trace.csv",
            function_trace_rows,
        )
        _write_aob_input_manifest(
            output_path / "aob_input_manifest.csv",
            function_aob_input_rows,
        )
        if config.enable_relation_dispatch:
            _write_raw_action_decision_rows(
                output_path / "action_decision.csv",
                function_action_decision_rows,
            )
        if function_action_mismatch_rows:
            _write_raw_action_mismatch_rows(
                output_path / "action_mismatch_audit.csv",
                function_action_mismatch_rows,
            )
        evaluation_record(output_data, str(output_path) + "/", record_FEs_list=(args.max_fes,))
        if not config.skip_plots:
            plot_evaluation_curve(output_data, str(output_path) + "/", font_size=12, log_scale=True)
            plot_evaluation_curve_best_so_far(
                output_data,
                str(output_path) + "/",
                font_size=12,
                log_scale=True,
                show_variance=True,
            )
        output_paths.append(output_path)
    return output_paths


if __name__ == "__main__":
    main()
