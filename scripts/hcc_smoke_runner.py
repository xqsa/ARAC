from __future__ import annotations

import argparse
import csv
import hashlib
import math
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
    select_evidence_action_controller_v3_mode,
    select_evidence_action_controller_v31_mode,
)
from src.arac.policy.relation_policy import (
    decide_actions_for_relations,
    decide_actions_for_relations_v2,
    decide_actions_for_relations_v21,
    decide_actions_for_relations_v22,
    decide_actions_for_relations_v23,
    decide_actions_for_relations_v24,
    decide_actions_for_relations_v25,
    decide_actions_for_relations_v26,
)
from src.arac.policy.search_state_policy import (
    CC_RESERVE_FRACTION,
    CONTINUE_CANONICAL_CC,
    CONTINUE_DIAGONAL_SEARCH_STATE,
    FIRST_PROBE_FRACTION,
    RESUME_PHASE_I_SEARCH_STATE,
    SEARCH_STATE_BLOCKED,
    PreHoldEvidence,
    SearchStateEvidence,
    SearchStateSchedulerState,
    build_pre_hold_evidence,
    normalized_gain_utility,
    plan_search_state_action,
    record_search_state_outcome,
)
from src.arac.policy.action_trust_policy import (
    ActionTrustDecision,
    ActionTrustPolicy,
    make_action_key,
    normalized_objective_credit,
    robust_damped_writeback,
)
from src.arac.policy.component_delayed_credit import (
    COMPONENT_CREDIT_TRACE_FIELDS,
    ComponentDelayedCreditTrace,
    calculate_scheduler_revisit_cap,
)
from src.arac.actions.controller_profiles import (
    controller_has_capability,
    controller_profile_by_action,
    controller_profile_by_version,
)
from arac.backends.hcc_car import (
    CARPlanDecision,
    CARRelationProposal,
    GroupOptimizationResult,
    allocate_component_horizon_budgets,
    freeze_component_writeback_plan,
    run_component_horizon,
    shuffled_component_writeback_plan,
)
from arac.policy.counterfactual_action_racing import (
    AuditEnvelope,
    BranchState,
    CARBudgetLedger,
    CARProbeExecutor,
    derive_probe_seed,
    fingerprint_branch_state,
)
from arac.policy.oracle_actionability import (
    CAR_ACTIONABILITY_HORIZON_LABELS,
    CAR_ACTIONABILITY_HORIZON_MULTIPLIERS,
    CAR_ACTIONABILITY_PROTOCOL_VERSION,
)
from src.arac.policy.trajectory_guard import (
    RecoveryCheckpoint,
    RecoveryResolution,
    make_recovery_checkpoint,
    preempt_recovery_checkpoint,
    resolve_recovery_checkpoint,
)
from src.arac.backends.diagonal_cma import (
    DiagonalCMAState,
    initialize_diagonal_cma_state,
    run_diagonal_cma_block,
)
from src.arac.backends.hcc import required_aob_data_files, validate_aob_data_root
from HCC.NDAs.MMES.state import MMESBlockResult, MMESState

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
FUNCTION_NAMES = ("elliptic", "schwefel", "rastrigin", "ackley")
PROBLEM_IDS = (1, 2, 3, 4, 5, 6)
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
    "search_state_non_coordinate_fraction",
    "search_state_active_intervention_fraction",
    "search_state_conflict_fraction",
    "search_state_writeback_unstable",
    "search_state_relative_writeback_max",
    "search_state_relative_writeback_unstable",
    "search_state_block_fe",
    "search_state_utility",
    "required_utility_ratio",
    "state_action_fe",
    "cc_reserve_fe",
    "state_fingerprint_before",
    "state_fingerprint_after",
    "abstain_reason",
    "pre_hold_phase_i_tail_utility",
    "pre_hold_group_count",
    "pre_hold_mean_group_size",
    "pre_hold_overlap_edge_count",
    "pre_hold_overlap_edge_fraction",
    "pre_hold_shared_variable_count",
    "pre_hold_shared_variable_ratio",
    "pre_hold_mean_overlap_width",
    "pre_hold_remaining_fes",
    "pre_hold_remaining_ratio",
    "pre_hold_scheduled_hold_fes",
    "pre_hold_projected_unheld_group_fes",
    "pre_hold_projected_held_group_fes",
    "pre_hold_budget_retention_ratio",
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
    "cma_sigma_reference",
    "cma_sigma_applied_factor",
    "cma_sigma_terminal",
    "cma_sigma_next_factor",
    "cma_sigma_route",
    "cma_restart_count",
    *COMPONENT_CREDIT_TRACE_FIELDS,
    "trajectory_guard_status",
    "trajectory_guard_pre_fitness",
    "trajectory_guard_post_writeback_fitness",
    "trajectory_guard_downstream_fitness",
    "trajectory_guard_recovery_credit",
    "trajectory_guard_restored",
]
V33_TRUST_TRACE_FIELDS = [
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
]
V34_RECOVERY_TRACE_FIELDS = [
    "trajectory_guard_status",
    "trajectory_guard_pre_fitness",
    "trajectory_guard_post_writeback_fitness",
    "trajectory_guard_downstream_fitness",
    "trajectory_guard_recovery_credit",
    "trajectory_guard_restored",
]
V36_MATURITY_TRACE_FIELDS = [
    "active_maturity_route",
    "sweep_evidence_relation_count",
    "sweep_evidence_active_count",
    "sweep_evidence_active_fraction",
    "sweep_evidence_support",
    "sweep_evidence_reason",
]
V37_RESOURCE_TRACE_FIELDS = [
    "phase_rescue_resource_route",
    "phase_rescue_rejected_before_maturity",
    "phase_rescue_productive_mature",
    "phase_rescue_retired",
]
V39_CMA_SIGMA_TRACE_FIELDS = [
    "cma_sigma_reference",
    "cma_sigma_applied_factor",
    "cma_sigma_terminal",
    "cma_sigma_next_factor",
    "cma_sigma_route",
    "cma_restart_count",
]
V40_COMPONENT_CREDIT_TRACE_FIELDS = list(COMPONENT_CREDIT_TRACE_FIELDS)
V33_ACTION_TRACE_FIELDS = [
    field
    for field in ACTION_TRACE_FIELDS
    if field not in V34_RECOVERY_TRACE_FIELDS
    and field not in V36_MATURITY_TRACE_FIELDS
    and field not in V37_RESOURCE_TRACE_FIELDS
    and field not in V39_CMA_SIGMA_TRACE_FIELDS
    and field not in V40_COMPONENT_CREDIT_TRACE_FIELDS
]
V34_ACTION_TRACE_FIELDS = [
    field
    for field in ACTION_TRACE_FIELDS
    if field not in V36_MATURITY_TRACE_FIELDS
    and field not in V37_RESOURCE_TRACE_FIELDS
    and field not in V39_CMA_SIGMA_TRACE_FIELDS
    and field not in V40_COMPONENT_CREDIT_TRACE_FIELDS
]
V36_ACTION_TRACE_FIELDS = [
    field
    for field in ACTION_TRACE_FIELDS
    if field not in V34_RECOVERY_TRACE_FIELDS
    and field not in V37_RESOURCE_TRACE_FIELDS
    and field not in V39_CMA_SIGMA_TRACE_FIELDS
    and field not in V40_COMPONENT_CREDIT_TRACE_FIELDS
]
V37_ACTION_TRACE_FIELDS = [
    field
    for field in ACTION_TRACE_FIELDS
    if field not in V34_RECOVERY_TRACE_FIELDS
    and field not in V39_CMA_SIGMA_TRACE_FIELDS
    and field not in V40_COMPONENT_CREDIT_TRACE_FIELDS
]
V39_ACTION_TRACE_FIELDS = [
    field
    for field in ACTION_TRACE_FIELDS
    if field not in V34_RECOVERY_TRACE_FIELDS
    and field not in V40_COMPONENT_CREDIT_TRACE_FIELDS
]
V40_ACTION_TRACE_FIELDS = [
    field
    for field in ACTION_TRACE_FIELDS
    if field not in V34_RECOVERY_TRACE_FIELDS
    and field not in V39_CMA_SIGMA_TRACE_FIELDS
]
LEGACY_ACTION_TRACE_FIELDS = [
    field
    for field in V33_ACTION_TRACE_FIELDS
    if field not in V33_TRUST_TRACE_FIELDS
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
CAR_PROBE_TRACE_FIELDS = [
    "problem_id",
    "seed",
    "pair_index",
    "channel",
    "graph_fingerprint",
    "component_fingerprint",
    "action_family",
    "candidate_mode",
    "fallback_fe",
    "candidate_fe",
    "seed_descriptor",
    "probe_seed",
    "phase1_probe_fitness_before",
    "fallback_after",
    "candidate_after",
    "normalized_delta",
    "lcb",
    "tail",
    "gate_result",
    "abstain_reason",
]
CAR_STATE_LEDGER_FIELDS = [
    "problem_id",
    "seed",
    "graph_fingerprint",
    "component_fingerprint",
    "candidate_action_name",
    "candidate_action_family",
    "candidate_mode",
    "evidence_sweeps",
    "checkpoint_fe",
    "probe_fe",
    "total_fe_after_probe",
    "probe_fe_limit",
    "adopted_branch",
    "committed_fitness",
    "evaluated_elite",
    "state_fingerprint",
    "gate_result",
    "abstain_reason",
]
CAR_BRANCH_MANIFEST_FIELDS = [
    "problem_id",
    "seed",
    "pair_index",
    "arm",
    "candidate_mode",
    "evaluator_id",
    "requested_fe",
    "actual_fe",
    "record_sha256",
    "record_best",
    "state_fingerprint_before",
    "state_fingerprint_after",
    "seed_descriptor",
    "probe_seed",
]
CAR_ACTIONABILITY_TRACE_FIELDS = [
    "protocol_version",
    "fresh_optimizer_execution",
    "problem_id",
    "seed",
    "audit_arm",
    "candidate_mode",
    "horizon_index",
    "horizon_label",
    "checkpoint_fe",
    "checkpoint_fitness",
    "configured_max_fes",
    "terminal_completion_tolerance_fe",
    "termination_reason",
    "terminal_fe_shortfall",
    "target_fe",
    "observed_fe",
    "best_error",
    "prefix_state_fingerprint",
    "prefix_record_sha256",
    "post_intervention_state_fingerprint",
    "graph_fingerprint",
    "component_fingerprint",
    "candidate_action_name",
    "candidate_action_family",
    "candidate_action_applied",
    "requested_fe",
    "actual_fe",
    "seed_descriptor",
    "probe_seed",
    "intervention_record_sha256",
    "fitness_prefix_sha256",
    "plan_status",
    "horizon_status",
    "abstain_reason",
]
CAR_W_MIN_EVIDENCE_SWEEPS = 2
CAR_W_PAIR_COUNT = 3
CAR_W_PROBE_BUDGET_FRACTION = 0.03
CAR_W2_FUTILITY_MIN_WRITEBACK_NORM = 1e-12
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
SEARCH_STATE_BIPOP_ACTION = "bipop_search_state_restart"
REPAIR_BIPOP_SEARCH_STATE_ACTION = "repair_bipop_search_state_restart"
PHASE_RESCUE_MULTISTART_ACTION = "phase_rescue_multistart"
REPAIR_PHASE_RESCUE_MULTISTART_ACTION = "repair_phase_rescue_multistart"
CC_HARM_GUARDED_SEP_REFRESH_ACTION = "cc_harm_guarded_sep_refresh"
SEPARABLE_CMAES_DISPATCH_ACTION = "separable_cmaes_dispatch_action"
REPAIR_PROTECT_REFINE_ACTION = "repair_protect_refine"
REPAIR_PROTECT_DEEP_REFINE_ACTION = "repair_protect_deep_refine"
POST_RETIREMENT_PRECISION_REANCHOR_ACTION = "post_retirement_precision_reanchor"
CROSS_SWEEP_CMA_SIGMA_CONTINUATION_ACTION = "cross_sweep_cma_sigma_continuation"
EVIDENCE_ACTION_CONTROLLER_V1 = "arac_evidence_action_controller_v1"
EVIDENCE_ACTION_CONTROLLER_V2 = "arac_evidence_action_controller_v2"
EVIDENCE_ACTION_CONTROLLER_V3 = "arac_evidence_action_controller_v3"
EVIDENCE_ACTION_CONTROLLER_V31 = "arac_evidence_action_controller_v31"
EVIDENCE_ACTION_CONTROLLER_V32 = "arac_evidence_action_controller_v32"
EVIDENCE_ACTION_CONTROLLER_V33 = controller_profile_by_version(33).action_name
EVIDENCE_ACTION_CONTROLLER_V34 = controller_profile_by_version(34).action_name
EVIDENCE_ACTION_CONTROLLER_V35 = controller_profile_by_version(35).action_name
EVIDENCE_ACTION_CONTROLLER_V36 = controller_profile_by_version(36).action_name
EVIDENCE_ACTION_CONTROLLER_V37 = controller_profile_by_version(37).action_name
EVIDENCE_ACTION_CONTROLLER_V38 = controller_profile_by_version(38).action_name
EVIDENCE_ACTION_CONTROLLER_V39 = controller_profile_by_version(39).action_name
EVIDENCE_ACTION_CONTROLLER_V40 = controller_profile_by_version(40).action_name
CAR_W_ACTION = controller_profile_by_action("arac_counterfactual_action_racing_w").action_name
CAR_W2_ACTION = controller_profile_by_action("arac_counterfactual_action_racing_w2").action_name
CAR_W3_ACTION = controller_profile_by_action("arac_counterfactual_action_racing_w3").action_name
TRAJECTORY_ACTION_NAMES = {
    "budget_shift_mean_blend",
    "budget_shift_only",
    "mean_blend_only",
    SEARCH_STATE_BIPOP_ACTION,
    REPAIR_BIPOP_SEARCH_STATE_ACTION,
    PHASE_RESCUE_MULTISTART_ACTION,
    REPAIR_PHASE_RESCUE_MULTISTART_ACTION,
    CC_HARM_GUARDED_SEP_REFRESH_ACTION,
    SEPARABLE_CMAES_DISPATCH_ACTION,
    REPAIR_PROTECT_REFINE_ACTION,
    REPAIR_PROTECT_DEEP_REFINE_ACTION,
    POST_RETIREMENT_PRECISION_REANCHOR_ACTION,
    CROSS_SWEEP_CMA_SIGMA_CONTINUATION_ACTION,
    EVIDENCE_ACTION_CONTROLLER_V1,
    EVIDENCE_ACTION_CONTROLLER_V2,
    EVIDENCE_ACTION_CONTROLLER_V3,
    EVIDENCE_ACTION_CONTROLLER_V31,
    EVIDENCE_ACTION_CONTROLLER_V32,
    EVIDENCE_ACTION_CONTROLLER_V33,
    EVIDENCE_ACTION_CONTROLLER_V34,
    EVIDENCE_ACTION_CONTROLLER_V35,
    EVIDENCE_ACTION_CONTROLLER_V36,
    EVIDENCE_ACTION_CONTROLLER_V37,
    EVIDENCE_ACTION_CONTROLLER_V38,
    EVIDENCE_ACTION_CONTROLLER_V39,
    CAR_W_ACTION,
    CAR_W2_ACTION,
    CAR_W3_ACTION,
    RESUME_PHASE_I_SEARCH_STATE,
    CONTINUE_DIAGONAL_SEARCH_STATE,
}
TRAJECTORY_BUDGET_SHIFT_STRENGTH = 0.35
TRAJECTORY_MEAN_BLEND_STRENGTH = 0.25
TRAJECTORY_MIN_POSITIVE_CREDIT_GROUPS = 2
REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER = 0.5
REPAIR_PROTECT_DEEP_REFINE_SIGMA_MULTIPLIER = 0.25
BIPOP_ESCAPE_BUDGET_FRACTION = 0.50
BIPOP_LARGE_POPULATION_MULTIPLIER = 2
BIPOP_LARGE_SIGMA_MULTIPLIER = 2.0
BIPOP_MIN_POPULATION_SIZE = 4
BIPOP_MIN_SIGMA_MULTIPLIER = 0.35
BIPOP_STAGNATION_EPSILON = 1e-8
BIPOP_STAGNATION_WINDOW = 2
BIPOP_RESTART_COOLDOWN = 1
BIPOP_ACCEPT_RELATIVE_IMPROVEMENT = 1e-4
BIPOP_REJECT_BACKOFF_SWEEP_CAP = 3
PHASE_RESCUE_START_COUNT = 3
PHASE_RESCUE_SIGMA_MULTIPLIER = 1.5
PHASE_RESCUE_ESCAPE_BUDGET_FRACTION = 0.60
PHASE_RESCUE_STAGNATION_WINDOW = 1
CC_HARM_MIN_GROUP_UPDATES = 3
CC_HARM_STAGNATED_FRACTION = 0.67
CC_HARM_CONFLICT_FRACTION = 0.50
CC_HARM_LOW_GAIN_RATIO = 1e-6
CC_HARM_WRITEBACK_NORM = 1e-9
RELATIVE_WRITEBACK_UNSTABLE_THRESHOLD = 0.10
CC_HARM_REFRESH_SIGMA_MULTIPLIER = 0.75
SEPARABLE_CMAES_INITIAL_SIGMA = 0.5
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
    car_branch_order: str = "fallback_first"
    car_candidate_mode: str = "graph"
    car_actionability_arm: str = "off"


@dataclass(frozen=True)
class RelationExecutionContext:
    overlap_indices: list[int]
    previous_values: np.ndarray
    current_values: np.ndarray
    previous_delta: float
    current_delta: float


@dataclass(frozen=True)
class BipopRestartPlan:
    restart_mode: str
    population_size: int
    sigma: float
    escape_budget: int


@dataclass
class PendingActionTrustObservation:
    decision: ActionTrustDecision
    pre_writeback_fitness: float
    unstable: bool
    trace_row: dict[str, str]


@dataclass
class PendingTrajectoryRecovery:
    checkpoint: RecoveryCheckpoint
    trace_row: dict[str, str]
    post_writeback_fitness: float | None = None


@dataclass
class EvidenceActionControllerV31RunState:
    dense_overlap: bool
    action_trust_policy: ActionTrustPolicy | None = field(default=None, repr=False)
    trajectory_guard_enabled: bool = False
    pending_trajectory_recovery: PendingTrajectoryRecovery | None = field(
        default=None,
        repr=False,
    )
    pending_action_trust: PendingActionTrustObservation | None = field(
        default=None,
        repr=False,
    )
    locked_policy_mode: str | None = None
    non_dense_repair_locked: bool = False
    non_dense_repair_lock_trigger: str = ""
    search_state_scheduler_state: SearchStateSchedulerState = field(
        default_factory=SearchStateSchedulerState
    )
    phase_i_optimizer: object | None = field(default=None, repr=False)
    phase_i_state: MMESState | None = field(default=None, repr=False)
    diagonal_cma_state: DiagonalCMAState | None = field(default=None, repr=False)
    phase_i_runtime_tail_utility: float = 0.0
    cc_utility_history: list[float] = field(default_factory=list)
    v36_enabled: bool = False
    v37_enabled: bool = False
    v38_enabled: bool = False
    v39_enabled: bool = False
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
    _v39_cma_sigma_factors: dict[tuple[int, ...], float] = field(
        default_factory=dict,
        repr=False,
    )
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

    def v39_cma_sigma_for_group(
        self,
        group_dims: list[int] | tuple[int, ...] | np.ndarray,
        reference_sigma: float,
    ) -> tuple[float, float, str]:
        reference = float(reference_sigma)
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference sigma must be finite and positive")
        if not self.v39_enabled:
            return reference, 1.0, ""
        key = tuple(int(index) for index in group_dims)
        factor = self._v39_cma_sigma_factors.get(key)
        if factor is None:
            return reference, 1.0, "cold_start"
        return reference * factor, factor, "continued"

    def observe_v39_cma_terminal_sigma(
        self,
        group_dims: list[int] | tuple[int, ...] | np.ndarray,
        *,
        reference_sigma: float,
        terminal_sigma: float,
    ) -> float:
        if not self.v39_enabled:
            return 1.0
        reference = float(reference_sigma)
        terminal = float(terminal_sigma)
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference sigma must be finite and positive")
        if not math.isfinite(terminal) or terminal <= 0.0:
            raise ValueError("terminal sigma must be finite and positive")
        lower_factor = (
            REPAIR_PROTECT_DEEP_REFINE_SIGMA_MULTIPLIER
            / REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER
        )
        next_factor = float(
            np.clip(
                terminal / reference,
                lower_factor,
                PHASE_RESCUE_SIGMA_MULTIPLIER,
            )
        )
        key = tuple(int(index) for index in group_dims)
        self._v39_cma_sigma_factors[key] = next_factor
        return next_factor

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

    def register_pending_trajectory_guard(
        self,
        *,
        candidate: np.ndarray,
        pre_writeback_fitness: float,
        trace_row: dict[str, str],
    ) -> RecoveryCheckpoint | None:
        if not self.trajectory_guard_enabled:
            return None
        if self.pending_trajectory_recovery is not None:
            raise RuntimeError("trajectory recovery checkpoint is already pending")
        checkpoint = make_recovery_checkpoint(candidate, pre_writeback_fitness)
        self.pending_trajectory_recovery = PendingTrajectoryRecovery(
            checkpoint=checkpoint,
            trace_row=trace_row,
        )
        trace_row.update(
            {
                "trajectory_guard_status": "pending",
                "trajectory_guard_pre_fitness": (
                    f"{checkpoint.fitness:.17e}"
                ),
                "trajectory_guard_post_writeback_fitness": "",
                "trajectory_guard_downstream_fitness": "",
                "trajectory_guard_recovery_credit": "",
                "trajectory_guard_restored": "",
            }
        )
        return checkpoint

    def observe_pending_trajectory_guard(
        self,
        *,
        post_writeback_fitness: float,
    ) -> float | None:
        pending = self.pending_trajectory_recovery
        if not self.trajectory_guard_enabled or pending is None:
            return None
        fitness = float(post_writeback_fitness)
        if not math.isfinite(fitness):
            raise ValueError("post-writeback fitness must be finite")
        pending.post_writeback_fitness = fitness
        pending.trace_row["trajectory_guard_post_writeback_fitness"] = (
            f"{fitness:.17e}"
        )
        return fitness

    def resolve_pending_trajectory_guard(
        self,
        *,
        downstream_candidate: np.ndarray,
        downstream_fitness: float,
    ) -> RecoveryResolution | None:
        pending = self.pending_trajectory_recovery
        if not self.trajectory_guard_enabled or pending is None:
            return None
        if pending.post_writeback_fitness is None:
            raise RuntimeError(
                "trajectory recovery requires a post-writeback observation"
            )
        resolved = resolve_recovery_checkpoint(
            pending.checkpoint,
            downstream_candidate=downstream_candidate,
            downstream_fitness=downstream_fitness,
        )
        pending.trace_row.update(
            {
                "trajectory_guard_status": resolved.status,
                "trajectory_guard_downstream_fitness": (
                    f"{float(downstream_fitness):.17e}"
                ),
                "trajectory_guard_recovery_credit": (
                    ""
                    if resolved.recovery_credit is None
                    else f"{resolved.recovery_credit:.6e}"
                ),
                "trajectory_guard_restored": str(int(resolved.restored)),
            }
        )
        self.pending_trajectory_recovery = None
        return resolved

    def preempt_pending_trajectory_guard(self) -> RecoveryResolution | None:
        pending = self.pending_trajectory_recovery
        if not self.trajectory_guard_enabled or pending is None:
            return None
        resolved = preempt_recovery_checkpoint(pending.checkpoint)
        pending.trace_row.update(
            {
                "trajectory_guard_status": resolved.status,
                "trajectory_guard_downstream_fitness": "",
                "trajectory_guard_recovery_credit": "",
                "trajectory_guard_restored": "1",
            }
        )
        self.pending_trajectory_recovery = None
        return resolved


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
        trajectory_guard_enabled=(
            action_name is not None
            and controller_has_capability(action_name, "trajectory_guard")
        ),
        v36_enabled=(
            action_name is not None
            and controller_has_capability(action_name, "maturity")
        ),
        v37_enabled=(
            action_name is not None
            and controller_has_capability(action_name, "rescue_retirement")
        ),
        v38_enabled=(
            action_name is not None
            and controller_has_capability(action_name, "precision_reanchor")
        ),
        v39_enabled=(
            action_name is not None
            and controller_has_capability(action_name, "sigma_continuation")
        ),
    )


def reconcile_trajectory_recovery_context(
    *,
    resolution: RecoveryResolution,
    checkpoint_candidate: np.ndarray,
    original_best: np.ndarray,
    original_fitness: float,
    current_delta: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    best = resolution.candidate.copy()
    if not resolution.restored:
        return (
            best,
            np.asarray(original_best, dtype=float).copy(),
            float(original_fitness),
            float(current_delta),
        )
    return (
        best,
        np.asarray(checkpoint_candidate, dtype=float).copy(),
        float(resolution.fitness),
        0.0,
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


def is_bipop_search_state_action(action_name: str) -> bool:
    return action_name in {SEARCH_STATE_BIPOP_ACTION, REPAIR_BIPOP_SEARCH_STATE_ACTION}


def is_phase_rescue_multistart_action(action_name: str) -> bool:
    return action_name in {
        PHASE_RESCUE_MULTISTART_ACTION,
        REPAIR_PHASE_RESCUE_MULTISTART_ACTION,
    }


def is_evidence_action_controller_v1(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V1


def is_evidence_action_controller_v2(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V2


def is_evidence_action_controller_v3(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V3


def is_evidence_action_controller_v31(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V31


def is_evidence_action_controller_v32(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V32


def is_evidence_action_controller_v33(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V33


def is_evidence_action_controller_v34(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V34


def is_evidence_action_controller_v35(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V35


def is_evidence_action_controller_v36(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V36


def is_evidence_action_controller_v37(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V37


def is_evidence_action_controller_v38(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V38


def is_evidence_action_controller_v39(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V39


def is_evidence_action_controller_v40(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V40


def is_car_w_action(action_name: str) -> bool:
    return action_name == CAR_W_ACTION


def is_car_w2_action(action_name: str) -> bool:
    return action_name == CAR_W2_ACTION


def is_car_w3_action(action_name: str) -> bool:
    return action_name == CAR_W3_ACTION


def is_car_w_family_action(action_name: str) -> bool:
    return (
        is_car_w_action(action_name)
        or is_car_w2_action(action_name)
        or is_car_w3_action(action_name)
    )


def is_risk_aware_evidence_action_controller(action_name: str) -> bool:
    return is_evidence_action_controller_v33(
        action_name
    ) or is_evidence_action_controller_v34(action_name) or is_car_w_family_action(action_name)


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


def is_guarded_evidence_action_controller(action_name: str) -> bool:
    return (
        is_evidence_action_controller_v3(action_name)
        or is_evidence_action_controller_v31(action_name)
        or is_evidence_action_controller_v32(action_name)
        or is_evidence_action_controller_v33(action_name)
        or is_evidence_action_controller_v34(action_name)
        or is_evidence_action_controller_v35(action_name)
        or is_evidence_action_controller_v36(action_name)
        or is_evidence_action_controller_v37(action_name)
        or is_evidence_action_controller_v38(action_name)
        or is_evidence_action_controller_v39(action_name)
        or is_evidence_action_controller_v40(action_name)
        or is_car_w_family_action(action_name)
    )


def is_evidence_action_controller(action_name: str) -> bool:
    return (
        is_evidence_action_controller_v1(action_name)
        or is_evidence_action_controller_v2(action_name)
        or is_evidence_action_controller_v3(action_name)
        or is_evidence_action_controller_v31(action_name)
        or is_evidence_action_controller_v32(action_name)
        or is_evidence_action_controller_v33(action_name)
        or is_evidence_action_controller_v34(action_name)
        or is_evidence_action_controller_v35(action_name)
        or is_evidence_action_controller_v36(action_name)
        or is_evidence_action_controller_v37(action_name)
        or is_evidence_action_controller_v38(action_name)
        or is_evidence_action_controller_v39(action_name)
        or is_evidence_action_controller_v40(action_name)
        or is_car_w_family_action(action_name)
    )


def uses_phase_rescue_controller(action_name: str) -> bool:
    return is_phase_rescue_multistart_action(action_name) or is_evidence_action_controller_v1(action_name)


def uses_cc_harm_guard_controller(action_name: str) -> bool:
    return is_cc_harm_guarded_sep_refresh_action(action_name) or is_evidence_action_controller_v1(action_name)


def uses_cc_harm_guard_during_run(
    action_name: str,
    *,
    evidence_controller_search_state_enabled: bool,
) -> bool:
    if uses_cc_harm_guard_controller(action_name):
        return True
    return (
        is_evidence_action_controller_v3(action_name)
        and evidence_controller_search_state_enabled
    )


def uses_phase_rescue_during_run(
    action_name: str,
    *,
    evidence_controller_search_state_enabled: bool,
) -> bool:
    return uses_phase_rescue_controller(action_name) or (
        (
            is_evidence_action_controller_v3(action_name)
            or is_evidence_action_controller_v32(action_name)
            or is_evidence_action_controller_v33(action_name)
            or is_evidence_action_controller_v34(action_name)
            or is_evidence_action_controller_v35(action_name)
            or is_evidence_action_controller_v36(action_name)
            or is_evidence_action_controller_v37(action_name)
            or is_evidence_action_controller_v38(action_name)
            or is_evidence_action_controller_v39(action_name)
            or is_evidence_action_controller_v40(action_name)
            or is_car_w_family_action(action_name)
        )
        and evidence_controller_search_state_enabled
    )


def uses_resumable_phase_i_state_during_run(action_name: str) -> bool:
    return is_evidence_action_controller_v31(action_name)


def uses_scheduled_search_state(config: SmokeConfig) -> bool:
    if config.search_state_backend == "diagonal_cma":
        return bool(
            is_evidence_action_controller_v31(config.arac_action)
            or is_evidence_action_controller_v32(config.arac_action)
            or is_evidence_action_controller_v33(config.arac_action)
            or is_evidence_action_controller_v34(config.arac_action)
            or is_evidence_action_controller_v35(config.arac_action)
            or is_evidence_action_controller_v36(config.arac_action)
            or is_evidence_action_controller_v37(config.arac_action)
            or is_evidence_action_controller_v38(config.arac_action)
            or is_evidence_action_controller_v39(config.arac_action)
            or is_evidence_action_controller_v40(config.arac_action)
            or is_car_w_family_action(config.arac_action)
        )
    return uses_resumable_phase_i_state_during_run(config.arac_action)


def trajectory_action_name_for_backend(config: SmokeConfig) -> str:
    if config.search_state_backend == "diagonal_cma":
        return CONTINUE_DIAGONAL_SEARCH_STATE
    if config.search_state_backend == "phase_i_mmes":
        return RESUME_PHASE_I_SEARCH_STATE
    raise ValueError(f"unsupported search_state_backend: {config.search_state_backend}")


def scheduled_search_state_hold_fes(
    config: SmokeConfig,
    state: SearchStateSchedulerState,
    *,
    overlap_edge_count: int | None = None,
) -> int:
    if not uses_scheduled_search_state(config) or state.phase == SEARCH_STATE_BLOCKED:
        return 0
    if overlap_edge_count is not None and int(overlap_edge_count) <= 0:
        return 0
    if config.search_state_backend == "diagonal_cma":
        return int(math.ceil(config.max_fes * FIRST_PROBE_FRACTION))
    return int(
        math.ceil(
            config.max_fes * (CC_RESERVE_FRACTION + FIRST_PROBE_FRACTION)
        )
    )


def is_cc_harm_guarded_sep_refresh_action(action_name: str) -> bool:
    return action_name == CC_HARM_GUARDED_SEP_REFRESH_ACTION


def is_separable_cmaes_dispatch_action(action_name: str) -> bool:
    return action_name == SEPARABLE_CMAES_DISPATCH_ACTION


def is_search_state_action(action_name: str) -> bool:
    return (
        is_bipop_search_state_action(action_name)
        or is_phase_rescue_multistart_action(action_name)
        or is_cc_harm_guarded_sep_refresh_action(action_name)
        or is_separable_cmaes_dispatch_action(action_name)
        or action_name
        in {RESUME_PHASE_I_SEARCH_STATE, CONTINUE_DIAGONAL_SEARCH_STATE}
        or is_evidence_action_controller(action_name)
    )


def overlap_action_name_for_lane(action_name: str) -> str:
    if action_name in {REPAIR_PROTECT_REFINE_ACTION, REPAIR_PROTECT_DEEP_REFINE_ACTION}:
        return "repair_shared_variable_binding"
    if action_name in {REPAIR_BIPOP_SEARCH_STATE_ACTION, REPAIR_PHASE_RESCUE_MULTISTART_ACTION}:
        return "repair_shared_variable_binding"
    if is_cc_harm_guarded_sep_refresh_action(action_name):
        return "conservative_no_action"
    if is_separable_cmaes_dispatch_action(action_name):
        return "conservative_no_action"
    if action_name in {
        SEARCH_STATE_BIPOP_ACTION,
        PHASE_RESCUE_MULTISTART_ACTION,
        RESUME_PHASE_I_SEARCH_STATE,
        CONTINUE_DIAGONAL_SEARCH_STATE,
    }:
        return "conservative_no_action"
    if is_evidence_action_controller(action_name):
        return "conservative_no_action"
    return action_name


def refine_sigma_for_action(
    action_name: str,
    base_sigma: float,
    *,
    controller_v31_run_state: EvidenceActionControllerV31RunState | None = None,
) -> float:
    if action_name == REPAIR_PROTECT_DEEP_REFINE_ACTION:
        return float(base_sigma) * REPAIR_PROTECT_DEEP_REFINE_SIGMA_MULTIPLIER
    if action_name in {REPAIR_PROTECT_REFINE_ACTION, REPAIR_PHASE_RESCUE_MULTISTART_ACTION}:
        return float(base_sigma) * REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER
    if uses_post_retirement_precision_reanchor(
        action_name,
        controller_v31_run_state,
    ):
        return float(base_sigma) * REPAIR_PROTECT_DEEP_REFINE_SIGMA_MULTIPLIER
    if (
        (
            is_evidence_action_controller_v31(action_name)
            or is_evidence_action_controller_v32(action_name)
            or is_evidence_action_controller_v33(action_name)
            or is_evidence_action_controller_v34(action_name)
            or is_evidence_action_controller_v35(action_name)
            or is_evidence_action_controller_v36(action_name)
            or is_evidence_action_controller_v37(action_name)
            or is_evidence_action_controller_v38(action_name)
            or is_evidence_action_controller_v39(action_name)
            or is_evidence_action_controller_v40(action_name)
            or is_car_w_family_action(action_name)
        )
        and controller_v31_run_state is not None
        and not controller_v31_run_state.dense_overlap
    ):
        return float(base_sigma) * REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER
    return float(base_sigma)


def uses_post_retirement_precision_reanchor(
    action_name: str,
    controller_run_state: EvidenceActionControllerV31RunState | None,
) -> bool:
    return bool(
        (
            is_evidence_action_controller_v38(action_name)
            or is_evidence_action_controller_v39(action_name)
            or is_evidence_action_controller_v40(action_name)
        )
        and controller_run_state is not None
        and controller_run_state.v38_enabled
        and not controller_run_state.dense_overlap
        and controller_run_state.phase_rescue_retired
    )


def should_trigger_bipop_restart(
    *,
    stagnation_count: int,
    cooldown_remaining: int,
    escape_budget: int,
) -> bool:
    return (
        int(stagnation_count) >= BIPOP_STAGNATION_WINDOW
        and int(cooldown_remaining) <= 0
        and int(escape_budget) > 0
    )


def bipop_relative_improvement(candidate_best: float, incumbent_fitness: float) -> float:
    denominator = max(abs(float(incumbent_fitness)), 1e-12)
    return max(0.0, (float(incumbent_fitness) - float(candidate_best)) / denominator)


def should_accept_bipop_restart(
    *,
    candidate_best: float,
    incumbent_fitness: float,
    min_relative_improvement: float = BIPOP_ACCEPT_RELATIVE_IMPROVEMENT,
) -> bool:
    return bipop_relative_improvement(candidate_best, incumbent_fitness) >= float(min_relative_improvement)


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


def build_bipop_restart_plan(
    *,
    group_index: int,
    restart_count: int,
    base_population_size: int,
    base_sigma: float,
    base_budget: int,
    remaining_fes: int,
    rng: np.random.Generator,
) -> BipopRestartPlan:
    base_population = max(2, int(base_population_size))
    if restart_count % 2 == 0:
        population_size = base_population * BIPOP_LARGE_POPULATION_MULTIPLIER
        sigma = float(base_sigma) * BIPOP_LARGE_SIGMA_MULTIPLIER
        restart_mode = "large_ipop"
    else:
        upper_small_population = max(BIPOP_MIN_POPULATION_SIZE, base_population)
        population_size = int(
            rng.integers(BIPOP_MIN_POPULATION_SIZE, upper_small_population + 1)
        )
        sigma_multiplier = float(
            rng.uniform(BIPOP_MIN_SIGMA_MULTIPLIER, BIPOP_LARGE_SIGMA_MULTIPLIER)
        )
        sigma = float(base_sigma) * sigma_multiplier
        restart_mode = "small_bipop"
    requested_budget = max(
        population_size,
        int(math.ceil(max(base_budget, population_size) * BIPOP_ESCAPE_BUDGET_FRACTION)),
    )
    escape_budget = bounded_population_budget(
        requested_fes=requested_budget,
        remaining_fes=remaining_fes,
        population_size=population_size,
    )
    return BipopRestartPlan(
        restart_mode=restart_mode,
        population_size=population_size,
        sigma=sigma,
        escape_budget=escape_budget,
    )


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


def cc_harm_conflict_fraction(fitness_deltas: list[float], reference_fitness: float) -> float:
    if len(fitness_deltas) <= 1:
        return 0.0
    threshold = max(BIPOP_STAGNATION_EPSILON, abs(float(reference_fitness)) * 1e-10)
    conflicts = 0
    for left, right in zip(fitness_deltas, fitness_deltas[1:]):
        left_active = float(left) > threshold
        right_active = float(right) > threshold
        if left_active != right_active:
            conflicts += 1
    return conflicts / max(1, len(fitness_deltas) - 1)


def phase_i_tail_utility(state: MMESState) -> float:
    window = state.recent_best[-3:]
    if len(window) < 2:
        return 0.0
    start_fe, start_best = window[0]
    end_fe, end_best = window[-1]
    return normalized_gain_utility(start_best, end_best, end_fe - start_fe)


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


def build_search_state_evidence(
    *,
    complete_sweep: bool,
    overlap_degree: float,
    phase_rescue_enabled: bool,
    repair_lock_active: bool,
    phase_i_tail_utility_value: float,
    relations: list[OverlapRelation],
    decisions: list[RelationActionDecision],
    writeback_norms: list[float],
    relative_writeback_norms: list[float],
    fitness_deltas: list[float],
    reference_fitness: float,
    cc_utility_history: list[float],
    remaining_fes: int,
    max_fes: int,
    population_size: int,
) -> SearchStateEvidence:
    canonical_actions = [
        _canonical_relation_action_name(decision) for decision in decisions
    ]
    non_coordinate = sum(
        action != "allow_beneficial_coordination" for action in canonical_actions
    ) / max(1, len(canonical_actions))
    active_intervention_actions = {
        "isolate_conflicting_relation",
        "repair_shared_variable_binding",
        "protect_high_margin_group",
    }
    active_intervention = sum(
        action in active_intervention_actions for action in canonical_actions
    ) / max(1, len(canonical_actions))
    conflict = cc_harm_conflict_fraction(fitness_deltas, reference_fitness)
    unstable = any(
        abs(float(norm)) > CC_HARM_WRITEBACK_NORM for norm in writeback_norms
    )
    relative_max = max(
        (max(0.0, float(norm)) for norm in relative_writeback_norms),
        default=0.0,
    )
    return SearchStateEvidence(
        complete_sweep=bool(complete_sweep and relations),
        overlap_degree=float(overlap_degree),
        phase_rescue_enabled=bool(phase_rescue_enabled),
        repair_lock_active=bool(repair_lock_active),
        phase_i_tail_utility=float(phase_i_tail_utility_value),
        non_coordinate_fraction=float(non_coordinate),
        conflict_fraction=float(conflict),
        writeback_unstable=bool(unstable),
        recent_cc_utilities=tuple(float(value) for value in cc_utility_history[-2:]),
        remaining_fes=int(remaining_fes),
        max_fes=int(max_fes),
        population_size=int(population_size),
        active_intervention_fraction=float(active_intervention),
        relative_writeback_max=float(relative_max),
        relative_writeback_unstable=(
            relative_max >= RELATIVE_WRITEBACK_UNSTABLE_THRESHOLD
        ),
    )


def run_resumed_phase_i_state_block(
    *,
    optimizer,
    state: MMESState,
    requested_fes: int,
    guard_individual: np.ndarray,
    guard_fitness: float,
    fun,
) -> tuple[MMESState, bool, np.ndarray, float, MMESBlockResult]:
    evaluations_before = current_fitness_evaluations(fun)
    block = optimizer.run_block(state, requested_fes)
    observed_fes = current_fitness_evaluations(fun) - evaluations_before
    if observed_fes != block.actual_fes:
        raise RuntimeError("stateful MMES FE mismatch")
    if int(block.actual_fes) < 0 or int(block.actual_fes) > max(0, int(requested_fes)):
        raise RuntimeError("stateful MMES exceeded requested FE budget")

    candidate = np.asarray(block.state.best_so_far_x, dtype=float).reshape(-1)
    candidate_fitness = float(block.state.best_so_far_y)
    guard = np.asarray(guard_individual, dtype=float).reshape(-1)
    if candidate.shape != guard.shape or not np.all(np.isfinite(candidate)):
        raise RuntimeError("stateful MMES returned invalid candidate")
    if not math.isfinite(candidate_fitness):
        raise RuntimeError("stateful MMES returned non-finite fitness")

    accepted = candidate_fitness < float(guard_fitness)
    if accepted:
        return block.state, True, candidate.copy(), candidate_fitness, block
    return block.state, False, guard.copy(), float(guard_fitness), block


def should_trigger_cc_harm_guard(
    *,
    fitness_deltas: list[float],
    overlap_writeback_norms: list[float],
    reference_fitness: float,
    remaining_fes: int,
    minimum_refresh_budget: int,
) -> tuple[bool, str]:
    if len(fitness_deltas) < CC_HARM_MIN_GROUP_UPDATES:
        return False, "insufficient_group_updates"
    if remaining_fes < minimum_refresh_budget:
        return False, "insufficient_refresh_budget"

    reference = max(abs(float(reference_fitness)), 1.0)
    positive_gain = sum(max(0.0, float(delta)) for delta in fitness_deltas)
    stagnated_count = sum(
        1 for delta in fitness_deltas
        if group_delta_stagnated(float(delta), reference)
    )
    stagnated_fraction = stagnated_count / max(1, len(fitness_deltas))
    conflict_fraction = cc_harm_conflict_fraction(fitness_deltas, reference)
    writeback_unstable = any(
        abs(float(norm)) > CC_HARM_WRITEBACK_NORM for norm in overlap_writeback_norms
    )
    low_gain = positive_gain <= reference * CC_HARM_LOW_GAIN_RATIO
    severe_stagnation = stagnated_fraction >= CC_HARM_STAGNATED_FRACTION
    high_conflict = conflict_fraction >= CC_HARM_CONFLICT_FRACTION

    if low_gain and (severe_stagnation or high_conflict or writeback_unstable):
        reasons = ["low_cc_gain"]
        if severe_stagnation:
            reasons.append("severe_group_stagnation")
        if high_conflict:
            reasons.append("high_relation_conflict")
        if writeback_unstable:
            reasons.append("unstable_overlap_writeback")
        return True, "+".join(reasons)
    return False, "cc_harm_evidence_below_threshold"


def run_guarded_nda_continuation(
    *,
    fun,
    info: dict,
    config: SmokeConfig,
    fun_name: str,
    fun_id: int,
    outer_iter: int,
    guard_individual: np.ndarray,
    guard_fitness: float,
    remaining_fes: int,
    requested_fes: int | None = None,
    search_state_action: str = CC_HARM_GUARDED_SEP_REFRESH_ACTION,
) -> tuple[bool, np.ndarray, float, int, float]:
    population_size = calculate_cmaes_population_size(int(info["dimension"]))
    requested_budget = remaining_fes if requested_fes is None else min(
        remaining_fes,
        max(0, int(requested_fes)),
    )
    refresh_budget = bounded_population_budget(
        requested_fes=requested_budget,
        remaining_fes=remaining_fes,
        population_size=population_size,
    )
    if refresh_budget <= 0:
        return False, guard_individual.copy(), float(guard_fitness), 0, math.inf
    backend_budget = refresh_budget - population_size
    if backend_budget <= 0:
        return False, guard_individual.copy(), float(guard_fitness), 0, math.inf

    problem = {
        "fitness_function": fun,
        "ndim_problem": info["dimension"],
        "lower_boundary": info["lower"] * np.ones((info["dimension"],)),
        "upper_boundary": info["upper"] * np.ones((info["dimension"],)),
    }
    options = {
        "max_function_evaluations": backend_budget,
        "mean": (np.asarray(guard_individual, dtype=float).copy(),),
        "sigma": float(config.sigma) * CC_HARM_REFRESH_SIGMA_MULTIPLIER,
        "n_individuals": population_size,
        "is_restart": config.mmes_restart,
        "verbose": config.verbose,
        "arac_search_state_action": search_state_action,
        "arac_guard_source": "phase_i_or_current_incumbent",
    }
    if config.seed is not None:
        options["seed_rng"] = derive_optimizer_seed(
            config.seed,
            fun_name,
            fun_id,
            outer_iter + 1,
            23011,
        )
    evaluations_before = current_fitness_evaluations(fun)
    results = MMES(problem, options).optimize()
    candidate_best = float(results["best_so_far_y"])
    candidate = np.asarray(results["best_so_far_x"], dtype=float).reshape(-1)
    reported_fes = int(results["n_function_evaluations"])
    if reported_fes < 0 or reported_fes > refresh_budget:
        raise RuntimeError("guarded NDA reported invalid FE usage")
    observed_fes = current_fitness_evaluations(fun) - evaluations_before
    if observed_fes < 0 or observed_fes > refresh_budget:
        raise RuntimeError("guarded NDA exceeded objective FE budget")
    used_fes = observed_fes if hasattr(fun, "fitness_record") else reported_fes
    if not math.isfinite(candidate_best):
        raise RuntimeError("guarded NDA returned non-finite fitness")
    guard_shape = np.asarray(guard_individual).reshape(-1).shape
    if candidate.shape != guard_shape:
        raise RuntimeError("guarded NDA returned invalid candidate shape")
    if not np.all(np.isfinite(candidate)):
        raise RuntimeError("guarded NDA returned non-finite candidate")
    accepted = candidate_best < float(guard_fitness)
    if accepted:
        return (
            True,
            candidate.copy(),
            candidate_best,
            used_fes,
            candidate_best,
        )
    return (
        False,
        guard_individual.copy(),
        float(guard_fitness),
        used_fes,
        candidate_best,
    )


def run_direct_separable_cmaes_dispatch(
    *,
    fun,
    info: dict,
    config: SmokeConfig,
    fun_name: str,
    fun_id: int,
    initial_mean: np.ndarray | None = None,
    incumbent_fitness: float | None = None,
    max_function_evaluations: int | None = None,
) -> dict[str, object]:
    dimension = int(info["dimension"])
    lower = float(info["lower"]) * np.ones((dimension,))
    upper = float(info["upper"]) * np.ones((dimension,))
    if initial_mean is None:
        mean = np.zeros((dimension,), dtype=float)
    else:
        raw_mean = np.asarray(initial_mean, dtype=float).reshape(-1)
        if raw_mean.size != dimension:
            raise ValueError(
                f"initial_mean dimension mismatch: expected {dimension}, got {raw_mean.size}"
            )
        mean = np.clip(raw_mean, lower, upper)
    population_size = calculate_cmaes_population_size(dimension)
    evaluation_budget = int(
        config.max_fes if max_function_evaluations is None else max_function_evaluations
    )
    if incumbent_fitness is None or not math.isfinite(float(incumbent_fitness)):
        raise ValueError("direct separable CMA dispatch requires a finite incumbent")
    optimizer_seed = derive_optimizer_seed(
        config.seed if config.seed is not None else 0,
        fun_name,
        fun_id,
        0,
        47011,
    )
    state = initialize_diagonal_cma_state(
        initial_mean=mean,
        sigma=SEPARABLE_CMAES_INITIAL_SIGMA,
        lower=lower,
        upper=upper,
        seed=optimizer_seed,
        population_size=population_size,
        incumbent_fitness=float(incumbent_fitness),
    )
    block = run_diagonal_cma_block(
        state,
        fun,
        requested_fes=evaluation_budget,
    )
    strategy_stds = np.asarray(state.strategy.stds, dtype=float).reshape(-1)

    return {
        "best_so_far_x": state.best_x.copy(),
        "best_so_far_y": float(state.best_y),
        "n_function_evaluations": int(block.actual_fes),
        "population_size": population_size,
        "sigma_mean": float(np.mean(strategy_stds)),
        "sigma_max": float(np.max(strategy_stds)),
        "success": bool(np.isfinite(state.best_y)),
        "optimizer_seed": optimizer_seed,
        "state_fingerprint_before": block.state_fingerprint_before,
        "state_fingerprint_after": block.state_fingerprint_after,
    }


def run_diagonal_search_state_block(
    *,
    state: DiagonalCMAState | None,
    requested_fes: int,
    guard_individual: np.ndarray,
    guard_fitness: float,
    fun,
    info: dict,
    config: SmokeConfig,
    fun_name: str,
    fun_id: int,
    outer_iter: int,
):
    dimension = int(info["dimension"])
    guard = np.asarray(guard_individual, dtype=float).reshape(-1)
    if guard.shape != (dimension,):
        raise ValueError("guard_individual dimension mismatch")
    optimizer_seed = derive_optimizer_seed(
        config.seed if config.seed is not None else 0,
        fun_name,
        fun_id,
        outer_iter,
        32011,
    )
    if state is None:
        lower = float(info["lower"]) * np.ones(dimension)
        upper = float(info["upper"]) * np.ones(dimension)
        search_mean = np.clip(guard, lower, upper)
        state = initialize_diagonal_cma_state(
            initial_mean=search_mean,
            sigma=float(config.sigma),
            lower=lower,
            upper=upper,
            seed=optimizer_seed,
            population_size=calculate_cmaes_population_size(dimension),
            incumbent_fitness=float(guard_fitness),
        )
        state.best_x = guard.copy()
        state.best_y = float(guard_fitness)
    elif float(guard_fitness) < float(state.best_y):
        state.best_x = guard.copy()
        state.best_y = float(guard_fitness)

    evaluations_before = current_fitness_evaluations(fun)
    block = run_diagonal_cma_block(
        state,
        fun,
        requested_fes=requested_fes,
    )
    if hasattr(fun, "fitness_record"):
        observed_fes = current_fitness_evaluations(fun) - evaluations_before
        if observed_fes != int(block.actual_fes):
            raise RuntimeError(
                "diagonal search-state FE mismatch: "
                f"observed={observed_fes}, reported={block.actual_fes}"
            )
    accepted = float(block.best_after) < float(guard_fitness)
    if accepted:
        candidate = np.asarray(block.state.best_x, dtype=float).reshape(-1).copy()
        candidate_fitness = float(block.best_after)
    else:
        candidate = guard.copy()
        candidate_fitness = float(guard_fitness)
    return (
        block.state,
        accepted,
        candidate,
        candidate_fitness,
        block,
        optimizer_seed,
    )


def is_trajectory_action(action_name: str) -> bool:
    return action_name in TRAJECTORY_ACTION_NAMES


def uses_trajectory_budget_shift(action_name: str) -> bool:
    return action_name in {"budget_shift_mean_blend", "budget_shift_only"}


def uses_trajectory_mean_blend(action_name: str) -> bool:
    return action_name in {"budget_shift_mean_blend", "mean_blend_only"}


def has_sufficient_trajectory_credit(contribution_credit: list[float]) -> bool:
    return sum(1 for value in contribution_credit if float(value) > 0.0) >= TRAJECTORY_MIN_POSITIVE_CREDIT_GROUPS


def calculate_group_overlap_support(
    grouping_result: list[list[int]],
    overlapping_elements: list,
) -> list[float]:
    support = [0.0 for _ in grouping_result]
    for left_index, shared in enumerate(overlapping_elements):
        right_index = left_index + 1
        if right_index >= len(grouping_result):
            break
        shared_count = len(shared) if not isinstance(shared, (int, np.integer)) else 1
        if shared_count <= 0:
            continue
        support[left_index] += shared_count / max(1, len(grouping_result[left_index]))
        support[right_index] += shared_count / max(1, len(grouping_result[right_index]))
    return support


def allocate_trajectory_group_budgets(
    total_budget: int,
    population_sizes: list[int],
    overlap_support: list[float],
    contribution_credit: list[float] | None = None,
) -> list[int]:
    group_count = len(population_sizes)
    if group_count == 0 or total_budget <= 0:
        return []
    min_budgets = [max(0, int(size)) for size in population_sizes]
    if total_budget <= sum(min_budgets):
        return _integer_weighted_split(total_budget, [1.0] * group_count)
    signal = overlap_support
    if contribution_credit is not None:
        signal = [
            max(0.0, float(overlap)) * max(0.0, float(credit))
            for overlap, credit in zip(overlap_support, contribution_credit)
        ]
    mean_signal = sum(signal) / max(1, len(signal))
    if mean_signal <= 0.0:
        weights = [1.0] * group_count
    else:
        weights = [
            max(
                0.25,
                1.0
                + TRAJECTORY_BUDGET_SHIFT_STRENGTH
                * ((float(value) / mean_signal) - 1.0),
            )
            for value in signal
        ]
    leftover = total_budget - sum(min_budgets)
    extras = _integer_weighted_split(leftover, weights)
    return [base + extra for base, extra in zip(min_budgets, extras)]


def _integer_weighted_split(total: int, weights: list[float]) -> list[int]:
    if not weights:
        return []
    if total <= 0:
        return [0 for _ in weights]
    safe_weights = [max(0.0, float(weight)) for weight in weights]
    weight_sum = sum(safe_weights)
    if weight_sum <= 0.0:
        safe_weights = [1.0 for _ in weights]
        weight_sum = float(len(weights))
    raw = [total * weight / weight_sum for weight in safe_weights]
    values = [int(math.floor(value)) for value in raw]
    remainder = total - sum(values)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - values[index], safe_weights[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        values[index] += 1
    return values


def blend_trajectory_mean(
    base_mean: np.ndarray,
    dims: list[int],
    variable_mean_cache: dict[int, float],
    lower: float,
    upper: float,
    strength: float = TRAJECTORY_MEAN_BLEND_STRENGTH,
) -> tuple[np.ndarray, int, float]:
    blended = np.asarray(base_mean, dtype=float).copy()
    before = blended.copy()
    applied_count = 0
    blend_weight = float(np.clip(strength, 0.0, 1.0))
    for local_index, variable_index in enumerate(dims):
        cached = variable_mean_cache.get(int(variable_index))
        if cached is None or not np.isfinite(cached):
            continue
        blended[local_index] = (
            (1.0 - blend_weight) * blended[local_index]
            + blend_weight * float(cached)
        )
        applied_count += 1
    blended = np.clip(blended, lower, upper)
    return blended, applied_count, float(np.linalg.norm(blended - before))


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


def apply_search_state_candidate(
    *,
    context_individual: np.ndarray,
    guard_individual: np.ndarray,
    guard_fitness: float,
    candidate: np.ndarray,
    candidate_fitness: float,
    accepted: bool,
    quarantine_context: bool,
) -> tuple[np.ndarray, np.ndarray, float, bool, bool]:
    next_context = np.asarray(context_individual, dtype=float).copy()
    next_guard = np.asarray(guard_individual, dtype=float).copy()
    next_guard_fitness = float(guard_fitness)
    if not accepted:
        return next_context, next_guard, next_guard_fitness, False, False

    protected_candidate = np.asarray(candidate, dtype=float).copy()
    next_guard = protected_candidate.copy()
    next_guard_fitness = float(candidate_fitness)
    if quarantine_context:
        return next_context, next_guard, next_guard_fitness, True, False
    return protected_candidate, next_guard, next_guard_fitness, True, True


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
    if is_trajectory_action(action_name):
        return blend_overlap_values(
            previous_values=previous_values,
            current_values=current_values,
            previous_delta=previous_delta,
            current_delta=current_delta,
        )
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
    if is_separable_cmaes_dispatch_action(action_name):
        return "full_space_diagonal_search"
    if is_cc_harm_guarded_sep_refresh_action(action_name):
        return "guarded_incumbent_refresh"
    if is_bipop_search_state_action(action_name):
        return "search_state_bipop_restart"
    if is_trajectory_action(action_name):
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
    if is_separable_cmaes_dispatch_action(action_name):
        return "full_space_diagonal_separable_search_takeover"
    if is_cc_harm_guarded_sep_refresh_action(action_name):
        return "cc_harm_guarded_sep_or_nda_refresh"
    if is_bipop_search_state_action(action_name):
        return "optimizer_search_state_restart"
    if is_trajectory_action(action_name):
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
    if is_cc_harm_guarded_sep_refresh_action(action_name):
        return "1"
    if is_bipop_search_state_action(action_name):
        return "1"
    if is_trajectory_action(action_name):
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
    if is_cc_harm_guarded_sep_refresh_action(action_name):
        return "1"
    if is_bipop_search_state_action(action_name):
        return "1"
    if is_trajectory_action(action_name):
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
    if is_trajectory_action(action_name):
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
    shuffled_source_action: RelationActionDecision | None = None,
) -> RelationActionDecision:
    if not relation.shared_vars:
        return action
    if relation_policy_mode in {
        "rule",
        "adaptive_v2",
        "adaptive_v21",
        "adaptive_v22",
        "adaptive_v23",
        "adaptive_v24",
        "adaptive_v25",
        "adaptive_v26",
    }:
        return action
    if relation_policy_mode == "shuffled":
        source_action_name = action.relation_action_name
        shuffled_action_name = SHUFFLED_NEGATIVE_CONTROL_ACTIONS[source_action_name]
        return RelationActionDecision(
            relation_id=relation.relation_id,
            action_name=shuffled_action_name,
            action_family=RELATION_ACTION_FAMILIES[shuffled_action_name],
            confidence=action.confidence if shuffled_action_name != "fallback" else 0.0,
            trigger_reason=(
                "deterministic_shuffled_negative_control_from:"
                f"{source_action_name}"
            ),
        )
    if relation_policy_mode != "lagged":
        raise ValueError(f"unsupported relation policy mode: {relation_policy_mode}")
    source_action = shuffled_source_action or RelationActionDecision(
        relation_id=relation.relation_id,
        action_name="fallback",
        action_family="fallback",
        confidence=0.0,
        trigger_reason="first_relation_has_no_previous_rule_action",
    )
    return RelationActionDecision(
        relation_id=relation.relation_id,
        action_name=source_action.relation_action_name,
        action_family=source_action.action_family,
        confidence=source_action.confidence,
        trigger_reason=(
            "deterministic_lagged_relation_policy_from:"
            f"{source_action.relation_action_name}"
        ),
    )


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
    search_state_block_fe: int | None = None,
    search_state_utility: float | None = None,
    required_utility_ratio: float | None = None,
    state_action_fe: int | None = None,
    cc_reserve_fe: int | None = None,
    state_fingerprint_before: str = "",
    state_fingerprint_after: str = "",
    abstain_reason: str = "",
    search_state_evidence: SearchStateEvidence | None = None,
    pre_hold_evidence: PreHoldEvidence | None = None,
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
    cma_sigma_reference: float | None = None,
    cma_sigma_applied_factor: float | None = None,
    cma_sigma_terminal: float | None = None,
    cma_sigma_next_factor: float | None = None,
    cma_sigma_route: str = "",
    cma_restart_count: int | None = None,
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
        "search_state_block_fe": ""
        if search_state_block_fe is None
        else str(search_state_block_fe),
        "search_state_utility": ""
        if search_state_utility is None
        else f"{search_state_utility:.6e}",
        "required_utility_ratio": ""
        if required_utility_ratio is None
        else f"{required_utility_ratio:.6e}",
        "state_action_fe": "" if state_action_fe is None else str(state_action_fe),
        "cc_reserve_fe": "" if cc_reserve_fe is None else str(cc_reserve_fe),
        "state_fingerprint_before": state_fingerprint_before,
        "state_fingerprint_after": state_fingerprint_after,
        "abstain_reason": abstain_reason,
        "search_state_non_coordinate_fraction": (
            ""
            if search_state_evidence is None
            else f"{search_state_evidence.non_coordinate_fraction:.6e}"
        ),
        "search_state_active_intervention_fraction": (
            ""
            if search_state_evidence is None
            else f"{search_state_evidence.active_intervention_fraction:.6e}"
        ),
        "search_state_conflict_fraction": (
            ""
            if search_state_evidence is None
            else f"{search_state_evidence.conflict_fraction:.6e}"
        ),
        "search_state_writeback_unstable": (
            ""
            if search_state_evidence is None
            else str(int(search_state_evidence.writeback_unstable))
        ),
        "search_state_relative_writeback_max": (
            ""
            if search_state_evidence is None
            else f"{search_state_evidence.relative_writeback_max:.6e}"
        ),
        "search_state_relative_writeback_unstable": (
            ""
            if search_state_evidence is None
            else str(int(search_state_evidence.relative_writeback_unstable))
        ),
    }
    pre_hold_values = {
        "phase_i_tail_utility": "{:.6e}",
        "group_count": "{}",
        "mean_group_size": "{:.6e}",
        "overlap_edge_count": "{}",
        "overlap_edge_fraction": "{:.6e}",
        "shared_variable_count": "{}",
        "shared_variable_ratio": "{:.6e}",
        "mean_overlap_width": "{:.6e}",
        "remaining_fes": "{}",
        "remaining_ratio": "{:.6e}",
        "scheduled_hold_fes": "{}",
        "projected_unheld_group_fes": "{}",
        "projected_held_group_fes": "{}",
        "budget_retention_ratio": "{:.6e}",
    }
    for name, value_format in pre_hold_values.items():
        row[f"pre_hold_{name}"] = (
            ""
            if pre_hold_evidence is None
            else value_format.format(getattr(pre_hold_evidence, name))
        )
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
            "cma_sigma_reference": ""
            if cma_sigma_reference is None
            else f"{cma_sigma_reference:.6e}",
            "cma_sigma_applied_factor": ""
            if cma_sigma_applied_factor is None
            else f"{cma_sigma_applied_factor:.6e}",
            "cma_sigma_terminal": ""
            if cma_sigma_terminal is None
            else f"{cma_sigma_terminal:.6e}",
            "cma_sigma_next_factor": ""
            if cma_sigma_next_factor is None
            else f"{cma_sigma_next_factor:.6e}",
            "cma_sigma_route": cma_sigma_route,
            "cma_restart_count": ""
            if cma_restart_count is None
            else str(cma_restart_count),
        }
    )
    row.update({field: "" for field in COMPONENT_CREDIT_TRACE_FIELDS})
    return row


def _write_action_trace(
    path: Path,
    rows: list[dict[str, str]],
    *,
    include_trust_fields: bool = True,
    include_recovery_fields: bool = False,
    include_maturity_fields: bool = False,
    include_resource_fields: bool = False,
    include_cma_sigma_fields: bool = False,
    include_component_credit_fields: bool = False,
) -> None:
    if include_component_credit_fields:
        fields = V40_ACTION_TRACE_FIELDS
    elif include_cma_sigma_fields:
        fields = V39_ACTION_TRACE_FIELDS
    elif include_resource_fields:
        fields = V37_ACTION_TRACE_FIELDS
    elif include_recovery_fields:
        fields = V34_ACTION_TRACE_FIELDS
    elif include_maturity_fields:
        fields = V36_ACTION_TRACE_FIELDS
    elif include_trust_fields:
        fields = V33_ACTION_TRACE_FIELDS
    else:
        fields = LEGACY_ACTION_TRACE_FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
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


def apply_relation_action_with_controller_v33(
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
]:
    """Apply v31 guards followed by v33 runtime-only trust damping."""

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
    return apply_v33_guard_to_executed_relation(
        relation=relation,
        executed_action=executed_action,
        adjusted_values=adjusted_values,
        action_value_delta_norm=action_value_delta_norm,
        current_values=current_values,
        controller_run_state=controller_run_state,
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


def apply_relation_action_with_controller_v35(
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
]:
    """Apply v31 active actions with the v33 topology fallback guard only."""

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
    adjusted_values, action_value_delta_norm, fallback_route = (
        apply_topology_scoped_fallback_guard(
            executed_action=executed_action,
            adjusted_values=adjusted_values,
            action_value_delta_norm=action_value_delta_norm,
            current_values=current_values,
            controller_run_state=controller_run_state,
        )
    )
    return (
        executed_action,
        adjusted_values,
        action_value_delta_norm,
        None,
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
    row = {field: str(raw.get(field, "")) for field in OVERLAP_RELATION_FIELDS}
    row["shared_vars"] = _format_shared_vars(relation.shared_vars)
    for field in (
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
        row[field] = f"{float(raw.get(field, 0.0)):.6f}"
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
    separable_continuation_fe: int = 0,
) -> None:
    budget_aligned_fe = min(max_fes, fitness_record_fe)
    stage_fe = (
        global_phase_fe
        + cc_phase_fe
        + rescue_fe
        + refresh_fe
        + search_state_fe
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
        writer = csv.DictWriter(handle, fieldnames=BUDGET_SUMMARY_FIELDS)
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


def _write_car_rows(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _car_controller_state_payload(
    controller: EvidenceActionControllerV31RunState | None,
    *,
    trajectory_mean_cache: dict[int, float],
    previous_group_contribution_credit: list[float],
) -> dict[str, object]:
    if controller is None:
        controller_payload: dict[str, object] = {}
    else:
        policy = controller.action_trust_policy
        trust_states = (
            {}
            if policy is None
            else {
                key: asdict(state)
                for key, state in sorted(policy._states.items())
            }
        )
        controller_payload = {
            "dense_overlap": bool(controller.dense_overlap),
            "locked_policy_mode": controller.locked_policy_mode,
            "non_dense_repair_locked": bool(controller.non_dense_repair_locked),
            "non_dense_repair_lock_trigger": controller.non_dense_repair_lock_trigger,
            "search_state_scheduler_state": asdict(
                controller.search_state_scheduler_state
            ),
            "trust_states": trust_states,
            "phase_i_state_fingerprint": (
                ""
                if controller.phase_i_state is None
                else controller.phase_i_state.fingerprint()
            ),
            "diagonal_cma_state_present": controller.diagonal_cma_state is not None,
        }
    return {
        "controller": controller_payload,
        "trajectory_mean_cache": {
            str(key): float(value)
            for key, value in sorted(trajectory_mean_cache.items())
        },
        "previous_group_contribution_credit": [
            float(value) for value in previous_group_contribution_credit
        ],
    }


def _run_car_cma_group(
    *,
    evaluator,
    background: np.ndarray,
    dims: tuple[int, ...],
    requested_fes: int,
    population_size: int,
    seed: int,
    info: dict[str, object],
    config: SmokeConfig,
) -> GroupOptimizationResult:
    objective = lambda x_batch: evaluator(combine(x_batch, background, list(dims)))
    problem = {
        "fitness_function": objective,
        "ndim_problem": len(dims),
        "lower_boundary": float(info["lower"]) * np.ones((len(dims),)),
        "upper_boundary": float(info["upper"]) * np.ones((len(dims),)),
    }
    options = {
        "max_function_evaluations": int(requested_fes),
        "mean": (np.asarray(background, dtype=float)[list(dims)].copy(),),
        "sigma": config.sigma,
        "n_individuals": int(population_size),
        "is_restart": config.cmaes_restart,
        "verbose": config.verbose,
        "early_stopping_evaluations": config.early_stopping_evaluations,
        "seed_rng": int(seed),
    }
    evaluations_before = current_fitness_evaluations(evaluator)
    result = CMAES(problem, options).optimize()
    actual_fes = current_fitness_evaluations(evaluator) - evaluations_before
    reported_fes = int(result["n_function_evaluations"])
    if actual_fes != reported_fes:
        raise RuntimeError(
            "CAR branch objective FE does not match CMA reported FE: "
            f"observed={actual_fes}, reported={reported_fes}"
        )
    return GroupOptimizationResult(
        best_x=np.asarray(result["best_so_far_x"], dtype=float).reshape(-1).copy(),
        best_y=float(result["best_so_far_y"]),
        actual_fes=actual_fes,
    )


@dataclass(frozen=True)
class CARBarrierResult:
    adopted_state: BranchState | None
    accounting_record: tuple[float, ...]
    probe_fe: int
    probe_trace_rows: tuple[dict[str, str], ...]
    state_ledger_rows: tuple[dict[str, str], ...]
    branch_manifest_rows: tuple[dict[str, str], ...]
    abstain_reason: str


@dataclass(frozen=True)
class CARActionabilityArmResult:
    state: BranchState | None
    accounting_record: tuple[float, ...]
    actual_fe: int
    trace_base_row: dict[str, str]
    abstain_reason: str


def _fitness_record_sha256(values: tuple[float, ...] | list[float]) -> str:
    normalized = tuple(float(value) for value in values)
    return hashlib.sha256(repr(normalized).encode("utf-8")).hexdigest()


def execute_car_actionability_arm_at_barrier(
    *,
    decision: CARPlanDecision,
    checkpoint: BranchState,
    checkpoint_fe: int,
    prefix_record: tuple[float, ...],
    fun_name: str,
    fun_id: int,
    output_path: Path,
    info: dict[str, object],
    config: SmokeConfig,
    problem_id: str,
) -> CARActionabilityArmResult:
    """Execute one offline audit arm from the frozen CAR-W3 checkpoint.

    Independent subprocess lanes provide the two long continuations.  This
    helper applies the candidate writeback only in the candidate lane's first
    component closure; both lanes then return to the same canonical runtime.
    """

    arm = config.car_actionability_arm
    if arm not in {"fallback", "candidate"}:
        raise ValueError("CAR actionability execution requires fallback or candidate arm")
    prefix_hash = _fitness_record_sha256(prefix_record)
    plan = decision.plan
    evidence = decision.evidence

    def base_row(*, reason: str, plan_status: str) -> dict[str, str]:
        return {
            "protocol_version": CAR_ACTIONABILITY_PROTOCOL_VERSION,
            "fresh_optimizer_execution": "1",
            "problem_id": problem_id,
            "seed": "" if config.seed is None else str(int(config.seed)),
            "audit_arm": arm,
            "candidate_mode": config.car_candidate_mode,
            "horizon_index": "",
            "horizon_label": "",
            "checkpoint_fe": str(int(checkpoint_fe)),
            "checkpoint_fitness": f"{checkpoint.committed_fitness:.17e}",
            "configured_max_fes": str(int(config.max_fes)),
            "terminal_completion_tolerance_fe": "",
            "termination_reason": "",
            "terminal_fe_shortfall": "",
            "target_fe": "",
            "observed_fe": "",
            "best_error": "",
            "prefix_state_fingerprint": checkpoint.state_fingerprint,
            "prefix_record_sha256": prefix_hash,
            "post_intervention_state_fingerprint": "",
            "graph_fingerprint": "" if evidence is None else evidence.graph_fingerprint,
            "component_fingerprint": (
                "" if evidence is None else evidence.component_fingerprint
            ),
            "candidate_action_name": (
                "" if evidence is None else evidence.candidate_action_name
            ),
            "candidate_action_family": (
                "" if evidence is None else evidence.candidate_action_family
            ),
            "candidate_action_applied": "0",
            "requested_fe": "0",
            "actual_fe": "0",
            "seed_descriptor": "",
            "probe_seed": "",
            "intervention_record_sha256": "",
            "fitness_prefix_sha256": "",
            "plan_status": plan_status,
            "horizon_status": "",
            "abstain_reason": reason,
        }

    if plan is None or evidence is None:
        reason = decision.abstain_reason or "missing_car_writeback_plan"
        return CARActionabilityArmResult(
            state=None,
            accounting_record=(),
            actual_fe=0,
            trace_base_row=base_row(reason=reason, plan_status="not_applicable"),
            abstain_reason=reason,
        )

    arm_cap = int(math.floor(config.max_fes * CAR_W_PROBE_BUDGET_FRACTION)) // (
        2 * CAR_W_PAIR_COUNT
    )
    remaining_fe = max(0, int(config.max_fes) - int(checkpoint_fe))
    budgets = allocate_component_horizon_budgets(
        max_arm_fes=min(arm_cap, remaining_fe),
        population_sizes=plan.group_population_sizes,
    )
    if not budgets:
        reason = "audit_budget_cannot_fit_complete_component_horizon"
        return CARActionabilityArmResult(
            state=None,
            accounting_record=(),
            actual_fe=0,
            trace_base_row=base_row(reason=reason, plan_status="abstain"),
            abstain_reason=reason,
        )

    requested_fe = 1 + sum(budgets)
    evaluator = Benchmark(
        str(output_path) + "/",
        data_dir=config.aob_data_root,
    ).get_function(fun_name, fun_id)
    seed_descriptor = derive_probe_seed(
        base_seed=0 if config.seed is None else int(config.seed),
        sweep_index=evidence.evidence_sweep_count,
        component_fingerprint=evidence.component_fingerprint,
        pair_index=0,
    )
    candidate_plan = (
        shuffled_component_writeback_plan(plan)
        if config.car_candidate_mode == "shuffled_graph"
        else plan
    )
    apply_candidate = (
        arm == "candidate" and config.car_candidate_mode != "paired_fallback"
    )
    state = run_component_horizon(
        checkpoint=checkpoint.clone(),
        evaluator=evaluator,
        seed_descriptor=seed_descriptor,
        requested_fes=requested_fe,
        plan=candidate_plan if apply_candidate else plan,
        apply_candidate=apply_candidate,
        optimize_group=lambda **kwargs: _run_car_cma_group(
            **kwargs,
            info=info,
            config=config,
        ),
    )
    record = tuple(float(value) for value in state.evaluator_record)
    if len(record) != requested_fe:
        raise RuntimeError("CAR actionability arm FE does not match reservation")
    row = base_row(reason="", plan_status="applied")
    row.update(
        {
            "post_intervention_state_fingerprint": state.state_fingerprint,
            "candidate_action_applied": "1" if apply_candidate else "0",
            "requested_fe": str(requested_fe),
            "actual_fe": str(len(record)),
            "seed_descriptor": seed_descriptor.canonical_key,
            "probe_seed": str(seed_descriptor.seed),
            "intervention_record_sha256": _fitness_record_sha256(record),
        }
    )
    return CARActionabilityArmResult(
        state=state.clone(),
        accounting_record=record,
        actual_fe=len(record),
        trace_base_row=row,
        abstain_reason="",
    )


def finalize_car_actionability_trace(
    *,
    trace_base_row: dict[str, str],
    fitness_record: list[float],
    max_fes: int,
) -> list[dict[str, str]]:
    """Materialize exact-FE nested and terminal labels after one fresh lane."""

    record = tuple(float(value) for value in fitness_record)
    rows: list[dict[str, str]] = []
    checkpoint_fe = int(trace_base_row.get("checkpoint_fe") or 0)
    intervention_fe = int(trace_base_row.get("actual_fe") or 0)
    plan_applied = trace_base_row.get("plan_status") == "applied"
    try:
        terminal_tolerance = int(
            str(trace_base_row.get("terminal_completion_tolerance_fe", "0"))
        )
    except ValueError:
        terminal_tolerance = 0
    # HCC can end adjacent lanes at different population boundaries. Retain
    # each natural endpoint in metadata, but pair the same absolute-FE prefix.
    closure_target = checkpoint_fe + intervention_fe
    terminal_target = max(
        closure_target,
        max(0, int(max_fes) - terminal_tolerance),
    )
    terminal_order_valid = not plan_applied or terminal_target > closure_target
    targets: list[tuple[int, str, int]] = []
    if plan_applied and intervention_fe > 0:
        for index, (multiplier, label) in enumerate(
            zip(
                CAR_ACTIONABILITY_HORIZON_MULTIPLIERS,
                CAR_ACTIONABILITY_HORIZON_LABELS,
                strict=True,
            )
        ):
            target = checkpoint_fe + multiplier * intervention_fe
            if (index == 0 and target <= terminal_target) or target < terminal_target:
                targets.append((index, label, target))
    targets.append((3, "terminal", terminal_target))

    for horizon_index, label, target_fe in targets:
        observed_fe = min(target_fe, len(record))
        prefix = record[:observed_fe]
        if label == "terminal":
            shortfall = max(0, int(max_fes) - len(record))
            reason = trace_base_row.get("termination_reason", "")
            horizon_status = (
                "complete"
                if observed_fe == target_fe
                and shortfall <= terminal_tolerance
                and reason != "early_guard"
                and terminal_order_valid
                else "incomplete"
            )
        else:
            horizon_status = "complete" if observed_fe == target_fe else "incomplete"
        row = dict(trace_base_row)
        if label == "terminal" and not terminal_order_valid:
            row["abstain_reason"] = (
                "terminal_target_has_no_post_intervention_continuation"
            )
        row.update(
            {
                "horizon_index": str(horizon_index),
                "horizon_label": label,
                "target_fe": str(target_fe),
                "observed_fe": str(observed_fe),
                "best_error": (
                    "" if not prefix else f"{min(prefix):.17e}"
                ),
                "fitness_prefix_sha256": (
                    "" if not prefix else _fitness_record_sha256(prefix)
                ),
                "horizon_status": horizon_status,
                "terminal_fe_shortfall": (
                    str(max(0, int(max_fes) - len(record)))
                    if label == "terminal"
                    else trace_base_row.get("terminal_fe_shortfall", "")
                ),
            }
        )
        rows.append(row)
    return rows


def execute_car_w_probe_at_barrier(
    *,
    decision: CARPlanDecision,
    checkpoint: BranchState,
    checkpoint_fe: int,
    fun_name: str,
    fun_id: int,
    output_path: Path,
    info: dict[str, object],
    config: SmokeConfig,
    problem_id: str,
    branch_order: tuple[str, str] = ("fallback", "candidate"),
    early_futility_abort: bool = False,
) -> CARBarrierResult:
    if config.car_candidate_mode not in {"graph", "shuffled_graph", "paired_fallback"}:
        raise ValueError(
            "unsupported CAR candidate mode: " + str(config.car_candidate_mode)
        )
    audit = AuditEnvelope(
        run_id=config.run_id,
        problem_id=problem_id,
        seed=0 if config.seed is None else int(config.seed),
    )
    probe_limit = int(math.floor(config.max_fes * CAR_W_PROBE_BUDGET_FRACTION))
    plan = decision.plan
    evidence = decision.evidence
    if plan is None or evidence is None:
        reason = decision.abstain_reason or "missing_car_writeback_plan"
        row = {
            "problem_id": audit.problem_id,
            "seed": "" if config.seed is None else str(audit.seed),
            "graph_fingerprint": "",
            "component_fingerprint": "",
            "candidate_action_name": "",
            "candidate_action_family": "",
            "candidate_mode": config.car_candidate_mode,
            "evidence_sweeps": "0",
            "checkpoint_fe": str(checkpoint_fe),
            "probe_fe": "0",
            "total_fe_after_probe": str(checkpoint_fe),
            "probe_fe_limit": str(probe_limit),
            "adopted_branch": "not_probed",
            "committed_fitness": f"{checkpoint.committed_fitness:.17e}",
            "evaluated_elite": "",
            "state_fingerprint": checkpoint.state_fingerprint,
            "gate_result": "abstain",
            "abstain_reason": reason,
        }
        return CARBarrierResult(None, (), 0, (), (row,), (), reason)

    probe_arm_cap = probe_limit // (2 * CAR_W_PAIR_COUNT)
    remaining_total_fe = max(0, int(config.max_fes) - int(checkpoint_fe))
    remaining_arm_cap = remaining_total_fe // (2 * CAR_W_PAIR_COUNT)
    max_arm_fes = min(probe_arm_cap, remaining_arm_cap)
    budgets = allocate_component_horizon_budgets(
        max_arm_fes=max_arm_fes,
        population_sizes=plan.group_population_sizes,
    )
    if not budgets:
        reason = (
            "remaining_total_budget_cannot_fit_complete_component_horizon"
            if remaining_arm_cap < probe_arm_cap
            else "probe_budget_cannot_fit_complete_component_horizon"
        )
        row = {
            "problem_id": audit.problem_id,
            "seed": "" if config.seed is None else str(audit.seed),
            "graph_fingerprint": evidence.graph_fingerprint,
            "component_fingerprint": evidence.component_fingerprint,
            "candidate_action_name": evidence.candidate_action_name,
            "candidate_action_family": evidence.candidate_action_family,
            "candidate_mode": config.car_candidate_mode,
            "evidence_sweeps": str(evidence.evidence_sweep_count),
            "checkpoint_fe": str(checkpoint_fe),
            "probe_fe": "0",
            "total_fe_after_probe": str(checkpoint_fe),
            "probe_fe_limit": str(probe_limit),
            "adopted_branch": "not_probed",
            "committed_fitness": f"{checkpoint.committed_fitness:.17e}",
            "evaluated_elite": "",
            "state_fingerprint": checkpoint.state_fingerprint,
            "gate_result": "abstain",
            "abstain_reason": reason,
        }
        return CARBarrierResult(None, (), 0, (), (row,), (), reason)

    arm_fes = 1 + sum(budgets)
    ledger = CARBudgetLedger(
        max_fes=config.max_fes,
        probe_fe_limit=probe_limit,
        committed_fe=checkpoint_fe,
    )
    evaluator_factory = lambda: Benchmark(
        str(output_path) + "/",
        data_dir=config.aob_data_root,
    ).get_function(fun_name, fun_id)

    candidate_plan = (
        shuffled_component_writeback_plan(plan)
        if config.car_candidate_mode == "shuffled_graph"
        else plan
    )

    def transition(apply_candidate: bool):
        return lambda state, evaluator, seed_descriptor, requested_fes: run_component_horizon(
            checkpoint=state,
            evaluator=evaluator,
            seed_descriptor=seed_descriptor,
            requested_fes=requested_fes,
            plan=candidate_plan if apply_candidate else plan,
            apply_candidate=(
                apply_candidate and config.car_candidate_mode != "paired_fallback"
            ),
            optimize_group=lambda **kwargs: _run_car_cma_group(
                **kwargs,
                info=info,
                config=config,
            ),
        )

    execution = CARProbeExecutor(
        evaluator_factory=evaluator_factory,
        ledger=ledger,
        base_seed=0 if config.seed is None else int(config.seed),
        sweep_index=evidence.evidence_sweep_count,
        graph_fingerprint=evidence.graph_fingerprint,
        component_fingerprint=evidence.component_fingerprint,
        action_family=evidence.candidate_action_family,
        arm_fes=arm_fes,
    ).execute(
        initial_checkpoint=checkpoint,
        fallback_transition=transition(False),
        candidate_transition=transition(True),
        branch_order=branch_order,
        early_futility_abort=early_futility_abort,
    )
    abstain_reason = ";".join(execution.gate.abstain_reasons)
    probe_rows = tuple(
        {
            "problem_id": audit.problem_id,
            "seed": "" if config.seed is None else str(audit.seed),
            "pair_index": str(observation.pair_index),
            "channel": "writeback",
            "graph_fingerprint": observation.graph_fingerprint,
            "component_fingerprint": observation.component_fingerprint,
            "action_family": observation.action_family,
            "candidate_mode": config.car_candidate_mode,
            "fallback_fe": str(observation.fallback_fe),
            "candidate_fe": str(observation.candidate_fe),
            "seed_descriptor": observation.seed_descriptor.canonical_key,
            "probe_seed": str(observation.seed_descriptor.seed),
            "phase1_probe_fitness_before": (
                f"{observation.phase1_probe_fitness_before:.17e}"
            ),
            "fallback_after": f"{observation.fallback_after:.17e}",
            "candidate_after": f"{observation.candidate_after:.17e}",
            "normalized_delta": f"{observation.normalized_delta:.17e}",
            "lcb": f"{execution.gate.lcb:.17e}",
            "tail": f"{execution.gate.tail:.17e}",
            "gate_result": "commit" if execution.gate.committed else "abstain",
            "abstain_reason": abstain_reason,
        }
        for observation in execution.observations
    )
    manifest_rows = tuple(
        {
            "problem_id": audit.problem_id,
            "seed": "" if config.seed is None else str(audit.seed),
            "pair_index": str(manifest.pair_index),
            "arm": manifest.arm,
            "candidate_mode": config.car_candidate_mode,
            "evaluator_id": manifest.evaluator_id,
            "requested_fe": str(manifest.requested_fe),
            "actual_fe": str(manifest.actual_fe),
            "record_sha256": manifest.record_sha256,
            "record_best": f"{manifest.record_best:.17e}",
            "state_fingerprint_before": manifest.state_fingerprint_before,
            "state_fingerprint_after": manifest.state_fingerprint_after,
            "seed_descriptor": manifest.seed_descriptor.canonical_key,
            "probe_seed": str(manifest.seed_descriptor.seed),
        }
        for manifest in execution.branch_manifests
    )
    evaluated_elite = min(manifest.record_best for manifest in execution.branch_manifests)
    state_row = {
        "problem_id": audit.problem_id,
        "seed": "" if config.seed is None else str(audit.seed),
        "graph_fingerprint": evidence.graph_fingerprint,
        "component_fingerprint": evidence.component_fingerprint,
        "candidate_action_name": evidence.candidate_action_name,
        "candidate_action_family": evidence.candidate_action_family,
        "candidate_mode": config.car_candidate_mode,
        "evidence_sweeps": str(evidence.evidence_sweep_count),
        "checkpoint_fe": str(checkpoint_fe),
        "probe_fe": str(ledger.probe_fe),
        "total_fe_after_probe": str(ledger.total_fe),
        "probe_fe_limit": str(probe_limit),
        "adopted_branch": execution.gate.adopted_arm,
        "committed_fitness": f"{execution.adopted_state.committed_fitness:.17e}",
        "evaluated_elite": f"{evaluated_elite:.17e}",
        "state_fingerprint": execution.adopted_state.state_fingerprint,
        "gate_result": "commit" if execution.gate.committed else "abstain",
        "abstain_reason": abstain_reason,
    }
    return CARBarrierResult(
        adopted_state=execution.adopted_state,
        accounting_record=execution.accounting_record,
        probe_fe=ledger.probe_fe,
        probe_trace_rows=probe_rows,
        state_ledger_rows=(state_row,),
        branch_manifest_rows=manifest_rows,
        abstain_reason=abstain_reason,
    )


def run_problem(fun_name: str, fun_id: int, output_path: Path, config: SmokeConfig) -> tuple[list[float], float, list[dict[str, str]]]:
    if config.budget_accounting not in {"strict", "source"}:
        raise ValueError(f"unsupported budget accounting mode: {config.budget_accounting}")
    if is_car_w_family_action(config.arac_action) and (
        not config.enable_relation_dispatch
        or config.relation_policy_mode != "controller_v31"
    ):
        raise ValueError(
            "CAR-W requires relation dispatch with the controller_v31 proposal policy"
        )
    if config.car_branch_order not in {"fallback_first", "candidate_first"}:
        raise ValueError("unsupported CAR branch order")
    if config.car_candidate_mode not in {"graph", "shuffled_graph", "paired_fallback"}:
        raise ValueError("unsupported CAR candidate mode")
    if config.car_actionability_arm not in {"off", "fallback", "candidate"}:
        raise ValueError("unsupported CAR actionability arm")
    if config.car_actionability_arm != "off" and not is_car_w3_action(
        config.arac_action
    ):
        raise ValueError("CAR actionability audit requires the frozen CAR-W3 action")
    time_start = time.time()
    bench = Benchmark(str(output_path) + "/", data_dir=config.aob_data_root)
    fun = bench.get_function(fun_name, fun_id)
    info = bench.get_info(fun_name, fun_id)
    problem_id = _problem_id(fun_name, fun_id)
    grouping_result = decompose_problem(fun_id, config.aob_data_root)
    _, overlap_groups, overlapping_elements = remove_overlapping_groups(grouping_result)
    terminal_completion_tolerance_fe = max(
        1,
        max(
            calculate_cmaes_population_size(len(group))
            for group in grouping_result
        ),
    )
    metadata = load_aob_metadata(fun_id, config.aob_data_root)
    degree = calculate_degree_of_overlap(overlap_groups, metadata["dimension"])
    global_fes = calculate_global_fes(config.max_fes, degree)
    controller_v31_run_state = (
        build_evidence_action_controller_v31_run_state(
            degree,
            action_name=config.arac_action,
        )
        if (
            is_risk_aware_evidence_action_controller(config.arac_action)
            or is_evidence_action_controller_v35(config.arac_action)
            or is_evidence_action_controller_v36(config.arac_action)
            or is_evidence_action_controller_v37(config.arac_action)
            or is_evidence_action_controller_v38(config.arac_action)
            or is_evidence_action_controller_v39(config.arac_action)
            or is_evidence_action_controller_v40(config.arac_action)
        )
        else build_evidence_action_controller_v31_run_state(degree)
        if (
            is_evidence_action_controller_v31(config.arac_action)
            or is_evidence_action_controller_v32(config.arac_action)
        )
        else None
    )
    if is_separable_cmaes_dispatch_action(config.arac_action):
        best_individual = np.zeros(info["dimension"])
        phase_i_fitness = math.inf
        sum_fes = 0
        global_phase_fe = 0
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
                "arac_search_state_action": SEPARABLE_CMAES_DISPATCH_ACTION,
                "arac_guard_source": "phase_i_mmes_incumbent",
            }
            if config.seed is not None:
                options["seed_rng"] = derive_optimizer_seed(config.seed, fun_name, fun_id, 0, 0)
            phase_i_evaluations_before = current_fitness_evaluations(fun)
            phase_i_results = MMES(problem, options).optimize()
            best_individual = np.asarray(phase_i_results["best_so_far_x"], dtype=float).copy()
            phase_i_fitness = float(phase_i_results["best_so_far_y"])
            global_phase_fe = observed_optimizer_fe(
                fun,
                evaluations_before=phase_i_evaluations_before,
                optimizer_reported_fe=phase_i_results["n_function_evaluations"],
            )
            sum_fes += global_phase_fe
        else:
            phase_i_fitness = float(fun(best_individual)[0])
            sum_fes += 1

        reported_current_fe = (
            sum_fes
            if config.budget_accounting == "source"
            else current_fitness_evaluations(fun)
        )
        current_fe = max(reported_current_fe, current_fitness_evaluations(fun))
        remaining_fes = max(0, config.max_fes - current_fe)
        result = run_direct_separable_cmaes_dispatch(
            fun=fun,
            info=info,
            config=config,
            fun_name=fun_name,
            fun_id=fun_id,
            initial_mean=best_individual,
            incumbent_fitness=phase_i_fitness,
            max_function_evaluations=remaining_fes,
        )
        separable_fe = int(result["n_function_evaluations"])
        sum_fes += separable_fe
        best_y = float(result["best_so_far_y"])
        candidate_x = np.asarray(result["best_so_far_x"], dtype=float).copy()
        action_value_delta_norm = float(np.linalg.norm(candidate_x - best_individual))
        accepted = bool(best_y < phase_i_fitness)
        if accepted:
            best_individual = candidate_x.copy()
        population_size = int(result["population_size"])
        action_trace_rows = [
            build_action_trace_row(
                problem_id=problem_id,
                seed=config.seed,
                outer_iter=0,
                group_index=0,
                selected_action_name=SEPARABLE_CMAES_DISPATCH_ACTION,
                overlap_size=0,
                previous_delta=0.0,
                current_delta=max(0.0, phase_i_fitness - best_y),
                state_mutated=accepted,
                action_value_delta_norm=action_value_delta_norm,
                downstream_consumed=False,
                search_state_action_type=SEPARABLE_CMAES_DISPATCH_ACTION,
                sigma_before=SEPARABLE_CMAES_INITIAL_SIGMA,
                sigma_after=float(result["sigma_mean"]),
                population_before=population_size,
                population_after=population_size,
                escape_budget=separable_fe,
                bipop_restart_mode="phase_i_warm_started_direct_full_space_diagonal_separable_cmaes",
                restart_triggered=separable_fe > 0,
                restart_accepted=accepted,
                best_before=phase_i_fitness,
                restart_candidate_best=best_y,
                restart_relative_improvement=bipop_relative_improvement(
                    candidate_best=best_y,
                    incumbent_fitness=phase_i_fitness,
                ),
                restart_acceptance_threshold=0.0,
                best_after=best_y if accepted else phase_i_fitness,
            )
        ]
        _write_overlap_relation_trace(
            case_artifact_path(output_path, problem_id, "overlap_relations.csv"),
            [],
        )
        _write_budget_summary(
            case_artifact_path(output_path, problem_id, "budget_summary.csv"),
            problem_id=problem_id,
            budget_accounting=config.budget_accounting,
            max_fes=config.max_fes,
            optimizer_reported_fe=sum_fes,
            fitness_record_fe=current_fitness_evaluations(fun),
            global_phase_fe=global_phase_fe,
            separable_continuation_fe=separable_fe,
        )
        print(
            f"{problem_id} separable CMA-ES warm-start dispatch completed: "
            f"phase_i={sum_fes - separable_fe} FEs, continuation={separable_fe} FEs"
        )
        return fun.fitness_record, time.time() - time_start, action_trace_rows
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
    car_artifacts_enabled = is_car_w_family_action(config.arac_action)
    car_probe_enabled = car_artifacts_enabled and any(overlapping_elements)
    car_probe_attempted = False
    car_proposal_sweeps: list[tuple[CARRelationProposal, ...]] = []
    car_current_proposals: list[CARRelationProposal] = []
    car_probe_trace_rows: list[dict[str, str]] = []
    car_state_ledger_rows: list[dict[str, str]] = []
    car_branch_manifest_rows: list[dict[str, str]] = []
    car_actionability_trace_base_row: dict[str, str] | None = None
    car_probe_fe = 0
    group_stagnation_counts = [0 for _ in grouping_result]
    bipop_global_cooldown = 0
    bipop_restart_count = 0
    bipop_rejected_restart_streak = 0
    guarded_incumbent = best_individual.copy()
    guarded_incumbent_fitness = math.inf
    cc_harm_guard_consumed = False
    evidence_controller_search_state_enabled = (
        controller_v31_run_state.phase_rescue_enabled
        if controller_v31_run_state is not None
        else False
    )
    component_credit_trace = (
        ComponentDelayedCreditTrace(
            grouping_result,
            lower=float(info["lower"]),
            upper=float(info["upper"]),
        )
        if is_evidence_action_controller_v40(config.arac_action)
        else None
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
        phase_i_optimizer = MMES(problem, options)
        phase_i_evaluations_before = current_fitness_evaluations(fun)
        capture_phase_i_state = (
            config.search_state_backend == "phase_i_mmes"
            and uses_resumable_phase_i_state_during_run(config.arac_action)
        )
        if not capture_phase_i_state:
            results = phase_i_optimizer.optimize()
        else:
            results, phase_i_state = phase_i_optimizer.optimize_with_state()
            controller_v31_run_state.phase_i_optimizer = phase_i_optimizer
            controller_v31_run_state.phase_i_state = phase_i_state
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
    elif is_cc_harm_guarded_sep_refresh_action(config.arac_action):
        guarded_incumbent_fitness = float(fun(best_individual)[0])

    pre_hold_evidence_snapshot: PreHoldEvidence | None = None
    overlap_edge_count = sum(1 for shared in overlapping_elements if shared)
    if controller_v31_run_state is not None:
        pre_hold_current_fes = (
            sum_fes
            if config.budget_accounting == "source"
            else current_fitness_evaluations(fun)
        )
        pre_hold_remaining_fes = max(0, config.max_fes - pre_hold_current_fes)
        pre_hold_evidence_snapshot = build_pre_hold_evidence(
            phase_i_tail_utility=(
                controller_v31_run_state.phase_i_runtime_tail_utility
            ),
            group_sizes=tuple(len(group) for group in grouping_result),
            overlapping_elements=tuple(
                tuple(int(variable) for variable in shared)
                for shared in overlapping_elements
            ),
            dimension=int(info["dimension"]),
            remaining_fes=pre_hold_remaining_fes,
            max_fes=config.max_fes,
            scheduled_hold_fes=scheduled_search_state_hold_fes(
                config,
                controller_v31_run_state.search_state_scheduler_state,
                overlap_edge_count=overlap_edge_count,
            ),
        )

    outer_iter = 0
    previous_rule_relation_action: RelationActionDecision | None = None
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
        sweep_incumbent_before = guarded_incumbent_fitness
        sweep_fes_before = current_fitness_evaluations(fun)
        sub_num = len(grouping_result)
        held_search_state_fes = (
            scheduled_search_state_hold_fes(
                config,
                controller_v31_run_state.search_state_scheduler_state,
                overlap_edge_count=overlap_edge_count,
            )
            if controller_v31_run_state is not None
            else 0
        )
        cc_budget_limit_fes = max(
            current_fes,
            config.max_fes
            - held_search_state_fes
            - (
                int(math.floor(config.max_fes * CAR_W_PROBE_BUDGET_FRACTION))
                if (
                    is_car_w_action(config.arac_action)
                    and car_probe_enabled
                    and not car_probe_attempted
                )
                else 0
            ),
        )
        if (
            is_car_w_action(config.arac_action)
            and car_probe_enabled
            and not car_probe_attempted
            and len(car_proposal_sweeps) < CAR_W_MIN_EVIDENCE_SWEEPS
        ):
            sweep_slots_remaining = CAR_W_MIN_EVIDENCE_SWEEPS - len(car_proposal_sweeps)
            sub_fes = math.ceil(
                max(0, cc_budget_limit_fes - current_fes)
                / max(1, sub_num * (sweep_slots_remaining + 1))
            )
        else:
            sub_fes = math.ceil(max(0, cc_budget_limit_fes - current_fes) / sub_num)
        population_sizes = [
            calculate_cmaes_population_size(len(dims)) for dims in grouping_result
        ]
        trajectory_budgets = []
        trajectory_credit_ready = has_sufficient_trajectory_credit(previous_group_contribution_credit)
        if uses_trajectory_budget_shift(config.arac_action) and trajectory_credit_ready:
            trajectory_budgets = allocate_trajectory_group_budgets(
                total_budget=config.max_fes - current_fes,
                population_sizes=population_sizes,
                overlap_support=calculate_group_overlap_support(
                    grouping_result,
                    overlapping_elements,
                ),
                contribution_credit=previous_group_contribution_credit,
            )
        fitness_delta_list: list[float] = []
        overlap_writeback_norms: list[float] = []
        relative_writeback_norms: list[float] = []
        current_outer_relations: list[OverlapRelation] = []
        current_outer_decisions: list[RelationActionDecision] = []
        car_current_proposals = []
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
            original_best = best_individual.copy()
            original_fitness = float(fun(best_individual)[0])
            if component_credit_trace is not None:
                component_credit_trace.resolve_group_revisit(
                    group_index=index,
                    resolution_fe=current_fitness_evaluations(fun),
                    current_fitness=original_fitness,
                    current_candidate=best_individual,
                )
            if controller_v31_run_state is not None:
                controller_v31_run_state.observe_pending_action_trust(
                    post_writeback_fitness=original_fitness,
                )
                controller_v31_run_state.observe_pending_trajectory_guard(
                    post_writeback_fitness=original_fitness,
                )
            if config.budget_accounting == "source":
                optimizer_budget = sub_fes
            else:
                requested_fes = max(sub_fes, population_size)
                if trajectory_budgets:
                    requested_fes = max(trajectory_budgets[index], population_size)
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
            if uses_trajectory_mean_blend(config.arac_action) and trajectory_credit_ready:
                cc_mean, _, _ = blend_trajectory_mean(
                    base_mean=cc_mean,
                    dims=list(dims),
                    variable_mean_cache=trajectory_mean_cache,
                    lower=info["lower"],
                    upper=info["upper"],
                )
            cma_sigma_reference = refine_sigma_for_action(
                config.arac_action,
                config.sigma,
                controller_v31_run_state=controller_v31_run_state,
            )
            cc_sigma = cma_sigma_reference
            cma_sigma_applied_factor = 1.0
            cma_sigma_route = ""
            if (
                controller_v31_run_state is not None
                and controller_v31_run_state.v39_enabled
            ):
                (
                    cc_sigma,
                    cma_sigma_applied_factor,
                    cma_sigma_route,
                ) = controller_v31_run_state.v39_cma_sigma_for_group(
                    dims,
                    cma_sigma_reference,
                )
            precision_reanchor_active = uses_post_retirement_precision_reanchor(
                config.arac_action,
                controller_v31_run_state,
            )
            objective_function = lambda x_batch, dims=dims: fun(combine(x_batch, best_individual, dims))
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
            primary_evaluations_before = current_fitness_evaluations(fun)
            scheduler_revisit_cap = (
                calculate_scheduler_revisit_cap(
                    sweep_start_fe=sweep_fes_before,
                    decision_fe=primary_evaluations_before,
                    cc_budget_limit_fe=cc_budget_limit_fes,
                    current_group_index=index,
                    current_sweep_group_budget_fe=sub_fes,
                    current_optimizer_budget_fe=optimizer_budget,
                    group_population_sizes=tuple(population_sizes),
                )
                if component_credit_trace is not None
                and precision_reanchor_active
                else None
            )
            results_cc = CMAES(problem_cc, options_cc).optimize()
            optimized_any_group = True
            primary_cc_fe = observed_optimizer_fe(
                fun,
                evaluations_before=primary_evaluations_before,
                optimizer_reported_fe=results_cc["n_function_evaluations"],
            )
            cc_phase_fe += primary_cc_fe
            sum_fes += primary_cc_fe
            cma_sigma_terminal: float | None = None
            cma_sigma_next_factor: float | None = None
            cma_restart_count: int | None = None
            if (
                controller_v31_run_state is not None
                and controller_v31_run_state.v39_enabled
            ):
                if "sigma" not in results_cc or "_n_restart" not in results_cc:
                    raise RuntimeError(
                        "v39 requires terminal CMA sigma and restart count"
                    )
                cma_sigma_terminal = float(results_cc["sigma"])
                cma_restart_count = int(results_cc["_n_restart"])
                cma_sigma_next_factor = (
                    controller_v31_run_state.observe_v39_cma_terminal_sigma(
                        dims,
                        reference_sigma=cma_sigma_reference,
                        terminal_sigma=cma_sigma_terminal,
                    )
                )
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
                if uses_trajectory_mean_blend(config.arac_action) and trajectory_credit_ready:
                    accepted_mean = np.asarray(
                        results_cc["best_so_far_x"],
                        dtype=float,
                    ).reshape(-1)
                    for local_index, variable_index in enumerate(dims):
                        if local_index < accepted_mean.size and np.isfinite(accepted_mean[local_index]):
                            trajectory_mean_cache[int(variable_index)] = float(accepted_mean[local_index])
            else:
                current_delta = 0.0
            if precision_reanchor_active:
                normal_refine_sigma = (
                    float(config.sigma) * REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER
                )
                precision_trace_row = build_action_trace_row(
                    problem_id=_problem_id(fun_name, fun_id),
                    seed=config.seed,
                    outer_iter=outer_iter,
                    group_index=index,
                    selected_action_name=(
                        POST_RETIREMENT_PRECISION_REANCHOR_ACTION
                    ),
                    overlap_size=0,
                    previous_delta=0.0,
                    current_delta=current_delta,
                    state_mutated=new_best_y < original_fitness,
                    action_value_delta_norm=abs(normal_refine_sigma - cc_sigma),
                    downstream_consumed=True,
                    downstream_consumption_scope="current_group_optimizer",
                    search_state_action_type=(
                        POST_RETIREMENT_PRECISION_REANCHOR_ACTION
                    ),
                    sigma_before=normal_refine_sigma,
                    sigma_after=cc_sigma,
                    population_before=population_size,
                    population_after=population_size,
                    best_before=original_fitness,
                    best_after=original_fitness - current_delta,
                    cc_block_fe=primary_cc_fe,
                    remaining_budget_ratio=(
                        max(0, config.max_fes - primary_evaluations_before)
                        / max(config.max_fes, 1)
                    ),
                    decision_point=f"group_optimizer:{outer_iter}:{index}",
                    state_action_fe=primary_cc_fe,
                )
                if scheduler_revisit_cap is not None:
                    precision_trace_row.update(
                        scheduler_revisit_cap.trace_fields()
                    )
                action_trace_rows.append(precision_trace_row)
                if component_credit_trace is not None:
                    component_credit_trace.register_search_action(
                        precision_trace_row,
                        action_name=POST_RETIREMENT_PRECISION_REANCHOR_ACTION,
                        outer_iter=outer_iter,
                        group_index=index,
                        decision_fe=primary_evaluations_before,
                        max_fes=config.max_fes,
                        pre_action_fitness=original_fitness,
                        post_action_fitness=original_fitness - current_delta,
                        pre_action_candidate=original_best,
                        post_action_candidate=best_individual,
                    )
            if (
                controller_v31_run_state is not None
                and controller_v31_run_state.v39_enabled
                and cma_sigma_terminal is not None
                and cma_sigma_next_factor is not None
                and cma_restart_count is not None
            ):
                action_trace_rows.append(
                    build_action_trace_row(
                        problem_id=_problem_id(fun_name, fun_id),
                        seed=config.seed,
                        outer_iter=outer_iter,
                        group_index=index,
                        selected_action_name=(
                            CROSS_SWEEP_CMA_SIGMA_CONTINUATION_ACTION
                        ),
                        overlap_size=0,
                        previous_delta=0.0,
                        current_delta=current_delta,
                        state_mutated=True,
                        action_value_delta_norm=abs(
                            cc_sigma - cma_sigma_reference
                        ),
                        downstream_consumed=True,
                        downstream_consumption_scope="current_group_optimizer",
                        search_state_action_type=(
                            CROSS_SWEEP_CMA_SIGMA_CONTINUATION_ACTION
                        ),
                        sigma_before=cma_sigma_reference,
                        sigma_after=cc_sigma,
                        population_before=population_size,
                        population_after=population_size,
                        best_before=original_fitness,
                        best_after=original_fitness - current_delta,
                        cc_block_fe=primary_cc_fe,
                        cma_sigma_reference=cma_sigma_reference,
                        cma_sigma_applied_factor=cma_sigma_applied_factor,
                        cma_sigma_terminal=cma_sigma_terminal,
                        cma_sigma_next_factor=cma_sigma_next_factor,
                        cma_sigma_route=cma_sigma_route,
                        cma_restart_count=cma_restart_count,
                    )
                )
            if is_bipop_search_state_action(config.arac_action):
                if bipop_global_cooldown > 0:
                    bipop_global_cooldown -= 1
                if group_delta_stagnated(current_delta, original_fitness):
                    group_stagnation_counts[index] += 1
                    outer_stagnation_streak += 1
                else:
                    group_stagnation_counts[index] = 0
                    outer_stagnation_streak = 0
                remaining_fes = config.max_fes - current_fitness_evaluations(fun)
                restart_rng = np.random.default_rng(
                    derive_optimizer_seed(
                        config.seed if config.seed is not None else 0,
                        fun_name,
                        fun_id,
                        outer_iter + 1,
                        (index + 1) * 1009 + bipop_restart_count,
                    )
                )
                restart_plan = build_bipop_restart_plan(
                    group_index=index,
                    restart_count=bipop_restart_count,
                    base_population_size=population_size,
                    base_sigma=config.sigma,
                    base_budget=optimizer_budget,
                    remaining_fes=remaining_fes,
                    rng=restart_rng,
                )
                if should_trigger_bipop_restart(
                    stagnation_count=max(group_stagnation_counts[index], outer_stagnation_streak),
                    cooldown_remaining=bipop_global_cooldown,
                    escape_budget=restart_plan.escape_budget,
                ):
                    stagnation_window_for_trace = max(
                        group_stagnation_counts[index],
                        outer_stagnation_streak,
                    )
                    primary_delta_for_trace = current_delta
                    post_primary_fitness = original_fitness - current_delta
                    restart_mean = perturb_bipop_restart_mean(
                        base_mean=np.asarray(best_individual[dims], dtype=float),
                        lower=info["lower"],
                        upper=info["upper"],
                        sigma=restart_plan.sigma,
                        rng=restart_rng,
                    )
                    restart_options = {
                        "max_function_evaluations": restart_plan.escape_budget,
                        "mean": (restart_mean,),
                        "sigma": restart_plan.sigma,
                        "n_individuals": restart_plan.population_size,
                        "is_restart": config.cmaes_restart,
                        "verbose": config.verbose,
                        "early_stopping_evaluations": config.early_stopping_evaluations,
                        "arac_search_state_action": SEARCH_STATE_BIPOP_ACTION,
                        "arac_bipop_restart_mode": restart_plan.restart_mode,
                    }
                    if config.seed is not None:
                        restart_options["seed_rng"] = derive_optimizer_seed(
                            config.seed,
                            fun_name,
                            fun_id,
                            outer_iter + 1,
                            (index + 1) * 7919 + bipop_restart_count,
                        )
                    restart_evaluations_before = current_fitness_evaluations(fun)
                    restart_results = CMAES(problem_cc, restart_options).optimize()
                    bipop_restart_count += 1
                    restart_fe = observed_optimizer_fe(
                        fun,
                        evaluations_before=restart_evaluations_before,
                        optimizer_reported_fe=restart_results["n_function_evaluations"],
                    )
                    rescue_fe += restart_fe
                    sum_fes += restart_fe
                    restart_best = float(restart_results["best_so_far_y"])
                    restart_relative_improvement = bipop_relative_improvement(
                        candidate_best=restart_best,
                        incumbent_fitness=post_primary_fitness,
                    )
                    restart_accepted = should_accept_bipop_restart(
                        candidate_best=restart_best,
                        incumbent_fitness=post_primary_fitness,
                    )
                    if restart_accepted:
                        best_individual[dims] = restart_results["best_so_far_x"].copy()
                        current_delta = original_fitness - restart_best
                        group_stagnation_counts[index] = 0
                        outer_stagnation_streak = 0
                        bipop_rejected_restart_streak = 0
                    else:
                        bipop_rejected_restart_streak += 1
                    bipop_global_cooldown = bipop_cooldown_after_restart(
                        restart_accepted=restart_accepted,
                        sub_num=sub_num,
                        rejected_restart_streak=bipop_rejected_restart_streak,
                    )
                    action_trace_rows.append(
                        build_action_trace_row(
                            problem_id=_problem_id(fun_name, fun_id),
                            seed=config.seed,
                            outer_iter=outer_iter,
                            group_index=index,
                            selected_action_name=config.arac_action,
                            overlap_size=0,
                            previous_delta=primary_delta_for_trace,
                            current_delta=0.0 if not restart_accepted else current_delta,
                            state_mutated=restart_accepted,
                            action_value_delta_norm=float(
                                np.linalg.norm(restart_mean - cc_mean)
                            ),
                            downstream_consumed=index < sub_num - 1,
                            search_state_action_type="bipop_restart",
                            stagnation_window=stagnation_window_for_trace,
                            delta_mean=float(np.linalg.norm(restart_mean - cc_mean)),
                            sigma_before=config.sigma,
                            sigma_after=restart_plan.sigma,
                            population_before=population_size,
                            population_after=restart_plan.population_size,
                            escape_budget=restart_plan.escape_budget,
                            bipop_restart_mode=restart_plan.restart_mode,
                            restart_triggered=True,
                            restart_accepted=restart_accepted,
                            best_before=post_primary_fitness,
                            restart_candidate_best=restart_best,
                            restart_relative_improvement=restart_relative_improvement,
                            restart_acceptance_threshold=BIPOP_ACCEPT_RELATIVE_IMPROVEMENT,
                            best_after=restart_best if restart_accepted else post_primary_fitness,
                        )
                    )
            if config.search_state_backend != "diagonal_cma" and uses_phase_rescue_during_run(
                config.arac_action,
                evidence_controller_search_state_enabled=(
                    evidence_controller_search_state_enabled
                ),
            ):
                if bipop_global_cooldown > 0:
                    bipop_global_cooldown -= 1
                if group_delta_stagnated(current_delta, original_fitness):
                    group_stagnation_counts[index] += 1
                    outer_stagnation_streak += 1
                else:
                    group_stagnation_counts[index] = 0
                    outer_stagnation_streak = 0
                remaining_fes = config.max_fes - current_fitness_evaluations(fun)
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
                        rescue_results = CMAES(problem_cc, rescue_options).optimize()
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
                            selected_action_name=PHASE_RESCUE_MULTISTART_ACTION
                            if is_evidence_action_controller(config.arac_action)
                            else config.arac_action,
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
            if (
                controller_v31_run_state is not None
                and controller_v31_run_state.pending_trajectory_recovery is not None
            ):
                checkpoint_candidate = (
                    controller_v31_run_state.pending_trajectory_recovery
                    .checkpoint.candidate.copy()
                )
                resolved_recovery = (
                    controller_v31_run_state.resolve_pending_trajectory_guard(
                        downstream_candidate=best_individual,
                        downstream_fitness=original_fitness - current_delta,
                    )
                )
                if resolved_recovery is not None:
                    (
                        best_individual,
                        original_best,
                        original_fitness,
                        current_delta,
                    ) = reconcile_trajectory_recovery_context(
                        resolution=resolved_recovery,
                        checkpoint_candidate=checkpoint_candidate,
                        original_best=original_best,
                        original_fitness=original_fitness,
                        current_delta=current_delta,
                    )
            fitness_delta_list.append(current_delta)
            if index > 0:
                overlap_indices = overlapping_elements[index - 1]
                if config.enable_relation_dispatch:
                    if config.relation_policy_mode not in {
                        "rule",
                        "adaptive_v2",
                        "adaptive_v21",
                        "adaptive_v22",
                        "adaptive_v23",
                        "adaptive_v24",
                        "adaptive_v25",
                        "adaptive_v26",
                        "controller_v3",
                        "controller_v31",
                        "shuffled",
                        "lagged",
                    }:
                        raise ValueError(
                            f"unsupported relation policy mode: {config.relation_policy_mode}"
                        )
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
                    if (
                        config.relation_policy_mode == "controller_v31"
                        and controller_v31_run_state is not None
                    ):
                        controller_v31_run_state.lock_from_runtime_prefix(
                            relation_policy_context
                        )
                        effective_policy_mode = (
                            controller_v31_run_state.effective_policy_mode
                        )
                        evidence_controller_search_state_enabled = (
                            controller_v31_run_state.phase_rescue_enabled
                        )
                    else:
                        effective_policy_mode = effective_relation_policy_mode(
                            config.relation_policy_mode,
                            relation_policy_context,
                        )
                        if config.relation_policy_mode == "controller_v3":
                            evidence_controller_search_state_enabled = (
                                select_evidence_action_controller_v3_mode(
                                    relation_policy_context
                                )
                                == "search_state_assisted"
                            )
                    if effective_policy_mode == "adaptive_v23":
                        rule_action = decide_actions_for_relations_v23(
                            relation_policy_context
                        )[-1]
                    elif effective_policy_mode == "adaptive_v26":
                        rule_action = decide_actions_for_relations_v26(
                            relation_policy_context
                        )[-1]
                    elif effective_policy_mode == "adaptive_v25":
                        rule_action = decide_actions_for_relations_v25(
                            relation_policy_context
                        )[-1]
                    elif effective_policy_mode == "adaptive_v24":
                        rule_action = decide_actions_for_relations_v24(
                            relation_policy_context
                        )[-1]
                    elif effective_policy_mode == "adaptive_v22":
                        rule_action = decide_actions_for_relations_v22(
                            relation_policy_context
                        )[-1]
                    elif effective_policy_mode == "adaptive_v21":
                        rule_action = decide_actions_for_relations_v21(
                            relation_policy_context
                        )[-1]
                    elif effective_policy_mode == "adaptive_v2":
                        rule_action = decide_actions_for_relations_v2(
                            relation_policy_context
                        )[-1]
                    else:
                        rule_action = decide_actions_for_relations(
                            relation_policy_context
                        )[-1]
                    shuffled_source_action = previous_rule_relation_action
                    if config.relation_policy_mode == "lagged":
                        previous_rule_relation_action, _, _ = (
                            apply_and_guard_action_to_relation(
                                relation=relation,
                                action=rule_action,
                                previous_values=context.previous_values,
                                current_values=context.current_values,
                                previous_delta=context.previous_delta,
                                current_delta=context.current_delta,
                            )
                        )
                    action = select_relation_action_for_policy(
                        relation=relation,
                        action=rule_action,
                        relation_policy_mode=effective_policy_mode
                        if config.relation_policy_mode in {"controller_v3", "controller_v31"}
                        else config.relation_policy_mode,
                        shuffled_source_action=shuffled_source_action,
                    )
                    if car_probe_enabled and relation.shared_vars:
                        forced_candidate = (
                            None
                            if controller_v31_run_state is None
                            else controller_v31_run_state.forced_relation_action(relation)
                        )
                        candidate_source_action = forced_candidate or action
                        (
                            candidate_proposal_action,
                            candidate_proposed_values,
                            candidate_writeback_norm,
                        ) = apply_and_guard_action_to_relation(
                            relation=relation,
                            action=candidate_source_action,
                            previous_values=context.previous_values,
                            current_values=context.current_values,
                            previous_delta=context.previous_delta,
                            current_delta=context.current_delta,
                        )
                        proposal_values = (
                            context.current_values
                            if candidate_proposed_values is None
                            else candidate_proposed_values
                        )
                        car_current_proposals.append(
                            CARRelationProposal(
                                sweep_index=len(car_proposal_sweeps),
                                group_left=relation.group_left,
                                group_right=relation.group_right,
                                shared_indices=tuple(int(value) for value in relation.shared_vars),
                                target_values=tuple(
                                    float(value)
                                    for value in np.asarray(
                                        proposal_values,
                                        dtype=float,
                                    ).reshape(-1)
                                ),
                                action_name=_canonical_relation_action_name(
                                    candidate_proposal_action
                                ),
                                action_family=candidate_proposal_action.action_family,
                                overlap_strength=float(relation.overlap_strength),
                                feature_coverage=float(relation.feature_coverage),
                                writeback_norm=float(candidate_writeback_norm),
                            )
                        )
                    trust_decision: ActionTrustDecision | None = None
                    fallback_route = ""
                    active_maturity_route = ""
                    if is_risk_aware_evidence_action_controller(config.arac_action):
                        (
                            action,
                            adjusted_values,
                            action_value_delta_norm,
                            trust_decision,
                            fallback_route,
                        ) = apply_relation_action_with_controller_v33(
                            relation=relation,
                            action=action,
                            previous_values=context.previous_values,
                            current_values=context.current_values,
                            previous_delta=context.previous_delta,
                            current_delta=context.current_delta,
                            controller_run_state=controller_v31_run_state,
                        )
                    elif is_evidence_action_controller_v35(config.arac_action):
                        (
                            action,
                            adjusted_values,
                            action_value_delta_norm,
                            trust_decision,
                            fallback_route,
                        ) = apply_relation_action_with_controller_v35(
                            relation=relation,
                            action=action,
                            previous_values=context.previous_values,
                            current_values=context.current_values,
                            previous_delta=context.previous_delta,
                            current_delta=context.current_delta,
                            controller_run_state=controller_v31_run_state,
                        )
                    elif is_evidence_action_controller_v36(
                        config.arac_action
                    ) or is_evidence_action_controller_v37(
                        config.arac_action
                    ) or is_evidence_action_controller_v38(
                        config.arac_action
                    ) or is_evidence_action_controller_v39(
                        config.arac_action
                    ) or is_evidence_action_controller_v40(config.arac_action):
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
                    else:
                        action, adjusted_values, action_value_delta_norm = (
                            apply_relation_action_with_controller_v31(
                                relation=relation,
                                action=action,
                                previous_values=context.previous_values,
                                current_values=context.current_values,
                                previous_delta=context.previous_delta,
                                current_delta=context.current_delta,
                                controller_v31_run_state=controller_v31_run_state,
                            )
                        )
                    trajectory_checkpoint_candidate = best_individual.copy()
                    if adjusted_values is not None:
                        best_individual[context.overlap_indices] = adjusted_values
                    overlap_writeback_norms.append(action_value_delta_norm)
                    relative_writeback_norm = scale_free_writeback_norm(
                        delta_norm=action_value_delta_norm,
                        shared_count=len(context.overlap_indices),
                        lower=float(info["lower"]),
                        upper=float(info["upper"]),
                    )
                    relative_writeback_norms.append(relative_writeback_norm)
                    trust_unstable = (
                        relative_writeback_norm
                        >= RELATIVE_WRITEBACK_UNSTABLE_THRESHOLD
                    )
                    canonical_action_name = _canonical_relation_action_name(action)
                    current_outer_relations.append(relation)
                    current_outer_decisions.append(action)
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
                    if component_credit_trace is not None:
                        component_credit_trace.annotate_relation_observation(
                            action_trace_row,
                            outer_iter=outer_iter,
                            group_left=relation.group_left,
                            group_right=relation.group_right,
                            previous_values=context.previous_values,
                            current_values=context.current_values,
                            decision_fe=current_fitness_evaluations(fun),
                            max_fes=config.max_fes,
                        )
                    action_trace_rows.append(action_trace_row)
                    if controller_v31_run_state is not None:
                        controller_v31_run_state.register_pending_action_trust(
                            decision=trust_decision,
                            pre_writeback_fitness=original_fitness - current_delta,
                            unstable=trust_unstable,
                            trace_row=action_trace_row,
                        )
                        if trust_writeback_active:
                            controller_v31_run_state.register_pending_trajectory_guard(
                                candidate=trajectory_checkpoint_candidate,
                                pre_writeback_fitness=(
                                    original_fitness - current_delta
                                ),
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
                    overlap_writeback_norms.append(overlap_writeback_norm)
                    relative_writeback_norms.append(
                        scale_free_writeback_norm(
                            delta_norm=overlap_writeback_norm,
                            shared_count=len(overlap_indices),
                            lower=float(info["lower"]),
                            upper=float(info["upper"]),
                        )
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
            if config.search_state_backend != "diagonal_cma" and uses_cc_harm_guard_during_run(
                config.arac_action,
                evidence_controller_search_state_enabled=(
                    evidence_controller_search_state_enabled
                ),
            ) and not cc_harm_guard_consumed:
                post_cc_fitness = original_fitness - current_delta
                if guarded_incumbent_fitness <= post_cc_fitness:
                    guard_individual = guarded_incumbent.copy()
                    guard_fitness = guarded_incumbent_fitness
                    guard_source = "phase_i_incumbent"
                else:
                    guard_individual = best_individual.copy()
                    guard_fitness = post_cc_fitness
                    guard_source = "current_cc_incumbent"
                remaining_fes = config.max_fes - current_fitness_evaluations(fun)
                minimum_refresh_budget = calculate_cmaes_population_size(int(info["dimension"]))
                guard_triggered, guard_reason = should_trigger_cc_harm_guard(
                    fitness_deltas=fitness_delta_list,
                    overlap_writeback_norms=overlap_writeback_norms,
                    reference_fitness=guard_fitness,
                    remaining_fes=remaining_fes,
                    minimum_refresh_budget=minimum_refresh_budget,
                )
                if guard_triggered:
                    accepted, guarded_candidate, guarded_best, refresh_fes, candidate_best = (
                        run_guarded_nda_continuation(
                            fun=fun,
                            info=info,
                            config=config,
                            fun_name=fun_name,
                            fun_id=fun_id,
                            outer_iter=outer_iter,
                            guard_individual=guard_individual,
                            guard_fitness=guard_fitness,
                            remaining_fes=remaining_fes,
                        )
                    )
                    sum_fes += refresh_fes
                    refresh_fe += refresh_fes
                    best_individual = guarded_candidate.copy()
                    guarded_incumbent = best_individual.copy()
                    guarded_incumbent_fitness = guarded_best
                    cc_harm_guard_consumed = True
                    action_trace_rows.append(
                        build_action_trace_row(
                            problem_id=_problem_id(fun_name, fun_id),
                            seed=config.seed,
                            outer_iter=outer_iter,
                            group_index=index,
                            selected_action_name=CC_HARM_GUARDED_SEP_REFRESH_ACTION,
                            overlap_size=0,
                            previous_delta=sum(max(0.0, delta) for delta in fitness_delta_list),
                            current_delta=max(0.0, guard_fitness - guarded_best),
                            state_mutated=accepted,
                            action_value_delta_norm=0.0,
                            downstream_consumed=False,
                            search_state_action_type=CC_HARM_GUARDED_SEP_REFRESH_ACTION,
                            stagnation_window=sum(
                                1 for delta in fitness_delta_list
                                if group_delta_stagnated(delta, guard_fitness)
                            ),
                            delta_mean=0.0,
                            sigma_before=config.sigma,
                            sigma_after=float(config.sigma) * CC_HARM_REFRESH_SIGMA_MULTIPLIER,
                            population_before=minimum_refresh_budget,
                            population_after=minimum_refresh_budget,
                            escape_budget=refresh_fes,
                            bipop_restart_mode=f"guarded_nda_continuation:{guard_source}:{guard_reason}",
                            restart_triggered=True,
                            restart_accepted=accepted,
                            best_before=guard_fitness,
                            restart_candidate_best=candidate_best,
                            restart_relative_improvement=bipop_relative_improvement(
                                candidate_best=candidate_best,
                                incumbent_fitness=guard_fitness,
                            ),
                            restart_acceptance_threshold=0.0,
                            best_after=guarded_best,
                        )
                    )
                    break
        if car_probe_enabled and not car_probe_attempted:
            expected_proposals = sum(1 for shared in overlapping_elements if shared)
            complete_evidence_sweep = (
                optimized_any_group
                and len(fitness_delta_list) == sub_num
                and len(car_current_proposals) == expected_proposals
            )
            if complete_evidence_sweep:
                car_proposal_sweeps.append(tuple(car_current_proposals))
            if len(car_proposal_sweeps) >= CAR_W_MIN_EVIDENCE_SWEEPS:
                car_probe_attempted = True
                lazy_car_mode = is_car_w2_action(
                    config.arac_action
                ) or is_car_w3_action(config.arac_action)
                if controller_v31_run_state is not None and not lazy_car_mode:
                    controller_v31_run_state.invalidate_pending_action_trust(
                        "car_component_barrier"
                    )
                car_decision = freeze_component_writeback_plan(
                    grouping_result=tuple(
                        tuple(int(value) for value in group)
                        for group in grouping_result
                    ),
                    overlapping_elements=tuple(
                        tuple(int(value) for value in shared)
                        for shared in overlapping_elements
                    ),
                    group_population_sizes=tuple(population_sizes),
                    proposal_sweeps=tuple(car_proposal_sweeps),
                    lower=float(info["lower"]),
                    upper=float(info["upper"]),
                    minimum_writeback_norm=(
                        CAR_W2_FUTILITY_MIN_WRITEBACK_NORM
                        if lazy_car_mode
                        else 0.0
                    ),
                )
                if car_decision.plan is not None:
                    if lazy_car_mode:
                        checkpoint_fitness = guarded_incumbent_fitness
                    else:
                        checkpoint_fitness = float(fun(best_individual)[0])
                        sum_fes += 1
                        cc_phase_fe += 1
                    checkpoint_incumbent = best_individual.copy()
                else:
                    checkpoint_fitness = guarded_incumbent_fitness
                    checkpoint_incumbent = guarded_incumbent.copy()
                if not math.isfinite(float(checkpoint_fitness)):
                    if current_fitness_evaluations(fun) >= config.max_fes:
                        raise RuntimeError(
                            "cannot establish a finite CAR actionability checkpoint"
                        )
                    checkpoint_fitness = float(fun(checkpoint_incumbent)[0])
                    sum_fes += 1
                    cc_phase_fe += 1
                checkpoint = BranchState(
                    incumbent=tuple(
                        float(value) for value in checkpoint_incumbent
                    ),
                    committed_fitness=checkpoint_fitness,
                    evaluator_record=[],
                    state_fingerprint="",
                    state_payload=_car_controller_state_payload(
                        controller_v31_run_state,
                        trajectory_mean_cache=trajectory_mean_cache,
                        previous_group_contribution_credit=(
                            previous_group_contribution_credit
                        ),
                    ),
                )
                checkpoint.state_fingerprint = fingerprint_branch_state(checkpoint)
                if config.car_actionability_arm != "off":
                    audit_result = execute_car_actionability_arm_at_barrier(
                        decision=car_decision,
                        checkpoint=checkpoint,
                        checkpoint_fe=current_fitness_evaluations(fun),
                        prefix_record=tuple(float(value) for value in fun.fitness_record),
                        fun_name=fun_name,
                        fun_id=fun_id,
                        output_path=output_path,
                        info=info,
                        config=config,
                        problem_id=problem_id,
                    )
                    car_actionability_trace_base_row = audit_result.trace_base_row
                    audit_state = audit_result.state
                    car_state_ledger_rows.append(
                        {
                            "problem_id": problem_id,
                            "seed": "" if config.seed is None else str(int(config.seed)),
                            "graph_fingerprint": car_actionability_trace_base_row[
                                "graph_fingerprint"
                            ],
                            "component_fingerprint": car_actionability_trace_base_row[
                                "component_fingerprint"
                            ],
                            "candidate_action_name": car_actionability_trace_base_row[
                                "candidate_action_name"
                            ],
                            "candidate_action_family": car_actionability_trace_base_row[
                                "candidate_action_family"
                            ],
                            "candidate_mode": config.car_candidate_mode,
                            "evidence_sweeps": (
                                "0"
                                if car_decision.evidence is None
                                else str(car_decision.evidence.evidence_sweep_count)
                            ),
                            "checkpoint_fe": car_actionability_trace_base_row[
                                "checkpoint_fe"
                            ],
                            "probe_fe": "0",
                            "total_fe_after_probe": str(
                                current_fitness_evaluations(fun) + audit_result.actual_fe
                            ),
                            "probe_fe_limit": "0",
                            "adopted_branch": (
                                "not_applied"
                                if audit_state is None
                                else f"offline_oracle_{config.car_actionability_arm}"
                            ),
                            "committed_fitness": (
                                f"{checkpoint.committed_fitness:.17e}"
                                if audit_state is None
                                else f"{audit_state.committed_fitness:.17e}"
                            ),
                            "evaluated_elite": "",
                            "state_fingerprint": (
                                checkpoint.state_fingerprint
                                if audit_state is None
                                else audit_state.state_fingerprint
                            ),
                            "gate_result": "offline_actionability_audit",
                            "abstain_reason": audit_result.abstain_reason,
                        }
                    )
                    if audit_state is not None:
                        fun.fitness_record.extend(audit_result.accounting_record)
                        sum_fes += audit_result.actual_fe
                        cc_phase_fe += audit_result.actual_fe
                        best_individual = np.asarray(
                            audit_state.incumbent,
                            dtype=float,
                        ).copy()
                        if audit_state.committed_fitness < guarded_incumbent_fitness:
                            guarded_incumbent = best_individual.copy()
                            guarded_incumbent_fitness = audit_state.committed_fitness
                        previous_group_contribution_credit = []
                else:
                    barrier = execute_car_w_probe_at_barrier(
                        decision=car_decision,
                        checkpoint=checkpoint,
                        checkpoint_fe=current_fitness_evaluations(fun),
                        fun_name=fun_name,
                        fun_id=fun_id,
                        output_path=output_path,
                        info=info,
                        config=config,
                        problem_id=problem_id,
                        branch_order=(
                            ("candidate", "fallback")
                            if config.car_branch_order == "candidate_first"
                            else ("fallback", "candidate")
                        ),
                        early_futility_abort=is_car_w3_action(config.arac_action),
                    )
                    car_probe_trace_rows.extend(barrier.probe_trace_rows)
                    car_state_ledger_rows.extend(barrier.state_ledger_rows)
                    car_branch_manifest_rows.extend(barrier.branch_manifest_rows)
                    if barrier.adopted_state is not None:
                        fun.fitness_record.extend(barrier.accounting_record)
                        car_probe_fe += barrier.probe_fe
                        sum_fes += barrier.probe_fe
                        best_individual = np.asarray(
                            barrier.adopted_state.incumbent,
                            dtype=float,
                        ).copy()
                        if (
                            barrier.adopted_state.committed_fitness
                            < guarded_incumbent_fitness
                        ):
                            guarded_incumbent = best_individual.copy()
                            guarded_incumbent_fitness = (
                                barrier.adopted_state.committed_fitness
                            )
                        previous_group_contribution_credit = []

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

        if (
            controller_v31_run_state is not None
            and uses_scheduled_search_state(config)
            and optimized_any_group
            and len(fitness_delta_list) == sub_num
        ):
            phase_state = controller_v31_run_state.phase_i_state
            phase_optimizer = controller_v31_run_state.phase_i_optimizer
            phase_state_available = phase_state is not None and phase_optimizer is not None
            search_state_available = (
                config.search_state_backend == "diagonal_cma"
                or phase_state_available
            )
            sweep_fes = current_fitness_evaluations(fun) - sweep_fes_before
            cc_utility = normalized_gain_utility(
                sweep_incumbent_before,
                guarded_incumbent_fitness,
                sweep_fes,
            )
            controller_v31_run_state.cc_utility_history.append(cc_utility)
            phase_population_size = (
                int(phase_state.n_individuals)
                if phase_state is not None
                else calculate_cmaes_population_size(int(info["dimension"]))
            )
            evidence = build_search_state_evidence(
                complete_sweep=(
                    len(current_outer_relations) == max(0, sub_num - 1)
                    and len(current_outer_decisions) == len(current_outer_relations)
                ),
                overlap_degree=degree,
                phase_rescue_enabled=(
                    (
                        controller_v31_run_state.phase_rescue_enabled
                        or config.search_state_backend == "diagonal_cma"
                    )
                    and search_state_available
                ),
                repair_lock_active=controller_v31_run_state.non_dense_repair_locked,
                phase_i_tail_utility_value=(
                    max(
                        phase_i_tail_utility(phase_state),
                        cc_utility,
                    )
                    if phase_state is not None
                    else max(
                        controller_v31_run_state.phase_i_runtime_tail_utility,
                        cc_utility,
                    )
                ),
                relations=current_outer_relations,
                decisions=current_outer_decisions,
                writeback_norms=overlap_writeback_norms,
                relative_writeback_norms=relative_writeback_norms,
                fitness_deltas=fitness_delta_list,
                reference_fitness=guarded_incumbent_fitness,
                cc_utility_history=controller_v31_run_state.cc_utility_history,
                remaining_fes=config.max_fes - current_fitness_evaluations(fun),
                max_fes=config.max_fes,
                population_size=phase_population_size,
            )
            scheduler_state_before = (
                controller_v31_run_state.search_state_scheduler_state
            )
            state_plan = plan_search_state_action(
                evidence,
                controller_v31_run_state.search_state_scheduler_state,
                new_complete_cc_sweep=True,
                trajectory_action_name=trajectory_action_name_for_backend(config),
                terminal_probe=(config.search_state_backend == "diagonal_cma"),
            )
            if (
                state_plan.action_name
                in {RESUME_PHASE_I_SEARCH_STATE, CONTINUE_DIAGONAL_SEARCH_STATE}
                and state_plan.requested_fes > 0
            ):
                preempted_recovery = (
                    controller_v31_run_state.preempt_pending_trajectory_guard()
                )
                if preempted_recovery is not None:
                    best_individual = preempted_recovery.candidate.copy()
                controller_v31_run_state.invalidate_pending_action_trust(
                    "search_state_intervened_before_credit"
                )
                if (
                    state_plan.action_name == RESUME_PHASE_I_SEARCH_STATE
                    and not phase_state_available
                ):
                    raise RuntimeError(
                        "stateful MMES action selected without a resumable Phase-I state"
                    )
                guard_before = guarded_incumbent_fitness
                guard_vector = guarded_incumbent.copy()
                optimizer_seed = None
                if state_plan.action_name == CONTINUE_DIAGONAL_SEARCH_STATE:
                    (
                        next_search_state,
                        accepted,
                        state_candidate,
                        state_candidate_fitness,
                        block,
                        optimizer_seed,
                    ) = run_diagonal_search_state_block(
                        state=controller_v31_run_state.diagonal_cma_state,
                        requested_fes=state_plan.requested_fes,
                        guard_individual=guard_vector,
                        guard_fitness=guard_before,
                        fun=fun,
                        info=info,
                        config=config,
                        fun_name=fun_name,
                        fun_id=fun_id,
                        outer_iter=outer_iter,
                    )
                    controller_v31_run_state.diagonal_cma_state = next_search_state
                    raw_candidate = np.asarray(
                        block.state.best_x,
                        dtype=float,
                    ).reshape(-1)
                    sigma_before = float(block.sigma_before)
                    sigma_after = float(block.sigma_after)
                    population_before = int(block.population_size)
                    population_after = int(block.population_size)
                    raw_candidate_fitness = float(block.candidate_best)
                else:
                    (
                        next_phase_state,
                        accepted,
                        state_candidate,
                        state_candidate_fitness,
                        block,
                    ) = run_resumed_phase_i_state_block(
                        optimizer=phase_optimizer,
                        state=phase_state,
                        requested_fes=state_plan.requested_fes,
                        guard_individual=guard_vector,
                        guard_fitness=guard_before,
                        fun=fun,
                    )
                    controller_v31_run_state.phase_i_state = next_phase_state
                    raw_candidate = np.asarray(
                        block.state.best_so_far_x,
                        dtype=float,
                    ).reshape(-1)
                    sigma_before = float(getattr(phase_state, "sigma", config.sigma))
                    sigma_after = float(
                        getattr(next_phase_state, "sigma", config.sigma)
                    )
                    population_before = int(
                        getattr(phase_state, "n_individuals", phase_population_size)
                    )
                    population_after = int(
                        getattr(
                            next_phase_state,
                            "n_individuals",
                            phase_population_size,
                        )
                    )
                    raw_candidate_fitness = float(block.state.best_so_far_y)
                actual_state_fes = int(block.actual_fes)
                search_state_fe += actual_state_fes
                sum_fes += actual_state_fes
                (
                    best_individual,
                    guarded_incumbent,
                    guarded_incumbent_fitness,
                    candidate_protected,
                    cc_context_replaced,
                ) = apply_search_state_candidate(
                    context_individual=best_individual,
                    guard_individual=guarded_incumbent,
                    guard_fitness=guarded_incumbent_fitness,
                    candidate=state_candidate,
                    candidate_fitness=state_candidate_fitness,
                    accepted=accepted,
                    quarantine_context=(
                        state_plan.action_name == CONTINUE_DIAGONAL_SEARCH_STATE
                    ),
                )
                state_utility = normalized_gain_utility(
                    guard_before,
                    state_candidate_fitness,
                    actual_state_fes,
                )
                controller_v31_run_state.search_state_scheduler_state = (
                    record_search_state_outcome(
                        controller_v31_run_state.search_state_scheduler_state,
                        stage=state_plan.stage,
                        accepted=accepted,
                        utility=state_utility,
                        required_utility_ratio=state_plan.required_utility_ratio,
                        cc_utility=cc_utility,
                        used_fes=actual_state_fes,
                    )
                )
                action_trace_rows.append(
                    build_action_trace_row(
                        problem_id=_problem_id(fun_name, fun_id),
                        seed=config.seed,
                        outer_iter=outer_iter,
                        group_index=sub_num - 1,
                        selected_action_name=state_plan.action_name,
                        overlap_size=0,
                        previous_delta=cc_utility,
                        current_delta=max(0.0, guard_before - state_candidate_fitness),
                        state_mutated=accepted,
                        action_value_delta_norm=float(
                            np.linalg.norm(raw_candidate - guard_vector)
                        ),
                        downstream_consumed=True,
                        downstream_consumption_scope="subsequent_outer_iterations",
                        search_state_action_type=state_plan.action_name,
                        search_state_backend=config.search_state_backend,
                        candidate_protected=candidate_protected,
                        cc_context_replaced=cc_context_replaced,
                        stagnation_window=0,
                        delta_mean=float(np.linalg.norm(raw_candidate - guard_vector)),
                        sigma_before=sigma_before,
                        sigma_after=sigma_after,
                        population_before=population_before,
                        population_after=population_after,
                        escape_budget=actual_state_fes,
                        bipop_restart_mode=state_plan.stage,
                        restart_triggered=True,
                        restart_accepted=accepted,
                        best_before=guard_before,
                        restart_candidate_best=raw_candidate_fitness,
                        restart_relative_improvement=bipop_relative_improvement(
                            candidate_best=raw_candidate_fitness,
                            incumbent_fitness=guard_before,
                        ),
                        restart_acceptance_threshold=0.0,
                        best_after=state_candidate_fitness,
                        trace_event=state_plan.stage,
                        remaining_budget_ratio=(
                            max(0, config.max_fes - current_fitness_evaluations(fun))
                            / max(config.max_fes, 1)
                        ),
                        shared_var_count=0,
                        repair_lock_active=(
                            controller_v31_run_state.non_dense_repair_locked
                        ),
                        refresh_budget=state_plan.requested_fes,
                        continuation_reserve=state_plan.cc_reserve_fes,
                        optimizer_seed=optimizer_seed,
                        scheduler_phase=scheduler_state_before.phase,
                        decision_point=f"complete_cc_sweep:{outer_iter}",
                        cc_block_fe=sweep_fes,
                        cc_utility=cc_utility,
                        search_state_block_fe=actual_state_fes,
                        search_state_utility=state_utility,
                        required_utility_ratio=state_plan.required_utility_ratio,
                        state_action_fe=(
                            controller_v31_run_state.search_state_scheduler_state.intervention_fe
                        ),
                        cc_reserve_fe=state_plan.cc_reserve_fes,
                        state_fingerprint_before=str(
                            getattr(block, "state_fingerprint_before", "")
                        ),
                        state_fingerprint_after=str(
                            getattr(block, "state_fingerprint_after", "")
                        ),
                        pre_hold_evidence=(
                            pre_hold_evidence_snapshot if outer_iter == 0 else None
                        ),
                        search_state_evidence=evidence,
                    )
                )
            else:
                action_trace_rows.append(
                    build_action_trace_row(
                        problem_id=_problem_id(fun_name, fun_id),
                        seed=config.seed,
                        outer_iter=outer_iter,
                        group_index=sub_num - 1,
                        selected_action_name=CONTINUE_CANONICAL_CC,
                        overlap_size=0,
                        previous_delta=cc_utility,
                        current_delta=0.0,
                        state_mutated=False,
                        action_value_delta_norm=0.0,
                        downstream_consumed=False,
                        downstream_consumption_scope="scheduler_abstention",
                        search_state_action_type=CONTINUE_CANONICAL_CC,
                        search_state_backend=config.search_state_backend,
                        restart_triggered=False,
                        restart_accepted=False,
                        best_before=guarded_incumbent_fitness,
                        restart_candidate_best=guarded_incumbent_fitness,
                        restart_relative_improvement=0.0,
                        restart_acceptance_threshold=0.0,
                        best_after=guarded_incumbent_fitness,
                        trace_event="decision",
                        remaining_budget_ratio=(
                            max(0, config.max_fes - current_fitness_evaluations(fun))
                            / max(config.max_fes, 1)
                        ),
                        repair_lock_active=(
                            controller_v31_run_state.non_dense_repair_locked
                        ),
                        scheduler_phase=scheduler_state_before.phase,
                        decision_point=f"complete_cc_sweep:{outer_iter}",
                        cc_block_fe=sweep_fes,
                        cc_utility=cc_utility,
                        search_state_block_fe=0,
                        required_utility_ratio=state_plan.required_utility_ratio,
                        state_action_fe=scheduler_state_before.intervention_fe,
                        cc_reserve_fe=state_plan.cc_reserve_fes,
                        abstain_reason=state_plan.trigger_reason,
                        pre_hold_evidence=(
                            pre_hold_evidence_snapshot if outer_iter == 0 else None
                        ),
                        search_state_evidence=evidence,
                    )
                )
                if config.search_state_backend == "diagonal_cma":
                    controller_v31_run_state.search_state_scheduler_state = (
                        SearchStateSchedulerState(
                            phase=SEARCH_STATE_BLOCKED,
                            probe_utilities=scheduler_state_before.probe_utilities,
                            intervention_fe=scheduler_state_before.intervention_fe,
                        )
                    )

        if (
            component_credit_trace is not None
            and len(fitness_delta_list) == sub_num
        ):
            component_credit_trace.complete_sweep(
                outer_iter=outer_iter,
                optimized_group_count=len(fitness_delta_list),
            )
        previous_group_contribution_credit = fitness_delta_list
        outer_iter += 1
        if cc_harm_guard_consumed:
            break

    if controller_v31_run_state is not None:
        preempted_recovery = (
            controller_v31_run_state.preempt_pending_trajectory_guard()
        )
        if preempted_recovery is not None:
            best_individual = preempted_recovery.candidate.copy()
    if component_credit_trace is not None:
        component_credit_trace.finalize_unresolved(
            resolution_fe=current_fitness_evaluations(fun)
        )

    problem_id = _problem_id(fun_name, fun_id)
    if car_artifacts_enabled:
        needs_terminal_checkpoint = (
            (
                config.car_actionability_arm != "off"
                and car_actionability_trace_base_row is None
            )
            or not car_state_ledger_rows
        )
        if needs_terminal_checkpoint and not math.isfinite(
            float(guarded_incumbent_fitness)
        ):
            if current_fitness_evaluations(fun) >= config.max_fes:
                raise RuntimeError(
                    "cannot establish a finite CAR actionability checkpoint"
                )
            guarded_incumbent_fitness = float(fun(guarded_incumbent)[0])
            sum_fes += 1
            cc_phase_fe += 1
        if car_actionability_trace_base_row is not None:
            terminal_fe = current_fitness_evaluations(fun)
            terminal_shortfall = max(0, config.max_fes - terminal_fe)
            termination_reason = (
                "early_guard"
                if cc_harm_guard_consumed
                else "early_termination"
                if terminal_shortfall > terminal_completion_tolerance_fe
                else "population_complete_budget_endpoint"
            )
            car_actionability_trace_base_row = {
                **car_actionability_trace_base_row,
                "terminal_completion_tolerance_fe": str(
                    terminal_completion_tolerance_fe
                ),
                "termination_reason": termination_reason,
                "terminal_fe_shortfall": str(terminal_shortfall),
            }
        if (
            config.car_actionability_arm != "off"
            and car_actionability_trace_base_row is None
        ):
            audit_checkpoint = BranchState(
                incumbent=tuple(float(value) for value in guarded_incumbent),
                committed_fitness=guarded_incumbent_fitness,
                evaluator_record=[],
                state_fingerprint="",
                state_payload=_car_controller_state_payload(
                    controller_v31_run_state,
                    trajectory_mean_cache=trajectory_mean_cache,
                    previous_group_contribution_credit=(
                        previous_group_contribution_credit
                    ),
                ),
            )
            audit_checkpoint.state_fingerprint = fingerprint_branch_state(
                audit_checkpoint
            )
            missing_audit = execute_car_actionability_arm_at_barrier(
                decision=CARPlanDecision(
                    plan=None,
                    evidence=None,
                    abstain_reason=(
                        "insufficient_complete_evidence_sweeps"
                        if car_probe_enabled
                        else "no_overlap_component_candidate"
                    ),
                ),
                checkpoint=audit_checkpoint,
                checkpoint_fe=current_fitness_evaluations(fun),
                prefix_record=tuple(float(value) for value in fun.fitness_record),
                fun_name=fun_name,
                fun_id=fun_id,
                output_path=output_path,
                info=info,
                config=config,
                problem_id=problem_id,
            )
            car_actionability_trace_base_row = {
                **missing_audit.trace_base_row,
                "terminal_completion_tolerance_fe": str(
                    terminal_completion_tolerance_fe
                ),
                "termination_reason": (
                    "early_guard"
                    if cc_harm_guard_consumed
                    else "early_termination"
                    if max(0, config.max_fes - current_fitness_evaluations(fun))
                    > terminal_completion_tolerance_fe
                    else "population_complete_budget_endpoint"
                ),
                "terminal_fe_shortfall": str(
                    max(0, config.max_fes - current_fitness_evaluations(fun))
                ),
            }
        if not car_state_ledger_rows:
            checkpoint = BranchState(
                incumbent=tuple(float(value) for value in guarded_incumbent),
                committed_fitness=guarded_incumbent_fitness,
                evaluator_record=[],
                state_fingerprint="",
                state_payload=_car_controller_state_payload(
                    controller_v31_run_state,
                    trajectory_mean_cache=trajectory_mean_cache,
                    previous_group_contribution_credit=(
                        previous_group_contribution_credit
                    ),
                ),
            )
            checkpoint.state_fingerprint = fingerprint_branch_state(checkpoint)
            incomplete = execute_car_w_probe_at_barrier(
                decision=CARPlanDecision(
                    plan=None,
                    evidence=None,
                    abstain_reason=(
                        "insufficient_complete_evidence_sweeps"
                        if car_probe_enabled
                        else "no_overlap_component_candidate"
                    ),
                ),
                checkpoint=checkpoint,
                checkpoint_fe=current_fitness_evaluations(fun),
                fun_name=fun_name,
                fun_id=fun_id,
                output_path=output_path,
                info=info,
                config=config,
                problem_id=problem_id,
            )
            car_state_ledger_rows.extend(incomplete.state_ledger_rows)
        _write_car_rows(
            case_artifact_path(output_path, problem_id, "car_probe_trace.csv"),
            fieldnames=CAR_PROBE_TRACE_FIELDS,
            rows=car_probe_trace_rows,
        )
        _write_car_rows(
            case_artifact_path(output_path, problem_id, "car_state_ledger.csv"),
            fieldnames=CAR_STATE_LEDGER_FIELDS,
            rows=car_state_ledger_rows,
        )
        _write_car_rows(
            case_artifact_path(output_path, problem_id, "car_branch_manifest.csv"),
            fieldnames=CAR_BRANCH_MANIFEST_FIELDS,
            rows=car_branch_manifest_rows,
        )
        if (
            config.car_actionability_arm != "off"
            and car_actionability_trace_base_row is not None
        ):
            _write_car_rows(
                case_artifact_path(
                    output_path,
                    problem_id,
                    "car_actionability_trace.csv",
                ),
                fieldnames=CAR_ACTIONABILITY_TRACE_FIELDS,
                rows=finalize_car_actionability_trace(
                    trace_base_row=car_actionability_trace_base_row,
                    fitness_record=fun.fitness_record,
                    max_fes=config.max_fes,
                ),
            )
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
    )
    print(f"{problem_id} overlap relations extracted: {len(relations)}")
    return fun.fitness_record, time.time() - time_start, action_trace_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARAC-owned HCC smoke runner.")
    parser.add_argument("--functions", nargs="+", choices=FUNCTION_NAMES, required=True)
    parser.add_argument("--ids", nargs="+", type=int, choices=PROBLEM_IDS, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--aob-data-root",
        type=lambda value: Path(value).resolve(),
        default=DATA_DIR.resolve(),
    )
    parser.add_argument("--timestamp", default="arac-hcc-smoke")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-fes", type=int, required=True)
    parser.add_argument("--verbose", type=int, default=1000)
    parser.add_argument("--early-stopping-evaluations", type=int, default=1000)
    parser.add_argument("--mmes-restart", dest="mmes_restart", action="store_true", default=True)
    parser.add_argument("--no-mmes-restart", dest="mmes_restart", action="store_false")
    parser.add_argument("--cmaes-restart", dest="cmaes_restart", action="store_true", default=True)
    parser.add_argument("--no-cmaes-restart", dest="cmaes_restart", action="store_false")
    parser.add_argument("--budget-accounting", default="strict", choices=["strict", "source"])
    parser.add_argument(
        "--search-state-backend",
        default="phase_i_mmes",
        choices=["phase_i_mmes", "diagonal_cma"],
    )
    parser.add_argument(
        "--car-branch-order",
        choices=["fallback_first", "candidate_first"],
        default="fallback_first",
    )
    parser.add_argument(
        "--lane-profile",
        default="runtime_smoke",
        choices=[
            "runtime_smoke",
            "targeted_ablation",
            "focused_core",
            "focused_compare",
            "landscape_escape",
            "repair_landscape_escape",
            "repair_refine",
            "evidence_routed_only",
            "evidence_routed_v2_only",
            "evidence_routed_v21_only",
            "evidence_routed_v22_only",
            "evidence_routed_v23_only",
            "evidence_routed_v24_only",
            "evidence_routed_v25_only",
            "evidence_routed_v26_only",
            "paper_best_win_push",
            "precision_refine_push",
            "phase_rescue_push",
            "repair_phase_rescue_push",
            "cc_harm_sep_refresh",
            "separable_cmaes_push",
            "evidence_action_controller_v1",
            "evidence_action_controller_v2",
            "evidence_action_controller_v3",
            "evidence_action_controller_v31",
        ],
        help=(
            "Accepted for experiment-runner CLI compatibility; lane expansion is "
            "handled by experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py."
        ),
    )
    parser.add_argument("--enable-relation-dispatch", action="store_true")
    parser.add_argument(
        "--relation-policy",
        default="rule",
        choices=[
            "rule",
            "adaptive_v2",
            "adaptive_v21",
            "adaptive_v22",
            "adaptive_v23",
            "adaptive_v24",
            "adaptive_v25",
            "adaptive_v26",
            "controller_v3",
            "controller_v31",
            "shuffled",
            "lagged",
        ],
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--car-candidate-mode",
        choices=["graph", "shuffled_graph", "paired_fallback"],
        default="graph",
    )
    parser.add_argument(
        "--car-actionability-arm",
        choices=["off", "fallback", "candidate"],
        default="off",
        help="Offline paired-lane actionability condition; never used for runtime dispatch.",
    )
    parser.add_argument("--arac-action-file", type=Path, default=None)
    parser.add_argument(
        "--arac-action",
        default="conservative_no_action",
        choices=[
            "conservative_no_action",
            "repair_shared_variable_binding",
            "isolate_conflicting_relation",
            "protect_high_margin_group",
            "allow_beneficial_coordination",
            "budget_shift_mean_blend",
            "budget_shift_only",
            "mean_blend_only",
            SEARCH_STATE_BIPOP_ACTION,
            REPAIR_BIPOP_SEARCH_STATE_ACTION,
            PHASE_RESCUE_MULTISTART_ACTION,
            REPAIR_PHASE_RESCUE_MULTISTART_ACTION,
            CC_HARM_GUARDED_SEP_REFRESH_ACTION,
            SEPARABLE_CMAES_DISPATCH_ACTION,
            REPAIR_PROTECT_REFINE_ACTION,
            REPAIR_PROTECT_DEEP_REFINE_ACTION,
            EVIDENCE_ACTION_CONTROLLER_V1,
            EVIDENCE_ACTION_CONTROLLER_V2,
            EVIDENCE_ACTION_CONTROLLER_V3,
            EVIDENCE_ACTION_CONTROLLER_V31,
            EVIDENCE_ACTION_CONTROLLER_V32,
            EVIDENCE_ACTION_CONTROLLER_V33,
            EVIDENCE_ACTION_CONTROLLER_V34,
            EVIDENCE_ACTION_CONTROLLER_V35,
            EVIDENCE_ACTION_CONTROLLER_V36,
            EVIDENCE_ACTION_CONTROLLER_V37,
            EVIDENCE_ACTION_CONTROLLER_V38,
            EVIDENCE_ACTION_CONTROLLER_V39,
            EVIDENCE_ACTION_CONTROLLER_V40,
            CAR_W_ACTION,
            CAR_W2_ACTION,
            CAR_W3_ACTION,
        ],
    )
    args = parser.parse_args(argv)
    if args.arac_action_file is not None:
        parser.error("--arac-action-file is not supported by the smoke runner yet")
    if args.car_actionability_arm != "off" and args.arac_action != CAR_W3_ACTION:
        parser.error("--car-actionability-arm requires --arac-action CAR-W3")
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
        car_branch_order=args.car_branch_order,
        car_candidate_mode=args.car_candidate_mode,
        car_actionability_arm=args.car_actionability_arm,
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
                include_trust_fields=uses_v33_trust_trace_schema(
                    config.arac_action
                ),
                include_recovery_fields=is_evidence_action_controller_v34(
                    config.arac_action
                ),
                include_maturity_fields=is_evidence_action_controller_v36(
                    config.arac_action
                )
                or is_evidence_action_controller_v37(config.arac_action)
                or is_evidence_action_controller_v38(config.arac_action),
                include_cma_sigma_fields=is_evidence_action_controller_v39(
                    config.arac_action
                ),
                include_resource_fields=(
                    is_evidence_action_controller_v37(config.arac_action)
                    or is_evidence_action_controller_v38(config.arac_action)
                    or is_evidence_action_controller_v39(config.arac_action)
                    or is_evidence_action_controller_v40(config.arac_action)
                ),
                include_component_credit_fields=is_evidence_action_controller_v40(
                    config.arac_action
                ),
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
            include_trust_fields=uses_v33_trust_trace_schema(
                config.arac_action
            ),
            include_recovery_fields=is_evidence_action_controller_v34(
                config.arac_action
            ),
            include_maturity_fields=is_evidence_action_controller_v36(
                config.arac_action
            )
            or is_evidence_action_controller_v37(config.arac_action)
            or is_evidence_action_controller_v38(config.arac_action),
            include_cma_sigma_fields=is_evidence_action_controller_v39(
                config.arac_action
            ),
            include_resource_fields=(
                is_evidence_action_controller_v37(config.arac_action)
                or is_evidence_action_controller_v38(config.arac_action)
                or is_evidence_action_controller_v39(config.arac_action)
                or is_evidence_action_controller_v40(config.arac_action)
            ),
            include_component_credit_fields=is_evidence_action_controller_v40(
                config.arac_action
            ),
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
