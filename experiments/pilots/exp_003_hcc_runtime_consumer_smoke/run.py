from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ARAC_REPO_ROOT = Path(__file__).resolve().parents[3]
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"
for import_root in (ARAC_REPO_ROOT, ARAC_SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.paths import (
    experiment_results_dir,
    repository_root,
    resolve_repository_path,
)

ARAC_REPO_ROOT = repository_root()
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"

from arac.actions import ActionDecision, ActionFamily
from arac.actions.controller_profiles import (
    CONTROLLER_PROFILES,
    ControllerProfile,
    controller_has_capability,
    controller_lane_profile_names,
    controller_profile_by_action,
    controller_profile_by_version,
)
from arac.audits import claim_gate
from arac.execution import BackendSemanticsDiff
from arac.execution.environment import (
    EnvironmentProbe,
    PINNED_HCC_RUNTIME_ENVIRONMENT,
    require_pinned_hcc_runtime_environment,
)
from arac.backends.hcc import (
    DEFAULT_AOB_DATA_ROOT,
    HCC_VENDOR_PATHS,
    HCC_VENDOR_ROOT,
    HccActionExecutionPlan,
    HccAobExecutionRequest,
    HccAobExecutionResult,
    HccVendorPaths,
    _find_hcc_action_trace,
    _parse_hcc_budget_summary,
    _parse_hcc_evaluation_record_with_optimizer_final_fe,
    build_hcc_action_execution_plan,
    hcc_backend_semantics_for,
    required_aob_data_files,
    resolve_hcc_vendor_paths,
    run_hcc_aob_smoke_execution,
)
from arac.evaluation import SameBudgetLedger
from arac.evaluation import classify_utility, relative_gain
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS, validate_runtime_payload
from arac.policy.counterfactual_action_racing import (
    AuditEnvelope,
    DispatchEvidence,
)
from arac.policy.component_delayed_credit import COMPONENT_CREDIT_TRACE_FIELDS
from arac.policy.causal_risk_scheduler import (
    FEATURE_SCHEMA_SHA256,
    PRE_ACTION_UTILITY_SCHEMA_VERSION,
    PRECISION_CAUSAL_DIAGNOSTIC_TRACE_FIELDS,
    UTILITY_FEATURE_NAMES,
)
from arac.policy.oracle_actionability import (
    CAR_ACTIONABILITY_HORIZON_LABELS,
    CAR_ACTIONABILITY_HORIZON_MULTIPLIERS,
    CAR_ACTIONABILITY_PROTOCOL_VERSION,
    log_actionability_advantage,
)

RUN_ID = "exp_003_hcc_runtime_consumer_smoke"
PROBLEM_ID = "E2"
DEFAULT_SEEDS = (1, 2, 3)
MAX_FES = 2_000
LOW_ACTIVE_DENSITY_THRESHOLD = 0.20
MEANINGFUL_GAIN_THRESHOLD = 0.05
FORMAL_SOTA_MIN_SEEDS = 25
FORMAL_SOTA_PROBLEMS = tuple(
    f"{prefix}{problem_id}"
    for prefix in ("E", "S", "R", "A")
    for problem_id in range(1, 7)
)
PRECISION_CAUSAL_PROTOCOL_VERSION = "precision-causal-logging-v1"
PRECISION_CAUSAL_RANDOMIZATION_SALT = "arac-precision-causal-logged-arm-v1"
PRECISION_CAUSAL_ARMS = ("baseline", "action")
PRECISION_CAUSAL_PREREGISTRATION_PATH = (
    "docs/superpowers/specs/2026-07-15-causal-risk-precision-scheduler-design.md"
)
PRECISION_CAUSAL_PREREGISTRATION_SHA256 = (
    "f566533ccd17c14fad2acf936c09668892183e872ebd6cf3ab57026b20797d26"
)
PRECISION_CAUSAL_PREREGISTRATION_COMMIT = (
    "f7960eafc27f64f519d0d2137f5a2c4152b715c3"
)
PRECISION_CAUSAL_FEATURE_FORMULAS = {
    "remaining_fe_ratio": "(max_fes - decision_fe) / max_fes",
    "revisit_cap_remaining_ratio": "scheduler_revisit_cap_fe / remaining_fe",
    "component_group_fraction": "component_group_count / total_group_count",
    "component_shared_variable_ratio": "component_shared_variable_count / dimension",
    "component_mean_overlap_ratio": "mean(pair_shared_count / min(pair_group_sizes)) over component overlap edges",
    "proposal_disagreement_mean_2": "mean(last two completed component-sweep normalized proposal disagreements)",
    "candidate_dose_ratio": "precision_sigma / v37_normal_refine_sigma",
    "phase_i_tail_progress_rate": "normalized Phase-I tail best-so-far progress per FE",
    "cc_progress_rate_last": "last completed CC sweep normalized progress per FE",
    "cc_progress_rate_slope_4": "OLS slope of last four completed CC progress rates",
    "cc_progress_rate_std_4": "population standard deviation of last four completed CC progress rates",
    "cc_stagnation_streak": "trailing completed CC progress rates <= 1e-8",
    "terminal_sigma_ratio_last": "last same-group terminal CMA sigma / initial sigma",
    "log_sigma_slope_3": "OLS slope of log terminal sigma ratios over last three same-group blocks",
    "success_generation_ratio_last": "generations improving running best / observed generations in last same-group block",
    "offspring_diversity_ratio_last": "mean offspring distance to batch centroid / parameter-space diagonal in last same-group block",
}


@dataclass(frozen=True)
class LaneConfig:
    lane_id: str
    action_family: ActionFamily
    selected_action_name: str
    runner_action_name: str
    dispatch_scope: str
    relation_dispatch_enabled: bool = False
    plan_action_name: str = ""
    relation_policy_mode: str = "rule"
    negative_control: bool = False
    car_candidate_mode: str = "graph"
    car_actionability_arm: str = "off"
    precision_causal_arm: str = "off"


LANES = (
    LaneConfig(
        "fallback",
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "conservative_no_action",
        "fixed_lane_runtime_consumer_smoke",
    ),
    LaneConfig(
        "fixed_repair",
        ActionFamily.REASSIGN_REPAIR,
        "repair_shared_variable_binding",
        "repair_shared_variable_binding",
        "fixed_lane_runtime_consumer_smoke",
    ),
    LaneConfig(
        "fixed_coordinate",
        ActionFamily.COORDINATE,
        "allow_beneficial_coordination",
        "allow_beneficial_coordination",
        "fixed_lane_runtime_consumer_smoke",
    ),
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_rule",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
    ),
    LaneConfig(
        "shuffled_relation_dispatch",
        ActionFamily.COORDINATE,
        "shuffled_relation_dispatch",
        "conservative_no_action",
        "shuffled_relation_dispatch_negative_control",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="shuffled",
        negative_control=True,
    ),
)
TARGETED_ABLATION_LANES = (
    *LANES,
    LaneConfig(
        "trajectory_budget_shift_mean_blend",
        ActionFamily.TRAJECTORY,
        "budget_shift_mean_blend",
        "budget_shift_mean_blend",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
LANDSCAPE_ESCAPE_LANES = (
    LANES[0],
    LANES[1],
    LANES[2],
    LaneConfig(
        "bipop_search_state_restart",
        ActionFamily.TRAJECTORY,
        "bipop_search_state_restart",
        "bipop_search_state_restart",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
REPAIR_LANDSCAPE_ESCAPE_LANES = (
    LANES[0],
    LANES[1],
    LaneConfig(
        "repair_bipop_search_state_restart",
        ActionFamily.TRAJECTORY,
        "repair_bipop_search_state_restart",
        "repair_bipop_search_state_restart",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
REPAIR_REFINE_LANES = (
    LANES[0],
    LANES[1],
    LaneConfig(
        "repair_protect_refine",
        ActionFamily.TRAJECTORY,
        "repair_protect_refine",
        "repair_protect_refine",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
PRECISION_REFINE_PUSH_LANES = (
    LANES[0],
    REPAIR_REFINE_LANES[-1],
    LaneConfig(
        "repair_protect_deep_refine",
        ActionFamily.TRAJECTORY,
        "repair_protect_deep_refine",
        "repair_protect_deep_refine",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
PHASE_RESCUE_PUSH_LANES = (
    LANES[0],
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v26",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v26",
    ),
    REPAIR_REFINE_LANES[-1],
    LaneConfig(
        "phase_rescue_multistart",
        ActionFamily.TRAJECTORY,
        "phase_rescue_multistart",
        "phase_rescue_multistart",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
REPAIR_PHASE_RESCUE_PUSH_LANES = (
    LANES[0],
    REPAIR_REFINE_LANES[-1],
    LaneConfig(
        "repair_phase_rescue_multistart",
        ActionFamily.TRAJECTORY,
        "repair_phase_rescue_multistart",
        "repair_phase_rescue_multistart",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
CC_HARM_SEP_REFRESH_LANES = (
    LANES[0],
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v26",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v26",
    ),
    LaneConfig(
        "cc_harm_guarded_sep_refresh",
        ActionFamily.TRAJECTORY,
        "cc_harm_guarded_sep_refresh",
        "cc_harm_guarded_sep_refresh",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
SEPARABLE_CMAES_PUSH_LANES = (
    LANES[0],
    LaneConfig(
        "separable_cmaes_dispatch_action",
        ActionFamily.TRAJECTORY,
        "separable_cmaes_dispatch_action",
        "separable_cmaes_dispatch_action",
        "fixed_lane_runtime_consumer_smoke",
    ),
)
FOCUSED_CORE_LANES = (
    LANES[0],
    LANES[1],
    LANES[2],
    LANES[3],
)
FOCUSED_COMPARE_LANES = (
    LANES[0],
    LANES[1],
    LANES[2],
)
EVIDENCE_ROUTED_ONLY_LANES = (
    LANES[3],
)
EVIDENCE_ROUTED_V2_ONLY_LANES = (
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v2",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v2",
    ),
)
EVIDENCE_ROUTED_V21_ONLY_LANES = (
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v21",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v21",
    ),
)
EVIDENCE_ROUTED_V22_ONLY_LANES = (
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v22",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v22",
    ),
)
EVIDENCE_ROUTED_V23_ONLY_LANES = (
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v23",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v23",
    ),
)
EVIDENCE_ROUTED_V24_ONLY_LANES = (
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v24",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v24",
    ),
)
EVIDENCE_ROUTED_V25_ONLY_LANES = (
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v25",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v25",
    ),
)
EVIDENCE_ROUTED_V26_ONLY_LANES = (
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_adaptive_v26",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v26",
    ),
)
RELATION_DISPATCH_V24_LANE = LaneConfig(
    "relation_dispatch_v24",
    ActionFamily.COORDINATE,
    "relation_dispatch_adaptive_v24",
    "conservative_no_action",
    "per_overlap_relation_runtime_dispatch",
    relation_dispatch_enabled=True,
    plan_action_name="allow_beneficial_coordination",
    relation_policy_mode="adaptive_v24",
)
RELATION_DISPATCH_LEGACY_LANE = LaneConfig(
    "relation_dispatch_legacy",
    ActionFamily.COORDINATE,
    "relation_dispatch_rule",
    "conservative_no_action",
    "per_overlap_relation_runtime_dispatch",
    relation_dispatch_enabled=True,
    plan_action_name="allow_beneficial_coordination",
    relation_policy_mode="rule",
)
PAPER_BEST_WIN_PUSH_LANES = (
    LANES[0],
    EVIDENCE_ROUTED_V26_ONLY_LANES[0],
    LANES[1],
    REPAIR_REFINE_LANES[-1],
    LANDSCAPE_ESCAPE_LANES[-1],
    SEPARABLE_CMAES_PUSH_LANES[-1],
)
PAPER_BEST_WIN_PUSH_V2_LANES = (
    LANES[0],
    EVIDENCE_ROUTED_V26_ONLY_LANES[0],
    RELATION_DISPATCH_V24_LANE,
    RELATION_DISPATCH_LEGACY_LANE,
    LANES[1],
    REPAIR_REFINE_LANES[-1],
    LANDSCAPE_ESCAPE_LANES[-1],
    SEPARABLE_CMAES_PUSH_LANES[-1],
)
HISTORICAL_ANCHOR_REFINE_PUSH_LANES = (
    LANES[0],
    EVIDENCE_ROUTED_V26_ONLY_LANES[0],
    RELATION_DISPATCH_LEGACY_LANE,
    LANES[1],
    REPAIR_REFINE_LANES[-1],
    PRECISION_REFINE_PUSH_LANES[-1],
    PHASE_RESCUE_PUSH_LANES[-1],
    REPAIR_PHASE_RESCUE_PUSH_LANES[-1],
    LANDSCAPE_ESCAPE_LANES[-1],
    SEPARABLE_CMAES_PUSH_LANES[-1],
)
HISTORICAL_13_PRESERVE_PUSH_LANES = (
    LANES[0],
    LANES[2],
    EVIDENCE_ROUTED_V26_ONLY_LANES[0],
    RELATION_DISPATCH_V24_LANE,
    RELATION_DISPATCH_LEGACY_LANE,
    LANES[1],
    REPAIR_REFINE_LANES[-1],
    PRECISION_REFINE_PUSH_LANES[-1],
    PHASE_RESCUE_PUSH_LANES[-1],
    REPAIR_PHASE_RESCUE_PUSH_LANES[-1],
    CC_HARM_SEP_REFRESH_LANES[-1],
    LANDSCAPE_ESCAPE_LANES[-1],
    SEPARABLE_CMAES_PUSH_LANES[-1],
)
HISTORICAL_13_FAST_PRESERVE_LANES = (
    LANES[0],
    LANES[2],
    EVIDENCE_ROUTED_V26_ONLY_LANES[0],
    RELATION_DISPATCH_V24_LANE,
    REPAIR_REFINE_LANES[-1],
    REPAIR_PHASE_RESCUE_PUSH_LANES[-1],
    LANDSCAPE_ESCAPE_LANES[-1],
)
HISTORICAL_13_RUNTIME_COMPOSITE_LANES = (
    LaneConfig(
        "arac_runtime_composite_v1",
        ActionFamily.TRAJECTORY,
        "repair_phase_rescue_multistart",
        "repair_phase_rescue_multistart",
        "single_run_relation_dispatch_plus_repair_phase_rescue",
        relation_dispatch_enabled=True,
        relation_policy_mode="adaptive_v26",
    ),
)
HISTORICAL_13_RUNTIME_COMPOSITE_V2_LANES = (
    LaneConfig(
        "arac_runtime_composite_v2",
        ActionFamily.TRAJECTORY,
        "cc_harm_guarded_sep_refresh",
        "cc_harm_guarded_sep_refresh",
        "single_run_relation_dispatch_plus_cc_harm_guarded_sep_refresh",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="adaptive_v26",
    ),
)
EVIDENCE_ACTION_CONTROLLER_V1_LANES = (
    LaneConfig(
        "arac_evidence_action_controller_v1",
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v1",
        "arac_evidence_action_controller_v1",
        "single_run_runtime_evidence_to_action_controller",
        relation_dispatch_enabled=True,
        plan_action_name="arac_evidence_action_controller_v1",
        relation_policy_mode="adaptive_v26",
    ),
)
EVIDENCE_ACTION_CONTROLLER_V2_LANES = (
    LaneConfig(
        "arac_evidence_action_controller_v2",
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v2",
        "arac_evidence_action_controller_v2",
        "single_run_relation_first_evidence_to_action_controller",
        relation_dispatch_enabled=True,
        plan_action_name="arac_evidence_action_controller_v2",
        relation_policy_mode="adaptive_v24",
    ),
)
EVIDENCE_ACTION_CONTROLLER_V3_LANES = (
    LaneConfig(
        "arac_evidence_action_controller_v3",
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v3",
        "arac_evidence_action_controller_v3",
        "single_run_runtime_evidence_controller_v3",
        relation_dispatch_enabled=True,
        plan_action_name="arac_evidence_action_controller_v3",
        relation_policy_mode="controller_v3",
    ),
)
EVIDENCE_ACTION_CONTROLLER_V31_LANES = (
    LaneConfig(
        "arac_evidence_action_controller_v31",
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v31",
        "arac_evidence_action_controller_v31",
        "single_run_guarded_runtime_evidence_controller_v31",
        relation_dispatch_enabled=True,
        plan_action_name="arac_evidence_action_controller_v31",
        relation_policy_mode="controller_v31",
    ),
)
EVIDENCE_ACTION_CONTROLLER_V32_LANES = (
    LaneConfig(
        "arac_evidence_action_controller_v32",
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v32",
        "arac_evidence_action_controller_v32",
        "single_run_guarded_runtime_evidence_controller_v32",
        relation_dispatch_enabled=True,
        plan_action_name="arac_evidence_action_controller_v32",
        relation_policy_mode="controller_v31",
    ),
)
def _lane_from_controller_profile(
    profile: ControllerProfile,
    *,
    lane_id: str | None = None,
    dispatch_scope: str | None = None,
    car_candidate_mode: str = "graph",
    car_actionability_arm: str = "off",
    precision_causal_arm: str = "off",
) -> LaneConfig:
    return LaneConfig(
        lane_id or profile.action_name,
        ActionFamily.TRAJECTORY,
        profile.action_name,
        profile.action_name,
        dispatch_scope or profile.dispatch_scope,
        relation_dispatch_enabled=True,
        plan_action_name=profile.action_name,
        relation_policy_mode=profile.relation_policy_mode,
        car_candidate_mode=car_candidate_mode,
        car_actionability_arm=car_actionability_arm,
        precision_causal_arm=precision_causal_arm,
    )


CONTROLLER_LANES_BY_PROFILE = {
    profile.lane_profile: (_lane_from_controller_profile(profile),)
    for profile in CONTROLLER_PROFILES
}
EVIDENCE_ACTION_CONTROLLER_V33_LANES = CONTROLLER_LANES_BY_PROFILE[
    "evidence_action_controller_v33"
]
EVIDENCE_ACTION_CONTROLLER_V34_LANES = CONTROLLER_LANES_BY_PROFILE[
    "evidence_action_controller_v34"
]
EVIDENCE_ACTION_CONTROLLER_V35_LANES = CONTROLLER_LANES_BY_PROFILE[
    "evidence_action_controller_v35"
]
EVIDENCE_ACTION_CONTROLLER_V36_LANES = CONTROLLER_LANES_BY_PROFILE[
    "evidence_action_controller_v36"
]
EVIDENCE_ACTION_CONTROLLER_V37_LANES = CONTROLLER_LANES_BY_PROFILE[
    "evidence_action_controller_v37"
]
EVIDENCE_ACTION_CONTROLLER_V38_LANES = CONTROLLER_LANES_BY_PROFILE[
    "evidence_action_controller_v38"
]
EVIDENCE_ACTION_CONTROLLER_V39_LANES = CONTROLLER_LANES_BY_PROFILE[
    "evidence_action_controller_v39"
]

PAIRED_V33_V36_RUNTIME_UTILITY_LANES = (
    _lane_from_controller_profile(
        controller_profile_by_version(33),
        lane_id="fallback",
        dispatch_scope="paired_v33_runtime_fallback_reference",
    ),
    _lane_from_controller_profile(
        controller_profile_by_version(36),
        lane_id="candidate",
        dispatch_scope="paired_v36_runtime_candidate",
    ),
    LANES[4],
    LaneConfig(
        "no_action_negative_control",
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "conservative_no_action",
        "paired_no_action_negative_control",
        negative_control=True,
    ),
)

CAR_W_DIAGNOSTIC_LANES = (
    _lane_from_controller_profile(
        controller_profile_by_version(33),
        lane_id="v33_fallback",
        dispatch_scope="car_w_diagnostic_v33_fallback_reference",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w"),
        lane_id="car_w",
        dispatch_scope="car_w_diagnostic_graph_candidate",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w"),
        lane_id="car_w_shuffled",
        dispatch_scope="car_w_diagnostic_shuffled_graph_control",
        car_candidate_mode="shuffled_graph",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w"),
        lane_id="car_w_paired_fallback",
        dispatch_scope="car_w_diagnostic_paired_fallback_control",
        car_candidate_mode="paired_fallback",
    ),
    LaneConfig(
        "no_action_negative_control",
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "conservative_no_action",
        "car_w_diagnostic_no_action_control",
        negative_control=True,
    ),
)
CAR_W2_DIAGNOSTIC_LANES = (
    _lane_from_controller_profile(
        controller_profile_by_version(33),
        lane_id="v33_fallback",
        dispatch_scope="car_w2_diagnostic_v33_fallback_reference",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w2"),
        lane_id="car_w2",
        dispatch_scope="car_w2_diagnostic_graph_candidate",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w2"),
        lane_id="car_w2_shuffled",
        dispatch_scope="car_w2_diagnostic_shuffled_graph_control",
        car_candidate_mode="shuffled_graph",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w2"),
        lane_id="car_w2_paired_fallback",
        dispatch_scope="car_w2_diagnostic_paired_fallback_control",
        car_candidate_mode="paired_fallback",
    ),
    LaneConfig(
        "no_action_negative_control",
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "conservative_no_action",
        "car_w2_diagnostic_no_action_control",
        negative_control=True,
    ),
)
CAR_W3_DIAGNOSTIC_LANES = (
    _lane_from_controller_profile(
        controller_profile_by_version(33),
        lane_id="v33_fallback",
        dispatch_scope="car_w3_diagnostic_v33_fallback_reference",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w3"),
        lane_id="car_w3",
        dispatch_scope="car_w3_diagnostic_graph_candidate",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w3"),
        lane_id="car_w3_shuffled",
        dispatch_scope="car_w3_diagnostic_shuffled_graph_control",
        car_candidate_mode="shuffled_graph",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w3"),
        lane_id="car_w3_paired_fallback",
        dispatch_scope="car_w3_diagnostic_paired_fallback_control",
        car_candidate_mode="paired_fallback",
    ),
    LaneConfig(
        "no_action_negative_control",
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "conservative_no_action",
        "car_w3_diagnostic_no_action_control",
        negative_control=True,
    ),
)
CAR_ACTIONABILITY_AUDIT_LANES = (
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w3"),
        lane_id="oracle_fallback",
        dispatch_scope="offline_actionability_fallback_continuation",
        car_actionability_arm="fallback",
    ),
    _lane_from_controller_profile(
        controller_profile_by_action("arac_counterfactual_action_racing_w3"),
        lane_id="oracle_candidate",
        dispatch_scope="offline_actionability_candidate_continuation",
        car_actionability_arm="candidate",
    ),
)
PRECISION_CAUSAL_LOGGING_LANES = (
    _lane_from_controller_profile(
        controller_profile_by_version(37),
        lane_id="precision_baseline",
        dispatch_scope="offline_precision_causal_baseline_continuation",
        precision_causal_arm="baseline",
    ),
    _lane_from_controller_profile(
        controller_profile_by_version(37),
        lane_id="precision_action",
        dispatch_scope="offline_precision_causal_action_continuation",
        precision_causal_arm="action",
    ),
)
CAR_W_ACTION_NAMES = frozenset(
    {
        "arac_counterfactual_action_racing_w",
        "arac_counterfactual_action_racing_w2",
        "arac_counterfactual_action_racing_w3",
    }
)
CANONICAL_EVIDENCE_CONTROLLER_V1_LANES = (
    LaneConfig(
        "canonical_evidence_controller_v1",
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v32",
        "arac_evidence_action_controller_v32",
        "single_run_canonical_runtime_evidence_controller",
        relation_dispatch_enabled=True,
        plan_action_name="arac_evidence_action_controller_v32",
        relation_policy_mode="controller_v31",
    ),
)


def lanes_for_profile(lane_profile: str) -> tuple[LaneConfig, ...]:
    controller_lanes = CONTROLLER_LANES_BY_PROFILE.get(lane_profile)
    if controller_lanes is not None:
        return controller_lanes
    if lane_profile == "paired_v33_v36_runtime_utility":
        return PAIRED_V33_V36_RUNTIME_UTILITY_LANES
    if lane_profile == "car_w_diagnostic":
        return CAR_W_DIAGNOSTIC_LANES
    if lane_profile == "car_w2_diagnostic":
        return CAR_W2_DIAGNOSTIC_LANES
    if lane_profile == "car_w3_diagnostic":
        return CAR_W3_DIAGNOSTIC_LANES
    if lane_profile == "car_actionability_audit":
        return CAR_ACTIONABILITY_AUDIT_LANES
    if lane_profile == "precision_causal_logging":
        return PRECISION_CAUSAL_LOGGING_LANES
    if lane_profile == "runtime_smoke":
        return LANES
    if lane_profile == "targeted_ablation":
        return TARGETED_ABLATION_LANES
    if lane_profile == "landscape_escape":
        return LANDSCAPE_ESCAPE_LANES
    if lane_profile == "repair_landscape_escape":
        return REPAIR_LANDSCAPE_ESCAPE_LANES
    if lane_profile == "repair_refine":
        return REPAIR_REFINE_LANES
    if lane_profile == "precision_refine_push":
        return PRECISION_REFINE_PUSH_LANES
    if lane_profile == "phase_rescue_push":
        return PHASE_RESCUE_PUSH_LANES
    if lane_profile == "repair_phase_rescue_push":
        return REPAIR_PHASE_RESCUE_PUSH_LANES
    if lane_profile == "cc_harm_sep_refresh":
        return CC_HARM_SEP_REFRESH_LANES
    if lane_profile == "separable_cmaes_push":
        return SEPARABLE_CMAES_PUSH_LANES
    if lane_profile == "focused_core":
        return FOCUSED_CORE_LANES
    if lane_profile == "focused_compare":
        return FOCUSED_COMPARE_LANES
    if lane_profile == "evidence_routed_only":
        return EVIDENCE_ROUTED_ONLY_LANES
    if lane_profile == "evidence_routed_v2_only":
        return EVIDENCE_ROUTED_V2_ONLY_LANES
    if lane_profile == "evidence_routed_v21_only":
        return EVIDENCE_ROUTED_V21_ONLY_LANES
    if lane_profile == "evidence_routed_v22_only":
        return EVIDENCE_ROUTED_V22_ONLY_LANES
    if lane_profile == "evidence_routed_v23_only":
        return EVIDENCE_ROUTED_V23_ONLY_LANES
    if lane_profile == "evidence_routed_v24_only":
        return EVIDENCE_ROUTED_V24_ONLY_LANES
    if lane_profile == "evidence_routed_v25_only":
        return EVIDENCE_ROUTED_V25_ONLY_LANES
    if lane_profile == "evidence_routed_v26_only":
        return EVIDENCE_ROUTED_V26_ONLY_LANES
    if lane_profile == "paper_best_win_push":
        return PAPER_BEST_WIN_PUSH_LANES
    if lane_profile == "paper_best_win_push_v2":
        return PAPER_BEST_WIN_PUSH_V2_LANES
    if lane_profile == "historical_anchor_refine_push":
        return HISTORICAL_ANCHOR_REFINE_PUSH_LANES
    if lane_profile == "historical_13_preserve_push":
        return HISTORICAL_13_PRESERVE_PUSH_LANES
    if lane_profile == "historical_13_fast_preserve":
        return HISTORICAL_13_FAST_PRESERVE_LANES
    if lane_profile == "historical_13_runtime_composite":
        return HISTORICAL_13_RUNTIME_COMPOSITE_LANES
    if lane_profile == "historical_13_runtime_composite_v2":
        return HISTORICAL_13_RUNTIME_COMPOSITE_V2_LANES
    if lane_profile == "evidence_action_controller_v1":
        return EVIDENCE_ACTION_CONTROLLER_V1_LANES
    if lane_profile == "evidence_action_controller_v2":
        return EVIDENCE_ACTION_CONTROLLER_V2_LANES
    if lane_profile == "evidence_action_controller_v3":
        return EVIDENCE_ACTION_CONTROLLER_V3_LANES
    if lane_profile == "evidence_action_controller_v31":
        return EVIDENCE_ACTION_CONTROLLER_V31_LANES
    if lane_profile == "evidence_action_controller_v32":
        return EVIDENCE_ACTION_CONTROLLER_V32_LANES
    if lane_profile == "canonical_evidence_controller_v1":
        return CANONICAL_EVIDENCE_CONTROLLER_V1_LANES
    raise ValueError(f"unsupported lane profile: {lane_profile}")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _write_claim_evidence_table(
    output_dir: Path,
    diagnosis_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# exp_003 Claim Evidence Table",
        "",
        "| Problem | Claim | Status | Evidence | Blocker | Source artifact |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in diagnosis_rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row.get(field, ""))
                for field in (
                    "problem_id",
                    "diagnostic_key",
                    "status",
                    "observed_value",
                    "blocker_reason",
                )
            )
            + " | policy_evidence_diagnosis.csv |"
        )
    (output_dir / "claim_evidence_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _decision(lane: LaneConfig) -> ActionDecision:
    action_name = lane.plan_action_name or lane.selected_action_name
    return ActionDecision(
        action_family=lane.action_family,
        action_name=action_name,
        decision="fallback" if lane.action_family == ActionFamily.FALLBACK else "allow",
        trigger_reason=f"exp_003_{lane.dispatch_scope}",
        utility_proxy=0.0 if lane.action_family == ActionFamily.FALLBACK else 1.0,
    )


def _require_hcc_action_preflight(
    lanes: tuple[LaneConfig, ...],
    problem_ids: tuple[str, ...],
) -> None:
    if not problem_ids:
        raise ValueError("at least one problem_id is required")
    failures = []
    for problem_id in problem_ids:
        for lane in lanes:
            plan = build_hcc_action_execution_plan(problem_id, _decision(lane))
            if plan.optimizer_consumed and plan.runtime_dispatch_allowed:
                continue
            failures.append(
                f"{problem_id}/{lane.lane_id}/{plan.selected_action_name}:"
                f"{plan.blocker_reason or plan.execution_mode}"
            )
    if failures:
        raise RuntimeError("HCC action preflight failed: " + ";".join(failures))


def _requires_pinned_environment(lanes: tuple[LaneConfig, ...]) -> bool:
    return any(
        controller_has_capability(lane.runner_action_name, "requires_pinned_environment")
        for lane in lanes
    )


def _effective_claim_gate_decision(
    lane: LaneConfig,
    decision: ActionDecision,
    trace_rows: list[dict[str, str]],
) -> ActionDecision:
    if not lane.relation_dispatch_enabled:
        return decision
    has_active_consumed_action = any(
        row.get("optimizer_consumed") == "1"
        and _trace_action(row) != "conservative_no_action"
        for row in trace_rows
    )
    if has_active_consumed_action:
        return decision
    return ActionDecision(
        action_family=ActionFamily.FALLBACK,
        action_name="conservative_no_action",
        decision="fallback",
        trigger_reason="relation_dispatch_no_active_optimizer_consumed_action",
        utility_proxy=0.0,
    )


def _same_budget_group_id(problem_id: str, seed: int, max_fes: int) -> str:
    return f"{problem_id}_seed{seed}_{max_fes}fe"


def _is_overlap_applicable_problem_id(problem_id: str) -> bool:
    level = "".join(character for character in problem_id if character.isdigit())
    return level != "1"


def _runtime_payload(
    problem_id: str,
    seed: int,
    lane_id: str,
    action_name: str,
    max_fes: int,
) -> dict[str, object]:
    payload = {
        "run_id": RUN_ID,
        "problem_id": problem_id,
        "seed": seed,
        "lane_id": lane_id,
        "selected_action_name": action_name,
        "benchmark": "AOB",
        "budget_limit": max_fes,
        "used_for_runtime": 1,
    }
    validate_runtime_payload(payload)
    return payload


def _ledger_for_result(result: HccAobExecutionResult) -> SameBudgetLedger:
    actual_fe_used = _actual_fe_used(result)
    phase_i_fe = min(actual_fe_used, max(0, result.global_phase_fe or 0))
    return SameBudgetLedger(
        phase_i_fe=phase_i_fe,
        phase_ii_fe=actual_fe_used - phase_i_fe,
        budget_limit=result.max_fes,
        fresh_execution=result.fresh_optimizer_execution,
        search_state_fe=max(0, result.search_state_fe or 0),
    )


def _actual_fe_used(result: HccAobExecutionResult) -> int:
    if result.optimizer_final_fe_used is None:
        return result.fe_used
    return result.optimizer_final_fe_used


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fingerprint_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _car_actionability_execution_dependencies(
    vendor_paths: HccVendorPaths,
) -> dict[str, str]:
    relative_paths = (
        "actions/contracts.py",
        "actions/controller_profiles.py",
        "backends/diagonal_cma.py",
        "backends/hcc.py",
        "backends/hcc_budget.py",
        "backends/hcc_car.py",
        "backends/hcc_plan.py",
        "backends/hcc_shared_writeback.py",
        "backends/hcc_trace.py",
        "evidence/__init__.py",
        "evidence/overlap_relation_builder.py",
        "policy/action_trust_policy.py",
        "policy/causal_risk_scheduler.py",
        "policy/component_delayed_credit.py",
        "policy/counterfactual_action_racing.py",
        "policy/oracle_actionability.py",
        "policy/relation_policy.py",
        "policy/search_state_policy.py",
        "policy/trajectory_guard.py",
    )
    paths = {
        "experiment_runner": Path(__file__).resolve(),
        "hcc_smoke_runner": vendor_paths.runner,
        **{
            f"src/arac/{relative}": ARAC_SRC_ROOT / "arac" / relative
            for relative in relative_paths
        },
        **{
            f"vendor/hcc/{path.relative_to(vendor_paths.vendor_root).as_posix()}": path
            for path in sorted(vendor_paths.vendor_root.rglob("*.py"))
        },
    }
    hashes = {name: _sha256_file(path) for name, path in sorted(paths.items())}
    missing = [name for name, digest in hashes.items() if digest == "missing"]
    if missing:
        raise RuntimeError(
            "missing CAR actionability execution dependencies: "
            + ",".join(missing)
        )
    return hashes


def _car_actionability_aob_inputs(
    request: HccAobExecutionRequest,
) -> dict[str, str]:
    suffix = str(request.problem_id)[1:]
    if not suffix.isdigit():
        raise ValueError("CAR actionability requires a canonical AOB problem id")
    paths = required_aob_data_files(request.aob_data_root, int(suffix))
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing CAR actionability AOB inputs: " + ",".join(sorted(missing))
        )
    return {
        path.name: _sha256_file(path)
        for path in sorted(paths, key=lambda item: item.name)
    }


def _car_actionability_request_payload(
    request: HccAobExecutionRequest,
) -> dict[str, object]:
    vendor_paths = resolve_hcc_vendor_paths(
        request.hcc_root,
        repo_root=request.hcc_repo_root,
        runner_path=request.hcc_runner,
    )
    execution_dependencies = _car_actionability_execution_dependencies(vendor_paths)
    aob_inputs = _car_actionability_aob_inputs(request)
    dependency_versions = {
        name: _dependency_version(name)
        for name in ("cma", "numpy", "scipy", "torch")
    }
    execution_context = {
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "thread_environment": _thread_environment(),
        "dependency_versions": dependency_versions,
        "execution_dependency_fingerprint": _fingerprint_payload(
            execution_dependencies
        ),
    }
    payload = {
        "protocol_version": CAR_ACTIONABILITY_PROTOCOL_VERSION,
        **execution_context,
        "execution_context_fingerprint": _fingerprint_payload(execution_context),
        "execution_dependency_sha256": execution_dependencies,
        "problem_id": request.problem_id,
        "seed": int(request.seed),
        "max_fes": int(request.max_fes),
        "timestamp": request.timestamp,
        "config_name": request.config_name,
        "python_executable": str(request.python_executable),
        "skip_plots": bool(request.skip_plots),
        "arac_action": request.arac_action,
        "enable_relation_dispatch": bool(request.enable_relation_dispatch),
        "relation_policy_mode": request.relation_policy_mode,
        "budget_accounting": request.budget_accounting,
        "cmaes_restart": bool(request.cmaes_restart),
        "mmes_restart": bool(request.mmes_restart),
        "search_state_backend": request.search_state_backend,
        "car_candidate_mode": request.car_candidate_mode,
        "car_actionability_arm": request.car_actionability_arm,
        "aob_data_root": str(Path(request.aob_data_root).resolve()),
        "aob_input_sha256": aob_inputs,
        "aob_input_fingerprint": _fingerprint_payload(aob_inputs),
        "hcc_vendor_root": str(vendor_paths.vendor_root),
        "hcc_runner": str(vendor_paths.runner),
        "hcc_runner_sha256": _sha256_file(vendor_paths.runner),
        "oracle_module_sha256": _sha256_file(
            ARAC_REPO_ROOT / "src" / "arac" / "policy" / "oracle_actionability.py"
        ),
    }
    return {
        "request": payload,
        "request_fingerprint": _fingerprint_payload(payload),
    }


def _car_actionability_provenance_path(request: HccAobExecutionRequest) -> Path:
    return Path(request.output_dir) / "car_actionability_provenance.json"


def _latest_car_artifact(output_root: Path, pattern: str) -> Path | None:
    paths = sorted(output_root.rglob(pattern))
    return paths[-1] if paths else None


def _provenance_artifact_path(
    value: object,
    output_root: Path,
) -> Path | None:
    if not value:
        return None
    try:
        path = Path(str(value)).resolve()
        path.relative_to(output_root.resolve())
    except (TypeError, ValueError, OSError):
        return None
    return path


def _prepare_car_actionability_provenance(
    request: HccAobExecutionRequest,
) -> dict[str, object]:
    path = _car_actionability_provenance_path(request)
    expected = _car_actionability_request_payload(request)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid CAR actionability provenance file") from exc
        if existing.get("request_fingerprint") != expected["request_fingerprint"]:
            raise RuntimeError("CAR actionability output belongs to a different request")
        if existing.get("status") == "complete":
            raise RuntimeError(
                "CAR actionability output is already complete; use a new output directory"
            )
    else:
        output_root = Path(request.output_dir)
        stale_patterns = (
            "action_trace.csv",
            "evaluation_record.txt",
            "*car_actionability_trace.csv",
        )
        if any(any(output_root.rglob(pattern)) for pattern in stale_patterns):
            raise RuntimeError(
                "CAR actionability output directory contains unprovenanced artifacts"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    scheduled = {
        **expected,
        "status": "scheduled",
        "output_dir": str(Path(request.output_dir).resolve()),
    }
    path.write_text(json.dumps(scheduled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return scheduled


def _complete_car_actionability_provenance(
    request: HccAobExecutionRequest,
    result: HccAobExecutionResult,
) -> None:
    if request.car_actionability_arm == "off":
        return
    if not result.fresh_optimizer_execution:
        raise RuntimeError("CAR actionability execution was not fresh")
    path = _car_actionability_provenance_path(request)
    expected = _car_actionability_request_payload(request)
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("missing CAR actionability provenance schedule") from exc
    if (
        current.get("request_fingerprint") != expected["request_fingerprint"]
        or current.get("request") != expected["request"]
    ):
        raise RuntimeError("CAR actionability provenance request mismatch")
    output_root = Path(result.output_root).resolve()
    trace_path = _latest_car_artifact(output_root, "*car_actionability_trace.csv")
    evaluation_path = _latest_car_artifact(output_root, "evaluation_record.txt")
    budget_path = _latest_car_artifact(output_root, "*budget_summary.csv")
    aob_manifest_path = _latest_car_artifact(
        output_root, "*aob_input_manifest.csv"
    )
    if (
        trace_path is None
        or evaluation_path is None
        or budget_path is None
        or aob_manifest_path is None
    ):
        raise RuntimeError("CAR actionability execution artifacts are incomplete")
    if result.action_trace_path is None or not result.action_trace_path.exists():
        raise RuntimeError("CAR actionability execution produced no action trace")
    action_trace_path = result.action_trace_path.resolve()
    try:
        action_trace_path.relative_to(output_root)
    except ValueError as exc:
        raise RuntimeError("CAR actionability action trace escaped output root") from exc
    completed = {
        **current,
        "status": "complete",
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": _sha256_file(trace_path),
        "evaluation_record_path": str(evaluation_path.resolve()),
        "evaluation_record_sha256": _sha256_file(evaluation_path),
        "budget_summary_path": str(budget_path.resolve()),
        "budget_summary_sha256": _sha256_file(budget_path),
        "aob_input_manifest_path": str(aob_manifest_path.resolve()),
        "aob_input_manifest_sha256": _sha256_file(aob_manifest_path),
        "action_trace_path": str(action_trace_path),
        "action_trace_sha256": _sha256_file(action_trace_path),
        "fresh_optimizer_execution": bool(result.fresh_optimizer_execution),
    }
    path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _precision_causal_request_payload(
    request: HccAobExecutionRequest,
) -> dict[str, object]:
    vendor_paths = resolve_hcc_vendor_paths(
        request.hcc_root,
        repo_root=request.hcc_repo_root,
        runner_path=request.hcc_runner,
    )
    execution_dependencies = _car_actionability_execution_dependencies(vendor_paths)
    aob_inputs = _car_actionability_aob_inputs(request)
    execution_context = {
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "thread_environment": _thread_environment(),
        "dependency_versions": {
            name: _dependency_version(name)
            for name in ("cma", "numpy", "scipy", "torch")
        },
        "execution_dependency_fingerprint": _fingerprint_payload(
            execution_dependencies
        ),
    }
    payload = {
        "protocol_version": PRECISION_CAUSAL_PROTOCOL_VERSION,
        **execution_context,
        "execution_context_fingerprint": _fingerprint_payload(execution_context),
        "execution_dependency_sha256": execution_dependencies,
        "problem_id": request.problem_id,
        "seed": int(request.seed),
        "max_fes": int(request.max_fes),
        "timestamp": request.timestamp,
        "config_name": request.config_name,
        "python_executable": str(request.python_executable),
        "skip_plots": bool(request.skip_plots),
        "arac_action": request.arac_action,
        "enable_relation_dispatch": bool(request.enable_relation_dispatch),
        "relation_policy_mode": request.relation_policy_mode,
        "budget_accounting": request.budget_accounting,
        "cmaes_restart": bool(request.cmaes_restart),
        "mmes_restart": bool(request.mmes_restart),
        "search_state_backend": request.search_state_backend,
        "precision_causal_arm": request.precision_causal_arm,
        "pair_id": precision_causal_pair_id(request.problem_id, request.seed),
        "logged_arm": precision_causal_logged_arm(
            request.problem_id, request.seed
        ),
        "randomization_salt": PRECISION_CAUSAL_RANDOMIZATION_SALT,
        "randomization_algorithm": "sha256_first_u64_mod2",
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "preregistration": {
            "path": PRECISION_CAUSAL_PREREGISTRATION_PATH,
            "sha256": PRECISION_CAUSAL_PREREGISTRATION_SHA256,
            "commit": PRECISION_CAUSAL_PREREGISTRATION_COMMIT,
        },
        "aob_data_root": str(Path(request.aob_data_root).resolve()),
        "aob_input_sha256": aob_inputs,
        "aob_input_fingerprint": _fingerprint_payload(aob_inputs),
        "hcc_vendor_root": str(vendor_paths.vendor_root),
        "hcc_runner": str(vendor_paths.runner),
        "hcc_runner_sha256": _sha256_file(vendor_paths.runner),
    }
    return {
        "request": payload,
        "request_fingerprint": _fingerprint_payload(payload),
    }


def _precision_causal_provenance_path(request: HccAobExecutionRequest) -> Path:
    return Path(request.output_dir) / "precision_causal_provenance.json"


def _prepare_precision_causal_provenance(
    request: HccAobExecutionRequest,
) -> None:
    path = _precision_causal_provenance_path(request)
    expected = _precision_causal_request_payload(request)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid precision causal provenance file") from exc
        if existing.get("request_fingerprint") != expected["request_fingerprint"]:
            raise RuntimeError("precision causal output belongs to a different request")
        if existing.get("status") == "complete":
            raise RuntimeError(
                "precision causal output is already complete; use a new output directory"
            )
    else:
        output_root = Path(request.output_dir)
        stale_patterns = (
            "action_trace.csv",
            "evaluation_record.txt",
            "*precision_causal_trace.csv",
        )
        if any(any(output_root.rglob(pattern)) for pattern in stale_patterns):
            raise RuntimeError(
                "precision causal output directory contains unprovenanced artifacts"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                **expected,
                "status": "scheduled",
                "output_dir": str(Path(request.output_dir).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _complete_precision_causal_provenance(
    request: HccAobExecutionRequest,
    result: HccAobExecutionResult,
) -> None:
    if request.precision_causal_arm == "off":
        return
    if not result.fresh_optimizer_execution:
        raise RuntimeError("precision causal execution was not fresh")
    path = _precision_causal_provenance_path(request)
    expected = _precision_causal_request_payload(request)
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("missing precision causal provenance schedule") from exc
    if (
        current.get("request_fingerprint") != expected["request_fingerprint"]
        or current.get("request") != expected["request"]
    ):
        raise RuntimeError("precision causal provenance request mismatch")
    output_root = Path(result.output_root).resolve()
    artifacts = {
        "trace": _latest_car_artifact(output_root, "*precision_causal_trace.csv"),
        "evaluation_record": _latest_car_artifact(output_root, "evaluation_record.txt"),
        "budget_summary": _latest_car_artifact(output_root, "*budget_summary.csv"),
        "aob_input_manifest": _latest_car_artifact(
            output_root, "*aob_input_manifest.csv"
        ),
        "action_trace": result.action_trace_path,
    }
    if any(path_value is None for path_value in artifacts.values()):
        raise RuntimeError("precision causal execution artifacts are incomplete")
    completed_artifacts: dict[str, dict[str, str]] = {}
    for name, artifact in artifacts.items():
        assert artifact is not None
        resolved = Path(artifact).resolve()
        try:
            resolved.relative_to(output_root)
        except ValueError as exc:
            raise RuntimeError(
                f"precision causal {name} artifact escaped output root"
            ) from exc
        completed_artifacts[name] = {
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
        }
    path.write_text(
        json.dumps(
            {
                **current,
                "status": "complete",
                "fresh_optimizer_execution": True,
                "artifacts": completed_artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _precision_causal_provenance_is_complete(
    request: HccAobExecutionRequest,
    *,
    action_trace_path: Path,
) -> bool:
    provenance_path = _precision_causal_provenance_path(request)
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = _precision_causal_request_payload(request)
    if (
        provenance.get("request_fingerprint") != expected["request_fingerprint"]
        or provenance.get("request") != expected["request"]
        or provenance.get("status") != "complete"
        or provenance.get("fresh_optimizer_execution") is not True
    ):
        return False
    output_root = Path(request.output_dir).resolve()
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    expected_latest = {
        "trace": _latest_car_artifact(output_root, "*precision_causal_trace.csv"),
        "evaluation_record": _latest_car_artifact(output_root, "evaluation_record.txt"),
        "budget_summary": _latest_car_artifact(output_root, "*budget_summary.csv"),
        "aob_input_manifest": _latest_car_artifact(
            output_root, "*aob_input_manifest.csv"
        ),
        "action_trace": action_trace_path.resolve(),
    }
    for name, expected_path in expected_latest.items():
        item = artifacts.get(name)
        if expected_path is None or not isinstance(item, dict):
            return False
        try:
            artifact_path = Path(str(item.get("path", ""))).resolve()
            artifact_path.relative_to(output_root)
        except (OSError, ValueError):
            return False
        if (
            artifact_path != Path(expected_path).resolve()
            or not artifact_path.exists()
            or item.get("sha256") != _sha256_file(artifact_path)
        ):
            return False
    trace_path = Path(str(artifacts["trace"]["path"]))
    rows = _read_csv_rows(trace_path)
    expected_row = {
        "protocol_version": PRECISION_CAUSAL_PROTOCOL_VERSION,
        "fresh_optimizer_execution": "1",
        "problem_id": request.problem_id,
        "seed": str(request.seed),
        "audit_arm": request.precision_causal_arm,
        "configured_max_fes": str(request.max_fes),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    }
    return len(rows) == 1 and all(
        rows[0].get(field) == value for field, value in expected_row.items()
    )


def _existing_completed_result(request: HccAobExecutionRequest) -> HccAobExecutionResult | None:
    action_trace_path, action_trace_rows = _find_hcc_action_trace(Path(request.output_dir))
    if action_trace_path is None:
        return None
    verified_fresh_audit = False
    if request.car_actionability_arm != "off":
        provenance_path = _car_actionability_provenance_path(request)
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected = _car_actionability_request_payload(request)
        if (
            provenance.get("request_fingerprint") != expected["request_fingerprint"]
            or provenance.get("request") != expected["request"]
            or provenance.get("status") != "complete"
            or provenance.get("fresh_optimizer_execution") is not True
        ):
            return None
        output_root = Path(request.output_dir).resolve()
        trace_path = _provenance_artifact_path(provenance.get("trace_path"), output_root)
        evaluation_path = _provenance_artifact_path(
            provenance.get("evaluation_record_path"), output_root
        )
        budget_path = _provenance_artifact_path(
            provenance.get("budget_summary_path"), output_root
        )
        aob_manifest_path = _provenance_artifact_path(
            provenance.get("aob_input_manifest_path"), output_root
        )
        if (
            trace_path is None
            or evaluation_path is None
            or budget_path is None
            or aob_manifest_path is None
        ):
            return None
        latest_evaluation = _latest_car_artifact(output_root, "evaluation_record.txt")
        latest_budget = _latest_car_artifact(output_root, "*budget_summary.csv")
        latest_trace = _latest_car_artifact(
            output_root, "*car_actionability_trace.csv"
        )
        latest_aob_manifest = _latest_car_artifact(
            output_root, "*aob_input_manifest.csv"
        )
        if (
            not trace_path.exists()
            or provenance.get("trace_sha256") != _sha256_file(trace_path)
            or trace_path != latest_trace
            or evaluation_path != latest_evaluation
            or budget_path != latest_budget
            or aob_manifest_path != latest_aob_manifest
            or not evaluation_path.exists()
            or not budget_path.exists()
            or not aob_manifest_path.exists()
            or provenance.get("evaluation_record_sha256")
            != _sha256_file(evaluation_path)
            or provenance.get("budget_summary_sha256")
            != _sha256_file(budget_path)
            or provenance.get("aob_input_manifest_sha256")
            != _sha256_file(aob_manifest_path)
            or provenance.get("action_trace_path") != str(action_trace_path.resolve())
            or provenance.get("action_trace_sha256") != _sha256_file(action_trace_path)
        ):
            return None
        audit_rows = _read_csv_rows(trace_path)
        if not audit_rows:
            return None
        expected = {
            "protocol_version": CAR_ACTIONABILITY_PROTOCOL_VERSION,
            "fresh_optimizer_execution": "1",
            "problem_id": request.problem_id,
            "seed": str(request.seed),
            "audit_arm": request.car_actionability_arm,
            "candidate_mode": request.car_candidate_mode,
            "configured_max_fes": str(request.max_fes),
        }
        if any(
            any(row.get(field) != value for field, value in expected.items())
            for row in audit_rows
        ):
            return None
        if _car_actionability_lane_semantic_failures(
            audit_rows,
            prefix=(
                f"{request.problem_id}/seed{request.seed}/"
                f"{request.car_actionability_arm}"
            ),
        ):
            return None
        verified_fresh_audit = True
    if request.precision_causal_arm != "off":
        if not _precision_causal_provenance_is_complete(
            request,
            action_trace_path=action_trace_path,
        ):
            return None
        verified_fresh_audit = True
    try:
        final_error, fe_used, optimizer_final_fe_used = (
            _parse_hcc_evaluation_record_with_optimizer_final_fe(
                Path(request.output_dir),
                budget_limit=request.max_fes,
            )
        )
    except (FileNotFoundError, ValueError):
        return None
    budget_breakdown = _parse_hcc_budget_summary(Path(request.output_dir))
    return HccAobExecutionResult(
        problem_id=request.problem_id,
        seed=request.seed,
        max_fes=request.max_fes,
        final_error=final_error,
        fe_used=fe_used,
        time_seconds=0.0,
        output_root=Path(request.output_dir),
        fresh_optimizer_execution=verified_fresh_audit,
        status="completed_existing_artifact",
        result_source=(
            "verified_fresh_precision_causal_artifact"
            if request.precision_causal_arm != "off" and verified_fresh_audit
            else "verified_fresh_car_actionability_artifact"
            if request.car_actionability_arm != "off" and verified_fresh_audit
            else "hcc_subprocess_smoke_execution_existing_artifact"
        ),
        action_trace_path=action_trace_path,
        action_trace_rows=action_trace_rows,
        optimizer_final_fe_used=optimizer_final_fe_used,
        global_phase_fe=budget_breakdown.get("global_phase_fe"),
        cc_phase_fe=budget_breakdown.get("cc_phase_fe"),
        rescue_fe=budget_breakdown.get("rescue_fe"),
        refresh_fe=budget_breakdown.get("refresh_fe"),
        search_state_fe=budget_breakdown.get("search_state_fe", 0),
        separable_continuation_fe=budget_breakdown.get(
            "separable_continuation_fe"
        ),
        overhead_fe=budget_breakdown.get("overhead_fe"),
    )


def _find_lane_artifact(result: HccAobExecutionResult, artifact_name: str) -> Path | None:
    root = Path(result.output_root)
    preferred = sorted(root.rglob(f"{result.problem_id}_{artifact_name}"))
    if preferred:
        return preferred[-1]
    generic = sorted(root.rglob(artifact_name))
    return generic[-1] if generic else None


def precision_causal_pair_id(problem_id: str, seed: int) -> str:
    material = (
        f"{PRECISION_CAUSAL_PROTOCOL_VERSION}|{str(problem_id).upper()}|{int(seed)}"
    )
    return "pair_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def precision_causal_logged_arm(problem_id: str, seed: int) -> str:
    material = (
        f"{PRECISION_CAUSAL_RANDOMIZATION_SALT}|"
        f"{str(problem_id).upper()}|{int(seed)}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return PRECISION_CAUSAL_ARMS[int.from_bytes(digest[:8], "big") % 2]


def _precision_trace_row(record: dict[str, object]) -> dict[str, str] | None:
    result = record["result"]
    lane = record["lane"]
    assert isinstance(result, HccAobExecutionResult)
    assert isinstance(lane, LaneConfig)
    if lane.precision_causal_arm == "off":
        return None
    rows = _read_csv_rows(
        _find_lane_artifact(result, "precision_causal_trace.csv")
    )
    if len(rows) != 1:
        return None
    return rows[0]


def _float_or_nan(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float("nan")


def _precision_causal_raw_rows(
    records: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[str],
]:
    grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for record in records:
        lane = record["lane"]
        result = record["result"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(result, HccAobExecutionResult)
        if lane.precision_causal_arm == "off":
            continue
        grouped.setdefault((result.problem_id, result.seed), {})[
            lane.precision_causal_arm
        ] = record

    feature_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    randomized_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for (problem_id, seed), arm_records in sorted(grouped.items()):
        pair_id = precision_causal_pair_id(problem_id, seed)
        logged_arm = precision_causal_logged_arm(problem_id, seed)
        arm_rows: dict[str, dict[str, str]] = {}
        for arm in PRECISION_CAUSAL_ARMS:
            record = arm_records.get(arm)
            if record is None:
                failures.append(f"{pair_id}:missing_{arm}_lane")
                continue
            trace = _precision_trace_row(record)
            if trace is None:
                failures.append(f"{pair_id}:missing_{arm}_trace")
                continue
            arm_rows[arm] = trace
            result = record["result"]
            lane = record["lane"]
            assert isinstance(result, HccAobExecutionResult)
            assert isinstance(lane, LaneConfig)
            branch_rows.append(
                {
                    "pair_id": pair_id,
                    "decision_id": trace.get("decision_id", ""),
                    "problem_id": problem_id,
                    "seed": seed,
                    "arm": arm,
                    "lane_id": lane.lane_id,
                    "fresh_optimizer_execution": int(
                        result.fresh_optimizer_execution
                    ),
                    "status": result.status,
                    "result_source": result.result_source,
                    "output_root": str(Path(result.output_root).resolve()),
                    "decision_status": trace.get("decision_status", ""),
                    "not_applicable_reason": trace.get(
                        "not_applicable_reason", ""
                    ),
                    "action_applied": trace.get("action_applied", ""),
                    "decision_fe": trace.get("decision_fe", ""),
                    "intervention_end_fe": trace.get(
                        "intervention_end_fe", ""
                    ),
                    "checkpoint_fitness": trace.get(
                        "checkpoint_fitness", ""
                    ),
                    "normal_sigma": trace.get("normal_sigma", ""),
                    "candidate_sigma": trace.get("candidate_sigma", ""),
                    "applied_sigma": trace.get("applied_sigma", ""),
                    "requested_fe": trace.get("requested_fe", ""),
                    "actual_fe": trace.get("actual_fe", ""),
                    "configured_max_fes": trace.get(
                        "configured_max_fes", ""
                    ),
                    "terminal_target_fe": trace.get("terminal_target_fe", ""),
                    "terminal_observed_fe": trace.get(
                        "terminal_observed_fe", ""
                    ),
                    "terminal_status": trace.get("terminal_status", ""),
                    "prefix_record_sha256": trace.get(
                        "prefix_record_sha256", ""
                    ),
                    "checkpoint_candidate_sha256": trace.get(
                        "checkpoint_candidate_sha256", ""
                    ),
                    "feature_sha256": trace.get("feature_sha256", ""),
                    "controller_state_sha256": trace.get(
                        "controller_state_sha256", ""
                    ),
                    "random_descriptor_sha256": trace.get(
                        "random_descriptor_sha256", ""
                    ),
                    "terminal_error": trace.get("terminal_error", ""),
                    "terminal_record_sha256": trace.get(
                        "terminal_record_sha256", ""
                    ),
                    "optimizer_fe_used": _actual_fe_used(result),
                    "same_budget_violation": int(
                        _actual_fe_used(result) > result.max_fes
                    ),
                }
            )

        if set(arm_rows) != set(PRECISION_CAUSAL_ARMS):
            continue
        baseline = arm_rows["baseline"]
        action = arm_rows["action"]
        status_match = baseline.get("decision_status") == action.get(
            "decision_status"
        )
        decision_id_match = baseline.get("decision_id") == action.get(
            "decision_id"
        )
        feature_match = baseline.get("feature_sha256") == action.get(
            "feature_sha256"
        )
        prefix_match = baseline.get("prefix_record_sha256") == action.get(
            "prefix_record_sha256"
        )
        controller_match = baseline.get("controller_state_sha256") == action.get(
            "controller_state_sha256"
        )
        checkpoint_candidate_match = baseline.get(
            "checkpoint_candidate_sha256"
        ) == action.get("checkpoint_candidate_sha256")
        random_descriptor_match = baseline.get(
            "random_descriptor_sha256"
        ) == action.get("random_descriptor_sha256")
        intervention_end_match = baseline.get("intervention_end_fe") == action.get(
            "intervention_end_fe"
        )
        reason_match = baseline.get("not_applicable_reason") == action.get(
            "not_applicable_reason"
        )
        pair_integrity = all(
            (
                status_match,
                decision_id_match,
                feature_match,
                prefix_match,
                controller_match,
                checkpoint_candidate_match,
                random_descriptor_match,
                intervention_end_match,
                reason_match,
            )
        )
        if not pair_integrity:
            failures.append(f"{pair_id}:preaction_pair_mismatch")

        decision_status = baseline.get("decision_status", "")
        decision_id = baseline.get("decision_id", "")
        if decision_status == "applicable" and pair_integrity:
            feature_rows.append(
                {
                    "decision_id": decision_id,
                    **{name: baseline.get(name, "") for name in UTILITY_FEATURE_NAMES},
                }
            )
        audit_rows.append(
            {
                "protocol_version": PRECISION_CAUSAL_PROTOCOL_VERSION,
                "pair_id": pair_id,
                "decision_id": decision_id,
                "problem_id": problem_id,
                "seed": seed,
                "decision_status": decision_status,
                "not_applicable_reason": baseline.get(
                    "not_applicable_reason", ""
                ),
                "logged_arm": logged_arm,
                "propensity": "0.5",
                "decision_fe": baseline.get("decision_fe", ""),
                "checkpoint_fitness": baseline.get("checkpoint_fitness", ""),
                "remaining_fe": baseline.get("remaining_fe", ""),
                "component_id": baseline.get("component_id", ""),
                "component_group_count": baseline.get(
                    "component_group_count", ""
                ),
                "component_shared_var_count": baseline.get(
                    "component_shared_var_count", ""
                ),
                "component_unlocked": baseline.get("component_unlocked", ""),
                "scheduler_revisit_reachable": baseline.get(
                    "scheduler_revisit_reachable", ""
                ),
                "scheduler_revisit_cap_fe": baseline.get(
                    "scheduler_revisit_cap_fe", ""
                ),
                "scheduler_revisit_reason": baseline.get(
                    "scheduler_revisit_reason", ""
                ),
                "source_phase_i_end_fe": baseline.get(
                    "source_phase_i_end_fe", ""
                ),
                "source_cc_history_end_fe": baseline.get(
                    "source_cc_history_end_fe", ""
                ),
                "source_disagreement_history_end_fe": baseline.get(
                    "source_disagreement_history_end_fe", ""
                ),
                "source_cma_history_end_fe": baseline.get(
                    "source_cma_history_end_fe", ""
                ),
                "source_end_fe": baseline.get("source_end_fe", ""),
                "prefix_record_sha256": baseline.get(
                    "prefix_record_sha256", ""
                ),
                "checkpoint_candidate_sha256": baseline.get(
                    "checkpoint_candidate_sha256", ""
                ),
                "controller_state_sha256": baseline.get(
                    "controller_state_sha256", ""
                ),
                "random_descriptor_sha256": baseline.get(
                    "random_descriptor_sha256", ""
                ),
                "feature_schema_sha256": baseline.get(
                    "feature_schema_sha256", ""
                ),
                "feature_sha256": baseline.get("feature_sha256", ""),
                "decision_status_match": int(status_match),
                "decision_id_match": int(decision_id_match),
                "feature_match": int(feature_match),
                "prefix_match": int(prefix_match),
                "controller_state_match": int(controller_match),
                "checkpoint_candidate_match": int(
                    checkpoint_candidate_match
                ),
                "random_descriptor_match": int(random_descriptor_match),
                "intervention_end_fe_match": int(intervention_end_match),
                "not_applicable_reason_match": int(reason_match),
                "pair_integrity": int(pair_integrity),
            }
        )

        baseline_error = _float_or_nan(baseline.get("terminal_error"))
        action_error = _float_or_nan(action.get("terminal_error"))
        baseline_result = arm_records["baseline"]["result"]
        action_result = arm_records["action"]["result"]
        assert isinstance(baseline_result, HccAobExecutionResult)
        assert isinstance(action_result, HccAobExecutionResult)
        equal_optimizer_fe = _actual_fe_used(baseline_result) == _actual_fe_used(
            action_result
        )
        checkpoint_baseline = _float_or_nan(baseline.get("checkpoint_fitness"))
        checkpoint_action = _float_or_nan(action.get("checkpoint_fitness"))
        equal_checkpoint = (
            math.isfinite(checkpoint_baseline)
            and checkpoint_baseline == checkpoint_action
        )
        equal_target = baseline.get("terminal_target_fe") == action.get(
            "terminal_target_fe"
        )
        equal_observed = baseline.get("terminal_observed_fe") == action.get(
            "terminal_observed_fe"
        )
        outcome_valid = bool(
            pair_integrity
            and decision_status == "applicable"
            and baseline.get("action_applied") == "0"
            and action.get("action_applied") == "1"
            and baseline.get("terminal_status") == "complete"
            and action.get("terminal_status") == "complete"
            and equal_checkpoint
            and equal_target
            and equal_observed
            and equal_optimizer_fe
            and math.isfinite(baseline_error)
            and math.isfinite(action_error)
        )
        if decision_status == "applicable" and not outcome_valid:
            failures.append(f"{pair_id}:invalid_paired_outcome")
        if decision_status != "applicable":
            if (
                baseline.get("terminal_record_sha256")
                != action.get("terminal_record_sha256")
                or not equal_optimizer_fe
            ):
                failures.append(f"{pair_id}:not_applicable_v37_parity_mismatch")
        floor = 1e-300
        baseline_y = (
            math.log(max(checkpoint_baseline, floor))
            - math.log(max(baseline_error, floor))
            if outcome_valid
            else float("nan")
        )
        action_y = (
            math.log(max(checkpoint_action, floor))
            - math.log(max(action_error, floor))
            if outcome_valid
            else float("nan")
        )
        paired_tau = action_y - baseline_y if outcome_valid else float("nan")
        catastrophic = (
            int(action_error >= 1.2 * baseline_error) if outcome_valid else ""
        )
        outcome_rows.append(
            {
                "pair_id": pair_id,
                "decision_id": decision_id,
                "problem_id": problem_id,
                "seed": seed,
                "decision_status": decision_status,
                "checkpoint_error": (
                    f"{checkpoint_baseline:.17e}" if equal_checkpoint else ""
                ),
                "baseline_terminal_error": (
                    f"{baseline_error:.17e}" if math.isfinite(baseline_error) else ""
                ),
                "action_terminal_error": (
                    f"{action_error:.17e}" if math.isfinite(action_error) else ""
                ),
                "baseline_log_progress": (
                    f"{baseline_y:.17e}" if outcome_valid else ""
                ),
                "action_log_progress": (
                    f"{action_y:.17e}" if outcome_valid else ""
                ),
                "paired_tau": f"{paired_tau:.17e}" if outcome_valid else "",
                "catastrophic": catastrophic,
                "equal_checkpoint": int(equal_checkpoint),
                "equal_terminal_target_fe": int(equal_target),
                "equal_terminal_observed_fe": int(equal_observed),
                "outcome_valid": int(outcome_valid),
            }
        )
        observed_row = baseline if logged_arm == "baseline" else action
        observed_error = baseline_error if logged_arm == "baseline" else action_error
        observed_y = baseline_y if logged_arm == "baseline" else action_y
        randomized_rows.append(
            {
                "pair_id": pair_id,
                "decision_id": decision_id,
                "problem_id": problem_id,
                "seed": seed,
                "logged_arm": logged_arm,
                "observed_treatment": int(logged_arm == "action"),
                "propensity": "0.5",
                "observed_terminal_error": (
                    f"{observed_error:.17e}" if outcome_valid else ""
                ),
                "observed_log_progress": (
                    f"{observed_y:.17e}" if outcome_valid else ""
                ),
                "terminal_target_fe": observed_row.get(
                    "terminal_target_fe", ""
                ),
                "terminal_observed_fe": observed_row.get(
                    "terminal_observed_fe", ""
                ),
                "outcome_valid": int(outcome_valid),
            }
        )
    return (
        feature_rows,
        audit_rows,
        branch_rows,
        outcome_rows,
        randomized_rows,
        failures,
    )


def _trace_rows_for_record(record: dict[str, object]) -> list[dict[str, str]]:
    result = record["result"]
    assert isinstance(result, HccAobExecutionResult)
    return _read_csv_rows(result.action_trace_path)


def _artifact_rows_for_record(
    record: dict[str, object],
    artifact_name: str,
) -> list[dict[str, str]]:
    result = record["result"]
    assert isinstance(result, HccAobExecutionResult)
    return _read_csv_rows(_find_lane_artifact(result, artifact_name))


def _with_lane_prefix(
    record: dict[str, object],
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    result = record["result"]
    assert isinstance(result, HccAobExecutionResult)
    return [
        {**row, "run_id": RUN_ID, "lane_id": record["lane_id"], "seed": result.seed}
        for row in rows
    ]


def _format_action_mix(
    rows: list[dict[str, str]],
    fallback_action: str,
    *,
    optimizer_consumed_only: bool = False,
) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        if optimizer_consumed_only and row.get("optimizer_consumed") != "1":
            continue
        action = row.get("canonical_action_name") or row.get("selected_action_name") or ""
        if not action:
            continue
        counts[action] = counts.get(action, 0) + 1
    if not counts:
        counts[fallback_action] = 1
    return ";".join(f"{action}={counts[action]}" for action in sorted(counts))


def _semantics_from_trace_rows(
    rows: list[dict[str, str]],
    fallback: BackendSemanticsDiff,
) -> BackendSemanticsDiff:
    if not rows:
        return fallback
    search_state_actions = {
        "bipop_search_state_restart",
        "phase_rescue_multistart",
        "repair_phase_rescue_multistart",
        "cc_harm_guarded_sep_refresh",
        "separable_cmaes_dispatch_action",
    }
    return BackendSemanticsDiff(
        variable_owner_changed=any(
            _trace_action(row) == "repair_shared_variable_binding"
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
        relation_handling_changed=any(
            _trace_action(row) == "isolate_conflicting_relation"
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
        coordination_mode_changed=any(
            _trace_action(row) == "allow_beneficial_coordination"
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
        budget_allocation_changed=any(
            _trace_action(row)
            in {"budget_shift_mean_blend", "budget_shift_only", *search_state_actions}
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
        update_order_changed=any(
            _trace_action(row) in search_state_actions
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
        acceptance_rule_changed=any(
            _trace_action(row) in search_state_actions
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
    )


def _trace_action(row: dict[str, str]) -> str:
    return row.get("canonical_action_name") or row.get("selected_action_name") or ""


def _relation_join_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        result = record["result"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(result, HccAobExecutionResult)
        if not lane.relation_dispatch_enabled:
            continue
        trace_ids = {
            row.get("relation_id", "")
            for row in _trace_rows_for_record(record)
            if row.get("relation_id")
        }
        decision_ids = {
            row.get("relation_id", "")
            for row in _artifact_rows_for_record(record, "action_decision.csv")
            if row.get("relation_id")
        }
        overlap_ids = {
            row.get("relation_id", "")
            for row in _artifact_rows_for_record(record, "overlap_relations.csv")
            if row.get("relation_id")
        }
        for relation_id in sorted(trace_ids | decision_ids | overlap_ids):
            has_trace = relation_id in trace_ids
            has_decision = relation_id in decision_ids
            has_overlap = relation_id in overlap_ids
            rows.append(
                {
                    "run_id": RUN_ID,
                    "lane_id": lane.lane_id,
                    "problem_id": result.problem_id,
                    "seed": result.seed,
                    "relation_id": relation_id,
                    "has_action_trace": int(has_trace),
                    "has_action_decision": int(has_decision),
                    "has_overlap_relation": int(has_overlap),
                    "audit_status": "pass"
                    if has_trace and has_decision and has_overlap
                    else "fail",
                }
            )
    return rows


def _relation_join_pass(record: dict[str, object]) -> bool:
    lane = record["lane"]
    assert isinstance(lane, LaneConfig)
    if not lane.relation_dispatch_enabled:
        return True
    rows = [
        row for row in _relation_join_rows([record])
        if row["lane_id"] == lane.lane_id
    ]
    return bool(rows) and all(row["audit_status"] == "pass" for row in rows)


def _records(
    output_dir: Path,
    execution_runner: Callable[[HccAobExecutionRequest], HccAobExecutionResult],
    hcc_root: Path,
    aob_data_root: Path,
    python_executable: str,
    seeds: tuple[int, ...],
    problem_ids: tuple[str, ...],
    max_fes: int,
    jobs: int = 1,
    budget_accounting: str = "strict",
    cmaes_restart: bool = True,
    mmes_restart: bool = True,
    lanes: tuple[LaneConfig, ...] = LANES,
    hcc_repo_root: Path | None = None,
    hcc_runner: Path | None = None,
    search_state_backend: str = "phase_i_mmes",
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    for problem_id in problem_ids:
        for seed in seeds:
            for lane in lanes:
                decision = _decision(lane)
                plan = build_hcc_action_execution_plan(problem_id, decision)
                semantics = hcc_backend_semantics_for(
                    decision,
                    optimizer_consumed=plan.optimizer_consumed,
                )
                payload = _runtime_payload(
                    problem_id,
                    seed,
                    lane.lane_id,
                    lane.selected_action_name,
                    max_fes,
                )
                lane_output = (
                    output_dir
                    / "_hcc_smoke"
                    / problem_id
                    / f"seed_{seed}"
                    / lane.lane_id
                ).resolve()
                contexts.append(
                    {
                        "lane": lane,
                        "lane_id": lane.lane_id,
                        "decision": decision,
                        "plan": plan,
                        "semantics": semantics,
                        "payload": payload,
                        "request": HccAobExecutionRequest(
                            problem_id=problem_id,
                            seed=seed,
                            max_fes=max_fes,
                            output_dir=lane_output,
                            hcc_root=hcc_root,
                            hcc_repo_root=hcc_repo_root,
                            hcc_runner=hcc_runner,
                            aob_data_root=aob_data_root,
                            python_executable=python_executable,
                            timestamp=f"{RUN_ID}-{problem_id}-seed{seed}-{lane.lane_id}",
                            arac_action=lane.runner_action_name,
                            enable_relation_dispatch=lane.relation_dispatch_enabled,
                            relation_policy_mode=lane.relation_policy_mode,
                            budget_accounting=budget_accounting,
                            cmaes_restart=cmaes_restart,
                            mmes_restart=mmes_restart,
                            search_state_backend=search_state_backend,
                            car_candidate_mode=lane.car_candidate_mode,
                            car_actionability_arm=lane.car_actionability_arm,
                            precision_causal_arm=lane.precision_causal_arm,
                            skip_plots=True,
                        ),
                    }
                )

    precision_requests = [
        context["request"]
        for context in contexts
        if isinstance(context["request"], HccAobExecutionRequest)
        and context["request"].precision_causal_arm != "off"
    ]
    if precision_requests:
        scheduled_pairs = [
            {
                "pair_id": precision_causal_pair_id(problem_id, seed),
                "problem_id": problem_id,
                "seed": seed,
                "logged_arm": precision_causal_logged_arm(problem_id, seed),
                "propensity": 0.5,
            }
            for problem_id in problem_ids
            for seed in seeds
        ]
        schedule = {
            "protocol_version": PRECISION_CAUSAL_PROTOCOL_VERSION,
            "status": "scheduled_before_subprocess",
            "randomization_salt": PRECISION_CAUSAL_RANDOMIZATION_SALT,
            "randomization_algorithm": "sha256_first_u64_mod2",
            "coin_material": "{salt}|{problem_id.upper()}|{int(seed)}",
            "arm_mapping": {"0": "baseline", "1": "action"},
            "preregistration": {
                "path": PRECISION_CAUSAL_PREREGISTRATION_PATH,
                "sha256": PRECISION_CAUSAL_PREREGISTRATION_SHA256,
                "commit": PRECISION_CAUSAL_PREREGISTRATION_COMMIT,
            },
            "pairs": scheduled_pairs,
        }
        schedule_path = output_dir / "causal_randomization_schedule.json"
        encoded_schedule = json.dumps(schedule, indent=2, sort_keys=True) + "\n"
        if schedule_path.exists():
            if schedule_path.read_text(encoding="utf-8") != encoded_schedule:
                raise RuntimeError(
                    "precision causal randomization schedule mismatch"
                )
        else:
            schedule_path.parent.mkdir(parents=True, exist_ok=True)
            schedule_path.write_text(encoded_schedule, encoding="utf-8")

    def run_context(context: dict[str, object]) -> dict[str, object]:
        request = context["request"]
        semantics = context["semantics"]
        plan = context["plan"]
        payload = context["payload"]
        decision = context["decision"]
        lane = context["lane"]
        assert isinstance(request, HccAobExecutionRequest)
        assert isinstance(semantics, BackendSemanticsDiff)
        assert isinstance(plan, HccActionExecutionPlan)
        assert isinstance(decision, ActionDecision)
        assert isinstance(lane, LaneConfig)
        result = _existing_completed_result(request)
        if result is None:
            if request.car_actionability_arm != "off":
                _prepare_car_actionability_provenance(request)
            if request.precision_causal_arm != "off":
                _prepare_precision_causal_provenance(request)
            result = execution_runner(request)
            if request.car_actionability_arm != "off":
                _complete_car_actionability_provenance(request, result)
            if request.precision_causal_arm != "off":
                _complete_precision_causal_provenance(request, result)
        trace_rows = _read_csv_rows(result.action_trace_path)
        semantics = _semantics_from_trace_rows(trace_rows, fallback=semantics)
        ledger = _ledger_for_result(result)
        effective_decision = _effective_claim_gate_decision(
            lane,
            decision,
            trace_rows,
        )
        allowed, blockers = claim_gate(
            runtime_payload=payload,
            decision=effective_decision,
            semantics_diff=semantics,
            ledger=ledger,
            utility_label="runtime_smoke_not_performance_claim",
            negative_control_pass=not lane.negative_control,
            optimizer_consumed=plan.optimizer_consumed,
        )
        return {
            "lane": lane,
            "lane_id": context["lane_id"],
            "decision": decision,
            "plan": plan,
            "semantics": semantics,
            "payload": payload,
            "ledger": ledger,
            "result": result,
            "claim_allowed": allowed,
            "claim_blockers": ";".join(blockers),
        }

    worker_count = max(1, int(jobs))
    if worker_count == 1:
        return [run_context(context) for context in contexts]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(run_context, contexts))


def _utility_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    fallback_by_case: dict[tuple[str, int], HccAobExecutionResult] = {}
    for record in records:
        if record["lane_id"] not in {"fallback", "v33_fallback"}:
            continue
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        fallback_by_case[(result.problem_id, result.seed)] = result
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        result = record["result"]
        ledger = record["ledger"]
        semantics = record["semantics"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(result, HccAobExecutionResult)
        assert isinstance(ledger, SameBudgetLedger)
        fallback_result = fallback_by_case.get((result.problem_id, result.seed))
        if fallback_result is None:
            utility_label = "no_fallback_reference"
            relative_gain_vs_fallback = "nan"
        else:
            utility_label = classify_utility(
                fallback_result.final_error,
                result.final_error,
            )
            relative_gain_vs_fallback = (
                f"{relative_gain(fallback_result.final_error, result.final_error):.6f}"
            )
        blockers: list[str] = []
        if fallback_result is None:
            blockers.append("no_fallback_reference")
        if lane.lane_id in {"fallback", "v33_fallback"}:
            blockers.append("comparison_lane_not_utility_claim")
        if lane.negative_control:
            blockers.append("negative_control_lane_not_utility_claim")
        if ledger.violation:
            blockers.append("same_budget_violation")
        if not ledger.fresh_execution:
            blockers.append("not_fresh_execution")
        if utility_label == "catastrophic_loss":
            blockers.append("catastrophic_loss")
        if lane.lane_id not in {"fallback", "v33_fallback"} and utility_label != "meaningful_win":
            blockers.append("utility_not_meaningful_win")
        if lane.relation_dispatch_enabled and not _relation_join_pass(record):
            blockers.append("relation_artifact_join_failed")
        claim_allowed = not blockers
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": lane.lane_id,
                "problem_id": result.problem_id,
                "seed": result.seed,
                "final_error": f"{result.final_error:.6e}",
                "fe_used": result.fe_used,
                "same_budget_violation": int(ledger.violation),
                "relative_gain_vs_fallback": relative_gain_vs_fallback,
                "utility_label": utility_label,
                "action_mix": _format_action_mix(
                    _trace_rows_for_record(record),
                    lane.selected_action_name,
                ),
                "optimizer_consumed_action_mix": _format_action_mix(
                    _trace_rows_for_record(record),
                    lane.selected_action_name,
                    optimizer_consumed_only=True,
                ),
                "runtime_connected_claim_allowed": int(
                    result.fresh_optimizer_execution
                    and bool(result.action_trace_rows)
                    and (not lane.relation_dispatch_enabled or _relation_join_pass(record))
                ),
                "backend_semantics_changed": int(semantics.changed),
                "claim_allowed": int(claim_allowed),
                "claim_blockers": ";".join(sorted(set(blockers))),
            }
        )
    return rows


def _our_result_rows(
    records: list[dict[str, object]],
    utility_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    utility_claim_by_lane = {
        (row["problem_id"], row["lane_id"], row["seed"]): row["claim_allowed"]
        for row in utility_rows
    }
    runtime_claim_by_lane = {
        (row["problem_id"], row["lane_id"], row["seed"]): row[
            "runtime_connected_claim_allowed"
        ]
        for row in utility_rows
    }
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        decision = record["decision"]
        result = record["result"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(decision, ActionDecision)
        assert isinstance(result, HccAobExecutionResult)
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": lane.lane_id,
                "problem_id": result.problem_id,
                "seed": result.seed,
                "selected_action_family": decision.action_family.value,
                "selected_action_name": lane.selected_action_name,
                "hcc_smoke_final_error": f"{result.final_error:.6e}",
                "hcc_smoke_fe_used": result.fe_used,
                "hcc_smoke_status": result.status,
                "fresh_optimizer_execution": int(result.fresh_optimizer_execution),
                "result_source": result.result_source,
                "action_trace_sha256": _sha256_file(result.action_trace_path)
                if result.action_trace_path is not None
                else "missing",
                "action_trace_rows": result.action_trace_rows,
                "runtime_dispatch_allowed": 1,
                "dispatch_scope": lane.dispatch_scope,
                "relation_dispatch_enabled": int(lane.relation_dispatch_enabled),
                "runtime_connected_claim_allowed": runtime_claim_by_lane[
                    (result.problem_id, lane.lane_id, result.seed)
                ],
                "utility_claim_allowed": utility_claim_by_lane[
                    (result.problem_id, lane.lane_id, result.seed)
                ],
                "performance_claim_allowed": 0,
            }
        )
    return rows


def _ledger_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        actual_fe_used = _actual_fe_used(result)
        phase_i_fe = min(actual_fe_used, max(0, result.global_phase_fe or 0))
        cc_phase_fe = max(0, result.cc_phase_fe or 0)
        rescue_fe = max(0, result.rescue_fe or 0)
        refresh_fe = max(0, result.refresh_fe or 0)
        search_state_fe = max(0, result.search_state_fe or 0)
        separable_continuation_fe = max(
            0,
            result.separable_continuation_fe or 0,
        )
        known_stage_fe = (
            phase_i_fe
            + cc_phase_fe
            + rescue_fe
            + refresh_fe
            + search_state_fe
            + separable_continuation_fe
        )
        overhead_fe = (
            max(0, actual_fe_used - known_stage_fe)
            if result.overhead_fe is None
            else max(0, result.overhead_fe)
        )
        if known_stage_fe + overhead_fe != actual_fe_used:
            raise RuntimeError(
                f"stage FE mismatch for {result.problem_id} seed {result.seed}: "
                f"stages={known_stage_fe + overhead_fe}, total={actual_fe_used}"
            )
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": result.problem_id,
                "seed": result.seed,
                "same_budget_group_id": _same_budget_group_id(
                    result.problem_id,
                    result.seed,
                    result.max_fes,
                ),
                "phase_i_fe": phase_i_fe,
                "phase_ii_fe": actual_fe_used - phase_i_fe,
                "cc_phase_fe": cc_phase_fe,
                "rescue_fe": rescue_fe,
                "refresh_fe": refresh_fe,
                "search_state_fe": search_state_fe,
                "separable_continuation_fe": separable_continuation_fe,
                "overhead_fe": overhead_fe,
                "total_fe": actual_fe_used,
                "budget_limit": result.max_fes,
                "configured_budget_limit": result.max_fes,
                "budget_aligned_fe_used": result.fe_used,
                "actual_fe_used": actual_fe_used,
                "budget_limit_source": "experiment_config",
                "same_budget_violation": int(actual_fe_used > result.max_fes),
                "fresh_execution": int(result.fresh_optimizer_execution),
            }
        )
    return rows


def _semantics_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        decision = record["decision"]
        result = record["result"]
        semantics = record["semantics"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(decision, ActionDecision)
        assert isinstance(result, HccAobExecutionResult)
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": result.problem_id,
                "seed": result.seed,
                "selected_action_name": lane.selected_action_name,
                "variable_owner_changed": int(semantics.variable_owner_changed),
                "relation_handling_changed": int(semantics.relation_handling_changed),
                "coordination_mode_changed": int(semantics.coordination_mode_changed),
                "budget_allocation_changed": int(semantics.budget_allocation_changed),
                "update_order_changed": int(semantics.update_order_changed),
                "acceptance_rule_changed": int(semantics.acceptance_rule_changed),
                "backend_semantics_changed": int(semantics.changed),
            }
        )
    return rows


def _action_execution_plan_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row = record["plan"].to_csv_row()
        row["run_id"] = RUN_ID
        row["lane_id"] = record["lane_id"]
        rows.append(row)
    return rows


def _action_trace_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(_with_lane_prefix(record, _trace_rows_for_record(record)))
    return rows


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


def action_trace_fields_for_lanes(lanes: tuple[LaneConfig, ...]) -> list[str]:
    fields: list[str] = []
    precision_causal_enabled = any(
        lane.precision_causal_arm != "off" for lane in lanes
    )
    if any(
        controller_has_capability(lane.runner_action_name, "trust_trace")
        for lane in lanes
    ):
        fields.extend(V33_TRUST_TRACE_FIELDS)
    if any(
        controller_has_capability(lane.runner_action_name, "trajectory_guard")
        for lane in lanes
    ):
        fields.extend(V34_RECOVERY_TRACE_FIELDS)
    if any(
        controller_has_capability(lane.runner_action_name, "maturity")
        for lane in lanes
    ):
        fields.extend(V36_MATURITY_TRACE_FIELDS)
    if any(
        controller_has_capability(lane.runner_action_name, "rescue_retirement")
        for lane in lanes
    ):
        fields.extend(V37_RESOURCE_TRACE_FIELDS)
    if any(
        controller_has_capability(lane.runner_action_name, "sigma_continuation")
        for lane in lanes
    ):
        fields.extend(V39_CMA_SIGMA_TRACE_FIELDS)
    if any(
        controller_has_capability(
            lane.runner_action_name,
            "component_delayed_credit_trace",
        )
        for lane in lanes
    ) or precision_causal_enabled:
        fields.extend(COMPONENT_CREDIT_TRACE_FIELDS)
    if precision_causal_enabled:
        fields.extend(PRECISION_CAUSAL_DIAGNOSTIC_TRACE_FIELDS)
    return fields


TRAJECTORY_GUARD_SUMMARY_FIELDS = [
    "run_id",
    "lane_id",
    "problem_id",
    "seed",
    "pending_count",
    "committed_count",
    "restored_count",
    "preempted_restored_count",
    "total_resolved_count",
    "restore_rate",
]


def _trajectory_guard_summary_rows(
    action_trace_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    status_fields = {
        "pending": "pending_count",
        "committed": "committed_count",
        "restored": "restored_count",
        "preempted_restored": "preempted_restored_count",
    }
    summaries: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for trace_row in action_trace_rows:
        status = str(trace_row.get("trajectory_guard_status", "")).strip()
        key = tuple(
            str(trace_row.get(field, ""))
            for field in ("run_id", "lane_id", "problem_id", "seed")
        )
        row = summaries.setdefault(
            key,
            {
                "run_id": key[0],
                "lane_id": key[1],
                "problem_id": key[2],
                "seed": key[3],
                "pending_count": 0,
                "committed_count": 0,
                "restored_count": 0,
                "preempted_restored_count": 0,
            },
        )
        if not status:
            continue
        if status not in status_fields:
            raise ValueError(f"unsupported trajectory guard status: {status}")
        count_field = status_fields[status]
        row[count_field] = int(row[count_field]) + 1

    rows: list[dict[str, object]] = []
    for row in summaries.values():
        total_resolved = sum(
            int(row[field])
            for field in (
                "committed_count",
                "restored_count",
                "preempted_restored_count",
            )
        )
        restored = int(row["restored_count"]) + int(
            row["preempted_restored_count"]
        )
        row["total_resolved_count"] = total_resolved
        row["restore_rate"] = (
            restored / total_resolved if total_resolved else ""
        )
        rows.append(row)
    return rows


PRE_HOLD_EVIDENCE_FIELDS = [
    "run_id",
    "lane_id",
    "problem_id",
    "seed",
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
]


def _pre_hold_evidence_rows(
    trace_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in trace_rows
        if str(row.get("pre_hold_group_count", "")).strip()
    ]


def _action_decision_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            _with_lane_prefix(record, _artifact_rows_for_record(record, "action_decision.csv"))
        )
    return rows


def _action_mismatch_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            _with_lane_prefix(
                record,
                _artifact_rows_for_record(record, "action_mismatch_audit.csv"),
            )
        )
    return rows


def _overlap_relation_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            _with_lane_prefix(record, _artifact_rows_for_record(record, "overlap_relations.csv"))
        )
    return rows


def _car_artifact_rows(
    records: list[dict[str, object]],
    artifact_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            _with_lane_prefix(
                record,
                _artifact_rows_for_record(record, artifact_name),
            )
        )
    return rows


def _car_actionability_rows_for_record(
    record: dict[str, object],
) -> list[dict[str, object]]:
    """Read audit rows only after validating their immutable lane identity."""

    result = record["result"]
    lane = record["lane"]
    assert isinstance(result, HccAobExecutionResult)
    assert isinstance(lane, LaneConfig)
    path = _find_lane_artifact(result, "car_actionability_trace.csv")
    provenance_path = Path(result.output_root) / "car_actionability_provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("missing CAR actionability provenance") from exc
    verified_path = _provenance_artifact_path(
        provenance.get("trace_path"), Path(result.output_root)
    )
    if (
        path is None
        or verified_path is None
        or path.resolve() != verified_path
        or provenance.get("status") != "complete"
        or provenance.get("fresh_optimizer_execution") is not True
        or not provenance.get("request_fingerprint")
        or provenance.get("trace_sha256") != _sha256_file(verified_path)
    ):
        raise ValueError("unverified CAR actionability trace provenance")
    request_payload = provenance.get("request")
    if not isinstance(request_payload, dict):
        raise ValueError("missing CAR actionability provenance request")
    if _fingerprint_payload(request_payload) != provenance.get("request_fingerprint"):
        raise ValueError("CAR actionability provenance request hash mismatch")
    execution_context_fingerprint = str(
        request_payload.get("execution_context_fingerprint", "")
    )
    aob_input_fingerprint = str(request_payload.get("aob_input_fingerprint", ""))
    if not execution_context_fingerprint or not aob_input_fingerprint:
        raise ValueError("incomplete CAR actionability provenance context")
    rows = _read_csv_rows(path)
    if not rows:
        return []
    expected_arm = lane.car_actionability_arm
    for row in rows:
        if row.get("protocol_version") != CAR_ACTIONABILITY_PROTOCOL_VERSION:
            raise ValueError("CAR actionability trace protocol version mismatch")
        if row.get("fresh_optimizer_execution") != "1":
            raise ValueError("CAR actionability trace is not marked fresh")
        if row.get("problem_id") != result.problem_id:
            raise ValueError("CAR actionability trace problem identity mismatch")
        if str(row.get("seed")) != str(result.seed):
            raise ValueError("CAR actionability trace seed identity mismatch")
        if row.get("audit_arm") != expected_arm:
            raise ValueError("CAR actionability trace arm identity mismatch")
        if row.get("candidate_mode") != lane.car_candidate_mode:
            raise ValueError("CAR actionability trace candidate mode mismatch")
        if row.get("configured_max_fes") != str(result.max_fes):
            raise ValueError("CAR actionability trace budget identity mismatch")
    return [
        {
            **row,
            "run_id": RUN_ID,
            "lane_id": record["lane_id"],
            "seed": result.seed,
            "request_fingerprint": provenance["request_fingerprint"],
            "execution_context_fingerprint": execution_context_fingerprint,
            "aob_input_fingerprint": aob_input_fingerprint,
        }
        for row in rows
    ]


def _aob_input_manifest_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            _with_lane_prefix(
                record,
                _artifact_rows_for_record(record, "aob_input_manifest.csv"),
            )
        )
    return rows


def _car_actionability_aob_pair_failures(
    rows: list[dict[str, object]],
    *,
    problem_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    lanes: tuple[LaneConfig, ...],
) -> list[str]:
    audit_lane_ids = tuple(
        lane.lane_id for lane in lanes if lane.car_actionability_arm != "off"
    )
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("problem_id", "")),
            str(row.get("seed", "")),
            str(row.get("lane_id", "")),
        )
        filename = str(row.get("file", ""))
        digest = str(row.get("sha256_before", ""))
        if filename:
            indexed.setdefault(key, {})[filename] = digest

    failures: list[str] = []
    for problem_id in problem_ids:
        for seed in seeds:
            lane_maps = {
                lane_id: indexed.get((problem_id, str(seed), lane_id), {})
                for lane_id in audit_lane_ids
            }
            missing = [lane_id for lane_id, mapping in lane_maps.items() if not mapping]
            prefix = f"{problem_id}/seed{seed}"
            if missing:
                failures.append(f"{prefix}:missing_aob_lane={','.join(missing)}")
                continue
            reference_lane = audit_lane_ids[0]
            reference = lane_maps[reference_lane]
            for lane_id in audit_lane_ids[1:]:
                candidate = lane_maps[lane_id]
                if set(reference) != set(candidate):
                    failures.append(f"{prefix}:{lane_id}:aob_file_set_mismatch")
                    continue
                mismatched = sorted(
                    filename
                    for filename in reference
                    if not reference[filename]
                    or reference[filename] == "missing"
                    or reference[filename] != candidate[filename]
                )
                if mismatched:
                    failures.append(
                        f"{prefix}:{lane_id}:aob_hash_mismatch="
                        + ",".join(mismatched)
                    )
    return failures


def _anti_leakage_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    payloads = [record["payload"] for record in records]
    artifact_rows = (
        _action_trace_rows(records)
        + _action_decision_rows(records)
        + _action_mismatch_rows(records)
        + _overlap_relation_rows(records)
    )
    rows: list[dict[str, object]] = []
    for field in sorted(FORBIDDEN_RUNTIME_FIELDS):
        found = any(field in payload for payload in payloads)
        artifact_found = any(field in row for row in artifact_rows)
        rows.append(
            {
                "run_id": RUN_ID,
                "artifact_path": (
                    "runtime_payload;action_trace.csv;"
                    "action_decision.csv;overlap_relations.csv"
                ),
                "forbidden_field": field,
                "found_in_runtime_payload": int(found or artifact_found),
                "runtime_dispatch_allowed": 0 if found or artifact_found else 1,
                "audit_status": "fail" if found or artifact_found else "pass",
            }
        )
    return rows


def _claim_gate_rows(
    records: list[dict[str, object]],
    utility_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    utility_by_case = {
        (row["problem_id"], row["lane_id"], row["seed"]): row
        for row in utility_rows
    }
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        decision = record["decision"]
        plan = record["plan"]
        ledger = record["ledger"]
        result = record["result"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(decision, ActionDecision)
        assert isinstance(ledger, SameBudgetLedger)
        assert isinstance(result, HccAobExecutionResult)
        utility_row = utility_by_case[(result.problem_id, lane.lane_id, result.seed)]
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": result.problem_id,
                "seed": result.seed,
                "selected_action_name": lane.selected_action_name,
                "optimizer_consumed": int(plan.optimizer_consumed),
                "same_budget_violation": int(ledger.violation),
                "runtime_connected_claim_allowed": int(record["claim_allowed"]),
                "runtime_claim_blockers": record["claim_blockers"],
                "utility_claim_allowed": utility_row["claim_allowed"],
                "utility_claim_blockers": utility_row["claim_blockers"],
                "performance_claim_allowed": 0,
                "claim_allowed": utility_row["claim_allowed"],
                "claim_blockers": utility_row["claim_blockers"],
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _parse_action_mix(value: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(value).split(";"):
        if not part or "=" not in part:
            continue
        action, count = part.rsplit("=", 1)
        counts[action] = counts.get(action, 0) + int(count)
    return counts


def _has_active_relation_action(row: dict[str, object]) -> bool:
    counts = _parse_action_mix(
        row.get("optimizer_consumed_action_mix") or row.get("action_mix", "")
    )
    return any(
        action != "conservative_no_action" and count > 0
        for action, count in counts.items()
    )


def _active_relation_density(row: dict[str, object]) -> float:
    counts = _parse_action_mix(row.get("action_mix", ""))
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    active = sum(
        count
        for action, count in counts.items()
        if action != "conservative_no_action"
    )
    return active / total


def _expects_backend_semantics(row: dict[str, object]) -> bool:
    lane_id = row["lane_id"]
    if lane_id in {"relation_dispatch_rule", "shuffled_relation_dispatch"}:
        return _has_active_relation_action(row)
    return lane_id != "fallback"


def _format_action_counts(counts: dict[str, int]) -> str:
    return ";".join(f"{action}={counts[action]}" for action in sorted(counts))


def _format_inline_counts(counts: dict[str, int]) -> str:
    return ",".join(f"{action}={counts[action]}" for action in sorted(counts))


def _mean_numeric(rows: list[dict[str, object]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return _mean(values)


def _format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _gain_bucket(gain: float) -> str:
    if gain > 0.0:
        return "win"
    if gain < 0.0:
        return "loss"
    return "tie"


def _aggregate_lane_action_mix(
    utility_rows: list[dict[str, object]],
    lane_id: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in utility_rows:
        if row["lane_id"] != lane_id:
            continue
        for action, count in _parse_action_mix(row.get("action_mix", "")).items():
            counts[action] = counts.get(action, 0) + count
    return counts


def _action_mix_for_gain_bucket(
    rows: list[dict[str, object]],
    bucket: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        gain = float(row["relative_gain_vs_fallback"])
        if (
            (bucket == "win" and gain <= 0.0)
            or (bucket == "loss" and gain >= 0.0)
            or (bucket == "tie" and gain != 0.0)
        ):
            continue
        for action, count in _parse_action_mix(row.get("action_mix", "")).items():
            counts[action] = counts.get(action, 0) + count
    return counts


def _action_value_delta_profile_for_gain_bucket(
    trace_rows: list[dict[str, object]],
    relation_rows: list[dict[str, object]],
    bucket: str,
) -> str:
    gain_by_case = {
        (str(row["problem_id"]), str(row["seed"])): float(
            row["relative_gain_vs_fallback"]
        )
        for row in relation_rows
    }
    values_by_action: dict[str, list[float]] = {}
    for row in trace_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        gain = gain_by_case.get((str(row.get("problem_id", "")), str(row.get("seed", ""))))
        if gain is None:
            continue
        if (
            (bucket == "win" and gain <= 0.0)
            or (bucket == "loss" and gain >= 0.0)
            or (bucket == "tie" and gain != 0.0)
        ):
            continue
        try:
            value = float(row.get("action_value_delta_norm", ""))
        except (TypeError, ValueError):
            continue
        action = str(row.get("canonical_action_name") or row.get("selected_action_name", ""))
        if not action:
            continue
        values_by_action.setdefault(action, []).append(value)
    return ";".join(
        f"{action}:n={len(values)},mean={_mean(values):.6f},max={max(values):.6f}"
        for action, values in sorted(values_by_action.items())
    )


def _result_by_problem_seed_and_lane(
    records: list[dict[str, object]],
) -> dict[tuple[str, int, str], HccAobExecutionResult]:
    indexed: dict[tuple[str, int, str], HccAobExecutionResult] = {}
    for record in records:
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        indexed[(result.problem_id, result.seed, str(record["lane_id"]))] = result
    return indexed


def _negative_control_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = _result_by_problem_seed_and_lane(records)
    lane_by_id = {
        str(record["lane_id"]): record["lane"]
        for record in records
        if isinstance(record.get("lane"), LaneConfig)
    }
    available_lane_ids = set(lane_by_id).union(
        lane_id for _problem_id, _seed, lane_id in indexed
    )
    candidate_lane_id = next(
        (
            lane_id
            for lane_id in (
                "candidate",
                "relation_dispatch_rule",
                "car_w",
                "car_w2",
                "car_w3",
            )
            if lane_id in available_lane_ids
        ),
        "",
    )
    negative_lane_ids = [
        lane_id
        for lane_id, lane in lane_by_id.items()
        if isinstance(lane, LaneConfig) and lane.negative_control
    ]
    sibling_suffixes = ("_shuffled", "_paired_fallback")
    negative_lane_ids.extend(
        f"{candidate_lane_id}{suffix}"
        for suffix in sibling_suffixes
        if candidate_lane_id
        and f"{candidate_lane_id}{suffix}" in available_lane_ids
    )
    negative_lane_ids = sorted(set(negative_lane_ids))
    if not negative_lane_ids:
        negative_lane_ids = sorted(
            lane_id
            for _problem_id, _seed, lane_id in indexed
            if lane_id in {
                "shuffled_relation_dispatch",
                "no_action_negative_control",
            }
        )
    problem_ids = sorted(
        problem_id
        for problem_id, _seed, lane_id in indexed
        if lane_id == candidate_lane_id
    )
    rows: list[dict[str, object]] = []
    for problem_id in sorted(set(problem_ids)):
        comparisons = []
        for negative_lane_id in negative_lane_ids:
            seeds = sorted(
                seed
                for indexed_problem_id, seed, lane_id in indexed
                if indexed_problem_id == problem_id
                and lane_id == candidate_lane_id
                and (problem_id, seed, negative_lane_id) in indexed
            )
            if not seeds:
                continue
            candidate_errors = [
                indexed[(problem_id, seed, candidate_lane_id)].final_error
                for seed in seeds
            ]
            negative_errors = [
                indexed[(problem_id, seed, negative_lane_id)].final_error
                for seed in seeds
            ]
            negative_win_count = sum(
                1
                for candidate_error, negative_error in zip(
                    candidate_errors,
                    negative_errors,
                    strict=True,
                )
                if classify_utility(candidate_error, negative_error) == "meaningful_win"
            )
            comparisons.append(
                (
                    negative_win_count,
                    -_mean(negative_errors),
                    negative_lane_id,
                    seeds,
                    candidate_errors,
                    negative_errors,
                )
            )
        if not comparisons:
            continue
        (
            shuffled_win_count,
            _negative_mean,
            negative_lane_id,
            seeds,
            relation_errors,
            shuffled_errors,
        ) = max(comparisons)
        total = len(seeds)
        stable_outperform = shuffled_win_count > total / 2
        legacy_shuffled = negative_lane_id in {
            "shuffled_relation_dispatch",
            "car_w_shuffled",
            "car_w2_shuffled",
            "car_w3_shuffled",
        }
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": problem_id,
                "seeds": ";".join(str(seed) for seed in seeds),
                "relation_dispatch_mean_final_error": f"{_mean(relation_errors):.6e}",
                "shuffled_mean_final_error": f"{_mean(shuffled_errors):.6e}",
                "shuffled_win_count": shuffled_win_count,
                "total_seeds": total,
                "stable_outperform_detected": int(stable_outperform),
                "negative_control_pass": int(not stable_outperform),
                "diagnostic": (
                    (
                        "shuffled_control_stably_outperforms_relation_dispatch"
                        if legacy_shuffled
                        else "negative_control_stably_outperforms_candidate"
                    )
                    if stable_outperform
                    else (
                        "shuffled_control_not_stably_better"
                        if legacy_shuffled
                        else "negative_control_not_stably_better"
                    )
                ),
            }
        )
    return rows


def _car_dispatch_boundary_rows() -> list[dict[str, object]]:
    """Materialize the CAR type boundary as an auditable artifact."""

    runtime_fields = set(DispatchEvidence.runtime_field_names())
    forbidden_fields = set(DispatchEvidence.forbidden_field_names())
    audit_fields = set(AuditEnvelope.__dataclass_fields__)
    overlap = runtime_fields & (forbidden_fields | audit_fields)
    rows: list[dict[str, object]] = []
    for field in sorted(runtime_fields):
        rows.append(
            {
                "boundary": "DispatchEvidence",
                "field_name": field,
                "field_owner": "runtime_dispatch",
                "present_in_runtime_type": 1,
                "audit_only": 0,
                "runtime_dispatch_allowed": int(field not in overlap),
                "audit_status": "pass" if field not in overlap else "fail",
            }
        )
    for field in sorted(audit_fields | forbidden_fields):
        owner = "AuditEnvelope" if field in audit_fields else "forbidden_runtime"
        rows.append(
            {
                "boundary": "DispatchEvidence",
                "field_name": field,
                "field_owner": owner,
                "present_in_runtime_type": int(field in runtime_fields),
                "audit_only": 1,
                "runtime_dispatch_allowed": 0,
                "audit_status": "pass" if field not in overlap else "fail",
            }
        )
    return rows


def _paired_runtime_utility_rows(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    indexed = _result_by_problem_seed_and_lane(records)
    pairs_by_problem: dict[str, list[tuple[float, float]]] = {}
    for problem_id, seed, lane_id in indexed:
        if lane_id != "candidate" or (problem_id, seed, "fallback") not in indexed:
            continue
        fallback = indexed[(problem_id, seed, "fallback")].final_error
        candidate = indexed[(problem_id, seed, "candidate")].final_error
        if not all(math.isfinite(value) and value >= 0.0 for value in (fallback, candidate)):
            raise ValueError(
                f"paired runtime utility requires finite non-negative errors: {problem_id}/seed{seed}"
            )
        pairs_by_problem.setdefault(problem_id, []).append((fallback, candidate))

    def summarize(problem_id: str, pairs: list[tuple[float, float]]) -> dict[str, object]:
        fallback_errors = [fallback for fallback, _candidate in pairs]
        candidate_errors = [candidate for _fallback, candidate in pairs]
        labels = [
            classify_utility(fallback, candidate)
            for fallback, candidate in pairs
        ]
        return {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "seed_count": len(pairs),
            "fallback_mean_error": f"{_mean(fallback_errors):.17e}",
            "candidate_mean_error": f"{_mean(candidate_errors):.17e}",
            "fallback_worst_error": f"{max(fallback_errors):.17e}",
            "candidate_worst_error": f"{max(candidate_errors):.17e}",
            "mean_log_error_delta": (
                f"{_mean([math.log1p(candidate) - math.log1p(fallback) for fallback, candidate in pairs]):.17e}"
            ),
            "meaningful_seed_wins": sum(label == "meaningful_win" for label in labels),
            "catastrophic_losses": sum(label == "catastrophic_loss" for label in labels),
            "mean_win": int(_mean(candidate_errors) < _mean(fallback_errors)),
            "worst_seed_win": int(max(candidate_errors) < max(fallback_errors)),
        }

    case_rows = [
        summarize(problem_id, pairs_by_problem[problem_id])
        for problem_id in sorted(pairs_by_problem)
    ]
    if not case_rows:
        return []
    all_pairs = [
        pair
        for problem_id in sorted(pairs_by_problem)
        for pair in pairs_by_problem[problem_id]
    ]
    aggregate = summarize("ALL", all_pairs)
    aggregate["mean_win"] = sum(int(row["mean_win"]) for row in case_rows)
    aggregate["worst_seed_win"] = sum(int(row["worst_seed_win"]) for row in case_rows)
    return [*case_rows, aggregate]


def _car_actionability_summary_rows(
    trace_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Pair offline oracle arms at identical absolute-FE horizons.

    This function is deliberately downstream of raw traces.  It never feeds
    terminal outcomes back into runtime evidence or dispatch.
    """

    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in trace_rows:
        key = (
            str(row.get("problem_id", "")),
            str(row.get("seed", "")),
            str(row.get("horizon_label", "")),
        )
        arm = str(row.get("audit_arm", ""))
        if arm not in {"fallback", "candidate"}:
            continue
        bucket = grouped.setdefault(key, {})
        if arm in bucket:
            raise ValueError(f"duplicate CAR actionability arm: {key}/{arm}")
        bucket[arm] = row

    def numeric(row: dict[str, str], field: str) -> float | None:
        value = str(row.get(field, "")).strip()
        if not value:
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        problem_id, seed, horizon_label = key
        arms = grouped[key]
        fallback = arms.get("fallback")
        candidate = arms.get("candidate")
        failures: list[str] = []
        if fallback is None:
            failures.append("missing_fallback_arm")
        if candidate is None:
            failures.append("missing_candidate_arm")
        if fallback is None or candidate is None:
            rows.append(
                {
                    "run_id": RUN_ID,
                    "problem_id": problem_id,
                    "seed": seed,
                    "horizon_label": horizon_label,
                    "prefix_match": 0,
                    "equal_fe": 0,
                    "fallback_error": "",
                    "candidate_error": "",
                    "log_advantage": "",
                    "relative_gain": "",
                    "numeric_win": 0,
                    "meaningful_win": 0,
                    "catastrophic_loss": 0,
                    "oracle_selected_arm": "",
                    "oracle_gain": "",
                    "integrity_status": "fail",
                    "integrity_failures": ";".join(failures),
                }
            )
            continue

        checkpoint_match = fallback.get("checkpoint_fe", "") == candidate.get(
            "checkpoint_fe", ""
        )
        target_match = fallback.get("target_fe", "") == candidate.get("target_fe", "")
        observed_match = fallback.get("observed_fe", "") == candidate.get(
            "observed_fe", ""
        )
        horizon_complete = (
            fallback.get("horizon_status") == "complete"
            and candidate.get("horizon_status") == "complete"
            and fallback.get("observed_fe") == fallback.get("target_fe")
            and candidate.get("observed_fe") == candidate.get("target_fe")
        )
        intervention_match = fallback.get("actual_fe", "") == candidate.get(
            "actual_fe", ""
        )
        plan_match = fallback.get("plan_status", "") == candidate.get(
            "plan_status", ""
        )
        plan_applied = fallback.get("plan_status") == "applied"
        crn_match = (
            bool(fallback.get("seed_descriptor", ""))
            and bool(fallback.get("probe_seed", ""))
            and fallback.get("seed_descriptor", "")
            == candidate.get("seed_descriptor", "")
            and fallback.get("probe_seed", "") == candidate.get("probe_seed", "")
        )
        action_match = all(
            bool(fallback.get(field, ""))
            and fallback.get(field, "") == candidate.get(field, "")
            for field in (
                "graph_fingerprint",
                "component_fingerprint",
                "candidate_action_name",
                "candidate_action_family",
            )
        )
        configured_budget_match = fallback.get(
            "configured_max_fes", ""
        ) == candidate.get("configured_max_fes", "")
        execution_context_match = (
            bool(fallback.get("execution_context_fingerprint", ""))
            and fallback.get("execution_context_fingerprint", "")
            == candidate.get("execution_context_fingerprint", "")
        )
        aob_input_match = (
            bool(fallback.get("aob_input_fingerprint", ""))
            and fallback.get("aob_input_fingerprint", "")
            == candidate.get("aob_input_fingerprint", "")
        )
        arm_semantics_valid = (
            (
                not plan_applied
                and fallback.get("candidate_action_applied") == "0"
                and candidate.get("candidate_action_applied") == "0"
            )
            or (
                plan_applied
                and fallback.get("candidate_action_applied") == "0"
                and candidate.get("candidate_action_applied") == "1"
            )
        )
        prefix_match = (
            bool(fallback.get("prefix_state_fingerprint"))
            and fallback.get("prefix_state_fingerprint")
            == candidate.get("prefix_state_fingerprint")
            and bool(fallback.get("prefix_record_sha256"))
            and fallback.get("prefix_record_sha256")
            == candidate.get("prefix_record_sha256")
        )
        if not prefix_match:
            failures.append("prefix_mismatch")
        if not checkpoint_match or not target_match or not observed_match:
            failures.append("unequal_absolute_horizon")
        if not intervention_match:
            failures.append("unequal_intervention_fe")
        if not configured_budget_match:
            failures.append("configured_budget_mismatch")
        if not execution_context_match:
            failures.append("execution_context_mismatch")
        if not aob_input_match:
            failures.append("aob_input_mismatch")
        if not plan_match:
            failures.append("plan_status_mismatch")
        if plan_applied and not crn_match:
            failures.append("crn_seed_mismatch")
        if plan_applied and not action_match:
            failures.append("action_identity_mismatch")
        if not arm_semantics_valid:
            failures.append("invalid_one_shot_arm_semantics")
        if not horizon_complete:
            failures.append("incomplete_horizon")
        fallback_error = numeric(fallback, "best_error")
        candidate_error = numeric(candidate, "best_error")
        if fallback_error is not None and fallback_error < 0.0:
            fallback_error = None
        if candidate_error is not None and candidate_error < 0.0:
            candidate_error = None
        if fallback_error is None or candidate_error is None:
            failures.append("missing_or_nonfinite_error")
        integrity_status = "pass" if not failures else "fail"
        if integrity_status != "pass":
            fallback_error = None
            candidate_error = None
            log_advantage = ""
            gain = ""
            numeric_win = meaningful_win = catastrophic_loss = 0
            selected = ""
            oracle_gain = ""
        else:
            log_advantage_value = log_actionability_advantage(
                fallback_error,
                candidate_error,
            )
            gain_value = relative_gain(fallback_error, candidate_error)
            utility = classify_utility(fallback_error, candidate_error)
            log_advantage = f"{log_advantage_value:.17e}"
            gain = f"{gain_value:.17e}"
            numeric_win = int(candidate_error < fallback_error)
            meaningful_win = int(utility == "meaningful_win")
            catastrophic_loss = int(utility == "catastrophic_loss")
            selected = (
                "candidate"
                if candidate_error < fallback_error
                else "fallback"
                if fallback_error < candidate_error
                else "tie"
            )
            oracle_gain = f"{max(gain_value, 0.0):.17e}"
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": problem_id,
                "seed": seed,
                "horizon_label": horizon_label,
                "horizon_index": fallback.get("horizon_index", ""),
                "checkpoint_fe": fallback.get("checkpoint_fe", ""),
                "configured_max_fes": fallback.get("configured_max_fes", ""),
                "fallback_request_fingerprint": fallback.get(
                    "request_fingerprint", ""
                ),
                "candidate_request_fingerprint": candidate.get(
                    "request_fingerprint", ""
                ),
                "execution_context_fingerprint": fallback.get(
                    "execution_context_fingerprint", ""
                ),
                "aob_input_fingerprint": fallback.get(
                    "aob_input_fingerprint", ""
                ),
                "target_fe": fallback.get("target_fe", ""),
                "observed_fe": fallback.get("observed_fe", ""),
                "prefix_match": int(prefix_match),
                "equal_fe": int(
                    checkpoint_match
                    and target_match
                    and observed_match
                    and intervention_match
                    and horizon_complete
                    and configured_budget_match
                    and execution_context_match
                    and aob_input_match
                    and plan_match
                    and (crn_match or not plan_applied)
                    and (action_match or not plan_applied)
                    and arm_semantics_valid
                ),
                "fallback_error": (
                    "" if fallback_error is None else f"{fallback_error:.17e}"
                ),
                "candidate_error": (
                    "" if candidate_error is None else f"{candidate_error:.17e}"
                ),
                "log_advantage": log_advantage,
                "relative_gain": gain,
                "numeric_win": numeric_win,
                "meaningful_win": meaningful_win,
                "catastrophic_loss": catastrophic_loss,
                "oracle_selected_arm": selected,
                "oracle_gain": oracle_gain,
                "integrity_status": integrity_status,
                "integrity_failures": ";".join(failures),
            }
        )

    # Add terminal sign agreement and rank reversal without changing the raw
    # fact source.  Missing/blocked rows remain explicitly blocked.
    by_run: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault((str(row["problem_id"]), str(row["seed"])), []).append(row)
    for run_rows in by_run.values():
        ordered = sorted(run_rows, key=lambda item: str(item.get("horizon_index", "")))
        terminal = next(
            (item for item in ordered if item.get("horizon_label") == "terminal"),
            None,
        )
        terminal_sign = 0
        if terminal is not None and terminal.get("integrity_status") == "pass":
            try:
                value = float(terminal.get("log_advantage", ""))
                terminal_sign = 1 if value > 0 else -1 if value < 0 else 0
            except (TypeError, ValueError):
                terminal_sign = 0
        previous_sign = 0
        for item in ordered:
            if item.get("integrity_status") != "pass":
                sign = 0
            else:
                try:
                    value = float(item.get("log_advantage", ""))
                    sign = 1 if value > 0 else -1 if value < 0 else 0
                except (TypeError, ValueError):
                    sign = 0
            item["terminal_sign_agreement"] = int(
                bool(sign) and bool(terminal_sign) and sign == terminal_sign
            )
            item["rank_reversal_from_previous"] = int(
                bool(sign) and bool(previous_sign) and sign != previous_sign
            )
            if sign:
                previous_sign = sign
    return rows


def _redact_car_actionability_summary_rows(
    rows: list[dict[str, object]],
    integrity_failures: list[str],
) -> list[dict[str, object]]:
    """Fail closed when any run-level integrity gate blocks the audit."""

    if not integrity_failures:
        return rows
    global_failure = "global_actionability_gate:" + "|".join(
        sorted(set(integrity_failures))
    )
    redacted: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        existing = str(row.get("integrity_failures", "")).strip(";")
        row.update(
            {
                "fallback_error": "",
                "candidate_error": "",
                "log_advantage": "",
                "relative_gain": "",
                "numeric_win": 0,
                "meaningful_win": 0,
                "catastrophic_loss": 0,
                "oracle_selected_arm": "",
                "oracle_gain": "",
                "terminal_sign_agreement": 0,
                "rank_reversal_from_previous": 0,
                "integrity_status": "fail",
                "integrity_failures": ";".join(
                    value for value in (existing, global_failure) if value
                ),
            }
        )
        redacted.append(row)
    return redacted


def _car_actionability_lane_semantic_failures(
    trace_rows: list[dict[str, object]],
    *,
    prefix: str,
) -> list[str]:
    """Validate one lane's v2 horizon set independently of paired outcomes."""

    failures: list[str] = []
    index_by_label = {
        **{
            label: str(index)
            for index, label in enumerate(CAR_ACTIONABILITY_HORIZON_LABELS)
        },
        "terminal": "3",
    }
    rows_by_label: dict[str, list[dict[str, object]]] = {}
    for row in trace_rows:
        label = str(row.get("horizon_label", ""))
        if label not in index_by_label:
            failures.append(f"{prefix}:unknown_horizon_label={label}")
            continue
        rows_by_label.setdefault(label, []).append(row)
        if str(row.get("horizon_index", "")) != index_by_label[label]:
            failures.append(f"{prefix}:{label}_horizon_index_mismatch")

    terminal_rows = rows_by_label.get("terminal", [])
    if len(terminal_rows) != 1:
        failures.append(f"{prefix}:terminal_count={len(terminal_rows)}")
        return failures
    terminal = terminal_rows[0]
    immutable_fields = (
        "checkpoint_fe",
        "actual_fe",
        "requested_fe",
        "candidate_action_applied",
        "plan_status",
        "configured_max_fes",
        "terminal_completion_tolerance_fe",
        "termination_reason",
        "terminal_fe_shortfall",
    )
    for label, label_rows in rows_by_label.items():
        for row in label_rows:
            for field in immutable_fields:
                if str(row.get(field, "")) != str(terminal.get(field, "")):
                    failures.append(f"{prefix}:{label}_{field}_mismatch")

    try:
        checkpoint_fe = int(str(terminal.get("checkpoint_fe", "")))
        intervention_fe = int(str(terminal.get("actual_fe", "")))
        requested_fe = int(str(terminal.get("requested_fe", "")))
        configured_max_fes = int(str(terminal.get("configured_max_fes", "")))
        terminal_tolerance = int(
            str(terminal.get("terminal_completion_tolerance_fe", ""))
        )
        terminal_shortfall = int(str(terminal.get("terminal_fe_shortfall", "")))
        terminal_target = int(str(terminal.get("target_fe", "")))
        terminal_observed = int(str(terminal.get("observed_fe", "")))
    except ValueError:
        failures.append(f"{prefix}:invalid_horizon_metadata")
        return failures
    if (
        checkpoint_fe < 0
        or intervention_fe < 0
        or requested_fe < 0
        or configured_max_fes <= 0
        or terminal_tolerance < 0
        or terminal_shortfall < 0
        or terminal_target < 0
        or terminal_observed < 0
    ):
        failures.append(f"{prefix}:invalid_horizon_metadata")
        return failures

    plan_status = str(terminal.get("plan_status", ""))
    if plan_status not in {"applied", "abstain", "not_applicable"}:
        failures.append(f"{prefix}:invalid_plan_status={plan_status}")
    if plan_status == "applied":
        if intervention_fe <= 0:
            failures.append(f"{prefix}:applied_intervention_fe_nonpositive")
        if requested_fe != intervention_fe:
            failures.append(f"{prefix}:applied_requested_actual_fe_mismatch")
    else:
        if intervention_fe != 0:
            failures.append(f"{prefix}:non_applied_intervention_fe_nonzero")
        if requested_fe != 0:
            failures.append(f"{prefix}:non_applied_requested_fe_nonzero")
        if str(terminal.get("candidate_action_applied", "")) != "0":
            failures.append(f"{prefix}:non_applied_candidate_action_applied")

    closure_target = checkpoint_fe + intervention_fe
    expected_terminal_target = max(
        closure_target,
        max(0, configured_max_fes - terminal_tolerance),
    )
    if terminal_target != expected_terminal_target:
        failures.append(f"{prefix}:terminal_target_mismatch")
    if plan_status == "applied" and terminal_target <= closure_target:
        failures.append(
            f"{prefix}:terminal_target_has_no_post_intervention_continuation"
        )
    if terminal_observed != terminal_target:
        failures.append(f"{prefix}:terminal_observed_fe_mismatch")
    if terminal_shortfall > terminal_tolerance:
        failures.append(f"{prefix}:terminal_shortfall_out_of_bounds")
    if configured_max_fes - terminal_shortfall < terminal_target:
        failures.append(f"{prefix}:terminal_endpoint_before_target")
    if terminal.get("termination_reason") != "population_complete_budget_endpoint":
        failures.append(
            f"{prefix}:terminal_termination_reason="
            f"{terminal.get('termination_reason', '')}"
        )
    if terminal.get("horizon_status") != "complete":
        failures.append(
            f"{prefix}:terminal_horizon_status="
            f"{terminal.get('horizon_status', '')}"
        )

    expected_targets: dict[str, int] = {}
    if plan_status == "applied":
        for index, (multiplier, label) in enumerate(
            zip(
                CAR_ACTIONABILITY_HORIZON_MULTIPLIERS,
                CAR_ACTIONABILITY_HORIZON_LABELS,
                strict=True,
            )
        ):
            target = checkpoint_fe + multiplier * intervention_fe
            if (index == 0 and target <= terminal_target) or target < terminal_target:
                expected_targets[label] = target

    for label in CAR_ACTIONABILITY_HORIZON_LABELS:
        label_rows = rows_by_label.get(label, [])
        count = len(label_rows)
        if label not in expected_targets:
            if count:
                failures.append(f"{prefix}:{label}_not_before_common_terminal")
            continue
        if count != 1:
            failures.append(f"{prefix}:{label}_count={count}")
            continue
        horizon = label_rows[0]
        expected_target = expected_targets[label]
        if str(horizon.get("target_fe", "")) != str(expected_target):
            failures.append(f"{prefix}:{label}_target_mismatch")
        if str(horizon.get("observed_fe", "")) != str(expected_target):
            failures.append(f"{prefix}:{label}_observed_fe_mismatch")
        if horizon.get("horizon_status") != "complete":
            failures.append(
                f"{prefix}:{label}_horizon_status="
                f"{horizon.get('horizon_status', '')}"
            )
    return failures


def _car_actionability_coverage_failures(
    trace_rows: list[dict[str, object]],
    *,
    problem_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    lanes: tuple[LaneConfig, ...],
) -> list[str]:
    """Require every pre-registered lane and reachable horizon exactly once."""

    indexed: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in trace_rows:
        key = (
            str(row.get("problem_id", "")),
            str(row.get("seed", "")),
            str(row.get("lane_id", "")),
        )
        indexed.setdefault(key, []).append(row)

    failures: list[str] = []
    audit_lanes = tuple(
        lane for lane in lanes if lane.car_actionability_arm != "off"
    )
    for problem_id in problem_ids:
        for seed in seeds:
            for lane in audit_lanes:
                prefix = f"{problem_id}/seed{seed}/{lane.lane_id}"
                failures.extend(
                    _car_actionability_lane_semantic_failures(
                        indexed.get((problem_id, str(seed), lane.lane_id), []),
                        prefix=prefix,
                    )
                )
    return failures


def _paired_runtime_utility_gate(
    paired_rows: list[dict[str, object]],
    *,
    negative_control_rows: list[dict[str, object]],
    integrity_failures: list[str] | None = None,
) -> dict[str, object]:
    case_rows = [row for row in paired_rows if row.get("problem_id") != "ALL"]
    aggregate = next(
        (row for row in paired_rows if row.get("problem_id") == "ALL"),
        None,
    )
    blockers = list(integrity_failures or [])
    integrity_status = "pass" if not blockers else "blocked"
    if aggregate is None:
        blockers.append("missing_paired_runtime_utility_summary")
        return {
            "status": "blocked",
            "blockers": blockers,
            "integrity_status": integrity_status,
            "integrity_failures": list(integrity_failures or []),
        }

    pair_count = int(aggregate["seed_count"])
    case_count = len(case_rows)
    mean_log_error_delta = float(aggregate["mean_log_error_delta"])
    mean_case_wins = int(aggregate["mean_win"])
    worst_seed_case_wins = int(aggregate["worst_seed_win"])
    meaningful_seed_wins = int(aggregate["meaningful_seed_wins"])
    catastrophic_losses = int(aggregate["catastrophic_losses"])
    negative_control_win_count = sum(
        int(row.get("shuffled_win_count", 0)) for row in negative_control_rows
    )
    negative_control_pair_count = sum(
        int(row.get("total_seeds", 0)) for row in negative_control_rows
    )
    negative_control_pass = (
        negative_control_pair_count > 0
        and negative_control_win_count <= negative_control_pair_count / 2
    )

    if case_count != 13 or pair_count != 65:
        blockers.append("expected_13_cases_65_pairs")
    if mean_log_error_delta >= 0.0:
        blockers.append("aggregate_mean_log_error_not_improved")
    if mean_case_wins < 7:
        blockers.append("mean_case_wins_below_7")
    if worst_seed_case_wins < 5:
        blockers.append("worst_seed_case_wins_below_5")
    if meaningful_seed_wins < 33:
        blockers.append("meaningful_seed_wins_below_33")
    if catastrophic_losses:
        blockers.append("catastrophic_paired_loss")
    if not negative_control_pass:
        blockers.append("negative_control_failed_or_missing")

    return {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "integrity_status": integrity_status,
        "integrity_failures": list(integrity_failures or []),
        "case_count": case_count,
        "pair_count": pair_count,
        "aggregate_mean_log_error_delta": mean_log_error_delta,
        "mean_case_wins": mean_case_wins,
        "worst_seed_case_wins": worst_seed_case_wins,
        "meaningful_seed_wins": meaningful_seed_wins,
        "catastrophic_losses": catastrophic_losses,
        "negative_control_pass": negative_control_pass,
        "negative_control_win_count": negative_control_win_count,
        "negative_control_pair_count": negative_control_pair_count,
    }


def _write_paired_runtime_utility_gate(
    output_dir: Path,
    gate: dict[str, object],
) -> None:
    (output_dir / "paired_runtime_utility_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    blockers = gate.get("blockers", [])
    lines = [
        "# Paired Runtime Utility Gate",
        "",
        f"Status: {gate.get('status', 'blocked')}",
        f"Integrity status: {gate.get('integrity_status', 'blocked')}",
        (
            "Integrity failures: "
            f"{','.join(str(value) for value in gate.get('integrity_failures', [])) or 'none'}"
        ),
        f"Cases / pairs: {gate.get('case_count', 0)} / {gate.get('pair_count', 0)}",
        f"Aggregate mean log-error delta: {gate.get('aggregate_mean_log_error_delta', 'missing')}",
        f"Mean-case wins: {gate.get('mean_case_wins', 0)}",
        f"Worst-seed case wins: {gate.get('worst_seed_case_wins', 0)}",
        f"Meaningful seed wins: {gate.get('meaningful_seed_wins', 0)}",
        f"Catastrophic paired losses: {gate.get('catastrophic_losses', 0)}",
        f"Negative-control pass: {int(bool(gate.get('negative_control_pass', False)))}",
        (
            "Negative-control wins / pairs: "
            f"{gate.get('negative_control_win_count', 0)} / "
            f"{gate.get('negative_control_pair_count', 0)}"
        ),
        f"Blockers: {','.join(str(value) for value in blockers) if blockers else 'none'}",
    ]
    (output_dir / "paired_runtime_utility_gate.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _policy_evidence_diagnosis_rows_for_problem(
    problem_id: str,
    utility_rows: list[dict[str, object]],
    negative_control: dict[str, object],
) -> list[dict[str, object]]:
    relation_rows = [
        row for row in utility_rows if row["lane_id"] == "relation_dispatch_rule"
    ]
    budget_violations = sum(
        1 for row in utility_rows if str(row["same_budget_violation"]) == "1"
    )
    relation_meaningful = sum(
        1 for row in relation_rows if row["utility_label"] == "meaningful_win"
    )
    relation_catastrophic = sum(
        1 for row in relation_rows if row["utility_label"] == "catastrophic_loss"
    )
    relation_gains = [
        float(row["relative_gain_vs_fallback"]) for row in relation_rows
    ]
    relation_positive = sum(1 for gain in relation_gains if gain > 0.0)
    relation_negative = sum(1 for gain in relation_gains if gain < 0.0)
    relation_mean_gain = _mean(relation_gains)
    fixed_coordinate_by_seed = {
        str(row["seed"]): float(row["final_error"])
        for row in utility_rows
        if row["lane_id"] == "fixed_coordinate"
    }
    relation_vs_fixed_coordinate_gains = [
        relative_gain(
            fixed_coordinate_by_seed[str(row["seed"])],
            float(row["final_error"]),
        )
        for row in relation_rows
        if str(row["seed"]) in fixed_coordinate_by_seed
    ]
    relation_beats_fixed_coordinate = sum(
        1 for gain in relation_vs_fixed_coordinate_gains if gain > 0.0
    )
    relation_loses_fixed_coordinate = sum(
        1 for gain in relation_vs_fixed_coordinate_gains if gain < 0.0
    )
    relation_vs_fixed_coordinate_mean_gain = _mean(relation_vs_fixed_coordinate_gains)
    relation_gating_pass = (
        bool(relation_vs_fixed_coordinate_gains)
        and relation_beats_fixed_coordinate > relation_loses_fixed_coordinate
        and relation_vs_fixed_coordinate_mean_gain > 0.0
    )
    fixed_coordinate_rows = [
        row for row in utility_rows if row["lane_id"] == "fixed_coordinate"
    ]
    fixed_coordinate_mean_gain = _mean(
        [float(row["relative_gain_vs_fallback"]) for row in fixed_coordinate_rows]
    )
    relation_directional_pass = (
        bool(relation_rows)
        and relation_positive > relation_negative
        and relation_mean_gain > 0.0
    )
    negative_control_pass = str(negative_control.get("negative_control_pass", "0")) == "1"
    blockers: list[str] = []
    if budget_violations:
        blockers.append("same_budget_violation")
    if relation_meaningful != len(relation_rows):
        blockers.append("relation_dispatch_not_meaningful_win")
    if relation_catastrophic:
        blockers.append("catastrophic_loss")
    if not negative_control_pass:
        blockers.append("negative_control_failed")
    if not relation_gating_pass:
        blockers.append("fixed_coordinate_baseline_not_beaten")
    pilot_utility_pass = (
        not budget_violations
        and relation_directional_pass
        and relation_catastrophic == 0
        and negative_control_pass
    )
    sota_allowed = not blockers
    rule_mix = _aggregate_lane_action_mix(utility_rows, "relation_dispatch_rule")
    shuffled_mix = _aggregate_lane_action_mix(
        utility_rows,
        "shuffled_relation_dispatch",
    )
    shuffled_fallbackized_isolate = min(
        rule_mix.get("isolate_conflicting_relation", 0),
        shuffled_mix.get("conservative_no_action", 0),
    )

    return [
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "same_budget_fe_status",
            "status": "blocked" if budget_violations else "pass",
            "observed_value": f"{budget_violations}/{len(utility_rows)}",
            "blocker_reason": "same_budget_violation" if budget_violations else "",
            "next_step": "fix_same_budget_accounting" if budget_violations else "continue",
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "relation_dispatch_utility",
            "status": (
                "pass" if relation_meaningful == len(relation_rows) else "blocked"
            ),
            "observed_value": f"{relation_meaningful}/{len(relation_rows)}",
            "blocker_reason": (
                "" if relation_meaningful == len(relation_rows)
                else "relation_dispatch_not_meaningful_win"
            ),
            "next_step": (
                "continue"
                if relation_meaningful == len(relation_rows)
                else "diagnose_policy_evidence_before_sota"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "relation_dispatch_directional_utility",
            "status": "pass" if relation_directional_pass else "blocked",
            "observed_value": (
                f"{relation_positive}/{len(relation_rows)};"
                f"mean_gain={relation_mean_gain:.6f}"
            ),
            "blocker_reason": ""
            if relation_directional_pass
            else "relation_dispatch_not_directionally_positive",
            "next_step": "continue"
            if relation_directional_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "pilot_utility_evidence",
            "status": "pass" if pilot_utility_pass else "blocked",
            "observed_value": (
                f"directional={relation_positive}/{len(relation_rows)};"
                f"mean_gain={relation_mean_gain:.6f};"
                f"negative_control={int(negative_control_pass)};"
                f"catastrophic={relation_catastrophic}/{len(relation_rows)}"
            ),
            "blocker_reason": ""
            if pilot_utility_pass
            else "pilot_utility_evidence_not_established",
            "next_step": "continue_to_multi_problem_protocol"
            if pilot_utility_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "relation_vs_fixed_coordinate_baseline",
            "status": "pass" if relation_gating_pass else "blocked",
            "observed_value": (
                f"win_count={relation_beats_fixed_coordinate}/"
                f"{len(relation_vs_fixed_coordinate_gains)};"
                "mean_gain_vs_fixed_coordinate="
                f"{relation_vs_fixed_coordinate_mean_gain:.6f};"
                "fixed_coordinate_mean_gain_vs_fallback="
                f"{fixed_coordinate_mean_gain:.6f}"
            ),
            "blocker_reason": ""
            if relation_gating_pass
            else "relation_gating_not_better_than_fixed_coordinate",
            "next_step": "continue"
            if relation_gating_pass
            else "diagnose_coordinate_gating_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "catastrophic_loss_gate",
            "status": "blocked" if relation_catastrophic else "pass",
            "observed_value": f"{relation_catastrophic}/{len(relation_rows)}",
            "blocker_reason": "catastrophic_loss" if relation_catastrophic else "",
            "next_step": (
                "diagnose_policy_evidence_before_sota"
                if relation_catastrophic
                else "continue"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "shuffled_negative_control",
            "status": "pass" if negative_control_pass else "blocked",
            "observed_value": str(negative_control.get("negative_control_pass", "")),
            "blocker_reason": "" if negative_control_pass else str(
                negative_control.get("diagnostic", "negative_control_failed")
            ),
            "next_step": (
                "continue"
                if negative_control_pass
                else "diagnose_policy_evidence_before_sota"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "negative_control_action_mix",
            "status": "pass" if negative_control_pass else "blocked",
            "observed_value": (
                "relation_dispatch_rule="
                f"{_format_action_counts(rule_mix)}|"
                "shuffled_relation_dispatch="
                f"{_format_action_counts(shuffled_mix)}"
            ),
            "blocker_reason": ""
            if negative_control_pass
            else (
                f"{negative_control.get('diagnostic', 'negative_control_failed')};"
                "rule_isolate_to_shuffled_fallback="
                f"{shuffled_fallbackized_isolate}"
            ),
            "next_step": (
                "continue"
                if negative_control_pass
                else "inspect_rule_vs_shuffled_action_mix"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "sota_escalation_allowed",
            "status": "pass" if sota_allowed else "blocked",
            "observed_value": str(int(sota_allowed)),
            "blocker_reason": ";".join(blockers),
            "next_step": "continue_to_sota_protocol"
            if sota_allowed
            else "diagnose_policy_evidence_before_sota",
        },
    ]


def _relation_policy_profile_row(
    problem_id: str,
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    trace_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
) -> dict[str, object]:
    relation_utility_rows = [
        row for row in utility_rows if row["lane_id"] == "relation_dispatch_rule"
    ]
    relation_decisions = [
        row
        for row in decision_rows
        if str(row.get("problem_id", "")) == problem_id
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
    ]
    relation_traces = [
        row
        for row in trace_rows
        if str(row.get("problem_id", "")) == problem_id
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
    ]
    relation_overlaps = [
        row
        for row in overlap_rows
        if str(row.get("problem_id", "")) == problem_id
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
    ]
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in relation_decisions:
        action_name = str(row.get("canonical_action_name", ""))
        if action_name:
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        trigger_reason = str(row.get("trigger_reason", ""))
        if trigger_reason:
            reason_counts[trigger_reason] = reason_counts.get(trigger_reason, 0) + 1
    active_decisions = [
        row
        for row in relation_decisions
        if str(row.get("canonical_action_name", "")) != "conservative_no_action"
    ]
    active_actions = sum(
        count
        for action_name, count in action_counts.items()
        if action_name != "conservative_no_action"
    )
    active_density = (
        active_actions / len(relation_decisions)
        if relation_decisions
        else float("nan")
    )
    downstream_consumed = sum(
        1 for row in relation_traces if str(row.get("downstream_consumed", "")) == "1"
    )
    optimizer_consumed = sum(
        1 for row in relation_traces if str(row.get("optimizer_consumed", "")) == "1"
    )
    profile_complete = bool(relation_decisions and relation_traces and relation_overlaps)
    utility_blocked = any(
        row["utility_label"] != "meaningful_win" for row in relation_utility_rows
    )
    return {
        "run_id": RUN_ID,
        "problem_id": problem_id,
        "diagnostic_key": "relation_policy_evidence_profile",
        "status": "pass" if profile_complete else "blocked",
        "observed_value": (
            f"relations={len(relation_decisions)};"
            f"active={active_actions};"
            f"active_density={_format_float(active_density)};"
            f"downstream={downstream_consumed}/{len(relation_traces)};"
            f"optimizer_consumed={optimizer_consumed}/{len(relation_traces)};"
            f"actions={_format_action_counts(action_counts)};"
            f"reasons={_format_action_counts(reason_counts)};"
            "mean_gain="
            f"{_format_float(_mean_numeric(relation_utility_rows, 'relative_gain_vs_fallback'))};"
            "mean_active_confidence="
            f"{_format_float(_mean_numeric(active_decisions, 'confidence'))};"
            "mean_fallback_margin="
            f"{_format_float(_mean_numeric(relation_overlaps, 'fallback_margin_proxy'))};"
            "mean_delta_ratio_gap="
            f"{_format_float(_mean_numeric(relation_overlaps, 'delta_ratio_gap'))};"
            "mean_rank_stability="
            f"{_format_float(_mean_numeric(relation_overlaps, 'rank_stability'))};"
            "mean_shared_var_support="
            f"{_format_float(_mean_numeric(relation_overlaps, 'shared_var_support_ratio'))}"
        ),
        "blocker_reason": "" if profile_complete else "relation_policy_profile_missing",
        "next_step": (
            "tune_policy_or_backend_effect_size"
            if profile_complete and utility_blocked
            else ("continue" if profile_complete else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_relation_policy_profile_row(
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
) -> dict[str, object]:
    relation_utility_rows = [
        row
        for row in utility_rows
        if row["lane_id"] == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row["problem_id"]))
    ]
    relation_decisions = [
        row
        for row in decision_rows
        if str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    ]
    relation_overlaps = [
        row
        for row in overlap_rows
        if str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    ]
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in relation_decisions:
        action_name = str(row.get("canonical_action_name", ""))
        if action_name:
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        trigger_reason = str(row.get("trigger_reason", ""))
        if trigger_reason:
            reason_counts[trigger_reason] = reason_counts.get(trigger_reason, 0) + 1
    active_decisions = [
        row
        for row in relation_decisions
        if str(row.get("canonical_action_name", "")) != "conservative_no_action"
    ]
    active_density = (
        len(active_decisions) / len(relation_decisions)
        if relation_decisions
        else float("nan")
    )
    utility_blocked = any(
        row["utility_label"] != "meaningful_win" for row in relation_utility_rows
    )
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_relation_policy_profile",
        "status": "pass" if relation_decisions else "blocked",
        "observed_value": (
            f"relations={len(relation_decisions)};"
            f"active={len(active_decisions)};"
            f"active_density={_format_float(active_density)};"
            f"actions={_format_action_counts(action_counts)};"
            f"reasons={_format_action_counts(reason_counts)};"
            "mean_gain="
            f"{_format_float(_mean_numeric(relation_utility_rows, 'relative_gain_vs_fallback'))};"
            "mean_active_confidence="
            f"{_format_float(_mean_numeric(active_decisions, 'confidence'))};"
            "mean_shared_var_support="
            f"{_format_float(_mean_numeric(relation_overlaps, 'shared_var_support_ratio'))}"
        ),
        "blocker_reason": "" if relation_decisions else "relation_policy_profile_missing",
        "next_step": (
            "diagnose_policy_evidence_before_sota"
            if relation_decisions and utility_blocked
            else ("continue" if relation_decisions else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_trigger_outcome_profile_row(
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> dict[str, object]:
    gain_by_case = {
        (str(row["problem_id"]), str(row["seed"])): float(
            row["relative_gain_vs_fallback"]
        )
        for row in utility_rows
        if row["lane_id"] == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row["problem_id"]))
    }
    counts: dict[str, dict[str, int]] = {}
    for row in decision_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        case_key = (str(row.get("problem_id", "")), str(row.get("seed", "")))
        if case_key not in gain_by_case:
            continue
        trigger_reason = str(row.get("trigger_reason", ""))
        if not trigger_reason:
            continue
        bucket = _gain_bucket(gain_by_case[case_key])
        counts.setdefault(trigger_reason, {"win": 0, "loss": 0, "tie": 0})[bucket] += 1
    observed_value = ";".join(
        (
            f"{trigger}=win:{bucket_counts['win']},"
            f"loss:{bucket_counts['loss']},"
            f"tie:{bucket_counts['tie']}"
        )
        for trigger, bucket_counts in sorted(counts.items())
    )
    losses = sum(bucket_counts["loss"] for bucket_counts in counts.values())
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_trigger_outcome_profile",
        "status": "blocked" if losses else ("pass" if counts else "blocked"),
        "observed_value": observed_value,
        "blocker_reason": (
            "relation_dispatch_lost_cases"
            if losses
            else ("" if counts else "relation_policy_profile_missing")
        ),
        "next_step": (
            "inspect_trigger_outcome_profile"
            if losses
            else ("continue" if counts else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_trigger_baseline_gap_profile_row(
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_case_lane = {
        (str(row["problem_id"]), str(row["seed"]), str(row["lane_id"])): float(
            row["final_error"]
        )
        for row in utility_rows
        if _is_overlap_applicable_problem_id(str(row["problem_id"]))
        and str(row["lane_id"])
        in {"relation_dispatch_rule", "fixed_repair", "fixed_coordinate"}
    }
    gaps: dict[str, dict[str, list[float]]] = {}
    for row in decision_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        problem_id = str(row.get("problem_id", ""))
        if not _is_overlap_applicable_problem_id(problem_id):
            continue
        seed = str(row.get("seed", ""))
        relation_error = by_case_lane.get((problem_id, seed, "relation_dispatch_rule"))
        repair_error = by_case_lane.get((problem_id, seed, "fixed_repair"))
        coordinate_error = by_case_lane.get((problem_id, seed, "fixed_coordinate"))
        trigger_reason = str(row.get("trigger_reason", ""))
        if (
            relation_error is None
            or repair_error is None
            or coordinate_error is None
            or not trigger_reason
        ):
            continue
        trigger_gaps = gaps.setdefault(
            trigger_reason,
            {"fixed_repair": [], "fixed_coordinate": []},
        )
        trigger_gaps["fixed_repair"].append(relative_gain(repair_error, relation_error))
        trigger_gaps["fixed_coordinate"].append(
            relative_gain(coordinate_error, relation_error)
        )
    observed_value = ";".join(
        (
            f"{trigger}=relations:{len(values['fixed_repair'])},"
            f"vs_fixed_repair_mean={_format_float(_mean(values['fixed_repair']))},"
            "vs_fixed_coordinate_mean="
            f"{_format_float(_mean(values['fixed_coordinate']))}"
        )
        for trigger, values in sorted(gaps.items())
    )
    has_negative_mean = any(
        _mean(values["fixed_repair"]) < 0.0
        or _mean(values["fixed_coordinate"]) < 0.0
        for values in gaps.values()
    )
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_trigger_baseline_gap_profile",
        "status": "blocked" if has_negative_mean else ("pass" if gaps else "blocked"),
        "observed_value": observed_value,
        "blocker_reason": (
            "trigger_baseline_gap_detected"
            if has_negative_mean
            else ("" if gaps else "relation_policy_profile_missing")
        ),
        "next_step": (
            "inspect_trigger_baseline_gap_profile"
            if has_negative_mean
            else ("continue" if gaps else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_action_baseline_gap_profile_row(
    utility_rows: list[dict[str, object]],
    action_trace_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_case_lane = {
        (str(row["problem_id"]), str(row["seed"]), str(row["lane_id"])): float(
            row["final_error"]
        )
        for row in utility_rows
        if _is_overlap_applicable_problem_id(str(row["problem_id"]))
        and str(row["lane_id"])
        in {"relation_dispatch_rule", "fixed_repair", "fixed_coordinate"}
    }
    gaps: dict[str, dict[str, list[float]]] = {}
    for row in action_trace_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        problem_id = str(row.get("problem_id", ""))
        if not _is_overlap_applicable_problem_id(problem_id):
            continue
        seed = str(row.get("seed", ""))
        relation_error = by_case_lane.get((problem_id, seed, "relation_dispatch_rule"))
        repair_error = by_case_lane.get((problem_id, seed, "fixed_repair"))
        coordinate_error = by_case_lane.get((problem_id, seed, "fixed_coordinate"))
        action_name = str(row.get("canonical_action_name", ""))
        if (
            relation_error is None
            or repair_error is None
            or coordinate_error is None
            or not action_name
        ):
            continue
        action_gaps = gaps.setdefault(
            action_name,
            {"fixed_repair": [], "fixed_coordinate": [], "value_delta": []},
        )
        action_gaps["fixed_repair"].append(relative_gain(repair_error, relation_error))
        action_gaps["fixed_coordinate"].append(
            relative_gain(coordinate_error, relation_error)
        )
        try:
            action_gaps["value_delta"].append(float(row["action_value_delta_norm"]))
        except (KeyError, TypeError, ValueError):
            pass
    observed_value = ";".join(
        (
            f"{action}=relations:{len(values['fixed_repair'])},"
            f"vs_fixed_repair_mean={_format_float(_mean(values['fixed_repair']))},"
            "vs_fixed_coordinate_mean="
            f"{_format_float(_mean(values['fixed_coordinate']))},"
            "mean_action_value_delta_norm="
            f"{_format_float(_mean(values['value_delta']))}"
        )
        for action, values in sorted(gaps.items())
    )
    has_negative_mean = any(
        _mean(values["fixed_repair"]) < 0.0
        or _mean(values["fixed_coordinate"]) < 0.0
        for values in gaps.values()
    )
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_action_baseline_gap_profile",
        "status": "blocked" if has_negative_mean else ("pass" if gaps else "blocked"),
        "observed_value": observed_value,
        "blocker_reason": (
            "action_baseline_gap_detected"
            if has_negative_mean
            else ("" if gaps else "relation_action_trace_missing")
        ),
        "next_step": (
            "inspect_action_baseline_gap_profile"
            if has_negative_mean
            else ("continue" if gaps else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_no_overlap_control_row(
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    def has_overlap_support(row: dict[str, object]) -> bool:
        saw_support_field = False
        for key in ("shared_var_count", "shared_vars_count", "overlap_strength"):
            if key not in row:
                continue
            saw_support_field = True
            raw_value = str(row.get(key, "")).strip()
            if not raw_value:
                continue
            try:
                if float(raw_value) > 0.0:
                    return True
            except ValueError:
                return True
        if "shared_vars" in row:
            saw_support_field = True
            if str(row.get("shared_vars", "")).strip():
                return True
        return not saw_support_field

    control_ids = sorted(
        {
            str(row["problem_id"])
            for row in utility_rows
            if not _is_overlap_applicable_problem_id(str(row["problem_id"]))
        }
    )
    if not control_ids:
        return None
    control_id_set = set(control_ids)
    control_utility_rows = [
        row for row in utility_rows if str(row["problem_id"]) in control_id_set
    ]
    control_overlap_rows = [
        row
        for row in overlap_rows
        if str(row.get("problem_id", "")) in control_id_set
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and has_overlap_support(row)
    ]
    active_decision_rows = [
        row
        for row in decision_rows
        if str(row.get("problem_id", "")) in control_id_set
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and str(row.get("canonical_action_name", "")) != "conservative_no_action"
    ]
    budget_violations = sum(
        1
        for row in control_utility_rows
        if str(row.get("same_budget_violation", "0")) == "1"
    )
    blockers: list[str] = []
    if control_overlap_rows:
        blockers.append("no_overlap_relation_rows_detected")
    if active_decision_rows:
        blockers.append("no_overlap_active_relation_actions_detected")
    if budget_violations:
        blockers.append("same_budget_violation")
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_no_overlap_control",
        "status": "blocked" if blockers else "pass",
        "observed_value": (
            f"controls={','.join(control_ids)};"
            f"relation_rows={len(control_overlap_rows)};"
            f"active_relation_actions={len(active_decision_rows)};"
            f"same_budget_violations={budget_violations}/{len(control_utility_rows)}"
        ),
        "blocker_reason": ";".join(blockers),
        "next_step": "inspect_no_overlap_controls" if blockers else "continue",
    }


def _multi_problem_action_mismatch_profile_row(
    mismatch_rows: list[dict[str, object]],
) -> dict[str, object]:
    relation_rows = [
        row
        for row in mismatch_rows
        if str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    ]
    final_counts: dict[str, int] = {}
    best_counts: dict[str, int] = {}
    abstain_counts: dict[str, int] = {}
    mismatch_count = 0
    margins: list[float] = []
    for row in relation_rows:
        final_action = str(row.get("final_action_name", ""))
        best_action = str(row.get("best_action_name", ""))
        if final_action:
            final_counts[final_action] = final_counts.get(final_action, 0) + 1
        if best_action:
            best_counts[best_action] = best_counts.get(best_action, 0) + 1
        if final_action and best_action and final_action != best_action:
            mismatch_count += 1
        abstain_reason = str(row.get("abstain_reason", ""))
        if abstain_reason:
            abstain_counts[abstain_reason] = abstain_counts.get(abstain_reason, 0) + 1
        try:
            margins.append(float(row.get("margin", "")))
        except (TypeError, ValueError):
            continue
    abstain_total = sum(abstain_counts.values())
    findings = mismatch_count or abstain_total
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_action_mismatch_profile",
        "status": "blocked" if findings or not relation_rows else "pass",
        "observed_value": (
            f"rows={len(relation_rows)};"
            f"final_best_mismatch={mismatch_count};"
            f"abstains={abstain_total};"
            f"mean_margin={_mean(margins):.6f};"
            f"final_actions={_format_inline_counts(final_counts)};"
            f"best_actions={_format_inline_counts(best_counts)};"
            f"abstain_reasons={_format_inline_counts(abstain_counts)}"
        ),
        "blocker_reason": (
            "relation_policy_profile_missing"
            if not relation_rows
            else ("action_mismatch_or_abstain_detected" if findings else "")
        ),
        "next_step": (
            "repair_relation_artifact_join"
            if not relation_rows
            else ("inspect_action_mismatch_audit" if findings else "continue")
        ),
    }


def _multi_problem_mismatch_baseline_gap_profile_row(
    utility_rows: list[dict[str, object]],
    mismatch_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_case_lane = {
        (str(row["problem_id"]), str(row["seed"]), str(row["lane_id"])): float(
            row["final_error"]
        )
        for row in utility_rows
        if _is_overlap_applicable_problem_id(str(row["problem_id"]))
        and str(row["lane_id"])
        in {"relation_dispatch_rule", "fixed_repair", "fixed_coordinate"}
    }
    gaps: dict[str, dict[str, list[float]]] = {}
    for row in mismatch_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        problem_id = str(row.get("problem_id", ""))
        if not _is_overlap_applicable_problem_id(problem_id):
            continue
        seed = str(row.get("seed", ""))
        relation_error = by_case_lane.get((problem_id, seed, "relation_dispatch_rule"))
        repair_error = by_case_lane.get((problem_id, seed, "fixed_repair"))
        coordinate_error = by_case_lane.get((problem_id, seed, "fixed_coordinate"))
        final_action = str(row.get("final_action_name", ""))
        best_action = str(row.get("best_action_name", ""))
        if (
            relation_error is None
            or repair_error is None
            or coordinate_error is None
            or not final_action
            or not best_action
        ):
            continue
        key = f"{final_action}->{best_action}"
        bucket = gaps.setdefault(key, {"fixed_repair": [], "fixed_coordinate": []})
        bucket["fixed_repair"].append(relative_gain(repair_error, relation_error))
        bucket["fixed_coordinate"].append(
            relative_gain(coordinate_error, relation_error)
        )
    observed_value = ";".join(
        (
            f"{key}=relations:{len(values['fixed_repair'])},"
            f"vs_fixed_repair_mean={_format_float(_mean(values['fixed_repair']))},"
            "vs_fixed_coordinate_mean="
            f"{_format_float(_mean(values['fixed_coordinate']))}"
        )
        for key, values in sorted(gaps.items())
    )
    has_negative_mean = any(
        _mean(values["fixed_repair"]) < 0.0
        or _mean(values["fixed_coordinate"]) < 0.0
        for values in gaps.values()
    )
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_mismatch_baseline_gap_profile",
        "status": "blocked" if has_negative_mean else ("pass" if gaps else "blocked"),
        "observed_value": observed_value,
        "blocker_reason": (
            "mismatch_baseline_gap_detected"
            if has_negative_mean
            else ("" if gaps else "action_mismatch_audit_missing")
        ),
        "next_step": (
            "inspect_action_mismatch_baseline_gaps"
            if has_negative_mean
            else ("continue" if gaps else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_relation_confidence_interval_row(
    utility_rows: list[dict[str, object]],
) -> dict[str, object]:
    indexed = {
        (str(row["problem_id"]), str(row["seed"]), str(row["lane_id"])): row
        for row in utility_rows
        if _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    }
    relation_rows = [
        row
        for row in utility_rows
        if str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    ]
    comparisons = [
        ("vs_fallback", "fallback"),
        ("vs_fixed_repair", "fixed_repair"),
        ("vs_fixed_coordinate", "fixed_coordinate"),
        ("vs_shuffled_relation_dispatch", "shuffled_relation_dispatch"),
    ]
    parts: list[str] = []
    missing = False
    for label, baseline_lane in comparisons:
        gains: list[float] = []
        for relation_row in relation_rows:
            case_key = (str(relation_row["problem_id"]), str(relation_row["seed"]))
            baseline_row = indexed.get((*case_key, baseline_lane))
            if baseline_row is None:
                continue
            if baseline_lane == "fallback":
                try:
                    gains.append(float(relation_row["relative_gain_vs_fallback"]))
                except (KeyError, TypeError, ValueError):
                    gains.append(
                        relative_gain(
                            float(baseline_row["final_error"]),
                            float(relation_row["final_error"]),
                        )
                    )
            else:
                gains.append(
                    relative_gain(
                        float(baseline_row["final_error"]),
                        float(relation_row["final_error"]),
                    )
                )
        if not gains:
            missing = True
        parts.append(f"{label}:{_confidence_interval_summary(gains)}")
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_relation_dispatch_confidence_interval",
        "status": "blocked" if missing else "pass",
        "observed_value": ";".join(parts),
        "blocker_reason": "confidence_interval_inputs_missing" if missing else "",
        "next_step": "repair_utility_audit_inputs" if missing else "continue",
    }


def _confidence_interval_summary(values: list[float]) -> str:
    if not values:
        return "n=0,mean=nan,ci95=[nan,nan]"
    mean = _mean(values)
    if len(values) == 1:
        half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        half_width = 1.96 * ((variance ** 0.5) / (len(values) ** 0.5))
    return (
        f"n={len(values)},mean={mean:.6f},"
        f"ci95=[{mean - half_width:.6f},{mean + half_width:.6f}]"
    )


def _multi_problem_diagnosis_rows(
    utility_rows: list[dict[str, object]],
    negative_control_rows: list[dict[str, object]],
    action_trace_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    problem_ids = sorted({str(row["problem_id"]) for row in utility_rows})
    seed_ids = sorted({str(row["seed"]) for row in utility_rows})
    formal_sota_protocol_pass = set(FORMAL_SOTA_PROBLEMS).issubset(problem_ids) and (
        len(seed_ids) >= FORMAL_SOTA_MIN_SEEDS
    )
    formal_sota_protocol_observed = (
        f"problems={len(set(problem_ids) & set(FORMAL_SOTA_PROBLEMS))}/"
        f"{len(FORMAL_SOTA_PROBLEMS)};seeds={len(seed_ids)}/{FORMAL_SOTA_MIN_SEEDS}"
    )
    if len(problem_ids) <= 1:
        return []
    overlap_applicable_ids = [
        problem_id
        for problem_id in problem_ids
        if _is_overlap_applicable_problem_id(problem_id)
    ]
    no_overlap_control_ids = [
        problem_id
        for problem_id in problem_ids
        if not _is_overlap_applicable_problem_id(problem_id)
    ]
    overlap_applicable_id_set = set(overlap_applicable_ids)
    scope_row = {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_claim_scope",
        "status": "pass" if overlap_applicable_ids else "blocked",
        "observed_value": (
            f"overlap_applicable={','.join(overlap_applicable_ids)};"
            f"no_overlap_controls={','.join(no_overlap_control_ids)}"
        ),
        "blocker_reason": "" if overlap_applicable_ids else "no_overlap_applicable_cases",
        "next_step": "continue" if overlap_applicable_ids else "add_overlap_applicable_cases",
    }
    utility_rows = [
        row
        for row in utility_rows
        if str(row["problem_id"]) in overlap_applicable_id_set
    ]
    negative_control_rows = [
        row
        for row in negative_control_rows
        if str(row["problem_id"]) in overlap_applicable_id_set
    ]
    if not utility_rows:
        return [scope_row]

    relation_rows = [
        row for row in utility_rows if row["lane_id"] == "relation_dispatch_rule"
    ]
    relation_gains = [
        float(row["relative_gain_vs_fallback"]) for row in relation_rows
    ]
    relation_lost_case_ids = [
        f"{row['problem_id']}_seed{row['seed']}"
        for row in relation_rows
        if float(row["relative_gain_vs_fallback"]) < 0.0
    ]
    relation_lost_rows = [
        row for row in relation_rows if float(row["relative_gain_vs_fallback"]) < 0.0
    ]
    relation_lost_action_mix: dict[str, int] = {}
    for row in relation_lost_rows:
        for action, count in _parse_action_mix(row.get("action_mix", "")).items():
            relation_lost_action_mix[action] = (
                relation_lost_action_mix.get(action, 0) + count
            )
    relation_lost_mean_gain = _mean(
        [float(row["relative_gain_vs_fallback"]) for row in relation_lost_rows]
    )
    relation_outcome_action_mix = {
        "wins": _action_mix_for_gain_bucket(relation_rows, "win"),
        "losses": _action_mix_for_gain_bucket(relation_rows, "loss"),
        "ties": _action_mix_for_gain_bucket(relation_rows, "tie"),
    }
    trace_rows = [] if action_trace_rows is None else action_trace_rows
    relation_action_value_delta_profile = {
        "wins": _action_value_delta_profile_for_gain_bucket(
            trace_rows,
            relation_rows,
            "win",
        ),
        "losses": _action_value_delta_profile_for_gain_bucket(
            trace_rows,
            relation_rows,
            "loss",
        ),
        "ties": _action_value_delta_profile_for_gain_bucket(
            trace_rows,
            relation_rows,
            "tie",
        ),
    }
    positive_cases = sum(1 for gain in relation_gains if gain > 0.0)
    mean_gain = _mean(relation_gains)
    loss_cases = len(relation_lost_rows)
    directional_pass = (
        bool(relation_rows) and positive_cases > loss_cases and mean_gain > 0.0
    )
    active_relation_rows = [
        row for row in relation_rows if _has_active_relation_action(row)
    ]
    active_relation_gains = [
        float(row["relative_gain_vs_fallback"]) for row in active_relation_rows
    ]
    active_relation_lost_case_ids = [
        f"{row['problem_id']}_seed{row['seed']}"
        for row in active_relation_rows
        if float(row["relative_gain_vs_fallback"]) < 0.0
    ]
    active_positive_cases = sum(1 for gain in active_relation_gains if gain > 0.0)
    active_mean_gain = _mean(active_relation_gains)
    active_loss_cases = len(active_relation_lost_case_ids)
    active_directional_pass = (
        bool(active_relation_rows)
        and active_positive_cases > active_loss_cases
        and active_mean_gain > 0.0
    )
    active_density_cases = [
        (
            f"{row['problem_id']}_seed{row['seed']}",
            _active_relation_density(row),
        )
        for row in relation_rows
    ]
    active_densities = [
        density for _case_id, density in active_density_cases if density == density
    ]
    low_active_density_case_ids = [
        case_id
        for case_id, density in active_density_cases
        if density == density and density <= LOW_ACTIVE_DENSITY_THRESHOLD
    ]
    low_active_density_cases = sum(
        1 for _case_id in low_active_density_case_ids
    )
    active_density_pass = low_active_density_cases == 0
    fixed_repair_by_case = {
        (str(row["problem_id"]), str(row["seed"])): float(row["final_error"])
        for row in utility_rows
        if row["lane_id"] == "fixed_repair"
    }
    fixed_repair_gain_cases = [
        (
            f"{row['problem_id']}_seed{row['seed']}",
            relative_gain(
                fixed_repair_by_case[(str(row["problem_id"]), str(row["seed"]))],
                float(row["final_error"]),
            ),
        )
        for row in relation_rows
        if (str(row["problem_id"]), str(row["seed"])) in fixed_repair_by_case
    ]
    fixed_repair_gains = [gain for _case_id, gain in fixed_repair_gain_cases]
    fixed_repair_lost_case_ids = [
        case_id for case_id, gain in fixed_repair_gain_cases if gain <= 0.0
    ]
    fixed_repair_win_count = sum(1 for gain in fixed_repair_gains if gain > 0.0)
    fixed_repair_mean_gain = _mean(fixed_repair_gains)
    fixed_repair_material_labels = [
        classify_utility(
            fixed_repair_by_case[(str(row["problem_id"]), str(row["seed"]))],
            float(row["final_error"]),
        )
        for row in relation_rows
        if (str(row["problem_id"]), str(row["seed"])) in fixed_repair_by_case
    ]
    fixed_repair_material_wins = fixed_repair_material_labels.count("meaningful_win")
    fixed_repair_material_losses = fixed_repair_material_labels.count(
        "catastrophic_loss"
    )
    fixed_repair_material_ties = (
        len(fixed_repair_material_labels)
        - fixed_repair_material_wins
        - fixed_repair_material_losses
    )
    fixed_repair_pass = (
        bool(fixed_repair_gains)
        and fixed_repair_win_count > len(fixed_repair_lost_case_ids)
        and fixed_repair_mean_gain > 0.0
    )
    fixed_coordinate_by_case = {
        (str(row["problem_id"]), str(row["seed"])): float(row["final_error"])
        for row in utility_rows
        if row["lane_id"] == "fixed_coordinate"
    }
    fixed_coordinate_gain_cases = [
        (
            f"{row['problem_id']}_seed{row['seed']}",
            relative_gain(
                fixed_coordinate_by_case[(str(row["problem_id"]), str(row["seed"]))],
                float(row["final_error"]),
            ),
        )
        for row in relation_rows
        if (str(row["problem_id"]), str(row["seed"])) in fixed_coordinate_by_case
    ]
    fixed_coordinate_gains = [gain for _case_id, gain in fixed_coordinate_gain_cases]
    fixed_coordinate_lost_case_ids = [
        case_id for case_id, gain in fixed_coordinate_gain_cases if gain <= 0.0
    ]
    fixed_coordinate_win_count = sum(1 for gain in fixed_coordinate_gains if gain > 0.0)
    fixed_coordinate_mean_gain = _mean(fixed_coordinate_gains)
    fixed_coordinate_pass = (
        bool(fixed_coordinate_gains)
        and fixed_coordinate_win_count > len(fixed_coordinate_lost_case_ids)
        and fixed_coordinate_mean_gain > 0.0
    )
    active_rows = [row for row in utility_rows if _expects_backend_semantics(row)]
    backend_semantics_changed = sum(
        1 for row in active_rows if str(row["backend_semantics_changed"]) == "1"
    )
    backend_semantics_pass = (
        bool(active_rows) and backend_semantics_changed == len(active_rows)
    )
    catastrophic = sum(
        1 for row in relation_rows if row["utility_label"] == "catastrophic_loss"
    )
    relation_meaningful = sum(
        1 for row in relation_rows if row["utility_label"] == "meaningful_win"
    )
    relation_material_losses = sum(
        1 for row in relation_rows if row["utility_label"] == "catastrophic_loss"
    )
    relation_material_ties = (
        len(relation_rows) - relation_meaningful - relation_material_losses
    )
    relation_materiality_pass = (
        bool(relation_rows)
        and relation_material_losses == 0
        and mean_gain >= MEANINGFUL_GAIN_THRESHOLD
    )
    relation_materiality_blocker = (
        ""
        if relation_materiality_pass
        else (
            "relation_dispatch_material_loss_detected"
            if relation_material_losses
            else "relation_dispatch_effect_size_below_threshold"
        )
    )
    budget_violations = sum(
        1 for row in utility_rows if str(row["same_budget_violation"]) == "1"
    )
    negative_failures = sum(
        1
        for row in negative_control_rows
        if str(row.get("negative_control_pass", "0")) != "1"
    )
    negative_failed_problem_ids = sorted(
        str(row["problem_id"])
        for row in negative_control_rows
        if str(row.get("negative_control_pass", "0")) != "1"
    )
    negative_pass_count = len(negative_control_rows) - negative_failures
    shuffled_win_count = sum(
        int(row.get("shuffled_win_count", 0)) for row in negative_control_rows
    )
    negative_total_seeds = sum(
        int(row.get("total_seeds", 0)) for row in negative_control_rows
    )
    negative_control_pass = (
        bool(negative_control_rows) and negative_failures == 0
    )
    rule_mix = _aggregate_lane_action_mix(utility_rows, "relation_dispatch_rule")
    shuffled_mix = _aggregate_lane_action_mix(
        utility_rows,
        "shuffled_relation_dispatch",
    )
    shuffled_repair_to_isolate = min(
        rule_mix.get("repair_shared_variable_binding", 0),
        shuffled_mix.get("isolate_conflicting_relation", 0),
    )
    shuffled_isolate_to_fallback = min(
        rule_mix.get("isolate_conflicting_relation", 0),
        shuffled_mix.get("conservative_no_action", 0),
    )
    blockers: list[str] = []
    if budget_violations:
        blockers.append("same_budget_violation")
    if not directional_pass:
        blockers.append("multi_problem_not_directionally_positive")
    if not relation_materiality_pass:
        blockers.append(relation_materiality_blocker)
    if catastrophic:
        blockers.append("catastrophic_loss")
    if negative_failures:
        blockers.append("negative_control_failed")
    if not fixed_repair_pass:
        blockers.append("fixed_repair_baseline_not_beaten")
    if not fixed_coordinate_pass:
        blockers.append("fixed_coordinate_baseline_not_beaten")
    if not backend_semantics_pass:
        blockers.append("backend_semantics_audit_failed")
    if not formal_sota_protocol_pass:
        blockers.append("formal_sota_protocol_incomplete")
    pilot_utility_pass = (
        not budget_violations
        and directional_pass
        and catastrophic == 0
        and negative_control_pass
    )
    sota_allowed = not blockers
    if sota_allowed:
        claim_tier = "sota_level_overlap_aware_cc_backend_optimization"
        claim_tier_blocker = ""
        claim_tier_next_step = "freeze_policy_and_run_final_protocol"
    elif pilot_utility_pass:
        claim_tier = (
            "runtime_evidence_driven_relation_dispatch_with_positive_utility_evidence"
        )
        claim_tier_blocker = "sota_gate_blocked"
        claim_tier_next_step = "report_positive_utility_or_continue_policy_diagnosis"
    else:
        claim_tier = "auditable_runtime_dispatch_framework"
        claim_tier_blocker = "pilot_utility_gate_blocked"
        claim_tier_next_step = "diagnose_policy_evidence_before_utility_claim"

    return [
        scope_row,
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_same_budget_fe_status",
            "status": "blocked" if budget_violations else "pass",
            "observed_value": f"{budget_violations}/{len(utility_rows)}",
            "blocker_reason": "same_budget_violation" if budget_violations else "",
            "next_step": "fix_same_budget_accounting" if budget_violations else "continue",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_dispatch_mean_gain",
            "status": "pass" if directional_pass else "blocked",
            "observed_value": (
                f"positive_cases={positive_cases}/{len(relation_rows)};"
                f"mean_gain={mean_gain:.6f};"
                f"lost_case_ids={','.join(relation_lost_case_ids)}"
            ),
            "blocker_reason": ""
            if directional_pass
            else "multi_problem_not_directionally_positive",
            "next_step": "continue" if directional_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_dispatch_materiality",
            "status": "pass" if relation_materiality_pass else "blocked",
            "observed_value": (
                f"material_wins={relation_meaningful}/{len(relation_rows)};"
                f"material_losses={relation_material_losses}/{len(relation_rows)};"
                f"ties={relation_material_ties}/{len(relation_rows)};"
                f"mean_gain={mean_gain:.6f};"
                f"threshold={MEANINGFUL_GAIN_THRESHOLD:.6f}"
            ),
            "blocker_reason": (
                "" if relation_materiality_pass else relation_materiality_blocker
            ),
            "next_step": (
                "continue"
                if relation_materiality_pass
                else "diagnose_policy_evidence_before_sota"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_lost_case_action_mix",
            "status": "blocked" if relation_lost_rows else "pass",
            "observed_value": (
                f"lost_cases={len(relation_lost_rows)};"
                f"mean_lost_gain={relation_lost_mean_gain:.6f};"
                f"actions={_format_action_counts(relation_lost_action_mix)}"
            ),
            "blocker_reason": (
                "relation_dispatch_lost_cases" if relation_lost_rows else ""
            ),
            "next_step": (
                "inspect_lost_case_action_mix" if relation_lost_rows else "continue"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_action_outcome_profile",
            "status": "blocked" if relation_lost_rows else "pass",
            "observed_value": (
                f"wins={_format_action_counts(relation_outcome_action_mix['wins'])}|"
                f"losses={_format_action_counts(relation_outcome_action_mix['losses'])}|"
                f"ties={_format_action_counts(relation_outcome_action_mix['ties'])}"
            ),
            "blocker_reason": (
                "relation_dispatch_lost_cases" if relation_lost_rows else ""
            ),
            "next_step": (
                "inspect_action_outcome_profile" if relation_lost_rows else "continue"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_action_value_delta_profile",
            "status": (
                "blocked"
                if relation_lost_rows and action_trace_rows is not None
                else "pass"
            ),
            "observed_value": (
                f"wins={relation_action_value_delta_profile['wins']}|"
                f"losses={relation_action_value_delta_profile['losses']}|"
                f"ties={relation_action_value_delta_profile['ties']}"
            ),
            "blocker_reason": (
                "relation_dispatch_lost_cases"
                if relation_lost_rows and action_trace_rows is not None
                else ""
            ),
            "next_step": (
                "inspect_action_value_delta_profile"
                if relation_lost_rows and action_trace_rows is not None
                else "continue"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_active_relation_dispatch_mean_gain",
            "status": "pass" if active_directional_pass else "blocked",
            "observed_value": (
                f"active_cases={len(active_relation_rows)};"
                f"positive_cases={active_positive_cases}/{len(active_relation_rows)};"
                f"mean_gain={active_mean_gain:.6f};"
                f"lost_case_ids={','.join(active_relation_lost_case_ids)}"
            ),
            "blocker_reason": ""
            if active_directional_pass
            else "active_relation_dispatch_not_directionally_positive",
            "next_step": "continue" if active_directional_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_pilot_utility_evidence",
            "status": "pass" if pilot_utility_pass else "blocked",
            "observed_value": (
                f"directional={positive_cases}/{len(relation_rows)};"
                f"mean_gain={mean_gain:.6f};"
                f"negative_control={negative_pass_count}/{len(negative_control_rows)};"
                f"catastrophic={catastrophic}/{len(relation_rows)}"
            ),
            "blocker_reason": ""
            if pilot_utility_pass
            else "multi_problem_pilot_utility_evidence_not_established",
            "next_step": "continue_to_sota_protocol"
            if pilot_utility_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_active_density_profile",
            "status": "pass" if active_density_pass else "blocked",
            "observed_value": (
                f"mean={_mean(active_densities):.6f};"
                f"min={min(active_densities) if active_densities else float('nan'):.6f};"
                f"low_density_cases={low_active_density_cases}/{len(active_densities)};"
                f"threshold={LOW_ACTIVE_DENSITY_THRESHOLD:.6f};"
                f"low_density_case_ids={','.join(low_active_density_case_ids)}"
            ),
            "blocker_reason": ""
            if active_density_pass
            else "low_relation_action_density_detected",
            "next_step": "continue"
            if active_density_pass
            else "inspect_low_active_density_problem_cases",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_dispatch_win_count",
            "status": "pass" if directional_pass else "blocked",
            "observed_value": (
                f"win_count={positive_cases}/{len(relation_rows)};"
                f"loss_count={loss_cases}/{len(relation_rows)}"
            ),
            "blocker_reason": ""
            if directional_pass
            else "multi_problem_not_directionally_positive",
            "next_step": "continue" if directional_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_fixed_repair_baseline",
            "status": "pass" if fixed_repair_pass else "blocked",
            "observed_value": (
                f"win_count={fixed_repair_win_count}/{len(fixed_repair_gains)};"
                f"mean_gain={fixed_repair_mean_gain:.6f};"
                f"lost_case_ids={','.join(fixed_repair_lost_case_ids)}"
            ),
            "blocker_reason": ""
            if fixed_repair_pass
            else "fixed_repair_baseline_not_beaten",
            "next_step": "continue" if fixed_repair_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_fixed_repair_materiality",
            "status": (
                "pass"
                if fixed_repair_material_labels
                and fixed_repair_material_losses == 0
                else "blocked"
            ),
            "observed_value": (
                f"material_wins={fixed_repair_material_wins}/"
                f"{len(fixed_repair_material_labels)};"
                f"material_losses={fixed_repair_material_losses}/"
                f"{len(fixed_repair_material_labels)};"
                f"ties={fixed_repair_material_ties}/"
                f"{len(fixed_repair_material_labels)}"
            ),
            "blocker_reason": (
                ""
                if fixed_repair_material_labels
                and fixed_repair_material_losses == 0
                else "fixed_repair_material_loss_detected"
            ),
            "next_step": (
                "continue"
                if fixed_repair_material_labels
                and fixed_repair_material_losses == 0
                else "diagnose_policy_evidence_before_sota"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_vs_fixed_coordinate_baseline",
            "status": "pass" if fixed_coordinate_pass else "blocked",
            "observed_value": (
                f"win_count={fixed_coordinate_win_count}/{len(fixed_coordinate_gains)};"
                f"mean_gain={fixed_coordinate_mean_gain:.6f};"
                f"lost_case_ids={','.join(fixed_coordinate_lost_case_ids)}"
            ),
            "blocker_reason": ""
            if fixed_coordinate_pass
            else "relation_gating_not_better_than_fixed_coordinate",
            "next_step": "continue" if fixed_coordinate_pass else "diagnose_coordinate_gating_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_backend_semantics_audit",
            "status": "pass" if backend_semantics_pass else "blocked",
            "observed_value": (
                f"changed={backend_semantics_changed}/{len(active_rows)}"
            ),
            "blocker_reason": ""
            if backend_semantics_pass
            else "backend_semantics_audit_failed",
            "next_step": "continue"
            if backend_semantics_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_negative_control",
            "status": "pass" if negative_control_pass else "blocked",
            "observed_value": (
                f"pass={negative_pass_count}/{len(negative_control_rows)};"
                f"shuffled_win_count={shuffled_win_count}/{negative_total_seeds};"
                f"failed_problem_ids={','.join(negative_failed_problem_ids)}"
            ),
            "blocker_reason": ""
            if negative_control_pass
            else "negative_control_failed",
            "next_step": "continue"
            if negative_control_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_negative_control_action_mix",
            "status": "pass" if negative_control_pass else "blocked",
            "observed_value": (
                "relation_dispatch_rule="
                f"{_format_action_counts(rule_mix)}|"
                "shuffled_relation_dispatch="
                f"{_format_action_counts(shuffled_mix)}"
            ),
            "blocker_reason": ""
            if negative_control_pass
            else (
                "negative_control_failed;"
                f"failed_problem_ids={','.join(negative_failed_problem_ids)};"
                "rule_repair_to_shuffled_isolate="
                f"{shuffled_repair_to_isolate};"
                "rule_isolate_to_shuffled_fallback="
                f"{shuffled_isolate_to_fallback}"
            ),
            "next_step": (
                "continue"
                if negative_control_pass
                else "inspect_rule_vs_shuffled_action_mix"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_catastrophic_loss_gate",
            "status": "blocked" if catastrophic else "pass",
            "observed_value": f"{catastrophic}/{len(relation_rows)}",
            "blocker_reason": "catastrophic_loss" if catastrophic else "",
            "next_step": "diagnose_policy_evidence_before_sota"
            if catastrophic
            else "continue",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_formal_sota_protocol_scope",
            "status": "pass" if formal_sota_protocol_pass else "blocked",
            "observed_value": formal_sota_protocol_observed,
            "blocker_reason": ""
            if formal_sota_protocol_pass
            else "formal_sota_protocol_incomplete",
            "next_step": "continue_to_sota_protocol"
            if formal_sota_protocol_pass
            else "run_full_aob_24_problem_25_seed_protocol_before_sota_claim",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_sota_escalation_allowed",
            "status": "pass" if sota_allowed else "blocked",
            "observed_value": str(int(sota_allowed)),
            "blocker_reason": ";".join(blockers),
            "next_step": "continue_to_sota_protocol"
            if sota_allowed
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_claim_tier_recommendation",
            "status": "pass",
            "observed_value": claim_tier,
            "blocker_reason": claim_tier_blocker,
            "next_step": claim_tier_next_step,
        },
    ]


def _policy_evidence_diagnosis_rows(
    records: list[dict[str, object]],
    utility_rows: list[dict[str, object]],
    negative_control_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    negative_by_problem = {
        str(row["problem_id"]): row for row in negative_control_rows
    }
    decision_rows = _action_decision_rows(records)
    mismatch_rows = _action_mismatch_rows(records)
    trace_rows = _action_trace_rows(records)
    overlap_rows = _overlap_relation_rows(records)
    rows: list[dict[str, object]] = []
    for problem_id in sorted({str(row["problem_id"]) for row in utility_rows}):
        problem_utility_rows = [
            row for row in utility_rows if str(row["problem_id"]) == problem_id
        ]
        rows.extend(
            _policy_evidence_diagnosis_rows_for_problem(
                problem_id,
                problem_utility_rows,
                negative_by_problem.get(problem_id, {}),
            )
        )
        rows.append(
            _relation_policy_profile_row(
                problem_id,
                problem_utility_rows,
                decision_rows,
                trace_rows,
                overlap_rows,
            )
        )
    rows.extend(
        _multi_problem_diagnosis_rows(
            utility_rows,
            negative_control_rows,
            trace_rows,
        )
    )
    if len({str(row["problem_id"]) for row in utility_rows}) > 1:
        no_overlap_row = _multi_problem_no_overlap_control_row(
            utility_rows,
            decision_rows,
            overlap_rows,
        )
        if no_overlap_row is not None:
            rows.append(no_overlap_row)
        rows.append(
            _multi_problem_relation_policy_profile_row(
                utility_rows,
                decision_rows,
                overlap_rows,
            )
        )
        rows.append(
            _multi_problem_trigger_outcome_profile_row(
                utility_rows,
                decision_rows,
            )
        )
        rows.append(
            _multi_problem_trigger_baseline_gap_profile_row(
                utility_rows,
                decision_rows,
            )
        )
        rows.append(
            _multi_problem_action_baseline_gap_profile_row(
                utility_rows,
                trace_rows,
            )
        )
        rows.append(_multi_problem_action_mismatch_profile_row(mismatch_rows))
        rows.append(
            _multi_problem_mismatch_baseline_gap_profile_row(
                utility_rows,
                mismatch_rows,
            )
        )
        rows.append(_multi_problem_relation_confidence_interval_row(utility_rows))
    return rows


def _diagnostic_observed_value(
    diagnosis_rows: list[dict[str, object]],
    diagnostic_key: str,
) -> str:
    for row in diagnosis_rows:
        if str(row["diagnostic_key"]) == diagnostic_key:
            return str(row["observed_value"])
    return ""


def _sota_claim_allowed(diagnosis_rows: list[dict[str, object]]) -> str:
    multi_value = _diagnostic_observed_value(
        diagnosis_rows,
        "multi_problem_sota_escalation_allowed",
    )
    if multi_value:
        return multi_value
    values = [
        str(row["observed_value"])
        for row in diagnosis_rows
        if str(row["diagnostic_key"]) == "sota_escalation_allowed"
    ]
    return "1" if values and all(value == "1" for value in values) else "0"


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ARAC_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _dependency_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _blas_summary() -> str:
    try:
        import numpy as np

        config = getattr(np.__config__, "CONFIG", {})
        blas = config.get("Build Dependencies", {}).get("blas", {})
        if blas:
            return json.dumps(
                {
                    key: blas.get(key)
                    for key in ("name", "version", "openblas configuration")
                    if blas.get(key) is not None
                },
                sort_keys=True,
            )
    except (AttributeError, ImportError, TypeError):
        pass
    return "unknown"


def _thread_environment() -> str:
    names = (
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    return ";".join(f"{name}={os.environ.get(name, 'unset')}" for name in names)


def _config_fingerprint(
    seeds: tuple[int, ...],
    problem_ids: tuple[str, ...],
    jobs: int,
    max_fes: int,
    budget_accounting: str,
    cmaes_restart: bool,
    mmes_restart: bool,
    lanes: tuple[LaneConfig, ...] = LANES,
    lane_profile: str = "runtime_smoke",
    aob_data_root: Path | str = DEFAULT_AOB_DATA_ROOT,
    vendor_paths: HccVendorPaths = HCC_VENDOR_PATHS,
    search_state_backend: str = "phase_i_mmes",
) -> str:
    mmes_path = vendor_paths.hcc_root / "NDAs" / "MMES" / "mmes.py"
    mmes_state_path = vendor_paths.hcc_root / "NDAs" / "MMES" / "state.py"
    cmaes_path = vendor_paths.hcc_root / "OPT" / "CMAES" / "cmaes.py"
    payload = {
        "budget_accounting": budget_accounting,
        "cmaes_restart": bool(cmaes_restart),
        "jobs": max(1, int(jobs)),
        "lane_profile": lane_profile,
        "lanes": [
            {
                "lane_id": lane.lane_id,
                "runner_action_name": lane.runner_action_name,
                "relation_policy_mode": lane.relation_policy_mode,
                "car_candidate_mode": lane.car_candidate_mode,
                "car_actionability_arm": lane.car_actionability_arm,
                "precision_causal_arm": lane.precision_causal_arm,
            }
            for lane in lanes
        ],
        "car_actionability_protocol": {
            "version": CAR_ACTIONABILITY_PROTOCOL_VERSION,
            "action_semantics": "one_shot_writeback_then_canonical_continuation",
            "horizons": ["closure_1", "budget_3x", "budget_9x", "terminal"],
            "terminal_semantics": "common_max_of_intervention_closure_and_cap_minus_tolerance_prefix_with_post_closure_gate",
        },
        "precision_causal_protocol": {
            "version": PRECISION_CAUSAL_PROTOCOL_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "randomization_salt": PRECISION_CAUSAL_RANDOMIZATION_SALT,
            "action_semantics": "one_v38_precision_sigma_group_block_then_v37_continuation",
        },
        "max_fes": int(max_fes),
        "mmes_restart": bool(mmes_restart),
        "problem_ids": list(problem_ids),
        "search_state_backend": search_state_backend,
        "seeds": list(seeds),
        "aob_data_root": str(Path(aob_data_root).resolve()),
        "hcc_vendor_root": str(vendor_paths.vendor_root),
        "hcc_aob_root": str(vendor_paths.aob_root),
        "hcc_source_root": str(vendor_paths.hcc_root),
        "hcc_smoke_runner": str(vendor_paths.runner),
        "hcc_smoke_runner_sha256": _sha256_file(vendor_paths.runner),
        "mmes_optimizer_sha256": _sha256_file(mmes_path),
        "mmes_state_sha256": _sha256_file(mmes_state_path),
        "cmaes_optimizer_sha256": _sha256_file(cmaes_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_manifest(
    output_dir: Path,
    seeds: tuple[int, ...],
    problem_ids: tuple[str, ...],
    diagnosis_rows: list[dict[str, object]],
    ledger_rows: list[dict[str, object]],
    jobs: int = 1,
    max_fes: int = MAX_FES,
    budget_accounting: str = "strict",
    cmaes_restart: bool = True,
    mmes_restart: bool = True,
    lanes: tuple[LaneConfig, ...] = LANES,
    lane_profile: str = "runtime_smoke",
    aob_data_root: Path | str = DEFAULT_AOB_DATA_ROOT,
    aob_input_rows: list[dict[str, object]] | None = None,
    python_executable: str = sys.executable,
    vendor_paths: HccVendorPaths = HCC_VENDOR_PATHS,
    search_state_backend: str = "phase_i_mmes",
    runtime_environment: dict[str, str] | None = None,
) -> None:
    aob_input_rows = [] if aob_input_rows is None else aob_input_rows
    mmes_path = vendor_paths.hcc_root / "NDAs" / "MMES" / "mmes.py"
    mmes_state_path = vendor_paths.hcc_root / "NDAs" / "MMES" / "state.py"
    cmaes_path = vendor_paths.hcc_root / "OPT" / "CMAES" / "cmaes.py"
    overlap_scope_same_budget_status = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_same_budget_fe_status",
        )
        or _diagnostic_observed_value(diagnosis_rows, "same_budget_fe_status")
        or "not_applicable"
    )
    same_budget_violations = sum(
        str(row.get("same_budget_violation")) != "0" for row in ledger_rows
    )
    same_budget_status = f"{same_budget_violations}/{len(ledger_rows)}"
    multi_problem_pilot = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_pilot_utility_evidence",
        )
        or "not_applicable"
    )
    multi_problem_claim_scope = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_claim_scope",
        )
        or "not_applicable"
    )
    multi_problem_active_density = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_active_density_profile",
        )
        or "not_applicable"
    )
    multi_problem_relation_materiality = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_relation_dispatch_materiality",
        )
        or "not_applicable"
    )
    multi_problem_fixed_repair = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_fixed_repair_baseline",
        )
        or "not_applicable"
    )
    multi_problem_fixed_repair_materiality = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_fixed_repair_materiality",
        )
        or "not_applicable"
    )
    multi_problem_fixed_coordinate = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_relation_vs_fixed_coordinate_baseline",
        )
        or "not_applicable"
    )
    multi_problem_relation_policy_profile = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_relation_policy_profile",
        )
        or "not_applicable"
    )
    multi_problem_claim_tier = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_claim_tier_recommendation",
        )
        or "not_applicable"
    )
    artifacts = [
        "our_result_by_case.csv",
        "same_budget_ledger.csv",
        "backend_semantics_diff.csv",
        "action_execution_plan.csv",
        "action_trace.csv",
        "trajectory_guard_summary.csv",
        "pre_hold_evidence.csv",
        "action_decision.csv",
        "action_mismatch_audit.csv",
        "overlap_relations.csv",
        "relation_join_audit.csv",
        "action_utility_audit.csv",
        "negative_control_comparison.csv",
        "policy_evidence_diagnosis.csv",
        "anti_leakage_audit.csv",
        "claim_gate.csv",
        "claim_evidence_table.md",
        "aob_input_manifest.csv",
    ]
    if runtime_environment is not None:
        artifacts.append("runtime_environment.json")
    if lane_profile == "paired_v33_v36_runtime_utility":
        artifacts.extend(
            [
                "paired_runtime_utility_summary.csv",
                "paired_runtime_utility_gate.json",
                "paired_runtime_utility_gate.md",
            ]
        )
    if any(
        lane.runner_action_name in CAR_W_ACTION_NAMES
        for lane in lanes
    ):
        artifacts.extend(
            [
                "car_dispatch_boundary_audit.csv",
                "car_probe_trace.csv",
                "car_state_ledger.csv",
                "car_branch_manifest.csv",
            ]
        )
    if any(lane.car_actionability_arm != "off" for lane in lanes):
        artifacts.extend(
            [
                "_hcc_smoke/**/car_actionability_provenance.json",
                "car_actionability_trace.csv",
                "car_actionability_summary.csv",
                "car_actionability_gate.json",
            ]
        )
    if any(lane.precision_causal_arm != "off" for lane in lanes):
        artifacts.extend(
            [
                "_hcc_smoke/**/precision_causal_provenance.json",
                "causal_decision_features.csv",
                "causal_decision_audit.csv",
                "causal_branch_manifest.csv",
                "causal_outcomes.csv",
                "randomized_log.csv",
                "causal_randomization_schedule.json",
                "feature_manifest.json",
                "causal_logging_manifest.json",
            ]
        )
    manifest = "\n".join(
        [
            "# exp_003_hcc_runtime_consumer_smoke Run Manifest",
            "",
            f"Date: {date.today().isoformat()}",
            "Executor: Codex",
            "",
            "Evidence posture: runtime dispatch + utility evidence",
            f"SOTA claim allowed: {_sota_claim_allowed(diagnosis_rows)}",
            "",
            "Command shape:",
            (
                "py -3 experiments\\pilots\\exp_003_hcc_runtime_consumer_smoke\\run.py "
                "--output-dir <output_dir> --seeds "
                f"{' '.join(str(seed) for seed in seeds)} --problems "
                f"{' '.join(problem_ids)} --jobs {max(1, int(jobs))} "
                f"--max-fes {max_fes} --budget-accounting {budget_accounting} "
                f"--lane-profile {lane_profile} "
                f"--hcc-root {vendor_paths.vendor_root} "
                f"--hcc-runner {vendor_paths.runner}"
                f"{'' if cmaes_restart else ' --no-cmaes-restart'}"
                f"{'' if mmes_restart else ' --no-mmes-restart'}"
            ),
            f"Budget: {max_fes} FE per lane/case",
            f"Budget accounting: {budget_accounting}",
            f"Lane profile: {lane_profile}",
            (
                "Optimizer restarts: "
                f"CMAES={'enabled' if cmaes_restart else 'disabled'}, "
                f"MMES={'enabled' if mmes_restart else 'disabled'}"
            ),
            f"Parallel jobs: {max(1, int(jobs))}",
            f"Lanes: {', '.join(lane.lane_id for lane in lanes)}",
            f"AOB data root: {Path(aob_data_root).resolve()}",
            f"HCC vendor root: {vendor_paths.vendor_root}",
            f"HCC AOB root: {vendor_paths.aob_root}",
            f"HCC source root: {vendor_paths.hcc_root}",
            f"HCC smoke runner: {vendor_paths.runner}",
            f"Actual cwd: {Path.cwd().resolve()}",
            f"Backend cwd: {vendor_paths.vendor_root}",
            f"Wrapper Python executable: {Path(sys.executable).resolve()}",
            f"Backend Python executable: {python_executable}",
            f"Search-state backend: {search_state_backend}",
            f"Python version: {platform.python_version()}",
            f"NumPy version: {_dependency_version('numpy')}",
            f"SciPy version: {_dependency_version('scipy')}",
            f"Torch version: {_dependency_version('torch')}",
            f"cma version: {_dependency_version('cma')}",
            f"BLAS: {_blas_summary()}",
            (
                "Pinned HCC runtime environment: "
                f"{'pass' if runtime_environment is not None else 'not_required'}"
            ),
            f"Thread environment: {_thread_environment()}",
            "",
            "Freeze evidence:",
            f"- git commit: {_git_commit()}",
            (
                "- config fingerprint: "
                f"{_config_fingerprint(seeds, problem_ids, jobs, max_fes, budget_accounting, cmaes_restart, mmes_restart, lanes, lane_profile, aob_data_root, vendor_paths, search_state_backend)}"
            ),
            f"- policy sha256: {_sha256_file(ARAC_SRC_ROOT / 'arac' / 'policy' / 'relation_policy.py')}",
            f"- search-state policy sha256: {_sha256_file(ARAC_SRC_ROOT / 'arac' / 'policy' / 'search_state_policy.py')}",
            f"- diagonal backend sha256: {_sha256_file(ARAC_SRC_ROOT / 'arac' / 'backends' / 'diagonal_cma.py')}",
            f"- experiment runner sha256: {_sha256_file(Path(__file__).resolve())}",
            f"- HCC smoke runner sha256: {_sha256_file(vendor_paths.runner)}",
            f"- MMES optimizer sha256: {_sha256_file(mmes_path)}",
            f"- MMES state model sha256: {_sha256_file(mmes_state_path)}",
            f"- CMAES optimizer sha256: {_sha256_file(cmaes_path)}",
            (
                "- AOB input hashes: aob_input_manifest.csv; "
                f"rows={len(aob_input_rows)}; "
                f"unchanged={int(bool(aob_input_rows) and all(str(row.get('unchanged')) == '1' for row in aob_input_rows))}"
            ),
            "",
            "Runtime boundary: final/reported/oracle values must not enter runtime dispatch.",
            "",
            "Key gates:",
            f"- claim scope: {multi_problem_claim_scope}",
            f"- same-budget violations: {same_budget_status}",
            (
                "- overlap-scope same-budget violations: "
                f"{overlap_scope_same_budget_status}"
            ),
            f"- pilot utility: {_diagnostic_observed_value(diagnosis_rows, 'pilot_utility_evidence')}",
            f"- multi-problem pilot utility: {multi_problem_pilot}",
            f"- multi-problem active density: {multi_problem_active_density}",
            f"- relation dispatch materiality: {multi_problem_relation_materiality}",
            f"- fixed repair baseline: {multi_problem_fixed_repair}",
            f"- fixed repair materiality: {multi_problem_fixed_repair_materiality}",
            f"- fixed coordinate baseline: {multi_problem_fixed_coordinate}",
            f"- multi-problem relation policy profile: {multi_problem_relation_policy_profile}",
            f"- claim tier recommendation: {multi_problem_claim_tier}",
            f"- SOTA escalation: {_sota_claim_allowed(diagnosis_rows)}",
            "",
            "Artifacts:",
            *[f"- {artifact}" for artifact in artifacts],
            "",
            "No performance or SOTA claim is made unless the relevant SOTA escalation gate is 1.",
        ]
    )
    (output_dir / "run_manifest.md").write_text(manifest + "\n", encoding="utf-8")


def run_hcc_runtime_consumer_smoke(
    output_dir: Path | str = experiment_results_dir(RUN_ID),
    execution_runner: Callable[[HccAobExecutionRequest], HccAobExecutionResult] = (
        run_hcc_aob_smoke_execution
    ),
    hcc_root: Path | str = HCC_VENDOR_ROOT,
    hcc_repo_root: Path | str | None = None,
    hcc_runner: Path | str | None = None,
    aob_data_root: Path | str = DEFAULT_AOB_DATA_ROOT,
    python_executable: str = sys.executable,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    problem_ids: tuple[str, ...] = (PROBLEM_ID,),
    jobs: int = 1,
    max_fes: int = MAX_FES,
    budget_accounting: str = "strict",
    cmaes_restart: bool = True,
    mmes_restart: bool = True,
    lane_profile: str = "runtime_smoke",
    search_state_backend: str = "phase_i_mmes",
    environment_probe: EnvironmentProbe | None = None,
) -> Path:
    worker_count = max(1, int(jobs))
    max_fes = int(max_fes)
    if max_fes <= 0:
        raise ValueError("max_fes must be positive")
    if budget_accounting not in {"strict", "source"}:
        raise ValueError("budget_accounting must be 'strict' or 'source'")
    if search_state_backend not in {"phase_i_mmes", "diagonal_cma"}:
        raise ValueError(
            "search_state_backend must be 'phase_i_mmes' or 'diagonal_cma'"
        )
    lanes = lanes_for_profile(lane_profile)
    precision_causal_profile_enabled = any(
        lane.precision_causal_arm != "off" for lane in lanes
    )
    if precision_causal_profile_enabled:
        preregistration_path = (
            ARAC_REPO_ROOT / PRECISION_CAUSAL_PREREGISTRATION_PATH
        )
        if (
            _sha256_file(preregistration_path)
            != PRECISION_CAUSAL_PREREGISTRATION_SHA256
        ):
            raise RuntimeError("precision causal preregistration hash mismatch")
    car_w_enabled = any(
        lane.runner_action_name in CAR_W_ACTION_NAMES
        for lane in lanes
    )
    problem_ids = tuple(problem_ids)
    seeds = tuple(seeds)
    _require_hcc_action_preflight(lanes, problem_ids)
    runtime_environment = None
    if _requires_pinned_environment(lanes):
        runtime_environment = require_pinned_hcc_runtime_environment(
            python_executable,
            environment_probe=environment_probe,
        )
    output = resolve_repository_path(output_dir).resolve()
    vendor_paths = resolve_hcc_vendor_paths(
        hcc_root,
        repo_root=hcc_repo_root,
        runner_path=hcc_runner,
    )
    resolved_aob_data_root = resolve_repository_path(aob_data_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if runtime_environment is not None:
        (output / "runtime_environment.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "expected": PINNED_HCC_RUNTIME_ENVIRONMENT,
                    "observed": runtime_environment,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    records = _records(
        output_dir=output,
        execution_runner=execution_runner,
        hcc_root=vendor_paths.vendor_root,
        aob_data_root=resolved_aob_data_root,
        python_executable=python_executable,
        seeds=seeds,
        problem_ids=problem_ids,
        max_fes=max_fes,
        jobs=worker_count,
        budget_accounting=budget_accounting,
        cmaes_restart=cmaes_restart,
        mmes_restart=mmes_restart,
        lanes=lanes,
        hcc_runner=vendor_paths.runner,
        search_state_backend=search_state_backend,
    )
    aob_input_rows = _aob_input_manifest_rows(records)
    ledger_rows = _ledger_rows(records)
    utility_rows = _utility_rows(records)
    negative_control_rows = _negative_control_rows(records)
    anti_leakage_rows = _anti_leakage_rows(records)
    action_execution_plan_rows = _action_execution_plan_rows(records)
    paired_runtime_utility_rows = (
        _paired_runtime_utility_rows(records)
        if lane_profile == "paired_v33_v36_runtime_utility"
        else []
    )
    action_trace_rows = _action_trace_rows(records)
    car_probe_trace_rows = _car_artifact_rows(records, "car_probe_trace.csv")
    car_state_ledger_rows = _car_artifact_rows(records, "car_state_ledger.csv")
    car_branch_manifest_rows = _car_artifact_rows(records, "car_branch_manifest.csv")
    car_actionability_enabled = any(
        lane.car_actionability_arm != "off" for lane in lanes
    )
    car_actionability_trace_rows = (
        [
            row
            for record in records
            for row in _car_actionability_rows_for_record(record)
        ]
        if car_actionability_enabled
        else []
    )
    car_actionability_summary_rows = (
        _car_actionability_summary_rows(car_actionability_trace_rows)
        if car_actionability_enabled
        else []
    )
    precision_causal_enabled = precision_causal_profile_enabled
    (
        causal_feature_rows,
        causal_audit_rows,
        causal_branch_rows,
        causal_outcome_rows,
        causal_randomized_rows,
        precision_causal_integrity_failures,
    ) = (
        _precision_causal_raw_rows(records)
        if precision_causal_enabled
        else ([], [], [], [], [], [])
    )
    car_dispatch_boundary_rows = (
        _car_dispatch_boundary_rows()
        if car_w_enabled
        else []
    )
    paired_integrity_failures: list[str] = []
    car_actionability_integrity_failures: list[str] = []
    if precision_causal_enabled:
        results = [record["result"] for record in records]
        expected_pair_count = len(problem_ids) * len(seeds)
        if not results or any(
            not isinstance(result, HccAobExecutionResult)
            or not result.fresh_optimizer_execution
            for result in results
        ):
            precision_causal_integrity_failures.append("not_all_runs_fresh")
        if any(
            str(row.get("same_budget_violation", "1")) != "0"
            for row in ledger_rows
        ):
            precision_causal_integrity_failures.append("same_budget_violation")
        if not aob_input_rows or any(
            str(row.get("unchanged", "0")) != "1" for row in aob_input_rows
        ):
            precision_causal_integrity_failures.append(
                "aob_input_changed_or_missing"
            )
        if not anti_leakage_rows or any(
            str(row.get("audit_status", "fail")) != "pass"
            for row in anti_leakage_rows
        ):
            precision_causal_integrity_failures.append("anti_leakage_violation")
        if len(causal_audit_rows) != expected_pair_count:
            precision_causal_integrity_failures.append(
                "causal_pair_count_mismatch"
            )
        if len(causal_branch_rows) != 2 * expected_pair_count:
            precision_causal_integrity_failures.append(
                "causal_branch_count_mismatch"
            )
        if any(
            str(row.get("terminal_status", "")) != "complete"
            for row in causal_branch_rows
        ):
            precision_causal_integrity_failures.append(
                "incomplete_terminal_branch"
            )
    if car_actionability_enabled:
        results = [record["result"] for record in records]
        if not results or any(
            not isinstance(result, HccAobExecutionResult)
            or not result.fresh_optimizer_execution
            for result in results
        ):
            car_actionability_integrity_failures.append("not_all_runs_fresh")
        if any(str(row.get("same_budget_violation", "1")) != "0" for row in ledger_rows):
            car_actionability_integrity_failures.append("same_budget_violation")
        if not aob_input_rows or any(
            str(row.get("unchanged", "0")) != "1" for row in aob_input_rows
        ):
            car_actionability_integrity_failures.append("aob_input_changed_or_missing")
        car_actionability_integrity_failures.extend(
            _car_actionability_aob_pair_failures(
                aob_input_rows,
                problem_ids=problem_ids,
                seeds=seeds,
                lanes=lanes,
            )
        )
        if not anti_leakage_rows or any(
            str(row.get("audit_status", "fail")) != "pass"
            for row in anti_leakage_rows
        ):
            car_actionability_integrity_failures.append("anti_leakage_violation")
        if not car_actionability_trace_rows:
            car_actionability_integrity_failures.append("missing_actionability_trace")
        car_actionability_integrity_failures.extend(
            _car_actionability_coverage_failures(
                car_actionability_trace_rows,
                problem_ids=problem_ids,
                seeds=seeds,
                lanes=lanes,
            )
        )
        car_actionability_integrity_failures.extend(
            f"{row['problem_id']}/seed{row['seed']}/{row['horizon_label']}:{row['integrity_failures']}"
            for row in car_actionability_summary_rows
            if str(row.get("integrity_status")) != "pass"
        )
        car_actionability_summary_rows = _redact_car_actionability_summary_rows(
            car_actionability_summary_rows,
            car_actionability_integrity_failures,
        )
    if lane_profile == "paired_v33_v36_runtime_utility":
        results = [record["result"] for record in records]
        plans = [record["plan"] for record in records]
        if runtime_environment is None:
            paired_integrity_failures.append("missing_pinned_environment_audit")
        if not results or any(
            not isinstance(result, HccAobExecutionResult)
            or not result.fresh_optimizer_execution
            for result in results
        ):
            paired_integrity_failures.append("not_all_runs_fresh")
        if any(str(row.get("same_budget_violation", "1")) != "0" for row in ledger_rows):
            paired_integrity_failures.append("same_budget_violation")
        if not aob_input_rows or any(
            str(row.get("unchanged", "0")) != "1" for row in aob_input_rows
        ):
            paired_integrity_failures.append("aob_input_changed_or_missing")
        if not anti_leakage_rows or any(
            str(row.get("audit_status", "fail")) != "pass"
            for row in anti_leakage_rows
        ):
            paired_integrity_failures.append("anti_leakage_violation")
        if any(
            not isinstance(plan, HccActionExecutionPlan)
            or not plan.optimizer_consumed
            or not plan.runtime_dispatch_allowed
            for plan in plans
        ):
            paired_integrity_failures.append("action_execution_plan_not_consumed")
        if not results or any(
            not isinstance(result, HccAobExecutionResult)
            or result.action_trace_path is None
            or result.action_trace_rows <= 0
            for result in results
        ):
            paired_integrity_failures.append("action_trace_missing_or_empty")
    diagnosis_rows = _policy_evidence_diagnosis_rows(
        records,
        utility_rows,
        negative_control_rows,
    )
    _write_csv(
        output / "our_result_by_case.csv",
        _our_result_rows(records, utility_rows),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "selected_action_family",
            "selected_action_name",
            "hcc_smoke_final_error",
            "hcc_smoke_fe_used",
            "hcc_smoke_status",
            "fresh_optimizer_execution",
            "result_source",
            "action_trace_sha256",
            "action_trace_rows",
            "runtime_dispatch_allowed",
            "dispatch_scope",
            "relation_dispatch_enabled",
            "runtime_connected_claim_allowed",
            "utility_claim_allowed",
            "performance_claim_allowed",
        ],
    )
    _write_csv(
        output / "same_budget_ledger.csv",
        ledger_rows,
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "same_budget_group_id",
            "phase_i_fe",
            "phase_ii_fe",
            "cc_phase_fe",
            "rescue_fe",
            "refresh_fe",
            "search_state_fe",
            "separable_continuation_fe",
            "overhead_fe",
            "total_fe",
            "budget_limit",
            "configured_budget_limit",
            "budget_aligned_fe_used",
            "actual_fe_used",
            "budget_limit_source",
            "same_budget_violation",
            "fresh_execution",
        ],
    )
    _write_csv(
        output / "backend_semantics_diff.csv",
        _semantics_rows(records),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "selected_action_name",
            "variable_owner_changed",
            "relation_handling_changed",
            "coordination_mode_changed",
            "budget_allocation_changed",
            "update_order_changed",
            "acceptance_rule_changed",
            "backend_semantics_changed",
        ],
    )
    _write_csv(
        output / "action_execution_plan.csv",
        action_execution_plan_rows,
        [
            "run_id",
            "lane_id",
            "problem_id",
            "selected_action_name",
            "selected_action_family",
            "backend_effect_kind",
            "optimizer_consumed",
            "optimizer_consumed_parameters",
            "execution_mode",
            "blocker_reason",
            "runtime_dispatch_allowed",
        ],
    )
    _write_csv(
        output / "action_trace.csv",
        action_trace_rows,
        [
            "run_id",
            "lane_id",
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
            *action_trace_fields_for_lanes(lanes),
        ],
    )
    if car_w_enabled:
        _write_csv(
            output / "car_dispatch_boundary_audit.csv",
            car_dispatch_boundary_rows,
            [
                "boundary",
                "field_name",
                "field_owner",
                "present_in_runtime_type",
                "audit_only",
                "runtime_dispatch_allowed",
                "audit_status",
            ],
        )
        _write_csv(
            output / "car_probe_trace.csv",
            car_probe_trace_rows,
            [
                "run_id",
                "lane_id",
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
            ],
        )
        _write_csv(
            output / "car_state_ledger.csv",
            car_state_ledger_rows,
            [
                "run_id",
                "lane_id",
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
            ],
        )
        _write_csv(
            output / "car_branch_manifest.csv",
            car_branch_manifest_rows,
            [
                "run_id",
                "lane_id",
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
            ],
        )
    if car_actionability_enabled:
        _write_csv(
            output / "car_actionability_trace.csv",
            car_actionability_trace_rows,
            [
                "run_id",
                "lane_id",
                "request_fingerprint",
                "execution_context_fingerprint",
                "aob_input_fingerprint",
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
            ],
        )
        _write_csv(
            output / "car_actionability_summary.csv",
            car_actionability_summary_rows,
            [
                "run_id",
                "problem_id",
                "seed",
                "horizon_label",
                "horizon_index",
                "checkpoint_fe",
                "configured_max_fes",
                "fallback_request_fingerprint",
                "candidate_request_fingerprint",
                "execution_context_fingerprint",
                "aob_input_fingerprint",
                "target_fe",
                "observed_fe",
                "prefix_match",
                "equal_fe",
                "fallback_error",
                "candidate_error",
                "log_advantage",
                "relative_gain",
                "numeric_win",
                "meaningful_win",
                "catastrophic_loss",
                "oracle_selected_arm",
                "oracle_gain",
                "terminal_sign_agreement",
                "rank_reversal_from_previous",
                "integrity_status",
                "integrity_failures",
            ],
        )
        (output / "car_actionability_gate.json").write_text(
            json.dumps(
                {
                    "status": (
                        "pass" if not car_actionability_integrity_failures else "blocked"
                    ),
                    "integrity_failures": car_actionability_integrity_failures,
                    "offline_only": True,
                    "estimand": "one_shot_component_actionability_with_canonical_continuation",
                    "horizons": ["closure_1", "budget_3x", "budget_9x", "terminal"],
                    "runtime_dispatch_inputs": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if precision_causal_enabled:
        _write_csv(
            output / "causal_decision_features.csv",
            causal_feature_rows,
            ["decision_id", *UTILITY_FEATURE_NAMES],
        )
        _write_csv(
            output / "causal_decision_audit.csv",
            causal_audit_rows,
            [
                "protocol_version",
                "pair_id",
                "decision_id",
                "problem_id",
                "seed",
                "decision_status",
                "not_applicable_reason",
                "logged_arm",
                "propensity",
                "decision_fe",
                "checkpoint_fitness",
                "remaining_fe",
                "component_id",
                "component_group_count",
                "component_shared_var_count",
                "component_unlocked",
                "scheduler_revisit_reachable",
                "scheduler_revisit_cap_fe",
                "scheduler_revisit_reason",
                "source_phase_i_end_fe",
                "source_cc_history_end_fe",
                "source_disagreement_history_end_fe",
                "source_cma_history_end_fe",
                "source_end_fe",
                "prefix_record_sha256",
                "checkpoint_candidate_sha256",
                "controller_state_sha256",
                "random_descriptor_sha256",
                "feature_schema_sha256",
                "feature_sha256",
                "decision_status_match",
                "decision_id_match",
                "feature_match",
                "prefix_match",
                "controller_state_match",
                "checkpoint_candidate_match",
                "random_descriptor_match",
                "intervention_end_fe_match",
                "not_applicable_reason_match",
                "pair_integrity",
            ],
        )
        _write_csv(
            output / "causal_branch_manifest.csv",
            causal_branch_rows,
            [
                "pair_id",
                "decision_id",
                "problem_id",
                "seed",
                "arm",
                "lane_id",
                "fresh_optimizer_execution",
                "status",
                "result_source",
                "output_root",
                "decision_status",
                "not_applicable_reason",
                "action_applied",
                "decision_fe",
                "intervention_end_fe",
                "checkpoint_fitness",
                "normal_sigma",
                "candidate_sigma",
                "applied_sigma",
                "requested_fe",
                "actual_fe",
                "configured_max_fes",
                "terminal_target_fe",
                "terminal_observed_fe",
                "terminal_status",
                "prefix_record_sha256",
                "checkpoint_candidate_sha256",
                "controller_state_sha256",
                "feature_sha256",
                "random_descriptor_sha256",
                "terminal_error",
                "terminal_record_sha256",
                "optimizer_fe_used",
                "same_budget_violation",
            ],
        )
        _write_csv(
            output / "causal_outcomes.csv",
            causal_outcome_rows,
            [
                "pair_id",
                "decision_id",
                "problem_id",
                "seed",
                "decision_status",
                "checkpoint_error",
                "baseline_terminal_error",
                "action_terminal_error",
                "baseline_log_progress",
                "action_log_progress",
                "paired_tau",
                "catastrophic",
                "equal_checkpoint",
                "equal_terminal_target_fe",
                "equal_terminal_observed_fe",
                "outcome_valid",
            ],
        )
        _write_csv(
            output / "randomized_log.csv",
            causal_randomized_rows,
            [
                "pair_id",
                "decision_id",
                "problem_id",
                "seed",
                "logged_arm",
                "observed_treatment",
                "propensity",
                "observed_terminal_error",
                "observed_log_progress",
                "terminal_target_fe",
                "terminal_observed_fe",
                "outcome_valid",
            ],
        )
        feature_manifest = {
            "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
            "feature_names": list(UTILITY_FEATURE_NAMES),
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "features": [
                {
                    "name": name,
                    "formula": PRECISION_CAUSAL_FEATURE_FORMULAS[name],
                    "source_timing": "strictly_pre_action",
                }
                for name in UTILITY_FEATURE_NAMES
            ],
            "identity_fields": ["problem_id", "seed", "component_id"],
            "identity_fields_location": "causal_decision_audit.csv_only",
            "forbidden_model_fields": sorted(
                {
                    "case",
                    "problem_id",
                    "seed",
                    "function_family",
                    "paper_best",
                    "historical_best",
                    "final_error",
                    "final_outcome",
                    "graph_fingerprint",
                    "component_id",
                    "group_index",
                    "raw_objective",
                    "incumbent",
                    "component_gain",
                    "neighbor_gain",
                    "overwrite",
                    "survival",
                }
            ),
            "immutable_snapshot": True,
        }
        (output / "feature_manifest.json").write_text(
            json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_artifact_names = (
            "causal_decision_features.csv",
            "causal_decision_audit.csv",
            "causal_branch_manifest.csv",
            "causal_outcomes.csv",
            "randomized_log.csv",
            "causal_randomization_schedule.json",
            "feature_manifest.json",
        )
        logging_manifest = {
            "protocol_version": PRECISION_CAUSAL_PROTOCOL_VERSION,
            "offline_only": True,
            "runtime_scheduler_authorized": False,
            "lane_profile": lane_profile,
            "baseline_action": controller_profile_by_version(37).action_name,
            "treatment_action": "post_retirement_precision_reanchor",
            "treatment_semantics": (
                "one_v38_precision_sigma_group_block_then_v37_continuation"
            ),
            "feature_schema": {
                "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "feature_names": list(UTILITY_FEATURE_NAMES),
            },
            "randomization": {
                "randomization_salt": PRECISION_CAUSAL_RANDOMIZATION_SALT,
                "randomization_algorithm": "sha256_first_u64_mod2",
                "coin_material": "{salt}|{problem_id.upper()}|{int(seed)}",
                "arm_mapping": {"0": "baseline", "1": "action"},
                "propensity": 0.5,
            },
            "estimand": {
                "unit": "first_complete_scheduler_reachable_unlocked_precision_opportunity_per_trajectory",
                "outcome": "log(checkpoint_error)-log(terminal_error)",
                "paired_tau": "log(baseline_terminal_error/action_terminal_error)",
                "catastrophic": "action_terminal_error >= 1.2 * baseline_terminal_error",
                "log_floor": 1e-300,
            },
            "preregistration": {
                "path": PRECISION_CAUSAL_PREREGISTRATION_PATH,
                "sha256": PRECISION_CAUSAL_PREREGISTRATION_SHA256,
                "commit": PRECISION_CAUSAL_PREREGISTRATION_COMMIT,
            },
            "matrix": {
                "problem_ids": list(problem_ids),
                "seeds": list(seeds),
                "arms": list(PRECISION_CAUSAL_ARMS),
                "max_fes": max_fes,
                "jobs": worker_count,
                "budget_accounting": budget_accounting,
            },
            "integrity": {
                "status": (
                    "pass" if not precision_causal_integrity_failures else "blocked"
                ),
                "failures": precision_causal_integrity_failures,
                "applicable_pairs": len(causal_feature_rows),
                "total_pairs": len(causal_audit_rows),
            },
            "git_commit": _git_commit(),
            "raw_artifact_sha256": {
                name: _sha256_file(output / name) for name in raw_artifact_names
            },
        }
        (output / "causal_logging_manifest.json").write_text(
            json.dumps(logging_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_csv(
        output / "trajectory_guard_summary.csv",
        _trajectory_guard_summary_rows(action_trace_rows),
        TRAJECTORY_GUARD_SUMMARY_FIELDS,
    )
    _write_csv(
        output / "pre_hold_evidence.csv",
        _pre_hold_evidence_rows(action_trace_rows),
        PRE_HOLD_EVIDENCE_FIELDS,
    )
    _write_csv(
        output / "action_decision.csv",
        _action_decision_rows(records),
        [
            "run_id",
            "lane_id",
            "seed",
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
        ],
    )
    _write_csv(
        output / "action_mismatch_audit.csv",
        _action_mismatch_rows(records),
        [
            "run_id",
            "lane_id",
            "seed",
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
        ],
    )
    _write_csv(
        output / "overlap_relations.csv",
        _overlap_relation_rows(records),
        [
            "run_id",
            "lane_id",
            "seed",
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
        ],
    )
    _write_csv(
        output / "relation_join_audit.csv",
        _relation_join_rows(records),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "relation_id",
            "has_action_trace",
            "has_action_decision",
            "has_overlap_relation",
            "audit_status",
        ],
    )
    _write_csv(
        output / "action_utility_audit.csv",
        utility_rows,
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "final_error",
            "fe_used",
            "same_budget_violation",
            "relative_gain_vs_fallback",
            "utility_label",
            "action_mix",
            "optimizer_consumed_action_mix",
            "runtime_connected_claim_allowed",
            "backend_semantics_changed",
            "claim_allowed",
            "claim_blockers",
        ],
    )
    _write_csv(
        output / "negative_control_comparison.csv",
        negative_control_rows,
        [
            "run_id",
            "problem_id",
            "seeds",
            "relation_dispatch_mean_final_error",
            "shuffled_mean_final_error",
            "shuffled_win_count",
            "total_seeds",
            "stable_outperform_detected",
            "negative_control_pass",
            "diagnostic",
        ],
    )
    if paired_runtime_utility_rows:
        _write_csv(
            output / "paired_runtime_utility_summary.csv",
            paired_runtime_utility_rows,
            [
                "run_id",
                "problem_id",
                "seed_count",
                "fallback_mean_error",
                "candidate_mean_error",
                "fallback_worst_error",
                "candidate_worst_error",
                "mean_log_error_delta",
                "meaningful_seed_wins",
                "catastrophic_losses",
                "mean_win",
                "worst_seed_win",
            ],
        )
        paired_gate = _paired_runtime_utility_gate(
            paired_runtime_utility_rows,
            negative_control_rows=negative_control_rows,
            integrity_failures=paired_integrity_failures,
        )
        _write_paired_runtime_utility_gate(output, paired_gate)
    _write_csv(
        output / "policy_evidence_diagnosis.csv",
        diagnosis_rows,
        [
            "run_id",
            "problem_id",
            "diagnostic_key",
            "status",
            "observed_value",
            "blocker_reason",
            "next_step",
        ],
    )
    _write_claim_evidence_table(output, diagnosis_rows)
    _write_csv(
        output / "aob_input_manifest.csv",
        aob_input_rows,
        [
            "run_id",
            "lane_id",
            "seed",
            "problem_id",
            "file",
            "path",
            "sha256_before",
            "sha256_after",
            "unchanged",
        ],
    )
    _write_csv(
        output / "anti_leakage_audit.csv",
        anti_leakage_rows,
        [
            "run_id",
            "artifact_path",
            "forbidden_field",
            "found_in_runtime_payload",
            "runtime_dispatch_allowed",
            "audit_status",
        ],
    )
    _write_csv(
        output / "claim_gate.csv",
        _claim_gate_rows(records, utility_rows),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "selected_action_name",
            "optimizer_consumed",
            "same_budget_violation",
            "runtime_connected_claim_allowed",
            "runtime_claim_blockers",
            "utility_claim_allowed",
            "utility_claim_blockers",
            "performance_claim_allowed",
            "claim_allowed",
            "claim_blockers",
        ],
    )
    _write_manifest(
        output,
        tuple(seeds),
        tuple(problem_ids),
        diagnosis_rows,
        ledger_rows,
        worker_count,
        max_fes,
        budget_accounting,
        cmaes_restart,
        mmes_restart,
        lanes,
        lane_profile,
        resolved_aob_data_root,
        aob_input_rows,
        python_executable,
        vendor_paths,
        search_state_backend,
        runtime_environment,
    )
    if car_actionability_integrity_failures:
        raise RuntimeError(
            "CAR actionability audit integrity gate blocked: "
            + ";".join(car_actionability_integrity_failures)
        )
    if precision_causal_integrity_failures:
        raise RuntimeError(
            "precision causal logging integrity gate blocked: "
            + ";".join(precision_causal_integrity_failures)
        )
    if paired_runtime_utility_rows:
        manifest_path = output / "run_manifest.md"
        manifest = manifest_path.read_text(encoding="utf-8")
        manifest += (
            "\nPaired runtime utility gate: "
            f"{paired_gate['status']}; "
            f"blockers={','.join(str(value) for value in paired_gate['blockers']) if paired_gate['blockers'] else 'none'}; "
            "details=paired_runtime_utility_gate.json\n"
        )
        manifest_path.write_text(manifest, encoding="utf-8")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp_003 HCC runtime consumer smoke.")
    parser.add_argument("--output-dir", default=str(experiment_results_dir(RUN_ID)))
    parser.add_argument(
        "--hcc-root",
        default=str(HCC_VENDOR_ROOT),
        help="canonical vendor/hcc runtime root, or a valid vendor snapshot with explicit runner context.",
    )
    parser.add_argument(
        "--hcc-repo-root",
        default=None,
        help="Repository root that owns scripts/hcc_smoke_runner.py for a vendor override.",
    )
    parser.add_argument(
        "--hcc-runner",
        default=None,
        help="Explicit ARAC-owned HCC smoke runner path for a vendor override.",
    )
    parser.add_argument("--aob-data-root", default=str(DEFAULT_AOB_DATA_ROOT))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--problems", nargs="+", default=[PROBLEM_ID])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-fes", type=int, default=MAX_FES)
    parser.add_argument("--budget-accounting", default="strict", choices=["strict", "source"])
    parser.add_argument(
        "--search-state-backend",
        default="phase_i_mmes",
        choices=["phase_i_mmes", "diagonal_cma"],
    )
    parser.add_argument("--cmaes-restart", dest="cmaes_restart", action="store_true", default=True)
    parser.add_argument("--no-cmaes-restart", dest="cmaes_restart", action="store_false")
    parser.add_argument("--mmes-restart", dest="mmes_restart", action="store_true", default=True)
    parser.add_argument("--no-mmes-restart", dest="mmes_restart", action="store_false")
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
            "precision_refine_push",
            "phase_rescue_push",
            "repair_phase_rescue_push",
            "cc_harm_sep_refresh",
            "separable_cmaes_push",
            "evidence_routed_only",
            "evidence_routed_v2_only",
            "evidence_routed_v21_only",
            "evidence_routed_v22_only",
            "evidence_routed_v23_only",
            "evidence_routed_v24_only",
            "evidence_routed_v25_only",
            "evidence_routed_v26_only",
            "paper_best_win_push",
            "paper_best_win_push_v2",
            "historical_anchor_refine_push",
            "historical_13_preserve_push",
            "historical_13_fast_preserve",
            "historical_13_runtime_composite",
            "historical_13_runtime_composite_v2",
            "evidence_action_controller_v1",
            "evidence_action_controller_v2",
            "evidence_action_controller_v3",
            "evidence_action_controller_v31",
            "evidence_action_controller_v32",
            *controller_lane_profile_names(),
            "paired_v33_v36_runtime_utility",
            "car_w_diagnostic",
            "car_w2_diagnostic",
            "car_w3_diagnostic",
            "car_actionability_audit",
            "precision_causal_logging",
            "canonical_evidence_controller_v1",
        ],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_hcc_runtime_consumer_smoke(
        output_dir=args.output_dir,
        hcc_root=Path(args.hcc_root),
        hcc_repo_root=(
            None if args.hcc_repo_root is None else Path(args.hcc_repo_root)
        ),
        hcc_runner=None if args.hcc_runner is None else Path(args.hcc_runner),
        aob_data_root=Path(args.aob_data_root),
        python_executable=str(args.python_executable),
        seeds=tuple(args.seeds),
        problem_ids=tuple(str(problem).upper() for problem in args.problems),
        jobs=int(args.jobs),
        max_fes=int(args.max_fes),
        budget_accounting=str(args.budget_accounting),
        cmaes_restart=bool(args.cmaes_restart),
        mmes_restart=bool(args.mmes_restart),
        lane_profile=str(args.lane_profile),
        search_state_backend=str(args.search_state_backend),
    )


if __name__ == "__main__":
    main()
