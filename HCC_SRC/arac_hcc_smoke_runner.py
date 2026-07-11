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
for import_root in (ARAC_REPO_ROOT, ARAC_SRC_ROOT):
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
    CONTINUE_CANONICAL_CC,
    RESUME_PHASE_I_SEARCH_STATE,
    SearchStateEvidence,
    SearchStateSchedulerState,
    normalized_gain_utility,
    plan_search_state_action,
    record_search_state_outcome,
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


DATA_DIR = ARAC_REPO_ROOT / "HCC_SRC" / "AOB" / "AOBG" / "datafile"
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
    "search_state_block_fe",
    "search_state_utility",
    "required_utility_ratio",
    "state_action_fe",
    "cc_reserve_fe",
    "state_fingerprint_before",
    "state_fingerprint_after",
    "abstain_reason",
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
ACTION_VALUE_DELTA_GUARD_THRESHOLD = 0.5
COORDINATE_ACTION_VALUE_DELTA_GUARD_THRESHOLD = 2.5
V31_NON_DENSE_PREFIX_RELATION_COUNT = 3
V31_NON_DENSE_PREFIX_SHARED_VAR_COUNT = 3
V31_NON_DENSE_PREFIX_REPAIR_TRIGGER = "controller_v31_non_dense_prefix_repair_lock"
V31_NON_DENSE_LARGE_FALLBACK_DELTA_RATIO_MAX = 0.15
V31_NON_DENSE_LARGE_FALLBACK_NORM_MIN = 10.0
V31_NON_DENSE_LARGE_FALLBACK_REPAIR_TRIGGER = (
    "controller_v31_non_dense_large_fallback_repair_lock"
)
BOUNDED_LATE_NDA_REFRESH_ACTION = "bounded_late_nda_refresh"
BOUNDED_REFRESH_REMAINING_RATIO_MIN = 0.08
BOUNDED_REFRESH_REMAINING_RATIO_MAX = 0.30
BOUNDED_REFRESH_BUDGET_FRACTION = 0.15
BOUNDED_REFRESH_CONTINUATION_FRACTION = 0.05
SEARCH_STATE_BIPOP_ACTION = "bipop_search_state_restart"
REPAIR_BIPOP_SEARCH_STATE_ACTION = "repair_bipop_search_state_restart"
PHASE_RESCUE_MULTISTART_ACTION = "phase_rescue_multistart"
REPAIR_PHASE_RESCUE_MULTISTART_ACTION = "repair_phase_rescue_multistart"
CC_HARM_GUARDED_SEP_REFRESH_ACTION = "cc_harm_guarded_sep_refresh"
SEPARABLE_CMAES_DISPATCH_ACTION = "separable_cmaes_dispatch_action"
REPAIR_PROTECT_REFINE_ACTION = "repair_protect_refine"
REPAIR_PROTECT_DEEP_REFINE_ACTION = "repair_protect_deep_refine"
EVIDENCE_ACTION_CONTROLLER_V1 = "arac_evidence_action_controller_v1"
EVIDENCE_ACTION_CONTROLLER_V2 = "arac_evidence_action_controller_v2"
EVIDENCE_ACTION_CONTROLLER_V3 = "arac_evidence_action_controller_v3"
EVIDENCE_ACTION_CONTROLLER_V31 = "arac_evidence_action_controller_v31"
TRAJECTORY_ACTION_NAMES = {
    "budget_shift_mean_blend",
    "budget_shift_only",
    "mean_blend_only",
    SEARCH_STATE_BIPOP_ACTION,
    REPAIR_BIPOP_SEARCH_STATE_ACTION,
    PHASE_RESCUE_MULTISTART_ACTION,
    REPAIR_PHASE_RESCUE_MULTISTART_ACTION,
    BOUNDED_LATE_NDA_REFRESH_ACTION,
    CC_HARM_GUARDED_SEP_REFRESH_ACTION,
    SEPARABLE_CMAES_DISPATCH_ACTION,
    REPAIR_PROTECT_REFINE_ACTION,
    REPAIR_PROTECT_DEEP_REFINE_ACTION,
    EVIDENCE_ACTION_CONTROLLER_V1,
    EVIDENCE_ACTION_CONTROLLER_V2,
    EVIDENCE_ACTION_CONTROLLER_V3,
    EVIDENCE_ACTION_CONTROLLER_V31,
    RESUME_PHASE_I_SEARCH_STATE,
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
CC_HARM_REFRESH_SIGMA_MULTIPLIER = 0.75
BOUNDED_REFRESH_GROUP_SPARSE_STAGNATED_FRACTION = 0.50
BOUNDED_REFRESH_GROUP_SPARSE_CONFLICT_FRACTION = 0.50
SEPARABLE_CMAES_INITIAL_SIGMA = 0.5
SEPARABLE_CMAES_SIGMA_ADAPTATION_RATE = 0.2
SEPARABLE_CMAES_MIN_SIGMA = 1e-12
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


@dataclass(frozen=True)
class BoundedLateNdaRefreshPlan:
    refresh_budget: int
    continuation_reserve: int
    remaining_budget_ratio: float
    shared_var_count: int
    trigger_reason: str


@dataclass
class EvidenceActionControllerV31RunState:
    dense_overlap: bool
    locked_policy_mode: str | None = None
    non_dense_repair_locked: bool = False
    non_dense_repair_lock_trigger: str = ""
    bounded_late_nda_refresh_consumed: bool = False
    search_state_scheduler_state: SearchStateSchedulerState = field(
        default_factory=SearchStateSchedulerState
    )
    phase_i_optimizer: object | None = field(default=None, repr=False)
    phase_i_state: MMESState | None = field(default=None, repr=False)
    cc_utility_history: list[float] = field(default_factory=list)
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
        return not self.dense_overlap and not self.non_dense_repair_locked

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


def build_evidence_action_controller_v31_run_state(
    degree_of_overlap: float,
) -> EvidenceActionControllerV31RunState:
    return EvidenceActionControllerV31RunState(
        dense_overlap=is_evidence_action_controller_v31_dense_overlap(
            degree_of_overlap
        )
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


def bounded_population_budget(
    requested_fes: int,
    remaining_fes: int,
    population_size: int,
) -> int:
    usable_fes = min(requested_fes, remaining_fes)
    if usable_fes <= 0 or population_size <= 0:
        return 0
    return (usable_fes // population_size) * population_size


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


def is_guarded_evidence_action_controller(action_name: str) -> bool:
    return is_evidence_action_controller_v3(action_name) or is_evidence_action_controller_v31(action_name)


def is_evidence_action_controller(action_name: str) -> bool:
    return (
        is_evidence_action_controller_v1(action_name)
        or is_evidence_action_controller_v2(action_name)
        or is_evidence_action_controller_v3(action_name)
        or is_evidence_action_controller_v31(action_name)
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
        is_evidence_action_controller_v3(action_name)
        and evidence_controller_search_state_enabled
    )


def is_cc_harm_guarded_sep_refresh_action(action_name: str) -> bool:
    return action_name == CC_HARM_GUARDED_SEP_REFRESH_ACTION


def is_bounded_late_nda_refresh_action(action_name: str) -> bool:
    return action_name == BOUNDED_LATE_NDA_REFRESH_ACTION


def is_separable_cmaes_dispatch_action(action_name: str) -> bool:
    return action_name == SEPARABLE_CMAES_DISPATCH_ACTION


def is_search_state_action(action_name: str) -> bool:
    return (
        is_bipop_search_state_action(action_name)
        or is_phase_rescue_multistart_action(action_name)
        or is_bounded_late_nda_refresh_action(action_name)
        or is_cc_harm_guarded_sep_refresh_action(action_name)
        or is_separable_cmaes_dispatch_action(action_name)
        or action_name == RESUME_PHASE_I_SEARCH_STATE
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
    if (
        is_evidence_action_controller_v31(action_name)
        and controller_v31_run_state is not None
        and not controller_v31_run_state.dense_overlap
    ):
        return float(base_sigma) * REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER
    return float(base_sigma)


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
    conflict = cc_harm_conflict_fraction(fitness_deltas, reference_fitness)
    unstable = any(
        abs(float(norm)) > CC_HARM_WRITEBACK_NORM for norm in writeback_norms
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


def plan_bounded_late_nda_refresh(
    *,
    controller_v31_run_state: EvidenceActionControllerV31RunState | None,
    current_outer_relations: list[OverlapRelation],
    fitness_deltas: list[float],
    overlap_writeback_norms: list[float],
    reference_fitness: float,
    remaining_fes: int,
    max_fes: int,
    population_size: int,
    expected_group_count: int,
) -> BoundedLateNdaRefreshPlan | None:
    state = controller_v31_run_state
    if (
        state is None
        or state.dense_overlap
        or state.non_dense_repair_locked
        or state.bounded_late_nda_refresh_consumed
        or not state.phase_rescue_enabled
        or max_fes <= 0
        or expected_group_count < CC_HARM_MIN_GROUP_UPDATES
        or len(fitness_deltas) != expected_group_count
        or len(current_outer_relations) != expected_group_count - 1
        or len(fitness_deltas) < CC_HARM_MIN_GROUP_UPDATES
        or len(current_outer_relations) < CC_HARM_MIN_GROUP_UPDATES - 1
    ):
        return None

    shared_counts = {len(relation.shared_vars) for relation in current_outer_relations}
    if shared_counts != {V31_NON_DENSE_PREFIX_SHARED_VAR_COUNT}:
        return None

    remaining_ratio = max(0.0, remaining_fes / max_fes)
    if not (
        BOUNDED_REFRESH_REMAINING_RATIO_MIN
        <= remaining_ratio
        <= BOUNDED_REFRESH_REMAINING_RATIO_MAX
    ):
        return None

    triggered, reason = should_trigger_cc_harm_guard(
        fitness_deltas=fitness_deltas,
        overlap_writeback_norms=overlap_writeback_norms,
        reference_fitness=reference_fitness,
        remaining_fes=remaining_fes,
        minimum_refresh_budget=population_size,
    )
    if triggered and (
        "high_relation_conflict" in reason
        or "severe_group_stagnation" in reason
    ):
        trigger_reason = reason
    else:
        reference = max(abs(float(reference_fitness)), 1.0)
        stagnated_count = sum(
            1
            for delta in fitness_deltas
            if group_delta_stagnated(float(delta), reference)
        )
        stagnated_fraction = stagnated_count / max(1, len(fitness_deltas))
        conflict_fraction = cc_harm_conflict_fraction(
            fitness_deltas,
            reference,
        )
        sparse_conflict = (
            stagnated_fraction >= BOUNDED_REFRESH_GROUP_SPARSE_STAGNATED_FRACTION
            and conflict_fraction >= BOUNDED_REFRESH_GROUP_SPARSE_CONFLICT_FRACTION
        )
        if not sparse_conflict:
            return None
        trigger_reason = "group_sparse_stagnation+high_relation_conflict"

    continuation_reserve = math.ceil(
        max_fes * BOUNDED_REFRESH_CONTINUATION_FRACTION
    )
    available_refresh_fes = remaining_fes - continuation_reserve
    if available_refresh_fes < population_size:
        return None
    refresh_cap = math.floor(max_fes * BOUNDED_REFRESH_BUDGET_FRACTION)
    refresh_budget = bounded_population_budget(
        requested_fes=min(refresh_cap, available_refresh_fes),
        remaining_fes=available_refresh_fes,
        population_size=population_size,
    )
    if refresh_budget <= 0:
        return None
    return BoundedLateNdaRefreshPlan(
        refresh_budget=refresh_budget,
        continuation_reserve=continuation_reserve,
        remaining_budget_ratio=remaining_ratio,
        shared_var_count=next(iter(shared_counts)),
        trigger_reason=trigger_reason,
    )


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
    sigma = np.full((dimension,), SEPARABLE_CMAES_INITIAL_SIGMA, dtype=float)
    population_size = calculate_cmaes_population_size(dimension)
    parents = max(1, population_size // 2)
    raw_weights = np.log(parents + 0.5) - np.log(np.arange(1, parents + 1))
    weights = raw_weights / np.sum(raw_weights)
    rng = np.random.default_rng(
        derive_optimizer_seed(
            config.seed if config.seed is not None else 0,
            fun_name,
            fun_id,
            0,
            47011,
        )
    )

    best_x = np.copy(mean)
    best_y = math.inf if incumbent_fitness is None else float(incumbent_fitness)
    evaluations = 0
    evaluation_budget = int(
        config.max_fes if max_function_evaluations is None else max_function_evaluations
    )
    while evaluations < evaluation_budget:
        batch_size = min(population_size, evaluation_budget - evaluations)
        z = rng.standard_normal((batch_size, dimension))
        candidates = np.clip(mean + sigma * z, lower, upper)
        y = np.asarray(fun(candidates), dtype=float).reshape(-1)
        evaluations += int(len(candidates))

        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        finite_indices = np.where(finite)[0]
        local_best = finite_indices[int(np.argmin(y[finite_indices]))]
        if float(y[local_best]) < best_y:
            best_y = float(y[local_best])
            best_x = np.copy(candidates[local_best])

        if batch_size < parents:
            continue
        order = np.argsort(np.where(finite, y, math.inf))[:parents]
        selected = candidates[order]
        selected_z = z[order]
        mean = np.dot(weights, selected)
        variance_signal = np.sqrt(np.dot(weights, np.square(selected_z)))
        sigma *= np.exp(
            SEPARABLE_CMAES_SIGMA_ADAPTATION_RATE * (variance_signal - 1.0)
        )
        sigma = np.clip(
            sigma,
            SEPARABLE_CMAES_MIN_SIGMA,
            np.maximum(upper - lower, 1.0),
        )

    return {
        "best_so_far_x": best_x,
        "best_so_far_y": best_y,
        "n_function_evaluations": evaluations,
        "population_size": population_size,
        "sigma_mean": float(np.mean(sigma)),
        "sigma_max": float(np.max(sigma)),
        "success": bool(np.isfinite(best_y)),
    }


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
    if is_bounded_late_nda_refresh_action(action_name):
        return "bounded_guarded_incumbent_refresh"
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
    if is_bounded_late_nda_refresh_action(action_name):
        return "bounded_late_nda_refresh_and_cc_continuation"
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
) -> dict[str, str]:
    canonical_action_name = canonical_action_name or selected_action_name
    action_family = action_family or _action_family_for_canonical(canonical_action_name)
    state_mutated_value = (
        _state_mutated(selected_action_name)
        if state_mutated is None
        else str(int(state_mutated))
    )
    return {
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
    }


def _write_action_trace(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTION_TRACE_FIELDS)
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


def run_problem(fun_name: str, fun_id: int, output_path: Path, config: SmokeConfig) -> tuple[list[float], float, list[dict[str, str]]]:
    if config.budget_accounting not in {"strict", "source"}:
        raise ValueError(f"unsupported budget accounting mode: {config.budget_accounting}")
    time_start = time.time()
    bench = Benchmark(str(output_path) + "/", data_dir=config.aob_data_root)
    fun = bench.get_function(fun_name, fun_id)
    info = bench.get_info(fun_name, fun_id)
    problem_id = _problem_id(fun_name, fun_id)
    grouping_result = decompose_problem(fun_id, config.aob_data_root)
    _, overlap_groups, overlapping_elements = remove_overlapping_groups(grouping_result)
    metadata = load_aob_metadata(fun_id, config.aob_data_root)
    degree = calculate_degree_of_overlap(overlap_groups, metadata["dimension"])
    global_fes = calculate_global_fes(config.max_fes, degree)
    controller_v31_run_state = (
        build_evidence_action_controller_v31_run_state(degree)
        if is_evidence_action_controller_v31(config.arac_action)
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
            phase_i_results = MMES(problem, options).optimize()
            best_individual = np.asarray(phase_i_results["best_so_far_x"], dtype=float).copy()
            phase_i_fitness = float(phase_i_results["best_so_far_y"])
            global_phase_fe = int(phase_i_results["n_function_evaluations"])
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
    group_stagnation_counts = [0 for _ in grouping_result]
    bipop_global_cooldown = 0
    bipop_restart_count = 0
    bipop_rejected_restart_streak = 0
    guarded_incumbent = best_individual.copy()
    guarded_incumbent_fitness = math.inf
    cc_harm_guard_consumed = False
    bounded_refresh_completion: dict[str, object] | None = None
    evidence_controller_search_state_enabled = (
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
        phase_i_optimizer = MMES(problem, options)
        if controller_v31_run_state is None:
            results = phase_i_optimizer.optimize()
        else:
            results, phase_i_state = phase_i_optimizer.optimize_with_state()
            controller_v31_run_state.phase_i_optimizer = phase_i_optimizer
            controller_v31_run_state.phase_i_state = phase_i_state
        best_individual = results["best_so_far_x"].copy()
        guarded_incumbent = best_individual.copy()
        guarded_incumbent_fitness = float(results["best_so_far_y"])
        global_phase_fe = int(results["n_function_evaluations"])
        sum_fes += global_phase_fe
    elif is_cc_harm_guarded_sep_refresh_action(config.arac_action):
        guarded_incumbent_fitness = float(fun(best_individual)[0])

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
        sub_fes = math.ceil((config.max_fes - current_fes) / sub_num)
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
        current_outer_relations: list[OverlapRelation] = []
        current_outer_decisions: list[RelationActionDecision] = []
        optimized_any_group = False
        outer_stagnation_streak = 0
        for index, dims in enumerate(grouping_result):
            population_size = population_sizes[index]
            if (
                config.budget_accounting == "strict"
                and config.max_fes - current_fitness_evaluations(fun) <= population_size
            ):
                break
            original_best = best_individual.copy()
            original_fitness = float(fun(best_individual)[0])
            if config.budget_accounting == "source":
                optimizer_budget = sub_fes
            else:
                requested_fes = max(sub_fes, population_size)
                if trajectory_budgets:
                    requested_fes = max(trajectory_budgets[index], population_size)
                optimizer_budget = bounded_population_budget(
                    requested_fes=requested_fes,
                    remaining_fes=config.max_fes - current_fitness_evaluations(fun),
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
            cc_sigma = refine_sigma_for_action(
                config.arac_action,
                config.sigma,
                controller_v31_run_state=controller_v31_run_state,
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
            results_cc = CMAES(problem_cc, options_cc).optimize()
            optimized_any_group = True
            primary_cc_fe = int(results_cc["n_function_evaluations"])
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
                    restart_results = CMAES(problem_cc, restart_options).optimize()
                    bipop_restart_count += 1
                    restart_fe = int(restart_results["n_function_evaluations"])
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
            if uses_phase_rescue_during_run(
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
                        rescue_results = CMAES(problem_cc, rescue_options).optimize()
                        total_rescue_fes += int(rescue_results["n_function_evaluations"])
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
                        )
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
                    if adjusted_values is not None:
                        best_individual[context.overlap_indices] = adjusted_values
                    overlap_writeback_norms.append(action_value_delta_norm)
                    canonical_action_name = _canonical_relation_action_name(action)
                    current_outer_relations.append(relation)
                    current_outer_decisions.append(action)
                    relations.append(relation)
                    action_decisions.append(action)
                    action_trace_rows.append(
                        build_action_trace_row(
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
                            state_mutated=adjusted_values is not None,
                            action_value_delta_norm=action_value_delta_norm,
                            downstream_consumed=index < sub_num - 1,
                        )
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
            if (
                controller_v31_run_state is not None
                and not is_evidence_action_controller_v31(config.arac_action)
                and not controller_v31_run_state.bounded_late_nda_refresh_consumed
                and current_outer_relations
            ):
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
                full_population_size = calculate_cmaes_population_size(
                    int(info["dimension"])
                )
                bounded_refresh_plan = plan_bounded_late_nda_refresh(
                    controller_v31_run_state=controller_v31_run_state,
                    current_outer_relations=current_outer_relations,
                    fitness_deltas=fitness_delta_list,
                    overlap_writeback_norms=overlap_writeback_norms,
                    reference_fitness=guard_fitness,
                    remaining_fes=remaining_fes,
                    max_fes=config.max_fes,
                    population_size=full_population_size,
                    expected_group_count=sub_num,
                )
                if bounded_refresh_plan is not None:
                    refresh_seed = (
                        derive_optimizer_seed(
                            config.seed,
                            fun_name,
                            fun_id,
                            outer_iter + 1,
                            23011,
                        )
                        if config.seed is not None
                        else None
                    )
                    (
                        accepted,
                        refreshed_individual,
                        refreshed_best,
                        used_fes,
                        candidate_best,
                    ) = run_guarded_nda_continuation(
                        fun=fun,
                        info=info,
                        config=config,
                        fun_name=fun_name,
                        fun_id=fun_id,
                        outer_iter=outer_iter,
                        guard_individual=guard_individual,
                        guard_fitness=guard_fitness,
                        remaining_fes=remaining_fes,
                        requested_fes=bounded_refresh_plan.refresh_budget,
                        search_state_action=BOUNDED_LATE_NDA_REFRESH_ACTION,
                    )
                    sum_fes += used_fes
                    refresh_fe += used_fes
                    best_individual = refreshed_individual.copy()
                    guarded_incumbent = best_individual.copy()
                    guarded_incumbent_fitness = refreshed_best
                    controller_v31_run_state.bounded_late_nda_refresh_consumed = True
                    action_trace_rows.append(
                        build_action_trace_row(
                            problem_id=_problem_id(fun_name, fun_id),
                            seed=config.seed,
                            outer_iter=outer_iter,
                            group_index=index,
                            selected_action_name=BOUNDED_LATE_NDA_REFRESH_ACTION,
                            overlap_size=bounded_refresh_plan.shared_var_count,
                            previous_delta=sum(
                                max(0.0, delta) for delta in fitness_delta_list
                            ),
                            current_delta=max(0.0, guard_fitness - refreshed_best),
                            state_mutated=accepted,
                            action_value_delta_norm=float(
                                np.linalg.norm(refreshed_individual - guard_individual)
                            ),
                            downstream_consumed=True,
                            downstream_consumption_scope=(
                                "subsequent_outer_iterations"
                            ),
                            search_state_action_type=BOUNDED_LATE_NDA_REFRESH_ACTION,
                            stagnation_window=sum(
                                1
                                for delta in fitness_delta_list
                                if group_delta_stagnated(delta, guard_fitness)
                            ),
                            delta_mean=0.0,
                            sigma_before=config.sigma,
                            sigma_after=(
                                float(config.sigma)
                                * CC_HARM_REFRESH_SIGMA_MULTIPLIER
                            ),
                            population_before=full_population_size,
                            population_after=full_population_size,
                            escape_budget=used_fes,
                            bipop_restart_mode=(
                                "bounded_late_nda_refresh:start:"
                                f"{guard_source}:{bounded_refresh_plan.trigger_reason}"
                            ),
                            restart_triggered=True,
                            restart_accepted=accepted,
                            best_before=guard_fitness,
                            restart_candidate_best=candidate_best,
                            restart_relative_improvement=bipop_relative_improvement(
                                candidate_best=candidate_best,
                                incumbent_fitness=guard_fitness,
                            ),
                            restart_acceptance_threshold=0.0,
                            best_after=refreshed_best,
                            trace_event="start",
                            remaining_budget_ratio=(
                                bounded_refresh_plan.remaining_budget_ratio
                            ),
                            shared_var_count=bounded_refresh_plan.shared_var_count,
                            repair_lock_active=(
                                controller_v31_run_state.non_dense_repair_locked
                            ),
                            refresh_budget=bounded_refresh_plan.refresh_budget,
                            continuation_reserve=(
                                bounded_refresh_plan.continuation_reserve
                            ),
                            optimizer_seed=refresh_seed,
                        )
                    )
                    bounded_refresh_completion = {
                        "outer_iter": outer_iter,
                        "group_index": index,
                        "best_after_refresh": refreshed_best,
                        "remaining_budget_ratio": (
                            bounded_refresh_plan.remaining_budget_ratio
                        ),
                        "shared_var_count": bounded_refresh_plan.shared_var_count,
                        "repair_lock_active": (
                            controller_v31_run_state.non_dense_repair_locked
                        ),
                        "refresh_budget": bounded_refresh_plan.refresh_budget,
                        "continuation_reserve": (
                            bounded_refresh_plan.continuation_reserve
                        ),
                        "optimizer_seed": refresh_seed,
                    }
                    break
            if uses_cc_harm_guard_during_run(
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
            and optimized_any_group
            and len(fitness_delta_list) == sub_num
        ):
            phase_state = controller_v31_run_state.phase_i_state
            phase_optimizer = controller_v31_run_state.phase_i_optimizer
            phase_state_available = phase_state is not None and phase_optimizer is not None
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
                    controller_v31_run_state.phase_rescue_enabled
                    and phase_state_available
                ),
                repair_lock_active=controller_v31_run_state.non_dense_repair_locked,
                phase_i_tail_utility_value=(
                    phase_i_tail_utility(phase_state)
                    if phase_state is not None
                    else 0.0
                ),
                relations=current_outer_relations,
                decisions=current_outer_decisions,
                writeback_norms=overlap_writeback_norms,
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
            )
            if (
                state_plan.action_name == RESUME_PHASE_I_SEARCH_STATE
                and state_plan.requested_fes > 0
            ):
                if not phase_state_available:
                    raise RuntimeError(
                        "stateful MMES action selected without a resumable Phase-I state"
                    )
                guard_before = guarded_incumbent_fitness
                guard_vector = guarded_incumbent.copy()
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
                actual_state_fes = int(block.actual_fes)
                search_state_fe += actual_state_fes
                sum_fes += actual_state_fes
                best_individual = state_candidate.copy()
                guarded_incumbent = state_candidate.copy()
                guarded_incumbent_fitness = state_candidate_fitness
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
                raw_candidate = np.asarray(
                    block.state.best_so_far_x,
                    dtype=float,
                ).reshape(-1)
                action_trace_rows.append(
                    build_action_trace_row(
                        problem_id=_problem_id(fun_name, fun_id),
                        seed=config.seed,
                        outer_iter=outer_iter,
                        group_index=sub_num - 1,
                        selected_action_name=RESUME_PHASE_I_SEARCH_STATE,
                        overlap_size=0,
                        previous_delta=cc_utility,
                        current_delta=max(0.0, guard_before - state_candidate_fitness),
                        state_mutated=accepted,
                        action_value_delta_norm=float(
                            np.linalg.norm(raw_candidate - guard_vector)
                        ),
                        downstream_consumed=True,
                        downstream_consumption_scope="subsequent_outer_iterations",
                        search_state_action_type=RESUME_PHASE_I_SEARCH_STATE,
                        stagnation_window=0,
                        delta_mean=float(np.linalg.norm(raw_candidate - guard_vector)),
                        sigma_before=float(getattr(phase_state, "sigma", config.sigma)),
                        sigma_after=float(
                            getattr(next_phase_state, "sigma", config.sigma)
                        ),
                        population_before=int(
                            getattr(phase_state, "n_individuals", phase_population_size)
                        ),
                        population_after=int(
                            getattr(
                                next_phase_state,
                                "n_individuals",
                                phase_population_size,
                            )
                        ),
                        escape_budget=actual_state_fes,
                        bipop_restart_mode=state_plan.stage,
                        restart_triggered=True,
                        restart_accepted=accepted,
                        best_before=guard_before,
                        restart_candidate_best=float(block.state.best_so_far_y),
                        restart_relative_improvement=bipop_relative_improvement(
                            candidate_best=float(block.state.best_so_far_y),
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
                    )
                )

        previous_group_contribution_credit = fitness_delta_list
        outer_iter += 1
        if cc_harm_guard_consumed:
            break

    problem_id = _problem_id(fun_name, fun_id)
    if bounded_refresh_completion is not None:
        best_after_refresh = float(
            bounded_refresh_completion["best_after_refresh"]
        )
        completion_best = min(float(value) for value in fun.fitness_record)
        continuation_improved = completion_best < best_after_refresh
        action_trace_rows.append(
            build_action_trace_row(
                problem_id=problem_id,
                seed=config.seed,
                outer_iter=int(bounded_refresh_completion["outer_iter"]),
                group_index=int(bounded_refresh_completion["group_index"]),
                selected_action_name=BOUNDED_LATE_NDA_REFRESH_ACTION,
                overlap_size=0,
                previous_delta=0.0,
                current_delta=max(0.0, best_after_refresh - completion_best),
                state_mutated=False,
                action_value_delta_norm=0.0,
                downstream_consumed=True,
                downstream_consumption_scope="run_completion",
                search_state_action_type=BOUNDED_LATE_NDA_REFRESH_ACTION,
                bipop_restart_mode="bounded_late_nda_refresh:completion",
                restart_triggered=False,
                restart_accepted=continuation_improved,
                best_before=best_after_refresh,
                restart_candidate_best=completion_best,
                restart_relative_improvement=bipop_relative_improvement(
                    candidate_best=completion_best,
                    incumbent_fitness=best_after_refresh,
                ),
                restart_acceptance_threshold=0.0,
                best_after=min(best_after_refresh, completion_best),
                trace_event="completion",
                remaining_budget_ratio=float(
                    bounded_refresh_completion["remaining_budget_ratio"]
                ),
                shared_var_count=int(
                    bounded_refresh_completion["shared_var_count"]
                ),
                repair_lock_active=bool(
                    bounded_refresh_completion["repair_lock_active"]
                ),
                refresh_budget=int(
                    bounded_refresh_completion["refresh_budget"]
                ),
                continuation_reserve=int(
                    bounded_refresh_completion["continuation_reserve"]
                ),
                optimizer_seed=(
                    None
                    if bounded_refresh_completion["optimizer_seed"] is None
                    else int(bounded_refresh_completion["optimizer_seed"])
                ),
            )
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
            "handled by experiments/exp_003_hcc_runtime_consumer_smoke/run.py."
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
        ],
    )
    args = parser.parse_args(argv)
    if args.arac_action_file is not None:
        parser.error("--arac-action-file is not supported by the smoke runner yet")
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
        _write_action_trace(output_path / "action_trace.csv", function_trace_rows)
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
