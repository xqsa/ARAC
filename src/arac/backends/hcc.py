"""HCC backbone extraction helpers.

This module is the first clean ARAC extraction layer for the historical
``E:\\HCC-main`` work. It models the data ARAC needs from HCC grouping and
optimization traces without importing legacy milestone runners or mutating the
HCC baseline.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from arac.action_space import ActionFamily
from arac.backend_adapter import BackendSemanticsDiff
from arac.evidence import EvidenceProfile, validate_runtime_payload
from arac.policy import ActionDecision

DEFAULT_HCC_MAIN_ROOT = Path("E:/HCC-main")
ARAC_REPO_ROOT = Path(__file__).resolve().parents[3]
ARAC_HCC_SMOKE_RUNNER = ARAC_REPO_ROOT / "HCC_SRC" / "arac_hcc_smoke_runner.py"
DEFAULT_AOB_DATA_ROOT = ARAC_REPO_ROOT / "HCC_SRC" / "AOB" / "AOBG" / "datafile"
TOTAL_AOB_FE = 3_000_000
AOB_FUNCTION_NAMES = {
    "E": "elliptic",
    "S": "schwefel",
    "R": "rastrigin",
    "A": "ackley",
}


@dataclass(frozen=True)
class HccGroupSignal:
    """Reference-blind signal exposed by one HCC decomposition group."""

    group_id: str
    fitness_delta: float
    rank: int
    shared_variable_count: int = 0


@dataclass(frozen=True)
class HccBackboneSnapshot:
    """Minimal HCC grouping/optimization state needed by ARAC.

    The snapshot deliberately excludes final error, oracle labels, reported
    baselines, problem-family labels, and prior outcome fields. ``problem_id``
    is retained only as execution identity and artifact grouping.
    """

    run_id: str
    problem_id: str
    seed: int
    dimension: int
    group_count: int
    overlap_group_count: int
    overlapping_element_count: int
    budget_remaining_ratio: float
    groups: tuple[HccGroupSignal, ...]
    runtime_payload_extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HccAobCaseTopology:
    """AOB topology read from the source HCC benchmark files.

    This is a source-grounded grouping probe only. It does not run MMES/CMAES,
    does not read final errors, and does not use paper-reported baselines.
    """

    problem_id: str
    function_name: str
    function_id: int
    dimension: int
    dimension_real: int
    overlap_gamma: int
    group_count: int
    overlap_group_count: int
    overlapping_element_count: int
    degree_of_overlap: float
    global_fes: int
    groups: tuple[HccGroupSignal, ...]
    source_level: str = "hcc_source_topology"
    fresh_optimizer_execution: bool = False

    def to_snapshot(
        self,
        *,
        run_id: str,
        seed: int,
        budget_remaining_ratio: float,
    ) -> HccBackboneSnapshot:
        return HccBackboneSnapshot(
            run_id=run_id,
            problem_id=self.problem_id,
            seed=seed,
            dimension=self.dimension,
            group_count=self.group_count,
            overlap_group_count=self.overlap_group_count,
            overlapping_element_count=self.overlapping_element_count,
            budget_remaining_ratio=budget_remaining_ratio,
            groups=self.groups,
            runtime_payload_extra={
                "benchmark": "AOB",
                "aob_function_id": self.function_id,
                "dimension_real": self.dimension_real,
                "overlap_gamma": self.overlap_gamma,
                "degree_of_overlap": self.degree_of_overlap,
                "global_fes": self.global_fes,
                "source_level": self.source_level,
                "fresh_optimizer_execution": int(self.fresh_optimizer_execution),
            },
        )


@dataclass(frozen=True)
class HccAobSmokeCommand:
    """Subprocess command for a bounded HCC-main smoke execution."""

    argv: tuple[str, ...]
    cwd: Path


@dataclass(frozen=True)
class HccAobExecutionRequest:
    """Request for a single AOB/HCC smoke execution.

    Full 3M-FE, 24-case pilots should be scheduled explicitly by experiment
    code. This request is intentionally single-case to keep HCC-main execution
    bridged through a narrow, auditable boundary.
    """

    problem_id: str
    seed: int
    max_fes: int
    output_dir: Path
    hcc_root: Path = DEFAULT_HCC_MAIN_ROOT
    aob_data_root: Path = DEFAULT_AOB_DATA_ROOT
    python_executable: str = "python"
    timestamp: str = "arac-hcc-smoke"
    config_name: str = "quick_smoke"
    arac_action: str = "conservative_no_action"
    enable_relation_dispatch: bool = False
    relation_policy_mode: str = "rule"
    arac_action_file: Path | None = None
    budget_accounting: str = "strict"
    cmaes_restart: bool = True
    mmes_restart: bool = True
    skip_plots: bool = False


@dataclass(frozen=True)
class HccAobExecutionResult:
    """Offline-only result from a fresh HCC optimizer smoke execution."""

    problem_id: str
    seed: int
    max_fes: int
    final_error: float
    fe_used: int
    time_seconds: float
    output_root: Path
    fresh_optimizer_execution: bool
    status: str
    result_source: str
    stdout_tail: str = ""
    stderr_tail: str = ""
    action_trace_path: Path | None = None
    action_trace_rows: int = 0
    optimizer_final_fe_used: int | None = None
    global_phase_fe: int | None = None
    cc_phase_fe: int | None = None
    rescue_fe: int | None = None
    refresh_fe: int | None = None
    separable_continuation_fe: int | None = None
    overhead_fe: int | None = None

    def to_offline_row(self) -> dict[str, str]:
        actual_fe_used = (
            self.fe_used
            if self.optimizer_final_fe_used is None
            else self.optimizer_final_fe_used
        )
        return {
            "problem_id": self.problem_id,
            "seed": str(self.seed),
            "max_fes": str(self.max_fes),
            "final_error": f"{self.final_error:.6e}",
            "fe_used": str(self.fe_used),
            "optimizer_final_fe_used": str(actual_fe_used),
            "time_seconds": f"{self.time_seconds:.6f}",
            "output_root": str(self.output_root),
            "fresh_optimizer_execution": "1" if self.fresh_optimizer_execution else "0",
            "status": self.status,
            "result_source": self.result_source,
            "action_trace_path": "" if self.action_trace_path is None else str(self.action_trace_path),
            "action_trace_rows": str(self.action_trace_rows),
            "runtime_dispatch_allowed": "0",
            "same_budget_violation": "1" if actual_fe_used > self.max_fes else "0",
            "performance_claim_allowed": "0",
        }


@dataclass(frozen=True)
class HccActionExecutionPlan:
    """Audit row describing whether an ARAC action reaches HCC runtime."""

    problem_id: str
    selected_action_name: str
    selected_action_family: str
    backend_effect_kind: str
    optimizer_consumed: bool
    optimizer_consumed_parameters: dict[str, object]
    execution_mode: str
    blocker_reason: str
    runtime_dispatch_allowed: bool

    def to_csv_row(self) -> dict[str, str]:
        return {
            "problem_id": self.problem_id,
            "selected_action_name": self.selected_action_name,
            "selected_action_family": self.selected_action_family,
            "backend_effect_kind": self.backend_effect_kind,
            "optimizer_consumed": "1" if self.optimizer_consumed else "0",
            "optimizer_consumed_parameters": _format_json_parameters(
                self.optimizer_consumed_parameters
            ),
            "execution_mode": self.execution_mode,
            "blocker_reason": self.blocker_reason,
            "runtime_dispatch_allowed": "1" if self.runtime_dispatch_allowed else "0",
        }


HCC_ACTION_EFFECTS = {
    "conservative_no_action": (
        "no_op_safe_fallback",
        {"backend": "repo_default_hcc_no_action"},
        "hcc_noop_baseline",
        True,
        "",
    ),
    "isolate_conflicting_relation": (
        "shared_variable_value_selection",
        {"runtime_hook": "overlap_value_selection_rule"},
        "hcc_relation_value_selection_consumed",
        True,
        "",
    ),
    "protect_high_margin_group": (
        "protect_resource_priority",
        {},
        "audit_only_not_executed",
        False,
        "no_hcc_runtime_consumer_yet",
    ),
    "budget_shift_mean_blend": (
        "optimizer_budget_and_mean_trajectory",
        {"runtime_hook": "budget_shift_mean_blend"},
        "hcc_trajectory_runtime_consumed",
        True,
        "",
    ),
    "bipop_search_state_restart": (
        "optimizer_search_state_restart",
        {"runtime_hook": "bipop_search_state_restart"},
        "hcc_search_state_runtime_consumed",
        True,
        "",
    ),
    "phase_rescue_multistart": (
        "optimizer_phase_rescue_multistart",
        {
            "runtime_hook": "phase_rescue_multistart",
            "acceptance_rule": "best_improving_candidate_only",
        },
        "hcc_phase_rescue_runtime_consumed",
        True,
        "",
    ),
    "repair_phase_rescue_multistart": (
        "repair_guided_phase_rescue_multistart",
        {
            "overlap_runtime_hook": "overlap_repair_rule",
            "search_state_runtime_hook": "phase_rescue_multistart",
            "acceptance_rule": "best_improving_candidate_only",
        },
        "hcc_repair_phase_rescue_runtime_consumed",
        True,
        "",
    ),
    "cc_harm_guarded_sep_refresh": (
        "cc_harm_guarded_sep_or_nda_refresh",
        {
            "runtime_hook": "cc_harm_guarded_sep_refresh",
            "guard": "phase_i_or_current_incumbent_no_harm",
            "refresh_backend": "full_space_mmes_nda_continuation",
            "acceptance_rule": "guarded_incumbent_improving_candidate_only",
        },
        "hcc_cc_harm_guarded_refresh_runtime_consumed",
        True,
        "",
    ),
    "separable_cmaes_dispatch_action": (
        "full_space_diagonal_separable_search_takeover",
        {
            "runtime_hook": "separable_cmaes_dispatch_action",
            "backend": "direct_separable_cmaes",
            "search_distribution": "diagonal_sigma_full_space",
            "acceptance_rule": "optimizer_best_so_far",
        },
        "hcc_direct_separable_cmaes_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v1": (
        "evidence_action_runtime_controller",
        {
            "relation_runtime_hook": "adaptive_v26_relation_dispatch",
            "overlap_runtime_hook": "evidence_triggered_overlap_action",
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
                "cc_harm_guarded_sep_refresh",
            ],
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v2": (
        "evidence_action_runtime_controller_v2",
        {
            "relation_runtime_hook": "adaptive_v24_relation_dispatch",
            "overlap_runtime_hook": "relation_first_evidence_triggered_overlap_action",
            "search_state_runtime_hooks": [],
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_v2_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v3": (
        "evidence_action_runtime_controller_v3",
        {
            "relation_runtime_hook": "controller_v3_relation_dispatch",
            "mode_selector": "early_runtime_overlap_relation_evidence",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
                "cc_harm_guarded_sep_refresh",
            ],
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_v3_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v31": (
        "evidence_action_runtime_controller_v31",
        {
            "relation_runtime_hook": "controller_v31_guarded_relation_dispatch",
            "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
                "cc_harm_guarded_sep_refresh",
            ],
            "guard": "stable_relation_first_no_harm_gate",
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_v31_runtime_consumed",
        True,
        "",
    ),
    "repair_bipop_search_state_restart": (
        "repair_guided_optimizer_search_state_restart",
        {
            "overlap_runtime_hook": "overlap_repair_rule",
            "search_state_runtime_hook": "bipop_search_state_restart",
        },
        "hcc_repair_bipop_runtime_consumed",
        True,
        "",
    ),
    "repair_protect_refine": (
        "repair_guided_local_refinement",
        {
            "overlap_runtime_hook": "overlap_repair_rule",
            "optimizer_runtime_hook": "protected_small_sigma_refine",
        },
        "hcc_repair_refine_runtime_consumed",
        True,
        "",
    ),
    "repair_protect_deep_refine": (
        "repair_guided_deep_local_refinement",
        {
            "overlap_runtime_hook": "overlap_repair_rule",
            "optimizer_runtime_hook": "protected_deep_sigma_refine",
        },
        "hcc_repair_deep_refine_runtime_consumed",
        True,
        "",
    ),
    "repair_shared_variable_binding": (
        "shared_variable_owner_rebinding",
        {"runtime_hook": "overlap_repair_rule"},
        "hcc_smoke_runtime_consumed",
        True,
        "",
    ),
    "allow_beneficial_coordination": (
        "coordination_mode_switch",
        {"runtime_hook": "overlap_clipped_consensus_blend"},
        "hcc_relation_runtime_consumed",
        True,
        "",
    ),
}


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _datafile_dir(hcc_root: Path) -> Path:
    return hcc_root / "HCC_SRC" / "AOB" / "AOBG" / "datafile"


def _parse_aob_info(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith("["):
            values[key] = ast.literal_eval(value)
            continue
        try:
            number = float(value)
        except ValueError:
            values[key] = value
            continue
        values[key] = int(number) if number.is_integer() else number
    return values


def required_aob_data_files(data_root: Path | str, function_id: int) -> tuple[Path, ...]:
    """Return the files consumed by one AOB function id."""

    root = Path(data_root).resolve()
    prefix = f"F{int(function_id)}"
    info_path = root / f"{prefix}-info.txt"
    files = [
        info_path,
        root / f"{prefix}-design.txt",
        root / f"{prefix}-p.txt",
        root / f"{prefix}-s.txt",
        root / f"{prefix}-w.txt",
        root / f"{prefix}-xopt.txt",
    ]
    if info_path.is_file():
        info = _parse_aob_info(info_path)
        rotation_sizes = {int(size) for size in info.get("subgroups_type", [])}
        files.extend(root / f"{prefix}-R{size}.txt" for size in sorted(rotation_sizes))
    return tuple(files)


def validate_aob_data_root(data_root: Path | str, function_id: int) -> Path:
    """Fail before optimizer execution when canonical AOB inputs are incomplete."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AOB data root does not exist: {root}")
    missing = [path.name for path in required_aob_data_files(root, function_id) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"AOB data root is incomplete for F{function_id}: {root}; "
            f"missing={','.join(missing)}"
        )
    return root


def _read_permutation(path: Path) -> list[int]:
    values = []
    for chunk in path.read_text(encoding="utf-8").replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(float(chunk)) - 1)
    return values


def _topology_groups(info: dict[str, object], permutation: list[int]) -> list[list[int]]:
    overlap = int(info["overlap_degree"])
    groups: list[list[int]] = []
    begin_index = 0
    for index, subgroup_size in enumerate(info["subgroups"]):
        size = int(subgroup_size)
        end_index = begin_index + size
        groups.append(permutation[begin_index:end_index])
        if index != len(info["subgroups"]) - 1:
            begin_index = end_index - overlap
    return groups


def _overlap_groups(groups: list[list[int]]) -> list[list[int]]:
    return [sorted(set(left) & set(right)) for left, right in zip(groups, groups[1:])]


def _calculate_global_fes(total_fes: int, degree_of_overlap: float) -> int:
    if degree_of_overlap == 0:
        return 0
    return int((0.2 + (4 / 5) * degree_of_overlap) * total_fes)


def _problem_parts(problem_id: str) -> tuple[str, str, int]:
    problem = str(problem_id).strip().upper()
    if len(problem) != 2 or problem[0] not in AOB_FUNCTION_NAMES or not problem[1].isdigit():
        raise ValueError(f"unsupported AOB problem_id: {problem_id}")
    function_id = int(problem[1])
    if function_id < 1 or function_id > 6:
        raise ValueError(f"unsupported AOB function id: {problem_id}")
    return problem, AOB_FUNCTION_NAMES[problem[0]], function_id


def load_hcc_aob_topology(
    problem_id: str,
    hcc_root: Path | str = DEFAULT_HCC_MAIN_ROOT,
    total_fes: int = TOTAL_AOB_FE,
) -> HccAobCaseTopology:
    """Read source-grounded AOB/HCC grouping topology without optimizer execution."""

    problem, function_name, function_id = _problem_parts(problem_id)
    data_dir = _datafile_dir(Path(hcc_root))
    info = _parse_aob_info(data_dir / f"F{function_id}-info.txt")
    permutation = _read_permutation(data_dir / f"F{function_id}-p.txt")
    topology_groups = _topology_groups(info, permutation)
    overlaps = _overlap_groups(topology_groups)
    overlapping_elements = {element for group in overlaps for element in group}
    dimension = int(info["dimension"])
    degree = _safe_divide(len(overlapping_elements), dimension)
    group_signals = tuple(
        HccGroupSignal(
            group_id=f"source_group_{index + 1:02d}",
            fitness_delta=1.0 / (index + 1),
            rank=index + 1,
            shared_variable_count=sum(1 for element in group if element in overlapping_elements),
        )
        for index, group in enumerate(topology_groups)
    )

    return HccAobCaseTopology(
        problem_id=problem,
        function_name=function_name,
        function_id=function_id,
        dimension=dimension,
        dimension_real=int(info["dimension_real"]),
        overlap_gamma=int(info["overlap_degree"]),
        group_count=len(topology_groups),
        overlap_group_count=sum(1 for group in overlaps if group),
        overlapping_element_count=len(overlapping_elements),
        degree_of_overlap=degree,
        global_fes=_calculate_global_fes(total_fes, degree),
        groups=group_signals,
    )


def build_hcc_action_execution_plan(
    problem_id: str,
    decision: ActionDecision,
) -> HccActionExecutionPlan:
    """Describe whether an ARAC action is optimizer-consumed by HCC today."""

    effect = HCC_ACTION_EFFECTS.get(decision.action_name)
    if effect is None:
        backend_effect_kind = "unknown_action"
        parameters: dict[str, object] = {}
        execution_mode = "audit_only_not_executed"
        optimizer_consumed = False
        blocker = "unknown_hcc_action_binding"
    else:
        backend_effect_kind, parameters, execution_mode, optimizer_consumed, blocker = effect

    return HccActionExecutionPlan(
        problem_id=_problem_parts(problem_id)[0],
        selected_action_name=decision.action_name,
        selected_action_family=decision.action_family.value,
        backend_effect_kind=backend_effect_kind,
        optimizer_consumed=bool(optimizer_consumed),
        optimizer_consumed_parameters=dict(parameters),
        execution_mode=execution_mode,
        blocker_reason=blocker,
        runtime_dispatch_allowed=bool(optimizer_consumed),
    )


def build_hcc_aob_smoke_command(request: HccAobExecutionRequest) -> HccAobSmokeCommand:
    """Build the subprocess command used to run HCC-main from its own cwd."""

    problem, function_name, function_id = _problem_parts(request.problem_id)
    if request.max_fes <= 0:
        raise ValueError("max_fes must be positive")
    if request.seed < 0:
        raise ValueError("seed must be non-negative")
    if request.arac_action_file is not None:
        raise ValueError("arac_action_file is not supported by the HCC smoke runner yet")
    if request.budget_accounting not in {"strict", "source"}:
        raise ValueError("budget_accounting must be 'strict' or 'source'")
    aob_data_root = validate_aob_data_root(request.aob_data_root, function_id)

    argv = [
        request.python_executable,
        str(ARAC_HCC_SMOKE_RUNNER),
        "--functions",
        function_name,
        "--ids",
        str(function_id),
        "--seed",
        str(request.seed),
        "--max-fes",
        str(request.max_fes),
        "--output-root",
        str(request.output_dir),
        "--aob-data-root",
        str(aob_data_root),
        "--timestamp",
        request.timestamp,
        "--arac-action",
        request.arac_action,
        "--budget-accounting",
        request.budget_accounting,
    ]
    if request.enable_relation_dispatch:
        argv.append("--enable-relation-dispatch")
    if request.relation_policy_mode:
        argv.extend(("--relation-policy", request.relation_policy_mode))
    if not request.cmaes_restart:
        argv.append("--no-cmaes-restart")
    if not request.mmes_restart:
        argv.append("--no-mmes-restart")
    if request.skip_plots:
        argv.append("--skip-plots")
    return HccAobSmokeCommand(argv=tuple(argv), cwd=Path(request.hcc_root))


def run_hcc_aob_smoke_execution(request: HccAobExecutionRequest) -> HccAobExecutionResult:
    """Run one bounded HCC-main smoke execution and parse its offline result.

    The subprocess is run with ``cwd=E:\\HCC-main`` because the historical AOB
    benchmark uses relative data-file paths. Returned final-error fields are
    offline evaluation outputs and must not be copied into runtime evidence.
    """

    command = build_hcc_aob_smoke_command(
        HccAobExecutionRequest(
            problem_id=request.problem_id,
            seed=request.seed,
            max_fes=request.max_fes,
            output_dir=Path(request.output_dir),
            hcc_root=Path(request.hcc_root),
            aob_data_root=Path(request.aob_data_root),
            python_executable=request.python_executable or sys.executable,
            timestamp=request.timestamp,
            config_name=request.config_name,
            arac_action=request.arac_action,
            enable_relation_dispatch=request.enable_relation_dispatch,
            relation_policy_mode=request.relation_policy_mode,
            arac_action_file=request.arac_action_file,
            budget_accounting=request.budget_accounting,
            cmaes_restart=request.cmaes_restart,
            mmes_restart=request.mmes_restart,
            skip_plots=request.skip_plots,
        )
    )
    start = time.time()
    completed = subprocess.run(
        command.argv,
        cwd=command.cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    if completed.returncode != 0:
        return HccAobExecutionResult(
            problem_id=_problem_parts(request.problem_id)[0],
            seed=request.seed,
            max_fes=request.max_fes,
            final_error=float("nan"),
            fe_used=0,
            time_seconds=elapsed,
            output_root=Path(request.output_dir),
            fresh_optimizer_execution=False,
            status=f"failed_returncode_{completed.returncode}",
            result_source="hcc_subprocess_smoke_execution",
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
            action_trace_path=None,
            action_trace_rows=0,
        )

    final_error, fe_used, optimizer_final_fe_used = (
        _parse_hcc_evaluation_record_with_optimizer_final_fe(
            Path(request.output_dir),
            budget_limit=request.max_fes,
        )
    )
    action_trace_path, action_trace_rows = _find_hcc_action_trace(Path(request.output_dir))
    budget_breakdown = _parse_hcc_budget_summary(Path(request.output_dir))
    return HccAobExecutionResult(
        problem_id=_problem_parts(request.problem_id)[0],
        seed=request.seed,
        max_fes=request.max_fes,
        final_error=final_error,
        fe_used=fe_used,
        time_seconds=elapsed,
        output_root=Path(request.output_dir),
        fresh_optimizer_execution=True,
        status="completed",
        result_source="hcc_subprocess_smoke_execution",
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        action_trace_path=action_trace_path,
        action_trace_rows=action_trace_rows,
        optimizer_final_fe_used=optimizer_final_fe_used,
        global_phase_fe=budget_breakdown.get("global_phase_fe"),
        cc_phase_fe=budget_breakdown.get("cc_phase_fe"),
        rescue_fe=budget_breakdown.get("rescue_fe"),
        refresh_fe=budget_breakdown.get("refresh_fe"),
        separable_continuation_fe=budget_breakdown.get(
            "separable_continuation_fe"
        ),
        overhead_fe=budget_breakdown.get("overhead_fe"),
    )


def _parse_hcc_evaluation_record(
    output_dir: Path,
    budget_limit: int | None = None,
) -> tuple[float, int]:
    final_error, fe_used, _optimizer_final_fe_used = (
        _parse_hcc_evaluation_record_with_optimizer_final_fe(
            output_dir,
            budget_limit=budget_limit,
        )
    )
    return final_error, fe_used


def _parse_hcc_evaluation_record_with_optimizer_final_fe(
    output_dir: Path,
    budget_limit: int | None = None,
) -> tuple[float, int, int]:
    records = sorted(Path(output_dir).rglob("evaluation_record.txt"))
    if not records:
        raise FileNotFoundError(f"missing HCC evaluation_record.txt under {output_dir}")
    text = records[-1].read_text(encoding="utf-8", errors="replace")
    final_match = re.search(
        r"Fin:\s*(?P<fe>[0-9.eE+-]+)\s+(?P<value>[0-9.eE+-]+)",
        text,
    )
    if not final_match:
        raise ValueError(f"could not parse final HCC error from {records[-1]}")
    optimizer_final_fe_used = _parse_hcc_budget_summary_final_fe(output_dir)
    if optimizer_final_fe_used is None:
        optimizer_final_fe_used = int(float(final_match.group("fe")))
    if budget_limit is not None:
        for checkpoint in re.finditer(
            r"^\s*(?P<fe>[0-9.eE+-]+)\s+(?P<value>[0-9.eE+-]+)",
            text,
            flags=re.MULTILINE,
        ):
            fe = int(float(checkpoint.group("fe")))
            if fe == budget_limit:
                return float(checkpoint.group("value")), fe, optimizer_final_fe_used

    return (
        float(final_match.group("value")),
        optimizer_final_fe_used,
        optimizer_final_fe_used,
    )


def _parse_hcc_budget_summary_final_fe(output_dir: Path) -> int | None:
    summary = _parse_hcc_budget_summary(output_dir)
    for field in ("fitness_record_fe", "optimizer_reported_fe"):
        if field in summary:
            return summary[field]
    return None


def _parse_hcc_budget_summary(output_dir: Path) -> dict[str, int]:
    summaries = sorted(Path(output_dir).rglob("*budget_summary.csv"))
    if not summaries:
        return {}
    with summaries[-1].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[-1]
    parsed: dict[str, int] = {}
    for field in (
        "fitness_record_fe",
        "optimizer_reported_fe",
        "global_phase_fe",
        "cc_phase_fe",
        "rescue_fe",
        "refresh_fe",
        "separable_continuation_fe",
        "overhead_fe",
    ):
        value = row.get(field)
        if value not in (None, ""):
            parsed[field] = int(float(value))
    return parsed


def _find_hcc_action_trace(output_dir: Path) -> tuple[Path | None, int]:
    traces = sorted(Path(output_dir).rglob("action_trace.csv"))
    if not traces:
        return None, 0
    trace_path = traces[-1]
    with trace_path.open(newline="", encoding="utf-8") as handle:
        row_count = max(0, sum(1 for _ in handle) - 1)
    return trace_path, row_count


def _tail(text: str, max_chars: int = 2000) -> str:
    return (text or "")[-max_chars:]


def _format_json_parameters(parameters: dict[str, object]) -> str:
    if not parameters:
        return ""
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _rank_stability(groups: tuple[HccGroupSignal, ...]) -> float:
    if len(groups) <= 1:
        return 1.0
    ranks = [group.rank for group in groups]
    if min(ranks) < 1:
        return 0.0
    unique_ratio = len(set(ranks)) / len(ranks)
    return _clamp_ratio(unique_ratio)


def _priority_spread(groups: tuple[HccGroupSignal, ...]) -> float:
    if not groups:
        return 0.0
    ranks = [group.rank for group in groups]
    span = max(ranks) - min(ranks)
    return _clamp_ratio(_safe_divide(span, max(len(groups), 1)))


def _gain_asymmetry(groups: tuple[HccGroupSignal, ...]) -> float:
    if not groups:
        return 0.0
    gains = [max(0.0, group.fitness_delta) for group in groups]
    return _clamp_ratio(_safe_divide(max(gains) - min(gains), max(gains) + 1e-12))


def _direction_disagreement(groups: tuple[HccGroupSignal, ...]) -> float:
    if not groups:
        return 0.0
    positives = sum(1 for group in groups if group.fitness_delta > 0)
    non_positives = len(groups) - positives
    minority = min(positives, non_positives)
    return _clamp_ratio(_safe_divide(minority, len(groups)))


def build_hcc_evidence_profile(snapshot: HccBackboneSnapshot) -> EvidenceProfile:
    """Convert HCC grouping/trace state into a runtime-legal ARAC evidence row."""

    payload = {
        "run_id": snapshot.run_id,
        "problem_id": snapshot.problem_id,
        "seed": snapshot.seed,
        "dimension": snapshot.dimension,
        "group_count": snapshot.group_count,
        "overlap_group_count": snapshot.overlap_group_count,
        "overlapping_element_count": snapshot.overlapping_element_count,
        "budget_remaining_ratio": snapshot.budget_remaining_ratio,
        **snapshot.runtime_payload_extra,
    }
    validate_runtime_payload(payload)

    overlap_degree = _clamp_ratio(
        _safe_divide(snapshot.overlap_group_count, max(snapshot.group_count, 1))
    )
    shared_var_support_ratio = _clamp_ratio(
        _safe_divide(snapshot.overlapping_element_count, max(snapshot.dimension, 1))
    )
    group_gain_asymmetry = _gain_asymmetry(snapshot.groups)
    priority_spread = _priority_spread(snapshot.groups)
    direction_disagreement = _direction_disagreement(snapshot.groups)
    harmful_coord_score = _clamp_ratio(
        max(overlap_degree, shared_var_support_ratio) * max(group_gain_asymmetry, 0.1)
    )

    return EvidenceProfile(
        run_id=snapshot.run_id,
        problem_id=snapshot.problem_id,
        seed=snapshot.seed,
        unit_type="problem",
        unit_id=f"hcc_backbone:{snapshot.problem_id}",
        feature_coverage=1.0 if snapshot.groups else 0.5,
        overlap_degree=overlap_degree,
        shared_var_support_ratio=shared_var_support_ratio,
        direction_disagreement=direction_disagreement,
        harmful_coord_score=harmful_coord_score,
        group_gain_asymmetry=group_gain_asymmetry,
        priority_spread=priority_spread,
        rank_stability=_rank_stability(snapshot.groups),
        budget_remaining_ratio=_clamp_ratio(snapshot.budget_remaining_ratio),
        fallback_margin_proxy=_clamp_ratio(1.0 - harmful_coord_score),
    )


def hcc_backend_semantics_for(
    decision: ActionDecision,
    *,
    optimizer_consumed: bool,
) -> BackendSemanticsDiff:
    """Map clean ARAC actions onto HCC optimizer-consumed semantic surfaces."""

    if not optimizer_consumed:
        return BackendSemanticsDiff()
    if decision.action_family == ActionFamily.ISOLATE:
        return BackendSemanticsDiff(relation_handling_changed=True)
    if decision.action_family == ActionFamily.PROTECT:
        return BackendSemanticsDiff(budget_allocation_changed=True)
    if decision.action_family == ActionFamily.REASSIGN_REPAIR:
        return BackendSemanticsDiff(variable_owner_changed=True)
    if decision.action_family == ActionFamily.COORDINATE:
        return BackendSemanticsDiff(coordination_mode_changed=True)
    if decision.action_family == ActionFamily.TRAJECTORY:
        if decision.action_name in {"repair_protect_refine", "repair_protect_deep_refine"}:
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                budget_allocation_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name in {
            "repair_bipop_search_state_restart",
            "repair_phase_rescue_multistart",
        }:
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "cc_harm_guarded_sep_refresh":
            return BackendSemanticsDiff(
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "separable_cmaes_dispatch_action":
            return BackendSemanticsDiff(
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "arac_evidence_action_controller_v1":
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                relation_handling_changed=True,
                coordination_mode_changed=True,
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "arac_evidence_action_controller_v2":
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                relation_handling_changed=True,
                coordination_mode_changed=True,
            )
        if decision.action_name == "arac_evidence_action_controller_v3":
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                relation_handling_changed=True,
                coordination_mode_changed=True,
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "arac_evidence_action_controller_v31":
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                relation_handling_changed=True,
                coordination_mode_changed=True,
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name in {"bipop_search_state_restart", "phase_rescue_multistart"}:
            return BackendSemanticsDiff(
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        return BackendSemanticsDiff(budget_allocation_changed=True)
    return BackendSemanticsDiff()
