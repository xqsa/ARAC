"""HCC backbone extraction helpers.

This module is the first clean ARAC extraction layer for the historical
``E:\\HCC-main`` work. It models the data ARAC needs from HCC grouping and
optimization traces without importing legacy milestone runners or mutating the
HCC baseline.
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from arac.evidence import EvidenceProfile, validate_runtime_payload

from .hcc_budget import (
    _parse_hcc_budget_summary,
    _parse_hcc_budget_summary_final_fe,
    _parse_hcc_evaluation_record,
    _parse_hcc_evaluation_record_with_optimizer_final_fe,
)
from .hcc_plan import (
    HCC_ACTION_EFFECTS,
    HccActionExecutionPlan,
    build_hcc_action_execution_plan,
)
from .hcc_shared_writeback import hcc_backend_semantics_for
from .hcc_trace import _find_hcc_action_trace, _find_hcc_named_csv, _tail


ARAC_REPO_ROOT = Path(__file__).resolve().parents[3]
HCC_VENDOR_ROOT = (ARAC_REPO_ROOT / "vendor" / "hcc").resolve()


@dataclass(frozen=True)
class HccVendorPaths:
    """Canonical HCC source paths derived from one explicit vendor root."""

    vendor_root: Path
    aob_root: Path
    hcc_root: Path
    aob_data_root: Path
    runner: Path


def resolve_hcc_vendor_paths(
    vendor_root: Path | str,
    *,
    repo_root: Path | str | None = None,
    runner_path: Path | str | None = None,
) -> HccVendorPaths:
    root = Path(vendor_root)
    if not root.is_absolute():
        root = ARAC_REPO_ROOT / root
    root = root.resolve()
    if root.name.casefold() == "hcc-main":
        raise ValueError(
            f"external HCC-main roots are offline-only and are not a vendor root: {root}"
        )
    for required_dir in ("AOB", "HCC"):
        path = root / required_dir
        if not path.is_dir():
            raise FileNotFoundError(
                f"not a valid HCC vendor root: {root}; missing {required_dir} directory"
            )

    if runner_path is not None:
        runner = Path(runner_path)
        if not runner.is_absolute():
            runner = ARAC_REPO_ROOT / runner
    elif repo_root is not None:
        resolved_repo_root = Path(repo_root)
        if not resolved_repo_root.is_absolute():
            resolved_repo_root = ARAC_REPO_ROOT / resolved_repo_root
        runner = resolved_repo_root / "scripts" / "hcc_smoke_runner.py"
    elif root == HCC_VENDOR_ROOT:
        runner = ARAC_REPO_ROOT / "scripts" / "hcc_smoke_runner.py"
    else:
        raise ValueError(
            "non-canonical HCC vendor roots require an explicit repo_root or runner_path"
        )
    runner = runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"HCC smoke runner does not exist: {runner}")

    return HccVendorPaths(
        vendor_root=root,
        aob_root=root / "AOB",
        hcc_root=root / "HCC",
        aob_data_root=root / "AOB" / "AOBG" / "datafile",
        runner=runner,
    )
HCC_VENDOR_PATHS = resolve_hcc_vendor_paths(HCC_VENDOR_ROOT)
ARAC_HCC_SMOKE_RUNNER = HCC_VENDOR_PATHS.runner
DEFAULT_AOB_DATA_ROOT = HCC_VENDOR_PATHS.aob_data_root
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
    """Subprocess command for a bounded canonical HCC vendor smoke execution."""

    argv: tuple[str, ...]
    cwd: Path


EVIDENCE_OVERLAY_MODES = frozenset(
    {"off", "native_audit", "paired_owner", "shuffled_owner"}
)
EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class HccAobExecutionRequest:
    """Request for a single AOB/HCC smoke execution.

    Full 3M-FE, 24-case pilots should be scheduled explicitly by experiment
    code. This request is intentionally single-case to keep canonical HCC vendor execution
    bridged through a narrow, auditable boundary.
    """

    problem_id: str
    seed: int
    max_fes: int
    output_dir: Path
    hcc_root: Path = HCC_VENDOR_ROOT
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
    early_stopping_evaluations: int = 1000
    skip_plots: bool = False
    search_state_backend: str = "phase_i_mmes"
    car_candidate_mode: str = "graph"
    car_actionability_arm: str = "off"
    precision_causal_arm: str = "off"
    precision_response_arm: str = "off"
    component_precision_arm: str = "off"
    hypergraph_trace_mode: str = "off"
    evidence_overlay_mode: str = "off"
    cma_sampling_mode: str = "iid"
    offline_frozen_replay: bool = False
    hcc_repo_root: Path | None = None
    hcc_runner: Path | None = None


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
    mos_sampling_audit_path: Path | None = None
    mos_sampling_audit_rows: int = 0
    mos_branch_provenance_path: Path | None = None
    mos_branch_provenance_rows: int = 0
    optimizer_final_fe_used: int | None = None
    global_phase_fe: int | None = None
    cc_phase_fe: int | None = None
    rescue_fe: int | None = None
    refresh_fe: int | None = None
    search_state_fe: int | None = None
    precision_probe_fe: int | None = None
    evidence_overlay_fe: int | None = None
    separable_continuation_fe: int | None = None
    overhead_fe: int | None = None
    native_terminal_error: float | None = None
    all_evaluation_best_error: float | None = None
    evidence_overlay_manifest_path: Path | None = None
    evidence_overlay_status: str = "off"

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




def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _datafile_dir(
    hcc_root: Path,
    *,
    hcc_repo_root: Path | None = None,
    hcc_runner: Path | None = None,
) -> Path:
    return resolve_hcc_vendor_paths(
        hcc_root,
        repo_root=hcc_repo_root,
        runner_path=hcc_runner,
    ).aob_data_root


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
    hcc_root: Path | str = HCC_VENDOR_ROOT,
    total_fes: int = TOTAL_AOB_FE,
    *,
    hcc_repo_root: Path | str | None = None,
    hcc_runner: Path | str | None = None,
) -> HccAobCaseTopology:
    """Read source-grounded AOB/HCC grouping topology without optimizer execution."""

    problem, function_name, function_id = _problem_parts(problem_id)
    data_dir = _datafile_dir(
        Path(hcc_root),
        hcc_repo_root=None if hcc_repo_root is None else Path(hcc_repo_root),
        hcc_runner=None if hcc_runner is None else Path(hcc_runner),
    )
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

def build_hcc_aob_smoke_command(request: HccAobExecutionRequest) -> HccAobSmokeCommand:
    """Build the subprocess command from the canonical HCC vendor boundary."""

    problem, function_name, function_id = _problem_parts(request.problem_id)
    if request.max_fes <= 0:
        raise ValueError("max_fes must be positive")
    if request.seed < 0:
        raise ValueError("seed must be non-negative")
    if request.arac_action_file is not None:
        raise ValueError("arac_action_file is not supported by the HCC smoke runner yet")
    if request.budget_accounting not in {"strict", "source"}:
        raise ValueError("budget_accounting must be 'strict' or 'source'")
    if request.search_state_backend not in {"phase_i_mmes", "diagonal_cma"}:
        raise ValueError(
            "search_state_backend must be 'phase_i_mmes' or 'diagonal_cma'"
        )
    if request.car_candidate_mode not in {"graph", "shuffled_graph", "paired_fallback"}:
        raise ValueError(
            "car_candidate_mode must be 'graph', 'shuffled_graph', or 'paired_fallback'"
        )
    if request.car_actionability_arm not in {"off", "fallback", "candidate"}:
        raise ValueError(
            "car_actionability_arm must be 'off', 'fallback', or 'candidate'"
        )
    if request.car_actionability_arm != "off" and request.arac_action != (
        "arac_counterfactual_action_racing_w3"
    ):
        raise ValueError("CAR actionability arms require the frozen CAR-W3 action")
    if request.precision_causal_arm not in {"off", "baseline", "action"}:
        raise ValueError(
            "precision_causal_arm must be 'off', 'baseline', or 'action'"
        )
    if request.precision_causal_arm != "off" and request.arac_action != (
        "arac_evidence_action_controller_v37"
    ):
        raise ValueError("precision causal logging requires the frozen v37 action")
    if request.precision_response_arm not in {
        "off",
        "a0_v37",
        "a1_probe_only",
        "a2_probe_gated",
    }:
        raise ValueError("unsupported precision response arm")
    if request.precision_response_arm != "off" and request.arac_action != (
        "arac_evidence_action_controller_v37"
    ):
        raise ValueError("precision response logging requires the frozen v37 action")
    if request.component_precision_arm not in {
        "off",
        "a0_v37",
        "a1_precision_component_once",
    }:
        raise ValueError("unsupported component precision arm")
    if request.component_precision_arm != "off" and request.arac_action != (
        "arac_evidence_action_controller_v37"
    ):
        raise ValueError("component precision logging requires the frozen v37 action")
    active_precision_arms = sum(
        arm != "off"
        for arm in (
            request.precision_causal_arm,
            request.precision_response_arm,
            request.component_precision_arm,
        )
    )
    if active_precision_arms > 1:
        raise ValueError(
            "causal, response, and component precision arms are mutually exclusive"
        )
    if request.hypergraph_trace_mode not in {"off", "observer"}:
        raise ValueError("hypergraph_trace_mode must be 'off' or 'observer'")
    if request.hypergraph_trace_mode == "observer" and request.arac_action != (
        "arac_evidence_action_controller_v37"
    ):
        raise ValueError("hypergraph observer requires the frozen v37 action")
    if request.hypergraph_trace_mode == "observer" and active_precision_arms:
        raise ValueError(
            "hypergraph observer and frozen precision experiment arms are mutually exclusive"
        )
    if request.evidence_overlay_mode not in EVIDENCE_OVERLAY_MODES:
        raise ValueError(
            "evidence_overlay_mode must be 'off', 'native_audit', "
            "'paired_owner', or 'shuffled_owner'"
        )
    evidence_overlay_enabled = request.evidence_overlay_mode != "off"
    if evidence_overlay_enabled:
        if request.arac_action != "arac_evidence_action_controller_v37":
            raise ValueError("evidence overlay requires the frozen v37 action")
        if not request.enable_relation_dispatch:
            raise ValueError("evidence overlay requires relation dispatch")
        if request.relation_policy_mode != "controller_v31":
            raise ValueError("evidence overlay requires controller_v31")
        if request.budget_accounting != "strict":
            raise ValueError("evidence overlay requires strict FE accounting")
        if not request.cmaes_restart or not request.mmes_restart:
            raise ValueError("evidence overlay requires frozen restart settings")
        if request.search_state_backend != "phase_i_mmes":
            raise ValueError("evidence overlay requires phase_i_mmes")
        if request.cma_sampling_mode != "iid":
            raise ValueError("evidence overlay requires iid CMA sampling")
        if request.car_candidate_mode != "graph":
            raise ValueError("evidence overlay requires the default CAR candidate mode")
        if active_precision_arms or request.hypergraph_trace_mode != "off":
            raise ValueError(
                "evidence overlay, hypergraph observer, and precision arms are "
                "mutually exclusive"
            )
        if request.car_actionability_arm != "off" or request.offline_frozen_replay:
            raise ValueError(
                "evidence overlay cannot combine with CAR actionability or frozen replay"
            )
    if request.cma_sampling_mode not in {"iid", "mirrored_orthogonal"}:
        raise ValueError(
            "cma_sampling_mode must be 'iid' or 'mirrored_orthogonal'"
        )
    mos_profile_enabled = request.config_name == "v37_mos_sampling"
    if request.cma_sampling_mode == "mirrored_orthogonal" and not mos_profile_enabled:
        raise ValueError("mirrored orthogonal sampling requires v37_mos_sampling")
    if mos_profile_enabled:
        if request.arac_action != "arac_evidence_action_controller_v37":
            raise ValueError("v37_mos_sampling requires frozen v37")
        if not request.enable_relation_dispatch:
            raise ValueError("v37_mos_sampling requires relation dispatch")
        if request.relation_policy_mode != "controller_v31":
            raise ValueError("v37_mos_sampling requires controller_v31")
        if request.budget_accounting != "strict":
            raise ValueError("v37_mos_sampling requires strict FE accounting")
        if not request.cmaes_restart or not request.mmes_restart:
            raise ValueError("v37_mos_sampling requires frozen restart settings")
        if request.search_state_backend != "phase_i_mmes":
            raise ValueError("v37_mos_sampling requires phase_i_mmes")
        if request.early_stopping_evaluations != 1000:
            raise ValueError("v37_mos_sampling requires frozen early stopping")
        if active_precision_arms or request.hypergraph_trace_mode != "off":
            raise ValueError("v37_mos_sampling cannot combine with frozen pilots")
        if request.car_actionability_arm != "off":
            raise ValueError("v37_mos_sampling cannot combine with CAR arms")
        if evidence_overlay_enabled:
            raise ValueError("v37_mos_sampling cannot combine with evidence overlay")
    frozen_action = request.arac_action == "arac_evidence_action_controller_v41"
    if frozen_action and not request.offline_frozen_replay:
        raise ValueError("v41 is frozen; use offline_frozen_replay for historical replay")
    if request.offline_frozen_replay and not frozen_action:
        raise ValueError("offline_frozen_replay is only valid for frozen v41")
    vendor_paths = resolve_hcc_vendor_paths(
        request.hcc_root,
        repo_root=request.hcc_repo_root,
        runner_path=request.hcc_runner,
    )
    requested_data_root = Path(request.aob_data_root)
    if not requested_data_root.is_absolute():
        requested_data_root = ARAC_REPO_ROOT / requested_data_root
    aob_data_root = validate_aob_data_root(requested_data_root, function_id)
    if mos_profile_enabled and aob_data_root != DEFAULT_AOB_DATA_ROOT:
        raise ValueError("v37_mos_sampling requires the canonical AOB data root")
    output_dir = _normalize_hcc_output_dir(request.output_dir)

    argv = [
        request.python_executable,
        str(vendor_paths.runner),
        "--functions",
        function_name,
        "--ids",
        str(function_id),
        "--seed",
        str(request.seed),
        "--max-fes",
        str(request.max_fes),
        "--output-root",
        str(output_dir),
        "--aob-data-root",
        str(aob_data_root),
        "--timestamp",
        request.timestamp,
        "--arac-action",
        request.arac_action,
        "--budget-accounting",
        request.budget_accounting,
        "--search-state-backend",
        request.search_state_backend,
    ]
    if mos_profile_enabled:
        argv.extend(
            (
                "--lane-profile",
                "v37_mos_sampling",
                "--cma-sampling-mode",
                request.cma_sampling_mode,
                "--early-stopping-evaluations",
                str(request.early_stopping_evaluations),
            )
        )
    if request.car_candidate_mode != "graph":
        argv.extend(("--car-candidate-mode", request.car_candidate_mode))
    if request.car_actionability_arm != "off":
        argv.extend(("--car-actionability-arm", request.car_actionability_arm))
    if request.precision_causal_arm != "off":
        argv.extend(("--precision-causal-arm", request.precision_causal_arm))
    if request.precision_response_arm != "off":
        argv.extend(("--precision-response-arm", request.precision_response_arm))
    if request.component_precision_arm != "off":
        argv.extend(("--component-precision-arm", request.component_precision_arm))
    if request.hypergraph_trace_mode != "off":
        argv.extend(("--hypergraph-trace-mode", request.hypergraph_trace_mode))
    if request.evidence_overlay_mode != "off":
        argv.extend(("--evidence-overlay-mode", request.evidence_overlay_mode))
    if request.offline_frozen_replay:
        argv.append("--offline-frozen-replay")
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
    return HccAobSmokeCommand(argv=tuple(argv), cwd=vendor_paths.vendor_root)


def _normalize_hcc_output_dir(output_dir: Path | str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ARAC_REPO_ROOT / path
    return path.resolve()


def _parse_hcc_evidence_overlay_manifest(
    output_dir: Path,
    *,
    problem_id: str,
) -> tuple[Path | None, str, float | None, float | None]:
    manifests = sorted(
        Path(output_dir).rglob(f"{problem_id}_evidence_overlay_manifest.json")
    )
    if not manifests:
        return None, "missing", None, None
    if len(manifests) != 1:
        return None, "ambiguous", None, None
    path = manifests[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest must be an object")
        native = float(payload["native_terminal_error"])
        all_best = float(payload["all_evaluation_best_error"])
        if not math.isfinite(native) or native < 0.0:
            raise ValueError("native_terminal_error must be finite and non-negative")
        if not math.isfinite(all_best) or all_best < 0.0:
            raise ValueError("all_evaluation_best_error must be finite and non-negative")
        barrier_status = payload.get("barrier_status")
        if not isinstance(barrier_status, str) or not barrier_status:
            raise ValueError("barrier_status must be a non-empty string")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return path, "invalid", None, None
    return path, barrier_status, native, all_best


def run_hcc_aob_smoke_execution(request: HccAobExecutionRequest) -> HccAobExecutionResult:
    """Run one bounded canonical HCC vendor smoke execution and parse its offline result.

    The subprocess runs from the canonical ``vendor/hcc`` root. All executable
    and AOB input paths are absolute, so execution does not depend on the caller's
    current working directory. Returned final-error fields are offline evaluation
    outputs and must not be copied into runtime evidence.
    """

    output_dir = _normalize_hcc_output_dir(request.output_dir)
    command = build_hcc_aob_smoke_command(
        HccAobExecutionRequest(
            problem_id=request.problem_id,
            seed=request.seed,
            max_fes=request.max_fes,
            output_dir=output_dir,
            hcc_root=Path(request.hcc_root),
            hcc_repo_root=(
                None if request.hcc_repo_root is None else Path(request.hcc_repo_root)
            ),
            hcc_runner=None if request.hcc_runner is None else Path(request.hcc_runner),
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
            early_stopping_evaluations=request.early_stopping_evaluations,
            skip_plots=request.skip_plots,
            search_state_backend=request.search_state_backend,
            car_candidate_mode=request.car_candidate_mode,
            car_actionability_arm=request.car_actionability_arm,
            precision_causal_arm=request.precision_causal_arm,
            precision_response_arm=request.precision_response_arm,
            component_precision_arm=request.component_precision_arm,
            hypergraph_trace_mode=request.hypergraph_trace_mode,
            evidence_overlay_mode=request.evidence_overlay_mode,
            cma_sampling_mode=request.cma_sampling_mode,
            offline_frozen_replay=request.offline_frozen_replay,
        )
    )
    start = time.time()
    subprocess_environment = None
    if request.evidence_overlay_mode != "off":
        subprocess_environment = {
            **os.environ,
            **EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT,
        }
    completed = subprocess.run(
        command.argv,
        cwd=command.cwd,
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_environment,
    )
    elapsed = time.time() - start
    if completed.returncode != 0:
        problem_id = _problem_parts(request.problem_id)[0]
        budget_breakdown: dict[str, int] = {}
        overlay_manifest_path = None
        overlay_status = (
            "off" if request.evidence_overlay_mode == "off" else "subprocess_failed"
        )
        native_terminal_error = None
        all_evaluation_best_error = None
        if request.evidence_overlay_mode != "off":
            try:
                budget_breakdown = _parse_hcc_budget_summary(output_dir)
            except (OSError, TypeError, ValueError):
                budget_breakdown = {}
            (
                overlay_manifest_path,
                parsed_overlay_status,
                native_terminal_error,
                all_evaluation_best_error,
            ) = _parse_hcc_evidence_overlay_manifest(
                output_dir,
                problem_id=problem_id,
            )
            if overlay_manifest_path is not None:
                overlay_status = parsed_overlay_status
        actual_fe = max(0, budget_breakdown.get("fitness_record_fe", 0))
        return HccAobExecutionResult(
            problem_id=problem_id,
            seed=request.seed,
            max_fes=request.max_fes,
            final_error=(
                float("nan")
                if all_evaluation_best_error is None
                else all_evaluation_best_error
            ),
            fe_used=actual_fe,
            time_seconds=elapsed,
            output_root=output_dir,
            fresh_optimizer_execution=bool(actual_fe or overlay_manifest_path),
            status=f"failed_returncode_{completed.returncode}",
            result_source="hcc_subprocess_smoke_execution",
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
            action_trace_path=None,
            action_trace_rows=0,
            optimizer_final_fe_used=(actual_fe if actual_fe > 0 else None),
            global_phase_fe=budget_breakdown.get("global_phase_fe"),
            cc_phase_fe=budget_breakdown.get("cc_phase_fe"),
            rescue_fe=budget_breakdown.get("rescue_fe"),
            refresh_fe=budget_breakdown.get("refresh_fe"),
            search_state_fe=budget_breakdown.get("search_state_fe", 0),
            precision_probe_fe=budget_breakdown.get("precision_probe_fe", 0),
            evidence_overlay_fe=budget_breakdown.get("evidence_overlay_fe", 0),
            separable_continuation_fe=budget_breakdown.get(
                "separable_continuation_fe"
            ),
            overhead_fe=budget_breakdown.get("overhead_fe"),
            native_terminal_error=native_terminal_error,
            all_evaluation_best_error=all_evaluation_best_error,
            evidence_overlay_manifest_path=overlay_manifest_path,
            evidence_overlay_status=overlay_status,
        )

    final_error, fe_used, optimizer_final_fe_used = (
        _parse_hcc_evaluation_record_with_optimizer_final_fe(
            output_dir,
            budget_limit=request.max_fes,
        )
    )
    action_trace_path, action_trace_rows = _find_hcc_action_trace(output_dir)
    problem_id = _problem_parts(request.problem_id)[0]
    mos_profile_enabled = request.config_name == "v37_mos_sampling"
    if mos_profile_enabled:
        mos_sampling_audit_path, mos_sampling_audit_rows = _find_hcc_named_csv(
            output_dir, f"{problem_id}_mos_sampling_audit.csv"
        )
        mos_branch_provenance_path, mos_branch_provenance_rows = (
            _find_hcc_named_csv(
                output_dir, f"{problem_id}_mos_branch_provenance.csv"
            )
        )
    else:
        mos_sampling_audit_path, mos_sampling_audit_rows = None, 0
        mos_branch_provenance_path, mos_branch_provenance_rows = None, 0
    budget_breakdown = _parse_hcc_budget_summary(output_dir)
    if request.evidence_overlay_mode == "off":
        overlay_manifest_path = None
        overlay_status = "off"
        native_terminal_error = None
        all_evaluation_best_error = None
    else:
        (
            overlay_manifest_path,
            overlay_status,
            native_terminal_error,
            all_evaluation_best_error,
        ) = _parse_hcc_evidence_overlay_manifest(
            output_dir,
            problem_id=problem_id,
        )
    return HccAobExecutionResult(
        problem_id=problem_id,
        seed=request.seed,
        max_fes=request.max_fes,
        final_error=final_error,
        fe_used=fe_used,
        time_seconds=elapsed,
        output_root=output_dir,
        fresh_optimizer_execution=True,
        status="completed",
        result_source="hcc_subprocess_smoke_execution",
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        action_trace_path=action_trace_path,
        action_trace_rows=action_trace_rows,
        mos_sampling_audit_path=mos_sampling_audit_path,
        mos_sampling_audit_rows=mos_sampling_audit_rows,
        mos_branch_provenance_path=mos_branch_provenance_path,
        mos_branch_provenance_rows=mos_branch_provenance_rows,
        optimizer_final_fe_used=optimizer_final_fe_used,
        global_phase_fe=budget_breakdown.get("global_phase_fe"),
        cc_phase_fe=budget_breakdown.get("cc_phase_fe"),
        rescue_fe=budget_breakdown.get("rescue_fe"),
        refresh_fe=budget_breakdown.get("refresh_fe"),
        search_state_fe=budget_breakdown.get("search_state_fe", 0),
        precision_probe_fe=budget_breakdown.get("precision_probe_fe", 0),
        evidence_overlay_fe=budget_breakdown.get("evidence_overlay_fe", 0),
        separable_continuation_fe=budget_breakdown.get(
            "separable_continuation_fe"
        ),
        overhead_fe=budget_breakdown.get("overhead_fe"),
        native_terminal_error=native_terminal_error,
        all_evaluation_best_error=all_evaluation_best_error,
        evidence_overlay_manifest_path=overlay_manifest_path,
        evidence_overlay_status=overlay_status,
    )




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
