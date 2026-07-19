"""Narrow HCC/AOB adapter used by the exp_018 evidence-overlay pilot."""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .hcc_budget import (
    _parse_hcc_budget_summary,
    _parse_hcc_evaluation_record_with_optimizer_final_fe,
)


ARAC_REPO_ROOT = Path(__file__).resolve().parents[3]
HCC_VENDOR_ROOT = (ARAC_REPO_ROOT / "vendor" / "hcc").resolve()


@dataclass(frozen=True)
class HccVendorPaths:
    """Canonical source paths derived from one explicit vendor root."""

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
    "A": "ackley",
}
EXP_018_CASES = frozenset({"E1", "E3", "A4", "S5"})

FROZEN_ARAC_ACTION = "arac_evidence_action_controller_v37"
FROZEN_RELATION_POLICY = "controller_v31"
FROZEN_BUDGET_ACCOUNTING = "strict"
FROZEN_SEARCH_STATE_BACKEND = "phase_i_mmes"

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
class HccGroupSignal:
    """Structural signal used only by the source-topology sanity check."""

    group_id: str
    fitness_delta: float
    rank: int
    shared_variable_count: int = 0


@dataclass(frozen=True)
class HccAobCaseTopology:
    """AOB topology read without running the optimizer."""

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


@dataclass(frozen=True)
class HccAobSmokeCommand:
    argv: tuple[str, ...]
    cwd: Path


@dataclass(frozen=True)
class HccAobExecutionRequest:
    """One frozen exp_018 HCC/AOB trajectory request."""

    problem_id: str
    seed: int
    max_fes: int
    output_dir: Path
    hcc_root: Path = HCC_VENDOR_ROOT
    aob_data_root: Path = DEFAULT_AOB_DATA_ROOT
    python_executable: str = "python"
    timestamp: str = "arac-hcc-smoke"
    evidence_overlay_mode: str = "off"


@dataclass(frozen=True)
class HccAobExecutionResult:
    """Offline result and FE ledger from one fresh HCC execution."""

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


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


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
    """Return every AOB input consumed by one function id."""

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
    """Fail before optimizer execution when AOB inputs are incomplete."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AOB data root does not exist: {root}")
    missing = [
        path.name
        for path in required_aob_data_files(root, function_id)
        if not path.is_file()
    ]
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
    min_fraction = 0.20
    min_global = int(min_fraction * total_fes)
    if degree_of_overlap == 0:
        return min_global
    return max(
        min_global,
        int((0.2 + (4 / 5) * degree_of_overlap) * total_fes),
    )


def _problem_parts(problem_id: str) -> tuple[str, str, int]:
    problem = str(problem_id).strip().upper()
    if problem not in EXP_018_CASES:
        raise ValueError(f"unsupported exp_018 AOB problem_id: {problem_id}")
    return problem, AOB_FUNCTION_NAMES[problem[0]], int(problem[1])


def load_hcc_aob_topology(
    problem_id: str,
    hcc_root: Path | str = HCC_VENDOR_ROOT,
    total_fes: int = TOTAL_AOB_FE,
) -> HccAobCaseTopology:
    """Read source-grounded AOB topology without optimizer execution."""

    problem, function_name, function_id = _problem_parts(problem_id)
    vendor_paths = resolve_hcc_vendor_paths(
        hcc_root,
        runner_path=ARAC_HCC_SMOKE_RUNNER,
    )
    data_dir = vendor_paths.aob_data_root
    info = _parse_aob_info(data_dir / f"F{function_id}-info.txt")
    permutation = _read_permutation(data_dir / f"F{function_id}-p.txt")
    topology_groups = _topology_groups(info, permutation)
    overlaps = _overlap_groups(topology_groups)
    overlapping_elements = {element for group in overlaps for element in group}
    dimension = int(info["dimension"])
    degree = _safe_divide(len(overlapping_elements), dimension)
    groups = tuple(
        HccGroupSignal(
            group_id=f"source_group_{index + 1:02d}",
            fitness_delta=1.0 / (index + 1),
            rank=index + 1,
            shared_variable_count=sum(
                1 for element in group if element in overlapping_elements
            ),
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
        groups=groups,
    )


def _normalize_hcc_output_dir(output_dir: Path | str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ARAC_REPO_ROOT / path
    return path.resolve()


def build_hcc_aob_smoke_command(request: HccAobExecutionRequest) -> HccAobSmokeCommand:
    """Build the exact frozen exp_018 subprocess command."""

    _, function_name, function_id = _problem_parts(request.problem_id)
    if isinstance(request.seed, bool) or not isinstance(request.seed, int) or request.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(request.max_fes, bool)
        or not isinstance(request.max_fes, int)
        or request.max_fes <= 0
    ):
        raise ValueError("max_fes must be a positive integer")
    if request.evidence_overlay_mode not in EVIDENCE_OVERLAY_MODES:
        raise ValueError(
            f"unsupported evidence_overlay_mode: {request.evidence_overlay_mode}"
        )

    vendor_paths = resolve_hcc_vendor_paths(
        request.hcc_root,
        runner_path=ARAC_HCC_SMOKE_RUNNER,
    )
    requested_data_root = Path(request.aob_data_root)
    if not requested_data_root.is_absolute():
        requested_data_root = ARAC_REPO_ROOT / requested_data_root
    aob_data_root = validate_aob_data_root(requested_data_root, function_id)
    output_dir = _normalize_hcc_output_dir(request.output_dir)
    argv = [
        request.python_executable or sys.executable,
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
        FROZEN_ARAC_ACTION,
        "--budget-accounting",
        FROZEN_BUDGET_ACCOUNTING,
        "--search-state-backend",
        FROZEN_SEARCH_STATE_BACKEND,
        "--enable-relation-dispatch",
        "--relation-policy",
        FROZEN_RELATION_POLICY,
        "--skip-plots",
    ]
    if request.evidence_overlay_mode != "off":
        argv.extend(("--evidence-overlay-mode", request.evidence_overlay_mode))
    return HccAobSmokeCommand(argv=tuple(argv), cwd=vendor_paths.vendor_root)


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


def _tail(text: str, max_chars: int = 2000) -> str:
    return (text or "")[-max_chars:]


def _ledger_fields(budget: dict[str, int]) -> dict[str, int | None]:
    return {
        "global_phase_fe": budget.get("global_phase_fe"),
        "cc_phase_fe": budget.get("cc_phase_fe"),
        "rescue_fe": budget.get("rescue_fe"),
        "refresh_fe": budget.get("refresh_fe"),
        "search_state_fe": budget.get("search_state_fe", 0),
        "precision_probe_fe": budget.get("precision_probe_fe", 0),
        "evidence_overlay_fe": budget.get("evidence_overlay_fe", 0),
        "separable_continuation_fe": budget.get("separable_continuation_fe"),
        "overhead_fe": budget.get("overhead_fe"),
    }


def run_hcc_aob_smoke_execution(request: HccAobExecutionRequest) -> HccAobExecutionResult:
    """Run one real HCC trajectory and parse its offline result and FE ledger."""

    output_dir = _normalize_hcc_output_dir(request.output_dir)
    command = build_hcc_aob_smoke_command(request)
    environment = None
    if request.evidence_overlay_mode != "off":
        environment = {**os.environ, **EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT}
    start = time.time()
    completed = subprocess.run(
        command.argv,
        cwd=command.cwd,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed = time.time() - start
    problem_id = _problem_parts(request.problem_id)[0]

    if completed.returncode != 0:
        try:
            budget = _parse_hcc_budget_summary(output_dir)
        except (OSError, TypeError, ValueError):
            budget = {}
        manifest_path = None
        native_terminal_error = None
        all_evaluation_best_error = None
        if request.evidence_overlay_mode != "off":
            (
                manifest_path,
                _manifest_status,
                native_terminal_error,
                all_evaluation_best_error,
            ) = _parse_hcc_evidence_overlay_manifest(
                output_dir,
                problem_id=problem_id,
            )
        actual_fe = max(0, budget.get("fitness_record_fe", 0))
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
            fresh_optimizer_execution=bool(actual_fe or manifest_path),
            status=f"failed_returncode_{completed.returncode}",
            result_source="hcc_subprocess_smoke_execution",
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
            optimizer_final_fe_used=actual_fe if actual_fe > 0 else None,
            native_terminal_error=native_terminal_error,
            all_evaluation_best_error=all_evaluation_best_error,
            evidence_overlay_manifest_path=manifest_path,
            **_ledger_fields(budget),
        )

    final_error, fe_used, optimizer_final_fe_used = (
        _parse_hcc_evaluation_record_with_optimizer_final_fe(
            output_dir,
            budget_limit=request.max_fes,
        )
    )
    budget = _parse_hcc_budget_summary(output_dir)
    manifest_path = None
    native_terminal_error = None
    all_evaluation_best_error = None
    if request.evidence_overlay_mode != "off":
        (
            manifest_path,
            _manifest_status,
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
        optimizer_final_fe_used=optimizer_final_fe_used,
        native_terminal_error=native_terminal_error,
        all_evaluation_best_error=all_evaluation_best_error,
        evidence_overlay_manifest_path=manifest_path,
        **_ledger_fields(budget),
    )
