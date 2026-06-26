"""HCC backbone extraction helpers.

This module is the first clean ARAC extraction layer for the historical
``E:\\HCC-main`` work. It models the data ARAC needs from HCC grouping and
optimization traces without importing legacy milestone runners or mutating the
HCC baseline.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from arac.action_space import ActionFamily
from arac.backend_adapter import BackendSemanticsDiff
from arac.evidence import EvidenceProfile, validate_runtime_payload
from arac.policy import ActionDecision

DEFAULT_HCC_MAIN_ROOT = Path("E:/HCC-main")
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


def hcc_backend_semantics_for(decision: ActionDecision) -> BackendSemanticsDiff:
    """Map clean ARAC actions onto HCC optimizer-consumed semantic surfaces."""

    if decision.action_family == ActionFamily.ISOLATE:
        return BackendSemanticsDiff(relation_handling_changed=True)
    if decision.action_family == ActionFamily.PROTECT:
        return BackendSemanticsDiff(budget_allocation_changed=True)
    if decision.action_family == ActionFamily.REASSIGN_REPAIR:
        return BackendSemanticsDiff(variable_owner_changed=True)
    if decision.action_family == ActionFamily.COORDINATE:
        return BackendSemanticsDiff(coordination_mode_changed=True)
    return BackendSemanticsDiff()
