from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import time
from dataclasses import asdict, dataclass
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
)
from src.arac.policy.relation_policy import decide_actions_for_relations

from AOB.AOB import Benchmark
from AOB.utils import (
    combine,
    evaluation_record,
    plot_evaluation_curve,
    plot_evaluation_curve_best_so_far,
    remove_overlapping_groups,
)
from HCC.NDAs.MMES.mmes import MMES
from HCC.OPT.CMAES.cmaes import CMAES
from HCC.RDDSM import Decomposition


PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "HCC_SRC" / "AOB" / "AOBG" / "datafile"
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
    "downstream_consumed",
    "downstream_consumption_scope",
    "optimizer_consumed",
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
REPAIR_ACTION_NAMES = {"repair_shared_variable_binding"}


@dataclass(frozen=True)
class SmokeConfig:
    max_fes: int
    seed: int | None
    run_id: str = "arac-hcc-smoke"
    sigma: float = 0.5
    verbose: int = 1000
    early_stopping_evaluations: int = 1000
    cmaes_restart: bool = False
    arac_action: str = "conservative_no_action"
    enable_relation_dispatch: bool = False
    relation_policy_mode: str = "rule"
    arac_action_file: Path | None = None


@dataclass(frozen=True)
class RelationExecutionContext:
    overlap_indices: list[int]
    previous_values: np.ndarray
    current_values: np.ndarray
    previous_delta: float
    current_delta: float


def load_aob_metadata(fun_id: int) -> dict:
    with (DATA_DIR / f"F{fun_id}-info.txt").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_design_matrix(fun_id: int) -> np.ndarray:
    return np.loadtxt(DATA_DIR / f"F{fun_id}-design.txt", delimiter=",")


def load_permutation_vector(fun_id: int) -> list[int]:
    return (np.loadtxt(DATA_DIR / f"F{fun_id}-p.txt", delimiter=",").reshape(-1).astype(int) - 1).tolist()


def build_aob_topology_groups(fun_id: int) -> list[list[int]]:
    metadata = load_aob_metadata(fun_id)
    permutation = load_permutation_vector(fun_id)
    overlap = int(metadata["overlap_degree"])
    groups: list[list[int]] = []
    begin_index = 0
    for index, subgroup_size in enumerate(metadata["subgroups"]):
        end_index = begin_index + int(subgroup_size)
        groups.append(permutation[begin_index:end_index])
        if index != len(metadata["subgroups"]) - 1:
            begin_index = end_index - overlap
    return groups


def order_grouping_by_aob_topology(grouping_result: list[list[int]], fun_id: int) -> list[list[int]]:
    topology_groups = build_aob_topology_groups(fun_id)
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


def decompose_problem(fun_id: int) -> list[list[int]]:
    grouping_result = Decomposition(load_design_matrix(fun_id)).decomposition()
    return order_grouping_by_aob_topology(grouping_result, fun_id)


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


def iteration_start_budget_remaining_ratio(max_fes: int, sum_fes: int) -> float:
    if max_fes <= 0:
        return 0.0
    return max(0.0, (max_fes - sum_fes) / max_fes)


def derive_optimizer_seed(base_seed: int, fun_name: str, fun_id: int, stage_index: int) -> int:
    payload = f"{base_seed}:{fun_name}:{fun_id}:{stage_index}".encode("utf-8")
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
    if action_name in {
        "repair_shared_variable_binding",
        "isolate_conflicting_relation",
        "allow_beneficial_coordination",
        "conservative_no_action",
    }:
        return "1"
    return "0"


def _optimizer_consumed(action_name: str, downstream_consumed: bool = True) -> str:
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


SHUFFLED_RELATION_ACTION = {
    "coordinate": ("reassign_repair", "reassign_repair"),
    "reassign_repair": ("isolate_conflicting_relation", "isolate"),
    "isolate_conflicting_relation": ("fallback", "fallback"),
    "fallback": ("coordinate", "coordinate"),
}


def select_relation_action_for_policy(
    relation: OverlapRelation,
    action: RelationActionDecision,
    relation_policy_mode: str,
) -> RelationActionDecision:
    if relation_policy_mode == "rule":
        return action
    if relation_policy_mode != "shuffled":
        raise ValueError(f"unsupported relation policy mode: {relation_policy_mode}")
    relation_action_name, action_family = SHUFFLED_RELATION_ACTION[
        action.relation_action_name
    ]
    return RelationActionDecision(
        relation_id=relation.relation_id,
        action_name=relation_action_name,
        action_family=action_family,
        confidence=action.confidence,
        trigger_reason=(
            "deterministic_shuffled_negative_control_from:"
            f"{action.relation_action_name}"
        ),
    )


def _canonical_relation_action_name(action: RelationActionDecision) -> str:
    if getattr(action, "canonical_action_name", ""):
        return action.canonical_action_name
    return RELATION_ACTION_ALIASES.get(action.action_name, action.action_name)


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
    downstream_consumed: bool = True,
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
        "downstream_consumed": str(int(downstream_consumed)),
        "downstream_consumption_scope": "same_outer_iteration",
        "optimizer_consumed": _optimizer_consumed(selected_action_name, downstream_consumed),
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
    time_start = time.time()
    bench = Benchmark(str(output_path) + "/")
    fun = bench.get_function(fun_name, fun_id)
    info = bench.get_info(fun_name, fun_id)
    grouping_result = decompose_problem(fun_id)
    _, overlap_groups, overlapping_elements = remove_overlapping_groups(grouping_result)
    metadata = load_aob_metadata(fun_id)
    degree = calculate_degree_of_overlap(overlap_groups, metadata["dimension"])
    global_fes = calculate_global_fes(config.max_fes, degree)
    best_individual = np.zeros(info["dimension"])
    sum_fes = 0
    action_trace_rows: list[dict[str, str]] = []
    relations: list[OverlapRelation] = []
    action_decisions: list[RelationActionDecision] = []

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
            "is_restart": True,
            "verbose": config.verbose,
        }
        if config.seed is not None:
            options["seed_rng"] = derive_optimizer_seed(config.seed, fun_name, fun_id, 0)
        results = MMES(problem, options).optimize()
        best_individual = results["best_so_far_x"].copy()
        sum_fes += results["n_function_evaluations"]

    outer_iter = 0
    while current_fitness_evaluations(fun) < config.max_fes:
        current_fes = current_fitness_evaluations(fun)
        iteration_budget_remaining_ratio = iteration_start_budget_remaining_ratio(
            max_fes=config.max_fes,
            sum_fes=current_fes,
        )
        sub_num = len(grouping_result)
        sub_fes = math.ceil((config.max_fes - current_fes) / sub_num)
        fitness_delta_list: list[float] = []
        optimized_any_group = False
        for index, dims in enumerate(grouping_result):
            population_size = calculate_cmaes_population_size(len(dims))
            if config.max_fes - current_fitness_evaluations(fun) <= population_size:
                break
            original_best = best_individual.copy()
            original_fitness = float(fun(best_individual)[0])
            optimizer_budget = bounded_population_budget(
                requested_fes=max(sub_fes, population_size),
                remaining_fes=config.max_fes - current_fitness_evaluations(fun),
                population_size=population_size,
            )
            if optimizer_budget <= 0:
                break
            objective_function = lambda x_batch, dims=dims: fun(combine(x_batch, best_individual, dims))
            problem_cc = {
                "fitness_function": objective_function,
                "ndim_problem": len(dims),
                "lower_boundary": info["lower"] * np.ones((len(dims),)),
                "upper_boundary": info["upper"] * np.ones((len(dims),)),
            }
            options_cc = {
                "max_function_evaluations": optimizer_budget,
                "mean": (best_individual[dims],),
                "sigma": config.sigma,
                "n_individuals": population_size,
                "is_restart": config.cmaes_restart,
                "verbose": config.verbose,
                "early_stopping_evaluations": config.early_stopping_evaluations,
            }
            if config.seed is not None:
                stage_index = outer_iter * sub_num + index + 1
                options_cc["seed_rng"] = derive_optimizer_seed(config.seed, fun_name, fun_id, stage_index)
            results_cc = CMAES(problem_cc, options_cc).optimize()
            optimized_any_group = True
            sum_fes += results_cc["n_function_evaluations"]
            new_best_y = float(results_cc["best_so_far_y"])
            if new_best_y < original_fitness:
                best_individual[dims] = results_cc["best_so_far_x"].copy()
                current_delta = original_fitness - new_best_y
            else:
                current_delta = 0.0
            fitness_delta_list.append(current_delta)
            if index > 0:
                overlap_indices = overlapping_elements[index - 1]
                if config.enable_relation_dispatch:
                    if config.relation_policy_mode not in {"rule", "shuffled"}:
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
                    action = select_relation_action_for_policy(
                        relation=relation,
                        action=decide_actions_for_relations([relation])[0],
                        relation_policy_mode=config.relation_policy_mode,
                    )
                    adjusted_values = apply_action_to_relation(
                        relation=relation,
                        action=action,
                        previous_values=context.previous_values,
                        current_values=context.current_values,
                        previous_delta=context.previous_delta,
                        current_delta=context.current_delta,
                    )
                    if adjusted_values is not None:
                        best_individual[context.overlap_indices] = adjusted_values
                    canonical_action_name = _canonical_relation_action_name(action)
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
                            relation_policy_source=(
                                "deterministic_shuffled_negative_control"
                                if config.relation_policy_mode == "shuffled"
                                else "rule_based_relation_policy"
                            ),
                            state_mutated=adjusted_values is not None,
                            downstream_consumed=index < sub_num - 1,
                        )
                    )
                else:
                    best_individual[overlap_indices] = apply_arac_overlap_action(
                        action_name=config.arac_action,
                        previous_values=original_best[overlap_indices],
                        current_values=best_individual[overlap_indices],
                        previous_delta=fitness_delta_list[index - 1],
                        current_delta=current_delta,
                    )
                    action_trace_rows.append(
                        build_action_trace_row(
                            problem_id=_problem_id(fun_name, fun_id),
                            seed=config.seed,
                            outer_iter=outer_iter,
                            group_index=index,
                            selected_action_name=config.arac_action,
                            overlap_size=len(overlap_indices),
                            previous_delta=fitness_delta_list[index - 1],
                            current_delta=current_delta,
                            state_mutated=True,
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
        outer_iter += 1

    problem_id = _problem_id(fun_name, fun_id)
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
    print(f"{problem_id} overlap relations extracted: {len(relations)}")
    return fun.fitness_record, time.time() - time_start, action_trace_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARAC-owned HCC smoke runner.")
    parser.add_argument("--functions", nargs="+", choices=FUNCTION_NAMES, required=True)
    parser.add_argument("--ids", nargs="+", type=int, choices=PROBLEM_IDS, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timestamp", default="arac-hcc-smoke")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-fes", type=int, required=True)
    parser.add_argument("--verbose", type=int, default=1000)
    parser.add_argument("--early-stopping-evaluations", type=int, default=1000)
    parser.add_argument("--cmaes-restart", action="store_true")
    parser.add_argument("--enable-relation-dispatch", action="store_true")
    parser.add_argument("--relation-policy", default="rule", choices=["rule", "shuffled"])
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
        ],
    )
    args = parser.parse_args(argv)
    if args.arac_action_file is not None:
        parser.error("--arac-action-file is not supported by the smoke runner yet")
    return args


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    config = SmokeConfig(
        run_id=args.timestamp,
        max_fes=args.max_fes,
        seed=args.seed,
        verbose=args.verbose,
        early_stopping_evaluations=args.early_stopping_evaluations,
        cmaes_restart=args.cmaes_restart,
        arac_action=args.arac_action,
        enable_relation_dispatch=args.enable_relation_dispatch,
        relation_policy_mode=args.relation_policy,
        arac_action_file=args.arac_action_file,
    )
    output_paths = []
    for fun_name in args.functions:
        output_path = Path(args.output_root) / args.timestamp / fun_name
        output_path.mkdir(parents=True, exist_ok=True)
        output_data = {}
        function_trace_rows: list[dict[str, str]] = []
        function_action_decision_rows: list[dict[str, str]] = []
        _remove_if_exists(output_path / "action_decision.csv")
        for fun_id in args.ids:
            algorithm = f"{fun_name}_{fun_id}"
            output_data[algorithm] = []
            output_data[f"{algorithm}_time"] = []
            record, elapsed, trace_rows = run_problem(fun_name, fun_id, output_path, config)
            output_data[algorithm].append(record)
            output_data[f"{algorithm}_time"].append(elapsed)
            problem_id = _problem_id(fun_name, fun_id)
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
            print(f"{algorithm} average time: {elapsed}")
        _write_action_trace(output_path / "action_trace.csv", function_trace_rows)
        if config.enable_relation_dispatch:
            _write_raw_action_decision_rows(
                output_path / "action_decision.csv",
                function_action_decision_rows,
            )
        evaluation_record(output_data, str(output_path) + "/", record_FEs_list=(args.max_fes,))
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
