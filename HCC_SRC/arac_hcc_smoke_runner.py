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
if str(ARAC_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ARAC_REPO_ROOT))

from src.arac.evidence.overlap_relation_builder import (
    OverlapRelation,
    build_overlap_relations,
)
from src.arac.policy.relation_policy import (
    ActionDecision as RelationActionDecision,
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
    "overlap_size",
    "previous_delta",
    "current_delta",
    "owner_selected",
    "semantic_surface",
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
    "action_name",
    "action_family",
    "confidence",
    "trigger_reason",
]
REPAIR_ACTION_NAMES = {"repair_shared_variable_binding", "reassign_repair"}


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
        return "previous"
    if action_name in {"coordinate", "fallback"}:
        return "current"
    return "weighted_blend"


def _semantic_surface(action_name: str) -> str:
    if action_name in REPAIR_ACTION_NAMES:
        return "shared_variable_owner_rebinding"
    if action_name == "isolate_conflicting_relation":
        return "relation_isolation_hook"
    if action_name == "coordinate":
        return "relation_coordinate_noop"
    if action_name == "fallback":
        return "relation_fallback_noop"
    return "native_overlap_blend"


def _optimizer_consumed(action_name: str) -> str:
    if action_name in {"repair_shared_variable_binding", "reassign_repair"}:
        return "1"
    if action_name == "isolate_conflicting_relation":
        return "1"
    return "0"


def build_action_trace_row(
    problem_id: str,
    seed: int | None,
    outer_iter: int,
    group_index: int,
    selected_action_name: str,
    overlap_size: int,
    previous_delta: float,
    current_delta: float,
) -> dict[str, str]:
    return {
        "problem_id": problem_id,
        "seed": "" if seed is None else str(seed),
        "outer_iter": str(outer_iter),
        "group_index": str(group_index),
        "selected_action_name": selected_action_name,
        "overlap_size": str(overlap_size),
        "previous_delta": f"{previous_delta:.6e}",
        "current_delta": f"{current_delta:.6e}",
        "owner_selected": _owner_selected(
            selected_action_name,
            previous_delta,
            current_delta,
        ),
        "semantic_surface": _semantic_surface(selected_action_name),
        "optimizer_consumed": _optimizer_consumed(selected_action_name),
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
        "budget_remaining_ratio": budget_remaining_ratio,
    }
    return build_overlap_relations(hcc_trace, problem_id)


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
    if action.action_name == "reassign_repair":
        return apply_arac_overlap_action(
            action_name="repair_shared_variable_binding",
            previous_values=previous_values,
            current_values=current_values,
            previous_delta=previous_delta,
            current_delta=current_delta,
        )
    if action.action_name == "isolate_conflicting_relation":
        return previous_values
    if action.action_name in {"coordinate", "fallback"}:
        return current_values
    raise ValueError(f"unknown relation action for {relation.relation_id}: {action.action_name}")


def _format_shared_vars(shared_vars: tuple[int, ...]) -> str:
    return ";".join(str(variable) for variable in shared_vars)


def _overlap_relation_row(relation: OverlapRelation) -> dict[str, str]:
    row = asdict(relation)
    row["shared_vars"] = _format_shared_vars(relation.shared_vars)
    row["overlap_strength"] = f"{relation.overlap_strength:.6f}"
    row["delta_signal"] = f"{relation.delta_signal:.6f}"
    row["rank_signal"] = f"{relation.rank_signal:.6f}"
    row["budget_remaining_ratio"] = f"{relation.budget_remaining_ratio:.6f}"
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
        "action_name": action.action_name,
        "action_family": action.action_family,
        "confidence": f"{action.confidence:.6f}",
        "trigger_reason": action.trigger_reason,
    }


def _append_action_decision_log(
    path: Path,
    run_id: str,
    relations: list[OverlapRelation],
    actions: list[RelationActionDecision],
) -> None:
    if len(relations) != len(actions):
        raise ValueError("relations and actions must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTION_DECISION_FIELDS)
        if write_header:
            writer.writeheader()
        for relation, action in zip(relations, actions, strict=True):
            writer.writerow(_action_decision_row(run_id, relation, action))


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
    while sum_fes < config.max_fes:
        sub_num = len(grouping_result)
        sub_fes = math.ceil((config.max_fes - sum_fes) / sub_num)
        fitness_delta_list: list[float] = []
        relation_contexts: dict[tuple[int, int], RelationExecutionContext] = {}
        for index, dims in enumerate(grouping_result):
            original_best = best_individual.copy()
            original_fitness = float(fun(best_individual)[0])
            objective_function = lambda x_batch, dims=dims: fun(combine(x_batch, best_individual, dims))
            problem_cc = {
                "fitness_function": objective_function,
                "ndim_problem": len(dims),
                "lower_boundary": info["lower"] * np.ones((len(dims),)),
                "upper_boundary": info["upper"] * np.ones((len(dims),)),
            }
            options_cc = {
                "max_function_evaluations": sub_fes,
                "mean": (best_individual[dims],),
                "sigma": config.sigma,
                "n_individuals": calculate_cmaes_population_size(len(dims)),
                "is_restart": config.cmaes_restart,
                "verbose": config.verbose,
                "early_stopping_evaluations": config.early_stopping_evaluations,
            }
            if config.seed is not None:
                stage_index = outer_iter * sub_num + index + 1
                options_cc["seed_rng"] = derive_optimizer_seed(config.seed, fun_name, fun_id, stage_index)
            results_cc = CMAES(problem_cc, options_cc).optimize()
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
                    relation_contexts[(index - 1, index)] = RelationExecutionContext(
                        overlap_indices=list(overlap_indices),
                        previous_values=original_best[overlap_indices].copy(),
                        current_values=best_individual[overlap_indices].copy(),
                        previous_delta=fitness_delta_list[index - 1],
                        current_delta=current_delta,
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
                        )
                    )
        budget_remaining_ratio = max(
            0.0,
            (config.max_fes - sum_fes) / config.max_fes,
        )
        iteration_relations = build_overlap_relation_trace(
            problem_id=_problem_id(fun_name, fun_id),
            outer_iter=outer_iter,
            grouping_result=grouping_result,
            overlapping_elements=overlapping_elements,
            fitness_delta_list=fitness_delta_list,
            budget_remaining_ratio=budget_remaining_ratio,
        )
        relations.extend(iteration_relations)
        if config.enable_relation_dispatch:
            actions = decide_actions_for_relations(iteration_relations)
            _append_action_decision_log(
                output_path / "action_decision.csv",
                config.run_id,
                iteration_relations,
                actions,
            )
            actions_by_relation_id = {action.relation_id: action for action in actions}
            for relation in iteration_relations:
                context = relation_contexts.get((relation.group_left, relation.group_right))
                if context is None:
                    continue
                action = actions_by_relation_id[relation.relation_id]
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
                action_trace_rows.append(
                    build_action_trace_row(
                        problem_id=_problem_id(fun_name, fun_id),
                        seed=config.seed,
                        outer_iter=outer_iter,
                        group_index=relation.group_right,
                        selected_action_name=action.action_name,
                        overlap_size=len(relation.shared_vars),
                        previous_delta=context.previous_delta,
                        current_delta=context.current_delta,
                    )
                )
        outer_iter += 1

    _write_overlap_relation_trace(output_path / "overlap_relations.csv", relations)
    print(f"{_problem_id(fun_name, fun_id)} overlap relations extracted: {len(relations)}")
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
    return parser.parse_args(argv)


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
    )
    output_paths = []
    for fun_name in args.functions:
        output_path = Path(args.output_root) / args.timestamp / fun_name
        output_path.mkdir(parents=True, exist_ok=True)
        output_data = {}
        for fun_id in args.ids:
            algorithm = f"{fun_name}_{fun_id}"
            output_data[algorithm] = []
            output_data[f"{algorithm}_time"] = []
            record, elapsed, trace_rows = run_problem(fun_name, fun_id, output_path, config)
            output_data[algorithm].append(record)
            output_data[f"{algorithm}_time"].append(elapsed)
            _write_action_trace(output_path / "action_trace.csv", trace_rows)
            print(f"{algorithm} average time: {elapsed}")
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
