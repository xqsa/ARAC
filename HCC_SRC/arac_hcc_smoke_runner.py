from __future__ import annotations

import argparse
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

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


@dataclass(frozen=True)
class SmokeConfig:
    max_fes: int
    seed: int | None
    sigma: float = 0.5
    verbose: int = 1000
    early_stopping_evaluations: int = 1000
    cmaes_restart: bool = False
    arac_action: str = "conservative_no_action"


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


def run_problem(fun_name: str, fun_id: int, output_path: Path, config: SmokeConfig) -> tuple[list[float], float]:
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
                best_individual[overlap_indices] = apply_arac_overlap_action(
                    action_name=config.arac_action,
                    previous_values=original_best[overlap_indices],
                    current_values=best_individual[overlap_indices],
                    previous_delta=fitness_delta_list[index - 1],
                    current_delta=current_delta,
                )
        outer_iter += 1

    return fun.fitness_record, time.time() - time_start


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
        max_fes=args.max_fes,
        seed=args.seed,
        verbose=args.verbose,
        early_stopping_evaluations=args.early_stopping_evaluations,
        cmaes_restart=args.cmaes_restart,
        arac_action=args.arac_action,
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
            record, elapsed = run_problem(fun_name, fun_id, output_path, config)
            output_data[algorithm].append(record)
            output_data[f"{algorithm}_time"].append(elapsed)
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
