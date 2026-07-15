from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
VENDOR_ROOT = REPO_ROOT / "vendor" / "hcc"
for import_root in (REPO_ROOT, SRC_ROOT, VENDOR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from HCC.NDAs.MMES.mmes import MMES
from HCC.OPT.CMAES.cmaes import CMAES
from AOB.AOB import Benchmark
from AOB.utils import combine

from scripts import hcc_smoke_runner as hcc_runner


RUN_ID = "exp_009_aob_e1_e4_convergence"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "exp_009_aob_e1_e4_convergence_seed1_3m"
DEFAULT_MAX_FES = 3_000_000
DEFAULT_CHECKPOINTS = 1_200
DEFAULT_SEED = 1
DEFAULT_CASES = ("E1", "E2", "E3", "E4")
METHOD_ORDER = ("ARAC-v33.8", "HCC_ES", "MMES", "RDDSM_CMAES")
CASE_TO_FUNCTION = {
    "E1": ("elliptic", 1),
    "E2": ("elliptic", 2),
    "E3": ("elliptic", 3),
    "E4": ("elliptic", 4),
}

METHOD_COLORS = {
    "ARAC-v33.8": "#d62728",
    "HCC_ES": "#9467bd",
    "MMES": "#2ca02c",
    "RDDSM_CMAES": "#1f77b4",
}


@dataclass(frozen=True)
class RunResult:
    case: str
    method: str
    seed: int
    elapsed_seconds: float
    actual_fes: int
    final_best: float
    curve_path: Path
    artifact_dir: Path


def parse_case(case: str) -> tuple[str, int]:
    normalized = str(case).strip().upper()
    if normalized not in CASE_TO_FUNCTION:
        raise ValueError(f"unsupported case: {case}; expected one of {DEFAULT_CASES}")
    return CASE_TO_FUNCTION[normalized]


def _derived_seed(seed: int, case: str, stage: int) -> int:
    payload = f"{seed}:exp009:{case}:{stage}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _best_so_far(record: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(record, dtype=float).reshape(-1)
    if values.size == 0:
        raise RuntimeError("optimizer returned an empty fitness record")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("fitness record contains non-finite values")
    return np.minimum.accumulate(values)


def downsample_curve(
    record: list[float] | np.ndarray,
    max_fes: int,
    checkpoints: int,
) -> list[tuple[int, float]]:
    """Return a deterministic, lossless-at-end best-so-far downsample."""
    best = _best_so_far(record)
    usable = min(int(max_fes), int(best.size))
    if usable < 1:
        raise RuntimeError("fitness record has no evaluations within the FE budget")
    count = min(max(2, int(checkpoints)), usable)
    fe_points = np.unique(np.linspace(1, usable, count, dtype=int))
    return [(int(fe), float(best[fe - 1])) for fe in fe_points]


def write_curve(path: Path, points: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fe", "best_so_far"))
        writer.writerows(points)


def _problem_payload(fun_name: str, fun_id: int, info: dict[str, object], fun) -> dict[str, object]:
    dimension = int(info["dimension"])
    return {
        "fitness_function": fun,
        "ndim_problem": dimension,
        "lower_boundary": float(info["lower"]) * np.ones(dimension),
        "upper_boundary": float(info["upper"]) * np.ones(dimension),
    }


def _run_mmes(case: str, seed: int, max_fes: int, artifact_dir: Path) -> list[float]:
    fun_name, fun_id = parse_case(case)
    bench = Benchmark(str(artifact_dir) + "/", data_dir=hcc_runner.DATA_DIR)
    fun = bench.get_function(fun_name, fun_id)
    info = bench.get_info(fun_name, fun_id)
    problem = _problem_payload(fun_name, fun_id, info, fun)
    options = {
        "max_function_evaluations": int(max_fes),
        "mean": (np.zeros(int(info["dimension"])),),
        "sigma": 0.5,
        "is_restart": True,
        "verbose": 0,
        "seed_rng": _derived_seed(seed, case, 0),
    }
    MMES(problem, options).optimize()
    return fun.fitness_record


def _run_rddsm_cmaes(case: str, seed: int, max_fes: int, artifact_dir: Path) -> list[float]:
    """Run pure RDDSM/CMA-ES CC with strict accounting and no paper values."""
    fun_name, fun_id = parse_case(case)
    bench = Benchmark(str(artifact_dir) + "/", data_dir=hcc_runner.DATA_DIR)
    fun = bench.get_function(fun_name, fun_id)
    info = bench.get_info(fun_name, fun_id)
    groups = hcc_runner.decompose_problem(fun_id, hcc_runner.DATA_DIR)
    best = np.zeros(int(info["dimension"]))
    group_count = len(groups)
    outer_iter = 0

    while len(fun.fitness_record) < max_fes:
        for group_index, dims in enumerate(groups):
            remaining = max_fes - len(fun.fitness_record)
            if remaining <= 0:
                break
            groups_left = group_count - group_index
            scheduled = max(1, int(np.ceil(remaining / groups_left)))
            before = best.copy()
            before_fitness = float(fun(best)[0])
            remaining_after_probe = max_fes - len(fun.fitness_record)
            optimizer_budget = min(scheduled - 1, remaining_after_probe)
            if optimizer_budget > 0:
                objective = lambda x_batch: fun(combine(x_batch, best, dims))
                problem = {
                    "fitness_function": objective,
                    "ndim_problem": len(dims),
                    "lower_boundary": float(info["lower"]) * np.ones(len(dims)),
                    "upper_boundary": float(info["upper"]) * np.ones(len(dims)),
                }
                options = {
                    "max_function_evaluations": optimizer_budget,
                    "mean": (best[dims],),
                    "sigma": 0.5,
                    "n_individuals": hcc_runner.calculate_cmaes_population_size(len(dims)),
                    "is_restart": True,
                    "verbose": 0,
                    "early_stopping_evaluations": 1000,
                    "seed_rng": _derived_seed(seed, case, 1 + outer_iter * group_count + group_index),
                }
                result = CMAES(problem, options).optimize()
                candidate = np.asarray(result["best_so_far_x"], dtype=float)
                candidate_best = float(result["best_so_far_y"])
                if candidate_best < before_fitness:
                    best[dims] = candidate
                else:
                    best = before
            if len(fun.fitness_record) >= max_fes:
                break
        outer_iter += 1
    return fun.fitness_record[:max_fes]


def _run_hcc_or_arac(
    case: str,
    method: str,
    seed: int,
    max_fes: int,
    artifact_dir: Path,
) -> list[float]:
    fun_name, fun_id = parse_case(case)
    if method == "ARAC-v33.8":
        action = hcc_runner.EVIDENCE_ACTION_CONTROLLER_V33
        enable_relation_dispatch = True
        relation_policy = "controller_v31"
    elif method == "HCC_ES":
        action = "conservative_no_action"
        enable_relation_dispatch = False
        relation_policy = "rule"
    else:
        raise ValueError(f"unsupported HCC-family method: {method}")
    config = hcc_runner.SmokeConfig(
        run_id=f"{RUN_ID}-{case}-{method}-seed{seed}",
        max_fes=int(max_fes),
        seed=int(seed),
        verbose=0,
        early_stopping_evaluations=1000,
        mmes_restart=True,
        cmaes_restart=True,
        arac_action=action,
        enable_relation_dispatch=enable_relation_dispatch,
        relation_policy_mode=relation_policy,
        budget_accounting="strict",
        skip_plots=True,
        aob_data_root=hcc_runner.DATA_DIR,
        search_state_backend="phase_i_mmes",
    )
    record, _elapsed, _trace = hcc_runner.run_problem(
        fun_name,
        fun_id,
        artifact_dir,
        config,
    )
    return record


def run_one(
    case: str,
    method: str,
    seed: int,
    max_fes: int,
    checkpoints: int,
    output_dir: Path,
) -> RunResult:
    if method not in METHOD_ORDER:
        raise ValueError(f"unsupported method: {method}")
    case = case.upper()
    artifact_dir = output_dir / "runs" / case / f"seed_{seed}" / method
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if method == "MMES":
        record = _run_mmes(case, seed, max_fes, artifact_dir)
    elif method == "RDDSM_CMAES":
        record = _run_rddsm_cmaes(case, seed, max_fes, artifact_dir)
    else:
        record = _run_hcc_or_arac(case, method, seed, max_fes, artifact_dir)
    budget_record = list(record[:max_fes])
    points = downsample_curve(budget_record, max_fes=max_fes, checkpoints=checkpoints)
    curve_path = artifact_dir / "convergence.csv"
    write_curve(curve_path, points)
    elapsed = time.perf_counter() - started
    return RunResult(
        case=case,
        method=method,
        seed=seed,
        elapsed_seconds=elapsed,
        actual_fes=len(budget_record),
        final_best=float(_best_so_far(budget_record)[-1]),
        curve_path=curve_path,
        artifact_dir=artifact_dir,
    )


def _run_one_worker(args: tuple[str, str, int, int, int, str]) -> RunResult:
    case, method, seed, max_fes, checkpoints, output_dir = args
    return run_one(case, method, seed, max_fes, checkpoints, Path(output_dir))


def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"empty convergence file: {path}")
    fe = np.atleast_1d(data["fe"]).astype(float)
    best = np.atleast_1d(data["best_so_far"]).astype(float)
    return fe, best


def plot_figure(
    output_dir: Path,
    cases: tuple[str, ...],
    seed: int,
    max_fes: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8), constrained_layout=True)
    axes = axes.reshape(-1)
    for index, case in enumerate(cases):
        ax = axes[index]
        for method in METHOD_ORDER:
            curve_path = output_dir / "runs" / case / f"seed_{seed}" / method / "convergence.csv"
            fe, best = _read_curve(curve_path)
            ax.plot(
                fe,
                best,
                color=METHOD_COLORS[method],
                linewidth=2.1 if method == "ARAC-v33.8" else 1.7,
                linestyle="-",
                label=method,
                solid_capstyle="round",
            )
        ax.set_title("Best-so-Far Evaluation Curves for Different Algorithms", fontsize=10)
        ax.text(
            -0.13,
            1.04,
            case,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.set_yscale("log")
        ax.set_xlim(0, max_fes)
        ax.xaxis.set_major_locator(MaxNLocator(7))
        x_formatter = ScalarFormatter(useMathText=False)
        x_formatter.set_powerlimits((6, 6))
        ax.xaxis.set_major_formatter(x_formatter)
        ax.grid(True, which="major", color="#b0b0b0", linewidth=0.7, alpha=0.55)
        ax.grid(True, which="minor", color="#d9d9d9", linewidth=0.45, alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("FEs", fontsize=11)
        ax.set_ylabel("Objective Value (log10)", fontsize=11)
        ax.legend(loc="best", fontsize=8.3, frameon=True, framealpha=0.9)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "aob_e1_e4_convergence.png",
        output_dir / "aob_e1_e4_convergence.pdf",
        output_dir / "aob_e1_e4_convergence.svg",
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)
    return paths


def _write_manifest(output_dir: Path, results: list[RunResult], args: argparse.Namespace) -> None:
    rows = []
    for result in sorted(results, key=lambda item: (item.case, METHOD_ORDER.index(item.method))):
        rows.append(
            {
                "case": result.case,
                "method": result.method,
                "seed": result.seed,
                "max_fes": args.max_fes,
                "actual_fes": result.actual_fes,
                "final_best": f"{result.final_best:.17e}",
                "elapsed_seconds": f"{result.elapsed_seconds:.6f}",
                "curve_csv": str(result.curve_path.relative_to(output_dir)),
                "artifact_dir": str(result.artifact_dir.relative_to(output_dir)),
                "curve_type": "single_seed_best_so_far",
            }
        )
    path = output_dir / "convergence_data_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "run_id": RUN_ID,
        "protocol": "real_3m_fe_single_seed_convergence",
        "seed": args.seed,
        "cases": list(args.cases),
        "methods": list(METHOD_ORDER),
        "max_fes": args.max_fes,
        "checkpoints": args.checkpoints,
        "solid_line_semantics": "one real seed; no mean or variance band",
        "runtime_dispatch_constraints": "paper values, case labels, function family and historical outcomes are offline only",
        "baseline_definitions": {
            "ARAC-v33.8": "evidence_action_controller_v33 with controller_v31 relation policy",
            "HCC_ES": "MMES global phase plus RDDSM/CMA-ES cooperative phase, conservative writeback",
            "MMES": "full-space MMES only",
            "RDDSM_CMAES": "pure RDDSM grouping plus CMA-ES cooperative phase, no MMES global phase",
        },
        "source_paths": {
            "aob_data_root": str(hcc_runner.DATA_DIR),
            "hcc_runner": str(REPO_ROOT / "scripts" / "hcc_smoke_runner.py"),
            "hcc_source_reference": str(Path("E:/HCC-main/2025_HCC_GECCO-main/HCC_SRC/HCC-ES.py")),
        },
        "results": [row for row in rows],
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-fes", type=int, default=DEFAULT_MAX_FES)
    parser.add_argument("--checkpoints", type=int, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="仅使用已经保存的 convergence.csv 生成图，不启动优化器",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_fes < 1 or args.checkpoints < 2:
        raise SystemExit("--max-fes must be >= 1 and --checkpoints must be >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    if not args.plot_only:
        tasks = [
            (case, method, args.seed, args.max_fes, args.checkpoints, str(args.output_dir))
            for case in args.cases
            for method in METHOD_ORDER
        ]
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
            for result in executor.map(_run_one_worker, tasks):
                results.append(result)
                print(
                    f"[{result.case}/{result.method}] FE={result.actual_fes} "
                    f"best={result.final_best:.6e} time={result.elapsed_seconds:.1f}s",
                    flush=True,
                )
        _write_manifest(args.output_dir, results, args)
    paths = plot_figure(args.output_dir, tuple(args.cases), args.seed, args.max_fes)
    print("Generated:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
