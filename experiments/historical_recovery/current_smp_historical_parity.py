"""Audit the exact EXP-052/current-independent SMP execution boundary."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "current-smp-historical-parity"
DEFAULT_AUDIT_PATH = TASK_ROOT / "raw" / "boundary_audit.json"
DEFAULT_TERMINAL_PROTOCOL = (
    REPOSITORY_ROOT
    / "experiments"
    / "historical_recovery"
    / "current_smp_historical_parity_protocol.json"
)

RECOVERY_ROOT = (
    REPOSITORY_ROOT
    / ".codex-tasks"
    / "historical-level-recovery"
    / "raw"
    / "exp052-e1-seed117-session-reproduction-v1"
)
RECOVERY_RUN_ROOT = (
    RECOVERY_ROOT
    / "runs"
    / "E1"
    / "candidate_smp"
    / "seed_117"
    / "exp_052_e_series_smp_paired_gate-e1-candidate_smp-seed117"
    / "elliptic"
)
RECOVERED_TREE = (
    REPOSITORY_ROOT
    / ".codex-tasks"
    / "historical-level-recovery"
    / "raw"
    / "replay-tree-candidate-v1"
)
AOB_DATA_ROOT = RECOVERED_TREE / "vendor" / "hcc" / "AOB" / "AOBG" / "datafile"
CURRENT_CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "historical_recovery_fixed_expert_v1"
    / "checkpoints"
    / "E1"
    / "seed_117"
    / "checkpoint.json"
)
CURRENT_RESULT_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "historical_recovery_fixed_expert_v1"
    / "arms"
    / "E1"
    / "seed_117"
    / "smp.json"
)
HISTORICAL_TARGET = 5.983267874603139e-7


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_groups() -> tuple[tuple[int, ...], ...]:
    permutation = tuple(
        int(value) - 1
        for value in (AOB_DATA_ROOT / "F1-p.txt").read_text(encoding="utf-8").split(",")
    )
    dimensions = tuple(
        int(float(value))
        for value in (AOB_DATA_ROOT / "F1-s.txt").read_text(encoding="utf-8").split()
    )
    if sum(dimensions) != len(permutation) or len(dimensions) != 20:
        raise ValueError("recovered E1 topology is malformed")
    groups: list[tuple[int, ...]] = []
    offset = 0
    for dimension in dimensions:
        groups.append(permutation[offset : offset + dimension])
        offset += dimension
    return tuple(groups)


def _group_mapping(
    historical: tuple[tuple[int, ...], ...],
    current: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    current_by_members = {frozenset(group): index for index, group in enumerate(current)}
    if len(current_by_members) != len(current):
        raise ValueError("current checkpoint contains duplicate blocks")
    try:
        return tuple(current_by_members[frozenset(group)] for group in historical)
    except KeyError as error:
        raise ValueError("historical/current block partitions differ") from error


def _historical_budget() -> dict[str, int | str]:
    path = RECOVERY_RUN_ROOT / "E1_budget_summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("expected one recovered E1 budget row")
    row = rows[0]
    integer_fields = (
        "max_fes",
        "global_phase_fe",
        "cc_phase_fe",
        "rescue_fe",
        "overhead_fe",
        "budget_aligned_fe",
        "same_budget_violation",
    )
    return {
        **{field: int(row[field]) for field in integer_fields},
        "budget_accounting": row["budget_accounting"],
    }


def build_boundary_audit() -> dict[str, Any]:
    reproduction = _read_json(RECOVERY_ROOT / "reproduction_summary.json")
    historical_action = _read_json(RECOVERY_RUN_ROOT / "smp_action.json")
    current_wrapper = _read_json(CURRENT_CHECKPOINT_PATH)
    current_result = _read_json(CURRENT_RESULT_PATH)
    checkpoint = current_wrapper["checkpoint"]
    action_result = current_result["action_result"]

    historical_groups = _historical_groups()
    current_groups = tuple(tuple(int(value) for value in group) for group in checkpoint["blocks"])
    historical_to_current = _group_mapping(historical_groups, current_groups)
    record_events = [event for event in historical_action["events"] if event["event"] == "record"]
    cold_events = [event for event in historical_action["events"] if event["event"] == "cold_start"]
    budget = _historical_budget()
    current_stateful_fes = int(str(action_result["route"]).split("stateful_block_visits_", 1)[1].split("_", 1)[0])

    source_paths = {
        "historical_runner": RECOVERED_TREE / "scripts" / "hcc_smoke_runner.py",
        "historical_cmaes": RECOVERED_TREE / "vendor" / "hcc" / "HCC" / "OPT" / "CMAES" / "cmaes.py",
        "historical_smp_cache": RECOVERED_TREE / "src" / "arac" / "actions" / "smp.py",
        "current_smp": REPOSITORY_ROOT / "src" / "arac" / "actions" / "smp.py",
        "current_execution": REPOSITORY_ROOT / "src" / "arac" / "actions" / "_execution.py",
    }
    return {
        "schema_version": "arac-current-smp-historical-boundary-audit-v1",
        "case_id": "E1",
        "seed": 117,
        "success_gate": {
            "max_fes": 3_000_000,
            "final_error_lte": HISTORICAL_TARGET,
        },
        "historical": {
            "phase1_fes": 0,
            "initial_incumbent": "zeros",
            "available_action_fes": 3_000_000,
            "final_error": reproduction["result"]["final_error"],
            "fitness_evaluations": reproduction["result"]["fitness_evaluations"],
            "group_dimensions": [len(group) for group in historical_groups],
            "group_count": len(historical_groups),
            "event_count": historical_action["event_count"],
            "visit_count": len(record_events),
            "cold_start_count": len(cold_events),
            "restore_count": historical_action["restore_count"],
            "reset_count": historical_action["reset_count"],
            "abstain_count": historical_action["abstain_count"],
            "state_fields": historical_action["action"]["state_fields"],
            "budget": budget,
            "native_cma_restart": False,
            "boundary_clipping": False,
            "terminal_fill": "deterministic_incumbent_reevaluation",
        },
        "current": {
            "phase1_fes": checkpoint["phase1_fes"],
            "initial_incumbent": "phase1_checkpoint",
            "initial_error": checkpoint["incumbent_error"],
            "available_action_fes": checkpoint["total_budget_fes"] - checkpoint["phase1_fes"],
            "final_error": current_result["final_error"],
            "group_dimensions": [len(group) for group in current_groups],
            "group_count": len(current_groups),
            "consumed_action_fes": action_result["consumed_fes"],
            "stateful_fes": current_stateful_fes,
            "route": action_result["route"],
            "native_cma_restart": False,
            "boundary_clipping": True,
            "terminal_fill": "sepcmaes",
        },
        "group_comparison": {
            "same_unordered_partition": True,
            "same_outer_order": historical_to_current == tuple(range(len(historical_groups))),
            "same_internal_order": all(
                historical == current_groups[current_index]
                for historical, current_index in zip(
                    historical_groups, historical_to_current, strict=True
                )
            ),
            "historical_to_current_indices": list(historical_to_current),
            "current_to_historical_indices": [
                historical_to_current.index(index) for index in range(len(current_groups))
            ],
        },
        "material_mismatches": [
            "phase_boundary_and_available_budget",
            "outer_and_inner_group_order",
            "optimizer_snapshot_restore_lifecycle",
            "per_visit_seed_and_rng_lifecycle",
            "offspring_boundary_handling",
            "incumbent_precheck_accounting",
            "stateful_vs_rescue_budget_ownership",
            "terminal_budget_fill",
        ],
        "confirmed_non_mismatches": [
            "case_id",
            "seed",
            "total_terminal_fes",
            "unordered_group_partition",
            "native_cma_restart_disabled",
            "stale_window_three",
            "initial_sigma_0.5",
        ],
        "budget_delta": {
            "historical_cc_minus_current_stateful_fes": budget["cc_phase_fe"]
            - current_stateful_fes,
        },
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "decision": {
            "current_checkpoint_is_exact_exp052_boundary": False,
            "production_change_authorized": False,
            "terminal_3m_run_authorized": False,
            "next_gate": "two-visit-first-group-lockstep",
        },
    }


def write_boundary_audit(path: Path = DEFAULT_AUDIT_PATH) -> dict[str, Any]:
    audit = build_boundary_audit()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit


def _historical_stage_seed(stage_index: int) -> int:
    payload = f"117:elliptic:1:0:{stage_index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _state_equal(
    historical_optimizer: Any,
    historical_state: tuple[np.ndarray, ...],
    current_session: Any,
) -> bool:
    mean, path_sigma, path_covariance, covariance, eigenvectors, eigenvalues = historical_state
    return (
        historical_optimizer.sigma == current_session.optimizer.sigma
        and historical_optimizer._n_generations == current_session.optimizer._n_generations
        and np.array_equal(mean, current_session.mean)
        and np.array_equal(path_sigma, current_session.path_sigma)
        and np.array_equal(path_covariance, current_session.path_covariance)
        and np.array_equal(covariance, current_session.covariance)
        and np.array_equal(eigenvectors, current_session.eigenvectors)
        and np.array_equal(eigenvalues, current_session.eigenvalues)
        and historical_optimizer.rng_optimization.bit_generator.state
        == current_session.optimizer.rng_optimization.bit_generator.state
    )


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    payload = array.dtype.str.encode("ascii") + str(array.shape).encode("ascii") + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _array_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    historical = np.asarray(left, dtype=np.float64)
    current = np.asarray(right, dtype=np.float64)
    if historical.shape != current.shape:
        return {
            "shape_equal": False,
            "historical_shape": list(historical.shape),
            "current_shape": list(current.shape),
            "bitwise_equal": False,
        }
    unequal = np.flatnonzero(historical.reshape(-1) != current.reshape(-1))
    return {
        "shape_equal": True,
        "historical_sha256": _array_sha256(historical),
        "current_sha256": _array_sha256(current),
        "bitwise_equal": bool(np.array_equal(historical, current)),
        "unequal_count": int(len(unequal)),
        "first_unequal_flat_index": None if len(unequal) == 0 else int(unequal[0]),
        "max_absolute_delta": float(np.max(np.abs(historical - current), initial=0.0)),
    }


def _generation_trace_comparison(
    historical: list[dict[str, np.ndarray]],
    current: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    fields = ("candidates", "full_candidates", "fitness", "steps")
    first_divergence: dict[str, Any] | None = None
    for generation in range(max(len(historical), len(current))):
        if generation >= len(historical) or generation >= len(current):
            first_divergence = {
                "generation": generation,
                "field": "generation_count",
                "historical_present": generation < len(historical),
                "current_present": generation < len(current),
            }
            break
        for field in fields:
            comparison = _array_comparison(
                historical[generation][field],
                current[generation][field],
            )
            if not comparison["bitwise_equal"]:
                first_divergence = {
                    "generation": generation,
                    "field": field,
                    **comparison,
                }
                break
        if first_divergence is not None:
            break

    field_comparisons: dict[str, dict[str, Any]] = {}
    for field in fields:
        historical_values = np.concatenate([row[field] for row in historical], axis=0)
        current_values = np.concatenate([row[field] for row in current], axis=0)
        field_comparisons[field] = _array_comparison(historical_values, current_values)
    return {
        "historical_generations": len(historical),
        "current_generations": len(current),
        "fields": field_comparisons,
        "first_divergence": first_divergence,
        "passed": first_divergence is None,
    }


def run_first_sweep_dual_track(
    group_count: int = 5,
    generations: int = 64,
    visit_fes: int | None = None,
    capture_generation_trace: bool = True,
) -> dict[str, Any]:
    """Compare independent historical/current incumbents over a short first sweep."""

    import sys

    from arac.actions._execution import (
        EARLY_STOPPING_EVALUATIONS,
        _PersistentBlockSession,
        _block_population_size,
        _historical_log_improvement,
        _run_block_visit,
    )
    from arac.benchmarks.aob import AobBenchmark

    groups = _historical_groups()
    if not 1 <= group_count <= len(groups):
        raise ValueError("group_count must be in 1..20")
    if isinstance(generations, bool) or generations <= 0:
        raise ValueError("generations must be a positive integer")
    historical_problem = AobBenchmark(
        vendor_root=REPOSITORY_ROOT / "vendor" / "aob",
        data_root=AOB_DATA_ROOT,
    ).load("E1")
    populations = tuple(_block_population_size(len(group)) for group in groups[:group_count])
    if visit_fes is None:
        visit_budgets = tuple(population * generations for population in populations)
    else:
        if isinstance(visit_fes, bool) or visit_fes <= 0:
            raise ValueError("visit_fes must be a positive integer")
        visit_budgets = tuple(
            visit_fes - visit_fes % population for population in populations
        )
    total_fes = group_count + sum(visit_budgets)
    current_context = _zero_start_context(total_fes)
    historical_incumbent = np.zeros(historical_problem.dimension, dtype=float)
    historical_total_fes = 0

    recovered_vendor = str(RECOVERED_TREE / "vendor" / "hcc")
    if recovered_vendor not in sys.path:
        sys.path.insert(0, recovered_vendor)
    from HCC.OPT.CMAES.cmaes import CMAES as HistoricalCMAES

    rows: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None

    for index, (group, population, visit_budget) in enumerate(
        zip(groups[:group_count], populations, visit_budgets, strict=True)
    ):
        dimensions = np.asarray(group, dtype=int)
        historical_base = historical_incumbent.copy()
        current_base = current_context.ledger.best_x
        base_comparison = _array_comparison(historical_base, current_base)
        historical_before = float(
            np.asarray(historical_problem.objective(historical_base)).reshape(-1)[0]
        )
        historical_total_fes += 1
        current_before = current_context.ledger.evaluate_incumbent(refresh_error=True)
        historical_trace: list[dict[str, np.ndarray]] = []

        def historical_objective(candidate: np.ndarray) -> float | np.ndarray:
            values = np.asarray(candidate, dtype=float)
            batch = values[None, :] if values.ndim == 1 else values
            full = np.repeat(historical_base[None, :], len(batch), axis=0)
            full[:, dimensions] = batch
            result = np.asarray(historical_problem.objective(full), dtype=float).reshape(-1)
            if capture_generation_trace:
                historical_trace.append(
                    {
                        "candidates": batch.copy(),
                        "full_candidates": full.copy(),
                        "fitness": result.copy(),
                    }
                )
            return float(result[0]) if values.ndim == 1 else result

        historical_optimizer = HistoricalCMAES(
            {
                "fitness_function": historical_objective,
                "ndim_problem": len(group),
                "lower_boundary": historical_problem.lower_array[dimensions],
                "upper_boundary": historical_problem.upper_array[dimensions],
            },
            {
                "max_function_evaluations": visit_budget,
                "mean": (historical_base[dimensions],),
                "sigma": 0.5,
                "n_individuals": population,
                "is_restart": False,
                "verbose": 0,
                "early_stopping_evaluations": EARLY_STOPPING_EVALUATIONS,
                "seed_rng": _historical_stage_seed(index + 1),
                "_save_state": True,
            },
        )
        historical_iterate = historical_optimizer.iterate

        def traced_historical_iterate(*args: Any, **kwargs: Any):
            x, fitness, steps = historical_iterate(*args, **kwargs)
            if np.size(fitness):
                historical_trace[-1]["steps"] = np.asarray(steps, dtype=float).copy()
            return x, fitness, steps

        if capture_generation_trace:
            historical_optimizer.iterate = traced_historical_iterate
        historical_result = historical_optimizer.optimize()
        historical_group_fes = int(historical_result["n_function_evaluations"])
        historical_total_fes += historical_group_fes
        historical_candidate = np.asarray(
            historical_result["best_so_far_x"], dtype=float
        ).copy()
        historical_after = float(historical_result["best_so_far_y"])
        historical_writeback = historical_after < historical_before
        if historical_writeback:
            historical_incumbent[dimensions] = historical_candidate
        historical_gain = _historical_log_improvement(
            historical_before,
            min(historical_before, historical_after),
        )

        session = _PersistentBlockSession(
            current_context,
            group,
            index,
            visit_budget,
            population_size=population,
            seed_factory=_historical_stage_seed,
            stage_index=index + 1,
            clip_offspring=False,
        )
        current_trace: list[dict[str, np.ndarray]] = []
        current_objective = session.optimizer.fitness_function

        def traced_current_objective(candidate: np.ndarray) -> float | np.ndarray:
            values = np.asarray(candidate, dtype=float)
            batch = values[None, :] if values.ndim == 1 else values
            full = np.repeat(current_base[None, :], len(batch), axis=0)
            full[:, dimensions] = batch
            result = current_objective(candidate)
            current_trace.append(
                {
                    "candidates": batch.copy(),
                    "full_candidates": full.copy(),
                    "fitness": np.asarray(result, dtype=float).reshape(-1).copy(),
                    "steps": session.steps.copy(),
                }
            )
            return result

        if capture_generation_trace:
            session.optimizer.fitness_function = traced_current_objective
        current_group_fes = _run_block_visit(session, visit_budget)
        current_candidate = np.asarray(
            session.optimizer.best_so_far_x, dtype=float
        ).copy()
        current_after = current_context.ledger.best_error
        current_writeback = current_after < current_before
        current_gain = _historical_log_improvement(current_before, current_after)
        historical_state = historical_result["optimizer_state"]
        trace_comparison = (
            _generation_trace_comparison(historical_trace, current_trace)
            if capture_generation_trace
            else {
                "captured": False,
                "first_divergence": None,
                "passed": True,
            }
        )
        candidate_comparison = _array_comparison(
            historical_candidate,
            current_candidate,
        )
        incumbent_comparison = _array_comparison(
            historical_incumbent,
            current_context.ledger.best_x,
        )
        state_fields = {
            "mean": _array_comparison(historical_state.mean, session.mean),
            "path_sigma": _array_comparison(historical_state.path_sigma, session.path_sigma),
            "path_covariance": _array_comparison(
                historical_state.path_covariance,
                session.path_covariance,
            ),
            "covariance": _array_comparison(
                historical_state.covariance,
                session.covariance,
            ),
            "eigenvectors": _array_comparison(
                historical_state.eigenvectors,
                session.eigenvectors,
            ),
            "eigenvalues": _array_comparison(
                historical_state.eigenvalues,
                session.eigenvalues,
            ),
        }
        state_scalars = {
            "sigma_equal": historical_optimizer.sigma == session.optimizer.sigma,
            "generation_equal": (
                historical_optimizer._n_generations
                == session.optimizer._n_generations
            ),
            "rng_state_equal": (
                historical_optimizer.rng_optimization.bit_generator.state
                == session.optimizer.rng_optimization.bit_generator.state
            ),
            "early_stopping_counter_equal": (
                historical_optimizer._counter_early_stopping
                == session.optimizer._counter_early_stopping
            ),
            "early_stopping_base_equal": (
                historical_optimizer._base_early_stopping
                == session.optimizer._base_early_stopping
            ),
        }
        state_bitwise_equal = all(
            comparison["bitwise_equal"] for comparison in state_fields.values()
        ) and all(state_scalars.values())
        historical_material = historical_gain > math.log(1.01)
        current_material = current_gain > math.log(1.01)
        row = {
            "group_index": index,
            "coordinates": list(group),
            "group_dimension": len(group),
            "population_size": population,
            "stage_index": index + 1,
            "seed": _historical_stage_seed(index + 1),
            "requested_generations": generations,
            "requested_group_fes": visit_budget,
            "incumbent_before": base_comparison,
            "historical_precheck_fitness": historical_before,
            "current_precheck_fitness": current_before,
            "precheck_fitness_bitwise_equal": historical_before == current_before,
            "historical_group_fes": historical_group_fes,
            "current_group_fes": current_group_fes,
            "historical_total_fes": historical_total_fes,
            "current_total_fes": current_context.ledger.count,
            "historical_after": min(historical_before, historical_after),
            "current_after": current_after,
            "historical_log_gain": historical_gain,
            "current_log_gain": current_gain,
            "historical_material_improvement": historical_material,
            "current_material_improvement": current_material,
            "historical_stagnation_streak": 0 if historical_material else 1,
            "current_stagnation_streak": 0 if current_material else 1,
            "historical_writeback": historical_writeback,
            "current_writeback": current_writeback,
            "writeback_equal": historical_writeback == current_writeback,
            "fitness_bitwise_equal": (
                historical_before == current_before
                and min(historical_before, historical_after) == current_after
            ),
            "generation_trace": trace_comparison,
            "candidate": candidate_comparison,
            "incumbent": incumbent_comparison,
            "state_fields": state_fields,
            "state_scalars": state_scalars,
            "state_bitwise_equal": state_bitwise_equal,
            "historical_termination_signal": int(historical_optimizer.termination_signal),
            "current_visit_early_stopped": session.early_stopped,
        }
        row["passed"] = bool(
            base_comparison["bitwise_equal"]
            and row["precheck_fitness_bitwise_equal"]
            and row["historical_group_fes"] == row["current_group_fes"]
            and row["historical_total_fes"] == row["current_total_fes"]
            and row["writeback_equal"]
            and row["fitness_bitwise_equal"]
            and trace_comparison["passed"]
            and candidate_comparison["bitwise_equal"]
            and incumbent_comparison["bitwise_equal"]
            and row["state_bitwise_equal"]
        )
        rows.append(row)
        if first_divergence is None and not row["passed"]:
            first_divergence = {
                "group_index": index,
                "generation_trace": trace_comparison["first_divergence"],
                "incumbent_before_bitwise_equal": base_comparison["bitwise_equal"],
                "candidate_bitwise_equal": candidate_comparison["bitwise_equal"],
                "incumbent_bitwise_equal": incumbent_comparison["bitwise_equal"],
                "fitness_bitwise_equal": row["fitness_bitwise_equal"],
                "state_bitwise_equal": row["state_bitwise_equal"],
                "writeback_equal": row["writeback_equal"],
            }

    return {
        "schema_version": "arac-current-smp-first-sweep-dual-track-v2",
        "case_id": "E1",
        "seed": 117,
        "group_count": group_count,
        "generations_per_group": generations if visit_fes is None else None,
        "requested_visit_fes": visit_fes,
        "generation_trace_captured": capture_generation_trace,
        "population_aligned_visit_budgets": list(visit_budgets),
        "expected_total_fes_per_track": total_fes,
        "historical_total_fes": historical_total_fes,
        "current_total_fes": current_context.ledger.count,
        "independent_problem_instances": historical_problem is not current_context.problem,
        "independent_incumbents": historical_incumbent is not current_context.ledger._best_x,
        "rows": rows,
        "first_divergence": first_divergence,
        "passed": bool(
            first_divergence is None
            and historical_total_fes == current_context.ledger.count
        ),
        "terminal_3m_run_authorized": False,
    }


def run_lockstep(visit_fes: int = 16) -> dict[str, Any]:
    """Compare two population-aligned visits against the recovered CMAES oracle."""

    import sys

    from arac.actions._execution import _PersistentBlockSession, _run_block_visit
    from arac.benchmarks.aob import AobBenchmark
    from arac.runtime.contracts import ActionContext, PhaseCheckpoint
    from arac.runtime.ledger import EvaluationLedger
    problem = AobBenchmark(
        vendor_root=REPOSITORY_ROOT / "vendor" / "aob",
        data_root=AOB_DATA_ROOT,
    ).load("E1")
    sys.path.insert(0, str(RECOVERED_TREE / "vendor" / "hcc"))
    from HCC.OPT.CMAES.cmaes import CMAES as HistoricalCMAES
    from HCC.OPT.CMAES.optimizer import Optimizer as HistoricalOptimizer

    historical_groups = _historical_groups()
    dimensions = np.asarray(historical_groups[0], dtype=int)
    lower = problem.lower_array[dimensions]
    upper = problem.upper_array[dimensions]
    zero = np.zeros(problem.dimension, dtype=float)
    zero_error = float(np.asarray(problem.objective(zero)).reshape(-1)[0])
    population = 4 + 3 * int(np.ceil(np.log(len(dimensions))))
    if visit_fes <= 0 or visit_fes % population:
        raise ValueError("visit_fes must be a positive population multiple")
    checkpoint = PhaseCheckpoint(
        protocol="current-smp-historical-lockstep-v1",
        run_seed=117,
        total_budget_fes=2 * visit_fes,
        phase1_fes=0,
        incumbent=tuple(zero),
        incumbent_error=zero_error,
        feature_names=("lockstep",),
        feature_values=(0.0,),
        blocks=historical_groups,
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=2 * visit_fes,
        phase1_fes=0,
        incumbent=tuple(zero),
        incumbent_error=zero_error,
        allow_out_of_bounds=True,
    )
    context = ActionContext("smp", checkpoint, problem, ledger, action_seed=117)
    session = _PersistentBlockSession(
        context,
        tuple(historical_groups[0]),
        0,
        2 * visit_fes,
        population_size=population,
        seed_factory=_historical_stage_seed,
        stage_index=1,
        clip_offspring=False,
    )

    def objective(base: np.ndarray):
        def evaluate(candidate: np.ndarray) -> float | np.ndarray:
            values = np.asarray(candidate, dtype=float)
            batch = values[None, :] if values.ndim == 1 else values
            full = np.repeat(base[None, :], len(batch), axis=0)
            full[:, dimensions] = batch
            result = np.asarray(problem.objective(full), dtype=float).reshape(-1)
            return float(result[0]) if values.ndim == 1 else result

        return evaluate

    def historical_visit(base: np.ndarray, stage_index: int, state: Any = None):
        optimizer = HistoricalCMAES(
            {
                "fitness_function": objective(base),
                "ndim_problem": len(dimensions),
                "lower_boundary": lower,
                "upper_boundary": upper,
            },
            {
                "max_function_evaluations": visit_fes,
                "mean": (base[dimensions],),
                "sigma": 0.5,
                "n_individuals": population,
                "is_restart": False,
                "verbose": 0,
                "seed_rng": _historical_stage_seed(stage_index),
            },
        )
        HistoricalOptimizer.optimize(optimizer)
        if state is None:
            x, mean, path_sigma, path_covariance, covariance, eigenvectors, eigenvalues, fitness, steps = optimizer.initialize()
        else:
            x, mean, path_sigma, path_covariance, covariance, eigenvectors, eigenvalues, fitness, steps = optimizer.restore_state(
                state,
                mean_override=base[dimensions],
            )
        x, fitness, steps = optimizer.iterate(
            x,
            mean,
            eigenvectors,
            eigenvalues,
            fitness,
            steps,
        )
        mean, path_sigma, path_covariance, covariance, eigenvectors, eigenvalues = optimizer.update_distribution(
            x,
            path_sigma,
            path_covariance,
            covariance,
            eigenvectors,
            eigenvalues,
            fitness,
            steps,
        )
        optimizer._n_generations += 1
        return optimizer, x, fitness, steps, mean, path_sigma, path_covariance, covariance, eigenvectors, eigenvalues

    historical, hx, hy, hd, hm, hps, hpc, hcov, hev, hea = historical_visit(zero, 1)
    _run_block_visit(session, visit_fes)
    first_equal = bool(np.array_equal(hx, session.x) and np.array_equal(hy, session.fitness))
    historical_state = historical.snapshot_state(
        mean=hm,
        path_sigma=hps,
        path_covariance=hpc,
        covariance=hcov,
        eigenvectors=hev,
        eigenvalues=hea,
    )
    first_state_equal = _state_equal(
        historical,
        (hm, hps, hpc, hcov, hev, hea),
        session,
    )
    current_base = ledger.best_x
    historical2, hx2, hy2, hd2, hm2, hps2, hpc2, hcov2, hev2, hea2 = historical_visit(
        current_base,
        21,
        state=historical_state,
    )
    _run_block_visit(session, visit_fes)
    second_equal = bool(np.array_equal(hx2, session.x) and np.array_equal(hy2, session.fitness))
    second_state_equal = _state_equal(
        historical2,
        (hm2, hps2, hpc2, hcov2, hev2, hea2),
        session,
    )
    return {
        "schema_version": "arac-current-smp-two-visit-lockstep-v1",
        "case_id": "E1",
        "seed": 117,
        "group_index": 0,
        "group_dimension": len(dimensions),
        "population_size": population,
        "visit_fes": visit_fes,
        "phase1_fes": 0,
        "historical_stage_indices": [1, 21],
        "current_clip_offspring": False,
        "first_visit": {
            "candidate_matrix_bitwise_equal": first_equal,
            "fitness_bitwise_equal": first_equal,
            "state_bitwise_equal": first_state_equal,
        },
        "second_visit": {
            "candidate_matrix_bitwise_equal": second_equal,
            "fitness_bitwise_equal": second_equal,
            "state_bitwise_equal": second_state_equal,
        },
        "ledger_consumed_fes": ledger.count,
        "passed": all(
            (
                first_equal,
                first_state_equal,
                second_equal,
                second_state_equal,
                ledger.count == 2 * visit_fes,
            )
        ),
        "production_hcc_import_allowed": False,
        "terminal_3m_run_authorized": False,
    }


def _zero_start_context(total_budget: int):
    from arac.benchmarks.aob import AobBenchmark
    from arac.runtime.contracts import ActionContext, PhaseCheckpoint
    from arac.runtime.ledger import EvaluationLedger

    problem = AobBenchmark(
        vendor_root=REPOSITORY_ROOT / "vendor" / "aob",
        data_root=AOB_DATA_ROOT,
    ).load("E1")
    groups = _historical_groups()
    zero = np.zeros(problem.dimension, dtype=float)
    zero_error = float(np.asarray(problem.objective(zero)).reshape(-1)[0])
    checkpoint = PhaseCheckpoint(
        protocol="current-smp-historical-schedule-v1",
        run_seed=117,
        total_budget_fes=total_budget,
        phase1_fes=0,
        incumbent=tuple(zero),
        incumbent_error=zero_error,
        feature_names=("lockstep",),
        feature_values=(0.0,),
        blocks=groups,
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=total_budget,
        phase1_fes=0,
        incumbent=tuple(zero),
        incumbent_error=zero_error,
        allow_out_of_bounds=True,
    )
    return ActionContext("smp", checkpoint, problem, ledger, action_seed=117)


def run_historical_contract_prefix(requested_fes: int = 3_200) -> dict[str, Any]:
    from arac.actions._execution import run_stateful_block_visits_with_sessions

    context = _zero_start_context(requested_fes)
    before = context.ledger.count
    consumed, visits, resets, sessions = run_stateful_block_visits_with_sessions(
        context,
        requested_fes=requested_fes,
        block_order=tuple(range(len(context.checkpoint.blocks))),
        seed_factory=_historical_stage_seed,
        clip_offspring=False,
        precheck_incumbent=True,
        strict_material_gain=True,
    )
    noop_fes = 0
    while context.ledger.remaining:
        context.ledger.evaluate(context.ledger.best_x)
        noop_fes += 1
    return {
        "schema_version": "arac-current-smp-historical-prefix-gate-v1",
        "requested_fes": requested_fes,
        "consumed_fes": consumed,
        "ledger_count": context.ledger.count,
        "precheck_fes": visits,
        "group_fes": consumed - visits,
        "noop_fes": noop_fes,
        "visit_count": visits,
        "restart_count": resets,
        "terminal_state_finite": all(
            np.all(np.isfinite(session.mean))
            and np.isfinite(session.optimizer.sigma)
            and np.all(np.isfinite(session.covariance))
            for session in sessions
        ),
        "zero_start": context.checkpoint.phase1_fes == 0,
        "historical_order": context.checkpoint.blocks == _historical_groups(),
        "seed_schedule": "elliptic:1:0:outer_iter*20+group_index+1",
        "clip_offspring": False,
        "stale_window": 3,
        "passed": (
            before == 0
            and consumed + noop_fes == requested_fes
            and context.ledger.count == requested_fes
            and visits > 0
            and all(
                np.all(np.isfinite(session.mean))
                and np.isfinite(session.optimizer.sigma)
                and np.all(np.isfinite(session.covariance))
                for session in sessions
            )
        ),
        "terminal_3m_run_authorized": False,
    }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_terminal_protocol(path: Path = DEFAULT_TERMINAL_PROTOCOL) -> dict[str, Any]:
    protocol = _read_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-current-smp-historical-parity-protocol-v3",
        "case_id": "E1",
        "seed": 117,
        "total_budget_fes": 3_000_000,
        "phase1_fes": 0,
        "historical_gate_final_error_lte": HISTORICAL_TARGET,
        "short_gate_fes": 3_200,
        "production_hcc_runtime_imports_allowed": False,
        "selector_execution_allowed": False,
        "multi_seed_execution_allowed": False,
        "terminal_run_count": 1,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"terminal protocol drifted: {key}")
    source_paths = protocol.get("source_paths")
    if not isinstance(source_paths, list) or not source_paths:
        raise ValueError("terminal protocol source_paths are missing")
    return protocol


def _production_hcc_imports() -> list[str]:
    matches: list[str] = []
    for path in (REPOSITORY_ROOT / "src" / "arac" / "actions").glob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        if any(token in source for token in ("from hcc", "import hcc", "vendor.hcc", "vendor/hcc")):
            matches.append(str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"))
    return matches


def preflight_terminal(path: Path = DEFAULT_TERMINAL_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_terminal_protocol(protocol_path)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    if output_root.exists():
        raise ValueError(f"terminal output already exists: {output_root}")
    missing = [
        source
        for source in protocol["source_paths"]
        if not (REPOSITORY_ROOT / str(source)).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"terminal protocol sources are missing: {missing}")
    hcc_imports = _production_hcc_imports()
    if hcc_imports:
        raise ValueError(f"production HCC imports remain: {hcc_imports}")
    audit = build_boundary_audit()
    lockstep = run_lockstep()
    short_gate = run_historical_contract_prefix(int(protocol["short_gate_fes"]))
    passed = (
        audit["historical"]["final_error"] == protocol["historical_gate_final_error_lte"]
        and lockstep["passed"] is True
        and short_gate["passed"] is True
        and short_gate["ledger_count"] == protocol["short_gate_fes"]
    )
    return {
        "schema_version": "arac-current-smp-historical-preflight-v1",
        "protocol_sha256": _sha256(protocol_path),
        "output_root": str(output_root.resolve()),
        "production_hcc_runtime_imports": hcc_imports,
        "audit_source_sha256": audit["source_sha256"],
        "lockstep": lockstep,
        "short_gate": short_gate,
        "passed": passed,
        "terminal_run_authorized": passed,
    }


def _terminal_manifest(
    protocol_path: Path,
    protocol: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": "arac-current-smp-historical-manifest-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha256(protocol_path),
        "preflight_sha256": _canonical_sha256(preflight),
        "source_sha256": {
            str(source): _sha256(REPOSITORY_ROOT / str(source))
            for source in protocol["source_paths"]
        },
        "production_hcc_runtime_imports": _production_hcc_imports(),
        "terminal_run_count": 1,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def _execute_terminal_contract(total_budget_fes: int) -> dict[str, Any]:
    from arac.actions._execution import run_stateful_block_visits_with_sessions

    context = _zero_start_context(total_budget_fes)
    events: list[dict[str, object]] = []
    consumed, visits, resets, sessions = run_stateful_block_visits_with_sessions(
        context,
        requested_fes=total_budget_fes,
        block_order=tuple(range(len(context.checkpoint.blocks))),
        seed_factory=_historical_stage_seed,
        clip_offspring=False,
        precheck_incumbent=True,
        strict_material_gain=True,
        event_trace=events,
    )
    noop_fes = 0
    while context.ledger.remaining:
        context.ledger.evaluate(context.ledger.best_x)
        noop_fes += 1
    return {
        "consumed_fes": context.ledger.count,
        "group_fes": consumed - visits,
        "precheck_fes": visits,
        "noop_fes": noop_fes,
        "visit_count": visits,
        "restart_count": resets,
        "cold_start_count": sum(event["route"] == "cold_start" for event in events),
        "restore_count": sum(event["route"] == "restore" for event in events),
        "events": events,
        "final_error": context.ledger.best_error,
        "terminal_state_finite": all(
            np.all(np.isfinite(session.mean))
            and np.isfinite(session.optimizer.sigma)
            and np.all(np.isfinite(session.covariance))
            for session in sessions
        ),
        "route": (
            f"historical_contract_group_{consumed - visits}_precheck_{visits}_"
            f"noop_{noop_fes}_visits_{visits}_resets_{resets}"
        ),
    }


def run_terminal(path: Path = DEFAULT_TERMINAL_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_terminal_protocol(protocol_path)
    preflight = preflight_terminal(protocol_path)
    if preflight["terminal_run_authorized"] is not True:
        raise RuntimeError("terminal preflight did not authorize the run")
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    output_root.mkdir(parents=True)
    manifest = _terminal_manifest(protocol_path, protocol, preflight)
    _write_json(output_root / "protocol.json", protocol)
    _write_json(output_root / "preflight.json", preflight)
    _write_json(output_root / "manifest.json", manifest)
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            pools = [
                {
                    "internal_api": item.get("internal_api"),
                    "num_threads": item.get("num_threads"),
                    "prefix": item.get("prefix"),
                }
                for item in threadpool_info()
            ]
            if not pools or any(item["num_threads"] != 1 for item in pools):
                raise RuntimeError(f"native thread limit is not one: {pools}")
            result = _execute_terminal_contract(int(protocol["total_budget_fes"]))
    except BaseException as error:
        _write_json(
            output_root / "failure.json",
            {
                "schema_version": "arac-current-smp-historical-failure-v1",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    body = {
        "schema_version": "arac-current-smp-historical-receipt-v1",
        "case_id": protocol["case_id"],
        "seed": protocol["seed"],
        "phase1_fes": protocol["phase1_fes"],
        "total_budget_fes": protocol["total_budget_fes"],
        "historical_gate_final_error_lte": protocol["historical_gate_final_error_lte"],
        "historical_level_recovered_or_exceeded": (
            result["final_error"] <= protocol["historical_gate_final_error_lte"]
        ),
        "result": result,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "threadpools": pools,
        "native_thread_limit_verified": True,
        "production_hcc_runtime_imports": _production_hcc_imports(),
        "selector_execution_allowed": False,
    }
    receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
    _write_json(output_root / "receipt.json", receipt)
    summary_body = {
        "schema_version": "arac-current-smp-historical-summary-v1",
        "final_error": result["final_error"],
        "historical_target": protocol["historical_gate_final_error_lte"],
        "absolute_delta": result["final_error"] - protocol["historical_gate_final_error_lte"],
        "terminal_fes": result["consumed_fes"],
        "historical_level_recovered_or_exceeded": receipt[
            "historical_level_recovered_or_exceeded"
        ],
        "receipt_sha256": receipt["receipt_sha256"],
    }
    summary = {**summary_body, "summary_sha256": _canonical_sha256(summary_body)}
    _write_json(output_root / "summary.json", summary)
    return summary


def verify_terminal(path: Path = DEFAULT_TERMINAL_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_terminal_protocol(protocol_path)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    manifest = _read_json(output_root / "manifest.json")
    manifest_hash = manifest.pop("manifest_sha256")
    if manifest_hash != _canonical_sha256(manifest):
        raise ValueError("terminal manifest hash drifted")
    if manifest["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("terminal protocol hash drifted")
    expected_sources = {
        str(source): _sha256(REPOSITORY_ROOT / str(source))
        for source in protocol["source_paths"]
    }
    if manifest["source_sha256"] != expected_sources:
        raise ValueError("terminal source hashes drifted")
    preflight = _read_json(output_root / "preflight.json")
    if preflight["passed"] is not True or preflight["terminal_run_authorized"] is not True:
        raise ValueError("stored terminal preflight did not pass")
    if manifest["preflight_sha256"] != _canonical_sha256(preflight):
        raise ValueError("stored terminal preflight hash drifted")
    receipt = _read_json(output_root / "receipt.json")
    receipt_hash = receipt.pop("receipt_sha256")
    if receipt_hash != _canonical_sha256(receipt):
        raise ValueError("terminal receipt hash drifted")
    if (
        receipt["case_id"] != "E1"
        or receipt["seed"] != 117
        or receipt["phase1_fes"] != 0
        or receipt["result"]["consumed_fes"] != 3_000_000
        or receipt["result"]["terminal_state_finite"] is not True
        or receipt["native_thread_limit_verified"] is not True
        or receipt["production_hcc_runtime_imports"] != []
        or receipt["selector_execution_allowed"] is not False
    ):
        raise ValueError("terminal receipt contract failed")
    summary = _read_json(output_root / "summary.json")
    summary_hash = summary.pop("summary_sha256")
    if summary_hash != _canonical_sha256(summary):
        raise ValueError("terminal summary hash drifted")
    if summary["receipt_sha256"] != receipt_hash:
        raise ValueError("terminal summary receipt binding drifted")
    return {**summary, "summary_sha256": summary_hash}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "lockstep",
            "prefix_diff",
            "short_gate",
            "preflight",
            "run",
            "verify",
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_TERMINAL_PROTOCOL)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = write_boundary_audit(args.output or DEFAULT_AUDIT_PATH)
    elif args.command == "lockstep":
        result = run_lockstep()
        output = args.output or (TASK_ROOT / "raw" / "lockstep.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "prefix_diff":
        result = run_first_sweep_dual_track()
        output = args.output or (TASK_ROOT / "raw" / "first_sweep_dual_track.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "short_gate":
        result = run_historical_contract_prefix()
        output = args.output or (TASK_ROOT / "raw" / "short_gate.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "preflight":
        result = preflight_terminal(args.protocol)
    elif args.command == "run":
        result = run_terminal(args.protocol)
    else:
        result = verify_terminal(args.protocol)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
