"""Run R1-R6 with the frozen GCB action for 25 seeds at exactly 3M FE."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from experiments.pilots.exp_026_arac_vs_hcc_paired import run as relation_run
from experiments.pilots.exp_027_r1_gcb import run as boundary_run


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_029_r_series_gcb_25seed"

PROTOCOL_VERSION = "r-series-gcb-terminal-25seed-v1"
EXPERIMENT_ID = "exp_029_r_series_gcb_25seed"
CASES = ("R1", "R2", "R3", "R4", "R5", "R6")
SEEDS = tuple(range(117, 142))
EXACT_MAX_FES = 3_000_000
ACTION = "gcb"
DEFAULT_JOBS = 20


@dataclass(frozen=True)
class RunSpec:
    case: str
    seed: int
    output_root: Path

    @property
    def trajectory_id(self) -> str:
        return f"{EXPERIMENT_ID}-{self.case.lower()}-seed{self.seed}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON config: {path}") from error
    _require(isinstance(payload, dict), "config must be a JSON object")
    return payload


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config = _read_json(path)
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "protocol changed")
    _require(config.get("experiment_id") == EXPERIMENT_ID, "experiment id changed")
    execution = config.get("execution")
    _require(isinstance(execution, dict), "execution config is missing")
    _require(tuple(execution.get("cases", ())) == CASES, "case matrix changed")
    _require(tuple(execution.get("seeds", ())) == SEEDS, "seed schedule changed")
    _require(execution.get("max_fes") == EXACT_MAX_FES, "FE budget changed")
    _require(execution.get("action") == ACTION, "GCB action changed")
    _require(execution.get("jobs") == DEFAULT_JOBS, "default concurrency changed")
    _require(execution.get("native_resume_sweeps") == 3, "native resume window changed")

    sources = config.get("source_protocols")
    _require(isinstance(sources, dict), "source protocol config is missing")
    _require(
        sources.get("r1") == boundary_run.PROTOCOL_VERSION,
        "R1 source protocol changed",
    )
    _require(
        sources.get("r2_r6") == relation_run.PROTOCOL_VERSION,
        "R2-R6 source protocol changed",
    )
    _require(
        sources.get("r1_artifact_schema")
        == boundary_run.GCB_ACTION_ARTIFACT_SCHEMA,
        "R1 artifact schema changed",
    )
    _require(
        sources.get("r2_r6_artifact_schema")
        == relation_run.GCB_ACTION_ARTIFACT_SCHEMA,
        "R2-R6 artifact schema changed",
    )
    return config


def build_specs(output_root: Path) -> tuple[RunSpec, ...]:
    specs = tuple(
        RunSpec(case=case, seed=seed, output_root=output_root)
        for seed in SEEDS
        for case in CASES
    )
    _require(len(specs) == 150, "experiment must contain exactly 150 trajectories")
    _require(
        len({(spec.case, spec.seed) for spec in specs}) == len(specs),
        "trajectory keys must be unique",
    )
    return specs


def _backend_spec(spec: RunSpec) -> object:
    if spec.case == "R1":
        return boundary_run.RunSpec(
            seed=spec.seed,
            output_root=spec.output_root,
            experiment_id=EXPERIMENT_ID,
        )
    return relation_run.RunSpec(
        experiment_id=EXPERIMENT_ID,
        case=spec.case,
        seed=spec.seed,
        action=ACTION,
        output_root=spec.output_root,
    )


def _run_backend(
    spec: RunSpec,
    *,
    boundary_config: Mapping[str, object],
    relation_config: Mapping[str, object],
    python_executable: str,
    run_subprocess: bool,
) -> dict[str, object]:
    backend_spec = _backend_spec(spec)
    if spec.case == "R1":
        _require(
            isinstance(backend_spec, boundary_run.RunSpec),
            "R1 backend spec mismatch",
        )
        return boundary_run.run_one(
            backend_spec,
            boundary_config,
            python_executable,
            run_subprocess=run_subprocess,
        )
    _require(
        isinstance(backend_spec, relation_run.RunSpec),
        "relation backend spec mismatch",
    )
    return relation_run.run_one(
        backend_spec,
        relation_config,
        python_executable,
        run_subprocess=run_subprocess,
    )


def run_one(
    spec: RunSpec,
    *,
    boundary_config: Mapping[str, object],
    relation_config: Mapping[str, object],
    python_executable: str,
    resume: bool,
    reuse_existing: bool,
) -> dict[str, object]:
    if resume or reuse_existing:
        existing = _run_backend(
            spec,
            boundary_config=boundary_config,
            relation_config=relation_config,
            python_executable=python_executable,
            run_subprocess=False,
        )
        if existing.get("ok") is True or reuse_existing:
            return existing
    else:
        existing = None

    result = _run_backend(
        spec,
        boundary_config=boundary_config,
        relation_config=relation_config,
        python_executable=python_executable,
        run_subprocess=True,
    )
    if resume:
        result["execution_source"] = "rerun_after_artifact_gate_failure"
        if existing is not None:
            result["resume_gate_error"] = existing.get("error", "missing artifact")
    return result


def build_case_summaries(
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for case in CASES:
        rows = sorted(
            (
                row
                for row in results
                if row.get("case") == case and row.get("ok") is True
            ),
            key=lambda row: int(row["seed"]),
        )
        _require(len(rows) == len(SEEDS), f"{case} requires 25 valid trajectories")
        _require(
            tuple(int(row["seed"]) for row in rows) == SEEDS,
            f"{case} seed schedule mismatch",
        )
        values = [float(row["final_error"]) for row in rows]
        summaries.append(
            {
                "case": case,
                "n": len(values),
                "mean_error": statistics.fmean(values),
                "sample_std_error": statistics.stdev(values),
                "seed_final_errors": [
                    {"seed": int(row["seed"]), "final_error": float(row["final_error"])}
                    for row in rows
                ],
            }
        )
    return summaries


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(payload))
    temporary.replace(path)


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _progress(result: Mapping[str, object], completed: int) -> None:
    suffix = (
        f" final_error={float(result['final_error']):.12e}"
        if result.get("ok") is True
        else f" error={result.get('error') or result.get('stderr_tail', '')}"
    )
    print(
        f"[{completed:03d}/150] {result['case']}/seed{result['seed']} "
        f"{result['status']} source={result['execution_source']}{suffix}",
        flush=True,
    )


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python_executable: str = sys.executable,
    jobs: int | None = None,
    resume: bool = False,
    reuse_existing: bool = False,
    progress_callback: Callable[[Mapping[str, object], int], None] = _progress,
) -> tuple[list[dict[str, object]], list[dict[str, object]] | None, dict[str, object]]:
    _require(not (resume and reuse_existing), "resume and reuse_existing are exclusive")
    config = load_config(config_path)
    boundary_config = boundary_run.load_config()
    relation_config = relation_run.load_config()
    execution = config["execution"]
    assert isinstance(execution, dict)
    workers = int(execution["jobs"]) if jobs is None else jobs
    _require(workers > 0, "jobs must be positive")

    results: list[dict[str, object]] = []
    specs = build_specs(output_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_one,
                spec,
                boundary_config=boundary_config,
                relation_config=relation_config,
                python_executable=python_executable,
                resume=resume,
                reuse_existing=reuse_existing,
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda row: (str(row["case"]), int(row["seed"])))
            progress_callback(result, len(results))
            _write_json(output_root / "run_progress.json", {"results": results})

    completed_count = sum(result.get("ok") is True for result in results)
    summaries = build_case_summaries(results) if completed_count == len(specs) else None
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "trajectory_count": len(results),
        "completed_trajectory_count": completed_count,
        "integrity_gate_passed": completed_count == 150 and summaries is not None,
        "cases": list(CASES),
        "seeds": list(SEEDS),
        "max_fes": EXACT_MAX_FES,
        "action": ACTION,
        "worker_count": workers,
        "execution_mode": (
            "offline_validation" if reuse_existing else "resume" if resume else "fresh"
        ),
        "config_sha256": _source_hash(config_path),
        "r1_source_config_sha256": _source_hash(boundary_run.DEFAULT_CONFIG_PATH),
        "r2_r6_source_config_sha256": _source_hash(relation_run.DEFAULT_CONFIG_PATH),
        "reused_trajectory_count": sum(
            result.get("execution_source") in {"offline_validation", "reused_valid_artifact"}
            for result in results
        ),
        "executed_trajectory_count": sum(
            result.get("execution_source")
            in {"fresh_execution", "rerun_after_artifact_gate_failure"}
            for result in results
        ),
    }
    _write_json(
        output_root / "run_summary.json",
        {**manifest, "case_summaries": summaries, "results": results},
    )
    return results, summaries, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs is not None and args.jobs <= 0:
        parser.error("--jobs must be positive")
    _results, summaries, manifest = run_experiment(
        config_path=args.config,
        output_root=args.output_root,
        python_executable=args.python_executable,
        jobs=args.jobs,
        resume=args.resume,
        reuse_existing=args.reuse_existing,
    )
    if summaries is not None:
        for summary in summaries:
            print(
                f"[{summary['case']}] n={summary['n']} "
                f"mean={float(summary['mean_error']):.12e} "
                f"std={float(summary['sample_std_error']):.12e}",
                flush=True,
            )
    print(f"Summary: {args.output_root / 'run_summary.json'}", flush=True)
    return 0 if manifest["integrity_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
