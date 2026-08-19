"""Run one family-mapped fixed expert from each shared Phase-I checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

import experiments.final.run as final_run
from arac.runtime.contracts import ACTION_NAMES
from experiments.audit_historical_recovery import parse_target


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("fixed_expert_config.json")
CONFIG_SCHEMA = "arac-historical-fixed-expert-config-v1"
SUMMARY_SCHEMA = "arac-historical-fixed-expert-summary-v1"
RUN_STATE_SCHEMA = "arac-historical-fixed-expert-run-state-v1"
EXPECTED_CASES = tuple(f"{family}{index}" for family in "AERS" for index in range(1, 7))
EXPECTED_SEEDS = tuple(range(117, 142))
EXPECTED_MAPPING = {"A": "aor", "E": "smp", "R": "gcb", "S": "ctp"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_json(Path(path).resolve())
    required = {
        "schema_version",
        "cases",
        "seeds",
        "expert_mapping",
        "max_fes",
        "max_workers",
        "output_root",
        "historical_table",
        "historical_target_column",
        "aggregate_precision",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("fixed-expert config schema or keys drifted")
    cases = tuple(str(value).upper() for value in config["cases"])
    seeds = tuple(int(value) for value in config["seeds"])
    mapping = {str(key): str(value) for key, value in config["expert_mapping"].items()}
    if cases != EXPECTED_CASES:
        raise ValueError("fixed-expert recovery requires all 24 AOB cases in fixed order")
    if seeds != EXPECTED_SEEDS:
        raise ValueError("fixed-expert recovery requires historical seeds 117..141")
    if mapping != EXPECTED_MAPPING or set(mapping.values()) != set(ACTION_NAMES):
        raise ValueError("fixed-expert mapping drifted")
    if int(config["max_fes"]) != 3_000_000:
        raise ValueError("fixed-expert recovery requires exactly 3,000,000 FE")
    workers = int(config["max_workers"])
    if not 1 <= workers <= 24:
        raise ValueError("fixed-expert recovery workers must be in 1..24")
    if config["aggregate_precision"] != ".2E":
        raise ValueError("historical aggregate precision drifted")
    return config


def build_contexts(
    cases: Sequence[str],
    seeds: Sequence[int],
    mapping: Mapping[str, str],
    *,
    max_fes: int,
    output_root: Path,
) -> tuple[tuple[final_run.CheckpointContext, ...], tuple[final_run.ArmContext, ...]]:
    checkpoints = tuple(
        final_run.CheckpointContext(str(case), int(seed), max_fes, output_root)
        for seed in seeds
        for case in cases
    )
    arms = tuple(
        final_run.ArmContext(
            str(case),
            int(seed),
            str(mapping[str(case)[0]]),
            max_fes,
            output_root,
        )
        for seed in seeds
        for case in cases
    )
    return checkpoints, arms


def _source_paths() -> dict[str, Path]:
    return {
        **final_run.SOURCE_PATHS,
        "historical_fixed_expert_campaign": Path(__file__).resolve(),
    }


def _source_hashes() -> dict[str, str]:
    return {
        name: final_run.file_sha256(path)
        for name, path in sorted(_source_paths().items())
    }


def _manifest(config: dict[str, Any], config_path: Path) -> dict[str, object]:
    return final_run._campaign_manifest(
        "historical_fixed_expert",
        config["cases"],
        config["seeds"],
        max_fes=int(config["max_fes"]),
        config_path=config_path,
        source_hashes=_source_hashes(),
        vendor_trees=final_run._vendor_tree_hashes(),
    )


def _freeze_inputs(
    output_root: Path,
    config_path: Path,
    config: dict[str, Any],
    *,
    resume: bool,
) -> None:
    protocol_root = output_root / "frozen_protocol"
    source_root = protocol_root / "sources"
    config_copy = protocol_root / "config.json"
    table_copy = protocol_root / "historical_targets.csv"
    source_paths = _source_paths()
    if resume:
        if final_run.file_sha256(config_copy) != final_run.file_sha256(config_path):
            raise ValueError("frozen fixed-expert config drifted")
        if final_run.file_sha256(table_copy) != final_run.file_sha256(
            REPOSITORY_ROOT / config["historical_table"]
        ):
            raise ValueError("frozen historical target table drifted")
        for name, source in source_paths.items():
            copy = source_root / f"{name}{source.suffix}"
            if final_run.file_sha256(copy) != final_run.file_sha256(source):
                raise ValueError(f"frozen fixed-expert source drifted: {name}")
        return
    source_root.mkdir(parents=True)
    shutil.copy2(config_path, config_copy)
    shutil.copy2(REPOSITORY_ROOT / config["historical_table"], table_copy)
    for name, source in source_paths.items():
        shutil.copy2(source, source_root / f"{name}{source.suffix}")


def _historical_targets(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    path = REPOSITORY_ROOT / config["historical_table"]
    targets: dict[str, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            case_id = row["case"].strip()
            if case_id in EXPECTED_CASES:
                targets[case_id] = parse_target(row[config["historical_target_column"]])
    if set(targets) != set(EXPECTED_CASES):
        raise ValueError("historical target table does not contain all 24 cases")
    return targets


def summarize_rows(rows: Sequence[Mapping[str, object]], config: dict[str, Any]) -> dict[str, object]:
    targets = _historical_targets(config)
    precision = str(config["aggregate_precision"])
    case_summaries = []
    for case_id in config["cases"]:
        case_rows = [row for row in rows if row["case_id"] == case_id]
        values = [float(row["final_error"]) for row in case_rows]
        if len(values) != len(config["seeds"]):
            raise ValueError(f"fixed-expert result coverage is incomplete: {case_id}")
        mean = fmean(values)
        sample_std = stdev(values)
        target_mean, target_std = targets[case_id]
        mean_match = format(mean, precision) == format(target_mean, precision)
        std_match = format(sample_std, precision) == format(target_std, precision)
        case_summaries.append(
            {
                "case": case_id,
                "action": config["expert_mapping"][case_id[0]],
                "seed_count": len(values),
                "mean": mean,
                "sample_std": sample_std,
                "historical_mean": target_mean,
                "historical_sample_std": target_std,
                "formatted_mean": format(mean, precision).upper(),
                "formatted_sample_std": format(sample_std, precision).upper(),
                "formatted_historical_mean": format(target_mean, precision).upper(),
                "formatted_historical_sample_std": format(target_std, precision).upper(),
                "mean_match": mean_match,
                "sample_std_match": std_match,
                "recovered": mean_match and std_match,
            }
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "context_count": len(rows),
        "case_count": len(case_summaries),
        "seed_count_per_case": len(config["seeds"]),
        "max_fes": int(config["max_fes"]),
        "max_workers": int(config["max_workers"]),
        "all_terminal_fes_exact": all(
            int(row["terminal_fes"]) == int(config["max_fes"]) for row in rows
        ),
        "recovered_case_count": sum(row["recovered"] for row in case_summaries),
        "gate_passed": all(row["recovered"] for row in case_summaries),
        "case_summaries": case_summaries,
    }
    summary.update(final_run._runtime_warning_summary(rows))
    return summary


def _write_results(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "case_id",
        "run_seed",
        "action_name",
        "phase1_fes",
        "terminal_fes",
        "final_error",
        "checkpoint_hash",
        "action_result_hash",
        "receipt_hash",
    )
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})
    temporary.replace(path)


def run_campaign(config_path: Path = DEFAULT_CONFIG, *, resume: bool = False) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    output_root = (REPOSITORY_ROOT / config["output_root"]).resolve()
    manifest = _manifest(config, config_path)
    final_run._prepare_campaign_root(output_root, manifest, resume=resume)
    _freeze_inputs(output_root, config_path, config, resume=resume)
    final_run._atomic_json(
        output_root / "run_state.json",
        {
            "schema_version": RUN_STATE_SCHEMA,
            "status": "running",
            "pid": os.getpid(),
            "max_workers": int(config["max_workers"]),
        },
    )
    checkpoints, arms = build_contexts(
        config["cases"],
        config["seeds"],
        config["expert_mapping"],
        max_fes=int(config["max_fes"]),
        output_root=output_root,
    )
    try:
        final_run._run_parallel(
            checkpoints,
            final_run._run_checkpoint,
            max_workers=int(config["max_workers"]),
            progress_path=output_root / "checkpoint_progress.json",
            receipt_path=lambda context: context.receipt_path,
            validator=final_run._validate_checkpoint,
            resume=resume,
        )
        rows = final_run._run_parallel(
            arms,
            final_run._run_arm,
            max_workers=int(config["max_workers"]),
            progress_path=output_root / "parallel_progress.json",
            receipt_path=lambda context: context.receipt_path,
            validator=final_run._validate_arm,
            resume=resume,
        )
        summary = summarize_rows(rows, config)
        final_run._atomic_json(output_root / "summary.json", summary)
        _write_results(output_root / "results.csv", rows)
        final_run._require_known_runtime_warnings(summary, stage="historical fixed-expert campaign")
    except BaseException as error:
        final_run._atomic_json(
            output_root / "run_state.json",
            {
                "schema_version": RUN_STATE_SCHEMA,
                "status": "failed",
                "pid": os.getpid(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    final_run._atomic_json(
        output_root / "run_state.json",
        {
            "schema_version": RUN_STATE_SCHEMA,
            "status": "completed",
            "pid": os.getpid(),
            "gate_passed": summary["gate_passed"],
        },
    )
    return summary


def preflight(config_path: Path = DEFAULT_CONFIG, *, resume: bool = False) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    output_root = (REPOSITORY_ROOT / config["output_root"]).resolve()
    if resume:
        final_run._prepare_campaign_root(
            output_root,
            _manifest(config, config_path),
            resume=True,
        )
    elif output_root.exists():
        raise ValueError(f"fixed-expert output already exists: {output_root}")
    return {
        "context_count": len(config["cases"]) * len(config["seeds"]),
        "checkpoint_count": len(config["cases"]) * len(config["seeds"]),
        "arm_count": len(config["cases"]) * len(config["seeds"]),
        "max_workers": int(config["max_workers"]),
        "output_root": str(output_root),
        "source_inputs_valid": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        preflight(args.config, resume=args.resume)
        if args.command == "preflight"
        else run_campaign(args.config, resume=args.resume)
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
