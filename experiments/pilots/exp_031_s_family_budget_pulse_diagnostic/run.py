"""Run and validate the five-seed S5 budget-pulse action diagnostic."""

from __future__ import annotations

# The standalone experiment entry point must register ``src`` before local imports.
# ruff: noqa: E402

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from random import Random
import subprocess
import sys
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arac.actions.budget_reallocation import (
    FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
)
from arac.actions.shrunk_budget_pulse import (
    SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
)
from arac.backends.hcc import required_aob_data_files
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
    ACTION_CEILING_HORIZONS,
    ACTION_CEILING_TIE_TOLERANCE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CATASTROPHIC_DELTA,
    MATERIAL_POSITIVE_DELTA,
    PRIMARY_HORIZON,
    S_FAMILY_BUDGET_PULSE_ARMS,
    S_FAMILY_BUDGET_PULSE_PROFILE,
    SPARSE_POSITIVE_THRESHOLD,
)
from experiments.pilots.exp_030_s_family_budget_pulse import run as exp030


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "results" / "exp_031_s_family_budget_pulse_diagnostic"
)

PROTOCOL_VERSION = "exp031-s-family-budget-pulse-diagnostic-v1"
EXPERIMENT_ID = "exp_031_s_family_budget_pulse_diagnostic"
SOURCE_ACTION_PROTOCOL = exp030.PROTOCOL_VERSION
COHORT = "real_aob"
CASE = "S5"
SEEDS = (117, 118, 119, 120, 121)
CONFIGURED_MAX_FES = 300_000
TERMINAL_FE_POLICY = exp030.TERMINAL_FE_POLICY
DEFAULT_JOBS = 1
EXPECTED_CONTEXTS_PER_SEED = 4
MINIMUM_VALID_CONTEXTS = 10
EXPECTED_TOTAL_CONTEXTS = EXPECTED_CONTEXTS_PER_SEED * len(SEEDS)
EXPECTED_TOTAL_ARM_ROWS = (
    EXPECTED_TOTAL_CONTEXTS
    * len(S_FAMILY_BUDGET_PULSE_ARMS)
    * len(ACTION_CEILING_HORIZONS)
)
BUDGET_PULSE_ARMS = (
    FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
    SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
)
REGISTERED_PULSE_ARMS = ("native_eq8", *BUDGET_PULSE_ARMS)
AGGREGATE_CONTEXT_NAME = "action_ceiling_contexts.csv"
AGGREGATE_ARM_NAME = "action_ceiling_arm_results.csv"
DIAGNOSTIC_NAME = "action_diagnostic.json"
MANIFEST_NAME = "manifest.json"
EXECUTION_RECEIPT_NAME = "exp031_execution_receipt.json"
EXECUTION_RECEIPT_VERSION = "exp031-execution-receipt-v1"

SOURCE_FILES = tuple(
    sorted(
        {Path(__file__).resolve(), *exp030.SOURCE_FILES},
        key=lambda path: path.as_posix(),
    )
)


@dataclass(frozen=True)
class RunSpec:
    seed: int
    output_root: Path

    @property
    def trajectory_id(self) -> str:
        return f"{EXPERIMENT_ID}-s5-seed{self.seed}"

    @property
    def artifact_dir(self) -> Path:
        return self.output_root.resolve() / self.trajectory_id / "schwefel"


@dataclass(frozen=True)
class ValidatedRun:
    spec: RunSpec
    artifacts: exp030.ValidatedArtifacts
    execution_source: str
    resume_gate_error: str = ""
    generation_source: str = ""
    generation_resume_gate_error: str = ""
    receipt_path: Path | None = None
    artifact_sha256: tuple[tuple[str, str], ...] = ()
    receipt_sha256: str = ""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    _require(isinstance(payload, dict), f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                tuple(reader.fieldnames or ()) == tuple(fields),
                f"CSV schema mismatch: {path}",
            )
            return list(reader)
    except OSError as error:
        raise ValueError(f"missing or unreadable CSV artifact: {path}") from error


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _sha256_file(path)
        for path in SOURCE_FILES
    }


def _aob_input_hashes() -> dict[str, str]:
    paths = required_aob_data_files(exp030.AOB_DATA_ROOT, 5)
    _require(
        bool(paths) and all(path.is_file() for path in paths),
        "required S5 AOB inputs are missing",
    )
    return {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): (
            _sha256_file(path)
        )
        for path in paths
    }


def _execution_identity(config_path: Path) -> dict[str, object]:
    resolved_config = config_path.resolve()
    return {
        "receipt_version": EXECUTION_RECEIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_action_protocol": SOURCE_ACTION_PROTOCOL,
        "experiment_id": EXPERIMENT_ID,
        "case": CASE,
        "configured_max_fes": CONFIGURED_MAX_FES,
        "profile": S_FAMILY_BUDGET_PULSE_PROFILE,
        "config_path": str(resolved_config),
        "config_sha256": _sha256_file(resolved_config),
        "source_sha256": _source_hashes(),
        "aob_input_sha256": _aob_input_hashes(),
    }


def _execution_receipt(
    spec: RunSpec,
    identity: Mapping[str, object],
    *,
    artifact_sha256: Mapping[str, str],
    generation_source: str,
    generation_resume_gate_error: str,
) -> dict[str, object]:
    _require(
        generation_source
        in {"fresh_execution", "rerun_after_artifact_gate_failure"},
        "invalid receipt generation source",
    )
    _require(
        (generation_source == "fresh_execution" and not generation_resume_gate_error)
        or (
            generation_source == "rerun_after_artifact_gate_failure"
            and bool(generation_resume_gate_error)
        ),
        "receipt generation error provenance is inconsistent",
    )
    return {
        **identity,
        "seed": spec.seed,
        "trajectory_id": spec.trajectory_id,
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "generation_source": generation_source,
        "generation_resume_gate_error": generation_resume_gate_error,
    }


def _validate_execution_receipt(
    path: Path,
    spec: RunSpec,
    identity: Mapping[str, object],
    artifact_sha256: Mapping[str, str],
) -> dict[str, object]:
    receipt = _read_json(path)
    generation_source = receipt.get("generation_source")
    generation_error = receipt.get("generation_resume_gate_error")
    expected_identity = {
        **identity,
        "seed": spec.seed,
        "trajectory_id": spec.trajectory_id,
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
    }
    identity_fields = tuple(expected_identity)
    _require(
        set(receipt)
        == set(expected_identity)
        | {"generation_source", "generation_resume_gate_error"}
        and all(receipt.get(field) == expected_identity[field] for field in identity_fields)
        and generation_source
        in {"fresh_execution", "rerun_after_artifact_gate_failure"}
        and isinstance(generation_error, str)
        and (
            (generation_source == "fresh_execution" and not generation_error)
            or (
                generation_source == "rerun_after_artifact_gate_failure"
                and bool(generation_error)
            )
        ),
        "execution receipt source/config/protocol mismatch",
    )
    return receipt


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config = _read_json(path)
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "protocol changed")
    _require(config.get("experiment_id") == EXPERIMENT_ID, "experiment id changed")
    _require(config.get("stage") == "action_validation_diagnostic", "stage changed")
    _require(
        config.get("source_action_protocol") == SOURCE_ACTION_PROTOCOL,
        "source action protocol changed",
    )
    _require(
        config.get("execution")
        == {
            "profile": S_FAMILY_BUDGET_PULSE_PROFILE,
            "cohort": COHORT,
            "case": CASE,
            "seeds": list(SEEDS),
            "max_fes": CONFIGURED_MAX_FES,
            "terminal_fe_policy": TERMINAL_FE_POLICY,
            "jobs": DEFAULT_JOBS,
            "expected_contexts_per_seed": EXPECTED_CONTEXTS_PER_SEED,
            "minimum_valid_contexts": MINIMUM_VALID_CONTEXTS,
            "expected_total_contexts": EXPECTED_TOTAL_CONTEXTS,
            "expected_total_arm_rows": EXPECTED_TOTAL_ARM_ROWS,
            "resume_policy": "validate_first_then_rerun_invalid",
        },
        "execution matrix changed",
    )
    _require(
        config.get("statistics")
        == {
            "primary_horizon": PRIMARY_HORIZON,
            "cluster_unit": "case_seed",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "material_positive_delta": MATERIAL_POSITIVE_DELTA,
            "catastrophic_delta": CATASTROPHIC_DELTA,
            "sparse_positive_threshold": SPARSE_POSITIVE_THRESHOLD,
            "tie_policy": "native_first",
        },
        "statistics contract changed",
    )
    _require(
        config.get("authorization")
        == {
            "action_ceiling_inference_authorized": 1,
            "real_aob_pilot_may_be_authorized": 1,
            "action_gate_authorized": 0,
            "evidence_separability_authorized": 0,
            "selector_authorized": 0,
            "runtime_authorized": 0,
            "primary_scope": "s5_budget_pulse_diagnostic_only",
        },
        "authorization contract changed",
    )
    return config


def build_specs(output_root: Path) -> tuple[RunSpec, ...]:
    specs = tuple(RunSpec(seed, output_root.resolve()) for seed in SEEDS)
    _require(len(specs) == 5, "diagnostic must contain exactly five trajectories")
    _require(len({spec.seed for spec in specs}) == len(specs), "duplicate seed spec")
    return specs


def build_worker_command(spec: RunSpec, python_executable: str) -> tuple[str, ...]:
    return (
        python_executable,
        "-m",
        exp030.WORKER_MODULE,
        "--cohort",
        COHORT,
        "--case",
        CASE,
        "--seed",
        str(spec.seed),
        "--max-fes",
        str(CONFIGURED_MAX_FES),
        "--output-root",
        str(spec.output_root),
        "--timestamp",
        spec.trajectory_id,
        "--profile",
        S_FAMILY_BUDGET_PULSE_PROFILE,
    )


def _run_worker(spec: RunSpec, python_executable: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(SOURCE_ROOT), environment.get("PYTHONPATH", ""))
        if value
    )
    subprocess.run(
        build_worker_command(spec, python_executable),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def _seed_artifact_sha256(
    spec: RunSpec,
    artifacts: exp030.ValidatedArtifacts,
) -> dict[str, str]:
    hashes = {
        str(path.relative_to(spec.output_root.resolve())).replace("\\", "/"): (
            _sha256_file(path)
        )
        for path in artifacts.artifact_paths
    }
    _require(
        len(hashes) == len(artifacts.artifact_paths),
        "seed artifact paths are duplicated",
    )
    return dict(sorted(hashes.items()))


def _validated_artifact_snapshot(
    spec: RunSpec,
) -> tuple[exp030.ValidatedArtifacts, dict[str, str]]:
    first = exp030.validate_artifacts(
        spec.artifact_dir,
        expected_seed=spec.seed,
        expected_run_id=spec.trajectory_id,
    )
    first_hashes = _seed_artifact_sha256(spec, first)
    second = exp030.validate_artifacts(
        spec.artifact_dir,
        expected_seed=spec.seed,
        expected_run_id=spec.trajectory_id,
    )
    second_hashes = _seed_artifact_sha256(spec, second)
    _require(
        first_hashes == second_hashes,
        "seed artifacts changed while being validated",
    )
    return second, second_hashes


def _validate_spec(
    spec: RunSpec,
    *,
    execution_source: str,
    execution_identity: Mapping[str, object],
    resume_gate_error: str = "",
) -> ValidatedRun:
    artifacts, artifact_sha256 = _validated_artifact_snapshot(spec)
    receipt_path = spec.artifact_dir / EXECUTION_RECEIPT_NAME
    try:
        receipt_sha256_before = _sha256_file(receipt_path)
    except OSError as error:
        raise ValueError(
            f"execution receipt is missing or unreadable: {receipt_path}"
        ) from error
    receipt = _validate_execution_receipt(
        receipt_path,
        spec,
        execution_identity,
        artifact_sha256,
    )
    try:
        receipt_sha256 = _sha256_file(receipt_path)
    except OSError as error:
        raise ValueError(
            f"execution receipt changed while being validated: {receipt_path}"
        ) from error
    _require(
        receipt_sha256 == receipt_sha256_before,
        "execution receipt changed while being validated",
    )
    return ValidatedRun(
        spec,
        artifacts,
        execution_source,
        resume_gate_error,
        str(receipt["generation_source"]),
        str(receipt["generation_resume_gate_error"]),
        receipt_path,
        tuple(artifact_sha256.items()),
        receipt_sha256,
    )


def run_one(
    spec: RunSpec,
    *,
    python_executable: str,
    config_path: Path,
    execution_identity: Mapping[str, object],
    resume: bool,
    reuse_existing: bool,
) -> ValidatedRun:
    _require(not (resume and reuse_existing), "resume and reuse_existing are exclusive")
    resume_gate_error = ""
    if resume or reuse_existing:
        try:
            return _validate_spec(
                spec,
                execution_source=(
                    "offline_validation" if reuse_existing else "reused_valid_artifact"
                ),
                execution_identity=execution_identity,
            )
        except ValueError as error:
            if reuse_existing:
                raise
            resume_gate_error = str(error)
            print(
                f"[resume-invalid] S5/seed{spec.seed}: {resume_gate_error}",
                flush=True,
            )
    _run_worker(spec, python_executable)
    _artifacts, artifact_sha256 = _validated_artifact_snapshot(spec)
    _require(
        _execution_identity(config_path) == dict(execution_identity),
        "source or config changed while generating a seed artifact",
    )
    generation_source = (
        "rerun_after_artifact_gate_failure" if resume else "fresh_execution"
    )
    receipt_path = spec.artifact_dir / EXECUTION_RECEIPT_NAME
    _write_json(
        receipt_path,
        _execution_receipt(
            spec,
            execution_identity,
            artifact_sha256=artifact_sha256,
            generation_source=generation_source,
            generation_resume_gate_error=resume_gate_error,
        ),
    )
    return _validate_spec(
        spec,
        execution_source=generation_source,
        execution_identity=execution_identity,
        resume_gate_error=resume_gate_error,
    )


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot average an empty sequence")
    return math.fsum(float(value) for value in values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _best_arm(deltas: Mapping[str, float], arms: Sequence[str]) -> tuple[str, float]:
    _require(set(deltas) >= set(arms), "context is missing a registered arm")
    best_value = max(float(deltas[arm]) for arm in arms)
    for arm in arms:
        if math.isclose(
            float(deltas[arm]),
            best_value,
            rel_tol=0.0,
            abs_tol=ACTION_CEILING_TIE_TOLERANCE,
        ):
            return arm, float(deltas[arm])
    raise RuntimeError("native-first arm tie-break failed")


def _vbs_values(
    contexts: Mapping[str, Mapping[str, object]],
    context_ids: Sequence[str],
    arms: Sequence[str],
) -> tuple[list[float], dict[str, int]]:
    values: list[float] = []
    winners = {arm: 0 for arm in arms}
    for context_id in context_ids:
        deltas = contexts[context_id]["deltas"]
        assert isinstance(deltas, dict)
        winner, value = _best_arm(deltas, arms)
        values.append(value)
        winners[winner] += 1
    return values, winners


def _metric_summary(
    values: Sequence[float],
    bootstrap_means: Sequence[float],
) -> dict[str, object]:
    material = [value > MATERIAL_POSITIVE_DELTA for value in values]
    catastrophic = [value <= CATASTROPHIC_DELTA for value in values]
    return {
        "mean_delta": _mean(values),
        "delta_lcb": _quantile(bootstrap_means, 0.025),
        "delta_ucb": _quantile(bootstrap_means, 0.975),
        "min_delta": min(values),
        "max_delta": max(values),
        "positive_count": sum(value > 0.0 for value in values),
        "positive_rate": _mean([float(value > 0.0) for value in values]),
        "material_positive_count": sum(material),
        "material_positive_rate": _mean([float(value) for value in material]),
        "catastrophic_count": sum(catastrophic),
        "catastrophic_rate": _mean([float(value) for value in catastrophic]),
    }


def summarize_diagnostic(
    arm_rows: Sequence[Mapping[str, str]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    _require(bootstrap_replicates > 0, "bootstrap_replicates must be positive")
    primary_rows = tuple(row for row in arm_rows if row.get("horizon") == PRIMARY_HORIZON)
    contexts: dict[str, dict[str, object]] = {}
    for row in primary_rows:
        context_id = str(row.get("context_id", ""))
        arm = str(row.get("arm", ""))
        seed = int(str(row.get("seed", "-1")))
        delta = float(str(row.get("delta", "nan")))
        _require(
            context_id
            and arm in S_FAMILY_BUDGET_PULSE_ARMS
            and seed in SEEDS
            and math.isfinite(delta),
            "primary arm observation is invalid",
        )
        context = contexts.setdefault(context_id, {"seed": seed, "deltas": {}})
        _require(context["seed"] == seed, "context seed changed across arms")
        deltas = context["deltas"]
        assert isinstance(deltas, dict)
        _require(arm not in deltas, "duplicate primary context arm")
        deltas[arm] = delta

    _require(len(contexts) == EXPECTED_TOTAL_CONTEXTS, "unexpected primary context count")
    clusters: dict[int, list[str]] = {seed: [] for seed in SEEDS}
    for context_id, context in contexts.items():
        deltas = context["deltas"]
        assert isinstance(deltas, dict)
        _require(
            set(deltas) == set(S_FAMILY_BUDGET_PULSE_ARMS),
            "primary context arm set is incomplete",
        )
        _require(
            float(deltas["native_eq8"]) == 0.0,
            "native arm must be the zero-delta reference",
        )
        clusters[int(context["seed"])].append(context_id)
    _require(
        all(len(clusters[seed]) == EXPECTED_CONTEXTS_PER_SEED for seed in SEEDS),
        "each seed must contribute exactly four contexts",
    )

    context_ids = tuple(sorted(contexts))
    arm_values = {
        arm: [float(contexts[context_id]["deltas"][arm]) for context_id in context_ids]
        for arm in S_FAMILY_BUDGET_PULSE_ARMS
    }
    rejected_arms = tuple(
        arm
        for arm in BUDGET_PULSE_ARMS
        if any(value <= CATASTROPHIC_DELTA for value in arm_values[arm])
    )
    eligible_pulse_arms = tuple(
        arm for arm in BUDGET_PULSE_ARMS if arm not in rejected_arms
    )
    eligible_vbs_arms = ("native_eq8", *eligible_pulse_arms)

    registered_vbs, registered_winners = _vbs_values(
        contexts,
        context_ids,
        S_FAMILY_BUDGET_PULSE_ARMS,
    )
    pulse_vbs, pulse_winners = _vbs_values(
        contexts,
        context_ids,
        REGISTERED_PULSE_ARMS,
    )
    eligible_vbs, eligible_winners = _vbs_values(
        contexts,
        context_ids,
        eligible_vbs_arms,
    )

    rng = Random(int(bootstrap_seed))
    bootstrap_arm_means = {arm: [] for arm in S_FAMILY_BUDGET_PULSE_ARMS}
    bootstrap_registered_vbs: list[float] = []
    bootstrap_pulse_vbs: list[float] = []
    bootstrap_eligible_vbs: list[float] = []
    bootstrap_eligible_material: list[float] = []
    cluster_keys = tuple(SEEDS)
    for _replicate in range(bootstrap_replicates):
        sampled_contexts: list[str] = []
        for _cluster in cluster_keys:
            sampled_seed = cluster_keys[rng.randrange(len(cluster_keys))]
            sampled_contexts.extend(clusters[sampled_seed])
        for arm in S_FAMILY_BUDGET_PULSE_ARMS:
            bootstrap_arm_means[arm].append(
                _mean(
                    [float(contexts[key]["deltas"][arm]) for key in sampled_contexts]
                )
            )
        sampled_registered, _ = _vbs_values(
            contexts,
            sampled_contexts,
            S_FAMILY_BUDGET_PULSE_ARMS,
        )
        sampled_pulse, _ = _vbs_values(
            contexts,
            sampled_contexts,
            REGISTERED_PULSE_ARMS,
        )
        sampled_eligible, _ = _vbs_values(
            contexts,
            sampled_contexts,
            eligible_vbs_arms,
        )
        bootstrap_registered_vbs.append(_mean(sampled_registered))
        bootstrap_pulse_vbs.append(_mean(sampled_pulse))
        bootstrap_eligible_vbs.append(_mean(sampled_eligible))
        bootstrap_eligible_material.append(
            _mean(
                [float(value > MATERIAL_POSITIVE_DELTA) for value in sampled_eligible]
            )
        )

    arm_statistics = {
        arm: _metric_summary(arm_values[arm], bootstrap_arm_means[arm])
        for arm in S_FAMILY_BUDGET_PULSE_ARMS
    }
    registered_sbs_arm, registered_sbs_mean = _best_arm(
        {arm: float(stats["mean_delta"]) for arm, stats in arm_statistics.items()},
        S_FAMILY_BUDGET_PULSE_ARMS,
    )
    eligible_sbs_arm, eligible_sbs_mean = _best_arm(
        {arm: float(arm_statistics[arm]["mean_delta"]) for arm in eligible_vbs_arms},
        eligible_vbs_arms,
    )
    positive_fixed_arms = tuple(
        arm
        for arm in eligible_pulse_arms
        if float(arm_statistics[arm]["delta_lcb"]) > 0.0
    )
    fixed_action_candidate = None
    if positive_fixed_arms:
        fixed_action_candidate, _ = _best_arm(
            {
                arm: float(arm_statistics[arm]["mean_delta"])
                for arm in positive_fixed_arms
            },
            positive_fixed_arms,
        )
    eligible_vbs_summary = _metric_summary(
        eligible_vbs,
        bootstrap_eligible_vbs,
    )
    eligible_vbs_summary.update(
        {
            "material_positive_lcb": _quantile(bootstrap_eligible_material, 0.025),
            "material_positive_ucb": _quantile(bootstrap_eligible_material, 0.975),
            "winner_counts": eligible_winners,
        }
    )

    if not eligible_pulse_arms or float(eligible_vbs_summary["mean_delta"]) <= 0.0:
        recommendation = "redesign_budget_pulse_actions"
    elif float(eligible_vbs_summary["delta_lcb"]) <= 0.0:
        recommendation = "collect_more_s5_ceiling_contexts"
    elif (
        float(eligible_vbs_summary["material_positive_ucb"])
        < SPARSE_POSITIVE_THRESHOLD
    ):
        recommendation = "force_abstain_sparse_s5_headroom"
    elif fixed_action_candidate is not None:
        recommendation = "advance_fixed_budget_pulse_to_real_aob_pilot"
    else:
        recommendation = "advance_budget_pulse_portfolio_to_real_aob_pilot"

    s5_diagnostic_passed = recommendation in {
        "advance_fixed_budget_pulse_to_real_aob_pilot",
        "advance_budget_pulse_portfolio_to_real_aob_pilot",
    }
    fixed_action_target = (
        fixed_action_candidate
        if recommendation == "advance_fixed_budget_pulse_to_real_aob_pilot"
        else None
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_action_protocol": SOURCE_ACTION_PROTOCOL,
        "cohort": COHORT,
        "problem_id": CASE,
        "primary_horizon": PRIMARY_HORIZON,
        "context_count": len(contexts),
        "cluster_count": len(clusters),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "arm_statistics": arm_statistics,
        "registered_sbs": {
            "arm": registered_sbs_arm,
            "mean_delta": registered_sbs_mean,
            "delta_lcb": arm_statistics[registered_sbs_arm]["delta_lcb"],
            "delta_ucb": arm_statistics[registered_sbs_arm]["delta_ucb"],
        },
        "runtime_eligible_budget_sbs": {
            "arm": eligible_sbs_arm,
            "mean_delta": eligible_sbs_mean,
            "delta_lcb": arm_statistics[eligible_sbs_arm]["delta_lcb"],
            "delta_ucb": arm_statistics[eligible_sbs_arm]["delta_ucb"],
        },
        "registered_vbs": {
            **_metric_summary(registered_vbs, bootstrap_registered_vbs),
            "winner_counts": registered_winners,
        },
        "registered_budget_pulse_vbs": {
            **_metric_summary(pulse_vbs, bootstrap_pulse_vbs),
            "winner_counts": pulse_winners,
        },
        "runtime_eligible_budget_pulse_vbs": eligible_vbs_summary,
        "rejected_catastrophic_budget_arms": list(rejected_arms),
        "runtime_eligible_budget_arms": list(eligible_pulse_arms),
        "positive_fixed_budget_arms": list(positive_fixed_arms),
        "fixed_action_validation_target": fixed_action_target,
        "control_headroom_only": bool(
            registered_winners.get("true_no_writeback", 0)
            and max(pulse_vbs) <= 0.0
        ),
        "primary_recommendation": recommendation,
        "s5_action_diagnostic_passed": int(s5_diagnostic_passed),
        "real_aob_pilot_authorized": int(s5_diagnostic_passed),
        "fixed_action_real_aob_pilot_authorized": int(
            s5_diagnostic_passed and fixed_action_target is not None
        ),
        "action_gate_authorized": 0,
        "fixed_action_validation_authorized": 0,
        "evidence_separability_authorized": 0,
        "selector_authorized": 0,
        "runtime_authorized": 0,
    }


def _aggregate_rows(
    runs: Sequence[ValidatedRun],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    _require(
        tuple(sorted(run.spec.seed for run in runs)) == SEEDS,
        "validated trajectory seed coverage is incomplete",
    )
    context_rows = [row for run in runs for row in run.artifacts.context_rows]
    arm_rows = [row for run in runs for row in run.artifacts.arm_rows]
    arm_order = {arm: index for index, arm in enumerate(S_FAMILY_BUDGET_PULSE_ARMS)}
    horizon_order = {horizon: index for index, horizon in enumerate(ACTION_CEILING_HORIZONS)}
    context_rows.sort(key=lambda row: (int(row["seed"]), row["context_id"]))
    arm_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            row["context_id"],
            arm_order[row["arm"]],
            horizon_order[row["horizon"]],
        )
    )
    _require(
        len(context_rows) == EXPECTED_TOTAL_CONTEXTS
        and len(context_rows) >= MINIMUM_VALID_CONTEXTS
        and len({row["context_id"] for row in context_rows}) == len(context_rows),
        "aggregate context coverage is invalid",
    )
    _require(len(arm_rows) == EXPECTED_TOTAL_ARM_ROWS, "aggregate arm matrix is invalid")
    return context_rows, arm_rows


def _aggregate_artifact_sha256(output_root: Path) -> dict[str, str]:
    return {
        name: _sha256_file(output_root / name)
        for name in (AGGREGATE_CONTEXT_NAME, AGGREGATE_ARM_NAME, DIAGNOSTIC_NAME)
    }


def _generation_mode(runs: Sequence[ValidatedRun]) -> str:
    sources = {run.generation_source for run in runs}
    _require(
        bool(sources)
        and sources
        <= {"fresh_execution", "rerun_after_artifact_gate_failure"},
        "seed generation provenance is incomplete",
    )
    return (
        "resume"
        if "rerun_after_artifact_gate_failure" in sources
        else "fresh"
    )


def _artifact_hashes(
    output_root: Path,
    runs: Sequence[ValidatedRun],
    aggregate_sha256: Mapping[str, str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for run in runs:
        for path, digest in run.artifact_sha256:
            _require(path not in hashes, "duplicate seed artifact hash path")
            hashes[path] = digest
        _require(
            run.receipt_path is not None and _is_sha256(run.receipt_sha256),
            "validated run is missing its frozen execution receipt hash",
        )
        receipt_key = str(run.receipt_path.relative_to(output_root)).replace("\\", "/")
        _require(receipt_key not in hashes, "duplicate execution receipt hash path")
        hashes[receipt_key] = run.receipt_sha256
    for path, digest in aggregate_sha256.items():
        _require(path not in hashes and _is_sha256(digest), "invalid aggregate hash")
        hashes[path] = digest
    return dict(sorted(hashes.items()))


def build_manifest(
    runs: Sequence[ValidatedRun],
    diagnostic: Mapping[str, object],
    *,
    config_path: Path,
    output_root: Path,
    execution_mode: str,
    worker_count: int,
    execution_identity: Mapping[str, object],
    aggregate_sha256: Mapping[str, str],
) -> dict[str, object]:
    per_seed = {
        str(run.spec.seed): {
            "context_count": len(run.artifacts.context_rows),
            "arm_row_count": len(run.artifacts.arm_rows),
            "configured_max_fes": run.artifacts.run_summary["configured_max_fes"],
            "terminal_fe": run.artifacts.run_summary["fitness_evaluations"],
            "comparison_fe": run.artifacts.run_summary["comparison_fe"],
            "terminal_shortfall_fes": (
                CONFIGURED_MAX_FES
                - int(run.artifacts.run_summary["fitness_evaluations"])
            ),
            "generation_source": run.generation_source,
            "generation_resume_gate_error": run.generation_resume_gate_error,
            "execution_receipt_sha256": run.receipt_sha256,
        }
        for run in sorted(runs, key=lambda item: item.spec.seed)
    }
    source_hashes = execution_identity.get("source_sha256")
    aob_input_hashes = execution_identity.get("aob_input_sha256")
    _require(
        isinstance(source_hashes, dict)
        and isinstance(aob_input_hashes, dict)
        and bool(aob_input_hashes)
        and all(_is_sha256(value) for value in aob_input_hashes.values())
        and execution_identity.get("config_path") == str(config_path.resolve())
        and execution_identity.get("config_sha256") == _sha256_file(config_path),
        "root manifest execution identity is invalid",
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_action_protocol": SOURCE_ACTION_PROTOCOL,
        "experiment_id": EXPERIMENT_ID,
        "stage": "action_validation_diagnostic",
        "status": "action_diagnostic_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "executor": "Codex",
        "git_head": _git_head(),
        "git_head_role": "assembly_annotation_source_sha256_is_authoritative",
        "execution_mode": execution_mode,
        "worker_count": worker_count,
        "config_path": execution_identity["config_path"],
        "config_sha256": execution_identity["config_sha256"],
        "cohort": COHORT,
        "case": CASE,
        "seeds": list(SEEDS),
        "configured_max_fes_per_seed": CONFIGURED_MAX_FES,
        "terminal_fe_policy": TERMINAL_FE_POLICY,
        "trajectory_count": len(runs),
        "cluster_count": len(SEEDS),
        "context_count": EXPECTED_TOTAL_CONTEXTS,
        "arm_row_count": EXPECTED_TOTAL_ARM_ROWS,
        "per_seed": per_seed,
        "arms": list(S_FAMILY_BUDGET_PULSE_ARMS),
        "horizons": list(ACTION_CEILING_HORIZONS),
        "primary_horizon": PRIMARY_HORIZON,
        "artifact_sha256": _artifact_hashes(
            output_root,
            runs,
            aggregate_sha256,
        ),
        "source_sha256": source_hashes,
        "aob_input_sha256": aob_input_hashes,
        "integrity_checks": {
            "five_seed_clusters": 1,
            "four_contexts_per_seed": 1,
            "minimum_context_gate": 1,
            "complete_four_arm_three_horizon_matrix": 1,
            "typed_action_reconstruction": 1,
            "native_three_sweep_parity": 1,
            "native_population_aligned_terminal_fe": 1,
            "case_seed_cluster_bootstrap": 1,
            "overlay_selected_relations_bound": 1,
            "overlay_observer_integrity": 1,
            "aob_inputs_unchanged": 1,
            "root_aob_input_identity": 1,
            "per_seed_execution_receipts": 1,
        },
        "integrity_gate_passed": 1,
        "action_ceiling_inference_authorized": 1,
        "action_gate_authorized": diagnostic["action_gate_authorized"],
        "s5_action_diagnostic_passed": diagnostic["s5_action_diagnostic_passed"],
        "real_aob_pilot_authorized": diagnostic["real_aob_pilot_authorized"],
        "fixed_action_real_aob_pilot_authorized": diagnostic[
            "fixed_action_real_aob_pilot_authorized"
        ],
        "fixed_action_validation_target": diagnostic[
            "fixed_action_validation_target"
        ],
        "fixed_action_validation_authorized": diagnostic[
            "fixed_action_validation_authorized"
        ],
        "evidence_separability_authorized": diagnostic[
            "evidence_separability_authorized"
        ],
        "selector_authorized": 0,
        "runtime_authorized": 0,
        "primary_recommendation": diagnostic["primary_recommendation"],
    }


def _validate_existing_outputs(
    runs: Sequence[ValidatedRun],
    *,
    config_path: Path,
    output_root: Path,
    execution_identity: Mapping[str, object],
) -> dict[str, object]:
    context_rows, arm_rows = _aggregate_rows(runs)
    diagnostic = summarize_diagnostic(arm_rows)
    _require(
        _read_csv(output_root / AGGREGATE_CONTEXT_NAME, ACTION_CEILING_CONTEXT_FIELDS)
        == context_rows
        and _read_csv(output_root / AGGREGATE_ARM_NAME, ACTION_CEILING_ARM_RESULT_FIELDS)
        == arm_rows,
        "existing aggregate CSVs differ from validated worker artifacts",
    )
    _require(
        _read_json(output_root / DIAGNOSTIC_NAME) == diagnostic,
        "existing diagnostic differs from validated arm rows",
    )
    aggregate_sha256 = _aggregate_artifact_sha256(output_root)
    manifest = _read_json(output_root / MANIFEST_NAME)
    execution_mode = _generation_mode(runs)
    _require(
        manifest.get("execution_mode") == execution_mode
        and manifest.get("worker_count") == DEFAULT_JOBS,
        "existing execution provenance is invalid",
    )
    expected = build_manifest(
        runs,
        diagnostic,
        config_path=config_path,
        output_root=output_root,
        execution_mode=execution_mode,
        worker_count=DEFAULT_JOBS,
        execution_identity=execution_identity,
        aggregate_sha256=aggregate_sha256,
    )
    stable_fields = tuple(
        field for field in expected if field not in {"generated_at_utc", "git_head"}
    )
    _require(
        set(manifest) == set(expected)
        and all(manifest.get(field) == expected[field] for field in stable_fields)
        and isinstance(manifest.get("generated_at_utc"), str)
        and bool(manifest["generated_at_utc"])
        and _is_git_hash(manifest.get("git_head")),
        "existing manifest contract or source/artifact hash changed",
    )
    return manifest


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python_executable: str = sys.executable,
    jobs: int | None = None,
    resume: bool = False,
    reuse_existing: bool = False,
) -> dict[str, object]:
    _require(not (resume and reuse_existing), "resume and reuse_existing are exclusive")
    config = load_config(config_path)
    execution = config["execution"]
    assert isinstance(execution, dict)
    worker_count = int(execution["jobs"]) if jobs is None else jobs
    _require(
        worker_count == DEFAULT_JOBS,
        "exp031 is frozen to one worker",
    )
    output_root = output_root.resolve()
    specs = build_specs(output_root)
    execution_identity = _execution_identity(config_path)

    if reuse_existing:
        runs = [
            run_one(
                spec,
                python_executable=python_executable,
                config_path=config_path,
                execution_identity=execution_identity,
                resume=False,
                reuse_existing=True,
            )
            for spec in specs
        ]
        return _validate_existing_outputs(
            runs,
            config_path=config_path,
            output_root=output_root,
            execution_identity=execution_identity,
        )

    runs: list[ValidatedRun] = []
    for spec in specs:
        run = run_one(
            spec,
            python_executable=python_executable,
            config_path=config_path,
            execution_identity=execution_identity,
            resume=resume,
            reuse_existing=False,
        )
        runs.append(run)
        progress = {
            "protocol_version": PROTOCOL_VERSION,
            "completed": [
                {
                    "seed": item.spec.seed,
                    "execution_source": item.execution_source,
                    "resume_gate_error": item.resume_gate_error,
                    "terminal_fe": item.artifacts.run_summary["fitness_evaluations"],
                }
                for item in runs
            ],
        }
        _write_json(output_root / "run_progress.json", progress)
        print(
            f"[{len(runs)}/{len(specs)}] S5/seed{run.spec.seed} "
            f"{run.execution_source}",
            flush=True,
        )

    context_rows, arm_rows = _aggregate_rows(runs)
    _require(
        _execution_identity(config_path) == execution_identity,
        "source or config changed during the diagnostic run",
    )
    _write_csv(
        output_root / AGGREGATE_CONTEXT_NAME,
        ACTION_CEILING_CONTEXT_FIELDS,
        context_rows,
    )
    _write_csv(
        output_root / AGGREGATE_ARM_NAME,
        ACTION_CEILING_ARM_RESULT_FIELDS,
        arm_rows,
    )
    diagnostic = summarize_diagnostic(arm_rows)
    _write_json(output_root / DIAGNOSTIC_NAME, diagnostic)
    aggregate_sha256 = _aggregate_artifact_sha256(output_root)
    _require(
        _execution_identity(config_path) == execution_identity,
        "source or config changed while aggregating the diagnostic",
    )
    manifest = build_manifest(
        runs,
        diagnostic,
        config_path=config_path,
        output_root=output_root,
        execution_mode=_generation_mode(runs),
        worker_count=worker_count,
        execution_identity=execution_identity,
        aggregate_sha256=aggregate_sha256,
    )
    _require(
        _execution_identity(config_path) == execution_identity
        and _aggregate_artifact_sha256(output_root) == aggregate_sha256,
        "source, config, or aggregate changed before manifest commit",
    )
    _write_json(output_root / MANIFEST_NAME, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs is not None and args.jobs <= 0:
        parser.error("--jobs must be positive")
    manifest = run_experiment(
        config_path=args.config,
        output_root=args.output_root,
        python_executable=args.python_executable,
        jobs=args.jobs,
        resume=args.resume,
        reuse_existing=args.reuse_existing,
    )
    print(
        f"[{manifest['status']}] contexts={manifest['context_count']} "
        f"recommendation={manifest['primary_recommendation']}",
        flush=True,
    )
    print(f"Manifest: {args.output_root / MANIFEST_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
