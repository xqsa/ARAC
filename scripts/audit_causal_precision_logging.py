"""Audit randomized precision logging and train the causal-risk gate.

This module is offline-only.  It deliberately keeps case/seed identity in the
validation and bootstrap code while exporting a runtime bundle whose inputs
are the exact sixteen identity-free pre-action features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARAC_REPO_ROOT = Path(__file__).resolve().parents[1]
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"
if str(ARAC_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(ARAC_SRC_ROOT))

from arac.policy.causal_risk_scheduler import (  # noqa: E402
    BOOTSTRAP_TREE_COUNT,
    CAUSAL_RISK_MODEL_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
    PRE_ACTION_UTILITY_SCHEMA_VERSION,
    UTILITY_FEATURE_NAMES,
    CausalRiskModelBundle,
    PreActionUtilityState,
    compute_model_sha256,
)

try:  # The causal-audit extra is mandatory; there is no fallback estimator.
    import numpy as np
    from scipy.stats import beta as beta_distribution
    from sklearn.tree import DecisionTreeRegressor
except ImportError as exc:  # pragma: no cover - exercised only in a broken env.
    raise RuntimeError(
        "causal audit dependencies are missing; install the pinned causal-audit extra"
    ) from exc


PROTOCOL_VERSION = "precision-causal-logging-v2"
RANDOMIZATION_SALT = "arac-precision-causal-logged-arm-v1"
RANDOMIZATION_ALGORITHM = "sha256_first_u64_mod2"
PROPENSITY = 0.5
ERROR_FLOOR = 1e-300
RISK_LIMIT = 0.05
DEFAULT_TREE_COUNT = BOOTSTRAP_TREE_COUNT
DEFAULT_POLICY_BOOTSTRAPS = 2000
DEFAULT_RANDOM_SEED = 20260715
PILOT_CASES = ("A4", "A5", "E1", "E2", "E3", "E4", "S2", "S5")
PILOT_SEEDS = tuple(range(40, 45))
FULL_CASES = tuple(
    f"{prefix}{index}" for prefix in ("E", "S", "R", "A") for index in range(1, 7)
)
FULL_SEEDS = tuple(range(40, 52))
CAUSAL_ARMS = ("baseline", "action")
STRICT_MAX_FES = 3_000_000
PRECISION_LANE_PROFILE = "precision_causal_logging"
PREREGISTRATION_PATH = (
    "docs/superpowers/specs/2026-07-15-causal-risk-precision-scheduler-design.md"
)
PREREGISTRATION_SHA256 = "9be1c021776c87cbc4e9ecfac1b91f97193f417ab1dd0a95f5f29afdd7b081a4"
PREREGISTRATION_COMMIT = "650d49126a27c48447ab4ab14e56d5e8ed847da2"

FEATURE_COLUMNS = ("decision_id", *UTILITY_FEATURE_NAMES)
RAW_FILENAMES = (
    "causal_decision_features.csv",
    "causal_decision_audit.csv",
    "causal_branch_manifest.csv",
    "causal_outcomes.csv",
    "randomized_log.csv",
    "causal_randomization_schedule.json",
)

AUDIT_REQUIRED_COLUMNS = (
    "protocol_version",
    "pair_id",
    "decision_id",
    "problem_id",
    "seed",
    "decision_status",
    "not_applicable_reason",
    "logged_arm",
    "propensity",
    "decision_fe",
    "checkpoint_fitness",
    "remaining_fe",
    "component_id",
    "component_group_count",
    "component_shared_var_count",
    "component_unlocked",
    "scheduler_revisit_reachable",
    "scheduler_revisit_cap_fe",
    "source_phase_i_end_fe",
    "source_cc_history_end_fe",
    "source_disagreement_history_end_fe",
    "source_cma_history_end_fe",
    "source_end_fe",
    "prefix_record_sha256",
    "checkpoint_candidate_sha256",
    "controller_state_sha256",
    "random_descriptor_sha256",
    "feature_sha256",
    "feature_schema_sha256",
    "decision_status_match",
    "decision_id_match",
    "feature_match",
    "prefix_match",
    "controller_state_match",
    "checkpoint_candidate_match",
    "random_descriptor_match",
    "requested_fe_match",
    "intervention_end_fe_match",
    "not_applicable_reason_match",
    "pair_integrity",
)

BRANCH_REQUIRED_COLUMNS = (
    "pair_id",
    "decision_id",
    "problem_id",
    "seed",
    "arm",
    "lane_id",
    "fresh_optimizer_execution",
    "status",
    "result_source",
    "output_root",
    "decision_status",
    "not_applicable_reason",
    "action_applied",
    "decision_fe",
    "intervention_end_fe",
    "checkpoint_fitness",
    "normal_sigma",
    "candidate_sigma",
    "applied_sigma",
    "requested_fe",
    "actual_fe",
    "configured_max_fes",
    "terminal_target_fe",
    "terminal_observed_fe",
    "terminal_status",
    "prefix_record_sha256",
    "checkpoint_candidate_sha256",
    "controller_state_sha256",
    "feature_sha256",
    "random_descriptor_sha256",
    "terminal_error",
    "terminal_record_sha256",
    "optimizer_fe_used",
    "same_budget_violation",
)

OUTCOME_REQUIRED_COLUMNS = (
    "pair_id",
    "decision_id",
    "problem_id",
    "seed",
    "decision_status",
    "checkpoint_error",
    "baseline_terminal_error",
    "action_terminal_error",
    "baseline_log_progress",
    "action_log_progress",
    "paired_tau",
    "catastrophic",
    "equal_checkpoint",
    "equal_terminal_target_fe",
    "equal_terminal_observed_fe",
    "outcome_valid",
)

RANDOMIZED_REQUIRED_COLUMNS = (
    "pair_id",
    "decision_id",
    "problem_id",
    "seed",
    "logged_arm",
    "propensity",
    "observed_treatment",
    "observed_terminal_error",
    "observed_log_progress",
    "terminal_target_fe",
    "terminal_observed_fe",
    "outcome_valid",
)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return tuple(reader.fieldnames), list(reader)


def _require_columns(
    header: Sequence[str], required: Sequence[str], *, artifact: str
) -> None:
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{artifact} missing required columns: {', '.join(missing)}")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _float(value: object, *, field: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float field {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite float field {field}: {value!r}")
    return number


def _int(value: object, *, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field {field}: {value!r}") from exc


def _flag(value: object, *, field: str) -> bool:
    if str(value) not in {"0", "1"}:
        raise ValueError(f"invalid binary field {field}: {value!r}")
    return str(value) == "1"


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def expected_pair_id(problem_id: str, seed: int) -> str:
    material = f"{PROTOCOL_VERSION}|{problem_id.upper()}|{int(seed)}".encode("utf-8")
    return "pair_" + hashlib.sha256(material).hexdigest()[:24]


def expected_logged_arm(
    problem_id: str,
    seed: int,
    *,
    salt: str = RANDOMIZATION_SALT,
) -> str:
    material = f"{salt}|{problem_id.upper()}|{int(seed)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2
    return "baseline" if value == 0 else "action"


def expected_decision_id(
    *,
    prefix_record_sha256: str,
    feature_sha256: str,
    not_applicable_reason: str,
) -> str:
    material = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "prefix_record_sha256": prefix_record_sha256,
            "feature_sha256": feature_sha256,
            "not_applicable_reason": not_applicable_reason,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "precision_" + hashlib.sha256(material).hexdigest()[:24]


def _log_progress(checkpoint_error: float, terminal_error: float) -> float:
    return math.log(max(checkpoint_error, ERROR_FLOOR)) - math.log(
        max(terminal_error, ERROR_FLOOR)
    )


def _paired_tau(baseline_error: float, action_error: float) -> float:
    return math.log(max(baseline_error, ERROR_FLOOR)) - math.log(
        max(action_error, ERROR_FLOOR)
    )


def _material_one_percent(tau: float) -> bool:
    # tau=log(baseline/action), so action/baseline=exp(-tau).
    return abs(math.expm1(-tau)) >= 0.01


@dataclass(frozen=True)
class DecisionPair:
    pair_id: str
    decision_id: str
    problem_id: str
    seed: int
    features: tuple[float, ...]
    feature_sha256: str
    logged_arm: str
    observed_treatment: int
    observed_y: float
    checkpoint_error: float
    baseline_error: float
    action_error: float
    y0: float
    y1: float
    tau: float
    catastrophic: int


@dataclass(frozen=True)
class ValidationObservation:
    pair_id: str
    decision_id: str
    problem_id: str
    seed: int
    features: tuple[float, ...]
    feature_sha256: str
    logged_arm: str
    observed_treatment: int
    observed_y: float


@dataclass(frozen=True)
class RobustSupport:
    median: tuple[float, ...]
    iqr: tuple[float, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    reference_scaled: tuple[tuple[float, ...], ...]
    knn_distance_threshold: float

    def scale(self, features: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            (value - median) / (iqr if iqr > 0.0 else 1.0)
            for value, median, iqr in zip(features, self.median, self.iqr, strict=True)
        )

    def evaluate(self, features: Sequence[float]) -> tuple[bool, tuple[str, ...]]:
        scaled = self.scale(features)
        reasons = [
            f"feature_out_of_range:{name}"
            for name, value, minimum, maximum in zip(
                UTILITY_FEATURE_NAMES,
                features,
                self.minimum,
                self.maximum,
                strict=True,
            )
            if value < minimum or value > maximum
        ]
        distances = sorted(math.dist(scaled, reference) for reference in self.reference_scaled)
        if len(distances) < 5 or distances[4] > self.knn_distance_threshold:
            reasons.append("knn_distance_exceeded")
        return not reasons, tuple(reasons)


@dataclass
class FoldModel:
    support: RobustSupport
    utility_trees: list[dict[str, object]]
    risk_trees: list[dict[str, object]]
    cp_tree: dict[str, object]
    conformal_margin: float
    nuisance_baseline: list[Any]
    nuisance_action: list[Any]


def _canonical_feature_state(values: Sequence[float]) -> PreActionUtilityState:
    return PreActionUtilityState.from_runtime_payload(
        {
            "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
            **dict(zip(UTILITY_FEATURE_NAMES, values, strict=True)),
        }
    )


def _one_by(
    rows: Iterable[dict[str, str]], field: str, *, artifact: str
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get(field, "")
        if not key:
            raise ValueError(f"{artifact} contains an empty {field}")
        if key in output:
            raise ValueError(f"{artifact} contains duplicate {field}: {key}")
        output[key] = row
    return output


def _group_by(
    rows: Iterable[dict[str, str]], field: str, *, artifact: str
) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row.get(field, "")
        if not key:
            raise ValueError(f"{artifact} contains an empty {field}")
        output[key].append(row)
    return output


def _manifest_value(payload: Mapping[str, object], *names: str) -> object | None:
    for name in names:
        if name in payload:
            return payload[name]
    feature_schema = payload.get("feature_schema")
    if isinstance(feature_schema, Mapping):
        for name in names:
            if name in feature_schema:
                return feature_schema[name]
    return None


def validate_manifests(input_root: Path) -> tuple[Mapping[str, object], list[str]]:
    blockers: list[str] = []
    feature_manifest = _read_json(input_root / "feature_manifest.json")
    logging_manifest = _read_json(input_root / "causal_logging_manifest.json")

    if _manifest_value(feature_manifest, "schema_version") != PRE_ACTION_UTILITY_SCHEMA_VERSION:
        blockers.append("feature_manifest_schema_version_mismatch")
    names = _manifest_value(feature_manifest, "feature_names", "features")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or tuple(names) != UTILITY_FEATURE_NAMES:
        blockers.append("feature_manifest_allowlist_mismatch")
    if _manifest_value(feature_manifest, "feature_schema_sha256", "sha256") != FEATURE_SCHEMA_SHA256:
        blockers.append("feature_manifest_sha256_mismatch")
    definitions = feature_manifest.get("features")
    if (
        not isinstance(definitions, Sequence)
        or isinstance(definitions, (str, bytes))
        or tuple(
            str(item.get("name", ""))
            for item in definitions
            if isinstance(item, Mapping)
        )
        != UTILITY_FEATURE_NAMES
        or any(
            not isinstance(item, Mapping)
            or not item.get("formula")
            or item.get("source_timing") != "strictly_pre_action"
            for item in definitions
        )
    ):
        blockers.append("feature_manifest_definitions_invalid")
    if feature_manifest.get("identity_fields_location") != "causal_decision_audit.csv_only":
        blockers.append("feature_manifest_identity_boundary_invalid")
    if feature_manifest.get("immutable_snapshot") is not True:
        blockers.append("feature_manifest_snapshot_not_immutable")

    if logging_manifest.get("protocol_version") != PROTOCOL_VERSION:
        blockers.append("logging_manifest_protocol_mismatch")
    if logging_manifest.get("offline_only") is not True:
        blockers.append("logging_manifest_not_offline_only")
    if logging_manifest.get("runtime_scheduler_authorized") is not False:
        blockers.append("raw_logging_manifest_claims_runtime_authorization")
    preregistration = logging_manifest.get("preregistration")
    if preregistration != {
        "path": PREREGISTRATION_PATH,
        "sha256": PREREGISTRATION_SHA256,
        "commit": PREREGISTRATION_COMMIT,
    }:
        blockers.append("logging_manifest_preregistration_mismatch")
    schema = logging_manifest.get("feature_schema")
    schema_names = schema.get("feature_names") if isinstance(schema, Mapping) else None
    if (
        not isinstance(schema, Mapping)
        or schema.get("schema_version") != PRE_ACTION_UTILITY_SCHEMA_VERSION
        or schema.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
        or not isinstance(schema_names, Sequence)
        or isinstance(schema_names, (str, bytes))
        or tuple(schema_names) != UTILITY_FEATURE_NAMES
    ):
        blockers.append("logging_manifest_feature_schema_mismatch")
    randomization = logging_manifest.get("randomization")
    if not isinstance(randomization, Mapping):
        randomization = {}
        blockers.append("logging_manifest_randomization_missing")
    if randomization.get("randomization_salt") != RANDOMIZATION_SALT:
        blockers.append("logging_manifest_randomization_salt_mismatch")
    if randomization.get("randomization_algorithm") != RANDOMIZATION_ALGORITHM:
        blockers.append("logging_manifest_randomization_algorithm_mismatch")
    try:
        propensity = _float(randomization.get("propensity"), field="manifest propensity")
    except ValueError:
        propensity = float("nan")
    if propensity != PROPENSITY:
        blockers.append("logging_manifest_propensity_mismatch")
    if randomization.get("coin_material") != "{salt}|{problem_id.upper()}|{int(seed)}":
        blockers.append("logging_manifest_coin_material_mismatch")
    if randomization.get("arm_mapping") != {"0": "baseline", "1": "action"}:
        blockers.append("logging_manifest_arm_mapping_mismatch")
    integrity = logging_manifest.get("integrity")
    if not isinstance(integrity, Mapping) or integrity.get("status") != "pass" or integrity.get("failures") not in ([], ()):
        blockers.append("logging_manifest_integrity_failed")

    hashes = logging_manifest.get("raw_artifact_sha256")
    if hashes is None:
        hashes = logging_manifest.get("artifact_sha256")
    if hashes is None:
        blockers.append("logging_manifest_artifact_hashes_missing")
    elif not isinstance(hashes, Mapping):
        blockers.append("logging_manifest_artifact_hashes_invalid")
    else:
        expected_hashed_files = (*RAW_FILENAMES, "feature_manifest.json")
        for filename in expected_hashed_files:
            expected = hashes.get(filename)
            if expected is None:
                blockers.append(f"logging_manifest_hash_missing:{filename}")
            elif expected != _file_sha256(input_root / filename):
                blockers.append(f"logging_manifest_hash_mismatch:{filename}")
    return logging_manifest, blockers


def load_decision_pairs(
    input_root: Path,
) -> tuple[list[DecisionPair], dict[str, object], list[str]]:
    """Load raw facts, recompute every pair fact, and return valid train rows."""

    logging_manifest, blockers = validate_manifests(input_root)
    feature_header, feature_rows = _read_csv(input_root / "causal_decision_features.csv")
    if feature_header != FEATURE_COLUMNS:
        raise ValueError(
            "causal_decision_features.csv must be exactly decision_id plus the 16-feature allowlist"
        )
    audit_header, audit_rows = _read_csv(input_root / "causal_decision_audit.csv")
    branch_header, branch_rows = _read_csv(input_root / "causal_branch_manifest.csv")
    outcome_header, outcome_rows = _read_csv(input_root / "causal_outcomes.csv")
    random_header, random_rows = _read_csv(input_root / "randomized_log.csv")
    _require_columns(audit_header, AUDIT_REQUIRED_COLUMNS, artifact="causal_decision_audit.csv")
    _require_columns(branch_header, BRANCH_REQUIRED_COLUMNS, artifact="causal_branch_manifest.csv")
    _require_columns(outcome_header, OUTCOME_REQUIRED_COLUMNS, artifact="causal_outcomes.csv")
    _require_columns(random_header, RANDOMIZED_REQUIRED_COLUMNS, artifact="randomized_log.csv")

    features_by_decision = _one_by(
        feature_rows, "decision_id", artifact="causal_decision_features.csv"
    )
    audit_by_pair = _one_by(audit_rows, "pair_id", artifact="causal_decision_audit.csv")
    branches_by_pair = _group_by(
        branch_rows, "pair_id", artifact="causal_branch_manifest.csv"
    )
    outcomes_by_pair = _one_by(outcome_rows, "pair_id", artifact="causal_outcomes.csv")
    randomized_by_pair = _one_by(random_rows, "pair_id", artifact="randomized_log.csv")
    schedule = _read_json(input_root / "causal_randomization_schedule.json")
    scheduled_rows = schedule.get("pairs")
    if not isinstance(scheduled_rows, Sequence) or isinstance(scheduled_rows, (str, bytes)):
        scheduled_rows = []
        blockers.append("randomization_schedule_pairs_invalid")
    try:
        scheduled_by_pair = _one_by(
            [dict(row) for row in scheduled_rows if isinstance(row, Mapping)],
            "pair_id",
            artifact="causal_randomization_schedule.json",
        )
    except ValueError as exc:
        scheduled_by_pair = {}
        blockers.append(f"randomization_schedule_invalid:{exc}")
    if (
        schedule.get("protocol_version") != PROTOCOL_VERSION
        or schedule.get("status") != "scheduled_before_subprocess"
        or schedule.get("randomization_salt") != RANDOMIZATION_SALT
        or schedule.get("randomization_algorithm") != RANDOMIZATION_ALGORITHM
        or schedule.get("coin_material") != "{salt}|{problem_id.upper()}|{int(seed)}"
        or schedule.get("arm_mapping") != {"0": "baseline", "1": "action"}
        or schedule.get("preregistration")
        != {
            "path": PREREGISTRATION_PATH,
            "sha256": PREREGISTRATION_SHA256,
            "commit": PREREGISTRATION_COMMIT,
        }
    ):
        blockers.append("randomization_schedule_contract_mismatch")

    if set(outcomes_by_pair) != set(audit_by_pair):
        blockers.append("outcome_audit_pair_matrix_mismatch")
    if set(randomized_by_pair) != set(audit_by_pair):
        blockers.append("randomized_audit_pair_matrix_mismatch")
    if set(branches_by_pair) != set(audit_by_pair):
        blockers.append("branch_audit_pair_matrix_mismatch")
    if set(scheduled_by_pair) != set(audit_by_pair):
        blockers.append("schedule_audit_pair_matrix_mismatch")

    matrix = logging_manifest.get("matrix")
    if not isinstance(matrix, Mapping):
        blockers.append("logging_manifest_matrix_missing")
    else:
        problem_ids = matrix.get("problem_ids")
        seeds = matrix.get("seeds")
        arms = matrix.get("arms")
        if (
            not isinstance(problem_ids, Sequence)
            or isinstance(problem_ids, (str, bytes))
            or not isinstance(seeds, Sequence)
            or isinstance(seeds, (str, bytes))
            or not isinstance(arms, Sequence)
            or isinstance(arms, (str, bytes))
        ):
            blockers.append("logging_manifest_matrix_invalid")
        else:
            try:
                expected_matrix = {
                    expected_pair_id(str(problem), int(seed))
                    for problem in problem_ids
                    for seed in seeds
                }
            except (TypeError, ValueError):
                expected_matrix = set()
                blockers.append("logging_manifest_matrix_invalid")
            if expected_matrix != set(audit_by_pair):
                blockers.append("logging_manifest_pair_matrix_mismatch")
            if tuple(arms) != ("baseline", "action"):
                blockers.append("logging_manifest_arm_matrix_mismatch")
    manifest_integrity = logging_manifest.get("integrity")
    if isinstance(manifest_integrity, Mapping):
        if manifest_integrity.get("total_pairs") != len(audit_by_pair):
            blockers.append("logging_manifest_total_pair_count_mismatch")

    pairs: list[DecisionPair] = []
    applicable_count = 0
    missing_feature_values = 0
    total_feature_values = 0
    for pair_id, audit in sorted(audit_by_pair.items()):
        problem_id = audit["problem_id"].upper()
        seed = _int(audit["seed"], field="seed")
        expected_pair = expected_pair_id(problem_id, seed)
        if pair_id != expected_pair:
            blockers.append(f"pair_id_assignment_drift:{pair_id}")
        if audit["protocol_version"] != PROTOCOL_VERSION:
            blockers.append(f"protocol_version_mismatch:{pair_id}")
        expected_arm = expected_logged_arm(problem_id, seed)
        if audit["logged_arm"] != expected_arm:
            blockers.append(f"logged_arm_assignment_drift:{pair_id}")
        scheduled = scheduled_by_pair.get(pair_id)
        if scheduled is None or (
            str(scheduled.get("problem_id", "")).upper() != problem_id
            or _int(scheduled.get("seed"), field="scheduled seed") != seed
            or scheduled.get("logged_arm") != expected_arm
            or _float(scheduled.get("propensity"), field="scheduled propensity")
            != PROPENSITY
        ):
            blockers.append(f"randomization_schedule_pair_mismatch:{pair_id}")
        try:
            propensity = _float(audit["propensity"], field="propensity")
        except ValueError:
            propensity = float("nan")
        if propensity != PROPENSITY:
            blockers.append(f"propensity_mismatch:{pair_id}")

        decision_status = audit["decision_status"]
        if decision_status not in {"applicable", "not_applicable"}:
            blockers.append(f"invalid_decision_status:{pair_id}")
            continue
        if decision_status == "not_applicable":
            if not audit["not_applicable_reason"]:
                blockers.append(f"missing_not_applicable_reason:{pair_id}")
            continue
        applicable_count += 1
        decision_id = audit["decision_id"]
        feature_row = features_by_decision.get(decision_id)
        if feature_row is None:
            blockers.append(f"applicable_feature_row_missing:{pair_id}")
            continue
        values: list[float] = []
        feature_complete = True
        for name in UTILITY_FEATURE_NAMES:
            total_feature_values += 1
            raw = feature_row.get(name, "")
            try:
                values.append(_float(raw, field=name))
            except ValueError:
                missing_feature_values += 1
                feature_complete = False
        if not feature_complete:
            blockers.append(f"critical_feature_missing_or_nonfinite:{pair_id}")
            continue
        state = _canonical_feature_state(values)
        if audit["feature_sha256"] != state.feature_sha256:
            blockers.append(f"feature_sha256_mismatch:{pair_id}")
        if audit["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256:
            blockers.append(f"feature_schema_sha256_mismatch:{pair_id}")
        if decision_id != expected_decision_id(
            prefix_record_sha256=audit["prefix_record_sha256"],
            feature_sha256=state.feature_sha256,
            not_applicable_reason=audit["not_applicable_reason"],
        ):
            blockers.append(f"decision_id_recompute_mismatch:{pair_id}")
        if any(
            audit[field] != "1"
            for field in (
                "decision_status_match",
                "decision_id_match",
                "feature_match",
                "prefix_match",
                "controller_state_match",
                "checkpoint_candidate_match",
                "random_descriptor_match",
                "requested_fe_match",
                "not_applicable_reason_match",
                "pair_integrity",
                "component_unlocked",
                "scheduler_revisit_reachable",
            )
        ):
            blockers.append(f"paired_pre_action_integrity_failed:{pair_id}")
        decision_fe = _int(audit["decision_fe"], field="decision_fe")
        watermarks = (
            "source_phase_i_end_fe",
            "source_cc_history_end_fe",
            "source_disagreement_history_end_fe",
            "source_cma_history_end_fe",
            "source_end_fe",
        )
        if any(_int(audit[field], field=field) >= decision_fe for field in watermarks):
            blockers.append(f"pre_action_watermark_violation:{pair_id}")
        if not _is_sha256(audit["prefix_record_sha256"]):
            blockers.append(f"invalid_prefix_sha256:{pair_id}")
        if not _is_sha256(audit["checkpoint_candidate_sha256"]):
            blockers.append(f"invalid_checkpoint_candidate_sha256:{pair_id}")
        if not _is_sha256(audit["controller_state_sha256"]):
            blockers.append(f"invalid_controller_state_sha256:{pair_id}")
        if not _is_sha256(audit["random_descriptor_sha256"]):
            blockers.append(f"invalid_random_descriptor_sha256:{pair_id}")

        branches = branches_by_pair.get(pair_id, [])
        by_arm = {row.get("arm", ""): row for row in branches}
        if len(branches) != 2 or set(by_arm) != {"baseline", "action"}:
            blockers.append(f"paired_branch_matrix_invalid:{pair_id}")
            continue
        baseline_branch = by_arm["baseline"]
        action_branch = by_arm["action"]
        for arm, branch in by_arm.items():
            if (
                branch["decision_id"] != decision_id
                or branch["problem_id"].upper() != problem_id
                or _int(branch["seed"], field="branch seed") != seed
                or branch["decision_status"] != decision_status
                or branch["not_applicable_reason"] != audit["not_applicable_reason"]
            ):
                blockers.append(f"branch_identity_mismatch:{pair_id}:{arm}")
            if branch["fresh_optimizer_execution"] != "1" or branch["status"] != "completed":
                blockers.append(f"branch_not_fresh_completed:{pair_id}:{arm}")
            if branch["terminal_status"] != "complete":
                blockers.append(f"branch_terminal_incomplete:{pair_id}:{arm}")
            if branch["same_budget_violation"] != "0":
                blockers.append(f"branch_same_budget_violation:{pair_id}:{arm}")
            if branch["feature_sha256"] != audit["feature_sha256"]:
                blockers.append(f"branch_feature_sha256_mismatch:{pair_id}:{arm}")
            if branch["prefix_record_sha256"] != audit["prefix_record_sha256"]:
                blockers.append(f"branch_prefix_sha256_mismatch:{pair_id}:{arm}")
            if (
                branch["checkpoint_candidate_sha256"]
                != audit["checkpoint_candidate_sha256"]
                or not _is_sha256(branch["checkpoint_candidate_sha256"])
            ):
                blockers.append(f"branch_checkpoint_candidate_sha256_mismatch:{pair_id}:{arm}")
            if branch["controller_state_sha256"] != audit["controller_state_sha256"]:
                blockers.append(f"branch_controller_sha256_mismatch:{pair_id}:{arm}")
            if (
                branch["random_descriptor_sha256"]
                != audit["random_descriptor_sha256"]
                or not _is_sha256(branch["random_descriptor_sha256"])
            ):
                blockers.append(f"branch_random_descriptor_sha256_mismatch:{pair_id}:{arm}")
            if not _is_sha256(branch["terminal_record_sha256"]):
                blockers.append(f"invalid_terminal_record_sha256:{pair_id}:{arm}")
        equal_fields = (
            "decision_fe",
            "checkpoint_fitness",
            "normal_sigma",
            "candidate_sigma",
            "requested_fe",
            "configured_max_fes",
            "terminal_target_fe",
            "terminal_observed_fe",
        )
        if any(baseline_branch[field] != action_branch[field] for field in equal_fields):
            blockers.append(f"paired_branch_budget_or_target_mismatch:{pair_id}")
        if audit["requested_fe_match"] != str(
            int(baseline_branch["requested_fe"] == action_branch["requested_fe"])
        ):
            blockers.append(f"requested_fe_match_flag_drift:{pair_id}")
        if audit["intervention_end_fe_match"] != str(
            int(
                baseline_branch["intervention_end_fe"]
                == action_branch["intervention_end_fe"]
            )
        ):
            blockers.append(f"intervention_end_fe_match_flag_drift:{pair_id}")
        if baseline_branch["action_applied"] != "0" or action_branch["action_applied"] != "1":
            blockers.append(f"paired_branch_action_contract_mismatch:{pair_id}")
        for arm, branch in by_arm.items():
            try:
                configured_max_fes = _int(
                    branch["configured_max_fes"], field="configured_max_fes"
                )
                intervention_end_fe = _int(
                    branch["intervention_end_fe"], field="intervention_end_fe"
                )
                requested_fe = _int(branch["requested_fe"], field="requested_fe")
                actual_fe = _int(branch["actual_fe"], field="actual_fe")
                terminal_target_fe = _int(
                    branch["terminal_target_fe"], field="terminal_target_fe"
                )
                terminal_observed_fe = _int(
                    branch["terminal_observed_fe"], field="terminal_observed_fe"
                )
                optimizer_fe_used = _int(
                    branch["optimizer_fe_used"], field="optimizer_fe_used"
                )
                remaining_fe = _int(audit["remaining_fe"], field="remaining_fe")
                revisit_cap_fe = _int(
                    audit["scheduler_revisit_cap_fe"], field="scheduler_revisit_cap_fe"
                )
                if (
                    intervention_end_fe - decision_fe != actual_fe
                    or actual_fe <= 0
                    or actual_fe > requested_fe
                    or terminal_observed_fe != terminal_target_fe
                    or terminal_target_fe <= intervention_end_fe
                    or optimizer_fe_used < terminal_target_fe
                    or optimizer_fe_used > configured_max_fes
                    or remaining_fe != configured_max_fes - decision_fe
                    or revisit_cap_fe > remaining_fe
                ):
                    blockers.append(f"branch_fe_contract_mismatch:{pair_id}:{arm}")
            except ValueError:
                blockers.append(f"branch_fe_contract_invalid:{pair_id}:{arm}")
        try:
            if not _close(
                _float(baseline_branch["applied_sigma"], field="baseline applied_sigma"),
                _float(baseline_branch["normal_sigma"], field="normal_sigma"),
            ):
                blockers.append(f"baseline_sigma_mismatch:{pair_id}")
            if not _close(
                _float(action_branch["applied_sigma"], field="action applied_sigma"),
                _float(action_branch["candidate_sigma"], field="candidate_sigma"),
            ):
                blockers.append(f"action_sigma_mismatch:{pair_id}")
        except ValueError:
            blockers.append(f"branch_sigma_invalid:{pair_id}")

        outcome = outcomes_by_pair.get(pair_id)
        randomized = randomized_by_pair.get(pair_id)
        if outcome is None or randomized is None:
            continue
        if (
            outcome["decision_id"] != decision_id
            or randomized["decision_id"] != decision_id
            or outcome["problem_id"].upper() != problem_id
            or randomized["problem_id"].upper() != problem_id
            or _int(outcome["seed"], field="outcome seed") != seed
            or _int(randomized["seed"], field="randomized seed") != seed
            or outcome["decision_status"] != decision_status
        ):
            blockers.append(f"outcome_or_randomized_identity_mismatch:{pair_id}")
        checkpoint = _float(outcome["checkpoint_error"], field="checkpoint_error")
        baseline_error = _float(outcome["baseline_terminal_error"], field="baseline_terminal_error")
        action_error = _float(outcome["action_terminal_error"], field="action_terminal_error")
        if checkpoint < 0.0 or baseline_error < 0.0 or action_error < 0.0:
            blockers.append(f"negative_error_value:{pair_id}")
            continue
        y0 = _log_progress(checkpoint, baseline_error)
        y1 = _log_progress(checkpoint, action_error)
        tau = _paired_tau(baseline_error, action_error)
        catastrophic = int(action_error >= 1.2 * baseline_error)
        branch_baseline_error = _float(
            baseline_branch["terminal_error"], field="baseline branch terminal_error"
        )
        branch_action_error = _float(
            action_branch["terminal_error"], field="action branch terminal_error"
        )
        if not _close(branch_baseline_error, baseline_error) or not _close(
            branch_action_error, action_error
        ):
            blockers.append(f"branch_outcome_terminal_error_mismatch:{pair_id}")
        if not _close(
            _float(baseline_branch["checkpoint_fitness"], field="branch checkpoint_fitness"),
            checkpoint,
        ) or not _close(
            _float(action_branch["checkpoint_fitness"], field="branch checkpoint_fitness"),
            checkpoint,
        ):
            blockers.append(f"branch_outcome_checkpoint_mismatch:{pair_id}")
        recomputed = (
            ("baseline_log_progress", y0),
            ("action_log_progress", y1),
            ("paired_tau", tau),
        )
        if any(not _close(_float(outcome[field], field=field), value) for field, value in recomputed):
            blockers.append(f"terminal_outcome_recompute_mismatch:{pair_id}")
        if outcome["catastrophic"] != str(catastrophic):
            blockers.append(f"catastrophic_label_mismatch:{pair_id}")
        if any(
            outcome[field] != "1"
            for field in (
                "equal_checkpoint",
                "equal_terminal_target_fe",
                "equal_terminal_observed_fe",
                "outcome_valid",
            )
        ):
            blockers.append(f"outcome_integrity_failed:{pair_id}")

        observed_treatment = int(expected_arm == "action")
        observed_error = action_error if observed_treatment else baseline_error
        observed_y = y1 if observed_treatment else y0
        observed_branch = action_branch if observed_treatment else baseline_branch
        if (
            randomized["logged_arm"] != expected_arm
            or _int(randomized["observed_treatment"], field="observed_treatment") != observed_treatment
            or _float(randomized["propensity"], field="randomized propensity") != PROPENSITY
            or not _close(_float(randomized["observed_terminal_error"], field="observed_terminal_error"), observed_error)
            or not _close(_float(randomized["observed_log_progress"], field="observed_log_progress"), observed_y)
            or randomized["terminal_target_fe"] != observed_branch["terminal_target_fe"]
            or randomized["terminal_observed_fe"] != observed_branch["terminal_observed_fe"]
            or randomized["outcome_valid"] != "1"
        ):
            blockers.append(f"randomized_observation_mismatch:{pair_id}")
        pairs.append(
            DecisionPair(
                pair_id=pair_id,
                decision_id=decision_id,
                problem_id=problem_id,
                seed=seed,
                features=tuple(values),
                feature_sha256=state.feature_sha256,
                logged_arm=expected_arm,
                observed_treatment=observed_treatment,
                observed_y=observed_y,
                checkpoint_error=checkpoint,
                baseline_error=baseline_error,
                action_error=action_error,
                y0=y0,
                y1=y1,
                tau=tau,
                catastrophic=catastrophic,
            )
        )

    unexpected_features = sorted(set(features_by_decision) - {pair.decision_id for pair in pairs})
    if unexpected_features:
        blockers.append("orphan_or_invalid_feature_rows")
    stats: dict[str, object] = {
        "logged_pair_count": len(audit_by_pair),
        "applicable_pair_count": applicable_count,
        "valid_pair_count": len(pairs),
        "critical_feature_value_count": total_feature_values,
        "critical_feature_missing_count": missing_feature_values,
        "critical_feature_missing_rate": (
            missing_feature_values / total_feature_values if total_feature_values else 1.0
        ),
        "source_logging_root": str(input_root.resolve()),
        "source_logging_manifest_sha256": _file_sha256(
            input_root / "causal_logging_manifest.json"
        ),
        "source_raw_artifact_sha256": (
            dict(logging_manifest["raw_artifact_sha256"])
            if isinstance(logging_manifest.get("raw_artifact_sha256"), Mapping)
            else {}
        ),
        "source_matrix": dict(matrix) if isinstance(matrix, Mapping) else {},
        "source_lane_profile": logging_manifest.get("lane_profile", ""),
        "source_git_commit": logging_manifest.get("git_commit", ""),
    }
    if isinstance(manifest_integrity, Mapping) and manifest_integrity.get(
        "applicable_pairs"
    ) != applicable_count:
        blockers.append("logging_manifest_applicable_pair_count_mismatch")
    return pairs, stats, sorted(set(blockers))


def _stage_matrix_matches(raw_stats: Mapping[str, object], stage: str) -> bool:
    matrix = raw_stats.get("source_matrix")
    if not isinstance(matrix, Mapping):
        return False
    expected_cases = PILOT_CASES if stage == "pilot" else FULL_CASES
    expected_seeds = PILOT_SEEDS if stage == "pilot" else FULL_SEEDS
    try:
        problems = {str(value).upper() for value in matrix["problem_ids"]}  # type: ignore[index]
        seeds = {int(value) for value in matrix["seeds"]}  # type: ignore[index]
        arms = tuple(str(value) for value in matrix["arms"])  # type: ignore[index]
        max_fes = int(matrix["max_fes"])
        jobs = int(matrix["jobs"])
        budget_accounting = str(matrix["budget_accounting"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        problems == set(expected_cases)
        and len(matrix["problem_ids"]) == len(expected_cases)  # type: ignore[index]
        and seeds == set(expected_seeds)
        and len(matrix["seeds"]) == len(expected_seeds)  # type: ignore[index]
        and arms == CAUSAL_ARMS
        and max_fes == STRICT_MAX_FES
        and jobs == 24
        and budget_accounting == "strict"
        and raw_stats.get("source_lane_profile") == PRECISION_LANE_PROFILE
    )


def fit_robust_support(features: Sequence[Sequence[float]]) -> RobustSupport:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(UTILITY_FEATURE_NAMES):
        raise ValueError("support fitting requires an n-by-16 feature matrix")
    if matrix.shape[0] < 6 or not np.all(np.isfinite(matrix)):
        raise ValueError("support fitting requires at least six finite rows")
    median = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0, method="linear")
    q75 = np.quantile(matrix, 0.75, axis=0, method="linear")
    iqr = q75 - q25
    scale = np.where(iqr > 0.0, iqr, 1.0)
    scaled = (matrix - median) / scale
    loo_fifth_distances: list[float] = []
    for index in range(len(scaled)):
        distances = sorted(
            math.dist(scaled[index], scaled[other])
            for other in range(len(scaled))
            if other != index
        )
        loo_fifth_distances.append(distances[4])
    threshold = float(np.quantile(loo_fifth_distances, 0.95, method="linear"))
    return RobustSupport(
        median=tuple(float(value) for value in median),
        iqr=tuple(float(value) for value in iqr),
        minimum=tuple(float(value) for value in np.min(matrix, axis=0)),
        maximum=tuple(float(value) for value in np.max(matrix, axis=0)),
        reference_scaled=tuple(
            tuple(float(value) for value in row) for row in scaled
        ),
        knn_distance_threshold=threshold,
    )


def _cluster_bootstrap_indices(
    labels: Sequence[str], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    clusters = np.asarray(sorted(set(labels)), dtype=object)
    if not len(clusters):
        raise ValueError("case-cluster bootstrap requires at least one case")
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    indices: list[int] = []
    for cluster in sampled:
        indices.extend(index for index, label in enumerate(labels) if label == cluster)
    sampled_set = set(str(value) for value in sampled)
    oob = np.asarray(
        [index for index, label in enumerate(labels) if label not in sampled_set],
        dtype=int,
    )
    return np.asarray(indices, dtype=int), oob


def _fit_regression_tree(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    min_samples_leaf: int,
    random_state: int,
) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(
        max_depth=2,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(features, targets)
    return model


def _serialize_tree(
    model: DecisionTreeRegressor,
    *,
    leaf_values: Mapping[int, float] | None = None,
    node: int = 0,
) -> dict[str, object]:
    tree = model.tree_
    left = int(tree.children_left[node])
    right = int(tree.children_right[node])
    if left == right:
        value = (
            float(leaf_values[node])
            if leaf_values is not None
            else float(np.ravel(tree.value[node])[0])
        )
        return {"value": value}
    feature_index = int(tree.feature[node])
    return {
        "feature": UTILITY_FEATURE_NAMES[feature_index],
        "threshold": float(tree.threshold[node]),
        "left": _serialize_tree(model, leaf_values=leaf_values, node=left),
        "right": _serialize_tree(model, leaf_values=leaf_values, node=right),
    }


def _predict_tree(tree: Mapping[str, object], features: Sequence[float]) -> float:
    node = tree
    while "value" not in node:
        name = str(node["feature"])
        child = "left" if features[UTILITY_FEATURE_NAMES.index(name)] <= float(node["threshold"]) else "right"
        next_node = node[child]
        if not isinstance(next_node, Mapping):
            raise ValueError("serialized tree contains an invalid child")
        node = next_node
    return float(node["value"])


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="linear"))


def clopper_pearson_upper(events: int, trials: int, *, alpha: float = 0.05) -> float:
    if trials < 0 or events < 0 or events > trials:
        raise ValueError("invalid exact-binomial counts")
    if trials == 0 or events == trials:
        return 1.0
    return float(beta_distribution.ppf(1.0 - alpha, events + 1, trials - events))


def _risk_leaf_values(
    model: DecisionTreeRegressor,
    features: np.ndarray,
    outcomes: np.ndarray,
    *,
    exact: bool,
) -> dict[int, float]:
    leaves = model.apply(features)
    output: dict[int, float] = {}
    for leaf in sorted(set(int(value) for value in leaves)):
        selected = outcomes[leaves == leaf]
        events = int(np.sum(selected))
        trials = int(len(selected))
        output[leaf] = (
            clopper_pearson_upper(events, trials)
            if exact
            else (events + 0.5) / (trials + 1.0)
        )
    return output


def _conformal_margin(
    oob_predictions: Sequence[Sequence[float]],
    targets: Sequence[float],
) -> float:
    scores: list[float] = []
    for predictions, target in zip(oob_predictions, targets, strict=True):
        if not predictions:
            raise ValueError("case bootstrap produced no OOB prediction for a training row")
        lower = _quantile(predictions, 0.05)
        scores.append(max(0.0, lower - float(target)))
    ordered = sorted(scores)
    index = min(len(ordered) - 1, math.ceil((len(ordered) + 1) * 0.95) - 1)
    return float(ordered[index])


def _random_state(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _fit_fold_model(
    pairs: Sequence[DecisionPair],
    *,
    tree_count: int,
    random_seed: int,
    fit_nuisance: bool = True,
) -> FoldModel:
    if tree_count <= 0:
        raise ValueError("tree_count must be positive")
    if len({pair.problem_id for pair in pairs}) < 2:
        raise ValueError("case-cluster fitting requires at least two training cases")
    support = fit_robust_support([pair.features for pair in pairs])
    matrix = np.asarray([support.scale(pair.features) for pair in pairs], dtype=float)
    tau = np.asarray([pair.tau for pair in pairs], dtype=float)
    risk = np.asarray([pair.catastrophic for pair in pairs], dtype=float)
    cases = [pair.problem_id for pair in pairs]
    min_leaf = max(8, math.ceil(0.1 * len(pairs)))
    rng = np.random.default_rng(random_seed)
    utility_trees: list[dict[str, object]] = []
    risk_trees: list[dict[str, object]] = []
    nuisance_baseline: list[Any] = []
    nuisance_action: list[Any] = []
    oob_predictions: list[list[float]] = [[] for _ in pairs]

    baseline_indices = [index for index, pair in enumerate(pairs) if pair.observed_treatment == 0]
    action_indices = [index for index, pair in enumerate(pairs) if pair.observed_treatment == 1]
    if fit_nuisance and (not baseline_indices or not action_indices):
        raise ValueError("both randomized treatment arms are required in every training fold")

    for _ in range(tree_count):
        sampled, oob = _cluster_bootstrap_indices(cases, rng)
        utility_model = _fit_regression_tree(
            matrix[sampled],
            tau[sampled],
            min_samples_leaf=min_leaf,
            random_state=_random_state(rng),
        )
        utility_trees.append(_serialize_tree(utility_model))
        if len(oob):
            predictions = utility_model.predict(matrix[oob])
            for index, prediction in zip(oob, predictions, strict=True):
                oob_predictions[int(index)].append(float(prediction))

        risk_model = _fit_regression_tree(
            matrix[sampled],
            risk[sampled],
            min_samples_leaf=min_leaf,
            random_state=_random_state(rng),
        )
        smoothed = _risk_leaf_values(
            risk_model,
            matrix[sampled],
            risk[sampled],
            exact=False,
        )
        risk_trees.append(_serialize_tree(risk_model, leaf_values=smoothed))

        if fit_nuisance:
            for arm, indices, destination in (
                (0, baseline_indices, nuisance_baseline),
                (1, action_indices, nuisance_action),
            ):
                labels = [pairs[index].problem_id for index in indices]
                local_sampled, _ = _cluster_bootstrap_indices(labels, rng)
                sampled_indices = np.asarray([indices[int(index)] for index in local_sampled], dtype=int)
                targets = np.asarray(
                    [pairs[index].observed_y for index in sampled_indices], dtype=float
                )
                destination.append(
                    _fit_regression_tree(
                        matrix[sampled_indices],
                        targets,
                        min_samples_leaf=min_leaf,
                        random_state=_random_state(rng) + arm,
                    )
                )

    conformal_margin = _conformal_margin(oob_predictions, tau)
    cp_model = _fit_regression_tree(
        matrix,
        risk,
        min_samples_leaf=min_leaf,
        random_state=_random_state(rng),
    )
    cp_values = _risk_leaf_values(cp_model, matrix, risk, exact=True)
    return FoldModel(
        support=support,
        utility_trees=utility_trees,
        risk_trees=risk_trees,
        cp_tree=_serialize_tree(cp_model, leaf_values=cp_values),
        conformal_margin=conformal_margin,
        nuisance_baseline=nuisance_baseline,
        nuisance_action=nuisance_action,
    )


def _fold_seed(base_seed: int, scheme: str, fold: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{scheme}|{fold}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _estimate_observation(
    model: FoldModel,
    observation: ValidationObservation,
) -> dict[str, object]:
    scaled = model.support.scale(observation.features)
    in_support, reasons = model.support.evaluate(observation.features)
    utility = [_predict_tree(tree, scaled) for tree in model.utility_trees]
    risks = [_predict_tree(tree, scaled) for tree in model.risk_trees]
    tau_hat = math.fsum(utility) / len(utility)
    tau_lcb = _quantile(utility, 0.05) - model.conformal_margin
    risk_ucb = max(_quantile(risks, 0.95), _predict_tree(model.cp_tree, scaled))
    if not model.nuisance_baseline or not model.nuisance_action:
        raise ValueError("cross-fitting requires nuisance ensembles")
    row = np.asarray([scaled], dtype=float)
    mu0 = math.fsum(float(tree.predict(row)[0]) for tree in model.nuisance_baseline) / len(model.nuisance_baseline)
    mu1 = math.fsum(float(tree.predict(row)[0]) for tree in model.nuisance_action) / len(model.nuisance_action)
    treatment = observation.observed_treatment
    y = observation.observed_y
    phi0 = mu0 + (y - mu0) / PROPENSITY if treatment == 0 else mu0
    phi1 = mu1 + (y - mu1) / PROPENSITY if treatment == 1 else mu1
    effect_score = phi1 - phi0
    utility_candidate = bool(in_support and tau_lcb > 0.0)
    safe_release = bool(utility_candidate and risk_ucb <= RISK_LIMIT)
    return {
        "pair_id": observation.pair_id,
        "decision_id": observation.decision_id,
        "problem_id": observation.problem_id,
        "seed": observation.seed,
        "logged_arm": observation.logged_arm,
        "observed_treatment": treatment,
        "observed_log_progress": y,
        "tau_hat": tau_hat,
        "tau_lcb": tau_lcb,
        "conformal_margin": model.conformal_margin,
        "catastrophic_risk_ucb": risk_ucb,
        "in_support": int(in_support),
        "ood_reasons": ";".join(reasons),
        "mu_baseline": mu0,
        "mu_action": mu1,
        "dr_effect_score": effect_score,
        "utility_candidate_selected": int(utility_candidate),
        "safe_release_selected": int(safe_release),
        "dr_utility_candidate": effect_score if utility_candidate else 0.0,
        "dr_safe_release": effect_score if safe_release else 0.0,
    }


def build_crossfit_predictions(
    pairs: Sequence[DecisionPair],
    *,
    scheme: str,
    tree_count: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], list[str]]:
    """Predict held-out rows without passing their sealed counterfactuals to fit."""

    if scheme not in {"LCO", "LSO"}:
        raise ValueError("cross-fitting scheme must be LCO or LSO")
    fold_of = (
        (lambda pair: pair.problem_id)
        if scheme == "LCO"
        else (lambda pair: str(pair.seed))
    )
    folds = sorted({fold_of(pair) for pair in pairs})
    output: list[dict[str, object]] = []
    blockers: list[str] = []
    for fold in folds:
        train = [pair for pair in pairs if fold_of(pair) != fold]
        held_out = [pair for pair in pairs if fold_of(pair) == fold]
        observations = [
            ValidationObservation(
                pair_id=pair.pair_id,
                decision_id=pair.decision_id,
                problem_id=pair.problem_id,
                seed=pair.seed,
                features=pair.features,
                feature_sha256=pair.feature_sha256,
                logged_arm=pair.logged_arm,
                observed_treatment=pair.observed_treatment,
                observed_y=pair.observed_y,
            )
            for pair in held_out
        ]
        try:
            model = _fit_fold_model(
                train,
                tree_count=tree_count,
                random_seed=_fold_seed(random_seed, scheme, fold),
            )
        except ValueError as exc:
            blockers.append(f"{scheme}_fold_fit_failed:{fold}:{exc}")
            continue
        for observation in observations:
            prediction = _estimate_observation(model, observation)
            prediction.update(
                {
                    "validation_scheme": scheme,
                    "fold_id": fold,
                    "training_pair_count": len(train),
                    "training_case_count": len({pair.problem_id for pair in train}),
                    "training_seed_count": len({pair.seed for pair in train}),
                }
            )
            output.append(prediction)

    # Counterfactual outcomes are joined only after all held-out predictions froze.
    pair_map = {pair.pair_id: pair for pair in pairs}
    for row in output:
        pair = pair_map[str(row["pair_id"])]
        row.update(
            {
                "exact_tau": pair.tau,
                "catastrophic": pair.catastrophic,
                "material_1pct": int(_material_one_percent(pair.tau)),
                "predicted_positive": int(float(row["tau_hat"]) > 0.0),
                "sign_correct": int((float(row["tau_hat"]) > 0.0) == (pair.tau > 0.0)),
                "exact_utility_candidate": (
                    pair.tau if int(row["utility_candidate_selected"]) else 0.0
                ),
                "exact_safe_release": (
                    pair.tau if int(row["safe_release_selected"]) else 0.0
                ),
            }
        )
    return output, blockers


def _balanced_accuracy(
    rows: Sequence[Mapping[str, object]],
    weights: Sequence[float] | None = None,
) -> float | None:
    material = [
        (index, row)
        for index, row in enumerate(rows)
        if int(row["material_1pct"]) == 1
    ]
    if not material:
        return None
    class_rates: list[float] = []
    for actual_positive in (0, 1):
        numerator = 0.0
        denominator = 0.0
        for index, row in material:
            actual = int(float(row["exact_tau"]) > 0.0)
            if actual != actual_positive:
                continue
            weight = 1.0 if weights is None else float(weights[index])
            denominator += weight
            numerator += weight * int(int(row["predicted_positive"]) == actual)
        if denominator <= 0.0:
            return None
        class_rates.append(numerator / denominator)
    return math.fsum(class_rates) / 2.0


def _multiway_weights(
    rows: Sequence[Mapping[str, object]], rng: np.random.Generator
) -> list[float]:
    cases = sorted({str(row["problem_id"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    case_counts = Counter(str(value) for value in rng.choice(cases, len(cases), replace=True))
    seed_counts = Counter(int(value) for value in rng.choice(seeds, len(seeds), replace=True))
    return [
        float(case_counts[str(row["problem_id"])] * seed_counts[int(row["seed"])])
        for row in rows
    ]


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    denominator = math.fsum(weights)
    if denominator <= 0.0:
        return None
    return math.fsum(value * weight for value, weight in zip(values, weights, strict=True)) / denominator


def _multiway_bootstrap_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    value_field: str,
    replicates: int,
    random_seed: int,
) -> tuple[float, float]:
    if replicates <= 0:
        raise ValueError("policy bootstrap count must be positive")
    rng = np.random.default_rng(random_seed)
    values = [float(row[value_field]) for row in rows]
    value_samples: list[float] = []
    accuracy_samples: list[float] = []
    for _ in range(replicates):
        weights = _multiway_weights(rows, rng)
        value = _weighted_mean(values, weights)
        if value is not None:
            value_samples.append(value)
        accuracy = _balanced_accuracy(rows, weights)
        # Missing either sign class is evidence of unsupported sign accuracy,
        # not a bootstrap replicate that may be discarded optimistically.
        accuracy_samples.append(0.0 if accuracy is None else accuracy)
    if not value_samples:
        raise ValueError("multiway bootstrap produced no non-empty replicate")
    return (
        _quantile(value_samples, 0.05),
        _quantile(accuracy_samples, 0.05),
    )


POLICY_FIELDS = {
    "utility_candidate_policy": (
        "utility_candidate_selected",
        "dr_utility_candidate",
        "exact_utility_candidate",
    ),
    "safe_release_policy": (
        "safe_release_selected",
        "dr_safe_release",
        "exact_safe_release",
    ),
}


def build_policy_value_rows(
    predictions: Sequence[Mapping[str, object]],
    *,
    bootstrap_count: int,
    random_seed: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    by_scheme: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        by_scheme[str(row["validation_scheme"])].append(row)
    for scheme, scheme_rows in sorted(by_scheme.items()):
        for policy_kind, (selected_field, dr_field, exact_field) in POLICY_FIELDS.items():
            dr_values = [float(row[dr_field]) for row in scheme_rows]
            exact_values = [float(row[exact_field]) for row in scheme_rows]
            selected = [row for row in scheme_rows if int(row[selected_field]) == 1]
            bootstrap_seed = _fold_seed(random_seed, scheme, policy_kind)
            dr_lcb, sign_lcb = _multiway_bootstrap_metrics(
                scheme_rows,
                value_field=dr_field,
                replicates=bootstrap_count,
                random_seed=bootstrap_seed,
            )
            sign_accuracy = _balanced_accuracy(scheme_rows)
            selected_catastrophes = sum(int(row["catastrophic"]) for row in selected)
            output.append(
                {
                    "validation_scheme": scheme,
                    "policy_kind": policy_kind,
                    "scope": "overall",
                    "fold_id": "",
                    "pair_count": len(scheme_rows),
                    "case_count": len({str(row["problem_id"]) for row in scheme_rows}),
                    "seed_count": len({int(row["seed"]) for row in scheme_rows}),
                    "in_support_count": sum(int(row["in_support"]) for row in scheme_rows),
                    "in_support_rate": sum(int(row["in_support"]) for row in scheme_rows) / len(scheme_rows),
                    "selected_count": len(selected),
                    "selected_case_count": len({str(row["problem_id"]) for row in selected}),
                    "selected_seed_count": len({int(row["seed"]) for row in selected}),
                    "selected_catastrophic_count": selected_catastrophes,
                    "selected_catastrophic_cp_ucb": clopper_pearson_upper(selected_catastrophes, len(selected)),
                    "dr_policy_value": math.fsum(dr_values) / len(dr_values),
                    "dr_policy_value_lcb_95": dr_lcb,
                    "exact_pair_policy_value": math.fsum(exact_values) / len(exact_values),
                    "material_pair_count": sum(int(row["material_1pct"]) for row in scheme_rows),
                    "sign_balanced_accuracy": "" if sign_accuracy is None else sign_accuracy,
                    "sign_balanced_accuracy_lcb_95": sign_lcb,
                }
            )
            grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
            for row in scheme_rows:
                grouped[str(row["fold_id"])].append(row)
            for fold, fold_rows in sorted(grouped.items()):
                fold_selected = [row for row in fold_rows if int(row[selected_field]) == 1]
                fold_sign = _balanced_accuracy(fold_rows)
                fold_cat = sum(int(row["catastrophic"]) for row in fold_selected)
                output.append(
                    {
                        "validation_scheme": scheme,
                        "policy_kind": policy_kind,
                        "scope": "fold",
                        "fold_id": fold,
                        "pair_count": len(fold_rows),
                        "case_count": len({str(row["problem_id"]) for row in fold_rows}),
                        "seed_count": len({int(row["seed"]) for row in fold_rows}),
                        "in_support_count": sum(int(row["in_support"]) for row in fold_rows),
                        "in_support_rate": sum(int(row["in_support"]) for row in fold_rows) / len(fold_rows),
                        "selected_count": len(fold_selected),
                        "selected_case_count": len({str(row["problem_id"]) for row in fold_selected}),
                        "selected_seed_count": len({int(row["seed"]) for row in fold_selected}),
                        "selected_catastrophic_count": fold_cat,
                        "selected_catastrophic_cp_ucb": clopper_pearson_upper(fold_cat, len(fold_selected)),
                        "dr_policy_value": math.fsum(float(row[dr_field]) for row in fold_rows) / len(fold_rows),
                        "dr_policy_value_lcb_95": "",
                        "exact_pair_policy_value": math.fsum(float(row[exact_field]) for row in fold_rows) / len(fold_rows),
                        "material_pair_count": sum(int(row["material_1pct"]) for row in fold_rows),
                        "sign_balanced_accuracy": "" if fold_sign is None else fold_sign,
                        "sign_balanced_accuracy_lcb_95": "",
                    }
                )
    return output


def _summary_map(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    return {
        (str(row["validation_scheme"]), str(row["policy_kind"])): row
        for row in rows
        if row["scope"] == "overall"
    }


def _positive_fold_fraction(
    rows: Sequence[Mapping[str, object]], *, scheme: str, policy_kind: str
) -> float:
    folds = [
        row
        for row in rows
        if row["validation_scheme"] == scheme
        and row["policy_kind"] == policy_kind
        and row["scope"] == "fold"
    ]
    if not folds:
        return 0.0
    return sum(float(row["dr_policy_value"]) > 0.0 for row in folds) / len(folds)


def _maximum_case_abs_gain_share(
    predictions: Sequence[Mapping[str, object]], *, selected_field: str
) -> float:
    rows = [
        row
        for row in predictions
        if row["validation_scheme"] == "LCO" and int(row[selected_field]) == 1
    ]
    absolute_by_case: dict[str, float] = defaultdict(float)
    for row in rows:
        absolute_by_case[str(row["problem_id"])] += abs(float(row["exact_tau"]))
    total = math.fsum(absolute_by_case.values())
    return max(absolute_by_case.values(), default=0.0) / total if total > 0.0 else 1.0


def build_identifiability_gate(
    *,
    stage: str,
    pairs: Sequence[DecisionPair],
    raw_stats: Mapping[str, object],
    integrity_blockers: Sequence[str],
    estimator_blockers: Sequence[str],
    predictions: Sequence[Mapping[str, object]],
    policy_rows: Sequence[Mapping[str, object]],
    tree_count: int,
    policy_bootstrap_count: int,
    random_seed: int,
    prior_pilot_gate_pass: bool = False,
    prior_pilot_gate_sha256: str = "",
) -> dict[str, object]:
    if stage not in {"pilot", "full"}:
        raise ValueError("stage must be pilot or full")
    summaries = _summary_map(policy_rows)
    required_summaries = {
        (scheme, policy)
        for scheme in ("LCO", "LSO")
        for policy in POLICY_FIELDS
    }
    missing_summaries = sorted(required_summaries - set(summaries))
    blockers = list(integrity_blockers) + list(estimator_blockers)
    blockers.extend(f"missing_policy_summary:{scheme}:{policy}" for scheme, policy in missing_summaries)

    pair_count = len(pairs)
    case_count = len({pair.problem_id for pair in pairs})
    seed_count = len({pair.seed for pair in pairs})
    material_count = sum(_material_one_percent(pair.tau) for pair in pairs)
    missing_rate = float(raw_stats["critical_feature_missing_rate"])
    integrity_pass = not integrity_blockers and pair_count == int(raw_stats["applicable_pair_count"])

    candidate_lco = summaries.get(("LCO", "utility_candidate_policy"), {})
    candidate_lso = summaries.get(("LSO", "utility_candidate_policy"), {})
    safe_lco = summaries.get(("LCO", "safe_release_policy"), {})
    safe_lso = summaries.get(("LSO", "safe_release_policy"), {})

    pilot_checks = {
        "integrity_pass": integrity_pass,
        "frozen_pilot_matrix_match": _stage_matrix_matches(raw_stats, "pilot"),
        "production_tree_count_1000": tree_count == DEFAULT_TREE_COUNT,
        "production_policy_bootstraps_2000": (
            policy_bootstrap_count == DEFAULT_POLICY_BOOTSTRAPS
        ),
        "frozen_random_seed_20260715": random_seed == DEFAULT_RANDOM_SEED,
        "applicable_pairs_ge_30": pair_count >= 30,
        "cases_ge_6": case_count >= 6,
        "seeds_ge_5": seed_count >= 5,
        "critical_feature_missing_rate_le_0_05": missing_rate <= 0.05,
        "material_pairs_ge_15": material_count >= 15,
        "candidate_lco_dr_positive": float(candidate_lco.get("dr_policy_value", 0.0)) > 0.0,
        "candidate_lso_dr_positive": float(candidate_lso.get("dr_policy_value", 0.0)) > 0.0,
        "candidate_sign_balanced_accuracy_gt_0_55": min(
            float(candidate_lco.get("sign_balanced_accuracy") or 0.0),
            float(candidate_lso.get("sign_balanced_accuracy") or 0.0),
        ) > 0.55,
        "both_schemes_in_support_ge_0_50": min(
            float(candidate_lco.get("in_support_rate", 0.0)),
            float(candidate_lso.get("in_support_rate", 0.0)),
        ) >= 0.50,
        "candidate_selected_catastrophic_zero_both_schemes": (
            int(candidate_lco.get("selected_catastrophic_count", 1)) == 0
            and int(candidate_lso.get("selected_catastrophic_count", 1)) == 0
        ),
    }
    pilot_pass = not missing_summaries and all(pilot_checks.values()) and not estimator_blockers

    safe_lco_predictions = [
        row
        for row in predictions
        if row["validation_scheme"] == "LCO" and int(row["safe_release_selected"]) == 1
    ]
    lco_exact = float(safe_lco.get("exact_pair_policy_value", 0.0))
    lso_exact = float(safe_lso.get("exact_pair_policy_value", 0.0))
    lco_dr = float(safe_lco.get("dr_policy_value", 0.0))
    lso_dr = float(safe_lso.get("dr_policy_value", 0.0))
    full_checks = {
        "prior_pilot_gate_pass": prior_pilot_gate_pass,
        "frozen_full_matrix_match": _stage_matrix_matches(raw_stats, "full"),
        "applicable_pairs_ge_80": pair_count >= 80,
        "cases_ge_8": case_count >= 8,
        "seeds_ge_8": seed_count >= 8,
        "lco_in_support_ge_0_60": float(safe_lco.get("in_support_rate", 0.0)) >= 0.60,
        "safe_lco_dr_lcb_positive": float(safe_lco.get("dr_policy_value_lcb_95", 0.0)) > 0.0,
        "safe_lso_dr_lcb_positive": float(safe_lso.get("dr_policy_value_lcb_95", 0.0)) > 0.0,
        "safe_lco_exact_dr_direction_agree": lco_exact != 0.0 and lco_exact * lco_dr > 0.0,
        "safe_lso_exact_dr_direction_agree": lso_exact != 0.0 and lso_exact * lso_dr > 0.0,
        "maximum_case_abs_gain_share_le_0_50": _maximum_case_abs_gain_share(
            predictions, selected_field="safe_release_selected"
        ) <= 0.50,
        "material_pairs_ge_30": material_count >= 30,
        "lco_sign_accuracy_lcb_gt_0_50": float(safe_lco.get("sign_balanced_accuracy_lcb_95", 0.0)) > 0.50,
        "positive_case_fold_fraction_ge_0_75": _positive_fold_fraction(
            policy_rows, scheme="LCO", policy_kind="safe_release_policy"
        ) >= 0.75,
        "positive_seed_fold_fraction_ge_0_60": _positive_fold_fraction(
            policy_rows, scheme="LSO", policy_kind="safe_release_policy"
        ) >= 0.60,
        "heldout_safe_releases_ge_59": len(safe_lco_predictions) >= 59,
        "safe_release_cases_ge_6": len({str(row["problem_id"]) for row in safe_lco_predictions}) >= 6,
        "safe_release_seeds_ge_5": len({int(row["seed"]) for row in safe_lco_predictions}) >= 5,
        "safe_release_catastrophic_zero_both_schemes": (
            int(safe_lco.get("selected_catastrophic_count", 1)) == 0
            and int(safe_lso.get("selected_catastrophic_count", 1)) == 0
        ),
        "safe_release_catastrophic_cp_ucb_le_0_05_both_schemes": (
            float(safe_lco.get("selected_catastrophic_cp_ucb", 1.0)) <= RISK_LIMIT
            and float(safe_lso.get("selected_catastrophic_cp_ucb", 1.0)) <= RISK_LIMIT
        ),
        "production_tree_count_1000": tree_count == DEFAULT_TREE_COUNT,
        "production_policy_bootstraps_2000": policy_bootstrap_count == DEFAULT_POLICY_BOOTSTRAPS,
        "frozen_random_seed_20260715": random_seed == DEFAULT_RANDOM_SEED,
    }
    full_pass = integrity_pass and not estimator_blockers and all(full_checks.values())
    runtime_authorized = bool(stage == "full" and full_pass and not blockers)
    active_checks = full_checks if stage == "full" else pilot_checks
    blockers.extend(name for name, passed in active_checks.items() if not passed)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "overall_status": "pass" if (full_pass if stage == "full" else pilot_pass) and not blockers else "fail",
        "runtime_scheduler_authorized": runtime_authorized,
        "policy_semantics": {
            "utility_candidate_policy": "pilot-only identifiability policy; not safe for runtime release",
            "safe_release_policy": "requires OOD pass, calibrated positive utility LCB, and catastrophic-risk UCB <= 0.05",
        },
        "raw_stats": dict(raw_stats),
        "valid_pair_count": pair_count,
        "case_count": case_count,
        "seed_count": seed_count,
        "material_pair_count": material_count,
        "integrity_pass": integrity_pass,
        "pilot_criteria_pass": pilot_pass,
        "full_criteria_pass": full_pass,
        "prior_pilot_gate_sha256": prior_pilot_gate_sha256,
        "source_logging_root": raw_stats.get("source_logging_root", ""),
        "source_logging_manifest_sha256": raw_stats.get(
            "source_logging_manifest_sha256", ""
        ),
        "source_raw_artifact_sha256": raw_stats.get(
            "source_raw_artifact_sha256", {}
        ),
        "source_matrix": raw_stats.get("source_matrix", {}),
        "source_matrix_sha256": _json_sha256(raw_stats.get("source_matrix", {})),
        "source_lane_profile": raw_stats.get("source_lane_profile", ""),
        "source_git_commit": raw_stats.get("source_git_commit", ""),
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": PREREGISTRATION_SHA256,
            "commit": PREREGISTRATION_COMMIT,
        },
        "estimator_configuration": {
            "tree_count": tree_count,
            "policy_bootstrap_count": policy_bootstrap_count,
            "tree_max_depth": 2,
            "registered_propensity": PROPENSITY,
            "random_seed": random_seed,
        },
        "pilot_checks": pilot_checks,
        "full_checks": full_checks,
        "utility_candidate_summary": {
            "LCO": dict(candidate_lco),
            "LSO": dict(candidate_lso),
        },
        "safe_release_summary": {
            "LCO": dict(safe_lco),
            "LSO": dict(safe_lso),
        },
        "blockers": sorted(set(blockers)),
    }


def build_model_payload(
    pairs: Sequence[DecisionPair],
    *,
    tree_count: int,
    random_seed: int,
) -> dict[str, object]:
    if tree_count != BOOTSTRAP_TREE_COUNT:
        raise ValueError("runtime model export requires exactly 1000 bootstrap trees")
    model = _fit_fold_model(
        pairs,
        tree_count=tree_count,
        random_seed=random_seed,
        fit_nuisance=False,
    )
    payload: dict[str, object] = {
        "schema_version": CAUSAL_RISK_MODEL_SCHEMA_VERSION,
        "feature_schema": {
            "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
            "feature_names": list(UTILITY_FEATURE_NAMES),
            "sha256": FEATURE_SCHEMA_SHA256,
        },
        "ood": {
            "median": list(model.support.median),
            "iqr": list(model.support.iqr),
            "minimum": list(model.support.minimum),
            "maximum": list(model.support.maximum),
            "reference_scaled": [list(row) for row in model.support.reference_scaled],
            "knn_k": 5,
            "knn_distance_threshold": model.support.knn_distance_threshold,
        },
        "utility": {
            "bootstrap_trees": model.utility_trees,
            "lcb_quantile": 0.05,
            "conformal_margin": model.conformal_margin,
        },
        "catastrophic_risk": {
            "bootstrap_trees": model.risk_trees,
            "bootstrap_quantile": 0.95,
            "clopper_pearson_tree": model.cp_tree,
        },
    }
    payload["model_sha256"] = compute_model_sha256(payload)
    CausalRiskModelBundle.from_mapping(payload)
    return payload


PAIR_FIELDS = (
    "pair_id",
    "decision_id",
    "problem_id",
    "seed",
    "feature_sha256",
    "logged_arm",
    "observed_treatment",
    "observed_log_progress",
    "checkpoint_error",
    "baseline_terminal_error",
    "action_terminal_error",
    "baseline_log_progress",
    "action_log_progress",
    "paired_tau",
    "catastrophic",
    "material_1pct",
)

FOLD_FIELDS = (
    "pair_id",
    "decision_id",
    "problem_id",
    "seed",
    "lco_fold",
    "lso_fold",
)

CROSSFIT_FIELDS = (
    "validation_scheme",
    "fold_id",
    "pair_id",
    "decision_id",
    "problem_id",
    "seed",
    "training_pair_count",
    "training_case_count",
    "training_seed_count",
    "logged_arm",
    "observed_treatment",
    "observed_log_progress",
    "tau_hat",
    "tau_lcb",
    "conformal_margin",
    "catastrophic_risk_ucb",
    "in_support",
    "ood_reasons",
    "mu_baseline",
    "mu_action",
    "dr_effect_score",
    "utility_candidate_selected",
    "safe_release_selected",
    "dr_utility_candidate",
    "dr_safe_release",
    "exact_tau",
    "catastrophic",
    "material_1pct",
    "predicted_positive",
    "sign_correct",
    "exact_utility_candidate",
    "exact_safe_release",
)

POLICY_SUMMARY_FIELDS = (
    "validation_scheme",
    "policy_kind",
    "scope",
    "fold_id",
    "pair_count",
    "case_count",
    "seed_count",
    "in_support_count",
    "in_support_rate",
    "selected_count",
    "selected_case_count",
    "selected_seed_count",
    "selected_catastrophic_count",
    "selected_catastrophic_cp_ucb",
    "dr_policy_value",
    "dr_policy_value_lcb_95",
    "exact_pair_policy_value",
    "material_pair_count",
    "sign_balanced_accuracy",
    "sign_balanced_accuracy_lcb_95",
)


def validate_prior_pilot_gate(path: Path) -> tuple[bool, str, list[str]]:
    blockers: list[str] = []
    try:
        gate = _read_json(path)
        gate_sha256 = _file_sha256(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, "", [f"prior_pilot_gate_unreadable:{exc}"]
    expected_estimator = {
        "tree_count": DEFAULT_TREE_COUNT,
        "policy_bootstrap_count": DEFAULT_POLICY_BOOTSTRAPS,
        "tree_max_depth": 2,
        "registered_propensity": PROPENSITY,
        "random_seed": DEFAULT_RANDOM_SEED,
    }
    if (
        gate.get("protocol_version") != PROTOCOL_VERSION
        or gate.get("stage") != "pilot"
        or gate.get("overall_status") != "pass"
        or gate.get("pilot_criteria_pass") is not True
        or gate.get("runtime_scheduler_authorized") is not False
        or gate.get("blockers") != []
        or gate.get("preregistration")
        != {
            "path": PREREGISTRATION_PATH,
            "sha256": PREREGISTRATION_SHA256,
            "commit": PREREGISTRATION_COMMIT,
        }
        or gate.get("estimator_configuration") != expected_estimator
        or not _stage_matrix_matches(gate, "pilot")
        or gate.get("source_matrix_sha256")
        != _json_sha256(gate.get("source_matrix", {}))
    ):
        blockers.append("prior_pilot_gate_contract_mismatch")
    checks = gate.get("pilot_checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        blockers.append("prior_pilot_gate_checks_not_all_passed")

    source_root_value = gate.get("source_logging_root")
    source_root = Path(str(source_root_value)) if source_root_value else None
    if source_root is None or not source_root.is_dir():
        blockers.append("prior_pilot_source_logging_root_missing")
    else:
        manifest_path = source_root / "causal_logging_manifest.json"
        try:
            manifest = _read_json(manifest_path)
            if _file_sha256(manifest_path) != gate.get(
                "source_logging_manifest_sha256"
            ):
                blockers.append("prior_pilot_logging_manifest_sha256_mismatch")
            if manifest.get("raw_artifact_sha256") != gate.get(
                "source_raw_artifact_sha256"
            ):
                blockers.append("prior_pilot_raw_artifact_hash_binding_mismatch")
            if manifest.get("matrix") != gate.get("source_matrix"):
                blockers.append("prior_pilot_matrix_binding_mismatch")
            _manifest, source_blockers = validate_manifests(source_root)
            blockers.extend(f"prior_pilot:{value}" for value in source_blockers)
            source_pairs, source_stats, pair_blockers = load_decision_pairs(source_root)
            blockers.extend(f"prior_pilot:{value}" for value in pair_blockers)
            _header, frozen_predictions = _read_csv(
                path.parent / "crossfit_predictions.csv"
            )
            if len(frozen_predictions) != 2 * len(source_pairs):
                blockers.append("prior_pilot_crossfit_matrix_incomplete")
            else:
                recomputed_policy = build_policy_value_rows(
                    frozen_predictions,
                    bootstrap_count=DEFAULT_POLICY_BOOTSTRAPS,
                    random_seed=DEFAULT_RANDOM_SEED,
                )
                recomputed_gate = build_identifiability_gate(
                    stage="pilot",
                    pairs=source_pairs,
                    raw_stats=source_stats,
                    integrity_blockers=pair_blockers,
                    estimator_blockers=[],
                    predictions=frozen_predictions,
                    policy_rows=recomputed_policy,
                    tree_count=DEFAULT_TREE_COUNT,
                    policy_bootstrap_count=DEFAULT_POLICY_BOOTSTRAPS,
                    random_seed=DEFAULT_RANDOM_SEED,
                )
                if (
                    recomputed_gate["overall_status"] != "pass"
                    or recomputed_gate["pilot_checks"] != gate.get("pilot_checks")
                ):
                    blockers.append("prior_pilot_gate_recomputation_mismatch")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"prior_pilot_source_validation_failed:{exc}")

    derived = gate.get("derived_artifact_sha256")
    if not isinstance(derived, Mapping):
        blockers.append("prior_pilot_derived_hashes_missing")
    else:
        for filename in (
            "causal_pairs.csv",
            "fold_assignments.csv",
            "crossfit_predictions.csv",
            "policy_value_summary.csv",
        ):
            artifact = path.parent / filename
            if not artifact.is_file() or derived.get(filename) != _file_sha256(artifact):
                blockers.append(f"prior_pilot_derived_hash_mismatch:{filename}")
    return not blockers, gate_sha256, sorted(set(blockers))


def write_reports(
    *,
    input_root: Path,
    output_root: Path,
    stage: str,
    tree_count: int = DEFAULT_TREE_COUNT,
    policy_bootstrap_count: int = DEFAULT_POLICY_BOOTSTRAPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    prior_pilot_gate: Path | None = None,
) -> tuple[Path, ...]:
    pairs, raw_stats, integrity_blockers = load_decision_pairs(input_root)
    predictions: list[dict[str, object]] = []
    estimator_blockers: list[str] = []
    for scheme in ("LCO", "LSO"):
        scheme_predictions, scheme_blockers = build_crossfit_predictions(
            pairs,
            scheme=scheme,
            tree_count=tree_count,
            random_seed=random_seed,
        )
        predictions.extend(scheme_predictions)
        estimator_blockers.extend(scheme_blockers)
    policy_rows: list[dict[str, object]] = []
    if predictions:
        expected_prediction_count = 2 * len(pairs)
        if len(predictions) != expected_prediction_count:
            estimator_blockers.append("crossfit_prediction_matrix_incomplete")
        else:
            try:
                policy_rows = build_policy_value_rows(
                    predictions,
                    bootstrap_count=policy_bootstrap_count,
                    random_seed=random_seed,
                )
            except ValueError as exc:
                estimator_blockers.append(f"policy_bootstrap_failed:{exc}")
    else:
        estimator_blockers.append("crossfit_predictions_empty")

    prior_pilot_gate_pass = False
    prior_pilot_gate_sha256 = ""
    prior_pilot_blockers: list[str] = []
    if prior_pilot_gate is not None:
        (
            prior_pilot_gate_pass,
            prior_pilot_gate_sha256,
            prior_pilot_blockers,
        ) = validate_prior_pilot_gate(prior_pilot_gate)
    elif stage == "full":
        prior_pilot_blockers.append("prior_pilot_gate_missing")
    estimator_blockers.extend(prior_pilot_blockers)
    gate = build_identifiability_gate(
        stage=stage,
        pairs=pairs,
        raw_stats=raw_stats,
        integrity_blockers=integrity_blockers,
        estimator_blockers=estimator_blockers,
        predictions=predictions,
        policy_rows=policy_rows,
        tree_count=tree_count,
        policy_bootstrap_count=policy_bootstrap_count,
        random_seed=random_seed,
        prior_pilot_gate_pass=prior_pilot_gate_pass,
        prior_pilot_gate_sha256=prior_pilot_gate_sha256,
    )
    pair_rows = [
        {
            "pair_id": pair.pair_id,
            "decision_id": pair.decision_id,
            "problem_id": pair.problem_id,
            "seed": pair.seed,
            "feature_sha256": pair.feature_sha256,
            "logged_arm": pair.logged_arm,
            "observed_treatment": pair.observed_treatment,
            "observed_log_progress": pair.observed_y,
            "checkpoint_error": pair.checkpoint_error,
            "baseline_terminal_error": pair.baseline_error,
            "action_terminal_error": pair.action_error,
            "baseline_log_progress": pair.y0,
            "action_log_progress": pair.y1,
            "paired_tau": pair.tau,
            "catastrophic": pair.catastrophic,
            "material_1pct": int(_material_one_percent(pair.tau)),
        }
        for pair in pairs
    ]
    fold_rows = [
        {
            "pair_id": pair.pair_id,
            "decision_id": pair.decision_id,
            "problem_id": pair.problem_id,
            "seed": pair.seed,
            "lco_fold": pair.problem_id,
            "lso_fold": pair.seed,
        }
        for pair in pairs
    ]

    paths = (
        output_root / "causal_pairs.csv",
        output_root / "fold_assignments.csv",
        output_root / "crossfit_predictions.csv",
        output_root / "policy_value_summary.csv",
        output_root / "causal_identifiability_gate.json",
    )
    _write_csv(paths[0], pair_rows, fieldnames=PAIR_FIELDS)
    _write_csv(paths[1], fold_rows, fieldnames=FOLD_FIELDS)
    _write_csv(paths[2], predictions, fieldnames=CROSSFIT_FIELDS)
    _write_csv(paths[3], policy_rows, fieldnames=POLICY_SUMMARY_FIELDS)
    gate["derived_artifact_sha256"] = {
        path.name: _file_sha256(path) for path in paths[:4]
    }
    _write_json(paths[4], gate)

    model_path = output_root / "causal_risk_precision_model.json"
    if gate["runtime_scheduler_authorized"]:
        model_payload = build_model_payload(
            pairs,
            tree_count=tree_count,
            random_seed=_fold_seed(random_seed, "FULL", "MODEL"),
        )
        _write_json(model_path, model_payload)
    else:
        model_path.unlink(missing_ok=True)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--tree-count", type=int, default=DEFAULT_TREE_COUNT)
    parser.add_argument(
        "--policy-bootstrap-count",
        type=int,
        default=DEFAULT_POLICY_BOOTSTRAPS,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--prior-pilot-gate",
        type=Path,
        help="required for --stage full; must be an independently passing pilot gate",
    )
    args = parser.parse_args(argv)
    if args.stage == "full" and args.prior_pilot_gate is None:
        parser.error("--stage full requires --prior-pilot-gate")
    paths = write_reports(
        input_root=args.input_dir,
        output_root=args.output_dir,
        stage=args.stage,
        tree_count=args.tree_count,
        policy_bootstrap_count=args.policy_bootstrap_count,
        random_seed=args.random_seed,
        prior_pilot_gate=args.prior_pilot_gate,
    )
    gate = _read_json(paths[-1])
    if gate["overall_status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
