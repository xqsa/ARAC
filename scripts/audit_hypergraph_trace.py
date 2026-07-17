"""Audit the frozen v37 overlap-hypergraph predictive trace protocol."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from arac.policy.overlap_hypergraph import (
    HyperedgeCycleState,
    HyperedgeScore,
    build_overlap_hypergraph,
    directional_survival,
    midrank_percentiles,
    score_hyperedge_states,
    unit_fe_contribution,
)
from arac.backends.hcc_hypergraph_trace import (
    HYPERGRAPH_AUDIT_FIELDS as RUNTIME_AUDIT_FIELDS,
    HYPERGRAPH_FEATURE_FIELDS as RUNTIME_FEATURE_FIELDS,
    HYPERGRAPH_NATIVE_SWEEP_END_STAGE,
    HYPERGRAPH_OUTCOME_FIELDS as RUNTIME_OUTCOME_FIELDS,
    HYPERGRAPH_PROPOSAL_FIELDS as RUNTIME_PROPOSAL_FIELDS,
)


CONFIG_PATH = ROOT / "configs" / "hypergraph_delayed_credit_v1.json"
SPEC_PATH = ROOT / "docs" / "design" / "hypergraph-delayed-credit-v1.md"
CONFIG_SHA256 = "b2be4e22d3ddc323199f36884d10e792b1e05e48e5bc10924d08a68ff93394f4"
SPEC_SHA256 = "0ab7aa3cabd5bb66292b441fce425e0a7d424109a36b7464c2e9a72138cce7f4"
PROTOCOL_VERSION = "hypergraph-delayed-credit-v1"
STAGES = ("screen", "full")

STATE_FIELDS = (
    "current_unit_fe_contribution",
    "ewma_unit_fe_contribution_3",
    "zero_gain_difficulty",
    "stagnation_ratio_3",
    "direct_owner_proposal_disagreement",
    "prior_next_sweep_overwrite",
)
SCORE_FIELDS = (
    "contribution_score",
    "need_score",
    "focal_priority",
    "owner_reliability",
)
FEATURE_FIELDS = ("decision_id", *STATE_FIELDS, *SCORE_FIELDS)
if FEATURE_FIELDS != tuple(RUNTIME_FEATURE_FIELDS):
    raise RuntimeError("auditor feature schema diverges from runtime")
AUDIT_FIELDS = tuple(RUNTIME_AUDIT_FIELDS)
PROPOSAL_FIELDS = tuple(RUNTIME_PROPOSAL_FIELDS)
OUTCOME_FIELDS = tuple(RUNTIME_OUTCOME_FIELDS)
METRIC_NAMES = (
    "trajectory_priority_spearman",
    "trajectory_focal_rank_advantage",
    "trajectory_owner_survival_spearman",
    "overwrite_balanced_accuracy",
)
TRAJECTORY_METRIC_FIELDS = METRIC_NAMES[:3]
FOLD_FIELDS = (
    "decision_id",
    "problem_id",
    "seed",
    "fold_type",
    "held_out_value",
    "trajectory_weight",
    "support_reason",
)
PREDICTION_FIELDS = (
    "decision_id",
    "problem_id",
    "seed",
    "fold_type",
    "held_out_value",
    "trajectory_weight",
    "focal_priority",
    "owner_reliability",
    "next_sweep_unit_fe_contribution",
    "next_sweep_survival",
    "next_sweep_overwrite",
    "overwrite_prediction",
    "reliability_training_median",
)
SUMMARY_FIELDS = (
    "validation",
    "fold",
    "row_count",
    *METRIC_NAMES,
    "bootstrap_lcb_95",
)

SOURCE_ARTIFACT_SCHEMAS = {
    "hyperedge_cycle_features.csv": FEATURE_FIELDS,
    "hyperedge_cycle_audit.csv": AUDIT_FIELDS,
    "shared_proposal_audit.csv": PROPOSAL_FIELDS,
    "hyperedge_cycle_outcomes.csv": OUTCOME_FIELDS,
}
SOURCE_BUNDLE_PATHS = (
    "src/arac/policy/overlap_hypergraph.py",
    "src/arac/backends/hcc_hypergraph_trace.py",
    "src/arac/backends/hcc.py",
    "scripts/hcc_smoke_runner.py",
    "experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py",
    "scripts/audit_hypergraph_trace.py",
)
SOURCE_MANIFEST_WRAPPER_FIELDS = frozenset(
    {
        "lane_id",
        "path",
        "sha256",
        "hcc_result_status",
        "hcc_result_max_fes",
        "hcc_result_actual_fe_used",
    }
)
_CONFIG_FORBIDDEN_POLICY_INPUTS = frozenset(
    str(value)
    for value in json.loads(CONFIG_PATH.read_text(encoding="utf-8"))[
        "forbidden_policy_inputs"
    ]
)
_STRUCTURAL_SCOPE_IDENTIFIERS = frozenset(
    {"group_index", "variable_index", "relation_index", "component_index"}
)
if not _STRUCTURAL_SCOPE_IDENTIFIERS <= _CONFIG_FORBIDDEN_POLICY_INPUTS:
    raise RuntimeError("structural policy exceptions diverge from frozen config")
POLICY_FORBIDDEN_IDENTIFIERS = (
    _CONFIG_FORBIDDEN_POLICY_INPUTS - _STRUCTURAL_SCOPE_IDENTIFIERS
)
POLICY_STATE_FIELDS = STATE_FIELDS
POLICY_SCORE_FIELDS = SCORE_FIELDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("git rev-parse HEAD did not return a full commit")
    return commit


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != tuple(fields):
            raise ValueError(f"{path.name} schema mismatch: {header}")
        return list(reader)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _int(row: Mapping[str, str], field: str, minimum: int = 0) -> int:
    try:
        value = int(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _flag(row: Mapping[str, str], field: str) -> bool:
    value = row.get(field)
    if value not in {"0", "1"}:
        raise ValueError(f"{field} must be 0 or 1")
    return value == "1"


def _hex64(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _hex40(value: object) -> bool:
    text = str(value)
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_bundle_from_paths(relative_paths: Sequence[str]) -> dict[str, object]:
    paths = tuple(str(path) for path in relative_paths)
    if len(paths) != len(set(paths)):
        raise ValueError("source bundle paths must be unique")
    file_sha256 = {
        relative_path: _sha256(ROOT / relative_path)
        for relative_path in sorted(paths)
    }
    return {
        "schema_version": 1,
        "file_sha256": file_sha256,
        "files": [
            {"path": relative_path, "sha256": digest}
            for relative_path, digest in file_sha256.items()
        ],
        "bundle_sha256": _canonical_sha256(file_sha256),
    }


def hypergraph_source_bundle() -> dict[str, object]:
    """Return the canonical read-only source bundle used by trace runs."""

    return _source_bundle_from_paths(SOURCE_BUNDLE_PATHS)


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        function = decorator.func
        if not isinstance(function, ast.Name) or function.id != "dataclass":
            continue
        return any(
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
    return False


def _policy_identifier_forbidden(
    value: str,
    *,
    include_structural: bool = False,
) -> bool:
    normalized = str(value).lower()
    parts = set(normalized.split("_"))
    forbidden_identifiers = (
        _CONFIG_FORBIDDEN_POLICY_INPUTS
        if include_structural
        else POLICY_FORBIDDEN_IDENTIFIERS
    )
    return bool(
        parts & {"case", "problem", "seed", "family"}
        or any(
            marker in normalized
            for marker in (
                "fingerprint",
                "raw_objective",
                "raw_incumbent",
                "final_outcome",
                "terminal_outcome",
                "paper_best",
                "historical_result",
            )
        )
        or normalized in forbidden_identifiers
    )


def _policy_ast_audit_from_source(source: str) -> dict[str, object]:
    blockers: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "status": "fail",
            "checks": {},
            "blockers": [f"policy_syntax_error:{exc}"],
        }
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    expected_classes = {
        "HyperedgeCycleState": POLICY_STATE_FIELDS,
        "HyperedgeScore": POLICY_SCORE_FIELDS,
    }
    expected_functions = {
        "build_hyperedge_cycle_states": (
            "topology",
            "snapshots",
            "closed_owner_credits",
            "decision_fe",
            "lower_bound",
            "upper_bound",
        ),
        "score_hyperedge_states": ("states",),
    }
    checks: dict[str, bool] = {}
    boundary_nodes: list[ast.AST] = []
    for name, expected_fields in expected_classes.items():
        node = classes.get(name)
        frozen = node is not None and _is_frozen_dataclass(node)
        observed_fields = (
            tuple(
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            )
            if node is not None
            else ()
        )
        checks[f"{name}_frozen"] = frozen
        checks[f"{name}_fields"] = observed_fields == tuple(expected_fields)
        if not frozen:
            blockers.append(f"policy_dataclass_not_frozen:{name}")
        if observed_fields != tuple(expected_fields):
            blockers.append(f"policy_dataclass_fields_changed:{name}")
        if node is not None:
            boundary_nodes.append(node)
    for name, expected_arguments in expected_functions.items():
        node = functions.get(name)
        observed_arguments = (
            tuple(
                argument.arg
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            )
            if node is not None
            else ()
        )
        boundary_valid = bool(
            node is not None
            and node.args.vararg is None
            and node.args.kwarg is None
            and observed_arguments == expected_arguments
        )
        checks[f"{name}_arguments"] = boundary_valid
        if not boundary_valid:
            blockers.append(f"policy_function_boundary_changed:{name}")
        if node is not None:
            boundary_nodes.append(node)
    module_definitions: dict[str, ast.FunctionDef | ast.ClassDef] = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    boundary_names = {
        node.name
        for node in boundary_nodes
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }

    def reachable_from(entry_names: set[str]) -> set[str]:
        reachable = set(entry_names) & set(module_definitions)
        pending = list(reachable)
        while pending:
            current = module_definitions[pending.pop()]
            for child in ast.walk(current):
                if not isinstance(child, ast.Call) or not isinstance(
                    child.func, ast.Name
                ):
                    continue
                called_name = child.func.id
                if called_name in module_definitions and called_name not in reachable:
                    reachable.add(called_name)
                    pending.append(called_name)
        return reachable

    reachable_names = reachable_from(boundary_names)
    reachable_nodes = [module_definitions[name] for name in sorted(reachable_names)]
    checks["same_module_call_graph_resolved"] = bool(reachable_nodes)

    def identifiers(nodes: Sequence[ast.AST]) -> set[str]:
        observed: set[str] = set()
        for boundary_node in nodes:
            for node in ast.walk(boundary_node):
                if isinstance(node, ast.Name):
                    observed.add(node.id)
                elif isinstance(node, ast.arg):
                    observed.add(node.arg)
                elif isinstance(node, ast.Attribute):
                    observed.add(node.attr)
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node.value)
                ):
                    observed.add(node.value)
        return observed

    # This module is the pure runtime policy boundary. Scan it in full so a
    # reachable attribute method cannot hide an identity-dependent helper.
    forbidden = {
        value for value in identifiers((tree,)) if _policy_identifier_forbidden(value)
    }
    checks["forbidden_identifiers_absent"] = not forbidden
    if forbidden:
        blockers.append("policy_forbidden_identifiers:" + ",".join(sorted(forbidden)))

    score_names = reachable_from({"HyperedgeScore", "score_hyperedge_states"})
    score_forbidden = {
        value
        for value in identifiers(
            tuple(module_definitions[name] for name in sorted(score_names))
        )
        if _policy_identifier_forbidden(value, include_structural=True)
    }
    checks["score_call_graph_forbidden_identifiers_absent"] = not score_forbidden
    if score_forbidden:
        blockers.append(
            "policy_score_forbidden_identifiers:" + ",".join(sorted(score_forbidden))
        )
    return {
        "status": "pass" if not blockers and all(checks.values()) else "fail",
        "checks": checks,
        "blockers": blockers,
        "reachable_definitions": sorted(reachable_names),
    }


def hypergraph_policy_ast_audit() -> dict[str, object]:
    """Independently audit the frozen pure state/score policy boundary."""

    policy_path = ROOT / "src" / "arac" / "policy" / "overlap_hypergraph.py"
    audit_result = _policy_ast_audit_from_source(
        policy_path.read_text(encoding="utf-8")
    )
    return {
        **audit_result,
        "policy_path": policy_path.relative_to(ROOT).as_posix(),
        "policy_sha256": _sha256(policy_path),
    }


def hypergraph_static_ast_audit() -> dict[str, object]:
    """Public stable alias used by the exp003 aggregate manifest."""

    return hypergraph_policy_ast_audit()


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _format_float(value: float) -> str:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("trace values must be finite")
    return f"{converted:.17e}"


def _raw_feature_payload(
    state: HyperedgeCycleState,
    score: HyperedgeScore,
) -> dict[str, str]:
    return {
        field: _format_float(float(getattr(state, field))) for field in STATE_FIELDS
    } | {
        field: _format_float(float(getattr(score, field))) for field in SCORE_FIELDS
    }


def _trajectory_key(row: Mapping[str, object]) -> tuple[str, str]:
    problem_id = str(row.get("problem_id", ""))
    try:
        seed = str(int(str(row.get("seed", ""))))
    except ValueError as exc:
        raise ValueError("seed must be an integer") from exc
    if not problem_id:
        raise ValueError("problem_id must be non-empty")
    return problem_id, seed


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _weighted_median(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("weighted median requires aligned non-empty values")
    pairs = sorted((float(value), float(weight)) for value, weight in zip(values, weights, strict=True))
    if any(weight < 0.0 for _, weight in pairs):
        raise ValueError("weighted median weights must be non-negative")
    total = math.fsum(weight for _, weight in pairs)
    if total <= 0.0:
        raise ValueError("weighted median requires positive total weight")
    threshold = total / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator <= 0.0:
        return None
    return math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(midrank_percentiles(left), midrank_percentiles(right))


def _balanced_accuracy(
    actual: Sequence[int],
    predicted: Sequence[int],
    weights: Sequence[float] | None = None,
) -> float | None:
    if len(actual) != len(predicted) or not actual:
        return None
    resolved_weights = (
        tuple(1.0 for _ in actual) if weights is None else tuple(float(w) for w in weights)
    )
    if len(resolved_weights) != len(actual) or any(weight < 0.0 for weight in resolved_weights):
        return None
    classes = set(actual)
    if classes != {0, 1}:
        return None
    recalls = []
    for label in (0, 1):
        indices = [index for index, value in enumerate(actual) if value == label]
        denominator = math.fsum(resolved_weights[index] for index in indices)
        if denominator <= 0.0:
            return None
        recalls.append(
            math.fsum(
                resolved_weights[index]
                for index in indices
                if predicted[index] == label
            )
            / denominator
        )
    return statistics.fmean(recalls)


def _metric(rows: Sequence[Mapping[str, object]], name: str) -> float | None:
    if name in TRAJECTORY_METRIC_FIELDS:
        if not rows:
            return None
        return statistics.fmean(float(row[name]) for row in rows)
    if name == "overwrite_balanced_accuracy":
        return _balanced_accuracy(
            [int(float(row["next_sweep_overwrite"]) > 0.5) for row in rows],
            [int(row["overwrite_prediction"]) for row in rows],
            [float(row["trajectory_weight"]) for row in rows],
        )
    raise KeyError(name)


def _crossfit(
    rows: Sequence[Mapping[str, object]],
    *,
    fold_type: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    identity_field = "problem_id" if fold_type == "lco" else "seed"
    held_out_values = sorted({str(row[identity_field]) for row in rows})
    assignments: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    blockers: list[str] = []
    for held_out in held_out_values:
        training = [row for row in rows if str(row[identity_field]) != held_out]
        testing = [row for row in rows if str(row[identity_field]) == held_out]
        if not testing:
            continue
        if not training:
            blockers.append(f"{fold_type}:{held_out}:empty_training_fold")
            continue
        training_counts = Counter(_trajectory_key(row) for row in training)
        testing_counts = Counter(_trajectory_key(row) for row in testing)
        reliability_median = _weighted_median(
            [float(row["owner_reliability"]) for row in training],
            [1.0 / training_counts[_trajectory_key(row)] for row in training],
        )
        for row in testing:
            assignments.append(
                {
                    "decision_id": row["decision_id"],
                    "problem_id": row["problem_id"],
                    "seed": row["seed"],
                    "fold_type": fold_type,
                    "held_out_value": held_out,
                    "trajectory_weight": 1.0
                    / testing_counts[_trajectory_key(row)],
                    "support_reason": "no_support_filter",
                }
            )
            predictions.append(
                {
                    **row,
                    "fold_type": fold_type,
                    "held_out_value": held_out,
                    "trajectory_weight": 1.0
                    / testing_counts[_trajectory_key(row)],
                    "overwrite_prediction": int(
                        float(row["owner_reliability"]) < reliability_median
                    ),
                    "reliability_training_median": reliability_median,
                }
            )
    return assignments, predictions, blockers


def _bootstrap_lcb(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    *,
    resamples: int,
    seed: int,
    quantile: float,
) -> float | None:
    if not rows:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = _pigeonhole_sample(rows, rng)
        value = _metric(sample, metric)
        if value is None and sample and metric == "overwrite_balanced_accuracy":
            value = 0.0
        values.append(float("-inf") if value is None else value)
    lcb = _quantile(values, quantile)
    return None if not math.isfinite(lcb) else lcb


def _pigeonhole_sample(
    rows: Sequence[Mapping[str, object]],
    rng: random.Random,
) -> list[Mapping[str, object]]:
    cases = sorted({str(row["problem_id"]) for row in rows})
    seeds = sorted({str(row["seed"]) for row in rows})
    case_counts = Counter(rng.choice(cases) for _ in cases)
    seed_counts = Counter(rng.choice(seeds) for _ in seeds)
    return [
        row
        for row in rows
        for _ in range(
            case_counts[str(row["problem_id"])]
            * seed_counts[str(row["seed"])]
        )
    ]


def _resolve_source_manifest_path(source_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("path is missing")
    posix_path = PurePosixPath(raw_path)
    if (
        posix_path.is_absolute()
        or Path(raw_path).is_absolute()
        or ".." in posix_path.parts
        or "\\" in raw_path
        or posix_path.as_posix() != raw_path
    ):
        raise ValueError("path must be a canonical relative POSIX path")
    root = source_root.resolve()
    resolved = (root / Path(*posix_path.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes source_root") from exc
    if not resolved.is_file():
        raise ValueError("path is not a file")
    return resolved


def _source_artifact_path(manifest_path: Path, artifact_name: str) -> Path:
    suffix = "hypergraph_manifest.json"
    if not manifest_path.name.endswith(suffix):
        raise ValueError("source manifest does not use the canonical filename")
    prefix = manifest_path.name[: -len(suffix)]
    return manifest_path.with_name(prefix + artifact_name)


def _source_manifest_index(
    aggregate_manifest: Mapping[str, object],
    *,
    source_root: Path,
    stage: str,
    config: Mapping[str, object],
    aggregate_rows: Mapping[
        str,
        Sequence[Mapping[str, str]],
    ]
    | None = None,
) -> tuple[dict[tuple[str, str], dict[str, object]], list[str]]:
    blockers: list[str] = []
    raw_manifests = aggregate_manifest.get("source_manifests")
    if not isinstance(raw_manifests, list):
        return {}, ["source_manifests_missing"]
    matrix = config["matrices"]["trace_screen" if stage == "screen" else "trace_full"]
    expected = {
        (str(problem_id), str(int(seed)))
        for problem_id in matrix["cases"]
        for seed in matrix["seeds"]
    }
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    reconstructed = {name: [] for name in SOURCE_ARTIFACT_SCHEMAS}
    target_fe = int(matrix["terminal_fe"])
    for payload in raw_manifests:
        if not isinstance(payload, dict):
            blockers.append("invalid_source_manifest_root")
            continue
        try:
            key = _trajectory_key(payload)
        except ValueError:
            blockers.append("invalid_source_manifest_identity")
            continue
        if key in indexed:
            blockers.append(f"duplicate_source_manifest:{key[0]}:seed{key[1]}")
            continue
        indexed[key] = payload
        prefix = f"{key[0]}:seed{key[1]}"
        try:
            manifest_path = _resolve_source_manifest_path(
                source_root,
                payload.get("path"),
            )
            expected_sha256 = payload.get("sha256")
            if not _hex64(expected_sha256) or _sha256(manifest_path) != expected_sha256:
                raise ValueError("file hash mismatch")
            source_payload = _read_json(manifest_path)
            embedded_payload = {
                field: value
                for field, value in payload.items()
                if field not in SOURCE_MANIFEST_WRAPPER_FIELDS
            }
            if embedded_payload != source_payload:
                raise ValueError("embedded payload mismatch")
            if payload.get("lane_id") != "hypergraph_v37_observer":
                raise ValueError("lane mismatch")
            artifact_hashes = source_payload.get("artifact_sha256")
            if not isinstance(artifact_hashes, dict):
                raise ValueError("source artifact hashes are missing")
            source_rows: dict[str, list[dict[str, str]]] = {}
            for artifact_name, fields in SOURCE_ARTIFACT_SCHEMAS.items():
                artifact_path = _source_artifact_path(manifest_path, artifact_name)
                if not artifact_path.is_file():
                    raise ValueError(f"missing source artifact {artifact_name}")
                expected_artifact_hash = artifact_hashes.get(
                    artifact_name,
                    artifact_hashes.get(artifact_path.name),
                )
                if (
                    not _hex64(expected_artifact_hash)
                    or _sha256(artifact_path) != expected_artifact_hash
                ):
                    raise ValueError(f"source artifact hash mismatch {artifact_name}")
                rows = _read_csv(artifact_path, fields)
                source_rows[artifact_name] = rows
                reconstructed[artifact_name].extend(rows)
            expected_counts = {
                "feature_row_count": len(
                    source_rows["hyperedge_cycle_features.csv"]
                ),
                "audit_row_count": len(source_rows["hyperedge_cycle_audit.csv"]),
                "shared_proposal_row_count": len(
                    source_rows["shared_proposal_audit.csv"]
                ),
                "outcome_row_count": len(
                    source_rows["hyperedge_cycle_outcomes.csv"]
                ),
            }
            if any(
                int(source_payload.get(field, -1)) != expected_count
                for field, expected_count in expected_counts.items()
            ):
                raise ValueError("source artifact row count mismatch")
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(f"source_manifest_provenance_failed:{prefix}:{exc}")
        try:
            hcc_status = str(payload["hcc_result_status"])
            hcc_max_fes = int(payload["hcc_result_max_fes"])
            hcc_actual_fes = int(payload["hcc_result_actual_fe_used"])
            terminal_target = int(payload["terminal_target_fe"])
            terminal_observed = int(payload["terminal_observed_fe"])
            terminal_tolerance = int(payload["terminal_completion_tolerance_fe"])
            if hcc_status != "completed":
                raise ValueError("HCC result status is not completed")
            if hcc_max_fes != target_fe or terminal_target != target_fe:
                raise ValueError("terminal target disagrees with preregistration")
            if hcc_actual_fes != terminal_observed:
                raise ValueError("terminal observed FE disagrees with HCC result")
            if not 1 <= terminal_tolerance <= target_fe:
                raise ValueError("terminal completion tolerance is invalid")
            if not target_fe - terminal_tolerance <= hcc_actual_fes <= target_fe:
                raise ValueError("terminal FE is outside the completion interval")
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"source_manifest_terminal_failed:{prefix}:{exc}")
        try:
            lower = float(payload["lower_bound"])
            upper = float(payload["upper_bound"])
            raw_hyperedges = payload["raw_hyperedges"]
            if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
                raise ValueError("invalid bounds")
            if not isinstance(raw_hyperedges, list):
                raise ValueError("raw_hyperedges must be a list")
            topology = build_overlap_hypergraph(raw_hyperedges)
            expected_topology_hash = _canonical_sha256(
                {
                    "hyperedges": topology.hyperedges,
                    "variable_owner_groups": topology.variable_owner_groups,
                }
            )
            owner_rows = [
                [variable, list(owners)]
                for variable, owners in topology.variable_owner_groups
            ]
            if payload.get("topology_sha256") != expected_topology_hash:
                raise ValueError("topology hash mismatch")
            if payload.get("variable_owner_groups") != owner_rows:
                raise ValueError("owner mapping mismatch")
            if int(payload.get("raw_group_count", -1)) != len(topology.hyperedges):
                raise ValueError("raw group count mismatch")
            if int(payload.get("eligible_hyperedge_count", -1)) != len(
                topology.eligible_group_indices
            ):
                raise ValueError("eligible group count mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"source_manifest_topology_failed:{prefix}:{exc}")
            continue
        expected_fields = {
            "protocol_version": PROTOCOL_VERSION,
            "hypergraph_trace_mode": "observer",
            "fresh_optimizer_execution": 1,
            "observer_status": "complete",
            "observer_integrity": 1,
            "observer_objective_calls": 0,
            "observer_rng_calls": 0,
            "observer_optimizer_calls": 0,
            "observer_fe": 0,
            "protocol_config_sha256": CONFIG_SHA256,
            "protocol_spec_sha256": SPEC_SHA256,
            "transitive_closure_used": 0,
        }
        for field, expected_value in expected_fields.items():
            if payload.get(field) != expected_value:
                blockers.append(f"source_manifest_{field}_failed:{prefix}")
        for field in (
            "runner_source_sha256",
            "rng_descriptor_sha256",
            "fitness_record_sha256",
        ):
            if not _hex64(payload.get(field, "")):
                blockers.append(f"source_manifest_{field}_failed:{prefix}")
        if payload.get("decision_status") not in {
            "pending_three_complete_sweeps",
            "applicable",
            "inapplicable",
        }:
            blockers.append(f"source_manifest_decision_status_failed:{prefix}")
        if payload.get("label_closure") not in {
            "closed",
            "terminal_censored",
            "not_applicable",
            "not_reached",
        }:
            blockers.append(f"source_manifest_label_closure_failed:{prefix}")
    if set(indexed) != expected:
        blockers.append("source_manifest_matrix_mismatch")
    if int(aggregate_manifest.get("source_manifest_count", -1)) != len(raw_manifests):
        blockers.append("source_manifest_count_mismatch")
    if aggregate_rows is not None:
        for artifact_name, expected_rows in reconstructed.items():
            observed_rows = [dict(row) for row in aggregate_rows.get(artifact_name, ())]
            if observed_rows != expected_rows:
                blockers.append(f"aggregate_source_rebuild_mismatch:{artifact_name}")
    return indexed, list(dict.fromkeys(blockers))


def _aggregate_manifest_blockers(
    manifest: Mapping[str, object],
    *,
    stage: str,
    matrix: Mapping[str, object],
) -> list[str]:
    blockers: list[str] = []
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        blockers.append("manifest_protocol_mismatch")
    if manifest.get("stage") != stage or manifest.get("status") != "pass":
        blockers.append("manifest_stage_or_integrity_failed")
    if manifest.get("hypergraph_trace_mode") != "observer":
        blockers.append("manifest_trace_mode_mismatch")
    if manifest.get("runtime_profile_authorized") is not False:
        blockers.append("manifest_runtime_profile_authorized")
    if manifest.get("runtime_model_bundle_allowed") is not False:
        blockers.append("manifest_runtime_model_bundle_allowed")
    if manifest.get("diagnostic_model_used") is not False:
        blockers.append("manifest_diagnostic_model_used")
    if manifest.get("source_bundle") != hypergraph_source_bundle():
        blockers.append("manifest_source_bundle_mismatch")
    try:
        current_commit = _git_commit()
        source_commit_matches = bool(
            _hex40(current_commit)
            and manifest.get("source_git_commit") == current_commit
        )
    except (OSError, RuntimeError, subprocess.SubprocessError):
        source_commit_matches = False
    if not source_commit_matches:
        blockers.append("manifest_source_git_commit_mismatch")
    static_ast_audit = hypergraph_static_ast_audit()
    if static_ast_audit.get("status") != "pass":
        blockers.append("current_policy_static_ast_failed")
    if manifest.get("static_ast_audit") != static_ast_audit:
        blockers.append("manifest_static_ast_audit_mismatch")
    if manifest.get("observer_calls") != {
        "objective": 0,
        "rng": 0,
        "optimizer": 0,
        "fe": 0,
    }:
        blockers.append("manifest_observer_calls_nonzero")
    if manifest.get("problem_ids") != matrix["cases"] or manifest.get(
        "seeds"
    ) != matrix["seeds"]:
        blockers.append("manifest_matrix_mismatch")
    try:
        terminal_fe_matches = int(manifest.get("max_fes", -1)) == int(
            matrix["terminal_fe"]
        )
    except (TypeError, ValueError):
        terminal_fe_matches = False
    if not terminal_fe_matches:
        blockers.append("manifest_terminal_fe_mismatch")
    config_entry = manifest.get("config")
    if not isinstance(config_entry, dict) or config_entry.get("sha256") != CONFIG_SHA256:
        blockers.append("manifest_config_hash_mismatch")
    spec_entry = manifest.get("spec")
    if not isinstance(spec_entry, dict) or spec_entry.get("sha256") != SPEC_SHA256:
        blockers.append("manifest_spec_hash_mismatch")
    return blockers


def _source_integrity(
    source: Path,
    *,
    stage: str,
    config: Mapping[str, object],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, object],
    list[str],
]:
    blockers: list[str] = []
    features = _read_csv(source / "hyperedge_cycle_features.csv", FEATURE_FIELDS)
    audits = _read_csv(source / "hyperedge_cycle_audit.csv", AUDIT_FIELDS)
    proposals = _read_csv(source / "shared_proposal_audit.csv", PROPOSAL_FIELDS)
    outcomes = _read_csv(source / "hyperedge_cycle_outcomes.csv", OUTCOME_FIELDS)
    manifest = _read_json(source / "hypergraph_trace_manifest.json")
    matrix = config["matrices"]["trace_screen" if stage == "screen" else "trace_full"]
    blockers.extend(
        _aggregate_manifest_blockers(manifest, stage=stage, matrix=matrix)
    )
    source_index, source_manifest_blockers = _source_manifest_index(
        manifest,
        source_root=source,
        stage=stage,
        config=config,
        aggregate_rows={
            "hyperedge_cycle_features.csv": features,
            "hyperedge_cycle_audit.csv": audits,
            "shared_proposal_audit.csv": proposals,
            "hyperedge_cycle_outcomes.csv": outcomes,
        },
    )
    blockers.extend(source_manifest_blockers)
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        blockers.append("manifest_artifact_hashes_missing")
    else:
        for name in (
            "hyperedge_cycle_features.csv",
            "hyperedge_cycle_audit.csv",
            "shared_proposal_audit.csv",
            "hyperedge_cycle_outcomes.csv",
        ):
            if artifact_hashes.get(name) != _sha256(source / name):
                blockers.append(f"artifact_hash_mismatch:{name}")

    ledger = _read_csv_any(source / "same_budget_ledger.csv")
    blockers.extend(
        _observer_ledger_blockers(
            ledger,
            source_index=source_index,
            matrix=matrix,
        )
    )
    aob_rows = _read_csv_any(source / "aob_input_manifest.csv")
    if not aob_rows or any(row.get("unchanged") != "1" for row in aob_rows):
        blockers.append("aob_input_integrity_failed")
    leakage_rows = _read_csv_any(source / "anti_leakage_audit.csv")
    if not leakage_rows or any(row.get("audit_status") != "pass" for row in leakage_rows):
        blockers.append("anti_leakage_failed")
    return features, audits, proposals, outcomes, manifest, blockers


def _observer_ledger_blockers(
    ledger: Sequence[Mapping[str, str]],
    *,
    source_index: Mapping[tuple[str, str], Mapping[str, object]],
    matrix: Mapping[str, object],
) -> list[str]:
    blockers: list[str] = []
    expected = {
        (str(problem_id), str(int(seed)))
        for problem_id in matrix["cases"]
        for seed in matrix["seeds"]
    }
    indexed: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in ledger:
        if row.get("lane_id") != "hypergraph_v37_observer":
            continue
        try:
            key = _trajectory_key(row)
        except ValueError:
            blockers.append("invalid_observer_ledger_identity")
            continue
        if key in indexed:
            blockers.append(f"duplicate_observer_ledger_route:{key[0]}:seed{key[1]}")
            continue
        indexed[key] = row
    if set(indexed) != expected or set(indexed) != set(source_index):
        blockers.append("observer_ledger_matrix_mismatch")

    for key in sorted(expected & set(indexed) & set(source_index)):
        prefix = f"{key[0]}:seed{key[1]}"
        row = indexed[key]
        source = source_index[key]
        try:
            actual_fe = int(row["actual_fe_used"])
            total_fe = int(row["total_fe"])
            budget_limit = int(row["budget_limit"])
            configured_budget_limit = int(row["configured_budget_limit"])
            hcc_max_fe = int(source["hcc_result_max_fes"])
            hcc_actual_fe = int(source["hcc_result_actual_fe_used"])
            target_fe = int(source["terminal_target_fe"])
            observed_fe = int(source["terminal_observed_fe"])
            tolerance_fe = int(source["terminal_completion_tolerance_fe"])
            if row.get("fresh_execution") != "1":
                raise ValueError("run is not fresh")
            if row.get("same_budget_violation") != "0":
                raise ValueError("same-budget violation")
            if actual_fe != hcc_actual_fe or actual_fe != observed_fe:
                raise ValueError("observed FE mismatch")
            if total_fe != actual_fe:
                raise ValueError("total FE mismatch")
            if (
                budget_limit != target_fe
                or configured_budget_limit != target_fe
                or hcc_max_fe != target_fe
            ):
                raise ValueError("terminal target mismatch")
            if not 1 <= tolerance_fe <= target_fe:
                raise ValueError("terminal tolerance is invalid")
            if not target_fe - tolerance_fe <= actual_fe <= target_fe:
                raise ValueError("terminal FE is outside the completion interval")
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"observer_ledger_integrity_failed:{prefix}:{exc}")
    return list(dict.fromkeys(blockers))


def _read_csv_any(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _contexts_from_source_manifests(
    source_manifests: Sequence[Mapping[str, object]],
) -> tuple[dict[tuple[str, str], dict[str, object]], list[str]]:
    contexts: dict[tuple[str, str], dict[str, object]] = {}
    blockers: list[str] = []
    for manifest in source_manifests:
        try:
            key = _trajectory_key(manifest)
            if key in contexts:
                raise ValueError("duplicate trajectory manifest")
            raw_hyperedges = manifest["raw_hyperedges"]
            if not isinstance(raw_hyperedges, list):
                raise ValueError("raw_hyperedges must be a list")
            topology = build_overlap_hypergraph(raw_hyperedges)
            lower = float(manifest["lower_bound"])
            upper = float(manifest["upper_bound"])
            if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
                raise ValueError("invalid bounds")
            contexts[key] = {
                "manifest": manifest,
                "topology": topology,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"invalid_source_context:{exc}")
    return contexts, blockers


def _proposal_credit(
    proposal_by_route: Mapping[tuple[str, str, int, int, int], Mapping[str, str]],
    *,
    problem_id: str,
    seed: str,
    sweep_index: int,
    group_index: int,
    shared_variables: Sequence[int],
    expected_resolution_fe: int,
    canonical_next_values: Mapping[int, float],
) -> tuple[float, float]:
    rows = [
        proposal_by_route[(problem_id, seed, sweep_index, group_index, variable)]
        for variable in shared_variables
    ]
    if any(not row.get("next_sweep_value") for row in rows):
        raise ValueError("proposal credit is not closed")
    if any(
        _int(row, "next_sweep_end_fe") != expected_resolution_fe for row in rows
    ):
        raise ValueError("proposal credit resolution FE mismatch")
    retained = directional_survival(
        anchor_values=tuple(_float(row, "anchor_value") for row in rows),
        candidate_values=tuple(_float(row, "proposed_value") for row in rows),
        next_sweep_values=tuple(
            float(canonical_next_values[variable]) for variable in shared_variables
        ),
    )
    return retained.survival, retained.overwrite


def _join_and_validate(
    features: Sequence[Mapping[str, str]],
    audits: Sequence[Mapping[str, str]],
    proposals: Sequence[Mapping[str, str]],
    outcomes: Sequence[Mapping[str, str]],
    *,
    source_manifests: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    blockers: list[str] = []
    contexts, context_blockers = _contexts_from_source_manifests(source_manifests)
    blockers.extend(context_blockers)

    audit_by_id: dict[str, Mapping[str, str]] = {}
    audit_by_route: dict[tuple[str, str, int, int], Mapping[str, str]] = {}
    recomputed_unit_fe: dict[tuple[str, str, int, int], float] = {}
    audits_by_sweep: dict[
        tuple[str, str, int], list[Mapping[str, str]]
    ] = defaultdict(list)
    for row in audits:
        decision_id = row.get("decision_id", "")
        try:
            problem_id, seed = _trajectory_key(row)
            sweep = _int(row, "sweep_index")
            group = _int(row, "group_index")
            context = contexts[(problem_id, seed)]
            topology = context["topology"]
            if group >= len(topology.hyperedges):
                raise ValueError("group is outside raw topology")
            route = (problem_id, seed, sweep, group)
            if not decision_id or decision_id in audit_by_id or route in audit_by_route:
                raise ValueError("duplicate or empty audit route")
            audit_by_id[decision_id] = row
            audit_by_route[route] = row
            audits_by_sweep[(problem_id, seed, sweep)].append(row)
            if row.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("protocol mismatch")
            if row.get("topology_sha256") != context["manifest"].get(
                "topology_sha256"
            ):
                raise ValueError("topology hash mismatch")
            if row.get("rng_descriptor_sha256") != context["manifest"].get(
                "rng_descriptor_sha256"
            ):
                raise ValueError("RNG descriptor mismatch")
            for field in (
                "cohort_locked",
                "state_complete",
                "unique_focal",
                "applicable",
                "all_raw_groups_completed",
                "native_sweep_end_completed",
                "watermark_valid",
                "observer_integrity",
            ):
                _flag(row, field)
            if not _flag(row, "observer_integrity"):
                raise ValueError("observer integrity failed")
            placeholder = not row.get("full_interval_start_fe")
            if placeholder:
                if not (
                    row.get("not_applicable_reason") == "incomplete_native_sweep"
                    and not _flag(row, "state_complete")
                    and not _flag(row, "applicable")
                    and not _flag(row, "watermark_valid")
                    and all(
                        not row.get(field, "")
                        for field in (
                            "source_end_fe",
                            "decision_fe",
                            "full_interval_end_fe",
                            "primary_requested_fe",
                            "primary_actual_fe",
                            "full_interval_actual_fe",
                            "pre_error",
                            "best_error",
                            "successful",
                            "unit_fe_contribution",
                            "feature_sha256",
                            "fitness_record_sha256",
                            "proposal_capture_watermark",
                        )
                    )
                ):
                    raise ValueError("invalid incomplete-sweep placeholder")
                continue

            start = _int(row, "full_interval_start_fe")
            end = _int(row, "full_interval_end_fe")
            actual = _int(row, "full_interval_actual_fe", 1)
            requested = _int(row, "primary_requested_fe", 1)
            primary_actual = _int(row, "primary_actual_fe")
            pre_error = _float(row, "pre_error")
            best_error = _float(row, "best_error")
            if end - start != actual or primary_actual > requested:
                raise ValueError("invalid group FE interval")
            if pre_error < 0.0 or best_error < 0.0:
                raise ValueError("negative objective error")
            expected_u = unit_fe_contribution(
                pre_error=pre_error,
                best_error=best_error,
                actual_fe=actual,
            )
            if row.get("unit_fe_contribution") != _format_float(expected_u):
                raise ValueError("unit-FE contribution encoding mismatch")
            if _flag(row, "successful") != (best_error < pre_error):
                raise ValueError("success flag mismatch")
            if row.get("proposal_capture_watermark") != (
                "after_group_local_rescue_recovery_before_relation_writeback"
            ):
                raise ValueError("proposal capture watermark mismatch")
            if not _flag(row, "watermark_valid"):
                raise ValueError("observed group watermark failed")
            if _flag(row, "native_sweep_end_completed") and not _flag(
                row, "all_raw_groups_completed"
            ):
                raise ValueError("native closure without all raw groups")
            if _flag(row, "all_raw_groups_completed") and _flag(
                row, "native_sweep_end_completed"
            ):
                decision_fe = _int(row, "decision_fe")
                if (
                    _int(row, "source_end_fe") != decision_fe
                    or end > decision_fe
                    or row.get("native_sweep_end_stage")
                    != HYPERGRAPH_NATIVE_SWEEP_END_STAGE
                    or not _hex64(row.get("fitness_record_sha256", ""))
                ):
                    raise ValueError("complete-sweep watermark mismatch")
            elif row.get("decision_fe") or row.get("fitness_record_sha256"):
                raise ValueError("incomplete sweep cannot expose decision state")
            if row.get("feature_sha256") and not _hex64(row["feature_sha256"]):
                raise ValueError("invalid feature hash")
            recomputed_unit_fe[route] = expected_u
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"audit_integrity_failed:{decision_id}:{exc}")

    complete_sweeps: set[tuple[str, str, int]] = set()
    complete_sweep_decision_fe: dict[tuple[str, str, int], int] = {}
    for sweep_route, rows in audits_by_sweep.items():
        context = contexts.get(sweep_route[:2])
        if context is None:
            continue
        topology = context["topology"]
        expected_groups = tuple(range(len(topology.hyperedges)))
        observed_groups = tuple(_int(row, "group_index") for row in rows)
        completed = [
            _flag(row, "all_raw_groups_completed")
            and _flag(row, "native_sweep_end_completed")
            for row in rows
        ]
        if any(completed):
            valid = True
            if not all(completed) or set(observed_groups) != set(expected_groups):
                blockers.append(
                    f"incomplete_raw_group_coverage:{sweep_route[0]}:"
                    f"seed{sweep_route[1]}:sweep{sweep_route[2]}"
                )
                valid = False
            if observed_groups != expected_groups:
                blockers.append(
                    f"noncanonical_raw_group_order:{sweep_route[0]}:"
                    f"seed{sweep_route[1]}:sweep{sweep_route[2]}"
                )
                valid = False
            try:
                decision_fes = {_int(row, "decision_fe") for row in rows}
                fitness_hashes = {row.get("fitness_record_sha256", "") for row in rows}
                if len(decision_fes) != 1:
                    blockers.append(
                        f"complete_sweep_decision_fe_mismatch:{sweep_route[0]}:"
                        f"seed{sweep_route[1]}:sweep{sweep_route[2]}"
                    )
                    valid = False
                if len(fitness_hashes) != 1:
                    blockers.append(
                        f"complete_sweep_fitness_hash_mismatch:{sweep_route[0]}:"
                        f"seed{sweep_route[1]}:sweep{sweep_route[2]}"
                    )
                    valid = False
                ordered_rows = sorted(rows, key=lambda row: _int(row, "group_index"))
                previous_end: int | None = None
                for row in ordered_rows:
                    start = _int(row, "full_interval_start_fe")
                    end = _int(row, "full_interval_end_fe")
                    if previous_end is not None and start < previous_end:
                        blockers.append(
                            f"complete_sweep_fe_interval_overlap:{sweep_route[0]}:"
                            f"seed{sweep_route[1]}:sweep{sweep_route[2]}"
                        )
                        valid = False
                        break
                    previous_end = end
            except (KeyError, TypeError, ValueError) as exc:
                blockers.append(
                    f"complete_sweep_structure_failed:{sweep_route[0]}:"
                    f"seed{sweep_route[1]}:sweep{sweep_route[2]}:{exc}"
                )
                valid = False
            if valid:
                complete_sweeps.add(sweep_route)
                complete_sweep_decision_fe[sweep_route] = next(iter(decision_fes))
        elif set(observed_groups) != set(expected_groups):
            blockers.append(
                f"incomplete_sweep_group_coverage:{sweep_route[0]}:"
                f"seed{sweep_route[1]}:sweep{sweep_route[2]}"
            )

    proposal_by_route: dict[
        tuple[str, str, int, int, int], Mapping[str, str]
    ] = {}
    for row in proposals:
        decision_id = row.get("decision_id", "")
        try:
            problem_id, seed = _trajectory_key(row)
            sweep = _int(row, "sweep_index")
            group = _int(row, "group_index")
            variable = _int(row, "variable_index")
            route = (problem_id, seed, sweep, group, variable)
            if route in proposal_by_route:
                raise ValueError("duplicate proposal route")
            proposal_by_route[route] = row
            audit = audit_by_route[(problem_id, seed, sweep, group)]
            context = contexts[(problem_id, seed)]
            topology = context["topology"]
            if variable not in topology.shared_for_group(group):
                raise ValueError("proposal variable is not shared by owner group")
            if decision_id != audit.get("decision_id"):
                raise ValueError("proposal decision route mismatch")
            if row.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("proposal protocol mismatch")
            if row.get("capture_watermark") != (
                "after_group_local_rescue_recovery_before_relation_writeback"
            ):
                raise ValueError("proposal capture watermark mismatch")
            if not _flag(row, "observer_integrity"):
                raise ValueError("proposal observer integrity failed")
            if row.get("topology_sha256") != context["manifest"].get(
                "topology_sha256"
            ):
                raise ValueError("proposal topology mismatch")
            if (
                _int(row, "proposal_source_end_fe")
                != _int(audit, "full_interval_end_fe")
                or _int(row, "sweep_end_fe") != _int(audit, "decision_fe")
            ):
                raise ValueError("proposal FE watermark mismatch")
            for field in (
                "anchor_value",
                "proposed_value",
                "sweep_end_value",
            ):
                _float(row, field)
            next_value_present = bool(row.get("next_sweep_value"))
            next_fe_present = bool(row.get("next_sweep_end_fe"))
            if next_value_present != next_fe_present:
                raise ValueError("partial next-sweep proposal closure")
            if next_value_present:
                _float(row, "next_sweep_value")
                if _int(row, "next_sweep_end_fe") <= _int(row, "sweep_end_fe"):
                    raise ValueError("proposal resolution FE did not advance")
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"proposal_integrity_failed:{decision_id}:{exc}")

    for problem_id, seed, sweep in complete_sweeps:
        topology = contexts[(problem_id, seed)]["topology"]
        for group in range(len(topology.hyperedges)):
            expected_variables = set(topology.shared_for_group(group))
            observed_variables = {
                variable
                for route_problem, route_seed, route_sweep, route_group, variable
                in proposal_by_route
                if (
                    route_problem,
                    route_seed,
                    route_sweep,
                    route_group,
                )
                == (problem_id, seed, sweep, group)
            }
            if observed_variables != expected_variables:
                blockers.append(
                    f"proposal_coverage_failed:{problem_id}:seed{seed}:"
                    f"sweep{sweep}:group{group}"
                )

    endpoint_values: dict[tuple[str, str, int, int], set[float]] = defaultdict(set)
    for route, row in proposal_by_route.items():
        endpoint_values[(route[0], route[1], route[2], route[4])].add(
            _float(row, "sweep_end_value")
        )
    for route, values in endpoint_values.items():
        if len(values) != 1:
            blockers.append(
                f"sweep_endpoint_disagreement:{route[0]}:seed{route[1]}:"
                f"sweep{route[2]}:variable{route[3]}"
            )

    endpoint_by_sweep_variable = {
        route: next(iter(values))
        for route, values in endpoint_values.items()
        if len(values) == 1
    }
    for route, row in proposal_by_route.items():
        problem_id, seed, sweep, _, variable = route
        next_sweep = (problem_id, seed, sweep + 1)
        next_value_present = bool(row.get("next_sweep_value"))
        if next_sweep not in complete_sweeps:
            if next_value_present or row.get("next_sweep_end_fe"):
                blockers.append(
                    f"unexpected_proposal_next_sweep_closure:{problem_id}:"
                    f"seed{seed}:sweep{sweep}:variable{variable}"
                )
            continue
        endpoint_key = (problem_id, seed, sweep + 1, variable)
        expected_endpoint = endpoint_by_sweep_variable.get(endpoint_key)
        if not next_value_present or expected_endpoint is None:
            blockers.append(
                f"missing_proposal_next_sweep_endpoint:{problem_id}:"
                f"seed{seed}:sweep{sweep}:variable{variable}"
            )
            continue
        try:
            if row.get("next_sweep_value") != _format_float(expected_endpoint):
                blockers.append(
                    f"proposal_next_sweep_endpoint_mismatch:{problem_id}:"
                    f"seed{seed}:sweep{sweep}:variable{variable}"
                )
            if _int(row, "next_sweep_end_fe") != complete_sweep_decision_fe[
                next_sweep
            ]:
                blockers.append(
                    f"proposal_next_sweep_fe_mismatch:{problem_id}:"
                    f"seed{seed}:sweep{sweep}:variable{variable}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(
                f"proposal_next_sweep_crosscheck_failed:{problem_id}:"
                f"seed{seed}:sweep{sweep}:variable{variable}:{exc}"
            )

    feature_by_id: dict[str, Mapping[str, str]] = {}
    features_by_trajectory: dict[
        tuple[str, str], list[Mapping[str, str]]
    ] = defaultdict(list)
    for row in features:
        decision_id = row.get("decision_id", "")
        try:
            if not decision_id or decision_id in feature_by_id:
                raise ValueError("duplicate or empty feature decision id")
            feature_by_id[decision_id] = row
            audit = audit_by_id[decision_id]
            key = _trajectory_key(audit)
            topology = contexts[key]["topology"]
            group = _int(audit, "group_index")
            if group not in topology.eligible_group_indices:
                raise ValueError("feature belongs to noneligible raw group")
            if not _flag(audit, "cohort_locked") or not _flag(
                audit, "state_complete"
            ):
                raise ValueError("feature does not belong to complete locked cohort")
            numeric_payload = {field: row[field] for field in FEATURE_FIELDS[1:]}
            if audit.get("feature_sha256") != _canonical_sha256(numeric_payload):
                raise ValueError("feature hash mismatch")
            values = {field: _float(row, field) for field in (*STATE_FIELDS, *SCORE_FIELDS)}
            if any(values[field] < 0.0 for field in STATE_FIELDS[:2]) or any(
                not 0.0 <= values[field] <= 1.0
                for field in (*STATE_FIELDS[2:], *SCORE_FIELDS)
            ):
                raise ValueError("feature outside frozen range")
            features_by_trajectory[key].append(row)
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"feature_integrity_failed:{decision_id}:{exc}")

    cohort_sweep_by_trajectory: dict[tuple[str, str], int] = {}
    expected_decision_status: dict[tuple[str, str], str] = {}
    expected_decision_reason: dict[tuple[str, str], str] = {}
    for key, context in contexts.items():
        cohort_rows = [
            row
            for (problem_id, seed, _, _), row in audit_by_route.items()
            if (problem_id, seed) == key and row.get("cohort_locked") == "1"
        ]
        manifest = context["manifest"]
        topology = context["topology"]
        observed_sweeps = sorted(
            sweep
            for problem_id, seed, sweep in audits_by_sweep
            if (problem_id, seed) == key
        )
        snapshot_sweep = next(
            (
                sweep
                for sweep in observed_sweeps
                if (key[0], key[1], sweep - 2) in complete_sweeps
                and (key[0], key[1], sweep - 1) in complete_sweeps
            ),
            None,
        )
        expected_lock_consumed = snapshot_sweep is not None
        try:
            actual_lock_consumed = int(
                manifest.get("decision_lock_consumed", 0)
            ) == 1
        except (TypeError, ValueError):
            actual_lock_consumed = False
        if actual_lock_consumed != expected_lock_consumed:
            blockers.append(f"decision_lock_recompute_mismatch:{key[0]}:seed{key[1]}")
        actual_snapshot = manifest.get("decision_snapshot_sweep")
        try:
            normalized_actual_snapshot = (
                None if actual_snapshot is None else int(actual_snapshot)
            )
        except (TypeError, ValueError):
            normalized_actual_snapshot = -1
        if normalized_actual_snapshot != snapshot_sweep or manifest.get(
            "cohort_locked_sweep"
        ) != snapshot_sweep:
            blockers.append(f"decision_snapshot_not_earliest:{key[0]}:seed{key[1]}")

        if snapshot_sweep is None:
            expected_decision_status[key] = "pending_three_complete_sweeps"
            expected_decision_reason[key] = ""
            if cohort_rows or features_by_trajectory.get(key):
                blockers.append(f"unexpected_decision_cohort:{key[0]}:seed{key[1]}")
            if manifest.get("decision_status") != expected_decision_status[key]:
                blockers.append(f"decision_status_mismatch:{key[0]}:seed{key[1]}")
            if manifest.get("decision_reason") != expected_decision_reason[key]:
                blockers.append(f"decision_reason_mismatch:{key[0]}:seed{key[1]}")
            continue

        cohort_sweep_by_trajectory[key] = snapshot_sweep
        cohort_sweeps = {_int(row, "sweep_index") for row in cohort_rows}
        if cohort_sweeps != {snapshot_sweep} or {
            _int(row, "group_index") for row in cohort_rows
        } != set(range(len(topology.hyperedges))):
            blockers.append(f"invalid_one_shot_cohort:{key[0]}:seed{key[1]}")
        feature_rows = features_by_trajectory.get(key, [])
        feature_groups = {
            _int(audit_by_id[row["decision_id"]], "group_index")
            for row in feature_rows
        }
        state_complete = (key[0], key[1], snapshot_sweep) in complete_sweeps
        if any(_flag(row, "state_complete") != state_complete for row in cohort_rows):
            blockers.append(f"cohort_state_complete_mismatch:{key[0]}:seed{key[1]}")
        expected_feature_groups = (
            set(topology.eligible_group_indices)
            if state_complete and len(topology.eligible_group_indices) >= 2
            else set()
        )
        if feature_groups != expected_feature_groups:
            blockers.append(f"feature_group_coverage_failed:{key[0]}:seed{key[1]}")
        if int(manifest.get("decision_feature_row_count", -1)) != len(feature_rows):
            blockers.append(f"manifest_feature_count_failed:{key[0]}:seed{key[1]}")
        if not state_complete:
            expected_decision_status[key] = "inapplicable"
            expected_decision_reason[key] = "incomplete_native_sweep"
        elif len(topology.eligible_group_indices) < 2:
            expected_decision_status[key] = "inapplicable"
            expected_decision_reason[key] = (
                "no_shared_hyperedge"
                if not topology.eligible_group_indices
                else "insufficient_comparison_hyperedges"
            )
        else:
            continue
        if manifest.get("decision_status") != expected_decision_status[key]:
            blockers.append(f"decision_status_mismatch:{key[0]}:seed{key[1]}")
        if manifest.get("decision_reason") != expected_decision_reason[key]:
            blockers.append(f"decision_reason_mismatch:{key[0]}:seed{key[1]}")
        for row in cohort_rows:
            if (
                _flag(row, "unique_focal")
                or _flag(row, "applicable")
                or row.get("not_applicable_reason") != expected_decision_reason[key]
            ):
                blockers.append(f"cohort_route_mismatch:{row['decision_id']}")

    raw_states_by_trajectory: dict[
        tuple[str, str], tuple[HyperedgeCycleState, ...]
    ] = {}
    raw_scores_by_trajectory: dict[
        tuple[str, str], tuple[HyperedgeScore, ...]
    ] = {}
    for key, feature_rows in features_by_trajectory.items():
        context = contexts[key]
        topology = context["topology"]
        sweep = cohort_sweep_by_trajectory.get(key)
        if sweep is None:
            continue
        history_sweeps = (sweep - 2, sweep - 1, sweep)
        if any((key[0], key[1], item) not in complete_sweeps for item in history_sweeps):
            blockers.append(f"feature_history_incomplete:{key[0]}:seed{key[1]}")
            continue
        decision_fe = complete_sweep_decision_fe[(key[0], key[1], sweep)]
        states: list[HyperedgeCycleState] = []
        try:
            for group in topology.eligible_group_indices:
                history = [
                    audit_by_route[(key[0], key[1], history_sweep, group)]
                    for history_sweep in history_sweeps
                ]
                contributions = [
                    recomputed_unit_fe[
                        (
                            key[0],
                            key[1],
                            history_sweep,
                            group,
                        )
                    ]
                    for history_sweep in history_sweeps
                ]
                ewma = contributions[0]
                for contribution in contributions[1:]:
                    ewma = 0.5 * contribution + 0.5 * ewma
                success_ratio = statistics.fmean(
                    float(_flag(row, "successful")) for row in history
                )
                trailing_zero = 0
                for contribution in reversed(contributions):
                    if contribution > 0.0:
                        break
                    trailing_zero += 1
                disagreements: list[float] = []
                for variable in topology.shared_for_group(group):
                    owners = topology.star_for_variable(variable).owner_group_indices
                    owner_values = [
                        _float(
                            proposal_by_route[
                                (key[0], key[1], sweep, owner, variable)
                            ],
                            "proposed_value",
                        )
                        for owner in owners
                    ]
                    disagreements.append(
                        min(
                            1.0,
                            max(
                                0.0,
                                (max(owner_values) - min(owner_values))
                                / (context["upper_bound"] - context["lower_bound"]),
                            ),
                        )
                    )
                _, prior_overwrite = _proposal_credit(
                    proposal_by_route,
                    problem_id=key[0],
                    seed=key[1],
                    sweep_index=sweep - 1,
                    group_index=group,
                    shared_variables=topology.shared_for_group(group),
                    expected_resolution_fe=decision_fe,
                    canonical_next_values={
                        variable: endpoint_by_sweep_variable[
                            (key[0], key[1], sweep, variable)
                        ]
                        for variable in topology.shared_for_group(group)
                    },
                )
                states.append(
                    HyperedgeCycleState(
                        current_unit_fe_contribution=contributions[-1],
                        ewma_unit_fe_contribution_3=ewma,
                        zero_gain_difficulty=1.0 - success_ratio,
                        stagnation_ratio_3=min(trailing_zero, 3) / 3.0,
                        direct_owner_proposal_disagreement=statistics.fmean(
                            disagreements
                        ),
                        prior_next_sweep_overwrite=prior_overwrite,
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"raw_state_recompute_failed:{key[0]}:seed{key[1]}:{exc}")
            continue
        raw_states = tuple(states)
        raw_states_by_trajectory[key] = raw_states
        expected_scores = score_hyperedge_states(raw_states)
        raw_scores_by_trajectory[key] = expected_scores
        row_by_group = {
            _int(audit_by_id[row["decision_id"]], "group_index"): row
            for row in feature_rows
        }
        for group, state, score in zip(
            topology.eligible_group_indices,
            raw_states,
            expected_scores,
            strict=True,
        ):
            row = row_by_group.get(group)
            if row is None:
                continue
            for field in STATE_FIELDS:
                if not _same_float(_float(row, field), float(getattr(state, field))):
                    blockers.append(
                        f"raw_feature_mismatch:{row['decision_id']}:{field}"
                    )
            for field in SCORE_FIELDS:
                if not _same_float(_float(row, field), float(getattr(score, field))):
                    blockers.append(
                        f"fixed_score_mismatch:{row['decision_id']}:{field}"
                    )
            expected_payload = _raw_feature_payload(state, score)
            expected_hash = _canonical_sha256(expected_payload)
            audit = audit_by_id[row["decision_id"]]
            if audit.get("feature_sha256") != expected_hash:
                blockers.append(
                    f"raw_feature_hash_mismatch:{row['decision_id']}"
                )
            if {field: row.get(field) for field in FEATURE_FIELDS[1:]} != expected_payload:
                blockers.append(
                    f"raw_feature_encoding_mismatch:{row['decision_id']}"
                )
        highest = max(score.focal_priority for score in expected_scores)
        unique_focal = sum(
            score.focal_priority == highest for score in expected_scores
        ) == 1
        expected_decision_status[key] = (
            "applicable" if unique_focal else "inapplicable"
        )
        expected_decision_reason[key] = "" if unique_focal else "focal_priority_tie"
        cohort_rows = [
            row
            for (problem_id, seed, row_sweep, _), row in audit_by_route.items()
            if (problem_id, seed, row_sweep) == (key[0], key[1], sweep)
        ]
        for audit in cohort_rows:
            group = _int(audit, "group_index")
            if _flag(audit, "unique_focal") != unique_focal:
                blockers.append(f"unique_focal_mismatch:{audit['decision_id']}")
            expected_applicable = unique_focal and group in topology.eligible_group_indices
            if _flag(audit, "applicable") != expected_applicable:
                blockers.append(f"applicable_route_mismatch:{audit['decision_id']}")
            expected_reason = (
                expected_decision_reason[key]
                if group in topology.eligible_group_indices
                else "no_shared_variables"
            )
            if audit.get("not_applicable_reason") != expected_reason:
                blockers.append(f"cohort_reason_mismatch:{audit['decision_id']}")
        if context["manifest"].get("decision_status") != expected_decision_status[key]:
            blockers.append(f"decision_status_mismatch:{key[0]}:seed{key[1]}")
        if context["manifest"].get("decision_reason") != expected_decision_reason[key]:
            blockers.append(f"decision_reason_mismatch:{key[0]}:seed{key[1]}")

    outcome_by_id: dict[str, Mapping[str, str]] = {}
    for row in outcomes:
        decision_id = row.get("decision_id", "")
        if not decision_id or decision_id in outcome_by_id:
            blockers.append("duplicate_or_empty_outcome_decision_id")
            continue
        outcome_by_id[decision_id] = row
    if set(outcome_by_id) != set(feature_by_id):
        blockers.append("feature_outcome_id_mismatch")

    completed_outcome_ids: set[str] = set()
    censored_outcome_ids: set[str] = set()
    recomputed_outcomes: dict[str, dict[str, float]] = {}
    for decision_id, feature in feature_by_id.items():
        outcome = outcome_by_id.get(decision_id)
        audit = audit_by_id.get(decision_id)
        if outcome is None or audit is None:
            continue
        try:
            key = _trajectory_key(audit)
            sweep = _int(audit, "sweep_index")
            group = _int(audit, "group_index")
            if (
                outcome.get("problem_id") != key[0]
                or str(int(outcome.get("seed", ""))) != key[1]
                or _int(outcome, "sweep_index") != sweep
            ):
                raise ValueError("outcome identity mismatch")
            complete = _flag(outcome, "outcome_complete")
            censored = _flag(outcome, "terminal_censored")
            if complete:
                if censored:
                    raise ValueError("complete outcome cannot be censored")
                resolution_sweep = _int(outcome, "resolution_sweep_index")
                resolution_fe = _int(outcome, "resolution_end_fe")
                if (
                    resolution_sweep != sweep + 1
                    or (key[0], key[1], resolution_sweep) not in complete_sweeps
                    or not _flag(outcome, "all_groups_completed")
                    or not _flag(outcome, "native_sweep_end_completed")
                ):
                    raise ValueError("outcome closure mismatch")
                next_audit = audit_by_route[
                    (key[0], key[1], resolution_sweep, group)
                ]
                if resolution_fe != _int(next_audit, "decision_fe"):
                    raise ValueError("outcome resolution FE mismatch")
                next_contribution = recomputed_unit_fe[
                    (key[0], key[1], resolution_sweep, group)
                ]
                if next_contribution < 0.0:
                    raise ValueError("negative next contribution")
                survival, overwrite = _proposal_credit(
                    proposal_by_route,
                    problem_id=key[0],
                    seed=key[1],
                    sweep_index=sweep,
                    group_index=group,
                    shared_variables=contexts[key]["topology"].shared_for_group(group),
                    expected_resolution_fe=resolution_fe,
                    canonical_next_values={
                        variable: endpoint_by_sweep_variable[
                            (key[0], key[1], resolution_sweep, variable)
                        ]
                        for variable in contexts[key]["topology"].shared_for_group(
                            group
                        )
                    },
                )
                expected_labels = {
                    "next_sweep_unit_fe_contribution": _format_float(
                        next_contribution
                    ),
                    "next_sweep_survival": _format_float(survival),
                    "next_sweep_overwrite": _format_float(overwrite),
                }
                if any(
                    outcome.get(field) != expected_value
                    for field, expected_value in expected_labels.items()
                ):
                    raise ValueError("raw outcome encoding mismatch")
                recomputed_outcomes[decision_id] = {
                    "next_sweep_unit_fe_contribution": next_contribution,
                    "next_sweep_survival": survival,
                    "next_sweep_overwrite": overwrite,
                }
                completed_outcome_ids.add(decision_id)
            else:
                if not censored:
                    raise ValueError("incomplete outcome must be censored")
                if _flag(outcome, "all_groups_completed") or _flag(
                    outcome, "native_sweep_end_completed"
                ):
                    raise ValueError("censored outcome cannot claim closure")
                if any(
                    outcome.get(field, "")
                    for field in (
                        "resolution_sweep_index",
                        "next_sweep_unit_fe_contribution",
                        "next_sweep_survival",
                        "next_sweep_overwrite",
                        "resolution_end_fe",
                    )
                ):
                    raise ValueError("censored outcome leaked a label")
                final_complete_sweep = max(
                    route_sweep
                    for route_problem, route_seed, route_sweep in complete_sweeps
                    if (route_problem, route_seed) == key
                )
                if sweep != final_complete_sweep:
                    raise ValueError("only terminal cohort may be censored")
                censored_outcome_ids.add(decision_id)
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"outcome_integrity_failed:{decision_id}:{exc}")

    for key, context in contexts.items():
        feature_ids = {
            row["decision_id"] for row in features_by_trajectory.get(key, [])
        }
        if key not in cohort_sweep_by_trajectory:
            expected_label_closure = "not_reached"
        elif not feature_ids:
            expected_label_closure = "not_applicable"
        elif feature_ids <= completed_outcome_ids:
            expected_label_closure = "closed"
        elif feature_ids <= censored_outcome_ids:
            expected_label_closure = "terminal_censored"
        else:
            expected_label_closure = "invalid"
            blockers.append(f"label_closure_incomplete:{key[0]}:seed{key[1]}")
        if context["manifest"].get("label_closure") != expected_label_closure:
            blockers.append(f"label_closure_mismatch:{key[0]}:seed{key[1]}")

    joined: list[dict[str, object]] = []
    runtime_applicable_trajectories: set[tuple[str, str]] = set()
    closed_applicable_trajectories: set[tuple[str, str]] = set()
    for key, feature_rows in features_by_trajectory.items():
        audits_for_features = [audit_by_id[row["decision_id"]] for row in feature_rows]
        unique = bool(audits_for_features) and all(
            _flag(row, "unique_focal") and _flag(row, "applicable")
            for row in audits_for_features
        )
        if not unique or len(feature_rows) < 2:
            continue
        runtime_applicable_trajectories.add(key)
        ids = {row["decision_id"] for row in feature_rows}
        if (
            not ids
            or not ids <= completed_outcome_ids
            or key not in raw_states_by_trajectory
            or key not in raw_scores_by_trajectory
        ):
            continue
        closed_applicable_trajectories.add(key)
        topology = contexts[key]["topology"]
        recomputed_by_group = {
            group: (state, score)
            for group, state, score in zip(
                topology.eligible_group_indices,
                raw_states_by_trajectory[key],
                raw_scores_by_trajectory[key],
                strict=True,
            )
        }
        highest_priority = max(
            score.focal_priority for _, score in recomputed_by_group.values()
        )
        for feature in feature_rows:
            decision_id = feature["decision_id"]
            audit = audit_by_id[decision_id]
            outcome = recomputed_outcomes[decision_id]
            state, score = recomputed_by_group[_int(audit, "group_index")]
            joined.append(
                {
                    "decision_id": decision_id,
                    "problem_id": key[0],
                    "seed": int(key[1]),
                    "sweep_index": _int(audit, "sweep_index"),
                    "selected_focal": int(score.focal_priority == highest_priority),
                    **{
                        field: float(getattr(state, field)) for field in STATE_FIELDS
                    },
                    **{
                        field: float(getattr(score, field)) for field in SCORE_FIELDS
                    },
                    "next_sweep_unit_fe_contribution": outcome[
                        "next_sweep_unit_fe_contribution"
                    ],
                    "next_sweep_survival": outcome["next_sweep_survival"],
                    "next_sweep_overwrite": outcome["next_sweep_overwrite"],
                }
            )

    closure_fraction = (
        len(closed_applicable_trajectories) / len(runtime_applicable_trajectories)
        if runtime_applicable_trajectories
        else 0.0
    )
    locked_overlap_trajectories = {
        key
        for key in cohort_sweep_by_trajectory
        if contexts[key]["topology"].eligible_group_indices
    }
    missing_state_trajectories = {
        key
        for key in locked_overlap_trajectories
        if key not in raw_states_by_trajectory
    }
    missing_state_fraction = (
        len(missing_state_trajectories) / len(locked_overlap_trajectories)
        if locked_overlap_trajectories
        else 1.0
    )
    coverage = {
        "applicable_rows": sum(
            len(features_by_trajectory[key])
            for key in runtime_applicable_trajectories
        ),
        "labeled_applicable_rows": len(joined),
        "runtime_applicable_trajectories": len(runtime_applicable_trajectories),
        "applicable_trajectories": len(runtime_applicable_trajectories),
        "labeled_applicable_trajectories": len(closed_applicable_trajectories),
        "applicable_cases": len({key[0] for key in runtime_applicable_trajectories}),
        "applicable_seeds": len({key[1] for key in runtime_applicable_trajectories}),
        "complete_next_sweep_label_fraction": closure_fraction,
        "required_state_missing_fraction": missing_state_fraction,
        "required_state_denominator_trajectories": len(
            locked_overlap_trajectories
        ),
        "required_state_missing_trajectories": len(missing_state_trajectories),
        "terminal_censored_trajectories": len(
            {
                _trajectory_key(audit_by_id[decision_id])
                for decision_id in censored_outcome_ids
            }
        ),
    }
    return joined, coverage, list(dict.fromkeys(blockers))


def _trajectory_summaries(
    joined: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    blockers: list[str] = []
    for row in joined:
        try:
            grouped[_trajectory_key(row)].append(row)
        except ValueError as exc:
            blockers.append(f"trajectory_summary_identity_failed:{exc}")
    summaries: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        selected = [row for row in rows if int(row.get("selected_focal", 0)) == 1]
        if len(rows) < 2 or len(selected) != 1:
            blockers.append(f"trajectory_summary_scope_failed:{key[0]}:seed{key[1]}")
            continue
        priority = [float(row["focal_priority"]) for row in rows]
        next_gain = [float(row["next_sweep_unit_fe_contribution"]) for row in rows]
        reliability = [float(row["owner_reliability"]) for row in rows]
        survival = [float(row["next_sweep_survival"]) for row in rows]
        priority_rho = _spearman(priority, next_gain)
        owner_rho = _spearman(reliability, survival)
        gain_ranks = midrank_percentiles(next_gain)
        selected_index = rows.index(selected[0])
        nonfocal_gain = [
            value for index, value in enumerate(next_gain) if index != selected_index
        ]
        summaries.append(
            {
                "problem_id": key[0],
                "seed": int(key[1]),
                "sweep_index": int(rows[0]["sweep_index"]),
                "row_count": len(rows),
                "trajectory_priority_spearman": (
                    0.0 if priority_rho is None else priority_rho
                ),
                "trajectory_focal_rank_advantage": gain_ranks[selected_index] - 0.5,
                "trajectory_owner_survival_spearman": (
                    0.0 if owner_rho is None else owner_rho
                ),
                "raw_focal_minus_nonfocal_gain": (
                    next_gain[selected_index] - statistics.fmean(nonfocal_gain)
                ),
            }
        )
    return summaries, blockers


def _fold_outputs(
    joined: Sequence[Mapping[str, object]],
    trajectory_summaries: Sequence[Mapping[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, object]],
    list[str],
]:
    all_assignments: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    pooled: dict[str, dict[str, object]] = {}
    blockers: list[str] = []
    for fold_type in ("lco", "lso"):
        assignments, predictions, fold_blockers = _crossfit(
            joined,
            fold_type=fold_type,
        )
        all_assignments.extend(assignments)
        all_predictions.extend(predictions)
        blockers.extend(fold_blockers)
        fold_values = sorted({str(row["held_out_value"]) for row in predictions})
        identity_field = "problem_id" if fold_type == "lco" else "seed"
        for fold in fold_values:
            fold_predictions = [
                row
                for row in predictions
                if str(row["held_out_value"]) == fold
            ]
            fold_trajectories = [
                row
                for row in trajectory_summaries
                if str(row[identity_field]) == fold
            ]
            metrics = {
                **{
                    name: _metric(fold_trajectories, name)
                    for name in TRAJECTORY_METRIC_FIELDS
                },
                "overwrite_balanced_accuracy": _metric(
                    fold_predictions,
                    "overwrite_balanced_accuracy",
                ),
            }
            summary_rows.append(
                {
                    "validation": fold_type,
                    "fold": fold,
                    "row_count": len(fold_predictions),
                    **{
                        name: "" if value is None else f"{value:.17e}"
                        for name, value in metrics.items()
                    },
                    "bootstrap_lcb_95": "",
                }
            )
        pooled_metrics = {
            **{
                name: _metric(trajectory_summaries, name)
                for name in TRAJECTORY_METRIC_FIELDS
            },
            "overwrite_balanced_accuracy": _metric(
                predictions,
                "overwrite_balanced_accuracy",
            ),
        }
        if pooled_metrics["overwrite_balanced_accuracy"] is None:
            blockers.append(f"{fold_type}:pooled_overwrite_single_class")
        pooled[fold_type] = {
            "row_count": len(predictions),
            "metrics": pooled_metrics,
            "fold_primary_directions": {
                fold: _metric(
                    [
                        row
                        for row in trajectory_summaries
                        if str(row[identity_field]) == fold
                    ],
                    "trajectory_focal_rank_advantage",
                )
                for fold in fold_values
            },
            "trajectory_rows": list(trajectory_summaries),
            "prediction_rows": predictions,
            "conditional_overwrite_ci": True,
        }
    return all_assignments, all_predictions, summary_rows, pooled, list(dict.fromkeys(blockers))


def _case_advantage_share(rows: Sequence[Mapping[str, object]]) -> float | None:
    if not rows:
        return None
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[str(row["problem_id"])].append(
            float(row["trajectory_focal_rank_advantage"])
        )
    contributions = {
        case: statistics.fmean(values) for case, values in by_case.items()
    }
    denominator = math.fsum(abs(value) for value in contributions.values())
    if denominator <= 0.0:
        return None
    return max(abs(value) for value in contributions.values()) / denominator


def _recompute_screen_gate(source_root: Path) -> dict[str, object]:
    return audit_hypergraph_trace(source_root, stage="screen", screen_gate=None)


def _prior_screen_gate_binding(
    screen_gate_path: Path,
    *,
    full_manifest: Mapping[str, object],
) -> tuple[str | None, dict[str, object], list[str]]:
    resolved_gate = screen_gate_path.resolve()
    checks: dict[str, bool] = {}
    binding: dict[str, object] = {
        "required": True,
        "provided": True,
        "status": "fail",
        "gate_path": str(resolved_gate),
        "source_root": None,
        "checks": checks,
    }
    try:
        gate_sha256 = _sha256(resolved_gate)
        screen_payload = _read_json(resolved_gate)
        raw_source_root = screen_payload.get("source_root")
        if not isinstance(raw_source_root, str) or not raw_source_root:
            raise ValueError("screen gate source_root is missing")
        screen_source_root = Path(raw_source_root).resolve()
        binding["source_root"] = str(screen_source_root)
        checks["canonical_gate_location"] = (
            resolved_gate.name == "hypergraph_identifiability_gate.json"
            and screen_source_root == resolved_gate.parent
        )
        recomputed = _recompute_screen_gate(screen_source_root)
        checks["recomputed_gate_exact_match"] = recomputed == screen_payload
        screen_checks = screen_payload.get("checks")
        checks["audited_screen_pass"] = bool(
            screen_payload.get("protocol_version") == PROTOCOL_VERSION
            and screen_payload.get("stage") == "screen"
            and screen_payload.get("status") == "screen_pass"
            and isinstance(screen_checks, dict)
            and screen_checks
            and all(value is True for value in screen_checks.values())
            and screen_payload.get("blockers") == []
        )
        current_bundle = hypergraph_source_bundle()
        current_ast_audit = hypergraph_static_ast_audit()
        checks["source_bundle_binding"] = bool(
            screen_payload.get("source_bundle") == current_bundle
            and full_manifest.get("source_bundle") == current_bundle
        )
        checks["static_ast_binding"] = bool(
            current_ast_audit.get("status") == "pass"
            and screen_payload.get("static_ast_audit") == current_ast_audit
            and full_manifest.get("static_ast_audit") == current_ast_audit
        )
        current_commit = _git_commit()
        checks["source_git_commit_binding"] = bool(
            _hex40(current_commit)
            and screen_payload.get("source_git_commit") == current_commit
            and full_manifest.get("source_git_commit") == current_commit
        )
        checks["frozen_protocol_binding"] = bool(
            screen_payload.get("config_sha256") == CONFIG_SHA256
            and screen_payload.get("spec_sha256") == SPEC_SHA256
            and isinstance(full_manifest.get("config"), dict)
            and full_manifest["config"].get("sha256") == CONFIG_SHA256
            and isinstance(full_manifest.get("spec"), dict)
            and full_manifest["spec"].get("sha256") == SPEC_SHA256
        )
        prior_binding = full_manifest.get("prior_screen_gate")
        checks["full_manifest_screen_binding"] = bool(
            isinstance(prior_binding, dict)
            and prior_binding.get("path") == str(resolved_gate)
            and prior_binding.get("sha256") == gate_sha256
            and prior_binding.get("source_root") == str(screen_source_root)
            and prior_binding.get("status") == "screen_pass"
            and prior_binding.get("source_git_commit") == current_commit
            and prior_binding.get("source_bundle") == current_bundle
            and prior_binding.get("config_sha256") == CONFIG_SHA256
            and prior_binding.get("spec_sha256") == SPEC_SHA256
        )
        if all(checks.values()):
            binding["status"] = "pass"
            return gate_sha256, binding, []
        failed = ",".join(name for name, passed in checks.items() if not passed)
        return gate_sha256, binding, [f"prior_screen_gate_binding_failed:{failed}"]
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        return None, binding, [f"prior_screen_gate_invalid:{exc}"]


def audit_hypergraph_trace(
    source_root: Path | str,
    *,
    stage: str,
    screen_gate: Path | str | None = None,
) -> dict[str, object]:
    source = Path(source_root).resolve()
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if stage == "screen" and screen_gate is not None:
        raise ValueError("screen stage does not accept a prior screen gate")
    if _sha256(CONFIG_PATH) != CONFIG_SHA256 or _sha256(SPEC_PATH) != SPEC_SHA256:
        raise RuntimeError("frozen hypergraph protocol/config hash mismatch")
    config = _read_json(CONFIG_PATH)
    assignments: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    coverage: dict[str, object] = {}
    pooled: dict[str, dict[str, object]] = {}
    source_git_commit: object = None
    prior_screen_gate_sha256: str | None = None
    prior_screen_binding: dict[str, object] = {
        "required": stage == "full",
        "provided": screen_gate is not None,
        "status": "not_required" if stage == "screen" else "fail",
        "gate_path": None if screen_gate is None else str(Path(screen_gate).resolve()),
        "source_root": None,
        "checks": {},
    }
    if stage == "full" and screen_gate is None:
        blockers.append("prior_screen_gate_required")
        checks["prior_screen_gate_binding"] = False
    input_names = (
        "hyperedge_cycle_features.csv",
        "hyperedge_cycle_audit.csv",
        "shared_proposal_audit.csv",
        "hyperedge_cycle_outcomes.csv",
        "hypergraph_trace_manifest.json",
        "same_budget_ledger.csv",
        "aob_input_manifest.csv",
        "anti_leakage_audit.csv",
    )
    try:
        features, audits, proposals, outcomes, manifest, integrity_blockers = (
            _source_integrity(source, stage=stage, config=config)
        )
        source_git_commit = manifest.get("source_git_commit")
        if stage == "full" and screen_gate is not None:
            (
                prior_screen_gate_sha256,
                prior_screen_binding,
                prior_screen_blockers,
            ) = _prior_screen_gate_binding(
                Path(screen_gate),
                full_manifest=manifest,
            )
            blockers.extend(prior_screen_blockers)
            checks["prior_screen_gate_binding"] = (
                prior_screen_binding.get("status") == "pass"
            )
        source_manifests = manifest.get("source_manifests")
        if not isinstance(source_manifests, list):
            raise ValueError("aggregate source manifests are missing")
        joined, coverage, join_blockers = _join_and_validate(
            features,
            audits,
            proposals,
            outcomes,
            source_manifests=source_manifests,
        )
        blockers.extend(integrity_blockers)
        blockers.extend(join_blockers)
        trajectory_rows, trajectory_blockers = _trajectory_summaries(joined)
        blockers.extend(trajectory_blockers)
        assignments, predictions, summaries, pooled, fold_blockers = _fold_outputs(
            joined,
            trajectory_rows,
        )
        blockers.extend(fold_blockers)
        gate_config = config["trace_gates"][stage]
        checks["integrity_fraction"] = not (
            integrity_blockers or join_blockers or trajectory_blockers
        )
        checks["minimum_applicable_trajectories"] = (
            int(coverage["applicable_trajectories"])
            >= int(gate_config["minimum_applicable_trajectories"])
        )
        checks["minimum_cases"] = int(coverage["applicable_cases"]) >= int(
            gate_config["minimum_cases"]
        )
        checks["minimum_seeds"] = int(coverage["applicable_seeds"]) >= int(
            gate_config["minimum_seeds"]
        )
        checks["complete_next_sweep_labels"] = (
            float(coverage["complete_next_sweep_label_fraction"])
            >= float(gate_config["minimum_complete_next_sweep_label_fraction"])
        )
        if stage == "screen":
            checks["required_state_missingness"] = (
                float(coverage["required_state_missing_fraction"])
                <= float(gate_config["maximum_required_state_missing_fraction"])
            )
            priority_mean = _metric(
                trajectory_rows,
                "trajectory_priority_spearman",
            )
            focal_rank_mean = _metric(
                trajectory_rows,
                "trajectory_focal_rank_advantage",
            )
            checks["positive_trajectory_priority_spearman"] = (
                priority_mean is not None
                and priority_mean
                > float(
                    gate_config[
                        "minimum_mean_trajectory_priority_spearman_strictly_above"
                    ]
                )
            )
            checks["positive_trajectory_focal_rank_advantage"] = (
                focal_rank_mean is not None
                and focal_rank_mean
                > float(
                    gate_config[
                        "minimum_mean_trajectory_focal_rank_advantage_strictly_above"
                    ]
                )
            )
            checks["overwrite_balanced_accuracy"] = all(
                pooled[name]["metrics"]["overwrite_balanced_accuracy"] is not None
                and pooled[name]["metrics"]["overwrite_balanced_accuracy"]
                > float(gate_config["minimum_overwrite_balanced_accuracy"])
                for name in ("lco", "lso")
            )
            case_directions = pooled.get("lco", {}).get(
                "fold_primary_directions",
                {},
            )
            seed_directions = pooled.get("lso", {}).get(
                "fold_primary_directions",
                {},
            )
            checks["positive_overlap_case_directions"] = sum(
                value is not None and value > 0.0 for value in case_directions.values()
            ) >= int(gate_config["minimum_positive_overlap_case_directions"])
            checks["positive_seed_directions"] = sum(
                value is not None and value > 0.0 for value in seed_directions.values()
            ) >= int(gate_config["minimum_positive_seed_directions"])
            coverage["trajectory_metric_means"] = {
                "trajectory_priority_spearman": priority_mean,
                "trajectory_focal_rank_advantage": focal_rank_mean,
                "trajectory_owner_survival_spearman": _metric(
                    trajectory_rows,
                    "trajectory_owner_survival_spearman",
                ),
            }
        else:
            trajectory_bootstrap = {
                metric: _bootstrap_lcb(
                    trajectory_rows,
                    metric,
                    resamples=int(config["numeric"]["bootstrap_count"]),
                    seed=int(config["numeric"]["bootstrap_seed"])
                    + metric_index,
                    quantile=float(config["numeric"]["bootstrap_lcb_quantile"]),
                )
                for metric_index, metric in enumerate(TRAJECTORY_METRIC_FIELDS)
            }
            overwrite_bootstrap = {
                fold_type: _bootstrap_lcb(
                    pooled[fold_type]["prediction_rows"],
                    "overwrite_balanced_accuracy",
                    resamples=int(config["numeric"]["bootstrap_count"]),
                    seed=int(config["numeric"]["bootstrap_seed"])
                    + 100
                    + fold_index,
                    quantile=float(config["numeric"]["bootstrap_lcb_quantile"]),
                )
                for fold_index, fold_type in enumerate(("lco", "lso"))
            }
            for metric in gate_config["trajectory_metric_lcb_strictly_positive"]:
                value = trajectory_bootstrap[metric]
                checks[f"{metric}_lcb"] = value is not None and value > 0.0
            for fold_type in ("lco", "lso"):
                ba = overwrite_bootstrap[fold_type]
                checks[f"{fold_type}_overwrite_balanced_accuracy_lcb"] = (
                    ba is not None
                    and ba > float(gate_config["overwrite_balanced_accuracy_lcb_strictly_above"])
                )
            case_directions = pooled["lco"]["fold_primary_directions"]
            seed_directions = pooled["lso"]["fold_primary_directions"]
            checks["positive_case_fold_fraction"] = (
                sum(value is not None and value > 0.0 for value in case_directions.values())
                / max(1, len(case_directions))
                >= float(gate_config["minimum_positive_case_fold_fraction"])
            )
            checks["positive_seed_folds"] = sum(
                value is not None and value > 0.0 for value in seed_directions.values()
            ) >= int(gate_config["minimum_positive_seed_folds"])
            case_share = _case_advantage_share(trajectory_rows)
            checks["single_case_advantage_share"] = (
                case_share is not None
                and case_share <= float(gate_config["maximum_single_case_absolute_advantage_share"])
            )
            coverage["maximum_single_case_absolute_advantage_share"] = case_share
            coverage["bootstrap_lcb"] = {
                "trajectory_metrics": trajectory_bootstrap,
                "overwrite_balanced_accuracy_conditional_oof": overwrite_bootstrap,
            }
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        blockers.append(f"audit_input_invalid:{exc}")

    checks["no_diagnostic_model"] = True
    blockers = list(dict.fromkeys(blockers))
    status = f"{stage}_pass" if checks and all(checks.values()) and not blockers else f"{stage}_no_go"
    _write_csv(source / "hypergraph_fold_assignments.csv", assignments, FOLD_FIELDS)
    _write_csv(
        source / "hypergraph_crossfit_predictions.csv",
        predictions,
        PREDICTION_FIELDS,
    )
    _write_csv(
        source / "hypergraph_predictive_summary.csv",
        summaries,
        SUMMARY_FIELDS,
    )
    gate = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "status": status,
        "checks": checks,
        "blockers": blockers,
        "coverage": coverage,
        "fixed_score_only": True,
        "diagnostic_model_used": False,
        "diagnostic_model_can_rescue": False,
        "support_filter_applied": False,
        "evaluation_scope": "all_decision_eligible_trajectories_no_support_filter",
        "runtime_profile_authorized": False,
        "action_implementation_authorized": bool(
            status == "full_pass"
            and prior_screen_binding.get("status") == "pass"
        ),
        "scheduler_authorized": False,
        "bootstrap": {
            "method": "case_seed_two_way_pigeonhole",
            "trajectory_scalar_scope": "equal_weight_trajectory_summaries",
            "overwrite_scope": "conditional_on_fixed_cross_fitted_predictions",
            "single_class_overwrite_replicate_value": 0.0,
            "resamples": int(config["numeric"]["bootstrap_count"]),
            "seed": int(config["numeric"]["bootstrap_seed"]),
            "lcb_quantile": float(config["numeric"]["bootstrap_lcb_quantile"]),
        },
        "config_sha256": CONFIG_SHA256,
        "spec_sha256": SPEC_SHA256,
        "source_git_commit": source_git_commit,
        "source_bundle": hypergraph_source_bundle(),
        "static_ast_audit": hypergraph_static_ast_audit(),
        "source_root": str(source),
        "prior_screen_gate_sha256": prior_screen_gate_sha256,
        "prior_screen_binding": prior_screen_binding,
        "input_artifact_sha256": {
            name: _sha256(source / name)
            for name in input_names
            if (source / name).is_file()
        },
        "output_artifact_sha256": {
            name: _sha256(source / name)
            for name in (
                "hypergraph_fold_assignments.csv",
                "hypergraph_crossfit_predictions.csv",
                "hypergraph_predictive_summary.csv",
            )
        },
    }
    _write_json(source / "hypergraph_identifiability_gate.json", gate)
    return gate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen v37 hypergraph trace identifiability."
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--screen-gate", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = audit_hypergraph_trace(
        args.source_root,
        stage=args.stage,
        screen_gate=args.screen_gate,
    )
    return 0 if gate["status"] == f"{args.stage}_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
