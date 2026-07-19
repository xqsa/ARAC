"""Frozen protocol and promotion gate for the RDDSM evidence-overlay pilot."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from arac.backends.hcc import (
    DEFAULT_AOB_DATA_ROOT,
    HCC_VENDOR_ROOT,
    HccAobExecutionRequest,
    required_aob_data_files,
)
from arac.backends.hcc_evidence_overlay import (
    CHECKPOINT_FIELDS,
    DELAYED_OUTCOME_FIELDS as DELAYED_FIELDS,
    EVIDENCE_OVERLAY_PROTOCOL_VERSION,
    PLAN_FIELDS,
    PROBE_EVIDENCE_FIELDS as PROBE_FIELDS,
    RUNTIME_ACTION_FIELDS as RUNTIME_FIELDS,  # noqa: F401 - aggregate schema export
    SHADOW_DECISION_FIELDS as SHADOW_FIELDS,
)

PROTOCOL_VERSION = EVIDENCE_OVERLAY_PROTOCOL_VERSION
SOURCE_MODE = "fresh_runtime_probe"
RUN_ID = "exp_018_rddsm_evidence_overlay_pilot"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rddsm_evidence_overlay_pilot_v1.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / RUN_ID

CASES = ("E1", "E3", "A4", "S5")
MECHANISM_SEEDS = (117, 118, 119, 120, 121)
LANE_MODE_PAIRS = (
    ("a_rddsm_original_order", "native_audit"),
    ("b_rddsm_evidence_overlay", "paired_owner"),
    ("c_rddsm_shuffled_overlay", "shuffled_owner"),
)
FROZEN_SECTION_SHA256 = {
    "execution": "e4cf584646f5ceed1b60fa9e2757d7e51977961993364b2be0260a27f720edee",
    "overlay": "c44ede1f990a297813b17a792e51ba997cbe6ca10c22f8a074de4fb26fc680f8",
    "promotion_gate": "faba9b13c460c35a1045bb0fd9cb6b053e400e3532180df530f9f43ced59ce56",
}
AGGREGATE_ARTIFACTS = (
    "run_manifest.md",
    "manifest.json",
    "same_budget_ledger.csv",
    "probe_plan.csv",
    "probe_evidence.csv",
    "delayed_outcomes.csv",
    "shadow_decisions.csv",
    "runtime_actions.csv",
    "run_results.csv",
    "lane_summary.csv",
    "promotion_gate.json",
    "aob_input_manifest.csv",
    "anti_leakage_audit.csv",
)
CHECKPOINT_PARITY_FIELDS = (
    "checkpoint_fe",
    "fitness_prefix_hash",
    "incumbent_hash",
    "rddsm_topology_hash",
    "rddsm_order_hash",
    "phase_boundary_fe",
    "history_sweeps",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    evidence_overlay_mode: str


@dataclass(frozen=True)
class RunSpec:
    stage: str
    cohort_id: str
    problem_id: str
    seed: int
    max_fes: int
    lane: LaneSpec

    @property
    def triplet_id(self) -> str:
        return f"{self.stage}:{self.cohort_id}:{self.problem_id}:seed{self.seed}:{self.max_fes}fe"

    @property
    def trajectory_id(self) -> str:
        return f"{self.triplet_id}:{self.lane.lane_id}"


@dataclass(frozen=True)
class GateInputs:
    run_results: Sequence[Mapping[str, object]]
    ledger_rows: Sequence[Mapping[str, object]]
    checkpoint_rows: Mapping[str, Mapping[str, object]]
    plan_rows: Sequence[Mapping[str, object]]
    probe_rows: Sequence[Mapping[str, object]]
    delayed_rows: Sequence[Mapping[str, object]]
    shadow_rows: Sequence[Mapping[str, object]]
    aob_rows: Sequence[Mapping[str, object]]
    anti_leakage_rows: Sequence[Mapping[str, object]]
    integrity_blockers: Sequence[str] = ()


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exp_018 config must be a JSON object")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("exp_018 protocol_version mismatch")
    for section, expected_sha256 in FROZEN_SECTION_SHA256.items():
        observed = payload.get(section)
        encoded = json.dumps(
            observed,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != expected_sha256:
            raise ValueError(f"exp_018 {section} section is frozen")
    execution = payload.get("execution")
    if not isinstance(execution, dict) or execution.get("source_mode") != SOURCE_MODE:
        raise ValueError(f"source_mode must be {SOURCE_MODE}")
    if execution.get("runtime_authorized") is not False:
        raise ValueError("exp_018 is observer-only")
    lanes = payload.get("lanes")
    observed_lanes = tuple(
        (str(row.get("lane_id")), str(row.get("evidence_overlay_mode")))
        for row in lanes
    ) if isinstance(lanes, list) and all(isinstance(row, dict) for row in lanes) else ()
    if observed_lanes != LANE_MODE_PAIRS:
        raise ValueError("exp_018 lane definitions are frozen")
    if tuple(payload.get("artifacts", ())) != AGGREGATE_ARTIFACTS:
        raise ValueError("exp_018 aggregate artifact contract mismatch")
    for stage in ("smoke", "mechanism"):
        build_run_matrix(payload, stage)
    return payload


def _lane_specs(config: Mapping[str, object]) -> tuple[LaneSpec, ...]:
    lanes = config["lanes"]
    assert isinstance(lanes, list)
    return tuple(
        LaneSpec(str(row["lane_id"]), str(row["evidence_overlay_mode"]))
        for row in lanes
        if isinstance(row, dict)
    )


def build_run_matrix(config: Mapping[str, object], stage: str) -> tuple[RunSpec, ...]:
    if stage not in {"smoke", "mechanism"}:
        raise ValueError("stage must be smoke or mechanism")
    matrix = config.get("matrix")
    stage_config = matrix.get(stage) if isinstance(matrix, dict) else None
    if not isinstance(stage_config, dict):
        raise ValueError(f"missing matrix for stage {stage}")
    cohorts = stage_config.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError(f"stage {stage} requires cohorts")
    lanes = _lane_specs(config)
    specs: list[RunSpec] = []
    for cohort in cohorts:
        if not isinstance(cohort, dict):
            raise ValueError("cohort must be an object")
        cases = cohort.get("cases")
        seeds = cohort.get("seeds")
        max_fes = cohort.get("max_fes")
        if (
            not isinstance(cases, list)
            or not isinstance(seeds, list)
            or isinstance(max_fes, bool)
            or not isinstance(max_fes, int)
            or max_fes <= 0
        ):
            raise ValueError("invalid exp_018 cohort")
        for problem_id in cases:
            for seed in seeds:
                if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                    raise ValueError("exp_018 seeds must be explicit non-negative integers")
                for lane in lanes:
                    specs.append(
                        RunSpec(
                            stage=stage,
                            cohort_id=str(cohort["cohort_id"]),
                            problem_id=str(problem_id).upper(),
                            seed=seed,
                            max_fes=max_fes,
                            lane=lane,
                        )
                    )
    expected = int(stage_config.get("expected_run_count", -1))
    if len(specs) != expected or len({spec.trajectory_id for spec in specs}) != expected:
        raise ValueError(f"stage {stage} matrix count does not match preregistration")
    if stage == "smoke":
        frozen = (
            ("mechanical_100k", CASES, (1,), 100_000),
            ("a4_3m", ("A4",), (1,), 3_000_000),
        )
    else:
        frozen = (("mechanism_3m", CASES, MECHANISM_SEEDS, 3_000_000),)
    observed = tuple(
        (
            str(cohort["cohort_id"]),
            tuple(str(case).upper() for case in cohort["cases"]),
            tuple(int(seed) for seed in cohort["seeds"]),
            int(cohort["max_fes"]),
        )
        for cohort in cohorts
    )
    if observed != frozen:
        raise ValueError(f"stage {stage} matrix is frozen")
    return tuple(specs)


def stage_jobs(config: Mapping[str, object], stage: str) -> int:
    matrix = config["matrix"]
    assert isinstance(matrix, dict) and isinstance(matrix[stage], dict)
    jobs = matrix[stage].get("jobs")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs <= 0:
        raise ValueError("jobs must be a positive integer")
    if stage == "mechanism" and jobs != 24:
        raise ValueError("mechanism jobs is frozen at 24")
    return jobs


def build_execution_request(
    spec: RunSpec,
    run_output_dir: Path | str,
    *,
    config: Mapping[str, object],
    python_executable: str,
    hcc_root: Path | str = HCC_VENDOR_ROOT,
    aob_data_root: Path | str = DEFAULT_AOB_DATA_ROOT,
) -> HccAobExecutionRequest:
    execution = config["execution"]
    assert isinstance(execution, dict)
    if execution.get("source_mode") != SOURCE_MODE:
        raise ValueError(f"source_mode must be {SOURCE_MODE}")
    return HccAobExecutionRequest(
        problem_id=spec.problem_id,
        seed=spec.seed,
        max_fes=spec.max_fes,
        output_dir=Path(run_output_dir),
        hcc_root=Path(hcc_root),
        aob_data_root=Path(aob_data_root),
        python_executable=python_executable,
        timestamp=spec.trajectory_id.replace(":", "-"),
        evidence_overlay_mode=spec.lane.evidence_overlay_mode,
    )


def config_sha256(path: Path | str = DEFAULT_CONFIG_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _int(row: Mapping[str, object], field: str, default: int = -1) -> int:
    try:
        value = row.get(field, default)
        if isinstance(value, bool):
            return int(value)
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _float(row: Mapping[str, object], field: str) -> float | None:
    try:
        value = float(str(row.get(field, "")))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _identity(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row.get("trajectory_id", "")), str(row.get("relation_id", ""))


def _midrank_unit(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end) / (2.0 * len(values))
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _midrank_unit(left)
    right_ranks = _midrank_unit(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    covariance = math.fsum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks)
    )
    left_ss = math.fsum((x - left_mean) ** 2 for x in left_ranks)
    right_ss = math.fsum((y - right_mean) ** 2 for y in right_ranks)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    return covariance / math.sqrt(left_ss * right_ss)


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float | None:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    if not positives or not negatives:
        return None
    sensitivity = sum(predictions[index] == 1 for index in positives) / len(positives)
    specificity = sum(predictions[index] == 0 for index in negatives) / len(negatives)
    return 0.5 * (sensitivity + specificity)


def _training_median(
    rows: Sequence[Mapping[str, object]],
    score_field: str,
) -> float | None:
    if not rows:
        return None
    return statistics.median(float(row[score_field]) for row in rows)


def _crossfit(
    rows: Sequence[Mapping[str, object]],
    fold_field: str,
) -> tuple[float | None, float | None, list[dict[str, object]]]:
    predictions: list[dict[str, object]] = []
    folds = sorted({str(row[fold_field]) for row in rows})
    for fold in folds:
        train = [row for row in rows if str(row[fold_field]) != fold]
        test = [row for row in rows if str(row[fold_field]) == fold]
        enhanced_threshold = _training_median(train, "enhanced_score")
        baseline_threshold = _training_median(train, "baseline_score")
        if enhanced_threshold is None or baseline_threshold is None:
            return None, None, []
        for row in test:
            predictions.append(
                {
                    **row,
                    "enhanced_prediction": int(
                        float(row["enhanced_score"]) < enhanced_threshold
                    ),
                    "baseline_prediction": int(
                        float(row["baseline_score"]) < baseline_threshold
                    ),
                }
            )
    labels = [int(row["label"]) for row in predictions]
    enhanced = _balanced_accuracy(labels, [int(row["enhanced_prediction"]) for row in predictions])
    baseline = _balanced_accuracy(labels, [int(row["baseline_prediction"]) for row in predictions])
    return enhanced, baseline, predictions


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _two_way_bootstrap(
    rows: Sequence[Mapping[str, object]],
    metric: Callable[[Sequence[Mapping[str, object]]], float | None],
    *,
    count: int,
    seed: int,
    quantile: float,
) -> float | None:
    cases = sorted({str(row["problem_id"]) for row in rows})
    seeds = sorted({str(row["seed"]) for row in rows})
    if not rows or not cases or not seeds:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(count):
        case_counts = Counter(rng.choices(cases, k=len(cases)))
        seed_counts = Counter(rng.choices(seeds, k=len(seeds)))
        replicate: list[Mapping[str, object]] = []
        for row in rows:
            weight = case_counts[str(row["problem_id"])] * seed_counts[str(row["seed"])]
            replicate.extend([row] * weight)
        value = metric(replicate)
        samples.append(0.0 if value is None else value)
    return _quantile(samples, quantile)


def _ba_improvement(rows: Sequence[Mapping[str, object]]) -> float | None:
    labels = [int(row["label"]) for row in rows]
    enhanced = _balanced_accuracy(labels, [int(row["enhanced_prediction"]) for row in rows])
    baseline = _balanced_accuracy(labels, [int(row["baseline_prediction"]) for row in rows])
    if enhanced is None or baseline is None:
        return None
    return enhanced - baseline


def _selected_plan_rows(
    rows: Sequence[Mapping[str, object]],
    trajectory_id: str | None = None,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if _int(row, "selected", 0) == 1
        and (
            trajectory_id is None
            or str(row.get("trajectory_id")) == trajectory_id
        )
    ]


def _probe_rows_by_relation(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, Mapping[str, object]]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[_identity(row)][str(row.get("candidate", ""))] = row
    return grouped


def _delayed_rows_by_relation(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, Mapping[str, object]]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[_identity(row)][str(row.get("owner", ""))] = row
    return grouped


def _owner_records(inputs: GateInputs) -> list[dict[str, object]]:
    plan = {
        _identity(row): row
        for row in _selected_plan_rows(inputs.plan_rows)
    }
    probes = _probe_rows_by_relation(inputs.probe_rows)
    delayed = _delayed_rows_by_relation(inputs.delayed_rows)
    records: list[dict[str, object]] = []
    by_trajectory: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(set(plan) & set(probes) & set(delayed)):
        trajectory_id, relation_id = key
        plan_row, probe_row, delayed_row = plan[key], probes[key], delayed[key]
        if str(plan_row.get("lane_id")) != "b_rddsm_evidence_overlay":
            continue
        for owner in ("left", "right"):
            reliability = _float(plan_row, f"{owner}_owner_reliability")
            utility = _float(probe_row.get(f"{owner}_owner", {}), "utility")
            overwrite = _float(delayed_row.get(owner, {}), "overwrite_label")
            if (
                reliability is None
                or utility is None
                or overwrite is None
                or not 0.0 <= overwrite <= 1.0
            ):
                continue
            by_trajectory[trajectory_id].append(
                {
                    "trajectory_id": trajectory_id,
                    "relation_id": relation_id,
                    "owner": owner,
                    "problem_id": str(plan_row.get("problem_id", "")),
                    "seed": str(plan_row.get("seed", "")),
                    "baseline_score": reliability,
                    "utility": utility,
                    "label": int(overwrite > 0.5),
                }
            )
    for trajectory_rows in by_trajectory.values():
        ranks = _midrank_unit([float(row["utility"]) for row in trajectory_rows])
        for row, rank in zip(trajectory_rows, ranks):
            row["utility_rank"] = rank
            row["enhanced_score"] = 0.5 * float(row["baseline_score"]) + 0.5 * rank
            records.append(row)
    return records


def _relation_rows(
    rows: Sequence[Mapping[str, object]],
    trajectory_id: str,
) -> list[Mapping[str, object]]:
    return [row for row in rows if str(row.get("trajectory_id")) == trajectory_id]


def _probe_value(rows: Sequence[Mapping[str, object]], probe_fe: int) -> float | None:
    if probe_fe <= 0 or not rows:
        return None
    total = 0.0
    by_relation = _probe_rows_by_relation(rows)
    for candidates in by_relation.values():
        values = [
            _float(candidates.get(candidate, {}), "utility")
            for candidate in ("left_owner", "right_owner", "bridge")
        ]
        if any(value is None for value in values):
            return None
        total += max(0.0, *(value for value in values if value is not None))
    return total / probe_fe


def _required_columns(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bool:
    return bool(rows) and all(all(field in row for field in fields) for row in rows)


def _probe_bundle_valid(
    rows: Sequence[Mapping[str, object]],
    *,
    relation_count: int,
    expected_fe: int,
) -> bool:
    grouped = _probe_rows_by_relation(rows)
    return bool(
        len(rows) == relation_count * 4
        and len(grouped) == relation_count
        and all(_int(row, "actual_fe", 0) == 1 for row in rows)
        and sum(_int(row, "actual_fe", 0) for row in rows) == expected_fe
        and all(
            set(candidates) == {"x0", "left_owner", "right_owner", "bridge"}
            for candidates in grouped.values()
        )
        and all(
            _float(candidates["x0"], "utility") == 0.0
            for candidates in grouped.values()
        )
    )


def _delayed_bundle_valid(
    rows: Sequence[Mapping[str, object]],
    *,
    relation_count: int,
) -> bool:
    grouped = _delayed_rows_by_relation(rows)
    return bool(
        len(rows) == relation_count * 2
        and len(grouped) == relation_count
        and all(set(owners) == {"left", "right"} for owners in grouped.values())
    )


def _shadow_bundle_valid(
    rows: Sequence[Mapping[str, object]],
    *,
    relation_count: int,
) -> bool:
    if len(rows) != relation_count:
        return False
    if len({str(row.get("relation_id", "")) for row in rows}) != relation_count:
        return False
    for row in rows:
        action = str(row.get("action", ""))
        winner = str(row.get("winner", ""))
        reason = str(row.get("reason", ""))
        utility = _float(row, "utility")
        if utility is None or _int(row, "runtime_authorized", 1) != 0:
            return False
        if action == "repair":
            if winner not in {"left_owner", "right_owner"} or reason != (
                "unique_probe_winner_above_one_percent"
            ):
                return False
        elif action == "coordinate":
            if winner != "bridge" or reason != "unique_probe_winner_above_one_percent":
                return False
        elif action == "fallback":
            if winner != "none" or reason not in {
                "non_unique_best_probe_utility",
                "probe_gain_below_one_percent",
            }:
                return False
        else:
            return False
    return True


def _shadow_probe_consistent(
    shadow_rows: Sequence[Mapping[str, object]],
    probe_rows: Sequence[Mapping[str, object]],
    *,
    material_log_utility: float,
) -> bool:
    """Recompute every observer-only decision from its recorded probe utilities."""

    probes = _probe_rows_by_relation(probe_rows)
    if len(shadow_rows) != len(probes):
        return False
    for row in shadow_rows:
        candidates = probes.get(_identity(row))
        if candidates is None:
            return False
        utilities = {
            candidate: _float(candidates.get(candidate, {}), "utility")
            for candidate in ("left_owner", "right_owner", "bridge")
        }
        if any(value is None for value in utilities.values()):
            return False
        numeric = {name: float(value) for name, value in utilities.items()}
        best_utility = max(numeric.values())
        winners = tuple(
            name for name, value in numeric.items() if value == best_utility
        )
        if len(winners) != 1:
            expected = (
                "fallback",
                "none",
                "non_unique_best_probe_utility",
            )
        elif best_utility < material_log_utility:
            expected = (
                "fallback",
                "none",
                "probe_gain_below_one_percent",
            )
        else:
            winner = winners[0]
            expected = (
                "coordinate" if winner == "bridge" else "repair",
                winner,
                "unique_probe_winner_above_one_percent",
            )
        observed_utility = _float(row, "utility")
        if (
            observed_utility is None
            or not math.isclose(
                observed_utility,
                best_utility,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or (
                str(row.get("action", "")),
                str(row.get("winner", "")),
                str(row.get("reason", "")),
            )
            != expected
        ):
            return False
    return True


def _checkpoint_row_valid(
    row: Mapping[str, object],
    *,
    spec: RunSpec | None = None,
) -> bool:
    checkpoint_fe = _int(row, "checkpoint_fe", -1)
    boundary_fe = _int(row, "phase_boundary_fe", -1)
    try:
        sweeps = tuple(
            int(value)
            for value in str(row.get("history_sweeps", "")).split(";")
            if value != ""
        )
    except ValueError:
        return False
    return bool(
        checkpoint_fe > 0
        and checkpoint_fe == boundary_fe
        and (spec is None or checkpoint_fe <= spec.max_fes)
        and len(sweeps) == 3
        and sweeps[0] >= 0
        and sweeps == tuple(range(sweeps[0], sweeps[0] + 3))
        and (
            spec is None
            or (
                str(row.get("problem_id", "")) == spec.problem_id
                and _int(row, "seed", -1) == spec.seed
                and str(row.get("mode", "")) == spec.lane.evidence_overlay_mode
            )
        )
        and _int(row, "previous_survival_closed", 0) == 1
        and _int(row, "runtime_authorized", 1) == 0
        and str(row.get("plan_status", "")) in {"selected", "abstained"}
        and bool(str(row.get("plan_reason", "")))
    )


def _aob_input_hash_binding(
    rows: Sequence[Mapping[str, object]],
    specs: Sequence[RunSpec],
) -> tuple[bool, str]:
    expected = {spec.trajectory_id: spec.problem_id for spec in specs}
    canonical_by_problem: dict[str, dict[str, tuple[str, str]]] = {}
    for problem_id in sorted(set(expected.values())):
        try:
            function_id = int(problem_id[1:])
            paths = required_aob_data_files(DEFAULT_AOB_DATA_ROOT, function_id)
        except (OSError, TypeError, ValueError):
            canonical_by_problem[problem_id] = {}
            continue
        canonical_by_problem[problem_id] = {
            path.name: (
                str(path.resolve()),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in paths
            if path.is_file()
        }
    by_trajectory: dict[str, dict[str, str]] = defaultdict(dict)
    valid = bool(rows)
    for row in rows:
        trajectory_id = str(row.get("trajectory_id", ""))
        problem_id = str(row.get("problem_id", ""))
        filename = str(row.get("file", ""))
        source_path = str(row.get("path", ""))
        before = str(row.get("sha256_before", ""))
        after = str(row.get("sha256_after", ""))
        canonical = canonical_by_problem.get(problem_id, {}).get(filename)
        if (
            trajectory_id not in expected
            or expected[trajectory_id] != problem_id
            or not filename
            or filename in by_trajectory[trajectory_id]
            or not SHA256_RE.fullmatch(before)
            or before != after
            or _int(row, "unchanged", 0) != 1
            or canonical is None
            or str(Path(source_path).resolve()) != canonical[0]
            or before != canonical[1]
        ):
            valid = False
            continue
        by_trajectory[trajectory_id][filename] = before
    if set(by_trajectory) != set(expected):
        valid = False

    problem_bindings: dict[str, dict[str, str]] = {}
    for problem_id in sorted(set(expected.values())):
        trajectory_maps = [
            by_trajectory.get(trajectory_id, {})
            for trajectory_id, expected_problem in expected.items()
            if expected_problem == problem_id
        ]
        if not trajectory_maps or not trajectory_maps[0]:
            valid = False
            continue
        reference = trajectory_maps[0]
        canonical_hashes = {
            filename: digest
            for filename, (_path, digest) in canonical_by_problem.get(
                problem_id,
                {},
            ).items()
        }
        if reference != canonical_hashes:
            valid = False
        if any(mapping != reference for mapping in trajectory_maps[1:]):
            valid = False
        problem_bindings[problem_id] = reference
    encoded = json.dumps(
        problem_bindings,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return valid, hashlib.sha256(encoded).hexdigest()


def _relation_bundle_ids_match(
    selected_plan: Sequence[Mapping[str, object]],
    probe_rows: Sequence[Mapping[str, object]],
    delayed_rows: Sequence[Mapping[str, object]],
    shadow_rows: Sequence[Mapping[str, object]],
) -> bool:
    plan_ids = {str(row.get("relation_id")) for row in selected_plan}
    return bool(
        plan_ids
        and {str(row.get("relation_id")) for row in probe_rows} == plan_ids
        and {str(row.get("relation_id")) for row in delayed_rows} == plan_ids
        and {str(row.get("relation_id")) for row in shadow_rows} == plan_ids
    )


def _phase_boundary_consistent(
    checkpoint: Mapping[str, object] | None,
    plan_rows: Sequence[Mapping[str, object]],
    probe_rows: Sequence[Mapping[str, object]],
) -> bool:
    if checkpoint is None:
        return False
    boundary = str(checkpoint.get("phase_boundary_fe", ""))
    return bool(
        boundary
        and all(str(row.get("phase_boundary_fe", "")) == boundary for row in plan_rows)
        and all(str(row.get("phase_boundary_fe", "")) == boundary for row in probe_rows)
    )


def _raw_runtime_unauthorized(inputs: GateInputs) -> bool:
    rows: list[Mapping[str, object]] = list(inputs.checkpoint_rows.values())
    rows.extend(inputs.plan_rows)
    rows.extend(inputs.probe_rows)
    rows.extend(inputs.delayed_rows)
    rows.extend(inputs.shadow_rows)
    return bool(rows) and all(_int(row, "runtime_authorized", 1) == 0 for row in rows)


def _is_catastrophic(
    candidate: float,
    comparator: float,
    multiplier: float,
) -> bool:
    if candidate < 0.0 or comparator < 0.0:
        raise ValueError("terminal errors must be non-negative")
    if comparator == 0.0:
        return candidate > 0.0
    return candidate >= multiplier * comparator


def build_promotion_gate(
    stage: str,
    config: Mapping[str, object],
    specs: Sequence[RunSpec],
    inputs: GateInputs,
) -> dict[str, object]:
    if stage not in {"smoke", "mechanism"}:
        raise ValueError("stage must be smoke or mechanism")
    gate_config = config.get("promotion_gate")
    if not isinstance(gate_config, dict):
        raise ValueError("promotion_gate config is missing")
    mechanical = gate_config.get("mechanical")
    integrity = gate_config.get("integrity")
    owner_gate = gate_config.get("owner_identifiability")
    delayed_gate = gate_config.get("delayed_alignment")
    negative_gate = gate_config.get("negative_control")
    shadow_gate = gate_config.get("shadow")
    risk_gate = gate_config.get("risk")
    overlay_config = config.get("overlay")
    if not all(
        isinstance(section, dict)
        for section in (
            mechanical,
            integrity,
            owner_gate,
            delayed_gate,
            negative_gate,
            shadow_gate,
            risk_gate,
            overlay_config,
        )
    ):
        raise ValueError("promotion_gate config sections are incomplete")
    assert isinstance(mechanical, dict)
    assert isinstance(integrity, dict)
    assert isinstance(owner_gate, dict)
    assert isinstance(delayed_gate, dict)
    assert isinstance(negative_gate, dict)
    assert isinstance(shadow_gate, dict)
    assert isinstance(risk_gate, dict)
    assert isinstance(overlay_config, dict)
    top_relations = int(overlay_config["top_relations"])
    maximum_probe_fe = int(overlay_config["maximum_probe_fe"])
    result_by_id = {str(row.get("trajectory_id")): row for row in inputs.run_results}
    ledger_by_id = {str(row.get("trajectory_id")): row for row in inputs.ledger_rows}
    blockers = list(dict.fromkeys(str(item) for item in inputs.integrity_blockers))
    checks: dict[str, bool] = {}
    expected_count = int(
        mechanical["required_completed_runs"]
        if stage == "smoke"
        else integrity["required_fresh_runs"]
    )
    checks["preregistered_run_count"] = (
        len(specs) == expected_count
        and len(result_by_id) == expected_count
        and set(result_by_id) == {spec.trajectory_id for spec in specs}
    )
    checks["all_runs_completed"] = checks["preregistered_run_count"] and all(
        str(row.get("status")) == "completed" for row in inputs.run_results
    )
    checks["all_runs_fresh"] = checks["preregistered_run_count"] and all(
        _int(row, "fresh_optimizer_execution", 0) == 1
        and str(row.get("source_mode")) == SOURCE_MODE
        for row in inputs.run_results
    )
    checks["zero_fe_overrun"] = bool(inputs.ledger_rows) and all(
        _int(row, "same_budget_violation", 1) == 0 for row in inputs.ledger_rows
    )
    checks["fe_ledger_closed"] = len(ledger_by_id) == expected_count and all(
        _int(row, "ledger_closed", 0) == 1 for row in inputs.ledger_rows
    )
    checks["terminal_tolerance"] = len(ledger_by_id) == expected_count and all(
        str(row.get("terminal_tolerance_rule"))
        == str(overlay_config["terminal_tolerance_rule"])
        and _int(row, "terminal_tolerance_fe", -1) >= 0
        and _int(row, "actual_fe_used", -1)
        >= _int(row, "budget_limit", 0) - _int(row, "terminal_tolerance_fe", -1)
        for row in inputs.ledger_rows
    )
    checks["aob_inputs_unchanged"] = bool(inputs.aob_rows) and all(
        str(row.get("unchanged")) == "1" for row in inputs.aob_rows
    )
    aob_binding_valid, aob_binding_sha256 = _aob_input_hash_binding(
        inputs.aob_rows,
        specs,
    )
    checks["aob_input_hash_consistency"] = aob_binding_valid
    checks["anti_leakage"] = (
        len(inputs.anti_leakage_rows) == expected_count
        and all(str(row.get("audit_status")) == "pass" for row in inputs.anti_leakage_rows)
    )
    checks["raw_schemas"] = (
        _required_columns(inputs.plan_rows, PLAN_FIELDS)
        and _required_columns(inputs.probe_rows, PROBE_FIELDS)
        and _required_columns(inputs.delayed_rows, DELAYED_FIELDS)
        and _required_columns(inputs.shadow_rows, SHADOW_FIELDS)
    )
    checks["all_raw_rows_runtime_unauthorized"] = _raw_runtime_unauthorized(inputs)

    specs_by_triplet: dict[str, list[RunSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_triplet[spec.triplet_id].append(spec)
    checkpoint_matches = 0
    applicable_triplets: list[str] = []
    applicable_details: list[dict[str, object]] = []
    closure_total = 0
    closure_closed = 0
    pair_integrity = True
    for triplet_id, triplet_specs in specs_by_triplet.items():
        checkpoint_rows = [inputs.checkpoint_rows.get(spec.trajectory_id) for spec in triplet_specs]
        checkpoint_valid = all(
            row is not None
            and all(field in row for field in CHECKPOINT_FIELDS)
            and _checkpoint_row_valid(row, spec=spec)
            and all(
                SHA256_RE.fullmatch(str(row.get(field, "")))
                for field in (
                    "fitness_prefix_hash",
                    "incumbent_hash",
                    "rddsm_topology_hash",
                    "rddsm_order_hash",
                )
            )
            for spec, row in zip(triplet_specs, checkpoint_rows, strict=True)
        )
        if checkpoint_valid:
            signatures = {
                tuple(str(row[field]) for field in CHECKPOINT_PARITY_FIELDS)
                for row in checkpoint_rows
                if row is not None
            }
            checkpoint_valid = len(signatures) == 1
        checkpoint_matches += int(checkpoint_valid)

        by_lane = {spec.lane.lane_id: spec for spec in triplet_specs}
        lane_a = by_lane[LANE_MODE_PAIRS[0][0]]
        lane_b = by_lane[LANE_MODE_PAIRS[1][0]]
        lane_c = by_lane[LANE_MODE_PAIRS[2][0]]
        result_a = result_by_id.get(lane_a.trajectory_id, {})
        result_b = result_by_id.get(lane_b.trajectory_id, {})
        result_c = result_by_id.get(lane_c.trajectory_id, {})
        a_fe = _int(result_a, "evidence_overlay_fe", -1)
        b_fe = _int(result_b, "evidence_overlay_fe", -1)
        c_fe = _int(result_c, "evidence_overlay_fe", -1)
        if (
            a_fe != 0
            or b_fe not in {0, maximum_probe_fe}
            or c_fe not in {0, maximum_probe_fe}
            or b_fe != c_fe
        ):
            pair_integrity = False
        b_plan = _selected_plan_rows(inputs.plan_rows, lane_b.trajectory_id)
        c_plan = _selected_plan_rows(inputs.plan_rows, lane_c.trajectory_id)
        b_probe = _relation_rows(inputs.probe_rows, lane_b.trajectory_id)
        c_probe = _relation_rows(inputs.probe_rows, lane_c.trajectory_id)
        b_delayed = _relation_rows(inputs.delayed_rows, lane_b.trajectory_id)
        c_delayed = _relation_rows(inputs.delayed_rows, lane_c.trajectory_id)
        b_shadow = _relation_rows(inputs.shadow_rows, lane_b.trajectory_id)
        c_shadow = _relation_rows(inputs.shadow_rows, lane_c.trajectory_id)
        b_relation_ids = {str(row.get("relation_id")) for row in b_plan}
        c_relation_ids = {str(row.get("relation_id")) for row in c_plan}
        applicable = (
            b_fe == c_fe == maximum_probe_fe
            and _int(result_b, "applicable", 0) == 1
            and _int(result_c, "applicable", 0) == 1
            and len(b_plan) == len(c_plan) == top_relations
            and _probe_bundle_valid(
                b_probe,
                relation_count=top_relations,
                expected_fe=maximum_probe_fe,
            )
            and _probe_bundle_valid(
                c_probe,
                relation_count=top_relations,
                expected_fe=maximum_probe_fe,
            )
            and _delayed_bundle_valid(
                b_delayed,
                relation_count=top_relations,
            )
            and _delayed_bundle_valid(
                c_delayed,
                relation_count=top_relations,
            )
            and len(b_relation_ids) == len(c_relation_ids) == top_relations
            and _relation_bundle_ids_match(
                b_plan,
                b_probe,
                b_delayed,
                b_shadow,
            )
            and _relation_bundle_ids_match(
                c_plan,
                c_probe,
                c_delayed,
                c_shadow,
            )
            and len(b_shadow) == len(c_shadow) == top_relations
            and _shadow_bundle_valid(
                b_shadow,
                relation_count=top_relations,
            )
            and _shadow_probe_consistent(
                b_shadow,
                b_probe,
                material_log_utility=float(
                    overlay_config["material_log_utility"]
                ),
            )
            and _shadow_bundle_valid(
                c_shadow,
                relation_count=top_relations,
            )
            and _shadow_probe_consistent(
                c_shadow,
                c_probe,
                material_log_utility=float(
                    overlay_config["material_log_utility"]
                ),
            )
            and _phase_boundary_consistent(
                inputs.checkpoint_rows.get(lane_b.trajectory_id),
                _relation_rows(inputs.plan_rows, lane_b.trajectory_id),
                b_probe,
            )
            and _phase_boundary_consistent(
                inputs.checkpoint_rows.get(lane_c.trajectory_id),
                _relation_rows(inputs.plan_rows, lane_c.trajectory_id),
                c_probe,
            )
            and all(
                len(str(row.get("owner_groups", "")).split(";")) == 2
                and bool(str(row.get("shared_variables", "")))
                for row in (*b_plan, *c_plan)
            )
            and b_relation_ids != c_relation_ids
        )
        if applicable:
            applicable_triplets.append(triplet_id)
            applicable_details.append(
                {
                    "triplet_id": triplet_id,
                    "problem_id": lane_b.problem_id,
                    "seed": lane_b.seed,
                    "cohort_id": lane_b.cohort_id,
                }
            )
            for row in (*b_delayed, *c_delayed):
                closure_total += 1
                closure_closed += int(_int(row, "label_closed", 0) == 1)
        elif b_fe != 0 or c_fe != 0:
            pair_integrity = False
    checkpoint_fraction = checkpoint_matches / max(1, len(specs_by_triplet))
    closure_fraction = closure_closed / max(1, closure_total)
    required_checkpoint_fraction = float(
        mechanical["required_checkpoint_pair_fraction"]
        if stage == "smoke"
        else 1.0
    )
    required_closure_fraction = float(
        mechanical["required_delayed_closure_fraction"]
        if stage == "smoke"
        else integrity["required_delayed_closure_fraction"]
    )
    checks["checkpoint_triplet_parity"] = (
        checkpoint_fraction >= required_checkpoint_fraction
    )
    checks["paired_overlay_integrity"] = pair_integrity
    checks["delayed_label_closure"] = (
        closure_total > 0 and closure_fraction >= required_closure_fraction
    )
    checks["e1_zero_probe"] = all(
        _int(result_by_id.get(spec.trajectory_id, {}), "evidence_overlay_fe", -1)
        == int(mechanical["required_e1_probe_fe"])
        for spec in specs
        if spec.problem_id == "E1"
    )

    coverage: dict[str, object] = {
        "expected_runs": expected_count,
        "observed_runs": len(result_by_id),
        "triplet_count": len(specs_by_triplet),
        "checkpoint_pair_fraction": checkpoint_fraction,
        "applicable_triplet_count": len(applicable_triplets),
        "applicable_triplets": applicable_details,
        "delayed_closure_fraction": closure_fraction,
        "aob_input_binding_sha256": aob_binding_sha256,
    }
    metrics: dict[str, object] = {}
    if stage == "smoke":
        checks["a4_active_lanes_have_16_probe_fe"] = all(
            _int(result_by_id.get(spec.trajectory_id, {}), "evidence_overlay_fe", -1)
            == int(mechanical["required_a4_overlay_fe_per_active_lane"])
            for spec in specs
            if spec.problem_id == "A4" and spec.lane.lane_id != LANE_MODE_PAIRS[0][0]
        )
    else:
        applicable_cases = {str(row["problem_id"]) for row in applicable_details}
        applicable_seeds = {str(row["seed"]) for row in applicable_details}
        checks["minimum_applicable_overlap_triplets"] = (
            len(applicable_triplets)
            >= int(integrity["minimum_applicable_overlap_triplets"])
        )
        checks["overlap_case_coverage"] = applicable_cases == set(
            str(case) for case in integrity["required_overlap_cases"]
        )
        checks["overlap_seed_coverage"] = (
            len(applicable_seeds) >= int(integrity["minimum_applicable_seeds"])
        )

        owner_rows = _owner_records(inputs)
        lco_enhanced, lco_baseline, lco_predictions = _crossfit(owner_rows, "problem_id")
        lso_enhanced, lso_baseline, lso_predictions = _crossfit(owner_rows, "seed")
        bootstrap_count = int(negative_gate["bootstrap_count"])
        bootstrap_seed = int(negative_gate["bootstrap_seed"])
        lco_bootstrap_seed = int(owner_gate["lco_bootstrap_seed"])
        lso_bootstrap_seed = int(owner_gate["lso_bootstrap_seed"])
        quantile = float(negative_gate["one_sided_lcb_quantile"])
        lco_lcb = _two_way_bootstrap(
            lco_predictions,
            _ba_improvement,
            count=bootstrap_count,
            seed=lco_bootstrap_seed,
            quantile=quantile,
        )
        lso_lcb = _two_way_bootstrap(
            lso_predictions,
            _ba_improvement,
            count=bootstrap_count,
            seed=lso_bootstrap_seed,
            quantile=quantile,
        )
        checks["lco_enhanced_owner_balanced_accuracy"] = (
            lco_enhanced is not None
            and lco_enhanced
            > float(owner_gate["minimum_lco_balanced_accuracy_strictly_above"])
        )
        checks["lso_enhanced_owner_balanced_accuracy"] = (
            lso_enhanced is not None
            and lso_enhanced
            > float(owner_gate["minimum_lso_balanced_accuracy_strictly_above"])
        )
        improvement_minimum = float(
            owner_gate["minimum_case_seed_bootstrap_improvement_lcb_strictly_above"]
        )
        checks["lco_improvement_bootstrap_lcb"] = (
            lco_lcb is not None and lco_lcb > improvement_minimum
        )
        checks["lso_improvement_bootstrap_lcb"] = (
            lso_lcb is not None and lso_lcb > improvement_minimum
        )
        metrics["owner_identifiability"] = {
            "owner_rows": len(owner_rows),
            "lco_enhanced_balanced_accuracy": lco_enhanced,
            "lco_baseline_balanced_accuracy": lco_baseline,
            "lco_improvement_lcb_95": lco_lcb,
            "lco_bootstrap_seed": lco_bootstrap_seed,
            "lso_enhanced_balanced_accuracy": lso_enhanced,
            "lso_baseline_balanced_accuracy": lso_baseline,
            "lso_improvement_lcb_95": lso_lcb,
            "lso_bootstrap_seed": lso_bootstrap_seed,
            "overwrite_positive_class": owner_gate["overwrite_positive_class"],
            "crossfit_threshold": owner_gate["crossfit_threshold"],
            "overwrite_prediction": owner_gate["overwrite_prediction"],
        }

        plan_by_key = {
            _identity(row): row
            for row in _selected_plan_rows(inputs.plan_rows)
        }
        probe_by_key = _probe_rows_by_relation(inputs.probe_rows)
        delayed_by_key = _delayed_rows_by_relation(inputs.delayed_rows)
        direction_matches = 0
        direction_total = 0
        trajectory_spearman: list[float] = []
        for spec in specs:
            if spec.lane.lane_id != LANE_MODE_PAIRS[1][0]:
                continue
            relation_keys = [key for key in plan_by_key if key[0] == spec.trajectory_id]
            vois: list[float] = []
            credits: list[float] = []
            for key in relation_keys:
                probe_candidates = probe_by_key.get(key)
                delayed_owners = delayed_by_key.get(key)
                if probe_candidates is None or delayed_owners is None:
                    continue
                left_u = _float(probe_candidates.get("left_owner", {}), "utility")
                right_u = _float(probe_candidates.get("right_owner", {}), "utility")
                left_s = _float(delayed_owners.get("left", {}), "survival_label")
                right_s = _float(delayed_owners.get("right", {}), "survival_label")
                comparable = (
                    None not in {left_u, right_u, left_s, right_s}
                    and left_u != right_u
                    and left_s != right_s
                )
                if comparable:
                    direction_total += 1
                    direction_matches += int((left_u > right_u) == (left_s > right_s))
                voi = _float(plan_by_key[key], "voi")
                left_credit = _float(
                    delayed_owners.get("left", {}),
                    "overwrite_penalized_credit",
                )
                right_credit = _float(
                    delayed_owners.get("right", {}),
                    "overwrite_penalized_credit",
                )
                if voi is not None and left_credit is not None and right_credit is not None:
                    vois.append(voi)
                    credits.append(max(left_credit, right_credit))
            rho = _spearman(vois, credits)
            if rho is not None:
                trajectory_spearman.append(rho)
        direction_agreement = direction_matches / max(1, direction_total)
        mean_spearman = statistics.fmean(trajectory_spearman) if trajectory_spearman else None
        checks["owner_preference_direction_agreement"] = (
            direction_total > 0
            and direction_agreement
            >= float(delayed_gate["minimum_owner_preference_direction_agreement"])
        )
        checks["positive_voi_delayed_credit_spearman"] = (
            mean_spearman is not None
            and mean_spearman
            > float(
                delayed_gate[
                    "minimum_mean_trajectory_voi_credit_spearman_strictly_above"
                ]
            )
        )
        metrics["delayed_alignment"] = {
            "direction_pairs": direction_total,
            "direction_agreement": direction_agreement,
            "trajectory_spearman_count": len(trajectory_spearman),
            "mean_trajectory_voi_credit_spearman": mean_spearman,
        }

        probe_value_rows: list[dict[str, object]] = []
        for detail in applicable_details:
            triplet_specs = specs_by_triplet[str(detail["triplet_id"])]
            by_lane = {spec.lane.lane_id: spec for spec in triplet_specs}
            b_spec = by_lane[LANE_MODE_PAIRS[1][0]]
            c_spec = by_lane[LANE_MODE_PAIRS[2][0]]
            b_value = _probe_value(
                _relation_rows(inputs.probe_rows, b_spec.trajectory_id),
                _int(result_by_id[b_spec.trajectory_id], "evidence_overlay_fe", 0),
            )
            c_value = _probe_value(
                _relation_rows(inputs.probe_rows, c_spec.trajectory_id),
                _int(result_by_id[c_spec.trajectory_id], "evidence_overlay_fe", 0),
            )
            if b_value is not None and c_value is not None:
                probe_value_rows.append(
                    {
                        "problem_id": b_spec.problem_id,
                        "seed": str(b_spec.seed),
                        "difference": b_value - c_value,
                    }
                )
        difference_metric = lambda rows: (
            statistics.fmean(float(row["difference"]) for row in rows) if rows else None
        )
        probe_lcb = _two_way_bootstrap(
            probe_value_rows,
            difference_metric,
            count=bootstrap_count,
            seed=bootstrap_seed,
            quantile=quantile,
        )
        probe_median = (
            statistics.median(float(row["difference"]) for row in probe_value_rows)
            if probe_value_rows else None
        )
        checks["probe_value_vs_shuffle_lcb"] = (
            probe_lcb is not None
            and probe_lcb > float(negative_gate["minimum_lcb_strictly_above"])
        )
        checks["probe_value_vs_shuffle_median"] = (
            probe_median is not None
            and probe_median >= float(negative_gate["minimum_median"])
        )
        metrics["negative_control"] = {
            "paired_triplets": len(probe_value_rows),
            "median_b_minus_c_probe_value_per_fe": probe_median,
            "lcb_95_b_minus_c_probe_value_per_fe": probe_lcb,
            "bootstrap_count": bootstrap_count,
            "bootstrap_seed": bootstrap_seed,
        }

        non_fallback = [
            row for row in inputs.shadow_rows
            if str(row.get("lane_id")) == LANE_MODE_PAIRS[1][0]
            and str(row.get("action")) in {"repair", "coordinate"}
            and _shadow_bundle_valid((row,), relation_count=1)
        ]
        checks["shadow_non_fallback_count"] = (
            len(non_fallback) >= int(shadow_gate["minimum_non_fallback_decisions"])
        )
        checks["shadow_case_coverage"] = (
            len({str(row.get("problem_id")) for row in non_fallback})
            >= int(shadow_gate["minimum_cases"])
        )
        checks["shadow_seed_coverage"] = (
            len({str(row.get("seed")) for row in non_fallback})
            >= int(shadow_gate["minimum_seeds"])
        )
        metrics["shadow"] = {
            "non_fallback_count": len(non_fallback),
            "case_count": len({str(row.get("problem_id")) for row in non_fallback}),
            "seed_count": len({str(row.get("seed")) for row in non_fallback}),
        }

        catastrophic: list[str] = []
        multiplier = float(risk_gate["catastrophic_multiplier"])
        for triplet_id, triplet_specs in specs_by_triplet.items():
            by_lane = {spec.lane.lane_id: spec for spec in triplet_specs}
            values: dict[str, float] = {}
            for lane_id, spec in by_lane.items():
                value = _float(result_by_id.get(spec.trajectory_id, {}), "native_terminal_error")
                if value is not None:
                    values[lane_id] = value
            if len(values) != 3:
                catastrophic.append(f"{triplet_id}:missing_terminal_error")
                continue
            b_value = values[LANE_MODE_PAIRS[1][0]]
            for comparator in (LANE_MODE_PAIRS[0][0], LANE_MODE_PAIRS[2][0]):
                try:
                    event = _is_catastrophic(
                        b_value,
                        values[comparator],
                        multiplier,
                    )
                except ValueError:
                    event = True
                if event:
                    catastrophic.append(f"{triplet_id}:vs:{comparator}")
        checks["zero_catastrophic_trajectories"] = (
            len(catastrophic) <= int(risk_gate["maximum_catastrophic_trajectories"])
        )
        metrics["catastrophic"] = {
            "multiplier": multiplier,
            "events": catastrophic,
        }

    for name, passed in checks.items():
        if not passed:
            blockers.append(f"gate_failed:{name}")
    blockers = list(dict.fromkeys(blockers))
    passed = bool(checks) and all(checks.values()) and not blockers
    if passed:
        status = "smoke_pass" if stage == "smoke" else "pilot_go"
    else:
        status = "pilot_no_go"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "status": status,
        "checks": checks,
        "blockers": blockers,
        "coverage": coverage,
        "metrics": metrics,
        "bootstrap": {
            "method": negative_gate["bootstrap_method"],
            "count": int(negative_gate["bootstrap_count"]),
            "seed": int(negative_gate["bootstrap_seed"]),
            "lco_owner_improvement_seed": int(owner_gate["lco_bootstrap_seed"]),
            "lso_owner_improvement_seed": int(owner_gate["lso_bootstrap_seed"]),
            "one_sided_lcb_quantile": float(
                negative_gate["one_sided_lcb_quantile"]
            ),
            "resampling_unit": "case_by_seed_two_way_pigeonhole",
        },
        "source_mode": SOURCE_MODE,
        "observer_only": True,
        "runtime_profile_authorized": False,
        "phase_two_action_authorized": False,
        "action_v2_design_authorized": bool(stage == "mechanism" and status == "pilot_go"),
        "lane_d_authorized": False,
        "threshold_tuning_authorized": False,
    }
