"""Audit whether recorded runtime state can explain delayed action utility.

This utility is offline-only. It joins terminal outcomes after execution and
must never be imported by runtime dispatch code.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev


FIELD_AUDIT = {
    "action_trace.csv": (
        "remaining_budget_ratio",
        "stagnation_window",
        "delta_mean",
        "sigma_before",
        "sigma_after",
        "best_before",
        "best_after",
        "decision_point",
        "cc_block_fe",
        "cc_utility",
        "search_state_conflict_fraction",
        "search_state_writeback_unstable",
        "search_state_relative_writeback_max",
        "search_state_block_fe",
        "search_state_utility",
        "trust_score",
        "trust_exposure",
        "trust_credit",
        "trust_pre_writeback_fitness",
        "trust_post_writeback_fitness",
        "trajectory_guard_status",
        "trajectory_guard_recovery_credit",
        "trajectory_guard_downstream_fitness",
        "sweep_evidence_active_fraction",
        "sweep_evidence_support",
        "phase_rescue_rejected_before_maturity",
        "phase_rescue_productive_mature",
        "phase_rescue_retired",
    ),
    "overlap_relations.csv": (
        "budget_remaining_ratio",
        "overlap_strength",
        "delta_abs_gap",
        "delta_ratio_gap",
        "one_side_zero",
        "both_positive",
        "rank_gap",
        "rank_stability",
        "shared_var_support_ratio",
        "feature_coverage",
        "fallback_margin_proxy",
    ),
    "car_state_ledger.csv": (
        "graph_fingerprint",
        "component_fingerprint",
        "candidate_action_name",
        "candidate_action_family",
        "evidence_sweeps",
        "checkpoint_fe",
        "state_fingerprint",
    ),
    "car_actionability_trace.csv": (
        "checkpoint_fe",
        "checkpoint_fitness",
        "post_intervention_state_fingerprint",
        "graph_fingerprint",
        "component_fingerprint",
        "candidate_action_name",
        "candidate_action_family",
        "candidate_action_applied",
    ),
}

CANDIDATE_FEATURES = (
    "state_mutation_rate",
    "stagnation_window_mean",
    "delta_mean_mean",
    "trust_credit_mean",
    "trust_credit_min",
    "trust_credit_negative_fraction",
    "trajectory_guard_credit_mean",
    "trajectory_guard_credit_min",
    "trajectory_guard_credit_negative_fraction",
    "trajectory_guard_restore_rate",
    "sweep_active_fraction_mean",
    "sweep_support_mean",
    "local_log_gain_mean",
    "precision_local_log_gain_mean",
    "phase_rescue_retired_rows",
    "precision_rows",
)

CAR_PREACTION_FEATURES = (
    "checkpoint_fe_ratio",
    "evidence_sweeps",
    "pre_relation_rows",
    "pre_sweep_count",
    "overlap_strength_mean",
    "delta_abs_gap_mean",
    "delta_abs_gap_cv",
    "delta_ratio_gap_abs_mean",
    "one_side_zero_fraction",
    "both_positive_fraction",
    "rank_gap_mean",
    "rank_stability_mean",
    "shared_var_support_ratio_mean",
    "feature_coverage_mean",
    "fallback_margin_proxy_mean",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric(rows: list[dict[str, str]], field: str) -> list[float]:
    values = (_float(row.get(field)) for row in rows)
    return [value for value in values if value is not None]


def _mean(values: list[float]) -> float | str:
    return fmean(values) if values else ""


def _minimum(values: list[float]) -> float | str:
    return min(values) if values else ""


def _fraction(predicate_count: int, total: int) -> float | str:
    return predicate_count / total if total else ""


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def within_case_concordance(
    rows: list[dict[str, object]], feature: str, outcome: str
) -> tuple[int, int, float | str]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        feature_value = _float(row.get(feature))
        outcome_value = _float(row.get(outcome))
        if feature_value is not None and outcome_value is not None:
            grouped[str(row["problem_id"])].append((feature_value, outcome_value))

    comparable = 0
    concordant = 0
    for values in grouped.values():
        for left, right in combinations(values, 2):
            feature_delta = left[0] - right[0]
            outcome_delta = left[1] - right[1]
            if feature_delta == 0.0 or outcome_delta == 0.0:
                continue
            comparable += 1
            concordant += int(feature_delta * outcome_delta > 0.0)
    return comparable, concordant, _fraction(concordant, comparable)


def build_field_coverage_rows(
    artifacts: list[tuple[str, Path]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for label, root in artifacts:
        for filename, fields in FIELD_AUDIT.items():
            path = root / filename
            if not path.exists():
                continue
            rows = _read_csv(path)
            header = set(rows[0]) if rows else set()
            for field in fields:
                present = field in header
                values = [
                    str(row.get(field, "")).strip()
                    for row in rows
                    if str(row.get(field, "")).strip()
                ]
                output.append(
                    {
                        "artifact": label,
                        "file": filename,
                        "field": field,
                        "total_rows": len(rows),
                        "field_present": int(present),
                        "nonempty_rows": len(values),
                        "coverage": len(values) / len(rows) if rows else 0.0,
                        "unique_nonempty": len(set(values)),
                    }
                )
    return output


def _result_map(root: Path) -> dict[tuple[str, int], float]:
    output: dict[tuple[str, int], float] = {}
    for row in _read_csv(root / "our_result_by_case.csv"):
        key = (row["problem_id"], int(row["seed"]))
        if key in output:
            raise ValueError(f"duplicate result row: {key} in {root}")
        value = _float(row.get("hcc_smoke_final_error"))
        if value is None or value <= 0.0:
            raise ValueError(f"invalid final error: {key} in {root}")
        output[key] = value
    return output


def _local_log_gains(rows: list[dict[str, str]]) -> list[float]:
    output: list[float] = []
    for row in rows:
        before = _float(row.get("best_before"))
        after = _float(row.get("best_after"))
        if before is not None and after is not None and before > 0.0 and after > 0.0:
            output.append(math.log(before / after))
    return output


def build_candidate_run_rows(
    baseline_root: Path,
    candidates: list[tuple[str, Path]],
    references: dict[str, Path] | None = None,
) -> list[dict[str, object]]:
    references = references or {}
    output: list[dict[str, object]] = []
    for label, root in candidates:
        reference_root = references.get(label, baseline_root)
        baseline = _result_map(reference_root)
        results = _result_map(root)
        trace_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(root / "action_trace.csv"):
            trace_by_key[(row["problem_id"], int(row["seed"]))].append(row)

        for key in sorted(results):
            if key not in baseline:
                raise KeyError(f"missing v33 baseline for {label} {key}")
            traces = trace_by_key[key]
            trust = _numeric(traces, "trust_credit")
            guard = _numeric(traces, "trajectory_guard_recovery_credit")
            sweep_active = _numeric(traces, "sweep_evidence_active_fraction")
            sweep_support = _numeric(traces, "sweep_evidence_support")
            stagnation = _numeric(traces, "stagnation_window")
            delta_mean = _numeric(traces, "delta_mean")
            local_gains = _local_log_gains(traces)
            precision = [
                row
                for row in traces
                if row.get("selected_action_name") == "post_retirement_precision_reanchor"
            ]
            guard_status = [
                row["trajectory_guard_status"]
                for row in traces
                if row.get("trajectory_guard_status")
            ]
            baseline_error = baseline[key]
            candidate_error = results[key]
            output.append(
                {
                    "artifact": label,
                    "reference_artifact": reference_root.name,
                    "problem_id": key[0],
                    "seed": key[1],
                    "reference_error": baseline_error,
                    "candidate_error": candidate_error,
                    "terminal_log_advantage_vs_reference": math.log(
                        baseline_error / candidate_error
                    ),
                    "terminal_catastrophic_vs_reference": int(
                        candidate_error >= 1.2 * baseline_error
                    ),
                    "trace_rows": len(traces),
                    "state_mutation_rate": _fraction(
                        sum(row.get("state_mutated") == "1" for row in traces), len(traces)
                    ),
                    "stagnation_window_count": len(stagnation),
                    "stagnation_window_mean": _mean(stagnation),
                    "delta_mean_count": len(delta_mean),
                    "delta_mean_mean": _mean(delta_mean),
                    "trust_credit_count": len(trust),
                    "trust_credit_mean": _mean(trust),
                    "trust_credit_min": _minimum(trust),
                    "trust_credit_negative_fraction": _fraction(
                        sum(value < 0.0 for value in trust), len(trust)
                    ),
                    "trajectory_guard_credit_count": len(guard),
                    "trajectory_guard_credit_mean": _mean(guard),
                    "trajectory_guard_credit_min": _minimum(guard),
                    "trajectory_guard_credit_negative_fraction": _fraction(
                        sum(value < 0.0 for value in guard), len(guard)
                    ),
                    "trajectory_guard_restore_rate": _fraction(
                        sum(value == "restored" for value in guard_status), len(guard_status)
                    ),
                    "sweep_active_fraction_count": len(sweep_active),
                    "sweep_active_fraction_mean": _mean(sweep_active),
                    "sweep_support_count": len(sweep_support),
                    "sweep_support_mean": _mean(sweep_support),
                    "local_log_gain_count": len(local_gains),
                    "local_log_gain_mean": _mean(local_gains),
                    "phase_rescue_retired_rows": sum(
                        row.get("phase_rescue_retired") == "1" for row in traces
                    ),
                    "precision_rows": len(precision),
                    "precision_local_log_gain_mean": _mean(_local_log_gains(precision)),
                }
            )
    return output


def _association_rows(
    rows: list[dict[str, object]],
    *,
    group_field: str,
    outcome: str,
    features: tuple[str, ...],
    association_kind: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(row)
    for group, group_rows in sorted(grouped.items()):
        for feature in features:
            valid_rows: list[tuple[dict[str, object], float, float]] = []
            for row in group_rows:
                feature_value = _float(row.get(feature))
                outcome_value = _float(row.get(outcome))
                if feature_value is not None and outcome_value is not None:
                    valid_rows.append((row, feature_value, outcome_value))
            pairs = [
                (feature_value, outcome_value)
                for _, feature_value, outcome_value in valid_rows
            ]
            left = [pair[0] for pair in pairs]
            right = [pair[1] for pair in pairs]
            comparable, concordant, agreement = within_case_concordance(
                group_rows, feature, outcome
            )
            correlation = spearman(left, right)
            case_abs_outcome: dict[str, float] = defaultdict(float)
            for row, _, outcome_value in valid_rows:
                case_abs_outcome[str(row["problem_id"])] += abs(outcome_value)
            total_abs_outcome = sum(case_abs_outcome.values())
            dominant_case = (
                max(case_abs_outcome, key=case_abs_outcome.__getitem__)
                if case_abs_outcome
                else ""
            )
            without_dominant = [
                (feature_value, outcome_value)
                for row, feature_value, outcome_value in valid_rows
                if str(row["problem_id"]) != dominant_case
            ]
            without_dominant_correlation = spearman(
                [pair[0] for pair in without_dominant],
                [pair[1] for pair in without_dominant],
            )
            leave_one_case_out = []
            for case in sorted(case_abs_outcome):
                kept = [
                    (feature_value, outcome_value)
                    for row, feature_value, outcome_value in valid_rows
                    if str(row["problem_id"]) != case
                ]
                value = spearman(
                    [pair[0] for pair in kept],
                    [pair[1] for pair in kept],
                )
                if value is not None:
                    leave_one_case_out.append(value)
            output.append(
                {
                    "association_kind": association_kind,
                    "group": group,
                    "feature": feature,
                    "outcome": outcome,
                    "n_runs": len(pairs),
                    "nonzero_outcome_runs": sum(value != 0.0 for value in right),
                    "n_cases": len(
                        {
                            str(row["problem_id"])
                            for row in group_rows
                            if _float(row.get(feature)) is not None
                            and _float(row.get(outcome)) is not None
                        }
                    ),
                    "spearman": "" if correlation is None else correlation,
                    "within_case_comparable_pairs": comparable,
                    "within_case_concordant_pairs": concordant,
                    "within_case_concordance": agreement,
                    "dominant_abs_outcome_case": dominant_case,
                    "dominant_abs_outcome_share": (
                        case_abs_outcome[dominant_case] / total_abs_outcome
                        if dominant_case and total_abs_outcome > 0.0
                        else ""
                    ),
                    "spearman_without_dominant_case": (
                        ""
                        if without_dominant_correlation is None
                        else without_dominant_correlation
                    ),
                    "leave_one_case_out_spearman_min": (
                        min(leave_one_case_out) if leave_one_case_out else ""
                    ),
                    "leave_one_case_out_spearman_max": (
                        max(leave_one_case_out) if leave_one_case_out else ""
                    ),
                }
            )
    return output


def build_candidate_association_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return _association_rows(
        rows,
        group_field="artifact",
        outcome="terminal_log_advantage_vs_reference",
        features=CANDIDATE_FEATURES,
        association_kind="candidate_runtime_state_vs_paired_reference_terminal",
    )


def _group_rows(
    rows: list[dict[str, str]], key_fields: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    output: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[tuple(row[field] for field in key_fields)].append(row)
    return output


def build_car_pair_rows(car_root: Path) -> list[dict[str, object]]:
    trace_rows = _read_csv(car_root / "car_actionability_trace.csv")
    summary = _group_rows(
        _read_csv(car_root / "car_actionability_summary.csv"), ("problem_id", "seed")
    )
    relations = _group_rows(
        _read_csv(car_root / "overlap_relations.csv"),
        ("problem_id", "seed", "lane_id"),
    )
    ledger = _group_rows(
        _read_csv(car_root / "car_state_ledger.csv"), ("problem_id", "seed", "lane_id")
    )

    applied: dict[tuple[str, str], dict[str, str]] = {}
    for row in trace_rows:
        if row.get("audit_arm") != "candidate" or row.get("candidate_action_applied") != "1":
            continue
        key = (row["problem_id"], row["seed"])
        previous = applied.setdefault(key, row)
        if previous["candidate_action_name"] != row["candidate_action_name"]:
            raise ValueError(f"multiple candidate actions for CAR pair: {key}")

    output: list[dict[str, object]] = []
    for key in sorted(applied):
        trace = applied[key]
        max_fes = _float(trace.get("configured_max_fes"))
        checkpoint_fe = _float(trace.get("checkpoint_fe"))
        if max_fes is None or checkpoint_fe is None or max_fes <= 0.0:
            raise ValueError(f"invalid CAR checkpoint: {key}")
        remaining_ratio = 1.0 - checkpoint_fe / max_fes
        pre_relations = [
            row
            for row in relations.get((*key, "oracle_fallback"), [])
            if (_float(row.get("budget_remaining_ratio")) or 0.0) + 1e-12 >= remaining_ratio
        ]
        horizons = {
            row["horizon_label"]: _float(row.get("log_advantage"))
            for row in summary[key]
        }
        if horizons.get("terminal") is None:
            raise ValueError(f"missing CAR terminal utility: {key}")
        evidence = _numeric(ledger.get((*key, "oracle_candidate"), []), "evidence_sweeps")
        overlap = _numeric(pre_relations, "overlap_strength")
        delta_abs = _numeric(pre_relations, "delta_abs_gap")
        delta_ratio_abs = [abs(value) for value in _numeric(pre_relations, "delta_ratio_gap")]
        rank_gap = _numeric(pre_relations, "rank_gap")
        rank_stability = _numeric(pre_relations, "rank_stability")
        support = _numeric(pre_relations, "shared_var_support_ratio")
        coverage = _numeric(pre_relations, "feature_coverage")
        margin = _numeric(pre_relations, "fallback_margin_proxy")
        sweeps = {row.get("outer_iter") for row in pre_relations if row.get("outer_iter")}
        terminal = horizons["terminal"]
        closure = horizons.get("closure_1")
        output.append(
            {
                "artifact": "car_actionability_v2",
                "problem_id": key[0],
                "seed": int(key[1]),
                "candidate_action_name": trace["candidate_action_name"],
                "candidate_action_family": trace["candidate_action_family"],
                "checkpoint_fe": checkpoint_fe,
                "checkpoint_fe_ratio": checkpoint_fe / max_fes,
                "checkpoint_fitness": _float(trace.get("checkpoint_fitness")) or "",
                "evidence_sweeps": max(evidence) if evidence else "",
                "pre_relation_rows": len(pre_relations),
                "pre_sweep_count": len(sweeps),
                "overlap_strength_mean": _mean(overlap),
                "delta_abs_gap_mean": _mean(delta_abs),
                "delta_abs_gap_cv": (
                    pstdev(delta_abs) / abs(fmean(delta_abs))
                    if len(delta_abs) >= 2 and fmean(delta_abs) != 0.0
                    else ""
                ),
                "delta_ratio_gap_abs_mean": _mean(delta_ratio_abs),
                "one_side_zero_fraction": _fraction(
                    sum(row.get("one_side_zero") == "1" for row in pre_relations),
                    len(pre_relations),
                ),
                "both_positive_fraction": _fraction(
                    sum(row.get("both_positive") == "1" for row in pre_relations),
                    len(pre_relations),
                ),
                "rank_gap_mean": _mean(rank_gap),
                "rank_stability_mean": _mean(rank_stability),
                "shared_var_support_ratio_mean": _mean(support),
                "feature_coverage_mean": _mean(coverage),
                "fallback_margin_proxy_mean": _mean(margin),
                "closure_log_advantage": "" if closure is None else closure,
                "budget_3x_log_advantage": (
                    "" if horizons.get("budget_3x") is None else horizons["budget_3x"]
                ),
                "budget_9x_log_advantage": (
                    "" if horizons.get("budget_9x") is None else horizons["budget_9x"]
                ),
                "terminal_log_advantage": terminal,
                "terminal_catastrophic": int(terminal <= math.log(1.0 / 1.2)),
                "closure_terminal_strict_sign_agreement": int(
                    closure is not None
                    and closure != 0.0
                    and terminal != 0.0
                    and closure * terminal > 0.0
                ),
            }
        )
    return output


def build_car_association_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = _association_rows(
        rows,
        group_field="artifact",
        outcome="terminal_log_advantage",
        features=CAR_PREACTION_FEATURES,
        association_kind="car_preaction_state_vs_terminal",
    )
    output.extend(
        _association_rows(
            rows,
            group_field="artifact",
            outcome="terminal_log_advantage",
            features=(
                "closure_log_advantage",
                "budget_3x_log_advantage",
                "budget_9x_log_advantage",
            ),
            association_kind="car_delayed_horizon_vs_terminal",
        )
    )
    return output


def _parse_artifact(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    return label.strip(), Path(path.strip())


def write_reports(
    *,
    baseline_root: Path,
    car_root: Path,
    candidates: list[tuple[str, Path]],
    references: dict[str, Path] | None,
    output_root: Path,
) -> tuple[Path, ...]:
    artifacts = [("v33_baseline", baseline_root), *candidates, ("car_actionability_v2", car_root)]
    coverage_rows = build_field_coverage_rows(artifacts)
    candidate_rows = build_candidate_run_rows(baseline_root, candidates, references)
    candidate_associations = build_candidate_association_rows(candidate_rows)
    car_rows = build_car_pair_rows(car_root)
    car_associations = build_car_association_rows(car_rows)
    paths = (
        output_root / "state_field_coverage.csv",
        output_root / "candidate_run_state_features.csv",
        output_root / "candidate_state_associations.csv",
        output_root / "car_applied_pair_state_features.csv",
        output_root / "car_state_associations.csv",
    )
    for path, rows in zip(
        paths,
        (coverage_rows, candidate_rows, candidate_associations, car_rows, car_associations),
        strict=True,
    ):
        _write_csv(path, rows)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--car-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_artifact,
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--reference",
        action="append",
        type=_parse_artifact,
        default=[],
        metavar="LABEL=PATH",
        help="optional per-candidate paired reference; defaults to --baseline-dir",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.candidate:
        parser.error("at least one --candidate is required")
    references = dict(args.reference)
    if len(references) != len(args.reference):
        parser.error("duplicate --reference label")
    unknown_references = sorted(set(references) - {label for label, _ in args.candidate})
    if unknown_references:
        parser.error(f"reference labels without candidates: {unknown_references}")
    write_reports(
        baseline_root=args.baseline_dir,
        car_root=args.car_dir,
        candidates=args.candidate,
        references=references,
        output_root=args.output_dir,
    )


if __name__ == "__main__":
    main()
