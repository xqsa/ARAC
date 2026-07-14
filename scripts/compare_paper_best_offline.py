"""Join fresh runtime results with the frozen paper-best table offline.

This utility is deliberately separate from the runtime runner. It must run
only after an experiment has completed and must never be imported by runtime
dispatch code.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SEED_COLUMNS = (1, 2, 3)
CATASTROPHIC_THRESHOLD_PCT = -20.0


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


def build_comparison_rows(
    result_rows: list[dict[str, str]],
    paper_rows: list[dict[str, str]],
    *,
    paper_source: str,
) -> list[dict[str, object]]:
    paper_best: dict[str, float] = {}
    for row in paper_rows:
        case = row.get("case", "").strip()
        value = row.get("paper_best", "").strip()
        if not case or not value:
            raise ValueError("paper rows require case and paper_best")
        if case in paper_best:
            raise ValueError(f"duplicate paper-best case: {case}")
        reference = float(value)
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError(f"paper-best value must be finite and positive: {case}")
        paper_best[case] = reference

    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in result_rows:
        case = row.get("problem_id", "").strip()
        if not case or not row.get("seed") or not row.get("hcc_smoke_final_error"):
            raise ValueError("result rows require problem_id, seed, and final error")
        seed = int(row["seed"])
        if seed not in SEED_COLUMNS:
            raise ValueError(f"unexpected seed for {case}: {seed}")
        if seed in grouped[case]:
            raise ValueError(f"duplicate result seed for {case}: {seed}")
        error = float(row["hcc_smoke_final_error"])
        if not math.isfinite(error):
            raise ValueError(f"final error must be finite: {case} seed {seed}")
        grouped[case][seed] = error

    output: list[dict[str, object]] = []
    for case in sorted(grouped):
        if case not in paper_best:
            raise KeyError(f"missing paper-best value for {case}")
        by_seed = grouped[case]
        missing = [seed for seed in SEED_COLUMNS if seed not in by_seed]
        if missing:
            raise ValueError(f"missing seeds for {case}: {missing}")
        reference = paper_best[case]
        errors = [by_seed[seed] for seed in SEED_COLUMNS]
        gains = [100.0 * (reference - error) / reference for error in errors]
        best_error = min(errors)
        worst_error = max(errors)
        mean_error = sum(errors) / len(errors)
        output.append(
            {
                "case": case,
                "paper_best": reference,
                "seed1": errors[0],
                "seed2": errors[1],
                "seed3": errors[2],
                "best_error": best_error,
                "best_gain_pct": 100.0 * (reference - best_error) / reference,
                "best_of_three_win": int(best_error < reference),
                "mean_error": mean_error,
                "mean_gain_pct": 100.0 * (reference - mean_error) / reference,
                "mean_win": int(mean_error < reference),
                "worst_error": worst_error,
                "worst_gain_pct": 100.0 * (reference - worst_error) / reference,
                "worst_win": int(worst_error < reference),
                "seed_win_count": sum(gain > 0.0 for gain in gains),
                "catastrophic_seed_count": sum(
                    gain <= CATASTROPHIC_THRESHOLD_PCT for gain in gains
                ),
                "catastrophic_threshold": f"{CATASTROPHIC_THRESHOLD_PCT:g}% relative gain",
                "paper_best_source": paper_source,
                "runtime_dispatch_used": 0,
            }
        )
    return output


def build_runtime_evidence_rows(
    *,
    result_rows: list[dict[str, str]],
    trace_rows: list[dict[str, str]],
    ledger_rows: list[dict[str, str]],
    aob_rows: list[dict[str, str]],
    leakage_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    result_by_case = defaultdict(list)
    for row in result_rows:
        result_by_case[row["problem_id"]].append(row)
    trace_by_case = defaultdict(list)
    for row in trace_rows:
        trace_by_case[row["problem_id"]].append(row)
    ledger_by_case = defaultdict(list)
    for row in ledger_rows:
        ledger_by_case[row["problem_id"]].append(row)
    aob_by_case = defaultdict(list)
    for row in aob_rows:
        aob_by_case[row["problem_id"]].append(row)

    leakage_failures = sum(row.get("audit_status") != "pass" for row in leakage_rows)
    output = []
    for case in sorted(result_by_case):
        traces = trace_by_case[case]
        continuation = [
            row
            for row in traces
            if row.get("selected_action_name")
            == "cross_sweep_cma_sigma_continuation"
        ]
        factors = [
            float(row["cma_sigma_next_factor"])
            for row in continuation
            if row.get("cma_sigma_next_factor")
        ]
        ledgers = ledger_by_case[case]
        output.append(
            {
                "case": case,
                "run_count": len(result_by_case[case]),
                "trace_rows": len(traces),
                "continuation_rows": len(continuation),
                "cold_start_rows": sum(
                    row.get("cma_sigma_route") == "cold_start" for row in continuation
                ),
                "continued_rows": sum(
                    row.get("cma_sigma_route") == "continued" for row in continuation
                ),
                "nonunit_applied_rows": sum(
                    abs(float(row["cma_sigma_applied_factor"]) - 1.0) > 1e-12
                    for row in continuation
                    if row.get("cma_sigma_applied_factor")
                ),
                "lower_clipped_rows": sum(
                    abs(value - 0.5) <= 1e-12 for value in factors
                ),
                "upper_clipped_rows": sum(
                    abs(value - 1.5) <= 1e-12 for value in factors
                ),
                "min_next_factor": min(factors) if factors else math.nan,
                "max_next_factor": max(factors) if factors else math.nan,
                "retirement_rows": sum(
                    row.get("phase_rescue_resource_route")
                    == "zero_yield_phase_rescue_retired"
                    for row in traces
                ),
                "precision_rows": sum(
                    row.get("selected_action_name")
                    == "post_retirement_precision_reanchor"
                    for row in traces
                ),
                "fe_min": min(int(row["actual_fe_used"]) for row in ledgers),
                "fe_max": max(int(row["actual_fe_used"]) for row in ledgers),
                "fresh_runs": sum(row.get("fresh_execution") == "1" for row in ledgers),
                "same_budget_violations": sum(
                    row.get("same_budget_violation") != "0" for row in ledgers
                ),
                "aob_changed_rows": sum(
                    row.get("unchanged") != "1" for row in aob_by_case[case]
                ),
                "anti_leakage_failures": leakage_failures,
                "runtime_dispatch_used": 0,
            }
        )
    return output


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "case",
        "best_of_three_win",
        "mean_win",
        "worst_win",
        "seed_win_count",
        "catastrophic_seed_count",
        "best_gain_pct",
        "mean_gain_pct",
        "worst_gain_pct",
    ]
    lines = [
        "# Offline Paper-Best Comparison",
        "",
        "Runtime dispatch used: 0. Paper-best values are joined after execution.",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(output_dir: Path, paper_path: Path) -> tuple[Path, Path, Path]:
    result_rows = _read_csv(output_dir / "our_result_by_case.csv")
    paper_rows = _read_csv(paper_path)
    comparison = build_comparison_rows(
        result_rows,
        paper_rows,
        paper_source=str(paper_path),
    )
    trace_rows = _read_csv(output_dir / "action_trace.csv")
    runtime = build_runtime_evidence_rows(
        result_rows=result_rows,
        trace_rows=trace_rows,
        ledger_rows=_read_csv(output_dir / "same_budget_ledger.csv"),
        aob_rows=_read_csv(output_dir / "aob_input_manifest.csv"),
        leakage_rows=_read_csv(output_dir / "anti_leakage_audit.csv"),
    )
    comparison_path = output_dir / "offline_paper_best_comparison.csv"
    markdown_path = output_dir / "offline_paper_best_comparison.md"
    runtime_path = output_dir / "runtime_evidence_case_summary.csv"
    _write_csv(comparison_path, comparison)
    _write_markdown(markdown_path, comparison)
    _write_csv(runtime_path, runtime)
    return comparison_path, markdown_path, runtime_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--paper-best",
        type=Path,
        default=Path("references/paper_reported_table2_best_by_case.csv"),
    )
    args = parser.parse_args(argv)
    write_reports(args.results_dir, args.paper_best)


if __name__ == "__main__":
    main()
