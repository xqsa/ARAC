from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

RUN_ID = "exp_004_hcc_main_historical_result_recovery"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HCC_RESULT_ROOT = Path("E:/HCC-main/HCC_SRC/result")
PAPER_REPORTED_CSV = ROOT / "references" / "paper_reported_table2_hcc_es.csv"
CASE_PATTERN = re.compile(r"^(?P<family>[ESRA])(?P<idx>[1-6])$")
SEED_PATTERN = re.compile(r"^seed-(?P<seed>\d+)$", re.IGNORECASE)
TARGETED_CASES = ("S4", "S5", "R4", "R5", "R6", "A1", "A2", "A3", "A4", "A5", "A6")
NEAR_TIE_REL_GAIN_ABS_PCT = 1.0


@dataclass(frozen=True)
class HistoricalRecord:
    source_file: Path
    source_root: Path
    experiment_label: str
    problem_id: str
    base_function: str
    seed: str
    fe_used: int
    final_error: float
    time_seconds: float
    parse_status: str


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paper_rows_by_problem() -> dict[str, dict[str, str]]:
    with PAPER_REPORTED_CSV.open(newline="", encoding="utf-8") as handle:
        return {row["problem_id"]: row for row in csv.DictReader(handle)}


def _parse_record_text(text: str) -> tuple[float, int, float]:
    final_match = re.search(
        r"Fin:\s*(?P<fe>[0-9.eE+-]+)\s+(?P<value>[0-9.eE+-]+)",
        text,
    )
    if not final_match:
        raise ValueError("missing Fin row")
    time_match = re.search(
        r"Run Time:\s*(?P<time>[0-9.eE+-]+)",
        text,
    )
    return (
        float(final_match.group("value")),
        int(float(final_match.group("fe"))),
        float(time_match.group("time")) if time_match else float("nan"),
    )


def _metadata_from_path(path: Path, result_root: Path) -> tuple[str, str, str, str]:
    relative_parts = path.relative_to(result_root).parts
    experiment_label = relative_parts[0] if relative_parts else ""
    problem_id = ""
    seed = ""
    for part in relative_parts:
        case_match = CASE_PATTERN.match(part.upper())
        if case_match:
            problem_id = part.upper()
        seed_match = SEED_PATTERN.match(part)
        if seed_match:
            seed = seed_match.group("seed")
    base_function = path.parent.name
    return experiment_label, problem_id, base_function, seed


def _iter_historical_records(result_root: Path) -> list[HistoricalRecord]:
    records: list[HistoricalRecord] = []
    for record_file in sorted(result_root.rglob("evaluation_record.txt")):
        experiment_label, problem_id, base_function, seed = _metadata_from_path(
            record_file,
            result_root,
        )
        try:
            final_error, fe_used, time_seconds = _parse_record_text(
                record_file.read_text(encoding="utf-8", errors="replace")
            )
            parse_status = "parsed"
        except ValueError:
            final_error = float("nan")
            fe_used = 0
            time_seconds = float("nan")
            parse_status = "parse_failed"
        records.append(
            HistoricalRecord(
                source_file=record_file,
                source_root=result_root,
                experiment_label=experiment_label,
                problem_id=problem_id,
                base_function=base_function,
                seed=seed,
                fe_used=fe_used,
                final_error=final_error,
                time_seconds=time_seconds,
                parse_status=parse_status,
            )
        )
    return records


def _inventory_rows(records: list[HistoricalRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "run_id": RUN_ID,
                "source_file": str(record.source_file),
                "source_root": str(record.source_root),
                "experiment_label": record.experiment_label,
                "problem_id": record.problem_id,
                "base_function": record.base_function,
                "seed": record.seed,
                "fe_used": str(record.fe_used),
                "final_error": f"{record.final_error:.6e}",
                "time_seconds": f"{record.time_seconds:.6f}",
                "parse_status": record.parse_status,
                "budget_match_status": _budget_match_status(record.fe_used),
                "protocol_match_status": (
                    "aob_case_detected" if record.problem_id else "unknown_case"
                ),
                "usable_for_runtime_dispatch": "0",
                "usable_for_offline_evaluation": "1" if record.parse_status == "parsed" else "0",
                "runtime_dispatch_allowed": "0",
                "notes": "historical HCC-main artifact; offline evidence only",
            }
        )
    return rows


def _comparison_rows(records: list[HistoricalRecord]) -> list[dict[str, object]]:
    paper = _paper_rows_by_problem()
    rows: list[dict[str, object]] = []
    for record in records:
        if record.problem_id not in paper or record.parse_status != "parsed":
            continue
        paper_row = paper[record.problem_id]
        reported_mean = float(paper_row["reported_mean"])
        rows.append(
            {
                "run_id": RUN_ID,
                "source_file": str(record.source_file),
                "experiment_label": record.experiment_label,
                "problem_id": record.problem_id,
                "base_function": record.base_function,
                "seed": record.seed,
                "fe_used": str(record.fe_used),
                "budget_match_status": _budget_match_status(record.fe_used),
                "historical_final_error": f"{record.final_error:.6e}",
                "paper_reported_mean": paper_row["reported_mean"],
                "paper_reported_std": paper_row["reported_std"],
                "better_than_paper_reported": "1" if record.final_error < reported_mean else "0",
                "offline_error_delta_vs_paper": f"{(record.final_error - reported_mean):.6e}",
                "runtime_dispatch_allowed": "0",
                "comparison_role": "offline_historical_hcc_main_vs_paper_reported",
            }
        )
    return rows


def _budget_match_status(fe_used: int) -> str:
    if 2_950_000 <= fe_used <= 3_050_000:
        return "near_paper_3m_fe"
    if fe_used <= 0:
        return "unknown_fe"
    return "non_3m_fe"


def _targeted_case_diagnostic_rows(
    comparison_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows_by_case: dict[str, list[dict[str, object]]] = {case: [] for case in TARGETED_CASES}
    for row in comparison_rows:
        problem_id = str(row["problem_id"])
        if problem_id in rows_by_case:
            rows_by_case[problem_id].append(row)

    paper = _paper_rows_by_problem()
    diagnostics: list[dict[str, object]] = []
    for problem_id in TARGETED_CASES:
        case_rows = rows_by_case[problem_id]
        paper_row = paper[problem_id]
        paper_mean = float(paper_row["reported_mean"])
        paper_std = float(paper_row["reported_std"])
        values = [float(row["historical_final_error"]) for row in case_rows]
        wins = sum(1 for value in values if value < paper_mean)
        best_row = min(
            case_rows,
            key=lambda row: float(row["historical_final_error"]),
            default=None,
        )
        if best_row is None:
            best_error = float("nan")
            mean_error = float("nan")
            sample_std = float("nan")
            rel_gain_pct = float("nan")
            best_experiment_label = ""
            best_fe_used = ""
            best_seed = ""
            relation_label = "missing_historical_evidence"
        else:
            best_error = float(best_row["historical_final_error"])
            mean_error = sum(values) / len(values)
            sample_std = _sample_std(values)
            rel_gain_pct = (paper_mean - best_error) / max(abs(paper_mean), 1e-12) * 100.0
            best_experiment_label = str(best_row["experiment_label"])
            best_fe_used = str(best_row["fe_used"])
            best_seed = str(best_row["seed"])
            relation_label = _target_relation_label(problem_id, rel_gain_pct)
        diagnostics.append(
            {
                "run_id": RUN_ID,
                "problem_id": problem_id,
                "base_function": paper_row["base_function"],
                "overlap_gamma": paper_row["overlap_gamma"],
                "historical_rows": str(len(case_rows)),
                "wins_vs_paper_mean": str(wins),
                "best_historical_final_error": _format_float(best_error),
                "mean_historical_final_error": _format_float(mean_error),
                "std_historical_final_error": _format_float(sample_std),
                "paper_reported_mean": paper_row["reported_mean"],
                "paper_reported_std": paper_row["reported_std"],
                "rel_gain_pct": _format_pct(rel_gain_pct),
                "best_historical_experiment_label": best_experiment_label,
                "best_historical_fe_used": best_fe_used,
                "best_historical_seed": best_seed,
                "relation_label": relation_label,
                "runtime_dispatch_allowed": "0",
                "diagnostic_role": "offline_targeted_best_row_vs_paper_reported_mean",
            }
        )
    return diagnostics


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _target_relation_label(problem_id: str, rel_gain_pct: float) -> str:
    if math.isnan(rel_gain_pct):
        return "missing_historical_evidence"
    if rel_gain_pct > 0.0:
        return "better_than_paper_mean"
    if problem_id.startswith("A") and abs(rel_gain_pct) <= NEAR_TIE_REL_GAIN_ABS_PCT:
        return "near_tie_pending_25_run"
    return "worse_than_paper_mean"


def _format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6e}"


def _format_pct(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.2f}"


def _write_audit(
    output_dir: Path,
    records: list[HistoricalRecord],
    comparison_rows: list[dict[str, object]],
    hcc_result_root: Path,
) -> None:
    parsed = [record for record in records if record.parse_status == "parsed"]
    detected_cases = sorted({record.problem_id for record in parsed if record.problem_id})
    better_rows = [row for row in comparison_rows if row["better_than_paper_reported"] == "1"]
    wins_by_case: dict[str, int] = {}
    for row in better_rows:
        wins_by_case[str(row["problem_id"])] = wins_by_case.get(str(row["problem_id"]), 0) + 1
    best_by_case: dict[str, dict[str, object]] = {}
    for row in comparison_rows:
        current = best_by_case.get(str(row["problem_id"]))
        if current is None or float(row["historical_final_error"]) < float(
            current["historical_final_error"]
        ):
            best_by_case[str(row["problem_id"])] = row

    lines = [
        "# HCC-main Historical Results Audit",
        "",
        "Date: 2026-06-26",
        "Executor: Codex",
        f"Source root: {hcc_result_root}",
        f"Evaluation records discovered: {len(records)}",
        f"Parsed records: {len(parsed)}",
        f"AOB cases detected: {', '.join(detected_cases) if detected_cases else 'none'}",
        f"Offline rows better than paper reported mean: {len(better_rows)}",
        "",
        "## Cases With Historical Rows Better Than Paper Reported Mean",
        "",
    ]
    if wins_by_case:
        lines.extend(
            f"- {problem_id}: {count}"
            for problem_id, count in sorted(wins_by_case.items())
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Best Historical Row By Case",
            "",
            "| case | best historical final error | paper reported mean | seed | experiment label |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for problem_id, row in sorted(best_by_case.items()):
        lines.append(
            "| "
            f"{problem_id} | "
            f"{row['historical_final_error']} | "
            f"{row['paper_reported_mean']} | "
            f"{row['seed']} | "
            f"{row['experiment_label']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime Boundary",
            "",
            "These artifacts are historical HCC-main evidence. They are usable for offline",
            "evaluation and provenance checks only. They must not enter runtime dispatch.",
            "",
        ]
    )
    output_dir.joinpath("hcc_main_historical_results_audit.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_hcc_main_historical_result_recovery(
    output_dir: Path | str = Path("results/exp_004_hcc_main_historical_result_recovery"),
    hcc_result_root: Path | str = DEFAULT_HCC_RESULT_ROOT,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_root = Path(hcc_result_root)
    records = _iter_historical_records(result_root)
    inventory = _inventory_rows(records)
    comparison = _comparison_rows(records)
    targeted_diagnostics = _targeted_case_diagnostic_rows(comparison)
    _write_csv(
        output / "hcc_main_historical_result_inventory.csv",
        inventory,
        [
            "run_id",
            "source_file",
            "source_root",
            "experiment_label",
            "problem_id",
            "base_function",
            "seed",
            "fe_used",
            "final_error",
            "time_seconds",
            "parse_status",
            "budget_match_status",
            "protocol_match_status",
            "usable_for_runtime_dispatch",
            "usable_for_offline_evaluation",
            "runtime_dispatch_allowed",
            "notes",
        ],
    )
    _write_csv(
        output / "hcc_main_vs_paper_reported_comparison.csv",
        comparison,
        [
            "run_id",
            "source_file",
            "experiment_label",
            "problem_id",
            "base_function",
            "seed",
            "fe_used",
            "budget_match_status",
            "historical_final_error",
            "paper_reported_mean",
            "paper_reported_std",
            "better_than_paper_reported",
            "offline_error_delta_vs_paper",
            "runtime_dispatch_allowed",
            "comparison_role",
        ],
    )
    _write_csv(
        output / "hcc_main_targeted_case_diagnostics.csv",
        targeted_diagnostics,
        [
            "run_id",
            "problem_id",
            "base_function",
            "overlap_gamma",
            "historical_rows",
            "wins_vs_paper_mean",
            "best_historical_final_error",
            "mean_historical_final_error",
            "std_historical_final_error",
            "paper_reported_mean",
            "paper_reported_std",
            "rel_gain_pct",
            "best_historical_experiment_label",
            "best_historical_fe_used",
            "best_historical_seed",
            "relation_label",
            "runtime_dispatch_allowed",
            "diagnostic_role",
        ],
    )
    _write_audit(output, records, comparison, result_root)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover HCC-main historical result artifacts.")
    parser.add_argument(
        "--output-dir",
        default="results/exp_004_hcc_main_historical_result_recovery",
    )
    parser.add_argument(
        "--hcc-result-root",
        default=str(DEFAULT_HCC_RESULT_ROOT),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_hcc_main_historical_result_recovery(
        output_dir=Path(args.output_dir),
        hcc_result_root=Path(args.hcc_result_root),
    )


if __name__ == "__main__":
    main()
