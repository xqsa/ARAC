"""Offline HCC output parsers kept separate from process execution."""

from __future__ import annotations

import csv
import re
from pathlib import Path

def _parse_hcc_evaluation_record(
    output_dir: Path,
    budget_limit: int | None = None,
) -> tuple[float, int]:
    final_error, fe_used, _optimizer_final_fe_used = (
        _parse_hcc_evaluation_record_with_optimizer_final_fe(
            output_dir,
            budget_limit=budget_limit,
        )
    )
    return final_error, fe_used


def _parse_hcc_evaluation_record_with_optimizer_final_fe(
    output_dir: Path,
    budget_limit: int | None = None,
) -> tuple[float, int, int]:
    records = sorted(Path(output_dir).rglob("evaluation_record.txt"))
    if not records:
        raise FileNotFoundError(f"missing HCC evaluation_record.txt under {output_dir}")
    text = records[-1].read_text(encoding="utf-8", errors="replace")
    final_match = re.search(
        r"Fin:\s*(?P<fe>[0-9.eE+-]+)\s+(?P<value>[0-9.eE+-]+)",
        text,
    )
    if not final_match:
        raise ValueError(f"could not parse final HCC error from {records[-1]}")
    optimizer_final_fe_used = _parse_hcc_budget_summary_final_fe(output_dir)
    if optimizer_final_fe_used is None:
        optimizer_final_fe_used = int(float(final_match.group("fe")))
    if budget_limit is not None:
        for checkpoint in re.finditer(
            r"^\s*(?P<fe>[0-9.eE+-]+)\s+(?P<value>[0-9.eE+-]+)",
            text,
            flags=re.MULTILINE,
        ):
            fe = int(float(checkpoint.group("fe")))
            if fe == budget_limit:
                return float(checkpoint.group("value")), fe, optimizer_final_fe_used

    return (
        float(final_match.group("value")),
        optimizer_final_fe_used,
        optimizer_final_fe_used,
    )


def _parse_hcc_budget_summary_final_fe(output_dir: Path) -> int | None:
    summary = _parse_hcc_budget_summary(output_dir)
    for field in ("fitness_record_fe", "optimizer_reported_fe"):
        if field in summary:
            return summary[field]
    return None


def _parse_hcc_budget_summary(output_dir: Path) -> dict[str, int]:
    summaries = sorted(Path(output_dir).rglob("*budget_summary.csv"))
    if not summaries:
        return {}
    with summaries[-1].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[-1]
    parsed: dict[str, int] = {}
    for field in (
        "fitness_record_fe",
        "optimizer_reported_fe",
        "global_phase_fe",
        "cc_phase_fe",
        "rescue_fe",
        "refresh_fe",
        "search_state_fe",
        "precision_probe_fe",
        "evidence_overlay_fe",
        "separable_continuation_fe",
        "overhead_fe",
    ):
        value = row.get(field)
        if value not in (None, ""):
            parsed[field] = int(float(value))
    parsed.setdefault("search_state_fe", 0)
    parsed.setdefault("precision_probe_fe", 0)
    parsed.setdefault("evidence_overlay_fe", 0)
    return parsed
