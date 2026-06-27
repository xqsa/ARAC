from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path


def _load_debug_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "debug_relation_dispatch.py"
    spec = importlib.util.spec_from_file_location("debug_relation_dispatch_for_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_action_decision(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "run_id",
        "problem_id",
        "relation_id",
        "group_left",
        "group_right",
        "shared_vars_count",
        "overlap_strength",
        "delta_signal",
        "rank_signal",
        "relation_action_name",
        "canonical_action_name",
        "action_family",
        "confidence",
        "trigger_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_find_latest_action_decision_csv_uses_mtime(tmp_path: Path) -> None:
    module = _load_debug_module()
    older = tmp_path / "old" / "action_decision.csv"
    newer = tmp_path / "new" / "action_decision.csv"
    _write_action_decision(older, [])
    _write_action_decision(newer, [])
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    assert module.find_latest_action_decision_csv(tmp_path) == newer


def test_summarize_rows_counts_histograms_and_means() -> None:
    module = _load_debug_module()
    rows = [
        {
            "group_left": "0",
            "overlap_strength": "1.0",
            "relation_action_name": "coordinate",
            "canonical_action_name": "allow_beneficial_coordination",
        },
        {
            "group_left": "0",
            "overlap_strength": "3.0",
            "relation_action_name": "coordinate",
            "canonical_action_name": "allow_beneficial_coordination",
        },
        {
            "group_left": "1",
            "overlap_strength": "2.0",
            "relation_action_name": "fallback",
            "canonical_action_name": "conservative_no_action",
        },
    ]

    summary = module.summarize_rows(rows)

    assert summary.total_relations == 3
    assert summary.action_counts == {
        "allow_beneficial_coordination": 2,
        "conservative_no_action": 1,
    }
    assert summary.group_left_counts == {"0": 2, "1": 1}
    assert summary.overlap_strength_mean_by_action == {
        "allow_beneficial_coordination": 2.0,
        "conservative_no_action": 2.0,
    }


def test_plot_action_frequency_writes_png(tmp_path: Path) -> None:
    module = _load_debug_module()
    summary = module.RelationDispatchSummary(
        total_relations=3,
        action_counts={"coordinate": 2, "fallback": 1},
        group_left_counts={"0": 2, "1": 1},
        overlap_strength_mean_by_action={"coordinate": 2.0, "fallback": 2.0},
    )
    output_path = tmp_path / "action_frequency.png"

    module.plot_action_frequency(summary, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
