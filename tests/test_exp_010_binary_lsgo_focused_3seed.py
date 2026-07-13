import csv
import json
from pathlib import Path

from experiments.exp_010_binary_lsgo_focused_3seed.run import (
    FOCUSED_PROBLEM_IDS,
    LANES,
    OPTIMIZER_SEEDS,
    run_focused_pilot,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_fixed_protocol_writes_45_same_budget_rows(tmp_path: Path):
    output = run_focused_pilot(tmp_path / "pilot", total_fes=40)
    rows = read_csv(output / "run_results.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 45
    assert tuple(FOCUSED_PROBLEM_IDS) == (
        "BLSGO-F07",
        "BLSGO-F08",
        "BLSGO-F09",
        "BLSGO-F14",
        "BLSGO-F15",
    )
    assert tuple(OPTIMIZER_SEEDS) == (20260713, 20260714, 20260715)
    assert tuple(LANES) == (
        "native_baseline",
        "arac_policy",
        "shuffled_evidence_negative_control",
    )
    assert {(row["problem_id"], int(row["optimizer_seed"])) for row in rows} == {
        (problem_id, seed)
        for problem_id in FOCUSED_PROBLEM_IDS
        for seed in OPTIMIZER_SEEDS
    }
    assert {row["total_fe"] for row in rows} == {"40"}
    assert {row["same_budget_violation"] for row in rows} == {"0"}
    assert manifest["execution_count"] == 45
