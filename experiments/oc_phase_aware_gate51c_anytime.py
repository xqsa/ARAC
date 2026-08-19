"""Post-process existing Gate 51c cells into fixed-grid anytime/AUC evidence."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.oc_phase_aware_gate51c import (
    ANYTIME_CHECKPOINTS,
    CASES,
    FRESH_SEEDS,
    STANDALONE_ARMS,
    TOTAL_FES,
    _anytime_points,
    _log_error_auc,
    _sample_anytime,
)

ROOT = Path("artifacts/oc_phase_aware_gate51c_v5_1")
OUTPUT = ROOT / "anytime_auc.json"


def _cell(root: Path, case: str, seed: int, arm: str) -> dict[str, object]:
    path = root / "cells" / f"{case}_{seed}_{arm}.json"
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def build(root: Path = ROOT) -> dict[str, object]:
    cells: dict[tuple[str, int, str], dict[str, object]] = {}
    for case in CASES:
        for seed in FRESH_SEEDS:
            phase1 = json.loads(
                (root / "phase1" / f"{case}_{seed}.json").read_text(encoding="utf-8")
            )["checkpoint"]
            checkpoint_error = float(phase1["incumbent_error"])
            for arm in (*STANDALONE_ARMS, "on", "off"):
                row = _cell(root, case, seed, arm)
                result = row["result"]
                points = _anytime_points(result, checkpoint_error)
                cells[(case, seed, arm)] = {
                    "anytime": _sample_anytime(points),
                    "log_error_auc": _log_error_auc(points),
                    "checkpoint_error": checkpoint_error,
                }

    per_case: dict[str, object] = {}
    for case in CASES:
        rows: list[dict[str, object]] = []
        for seed in FRESH_SEEDS:
            standalone = {
                arm: cells[(case, seed, arm)] for arm in STANDALONE_ARMS
            }
            best_arm = min(
                STANDALONE_ARMS,
                key=lambda arm: float(standalone[arm]["log_error_auc"]),
            )
            on = cells[(case, seed, "on")]
            best = standalone[best_arm]
            rows.append(
                {
                    "seed": seed,
                    "best_standalone_by_auc": best_arm,
                    "on_anytime": on["anytime"],
                    "best_anytime": best["anytime"],
                    "on_vs_best_anytime": {
                        key: float(on["anytime"][key]) / max(float(best["anytime"][key]), 1e-300)
                        for key in on["anytime"]
                    },
                    "on_log_error_auc": on["log_error_auc"],
                    "best_log_error_auc": best["log_error_auc"],
                    "on_vs_best_auc": float(on["log_error_auc"]) / max(float(best["log_error_auc"]), 1e-300),
                }
            )
        per_case[case] = rows
    return {
        "schema_version": "arac-oc-gate51c-anytime-auc-v1",
        "source": str(root),
        "grid_total_fes": list(ANYTIME_CHECKPOINTS),
        "phase1_fes": 180_000,
        "total_fes": TOTAL_FES,
        "per_case": per_case,
    }


def main() -> int:
    payload = build()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
