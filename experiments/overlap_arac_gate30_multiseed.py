"""Gate 30: tolerance-aware three-seed confirmation for overlap ARAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = (20260829, 20260830, 20260831)
MODES = ("conforming", "conflicting")
ATOL = 1.0e-9
RTOL = 1.0e-9
DEFAULT_SOURCES = (
    Path("artifacts/overlap_arac_gate29_screening/confirmation_fresh.json"),
    Path("artifacts/overlap_arac_gate30_multiseed/seed20260830/confirmation_fresh.json"),
    Path("artifacts/overlap_arac_gate30_multiseed/seed20260831/confirmation_fresh.json"),
)


def _arm(row: dict[str, object], mode: str) -> dict[str, object]:
    return next(arm for arm in row["arms"] if arm["mode"] == mode)


def practical_outcome(left: float, right: float) -> tuple[str, float, float]:
    """Compare minimization errors as left vs right with a numerical tie band."""

    gain = float(right) - float(left)
    tolerance = ATOL + RTOL * max(abs(float(left)), abs(float(right)))
    if gain > tolerance:
        return "win", gain, tolerance
    if gain < -tolerance:
        return "loss", gain, tolerance
    return "tie", gain, tolerance


def _load_sources(paths: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    payloads = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    if any(payload.get("schema_version") != "arac-overlap-gate29-screening-v1" for payload in payloads):
        raise RuntimeError("Gate30 source schema drifted")
    return payloads


def _comparison(rows: tuple[dict[str, object], ...], right_mode: str) -> dict[str, object]:
    records = []
    for row in rows:
        left = float(_arm(row, "proposal_neighborhood")["final_error"])
        right = float(_arm(row, right_mode)["final_error"])
        outcome, gain, tolerance = practical_outcome(left, right)
        records.append(
            {
                "cell": row["cell"],
                "outcome": outcome,
                "gain": gain,
                "tolerance": tolerance,
                "practical_gain": 0.0 if outcome == "tie" else gain,
            }
        )
    gains = np.asarray([record["gain"] for record in records], dtype=float)
    practical = np.asarray([record["practical_gain"] for record in records], dtype=float)
    counts = {outcome: sum(record["outcome"] == outcome for record in records) for outcome in ("win", "tie", "loss")}
    return {
        "count": len(records),
        "wins": counts["win"],
        "ties": counts["tie"],
        "losses": counts["loss"],
        "win_or_tie_rate": (counts["win"] + counts["tie"]) / len(records),
        "median_gain": float(np.median(gains)),
        "median_practical_gain": float(np.median(practical)),
        "records": records,
    }


def run_gate(*, sources: tuple[Path, ...] = DEFAULT_SOURCES) -> dict[str, object]:
    payloads = _load_sources(sources)
    rows = tuple(row for payload in payloads for row in payload["cells"])
    seeds = tuple(sorted({int(row["cell"]["seed"]) for row in rows}))
    identities = {
        (
            row["cell"]["mode"],
            row["cell"]["topology"],
            int(row["cell"]["overlap_budget"]),
            int(row["cell"]["seed"]),
        )
        for row in rows
    }
    proposal = _comparison(rows, "proposal_only")
    full = _comparison(rows, "full_context")
    proposal_by_mode = {
        mode: _comparison(tuple(row for row in rows if row["cell"]["mode"] == mode), "proposal_only")
        for mode in MODES
    }
    full_by_mode = {
        mode: _comparison(tuple(row for row in rows if row["cell"]["mode"] == mode), "full_context")
        for mode in MODES
    }
    integrity = all(
        row["checkpoint_parity"]
        and row["proposal_budget_parity"]
        and row["terminal_exact"]
        and row["strict_best"]
        and row["phase1_consumed_fes"] == 180_000
        and row["truth_shared_count"] == row["inferred_shared_count"]
        for row in rows
    )
    coordination_by_seed_mode = {
        f"{seed}:{mode}": max(
            float(_arm(row, "proposal_neighborhood")["total_coordination_gain"])
            for row in rows
            if row["cell"]["seed"] == seed and row["cell"]["mode"] == mode
        )
        for seed in SEEDS
        for mode in MODES
    }
    checks = {
        "cell_count_36": len(rows) == 36,
        "cell_identities_unique": len(identities) == 36,
        "seeds_exact": seeds == SEEDS,
        "source_integrity": integrity,
        "coordination_nonzero_each_seed_mode": all(value > 0.0 for value in coordination_by_seed_mode.values()),
        "proposal_win_tie_ge_0_60": proposal["win_or_tie_rate"] >= 0.60,
        "proposal_median_gain_positive": proposal["median_gain"] > 0.0,
        "proposal_mode_win_tie_ge_0_60": all(item["win_or_tie_rate"] >= 0.60 for item in proposal_by_mode.values()),
        "proposal_mode_median_gain_positive": all(item["median_gain"] > 0.0 for item in proposal_by_mode.values()),
        "full_noninferior_win_tie_ge_0_60": full["win_or_tie_rate"] >= 0.60,
        "full_noninferior_median_practical_nonnegative": full["median_practical_gain"] >= 0.0,
        "full_mode_noninferior_win_tie_ge_0_60": all(item["win_or_tie_rate"] >= 0.60 for item in full_by_mode.values()),
        "full_mode_median_practical_nonnegative": all(item["median_practical_gain"] >= 0.0 for item in full_by_mode.values()),
    }
    superiority = full["median_practical_gain"] > 0.0 and full["wins"] > full["losses"]
    return {
        "schema_version": "arac-overlap-gate30-multiseed-v1",
        "protocol": {"seeds": SEEDS, "cell_count": 36, "atol": ATOL, "rtol": RTOL},
        "coordination_by_seed_mode": coordination_by_seed_mode,
        "proposal_neighborhood_vs_proposal_only": proposal,
        "proposal_neighborhood_vs_full_context": full,
        "proposal_by_mode": proposal_by_mode,
        "full_by_mode": full_by_mode,
        "neighborhood_superiority_supported": superiority,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_arac_gate30_multiseed/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_passed": payload["gate_passed"],
                "gate_checks": payload["gate_checks"],
                "proposal_comparison": {
                    key: value
                    for key, value in payload["proposal_neighborhood_vs_proposal_only"].items()
                    if key != "records"
                },
                "full_comparison": {
                    key: value
                    for key, value in payload["proposal_neighborhood_vs_full_context"].items()
                    if key != "records"
                },
                "neighborhood_superiority_supported": payload["neighborhood_superiority_supported"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

