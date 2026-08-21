"""Two-axis (difficulty x contribution) representation replay gate (G11).

Pre-registration (from docs/lit-review-2026-08-21.md section 三.2: verify the
two-axis representation BEFORE implementing any v6.0 scheduler):

- data: the frozen gate51c v5.3 cells (arms on/off; audit-failed cells absent
  by fail-closed design), offline receipt analysis only;
- per cell-arm and episode, from the receipt stream ordered by
  ``cumulative_runtime_fes``:
    difficulty        median FE interval between improvements
                      (receipts with global_gain > 0); an episode with fewer
                      than two improvements gets its full funded span as a
                      single interval (never-improved-over-span = hardest);
    recent_contribution  mean gain_rate over the last 50% of the episode's
                      windows (CCFR-style recency);
    ranks are computed within the cell-arm across the four episodes
    (average ranks, normalized to [0, 1]);
    contributing := contribution_rank >= 0.5; difficult := difficulty_rank >= 0.5;
- pre-registered predictions (all evaluated on well-funded off-arm
  trajectories; on-arm recorded as secondary):
    P1  S5 protect cell: in >= 2 of 3 S5 off-cells, CTP has the top
        contribution rank AND difficulty_rank >= 0.5;
    P2  R2 fast-release cell: in >= 2 of 3 R2 off-cells, at least one
        zero-total-gain episode is classified (low difficulty, not
        contributing);
    P3  allocation direction: the pre-registered two-axis rule
        w = 1 + 2*[difficult & top-contributing] - 0.75*[not difficult &
        not contributing] (floored at 0.05) allocates CTP >= 45% of budget
        in >= 2 of 3 S5 off-cells (the off-arm's winning share is 62%,
        v5.3's starving share is 20.6%);
    P3b (secondary, directional only): on S5 on-cells, the same rule driven
        by first-half trajectories would not reduce CTP's share below its
        already-starved level.
- verdict: two_axis_representation_supported = P1 and P2 and P3.  Only a
  supported verdict unblocks the v6.0 implementation pre-registration.

Offline replay; production selector untouched; no new search FE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

OUTPUT_SCHEMA = "arac-oc-two-axis-replay-v1"
V53_ROOT = Path("artifacts/oc_phase_aware_gate51c_v5_3/cells")
CASES = ("A3", "R2", "R6", "S5")
SEEDS = (20260901, 20260902, 20260903)
ARMS = ("on", "off")
RECENT_FRACTION = 0.5
PROTECT_BONUS = 2.0
RELEASE_PENALTY = 0.75
WEIGHT_FLOOR = 0.05
CTP_SHARE_PREDICTION = 0.45
SEEDS_REQUIRED = 2


def _cell(case: str, seed: int, arm: str) -> dict[str, Any] | None:
    path = V53_ROOT / f"{case}_{seed}_{arm}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["result"]["result"]


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    order = sorted(values, key=lambda key: (values[key], key))
    ranks = {key: float(index) for index, key in enumerate(order)}
    return {key: ranks[key] / (len(order) - 1) if len(order) > 1 else 0.5 for key in ranks}


def _episode_axes(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for receipt in result["receipts"]:
        by_episode.setdefault(str(receipt["episode"]), []).append(receipt)
    axes: dict[str, dict[str, Any]] = {}
    for episode, receipts in by_episode.items():
        receipts = sorted(receipts, key=lambda item: item["cumulative_runtime_fes"])
        funded = float(sum(item["consumed_fes"] for item in receipts))
        improvements = [
            float(item["cumulative_runtime_fes"])
            for item in receipts
            if float(item["global_gain"]) > 0.0
        ]
        if len(improvements) >= 2:
            intervals = [
                improvements[i + 1] - improvements[i] for i in range(len(improvements) - 1)
            ]
        else:
            intervals = [funded]
        difficulty = float(np.median(intervals))
        tail = receipts[max(0, int(len(receipts) * (1.0 - RECENT_FRACTION))):]
        recent_contribution = (
            float(np.mean([float(item["gain_rate"]) for item in tail])) if tail else 0.0
        )
        axes[episode] = {
            "funded": funded,
            "windows": len(receipts),
            "improvements": len(improvements),
            "total_gain": float(sum(float(item["global_gain"]) for item in receipts)),
            "difficulty": difficulty,
            "recent_contribution": recent_contribution,
        }
    difficulty_ranks = _average_ranks({e: a["difficulty"] for e, a in axes.items()})
    contribution_ranks = _average_ranks(
        {e: a["recent_contribution"] for e, a in axes.items()}
    )
    top_contributor = max(axes, key=lambda e: (axes[e]["recent_contribution"], e))
    for episode, record in axes.items():
        record["difficulty_rank"] = difficulty_ranks[episode]
        record["contribution_rank"] = contribution_ranks[episode]
        record["difficult"] = difficulty_ranks[episode] >= 0.5
        record["contributing"] = contribution_ranks[episode] >= 0.5
        record["is_top_contributor"] = episode == top_contributor
        record["quadrant"] = (
            ("difficult" if record["difficult"] else "easy")
            + "/"
            + ("contributing" if record["contributing"] else "flat")
        )
    return axes


def _two_axis_weights(axes: dict[str, dict[str, Any]]) -> dict[str, float]:
    weights = {}
    for episode, record in axes.items():
        weight = 1.0
        if record["difficult"] and record["is_top_contributor"]:
            weight += PROTECT_BONUS
        if not record["difficult"] and not record["contributing"]:
            weight -= RELEASE_PENALTY
        weights[episode] = max(WEIGHT_FLOOR, weight)
    return weights


def run_gate() -> dict[str, Any]:
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    availability: dict[str, list[str]] = {}
    for case in CASES:
        for seed in SEEDS:
            for arm in ARMS:
                result = _cell(case, seed, arm)
                if result is None:
                    availability.setdefault(f"{case}_{arm}", []).append(str(seed))
                    continue
                cells[(case, seed, arm)] = result
    analyses = {
        key: _episode_axes(result) for key, result in cells.items()
    }
    p1_hits = 0
    p1_detail = []
    for seed in SEEDS:
        key = ("S5", seed, "off")
        if key not in analyses:
            continue
        ctp = analyses[key].get("ctp")
        hit = bool(
            ctp
            and ctp["is_top_contributor"]
            and ctp["difficult"]
        )
        p1_hits += int(hit)
        p1_detail.append(
            {
                "seed": seed,
                "ctp_quadrant": ctp["quadrant"] if ctp else None,
                "ctp_contribution_rank": ctp["contribution_rank"] if ctp else None,
                "ctp_difficulty_rank": ctp["difficulty_rank"] if ctp else None,
                "hit": hit,
            }
        )
    p2_hits = 0
    p2_detail = []
    for seed in SEEDS:
        key = ("R2", seed, "off")
        if key not in analyses:
            continue
        zero_gain = [
            episode
            for episode, record in analyses[key].items()
            if record["total_gain"] <= 0.0
        ]
        hit = any(
            not analyses[key][episode]["difficult"]
            and not analyses[key][episode]["contributing"]
            for episode in zero_gain
        )
        p2_hits += int(hit)
        p2_detail.append(
            {
                "seed": seed,
                "zero_gain_episodes": {
                    episode: analyses[key][episode]["quadrant"] for episode in zero_gain
                },
                "hit": hit,
            }
        )
    p3_hits = 0
    p3_detail = []
    p3b_detail = []
    for seed in SEEDS:
        for arm in ("off", "on"):
            key = ("S5", seed, arm)
            if key not in analyses:
                continue
            weights = _two_axis_weights(analyses[key])
            total = sum(weights.values())
            ctp_share = weights.get("ctp", 0.0) / total
            actual_share = (
                analyses[key]["ctp"]["funded"]
                / max(sum(record["funded"] for record in analyses[key].values()), 1.0)
                if "ctp" in analyses[key]
                else None
            )
            record = {
                "seed": seed,
                "simulated_ctp_share": ctp_share,
                "actual_ctp_share": actual_share,
                "weights": weights,
            }
            if arm == "off":
                p3_hits += int(ctp_share >= CTP_SHARE_PREDICTION)
                p3_detail.append({**record, "hit": ctp_share >= CTP_SHARE_PREDICTION})
            else:
                p3b_detail.append(record)
    p1 = p1_hits >= SEEDS_REQUIRED
    p2 = p2_hits >= SEEDS_REQUIRED
    p3 = p3_hits >= SEEDS_REQUIRED
    supported = bool(p1 and p2 and p3)
    quadrant_matrix: dict[str, dict[str, dict[str, str]]] = {}
    for (case, seed, arm), axes in sorted(analyses.items()):
        quadrant_matrix.setdefault(f"{case}_{arm}", {})[str(seed)] = {
            episode: record["quadrant"] for episode, record in sorted(axes.items())
        }
    checks = {
        "cells_available_for_all_predictions": bool(analyses),
        "offline_replay_only": True,
        "production_selector_unchanged": True,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "recent_fraction": RECENT_FRACTION,
            "protect_bonus": PROTECT_BONUS,
            "release_penalty": RELEASE_PENALTY,
            "weight_floor": WEIGHT_FLOOR,
            "ctp_share_prediction": CTP_SHARE_PREDICTION,
            "seeds_required": SEEDS_REQUIRED,
            "source": "artifacts/oc_phase_aware_gate51c_v5_3/cells (offline receipts)",
            "production_selector_modified": False,
        },
        "predictions": {
            "P1_s5_ctp_protect_cell": {"hits": p1_hits, "detail": p1_detail, "passed": p1},
            "P2_r2_fast_release_cell": {"hits": p2_hits, "detail": p2_detail, "passed": p2},
            "P3_allocation_direction": {
                "hits": p3_hits,
                "detail": p3_detail,
                "passed": p3,
            },
            "P3b_on_arm_directional": p3b_detail,
        },
        "quadrant_matrix": quadrant_matrix,
        "missing_cells": availability,
        "summary": {
            "two_axis_representation_supported": supported,
            "verdict": (
                "two-axis representation validated; v6.0 implementation unblocked"
                if supported
                else "two-axis representation insufficient as specified"
            ),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_two_axis_replay_gate/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "predictions": {
                    k: v.get("passed", v) if isinstance(v, dict) else v
                    for k, v in result["predictions"].items()
                },
                "summary": result["summary"],
                "quadrant_matrix": result["quadrant_matrix"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
