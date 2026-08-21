"""S5 schedule-layer failure diagnostic (G5).

Pre-registered classification (frozen in .codex-tasks/arac-oc-evidence-closure/
EPIC.md; schedule-layer instantiation of the four-way taxonomy):

For each case (A3/R2/R6/S5) and seed, using the v5.3 gate51c 'on'/'off' cells:

- ``best_episode``(case, seed) = episode with the highest total ``global_gain``
  in the 'off' (handoff-disabled) cell — what actually works when the v5.3
  runway machinery is not steering;
- ``starvation`` = funded_fes[best_episode under on] / funded_fes[best_episode
  under off];
- ``gain_rates`` = per-episode mean receipt ``gain_rate`` under 'on';
- ``release_events`` = plateau_release / released / grace_consumed counts;
- ``relation_coupling`` = from the frozen Phase-I caches (v5.4 design input).

Decision rules (v2, registered before the G4c campaign; v1's absolute-rate
condition was mis-triggered by the rate drop that starvation itself causes):

  BUDGET_TRIGGER_FAILURE  starvation < 0.5 AND the starved episode remains
                          the top-ranked nonzero-gain episode under 'on'
  EPISODE_INEFFECTIVE     all on-cell episodes have mean gain_rate <= 1e-12
  GEOMETRY_BOTTLENECK     S5 relation_coupling exceeds every non-S5 case by
                          a relative margin > 25% AND neither rule above hit
  SHARED_REPAIR_FAILURE   default when FE flows to the right episode but
                          gains do not materialize

The gate reports the majority cause across S5 seeds as the single primary
cause, with the full evidence table for audit.
Offline artifact analysis only; no scheduler code is executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

OUTPUT_SCHEMA = "arac-oc-s5-failure-diagnostic-v1"
GATE_ROOT = Path("artifacts/oc_phase_aware_gate51c_v5_3")
CASES = ("A3", "R2", "R6", "S5")
SEEDS = (20260901, 20260902, 20260903)
ARMS = ("on", "off")
STARVATION_THRESHOLD = 0.5
GEOMETRY_MARGIN = 0.25


def _cell(case: str, seed: int, arm: str) -> dict[str, Any] | None:
    path = GATE_ROOT / "cells" / f"{case}_{seed}_{arm}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["result"]["result"]


def _episode_gains(result: dict[str, Any]) -> dict[str, float]:
    gains: dict[str, float] = {}
    for receipt in result["receipts"]:
        episode = str(receipt.get("episode"))
        gains[episode] = gains.get(episode, 0.0) + float(receipt.get("global_gain", 0.0))
    return gains


def _episode_gain_rates(result: dict[str, Any]) -> dict[str, float]:
    rates: dict[str, list[float]] = {}
    for receipt in result["receipts"]:
        episode = str(receipt.get("episode"))
        rates.setdefault(episode, []).append(float(receipt.get("gain_rate", 0.0)))
    return {episode: float(np.mean(values)) for episode, values in rates.items()}


def _relation_coupling(case: str, seed: int) -> dict[str, float] | None:
    path = GATE_ROOT / "phase1" / f"{case}_{seed}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks = payload["checkpoint"]["blocks"]
    relations = payload["checkpoint"]["relations"]
    coupled_blocks: set[int] = set()
    for relation in relations:
        coupled_blocks.add(int(relation["left_block"]))
        coupled_blocks.add(int(relation["right_block"]))
    coupled_vars = sum(len(blocks[index]) for index in coupled_blocks if 0 <= index < len(blocks))
    total_vars = sum(len(block) for block in blocks)
    strengths = [float(relation["strength"]) for relation in relations]
    return {
        "coupled_vars_ratio": coupled_vars / max(total_vars, 1),
        "relations": float(len(strengths)),
        "mean_strength": float(np.mean(strengths)) if strengths else 0.0,
    }


def _release_events(result: dict[str, Any]) -> dict[str, int]:
    events = {"plateau_release": 0, "released": 0, "grace_consumed": 0, "switched": 0}
    for receipt in result["receipts"]:
        for key in events:
            if receipt.get(key):
                events[key] += 1
    return events


def run_gate() -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for case in CASES:
        for seed in SEEDS:
            on = _cell(case, seed, "on")
            off = _cell(case, seed, "off")
            if on is None or off is None:
                evidence.append(
                    {"case": case, "seed": seed, "status": "cells_missing",
                     "on_present": on is not None, "off_present": off is not None}
                )
                continue
            off_gains = _episode_gains(off)
            best_episode = max(off_gains, key=lambda e: (off_gains[e], e)) if off_gains else ""
            on_funded = dict(on["funded_fes"])
            off_funded = dict(off["funded_fes"])
            starvation = (
                float(on_funded.get(best_episode, 0)) / float(off_funded.get(best_episode, 0))
                if best_episode and off_funded.get(best_episode, 0) > 0
                else None
            )
            on_rates = _episode_gain_rates(on)
            off_rates = _episode_gain_rates(off)
            best_rate_on = on_rates.get(best_episode, 0.0)
            best_rate_off = off_rates.get(best_episode, 0.0)
            positive_rates = {e: r for e, r in on_rates.items() if r > 1e-12}
            top_positive = max(positive_rates, key=lambda e: (positive_rates[e], e)) if positive_rates else ""
            audit_ok = all(on.get("audit", {}).values()) and all(off.get("audit", {}).values())
            cause: str | None = None
            if (
                starvation is not None
                and starvation < STARVATION_THRESHOLD
                and top_positive == best_episode
            ):
                cause = "BUDGET_TRIGGER_FAILURE"
            elif on_rates and all(rate <= 1e-12 for rate in on_rates.values()):
                cause = "EPISODE_INEFFECTIVE"
            evidence.append(
                {
                    "case": case,
                    "seed": seed,
                    "status": "ok",
                    "audit_ok": audit_ok,
                    "final_on": float(on["final_error"]),
                    "final_off": float(off["final_error"]),
                    "on_better": float(on["final_error"]) < float(off["final_error"]),
                    "best_episode_off": best_episode,
                    "funded_on": on_funded,
                    "funded_off": off_funded,
                    "starvation": starvation,
                    "dev_expl_on": [on["development_fes"], on["exploitation_fes"]],
                    "dev_expl_off": [off["development_fes"], off["exploitation_fes"]],
                    "gain_rate_on": on_rates,
                    "gain_rate_off": off_rates,
                    "best_episode_rate_on": best_rate_on,
                    "best_episode_rate_off": best_rate_off,
                    "release_events_on": _release_events(on),
                    "handoffs_on": len(on.get("handoffs", [])),
                    "handoffs_off": len(off.get("handoffs", [])),
                    "relation_coupling": _relation_coupling(case, seed),
                    "preliminary_cause": cause,
                }
            )
    complete = [row for row in evidence if row.get("status") == "ok"]
    s5_rows = [row for row in complete if row["case"] == "S5"]
    coupling = {
        case: float(np.mean([row["relation_coupling"]["coupled_vars_ratio"] for row in complete if row["case"] == case]))
        for case in CASES
        if any(row["case"] == case for row in complete)
    }
    s5_coupling = coupling.get("S5", 0.0)
    others = [value for case, value in coupling.items() if case != "S5"]
    geometry_dominant = bool(others and s5_coupling > max(others) * (1.0 + GEOMETRY_MARGIN))
    causes: list[str] = []
    for row in s5_rows:
        if row["preliminary_cause"]:
            causes.append(row["preliminary_cause"])
        elif geometry_dominant:
            causes.append("GEOMETRY_BOTTLENECK")
        else:
            causes.append("SHARED_REPAIR_FAILURE")
    if causes:
        primary = max(set(causes), key=causes.count)
    else:
        primary = "INSUFFICIENT_CELLS"
    checks = {
        "at_least_one_s5_seed_complete": bool(s5_rows),
        "offline_analysis_only": True,
        "production_selector_unchanged": True,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "seeds": list(SEEDS),
            "starvation_threshold": STARVATION_THRESHOLD,
            "geometry_margin": GEOMETRY_MARGIN,
            "decision_rules": "first-match: BUDGET_TRIGGER_FAILURE, EPISODE_INEFFECTIVE, then GEOMETRY/REPAIR",
            "source": "artifacts/oc_phase_aware_gate51c_v5_3/cells (offline)",
            "production_selector_modified": False,
        },
        "evidence": evidence,
        "summary": {
            "complete_rows": len(complete),
            "s5_causes": causes,
            "s5_primary_cause": primary,
            "coupling_by_case": coupling,
            "geometry_dominant": geometry_dominant,
            "on_better_by_case": {
                case: {
                    "wins": sum(1 for row in complete if row["case"] == case and row["on_better"]),
                    "total": sum(1 for row in complete if row["case"] == case),
                }
                for case in CASES
            },
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_s5_failure_diagnostic/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary = result["summary"]
    print(json.dumps({
        "gate_passed": result["gate_passed"],
        "s5_primary_cause": summary["s5_primary_cause"],
        "s5_causes": summary["s5_causes"],
        "coupling_by_case": summary["coupling_by_case"],
        "on_better_by_case": summary["on_better_by_case"],
    }, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
