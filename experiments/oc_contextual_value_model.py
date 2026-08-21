"""Contextual value model shadow (G9) — unlocked by the G8 pass.

Pre-registration (this gate runs only because the G8 pairwise ranker passed
its held-out rule; frozen in .codex-tasks/arac-oc-evidence-closure/EPIC.md):

- prediction target: per-(context, arm) ``end_to_end_gain / checkpoint_error``
  (normalized end-to-end gain; NOT raw action gain);
- model: ridge regression (closed form, L2 = 1.0) on features
  [arm one-hot (5)] + [arm one-hot x context features] — the same ex-ante
  feature family as G8, no model selection;
- grouped cross-fitting: leave-one-seed-out over the fresh seeds;
- selection: argmax predicted value per held-out context; reported against
  the G8 ranker (from its artifact), the production baseline (G2 persistent
  scenario), the best fixed arm, random and the oracle;
- this model may only ever enter a challenger lane; the production selector
  stays untouched regardless of the outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.oc_lagged_coupling_shadow import ARMS, validate_input
from experiments.oc_pairwise_ranking_shadow import (
    BEST_FIXED_ARM,
    CHANCE_HIT_RATE,
    PRODUCTION_BASELINE,
    _bootstrap_ci_low,
    _context_features,
    _design,
    _kappa,
)

INPUT_SCHEMA = "arac-oc-action-semantic-gate-v3"
G8_ARTIFACT = Path("artifacts/oc_pairwise_ranking_shadow/confirmation.json")
OUTPUT_SCHEMA = "arac-oc-contextual-value-model-v1"
RIDGE_L2 = 1.0


def _features(x: np.ndarray) -> np.ndarray:
    rows = []
    for context_features in x:
        block = list(context_features)
        row = []
        for arm_index in range(len(ARMS)):
            one_hot = [0.0] * len(ARMS)
            one_hot[arm_index] = 1.0
            interactions = [value * flag for value in block for flag in one_hot]
            row.append(one_hot + interactions)
        rows.append(row)
    return np.asarray(rows, dtype=float)  # (contexts, arms, features)


def run_gate(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validate_input(payload)
    if not G8_ARTIFACT.exists():
        raise RuntimeError("G8 artifact missing; G9 is only registered after a G8 pass")
    g8 = json.loads(G8_ARTIFACT.read_text(encoding="utf-8"))
    if g8["summary"]["supports_arm_ranking"] is not True:
        raise RuntimeError("G8 did not pass; G9 protocol forbids running it")
    x, gains, seeds, contexts, topologies = _design(payload)
    errors = np.asarray([float(c["checkpoint_error"]) for c in contexts])
    normalized = gains / errors[:, None]
    oracle_index = gains.argmax(axis=1)
    design = _features(x)
    production_rows: dict[tuple, str] = {}
    production_payload = json.loads(PRODUCTION_BASELINE.read_text(encoding="utf-8"))
    for row in production_payload["rows"]:
        key = (str(row["key"][0]), int(row["key"][1]), int(row["key"][2]), str(row["key"][3]))
        production_rows[key] = row["persistent_mapped_arm"] or "owner_control"

    predicted: list[str] = []
    hits: list[bool] = []
    regrets: list[float] = []
    prod_diffs: list[float] = []
    fixed_diffs: list[float] = []
    for held_out in sorted(set(int(s) for s in seeds)):
        train_mask = seeds != held_out
        test_mask = seeds == held_out
        train_x = design[train_mask].reshape(-1, design.shape[2])
        train_y = normalized[train_mask].reshape(-1)
        gram = train_x.T @ train_x + RIDGE_L2 * np.eye(train_x.shape[1])
        weights = np.linalg.solve(gram, train_x.T @ train_y)
        for local_index, global_index in enumerate(np.flatnonzero(test_mask)):
            scores = design[global_index] @ weights
            arm_index = int(np.argmax(scores))
            arm = ARMS[arm_index]
            context = contexts[global_index]
            key = (
                str(context["topology"]),
                int(context["overlap_budget"]),
                int(context["seed"]),
                str(context["mode"]),
            )
            predicted.append(arm)
            hits.append(arm_index == int(oracle_index[global_index]))
            regrets.append(
                float(gains[global_index, oracle_index[global_index]] - gains[global_index, arm_index])
            )
            prod_arm = production_rows[key]
            prod_diffs.append(
                float(gains[global_index, arm_index] - gains[global_index, ARMS.index(prod_arm)])
            )
            fixed_diffs.append(
                float(
                    gains[global_index, arm_index]
                    - gains[global_index, ARMS.index(BEST_FIXED_ARM)]
                )
            )
    hit_rate = float(np.mean([float(h) for h in hits]))
    prod_ci_low = _bootstrap_ci_low(prod_diffs)
    fixed_ci_low = _bootstrap_ci_low(fixed_diffs)
    checks = {
        "context_count_60": len(contexts) == 60,
        "g8_pass_confirmed": True,
        "folds_cover_all_contexts": len(predicted) == len(contexts),
        "production_selector_unchanged": True,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "input_schema": INPUT_SCHEMA,
            "input_artifact": str(input_path),
            "g8_artifact": str(G8_ARTIFACT),
            "target": "end_to_end_gain / checkpoint_error",
            "model": "ridge regression (L2=1.0), arm one-hot + arm x context interactions",
            "cv": "leave-one-seed-out",
            "deployment_rule": "challenger lane only; production selector unchanged",
            "production_selector_modified": False,
        },
        "summary": {
            "held_out_rows": len(predicted),
            "top1_hit_rate": hit_rate,
            "chance_hit_rate": CHANCE_HIT_RATE,
            "cohens_kappa_vs_oracle": _kappa(predicted, [ARMS[int(i)] for i in oracle_index]),
            "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
            "paired_gain_vs_production": float(np.mean(prod_diffs)),
            "paired_gain_vs_production_ci95_low": prod_ci_low,
            "paired_gain_vs_best_fixed": float(np.mean(fixed_diffs)),
            "paired_gain_vs_best_fixed_ci95_low": fixed_ci_low,
            "beats_production_baseline": bool(prod_ci_low > 0.0),
            "beats_best_fixed_arm": bool(fixed_ci_low > 0.0),
            "g8_reference": {
                k: g8["summary"][k]
                for k in ("top1_hit_rate", "mean_regret", "paired_gain_vs_production")
            },
            "challenger_lane_recommended": bool(prod_ci_low > 0.0 and fixed_ci_low > 0.0),
            "predicted_arm_distribution": {arm: predicted.count(arm) for arm in ARMS},
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/oc_action_semantic_gate_v3/confirmation_fresh.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_contextual_value_model/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": result["gate_passed"], "summary": result["summary"]}, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
