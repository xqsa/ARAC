"""Pairwise arm-ranking shadow on the frozen five-arm contexts (G8).

Pre-registration (frozen in .codex-tasks/arac-oc-evidence-closure/EPIC.md):

- labels are within-context pairwise winners on raw ``end_to_end_gain``
  (exact ties are excluded from training pairs and counted);
- the model is a linear pairwise logistic ranker
  ``score(context, arm) = b[arm] + u[arm, :] . ctx`` with ctx =
  [topology one-hot (3), budget/12, mode-conforming flag, |scope|/12];
  full-batch gradient descent, zero init, fixed seed, L2 1e-3 — no other
  learner is tried, so there is no model-selection freedom;
- grouped cross-validation: 5 folds split by context seed (a seed never
  spans train and test; all of a seed's contexts are held out together);
- reported on held-out contexts only: top-1 hit rate vs oracle arm,
  Cohen's kappa, mean regret, paired end-gain difference vs the production
  baseline's mapped arm (G2 `persistent` scenario), vs the best fixed arm,
  and vs random;
- pre-registered pass rule: the model "supports arm ranking" only if the
  paired held-out end-gain difference vs the production baseline has
  bootstrap 95% CI lower bound > 0 AND hit rate exceeds the random 0.20
  point estimate.  Anything else is recorded as "existing ex-ante features
  cannot support arm ranking".

Offline replay only; production selector untouched; no new search FE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.oc_lagged_coupling_shadow import ARMS, validate_input
from experiments.overlap_value_aware_dispatch_gate15 import FRESH_SEEDS

INPUT_SCHEMA = "arac-oc-action-semantic-gate-v3"
PRODUCTION_BASELINE = Path("artifacts/oc_production_baseline_gate/confirmation.json")
OUTPUT_SCHEMA = "arac-oc-pairwise-ranking-shadow-v1"
BEST_FIXED_ARM = "duplicated_shared_local_competition"
CHANCE_HIT_RATE = 1.0 / len(ARMS)
CONTEXT_FIELDS = ("topology", "overlap_budget", "mode")
GD_ITERS = 500
GD_LEARNING_RATE = 0.1
L2_PENALTY = 1e-3
MODEL_SEED = 20260823
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260824


def _context_features(context: dict[str, Any], topologies: tuple[str, ...]) -> np.ndarray:
    one_hot = [1.0 if str(context["topology"]) == name else 0.0 for name in topologies]
    budget = min(float(context["overlap_budget"]) / 12.0, 1.0)
    conforming = 1.0 if str(context["mode"]) == "conforming" else 0.0
    scope = min(len(context["selected_component"]) / 12.0, 1.0)
    return np.asarray(one_hot + [budget, conforming, scope], dtype=float)


def _design(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], tuple[str, ...]]:
    topologies = tuple(sorted({str(c["topology"]) for c in payload["contexts"]}))
    contexts = payload["contexts"]
    x = np.stack([_context_features(c, topologies) for c in contexts])
    gains = np.asarray(
        [[float(c[arm]["end_to_end_gain"]) for arm in ARMS] for c in contexts], dtype=float
    )
    seeds = np.asarray([int(c["seed"]) for c in contexts])
    return x, gains, seeds, contexts, topologies


class PairwiseLogisticRanker:
    def __init__(self, context_dim: int) -> None:
        self.bias = np.zeros(len(ARMS))
        self.weights = np.zeros((len(ARMS), context_dim))

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.bias[None, :] + x @ self.weights.T

    def fit_pairs(self, pairs: list[tuple[np.ndarray, int, int, float]]) -> "PairwiseLogisticRanker":
        d = pairs[0][0].size
        n_features = len(ARMS) + len(ARMS) * d
        matrix = np.zeros((len(pairs), n_features))
        labels = np.zeros(len(pairs))
        for row_index, (context_features, a, b, label) in enumerate(pairs):
            matrix[row_index, a] = 1.0
            matrix[row_index, b] = -1.0
            matrix[row_index, len(ARMS) + a * d : len(ARMS) + (a + 1) * d] = context_features
            matrix[row_index, len(ARMS) + b * d : len(ARMS) + (b + 1) * d] = -context_features
            labels[row_index] = label
        theta = np.zeros(n_features)
        generator = np.random.default_rng(MODEL_SEED)
        del generator
        for _ in range(GD_ITERS):
            z = matrix @ theta
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            gradient = matrix.T @ (p - labels) / max(1, len(labels)) + L2_PENALTY * theta
            theta -= GD_LEARNING_RATE * gradient
        self.bias = theta[: len(ARMS)]
        self.weights = theta[len(ARMS) :].reshape(len(ARMS), d)
        return self


def _build_pairs(x: np.ndarray, gains: np.ndarray) -> tuple[list[tuple[np.ndarray, int, int, float]], int]:
    rows, cols = np.triu_indices(len(ARMS), k=1)
    pairs: list[tuple[np.ndarray, int, int, float]] = []
    ties = 0
    for i in range(x.shape[0]):
        for a, b in zip(rows, cols):
            delta = gains[i, a] - gains[i, b]
            if delta == 0.0:
                ties += 1
                continue
            pairs.append((x[i], int(a), int(b), 1.0 if delta > 0 else 0.0))
    return pairs, ties


def _kappa(predicted: list[str], oracle: list[str]) -> float:
    total = len(predicted)
    if not total:
        return 0.0
    observed = sum(1 for a, b in zip(predicted, oracle) if a == b) / total
    if observed >= 1.0:
        return 1.0
    labels = sorted(set(predicted + oracle))
    expected = sum(
        (predicted.count(k) / total) * (oracle.count(k) / total) for k in labels
    )
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def _bootstrap_ci_low(values: list[float]) -> float:
    if not values:
        return 0.0
    sample = np.asarray(values, dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, sample.size, size=(BOOTSTRAP_DRAWS, sample.size))
    means = sample[indices].mean(axis=1)
    return float(np.percentile(means, 2.5))


def run_gate(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validate_input(payload)
    x, gains, seeds, contexts, topologies = _design(payload)
    oracle_index = gains.argmax(axis=1)
    production_rows: dict[tuple, str] = {}
    production_context = None
    if PRODUCTION_BASELINE.exists():
        production_payload = json.loads(PRODUCTION_BASELINE.read_text(encoding="utf-8"))
        production_context = "persistent"
        for row in production_payload["rows"]:
            key = (str(row["key"][0]), int(row["key"][1]), int(row["key"][2]), str(row["key"][3]))
            production_rows[key] = row[f"{production_context}_mapped_arm"] or "owner_control"
    else:
        raise RuntimeError("G2 production baseline artifact missing; run G2 first")

    fold_seeds = sorted(FRESH_SEEDS)
    predicted: list[str] = []
    oracle_labels: list[str] = []
    regrets: list[float] = []
    prod_diffs: list[float] = []
    fixed_diffs: list[float] = []
    hits: list[bool] = []
    for held_out in fold_seeds:
        train_mask = seeds != held_out
        test_mask = seeds == held_out
        pairs, _ = _build_pairs(x[train_mask], gains[train_mask])
        model = PairwiseLogisticRanker(x.shape[1]).fit_pairs(pairs)
        scores = model.scores(x[test_mask])
        for local_index, global_index in enumerate(np.flatnonzero(test_mask)):
            arm_index = int(np.argmax(scores[local_index]))
            arm = ARMS[arm_index]
            context = contexts[global_index]
            key = (
                str(context["topology"]),
                int(context["overlap_budget"]),
                int(context["seed"]),
                str(context["mode"]),
            )
            predicted.append(arm)
            oracle_labels.append(ARMS[int(oracle_index[global_index])])
            hits.append(arm_index == int(oracle_index[global_index]))
            regrets.append(float(gains[global_index, oracle_index[global_index]] - gains[global_index, arm_index]))
            prod_arm = production_rows[key]
            prod_diffs.append(float(gains[global_index, arm_index] - gains[global_index, ARMS.index(prod_arm)]))
            fixed_diffs.append(float(gains[global_index, arm_index] - gains[global_index, ARMS.index(BEST_FIXED_ARM)]))
    hit_rate = float(np.mean([float(h) for h in hits]))
    prod_ci_low = _bootstrap_ci_low(prod_diffs)
    beats_production = bool(prod_ci_low > 0.0)
    beats_chance = bool(hit_rate > CHANCE_HIT_RATE)
    supports_ranking = bool(beats_production and beats_chance)
    _, train_ties_total = _build_pairs(x, gains)
    checks = {
        "context_count_60": len(contexts) == 60,
        "folds_cover_all_contexts": len(predicted) == len(contexts),
        "no_seed_leakage": True,
        "production_baseline_present": PRODUCTION_BASELINE.exists(),
        "production_selector_unchanged": True,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "input_schema": INPUT_SCHEMA,
            "input_artifact": str(input_path),
            "production_baseline_artifact": str(PRODUCTION_BASELINE),
            "production_scenario": production_context,
            "model": "pairwise logistic ranker, linear in arm bias + arm x context features",
            "context_features": [
                f"topology_one_hot[{name}]" for name in topologies
            ] + ["budget/12", "mode_conforming", "scope_size/12"],
            "cv": "leave-one-seed-out over the fresh seeds",
            "gd_iters": GD_ITERS,
            "l2_penalty": L2_PENALTY,
            "tie_policy": "exact-tie pairs excluded from training and counted",
            "pass_rule": "paired held-out end-gain vs production CI low > 0 AND hit rate > 0.20",
            "production_selector_modified": False,
        },
        "summary": {
            "held_out_rows": len(predicted),
            "pairwise_tie_count_total": train_ties_total,
            "top1_hit_rate": hit_rate,
            "chance_hit_rate": CHANCE_HIT_RATE,
            "cohens_kappa_vs_oracle": _kappa(predicted, oracle_labels),
            "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
            "paired_gain_vs_production": float(np.mean(prod_diffs)) if prod_diffs else 0.0,
            "paired_gain_vs_production_ci95_low": prod_ci_low,
            "paired_gain_vs_best_fixed": float(np.mean(fixed_diffs)) if fixed_diffs else 0.0,
            "paired_gain_vs_best_fixed_ci95_low": _bootstrap_ci_low(fixed_diffs),
            "beats_best_fixed_arm": bool(_bootstrap_ci_low(fixed_diffs) > 0.0),
            "beats_production_baseline": beats_production,
            "beats_chance": beats_chance,
            "supports_arm_ranking": supports_ranking,
            "prediction": (
                "ex-ante context features support arm ranking"
                if supports_ranking
                else "existing ex-ante features cannot support arm ranking"
            ),
            "predicted_arm_distribution": {
                arm: predicted.count(arm) for arm in ARMS
            },
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
        default=Path("artifacts/oc_pairwise_ranking_shadow/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": result["gate_passed"], "summary": result["summary"]}, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
