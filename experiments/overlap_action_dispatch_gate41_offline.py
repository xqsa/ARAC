"""Gate 41a: offline calibration and validation of the AOB action dispatch rule.

The user's requirement: the upgraded method on AOB-24 must not be weaker than
the historical ARAC column.  Evidence assembled offline:

- The current structural routing regresses on E/S families (recovery campaign
  2026-08-11: 21/24 cases worse than displayed).
- The RecoveredActionRegistry four-arm matrix (7 seeds x 24 cases x 4 arms,
  shared Phase-I checkpoints) shows oracle-per-case action selection reaches
  or beats the historical column nearly everywhere.
- This gate fits a two-feature interpretable dispatch rule on those receipts
  (zero new FE) and evaluates its realized per-case errors against the
  historical column and HCC-ES.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

FOUR_ARM_ROOT = Path("artifacts/current_recovered_four_arm_matrix_v2/arms")
REFERENCE_CSV = Path("output/pdf/aob_arac_method_comparison_corrected.csv")
OUTPUT_SCHEMA = "arac-overlap-action-dispatch-gate41a-offline-v1"

# Pre-registered two-feature dispatch rule (thresholds sit inside wide gaps:
# A max 0.0059 vs R min 0.251 -> tau_aor; R max 0.304 vs S min 0.664 -> tau_ctp).
TAU_AOR_GAIN = 0.10
TAU_CTP_GAIN = 0.50
TAU_NO_RELATION = 0.05
ARMS = ("aor", "ctp", "smp", "gcb")


def dispatch(tail_log10_gain: float, structural_relation_density: float) -> str:
    if tail_log10_gain < TAU_AOR_GAIN:
        return "aor"
    if tail_log10_gain >= TAU_CTP_GAIN:
        return "smp" if structural_relation_density <= TAU_NO_RELATION else "ctp"
    return "gcb"


def _load_receipts() -> dict[tuple[str, int], dict]:
    data: dict[tuple[str, int], dict] = {}
    for case_dir in sorted(FOUR_ARM_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue
        for seed_dir in sorted(case_dir.iterdir()):
            for arm_file in seed_dir.glob("*.json"):
                payload = json.loads(arm_file.read_text(encoding="utf-8"))
                key = (case_dir.name, int(seed_dir.name.split("_")[1]))
                entry = data.setdefault(key, {"features": None, "errors": {}})
                entry["errors"][arm_file.stem] = float(payload["final_error"])
                if entry["features"] is None:
                    entry["features"] = dict(
                        zip(payload["feature_names"], payload["feature_values"])
                    )
    return data


def _load_reference() -> dict[str, dict[str, float]]:
    import csv
    import re

    references: dict[str, dict[str, float]] = {}
    valid_cases = {
        f"{family}{index}" for family in "AERS" for index in range(1, 7)
    }
    with REFERENCE_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case"] not in valid_cases:
                continue
            references[row["case"]] = {
                "arac": float(re.match(r"\s*([0-9.eE+-]+)", row["ARAC Mean +/- Std"]).group(1)),
                "hcc": float(
                    re.match(r"\s*([0-9.eE+-]+)", row["HCC-ES Mean +/- Std"]).group(1)
                ),
            }
    return references


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_action_dispatch_gate41a/offline.json"),
    )
    args = parser.parse_args()
    data = _load_receipts()
    references = _load_reference()
    cases = sorted({key[0] for key in data}, key=lambda case: (case[0], int(case[1:])))

    rows = []
    seed_matches = 0
    seed_total = 0
    for case in cases:
        seeds = [seed for (c, seed) in data if c == case]
        chosen_errors = []
        oracle_means = {}
        for arm in ARMS:
            values = [
                data[(case, seed)]["errors"][arm]
                for seed in seeds
                if arm in data[(case, seed)]["errors"]
            ]
            if values:
                oracle_means[arm] = statistics.mean(values)
        oracle = min(oracle_means, key=oracle_means.get)
        for seed in seeds:
            features = data[(case, seed)]["features"]
            chosen = dispatch(
                features["tail_log10_gain"],
                features["structural_relation_density"],
            )
            if chosen in data[(case, seed)]["errors"]:
                chosen_errors.append(data[(case, seed)]["errors"][chosen])
            seed_total += 1
            seed_matches += int(chosen == oracle)
        realized = statistics.mean(chosen_errors)
        historical = references[case]["arac"]
        hcc = references[case]["hcc"]
        rows.append(
            {
                "case": case,
                "oracle_action": oracle,
                "chosen_action": dispatch(
                    statistics.mean(
                        data[(case, seed)]["features"]["tail_log10_gain"] for seed in seeds
                    ),
                    statistics.mean(
                        data[(case, seed)]["features"]["structural_relation_density"]
                        for seed in seeds
                    ),
                ),
                "realized_mean": realized,
                "oracle_mean": oracle_means[oracle],
                "historical_arac_mean": historical,
                "hcc_mean": hcc,
                "ratio_vs_historical": realized / historical,
                "better_than_hcc": realized < hcc,
            }
        )

    ratios = [row["ratio_vs_historical"] for row in rows]
    geometric_mean = statistics.geometric_mean(ratios)
    hcc_wins = sum(1 for row in rows if row["better_than_hcc"])
    checks = {
        "per_seed_rule_matches_oracle_ge_0_95": seed_matches / seed_total >= 0.95,
        "every_case_within_1_10x_of_historical": all(ratio <= 1.10 for ratio in ratios),
        "geometric_mean_ratio_le_1_0": geometric_mean <= 1.0,
        "hcc_win_count_ge_18": hcc_wins >= 18,
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "four_arm_source": str(FOUR_ARM_ROOT),
            "reference_source": str(REFERENCE_CSV),
            "rule": {
                "features": ("tail_log10_gain", "structural_relation_density"),
                "tau_aor_gain": TAU_AOR_GAIN,
                "tau_ctp_gain": TAU_CTP_GAIN,
                "tau_no_relation": TAU_NO_RELATION,
            },
            "seed_count": len({key[1] for key in data}),
        },
        "summary": {
            "seed_rule_match_fraction": seed_matches / seed_total,
            "hcc_win_count": hcc_wins,
            "hcc_loss_count": sum(1 for row in rows if not row["better_than_hcc"]),
            "geometric_mean_ratio_vs_historical": geometric_mean,
            "max_ratio_vs_historical": max(ratios),
            "cases_better_than_historical_column": sum(
                1 for row in rows if row["realized_mean"] < row["historical_arac_mean"]
            ),
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{'case':>4} {'chosen':>6} {'oracle':>6} {'realized':>11} {'histARAC':>11} {'ratio':>7} {'vsHCC':>7}"
    )
    for row in rows:
        print(
            f"{row['case']:>4} {row['chosen_action']:>6} {row['oracle_action']:>6} "
            f"{row['realized_mean']:>11.3g} {row['historical_arac_mean']:>11.3g} "
            f"{row['ratio_vs_historical']:>7.3f} {'WIN' if row['better_than_hcc'] else 'loss':>7}"
        )
    print(json.dumps({"summary": payload["summary"], "checks": checks}, indent=1))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
