"""Gate49: paired AOB pilot with a real operator episode reservation.

Gate48b only proved that the operator path could fire; its reservation was
8 FE, while the recovered CTP route used the whole Phase-II budget. Gate49
keeps the same Phase-I and MMES control protocol, caps sense at 10%, and
registers a 4096-FE minimum CTP episode. It is a calibration pilot, not a
claim of universal non-regression.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from arac.coordination.contract import OcCoordinatorConfig
from oc_aob_paired_gate48b import CASES, run_case

SENSE_BUDGET_SHARE = 0.10
EPISODE_MIN_FES = 4096
EPISODE_MAX_FES = 65536
OUTPUT_ROOT = Path("artifacts/oc_aob_gate49_episode")
SCHEMA = "arac-oc-gate49-episode-v1"


def _run(case_id: str) -> dict[str, object]:
    return run_case(
        case_id,
        sense_budget_share=SENSE_BUDGET_SHARE,
        oc_config=OcCoordinatorConfig(
            pulse_min_fes=EPISODE_MIN_FES,
            pulse_max_fes=EPISODE_MAX_FES,
            operator_episode_min_fes=EPISODE_MIN_FES,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cells_dir = OUTPUT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run, case): case for case in CASES}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            (cells_dir / f"{row['case_id']}.json").write_text(
                json.dumps({"schema_version": SCHEMA, "result": row}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                f"{row['case_id']}: control={row['control']['final_error']:.6g} "
                f"unified={row['unified']['final_error']:.6g} "
                f"operator_fes={row['unified']['budget_flow']['operator']}",
                flush=True,
            )
    rows.sort(key=lambda row: row["case_id"])
    checks = {
        "paired_checkpoint_all": all(
            row["checkpoint_hash"]
            == row["control"]["checkpoint_hash"]
            == row["unified"]["checkpoint_hash"]
            for row in rows
        ),
        "phase1_exact_all": all(row["phase1_fes"] == 180_000 for row in rows),
        "terminal_exact_all": all(
            row["control"]["terminal_fes"] == 3_000_000
            and row["unified"]["terminal_fes"] == 3_000_000
            for row in rows
        ),
        "strict_best_all": all(row["unified"]["strict_best"] for row in rows),
        "receipts_valid_all": all(
            row["unified"]["receipt_parity"] and row["unified"]["state_hash_chain"]
            for row in rows
        ),
        "operator_episode_fired_all": all(
            row["unified"]["budget_flow"]["operator"] >= EPISODE_MIN_FES
            for row in rows
        ),
    }
    payload = {
        "schema_version": SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "phase1_fes": 180_000,
            "total_fes": 3_000_000,
            "sense_budget_share": SENSE_BUDGET_SHARE,
            "operator_episode_min_fes": EPISODE_MIN_FES,
            "operator_episode_max_fes": EPISODE_MAX_FES,
            "control": "same Phase-I checkpoint, full Phase-II MMES",
            "purpose": "episode calibration; not a formal non-regression claim",
        },
        "checks": checks,
        "pilot_passed": all(checks.values()),
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "pilot.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(checks, indent=1))
    return 0 if payload["pilot_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
