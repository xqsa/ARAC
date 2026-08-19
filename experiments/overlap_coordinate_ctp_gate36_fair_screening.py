"""Gate 36: fair-seed rerun of the coordinate-wise persistent CTP screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.overlap_coordinate_ctp_gate35_screening import run_gate


TOLERANCE = 1e-9


def _corrected_checks(payload: dict[str, object]) -> None:
    rows = payload["cells"]
    proposal_gain = np.asarray(
        [
            row["baselines"]["proposal_neighborhood"]["final_error"] - row["final_error"]
            for row in rows
        ],
        dtype=float,
    )
    full_gain = np.asarray(
        [row["baselines"]["full_context"]["final_error"] - row["final_error"] for row in rows],
        dtype=float,
    )
    conflicting_gain = np.asarray(
        [gain for gain, row in zip(proposal_gain, rows, strict=True) if row["cell"]["mode"] == "conflicting"],
        dtype=float,
    )
    payload["schema_version"] = "arac-overlap-coordinate-ctp-gate36-fair-screen-v1"
    payload["protocol"]["comparison_tolerance"] = TOLERANCE
    payload["summary"].update(
        {
            "vs_proposal_win_or_tie": float(np.mean(proposal_gain >= -TOLERANCE)),
            "vs_proposal_median_gain": float(np.median(proposal_gain)),
            "vs_proposal_min_gain": float(np.min(proposal_gain)),
            "vs_full_win_or_tie": float(np.mean(full_gain >= -TOLERANCE)),
            "vs_full_median_gain": float(np.median(full_gain)),
            "conflicting_max_gain": float(np.max(conflicting_gain)),
        }
    )
    payload["screening_checks"] = {
        "vs_proposal_win_tie_ge_0_75": float(np.mean(proposal_gain >= -TOLERANCE)) >= 0.75,
        "vs_proposal_median_nonnegative": float(np.median(proposal_gain)) >= -TOLERANCE,
        "vs_proposal_no_material_regression": float(np.min(proposal_gain)) >= -1e-6,
        "conflicting_positive_gain": float(np.max(conflicting_gain)) > TOLERANCE,
        "vs_full_win_tie_ge_0_50": float(np.mean(full_gain >= -TOLERANCE)) >= 0.50,
        "vs_full_median_nonnegative": float(np.median(full_gain)) >= -TOLERANCE,
    }
    payload["gate_passed"] = all(payload["protocol_checks"].values()) and all(
        payload["screening_checks"].values()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cell-dir",
        type=Path,
        default=Path("artifacts/overlap_coordinate_ctp_gate36/cells"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_coordinate_ctp_gate36/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers, cell_dir=args.cell_dir)
    _corrected_checks(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_passed": payload["gate_passed"],
                "protocol_checks": payload["protocol_checks"],
                "screening_checks": payload["screening_checks"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
