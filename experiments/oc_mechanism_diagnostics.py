"""Receipt-only diagnostics for the S5 runway tax and R2 AOR horizon.

The analyzer deliberately does not rerun an optimizer.  It reconstructs all
claims from the paired gate51c JSON cells so a diagnosis cannot drift from the
experiment that produced the headline result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/oc_phase_aware_gate51c_v5_1")
CASES = ("S5", "R2")
SEEDS = (20260901, 20260902, 20260903)
HORIZON_FES = 450_000
PHASE1_FES = 180_000


def _load(root: Path, case: str, seed: int, arm: str) -> dict[str, Any]:
    path = root / "cells" / f"{case}_{seed}_{arm}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get("result", payload)
    if row.get("case_id") != case or row.get("seed") != seed or row.get("arm") != arm:
        raise ValueError(f"cell identity mismatch: {path}")
    return row


def _receipt_fes(receipts: list[dict[str, Any]], index: int) -> int:
    return sum(int(receipt["consumed_fes"]) for receipt in receipts[: index + 1])


def _step_error(segments: list[dict[str, Any]], phase2_fes: int) -> float:
    """Read a standalone piecewise-constant trace at a Phase-II FE."""

    error = float(segments[0]["error_before"]) if segments else math.inf
    for segment in segments:
        if int(segment["cumulative_phase2_fes"]) <= phase2_fes:
            error = float(segment["error_after"])
        else:
            break
    return error


def diagnose_s5(root: Path = ROOT, seed: int = 20260901) -> dict[str, Any]:
    on = _load(root, "S5", seed, "on")["result"]
    standalone = _load(root, "S5", seed, "ctp")["result"]
    receipts = list(on.get("receipts", []))
    protected = [
        receipt for receipt in receipts if receipt.get("reservation_kind") == "protected_runway"
    ]
    release = [receipt for receipt in protected if receipt.get("plateau_release")]
    all_releases = [receipt for receipt in receipts if receipt.get("plateau_release")]
    ctp_receipts = [receipt for receipt in receipts if receipt.get("episode") == "ctp"]
    matched_budget = []
    for receipt in ctp_receipts:
        runtime = int(receipt.get("cumulative_runtime_fes", 0))
        if runtime <= 0:
            continue
        standalone_error = _step_error(standalone.get("segments", []), runtime)
        matched_budget.append(
            {
                "segment_index": receipt["segment_index"],
                "runtime_fes": runtime,
                "oc_local_error": float(receipt["local_error_after"]),
                "standalone_ctp_error": standalone_error,
                "oc_minus_standalone": float(receipt["local_error_after"]) - standalone_error,
                "reservation_kind": receipt.get("reservation_kind", ""),
                "plateau_release": bool(receipt.get("plateau_release", False)),
            }
        )
    protected_tax = sum(int(receipt["consumed_fes"]) for receipt in protected)
    post_release_fes = sum(
        int(receipt["consumed_fes"])
        for receipt in receipts
        if release and receipt["segment_index"] > release[0]["segment_index"]
        and receipt.get("episode") != "ctp"
    )
    return {
        "case": "S5",
        "seed": seed,
        "final_error": float(on["final_error"]),
        "standalone_ctp_final_error": float(standalone["final_error"]),
        "adaptive_lock": [
            {
                "segment_index": receipt["segment_index"],
                "consumed_fes": receipt["consumed_fes"],
                "global_gain": receipt["global_gain"],
                "global_error_after": receipt["global_error_after"],
            }
            for receipt in receipts
            if receipt.get("reservation_kind") == "adaptive_lock"
        ],
        "protected_runway_fes": protected_tax,
        "protected_runway_count": len(protected),
        "plateau_release_count": len(release),
        "all_plateau_release_count": len(all_releases),
        "release_receipts": [
            {
                "segment_index": receipt["segment_index"],
                "episode": receipt["episode"],
                "global_gain": receipt["global_gain"],
                "released": receipt.get("released", receipt.get("plateau_release", False)),
                "next_episode": (
                    receipts[index + 1]["episode"] if index + 1 < len(receipts) else ""
                ),
            }
            for index, receipt in enumerate(receipts)
            if receipt.get("plateau_release")
            and receipt.get("reservation_kind") == "protected_runway"
        ],
        "post_release_non_ctp_fes": post_release_fes,
        "ctp_matched_budget": matched_budget,
        "handoff_count": (
            len(on.get("handoffs", []))
            if on.get("handoffs") is not None
            else sum(bool(receipt.get("switched")) for receipt in receipts)
        ),
    }


def diagnose_r2(root: Path = ROOT, seed: int = 20260901) -> dict[str, Any]:
    on = _load(root, "R2", seed, "on")["result"]
    standalone = _load(root, "R2", seed, "aor")["result"]
    receipts = [receipt for receipt in on.get("receipts", []) if receipt.get("episode") == "aor"]
    horizon_receipts = [
        receipt for receipt in receipts if receipt.get("reservation_kind") == "horizon"
    ]
    crossing = next(
        (
            receipt for receipt in receipts
            if int(receipt.get("cumulative_runtime_fes", 0)) >= HORIZON_FES
        ),
        None,
    )
    material = next(
        (receipt for receipt in receipts if bool(receipt.get("material"))),
        None,
    )
    standalone_crossing = next(
        (
            segment for segment in standalone.get("segments", [])
            if int(segment.get("cumulative_phase2_fes", 0)) >= HORIZON_FES
        ),
        None,
    )
    return {
        "case": "R2",
        "seed": seed,
        "final_error": float(on["final_error"]),
        "standalone_aor_final_error": float(standalone["final_error"]),
        "horizon_fes": HORIZON_FES,
        "aor_runtime_fes": sum(int(receipt["consumed_fes"]) for receipt in receipts),
        "horizon_receipt_count": len(horizon_receipts),
        "horizon_crossing": (
            {
                "segment_index": crossing["segment_index"],
                "runtime_fes": crossing["cumulative_runtime_fes"],
                "global_error_after": crossing["global_error_after"],
            }
            if crossing else None
        ),
        "first_material_global_receipt": (
            {
                "segment_index": material["segment_index"],
                "runtime_fes": material["cumulative_runtime_fes"],
                "global_gain": material["global_gain"],
                "reservation_kind": material.get("reservation_kind", ""),
            }
            if material else None
        ),
        "standalone_horizon_segment": standalone_crossing,
        "aor_receipts": [
            {
                "segment_index": receipt["segment_index"],
                "grant_kind": receipt["grant_kind"],
                "reservation_kind": receipt.get("reservation_kind", ""),
                "runtime_fes": receipt["cumulative_runtime_fes"],
                "global_gain": receipt["global_gain"],
                "material": receipt["material"],
            }
            for receipt in receipts
        ],
    }


def build_report(root: Path = ROOT, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    rows = {
        "S5": [diagnose_s5(root, seed) for seed in seeds],
        "R2": [diagnose_r2(root, seed) for seed in seeds],
    }
    return {
        "schema_version": "arac-oc-mechanism-diagnostics-v1",
        "source": str(root),
        "phase1_fes": PHASE1_FES,
        "rows": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ARAC-OC Mechanism Diagnostics",
        "",
        "Receipt-only analysis of gate51c fresh-seed cells.",
        "",
        "## S5 runway",
        "",
        "| seed | protected runway FE | plateau releases | post-release non-CTP FE | ON final | CTP standalone |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]["S5"]:
        lines.append(
            f"| {row['seed']} | {row['protected_runway_fes']} | "
            f"{row['plateau_release_count']} | {row['post_release_non_ctp_fes']} | "
            f"{row['final_error']:.6g} | {row['standalone_ctp_final_error']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## R2 AOR horizon",
            "",
            "| seed | AOR runtime FE | horizon receipts | horizon crossing | first material receipt | ON final | AOR standalone |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["rows"]["R2"]:
        crossing = row["horizon_crossing"]
        material = row["first_material_global_receipt"]
        lines.append(
            f"| {row['seed']} | {row['aor_runtime_fes']} | {row['horizon_receipt_count']} | "
            f"{crossing['runtime_fes'] if crossing else 'not reached'} | "
            f"{material['runtime_fes'] if material else 'none'} | "
            f"{row['final_error']:.6g} | {row['standalone_aor_final_error']:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "mechanism_diagnostics.json")
    args = parser.parse_args()
    report = build_report(args.root)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
