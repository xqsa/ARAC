"""Audit independent ARAC action semantics against frozen and historical references.

This audit never imports or executes HCC code. Historical HCC-backed artifacts are
treated as numerical golden references, while the frozen independent v3 sources are
the closest executable mechanism reference for the current ARAC runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CURRENT_ACTION_ROOT = REPOSITORY_ROOT / "src" / "arac" / "actions"
FROZEN_V3_ROOT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "final_24x25_v3_bounded"
    / "frozen_protocol"
    / "sources"
)
HISTORICAL_REPORT = (
    REPOSITORY_ROOT
    / "experiments"
    / "historical_recovery"
    / "historical_protocol_reconstruction.json"
)
DEFAULT_OUTPUT_JSON = (
    REPOSITORY_ROOT
    / "experiments"
    / "historical_recovery"
    / "independent_action_semantic_parity.json"
)
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_JSON.with_suffix(".md")


ACTION_SPECS: dict[str, dict[str, Any]] = {
    "AOR": {
        "file": "aor.py",
        "v3_markers": ("evidence_routed_", "run_full_space", "sepcmaes", "mmes"),
        "current_markers": ("_optimizer_route", "run_full_space", "sepcmaes", "mmes"),
        "v3_semantics": "Phase-I evidence routes one full-space Sep-CMA or MMES continuation.",
        "current_semantics": "The same evidence route is retained inside the independent runtime.",
        "v3_parity": "compatible",
        "historical_parity": "not_equivalent",
        "historical_reason": (
            "Historical AOR is a fresh 3,000,000-FE vendor Sep-CMA run from mean 0; "
            "current AOR starts from a Phase-I checkpoint and receives only the remaining budget."
        ),
    },
    "CTP": {
        "file": "ctp.py",
        "v3_markers": ("_POLISH_SWEEPS = 8", "coverage_sweeps = 4", "run_persistent_blocks"),
        "current_markers": ("_COVERAGE_FRACTION = 0.20", "run_sequential_blocks", "ctp-terminal"),
        "v3_semantics": (
            "Use 4 zero-relation or 2 positive-relation coverage sweeps, then 8 block "
            "polish sweeps, then terminal full-space MMES."
        ),
        "current_semantics": (
            "Spend about 20% on persistent coverage and then allocate nearly all remaining "
            "budget to sequential block or relation-cover polish."
        ),
        "v3_parity": "different",
        "historical_parity": "not_equivalent",
        "historical_reason": (
            "Historical CTP binds four coverage sweeps at a separate HCC decision boundary "
            "before full-space polish; current scheduling and checkpoint lifecycle differ."
        ),
    },
    "SMP": {
        "file": "smp.py",
        "v3_markers": ("run_persistent_blocks", "persistent_block_cma_"),
        "current_markers": ("run_stateful_block_visits", "run_zero_relation_hybrid_rescue", "global_polish"),
        "v3_semantics": "Persist one block-CMA state per block for the available Phase-II budget.",
        "current_semantics": (
            "Use stateful visits, stale-state restarts, directional rescue, and conditional "
            "full-space polish."
        ),
        "v3_parity": "different",
        "historical_parity": "unresolved",
        "historical_reason": (
            "The complete historical 25-seed SMP lane and its exact action lifecycle are absent."
        ),
    },
    "GCB": {
        "file": "gcb.py",
        "v3_markers": ("zero_relation_global_coordination", "run_persistent_blocks", "// 10"),
        "current_markers": ("run_cold_start_block_sweeps", "gcb-global-coordination", "gcb-continuation"),
        "v3_semantics": (
            "Use full-space Sep-CMA immediately for zero relations; otherwise spend about "
            "10% on graph-ordered persistent blocks before global coordination."
        ),
        "current_semantics": (
            "Run up to three cold block sweeps, one short full-space coordination burst, "
            "then restart cold block sweeps to the terminal budget."
        ),
        "v3_parity": "different",
        "historical_parity": "not_equivalent",
        "historical_reason": (
            "Historical GCB uses a native phase-boundary burst followed by persistent native "
            "resume sweeps; current cold-start sessions do not preserve that lifecycle."
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_evidence(path: Path, markers: tuple[str, ...]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        "sha256": _sha256(path),
        "required_markers": list(markers),
        "missing_markers": missing,
        "auditable": not missing,
    }


def _historical_lanes() -> dict[str, dict[str, Any]]:
    payload = json.loads(HISTORICAL_REPORT.read_text(encoding="utf-8"))
    return {str(lane["lane"]): lane for lane in payload["lanes"]}


def _hcc_runtime_imports() -> list[str]:
    forbidden = (
        "vendor.hcc",
        "vendor/hcc",
        "vendor\\hcc",
        "HCC.",
        "arac.backends.hcc",
        "src.arac.backends.hcc",
    )
    matches: list[str] = []
    for path in sorted((REPOSITORY_ROOT / "src" / "arac").rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(("from ", "import ")):
                continue
            if any(token in stripped for token in forbidden):
                relative = str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
                matches.append(f"{relative}:{line_number}:{stripped}")
    return matches


def build_report() -> dict[str, Any]:
    historical = _historical_lanes()
    lanes = []
    for lane_name, spec in ACTION_SPECS.items():
        current = _source_evidence(
            CURRENT_ACTION_ROOT / spec["file"],
            spec["current_markers"],
        )
        frozen_v3 = _source_evidence(
            FROZEN_V3_ROOT / spec["file"],
            spec["v3_markers"],
        )
        historical_lane = historical[lane_name]
        lanes.append(
            {
                "lane": lane_name,
                "action": historical_lane["action"],
                "current_source": current,
                "frozen_v3_source": frozen_v3,
                "source_hash_equal": current["sha256"] == frozen_v3["sha256"],
                "frozen_v3_semantics": spec["v3_semantics"],
                "current_semantics": spec["current_semantics"],
                "independent_v3_parity": spec["v3_parity"],
                "historical_hcc_parity": spec["historical_parity"],
                "historical_reason": spec["historical_reason"],
                "historical_protocol_verdict": historical_lane["verdict"],
                "historical_replay_authorized": historical_lane["replay_authorized"],
                "ready_for_selector_evaluation": False,
            }
        )
    runtime_imports = _hcc_runtime_imports()
    return {
        "schema_version": "arac-independent-action-semantic-parity-v1",
        "reference_policy": {
            "historical_hcc": "numerical_golden_reference_only",
            "frozen_independent_v3": "mechanism_reference",
            "current": "production_candidate",
        },
        "production_hcc_runtime_imports": runtime_imports,
        "production_hcc_runtime_clean": not runtime_imports,
        "lanes": lanes,
        "selector_evaluation_authorized": False,
        "next_gate": "fixed_checkpoint_single_case_mechanism_screen",
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Independent action semantic parity audit",
        "",
        "Historical HCC-backed results are numerical golden references only. Production",
        "ARAC remains independent and must recover useful action semantics without importing",
        "or renaming HCC code.",
        "",
        f"- Production HCC runtime imports: `{len(report['production_hcc_runtime_imports'])}`",
        f"- Selector evaluation authorized: `{report['selector_evaluation_authorized']}`",
        f"- Next gate: `{report['next_gate']}`",
        "",
        "| Action | Frozen independent v3 | Historical HCC | Selector ready |",
        "|---|---|---|---|",
    ]
    for lane in report["lanes"]:
        lines.append(
            f"| {lane['lane']} | {lane['independent_v3_parity']} | "
            f"{lane['historical_hcc_parity']} | no |"
        )
    lines.extend(["", "## Mechanism differences", ""])
    for lane in report["lanes"]:
        lines.extend(
            [
                f"### {lane['lane']}",
                "",
                f"- Frozen v3: {lane['frozen_v3_semantics']}",
                f"- Current: {lane['current_semantics']}",
                f"- Historical boundary: {lane['historical_reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "Do not restore an HCC production runner and do not evaluate the selector yet.",
            "First screen one representative fixed checkpoint per action, then promote only",
            "mechanisms that preserve the independent runtime and pass the action-level gate.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    json_path: Path = DEFAULT_OUTPUT_JSON,
    markdown_path: Path = DEFAULT_OUTPUT_MD,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)
    report = build_report()
    write_report(report, args.json, args.markdown)
    if args.check:
        if not report["production_hcc_runtime_clean"]:
            raise SystemExit("production HCC runtime imports remain")
        if any(
            not lane["current_source"]["auditable"]
            or not lane["frozen_v3_source"]["auditable"]
            for lane in report["lanes"]
        ):
            raise SystemExit("one or more action mechanism markers are missing")
        print(f"Wrote {args.json}")
        print(f"Wrote {args.markdown}")
        print("Selector evaluation authorized: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
