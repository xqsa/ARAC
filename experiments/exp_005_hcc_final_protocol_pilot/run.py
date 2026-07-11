from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ARAC_REPO_ROOT = Path(__file__).resolve().parents[2]
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"
for import_root in (ARAC_REPO_ROOT, ARAC_SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
    run_hcc_aob_smoke_execution,
    run_hcc_runtime_consumer_smoke,
)
from arac.backends.hcc import DEFAULT_AOB_DATA_ROOT

RUN_ID = "exp_005_hcc_final_protocol_pilot"
DEFAULT_EXECUTION_RUNNER = run_hcc_aob_smoke_execution
DEFAULT_MAX_FES = 3_000_000
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_PROBLEMS = (
    "E1",
    "E2",
    "E3",
    "E4",
    "E6",
    "S2",
    "S3",
    "S6",
    "R1",
    "R2",
    "R3",
    "A4",
    "A5",
)
DEFAULT_OUTPUT_DIR = Path("results/exp_005_hcc_final_protocol_pilot")
DEFAULT_PAPER_BEST_MATRIX = (
    ARAC_REPO_ROOT / "references" / "aob_paper_best_win_replay_matrix.csv"
)
AOB_AUDIT_FILES = (
    "Benchmarks.py",
    "elliptic.py",
    "schwefel.py",
    "rastrigin.py",
    "ackley.py",
)
PINNED_FINAL_PROTOCOL_ENVIRONMENT = {
    "python": "3.12.13",
    "numpy": "2.3.5",
    "matplotlib": "3.11.0",
    "PyYAML": "6.0.3",
    "scipy": "1.18.0",
    "torch": "2.12.1",
    "blas_name": "scipy-openblas",
    "blas_version": "0.3.30",
}
EnvironmentProbe = Callable[[str], dict[str, str]]
_ENVIRONMENT_PROBE_SOURCE = """
import importlib.metadata as metadata
import json
import platform

import numpy as np

blas = getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas", {})
print(json.dumps({
    "python": platform.python_version(),
    "numpy": metadata.version("numpy"),
    "matplotlib": metadata.version("matplotlib"),
    "PyYAML": metadata.version("PyYAML"),
    "scipy": metadata.version("scipy"),
    "torch": metadata.version("torch"),
    "blas_name": str(blas.get("name", "missing")),
    "blas_version": str(blas.get("version", "missing")),
}, sort_keys=True))
"""


def _probe_final_protocol_environment(python_executable: str) -> dict[str, str]:
    try:
        completed = subprocess.run(
            [python_executable, "-c", _ENVIRONMENT_PROBE_SOURCE],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        observed = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"unable to audit final protocol environment via {python_executable}: {exc}"
        ) from exc
    if not isinstance(observed, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in observed.items()
    ):
        raise RuntimeError("final protocol environment probe returned an invalid payload")
    return observed


def _final_protocol_environment_failures(observed: dict[str, str]) -> list[str]:
    return [
        f"{name}:expected={expected},observed={observed.get(name, 'missing')}"
        for name, expected in PINNED_FINAL_PROTOCOL_ENVIRONMENT.items()
        if observed.get(name) != expected
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_or_missing(path: Path) -> str:
    if not path.exists():
        return "missing"
    return _sha256_file(path)


def _match_flag(left_hash: str, right_hash: str) -> str:
    if left_hash == "missing" or right_hash == "missing":
        return "missing"
    return "1" if left_hash == right_hash else "0"


def _write_aob_protocol_audit(output_dir: Path, hcc_root: Path) -> Path:
    runtime_source = ARAC_REPO_ROOT / "HCC_SRC" / "AOB"
    canonical_source = hcc_root / "2025_HCC_GECCO-main" / "HCC_SRC" / "AOB"
    mutable_source = hcc_root / "HCC_SRC" / "AOB"
    audit_path = output_dir / "aob_protocol_audit.csv"

    rows = []
    for filename in AOB_AUDIT_FILES:
        runtime_hash = _hash_or_missing(runtime_source / filename)
        canonical_hash = _hash_or_missing(canonical_source / filename)
        mutable_hash = _hash_or_missing(mutable_source / filename)
        rows.append(
            {
                "file": filename,
                "runtime_source": str(runtime_source),
                "canonical_source": str(canonical_source),
                "mutable_hcc_source": str(mutable_source),
                "runtime_sha256": runtime_hash,
                "canonical_sha256": canonical_hash,
                "mutable_hcc_sha256": mutable_hash,
                "runtime_matches_canonical": _match_flag(runtime_hash, canonical_hash),
                "runtime_matches_mutable_hcc_src": _match_flag(runtime_hash, mutable_hash),
            }
        )

    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return audit_path


def _all_match(rows: list[dict[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    if values == {"1"}:
        return "1"
    if "missing" in values:
        return "missing"
    return "0"


def _read_aob_protocol_audit(audit_path: Path) -> list[dict[str, str]]:
    with audit_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_best_of_three_vs_paper_best(
    output_dir: Path,
    paper_best_matrix: Path,
) -> Path:
    thresholds = _read_csv(paper_best_matrix)
    if not thresholds:
        raise FileNotFoundError(
            f"offline paper-best matrix is missing or empty: {paper_best_matrix}"
        )
    threshold_by_case = {
        row["case"]: float(row["paper_best"])
        for row in thresholds
        if row.get("case") and row.get("paper_best")
    }
    result_rows = _read_csv(output_dir / "our_result_by_case.csv")
    canonical_rows = [
        row
        for row in result_rows
        if row.get("lane_id") == "canonical_evidence_controller_v1"
    ]
    if not canonical_rows:
        raise RuntimeError("canonical execution rows are required for offline comparison")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in canonical_rows:
        grouped.setdefault(row["problem_id"], []).append(row)
    output_path = output_dir / "best_of_three_vs_paper_best.csv"
    rows: list[dict[str, str]] = []
    for problem_id in sorted(grouped):
        if problem_id not in threshold_by_case:
            raise KeyError(f"missing offline paper-best threshold for {problem_id}")
        case_rows = grouped[problem_id]
        errors = [float(row["hcc_smoke_final_error"]) for row in case_rows]
        best_error = min(errors)
        seed_count = len({row["seed"] for row in case_rows})
        paper_best = threshold_by_case[problem_id]
        rows.append(
            {
                "problem_id": problem_id,
                "seed_count": str(seed_count),
                "best_error": f"{best_error:.6e}",
                "paper_best": f"{paper_best:.6e}",
                "best_of_three_win": str(int(seed_count >= 3 and best_error < paper_best)),
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "problem_id",
                "seed_count",
                "best_error",
                "paper_best",
                "best_of_three_win",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _canonical_protocol_gate_failures(
    *,
    aob_input_rows: list[dict[str, str]],
    ledger_rows: list[dict[str, str]],
    anti_leakage_rows: list[dict[str, str]],
    action_trace_rows: list[dict[str, str]],
) -> list[str]:
    failures = []
    if not aob_input_rows or any(row.get("unchanged") != "1" for row in aob_input_rows):
        failures.append("aob_input_changed_or_missing")
    if not ledger_rows or any(
        row.get("same_budget_violation") != "0" for row in ledger_rows
    ):
        failures.append("same_budget_violation")
    if not anti_leakage_rows or any(
        row.get("audit_status") != "pass" for row in anti_leakage_rows
    ):
        failures.append("anti_leakage_violation")
    for row in action_trace_rows:
        before = str(row.get("best_before", "")).strip()
        after = str(row.get("best_after", "")).strip()
        if before and after and float(after) > float(before):
            failures.append("no_harm_violation")
            break
    return failures


def run_hcc_final_protocol_pilot(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    execution_runner=None,
    hcc_root: Path | str = Path("E:/HCC-main"),
    aob_data_root: Path | str = DEFAULT_AOB_DATA_ROOT,
    python_executable: str = sys.executable,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    problem_ids: tuple[str, ...] = DEFAULT_PROBLEMS,
    jobs: int = 1,
    max_fes: int = DEFAULT_MAX_FES,
    budget_accounting: str = "strict",
    cmaes_restart: bool = True,
    mmes_restart: bool = True,
    lane_profile: str = "canonical_evidence_controller_v1",
    environment_probe: EnvironmentProbe | None = None,
    paper_best_matrix: Path | str = DEFAULT_PAPER_BEST_MATRIX,
) -> Path:
    observed_environment = (environment_probe or _probe_final_protocol_environment)(
        python_executable
    )
    environment_failures = _final_protocol_environment_failures(observed_environment)
    if environment_failures:
        raise RuntimeError(
            "final protocol environment gate failed: " + ";".join(environment_failures)
        )
    output = run_hcc_runtime_consumer_smoke(
        output_dir=output_dir,
        execution_runner=execution_runner or DEFAULT_EXECUTION_RUNNER,
        hcc_root=hcc_root,
        aob_data_root=aob_data_root,
        python_executable=python_executable,
        seeds=tuple(seeds),
        problem_ids=tuple(problem_ids),
        jobs=jobs,
        max_fes=max_fes,
        budget_accounting=budget_accounting,
        cmaes_restart=cmaes_restart,
        mmes_restart=mmes_restart,
        lane_profile=lane_profile,
    )
    manifest_path = Path(output) / "run_manifest.md"
    manifest = manifest_path.read_text(encoding="utf-8")
    manifest = (
        manifest.replace(
            "# exp_003_hcc_runtime_consumer_smoke Run Manifest",
            "# exp_005_hcc_final_protocol_pilot Run Manifest",
        )
        .replace("Evidence posture: runtime dispatch + utility evidence", "Final protocol pilot wrapper: exp_005_hcc_final_protocol_pilot")
    )
    environment_audit = {
        "status": "pass",
        "expected": PINNED_FINAL_PROTOCOL_ENVIRONMENT,
        "observed": observed_environment,
    }
    (Path(output) / "final_protocol_environment.json").write_text(
        json.dumps(environment_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest += (
        "\nFinal protocol environment gate: pass; "
        "details=final_protocol_environment.json\n"
    )
    audit_path = _write_aob_protocol_audit(Path(output), Path(hcc_root))
    audit_rows = _read_aob_protocol_audit(audit_path)
    manifest += (
        "\n"
        "AOB protocol audit: "
        f"runtime_matches_canonical={_all_match(audit_rows, 'runtime_matches_canonical')}; "
        f"runtime_matches_mutable_hcc_src={_all_match(audit_rows, 'runtime_matches_mutable_hcc_src')}; "
        "details=aob_protocol_audit.csv\n"
    )
    if lane_profile == "canonical_evidence_controller_v1":
        gate_failures = _canonical_protocol_gate_failures(
            aob_input_rows=_read_csv(Path(output) / "aob_input_manifest.csv"),
            ledger_rows=_read_csv(Path(output) / "same_budget_ledger.csv"),
            anti_leakage_rows=_read_csv(Path(output) / "anti_leakage_audit.csv"),
            action_trace_rows=_read_csv(Path(output) / "action_trace.csv"),
        )
        manifest += (
            "Canonical protocol gate: "
            f"{'pass' if not gate_failures else 'fail'}; "
            f"failures={','.join(gate_failures) if gate_failures else 'none'}\n"
        )
        manifest_path.write_text(manifest, encoding="utf-8")
        if gate_failures:
            raise RuntimeError(
                "canonical final protocol gate failed: " + ",".join(gate_failures)
            )
        comparison_path = _write_best_of_three_vs_paper_best(
            Path(output),
            Path(paper_best_matrix),
        )
        manifest += (
            "Offline best-of-three comparison: "
            f"{comparison_path.name}; source={Path(paper_best_matrix)}; "
            "runtime_dispatch_used=0\n"
        )
    manifest_path.write_text(manifest, encoding="utf-8")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3M-FE final protocol pilot.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hcc-root", default="E:/HCC-main")
    parser.add_argument(
        "--aob-data-root",
        type=lambda value: Path(value).resolve(),
        default=DEFAULT_AOB_DATA_ROOT.resolve(),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-fes", type=int, default=DEFAULT_MAX_FES)
    parser.add_argument("--budget-accounting", default="strict", choices=["strict", "source"])
    parser.add_argument("--cmaes-restart", dest="cmaes_restart", action="store_true", default=True)
    parser.add_argument("--no-cmaes-restart", dest="cmaes_restart", action="store_false")
    parser.add_argument("--mmes-restart", dest="mmes_restart", action="store_true", default=True)
    parser.add_argument("--no-mmes-restart", dest="mmes_restart", action="store_false")
    parser.add_argument(
        "--lane-profile",
        default="canonical_evidence_controller_v1",
        choices=[
            "focused_compare",
            "landscape_escape",
            "repair_landscape_escape",
            "repair_refine",
            "precision_refine_push",
            "phase_rescue_push",
            "repair_phase_rescue_push",
            "cc_harm_sep_refresh",
            "separable_cmaes_push",
            "evidence_routed_only",
            "evidence_routed_v2_only",
            "evidence_routed_v21_only",
            "evidence_routed_v22_only",
            "evidence_routed_v23_only",
            "evidence_routed_v24_only",
            "evidence_routed_v25_only",
            "evidence_routed_v26_only",
            "paper_best_win_push",
            "paper_best_win_push_v2",
            "historical_anchor_refine_push",
            "historical_13_preserve_push",
            "historical_13_fast_preserve",
            "historical_13_runtime_composite",
            "historical_13_runtime_composite_v2",
            "evidence_action_controller_v1",
            "evidence_action_controller_v2",
            "evidence_action_controller_v3",
            "evidence_action_controller_v31",
            "canonical_evidence_controller_v1",
        ],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_hcc_final_protocol_pilot(
        output_dir=Path(args.output_dir),
        hcc_root=Path(args.hcc_root),
        aob_data_root=Path(args.aob_data_root),
        python_executable=str(args.python_executable),
        seeds=tuple(args.seeds),
        problem_ids=tuple(str(problem).upper() for problem in args.problems),
        jobs=int(args.jobs),
        max_fes=int(args.max_fes),
        budget_accounting=str(args.budget_accounting),
        cmaes_restart=bool(args.cmaes_restart),
        mmes_restart=bool(args.mmes_restart),
        lane_profile=str(args.lane_profile),
    )


if __name__ == "__main__":
    main()
