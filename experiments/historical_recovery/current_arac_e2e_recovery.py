"""Run and verify the real current Phase-I -> ARAC-Core -> Phase-II chain."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import traceback

from threadpoolctl import threadpool_info, threadpool_limits

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "current_arac_e2e_recovery_protocol.json"
TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "current-arac-e2e-recovery"
OUTPUT_DEFAULT = REPOSITORY_ROOT / "artifacts" / "current_arac_e2e_recovery_v1"
SAME_BOUNDARY_REFERENCE = 72195.19251439234
ZERO_START_REFERENCE = 5.983267874603139e-7


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol = _read_json(path)
    expected = {
        "schema_version": "arac-current-arac-e2e-recovery-protocol-v1",
        "case_id": "E1",
        "run_seed": 117,
        "action_seed": 117,
        "total_budget_fes": 3_000_000,
        "expected_phase1_fes": 180_000,
        "same_boundary_reference_action": "smp",
        "same_boundary_reference_final_error": SAME_BOUNDARY_REFERENCE,
        "zero_start_historical_reference_final_error": ZERO_START_REFERENCE,
        "terminal_run_count": 1,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol drifted: {key}")
    sources = protocol.get("source_paths")
    if not isinstance(sources, list) or not sources:
        raise ValueError("protocol source paths are missing")
    return protocol


def _production_hcc_imports() -> list[str]:
    matches = []
    for path in (REPOSITORY_ROOT / "src" / "arac").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        if any(token in source for token in ("from hcc", "import hcc", "vendor.hcc", "vendor/hcc")):
            matches.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    return matches


def preflight(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    if output_root.exists():
        raise ValueError(f"fresh output already exists: {output_root}")
    missing = [source for source in protocol["source_paths"] if not (REPOSITORY_ROOT / str(source)).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source inputs: {missing}")
    imports = _production_hcc_imports()
    if imports:
        raise ValueError(f"production HCC imports remain: {imports}")
    return {
        "schema_version": "arac-current-arac-e2e-recovery-preflight-v1",
        "protocol_sha256": _sha256(protocol_path),
        "output_root": str(output_root.resolve()),
        "source_sha256": {str(source): _sha256(REPOSITORY_ROOT / str(source)) for source in protocol["source_paths"]},
        "production_hcc_runtime_imports": imports,
        "selector_execution_allowed": False,
        "passed": True,
    }


def run(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    gate = preflight(protocol_path)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    output_root.mkdir(parents=True)
    _write(output_root / "protocol.json", protocol)
    _write(output_root / "preflight.json", gate)
    manifest_body = {
        "schema_version": "arac-current-arac-e2e-recovery-manifest-v1",
        "protocol_sha256": _sha256(protocol_path),
        "preflight_sha256": _canonical(gate),
        "source_sha256": gate["source_sha256"],
        "terminal_run_count": 1,
    }
    _write(output_root / "manifest.json", {**manifest_body, "manifest_sha256": _canonical(manifest_body)})
    try:
        from arac.benchmarks.aob import AobBenchmark
        from arac.core import run_arac

        problem = AobBenchmark().load("E1", output_directory=output_root / "benchmark")
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            result = run_arac(problem, total_budget_fes=3_000_000, run_seed=117, action_seed=117)
        payload = {
            "schema_version": "arac-current-arac-e2e-recovery-receipt-v1",
            "case_id": "E1",
            "run_seed": 117,
            "action_seed": 117,
            "total_budget_fes": 3_000_000,
            "phase1_fes": result.phase1.checkpoint.phase1_fes,
            "phase1_checkpoint_hash": result.phase1.checkpoint.checkpoint_hash,
            "action_checkpoint_hash": result.core.action_result.checkpoint_hash,
            "selected_action": result.core.decision.action_name,
            "selection_reason": result.core.decision.reason,
            "selection_scores": list(result.core.decision.scores),
            "relation_count": result.core.decision.relation_count,
            "largest_component_fraction": result.core.decision.largest_component_fraction,
            "action_result": result.core.action_result.payload(),
            "terminal_fes": result.core.action_result.terminal_fes,
            "final_error": result.core.action_result.final_error,
            "same_boundary_reference_action": "smp",
            "same_boundary_reference_final_error": SAME_BOUNDARY_REFERENCE,
            "same_boundary_recovered_or_exceeded": result.core.action_result.final_error <= SAME_BOUNDARY_REFERENCE,
            "zero_start_historical_reference_final_error": ZERO_START_REFERENCE,
            "zero_start_reference_comparable": False,
            "terminal_state_finite": True,
            "production_hcc_runtime_imports": _production_hcc_imports(),
            "selector_execution_allowed": False,
            "threadpools": pools,
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        _write(output_root / "receipt.json", {**payload, "receipt_sha256": _canonical(payload)})
        summary_body = {
            "schema_version": "arac-current-arac-e2e-recovery-summary-v1",
            "selected_action": payload["selected_action"],
            "phase1_fes": payload["phase1_fes"],
            "terminal_fes": payload["terminal_fes"],
            "final_error": payload["final_error"],
            "same_boundary_reference_final_error": SAME_BOUNDARY_REFERENCE,
            "same_boundary_recovered_or_exceeded": payload["same_boundary_recovered_or_exceeded"],
            "receipt_sha256": _canonical(payload),
        }
        _write(output_root / "summary.json", {**summary_body, "summary_sha256": _canonical(summary_body)})
        return summary_body
    except BaseException as error:
        _write(output_root / "failure.json", {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise


def verify(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    manifest = _read_json(output_root / "manifest.json")
    claimed = manifest.pop("manifest_sha256")
    if claimed != _canonical(manifest):
        raise ValueError("manifest hash drifted")
    if manifest["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("protocol hash drifted")
    receipt = _read_json(output_root / "receipt.json")
    receipt_claimed = receipt.pop("receipt_sha256")
    if receipt_claimed != _canonical(receipt):
        raise ValueError("receipt hash drifted")
    if receipt["phase1_fes"] != 180_000 or receipt["terminal_fes"] != 3_000_000:
        raise ValueError("end-to-end FE contract failed")
    if receipt["phase1_checkpoint_hash"] != receipt["action_checkpoint_hash"]:
        raise ValueError("checkpoint binding failed")
    summary = _read_json(output_root / "summary.json")
    summary_claimed = summary.pop("summary_sha256")
    if summary_claimed != _canonical(summary) or summary["receipt_sha256"] != receipt_claimed:
        raise ValueError("summary binding failed")
    return {**summary, "receipt_sha256": receipt_claimed, "summary_sha256": summary_claimed}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    result = preflight(args.protocol) if args.command == "preflight" else run(args.protocol) if args.command == "run" else verify(args.protocol)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
