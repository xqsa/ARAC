"""U0 baseline guard for the shared_patch_v1 upgrade candidate.

U0 is the entry rung of the stepwise upgrade ladder
(docs/arac-oc-stepwise-upgrade-plan-v2.1.md).  It re-runs the frozen
recovered-baseline verifier in-process, pins the anchor identity and the
three production switches, and freezes a level manifest for the upgrade
candidate.  It performs no optimization runs and adds no behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from arac.runtime.contracts import canonical_sha256
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("u0_baseline_guard_protocol_v1.json")
DEFAULT_CANDIDATE_PROTOCOL = Path(__file__).with_name("protocol.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-upgrade-u0-baseline-guard-protocol-v1",
        "candidate_id": "shared_patch_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "verifier_module": "experiments.historical_recovery.verify_recovered_baseline_freeze",
        "output_root": "artifacts/upgrade_u0_baseline_guard_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"U0 protocol drifted: {key}")
    for source in protocol["sources"]:
        if not (REPOSITORY_ROOT / source).is_file():
            raise ValueError(f"U0 source is missing: {source}")
    return protocol


def run_guard(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    candidate_path = DEFAULT_CANDIDATE_PROTOCOL.resolve()
    candidate = _load_json(candidate_path)
    if candidate.get("candidate_id") != protocol["candidate_id"] or candidate.get("freeze_anchor") != protocol["freeze_anchor"]:
        raise ValueError("U0 candidate protocol and level protocol disagree on identity")
    verifier_report = freeze_verifier.verify()
    expected = protocol["expected_verifier"]
    checks = {
        "verifier_status_frozen": verifier_report.get("status") == expected["status"],
        "freeze_id_matches_anchor": verifier_report.get("freeze_id") == expected["freeze_id"],
        "smp_smoke_green": verifier_report.get("smp_smoke_green") is True,
        "e1_preservation_green": verifier_report.get("e1_preservation_green") is True,
        "screen_contract_green": verifier_report.get("screen_contract_green") is True,
        "patch_disabled": verifier_report.get("patch_enabled") is False,
        "soft_routing_disabled": verifier_report.get("soft_routing_enabled") is False,
        "selector_disabled": verifier_report.get("selector_enabled") is False,
    }
    manifest = {
        "schema_version": "arac-upgrade-u0-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(Path(protocol_path).resolve()),
        "candidate_protocol_sha256": _sha256(candidate_path),
        "source_sha256": {source: _sha256(REPOSITORY_ROOT / source) for source in sorted(protocol["sources"])},
        "verifier_report": verifier_report,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    summary = {
        "schema_version": "arac-upgrade-u0-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "manifest_sha256": manifest["manifest_sha256"],
        "verifier_report": verifier_report,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "upgrade_authorized_to": "u1",
    }
    summary["result_hash"] = canonical_sha256(summary)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    summary = run_guard(args.protocol)
    print(json.dumps({"stage": "u0", "gate_passed": summary["gate_passed"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["DEFAULT_PROTOCOL", "load_protocol", "run_guard"]


if __name__ == "__main__":
    raise SystemExit(main())
