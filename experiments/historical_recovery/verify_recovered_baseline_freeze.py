"""Verify the immutable recovered-action baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("recovered_baseline_freeze_protocol_v1.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    if protocol.get("schema_version") != "arac-recovered-baseline-freeze-v1":
        raise ValueError("recovered baseline freeze schema drifted")
    if protocol.get("status") != "frozen":
        raise ValueError("recovered baseline is not marked frozen")
    contract = protocol.get("runtime_contract")
    if not isinstance(contract, dict):
        raise ValueError("freeze runtime contract is missing")
    expected_flags = {
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
        "topology_conditioned_smp": True,
    }
    for key, expected in expected_flags.items():
        if contract.get(key) is not expected:
            raise ValueError(f"freeze runtime contract drifted: {key}")
    for section in ("source_files", "protocol_files", "evidence_artifacts"):
        entries = protocol.get(section)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"freeze manifest section is empty: {section}")
        for relative, expected_hash in entries.items():
            if not _resolved(str(relative)).is_file():
                raise FileNotFoundError(f"frozen file is missing: {relative}")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"invalid frozen hash: {relative}")
    return protocol


def _verify_hashes(entries: Mapping[str, str]) -> list[dict[str, str]]:
    drifted: list[dict[str, str]] = []
    for relative, expected in entries.items():
        actual = _sha256(_resolved(relative))
        if actual != expected:
            drifted.append({"path": relative, "expected": expected, "actual": actual})
    return drifted


def verify(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    sections = {
        section: _verify_hashes(protocol[section])
        for section in ("source_files", "protocol_files", "evidence_artifacts")
    }
    drifted = [item for values in sections.values() for item in values]
    if drifted:
        raise ValueError(json.dumps({"frozen_file_drift": drifted}, sort_keys=True))

    screen_summary = _load_json(_resolved("artifacts/recovery_first_screen_smp_topology_v3/summary.json"))
    smoke_summary = _load_json(_resolved("artifacts/recovery_smp_lifecycle_smoke_v1/summary.json"))
    e1_summary = _load_json(_resolved("artifacts/recovery_smp_zero_relation_preservation_v1/summary.json"))
    if not all(screen_summary.get(key) is True for key in (
        "all_checkpoint_bindings_exact", "all_receipt_hashes_valid", "all_terminal_fes_exact"
    )):
        raise ValueError("frozen screen contract is not green")
    if screen_summary.get("context_count") != 120:
        raise ValueError("frozen screen context count drifted")
    if smoke_summary.get("smoke_gate_passed") is not True or e1_summary.get("preservation_gate_passed") is not True:
        raise ValueError("frozen SMP recovery evidence is not green")

    return {
        "freeze_id": protocol["freeze_id"],
        "status": protocol["status"],
        "checked_file_count": sum(len(protocol[section]) for section in sections),
        "source_file_count": len(protocol["source_files"]),
        "protocol_file_count": len(protocol["protocol_files"]),
        "evidence_artifact_count": len(protocol["evidence_artifacts"]),
        "screen_contract_green": True,
        "smp_smoke_green": True,
        "e1_preservation_green": True,
        "patch_enabled": protocol["runtime_contract"]["patch_enabled"],
        "soft_routing_enabled": protocol["runtime_contract"]["soft_routing_enabled"],
        "selector_enabled": protocol["runtime_contract"]["selector_enabled"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.protocol), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_PROTOCOL", "load_protocol", "verify"]
