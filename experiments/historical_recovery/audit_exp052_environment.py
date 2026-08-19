"""Audit the EXP-052 numerical environment without running the campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tomllib
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_REF = "c7505d91"
DEFAULT_OUTPUT_JSON = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "exp052_environment.json"
DEFAULT_OUTPUT_MD = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "exp052_environment.md"
RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "exp_052_e_series_smp_paired_gate"
    / "validation"
    / "runs"
    / "E1"
    / "candidate_smp"
    / "seed_117"
    / "exp_052_e_series_smp_paired_gate-e1-candidate_smp-seed117"
    / "elliptic"
    / "exp052_execution_receipt.json"
)
PYTHON_EXECUTABLE = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"
EXPECTED_DISTRIBUTIONS = (
    "cma",
    "numpy",
    "PyYAML",
    "scipy",
    "llvmlite",
    "numba",
    "pypop7",
)
RUNTIME_DISTRIBUTIONS = ("numpy", "scipy", "PyYAML")
SESSION_ROOT = Path.home() / ".codex" / "sessions"
FORMAL_START_LINE = 2628
FORMAL_START_SESSION = (
    SESSION_ROOT
    / "2026"
    / "07"
    / "26"
    / "rollout-2026-07-26T17-04-21-019f9dab-0f42-70f2-8f45-5cf51411e668.jsonl"
)
FORMAL_START_UTC = "2026-07-26T12:40:23.355Z"
RECEIPT_ENVIRONMENT_KEYS = (
    "python_version",
    "python_executable_sha256",
    "environment_manifest_sha256",
    "dependency_versions",
    "package_versions",
)


SESSION_EVIDENCE = (
    {
        "id": "venv_absent_before_creation",
        "path": "2026/07/18/rollout-2026-07-18T21-51-16-019f757e-ca50-7513-af12-dd6c4227fff0.jsonl",
        "line": 113,
        "kind": "precondition",
        "expected": (".venv\\Scripts\\python.exe", "Cannot find path"),
    },
    {
        "id": "venv_created",
        "path": "2026/07/18/rollout-2026-07-18T21-51-16-019f757e-ca50-7513-af12-dd6c4227fff0.jsonl",
        "line": 160,
        "kind": "command",
        "expected": ("py -3.12 -m venv", ".venv"),
    },
    {
        "id": "venv_creation_succeeded",
        "path": "2026/07/18/rollout-2026-07-18T21-51-16-019f757e-ca50-7513-af12-dd6c4227fff0.jsonl",
        "line": 165,
        "kind": "output",
        "expected": ('"exit_code":0',),
    },
    {
        "id": "hcc_pins_installed",
        "path": "2026/07/18/rollout-2026-07-18T21-51-16-019f757e-ca50-7513-af12-dd6c4227fff0.jsonl",
        "line": 197,
        "kind": "output",
        "expected": (
            "Successfully installed",
            "cma-4.4.4",
            "numpy-2.3.5",
            "PyYAML-6.0.3",
            "scipy-1.18.0",
        ),
    },
    {
        "id": "hcc_versions_imported",
        "path": "2026/07/18/rollout-2026-07-18T21-51-16-019f757e-ca50-7513-af12-dd6c4227fff0.jsonl",
        "line": 206,
        "kind": "output",
        "expected": ("4.4.4 2.3.5 1.18.0 6.0.3", "E:\\ARAC\\.venv"),
    },
    {
        "id": "baselines_installed",
        "path": "2026/07/21/rollout-2026-07-21T21-20-29-019f84d5-aa92-7770-a6d2-f9e4ef551b47.jsonl",
        "line": 2153,
        "kind": "command",
        "expected": (".venv\\Scripts\\python.exe", "pip install", ".[baselines]"),
    },
    {
        "id": "baselines_pins_resolved",
        "path": "2026/07/21/rollout-2026-07-21T21-20-29-019f84d5-aa92-7770-a6d2-f9e4ef551b47.jsonl",
        "line": 2162,
        "kind": "output",
        "expected": (
            "llvmlite==0.48.0",
            "numba==0.66.0",
            "pypop7==0.0.82",
            "(2.3.5)",
            "(1.18.0)",
        ),
    },
    {
        "id": "pypop7_installed_location",
        "path": "2026/07/21/rollout-2026-07-21T21-20-29-019f84d5-aa92-7770-a6d2-f9e4ef551b47.jsonl",
        "line": 2252,
        "kind": "output",
        "expected": ("pypop7", "0.0.82", "E:\\ARAC\\.venv\\Lib\\site-packages"),
    },
    {
        "id": "python_version_observed",
        "path": "2026/07/21/rollout-2026-07-21T21-20-29-019f84d5-aa92-7770-a6d2-f9e4ef551b47.jsonl",
        "line": 3320,
        "kind": "output",
        "expected": ("Python 3.12.7", "pypop7==0.0.82", "llvmlite==0.48.0", "numba==0.66.0"),
    },
    {
        "id": "blas_backend_observed",
        "path": "2026/07/21/rollout-2026-07-21T19-16-44-019f8464-6384-7c70-ad66-99e5d3cc6d3f.jsonl",
        "line": 18775,
        "kind": "output",
        "expected": ("OpenBLAS 0.3.30", "USE64BITINT", "DYNAMIC_ARCH"),
    },
    {
        "id": "same_venv_rechecked_before_formal_start",
        "path": "2026/07/23/rollout-2026-07-23T11-11-42-019f8cf5-0c8b-7c30-b530-157a28a664db.jsonl",
        "line": 17001,
        "kind": "output",
        "expected": ("E:\\ARAC\\.venv\\Scripts\\python.exe", "2.3.5 6.0.3"),
    },
    {
        "id": "scipy_rechecked_before_formal_start",
        "path": "2026/07/23/rollout-2026-07-23T11-11-42-019f8cf5-0c8b-7c30-b530-157a28a664db.jsonl",
        "line": 17401,
        "kind": "output",
        "expected": ("1.18.0", "True"),
    },
    {
        "id": "last_editable_install_only_arac",
        "path": "2026/07/26/rollout-2026-07-26T17-04-21-019f9dab-0f42-70f2-8f45-5cf51411e668.jsonl",
        "line": 528,
        "kind": "output",
        "expected": ("Installing collected packages: arac", "Successfully installed arac-0.1.0"),
        "forbidden": ("cma", "numpy", "PyYAML", "scipy", "pypop7", "llvmlite", "numba"),
    },
    {
        "id": "formal_exp052_start",
        "path": "2026/07/26/rollout-2026-07-26T17-04-21-019f9dab-0f42-70f2-8f45-5cf51411e668.jsonl",
        "line": FORMAL_START_LINE,
        "kind": "command",
        "expected": (".venv\\Scripts\\python.exe", "exp_052_e_series_smp_paired_gate.run", "--jobs 10"),
    },
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def _payload_text(payload: object) -> str:
    fragments: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            fragments.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return "\n".join(fragments)


def _session_line(path: Path, line_number: int) -> tuple[str, str]:
    if not path.is_file():
        return "", ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for current, raw in enumerate(handle, start=1):
            if current != line_number:
                continue
            try:
                payload = json.loads(raw).get("payload", {})
                text = _payload_text(payload).replace("\\\\", "\\")
                timestamp = str(json.loads(raw).get("timestamp", ""))
            except (TypeError, ValueError, json.JSONDecodeError):
                text = raw
                timestamp = ""
            return timestamp, text
    return "", ""


def _audit_session_evidence() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in SESSION_EVIDENCE:
        path = SESSION_ROOT.joinpath(*str(item["path"]).split("/"))
        timestamp, text = _session_line(path, int(item["line"]))
        expected = tuple(str(value) for value in item.get("expected", ()))
        forbidden = tuple(str(value) for value in item.get("forbidden", ()))
        rows.append(
            {
                "id": item["id"],
                "session_path": str(path),
                "line": int(item["line"]),
                "kind": item["kind"],
                "timestamp": timestamp,
                "expected": list(expected),
                "observed": bool(text) and all(value in text for value in expected),
                "forbidden": list(forbidden),
                "forbidden_absent": all(value not in text for value in forbidden),
                "payload_text_sha256": _sha256(text.encode("utf-8")) if text else None,
                "text_excerpt": text[:1200],
            }
        )
    return rows


def _scan_formal_session_for_dependency_mutations() -> dict[str, Any]:
    path = FORMAL_START_SESSION
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return {
            "session_path": str(path),
            "start_line": FORMAL_START_LINE - 1,
            "end_line": FORMAL_START_LINE - 1,
            "events": events,
            "scan_complete": False,
        }
    command_pattern = re.compile(
        r"(?i)(?:pip(?:\.exe)?\s+(?:install|uninstall)|python(?:\.exe)?\s+-m\s+pip|"
        r"py\s+-3\.12\s+-m\s+venv|Remove-Item[^\n]*\.venv)"
    )
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if line_number >= FORMAL_START_LINE:
                break
            if line_number <= 528:
                continue
            try:
                record = json.loads(raw)
                payload = record.get("payload", {})
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("type") != "custom_tool_call":
                continue
            text = _payload_text(payload.get("input", ""))
            match = command_pattern.search(text)
            if match:
                events.append(
                    {
                        "line": line_number,
                        "timestamp": record.get("timestamp", ""),
                        "command_excerpt": text[max(0, match.start() - 160) : match.end() + 300],
                    }
                )
    return {
        "session_path": str(path),
        "start_line": 529,
        "end_line": FORMAL_START_LINE - 1,
        "events": events,
        "scan_complete": True,
    }


def _git_file(ref: str, path: str) -> bytes:
    completed = subprocess.run(
        ("git", "show", f"{ref}:{path}"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FileNotFoundError(f"Git file missing: {ref}:{path}")
    return completed.stdout


def _historical_requirements() -> tuple[dict[str, str], str]:
    data = _git_file(HISTORICAL_REF, "pyproject.toml")
    document = tomllib.loads(data.decode("utf-8"))
    optional = document["project"]["optional-dependencies"]
    requirements: dict[str, str] = {}
    for group in ("hcc", "baselines"):
        for requirement in optional[group]:
            name, version = requirement.split("==", 1)
            requirements[name] = version
    return requirements, _sha256(data)


def _capture_environment() -> dict[str, Any]:
    if not PYTHON_EXECUTABLE.is_file():
        return {"available": False, "reason": "project_venv_missing"}
    code = """
import importlib.metadata as metadata
import json
import pathlib
import platform
import sys

names = {name.lower(): name for name in %r}
packages = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    if name and name.lower() in names:
        packages[names[name.lower()]] = distribution.version
print(json.dumps({
    "available": True,
    "python_version": sys.version,
    "python_implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "executable": sys.executable,
    "packages": packages,
}, sort_keys=True))
""" % (EXPECTED_DISTRIBUTIONS,)
    completed = subprocess.run(
        (str(PYTHON_EXECUTABLE), "-c", code),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": "environment_probe_failed",
            "stderr": completed.stderr[-1000:],
        }
    payload = json.loads(completed.stdout)
    stable = dict(payload)
    stable["python_version"] = str(payload["python_version"]).split()[0]
    stable["executable_sha256"] = _sha256(PYTHON_EXECUTABLE.read_bytes())
    pyvenv = PYTHON_EXECUTABLE.parent.parent / "pyvenv.cfg"
    stable["pyvenv_cfg_sha256"] = _sha256(pyvenv.read_bytes()) if pyvenv.is_file() else None
    stable["manifest_sha256"] = _canonical_sha256(stable)
    stable["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    return stable


def _read_receipt() -> dict[str, Any]:
    if not RECEIPT_PATH.is_file():
        return {"present": False}
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    return {
        "present": True,
        "runner_sha256": payload.get("runner_sha256"),
        "environment_keys_present": [key for key in RECEIPT_ENVIRONMENT_KEYS if key in payload],
    }


def build_report() -> dict[str, Any]:
    requirements, pyproject_sha256 = _historical_requirements()
    environment = _capture_environment()
    packages = environment.get("packages", {}) if environment.get("available") else {}
    package_match = {
        name: packages.get(name) == version for name, version in requirements.items()
    }
    runtime_package_match = {
        name: package_match[name]
        for name in RUNTIME_DISTRIBUTIONS
        if name in package_match
    }
    all_pinned_packages_match = bool(package_match) and all(package_match.values())
    receipt = _read_receipt()
    receipt_bound = bool(receipt.get("environment_keys_present"))
    session_evidence = _audit_session_evidence()
    mutation_scan = _scan_formal_session_for_dependency_mutations()
    session_evidence_complete = bool(session_evidence) and all(
        row["observed"] and row["forbidden_absent"] for row in session_evidence
    )
    session_observed_binding = (
        all_pinned_packages_match
        and session_evidence_complete
        and mutation_scan["scan_complete"]
        and not mutation_scan["events"]
    )
    session_manifest = {
        "formal_start_utc": FORMAL_START_UTC,
        "formal_start_session": str(FORMAL_START_SESSION),
        "formal_start_line": FORMAL_START_LINE,
        "evidence": [
            {
                key: row[key]
                for key in (
                    "id",
                    "session_path",
                    "line",
                    "timestamp",
                    "payload_text_sha256",
                    "observed",
                    "forbidden_absent",
                )
            }
            for row in session_evidence
        ],
        "mutation_scan": mutation_scan,
    }
    return {
        "schema_version": "arac-exp052-environment-audit-v2",
        "historical_ref": HISTORICAL_REF,
        "historical_pyproject_sha256": pyproject_sha256,
        "historical_requirements": requirements,
        "environment": environment,
        "package_match": package_match,
        "all_expected_packages_match": bool(runtime_package_match)
        and all(runtime_package_match.values()),
        "runtime_package_match": runtime_package_match,
        "all_pinned_packages_match": all_pinned_packages_match,
        "optional_only_mismatches": [
            name for name, matched in package_match.items() if not matched and name not in runtime_package_match
        ],
        "session_evidence": session_evidence,
        "session_evidence_complete": session_evidence_complete,
        "formal_session_dependency_mutation_scan": mutation_scan,
        "session_environment_manifest_sha256": _canonical_sha256(session_manifest),
        "session_observed_environment_binding": session_observed_binding,
        "receipt": receipt,
        "receipt_environment_binding": receipt_bound,
        "environment_binding_complete": bool(environment.get("available")) and receipt_bound,
        "replay_authorized": False,
        "decision": (
            "session_provenance_complete_receipt_binding_missing_manual_authorization_required"
            if session_observed_binding
            else "session_provenance_incomplete_replay_blocked"
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    environment = report["environment"]
    lines = [
        "# EXP-052 numerical environment audit",
        "",
        f"- Historical Git ref: `{report['historical_ref']}`",
        f"- Historical pyproject SHA-256: `{report['historical_pyproject_sha256']}`",
        f"- Current candidate all-pinned package match: **{'yes' if report['all_pinned_packages_match'] else 'no'}**",
        f"- Session-observed environment binding: **{'yes' if report['session_observed_environment_binding'] else 'no'}**",
        f"- Receipt environment binding: **{'yes' if report['receipt_environment_binding'] else 'no'}**",
        "- Replay authorized: **no**",
        "",
        "| Distribution | Historical pin | Current candidate | Match |",
        "|---|---:|---:|---|",
    ]
    for name, version in report["historical_requirements"].items():
        current = environment.get("packages", {}).get(name)
        lines.append(f"| `{name}` | `{version}` | `{current or '-'}` | {'yes' if report['package_match'][name] else 'no'} |")
    lines.extend(
        [
            "",
            "## Session provenance",
            "",
            "| Evidence | UTC timestamp | Session line | Verified |",
            "|---|---|---:|---|",
        ]
    )
    for row in report["session_evidence"]:
        verified = row["observed"] and row["forbidden_absent"]
        lines.append(
            f"| `{row['id']}` | `{row['timestamp'] or '-'}` | `{row['line']}` "
            f"| {'yes' if verified else 'no'} |"
        )
    mutation_scan = report["formal_session_dependency_mutation_scan"]
    lines.extend(
        [
            "",
            f"Formal-session dependency mutations after the last editable install and before "
            f"EXP-052 start: **{len(mutation_scan['events'])}**.",
            "",
            "The retained sessions directly bind the project `.venv` creation, Python 3.12.7,",
            "the pinned package versions, OpenBLAS 0.3.30 build configuration, the final",
            "editable-only `arac` reinstall, and the formal EXP-052 launch command. This supports",
            "a version-level isolated reproduction. The historical receipt itself records neither",
            "an environment manifest hash nor a launch-time runtime-library fingerprint, so it",
            "does not support a bitwise/receipt-bound replay claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(
    report: dict[str, Any],
    json_path: Path = DEFAULT_OUTPUT_JSON,
    markdown_path: Path = DEFAULT_OUTPUT_MD,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
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
        print(f"Historical pins match current candidate: {'yes' if report['all_pinned_packages_match'] else 'no'}")
        print(
            "Session-observed environment binding: "
            f"{'yes' if report['session_observed_environment_binding'] else 'no'}"
        )
        print(f"Receipt environment binding: {'yes' if report['receipt_environment_binding'] else 'no'}")
        print("Replay authorized: no")
        if not report["session_observed_environment_binding"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
