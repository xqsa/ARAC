from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


ALLOWED_TOP_LEVEL_DIRECTORIES = frozenset(
    {
        ".codex",
        ".github",
        "analysis",
        "archive",
        "configs",
        "data",
        "docs",
        "experiments",
        "logs",
        "paper",
        "references",
        "results",
        "scripts",
        "src",
        "tests",
        "vendor",
    }
)
IGNORED_TOP_LEVEL_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
)
IGNORED_SOURCE_DIRECTORY_NAMES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
)
OFFLINE_PATH_PATTERN = re.compile(
    r"(?i)(?:references[\\/](?:paper|historical)|paper|historical|archive)[\\/]"
)
RESULTS_TRACKED_FILES = frozenset({"results/.gitkeep", "results/README.md"})


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str


@dataclass(frozen=True)
class AuditReport:
    errors: tuple[Finding, ...]
    warnings: tuple[Finding, ...]
    generated_paths: tuple[str, ...]


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _tracked_paths(root: Path) -> tuple[list[str], Finding | None]:
    completed = _run_git(root, "ls-files", "-z")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git ls-files failed"
        return [], Finding(".", f"tracked-file audit unavailable: {detail}")
    return [path for path in completed.stdout.split("\0") if path], None


def _is_ignored(root: Path, relative_path: str) -> tuple[bool, Finding | None]:
    completed = _run_git(root, "check-ignore", "--no-index", "--quiet", "--", relative_path)
    if completed.returncode == 0:
        return True, None
    if completed.returncode == 1:
        return False, None
    detail = completed.stderr.strip() or "git check-ignore failed"
    return False, Finding(".gitignore", f"ignore audit unavailable: {detail}")


def audit_project_structure(root: Path) -> AuditReport:
    root = root.resolve()
    errors: list[Finding] = []
    warnings: list[Finding] = []

    if not root.is_dir():
        return AuditReport((Finding(str(root), "repository root is not a directory"),), (), ())

    for child in root.iterdir():
        if not child.is_dir() or child.name in IGNORED_TOP_LEVEL_DIRECTORIES:
            continue
        if child.name == "HCC_SRC":
            warnings.append(
                Finding(
                    "HCC_SRC",
                    "warning: Task 3 pending transitional legacy source; retained because current "
                    "compatibility paths still resolve the HCC backend",
                )
            )
            continue
        if child.name not in ALLOWED_TOP_LEVEL_DIRECTORIES:
            errors.append(Finding(child.name, "unexpected top-level directory"))

    tracked_paths, git_error = _tracked_paths(root)
    if git_error:
        errors.append(git_error)
    else:
        for tracked_path in tracked_paths:
            path = PurePosixPath(tracked_path)
            if "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
                errors.append(Finding(tracked_path, "tracked Python cache artifact"))
            if tracked_path.startswith("results/") and tracked_path not in RESULTS_TRACKED_FILES:
                errors.append(Finding(tracked_path, "generated results payload is tracked by Git"))

    runtime_root = root / "src" / "arac"
    if runtime_root.is_dir():
        for source_path in runtime_root.rglob("*.py"):
            relative_parts = source_path.relative_to(runtime_root).parts
            if any(part in IGNORED_SOURCE_DIRECTORY_NAMES for part in relative_parts):
                continue
            try:
                source = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                errors.append(
                    Finding(
                        source_path.relative_to(root).as_posix(),
                        f"runtime source is not readable UTF-8: {exc}",
                    )
                )
                continue
            if OFFLINE_PATH_PATTERN.search(source):
                errors.append(
                    Finding(
                        source_path.relative_to(root).as_posix(),
                        "runtime source references a paper/historical/archive offline path",
                    )
                )

    results_probe = "results/__arac_structure_audit_probe__.tmp"
    results_ignored, ignore_error = _is_ignored(root, results_probe)
    if ignore_error:
        errors.append(ignore_error)
    elif not results_ignored:
        errors.append(Finding(".gitignore", "results generated payload is not ignored by Git"))

    for trackable_path in ("results/.gitkeep", "results/README.md"):
        ignored, ignore_error = _is_ignored(root, trackable_path)
        if ignore_error:
            errors.append(ignore_error)
        elif ignored:
            errors.append(Finding(".gitignore", f"{trackable_path} must remain trackable"))

    generated_paths = ("results",) if (root / "results").is_dir() else ()
    return AuditReport(
        tuple(sorted(set(errors), key=lambda finding: (finding.path, finding.rule))),
        tuple(sorted(set(warnings), key=lambda finding: (finding.path, finding.rule))),
        generated_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the ARAC scientific project structure.")
    parser.add_argument("--root", required=True, type=Path, help="repository root to audit")
    args = parser.parse_args(argv)

    report = audit_project_structure(args.root)
    for finding in report.errors:
        print(f"{finding.path}: {finding.rule}")
    for warning in report.warnings:
        print(f"{warning.path}: {warning.rule}")
    for generated_path in report.generated_paths:
        print(f"{generated_path}: generated directory; payload scan skipped; Git ignore required")

    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
