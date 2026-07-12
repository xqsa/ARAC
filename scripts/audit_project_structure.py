from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


ALLOWED_TOP_LEVEL_DIRECTORIES = frozenset(
    {
        ".codex",
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
OFFLINE_PATH_PARTS = frozenset({"archive", "historical", "paper", "results"})
PATH_CONSTRUCTORS = frozenset(
    {
        "Path",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
        "pathlib.Path",
        "pathlib.PurePath",
        "pathlib.PurePosixPath",
        "pathlib.PureWindowsPath",
    }
)
PATH_JOIN_FUNCTIONS = frozenset({"ntpath.join", "os.path.join", "posixpath.join"})
TRACKED_RESULT_POLICIES = (
    ("results/", frozenset({"results/.gitkeep", "results/README.md"})),
    ("HCC_SRC/result/", frozenset({"HCC_SRC/result/README.md"})),
    ("vendor/hcc/result/", frozenset({"vendor/hcc/result/README.md"})),
)
GENERATED_IGNORE_PROBES = (
    ("results", "results/__arac_structure_audit_probe__.tmp"),
    ("HCC_SRC/result", "HCC_SRC/result/__arac_structure_audit_probe__.tmp"),
    ("vendor/hcc/result", "vendor/hcc/result/__arac_structure_audit_probe__.tmp"),
)
TRACKABLE_RESULT_FILES = (
    "results/.gitkeep",
    "results/README.md",
    "HCC_SRC/result/README.md",
    "vendor/hcc/result/README.md",
)


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


def _repo_root_error(root: Path) -> Finding | None:
    completed = _run_git(root, "rev-parse", "--show-toplevel")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git rev-parse failed"
        return Finding(".", f"--root must equal Git top-level: {detail}")

    git_root = Path(completed.stdout.strip()).resolve()
    if git_root != root:
        return Finding(".", f"--root must equal Git top-level: {git_root}")
    return None


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _split_path(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.replace("\\", "/").split("/") if part and part != ".")


def _static_path_parts(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _split_path(node.value)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _static_path_parts(node.left)
        right = _static_path_parts(node.right)
        return left + right if left is not None and right is not None else None

    if not isinstance(node, ast.Call):
        return None

    function_name = _dotted_name(node.func)
    if function_name not in PATH_CONSTRUCTORS and function_name not in PATH_JOIN_FUNCTIONS:
        return None

    parts: tuple[str, ...] = ()
    for argument in node.args:
        argument_parts = _static_path_parts(argument)
        if argument_parts is None:
            return None
        parts += argument_parts
    return parts


def _candidate_path_parts(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if "/" not in node.value and "\\" not in node.value:
            return None
        return _split_path(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _static_path_parts(node)
    if isinstance(node, ast.Call):
        function_name = _dotted_name(node.func)
        if function_name in PATH_CONSTRUCTORS or function_name in PATH_JOIN_FUNCTIONS:
            return _static_path_parts(node)
    return None


def _contains_offline_path(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        parts = _candidate_path_parts(node)
        if parts and any(part.casefold() in OFFLINE_PATH_PARTS for part in parts):
            return True
    return False


def audit_project_structure(root: Path) -> AuditReport:
    root = root.resolve()
    errors: list[Finding] = []
    warnings: list[Finding] = []

    if not root.is_dir():
        return AuditReport((Finding(str(root), "repository root is not a directory"),), (), ())

    root_error = _repo_root_error(root)
    if root_error:
        return AuditReport((root_error,), (), ())

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
            for prefix, allowed_paths in TRACKED_RESULT_POLICIES:
                if tracked_path.startswith(prefix) and tracked_path not in allowed_paths:
                    errors.append(
                        Finding(tracked_path, f"generated {prefix.rstrip('/')} payload is tracked")
                    )
                    break

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
            try:
                contains_offline_path = _contains_offline_path(source)
            except SyntaxError as exc:
                errors.append(
                    Finding(
                        source_path.relative_to(root).as_posix(),
                        f"runtime source is not valid Python: {exc.msg} at line {exc.lineno}",
                    )
                )
                continue
            if contains_offline_path:
                errors.append(
                    Finding(
                        source_path.relative_to(root).as_posix(),
                        "runtime source references a paper/historical/archive/results offline path",
                    )
                )

    for result_path, probe in GENERATED_IGNORE_PROBES:
        results_ignored, ignore_error = _is_ignored(root, probe)
        if ignore_error:
            errors.append(ignore_error)
        elif not results_ignored:
            errors.append(
                Finding(".gitignore", f"{result_path} generated payload is not ignored by Git")
            )

    for trackable_path in TRACKABLE_RESULT_FILES:
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
