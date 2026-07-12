from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.audit_project_structure import audit_project_structure, main


TARGET_DIRECTORIES = (
    "analysis",
    "archive",
    "configs",
    "data/raw",
    "docs",
    "experiments",
    "paper",
    "references/historical",
    "references/paper",
    "results",
    "scripts",
    "src/arac",
    "tests",
    "vendor/hcc",
)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _project_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "arac"
    root.mkdir()
    for directory in TARGET_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)

    (root / ".gitignore").write_text(
        "results/*\n"
        "!results/.gitkeep\n"
        "!results/README.md\n"
        "analysis/generated/\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    (root / "results" / ".gitkeep").touch()
    (root / "results" / "README.md").write_text("generated outputs\n", encoding="utf-8")
    (root / "src" / "arac" / "__init__.py").write_text("", encoding="utf-8")

    _git(root, "init", "-q")
    _git(root, "add", ".")
    return root


def _error_paths(root: Path) -> set[str]:
    return {finding.path for finding in audit_project_structure(root).errors}


def test_target_scientific_top_level_directories_are_allowed(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)

    report = audit_project_structure(root)

    assert report.errors == ()


def test_tracked_python_cache_under_source_is_reported(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    cache = root / "src" / "arac" / "__pycache__" / "policy.cpython-311.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"tracked cache")
    _git(root, "add", "-f", "src/arac/__pycache__/policy.cpython-311.pyc")

    assert "src/arac/__pycache__/policy.cpython-311.pyc" in _error_paths(root)


def test_runtime_source_reference_to_historical_paper_path_is_reported(
    tmp_path: Path,
) -> None:
    root = _project_fixture(tmp_path)
    runtime_source = root / "src" / "arac" / "leaky_policy.py"
    runtime_source.write_text(
        'HISTORICAL_TABLE = "references/historical/final.csv"\n',
        encoding="utf-8",
    )

    assert "src/arac/leaky_policy.py" in _error_paths(root)


def test_results_are_generated_and_payload_is_not_scanned(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    payload_cache = root / "results" / "run-001" / "__pycache__" / "payload.pyc"
    payload_cache.parent.mkdir(parents=True)
    payload_cache.write_bytes(b"generated payload")

    report = audit_project_structure(root)

    assert "results" in report.generated_paths
    assert report.errors == ()


def test_results_generated_payload_must_be_ignored_by_git(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    (root / ".gitignore").write_text("", encoding="utf-8")

    assert ".gitignore" in _error_paths(root)


def test_hcc_src_is_an_explicit_nonfatal_task3_transition(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    legacy_readme = root / "HCC_SRC" / "README.md"
    legacy_readme.parent.mkdir()
    legacy_readme.write_text("Task 3 will migrate this source.\n", encoding="utf-8")
    _git(root, "add", "HCC_SRC/README.md")

    report = audit_project_structure(root)

    assert report.errors == ()
    assert any(
        warning.path == "HCC_SRC"
        and "Task 3" in warning.rule
        and "compatibility" in warning.rule
        for warning in report.warnings
    )


def test_cli_prints_path_rule_and_returns_nonzero(
    tmp_path: Path,
    capsys,
) -> None:
    root = _project_fixture(tmp_path)
    source = root / "src" / "arac" / "leaky_policy.py"
    source.write_text('PAPER = "paper/tables/final.csv"\n', encoding="utf-8")

    exit_code = main(["--root", str(root)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "src/arac/leaky_policy.py: " in output
