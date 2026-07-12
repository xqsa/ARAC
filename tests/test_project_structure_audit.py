from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.audit_project_structure as audit_module
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
        "analysis/generated/\n"
        "vendor/hcc/result/*\n"
        "!vendor/hcc/result/README.md\n",
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


def _error_rules(root: Path) -> set[str]:
    return {finding.rule for finding in audit_project_structure(root).errors}


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


@pytest.mark.parametrize(
    "source",
    [
        'from pathlib import Path\nTABLE = Path("references") / "historical"\n',
        'import os\nTABLE = os.path.join("paper", "tables")\n',
        'from pathlib import Path\nRUN = Path("results") / "run.csv"\n',
        'TABLE = r"references\\paper\\table.csv"\n',
    ],
)
def test_structured_runtime_offline_paths_are_reported(tmp_path: Path, source: str) -> None:
    root = _project_fixture(tmp_path)
    runtime_source = root / "src" / "arac" / "structured_leak.py"
    runtime_source.write_text(source, encoding="utf-8")

    assert "src/arac/structured_leak.py" in _error_paths(root)


def test_similar_runtime_path_names_are_not_reported(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    runtime_source = root / "src" / "arac" / "valid_paths.py"
    runtime_source.write_text(
        'PATHS = ("notpaper/x", "myarchive/x", "myhistorical/x")\n',
        encoding="utf-8",
    )

    assert "src/arac/valid_paths.py" not in _error_paths(root)


@pytest.mark.parametrize(
    "source",
    [
        'from pathlib import Path as P\nRUN = P("results") / run_id\n',
        'from os.path import join\nTABLE = join("references", "historical", name)\n',
        'import pathlib as pl\nRUN = pl.Path("results") / run_id\n',
    ],
)
def test_import_aliases_and_dynamic_prefixes_are_reported(
    tmp_path: Path,
    source: str,
) -> None:
    root = _project_fixture(tmp_path)
    runtime_source = root / "src" / "arac" / "aliased_leak.py"
    runtime_source.write_text(source, encoding="utf-8")

    assert "src/arac/aliased_leak.py" in _error_paths(root)


def test_f_string_runtime_prefix_is_reported(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    runtime_source = root / "src" / "arac" / "dynamic_leak.py"
    runtime_source.write_text(
        'from pathlib import Path\nRUN = Path(f"results/{run_id}")\n',
        encoding="utf-8",
    )

    assert "src/arac/dynamic_leak.py" in _error_paths(root)


@pytest.mark.parametrize(
    "source",
    [
        'from pathlib import Path\nRUN = Path("results" + "/" + run_id)\n',
        'from pathlib import Path\nTABLE = Path("paper" + suffix)\n',
    ],
)
def test_string_concatenation_static_prefix_is_reported(
    tmp_path: Path,
    source: str,
) -> None:
    root = _project_fixture(tmp_path)
    runtime_source = root / "src" / "arac" / "concatenated_leak.py"
    runtime_source.write_text(source, encoding="utf-8")

    assert "src/arac/concatenated_leak.py" in _error_paths(root)


def test_utf8_bom_runtime_source_is_valid_python(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    runtime_source = root / "src" / "arac" / "bom_source.py"
    runtime_source.write_bytes("from pathlib import Path\nVALUE = Path('src')\n".encode("utf-8-sig"))

    assert "src/arac/bom_source.py" not in _error_paths(root)


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


def test_vendor_result_generated_payload_must_be_ignored_by_git(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    ignore_file = root / ".gitignore"
    ignore_file.write_text(
        ignore_file.read_text(encoding="utf-8").replace(
            "vendor/hcc/result/*\n!vendor/hcc/result/README.md\n",
            "",
        ),
        encoding="utf-8",
    )

    assert any("vendor/hcc/result" in rule for rule in _error_rules(root))


def test_tracked_hcc_result_payload_is_reported(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    payload_path = "vendor/hcc/result/run.csv"
    payload = root / payload_path
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("generated\n", encoding="utf-8")
    _git(root, "add", "-f", payload_path)

    assert payload_path in _error_paths(root)


def test_hcc_result_readme_is_allowed(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    readme_path = "vendor/hcc/result/README.md"
    readme = root / readme_path
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("generated result contract\n", encoding="utf-8")
    _git(root, "add", readme_path)

    assert readme_path not in _error_paths(root)


def test_canonical_hcc_tree_has_one_external_runner_and_no_legacy_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "vendor" / "hcc"
    source = (repo_root / "vendor" / "hcc" / "HCC-ES.py").read_text(encoding="utf-8")

    assert not (repo_root / "HCC_SRC").exists()
    assert (repo_root / "scripts" / "hcc_smoke_runner.py").is_file()
    assert list(vendor_root.glob("*runner*.py")) == []
    assert "HCC_SRC/" not in source


@pytest.mark.parametrize("root_selector", ["nested", "parent"])
def test_root_must_equal_git_toplevel(tmp_path: Path, root_selector: str) -> None:
    root = _project_fixture(tmp_path)
    candidate = root / "src" if root_selector == "nested" else root.parent

    assert any("must equal Git top-level" in rule for rule in _error_rules(candidate))


def test_external_hcc_symlink_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project_fixture(tmp_path)
    hcc_src = root / "HCC_SRC"
    hcc_src.mkdir()
    outside = (tmp_path / "outside-hcc").resolve()
    outside.mkdir()
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve

    def fake_is_symlink(path: Path) -> bool:
        return path == hcc_src or original_is_symlink(path)

    def fake_resolve(path: Path, strict: bool = False) -> Path:
        if path == hcc_src:
            return outside
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "resolve", fake_resolve)

    report = audit_project_structure(root)

    assert any(
        finding.path == "HCC_SRC" and "outside repository" in finding.rule
        for finding in report.errors
    )
    assert not any(warning.path == "HCC_SRC" for warning in report.warnings)


def test_hcc_lstat_error_is_fatal_and_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project_fixture(tmp_path)
    hcc_src = root / "HCC_SRC"
    hcc_src.mkdir()
    original_lstat = audit_module.os.lstat

    def fail_lstat(path):
        if Path(path).name == "HCC_SRC":
            raise PermissionError("HCC_SRC metadata denied")
        return original_lstat(path)

    monkeypatch.setattr(audit_module.os, "lstat", fail_lstat)

    report = audit_project_structure(root)

    assert any(
        finding.path == "HCC_SRC" and "cannot inspect HCC_SRC" in finding.rule
        for finding in report.errors
    )
    assert not any(warning.path == "HCC_SRC" for warning in report.warnings)


def test_hcc_src_regular_file_is_fatal_legacy_path(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    (root / "HCC_SRC").write_text("not a directory\n", encoding="utf-8")

    report = audit_project_structure(root)

    assert any(
        finding.path == "HCC_SRC" and "legacy" in finding.rule
        for finding in report.errors
    )
    assert not any(warning.path == "HCC_SRC" for warning in report.warnings)


@pytest.mark.parametrize(
    "git_failure",
    [FileNotFoundError("git executable missing"), OSError("git cannot start")],
    ids=["file-not-found", "os-error"],
)
def test_cli_reports_git_execution_error_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    git_failure: OSError,
) -> None:
    root = _project_fixture(tmp_path)

    def fail_git(*args, **kwargs):
        raise git_failure

    monkeypatch.setattr(audit_module.subprocess, "run", fail_git)

    exit_code = main(["--root", str(root)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "audit error" in output
    assert "Traceback" not in output


def test_hcc_src_directory_is_fatal_after_vendor_migration(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    legacy_readme = root / "HCC_SRC" / "README.md"
    legacy_readme.parent.mkdir()
    legacy_readme.write_text("legacy source\n", encoding="utf-8")
    _git(root, "add", "HCC_SRC/README.md")

    report = audit_project_structure(root)

    assert any(
        finding.path == "HCC_SRC" and "legacy" in finding.rule
        for finding in report.errors
    )
    assert not any(warning.path == "HCC_SRC" for warning in report.warnings)


def test_hcc_src_does_not_hide_unknown_top_level_directory(
    tmp_path: Path,
    capsys,
) -> None:
    root = _project_fixture(tmp_path)
    (root / "HCC_SRC").mkdir()
    (root / "unexpected-output").mkdir()

    exit_code = main(["--root", str(root)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "HCC_SRC: legacy" in output
    assert "unexpected-output: unexpected top-level directory" in output


def test_cli_with_hcc_src_returns_nonzero_and_prints_fatal_error(
    tmp_path: Path,
    capsys,
) -> None:
    root = _project_fixture(tmp_path)
    (root / "HCC_SRC").mkdir()

    exit_code = main(["--root", str(root)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "HCC_SRC: legacy" in output


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
