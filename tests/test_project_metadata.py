from __future__ import annotations

import csv
import re
import tomllib
from pathlib import Path


def test_hcc_optional_dependencies_cover_source_execution_imports() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    hcc_dependencies = metadata["project"]["optional-dependencies"]["hcc"]
    normalized = {re.split(r"[<>=!~]", dependency, maxsplit=1)[0] for dependency in hcc_dependencies}

    assert {"matplotlib", "numpy", "PyYAML", "scipy", "torch"} <= normalized


def test_hcc_vendor_migration_record_matches_completed_worktree_state() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    migration_path = repo_root / "docs" / "migrations" / "2026-07-12-path-migration.csv"
    with migration_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    row = next(item for item in rows if item["old_path"] == "HCC_SRC/")

    assert row["new_path"] == "vendor/hcc/"
    assert row["source_root"] == "canonical_worktree"
    assert row["source_state"] == "tracked"
    assert row["action"] == "migrated-task-3"
    assert "vendor/hcc exists" in row["verification"]
    assert "HCC_SRC absent" in row["verification"]
    assert "vendor/hcc/AOB exists" in row["verification"]
    assert "vendor/hcc/HCC exists" in row["verification"]
    assert "scripts/hcc_smoke_runner.py exists" in row["verification"]
    assert not (repo_root / "HCC_SRC").exists()
    assert (repo_root / "vendor" / "hcc").is_dir()
    assert (repo_root / "vendor" / "hcc" / "AOB").is_dir()
    assert (repo_root / "vendor" / "hcc" / "HCC").is_dir()
    assert (repo_root / "scripts" / "hcc_smoke_runner.py").is_file()
