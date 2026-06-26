from __future__ import annotations

import tomllib
from pathlib import Path


def test_hcc_optional_dependencies_cover_source_execution_imports() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    hcc_dependencies = metadata["project"]["optional-dependencies"]["hcc"]
    normalized = {dependency.split(">=", 1)[0] for dependency in hcc_dependencies}

    assert {"matplotlib", "numpy", "PyYAML", "scipy", "torch"} <= normalized
