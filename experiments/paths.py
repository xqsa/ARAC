"""Repository-relative paths shared by experiment entrypoints."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    """Return the ARAC repository root independently of the caller's cwd."""

    return Path(__file__).resolve().parents[1]


def results_root() -> Path:
    """Return the generated-results directory for this repository."""

    return repository_root() / "results"


def experiment_results_dir(experiment_id: str) -> Path:
    """Return the canonical generated-results directory for one experiment."""

    return results_root() / experiment_id


def resolve_repository_path(path: Path | str) -> Path:
    """Resolve a relative experiment path against the repository root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate
