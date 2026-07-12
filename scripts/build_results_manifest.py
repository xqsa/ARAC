"""Build a deterministic, read-only index of generated experiment outputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FIELDNAMES = (
    "experiment_id",
    "protocol",
    "git_commit",
    "config_path",
    "seed",
    "case_id",
    "total_fe",
    "status",
    "claim_level",
    "output_path",
)

_METADATA_PATTERNS = {
    "protocol": re.compile(r"^Protocol:\s*(?P<value>.+?)\s*$", re.MULTILINE),
    "config_path": re.compile(r"^Config(?:uration)?:\s*(?P<value>.+?)\s*$", re.MULTILINE),
    "git_commit": re.compile(r"^-\s*git commit:\s*(?P<value>.+?)\s*$", re.MULTILINE),
    "claim_level": re.compile(r"^Claim level:\s*(?P<value>.+?)\s*$", re.MULTILINE),
    "budget": re.compile(r"^Budget:\s*(?P<value>[0-9]+)\s*FE", re.MULTILINE),
}


def _manifest_metadata(path: Path) -> dict[str, str]:
    manifest = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for field, pattern in _METADATA_PATTERNS.items():
        match = pattern.search(manifest)
        if match:
            values[field] = match.group("value").strip()
    return values


def _ledger_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return [{}]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows or [{}]


def _row(
    results_root: Path,
    run_dir: Path,
    metadata: dict[str, str],
    ledger: dict[str, str],
) -> dict[str, str]:
    case_id = ledger.get("problem_id", "")
    seed = ledger.get("seed", "")
    total_fe = ledger.get("total_fe", "") or metadata.get("budget", "")
    required = (
        metadata.get("protocol"),
        metadata.get("git_commit"),
        metadata.get("claim_level"),
        metadata.get("budget"),
        case_id,
        seed,
        total_fe,
    )
    status = "complete" if all(required) and ledger.get("same_budget_violation", "") == "0" else "partial"
    return {
        "experiment_id": run_dir.name,
        "protocol": metadata.get("protocol", ""),
        "git_commit": metadata.get("git_commit", ""),
        "config_path": metadata.get("config_path", ""),
        "seed": seed,
        "case_id": case_id,
        "total_fe": total_fe,
        "status": status,
        "claim_level": metadata.get("claim_level", ""),
        "output_path": run_dir.relative_to(results_root).as_posix(),
    }


def build_manifest(results_root: Path | str, output: Path | str) -> Path:
    root = Path(results_root).resolve()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        manifest_path = run_dir / "run_manifest.md"
        metadata = _manifest_metadata(manifest_path) if manifest_path.is_file() else {}
        for ledger in _ledger_rows(run_dir / "same_budget_ledger.csv"):
            rows.append(_row(root, run_dir, metadata, ledger))
    rows.sort(key=lambda row: (row["experiment_id"], row["case_id"], row["seed"], row["output_path"]))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    results_root = args.results if args.results.is_absolute() else args.root / args.results
    output = args.output if args.output.is_absolute() else args.root / args.output
    build_manifest(results_root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
