from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = (
    "run_id",
    "problem_id",
    "relation_id",
    "group_left",
    "group_right",
    "shared_vars_count",
    "overlap_strength",
    "delta_signal",
    "rank_signal",
    "action_name",
    "action_family",
    "confidence",
    "trigger_reason",
)


@dataclass(frozen=True)
class RelationDispatchSummary:
    total_relations: int
    action_counts: dict[str, int]
    group_left_counts: dict[str, int]
    overlap_strength_mean_by_action: dict[str, float]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_latest_action_decision_csv(search_root: Path) -> Path:
    candidates = sorted(search_root.rglob("action_decision.csv"))
    if not candidates:
        raise FileNotFoundError(f"no action_decision.csv found under {search_root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def read_action_decision_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        return list(reader)


def summarize_rows(rows: list[dict[str, str]]) -> RelationDispatchSummary:
    action_counts: Counter[str] = Counter()
    group_left_counts: Counter[str] = Counter()
    overlap_totals: defaultdict[str, float] = defaultdict(float)
    overlap_counts: Counter[str] = Counter()

    for row in rows:
        action_name = row["action_name"]
        group_left = row["group_left"]
        overlap_strength = _as_float(row["overlap_strength"])
        action_counts[action_name] += 1
        group_left_counts[group_left] += 1
        overlap_totals[action_name] += overlap_strength
        overlap_counts[action_name] += 1

    means = {
        action_name: overlap_totals[action_name] / overlap_counts[action_name]
        for action_name in sorted(overlap_counts)
    }
    return RelationDispatchSummary(
        total_relations=len(rows),
        action_counts=dict(sorted(action_counts.items())),
        group_left_counts=dict(sorted(group_left_counts.items(), key=_group_left_sort_key)),
        overlap_strength_mean_by_action=means,
    )


def print_summary(path: Path, summary: RelationDispatchSummary) -> None:
    print(f"action_decision_csv: {path}")
    print(f"total relations: {summary.total_relations}")
    print("distribution of actions:")
    for action_name, count in summary.action_counts.items():
        print(f"  {action_name}: {count}")
    print("per-group-left histogram:")
    for group_left, count in summary.group_left_counts.items():
        print(f"  {group_left}: {count}")
    print("per-action overlap_strength mean:")
    for action_name, mean_value in summary.overlap_strength_mean_by_action.items():
        print(f"  {action_name}: {mean_value:.6f}")


def plot_action_frequency(summary: RelationDispatchSummary, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actions = list(summary.action_counts)
    counts = [summary.action_counts[action] for action in actions]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(actions, counts, color="#4C78A8")
    ax.set_title("Relation Dispatch Action Frequency")
    ax.set_xlabel("action_name")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug ARAC relation dispatch decisions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Specific action_decision.csv to inspect. Defaults to latest under --search-root.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=repo_root() / "results",
        help="Directory searched for the latest action_decision.csv.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="PNG path for the action frequency chart.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input if args.input is not None else find_latest_action_decision_csv(args.search_root)
    rows = read_action_decision_rows(input_path)
    summary = summarize_rows(rows)
    print_summary(input_path, summary)

    plot_output = args.plot_output
    if plot_output is None:
        plot_output = input_path.with_name("action_frequency.png")
    plot_action_frequency(summary, plot_output)
    print(f"action frequency plot: {plot_output}")
    return 0


def _as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"expected numeric overlap_strength, got {value!r}") from exc


def _group_left_sort_key(item: tuple[str, int]) -> tuple[int, int | str]:
    group_left, _ = item
    try:
        return (0, int(group_left))
    except ValueError:
        return (1, group_left)


if __name__ == "__main__":
    raise SystemExit(main())
