"""Offline HCC action-trace readers."""

from __future__ import annotations

from pathlib import Path

def _find_hcc_action_trace(output_dir: Path) -> tuple[Path | None, int]:
    traces = sorted(Path(output_dir).rglob("action_trace.csv"))
    if not traces:
        return None, 0
    trace_path = traces[-1]
    with trace_path.open(newline="", encoding="utf-8") as handle:
        row_count = max(0, sum(1 for _ in handle) - 1)
    return trace_path, row_count


def _tail(text: str, max_chars: int = 2000) -> str:
    return (text or "")[-max_chars:]
