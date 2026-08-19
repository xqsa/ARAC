from __future__ import annotations

from experiments.historical_recovery.audit_retained_historical_sources import (
    _lane_verdict,
)


def test_exact_runner_without_checkpoint_is_only_partial() -> None:
    assert (
        _lane_verdict(
            receipt_present=True,
            exact_source_recovered=True,
            checkpoint_bound=False,
        )
        == "partial"
    )


def test_missing_receipt_and_source_is_missing() -> None:
    assert (
        _lane_verdict(
            receipt_present=False,
            exact_source_recovered=False,
            checkpoint_bound=False,
        )
        == "missing"
    )


def test_recovered_requires_all_three_bindings() -> None:
    assert (
        _lane_verdict(
            receipt_present=True,
            exact_source_recovered=True,
            checkpoint_bound=True,
        )
        == "recovered"
    )
