from __future__ import annotations

from pathlib import Path

from experiments.historical_recovery.recovery_first_screen_campaign import (
    DEFAULT_PROTOCOL,
    EXPECTED_CASES,
    EXPECTED_MAPPING,
    EXPECTED_SEEDS,
    _contexts,
    _manifest,
    load_protocol,
)


def test_screen_protocol_freezes_120_mapped_action_arms() -> None:
    protocol = load_protocol()
    assert tuple(protocol["cases"]) == EXPECTED_CASES
    assert tuple(protocol["screen_seeds"]) == EXPECTED_SEEDS
    assert protocol["historical_action_mapping"] == EXPECTED_MAPPING
    assert protocol["patch_enabled"] is False
    assert protocol["soft_routing_enabled"] is False
    assert protocol["selector_enabled"] is False
    assert protocol["max_workers"] == 24


def test_screen_contexts_have_one_historical_action_per_case_seed(tmp_path: Path) -> None:
    protocol = load_protocol()
    manifest = _manifest(DEFAULT_PROTOCOL.resolve(), protocol)
    contexts = _contexts(protocol, tmp_path, str(manifest["manifest_sha256"]))
    assert len(contexts) == 120
    assert len({(context.case_id, context.run_seed) for context in contexts}) == 120
    assert {(context.case_id, context.action_name) for context in contexts} == set(EXPECTED_MAPPING.items())
    assert all(context.action_name == EXPECTED_MAPPING[context.case_id] for context in contexts)


def test_screen_manifest_binds_both_retained_source_trees() -> None:
    protocol = load_protocol()
    manifest = _manifest(DEFAULT_PROTOCOL.resolve(), protocol)
    assert manifest["checkpoint_tree"]["file_count"] == 600
    assert manifest["current_receipt_tree"]["file_count"] == 600
    assert len(manifest["manifest_sha256"]) == 64
