"""Tests for the pairwise_edge_ctp_v1 upgrade candidate (P0 / chain diagnostic)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arac.runtime.contracts import canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
P0_SUMMARY = REPOSITORY_ROOT / "artifacts/upgrade_pairwise_edge_ctp_v1_p0_v1/summary.json"
CHAIN_SUMMARY = REPOSITORY_ROOT / "artifacts/upgrade_pairwise_edge_ctp_v1_chain_diagnostic_v1/summary.json"


class _Interaction:
    def __init__(self, variable: int, source: int, target: int):
        self.variable = variable
        self.source_region = source
        self.target_region = target


class _Hyperedge:
    def __init__(self, variable: int, regions):
        self.variable = variable
        self.regions = tuple(regions)
        self.evidence = tuple(_Interaction(variable, regions[0], region) for region in regions[1:])


class _Evidence:
    def __init__(self, hyperedges, interactions):
        self.resolved_hyperedges = tuple(hyperedges)
        self.variable_region_interactions = tuple(interactions)


def test_pairwise_certificates_accept_only_two_region_edges() -> None:
    from experiments.upgrade.pairwise_edge_ctp_v1.p0_pairwise_evidence import pairwise_certificates

    evidence = _Evidence(
        hyperedges=(_Hyperedge(1, (10, 11)), _Hyperedge(2, (10, 11, 12))),
        interactions=(_Interaction(1, 10, 11),),
    )
    certificates, rejected = pairwise_certificates(evidence)
    assert [certificate["variable"] for certificate in certificates] == [1]
    assert rejected[0]["reason"] == "not_exactly_two_regions"
    assert rejected[0]["variable"] == 2


def test_pairwise_certificates_reject_unexplained_third_region_evidence() -> None:
    from experiments.upgrade.pairwise_edge_ctp_v1.p0_pairwise_evidence import pairwise_certificates

    evidence = _Evidence(
        hyperedges=(_Hyperedge(5, (10, 11)),),
        interactions=(_Interaction(5, 10, 11), _Interaction(5, 10, 12)),
    )
    certificates, rejected = pairwise_certificates(evidence)
    assert certificates == []
    assert rejected[0]["reason"] == "unexplained_third_region_evidence"
    assert rejected[0]["unexplained_targets"] == [12]


def test_p0_protocol_pins_preregistered_values() -> None:
    from experiments.upgrade.pairwise_edge_ctp_v1.p0_pairwise_evidence import DEFAULT_PROTOCOL, load_protocol

    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol["cell"] == "pairs3-strong"
    assert protocol["discovery_seeds"] == [20270401, 20270402, 20270403, 20270404, 20270405]
    assert protocol["dsm_budget_cap"] == 100_000
    assert protocol["precision_required"] == 1.0
    assert protocol["recall_required"] == 0.9


@pytest.mark.skipif(not P0_SUMMARY.is_file(), reason="P0 summary artifact not present")
def test_p0_summary_contract() -> None:
    summary = json.loads(P0_SUMMARY.read_text(encoding="utf-8"))
    claimed = summary.pop("result_hash")
    assert claimed == canonical_sha256(summary)
    assert summary["patch_stage_authorized"] is summary["gate_passed"]
    assert summary["performance_comparison_run"] is False
    assert len(summary["seed_rows"]) == 5
    for row in summary["seed_rows"]:
        # certificates never lie, whatever the recall
        assert row["precision"] in (0.0, 1.0)


@pytest.mark.skipif(not CHAIN_SUMMARY.is_file(), reason="chain diagnostic artifact not present")
def test_chain_diagnostic_summary_contract() -> None:
    summary = json.loads(CHAIN_SUMMARY.read_text(encoding="utf-8"))
    claimed = summary.pop("result_hash")
    assert claimed == canonical_sha256(summary)
    assert summary["diagnostic_only"] is True
    assert summary["performance_claim_authorized"] is False
    assert summary["conclusion"] == "pair-specific residual evidence separates chain links"
    for row in summary["seed_rows"]:
        assert row["all_links_separable"] is True
        assert row["consumed_fes"] > 0
