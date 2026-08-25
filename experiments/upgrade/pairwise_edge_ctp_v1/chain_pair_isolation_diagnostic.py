"""chain_pair_isolation_diagnostic_v1 - v5.1 chain separability diagnostic.

Diagnostic only; no performance claim and no gate.  Question (v5.1 plan):
can the middle blocks' left-side and right-side shared variables of a chain
topology be separated by pair-specific residual evidence, or is chain
overlap fundamentally uncertifiable under the current evidence model?

For every planted link (a, b) of ``chain4-strong`` and every shared
variable j of that link, this diagnostic measures three mixed-difference
residual interactions on a fresh diagnostic ledger:

- j vs block a's residual (block a minus every shared variable): expected
  interaction (j enters block a's linkage sum);
- j vs block b's residual: expected interaction (j enters block b's
  linkage sum);
- j vs the residual of a block j does NOT belong to (the far side of the
  chain): expected NO interaction.

A link is ``pairwise_separable`` when every one of its shared variables
passes all three tests.  Ground truth is used to construct the probes -
this is an offline instrument diagnostic, not a runtime path.  The result
explains the v5.0 chain failure: pairwise residual evidence separates the
links even though the transitive resolved-hyperedge construction does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from arac.evidence.soft_rddsm import _rdg_interact
from arac.evidence.structural import _bounded_steps
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.upgrade.hyperedge_ctp_v1.h0_sidecar import GENERATOR_FREEZE
from experiments.upgrade.shared_patch_v1.conflicting_generator_v3 import build_v3_problem


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("chain_pair_isolation_diagnostic_protocol_v1.json"
)
CELL = "chain4-strong"
DIAGNOSTIC_SEEDS = (20270421, 20270422, 20270423)
THRESHOLD = 1e-13


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-upgrade-chain-pair-isolation-diagnostic-protocol-v1",
        "candidate_id": "pairwise_edge_ctp_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "cell": CELL,
        "diagnostic_seeds": list(DIAGNOSTIC_SEEDS),
        "threshold": THRESHOLD,
        "output_root": "artifacts/upgrade_pairwise_edge_ctp_v1_chain_diagnostic_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"chain diagnostic protocol drifted: {key}")
    return protocol


def run_diagnostic_seed(cell_id: str, seed: int) -> dict[str, Any]:
    problem, truth = build_v3_problem(
        cell_id,
        seed,
        conditioning=GENERATOR_FREEZE["conditioning"],
        shared_width=int(GENERATOR_FREEZE["shared_width"]),
        linkage_lambda=float(GENERATOR_FREEZE["linkage_lambda"]),
    )
    ledger = EvaluationLedger(problem, total_budget=3_000_000)
    centre = (problem.lower_array + problem.upper_array) / 2.0
    centre_value = float(np.asarray(ledger.evaluate(centre[np.newaxis, :])).reshape(-1)[0])
    rng = np.random.default_rng(seed ^ 0x57A6C7)
    steps = _bounded_steps(problem, centre, rng)
    shared_by_link: dict[tuple[int, int], list[int]] = {}
    for variable, left, right in truth.shared_owner_pairs:
        shared_by_link.setdefault((left, right), []).append(variable)
    all_shared = set(truth.shared_variables)
    link_rows = []
    for link in sorted(shared_by_link):
        left, right = link
        members = sorted(shared_by_link[link])
        far_blocks = [block for block in range(10) if block not in {left, right}]
        var_rows = []
        for variable in members:
            def residual(block_index: int) -> list[int]:
                return [v for v in truth.planted_blocks[block_index] if v not in all_shared]

            near_left = _rdg_interact(
                problem, ledger, set_a=[variable], set_b=residual(left),
                base_point=centre, base_value=centre_value, threshold=THRESHOLD,
            )
            near_right = _rdg_interact(
                problem, ledger, set_a=[variable], set_b=residual(right),
                base_point=centre, base_value=centre_value, threshold=THRESHOLD,
            )
            far_results = {}
            for far in far_blocks[:2]:
                far_results[far] = _rdg_interact(
                    problem, ledger, set_a=[variable], set_b=residual(far),
                    base_point=centre, base_value=centre_value, threshold=THRESHOLD,
                )
            var_rows.append(
                {
                    "variable": variable,
                    "interacts_own_left_residual": bool(near_left),
                    "interacts_own_right_residual": bool(near_right),
                    "interacts_far_residuals": {str(far): bool(value) for far, value in far_results.items()},
                    "isolated": bool(near_left and near_right and not any(far_results.values())),
                }
            )
        link_rows.append(
            {
                "link": list(link),
                "shared_variable_count": len(members),
                "isolated_count": sum(1 for row in var_rows if row["isolated"]),
                "pairwise_separable": all(row["isolated"] for row in var_rows),
                "variables": var_rows,
            }
        )
    return {
        "seed": seed,
        "consumed_fes": ledger.count,
        "link_rows": link_rows,
        "all_links_separable": all(row["pairwise_separable"] for row in link_rows),
    }


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("chain diagnostic refuses to run: the freeze verifier is not green")
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    if output_root.exists():
        raise FileExistsError(f"chain diagnostic output already exists: {output_root}")
    manifest = {
        "schema_version": "arac-upgrade-chain-diagnostic-manifest-v1",
        "protocol_sha256": _sha256(resolved),
        "verifier_report": verifier_report,
        "diagnostic_only": True,
        "performance_claim": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    output_root.mkdir(parents=True)
    _write_json(output_root / "manifest.json", manifest)
    rows = []
    with threadpool_limits(limits=1):
        for seed in protocol["diagnostic_seeds"]:
            rows.append(run_diagnostic_seed(protocol["cell"], int(seed)))
    body = {
        "schema_version": "arac-upgrade-chain-diagnostic-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "cell": protocol["cell"],
        "seed_rows": rows,
        "diagnostic_only": True,
        "performance_claim_authorized": False,
        "conclusion": (
            "pair-specific residual evidence separates chain links"
            if all(row["all_links_separable"] for row in rows)
            else "chain overlap is not separable by pair-specific residual evidence under the current budget and evidence model"
        ),
    }
    body["result_hash"] = canonical_sha256(body)
    _write_json(output_root / "summary.json", body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    summary = run_stage(args.protocol)
    print(json.dumps({"stage": "chain_diagnostic", "all_seeds_separable": all(row["all_links_separable"] for row in summary["seed_rows"]), "conclusion": summary["conclusion"]}, indent=2, sort_keys=True))
    return 0


__all__ = ["load_protocol", "run_diagnostic_seed", "run_stage"]


if __name__ == "__main__":
    raise SystemExit(main())
