"""Matched-host M0-M2 validation for the shared-patch kernel.

This is an offline attribution experiment.  It creates a known conflicting
overlap instance, freezes the proposal/checkpoint/ledger inputs, and forces the
operator host label to CTP or GSS.  The force is local to this experiment and
never calls the production planner or selector.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from arac.coordination.shared_patch import K_PATCH_FES, SharedPatchKernel, patch_stable_hash
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.overlap_value_aware_dispatch_gate15 import (
    DIMENSION,
    _combined_problem,
    _new_scheduler,
    _proposal_payload,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = Path(__file__).with_name("overlap_shared_patch_matched_host_gate_protocol.json")
HOSTS = ("ctp", "gss")
MODES = ("a0", "a1", "a2", "a3", "a4")
VISITS = 2


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-overlap-shared-patch-matched-host-protocol-v1",
        "dimension": DIMENSION,
        "conflict_mode": "conflicting",
        "hosts": list(HOSTS),
        "modes": list(MODES),
        "patch_lane_fes": K_PATCH_FES,
        "patch_rounds_per_visit": 4,
        "visits_per_arm": VISITS,
        "operator_reservation_fes": K_PATCH_FES * VISITS,
        "selector_participates": False,
        "production_planner_modified": False,
        "soft_routing_enabled": False,
        "soft_routing_stage": "post_m1_only",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"matched-host protocol drifted: {key}")
    if set(protocol.get("mode_definition", {})) != set(MODES):
        raise ValueError("matched-host ablation modes are incomplete")
    if tuple(protocol.get("fresh_seeds", ())) != (7301, 7302, 7303, 7304):
        raise ValueError("matched-host fresh seeds drifted")
    return protocol


def _proposal_hash(proposals: Sequence[object]) -> str:
    payload = []
    for proposal in proposals:
        payload.append(
            {
                "group": int(proposal.group),
                "values": [(int(variable), float(value)) for variable, value in proposal.values],
                "improvement": float(proposal.improvement),
                "uncertainty": [(int(variable), float(value)) for variable, value in proposal.uncertainty],
            }
        )
    return canonical_sha256(payload)


def _baseline_candidates(ledger: EvaluationLedger, structure, proposals, component, seed: int) -> np.ndarray:
    selected = tuple(proposal for proposal in proposals if proposal.group in component)
    if not selected:
        raise ValueError("selected component has no proposals")
    rng = np.random.default_rng(seed)
    base = np.asarray(ledger.best_x, dtype=float)
    candidates = np.repeat(base[np.newaxis, :], K_PATCH_FES * VISITS, axis=0)
    variables = tuple(sorted({variable for group in component for variable in structure.groups[group]}))
    owners = {
        variable: tuple(proposal for proposal in selected if any(item == variable for item, _ in proposal.values))
        for variable in variables
    }
    if any(not proposals_for_variable for proposals_for_variable in owners.values()):
        raise RuntimeError("baseline candidates lack a proposal for a component variable")
    for row in range(candidates.shape[0]):
        for variable in variables:
            proposal = owners[variable][row % len(owners[variable])]
            candidates[row, variable] = proposal.value(variable) + float(rng.normal(0.0, max(1e-12, proposal.sigma(variable))))
    np.clip(candidates, ledger.problem.lower_array, ledger.problem.upper_array, out=candidates)
    return candidates


@dataclass(frozen=True)
class ArmResult:
    host: str
    mode: str
    topology: str
    overlap_budget: int
    seed: int
    component: tuple[int, ...]
    scope: tuple[int, ...]
    checkpoint_hash: str
    proposal_hash: str
    route: str
    arbitration_only: bool
    patch_receipt_count: int
    patch_lane_fes: int
    consumed_fes: int
    expected_operator_fes: int
    candidate_trace_count: int
    state_trace_count: int
    radius_trace_count: int
    accepted_count: int
    final_error_before: float
    final_error: float
    anytime_auc: float
    strict_best: bool
    exact_fe: bool
    state_hashes: tuple[str, ...]
    radius_min: float
    radius_max: float
    u_max: float
    fe_classification: dict[str, int]


@dataclass(frozen=True)
class MatchedInputs:
    problem: object
    structure: object
    checkpoint_x: np.ndarray
    checkpoint_error: float
    proposals: tuple[object, ...]
    checkpoint_fes: int
    checkpoint_hash: str
    proposal_hash: str


def _build_inputs(topology: str, overlap_budget: int, seed: int) -> MatchedInputs:
    problem, structure, _ = _combined_problem("conflicting", topology, overlap_budget, seed)
    checkpoint_x, checkpoint_error, proposals, checkpoint_fes = _proposal_payload(problem, structure, seed)
    return MatchedInputs(
        problem=problem,
        structure=structure,
        checkpoint_x=np.asarray(checkpoint_x, dtype=float).copy(),
        checkpoint_error=float(checkpoint_error),
        proposals=tuple(proposals),
        checkpoint_fes=int(checkpoint_fes),
        checkpoint_hash=canonical_sha256({"x": [float(value) for value in checkpoint_x], "error": float(checkpoint_error), "fes": int(checkpoint_fes)}),
        proposal_hash=_proposal_hash(proposals),
    )


def _run_arm(inputs: MatchedInputs, topology: str, overlap_budget: int, seed: int, host: str, mode: str) -> ArmResult:
    problem = inputs.problem
    structure = inputs.structure
    checkpoint_x = inputs.checkpoint_x
    checkpoint_error = inputs.checkpoint_error
    proposals = inputs.proposals
    checkpoint_fes = inputs.checkpoint_fes
    ledger, scheduler = _new_scheduler(problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes)
    coordinator = scheduler.coordinator
    components = tuple(
        component
        for component in structure.connected_components()
        if len(component) > 1
        and set().union(*(set(structure.groups[group]) for group in component))
        & set(structure.shared_variables)
    )
    if not components:
        raise RuntimeError("matched host generator produced no overlap component")
    component = tuple(components[0])
    scope = tuple(sorted(set(coordinator._component_variables(component)) & set(structure.shared_variables)))
    if not scope:
        raise RuntimeError("matched host component has no shared variable scope")
    proposal_hash = inputs.proposal_hash
    checkpoint_hash = inputs.checkpoint_hash
    route = f"forced_{host}"
    before = float(ledger.best_error)
    accepted = 0
    state_hashes: list[str] = []
    candidate_trace_count = 0
    state_trace_count = 0
    radius_trace_count = 0
    radius_values: list[float] = []
    u_values: list[float] = []
    patch_receipt_count = 0
    patch_lane_fes = 0
    if mode == "a0":
        ledger.evaluate(_baseline_candidates(ledger, structure, proposals, component, seed ^ 0xA0))
    else:
        kernel = SharedPatchKernel()
        kernel_mode = {"a1": "v2", "a2": "candidates", "a3": "state", "a4": "full"}[mode]
        context_hash = patch_stable_hash(checkpoint_hash, proposal_hash, route)
        for visit in range(VISITS):
            result = kernel.apply(
                component,
                proposals,
                scope,
                context_hash,
                structure=structure,
                ledger=ledger,
                budget_fes=K_PATCH_FES,
                seed=seed ^ (visit + 1) ^ 0x51ED,
                mode=kernel_mode,
            )
            if result.budget_status != "executed" or result.consumed_fes != K_PATCH_FES:
                raise RuntimeError(f"{host}/{mode} patch lane did not execute exactly {K_PATCH_FES} FE")
            patch_receipt_count += 1
            patch_lane_fes += result.consumed_fes
            candidate_trace_count += len(result.candidate_trace)
            state_trace_count += len(result.u_trace)
            radius_trace_count += len(result.radius_trace)
            accepted += sum(int(trace.accepted) for trace in result.candidate_trace)
            radius_values.extend(float(value) for value in result.radius_trace)
            u_values.extend(float(value) for value in result.u_trace)
            state_hashes.append(result.state_hash)
    consumed = ledger.count - checkpoint_fes
    expected_operator_fes = K_PATCH_FES * VISITS
    final_error = float(ledger.best_error)
    return ArmResult(
        host=host,
        mode=mode,
        topology=topology,
        overlap_budget=overlap_budget,
        seed=seed,
        component=component,
        scope=scope,
        checkpoint_hash=checkpoint_hash,
        proposal_hash=proposal_hash,
        route=route,
        arbitration_only=False,
        patch_receipt_count=patch_receipt_count,
        patch_lane_fes=patch_lane_fes,
        consumed_fes=consumed,
        expected_operator_fes=expected_operator_fes,
        candidate_trace_count=candidate_trace_count,
        state_trace_count=state_trace_count,
        radius_trace_count=radius_trace_count,
        accepted_count=accepted,
        final_error_before=before,
        final_error=final_error,
        anytime_auc=0.5 * (before + final_error),
        strict_best=final_error <= before,
        exact_fe=consumed == expected_operator_fes,
        state_hashes=tuple(state_hashes),
        radius_min=min(radius_values, default=0.0),
        radius_max=max(radius_values, default=0.0),
        u_max=max(u_values, default=0.0),
        fe_classification={
            "proposal_fes": checkpoint_fes,
            "operator_reservation_fes": expected_operator_fes,
            "patch_lane_fes": patch_lane_fes,
            "arbitration_fes": 0,
            "sense_fes": 0,
            "tail_fes": 0,
        },
    )


def _context(protocol: Mapping[str, Any], topology: str, overlap_budget: int, seed: int, host: str) -> dict[str, Any]:
    inputs = _build_inputs(topology, overlap_budget, seed)
    arms = tuple(_run_arm(inputs, topology, overlap_budget, seed, host, mode) for mode in MODES)
    by_mode = {arm.mode: arm for arm in arms}
    return {
        "host": host,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "arms": [asdict(arm) for arm in arms],
        "matched_inputs": len({(arm.checkpoint_hash, arm.proposal_hash, arm.component, arm.scope) for arm in arms}) == 1,
        "route_is_forced_host": all(arm.route == f"forced_{host}" for arm in arms),
        "not_arbitration_only": all(not arm.arbitration_only for arm in arms),
        "exact_fe": all(arm.exact_fe for arm in arms),
        "strict_best": all(arm.strict_best for arm in arms),
        "patch_receipts_nonzero_a1_a4": all(by_mode[mode].patch_receipt_count > 0 for mode in MODES[1:]),
        "candidate_trace_nonempty_a1_a4": all(by_mode[mode].candidate_trace_count > 0 for mode in MODES[1:]),
        "state_trace_nonempty_a3": by_mode["a3"].state_trace_count > 0,
        "radius_trace_nonempty_a4": by_mode["a4"].radius_trace_count > 0,
        "state_hash_chain_present": all(len(by_mode[mode].state_hashes) == VISITS for mode in MODES[1:]),
        "u_bounded": all(arm.u_max <= 4.0 for arm in arms),
    }


def run_gate(protocol_path: Path = DEFAULT_PROTOCOL, *, stage: str = "m0") -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if stage not in {"m0", "m1", "m2"}:
        raise ValueError("stage must be m0, m1, or m2")
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    if stage == "m2":
        m1_path = output_root / "m1.json"
        if not m1_path.is_file() or not bool(_load_json(m1_path).get("gate_passed")):
            blocked = {
                "schema_version": "arac-overlap-shared-patch-matched-host-m2-v1",
                "stage": "m2",
                "context_count": 0,
                "contexts": [],
                "checks": {"m1_gate_required": False},
                "gate_passed": False,
                "performance_comparison_authorized": False,
                "blocked_reason": "M1 attribution gate must pass before M2 fresh-seed comparison",
            }
            blocked["result_hash"] = canonical_sha256(blocked)
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "m2.json").write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return blocked
    if stage == "m2":
        topologies = tuple(protocol["topologies"])
        budgets = tuple(int(value) for value in protocol["overlap_budgets"])
        seeds = tuple(int(value) for value in protocol["fresh_seeds"])
    else:
        topologies = tuple(protocol["preflight_topologies"])
        budgets = tuple(int(value) for value in protocol["preflight_overlap_budgets"])
        seeds = tuple(int(value) for value in protocol["preflight_seeds"])
    contexts = [
        _context(protocol, topology, budget, seed, host)
        for topology in topologies
        for budget in budgets
        for seed in seeds
        for host in HOSTS
    ]
    checks = {
        "context_count": len(contexts) == len(topologies) * len(budgets) * len(seeds) * len(HOSTS),
        "matched_inputs": all(row["matched_inputs"] for row in contexts),
        "forced_host_route": all(row["route_is_forced_host"] for row in contexts),
        "not_arbitration_only": all(row["not_arbitration_only"] for row in contexts),
        "exact_fe": all(row["exact_fe"] for row in contexts),
        "strict_best": all(row["strict_best"] for row in contexts),
        "patch_receipts_nonzero": all(row["patch_receipts_nonzero_a1_a4"] for row in contexts),
        "candidate_trace_nonempty": all(row["candidate_trace_nonempty_a1_a4"] for row in contexts),
        "a3_state_trace": all(row["state_trace_nonempty_a3"] for row in contexts),
        "a4_radius_trace": all(row["radius_trace_nonempty_a4"] for row in contexts),
        "state_hash_chain": all(row["state_hash_chain_present"] for row in contexts),
        "u_bounded": all(row["u_bounded"] for row in contexts),
    }
    if stage == "m1":
        checks["nested_ablation_declared"] = protocol["mode_definition"]["a3"]["candidate"] == protocol["mode_definition"]["a2"]["candidate"]
        checks["nested_ablation_declared"] = checks["nested_ablation_declared"] and protocol["mode_definition"]["a4"]["state"] == protocol["mode_definition"]["a3"]["state"]
    summary: dict[str, Any] = {
        "schema_version": f"arac-overlap-shared-patch-matched-host-{stage}-v1",
        "stage": stage,
        "protocol": {key: protocol[key] for key in ("hosts", "modes", "patch_lane_fes", "visits_per_arm", "selector_participates", "production_planner_modified", "soft_routing_enabled", "soft_routing_stage")},
        "context_count": len(contexts),
        "contexts": contexts,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "performance_comparison_authorized": stage == "m2" and all(checks.values()),
    }
    if stage == "m2" and contexts:
        comparisons = []
        for context in contexts:
            by_mode = {row["mode"]: row for row in context["arms"]}
            comparisons.append({
                "host": context["host"],
                "topology": context["topology"],
                "overlap_budget": context["overlap_budget"],
                "seed": context["seed"],
                "a4_vs_a1_final_error_delta": by_mode["a1"]["final_error"] - by_mode["a4"]["final_error"],
                "a4_vs_a2_final_error_delta": by_mode["a2"]["final_error"] - by_mode["a4"]["final_error"],
                "a4_acceptance": by_mode["a4"]["accepted_count"],
                "a4_radius_min": by_mode["a4"]["radius_min"],
                "a4_radius_max": by_mode["a4"]["radius_max"],
                "a4_u_max": by_mode["a4"]["u_max"],
            })
        summary["fresh_comparisons"] = comparisons
    output_root.mkdir(parents=True, exist_ok=True)
    summary["result_hash"] = canonical_sha256(summary)
    (output_root / f"{stage}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage", choices=("m0", "m1", "m2"), default="m0")
    args = parser.parse_args(argv)
    result = run_gate(args.protocol, stage=args.stage)
    print(json.dumps({"stage": args.stage, "gate_passed": result["gate_passed"], "checks": result["checks"]}, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_PROTOCOL", "load_protocol", "run_gate"]
