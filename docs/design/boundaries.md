# Method Boundaries

Date: 2026-07-18
Status: exp_018 observer-only boundary

## In Scope

- RDDSM structural grouping from the AOB design matrix.
- Reference-blind ordering of overlapping groups.
- Same-checkpoint owner and bridge probes at the Phase-I boundary.
- Strict accounting of native, probe, and overhead FE.
- Delayed owner survival and overwrite evaluation.
- Shuffled negative control and catastrophic-risk gates.

## Out of Scope

- Replacing RDDSM with a new grouping algorithm.
- Dynamic regrouping after the Phase-I boundary.
- Writing probe candidates into the incumbent or cooperative context.
- Migrating CMA/MMES state between topologies.
- Selecting a runtime action from oracle labels or final outcomes.
- Claiming terminal improvement from an observer-only pilot.

## Runtime Inputs

Allowed inputs are current-trajectory quantities: group membership reconstructed
from the design matrix, group intersections, proposal disagreement, owner
priority and reliability, completed-sweep state, checkpoint hashes, and the
strict remaining-FE ledger.

Forbidden inputs include `Pvector`, `subgroups`, AOB truth labels, final error,
relative gain, paper-reported baselines, problem-family labels, prior pilot
outcomes, and any offline promotion label. Problem IDs may identify files and
offline joins but may not trigger policy behavior.

## Isolation Contract

Probe calls may append evaluation records only. They must not alter native RNG,
incumbent, cooperative context, optimizer state, grouping result, controller,
or Phase-II topology. The topology is frozen after the checkpoint. Every shadow
Shadow decision artifacts have `runtime_authorized=0`. A separately compiled
`RuntimeProbeAction` may be authorized only by the runtime ledger, with exact
shared values, local anchor/checkpoint validation, and one-shot consumption.

## Claim Ladder

1. Unit and contract tests pass.
2. Mechanical smoke completes with paired checkpoints and exact FE ledgers.
3. Mechanism pilot completes with fresh trajectories and closed delayed labels.
4. The frozen promotion gate returns pass.
5. A separate action-v2 may then be designed on fresh seeds.

Steps 1-4 do not themselves authorize runtime action or establish performance
improvement.
