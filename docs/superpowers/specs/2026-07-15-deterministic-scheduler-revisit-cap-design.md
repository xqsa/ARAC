# Deterministic Scheduler Revisit-Cap Design

Date: 2026-07-15
Author: Codex
Status: frozen before new optimizer FE

## Decision

S22 is the final allowed correction to the component-lease route. It replaces
S21's nonstationary historical-cycle projection with an action-time scheduler
bound. It does not change v40 dispatch, candidates, optimizer state, RNG, FE,
or action dose. v40 remains trace-only until the fresh held-out coverage gate
passes.

## Runtime Inputs

The cap may use only values already available before the current group CMAES
starts:

- current objective FE and sweep-start FE;
- strict CC budget limit;
- current group index and current sweep uniform `sub_fes`;
- current optimizer's population-rounded FE cap;
- the ordered current-run group population sizes.

Problem/case/seed identity, function family, paper-best, historical best,
current action execution FE, resolution status or FE, local/component/neighbor
gain, overwrite/survival, and terminal outcome are prohibited.

## Bound

Let `R` be FE remaining at the current action decision. First simulate the
current group and every later group at their full scheduler caps, including
the one objective evaluation before each later optimizer. If this path hits
the strict `remaining <= population` guard, the revisit is not guaranteed.

Let `U` be that current-tail cap and `R_min = R - U`. The smallest possible
next-sweep uniform budget is `q0 = ceil(R_min / G)`. Because one FE less in the
current tail can cross a ceiling boundary, evaluate both `q0` and `q0 + 1`.
For each candidate `q`, bound every prefix optimizer by `max(q, population)`;
this dominates population rounding. The upper envelope decreases after the
first boundary because the lost current-tail interval is `G` FE while at most
`i < G` prefix budgets increase. The larger of the two boundary values is the
deterministic revisit cap.

The target is reachable only when the cap leaves more FE than the largest
population guard among the next-sweep prefix and target group. Otherwise the
controller must abstain. Unsupported shifted budgets also abstain.

## Trace Contract

Every v40 precision row must add exact integer state:

- `component_scheduler_sweep_start_fe`;
- `component_scheduler_cc_budget_limit_fe`;
- `component_scheduler_group_budget_fe`;
- `component_scheduler_optimizer_budget_fe`;
- `component_scheduler_population_sizes`;
- `component_scheduler_revisit_cap_fe`;
- `component_scheduler_revisit_reachable`;
- `component_scheduler_revisit_reason`.

The cap is computed before CMAES. Serialization may happen afterward, but no
post-action value may enter the computation.

## Staged Gate

1. TDD the E2 seed32 state, strict-tail rejection, unsupported-budget
   abstention, and exhaustive small discrete early-stop paths.
2. Prove v38/v40 CLI/5k behavior parity, schema isolation, AOB immutability,
   same-budget accounting, and anti-leakage.
3. Commit and freeze the implementation before new 3M FE.
4. Run fresh v40 A4/S2/E2 seeds 34/35/36 and v38 seed34 parity anchors at 3M.
5. Offline replay requires exact cap recomputation, no selected component
   overlap, 100% selected resolution, zero cap underprediction, at least six
   selected runs, three selected cases, and at least two cases with two
   selected seeds.

Any failure permanently stops this controller route. Passing permits only a
separately registered runtime mutex/cap pilot; it is not performance evidence.
