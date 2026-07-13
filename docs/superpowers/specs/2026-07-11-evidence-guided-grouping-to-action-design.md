# Evidence-Guided Grouping-to-Action Runtime Design

Date: 2026-07-11
Executor: Codex
Status: approved direction; written-spec review pending

## Canonical Research Direction

User-approved research direction:

> 面向大规模重叠全局优化，研究如何把协同进化得到的变量分组和重叠关系转化为运行时动作，动态调整共享变量写回、子问题资源分配和搜索起点，从而提升复杂重叠问题的优化性能。

English working formulation:

> For large-scale overlapping global optimization, transform cooperative-coevolution variable groups and overlap relations into runtime actions that dynamically adjust shared-variable writeback, subproblem resource allocation, and search starting states.

The project terminology is `grouping-to-action controller`. The method is not
defined as an optimizer portfolio, a collection of case-specific routes, or a
static grouping heuristic.

## Objective

Upgrade the existing single-lane canonical runtime so that grouping and overlap
evidence can select and execute three kinds of intervention:

1. shared-variable writeback actions;
2. subproblem resource-allocation actions;
3. search-state and search-start actions.

The immediate implementation target is a stateful search-state action and its
conservative budget arbitration. Existing relation writeback and group-budget
mechanisms remain the action executors around it. The final experiment entry
remains `canonical_evidence_controller_v1`; no parallel final runner or oracle
lane is introduced.

The performance target is to investigate the remaining R3 gap while preserving
the current 12/13 paper-best wins. Preservation is evaluated by three-seed
best-of-three, not by a historical single row or a fabricated mean.

## Evidence And Root Cause

The current evidence establishes four facts:

- The pinned canonical protocol has 12/13 best-of-three wins; R3 is the remaining
  miss.
- The recent bounded late refresh started a new MMES from an incumbent. It did
  not resume the Phase-I distribution state because MMES kept its distribution
  variables in local `optimize()` variables.
- A refresh can improve the incumbent immediately and still lose overall because
  its FE replaces a more productive continuation of canonical CC.
- R3's canonical CC gain rate can increase in later complete sweeps, so static
  conflict or stagnation gates are not sufficient evidence for taking budget
  away from CC.

The root problem is therefore not simply "the wrong threshold". The runtime is
missing an auditable stateful action boundary and an opportunity-cost comparison
between the competing search states.

## Non-Negotiable Boundaries

Runtime dispatch may use only current-run evidence:

- grouping topology, group sizes, overlap degree, and shared-variable support;
- group contribution and contribution dispersion;
- relation conflict, writeback magnitude, and writeback stability;
- recent gain, function evaluations, stagnation, and gain-rate trend;
- remaining budget and optimizer state statistics.

Runtime dispatch must never use:

- case identifiers or function-family labels;
- paper means, paper-best values, or relative gains to paper;
- historical final errors, historical experiment labels, or prior run outcomes;
- an unexecuted candidate's final result;
- multiple complete lanes followed by final-result selection.

Paper and historical values are offline evaluation inputs only.

## Considered Approaches

### A. Continue threshold tuning on cold-start refresh

Rejected. The observed failure is opportunity cost plus loss of optimizer state,
so changing conflict thresholds does not repair the execution semantics.

### B. Fixed CC/NDA budget split

Rejected as the main method. It is easy to implement but is not evidence-driven,
and a fixed split can remove useful CC budget from the existing winning cases.

### C. Stateful single-track grouping-to-action controller

Selected. Preserve the Phase-I MMES state, execute one sequential trajectory,
compare normalized marginal utility in matched blocks, and allocate a bounded
search-state budget only when grouping and runtime evidence justify it. The
canonical CC path retains a protected reserve and remains the fallback action.

## Runtime Architecture

### 1. Evidence snapshot

At the end of each complete CC outer sweep, construct an immutable evidence
snapshot. It contains structural evidence, relation evidence, and optimization
evidence. The snapshot is the only input to the action decision function.

The action decision is pure with respect to the snapshot and scheduler state. It
returns an action plus a reason and a budget plan; it does not execute an
optimizer or inspect offline comparison data.

### 2. Action families

The controller exposes three action surfaces:

- `writeback`: coordinate, isolate, repair, or protect shared-variable values;
- `resource`: retain canonical CC reserve or allocate a bounded block to a
  group/search state according to observed contribution and marginal utility;
- `search_state`: continue canonical CC, resume the saved Phase-I MMES state, or
  abstain and protect the incumbent.

The current iteration implements the stateful `search_state` executor and its
resource guard. Relation actions continue to use the existing v24/v26 controller
and are not retuned in this change.

### 3. Stateful MMES executor

The MMES backend gains an explicit state object containing all variables needed
for deterministic continuation:

```text
MMESState
  x, mean, p, w, q, t, v, y
  sigma, n_individuals, n_parents
  generation and restart counters
  best_so_far_x, best_so_far_y
  function-evaluation counters
  optimization and initialization RNG states
```

The executor provides:

- `initialize_state(...)`;
- `run_block(state, additional_function_evaluations)`;
- `state_to_result(state)`;
- the existing `optimize()` wrapper for backward compatibility.

Blocks stop only at a complete population boundary. A block reports its initial
and final incumbent, actual FE, normalized gain rate, termination reason, and a
state fingerprint. Invalid state shapes, non-finite values, RNG state mismatch,
or FE overrun invalidate the run instead of silently restarting.

### 4. Single-track phase transition

The run has one global incumbent and two sequential search states:

- canonical CC state, executed by the existing group loop;
- the saved Phase-I MMES state, resumed only when selected by runtime evidence.

The states are not run in parallel and are not treated as independent final
lanes. A decision point is reached only after a complete CC sweep or a complete
MMES block. Every accepted incumbent update and every rejected candidate remains
part of the same trajectory and FE ledger.

The first stateful probe is deliberately small:

- round the probe to a complete MMES population;
- cap each probe at 1% of total FE, subject to the remaining budget;
- allow at most two probe blocks before an expansion decision;
- cap cumulative search-state intervention at 15% of total FE in this version;
- retain a 10% total-FE reserve for canonical CC continuation whenever the
  intervention is eligible.

Probe eligibility requires a complete sweep, nonzero overlap, an enabled
phase-rescue boundary, no active repair lock, positive Phase-I tail utility, and
at least one current-run structural intervention signal. A structural signal is
either a non-coordinate relation majority, relation-conflict fraction of at
least 0.50, or the existing writeback-instability detector. These fields already
belong to the runtime controller; no case-specific rule is added.

The execution order is fixed:

1. complete one canonical CC sweep and record its actual gain and FE;
2. run one stateful probe only if the structural eligibility gate passes;
3. return to canonical CC for another complete sweep;
4. run one confirmation probe only if the first probe strictly improved the
   incumbent and still beats the updated CC opportunity-cost gate;
5. expand in population-rounded blocks only if both probes pass.

The base opportunity-cost gate requires MMES utility to be at least `1.5` times
the most recent complete-sweep CC utility. If CC utility increased across the
last two complete sweeps, the required ratio is `2.0`. A zero-gain or rejected
probe fails the gate. Any failed gate returns control to canonical CC and blocks
further stateful expansion for that run.

Normalized utility is computed from current-run values only:

```text
utility = max(0, incumbent_before - incumbent_after)
          / (max(abs(incumbent_before), 1) * max(actual_fe, 1))
```

The controller never compares a candidate to a paper threshold. A non-improving
candidate cannot replace the global incumbent, and an ineligible or ambiguous
decision is `continue_canonical_cc`.

### 5. Search-start semantics

The first implementation uses exact Phase-I state resumption as the safe search
start action. It deliberately does not mix a stale MMES distribution with a
later CC incumbent. Re-anchoring the distribution mean, resetting evolution
paths, or changing population size is a separate action and requires its own
ablation; it is out of scope for this implementation.

This keeps the first stateful action attributable: any effect comes from
continuing learned search state and reallocating a bounded budget, not from an
untracked collection of restart heuristics.

## Incumbent And Failure Policy

- The best protected incumbent is copied before every stateful action.
- Only strict objective improvement updates the global incumbent.
- A rejected candidate still consumes and reports its actual FE; it does not
  silently revert the budget ledger.
- State or objective exceptions fail the run audit explicitly.
- If evidence is incomplete, the scheduler abstains and canonical CC continues.
- If the stateful API is disabled, the canonical execution path and seed stream
  remain unchanged.

This is a safety policy, not a claim that runtime can know the paper-best value.
The 12/12 preservation decision is made offline after fresh executions.

## Audit Surface

Extend the existing action trace and same-budget ledger rather than creating a
second result schema. Each stateful decision records:

- decision point and completed sweep/block index;
- structural and relation evidence summary;
- CC and MMES block FE and normalized utility;
- selected action, abstain reason, and budget cap;
- protected CC reserve and cumulative state-action FE;
- state fingerprint before and after execution;
- incumbent before, candidate, acceptance, and incumbent after;
- optimizer seed, termination reason, and actual FE.

The manifest must identify the stateful MMES module hash, runner hash, AOB input
hashes, environment, thread settings, and anti-leakage result.

## Verification Plan

### Code-level tests

1. State round-trip preserves all continuation fields and RNG state.
2. One-block execution never exceeds its population-rounded FE cap.
3. `optimize()` and state-block execution are equivalent when the scheduler is
   disabled.
4. Non-finite, malformed, or mismatched state fails explicitly.
5. Pure action decisions reject forbidden dispatch fields.
6. Probe cap, cumulative cap, and CC reserve reconcile exactly.
7. A rejected stateful candidate preserves the protected incumbent.
8. Existing relation-policy and canonical trace tests remain green.

### Runtime pilot order

1. Run a small deterministic state-equivalence pilot.
2. Run one 3M-FE R3 diagnostic with stateful tracing enabled.
3. Run E6, S6, R2, and A4 preservation controls before broadening the action.
4. Run the 12 existing winning cases at three seeds only after all controls are
   fresh, same-budget clean, and anti-leakage clean.
5. Run the 13-case target protocol only if every existing best-of-three win is
   preserved.

### Adoption gates

- Existing 12 cases retain 12/12 best-of-three wins against their offline
  paper-best thresholds.
- R3 is reported separately as the target case; no R3 improvement is claimed
  from a historical row alone.
- No run has a `catastrophic_loss` under the existing 20% relative-loss audit
  threshold against its matched canonical execution.
- All runs have exact FE reconciliation, unchanged AOB inputs, clean
  anti-leakage audits, and recorded environment hashes.

## Scope And Follow-Ups

This design does not add a new optimizer, paper-specific route, or final-result
selector. It upgrades the existing controller so that grouping evidence can
select a real, auditable search-state action.

After the stateful action passes the preservation gate, a separate design may
evaluate evidence-driven re-anchoring and richer per-group resource allocation.
Those changes must not be bundled into the first state-continuation experiment.
