# ARAC-CAR-W2: Lazy Zero-Regret Probe Lease

Date: 2026-07-15
Executor: Codex
Status: frozen for CLI/5k parity only; no 3M utility claim

## Motivation

The CAR-W v1 diagnostic was integrity-valid but utility-invalid. It reserved
probe capacity and reshaped the early sweep schedule before a stable candidate
existed. All candidate probes then abstained, so the run paid opportunity cost
without deploying a candidate. CAR-W2 changes that control flow while keeping
the v1 risk gates unchanged.

The recent literature review establishes that overlap-aware writeback,
contribution-driven resource allocation, probing, common-random-number racing,
and lower-tail risk rules are established ingredients. The defensible ARAC
delta remains the combination of a same-checkpoint intervention, native
fallback pairing, one FE ledger, and atomic abstention. W2 therefore claims a
zero-regret lease protocol, not a new overlap heuristic.

## Runtime boundary

Dispatch may consume only current-run Phase-I/runtime evidence:

- grouping and overlap graph fingerprints;
- current component support and shared-variable count;
- two complete proposal sweeps;
- current overlap strength, feature coverage, and bounded writeback norm.

It must not consume case labels, function families, paper-best values,
historical results, final outcomes/errors, lane names, or any offline comparison.
The `DispatchEvidence` type boundary remains unchanged.

## W2 protocol

1. **Native prefix.** Before a stable component plan exists, execute the v33.8
   route with the same sweep budgets and ordering. W2 does not reserve probe FE,
   alter `sub_fes`, or change v33 trust/search-state transitions.
2. **Lazy discovery.** Capture proposal metadata as trace-only evidence during
   complete native sweeps. After two complete sweeps, freeze the same stable
   support-subgraph plan used by W1.
3. **Zero-FE futility screen.** Reject a plan if it has no stable non-fallback
   action or its bounded aggregate writeback norm is at most
   `1e-12`. This is a fixed structural screen, not a final-outcome filter. A
   rejected plan consumes zero probe FE and returns the untouched v33 state.
4. **Lease.** Only a surviving plan may request the fixed `3% * max_FE` probe
   cap. The cap is charged lazily from remaining FE; no reservation is made
   before discovery. If the complete paired horizon cannot fit, abstain with
   zero probe FE.
5. **Paired risk gate.** Use the existing three paired fallback/candidate
   component horizons, common-random-number descriptors, positive LCB, lower
   tail, endpoint, equal-FE, state-fingerprint, and branch-isolation gates.
   Do not relax or retune the W1 thresholds.
6. **Atomic adoption.** Adopt only the pre-registered final candidate pair when
   every gate passes. Otherwise adopt the paired fallback state. Never select
   the numerically best probe endpoint.

## Zero-regret invariant

For a run with no surviving W2 plan, the committed state, FE ledger,
optimizer route, and pending controller state are bit-equivalent to v33.8.
CAR artifacts may record an abstain row, but they cannot change the primary
trajectory. A no-overlap graph is a special case of this invariant.

## Falsifiable gates

The CLI/5k parity gate must show:

- no-candidate probe FE exactly zero;
- no-candidate v33/CAR final state and FE bit-equivalent;
- no early probe reservation or sweep-budget reshaping;
- AOB hashes unchanged, equal-FE, branch isolation, and anti-leakage pass;
- W1 behavior unchanged when the W1 action is selected.

Only after this gate passes may a fresh W2 3M diagnostic be registered. The
W2 diagnostic must report candidate coverage, commits, paired mean/median,
worst seed, catastrophic losses, and lease overhead. It cannot start R/S or
the full 24-case protocol until its pre-registered W gate passes.
