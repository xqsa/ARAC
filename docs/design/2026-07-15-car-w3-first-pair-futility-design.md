# ARAC-CAR-W3: First-Pair Futility Abort

Date: 2026-07-15
Executor: Codex
Status: frozen for CLI/5k parity and executor validation only

## Evidence basis

CAR-W2 passed every integrity gate but failed utility. Eleven stable graph
plans consumed a three-pair lease; all eleven abstained. The first pair was
already non-positive in 9/11 graph probes. Continuing those nine probes could
not satisfy the frozen lower-tail gate because the first pair would remain the
minimum or a negative member of the final K=3 sample.

This redesign is therefore a logical early stop, not a threshold fitted to
case outcomes. It preserves the same candidate, candidate alpha, branch
horizon, common-random-number seed, LCB, lower-tail, endpoint, and final-pair
deployment rules.

## Runtime boundary

W3 consumes only the same identity-free `DispatchEvidence` and paired branch
observations already admitted by CAR. Case labels, function families,
paper-best, historical results, final errors/outcomes, and offline win or
catastrophic labels remain forbidden runtime inputs.

## Protocol

1. Run the native v33.8 prefix with no pre-candidate FE reservation.
2. Freeze a stable component plan after two complete native sweeps. A missing
   or structurally futile plan consumes zero probe FE and preserves v33 state.
3. Allocate the same fixed 3% maximum lease and the same equal arm horizon as
   W2, but reserve only one pair at a time.
4. Execute pair 0 with isolated fallback and candidate evaluators and the same
   counter-based random descriptor.
5. Abort immediately and adopt pair-0 fallback when either condition holds:
   `normalized_delta <= 1e-12`, or `candidate_after > checkpoint + 1e-12`.
   Emit `futility_pair_not_positive` and/or
   `candidate_endpoint_worse_than_start`. Do not reserve pairs 1 or 2.
6. Only a positive, endpoint-safe pair 0 may execute pairs 1 and 2. The
   original K=3 LCB, lower-tail, endpoint, fingerprint, and atomic final-pair
   gate then decides commit versus fallback.

The futility rule is safe with respect to the frozen W gate: a non-positive
pair 0 cannot pass `Tail >= 0` and `LCB > epsilon`. Early abort therefore
removes executions that are already logically unable to commit; it does not
convert a rejected candidate into an accepted one.

## Required verification

- W1 and W2 still execute three pairs when their existing actions are used.
- W3 negative pair 0 produces one observation, two branch manifests, equal
  arm FE, one-pair ledger charge, and fallback adoption.
- W3 positive pair 0 proceeds to all three pairs and the unchanged risk gate.
- Branch-order swap is invariant.
- No-plan CLI/5k W3 is bit-equivalent to v33 with zero probe FE.
- AOB input, anti-leakage, type boundary, and total FE gates pass.

No W3 3M diagnostic is authorized by this document. A fresh diagnostic needs
a separate preregistration after the verification gate passes.
