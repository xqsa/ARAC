---
session_no: S11
suggested_title: "[ARAC] S12 CAR-W2 fresh diagnostic preregistration"
parent_session: S10
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

CAR-W2 lazy zero-regret lease is implemented and passed CLI/5k parity. The
fresh 5k E2 run had no stable two-sweep plan, so the intended W2 behavior was
observed: zero probe FE and v33-equivalent native trajectory. No 3M utility
diagnostic has started.

## Completed

- `docs/design/2026-07-15-car-w2-zero-regret-design.md` — frozen W2 protocol.
- `src/arac/actions/controller_profiles.py` — registered
  `arac_counterfactual_action_racing_w2`.
- `scripts/hcc_smoke_runner.py` — W2 native-prefix/lazy-lease route.
- `src/arac/backends/hcc_car.py` — optional fixed zero-FE structural futility
  screen.
- `results/car_w2_parity_5k_20260715_w2` and `_v33` — fresh E2 seed9 parity.

## Verification

- 158 focused tests passed; full tracked suite: 727 passed, 1 skipped.
- compileall passed; `git diff --check` passed.
- W2 and v33 E2 seed9 5k final error: `6.772894435189659e+12` for both.
- Budget summaries and FE: `fitness_record_fe=4996`, `same_budget_violation=0`
  for both.
- W2 action/overlap traces match v33; AOB manifest SHA256 matches.
- W2 ledger: `probe_fe=0`, `gate_result=abstain`,
  `abstain_reason=insufficient_complete_evidence_sweeps`.

## Next step

1. Review the W2 spec and commit locally.
2. Register a new 3M diagnostic only after confirming the frozen command and
   output schema. Use fresh seeds/cases distinct from the W1 diagnostic.
3. Require candidate coverage, commit count, paired utility, worst seed,
   catastrophic loss, lease overhead, AOB, FE, and anti-leakage gates before
   considering R/S or full-24.

## Blockers and risks

- W2 candidate coverage is zero at 5k when two native complete sweeps do not
  fit; this is expected but does not establish candidate utility.
- W1 remains a failed integrity-valid historical result and is immutable.
- Do not relax positive LCB, lower-tail, endpoint, or catastrophic gates.
- Do not use case labels, function families, paper-best, historical results,
  or final outcomes as runtime dispatch inputs.
- Do not start R/S or full-24 before a fresh W2 W gate passes.

## Must read

1. This card
2. `.light/passport.yaml`
3. `docs/design/2026-07-15-car-w2-zero-regret-design.md`
4. `docs/design/car-w-diagnostic-result-20260715.md`
5. `docs/design/core-method.md`
