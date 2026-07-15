---
session_no: S15
suggested_title: "[ARAC] S16 CAR actionability 3M audit"
parent_session: S14
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

The CAR-W3 oracle actionability implementation and provenance gate are complete.
The latest 5k smoke passed; the next stage is the pre-registered six-case,
three-seed, 3M-FE offline audit. No long-horizon utility claim exists yet.

## Completed

- `src/arac/policy/oracle_actionability.py` and
  `docs/design/car-actionability-audit-protocol.md` define the offline-only
  estimand, fixed horizons, zero-safe log contrast, and leakage boundary.
- `scripts/hcc_smoke_runner.py` executes independent fallback/candidate lanes,
  applies the candidate writeback once, then resumes canonical HCC.
- `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py` requires a
  complete request fingerprint and SHA-256 binding for actionability trace,
  action trace, evaluation record, and budget summary before artifact reuse.
- `results/car_actionability_smoke_20260715_v5` passed: 4/4 fresh lanes,
  FE 4996..4999/5000 with zero violations, AOB 38/38 unchanged,
  anti-leakage 16/16, provenance 4/4 valid, and zero-cost E1/E2 abstention.
- Verification: tracked tests 753 passed and 1 skipped; focused actionability
  and runtime-boundary tests 357 passed and 1 skipped; compileall and diff
  check passed.

## Workspace state

The actionability freeze is prepared for a local commit on `main`. Results are
untracked/ignored. User-owned untracked FlyKI, exp006-008, manuscript, and
historical-report files must remain untouched. Refresh Git status and log before
using this card because the freeze commit is created after the card is written.

## Next steps

1. Confirm the freeze commit and an empty output directory, then run E1/E2/S3/
   R4/A5/E6, seeds 1/2/3, two actionability lanes, strict 3M FE, jobs 4.
2. Validate fresh provenance, FE, AOB hashes, anti-leakage, prefix/CRN/action
   identity, all reachable horizons, terminal completeness, and invalid-row
   metric blanking before reading utility.
3. Report terminal headroom, horizon sign agreement/rank reversal, mean, worst
   seed, meaningful wins, and catastrophic losses; do not start a critic unless
   the integrity-clean result supports it.

## Risks

- The first stable checkpoint is expected late in the trajectory, near 2.5M FE;
  5k ties contain no evidence about long-horizon action value.
- The 3M audit is an offline causal-label diagnostic, not deployable runtime
  utility. Oracle outcomes and paper-best values remain forbidden dispatch data.
- A positive oracle gap still does not pay for online label acquisition; a
  non-positive or unstable terminal gap ends selector development for July.

## Required reading

1. `.light/handoff/S15-arac-actionability-smoke-pass.md`
2. `.light/passport.yaml`
3. `docs/design/car-actionability-audit-protocol.md`
4. `docs/design/core-method.md`
5. `docs/design/boundaries.md`

## Prohibited

- Do not treat the handoff as current Git reality; refresh status/log first.
- Do not reuse old output directories or unprovenanced raw artifacts.
- Do not use case labels, function family, paper-best, historical best, final
  outcome, or oracle labels as runtime dispatch inputs.
- Do not tune thresholds or start full-24 before the six-case integrity and
  actionability result is read.
