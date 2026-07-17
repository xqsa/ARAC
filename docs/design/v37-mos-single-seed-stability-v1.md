# v37 MOS Single-Seed Stability Protocol

Date: 2026-07-17
Executor: Codex

## Claim boundary

This protocol tests one claim: replacing only the group-local CMA Gaussian
direction construction in frozen v37 with mirrored orthogonal sampling can
make every preregistered fresh seed beat paper-best on at least 13 of the 24
AOB cases. The winning case set may differ by seed.

The experiment does not revive overlap scheduling or precision actions. It
does not tune sigma, thresholds, grouping, CSA, restarts, budgets, resources,
or controller behavior. `E:\HCC-main` and `vendor/hcc` remain read-only.

## Frozen sampler

For every allowed CMA generation, draw exactly one Gaussian matrix of shape
`(population, dimension)`, matching iid v37 RNG advancement. Use the first
`ceil(population / 2)` rows as base directions. Split them into blocks of at
most `dimension` rows, apply reduced QR to each block transpose, correct each
QR column by the sign of the corresponding `R` diagonal (zero is positive),
and restore the original Gaussian row radius.

Emit directions as `z0,-z0,z1,-z1,...`. For an odd population, append the last
unpaired base direction. Multiple blocks need only be orthogonal internally.
All unused Gaussian rows are still drawn and discarded. Numerical validation
failure raises an error; there is no iid fallback.

MOS applies only to primary group CMA, phase-rescue multistart CMA, and their
internal vendor restarts while the v37 profile is active. The runner's
independent search-state BIPOP branch is not part of v37 and therefore remains
iid. Phase-I MMES, CAR, full-space/diagonal backends, old precision protocols,
and all other non-v37 profiles also remain iid.

## Matrices and ordering

1. Complete seeds 91–95 by running the missing 16 cases under the frozen v37
   observer-parity lane, then merge with the existing 40 rows.
2. After MOS code and tests are committed, run E1/E2 seeds 1/2 at 5k for both
   arms. Use A4 seed 1 at 100k only if a real MOS generation was not observed;
   it may be escalated once to 3M.
3. Run all 24 cases for seeds 96–100 under paired iid/MOS arms.
4. Run all 24 cases for seeds 101–108 only after the development gate passes.

Seeds 96–108 had no local result artifacts when this protocol was frozen.
Their appearance in an unexecuted older matrix does not expose outcomes; that
older protocol is retired and will not consume them.

## Offline metrics

Paper-best is joined only after raw result hashes are frozen. For case `c` and
seed `s`, define `r(c,s)=error_mos(c,s)/paper_best(c)`. The thirteenth order
statistic `q13(s)` must be strictly below one for every seed. Ties are losses.

Case-level cross-seed comparison uses arithmetic mean error. The stable core
requires at least 10 cases winning in four of five development seeds and six
of eight confirmation seeds. The strict all-seed intersection and pairwise
Jaccard minimum/mean are reported separately.

Paired tail effect is
`tau=log(max(error_v37,1e-300)/max(error_mos,1e-300))`. Catastrophic means
`error_mos >= 1.2*max(error_v37,1e-300)`. Confirmation CVaR is the arithmetic
mean of the 20 smallest tau values and must be strictly positive.

Upper-tail preservation compares paired best-of-K normalized errors on the
same seeds. MOS must preserve both the number of cases below paper-best and
the 24-case mean log normalized best.

## Stop rule

Any development hard-gate failure permanently rejects this MOS v1 and stops
confirmation. A confirmation failure burns seeds 101–108 and forbids runtime
registration. No sampler, threshold, seed, or case adjustment is allowed
after observing an outcome under this protocol.
