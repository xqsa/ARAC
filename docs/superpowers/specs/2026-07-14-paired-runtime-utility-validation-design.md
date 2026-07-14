# Paired Runtime Utility Validation Design

Date: 2026-07-14
Executor: Codex
Status: pre-registered remediation protocol

## Purpose

Repair the experimental contract before introducing another controller version.
The protocol isolates three confounds that make v35-v39 unsuitable as utility
evidence: runtime environment drift, missing high-level HCC action bindings,
and the absence of a same-environment, same-seed, same-FE fallback reference.

This protocol does not change runtime action logic. It validates whether the
existing v36 first-sweep maturity candidate has held-out utility over the
canonical v33.8 controller after both are run under an identical contract.

## Root-Cause Basis

1. The same code and seed produced materially different final errors under the
   `D:/python` and `E:/ARAC/.venv` numerical stacks.
2. v35-v38 emitted inner action traces, but `action_execution_plan.csv` marked
   them as `unknown_action`, `optimizer_consumed=0`, and
   `runtime_dispatch_allowed=0`.
3. Single-controller profiles emitted `no_fallback_reference`, so paper-best
   comparisons could not establish same-budget action utility.
4. Seeds 1/2/3 were repeatedly used for development and are not an admissible
   release set for another method change.

## Frozen Runtime Environment

Every controller profile in this protocol must run through the repository
`.venv` and match all of these values before an output directory is created:

```text
Python 3.12.13
NumPy 2.3.5
SciPy 1.18.0
Torch 2.12.1
matplotlib 3.11.0
PyYAML 6.0.3
cma 4.4.4
scipy-openblas 0.3.30
```

The environment probe is a shared stable library contract consumed by both
exp003 and exp005. An environment mismatch is a hard pre-execution failure.

## Single Metadata Source

Controller action metadata for v33-v39 must be registered once and consumed by:

- `src/arac/backends/hcc_plan.py` for optimizer-consumed action bindings;
- `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py` for lanes and
  trace capabilities;
- `scripts/hcc_smoke_runner.py` for cumulative controller capabilities.

Unknown or non-consumed actions are hard failures before optimizer execution.
The registry contains no case identifier, function family, reported value,
historical result, relative gain, or final outcome.

## Paired Lanes

The `paired_v33_v36_runtime_utility` profile contains four same-budget lanes:

| Lane | Runtime action | Role |
| --- | --- | --- |
| `fallback` | `arac_evidence_action_controller_v33` | canonical paired reference |
| `candidate` | `arac_evidence_action_controller_v36` | action candidate |
| `shuffled_relation_dispatch` | deterministic shuffled relation policy | shuffled negative control |
| `no_action_negative_control` | `conservative_no_action` | no-action negative control |

All lanes use the same case, seed, FE limit, AOB inputs, backend source, and
numerical environment. Paper values are joined only after runtime completion.

## Verification Ladder

1. Focused tests must prove exact environment matching, fail-fast mismatch,
   v35-v38 HCC bindings, registry-derived lanes, and fail-fast unknown actions.
2. The tracked test suite, compileall, and diff checks must pass.
3. A real-HCC CLI probe and 5k smoke must run the paired profile on dense,
   non-dense, and no-overlap routes.
4. Only after every integrity gate passes, run the current-winning 13 cases
   (`A4`, `A5`, `E1`, `E2`, `E3`, `E4`, `E6`, `R1`, `R2`, `S2`, `S3`,
   `S5`, `S6`) with held-out seeds 4/5/6/7/8 and strict 3M FE.
5. Run full-24 only if the held-out utility gate below passes.

## Integrity Gate

Every configured trajectory must satisfy all of the following:

- fresh optimizer execution;
- no configured or observed FE overspend;
- unchanged AOB input hashes;
- all anti-leakage rows pass;
- all action plans are known, optimizer-consumed, and dispatch-allowed;
- expected backend semantics and required trace capabilities are present;
- the pinned environment audit passes;
- no case-specific or outcome-specific runtime dispatch exists.

Any failure blocks the held-out or full-24 stage.

## Held-Out Utility Gate

For each case/seed pair define the primary paired statistic:

```text
paired_log_error_delta = log1p(candidate_error) - log1p(v33_fallback_error)
```

Lower is better. The held-out candidate passes only if all conditions hold:

- aggregate mean paired log-error delta is strictly below zero;
- candidate arithmetic-mean error wins at least 7/13 cases;
- candidate worst-seed error wins at least 5/13 cases;
- candidate records meaningful wins on at least 33/65 paired trajectories;
- candidate records zero catastrophic losses versus v33.8, where catastrophic
  means at least 20% relative degradation under the existing utility classifier;
- neither negative control stably outperforms the candidate on a majority of
  the 65 paired trajectories;
- every integrity gate remains green.

This is a release gate, not a significance claim. Failure stops execution and
must be reported as evidence that the current Phase-I evidence-to-action mapping
does not identify safe long-horizon utility.

## Conditional Full-24 Reporting

If the held-out gate passes, run all E1-E6/S1-S6/R1-R6/A1-A6 with the same
four lanes, held-out seeds 4-8, and strict 3M FE. Report paired log-error,
arithmetic mean, worst seed, seed wins, catastrophic losses, and negative
controls as primary results. Join frozen paper-best values offline as a
secondary report and state whether best-of-five reaches at least 13/24.

## Runtime Legality

Runtime dispatch may consume only current-run Phase-I structure, relation,
group, budget, trust, maturity, and optimizer state already registered by the
controller. Case labels are execution identities only. Function family,
paper-best, historical or prior-run outcomes, final error, relative gain, and
offline success labels are forbidden dispatch inputs.

`[CONTRACT-ACKNOWLEDGED]`
