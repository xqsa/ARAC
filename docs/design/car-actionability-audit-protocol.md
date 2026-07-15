# CAR Actionability Audit Protocol

Date: 2026-07-15
Executor: Codex
Status: frozen for the first CAR-W3 audit

## Purpose

The audit separates three questions that the runtime CAR-W3 gate cannot answer
from short overlap evidence alone:

1. Does the frozen one-shot writeback action have any long-horizon headroom?
2. Does a short post-checkpoint signal identify that headroom?
3. Does online probing retain positive utility after its FE opportunity cost?

This first protocol measures only question 1. The raw terminal outcomes are
offline labels and are never runtime dispatch inputs.

## Estimand and Arms

At the first checkpoint with two complete CAR-W3 evidence sweeps, two fresh
subprocess lanes are started with the same seed, AOB input hashes, canonical
prefix, checkpoint fingerprint, and optimizer configuration.

The fallback lane resumes the canonical policy without writeback. The
candidate lane applies the frozen graph action exactly once at the checkpoint,
then resumes the same canonical policy. Both lanes charge every evaluator call
to their own same-budget ledger. The actionability contrast at horizon `h` is:

```text
Y_h = log(error_fallback,h) - log(error_candidate,h)
```

Positive `Y_h` means the candidate has lower error. A configured zero error is
represented with the pre-registered `log_floor=1e-300` only for finite logging;
the raw error remains zero.

## Horizons

The intervention itself is one complete population-valid component horizon.
After it, labels are sampled at absolute FE targets `1x`, `3x`, and `9x` of
that intervention budget, followed by the terminal population-complete endpoint.
The trace labels are therefore `closure_1`, `budget_3x`, `budget_9x`, and
`terminal`; the latter three are canonical continuation horizons, not cloned
MMES/CMA state checkpoints.

The configured cap is recorded separately from the observed endpoint. HCC may
stop a few FE below the cap at a complete population boundary. A paired
horizon is valid only when both lanes have the same checkpoint/target/observed
FE, matching prefix state and prefix-record hashes, equal intervention FE, and
`horizon_status=complete`. Otherwise the offline summary is blocked.

## Integrity Gates

- Fresh optimizer execution for every lane.
- A per-lane `car_actionability_provenance.json` must be `complete`, match the
  immutable request fingerprint, and bind the execution dependency hashes,
  Python/package environment, AOB input-content snapshot, actionability trace,
  action trace, evaluation record, budget summary, and AOB input manifest.
  Artifact paths and contents are checked by resolved path and SHA-256. A raw
  trace's self-reported fresh flag is never sufficient for resume.
- No FE overrun; equal absolute FE at every paired horizon.
- Exact AOB input hashes before and after each lane.
- Fallback and candidate lanes must expose identical required-file sets and
  identical `sha256_before` values for every AOB input.
- Prefix state and evaluator-record hash equality before intervention.
- Distinct branch-local evaluator records and one-shot candidate application.
- CRN seed descriptor shared by the two arms and containing no arm/case/outcome
  identity.
- `DispatchEvidence` and runtime payloads contain no terminal/oracle/final
  outcome fields.
- No plan or no-overlap control has zero intervention cost and still records a
  paired terminal control row.

## Interpretation

`car_actionability_summary.csv` is a paired offline fact table. It reports
numeric/meaningful wins, catastrophic losses, terminal sign agreement, rank
reversal, and integrity status. A positive oracle headroom does not establish
deployable runtime utility: the two full lanes are label-acquisition cost, and
the cost-adjusted online decision remains a separate experiment. A non-positive
terminal oracle headroom stops selector development; a positive but
short-horizon-misaligned result calls for a calibrated horizon/credit model;
only a positive, stable, integrity-clean result justifies a small selective
critic.
