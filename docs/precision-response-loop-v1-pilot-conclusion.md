# Precision Response Loop v1 Pilot Conclusion

Date: 2026-07-16
Executor: Codex
Protocol: `precision-response-loop-v1`
Decision: `pilot_no_go`

## Conclusion

The preregistered coverage gate passed, but the three-arm pilot gate failed.
The response loop must therefore stop at the pilot: it is not registered as a
runtime scheduler, and no full-24 experiment is authorized.

This is not a failure to find structural opportunities. A0 found 35 applicable
trajectories out of 40, spanning seven cases and all five registered seeds. The
failure is utility identification: the paired probe rarely authorized a lease,
and its only authorized lease had negative terminal value.

## Frozen Experiment

| Stage | Matrix | Fresh runs | Result |
|---|---:|---:|---|
| CLI/5k | E1/E2, seeds 1/2, A0/A1/A2 | 12/12 | Integrity pass; no opportunity |
| Trace smoke | A4 seed 1, 100k, A0/A1/A2 | 3/3 | Opportunity found; gate abstained |
| Coverage | 8 cases, seeds 60-64, A0 | 40/40 | 35/40 applicable; coverage pass |
| Treatment | 8 cases, seeds 60-64, A1/A2 | 80/80 | 35 paired probes; one release |

All runs used strict absolute terminal FE accounting. There were no FE
violations, AOB changes, anti-leakage failures, forbidden model bundles, or
triplet-integrity failures. All 39 A2 abstain/non-applicable pairs matched A1
in public action-trace hash and terminal error.

## Pilot Gate

| Criterion | Observed | Status |
|---|---:|---|
| Applicable coverage | 35/40, 7 cases, 5 seeds | Pass |
| Releases | 1, A5 seed 64 | Fail: requires at least 10 |
| Release coverage | 1 case, 1 seed | Fail: requires 4 cases, 3 seeds |
| Mean A2-A0 log advantage | `9.506984e-05` | Inconclusive |
| A2-A0 one-sided 95% LCB | `-1.631406e-03` | Fail |
| Released A2-A1 log advantage | `-2.722189e-03` | Fail |
| Median A2-A0 | `0` | Pass |
| Released wins/losses | 0/1 | Fail |
| Positive lease effects at least 1% | 0 | Fail |
| Catastrophic losses | 0 | Pass |
| Delayed-credit closure | 1/1 | Pass |

The confidence bounds use the frozen 2,000-resample case-by-seed two-way
cluster bootstrap. A lease confidence bound is undefined with only one
released case-seed and cannot support a positive claim.

## Runtime-Evidence Diagnosis

The structural layer worked: every non-E1 coverage trajectory produced the
registered safe overlap revisit. The bottleneck is the mapping from immediate
paired response to durable terminal utility.

Of the 35 probes, 17 reached at least 13/16 precision wins and all 35 had zero
large losses. However, 14 decisions were rejected for excessive precision
boundary hits, 18 for insufficient Wilson win LCB, and two because the best
precision candidate did not improve the checkpoint. Only one decision passed
all gates. The high-win boundary cases are not trustworthy evidence for a
precision lease under `legacy_none`; their response is entangled with domain
geometry and was correctly rejected by the frozen safety gate.

The sole release, A5 seed 64, had 15/16 wins, no boundary hits, and no large
losses, but its response magnitude was extremely small: median paired relative
advantage `7.10e-08` and best relative gain `6.68e-08`. Its terminal A2-A1 log
advantage was `-2.722189e-03`, about a 0.272% loss. Delayed credit closed at the
next same-group revisit, yet shared-variable overwrite was 100%, survival was
zero, and `review_positive=0`. This directly falsifies the assumption that a
high paired win rate alone identifies persistent lease value.

The probe itself was also seed-unstable. Across 35 applicable triplets, A1-A0
had four wins, five losses, and 26 exact ties; its mean log advantage was
`1.728467e-04`, but seed means changed sign. The complete loop had four wins,
six losses, and 25 ties. There was no catastrophic loss, but there was also no
stable positive lower bound.

## Decision And Scope

- Keep v37 as the runtime baseline.
- Do not relax Wilson, boundary, or release-count thresholds after observing
  this pilot.
- Do not run full-24, generate `causal_risk_precision_model.json`, or register a
  response-loop runtime profile.
- Treat the result as evidence that current-trajectory paired ordering is not a
  sufficient proxy for long-horizon survival.
- Any future v2 must be a new preregistration. It should test domain-valid
  paired sampling and a minimum response-effect magnitude, then separately
  identify survival after writeback; these changes cannot be inferred or tuned
  from this pilot.

## Audit Artifacts

- Coverage source: `results/precision_response_coverage_a0_cases8_seeds60_64_3m_20260716_1125`
- Treatment source: `results/precision_response_treatment_a1a2_cases8_seeds60_64_3m_20260716_1215`
- Assembled audit: `results/precision_response_pilot_assembled_cases8_seeds60_64_3m_20260716_1400`
- Assembled manifest SHA-256: `0156843379193144ddcf80c3eddc2423585dd50a8f27363823727658fbf218b1`
- Triplets SHA-256: `0d1e7150dd4a9aa5616cc2a25afe029aaeb6ae0c3d0f97d64684a8e88d437ffc`
- Final gate SHA-256: `8ee7aea525f9f9f157a9f1657ff4bd781453296ff8a2db367b113b4e85b9576d`
- Frozen preregistration commit: `7c62fe5`
- Runtime implementation commit: `140fed2`
- Probe-FE aggregation fix: `1eac987`
- Phased assembly fix: `b8e6e64`

Raw `results/` artifacts remain untracked and are not committed. The assembled
manifest records SHA-256 hashes of both source manifests, and
`precision_response_pilot_gate.json` is the canonical machine-readable final
decision.
