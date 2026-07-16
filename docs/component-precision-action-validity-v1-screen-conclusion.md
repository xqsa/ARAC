# Component Precision Action Validity v1 Screen Conclusion

Date: 2026-07-16
Executor: Codex
Protocol: `component-precision-action-validity-v1`
Decision: `screen_no_go`

## Conclusion

The component-atomic precision action passed every runtime integrity check but
failed the preregistered eight-case screen. The frozen `kappa=0.5` action is
therefore retired as an action without a demonstrated stable main effect. The
24-case confirmation, dual-kernel portfolio, scheduler, and runtime profile
are not authorized and were not run or implemented.

This is not evidence that the action is always harmful. Its descriptive mean
effect was positive and no catastrophic event occurred. The failure is that
safe applicability was too sparse, material effects were too rare and
case-concentrated, the terminal lower confidence bound remained negative, and
shared-variable survival was not directionally stable.

## Frozen Experiment

| Stage | Matrix | Fresh runs | Result |
|---|---:|---:|---|
| CLI/5k | E1/E2, seeds 1/2, two arms | 8/8 | Integrity pass; no opportunity |
| v37 parity | E1/E2, seeds 1/2 | 4/4 | Terminal, FE, and public trace matched |
| Trace smoke | A4 seed 1, 100k | 2/2 | Active auxiliary route; abstain parity |
| Trace escalation | A4 seed 1, 3M | 2/2 | One complete action and delayed closure |
| Screen | 8 cases, seeds 65-69, two arms | 80/80 | Integrity pass; action-validity no-go |

The 100k smoke first exposed a budget-integrity defect: native v37 phase rescue
could consume FE inside a frozen component horizon. Before screen execution,
commit `f234f92` preregistered auxiliary-FE quiescence and commit `87f3fe6`
implemented it. The correction changed no sigma, threshold, estimand, case,
seed, or matrix, and no action-utility result was available when it was made.

## Integrity

| Check | Result |
|---|---:|
| Fresh branches | PASS, 80/80 |
| Pair integrity | PASS, 40/40 |
| Strict terminal FE | PASS, zero overspend and 40/40 terminal closure |
| Prefix/checkpoint/frozen-plan integrity | PASS |
| AOB inputs and source hashes | PASS |
| Anti-leakage | PASS |
| Component and delayed closure | PASS, 15/15 applicable pairs |
| Component or terminal catastrophic events | 0 |

Case, seed, family, component fingerprint, raw/final outcome, paper-best, and
historical performance were not runtime dispatch inputs. All 25 non-applicable
pairs followed native v37 and passed abstain parity.

## Screen Gate

| Criterion | Observed | Status |
|---|---:|---|
| Applicable pairs | 15/40 | Fail: requires at least 30 |
| Applicable cases | 5/8 | Fail: requires at least 6 |
| Applicable seeds | 5/5 | Pass |
| Terminal ATT mean / median | `+2.436276e-2` / `+8.588149e-5` | Pass descriptively |
| Terminal ATT one-sided 95% LCB | `-2.414432e-4` | Not positive |
| Component ATT mean / median | `+2.269112e-2` / `+1.132945e-4` | Pass descriptively |
| Terminal ATT wins / losses | 9 / 6 | Mixed |
| Terminal effects at least 1% | 4 pairs, 2 cases | Fail: requires 10 pairs |
| Positive seed means | 4/5 | Pass |
| Shared survival `delta_S_H` mean / median | `+7.393415e-2` / `-6.411260e-4` | Fail: median negative |
| Shared survival `delta_S_H` 95% LCB | `-9.217411e-2` | Not positive |
| Catastrophic events | 0 | Pass |

The four material effects covered only two cases. S2 contributed about 88.8%
of the total absolute terminal effect, so the positive overall mean does not
represent broad action stability. The terminal worst-10% ATT CVaR was
`-8.589096e-3`.

## Runtime-Evidence Diagnosis

The component-atomic formulation fixed the earlier estimand problem: every
applicable action covered the complete overlap component, produced one unique
component endpoint, and closed delayed shared-variable credit. The remaining
failure is therefore not a local-credit or FE-accounting artifact.

Only A4, A5, E2, S2, and S5 produced applicable pairs. E1 was the registered
no-overlap control. Across the other abstentions, 18 pairs still had an active
native-v37 auxiliary-FE route and two could not close a full component horizon.
Relaxing these conditions after observing the screen would either modify A0 or
break frozen requested-budget identity, so it is not an admissible coverage
fix.

Among the 15 valid actions, the terminal sign still changed across pairs and
the median paired shared-survival effect was negative. This means that making
the action atomic removed attribution ambiguity but did not create a stable
precision main effect. A scheduler or portfolio cannot be justified from this
screen because it would be asked to rescue an action whose safe support and
durable benefit are not established.

## Decision And Scope

- Keep v37 as the runtime baseline.
- Retire the frozen `0.5 * normal sigma` complete-component action.
- Do not run the 384-run confirmation matrix.
- Do not implement `baseline-anchored-component-portfolio-v1`.
- Do not register a scheduler, portfolio, model bundle, or runtime profile.
- Do not tune sigma, opportunity rules, quiescence, thresholds, cases, or seeds
  against this screen.
- Treat the result as a falsification of this precision action channel, not of
  overlap/grouping evidence as a scope and risk signal.

Any future action must be a new preregistered mechanism with an independently
validated main effect before scheduler research resumes.

## Audit Artifacts

- Screen root:
  `results/component_precision_screen_8case_seed65_69_3m_87f3fe6_20260716T174428171`
- Canonical gate: `component_action_gate.json`
- Gate SHA-256:
  `14f2d076a9c60f8928000df68dd7a193da1ad251f4ec13a222d4f3edade83be9`
- Source implementation commit: `87f3fe6b70c85bb81e58f03bf4b2333134c6448f`
- Corrective preregistration commit: `f234f9205f92d3a68ca42e54fb29954c05cdc9e8`
- Frozen spec SHA-256:
  `3e97419d89c9cc0c42b2c4e12ac088225bbbe864c93f8995394d429d0cca714e`
- Frozen config SHA-256:
  `80d848494dc15ee8d46b7d3dc5b826c5f09f408015f99f198b525064a487b92f`

Raw `results/` artifacts remain untracked. The machine-readable gate is the
canonical decision and has `full_24_authorized=false` and
`runtime_scheduler_authorized=false`.
