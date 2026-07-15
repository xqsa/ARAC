# Causal-Risk Precision Pilot Result: 2026-07-16

Status: integrity-valid randomized pilot; causal identifiability gate failed.
No model bundle or runtime scheduler is authorized.

Executor: Codex. Implementation commit: `852a268`. Preregistration amendment:
`650d491`. Protocol: `precision-causal-logging-v2`.

## Frozen run

- Entry: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke`
- Profile: `precision_causal_logging`
- Cases: `A4/A5/E1/E2/E3/E4/S2/S5`
- Seeds: `40/41/42/43/44`
- Arms: v37 normal-sigma baseline and one v38 precision-sigma action
- Budget: strict `3,000,000` FE per arm, `jobs=24`
- Raw root:
  `results/causal_precision_logging_v2_pilot_8case_seed40_44_3m_jobs24_852a268_20260716T025226`
- Audit root:
  `results/causal_precision_logging_v2_pilot_8case_seed40_44_3m_jobs24_852a268_20260716T025226_audit`

The treatment is the first complete scheduler-reachable and unlocked
post-retirement precision opportunity. Each trajectory contributes at most
one action. All labels use the same absolute terminal FE prefix. Case, seed,
function family, fingerprints, paper-best values, historical outcomes, and
current/final outcomes are excluded from model inputs.

## Mechanical correction before pilot

The first post-history-fix A4 diagnostic reached a real opportunity but showed
that normal and precision sigma can trigger CMA early stopping after different
numbers of group evaluations. Protocol v1 incorrectly treated this
post-treatment stopping time as a pre-action equality requirement.

Commit `650d491` preregistered protocol v2 before randomized pilot FE. Commit
`852a268` implemented it. v2 requires equal requested reservation and common
terminal endpoint, while checking each arm's actual intervention FE
independently. It does not disable early stopping, pad an arm, or change the
action. Four hundred twenty-one focused/adjacent tests passed with one skip;
the tracked full suite passed 868 tests with one skip.

## Integrity

| Check | Result |
|---|---:|
| Fresh branches | PASS, 80/80 completed and fresh |
| Pair integrity | PASS, 40/40 |
| Strict FE | PASS, 0/80 overrun; actual endpoints 2,999,984..3,000,000 |
| Common terminal labels | PASS for all 16 applicable pairs |
| Requested intervention reservation | PASS, 40/40 matched |
| AOB inputs | PASS, 790/790 unchanged |
| Anti-leakage | PASS, 16/16 checks |
| Feature completeness | PASS, 0/256 missing values |
| Raw/derived hashes | PASS |

Fourteen applicable pairs had different natural intervention endpoints by
arm, as expected for an action-dependent early-stopping mediator. All were
individually FE-consistent and reached the same terminal prefix.

## Feasibility coverage

Only 16 of 40 registered pairs were applicable:

| Case | Applicable | Not applicable |
|---|---:|---:|
| A4 | 5 | 0 |
| A5 | 5 | 0 |
| E1 | 0 | 5 |
| E2 | 1 | 4 |
| E3 | 0 | 5 |
| E4 | 0 | 5 |
| S2 | 5 | 0 |
| S5 | 0 | 5 |

The 24 non-applicable pairs were blocked by:

- `precision_retirement_not_reached`: 12;
- `no_scheduler_reachable_unlocked_precision_opportunity`: 6;
- `no_overlap_component_candidate`: 5;
- `missing_pre_action_history:cc_progress_history_4`: 1.

The frozen continuation minimum was 30 applicable pairs from six cases. The
observed coverage was 16 pairs from four cases, so the pilot cannot support the
registered full logging extension.

## Descriptive paired action effect

Positive `tau = log(error_baseline / error_action)` favors the precision arm.
These values are descriptive terminal facts, not runtime inputs.

| Case | n | Mean tau | Median tau | Wins / losses | >=1% material | Catastrophic | Worst tau |
|---|---:|---:|---:|---:|---:|---:|---:|
| A4 | 5 | -4.064e-06 | -4.089e-06 | 2 / 3 | 0 | 0 | -1.341e-05 |
| A5 | 5 | +1.134e-05 | +9.014e-06 | 4 / 1 | 0 | 0 | -1.838e-06 |
| E2 | 1 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| S2 | 5 | -2.493e-02 | -2.541e-02 | 2 / 3 | 4 | 0 | -9.173e-02 |
| All | 16 | -7.790e-03 | +5.574e-07 | 8 / 7, one tie | 4 | 0 | -9.173e-02 |

A4 and A5 effects were effectively zero at terminal. All four material effects
came from S2, where signs varied across seeds and three of five pairs lost.
The absence of a 20% catastrophic event does not make this signal predictable.

## Frozen identifiability gate

The production audit used 1,000 case-cluster bootstrap depth-2 trees, 2,000
case-by-seed policy bootstraps, and random seed `20260715`.

| Pilot criterion | Result |
|---|---:|
| Integrity and frozen matrix | PASS |
| Seeds >=5 | PASS, 5 |
| Missing features <=5% | PASS, 0% |
| No policy-selected catastrophe | PASS, no action selected |
| Applicable pairs >=30 | FAIL, 16 |
| Cases >=6 | FAIL, 4 |
| Material pairs >=15 | FAIL, 4 |
| LCO/LSO DR value positive | FAIL, 0 / 0 |
| Sign balanced accuracy >0.55 | FAIL, 0.5 |
| In-support >=50% | FAIL, LCO 0%; LSO 31.25% |

The conservative policy selected zero held-out actions. LCO rejected every
state as out of support; LSO retained only five of sixteen. Consequently, the
safe policy value is zero and the zero-release catastrophic upper bound is
uninformative (`1.0`), not evidence of safety.

Gate artifact:
`causal_identifiability_gate.json`, SHA-256
`0fcf25cfaaac4d9a75bd9dd7c55828c6e7a3b42a815424f95979000e35620d73`.
Its `runtime_scheduler_authorized` field is `false`. No
`causal_risk_precision_model.json` exists.

## Runtime-evidence conclusion

The utility layer failed for two independent reasons:

1. The first-opportunity estimand is too sparse. Feasibility reaches only four
   cases in the registered matrix, so the logger cannot supply the required
   cross-case support without changing the action or estimand after seeing the
   data.
2. Where the action is observable, most effects are negligible and the only
   material case, S2, changes sign across seeds. The sixteen state features do
   not transport across held-out cases: their ranges identify different
   state regimes even though case identity itself is absent.

This is not another threshold, mutex, cooldown, FE, or leakage bug. Structure
and state evidence can propose a precision action, but the available
first-opportunity data do not identify a positive long-horizon utility lower
bound. Under the registered safety rule, the correct stable scheduler is to
abstain and execute v37.

## Release decision

- Stop before full 24-case logging.
- Do not export a model or add a runtime profile.
- Do not run shadow, live pilot, or scheduler full-24.
- Do not tune coverage, OOD, LCB, risk, feature, case, seed, or opportunity
  thresholds against this pilot.
- Keep v33.8 as the July paper's existing performance result and report this
  pilot as a mechanism/limitation result.

Any future attempt must define a new estimand and collect new randomized data
before inspecting outcomes. Repeat leases, writeback, and resource channels
remain separate experiments and cannot reuse this failed precision model.
