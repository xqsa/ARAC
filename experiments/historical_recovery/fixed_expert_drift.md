# Fixed-expert historical drift diagnosis

## Current v9 campaign

- Summary: `artifacts/historical_recovery_fixed_expert_v1/summary.json`
- Complete contexts: **600**
- Exact terminal FE: **true**
- Phase-I FE in the shared checkpoint: **180000**
- Mapped action FE after Phase-I: **2820000**
- Current optimizer port: `pypop7==0.0.82`

## Historical representatives

| Lane | Evidence | Key protocol fields |
|---|---|---|
| AOR | `results\exp_057_a_series_aor_25seed\a1-a6-25seed-v1\runs\A1\seed_117\run_summary.json` | worker_protocol_version=exp057-aor-worker-v1; backend=vendor.HCC.OPT.CMAES.sepcmaes.SEPCMAES; optimizer_route=full_space_sep_cmaes; policy_action=aor; policy_protocol=a-series-aor-v1; initial_mean=0.0; configured_max_fes=3000000; fitness_evaluations=3000000 |
| CTP | `results\exp_058_ctp_stable_v2_25seed\validation\runs\S1\seed_117\exp_058_ctp_stable_v2_25seed-s1-seed117\schwefel\run_summary.json` | protocol_version=hcc-run-summary-v3; decision_available_fes=1610403; decision_fe=1389598; configured_max_fes=3000000; fitness_evaluations=3000000; runtime_action=ctp_stable |
| GCB | `results\exp_059_gcb_stable_all_case_25seed\validation\runs\R1\seed_117\exp_059_gcb_stable_all_case_25seed-r1-seed117\rastrigin\run_summary.json` | protocol_version=hcc-run-summary-v3; configured_max_fes=3000000; fitness_evaluations=3000000; runtime_action=gcb; runtime_policy_action=gcb_stable; runtime_policy_protocol=r-series-gcb-stable-all-case-validation-v2 |
| SMP | `artifacts/frozen_actions/smp_v26/manifest.json` | historical_complete_lane_absent |

## Source and protocol drift

- Current Phase-I protocol: `arac-identity-blind-evidence-v9`
- Frozen matrix Phase-I protocol: `arac-identity-blind-evidence-v8`
- Common source components: **3/14** match

| Component | Match |
|---|---|
| action_execution | True |
| action_registry | False |
| aor | False |
| benchmark | True |
| contracts | False |
| ctp | False |
| experiment | False |
| gcb | False |
| ledger | False |
| optimizers | False |
| phase1 | False |
| selector | False |
| smp | False |
| structural_evidence | True |

## Confirmed findings

- The historical AOR representative is a fresh full-space 3,000,000-FE run from initial_mean=0 using vendor.HCC.OPT.CMAES.sepcmaes.
- The current fixed-expert campaign spends 180,000 FE in an identity-blind Phase-I checkpoint and gives the mapped action only the remaining 2,820,000 FE.
- Historical CTP and GCB representatives use hcc-run-summary-v3 action-specific decision and routing fields that are not represented by the v9 shared checkpoint contract.
- The current fixed-expert manifest and frozen v5 matrix disagree on the Phase-I protocol and on most common source hashes.
- The completed v9 campaign therefore establishes a valid current Phase-I fixed-action result, but it is not a fair bitwise or aggregate replay oracle for the historical table.

## Decision

The v9 fixed-action campaign is valid evidence for the current Phase-I-plus-mapped-action protocol, but it cannot certify recovery of the historical action-specific table. Reconstruct the historical action protocol, or explicitly rebind the target table to v9, before running selector correctness or ARAC-Core end-to-end experiments.
