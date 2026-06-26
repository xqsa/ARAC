# HCC-main Historical Results Audit

Date: 2026-06-26
Executor: Codex
Source root: E:\HCC-main\HCC_SRC\result
Evaluation records discovered: 233
Parsed records: 233
AOB cases detected: A1, A2, A3, A4, A5, A6, E1, E2, E3, E4, E5, E6, R1, R2, R3, R4, R5, R6, S1, S2, S3, S4, S5, S6
Offline rows better than paper reported mean: 59

## Cases With Historical Rows Better Than Paper Reported Mean

- A3: 2
- A4: 1
- A5: 1
- E1: 8
- E2: 7
- E3: 7
- E4: 3
- E6: 3
- R1: 8
- R2: 9
- R3: 4
- S2: 1
- S3: 4
- S6: 1

## Best Historical Row By Case

| case | best historical final error | paper reported mean | seed | experiment label |
| --- | ---: | ---: | --- | --- |
| A1 | 7.836371e+04 | 7.76E4 |  | paper_like_seed1_function_pipeline_fix |
| A2 | 7.847626e+04 | 7.81E4 | 3 | mi_arac_action_clean_hcc_anchor_5seed |
| A3 | 7.836237e+04 | 7.86E4 | 4 | mi_arac_action_clean_hcc_anchor_5seed |
| A4 | 7.825342e+04 | 7.83E4 | 2 | mi_arac_action_clean_hcc_anchor_5seed |
| A5 | 7.825517e+04 | 7.85E4 | 4 | mi_arac_action_clean_hcc_anchor_5seed |
| A6 | 7.815793e+04 | 7.80E4 | 2 | mi_arac_action_clean_hcc_anchor_5seed |
| E1 | 2.672356e+05 | 2.84E6 | 4 | mi_arac_action_clean_hcc_anchor_5seed |
| E2 | 2.206362e+06 | 6.87E6 | 3 | mi_arac_action_clean_hcc_anchor_5seed |
| E3 | 7.254322e+06 | 1.60E7 | 2 | mi_arac_action_clean_hcc_anchor_5seed |
| E4 | 1.510332e+07 | 2.26E7 |  | paper_like_seed1 |
| E5 | 1.396273e+07 | 7.76E6 |  | mi_arac_action_prefix10_dispatch_phaseii_policy |
| E6 | 2.644759e+07 | 4.32E7 |  | paper_like_seed1 |
| R1 | 1.375464e+05 | 1.74E5 |  | mi_arac_action_prefix01_dispatch_phaseii_policy |
| R2 | 1.953551e+05 | 3.72E5 |  | mi_arac_action_prefix01_dispatch_phaseii_policy |
| R3 | 4.076996e+05 | 5.06E5 |  | paper_like_seed1_function_pipeline_fix |
| R4 | 6.813109e+05 | 5.59E5 |  | paper_like_seed1_function_pipeline_fix |
| R5 | 9.358130e+05 | 6.84E5 |  | mi_arac_action_prefix10_dispatch_phaseii_policy |
| R6 | 1.003215e+06 | 8.15E5 |  | mi_arac_action_prefix01_dispatch_phaseii_policy |
| S1 | 3.170804e+01 | 1.92E-3 |  | mi_arac_action_prefix01_dispatch_phaseii_policy |
| S2 | 3.390253e+03 | 5.58E3 |  | paper_like_seed1_current_fix_24case |
| S3 | 7.769275e+03 | 9.72E3 |  | mi_arac_action_m179_route_a_s_family_low_risk_same_budget_smoke_run |
| S4 | 1.587667e+04 | 1.24E3 |  | paper_like_seed1_topology_fix_schwefel |
| S5 | 1.316993e+05 | 9.23E3 |  | paper_like_seed1_topology_fix_schwefel |
| S6 | 4.091427e+04 | 6.65E4 |  | paper_like_seed1 |

## Runtime Boundary

These artifacts are historical HCC-main evidence. They are usable for offline
evaluation and provenance checks only. They must not enter runtime dispatch.
