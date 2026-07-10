# exp_005_hcc_final_protocol_pilot

This experiment is the auditable 3M-FE entry for the canonical ARAC runtime
controller. It runs one optimizer trajectory per case/seed; it does not execute
an action portfolio and select by final outcome.

Default protocol:

- problems: `E1 E2 E3 E4 E6 S2 S3 S6 R1 R2 R3 A4 A5`
- seeds: `1 2 3`
- budget: `3,000,000` FE per case
- lane profile: `canonical_evidence_controller_v1`
- lane: one `arac_evidence_action_controller_v31` trajectory
- AOB data root: `E:\ARAC\HCC_SRC\AOB\AOBG\datafile`

Run:

```powershell
py -3 experiments\exp_005_hcc_final_protocol_pilot\run.py --output-dir results\exp_005_hcc_final_protocol_pilot --aob-data-root E:\ARAC\HCC_SRC\AOB\AOBG\datafile --jobs 3 --budget-accounting strict
```

For Ackley/platform-escape pilots, use the same 3M-FE wrapper with
`--lane-profile landscape_escape` and explicit A-series cases. This adds the
`bipop_search_state_restart` lane while keeping fallback/repair/coordinate for
same-budget comparison:

```powershell
py -3 experiments\exp_005_hcc_final_protocol_pilot\run.py --output-dir results\exp_005_hcc_ackley_landscape_escape --problems A1 A2 A3 A4 A5 A6 --lane-profile landscape_escape --jobs 3 --budget-accounting strict
```

The canonical gate rejects missing or changed AOB inputs, same-budget
violations, forbidden runtime fields, and action traces that report a worse
accepted incumbent. Paper values and historical outcomes are offline-only and
never enter controller dispatch.
