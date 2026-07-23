# exp_030 S-family budget pulse

Date: 2026-07-23
Executor: Codex

This entry point runs one mechanical action-validation trajectory only:

- real AOB `S5`, seed `117`, with a configured upper bound of `300000` FE;
- profile `s_family_budget_pulse`;
- four captured relation contexts;
- `native_eq8`, `true_no_writeback`, raw frozen efficiency allocation, and the
  fixed 50/50 shrunk pulse at three horizons (48 rows total).

The runner delegates numerical execution to the existing exp019 diagnostic worker. Its
local validator then reconstructs both typed budget actions and their one-shot lifecycle,
checks native parity, FE/order/budget traces, current AOB input hashes, and the native
population-aligned terminal window. Native HCC may stop below the configured upper bound when
the remaining budget cannot admit another optimizer population; the manifest records both the
observed terminal FE and the fixed comparison FE. Passing this smoke does not establish positive
action effect and does not authorize the action, selector, inference, or runtime gates.

For the frozen context population sizes, the terminal contract is
`comparison_fe = max(1, configured_max_fes - max(population_sizes))` and
`comparison_fe <= fitness_evaluations <= configured_max_fes`. The runner must not pad the record
with repeated incumbent evaluations to reach the configured upper bound.

Protocol v1 does not serialize the dispatch-time `current_values` preimage. Its
`dispatch_anchor_hash` is therefore validated as a unique, well-formed runtime-issued receipt and
is not incorrectly recomputed from the separately recorded `right_values` candidate.

Run a fresh smoke:

```powershell
.\.venv\Scripts\python.exe -m experiments.pilots.exp_030_s_family_budget_pulse.run
```

Validate an existing trajectory without optimizer execution:

```powershell
.\.venv\Scripts\python.exe -m experiments.pilots.exp_030_s_family_budget_pulse.run --reuse-existing
```

Reuse mode is read-only. It requires the existing root aggregate CSVs and manifest to match
the worker artifacts, current config, source hashes, and closed authorization fields; it does
not regenerate or repair any artifact.

Outputs are written under `results/exp_030_s_family_budget_pulse/`. The root aggregate CSVs
are exact single-trajectory copies of the validated worker artifacts; `manifest.json` records
their hashes, AOB input hashes, FE summary, and closed authorization gates.
