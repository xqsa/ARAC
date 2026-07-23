# exp_031 S-family budget-pulse diagnostic

Date: 2026-07-23
Executor: Codex

This experiment is the inferential step after the exp030 mechanical smoke. It runs real AOB
`S5` for seeds `117..121`, with a configured upper bound of `300000` FE per trajectory. Each
trajectory must yield four validated contexts, for five case-seed clusters, 20 contexts, and
240 arm-horizon rows.

The worker, typed actions, deterministic executor, and trajectory truth checks remain owned by
the frozen exp030 action protocol. This entry adds validate-first resume with per-seed
source/config/AOB-input receipts, deterministic aggregation, and case-seed cluster-bootstrap
statistics.
The primary horizon is `sweep_1`; `immediate` and `sweep_3` remain diagnostic labels.

The report separates the registered four-arm oracle from the budget-pulse action ceiling. A win
by `true_no_writeback` is control headroom and cannot validate either raw or 50/50-shrunk budget
pulse. Catastrophic loss rejects only the affected arm before the runtime-eligible pulse ceiling
is recomputed. Selector and runtime authorization remain closed.

Run or resume sequentially:

```powershell
.\.venv\Scripts\python.exe -m experiments.pilots.exp_031_s_family_budget_pulse_diagnostic.run --resume --jobs 1
```

Strictly validate existing outputs without writes:

```powershell
.\.venv\Scripts\python.exe -m experiments.pilots.exp_031_s_family_budget_pulse_diagnostic.run --reuse-existing
```

Outputs are written under `results/exp_031_s_family_budget_pulse_diagnostic/`. Passing this
single-case diagnostic can only advance the surviving budget-pulse action or portfolio to the
registered `E1/E3/A4/R4/S5` real-AOB pilot. Project-level action, evidence-separability, selector,
and runtime gates remain closed.
