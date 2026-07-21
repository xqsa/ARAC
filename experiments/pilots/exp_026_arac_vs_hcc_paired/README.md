# exp_026 R/S Phase2 Action Validation

This is a five-seed descriptive validation of one explicit Phase2 action per
trajectory on `R2-R6` and `S2-S6`, at exactly 3,000,000 FE. The 50 trajectories
use seeds `117-121`.

- R cases run `full_space_sep_cma` once, with a budget equal to the previous
  complete native sweep, accept only a strict improvement, and then resume
  three frozen native sweeps before continuing native HCC to the terminal FE.
- S cases run `persistent_frozen_efficiency_budget_reallocation`: the frozen
  per-group budgets remain caps across Phase2 until the lifecycle closes at the
  exact terminal FE. Legal CMA-ES early stopping may consume less than a cap.

The runner must be called with `persistent_phase2`, `paired_owner`, relation
dispatch, and the case-specific `--persistent-phase2-action`. Every successful
trajectory must write `run_summary.json` with `hcc-run-summary-v3` and an
adjacent `persistent_phase2_action.json` with schema
`phase2-action-v2`. The latter is fail-closed: one selected action, authorized
and consumed runtime execution, terminal 3M FE, matching SHA-256
action/lifecycle hashes, and the case-specific R/S lifecycle semantics above
are mandatory. Version-1 artifacts are stale and `--resume` reruns them.

`run.py` reports per-case mean, median, sample standard deviation, and a 2,000
replicate within-case seed bootstrap mean 95% interval. It also reports the
observed mean divided by the Table 2 reported bold and numeric-best means. Those
ratios are descriptive only: neither native HCC nor a paper baseline is run,
and this five-seed cohort makes no inferential comparison with the paper's
25-run results.

Run with:

```powershell
python experiments/pilots/exp_026_arac_vs_hcc_paired/run.py
```

Each trajectory streams merged runner output to its own `runner.log`. If a run
is interrupted, resume with:

```powershell
python experiments/pilots/exp_026_arac_vs_hcc_paired/run.py --resume
```

Resume reuses a trajectory only after the full artifact gate passes; missing or
invalid trajectories are executed again. Use `--reuse-existing` for read-only
validation that never launches a runner subprocess.
