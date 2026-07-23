# exp029: R-series GCB 25-seed terminal validation

This experiment runs the frozen `gcb` action on R1-R6 for seeds 117-141 at
exactly 3,000,000 FE. R1 uses the phase-boundary adapter from exp027; R2-R6 use
the relation-dispatch adapter from exp026. Both paths execute the same
`GcbAction` through `RuntimeActionDispatcher`.

The experiment reports the final best-so-far error as mean and sample standard
deviation over 25 seeds. It does not run a paired native branch and therefore
does not establish a selector or same-checkpoint action-ceiling claim.

Run or resume the complete matrix:

```powershell
.\.venv\Scripts\python.exe -m experiments.pilots.exp_029_r_series_gcb_25seed.run --resume
```

Validate existing artifacts without launching trajectories:

```powershell
.\.venv\Scripts\python.exe -m experiments.pilots.exp_029_r_series_gcb_25seed.run --reuse-existing
```
