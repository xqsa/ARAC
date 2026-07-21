# exp027: R1 forced global Sep-CMA action validation

This pilot runs `R1` with seeds `117-121` to exactly `3,000,000` function
evaluations. R1 has no overlap relation, so this is deliberately not a
relation-selected action experiment. The runner issues one immutable
`full_space_sep_cma` action after three complete native HCC sweeps, consumes it
at the next sweep boundary before any group evaluation, resumes three frozen
native sweeps, and then continues native HCC to the terminal FE.

The pilot requires `global_phase2_action.json` with schema
`phase2-global-action-v1`. Validation fails closed unless the artifact records
a `phase_boundary` trigger, `relation: null`, exact burst FE accounting, a
completed lifecycle, three completed native resume sweeps, exact terminal FE,
and a SHA-256 reference that agrees with `run_summary.json`.

Run all five trajectories with the configured five workers:

```powershell
python experiments/pilots/exp_027_r1_global_sep_cma/run.py
```

Resume an interrupted cohort, reusing only artifacts that pass the complete
gate:

```powershell
python experiments/pilots/exp_027_r1_global_sep_cma/run.py --resume
```

Validate existing output without launching the HCC runner:

```powershell
python experiments/pilots/exp_027_r1_global_sep_cma/run.py --reuse-existing
```

The aggregate `run_summary.json` contains every seed's terminal error plus the
five-seed descriptive summary. These results can assess this forced global
action, but they do not establish that Phase1 relation evidence can select it.
