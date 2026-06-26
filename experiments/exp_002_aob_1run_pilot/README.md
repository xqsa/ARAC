# exp_002_aob_1run_pilot

This experiment is the first AOB pilot entrypoint for ARAC-on-HCC.

It covers the 24 AOB cases from the Two-Phase CC protocol:
`E1-E6`, `S1-S6`, `R1-R6`, and `A1-A6`, with `D=1000`,
search range `[-100, 100]^D`, one independent pilot run, and a total budget of
`3,000,000` function evaluations.

## Claim Level

This is an HCC-source-grounded grouping probe, not a final optimizer
performance run. Rows in `our_result_by_case.csv` use
`source_level=hcc_source_topology` and
`pilot_result_source=hcc_source_grounded_grouping_probe`.

The runner reads AOB metadata, overlap gamma, real dimension, topology groups,
and overlap-derived FE allocation from `E:\HCC-main`. It still does not run
MMES/CMAES or claim fresh optimizer performance.

Paper-reported HCC-ES Table 2 values are joined only in
`paper_reported_comparison.csv` for offline evaluation. They must not enter
runtime dispatch.

## Run

```powershell
$env:PYTHONPATH='src'; & 'C:\Users\83718\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m experiments.exp_002_aob_1run_pilot.run
```

The default output directory is `results/exp_002_aob_1run_pilot/`.
