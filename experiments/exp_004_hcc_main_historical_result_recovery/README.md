# exp_004_hcc_main_historical_result_recovery

This experiment recovers historical `evaluation_record.txt` artifacts from
`E:\HCC-main\HCC_SRC\result` and aligns detected AOB cases with the paper
reported HCC-ES Table 2 anchors.

The goal is to preserve useful historical HCC-main evidence without leaking it
into runtime dispatch. Historical final errors, reported baselines, and win
flags are offline evaluation artifacts only.

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\83718\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m experiments.exp_004_hcc_main_historical_result_recovery.run --output-dir results\exp_004_hcc_main_historical_result_recovery
```

Outputs:

- `hcc_main_historical_result_inventory.csv`
- `hcc_main_vs_paper_reported_comparison.csv`
- `hcc_main_historical_results_audit.md`
