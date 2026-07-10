# exp_004 Expected Outputs

`hcc_main_historical_result_inventory.csv` contains one row per discovered
`evaluation_record.txt`.

Important columns:

- `source_file`
- `experiment_label`
- `problem_id`
- `seed`
- `fe_used`
- `final_error`
- `parse_status`
- `usable_for_runtime_dispatch`
- `usable_for_offline_evaluation`
- `runtime_dispatch_allowed`

`hcc_main_vs_paper_reported_comparison.csv` contains rows where a historical
record can be mapped to one of the 24 AOB cases in
`references/paper_reported_table2_hcc_es.csv`.

`hcc_main_targeted_case_diagnostics.csv` summarizes the 11 cases that need
special discussion in the current report: S4, S5, R4, R5, R6, and A1-A6.
It is an offline-only table that classifies the best historical row against the
paper mean and flags Ackley near-ties separately from real losses.

All rows must keep `runtime_dispatch_allowed=0`.
