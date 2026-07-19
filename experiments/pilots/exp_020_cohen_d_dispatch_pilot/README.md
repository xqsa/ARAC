# exp_020 Cohen's d relation dispatch pilot

This experiment runs fresh HCC trajectories for E3, A4, S5, and the conforming-overlap
R4 control at 100,000 FE for seeds 1 through 5. Runtime dispatch is enabled and uses
only per-relation top-5 owner distributions:

```text
cohen_d > 0.8  -> repair_shared_variable_binding
cohen_d <= 0.8 -> conservative_no_action
```

The experiment audits every relation by joining `overlap_relations.csv` to
`action_trace.csv`. It reports the Cohen's d distribution per case and fails closed
if the selected action, threshold, top-k counts, or runtime evidence disagree.

Run from the repository root:

```powershell
python -m experiments.pilots.exp_020_cohen_d_dispatch_pilot.run
```

Generated outputs are written under `results/exp_020_cohen_d_dispatch_pilot/` and are
not committed.
