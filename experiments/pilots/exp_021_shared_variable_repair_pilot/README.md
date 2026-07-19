# Exp 021: shared-variable repair pilot

This experiment repeats the Exp 020 paired protocol with one intended change:
the target lane uses `repair_shared_variable_binding` instead of
`allow_beneficial_coordination`.

E3, A4, and S5 each run at seed 1 and 100,000 FE against a fresh native HCC
`conservative_no_action` baseline. Relation dispatch and the evidence overlay
remain disabled. The existing repair action selects the shared-variable values
from the group with the larger local fitness improvement; its implementation is
not changed by this experiment.

Run from the repository root:

```powershell
python -m experiments.pilots.exp_021_shared_variable_repair_pilot.run
```

Regenerable output is written under
`results/exp_021_shared_variable_repair_pilot/`. The promotion rule and FE audit
are identical to Exp 020, so this remains a single-seed pilot rather than a final
statistical claim.
