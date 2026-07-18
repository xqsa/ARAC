# Source Index

Date: 2026-07-18

`E:\HCC-main` is the read-only source and evidence workspace. The following
paths were rechecked while pruning this repository:

- `E:\HCC-main\HCC_SRC\HCC\RDDSM.py`: original RDDSM grouping source.
- `E:\HCC-main\HCC_SRC\HCC-ES.py`: original HCC optimizer entrypoint.
- `E:\HCC-main\HCC_SRC\arac_hcc_smoke_runner.py`: historical ARAC/HCC runner
  from which the retained execution boundary was extracted.
- `E:\HCC-main\HCC_SRC\AOB\`: original AOB objective and data implementation.
- `E:\HCC-main\HCC_SRC\HCC\MI_ARAC_ACTION\README.md`: historical method and
  runtime notes; useful as provenance, not as the current specification.
- `C:\Users\83718\Desktop\前沿\Two-Phase CC.pdf`: HCC/two-phase paper source.

The current repository truth is:

- `docs/design/core-method.md`: exp_018 method contract.
- `configs/rddsm_evidence_overlay_pilot_v1.json`: frozen experiment config.
- `experiments/pilots/exp_018_rddsm_evidence_overlay_pilot/protocol.py`: gate
  implementation.
- `scripts/hcc_smoke_runner.py`: canonical executable HCC/AOB boundary.

Historical final errors, paper tables, old milestone controllers, and previous
pilot outputs are not runtime inputs and are not retained here.
