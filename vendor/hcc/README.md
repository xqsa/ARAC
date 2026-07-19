# HCC Vendor Boundary

`vendor/hcc/` is the reviewed runtime snapshot extracted from the read-only
provenance repository `E:\HCC-main`. ARAC retains only the HCC/AOB components
required by the RDDSM evidence-overlay experiment:

- `HCC/RDDSM.py`: structural decomposition from the AOB design matrix.
- `HCC/NDAs/MMES/`: Phase-I MMES optimizer and resumable state.
- `HCC/OPT/CMAES/`: cooperative group-level CMA-ES optimizer.
- `AOB/`: benchmark objectives, shared utilities, and AOBG data-bundle IDs
  `1/3/4/5` (stored as `F*-*.txt`) used by the real cases `E1`, `E3`, `A4`,
  `R4`, and `S5`. The `F*` prefix is not a benchmark case name.

The ARAC-owned execution entry remains `scripts/hcc_smoke_runner.py`. Runtime
adapters pass explicit vendor and AOB data roots, so execution does not depend
on the process working directory or read from `E:\HCC-main`.

Vendored optimizer behavior is read-only by default. Any required patch must be
isolated, justified against `E:\HCC-main`, and covered by a focused regression
test. Changes to optimizer semantics, random-number flow, or FE accounting need
an explicit protocol change.
