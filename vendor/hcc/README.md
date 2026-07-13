# HCC Vendor Boundary

`vendor/hcc/` is the canonical HCC/AOB source boundary for ARAC v3.2. It is a reviewed snapshot
of the patched runtime source extracted from the read-only provenance repository
`E:\HCC-main`; it is not a live mirror of that workspace. The original HCC project README is
preserved as `UPSTREAM_README.md`.

## Ownership

- `AOB/`, `HCC/`, and `HCC-ES.py` are vendored upstream-derived source and are read-only by
  default.
- The ARAC-owned runner lives at `scripts/hcc_smoke_runner.py`; no ARAC runner is kept inside
  the vendor tree.
- Runtime adapters pass an explicit vendor root and AOB data root. They do not depend on the
  process working directory or on `E:\HCC-main`.
- Generated optimizer payload under `result/` is ignored. Only `result/README.md` is tracked.

## Smoke Command

Run from any working directory by using absolute paths, or from the repository root with:

```powershell
py scripts/hcc_smoke_runner.py --functions elliptic --ids 1 --seed 1 --max-fes 2000 --output-root results/hcc-smoke --aob-data-root vendor/hcc/AOB/AOBG/datafile --skip-plots
```

The selected interpreter must provide the optional HCC dependencies declared in
`pyproject.toml`.

## Patch Rules

Do not edit vendor optimizer behavior as part of ARAC policy or experiment work. A required HCC
patch must be isolated, justified against `E:\HCC-main`, covered by focused regression tests,
and documented in the commit that introduces it. Changes to optimizer semantics, random-number
flow, or FE accounting require an explicit protocol task; path and packaging work must remain
behavior-preserving.
