# HCC/AOB Backbone Boundary

Date: 2026-07-18
Executor: Codex

ARAC is extracted from the read-only `E:\HCC-main` workspace. The retained
backbone is deliberately small:

```text
HCC RDDSM + MMES/CMA-ES + AOB objective
  -> Phase-I group proposals and trace evidence
  -> ARAC evidence overlay
  -> unchanged HCC Phase II
```

## Retained Runtime

- `vendor/hcc/HCC/RDDSM.py`: NumPy RDDSM implementation with grouping behavior
  equivalent to the extracted PyTorch version.
- `vendor/hcc/HCC/NDAs/MMES/`: Phase-I optimizer state.
- `vendor/hcc/HCC/OPT/CMAES/`: cooperative group optimizer.
- `vendor/hcc/AOB/`: objective functions and only the F1/F3/F4/F5 inputs needed
  by `E1/E3/A4/S5`.
- `scripts/hcc_smoke_runner.py`: canonical subprocess runner and FE ledger.

`src/arac/backends/hcc.py` validates AOB inputs, constructs the real runner
command, enforces the frozen overlay profile, parses native and overlay FE, and
returns offline execution results. `src/arac/backends/hcc_evidence_overlay.py`
contains the runtime observer integration. `src/arac/policy/evidence_overlay.py`
contains the pure selection and four-point policy logic.

## Extraction Rules

- `E:\HCC-main` remains read-only.
- Historical milestones, result artifacts, plotting code, binary LSGO, and
  unused F2/F6 data are not part of the retained runtime.
- AOB truth and final outcomes are offline evaluation inputs only.
- The overlay is disabled unless explicitly requested; the default HCC command
  surface remains unchanged.
- Phase-II topology is frozen and no probe state is adopted.

## Validation Boundary

Adapter and integration tests cover AOB data completeness, RDDSM topology,
strict FE accounting, subprocess environment isolation, checkpoint pairing,
native state parity, and exp_018 aggregation. Passing those tests establishes a
reproducible observer connection, not a benchmark-performance claim.
