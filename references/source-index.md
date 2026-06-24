# Source Index

This folder is a clean extraction. The original project evidence remains in
`E:\HCC-main`.

Use these source files only as references, not as code to copy wholesale.

## Core Sources

- `E:\HCC-main\HCC_SRC\HCC\MI_ARAC_ACTION\README.md`
  - Current module-level summary, runtime boundary, current evidence chain, and
    active layout.

- `E:\HCC-main\artifacts\mi_arac_action_MAINLINE_INDEX.md`
  - Durable evidence index and latest status. Important for claim boundaries.

- `E:\HCC-main\mi-arac-trace-replay-dfdp-schema.md`
  - Earlier schema draft for trace replay, action labels, negative controls,
    fallback, and termination.

- `E:\HCC-main\HCC_SRC\HCC\MI_ARAC_ACTION\final_executor.py`
  - Historical implementation of action taxonomy fields, active effect fields,
    and final gate surfaces.

- `E:\HCC-main\HCC_SRC\HCC\MI_ARAC_ACTION\mapping_guard_v2.py`
  - Historical mapping guard with many repair stages. Treat as evidence of
    what to simplify, not as the clean policy source.

## What Not To Copy

- `scripts\mi_arac_action_legacy\`
- `tests\mi_arac_action_legacy\`
- `artifacts\archive\`
- milestone-specific runner names
- Route A repair chain as a default branch
- problem-specific threshold patches

## Portable Lessons

- Keep runtime reference-blind.
- Separate backend/support labels from core actions.
- Require fresh same-budget execution for runtime claims.
- Use negative controls and catastrophic-loss gates.
- Treat final/reference evidence as evaluation-only.

