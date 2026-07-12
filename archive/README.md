# Archive Contract

`archive/` records superseded implementations, failed experiments, and decisions that must stay
discoverable without becoming an alternative runtime source of truth.

## Inputs And Outputs

- Inputs: reviewed obsolete material with origin commit/path, replacement or failure reason,
  relevant protocol, result location, and the gate that blocked promotion.
- Outputs: concise historical records and, only when necessary, approved frozen source needed
  to reproduce a decision. Large run payload remains under ignored `results/` storage.

## Git, Claims, And Runtime Boundary

Track small records and explicitly approved source; do not track caches, logs, duplicated result
payload, or unreviewed worktree diffs. Archived negative evidence keeps its original scaffold,
smoke, pilot, or final-evaluation claim level and cannot be relabeled as success. `src/arac/`,
configs, and active experiment runners must not import or read `archive/`; restoration requires a
new reviewed change with fresh tests rather than an archive path dependency.
