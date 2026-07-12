# Results Contract

`results/` is the generated output boundary for experiments. It is not a source-code or runtime
dispatch input.

## Inputs And Outputs

- Inputs to generation: Git commit, experiment ID and protocol, config path, seed, case ID,
  benchmark raw input, runtime trace, and same-budget FE accounting.
- Outputs: evidence profiles, action decisions, backend traces, semantic diffs, FE ledgers,
  audit tables, run metadata, and reproducible manifests.

Each output must preserve its source commit and claim level. Scaffold/proxy, smoke, pilot, and
final outputs must remain distinguishable; a file named `final` does not establish final success.

## Git Policy

Run payload is generated and ignored by `results/*`. Only `results/README.md` and
`results/.gitkeep` are tracked in Task 2. Do not commit logs, caches, manifests produced for a
single local run, or large output tables. Schemas and reusable generators belong under `docs/`
or `scripts/` and remain trackable. Paper and historical values may be joined only by offline
analysis after reference-blind runtime execution finishes.
