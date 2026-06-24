# Experiments

Future experiments should be small, resumable, and table-backed.

## Recommended Layout

```text
experiments/
  exp_001_schema_smoke/
  exp_002_backend_binding_smoke/
  exp_003_same_budget_smoke/
```

Each experiment should contain:

- `README.md`: purpose, scope, and claim level.
- `config.yaml`: frozen parameters.
- `run.ps1` or `run.py`: executable entrypoint.
- `expected_outputs.md`: required truth tables.

Results should go under `results/<experiment_id>/`, not in the project root.

## Minimum Gates

An experiment is not complete unless it states:

- claim level
- allowed runtime inputs
- forbidden runtime inputs checked
- same-budget FE status
- backend semantics status
- negative-control status
- catastrophic-loss status

