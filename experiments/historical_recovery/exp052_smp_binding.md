# EXP-052 SMP binding audit

This is a read-only audit of the retained seed117 trajectory. It does not
execute the recovered runner.

- External checkpoint required: **no**
- Exact receipt/config/action/input/budget binding: **yes**
- Optimizer dependency sources: **recovered from the session timeline**
- Optimizer dependency receipt hashes: **not recorded**
- Runtime package pins match the current candidate `.venv`: **yes**
- Runtime environment manifest bound in receipt: **no**
- Replay authorized: **no**

## Evidence

The command starts a fresh seed117 run. The runner constructs an empty
SMP cache and records/validates state internally on each group visit; the
retained `smp_action.json` contains the state schema, group dimensions,
restore/reset counts, and per-event state hashes.

The optimizer source closure is recovered. The remaining gate is provenance
binding: EXP-052 does not hash those optimizer files or the Python/numerical
environment in its receipt. The current `.venv` matches the runtime pins from
the historical `pyproject.toml`, but that is a reconstruction candidate, not a
receipt-bound environment. See `exp052_environment.md`.
