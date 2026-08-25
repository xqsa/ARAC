# shared_patch_v1 upgrade candidate

Stepwise upgrade candidate against the frozen recovered baseline
`arac-recovered-baseline-20260823-v1`, following
`docs/arac-oc-stepwise-upgrade-plan-v2.1.md` (outer U0-U4 ladder, inner
S1-S5b mechanism ladder).

Rules inherited from the freeze and the plan:

- frozen sources, protocols and evidence artifacts are never modified or
  overwritten; every level writes to its own new artifact directory;
- mounts exist only inside the ctp/gcb block-sweep hosts (gcb is the
  recovered-lane name of the historical GSS episode);
- patch / soft routing / new selector stay off in every baseline arm;
- each stage preregisters its protocol before execution and rolls back to
  the previous passed stage on failure.

Levels:

- `u0_baseline_guard.py` - re-runs the freeze verifier and pins the level
  manifest (`artifacts/upgrade_u0_baseline_guard_v1`).  **PASSED 2026-08-23.**
- `u1_host_reachability.py` - per-case host reachability table over AOB
  mapped hosts and the preregistered conflicting overlap generator
  (`artifacts/upgrade_u1_host_reachability_v1`).  **PASSED 2026-08-23**;
  all twelve AOB instrumented reruns were bit-identical to the frozen
  recovery-screen receipts.
- `s1_leverage_sweep.py` - leverage-priority first-sweep reorder
  (`artifacts/upgrade_s1_leverage_sweep_v1`).  **GATE FAILED 2026-08-23**:
  the ctp host is structurally inert (coverage sessions never re-anchor),
  the generator cells are leverage-degenerate (complete-graph discovery),
  and the only active lane (gcb, R2-R6) breached the final-error
  non-inferiority margin.  Per the plan stop-loss the reorder stays out of
  every candidate; the frozen baseline is untouched.
- `s2_propagation_handoff_campaign.py` - slot handoff.  Protocol frozen but
  **blocked**: its loader refuses to run while the S1 gate is failed.

Environment note: every command must run under `./.venv/Scripts/python`;
a plain `python` on PATH may resolve to an unrelated venv whose numpy/BLAS
stack silently breaks bit-identity with the frozen receipts.

Run pattern (Git Bash).  Always use the project interpreter — a plain
`python` on PATH may resolve to an unrelated venv whose numpy/BLAS stack
breaks bit-identity with the frozen receipts:

```text
PYTHONPATH='.;src' ./.venv/Scripts/python -m experiments.upgrade.shared_patch_v1.u0_baseline_guard
```
