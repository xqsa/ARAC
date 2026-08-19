# Historical-to-independent action semantic delta

日期：2026-08-10  
执行者：Codex  
目的：解释旧 25-seed 结果与当前独立 ARAC 结果的差异；不授权 HCC replay。

## Evidence

- Exact source recovery: `session_source_recovery.md`。
- Cross-session provenance scan: `retained_source_recovery.md`。
- Historical action configurations and protocols: recovered EXP-052/058/059
  campaign sources under `.codex-tasks/historical-level-recovery/raw/session-source-recovery/`。
- Current independent implementations: `src/arac/actions/aor.py`, `ctp.py`,
  `smp.py`, `gcb.py`, and `_execution.py`。

## Delta table

| Action | Historical evidence | Current independent path | Parity status |
|---|---|---|---|
| AOR | EXP-057 worker runs a full-space Sep-CMA route through `HCC.OPT.CMAES.sepcmaes.SEPCMAES`; route metadata is produced by `resolve_aor_route(case)`. | `AorExecutor` routes from the current checkpoint evidence and uses the independent optimizer port. | Same action intent, different runtime/source contract; no bitwise parity claim. |
| CTP | EXP-058 freezes S1 as a zero-overlap group-polish route and S2-S6 as a positive-overlap MMES-tail route; the frozen campaign sets `cmaes_restart=false`. | Current CTP performs evidence coverage, relation-cover/block polish, and a terminal MMES path with new budgets, block objects, and seeds. | Budget/schedule and optimizer lifecycle differ; current result is not EXP-058-equivalent. |
| GCB | EXP-059 uses a phase-boundary burst for R1, relation dispatch for R2-R6, and three native resume sweeps under a frozen authorization protocol. | Current GCB uses an evidence-derived block order and a relation-count route with current terminal alignment; it does not reproduce the historical dispatch artifact/lifecycle. | Trigger, state handoff, and continuation contract differ; current result is not EXP-059-equivalent. |
| SMP | EXP-052 is a paired native-full-CMAES versus persistent-state protocol, with retained stop/reset evidence; its receipt is bound to the historical runner and not to the current checkpoint. | Current SMP uses independent persistent block sessions, current group dimensions, rescue branches, global polish, and a terminal optimizer. | Internal components are reproducible, but historical checkpoint, grouping, and lifecycle binding are absent. |

## Scientific conclusion

The July-30 independentization succeeded at the architectural boundary: AOB can
remain only a benchmark adapter, Phase-I can be identity-blind, and ARAC can
choose one action through its own runtime. It did not preserve the numerical
behavior of the promoted historical action implementations. Re-training the
selector on the rewritten actions therefore labels the new behavior; it cannot
restore the old table.

The exact historical runner bytes are now available for forensic comparison, but
they still import the HCC runtime and do not yet have a verified dependency closure
or numerical-environment binding. EXP-052 itself starts from a fresh seed rather
than an external checkpoint; the checkpoint mismatch applies when comparing its
internal Phase-I state with a current independent ARAC checkpoint. Copying the
runner back into production would violate the method boundary and would still not
prove a fair replay.

## Required next gate

1. Freeze the current independent AOR/CTP/SMP/GCB implementations as a new
   versioned action contract.
2. Port only the evidenced mechanisms above into that contract: CTP's route
   distinction, GCB's trigger/continuation semantics, SMP's persistent state and
   reset accounting, and AOR's full-space route.
3. On fresh shared Phase-I checkpoints, compare each fixed action and ARAC-Core
   under identical total FE, seed, and ledger rules.
4. Report the result as current-independent ARAC evidence. Use EXP-052/057/058/059
   only as historical golden references; do not call the comparison HCC replay.
