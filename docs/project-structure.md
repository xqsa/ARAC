# ARAC Scientific Project Structure

This contract separates stable runtime code, immutable inputs, generated evidence,
offline evaluation, and research claims. A path's location determines whether it may
participate in runtime dispatch; moving a file does not relax the reference-blind boundary.

## Directory Contract

| Path | Inputs | Outputs | Git policy |
| --- | --- | --- | --- |
| `.codex/` | Local audit and extraction inputs | Disposable Task-local intermediates under `.codex/tmp/` | Ignore `.codex/tmp/`; do not use it as a project fact source |
| `src/arac/` | Trace-derived evidence, explicit configs, backend interfaces | Reusable evidence, policy, action, execution, evaluation, and audit APIs | Track source and tests; never track caches |
| `vendor/hcc/` | Reviewed HCC/AOB snapshot from `E:\HCC-main` | Read-only backend source consumed through adapters; generated backend results stay under its ignored `result/` boundary | Track approved source, provenance, and `result/README.md`; never track result payload or cache |
| `experiments/` | Versioned config plus stable APIs | Reproducible run requests and outputs under `results/` | Track entrypoints, protocol README, and expected schemas, not run payloads |
| `configs/` | Human-reviewed experiment parameters | Current executable configuration | Track current configs; archive or remove obsolete alternatives |
| `data/raw/` | Immutable external benchmark inputs | No in-place transformation | Ignore payload by default; track only approved metadata and sentinels |
| `analysis/` | Completed `results/` artifacts and offline references | Statistical tables, plots, and claim-gate summaries | Track reusable analysis code; ignore `analysis/generated/` |
| `results/` | Runtime traces, metadata, seeds, configs, commit IDs | Rebuildable run payload and manifests | Ignore payload; track only `README.md` and `.gitkeep` in Task 2 |
| `references/` | Paper-reported values, historical records, source indexes | Frozen offline comparison evidence | Track small reviewed evidence; never import it into runtime dispatch |
| `paper/` | Approved analysis outputs and claim-gate evidence | Draft text, table sources, and figure sources | Track reviewable sources; generated evidence remains reproducible upstream |
| `docs/` | Decisions, protocols, schemas, audits, research log | Maintained project contracts and handoff records | Track; update an existing source of truth instead of adding parallel versions |
| `scripts/` | Explicit repository paths and versioned inputs | Audits and deterministic maintenance outputs | Track reusable scripts; temporary `*.manifest.tmp` files are ignored |
| `logs/` | Runtime and maintenance command events | Disposable diagnostic logs | Ignore all log payload; evidence required for claims belongs in auditable result tables |
| `archive/` | Superseded or failed material with provenance | Non-runtime historical record | Track concise records or approved source only; large payload stays in `results/` |

`vendor/hcc/` is the sole canonical v3.2 HCC source tree. The ARAC-owned smoke runner lives in
`scripts/hcc_smoke_runner.py`; a top-level `HCC_SRC/` path is a fatal structure violation.

## Raw, Generated, And Reference Evidence

- **Raw** means externally supplied and immutable in this repository. Scripts may read
  `data/raw/` but must write transformations to generated locations.
- **Generated** means reproducible from a Git commit, config, seed, raw input, and runtime
  metadata. `results/` and `analysis/generated/` may be deleted and rebuilt and therefore do
  not become source facts.
- **Reference evidence** means reviewed paper or historical material under `references/`.
  It is valid only after runtime execution for offline comparison, blocker classification,
  and reporting.
- A generated result is not promoted to raw data, and a paper value is not promoted to a
  runtime feature. Manifests record provenance; they do not change claim status.

## Claim Levels

Every experiment or report must use the narrowest supported level:

1. `schema_complete`: required fields and output schemas exist.
2. `preflight_valid`: inputs, paths, config, and budget checks pass.
3. `runtime_connected`: ARAC decisions reach the backend boundary.
4. `fresh_same_budget_smoke`: a fresh smoke run has auditable FE accounting.
5. `pilot_evidence`: the declared pilot protocol completed and passed its gates.
6. `final_evaluation_completed`: the full declared protocol completed.
7. `final_success_claim`: meaningful wins, catastrophic-loss, same-budget, leakage, and
   backend-semantic gates all pass.

Scaffold or synthetic proxy outputs must identify themselves and cannot support a higher claim.
Neither `results/` existence nor a paper comparison upgrades a claim level.

## Reference-Blind Runtime Boundary

Runtime dispatch may consume trace-derived overlap, shared-variable, disagreement, group-gain,
priority, rank-stability, coverage, and remaining-budget evidence. It must not read `paper/`,
`references/paper/`, `references/historical/`, `archive/`, or prior `results/`; nor may it
consume final errors, relative gains, reported baselines, oracle labels, problem-family labels,
problem-ID special cases, or prior pilot/final outcomes. Offline analysis runs only after action
and same-budget execution records have been written.

The structure audit parses `src/arac/**/*.py`, resolves common module-level `Path`/`join` import
aliases, and evaluates literal `Path` composition, constant-string concatenation,
`os.path.join`, f-string static prefixes, and slash-separated strings for exact offline path
components. Runtime files are read as `utf-8-sig`, so a legal UTF-8 BOM is accepted. This
structural guard avoids substring matching and is not a replacement for semantic anti-leakage
tests.

Any top-level `HCC_SRC/` directory, file, symbolic link, or Windows reparse point is fatal.
Links targeting outside the repository and filesystem metadata inspection failures receive
more specific findings. Git executable startup failures are reported as audit findings with a
nonzero CLI status rather than an unhandled traceback.

## Audit

Run from any working directory:

```powershell
python scripts/audit_project_structure.py --root <repo-root>
```

The CLI uses only the Python standard library and Git. It skips `.git`, `.venv`,
`.pytest_cache`, and generated `results/` payload traversal. Each violation is emitted as
`path: rule` and returns a nonzero status. Generated paths are printed explicitly and never
silently treated as source. `--root` must resolve exactly to
`git rev-parse --show-toplevel`; parent and nested paths are rejected.
