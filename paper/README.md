# Paper Contract

`paper/` holds reviewable manuscript, table-source, and figure-source material. It does not own
runtime code, benchmark data, or experiment payload.

## Inputs And Outputs

- Inputs: versioned method/protocol documents, approved `analysis/generated/` summaries,
  manifests linking rows to runs, and reviewed references.
- Outputs: drafts, table sources, figure sources, and claim text whose evidence can be traced to
  an experiment, commit, config, seed set, FE budget, and claim gate.

## Git And Runtime Boundary

Track concise source material that is needed for review and reproduction. Keep rebuildable
intermediates in generated locations. Runtime modules under `src/arac/` must never read this
directory or use manuscript-reported values, final outcomes, family labels, or prior results for
dispatch. Paper language must preserve the measured claim level; 1-run pilots and synthetic
proxies cannot be presented as final optimizer performance.
