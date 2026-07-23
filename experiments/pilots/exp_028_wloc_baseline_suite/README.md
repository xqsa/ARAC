# exp_028 Continuous WLOC Baseline Suite

This experiment entry binds the 18 frozen 1000-dimensional WLOC cases to the nine AOB comparison methods. It is synthetic-only and cannot enter the real-AOB SBS, VBS, bootstrap, or action-gate summaries.

The frozen task matrix uses measured DG2/RDG3 decomposition, generated 20-way Random grouping, and the vendored RDDSM design-matrix decomposition. Decomposition FEs and optimization FEs remain separate.

The mechanical smoke uses 31 optimization FEs. For DG2-CMAES and RDG3-CMAES only, it supplies the catalog topology and records `provided_catalog_topology_smoke`; this validates the 1000-dimensional adapter without claiming a measured decomposition. Focused small-dimensional tests exercise the actual DG2 and RDG3 implementations.

```powershell
.venv\Scripts\python.exe -m experiments.pilots.exp_028_wloc_baseline_suite.run --emit-matrix results\exp_028_wloc_baseline_suite\task_matrix.json
.venv\Scripts\python.exe -m experiments.pilots.exp_028_wloc_baseline_suite.run --mechanical-smoke --case WLOC01 --method all
```

These commands produce implementation artifacts only. They do not authorize a performance comparison or multi-seed conclusion.

