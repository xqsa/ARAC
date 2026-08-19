# AOB Phase-I landscape-family separability audit

- Frozen contexts: **600/600**
- Current E2E checkpoint bindings: **600/600**
- Phase-I budget per context: **180,000 FE**
- New optimizer/objective evaluations: **0**
- Primary separability gate: **true**
- Predefined mapped-action labels correct: **587/600**
- Correct by per-case majority: **24/24**

## Feature-group ablation

| Feature group | Role | Features | Accuracy | Balanced accuracy | Minimum family recall |
|---|---|---:|---:|---:|---:|
| landscape_probe_30 | fixed_ablation | 30 | 0.957 | 0.957 | 0.907 |
| structure_5 | fixed_ablation | 5 | 0.412 | 0.412 | 0.167 |
| progress_5 | fixed_ablation | 5 | 0.953 | 0.953 | 0.847 |
| all_40 | primary | 40 | 0.978 | 0.978 | 0.940 |
| landscape_shape_28 | post_hoc_scale_sensitivity | 28 | 0.755 | 0.755 | 0.573 |
| all_without_level_38 | post_hoc_scale_sensitivity | 38 | 0.928 | 0.928 | 0.807 |

## Primary all-40 result

- Accuracy: **0.978**
- Balanced accuracy: **0.978**
- Label order: `A, E, R, S`

| True / predicted | A | E | R | S | Recall |
|---|---:|---:|---:|---:|---:|
| A | 150 | 0 | 0 | 0 | 1.000 |
| E | 0 | 141 | 0 | 9 | 0.940 |
| R | 0 | 0 | 147 | 3 | 0.980 |
| S | 0 | 0 | 1 | 149 | 0.993 |

## Held-out variant folds

| Variant | Test cases | Accuracy | Balanced accuracy |
|---:|---|---:|---:|
| 1 | A1, E1, R1, S1 | 1.000 | 1.000 |
| 2 | A2, E2, R2, S2 | 0.910 | 0.910 |
| 3 | A3, E3, R3, S3 | 1.000 | 1.000 |
| 4 | A4, E4, R4, S4 | 0.990 | 0.990 |
| 5 | A5, E5, R5, S5 | 0.990 | 0.990 |
| 6 | A6, E6, R6, S6 | 0.980 | 0.980 |

## Main standardized coefficients

| Feature | Mean absolute coefficient | Fold std |
|---|---:|---:|
| tail_log10_gain | 1.5865 | 0.0338 |
| log10_best_probe_error | 1.0589 | 0.0413 |
| log10_center_error | 1.0571 | 0.0413 |
| warmup_log10_gain | 0.6317 | 0.0359 |
| phase1_log10_improvement | 0.6190 | 0.0223 |
| line_high_frequency_fraction_minimum | 0.5687 | 0.0365 |
| line_high_frequency_fraction_median | 0.4429 | 0.0141 |
| late_gain_fraction | 0.3684 | 0.0196 |
| line_high_frequency_fraction_maximum | 0.3542 | 0.0141 |
| structural_relation_density | 0.3269 | 0.1255 |

## Scientific boundary

- Tests four known AOB base-function families on held-out variant indices only.
- Does not establish generalization to unseen base-function families or external suites.
- Mapped-action label accuracy means agreement with A-to-AOR, E-to-SMP, R-to-GCB, and S-to-CTP; it is not an oracle terminal-performance label.
- Does not establish that the historical family-to-action mapping is optimal at the current 180000-FE checkpoint.
- Does not modify or validate the production ARAC-Core action selector.
