# EXP-052 numerical environment audit

- Historical Git ref: `c7505d91`
- Historical pyproject SHA-256: `d79600e659b06ef146d86a91937706d77a344fbd97727248ba5bfecf65a7329a`
- Current candidate all-pinned package match: **yes**
- Session-observed environment binding: **yes**
- Receipt environment binding: **no**
- Replay authorized: **no**

| Distribution | Historical pin | Current candidate | Match |
|---|---:|---:|---|
| `cma` | `4.4.4` | `4.4.4` | yes |
| `numpy` | `2.3.5` | `2.3.5` | yes |
| `PyYAML` | `6.0.3` | `6.0.3` | yes |
| `scipy` | `1.18.0` | `1.18.0` | yes |
| `llvmlite` | `0.48.0` | `0.48.0` | yes |
| `numba` | `0.66.0` | `0.66.0` | yes |
| `pypop7` | `0.0.82` | `0.0.82` | yes |

## Session provenance

| Evidence | UTC timestamp | Session line | Verified |
|---|---|---:|---|
| `venv_absent_before_creation` | `2026-07-18T14:01:29.051Z` | `113` | yes |
| `venv_created` | `2026-07-18T14:03:15.656Z` | `160` | yes |
| `venv_creation_succeeded` | `2026-07-18T14:03:33.281Z` | `165` | yes |
| `hcc_pins_installed` | `2026-07-18T14:05:04.464Z` | `197` | yes |
| `hcc_versions_imported` | `2026-07-18T14:05:47.425Z` | `206` | yes |
| `baselines_installed` | `2026-07-23T03:03:53.823Z` | `2153` | yes |
| `baselines_pins_resolved` | `2026-07-23T03:04:33.051Z` | `2162` | yes |
| `pypop7_installed_location` | `2026-07-23T03:09:53.527Z` | `2252` | yes |
| `python_version_observed` | `2026-07-24T15:27:35.344Z` | `3320` | yes |
| `blas_backend_observed` | `2026-07-21T11:16:48.795Z` | `18775` | yes |
| `same_venv_rechecked_before_formal_start` | `2026-07-26T09:26:53.181Z` | `17001` | yes |
| `scipy_rechecked_before_formal_start` | `2026-07-26T09:59:32.766Z` | `17401` | yes |
| `last_editable_install_only_arac` | `2026-07-26T09:45:33.603Z` | `528` | yes |
| `formal_exp052_start` | `2026-07-26T12:40:23.355Z` | `2628` | yes |

Formal-session dependency mutations after the last editable install and before EXP-052 start: **0**.

The retained sessions directly bind the project `.venv` creation, Python 3.12.7,
the pinned package versions, OpenBLAS 0.3.30 build configuration, the final
editable-only `arac` reinstall, and the formal EXP-052 launch command. This supports
a version-level isolated reproduction. The historical receipt itself records neither
an environment manifest hash nor a launch-time runtime-library fingerprint, so it
does not support a bitwise/receipt-bound replay claim.
