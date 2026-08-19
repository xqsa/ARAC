# ARAC historical recovery audit

- Gate passed: **false**
- Recovered: **18/24**
- Failed: **0/24**
- Missing: **6/24**
- Source: `output/pdf/aob_arac_method_comparison_corrected.csv`
- Current/frozen-source runtime parity: **4/4**

| Case | Expert | Status | Mean | Historical mean | Sample std | Historical std |
|---|---|---|---:|---:|---:|---:|
| A1 | AOR | recovered | 7.80E+04 | 7.80E+04 | 9.33E+01 | 9.33E+01 |
| A2 | AOR | recovered | 7.80E+04 | 7.80E+04 | 9.27E+01 | 9.27E+01 |
| A3 | AOR | recovered | 7.81E+04 | 7.81E+04 | 9.28E+01 | 9.28E+01 |
| A4 | AOR | recovered | 7.82E+04 | 7.82E+04 | 1.07E+02 | 1.07E+02 |
| A5 | AOR | recovered | 7.84E+04 | 7.84E+04 | 6.63E+01 | 6.63E+01 |
| A6 | AOR | recovered | 7.80E+04 | 7.80E+04 | 7.73E+01 | 7.73E+01 |
| E1 | SMP | missing | NA | 5.69E+05 | NA | 1.57E+06 |
| E2 | SMP | missing | NA | 5.62E+06 | NA | 3.78E+06 |
| E3 | SMP | missing | NA | 1.34E+07 | NA | 5.20E+06 |
| E4 | SMP | missing | NA | 2.61E+07 | NA | 9.35E+06 |
| E5 | SMP | missing | NA | 2.98E+07 | NA | 9.18E+06 |
| E6 | SMP | missing | NA | 3.19E+07 | NA | 6.54E+06 |
| R1 | GCB | recovered | 1.56E+05 | 1.56E+05 | 1.53E+04 | 1.53E+04 |
| R2 | GCB | recovered | 2.24E+05 | 2.24E+05 | 2.95E+04 | 2.95E+04 |
| R3 | GCB | recovered | 2.69E+05 | 2.69E+05 | 3.29E+04 | 3.29E+04 |
| R4 | GCB | recovered | 3.15E+05 | 3.15E+05 | 3.25E+04 | 3.25E+04 |
| R5 | GCB | recovered | 3.34E+05 | 3.34E+05 | 4.10E+04 | 4.10E+04 |
| R6 | GCB | recovered | 3.61E+05 | 3.61E+05 | 4.19E+04 | 4.19E+04 |
| S1 | CTP | recovered | 1.04E+00 | 1.04E+00 | 5.12E+00 | 5.12E+00 |
| S2 | CTP | recovered | 8.12E+02 | 8.12E+02 | 5.82E+02 | 5.82E+02 |
| S3 | CTP | recovered | 2.46E+03 | 2.46E+03 | 1.35E+03 | 1.35E+03 |
| S4 | CTP | recovered | 5.11E+03 | 5.11E+03 | 3.49E+03 | 3.49E+03 |
| S5 | CTP | recovered | 8.38E+03 | 8.38E+03 | 8.69E+03 | 8.69E+03 |
| S6 | CTP | recovered | 4.18E+03 | 4.18E+03 | 1.99E+03 | 1.99E+03 |

## Current fixed-expert campaign

- Status: **failed**
- Campaign gate passed: **false**
- Complete arms: **600/600**
- Exact terminal FE: **true**
- Mean matches at displayed precision: **6/24**
- Sample-std matches at displayed precision: **0/24**
- Recovered cases: **0/24**

| Case | Expert | Status | Mean | Historical mean | Sample std | Historical std |
|---|---|---|---:|---:|---:|---:|
| A1 | AOR | failed | 7.80E+04 | 7.80E+04 | 1.51E+02 | 9.33E+01 |
| A2 | AOR | failed | 7.80E+04 | 7.80E+04 | 9.99E+01 | 9.27E+01 |
| A3 | AOR | failed | 7.81E+04 | 7.81E+04 | 1.03E+02 | 9.28E+01 |
| A4 | AOR | failed | 7.82E+04 | 7.82E+04 | 8.75E+01 | 1.07E+02 |
| A5 | AOR | failed | 7.84E+04 | 7.84E+04 | 1.10E+02 | 6.63E+01 |
| A6 | AOR | failed | 7.80E+04 | 7.80E+04 | 1.06E+02 | 7.73E+01 |
| E1 | SMP | failed | 5.13E+06 | 5.69E+05 | 8.07E+06 | 1.57E+06 |
| E2 | SMP | failed | 3.71E+06 | 5.62E+06 | 1.02E+06 | 3.78E+06 |
| E3 | SMP | failed | 6.24E+06 | 1.34E+07 | 1.36E+06 | 5.20E+06 |
| E4 | SMP | failed | 9.17E+06 | 2.61E+07 | 1.66E+06 | 9.35E+06 |
| E5 | SMP | failed | 1.13E+07 | 2.98E+07 | 2.28E+06 | 9.18E+06 |
| E6 | SMP | failed | 1.46E+07 | 3.19E+07 | 2.65E+06 | 6.54E+06 |
| R1 | GCB | failed | 1.67E+05 | 1.56E+05 | 2.02E+04 | 1.53E+04 |
| R2 | GCB | failed | 2.20E+05 | 2.24E+05 | 3.26E+04 | 2.95E+04 |
| R3 | GCB | failed | 3.00E+05 | 2.69E+05 | 4.48E+04 | 3.29E+04 |
| R4 | GCB | failed | 3.17E+05 | 3.15E+05 | 5.97E+04 | 3.25E+04 |
| R5 | GCB | failed | 3.36E+05 | 3.34E+05 | 4.51E+04 | 4.10E+04 |
| R6 | GCB | failed | 3.79E+05 | 3.61E+05 | 7.44E+04 | 4.19E+04 |
| S1 | CTP | failed | 1.60E+04 | 1.04E+00 | 7.55E+04 | 5.12E+00 |
| S2 | CTP | failed | 6.19E+02 | 8.12E+02 | 4.66E+02 | 5.82E+02 |
| S3 | CTP | failed | 6.31E+03 | 2.46E+03 | 4.73E+03 | 1.35E+03 |
| S4 | CTP | failed | 1.15E+04 | 5.11E+03 | 8.44E+03 | 3.49E+03 |
| S5 | CTP | failed | 3.10E+04 | 8.38E+03 | 2.27E+04 | 8.69E+03 |
| S6 | CTP | failed | 2.52E+04 | 4.18E+03 | 1.42E+04 | 1.99E+03 |

## Frozen independent-action matrix

- Historical mean met at displayed precision: **12/24**
- Historical mean not met: **12/24**
- Current source hashes matching the frozen matrix: **3/12**

| Case | Expert | Frozen mean | Historical mean | Ratio | Status |
|---|---|---:|---:|---:|---|
| A1 | AOR | 7.80E+04 | 7.80E+04 | 1.000 | historical_mean_met |
| A2 | AOR | 7.80E+04 | 7.80E+04 | 1.000 | historical_mean_met |
| A3 | AOR | 7.81E+04 | 7.81E+04 | 1.000 | historical_mean_met |
| A4 | AOR | 7.82E+04 | 7.82E+04 | 1.000 | historical_mean_met |
| A5 | AOR | 7.84E+04 | 7.84E+04 | 1.000 | historical_mean_met |
| A6 | AOR | 7.80E+04 | 7.80E+04 | 1.000 | historical_mean_met |
| E1 | SMP | 3.11E+06 | 5.69E+05 | 5.457 | historical_mean_not_met |
| E2 | SMP | 3.74E+06 | 5.62E+06 | 0.665 | historical_mean_met |
| E3 | SMP | 6.02E+06 | 1.34E+07 | 0.449 | historical_mean_met |
| E4 | SMP | 9.74E+06 | 2.61E+07 | 0.373 | historical_mean_met |
| E5 | SMP | 1.07E+07 | 2.98E+07 | 0.358 | historical_mean_met |
| E6 | SMP | 1.47E+07 | 3.19E+07 | 0.461 | historical_mean_met |
| R1 | GCB | 1.60E+05 | 1.56E+05 | 1.024 | historical_mean_not_met |
| R2 | GCB | 2.32E+05 | 2.24E+05 | 1.037 | historical_mean_not_met |
| R3 | GCB | 2.84E+05 | 2.69E+05 | 1.056 | historical_mean_not_met |
| R4 | GCB | 3.22E+05 | 3.15E+05 | 1.021 | historical_mean_not_met |
| R5 | GCB | 3.41E+05 | 3.34E+05 | 1.021 | historical_mean_not_met |
| R6 | GCB | 3.64E+05 | 3.61E+05 | 1.008 | historical_mean_not_met |
| S1 | CTP | 1.04E+03 | 1.04E+00 | 1003.836 | historical_mean_not_met |
| S2 | CTP | 7.00E+02 | 8.12E+02 | 0.863 | historical_mean_met |
| S3 | CTP | 6.55E+03 | 2.46E+03 | 2.663 | historical_mean_not_met |
| S4 | CTP | 1.11E+04 | 5.11E+03 | 2.175 | historical_mean_not_met |
| S5 | CTP | 3.10E+04 | 8.38E+03 | 3.697 | historical_mean_not_met |
| S6 | CTP | 3.48E+04 | 4.18E+03 | 8.320 | historical_mean_not_met |

## Current-code checkpoint replay

- Replay status: **failed**
- Exact replay passed: **1/4**

| Case | Expert | Current error | Frozen error | Ratio | FE | Error | Hash |
|---|---|---:|---:|---:|---|---|---|
| A1 | AOR | 7.821611E+04 | 7.821611E+04 | 1.000 | True | True | True |
| E1 | SMP | 4.146293E+07 | 3.387152E+06 | 12.241 | True | False | False |
| R1 | GCB | 1.957385E+05 | 1.737794E+05 | 1.126 | True | False | False |
| S1 | CTP | 1.807955E-07 | 1.630005E-07 | 1.109 | True | False | False |

## Frozen-source runtime control

- Control status: **matched**
- Current results matching the manifest-bound frozen source: **4/4**
- Frozen source files matching their manifest: **13/13**

| Case | Expert | Current/frozen source | Stored v5 arm |
|---|---|---|---|
| A1 | AOR | True | True |
| E1 | SMP | True | False |
| R1 | GCB | True | False |
| S1 | CTP | True | False |

## Decision

The current fixed-expert campaign completed all 600 mapped arms at the exact 3,000,000-FE budget, but its aggregate gate remains closed: only the six A-series means match the displayed historical precision, and no case matches both mean and sample standard deviation. The historical artifact layer still has an incomplete E/SMP lane, while the frozen independent matrix misses historical means on multiple cases. The current legacy path matches its manifest-bound frozen source on all four representative contexts, but three stored v5 block-action arms do not reproduce from that same source. Selector correctness and ARAC-Core end-to-end claims must remain deferred.
