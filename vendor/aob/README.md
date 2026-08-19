# AOB Benchmark Boundary

This directory contains the AOB benchmark implementation and the AOBG
1000-dimensional input bundle separated from the frozen historical optimizer
tree on 2026-07-30 by Codex.

The benchmark Python files and every `AOBG/datafile/F*` input are preserved
byte-for-byte from the reviewed snapshot originally extracted from
`E:\HCC-main`. ARAC treats them as third-party benchmark material, not as part
of the proposed method.

Production code accesses the benchmark only through `arac.benchmarks.aob`.
Benchmark case identity is experiment metadata; the adapter exposes only the
objective callable, dimension, public bounds, and optimum value to the runtime.

