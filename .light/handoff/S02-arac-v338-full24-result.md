---
session_no: S02
suggested_title: "[ARAC] S03 诊断 v33.8 跨 seed catastrophic loss"
parent_session: S01
project: arac
date: 2026-07-14
---

## 当前阶段

v33.8 的 raw evidence 恢复和完整 24-case 三种子 3M-FE 验证已经完成。
best-of-three 正好达到 13/24，但跨 seed 稳定性和 catastrophic-loss 硬门
未通过。下一阶段只能按 runtime evidence 做 no-harm 诊断或消融，不能按 case
继续调 selector。

## 已完成（产物路径 + 验证摘要）

- `results/controller_v338_replay_5k_smoke_20260713` — R2/S6 6/6 fresh，
  FE 0/6，AOB 60/60 unchanged，anti-leakage 16/16；non-dense 最大范数
  0.5，dense 最大范数 7.450186。
- `results/controller_v338_replay_8case_seed123_3m_20260713` — 24/24 fresh，
  8/8 best-of-three，FE 0/24，AOB 237/237 unchanged，anti-leakage 16/16；
  mean 1/8，worst 0/8，catastrophic 7/24。
- `results/controller_v338_full24_seed123_3m_20260713` — 72/72 fresh，
  FE 0/72，AOB 708/708 unchanged，anti-leakage 16/16，backend semantics
  changed 72/72。
- `results/controller_v338_full24_seed123_3m_20260713/offline_paper_best_comparison.md`
  — best 13/24，mean 4/24，worst 2/24，seed wins 21/72，catastrophic
  31/72；paper-best 仅离线 join，runtime dispatch used=0。
- `results/controller_v338_full24_seed123_3m_20260713/runtime_evidence_case_summary.csv`
  — 1,238 credit rows 中 1,080 negative；1,478 trust rows 中 7 trusted、
  241 quarantined；non-dense fallback 1,352 行、最大 0.5，dense fallback
  327 行、最大 457.486。
- `docs/superpowers/specs/2026-07-13-risk-aware-action-guard-design.md` —
  已追加 full-24 的权威 tracked 摘要和 claim boundary。

## 工作区状态

本卡、`.light/passport.yaml` 和 v33.8 spec 将随本轮同一收尾提交落盘；接手后
以 `git status --short --branch` 和 `git log --oneline -3` 刷新实际 commit。
用户原有未跟踪论文、FlyKi、exp006-exp008 和外部源码材料未改动、未纳入提交。

## 下一步（<=3 条，最小动作）

1. 从 `runtime_evidence_case_summary.csv` 切片 negative credit、quarantine、
   dense/non-dense fallback 与 no-overlap controls，形成 no-harm 消融设计。
2. 优先验证“弱/负 credit 时更早 abstain”是否降低 catastrophic seeds；只用
   Phase-I/current-run evidence，不用 paper-best、case label 或历史 final。
3. 新 selector 只有在受保护 8-case 不退化、完整 24-case 至少 13/24，且
   mean/worst/catastrophic 明显改善后，才考虑 5-seed 稳定性扩展。

## 阻塞/风险

- 13/24 只是三种子 best-of-three pilot 门，不是 robust final success。
- mean 4/24、worst 2/24、catastrophic 31/72 是当前主要硬风险。
- exp003 是单 action lane，没有 runtime fallback utility 对照；claim gate 因
  `no_fallback_reference` 保持 performance claim blocked。

## 必读文件（按序）

1. 本卡。
2. `.light/passport.yaml`。
3. `docs/design/core-method.md` 与 `docs/design/boundaries.md`。
4. `docs/superpowers/specs/2026-07-13-risk-aware-action-guard-design.md`。
5. full-24 目录内 `run_manifest.md`、`offline_paper_best_comparison.md` 和
   `runtime_evidence_case_summary.csv`。

## 禁止

- 不要把 13/24 best-of-three 写成 mean、25-run、SOTA 或 robust success。
- 不要隐去 31/72 catastrophic seeds，也不要从最优 seed 反推稳定性。
- 不要把 case label、function family、paper-best、历史结果或 final outcome
  加入 runtime dispatch。
- 不要修改 `E:\HCC-main`；不要覆盖或提交用户未跟踪材料。
- 不要重跑本卡已列出的 fresh runs，除非证据损坏或用户明确要求。
