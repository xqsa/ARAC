---
session_no: S03
suggested_title: "[ARAC] S04 设计可辨识的长期 utility runtime action"
parent_session: S02
project: arac
date: 2026-07-14
---

## 当前阶段

v33.8 的 fresh full-24 证据仍完整且达到 best-of-three `13/24`。为降低其
跨 seed 风险实现了 v34 downstream recovery checkpoint，但两轮 protected
验证均失败。修正后的 v34 为 `6/8`，未达到强制 `8/8`，因此没有启动 v34
full-24，v34 未被采用。

## 已完成（产物路径 + 验证摘要）

- `results/controller_v34_recovery_5k_smoke_20260714`：首轮 v34 real-HCC
  5k smoke。
- `results/controller_v34_recovery_8case_seed123_3m_20260714`：首轮 protected
  `24/24` fresh、FE `0/24`、AOB `237/237` unchanged、anti-leakage `16/16`；
  best `5/8`、mean `1/8`、worst `1/8`、seed wins `8/24`、catastrophic `8/24`。
- commit `52a3b0d` 修正 commit 后局部 CC evidence 被 composite delta 覆盖的
  语义错误；v33 行为和 FE 账本保持隔离。
- `results/controller_v34_recovery_local_evidence_5k_20260714`：修正后 6/6
  fresh smoke，FE/AOB/anti-leakage 通过，recovery pending=0。
- `results/controller_v34_recovery_local_evidence_8case_seed123_3m_20260714`：
  第二轮 `24/24` fresh、FE 违规/超支 0、AOB `237/237` unchanged、
  anti-leakage `16/16`，五类 raw artifacts 各 24；checkpoint 为 commit 467、
  restore 226、preempt 1、pending 0。
- 同目录 `offline_paper_best_comparison.csv/.md`：best `6/8`、mean `1/8`、
  worst `0/8`、seed wins `9/24`、catastrophic `5/24`；失败 case 为 S6、R2。

## 失败原因（runtime evidence）

- R1 无 recovery control：`705/705` action decisions 与 v33.8 相同，三个
  final error 完全相同，说明分叉来自 recovery 路径而非 runner 漂移。
- overlap runs 相对 v33.8 有 364 个 shared-id action 分叉，另有 140 个
  new-only、65 个 old-only relation ids。
- S6 seed2 相对 v33.8 恶化 `200.33%`，seed3 却改善 `22.76%`；两者都在
  outer iteration 0 首次 restore 后分叉。
- R2 seed1/3 的 restored mean credit 几乎相同（`-9.7e-5` / `-1.01e-4`），
  但长期结果方向相反（`-63.82%` / `+18.07%`）。单 downstream-group credit
  无法辨识 final utility，按离线结果调 tolerance 会形成泄漏式过拟合。

## 下一步（<=3 条，最小动作）

1. 新候选必须先提出一个独立、reference-blind 的长期 utility 可辨识假设；
   不在 v34 上继续堆阈值或 case/family 特例。
2. 先用现有 raw trace 做离线可辨识性审计，验证候选 runtime feature 在同一
   case 跨 seed 上能否区分有害/有益分叉，再决定是否实现。
3. 仍按 5k smoke -> protected 8/8 -> full-24 的梯子执行；未过 protected
   不消耗 full-24 预算。

## 阻塞/风险

- v34 只保证一个 downstream group 的局部 checkpoint no-harm，不保证长期
  optimizer utility；当前 trace 已给出同 signal、相反 final outcome 的反例。
- v33.8 的 `13/24` 只是三 seed best-of-three pilot；mean `4/24`、worst
  `2/24`、catastrophic `31/72` 仍阻止 robust/SOTA 声称。
- v34 full-24 不存在，不能把 v33.8 full-24 指标写成 v34 指标。

## 必读文件（按序）

1. 本卡。
2. `.light/passport.yaml`。
3. `docs/design/core-method.md` 与 `docs/design/boundaries.md`。
4. `docs/superpowers/specs/2026-07-14-downstream-recovery-checkpoint-design.md`。
5. 两个 v34 protected 目录内的 comparison、trajectory guard 和 action trace。

## 禁止

- 不得使用 case、function family、paper-best、历史最优、gain label 或 final
  outcome 作为 runtime dispatch 或阈值选择输入。
- 不得跳过 protected `8/8` 门直接运行 full-24。
- 不得隐藏 losing seeds/catastrophic loss，或把三 seed 写成稳定性结论。
- 不修改 `E:\HCC-main`，不覆盖或提交用户未跟踪材料。
