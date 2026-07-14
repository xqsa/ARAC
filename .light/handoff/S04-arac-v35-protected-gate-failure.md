---
session_no: S04
suggested_title: "[ARAC] S05 识别长期 utility runtime signal"
parent_session: S03
project: arac
date: 2026-07-14
---

## 当前阶段

canonical v33.8 的 fresh full-24 证据仍为 best-of-three `13/24`。本会话实现并
验证了 v35 transparent-trust topology guard；它恢复了 S2/S3，但未通过
protected gate，因此没有运行 v35 full-24，v35 未被采用。

## 已完成（产物路径 + 验证摘要）

- commits `8567a17`、`e422719`、`5b582c0`、`933368d`：v35 纯语义、runtime
  注册、exp003 lane 与 matched-FE 测试。
- 聚焦测试 `268 passed, 1 skipped`；Git 跟踪全量测试
  `602 passed, 1 skipped`。
- `results/controller_v35_transparent_trust_5k_20260714`：`9/9` fresh，FE
  violation/overspend 0，AOB `90/90` unchanged，anti-leakage `16/16`，trust/
  recovery 非空值 0，两类 fallback route 均出现。
- `results/controller_v35_transparent_trust_10case_seed123_3m_20260714`：
  `30/30` fresh，FE violation/overspend 0，AOB `297/297` unchanged，
  anti-leakage `16/16`，每类 raw artifact 各 30。
- 同目录 `offline_paper_best_comparison.csv/.md`：原 protected 八个 best
  `6/8`，S2/S3 seed wins `6/6`；aggregate best `8/10`、mean `3/10`、worst
  `2/10`、seed wins `15/30`、catastrophic `7/30`。失败 case 是 A4、R2。
- v35 full-24 不存在：protected gate 未过，按协议未启动。

## 工作区状态

v35 代码和失败证据已本地提交、未 push；接手后以 `git status --short --branch`
和 `git log --oneline -3` 刷新实际 commit。`results/` 不入 Git。用户原有未跟踪
论文、FlyKi、exp006-exp008 等材料保持不动。

## 下一步（≤3 条，最小动作）

1. 不在 v35 上按 A4/R2 调阈值；先提出一个 independent、reference-blind 的
   long-horizon utility 可辨识假设。
2. 用现有 v33.8/v34/v35 raw traces 做 within-case、cross-seed 可辨识性审计，
   目标是证明候选 early signal 能区分相反长期方向后再实现。
3. 新候选仍走 5k smoke -> protected gate -> conditional full-24，失败不得削弱门。

## 阻塞/风险

- v35 首次 matched active-action 分叉在 A4/R2 所有 seed 都把 v33.8
  probation-limited 写回放大为 5 倍，但长期效果方向混合。
- A4 是 `2.51` 的极小 best-margin 失败；R2 seed1 相对 v33.8 恶化
  `15.78%`、seed2 改善 `4.36%`、seed3 近似不变，说明 unconditional
  transparency 不是稳定策略。
- 三 seed 只允许称 pilot evidence；canonical v33.8 的 mean/worst/catastrophic
  风险仍是 `4/24`、`2/24`、`31/72`。

## 必读文件（按序）

1. 本卡。
2. `.light/passport.yaml`。
3. `docs/design/core-method.md` 与 `docs/design/boundaries.md`。
4. `docs/superpowers/specs/2026-07-14-transparent-trust-topology-guard-design.md`。
5. v35 5k 与 protected 目录中的 manifest、comparison 和 raw action traces。

## 禁止

- 不使用 case、family、paper-best、历史最优、gain label 或 final outcome 做
  runtime dispatch、阈值选择或 seed 分类。
- 不把 v33.8 full-24 指标写成 v35 指标；v35 没有 full-24。
- 不隐藏 A4/R2 失败或 losing/catastrophic seeds，不把三 seed 写成稳健/SOTA。
- 不修改 `E:\HCC-main`，不提交 `results/` 或用户原有未跟踪材料。
