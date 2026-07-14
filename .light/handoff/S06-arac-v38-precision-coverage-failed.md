---
session_no: S06
suggested_title: "[ARAC] S07 重新设计 evidence coverage"
parent_session: S05
project: arac
date: 2026-07-14
---
## 当前阶段
v38 post-retirement precision reanchor 已完成实现、真实 HCC 验证和
current-winning-13 门控分析。实现与审计均通过，但离线 mean gate 仍为
`5/13 < 6/13`，因此按预注册规则停止，未运行 v38 full-24。

## 已完成（产物路径 + 验证摘要）
- `docs/superpowers/specs/2026-07-14-post-retirement-precision-reanchor-design.md` — 预注册规则与 observed result；实现固定在 `ee6f28c`。
- `results/controller_v38_post_retirement_precision_reanchor_5k_20260714` — `21/21 fresh`；FE/AOB/leakage 全过。
- `results/controller_v38_post_retirement_precision_reanchor_a4_seed1_3m_probe_20260714` — `695` precision rows 全部在 retirement 后；FE `2,999,998`；AOB/leakage 全过。
- `results/controller_v38_post_retirement_precision_reanchor_13win_seed123_3m_20260714` — `39/39 fresh`；FE `2,999,984..3,000,000`；AOB `384/384 unchanged`；anti-leakage `16/16`；best/mean/worst `13/5/4`；seed wins `24/39`；catastrophic `9/39`。
- `results/controller_v338_full24_seed123_3m_20260713` — 原始 v33.8 full-24 已完成且 best-of-three `13/24`；不要把 v38 未跑 full-24 误写成项目从未完成 full-24。

## 工作区状态
v38 实现已提交为 `ee6f28c`。tracked 结果文档、passport 和本卡在本阶段结果提交中；`results/` 被忽略但 artifacts 完整。`main` 仍显著领先 `origin/main`，不要 push。大量既有未跟踪 FlyKI/文稿文件属于用户，未触碰。

## 下一步（≤3 条，最小动作）
1. 不运行 v38 full-24；若继续方法探索，先预注册一个能覆盖“未 retirement 的失败 run”的纯 runtime-evidence 路由。
2. 新候选仍先过 current-winning-13 gate：best `13/13`、mean `>=6/13`、worst `>=4/13`、seed wins `>=24/39`、catastrophic `<=9/39`、S2/S3 `6/6`。
3. 若不再探索，直接以 v33.8 full-24 `13/24` 和 v36-v38 负消融形成审计报告；不得把三 seed 包装成显著性结论。

## 阻塞/风险
v38 的失败是 evidence coverage 不匹配：`4,397` precision rows 中 `3,984`
集中于 A4/A5；S2 三 seed 明显改善但原本已 `3/3` 胜；R1/R2/E6/S6
没有 retirement，未收到 precision，因此 mean/catastrophic 门未改善。继续探索会复用同三
seed，存在开发集过拟合风险，任何新机制必须先预注册且禁止 outcome/case dispatch。

## 必读文件（按序）
1. `.light/handoff/S06-arac-v38-precision-coverage-failed.md`
2. `.light/passport.yaml`
3. `AGENTS.md`
4. `docs/design/core-method.md`
5. `docs/design/boundaries.md`
6. `docs/superpowers/specs/2026-07-14-post-retirement-precision-reanchor-design.md`

## 禁止
- 别重跑已完成的 v33.8/v36/v37/v38 artifacts；别凭记忆补写未验证结论。
- 别把 paper-best、case label、function family、历史/最终 outcome 放进 runtime dispatch。
- 别修改 `E:\HCC-main`；别 push；别覆盖用户未跟踪文件。
- 别把本卡当作当前事实——接手后先用 git status / git log 刷新现实再动手。
