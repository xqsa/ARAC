---
session_no: S07
suggested_title: "[ARAC] S08 重构 Phase-I 动作可识别性"
parent_session: S06
project: arac
date: 2026-07-15
---
## 当前阶段
paired v33/v36 held-out 3M 验证已完成。运行完整性全部通过，但预注册
utility gate 失败，因此按协议停止，未启动 full-24。

## 已完成（产物路径 + 验证摘要）
- `results/controller_paired_v33_v36_pinned_5k_seed4_20260714` - 12/12 fresh；FE、AOB、anti-leakage、action plan、pinned environment 完整性通过。
- `results/controller_paired_v33_v36_13case_seed45678_3m_20260714_retry` - 13 cases x 5 seeds x 4 lanes = 260/260 fresh；FE 2,999,984..3,000,000，超支 0/260；AOB 2,560/2,560 unchanged；anti-leakage 16/16；action plan 260/260 consumed/allowed。
- `results/controller_paired_v33_v36_13case_seed45678_3m_20260714_retry/paired_runtime_utility_gate.json` - integrity pass，utility blocked：mean log delta -0.002233，mean wins 3/13，worst wins 1/13，meaningful wins 2/65，catastrophic 1/65。
- `docs/superpowers/specs/2026-07-14-paired-runtime-utility-validation-design.md` - 已追加 observed result 和 runtime-evidence 失败诊断。

## 工作区状态
代码固定在 `dd5ffec`；本阶段仅更新 tracked 协议、passport、autonomous 台账和本卡。`results/` 被忽略但 raw artifacts 完整。用户未跟踪的 FlyKI、论文和外部源码文件未触碰。最终验证、提交和 push 尚待本会话完成。

## 下一步（<=3 条，最小动作）
1. 不运行本候选 full-24；先以 paired artifact 为依据重构 Phase-I evidence 到动作的可识别性。
2. 新设计必须同时提高 action coverage，并加入 current-run、outcome-free 的风险校准；先用新的 held-out seed 预注册 paired fallback gate。
3. 禁止继续在 seeds 1-8 上调参；这些 seed 已成为开发/验证证据，下一候选必须冻结后使用新 seed。

## 阻塞/风险
v36 与 v33 在 59/65 case-seed pair 上最终结果完全一致，说明当前动作覆盖过低；仅六个变化 pair 中既有 S2/S3 改善，也有 E2 退化和 S3 seed-7 约 35% catastrophic loss。现有 Phase-I evidence-to-action 映射没有识别出广泛且安全的 long-horizon utility。

## 必读文件（按序）
1. `.light/handoff/S07-arac-paired-runtime-utility-blocked.md`
2. `.light/passport.yaml`
3. `AGENTS.md`
4. `docs/design/core-method.md`
5. `docs/design/boundaries.md`
6. `docs/superpowers/specs/2026-07-14-paired-runtime-utility-validation-design.md`
7. `results/controller_paired_v33_v36_13case_seed45678_3m_20260714_retry/paired_runtime_utility_gate.json`

## 禁止
- 别把 260 条 trajectory 写成 260 个 case；实际是 13 case、65 case-seed pair、4 lane。
- 别绕过 held-out gate 运行 full-24，别把 integrity pass 写成 utility pass。
- 别把 case label、function family、paper-best、历史最优、final outcome 或 relative gain 放进 runtime dispatch。
- 别修改 `E:\HCC-main`；别覆盖用户未跟踪文件。
- 别把本卡当作当前事实，接手后先用 git status / git log 刷新现实再动手。
