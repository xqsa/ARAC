---
session_no: S05
suggested_title: "[ARAC] S06 设计第二稳定性机制"
parent_session: S04
project: arac
date: 2026-07-14
---
## 当前阶段

v36 first-sweep evidence maturity 候选已完成实现、smoke、3M route probe 和
current-winning-13 阶段门。阶段门失败，v33.8 继续作为 canonical；尚未开始
full-24。

## 已完成（产物路径 + 验证摘要）

- `scripts/hcc_smoke_runner.py`、`src/arac/actions/contracts.py`、
  `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py` — v36 单一
  action/lane、maturity state/trace、matched-FE 和 runtime state 接线完成；
  Git-tracked tests `628 passed, 1 skipped`，compileall pass。
- `results/controller_v36_sweep_maturity_5k_fixed_20260714` — 12/12 fresh，
  FE 超支 0，AOB 120/120 unchanged，anti-leakage 16/16。
- `results/controller_v36_sweep_maturity_s6_dense_5k_20260714` — 3/3 fresh，
  dense fallback 45 行，FE/AOB/leakage pass。
- `results/controller_v36_sweep_maturity_s3_seed1_3m_probe_20260714` —
  repair transparency 158 行，1/1 fresh exact-3M，FE/AOB/leakage pass。
- `results/controller_v36_sweep_maturity_13win_seed123_3m_20260714` — 39/39
  fresh，raw artifact sets 39/39，FE 超支 0，AOB 384/384 unchanged，
  anti-leakage 16/16；offline best/mean/worst `10/4/2` of 13，seed wins
  17/39，catastrophic 14/39，阶段门 FAIL。
- `docs/superpowers/specs/2026-07-14-first-sweep-evidence-maturity-guard-design.md`
  — 记录实际 route、失败原因和不运行 full-24 的决定。

## 工作区状态

v36 implementation commits through `2c40937`; 本卡、spec 和 passport 位于
本地 evidence commit。`results/` 被忽略；仅保留用户既有未跟踪材料。当前
`main` 未 push。

## 下一步（<=3 条，最小动作）

1. 对 E2/S5 等无 maturity route 的 v33.8/v36 当前代码做最小同 seed 诊断，
   区分 canonical drift 与优化器重复运行方差。
2. 设计一个与 v36 阈值调节不同、可独立归因的稳定性机制；先做 runtime
   legality、matched-FE 和 protected controls 的盲审。
3. 只在小型 protected gate 通过后运行 current-winning-13；满足项目级
   mean>=6、worst>=4、catastrophic<=27/72 的充分证据前不得 full-24。

## 阻塞/风险

- v36 合法 evidence 不等于长期 utility：R2 seed3 是 false-positive maturity，
  S2/S3 仍有 catastrophic seed。
- 三 seed 仅是 pilot 级证据；无 maturity route 的 case 也出现跨 fresh run
  波动，归因前必须排除 current-canonical drift/随机性。
- v36 阶段门已失败；不能事后改边界并沿用 v36 名义。

## 必读文件（按序）

1. `.light/handoff/S05-arac-v36-sweep-maturity-failed-candidate.md`
2. `.light/passport.yaml`
3. `docs/superpowers/specs/2026-07-14-first-sweep-evidence-maturity-guard-design.md`
4. `docs/design/core-method.md`
5. `docs/design/boundaries.md`

## 禁止

- 不直接运行 full-24，不降低 final gate。
- 不把 case ID、function family、paper-best、历史结果、relative gain 或 final
  outcome 放入 runtime dispatch。
- 不把 v36 的 outcome 后验阈值修改包装成原候选；新机制必须新版本、单独归因。
- 不修改 `E:\HCC-main`，不覆盖既有 results，不 push，除非用户另行授权。
- 别重做已完成清单；接手后先刷新 git/status/passport，再行动。
