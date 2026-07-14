---
session_no: S09
suggested_title: "[ARAC] S09 CAR-W 实现与 5k 审计收口"
parent_session: S08
project: arac
date: 2026-07-15
author: Codex
---

## 当前阶段

CAR-W implementation freeze 已完成并提交于 `f5cd749`。当前 canonical
fallback 仍是 v33.8；CAR-W 只在显式 lane profile 中启用，不能据此作性能
结论。

## 已完成

- 实现 `arac_counterfactual_action_racing_w`：K=3 sequential paired probes、
  equal-FE、counter-based CRN、single FE ledger、LCB runtime safety score、
  worst-tail gate、最终 pair 原子 adoption 和 discarded branch 隔离。
- component horizon 使用完整 overlap connected component；只有连续两个
  sweep 同一非-fallback action name/family 的稳定支持子图写回；候选只拥有
  一个 component-horizon lease，后续恢复 v33.8 fallback。
- 新增 `car_probe_trace.csv`、`car_state_ledger.csv`、
  `car_branch_manifest.csv` 和类型级 `car_dispatch_boundary_audit.csv`。
- `results/car_w_exp003_5k_20260715_final`：E2 seed1 fresh，4989/5000 FE，
  same-budget violation=0，AOB 10/10 unchanged，通用 anti-leakage 16/16，
  类型边界 28/28；3% probe cap 无法容纳完整 horizon，按协议 abstain。
- focused tests `131 passed`；Git 跟踪测试集合 `718 passed, 1 skipped`；
  compileall pass；passport 已更新到 stage 19。

## 工作区状态

- `main` 已包含 CAR-W commit `f5cd749`，尚未确认远端同步。
- 用户既有未跟踪的 FlyKI、exp006-exp008、论文和历史材料保留在工作区，
  不属于本轮提交。
- 全量 pytest 额外发现 37 个未跟踪 exp008 测试因旧的
  `experiments.exp_005_hcc_final_protocol_pilot.run` 路径缺失而失败；未修改
  这些无关文件。
- `ruff` 未安装；compileall、pytest、diff check 已执行。

## 下一步

1. 核对并推送 `f5cd749` 到 `origin/main`。
2. 按冻结协议用 seeds 9/10/11 跑六个 topology-stratified diagnostic cases，
   每个包含 v33、CAR-W、shuffled-W、paired-fallback-probe 和 no-action controls。
3. 只在至少 6 commits 覆盖 3 cases/2 topology strata、probe-to-3M sign
   agreement >=60%、mean<0、median<=0、zero catastrophic、overhead<=6% 时
   放行 R/S；否则记录 W 失败原因并停止扩展。

## 必读文件

- `docs/design/core-method.md`
- `docs/literature_review.md`
- `src/arac/policy/counterfactual_action_racing.py`
- `src/arac/backends/hcc_car.py`
- `scripts/hcc_smoke_runner.py`
- `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`
- `.light/passport.yaml`

## 禁止

- 不把 case label、function family、paper-best、历史结果、final outcome
  接入 runtime dispatch。
- 不把 5k smoke 或 100k diagnostic 当作 3M utility/performance 结论。
- 不在 W gate 未通过前实现或叠加 R/S channel。
- 不纳入或覆盖用户现有未跟踪文件。
