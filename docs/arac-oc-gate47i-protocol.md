# ARAC-OC Gate47i 协议

日期：2026-08-16  
执行者：Codex  
目的：确认 shared-core CTP 的预算可行性检查能避免结构上不足的低价值 pulse。

## 预注册规则

- 复用 Gate47b 的 12 个 Layer-1、6 个 Layer-2 配对矩阵；Layer-2 使用新鲜
  `seed=20260843`。
- 对 shared-core CTP，最小 operator 窗口为 `2 * len(scope)` FE。当前 pulse 或
  operator pool 不足时输出 `shared_core_budget_unavailable` arbitration-only，
  不预留、不消耗 operator FE。
- 继续使用 `arbitration_value_ratio=0.01` 和 `operator_value_ratio=0.01`；所有
  sensing/probe/writeback/terminal tail 及其他 operator contract 保持不变。
- 规则不得检查 benchmark 名称、topology 或最终结果；Gate47g/h 产物保持不变。

## 通过条件

协议检查、receipt parity、strict-best、state-hash chain、
`path_fires_on_chain`、`not_worse_than_kernel_all` 和
`star_no_regression_vs_proposal` 全部通过。

产物：`artifacts/oc_streak_confirmation_gate47i/`。
