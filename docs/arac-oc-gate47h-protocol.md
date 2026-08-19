# ARAC-OC Gate47h 协议

日期：2026-08-16  
执行者：Codex  
目的：确认仲裁后价值门与 operator 后验价值门能阻止低价值 archive 写回，同时
保留真实 operator 路径和 exact-FE 语义。

## 预注册规则

- 复用 Gate47b 的 12 个 Layer-1、6 个 Layer-2 配对矩阵；Layer-2 使用新鲜
  `seed=20260842`。
- `arbitration_value_ratio=0.01`：仲裁相对改进达到 1% 时跳过同周期 operator。
- `operator_value_ratio=0.01`：operator 实际相对改进低于 1% 时恢复 archive，
  不退还已消费 FE，receipt 标记 `no_gain`。
- 两个门都不得检查 benchmark 名称、topology 或最终结果；所有 sensing/probe/
  writeback/terminal tail 和 operator reservation 保持原协议。

## 通过条件

协议检查、receipt parity、strict-best、state-hash chain、
`path_fires_on_chain`、`not_worse_than_kernel_all` 和
`star_no_regression_vs_proposal` 全部通过。

产物：`artifacts/oc_streak_confirmation_gate47h/`。
