# ARAC-OC Gate47g 协议

日期：2026-08-16  
执行者：Codex  
目的：在不覆盖 Gate47b/c/d/e/f 的前提下，确认版本化 `arbitration_value_ratio=0.01`
是否能消除低价值 operator pulse 导致的尾部机会成本。

## 预注册规则

- 使用 Gate47b 的 12 个 Layer-1 和 6 个 Layer-2 cell 协议；Layer-2 使用新鲜
  `seed=20260841`。
- Phase-I 必须精确消耗 180,000 FE，终端必须精确到 3,000,000 FE。
- ARAC-OC 仍执行完整的 SMP sense、counted probe、proposal arbitration、GCB
  dispatch、operator contract、strict-best、writeback 和 terminal tail。
- 价值门只使用当前周期 `max(0, error_after_sense-error_after_arbitration) /
  max(abs(error_after_sense),1)`，达到 0.01 时把当前 operator 改为
  `arbitration_value_gate`；不得检查 benchmark 名称、topology 或最终结果。
- 旧 Gate47f 结果只作历史参照，不从旧 cell 收据重出新判定。

## 通过条件

协议检查、receipt parity、strict-best 和 state-hash chain 全部通过；
`path_fires_on_chain`、`not_worse_than_kernel_all` 和
`star_no_regression_vs_proposal` 全部通过。该 gate 只验证本门控机制和配对性能
筛查，不宣称已完成 AOB 生产性能校准。

产物：`artifacts/oc_streak_confirmation_gate47g/`。
