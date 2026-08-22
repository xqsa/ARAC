# Gate P0：Shared-Patch Kernel 契约协议（预注册记录）

日期：2026-08-22
依据：用户方案"ARAC-OC Shared-Patch 完整落地方案"§4 Gate P0。
实现：`src/arac/coordination/shared_patch.py`（内核）、`state.py`（schema v2）、
`contract.py`（receipt v2 字段）、`loop.py`（CTP 挂载，`patch_config` 默认
None = 原行为逐位不变）。

## 范围

toy overlap + 统一循环集成；不涉及性能结论。

## 十二项检查 → 测试映射（tests/test_shared_patch.py，9 项测试全部通过）

| 检查 | 测试 |
|---|---|
| 候选始终在界内 | test_candidates_within_bounds（ledger 越界即 raise） |
| exact FE | test_exact_fe_and_strict_best_monotone（8 FE 精确） |
| strict-best 单调 | 同上 |
| 车道只消费预留 | test_budget_unavailable_is_explicit（<8 FE 拒绝、0 消耗） |
| patch_budget_unavailable 显式 | 同上 |
| u_j 不进候选方向 | test_u_never_enters_candidate_direction（同 seed/上下文重放逐位一致） |
| context hash 变化 reset | test_context_reset_reinitializes_and_decays |
| 成功扩大/失败缩小半径 | test_radius_expands_on_success_and_shrinks_on_failure（trace 峰值判定 + flat 目标确定性缩小） |
| state hash 可重复 | test_state_hash_reproducible |
| v1 snapshot 恢复到 v2 | test_state_schema_v2_and_v1_restore |
| 异常 fail-closed | test_invalid_scope_fails_closed（非法 scope/mode raise） |
| 重锚后首个 patch 正常 | test_context_reset_reinitializes_and_decays（reset 后 consumed=8） |

补充：统一循环级集成（patch on/off 各跑一遍）在 test_shared_patch.py 的
loop 集成测试中断言 exact terminal FE 与收据完整。

## 判定

**P0 通过**（2026-08-22，9/9 + 集成测试通过，loop 既有回归 18/18 不变）。
按方案继续 Gate P1。
