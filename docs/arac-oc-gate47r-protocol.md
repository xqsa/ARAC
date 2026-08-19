# Gate 47-R 协议：proposal residual + topology 路由确认

日期：2026-08-16  
执行者：Codex  
状态：已执行，通过（2026-08-16）

## 1. 目的

Gate 46 证明统一循环在 24 个稀疏 cell 上不劣，但统一臂的 operator FE
全部为零。Gate 47 随后证明 counted probe 的 `C_j` 幅值在 chain/star
之间不存在可用阈值间隙。因此 Gate 47-R 不再尝试降低 `tau_enter`，而是
确认 ARAC-OC 的实际调度链：

```text
SMP.sense
  -> proposal residual persistence
  -> relative hub / qhat / escalation
  -> GCB OperatorPlan
  -> SMP | restricted CTP | shared-core CTP | AOR
  -> strict-best + receipt/state feedback
```

本 gate 只验证调度路径、预算契约和状态反馈，不宣称 AOB 性能优越性，也不
修改 Gate 47 的历史 artifacts。

## 2. 冻结规则

1. `C_j` 只用于 scope 排序、探针尺度和诊断收据；其幅值不能直接打开动作。
2. 连续 `persistent_streak` 次高 proposal residual 才允许从 arbitration-only
   进入算子派发。
3. `relative_hub < complex_hub_ratio` 路由到 restricted CTP；达到阈值路由
   到 shared-core CTP。
4. `qhat_mean < smp_trust_floor` 路由到 SMP state rebuild。
5. `conflict_streak >= escalation_streak` 且尚未使用 AOR 时路由到 AOR；AOR
   不是异常 fallback。
6. 所有动作通过同一 `OperatorPlan`/`OperatorReceipt` contract；exact FE、
   strict-best、状态 hash 链和 fail-closed 保持不变。
7. 使用新鲜 seeds。旧 Gate 47 的 `phaseA/phaseB` 只能作为失败背景，不能被
   读取为本 gate 的通过证据。

## 3. 实验单元

Phase-I 使用与 Gate 29/38/39 相同的稀疏 conflicting overlap 构造，包含：

- chain/overlap=3：应产生 restricted CTP；
- star/overlap=6：应产生 shared-core CTP；
- chain/overlap=6：在低 qhat 注入后应产生 SMP；
- chain/overlap=3：在达到 escalation streak 后应产生 AOR。

fresh-seed 层固定为 `20260836`（路由路径）和 `20260837`（重放/确定性层）。
每个 cell 仍先完成 Phase-I 180,000 FE，随后使用统一 ARAC-OC 入口；路由
诊断另外使用同一 checkpoint 上的 plan-only 状态回放，避免把“没有自然触发”
误报成 GCB 能力不存在。

## 4. 判定

协议检查：

- Phase-I FE 精确为 180,000；
- 终端 FE 精确到总预算；
- 每个正常 receipt 的 `actual_fes == reserved_fes`；
- 所有 cycle 和 receipt 保持 strict-best；
- receipt 携带 64 位 state hash，重放 hash 链一致；
- operator 异常只产生 `operator_failed` receipt，不自动切换动作。

路由检查：

- 至少一个 fresh chain cell 产生 `ctp_restricted`；
- 至少一个 fresh star cell 产生 `ctp_shared_core`；
- qhat 低的 plan 产生 `smp`；
- escalation streak 的 plan 产生 `aor`；
- counted probe 高幅值但 residual 未持续时仍为 `arbitration_only`。

失败处理：保留所有 cell receipt 和状态快照，按具体缺失路径报告；不回调
`tau_enter`，不改写旧 Gate 47 结果，不通过隐藏重试补齐动作计数。

本次执行的 3 个真实 fresh cell（seed `20260836`）为 chain/ov3、chain/ov6
和 star/ov6。三者均通过 exact FE、strict-best 和 receipt parity；自然运行
的 operator FE 分别为 215、308、20，实际动作覆盖了 restricted CTP、shared-core
CTP、SMP 和 AOR。两个 fresh witness seed 均覆盖五个预期分支，其中高幅值
`C_j` 但无 residual persistence 的场景仍保持 `arbitration_only`。

## 5. 运行与产出

```powershell
.venv\Scripts\python.exe -m experiments.oc_residual_topology_gate47r --workers 4
```

产出目录：`artifacts/oc_residual_topology_gate47r/`，包括每个 fresh cell 的
receipt 摘要、plan-only 路由证明、协议检查和最终 `confirmation.json`。
