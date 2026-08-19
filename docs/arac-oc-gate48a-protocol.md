# Gate 48a 协议：AOB 接线 pilot（适用性与预算流向遥测）

日期：2026-08-16
状态：预注册（观察性 pilot，无通过/失败性能判据——那是 Gate 48 正式门的职责）

## 1. 研究问题

稀疏域已收口（Gate 47i 生产配置）。本 pilot 回答 AOB 主线的前两个
问题：

> (a) **接线是否成立**：v3 soft-RDDSM 证据 → Gate 42 超边门 →
> `OverlapStructure` → 统一循环，在 AOB 重叠 case 上能否端到端跑通
> 且逐 FE 契约成立？（b) **预算流向与派发行为**：sense/probe/算子/
> 终端尾各占多少、算子是否开火、与 41b 动作级派发列的量级差距？

## 2. 设计

- Case：R2（ov1，19 共享）、A3（ov3，57）、S5（ov7，133）、
  R6（ov10，190）——重叠度梯度覆盖；零重叠 case（R1/E1）预期在
  超边门 fail-closed，属正确行为，不在本 pilot 范围。
- seed：20260845（与 v3 基线冻结同 seed，Phase-I 侧可直接对照）。
- Phase-I：`discover_hierarchical_soft`（v3 默认配置，180k 契约），
  剩余 Phase-I 预算由 MMES incumbent 段耗至精确 180k；
  `to_overlap_structure`（Gate 42 超边门，无超边即 fail-closed）。
- Phase-II：`run_oc_unified_from_structure`，3M 总预算、16 周期、
  sense 预算 = `_proposal_budget` 同式、生产配置 = Gate 47i 冻结值
  （config v3，脉冲界 8-32，价值门 0.01）。
- 参照线（只报告不判定）：Gate 41b 各 case 25-seed 均值。

## 3. 判定（适用性判据，非性能判据）

1. `wiring_succeeds_4_4`：四个 case 全部完成 超边门转换 +
   统一循环，无 fail-closed、无 OperatorFailure；
2. `terminal_exact_all`：Phase-I 精确 180k、终端精确 3M；
3. `strict_best_all`；
4. `receipts_valid_all`：消耗平价 + 状态哈希链。

报告项：逐 case 共享召回/精确率（vs 真值，离线审计）、结构统计
（组数/组件数/scope 大小）、循环遥测（动作分布、价值门触发次数、
预算流向）、最终误差与 41b 均值之比。

## 4. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.oc_aob_wiring_pilot_gate48a --workers 4
```

产出：`artifacts/oc_aob_wiring_pilot_gate48a/`（cells/ + pilot.json）。
