# Gate 47b 协议：streak 触发派发的稀疏域确认

日期：2026-08-16
状态：预注册（正式运行待执行）

## 1. 研究问题

Gate 47 判定 counted C_j 幅值对拓扑可修复性无判别力后，派发信号已
重构为 **proposal 残差 streak 触发 + 相对 hub 路由**（与冻结内核
Gate 38 v2 表同源），C_j 降级为诊断性 EMA。Gate 46 的 24/24 不劣
证据覆盖的是旧保守配置（零派发）；本 gate 回答：

> (a) 新机制下算子派发是否实际开火？（b) 开火后统一臂对冻结内核
> 是否保持不劣？（c) star 对 proposal 基线是否无回归？

## 2. 设计

- **层 1（12 cell）**：gate40 矩阵（ackley/elliptic/schwefel ×
  chain/star × ov3/6），seed 20260832。只跑 `oc_unified`(streak)
  臂；内核与 proposal 参照直接取 Gate 46 冻结收据
  （`oc_unified_sparse_regression_gate46/cells/`，其中内核重跑已
  逐位复现 gate40）。
- **层 2（6 cell）**：rastrigin 族（chain/star/random × ov3/6，
  Gate 29 `build_cell`，conflicting），seed 20260835（从未使用）。
  三臂配对共享 checkpoint：内核（`gcb_coordinated` v2）、
  `oc_unified`(streak)、proposal 基线。
- **streak 生产配置**：`OcCoordinatorConfig()` 默认（persistent
  streak 2 / escalation 6 / relative hub 0.9 / SMP 信任地板 0.5），
  脉冲界 (8, 32)（延续 Gate 46 预算纪律）；EMA/τ 保留为诊断通道。
- 共 12 + 6 cell，速度优先全并行（workers 16）。

## 3. 判定（预注册，容差 1e-9）

协议检查：`layer1_cell_count_12`、`layer2_cell_count_6`、
`phase1_exact`、`terminal_exact`、`strict_best`、
`unified_receipts_ok`（消耗平价 + 状态哈希链，两层全部运行）。

筛查判据：

1. `path_fires_on_chain`：≥1 个层 2 chain cell 的统一臂
   operator FE > 0（新机制生死判据）；
2. `not_worse_than_kernel_all`：18/18
   `unified ≤ kernel × 1.05 + tol`（层 1 用 Gate 46 冻结内核值）；
3. `star_no_regression_vs_proposal`：8 个 star 实例
   `unified ≤ proposal + tol`（层 1 用冻结 proposal 值）。

报告项（不判定）：chain/ov3 血统收益捕获（统一臂 vs 内核配对
收益）、派发动作分布、预算流向。失败处置：按收据归因，不回调
阈值。

## 4. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.oc_streak_confirmation_gate47b --workers 16
```

产出：`artifacts/oc_streak_confirmation_gate47b/`（cells/ +
confirmation.json）。
