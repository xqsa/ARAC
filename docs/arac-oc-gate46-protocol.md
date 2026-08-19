# Gate 46 协议：统一 ARAC-OC 循环对冻结最小内核的稀疏域回归

日期：2026-08-15
状态：预注册（正式运行待执行）

## 1. 研究问题

四阶段适配完成后的主线
`Phase-I → Overlap Coordinator → GCB OperatorPlan → scoped SMP/CTP/AOR → strict-best → CoordinatorState`
（`arac.coordination.loop.run_oc_unified`）作为**系统**尚未在任何基准上
对打过。本 gate 回答：

> 在 Gate 37-40 已验证的稀疏重叠网格上，统一循环是否（a）不劣于
> 冻结的最小内核（`gcb_coordinated` v2 相对 hub 配置），（b）不劣于
> proposal 基线（star 无回归判据），（c）重跑的内核是否逐位复现
> Gate 40 的冻结收据（决定性审计）？

## 2. 设计

- Cell 矩阵：3 族（ackley / elliptic / schwefel）× {chain, star} ×
  overlap {3, 6}，conflicting 模式，共 12 cell（与 Gate 40 完全同构）。
- 两层 seed：
  - **复现层** seed 20260832（Gate 40 的 seed）：跑 `coordinator`
    （重跑审计）与 `oc_unified` 两 arm；proposal/persistent 参照取
    Gate 40 冻结收据。
  - **新鲜层** seed 20260833（从未用于任何 gate/校准/审计）：跑
    `coordinator`、`oc_unified`、`proposal_neighborhood` 三 arm。
- 每 cell 一次 Phase-I pilot（Gate 40 同参数：anchor 5 / step 0.25 /
  rounds 12 / bucket 16 / max_pairs 128，180k FE）；同 cell 全部 arm
  共享同一 checkpoint（记录 checkpoint_hash 平价），各 3M FE。
- 统一臂预算架构（预算配对，隔离决策机制差异）：
  - `refresh_cycles = 16`（与内核同）；
  - `sense_budget_fes` = `_proposal_budget` 同式同输入计算——sense
    阶段与内核 proposal 阶段**同预算、同种子派生、同原语**；
  - 脉冲界 `pulse_min_fes=8 / pulse_max_fes=32`（镜像内核 32 FE/
    组件/周期 envelope）；其余常数为 v1 未校准默认（EMA α=0.3、
    τ 0.5/0.2、k 3/3、γ 1.5/0.5）——`calibration_status` 入收据。
- 共 24 cell × (2~3) arm。

## 3. 判定（预注册，容差 1e-9）

协议检查（全部 arm）：`cell_count_24`、`phase1_exact`（180k）、
`terminal_exact`（3M）、`strict_best`、`checkpoint_parity`（同 cell
同 hash）；内核臂 `envelope_no_encroachment` + `消耗=预留`；
统一臂 `receipt_parity`（全部 OperatorReceipt actual==reserved）+
`state_hash_chain`（每周期收据/trace 携带非空 state hash）。

复现审计（复现层）：`kernel_rerun_matches_gate40`——12/12 cell 重跑
coordinator 终值与 `artifacts/overlap_family_gate40/cells/` 冻结值
逐位一致。

筛查判据：

1. `not_worse_than_kernel_all`：24/24 实例
   `unified.final_error ≤ coordinator.final_error × 1.05`
   （实质性不劣容差，预注册；其中平局/胜数另行报告）；
2. `star_no_regression_vs_proposal`：12 个 star 实例（两层各 6）
   `unified.final_error ≤ proposal.final_error + 1e-9`（不劣于基线；
   复现层用冻结 proposal 值）。

失败处置：按收据归因，不回调阈值；统一臂 `OperatorFailure`
（fail-closed）记为该实例失败。报告项（不判定）：逐实例 win/tie/loss、
动作分布、预算流向（sense/probe/operator/tail 占比）——作为后续
校准门的定量输入。

## 4. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.oc_unified_sparse_regression_gate46 --workers 8
```

产出：`artifacts/oc_unified_sparse_regression_gate46/`（cells/ +
confirmation.json）。
