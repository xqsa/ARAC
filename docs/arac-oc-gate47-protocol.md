# Gate 47 协议：统一循环 τ 阈值校准门

日期：2026-08-15
状态：**已执行，fail-closed 判定：τ 间隙不存在，阈值未选定**
（Phase B 按预注册规则未运行）
注：原"Gate 47 AOB 端到端"顺延为 Gate 48。

> **执行修订（运行前）**：cell 矩阵实际为 6（gate29 的 TOPOLOGIES 含
> chain/star/random 三拓扑 × ov3/6）。random 为 Gate 38 已声明的混合
> 类，仅作遥测记录，不入正负类。正类 = chain（2 cell），负类 =
> star（2 cell），τ 规则不变。此修订在查看任何 C 值之前做出。

## 1. 研究问题

Gate 46 判明：统一臂全程零算子派发——counted 冲突分 C_j 在未校准的
`tau_enter=0.5` 下从未越带，滞回机保持在 low。本 gate 回答：

> (a) 真实稀疏网格上 C_j 的分布是否存在可分离的"应升级/不应升级"间隙？
> (b) 按预注册规则选定 τ 后，算子派发路径能否实际开火？
> (c) 开火后统一臂对冻结内核是否仍保持不劣？

## 2. 设计

- **样本来源**：rastrigin 族稀疏网格（Gate 29/39 的 `build_cell`，
  conflicting 模式）——这是唯一有书面派发收益记载的族（Gate 38
  chain/ov3 +3.57、Gate 39 +31.41）。
- **Phase A（遥测）**：4 cell（chain/star × ov3/6）× seed 20260834
  （从未使用）。以**不可升级配置**（`tau_enter=2.0 > 1 ≥ C`，
  探针照常运行）跑统一臂，遥测每周期 `probe_max_c`（本次新增的
  trace 字段）。统计量：cell 级 `s = max_cycles probe_max_c`。
- **正负类**：正类 = chain cell（有派发潜力），负类 = star cell
  （Gate 39 证实数值平局）。
- **τ 选择规则（预注册）**：令 P = {chain 的 s}，N = {star 的 s}。
  若 `min(P) ≤ max(N)` → **校准失败**（报告分布，不强行定阈值）。
  否则 `tau_enter = sqrt(min(P)·max(N))`（几何中点），
  `tau_exit = tau_enter/2.5`，`k_enter = k_exit = 2`，EMA α=0.3，
  脉冲界 (8, 32) 不变；配置版本升级为
  `calibration_status = "tau-calibrated-gate47"`。
- **Phase B（确认）**：同 4-cell 矩阵 × seed 20260835（从未使用），
  三 arm 配对共享 checkpoint：冻结内核（`gcb_coordinated` v2）、
  统一臂（校准后配置）、proposal 基线。

## 3. 判定（预注册，容差 1e-9）

协议检查：`phaseA_cell_count_4`、`phaseB_cell_count_4`、`phase1_exact`、
`terminal_exact`、`strict_best`、`unified_receipt_parity`（含状态哈希链）。

筛查判据：

1. `tau_gap_exists`：Phase A 间隙成立（否则门判失败并报告分布）；
2. `path_fires_on_chain`：≥1 个 chain cell 的统一臂实际派发算子
   （operator FE > 0）；
3. `not_worse_than_kernel_all`：4/4 `unified ≤ kernel × 1.05 + tol`；
4. `star_no_regression_vs_proposal`：2 star cell
   `unified ≤ proposal + tol`。

报告项（不判定）：chain cell 上统一臂 vs 内核的配对收益（是否捕获
Gate 38/39 型收益）、动作分布、预算流向。

失败处置：按收据归因，不回调阈值。

## 4. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.oc_calibration_gate47 --workers 4
```

产出：`artifacts/oc_calibration_gate47/`（phaseA/、phaseB/、
confirmation.json）。
