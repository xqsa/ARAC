# Gate 40 协议：协调器跨底函数族泛化（AOB 四族中的三新族）

日期：2026-08-15
状态：预注册（AOB24 适用性审计与阈值迁移校准已完成，正式运行待执行）

## 1. 背景与两条前置证据

**AOB24 适用性审计**（`artifacts/aob24_overlap_applicability_audit/`）：24 个
AOB case 上，Phase-I 稀疏重叠发现全部 fail-closed（`candidate_pair_cap_
exceeded`，候选交互对 49.7 万+/49.95 万 ≈ 100%）。AOB 的真实结构为
755 组、变量平均属 18–79 组、6.8% 变量对同组交互——稠密重叠域，超出
稀疏协调器适用条件。**结论：AOB-24 原套件不是当前协调器的合法泛化
目标；直接套用属于违反适用假设。** 协调器在 AOB 上的正确行为是
fail-closed 转 AOR（ARAC-Core 路由规则），已由审计凭证化。

**本 gate 的泛化对象**：AOB 的四个底函数族中未经检验的三族
（ackley / elliptic / schwefel；rastrigin 已由 Gate 38/39 覆盖），加载到
本仓库的稀疏重叠网格（1000-D、24 active、4 组）上。

**阈值迁移校准**（`artifacts/overlap_family_calibration_gate40/`，36 个
Phase-I，2 新 seed）：三族上 adapter 36/36 ready；chain rel_hub
0.667–0.824、star 全部 1.000——**Gate 38 冻结的 `relative_hub ≥ 0.9`
阈值无需重校直接迁移**。random 在这些 seed 上全部饱和（归入 star 类，
接受为已声明边界）。

## 2. 设计

- 配置：3 族 × {chain, star} × overlap {3, 6}，conflicting 模式 = 12 cell。
- seed：20260832（未用于任何 gate、校准或审计）。
- 每 cell：一次 Phase-I pilot；三 arm 配对（`gcb_coordinated` v2 配置、
  `proposal_neighborhood`、`persistent_ctp`），共享 checkpoint，各 3M FE。

## 3. 判定（预注册，容差 1e-9）

协议检查：cell_count_12、phase1_exact、terminal_exact（三 arm）、
strict_best（三 arm）、envelope 不侵占、消耗=预留。

筛查判据（chain 判据采用 Gate 39 归因修正后的形式，在本次运行前冻结）：

1. `star_no_regression_all`：6 个 star cell 对配对 proposal gain ≥ −tol；
2. `chain_positive_where_potential`：对每个 chain cell，gain > tol，或
   （固定 CTP 潜力 ≤ tol 且 coordinator ≥ 固定 CTP − tol）；
3. `win_or_tie_vs_proposal_ge_0_75`：≥ 9/12；
4. `not_worse_than_persistent_ctp_all`：12/12。

失败处置：按收据归因，不回调阈值。

## 4. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.overlap_family_gate40_screening --workers 4
```

产出：`artifacts/overlap_family_gate40/confirmation_fresh.json`。
