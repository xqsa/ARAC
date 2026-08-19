# Gate 50 协议：episode 级 ARAC-OC 对四完整动作的非回归

日期：2026-08-16
状态：预注册（正式运行待执行）

## 1. 研究问题

四动作已完整收编为 v2 episode（ctp/gcb：Gate 50a 逐位；smp/aor
recovered：双逐位）。本 gate 回答收编后的核心问题：

> 同一 Phase-I checkpoint、同总预算下，episode 级 ARAC-OC 调度
> （GCB 分段分配 + 停滞切换）是否不劣于**每一条**单独跑的完整
> episode，且预算/收据契约成立？

## 2. 设计

- Case：R2 / A3 / S5 / R6（48a/48b/49 血统），Phase-I = v3 发现 +
  MMES 填充至精确 180k（seed 20260845，确定性复现）。
- Episode checkpoint：blocks = v3 区域树互斥叶（覆盖全部变量），
  relations = 叶间关系（Gate 42 证据），features 按可计算项
  （log10 incumbent 误差）。
- **五臂配对**（同 checkpoint、各 3M、独立账本）：
  1-4. 四条 standalone episode（v2 状态机单发跑满）；
  5. `oc_schedule`（`run_oc_episode_schedule`：四 episode 常驻
  镜像账本、GCB 300k 分段、非实质增益即切换、max 6 切换、
  初始 episode = 41b 冻结派发表的同 case 输出——R2→gcb、
  A3→ctp、S5→ctp、R6→gcb）。
- 选择开销为零 FE（切换决策读收据不花评价）；ε 容忍的是初始
  选择错误时烧掉的段预算。

## 3. 判定（预注册，容差 1e-9）

协议检查：`arm_count_20`、`phase1_exact_180k`、`terminal_exact_all`
（5 臂全部 3M）、`strict_best_all`、`oc_receipts_valid`（调度收据
非空 + schedule_hash + 每段状态哈希链）。

筛查判据：

1. `not_worse_than_best_episode_all`：4/4 case
   `oc_final ≤ min(四 standalone) × 1.05 + tol`；
2. `oc_beats_or_ties_initial_episode_all`：4/4
   `oc_final ≤ 对应初始 episode standalone × 1.05 + tol`（调度
   不得比"直接跑预选动作"差——ε 的实质检验）。

报告项（不判定）：funded 份额、切换轨迹、与 41b 均值之比、
交错上行实例（oc 严格优于全部 standalone 的 case）。

失败处置：按收据归因；切换阈值（material = log(1.01) 冻结值）不
回调。

## 4. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.oc_action_episode_gate50 --workers 20
```

产出：`artifacts/oc_action_episode_gate50/`（cells/ + confirmation.json）。
