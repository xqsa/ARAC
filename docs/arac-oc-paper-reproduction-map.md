# ARAC 论文主张复现地图（current-HEAD 活体状态）

日期：2026-08-24。目的：论文的每条主张由哪个入口产出、用什么命令复现、
冻结产物在哪、以及**当前 HEAD 上的活体验证状态**。今天完成的活体验证
均标注 ✅ 并给出产物目录。

## 主张 → 入口 → 复现状态

| 主张 | 产出入口 | 复现命令（`PYTHONPATH='.;src'`） | 冻结产物 | 当前 HEAD 状态 |
|---|---|---|---|---|
| C1/C2 黑盒重叠发现 | soft-RDDSM 基线（AOB 六 case） | `python experiments/soft_rddsm_aob_baseline_v2.py --workers 6` | `artifacts/soft_rddsm_aob_baseline_v2,v3` | ✅ **今日活体复现**（`artifacts/upgrade_paper_entry_c1_headrev_v1`）：召回 0.789(R6)–1.000(R2)、精度 1.0 全程、E1/R1 零误报、预算逐笔对账——与冻结区间逐端点吻合 |
| C3 证据驱动派发 | Gate 41a 离线评估（读冻结四臂矩阵收据，零新 FE）+ 41b 在线确认 | `python experiments/overlap_action_dispatch_gate41_offline.py` | `artifacts/overlap_action_dispatch_gate41_offline`、`current_recovered_four_arm_matrix_v2` | ✅ **今日活体复现**（`artifacts/upgrade_paper_entry_c3_headrev_v1`）：21 胜 3 负 vs HCC、geo 0.0645、规则命中 oracle ≥0.95、全 case 历史列 1.10× 内，四项检查全绿 |
| C3 的活体基质 | RecoveredActionRegistry 四臂执行（AOB） | `execute_phase2_action + RecoveredActionRegistry`（见 historical_recovery replay） | recovery screen 120/120 | ✅ 早已活体验证：T1 identity 臂（S3/ctp、E3/smp）与冻结收据逐位一致；E1 战役 60 臂全过终端契约 |
| C4 完整目标仲裁 | Gate 29/30（**24 维受控重叠生成器**，非 AOB） | `python experiments/overlap_arac_gate29_screening.py --seed <s>` | `artifacts/overlap_arac_gate29,30*` | 入口 import OK；**AOB 非其主张域**（诚实边界 #5：仲裁收益在高重叠稀疏域以外未验证） |
| C5 组合调度存在性 | Gate 50c episodes（cut-2 已删除 episodes 模块） | 无法从 HEAD 重跑；查 `git tag v5.3-prealation` | 冻结 gate 产物 | ❌ 入口断裂（ModuleNotFoundError: arac.coordination.episodes）——证据只在冻结产物与 tag |
| C6 工程契约 | tests/ 契约测试 | `python -m pytest tests/` | — | ✅ 核心契约 17/17 绿 |

## 关键边界（写进论文，勿混）

1. **统一 OC 循环（`run_arac_oc`）不是 AOB 入口**：AOB dense-overlap 上
   Phase-I membership 证据 fail-closed（诚实边界 #2，历史行为）。它在
   Gate-40/46 校准 cell 上运行，且必须携带 blessed pilot 参数
   （`anchor_count=5, step=0.25, rounds=12, bucket_size=16,
   max_candidate_pairs=128`）——默认参数在 1000 维需 315,040 FE >
   180k Phase-I 上限，直接崩溃（P0-4，已实证）。
2. **C4 是受控域机制主张**，不是 AOB 主张；如需 AOB 化，是把 soft-RDDSM
   证据接入协调器的接口工程（arac-core-method §8 的未来工作），不是复现。
3. C5 只能从 tag `v5.3-prealation` 复现。

## 外审 P0 裁定补充（今日核实）

- P0-5（value gate 回滚）：`operator_value_ratio` 默认 0.0，回滚分支
  默认不触发，best-observed 语义在默认配置下安全。
- P0-7（`dataclasses.replace` NameError）：仅在 patch 执行分支可达，
  patch 默认关闭（退役），分支不可达。
