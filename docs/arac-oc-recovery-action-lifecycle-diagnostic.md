# Recovery Action Lifecycle Diagnostic

日期：2026-08-23  
执行者：Codex  
产物：`artifacts/recovery_action_lifecycle_diagnostic_v1/summary.json`

## 实验边界

本诊断固定复用已有 v9 Phase-I checkpoint，不重新运行 Phase-I。覆盖：

- SMP：E2-E6；GCB：R1-R6；
- seeds：117、123、129、135、141；
- 每个 case/seed 成对运行 `current` 与 `historical_compatible`；
- 共 55 pairs、110 arms；24 workers、native threads=1；
- patch、soft routing、selector 全部关闭。

`historical_compatible` 使用仓库冻结的 action source schedule。它是生命周期隔离对照，不是 bitwise historical replay；冻结 source 仍依赖当前公共 execution/runtime helper。因此本报告不授权历史 superiority 或最终恢复声明。

## Contract 结果

- 110/110 arms 完成，0 failures；
- 55/55 pairs 使用相同 checkpoint hash；
- terminal FE、Phase-II FE 全部精确；
- 110/110 receipt、action-result hash 通过；
- 25 个已记录的 RuntimeWarning（5 次 `exp` overflow、20 次 scalar multiply overflow），未导致非有限终值或 contract 失败。

## SMP 结果

| case | current mean | historical-compatible mean | geometric ratio | compatible better |
|---|---:|---:|---:|---:|
| E2 | 1.00e7 | 4.61e6 | 1.789 | 4/5 |
| E3 | 2.81e7 | 5.54e6 | 4.559 | 5/5 |
| E4 | 8.61e7 | 9.58e6 | 8.687 | 5/5 |
| E5 | 1.27e8 | 1.04e7 | 11.756 | 5/5 |
| E6 | 1.76e8 | 1.54e7 | 10.663 | 5/5 |

根因已被 paired 对照隔离：当前 SMP 把几乎全部 2.82M FE 用于 stateful visits，尾部只剩 no-op 填充；route 形如 `recovered_stateful_visits_2819xxx ... noop_xx`。历史兼容 schedule 将预算拆为约 1.128M stateful visits、约 0.282M rescue、约 1.41M MMES global polish，并保留 terminal Sep-CMAES。对应冻结 source 为 `artifacts/historical_recovery_fixed_expert_v1/frozen_protocol/sources/smp.py`，当前实现为 `src/arac/actions/recovered.py`。

结论：SMP E2-E6 的系统性回归主要来自 action lifecycle/budget ownership，不是 checkpoint、FE 记账或五 seed 噪声。SMP 应先恢复 rescue + global-polish 生命周期，再做小规模复测。

## GCB 结果

| case | current mean | historical-compatible mean | geometric ratio | compatible better |
|---|---:|---:|---:|---:|
| R1 | 1.69e5 | 1.83e5 | 0.917 | 1/5 |
| R2 | 2.36e5 | 2.07e5 | 1.143 | 4/5 |
| R3 | 2.99e5 | 3.23e5 | 0.927 | 2/5 |
| R4 | 3.52e5 | 3.30e5 | 1.064 | 3/5 |
| R5 | 3.41e5 | 3.40e5 | 0.994 | 2/5 |
| R6 | 3.81e5 | 3.74e5 | 1.019 | 4/5 |

GCB current schedule 在 R1/R3 明显更好，在 R2/R4/R6 略差，R5 基本持平。当前 route 是 three source sweeps + derived-seed native windows；历史兼容 route 是 three-sweep warmup + last-sweep-FE coordination + cold continuation。差异集中在阶段 FE 分配、stage seed namespace 和 continuation 生命周期，不能归结为统一的 GCB 退化。

结论：GCB 需要 schedule ablation 和更大 seed 数量后才能决定是否替换生产；当前没有足够证据把历史兼容 schedule 写回生产。

## 决策

1. SMP：恢复历史兼容的 rescue/global-polish budget ownership，先跑 E2-E6 五 seed paired smoke；通过后再重跑 24x5 screen。
2. GCB：保持当前生产实现冻结，暂不回退；后续只做 schedule attribution，不启动生产 selector/patch 接入。
3. AOR/CTP：本诊断未覆盖，不因 SMP/GCB 结论修改。
4. B1-Final、soft routing、production patch 继续停止，直到恢复 screen 重新通过。

## SMP lifecycle repair and recovery screen (2026-08-23)

实现修复保持 `RecoveredSmpExecutor` 的 v2 `initialize/resume` 状态契约不变，
仅让 `RecoveredActionRegistry` 的 one-shot SMP 路由使用冻结的 `SmpExecutor`
生命周期。该路由显式记录 `historical_compatible_smp_v1_clip_offspring_true`，
并恢复 rescue、MMES global polish 与 terminal Sep-CMAES 尾部；patch、soft
routing、selector 仍关闭。

paired smoke 产物：`artifacts/recovery_smp_lifecycle_smoke_v1/summary.json`

- 25 pairs、50 arms、24 workers、0 failures；
- 25/25 pairs checkpoint 一致，terminal FE 与 receipt/action-result hash 全通过；
- 所有 current route 均包含 `rescue_` 与 `global_polish_`，无 `noop_` 尾部；
- E2-E6 的 current 与 frozen historical final error 逐位一致，smoke gate 通过；
- 该 smoke 只证明生命周期接入和可归因，不授权 superiority 声明。

重新运行的 mapped-action screen 产物：
`artifacts/recovery_first_screen_smp_lifecycle_v2/summary.json`

- 120/120 arms、0 failures；checkpoint、terminal FE、receipt hash 全通过；
- SMP E2-E6 五个 case 均通过 displayed-mean 筛查；E1 仍未通过；
- AOR A4/A6、GCB R1-R6、CTP S6 仍未通过 displayed-mean 筛查；
- screen gate 仍为 false，因此 B1-Final、soft routing、production patch 继续关闭；
- 未通过项仍需按动作族分别做 fixed-action lifecycle/schedule attribution，不能用统一平均数掩盖。

## Zero-relation E1 preservation (2026-08-23)

24x5 screen 暴露了一个新的边界：把 historical-compatible SMP 无条件用于零关系
case 会损失原 E1 的 near-zero 结果。修复改为按 checkpoint relation topology 分流：

- `overlap_relation_count > 0`：historical-compatible SMP lifecycle；
- `overlap_relation_count == 0`：保留原 recovered zero-relation hybrid lifecycle。

E1 preservation 产物：`artifacts/recovery_smp_zero_relation_preservation_v1/summary.json`

- 5/5 seeds、0 failures；
- checkpoint、terminal FE、receipt/action-result hash 全通过；
- 5/5 final error 与旧 screen E1 receipts 逐位一致；
- route 显式标记 `zero_relation_recovered_smp_v1_clip_offspring_false`。

因此 SMP 的条件路由已通过 E1 preservation gate。此前的 24x5 总 screen 仍是
条件路由修复前的结果，不能据此宣称新的整体 B1 状态；若需要更新 B1，必须用
新 manifest 重跑完整 24x5 screen。

## Topology-conditioned recovery screen (2026-08-23)

使用条件路由重新生成的完整 screen：
`artifacts/recovery_first_screen_smp_topology_v3/summary.json`。

- 120/120 arms、0 failures；checkpoint、terminal FE、receipt hash 全通过；
- SMP E1-E6 全部通过 displayed-mean 筛查，E1 恢复到 `7.75e-06` 均值；
- AOR A4/A6、GCB R1-R6、CTP S6 仍未通过；其余 case 通过；
- overall screen gate 仍为 false，`final_recovery_claim_authorized=false`。

结论：SMP 恢复链已闭合，但四动作历史恢复尚未闭合。后续应冻结 SMP 当前条件
路由，分别对 AOR A4/A6、GCB R1-R6、CTP S6 做 fixed-action lifecycle/schedule
attribution；在这些动作恢复前不启动 B1-Final、soft routing 或 production patch。

## GCB schedule attribution (2026-08-23)

使用既有 `R1-R6 × 5 seeds` paired diagnostic receipts 生成无重算归因报告：
`artifacts/gcb_recovery_attribution_v1/report.json`。

- 60 arms、30 pairs；checkpoint、terminal FE、route 解析全部通过；
- current 相对 historical-compatible 平均少约 `235k FE` warmup/source、少约
  `74k FE` coordination，多约 `309k FE` continuation/native；
- 性能方向混合：current 在 R1/R3 更好，historical-compatible 在 R2/R4/R6
  更好，R5 近似持平；
- `uniform_historical_rollback_supported=false`，`production_gcb_change_authorized=false`。

结论：GCB 的回归可以归因到 schedule ownership/lifecycle 差异，但现有证据不支持
统一回退。下一步若继续 GCB，只能做预注册 fresh-seed schedule ablation；生产 GCB
保持冻结。

## AOR A4/A6 and CTP S6 fixed-action attribution (2026-08-23)

协议与可复核产物：

- protocol：`experiments/historical_recovery/aor_ctp_recovery_attribution_protocol_v1.json`；
- runner：`experiments/historical_recovery/aor_ctp_recovery_attribution.py`；
- report：`artifacts/aor_ctp_recovery_attribution_v1/report.json`；
- screen receipts：`recovery_first_screen_smp_topology_v3` 的 A4、A6、S6，5 个
  screen seeds（117、123、129、135、141）；
- CTP matched-tail receipts：S6 seeds 31001、31002、31003，共享 checkpoint、
  operator reservation 和 strict-best ledger。

### AOR：排除 lifecycle/source delta

当前与冻结历史 source 的 SHA-256 完全一致：
`15f7d567351ce660658079b70f3cc00d9feceeab7c962a4779246e1f442e2805`。
因此 A4/A6 的 screen 残差不能归因于 AOR action source 或 lifecycle 改动：

| case | screen mean | historical target | ratio |
|---|---:|---:|---:|
| A4 | 78,257.56 | 78,200 | 1.000736 |
| A6 | 78,123.18 | 78,000 | 1.001579 |

两项都是 5-seed fixed-action screen 的轻微 displayed-mean 回归，receipt、FE、
checkpoint 和 action-result hash 均有效。结论是 `aor_code_change_authorized=false`：
不修改生产 AOR，不把这两个 case 的残差归因给 shared-patch 或 selector。

### CTP：tail 机制有正向 matched-host 证据，但尚未恢复 screen

当前 CTP source 与冻结历史 source 不同；差异集中在正关系路径的 20% MMES tail。
S6 screen 已派发 reserved tail，但 5-seed screen mean 仍为 `5,283.95`，高于
历史目标 `4,180`（ratio `1.264102`，sample std `2,670.07`）。

在 matched checkpoint 的独立 tail ablation 中，candidate tail 在 3/3 seed 胜出，
几何均值 ratio 为 `0.289733`。candidate tail FE 为 `564,048/564,228/564,108`，
baseline tail FE 为 `204/84/312`。这证明 tail 机制在受控 paired host 上具有可观测
增量，但这些 seeds（31001-31003）不是 screen seeds（117-141），不能据此宣称
S6 screen 已恢复，也不能直接授权生产 CTP source 回退或替换。

### Attribution decision

- AOR：source identity 已完成归因；保持当前实现，不做统一回退。
- CTP：保留当前实现和 matched-tail 正向证据；生产修改保持关闭。
- 下一步：如继续 CTP，只运行预注册的 fresh matched S6 screen-seed tail
  attribution；在此之前保持 patch、soft routing、selector 和 B1-Final 关闭。
