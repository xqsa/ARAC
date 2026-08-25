# ARAC-OC Recovered Baseline Freeze

日期：2026-08-23  
执行者：Codex  
状态：`FROZEN`

## 冻结对象

当前恢复后的四动作执行锚点冻结为：

```text
arac-recovered-baseline-20260823-v1
```

唯一真值协议是：

`experiments/historical_recovery/recovered_baseline_freeze_protocol_v1.json`

校验入口是：

```text
$env:PYTHONPATH='.;src'; python -m experiments.historical_recovery.verify_recovered_baseline_freeze verify
```

冻结内容包括：

- `RecoveredActionRegistry` 及其 AOR/SMP/GCB/CTP action implementation；
- 1000 维、Phase-I `180,000 FE`、terminal `3,000,000 FE`、native threads=1；
- 正关系 SMP 的 historical-compatible lifecycle；
- 零关系 SMP 的 recovered hybrid lifecycle；
- CTP 正关系 MMES tail、AOR 当前实现和当前 GCB schedule；
- recovery screen、SMP lifecycle smoke、E1 preservation、GCB/AOR/CTP attribution
  的已验证产物哈希。

## 冻结时的开关

```text
patch = off
soft routing = off
new selector = off
```

这些开关是生产默认值。冻结不把当前 screen 的残余误写成新的 superiority
证据：AOR A4/A6、GCB R1-R6、CTP S6 的历史均值差异仍按各自 attribution 报告
记录；冻结的含义是“恢复后的稳定 anchor”，不是宣称 B1-Final 已通过。

## 不可变规则

1. 不直接修改 freeze manifest 中列出的 source、protocol 和 evidence artifacts。
2. 不在冻结 registry 中接入 patch、soft routing 或新 selector。
3. 不覆盖现有 artifacts；新实验使用新的版本目录和新的 seed manifest。
4. 任何升级候选先通过 baseline verifier，再运行自己的 U0-U4 gate。
5. 候选失败时保留本冻结锚点，不能用候选结果改写历史恢复结论。

## 已通过的冻结前检查

- SMP lifecycle paired smoke：25 pairs / 50 arms，exact final error；
- E1 zero-relation preservation：5/5 exact match；
- topology-conditioned screen：120/120 arms，FE/checkpoint/receipt contract 全通过；
- attribution reports：GCB 30 pairs，AOR/CTP source and matched-tail evidence；
- 当前相关回归测试与 freeze verifier 均通过。

## 解冻条件

只有新升级候选通过 U0-U4，并且用户明确选择将其提升为新 baseline，才允许
生成 `arac-recovered-baseline-<date>-v2`。在此之前，任何代码变更都应视为
upgrade candidate，不得修改本冻结点。
