# 二进制 LSGO 聚焦 3-Seed Pilot 设计

日期：2026-07-13
执行者：Codex
状态：已实现并验证；晋级门未通过

## 1. 目标与边界

新增一个只针对二进制重叠 LSGO 测试集的独立实验入口
`experiments/exp_010_binary_lsgo_focused_3seed/`，验证首次单 seed pilot 中
`F08`、`F15` 的 ARAC isolate 动作是否具有跨 seed 的稳定性，并用同族邻近 case
作为诊断对照。

本实验不修改二进制 benchmark 生成器、`BinaryLsgoBackend`、ARAC policy 或 HCC/AOB
连续优化链路。`exp_009` 保持原有 18-case 单 seed 协议不变；`exp_010` 只调用现有
backend 和 policy，并生成自己的聚合审计产物。

## 2. 固定协议

固定 case 顺序为：

```text
BLSGO-F07, BLSGO-F08, BLSGO-F09, BLSGO-F14, BLSGO-F15
```

固定 optimizer seeds 为：

```text
20260713, 20260714, 20260715
```

每个 case/seed 运行三条 lane：

- `native_baseline`
- `arac_policy`
- `shuffled_evidence_negative_control`

因此 canonical 协议包含 `5 × 3 × 3 = 45` 次独立 backend execution。每次 execution
使用 `total_fes=2000` 和 `phase_one_fraction=0.20`，Phase I 为 400 FE，Phase II 为
1600 FE。三条 lane 对同一个 case/seed 使用同一个生成问题、初始向量和 Phase-I 状态。

CLI 只允许改变输出目录和测试用 `--total-fes`；当预算不是 2000 时，实验仍可运行和测试，
但 `promotion_gate.json` 必须标记为 `canonical_budget_failed`，不能产生晋级通过。

## 3. 实验入口与数据流

新增 `experiments/exp_010_binary_lsgo_focused_3seed/run.py`，直接调用：

```text
standard_binary_lsgo_specs()
    -> 固定 5 个 problem spec
    -> generate_binary_lsgo()
    -> run_binary_lsgo() × 3 seeds × 3 lanes
    -> run_results.csv
    -> case_summary.csv + promotion_gate.json
```

实验入口不复制优化逻辑，不读取历史结果，不把 final objective 或相对收益传回
`EvidenceProfile` 或 `decide_action()`。所有收益、均值、中位数和灾难性损失只在 execution
完成后用于 offline aggregation。

## 4. 输出契约

### 4.1 `run_results.csv`

每次 execution 一行，共 45 行。必须包含：

- run/case/seed/lane 标识和输入 hash；
- Phase-I objective、final objective、selected action、optimizer-consumed；
- evidence 字段、backend semantic diff 和 action trace 摘要；
- `phase_i_fe`、`phase_ii_fe`、`total_fe`、`same_budget_violation`；
- 相对 native baseline 的 offline gain、utility label 和 catastrophic flag；
- runtime forbidden-field audit 和 `claim_allowed=0`。

### 4.2 `case_summary.csv`

每个 case 一行，共 5 行。只聚合 `arac_policy` 相对 `native_baseline` 的结果，同时保留
对照诊断字段：

- 3 个 seed 中 action consumed 的次数；
- ARAC 结果的 mean、median、minimum relative gain；
- action-consumed seed 的 median relative gain；
- catastrophic loss 次数；
- shuffled negative-control evidence 改变次数；
- 是否所有 lane 同预算和 runtime 审计通过。

F07、F09、F14 的 fallback/action 分布只作为诊断字段，不因一次触发或不触发直接改变
晋级门定义。

### 4.3 `promotion_gate.json`

写入每条 gate 的 `passed`、观测值、阈值和失败原因，并写入总状态：

```json
{
  "canonical_budget": {"passed": true},
  "target_action_frequency": {"passed": true},
  "target_action_median_gain": {"passed": true},
  "no_catastrophic_loss": {"passed": true},
  "same_budget": {"passed": true},
  "runtime_boundary": {"passed": true},
  "negative_control": {"passed": true},
  "overall_pass": false
}
```

## 5. 晋级门

canonical 预算下，以下条件全部满足才允许 `overall_pass=true`：

1. F08 和 F15 各自在至少 3 个 seed 中的 2 个实际消费非 fallback ARAC action；
2. F08、F15 各自的 action-consumed seed 相对收益中位数均不小于 0；
3. 15 条 `arac_policy` 结果中没有相对 native gain `<= -0.20` 的 catastrophic loss；
4. 45 条 execution 的总 FE 均为 2000，`same_budget_violation=0`；
5. 45 条 runtime evidence 均不含 `FORBIDDEN_RUNTIME_FIELDS`；
6. 15 条 negative-control execution 均有对应记录、`claim_allowed=0`，且每个 case/seed
   至少有一个 identity-sensitive evidence 字段（优先 `priority_spread`）与 policy lane
   不同。

这些门只决定是否值得进入更大规模实验，不构成最终性能结论。即使通过，也只能称为
“focused 3-seed pilot gate passed”。

## 6. 错误处理

- 固定 case 不存在、seed 非整数、预算小于 2 或输出目录不可写时立即失败；
- 某 case/seed/lane 的初始向量 hash 或 Phase-I objective 不一致时立即失败；
- 某 execution 抛出 backend 异常时，实验命令失败，不生成伪造的 summary 通过状态；
- promotion gate 缺少目标 case、action seed 为空或 canonical budget 不匹配时显式失败；
- 不捕获宽泛异常后继续运行，不写入历史 final outcome 或 paper baseline。

## 7. 测试策略

新增 `tests/test_exp_010_binary_lsgo_focused_3seed.py`，覆盖：

1. 固定 case/seed/lane 数量为 45；
2. 测试预算下的 CSV 行数、字段和 lane 初始状态一致；
3. 相同输入运行两次的 `run_results.csv`、`case_summary.csv`、`promotion_gate.json` 和
   `manifest.json` 字节完全一致；
4. summary 的 mean/median/minimum 与逐行结果一致；
5. action frequency、action median gain、catastrophic、same-budget、runtime boundary
   和 negative-control gate 分别可被构造数据触发通过/失败；
6. 非 2000-FE 运行不能产生 `overall_pass=true`；
7. 完整现有测试集不回归。

## 8. 运行命令与 claim level

canonical 命令：

```powershell
$env:PYTHONPATH='src'
& 'E:\ARAC\.venv\Scripts\python.exe' -m experiments.exp_010_binary_lsgo_focused_3seed.run `
  --output-dir results/exp_010_binary_lsgo_focused_3seed `
  --total-fes 2000
```

claim level 固定为 focused 3-seed pilot。生成的 `results/` 只作为可重生成离线产物，
不提交 Git；只有代码、测试、文档和 manifest schema 进入版本控制。

## 9. Canonical 验证结果

2026-07-13 使用固定协议完成 45 次 execution：

- `run_results.csv`：45 行；
- `case_summary.csv`：5 行；
- 所有 execution 均为 2000 FE，预算违规为 0；
- runtime forbidden-field 记录为 0；
- catastrophic loss 为 0；
- 15 条 negative control 均存在、禁止用于 claim，且 evidence 均发生预期变化；
- F08 action consumed 为 0/3，F15 为 1/3；
- `target_action_frequency` 和 `target_action_median_gain` 未通过；
- `overall_pass=false`。

因此本轮只能得出“动作触发跨 seed 不稳定，暂不进入更大规模实验”的结论。该失败结果
保留原阈值和固定 seed，不做事后调参。
