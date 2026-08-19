# Gate 39 协议：相对 hub 派发协调器的多 seed 配对确认

日期：2026-08-15
状态：已执行完毕。判定：gate_passed = false（4 项筛查过 3 项）；
失败项归因见 §6——结论对主线实际是有利的。

## 1. 研究问题

Gate 37/38 的全部对照都依赖 seed 20260829 的冻结基线。本 gate 在新 seed
上做完全配对的确认：

> 在未见过的 seed 上（Phase-I checkpoint、proposal 基线、固定 CTP 臂都
> 重新配对计算），相对 hub 派发协调器是否仍然（a）不劣于 proposal
> 基线，（b）不劣于固定 coordinate CTP，（c）在 chain/ov=3 上产生正收益、
> 在 star 上不回归？

## 2. 设计

- 配置：6 个 conflicting 配置（chain/star/random × overlap 3/6）。
  conforming 配置不含：派发几乎不触发，协调器与 proposal 基线等价
  （Gate 37 已结构性证明），不属于本 gate 的科学问题。
- seed：20260830、20260831（从未用于任何 gate、校准或基线）。
- 每 cell：跑一次 Phase-I pilot；从同一 pilot 跑三个 arm——
  `gcb_coordinated`（v2 相对 hub 配置）、`proposal_neighborhood`、
  `persistent_ctp`（固定 coordinate CTP，Gate 36 机制）。三 arm 共享
  checkpoint 与总预算 3M FE，消耗独立 ledger。
- 共 12 cell × 3 arm = 36 次 2.82M 相位 + 12 次 Phase-I。

## 3. 判定（预注册，容差 1e-9）

协议检查（每 arm）：terminal 精确 3M、strict-best、（coordinator arm）
envelope 不侵占与消耗=预留；每 cell 三 arm 共享同一 checkpoint hash。

筛查判据：

1. `star_no_regression_all`：4 个 star cell 实例（2 seed × ov3/ov6）对
   配对 proposal 基线 gain ≥ −1e-9；
2. `chain_ov3_positive_both_seeds`：2 个 chain/ov=3 实例 gain > 1e-9；
3. `win_or_tie_vs_proposal_ge_0_75`：12 个实例中 ≥ 9 个 gain ≥ −1e-9；
4. `not_worse_than_persistent_ctp_all`：12 个实例 coordinator 终值 ≤
   配对 persistent_ctp 终值 + 1e-9。

## 4. 与 Gate 38 的关系

Gate 38 通过不是本 gate 的前提：若 Gate 38 在 seed 20260829 上失败而本
gate 通过，说明单 seed 冻结批上的失败是 seed 特异的，证据反而更强；
若两者都失败，则相对 hub 信号方案被否定，转入 counted probe 路线
（Gate 40）。两种走向都记录，不挑选。

## 5. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.overlap_gcb_multiseed_gate39_screening --workers 4
```

产出：`artifacts/overlap_gcb_multiseed_gate39/confirmation_fresh.json`。

## 6. 结果与归因（2026-08-15 执行）

协议检查 6/6 通过。筛查 3/4 通过：

- `star_no_regression_all`：通过（4 实例全部数值平局）；
- `not_worse_than_persistent_ctp_all`：通过（12 实例 0 负，其中 2 个小胜
  +0.019 / +5.99）；
- `win_or_tie_vs_proposal_ge_0_75`：通过（实际 12/12，2 个正收益
  +31.41 / +7.57）；
- `chain_ov3_positive_both_seeds`：**失败**（seed 20260831 = +31.41，
  seed 20260830 = 数值平局）。

失败归因（收据与配对臂证据）：seed 20260830 的 chain/ov=3 上，**固定
CTP 臂与 proposal 基线也完全相同**（三 arm 终值一致）——即坐标 CTP 算子
在该 seed 实例上净收益潜力为零，任何派发器都无法提取不存在的收益。该
判据预注册时隐含"算子在每个 seed 上都有正潜力"的假设，此假设不成立。
在潜力存在的实例上（20260829 的 +3.57、20260831 的 +31.41），协调器
全部捕获（与固定 CTP 精确持平）。

事后分析（声明为 post-hoc，不作为 gate 通过）："coordinator ≥ 固定 CTP
于全部实例，且在固定 CTP 为正处为正"在 12/12 实例上成立。若需正式确认，
该修正判据应进入下一个预注册 gate，而非回调本 gate。

附带发现：固定 CTP 的 star 回归模式在新 seed 上复现（star/ov=6/20260831
固定臂比协调器差 +5.99），协调器再次规避。

三 seed 汇总（20260829 + 两个新 seed）：协调器对 proposal 基线 24/24
win-or-tie（3 正收益），对固定 CTP 24/24 不劣（4 个严格更优）。
