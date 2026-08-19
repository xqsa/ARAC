# Gate 51c 协议：v5.1 fresh-seed 确认门

日期：2026-08-18（夜，用户授权自主推进）
更新：2026-08-19（receipt-only 诊断与 anytime/AUC 字段）
状态：预注册（运行前冻结）
前置：Gate 51b（v5.1）单 seed 通过（A3 1.004✓ / R6 1.028✓ /
R2 1.132✗ / S5 2.89✗；ON<OFF 全域；对 HCC 3 胜 1 负）。

## 1. 研究问题

v5.1 的全部数字来自设计 seed（20260845）。本门回答：

> (a) 非劣结论（A3/R6）是稳定的还是 seed 运气？
> (b) R2 的 1.13× 中有多少是方差？
> (c) ON<OFF 的全域因果是否经得起重复？

## 2. 设计

- Fresh seeds：**20260901 / 20260902 / 20260903**（未参与任何
  设计决策、校准或调参；校准只用了 20260845）。
- 每 (case, seed)：全新 Phase-I checkpoint（该 seed 的发现 +
  MMES 填充 180k，落盘缓存）+ 四个 standalone（150k 分段轨迹）
  + OC handoff ON/OFF（v5.1，同冻结定标表）。
- action_seed = Phase-I seed（镜像主管线约定）。
- 参照：**同 seed 的四个 standalone 的最优者**（不跨 seed 比较，
  无 Gate 50 逐位要求——不同 seed 无从对账）；HCC 参照仅作
  信息报告。
- 产出：`artifacts/oc_phase_aware_gate51c_v5_1/`（cells/ +
  confirmation.json），逐 cell manifest 戳 + 复用守卫。每个 cell 同时
  保存 Phase-I 起点、分段 strict-best 轨迹、固定总 FE 网格
  `600k/1M/2M/3M` 的 anytime 值和 Phase-II 归一化 `log10(error)` AUC。
  对历史 cell 可运行 `python -m experiments.oc_phase_aware_gate51c_anytime`，
  生成独立的 `anytime_auc.json`，不覆盖原始收据。

## 3. 判定（预注册，v4 计划 51c 条款）

1. 每 case **中位数** ratio（ON / 同 seed 最优 standalone）≤ 1.05；
2. 每 case **最差 seed** ratio ≤ 1.10；
3. ≥2 case 中位数严格胜（≤ 0.98×）；
4. 组合胜例（中位数严格胜的 case）在 ≥2 fresh seed 上 ON < OFF；
5. 全部 cell 协议审计通过（终端精确、13 项调度审计、FE 对账）。

gate_passed = 1∧2∧4∧5；条款 3 单独报告（严格胜不足但非劣稳健
时，如实记为"非劣确认、严格胜未达"）。

当前已落盘的 72-cell 结果满足零失败和全审计通过，但 Gate 51c 未过：
A3 为稳定非劣，R2 有一个 fresh-seed 严格胜且三 seed handoff 均优于
OFF，R6 边缘，S5 存在一个 protected-runway 预算税灾难 seed。该结论
不应写成“v5.1 已全面通过”，而应写成“协议通过、性能门未通过，机制
归因已定位”。

## 4. 失败条款

任一条款失败 → 按收据归因 seed 敏感机制，不针对单 seed 写特判；
v5.2 杠杆（R2 horizon 梯档 450k、S5 plateau 释放阈值）已预注册，
仅在 fresh 证据指向时启用。本门结果无论如何不回调 v5.1 机制。

## 5. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.oc_phase_aware_gate51c --workers 8 --pin-p-cores
```
