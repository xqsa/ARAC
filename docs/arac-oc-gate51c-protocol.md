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

## 6. v5.2 杠杆最终定义（2026-08-19 用户裁决，跑批前冻结）

诊断（`artifacts/oc_phase_aware_gate51c_v5_1/mechanism_diagnostics.md`）：
三个 fresh seed 的 AOR 运行时恰好停在 450k 穿越线，但仅 20260902 在
450k 产生 material gain（0.0511），另两 seed 为零且后续主要收益来自
GSS。据此本节取代 §4 中"R2 horizon 梯档 450k"的原表述——不做无条件
顶档扩展，v5.2 定稿为：

1. **有界验证窗（杠杆 1）**：adaptive lock 后的 protected runway 与
   material leader 续跑窗均以 w1 为上界；plateau 在 w1 FE 后释放，
   released/plateau_release 入收据与审计；
2. **material horizon promotion（杠杆 2）**：horizon 保留段产生全局
   material gain 的 episode **立即**获得一个 2×w1 显现后验证 exploit
   （reservation_kind=horizon_promotion、exploitation 账本、每 episode
   每 run 一次）。material 验证窗写入首个 exploit 速率样本、按常规
   排名竞争 leadership；平坦验证窗置 released——损失上界 2×w1。根因
   修复：leader 选举只认 exploit_history，challenger 通道的发现原本
   无法进入利用阶段。

复判入口：`experiments/oc_phase_aware_gate51c_v5_2.py`（版本隔离目录
`oc_phase_aware_gate51c_v5_2`；OC 臂重跑，standalone/Phase-I 复用冻结
v5.1 产物，锚 = v5.1 confirmation manifest + checkpoint hash；冻结 cell
的 anytime 轨迹从 segments 确定性重算并经 anytime_auc.json 12/12
逐位验证）。
