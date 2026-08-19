# Gate 41 协议：AOB-24 升级无回归（证据派发动作选择）

日期：2026-08-15
状态：41a（离线校准与验证）已完成并通过；41b（在线 25-seed 正式表）待运行

## 1. 需求与背景

用户要求：升级后的方法在 AOB-24 上**不弱于历史 ARAC 列**（对 HCC-ES
18/1/5 的那列数字）。前置事实：

- 当前结构路由在 E/S 族上回归（08-11 恢复 campaign：21/24 差于显示值），
  回归源是把稠密关系图全部路由到 GCB；
- `RecoveredActionRegistry` 四动作矩阵（24 case × 7 seed × 4 arm，共享
  Phase-I checkpoint）显示按 case 选 oracle 动作几乎全面达到或超过历史列；
- AOB 上重叠协调器 fail-closed（Gate 40 前审计），升级方法的 AOB 行为
  完全由动作派发层决定。

## 2. 派发规则（预注册，两特征可解释规则）

```text
tail_log10_gain < 0.10                     -> aor
tail_log10_gain >= 0.50 且 relation密度<=0.05 -> smp   （E1/S1：各族无重叠实例）
tail_log10_gain >= 0.50（稠密关系）          -> ctp
其余                                        -> gcb
```

校准证据（`artifacts/overlap_action_dispatch_gate41a/offline.json`）：
`tail_log10_gain` 逐 seed 完全分离四族（A 0.000–0.010 / R 0.262–0.315 /
S 0.619–0.786 / E 0.754–0.895），阈值位于空隙内，留一 case 不改变判定。
规则只用身份盲 Phase-I 特征，不读函数编号/族标签。

## 3. 41a 离线验证结果（已通过，零新 FE）

在 four-arm 矩阵（7 seed）上模拟规则实现误差：

- 24/24 case 相对历史列 ratio ≤ 1.005（最差 R1 +0.5%）；
- 17 case 严格更好；E1 8.1e-6（历史 5.69e5）、S1 8.9e-14（历史 1.04）；
- 对 HCC-ES：21 胜 / 3 负（历史 18/1/5 → S4/E1/S1 翻胜）；
- 逐 seed 规则-oracle 一致率 95.8%。

## 4. 41b 在线正式运行（预注册）

- 每 (case, seed)：加载冻结 Phase-I checkpoint（25 seed 全存在），
  在线应用派发规则，经 `RecoveredActionRegistry` 执行唯一动作至 3M FE；
  four-arm 已有同 (case, seed, action) 收据的直接复用；
- 判定（容差同 41a）：每 case 均值 ≤ 历史 × 1.10；几何均值比 ≤ 1.0；
  对 HCC-ES 胜数 ≥ 18；协议检查（terminal 精确、契约、派发收据完整）。

## 5. 声明边界

- 派发规则在同一 AOB 函数分布内校准（与历史 RF v3 同边界）；不宣称
  跨函数族外推；
- 25-seed 表与历史列同 seed 集（117–141），配对可比。

## 6. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.overlap_action_dispatch_gate41_online --workers 4
```

产出：`artifacts/overlap_action_dispatch_gate41_online/confirmation.json`。
