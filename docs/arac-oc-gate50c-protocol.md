# Gate 50c 协议：3M 六臂配对——ARAC-OC 组合调度 vs 四 standalone + handoff 消融

日期：2026-08-16（夜间自主执行）
状态：预注册

## 1. 研究问题

handoff 机制闭环（600k 冒烟+消融通过）后，3M 全量预算下的性能判定：

> (a) OC(handoff ON) 是否在全部 case 不劣于最优 standalone（×1.05）？
> (b) 是否至少一个 case 严格优于全部 standalone（超越单臂直接证据）？
> (c) 严格优于的 case 中，胜利能否归因于组合接力（≥2 episode
>     material global gain + ON 优于 OFF）？

## 2. 设计

- Case：A3 / R2 / R6 / S5；Phase-I = v3 发现 + MMES 填充 180k
  （seed 20260845，checkpoint 哈希与 Gate 50 standalone 收据逐位
  一致，已核验）；action_seed = 20260845。
- 六臂（同 checkpoint、独立 ledger、精确 3M）：
  - CTP / GSS(gcb) / SMP / AOR standalone —— **复用 Gate 50 收据**
    （`oc_action_episode_gate50/cells/`，哈希一致）；
  - OC handoff ON —— 候选方法；
  - OC handoff OFF —— 因果消融。
- OC 参数（与 600k 同规则，比例固定）：segment 300k、
  probe_share 0.10（3M 探针窗 ~282k/episode）、probe_min 20k、
  全局 materiality 阈 log(1.01)、max 6 切换。

## 3. 判定（三层，预注册）

**协议层**（全部臂）：terminal 3M 精确；strict-best 单调；
FE 对账（phase1 + sensing + funded == terminal）；OC 臂 handoff
链/snapshot hash 完整（含 refusal 语义合法）；收据哈希重算一致。

**性能层**：
1. `not_worse_than_best_all`：4/4 `OC_ON ≤ 1.05 × best standalone`；
2. `strictly_better_exists`：≥1 case `OC_ON < best standalone`。

**互补层**（在严格优于的 case 上）：
3. `two_episodes_material`：≥2 个不同 episode 在 handoff 后产生
   material global gain（探针或 exploitation 段）；
4. `on_beats_off`：该 case `OC_ON < OC_OFF`。

判定：性能层两条 + 互补层两条全部成立 → gate_passed；仅性能层
成立 → "单臂级非回归，组合归因不足"如实记录；性能层 1 失败 →
按收据归因（探针税/误配），不回调 handoff。

## 4. R2 特别条款（用户指示）

若 R2 上 ON 明显差于 OFF 且 AOR standalone 最优：记录为
"晚熟动作的时间尺度问题"，未来方向为 AOR correction tail 预留或
多时间尺度收益估计；**不修改 handoff 机制**。

## 5. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.oc_action_episode_gate50c --workers 8
```

产出：`artifacts/oc_action_episode_gate50c/`（cells/ + confirmation.json）。
