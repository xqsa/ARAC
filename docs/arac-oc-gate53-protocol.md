# Gate 53 协议（仲裁接进派发路径）——完整预注册

日期：2026-08-21
上游：`docs/arac-oc-completion-plan.md` §5 骨架 → 本文档定稿。
本文档在 Gate 53 任何运行之前冻结；运行后不许回调。

## 0. 相对 §5 骨架的两处预注册偏离（先声明）

- **D1 谱系家族**：骨架写 F1–F6；F 家族无冻结 checkpoint/派发收据。重叠度由
  index 决定（vendor 元数据：F{i}-info.txt，1–6 → ov 0/1/3/5/7/10），四家族
  A/E/R/S 共享同一 index 语义。取 **S1–S6**（骨架自身的高重叠引用即 S 家族）。
- **D2 配对谱系**：仲裁需要逐变量多 owner 结构，仅 soft-RDDSM 谱系具备
  （recovered 谱系块不相交、无共享变量集，仲裁在该谱系上无定义）。因此两臂
  均执行**同一 seed 的 soft-RDDSM v3 Phase-I**（gate48b 冻结配方：
  `discover_hierarchical_soft` + MMES 烧到精确 180k），动作名取自**冻结的
  gate41 派发收据**（同 case/seed 的 dispatch 决策，保证派发规则不变）。
  历史 checkpoint 执行结果仅作 context 报告，不进配对判定。
  配对内唯一差异 = 仲裁 ledger 包装，隔离即仲裁净效应。

## 1. 网格与预算

- cases：S1–S6（ov 0/1/3/5/7/10）；seeds：{117, 118, 119, 120}（4 seed；
  相对冻结四臂收据全部为 fresh 执行）。
- 每 (case, seed)：Phase-I 180k（两臂共享）+ 每臂精确 2,820,000 Phase-II，
  终点精确 3,000,000；fail-closed（任何契约失败即该 cell 无效，不重试）。

## 2. 两臂定义

- **A（裸派发）**：`execute_phase2_action(action, checkpoint, problem, ledger)`
  with plain `EvaluationLedger`。
- **B（派发 + 仲裁）**：同 action、同 checkpoint、同 seed，ledger 换为
  `ArbitrationLedger`（`EvaluationLedger` 子类；仅重载 `evaluate`，在周期边界
  插入仲裁；其余全透传）。

## 3. 仲裁周期（预注册参数）

- `cycle_fe = 150_000`（约 18 个周期）；周期索引 k 的 rng =
  `default_rng(seed ^ 0x53C0 ^ (7919·k))`。
- 每周期（剩余预算 ≥ 8 才触发；否则记 skipped）：
  1. 块提案：x_g = 当前 strict-best incumbent 在 group g 坐标上加
     N(0, σ)，σ = 0.05·(upper−lower)，逐坐标独立，clip 到界；
  2. 共享变量 = owners ≥ 2 的变量；**无共享变量（S1/ov0）→ 零 FE 静默**；
  3. 三候选（一次 batch、精确 3 FE，经真实完整目标）：
     - owner-competition：每个共享变量取 |Δincumbent| 最大的 owner 值；
     - weighted-consensus：owner 值按成员置信度 q 加权平均；
     - weighted-median：owner 值的 q 加权 50% 分位；
     非共享坐标一律取 incumbent；
  4. strict-best：接受与否由 ledger 自身裁决；收据记录三候选误差与接受位。
- 全程仲裁 FE 上限 ≈ 18×3 = 54 FE（对 2.82M 的税 < 0.002%）。

## 4. 判定（全部同时满足才通过；预注册后不许改）

1. **无税条款（S1）**：|mean log-ratio(B/A)| ≤ ln(1.02)，且 S1 的仲裁 FE = 0；
2. **高重叠严格更优**：S4–S6 中至少 1 case 的几何均值比 (B/A) < 1.0；
3. **全局非劣**：每个 case 的几何均值比 ≤ 1.05；
4. **核心曲线**：per-case median log-ratio 对 ov 的 Spearman ≥ 0，
   单侧 permutation 检验 p < 0.05（10,000 次置换，seed=20260822）；
5. **契约审计**：两臂 terminal = 3,000,000 精确；strict-best 单调；
   仲裁收据完整（周期数、候选误差、接受位、skipped 原因）；
   每 cell 的 checkpoint_hash、结构共享数、动作名与冻结收据一致。

## 5. 失败语义

任一条失败 → 仲裁按 case 关闭（回退纯派发），失败项如实写入论文边界；
不允许 gate 内调参重试。仲裁实现仅存在于实验脚本（零 src 改动），
production 全程不变。

## 6. 产物

`artifacts/oc_gate53_arbitration_wire/cells/{case}_{seed}.json`（两臂同文件）+
`confirmation.json`（判定 + 五条 check + per-case 曲线数据）。

## 7. 修订 v2（2026-08-21，campaign 启动前登记；v1 语义对齐 G11-v2 先例）

- **触发**：v1 的探索性冒烟（S5/117，未产生任何 campaign cell）证明"窗口内
  插入仲裁 FE"与 recovered 动作的 `consumed == aligned` 契约**结构性冲突**
  （`run_persistent_blocks`/`run_sequential_blocks` 把窗口内任何第三方 FE 记为
  漂移并 fail-closed）——这正是本项目预算车道纪律的自我执行。
- **v2 接线**：仲裁改在**相位边界**开火——
  - ctp cell（S2–S6）：arm B 用与 `CtpExecutor.execute` 完全相同的公共 helper
    和预算数学复现三相位（coverage → sequential polish → mmes terminal），
    仅在相位之间 drain 到期仲裁周期；
  - smp cell（S1）：arm B ≡ arm A（smp 无可插入相位；S1 本就是预期静默的
    无税 case）。S1 上 discovery 的共享变量计数作为 Phase-I 精度观测报告。
- **新增 D3**：两臂 checkpoint 的 `relations=()`（软 RDDSM evidence 的
  region-relation 到 block 索引映射不在本 gate 范围）→ ctp 走无关系路由。
  配对内部一致（两臂同路由同块集），绝对值与历史收据不可比（D2 已声明）。
- 判定五条不变；S1 无税条款的"仲裁 FE=0"由 smp 同臂保证，另加报告项：
  S1 discovery 共享变量计数（若 >0 即 Phase-I 精度信号，单独披露）。

## 8. 修订 v3（D5，2026-08-21 晚，campaign 第二次启动前登记）

- **触发**：S1/smp 臂 A 在新 Phase-I 谱系上首跑即遇 AOB 目标函数在有限界内点
  溢出为 inf/nan，ledger 有限性契约 fail-closed（历史 S 家族收据全部来自
  four_arm_reuse 另一条管线，该路径从未在线跑过）。
- **D5**：gate 层 objective 净化包装——非有限返回值映射为 1e300 哨兵
  （语义：溢出点比任何有限值都差，保持排序与 strict-best 有限性），
  对 Phase-I、臂 A、臂 B **一致**适用；真实契约违反（形状错误、输入非有限）
  仍在 ledger 内 fail-closed。零 src 改动。
- 同时给 campaign 加 per-cell fail-soft：单 cell 异常记 `protocol_invalid`
  并继续，其余 cell 照常完成与判定。


