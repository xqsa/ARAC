# 论文素材汇总（总纲 #7）

日期：2026-08-21（cut-2 之后）
来源：全部 gate artifact（`artifacts/`）+ CONCLUSION.md + related-work-gaps.md。
定位叙事与引用清单见 `docs/arac-oc-related-work-gaps.md`（FEA 必引、HCC
开放问题接续、AOS extreme-credit lineage）。

## 1. 主张表（每条均有冻结 gate 证据）

| # | 主张 | 证据 | 数字 |
|---|---|---|---|
| C1 | 黑盒重叠发现：高召回、完美精度、固定预算 | soft-RDDSM v3 基线 | AOB 六 case 召回 0.789–1.000、精度 1.000、ov0 假阳性 0、180k FE 逐笔对账 |
| C2 | 共享变量谱随真值重叠度单调、跨 seed 稳定 | Gate 53 cells | 0/18/51/80/119/130 对 ov 0/1/3/5/7/10，4 seed 完全一致，S1=0 |
| C3 | 证据驱动派发对 oracle-上界方法取胜 | Gate 41a/41b | 24 case × 25 seed，对 HCC 20 胜 4 负，几何均值比 0.244 |
| C4 | 共享变量完整目标仲裁在循环语境有效 | Gate 29/30 | proposal-neighborhood 对 proposal-only 36/36 全胜，中位增益 37.9514；对 full-context 非劣 |
| C5 | 存在性：在线组合调度可以严格胜全部 standalone | Gate 50c | R6 0.870×，smp/ctp/aor 三路 material，ON<OFF 因果 |
| C6 | 工程契约可验证 | 全部 gates | strict-best 单调、exact FE、fail-closed、receipt/state hash 链 |

## 2. 分析节素材（调度传奇的降级叙事）

1. **跷跷板的收据级刻画**：v4.0→v5.3 每版修一边破另一边；v5.3 终局
   6/12 on-cell 违反自身审计、S5 median 73×（`oc_phase_aware_gate51c_v5_3`）。
2. **根因：单轴规则 vs 双轴状态**（lit-review §一 + G11 回放）：
   双轴象限放置 S5-CTP=protect 3/3、两轴规则给 CTP 57.1%（正确侧）；
   喂血/释放判别仍开放（间隔代理无判别力）。
3. **象限稳定性与调度价值挂钩**：S5 放置跨 seed 完全稳定（政策清晰）、
   R2/A3 不稳定（谁也赢不了）。
4. **调度信号的三段闭案**：production 规则/固定臂/lagged EMA 统计不可区分
   （G1/G2）；oracle 上限仅 ~1.6%（G3a）；但简单事前特征可提取其中 ~38%
   ——线性 ranker/价值模型在 leave-one-seed-out 下显著超过 production 与
   best-fixed（G8 hit 0.383、G9 0.433，CI 下界均 >0）→ challenger lane。
5. **SACC 式预算敏感**：sense 份额 30% 对 45% 在 S5 −42% 但 R6 +50%，
   固定份额无安全值；减 sense 的 FE 流入 tail 而非算子（G7）。

## 3. 诚实边界（必须写，全部有 gate 编号）

1. **仲裁不可移植**：经代理提案（块扰动）接入派发路径时零接受——
   Gate 53（24/24 cell，ratio 全 ≥1.0，Spearman 0.143 p=0.463）；
   仲裁价值依赖 sense 供给的信息性提案（Gate 29/30 语境）。
2. dense-overlap 的 Phase-I 曾是 fail-closed 边界（历史）。
3. S5 在线调度未过门（Gate 50 的 1.000 恰是调度器不作为时取得）。
4. C_j / G_coupled / G_int 不可调度的阴性结论（Gate 47、fresh 200-cell、
   two-baseline）+ 尺度归一后 lagged EMA 仍无信号（G1）。
5. 仲裁收益在高重叠稀疏域以外未验证；G8/G9 的天花板 ~1.6%。

## 4. 核心图表数据源

- **优势-重叠度曲线**：Gate 53 `confirmation.json` per_case（注意：该 gate
  判定未过，曲线应改用 C2 的发现谱 + C4 的循环语境增益呈现，Gate 53 数据
  进诚实边界）。
- **HCC 对比表**：Gate 41b confirmation（20/4、0.244）。
- **36/36 配对图**：Gate 29/30 artifacts。
- **机制消融矩阵**：`oc_mechanism_ablation_screening`（v4 12/12 审计稳 vs
  v5.3 6/12 违例 + S5 5× vs 73×）。
- **双轴象限热图**：`oc_two_axis_replay_gate` quadrant_matrix。

## 5. 收尾状态

- 主线代码：Phase-I（evidence/）+ 派发（dispatch_policy/actions）+ 统一循环
  （coordination/loop，第三刀范围）；v4–v5.3 调度线已删除（cut-2，a571192，
  tag `v5.3-prealation` 保有全部历史）。
- 可选后续（不阻塞论文）：Gate 53b（信息性提案源仲裁）、v6.0 双轴 + CCFR3
  （Gate 52 预注册已冻结）、G8/G9 challenger lane 晋级 gate。

---

## Shared-Patch 主线素材（2026-08-22 修订版方案，进行中）

### 证据链重划（恢复优先）

论文不再声称“AOB 上 patch 已产生性能收益”。结果按来源拆分：

1. **AOB preservation**：B0-B3 恢复历史四动作、selector parity 和
   patch off/on 的非劣/零税；conforming case 不要求 patch 触发或改善。
2. **Conflicting efficacy**：matched-host M0-M2 在已知 conflicting overlap
   generator 上证明 shared-patch 的可达性、嵌套归因和 fresh-seed 增量。
3. **Production end-to-end**：仅在 B1、B2、B3、M0、M1 全部通过后报告恢复
   基线与 patch-enabled CTP/GSS 的统一 loop 结果。

P1 的 12/12 formal pass 记录为实质不可评估：AOB conforming 构造导致
`arbitration_only`，没有生产 patch 挂载点；该结果不能进入 superiority 表。

### 新协议与止损线

- recovery-first：`experiments/historical_recovery/recovery_first_protocol_v1.json`
  和 `recovery_first_campaign.py`；B1 不通过时停止创新实验。
- 2026-08-23 recovery screen：SMP E2-E6 lifecycle repair passed a 25-pair
  paired smoke with exact final errors and no-op tail removed. The independent
  `24x5` mapped-action screen still failed overall (`E1`, `A4/A6`, `R1-R6`,
  `S6`), so this is recovery evidence only and does not authorize B1-Final.
- Follow-up E1 preservation gate passed 5/5 seeds after topology-conditioned SMP
  dispatch: positive-relation cases use the historical-compatible lifecycle,
  while zero-relation E1 retains the recovered hybrid lifecycle. The subsequent
  full 24x5 rerun is recorded below; the aggregate B1 result remains closed.
- The topology-conditioned `24x5` rerun completed with 120/120 valid arms:
  SMP E1-E6 all pass the displayed-mean screen, while AOR A4/A6, GCB R1-R6,
  and CTP S6 remain failures. Overall B1 remains closed.
- GCB receipt-level attribution found a schedule ownership shift (about 235k
  fewer warmup FE, 74k fewer coordination FE, and 309k more continuation FE in
  current versus historical-compatible), but paired performance is mixed; no
  uniform GCB rollback is supported.
- matched-host：`experiments/overlap_shared_patch_matched_host_gate.py`；M0
  不通过时禁止性能比较，M1 不通过时保留历史基线并放弃当前 patch 版本。
- 软路由只在进入 CTP/GSS 后控制 scope 排序、candidate strength 和 radius
  upper bound；不得改变 Phase-I、外层 selector、动作类型、sense/probe/tail
  预算或 component 停用。
- AOB preservation 不通过时 patch 默认关闭；生产端到端不通过时只保留
  matched-host 机制结论，不宣称生产 superiority。

- 机制：CTP 内部 Stateful Shared-Patch Kernel（owner/consensus/disagreement
  候选、改进度加权共识、conforming 静默 base_radius=eps、逐变量局部
  context hash（写集外坐标，自接受不 reset）、z/u/r 持久状态、u 唯一因果
  通道 = (-u,-priority,j) scope 排序、固定 8 FE 车道从算子预留划出）。
- 嵌套消融 A0-A4（A2 强制：隔离候选增量 vs 状态增量 vs 半径增量）。
- Gate 链：P0（修订版 12/12 契约通过）→ P1（归因，运行中）→ P2/P3
  （fresh-seed 筛查 + selector 首决策 parity）→ P4（24×25，conforming 层
  只要求非劣、conflicting 层要求增量，按构造标签分层）。
- 论文主张顺序按修订方案 §13（1-8）；conforming 自动静默与 conflicting
  局部 trust-region 增量是核心新主张；失败/回退边界如实报告。

### Recovery fixed-action attribution update (2026-08-23)

本轮 AOR/CTP 归因使用冻结 screen receipts、source manifest 和 matched-host
tail ablation，不改变 Phase-I、selector、patch 或 soft-routing 开关。结果必须
拆成“排除代码归因”和“机制有正证据但未恢复”两类，不能合并成历史恢复声明：

| action/case | evidence | interpretation | paper status |
|---|---|---|---|
| AOR A4/A6 | 当前与历史 source byte-identical；screen ratio 1.000736/1.001579 | residual is not an AOR lifecycle/source delta | 报告非归因，不声称回退收益 |
| CTP S6 screen | mean 5,283.95 vs target 4,180；ratio 1.264102 | screen recovery 未完成 | 不报告 AOB/production superiority |
| CTP matched tail | same-checkpoint 3/3 wins；geometric ratio 0.289733 | tail kernel 在受控 host 上有增量证据 | 可作为机制候选证据，不能外推到 screen |

CTP current/historical source hash 不同，差异集中在正关系路径的 MMES tail；但
matched ablation 使用 seeds 31001-31003，而 screen 使用 117-141，因此论文只能写
“matched-host tail efficacy”，不能写“S6 historical recovery”。在 fresh matched
screen-seed attribution 之前，B1-Final、AOB preservation、生产端到端和任何
shared-patch/soft-routing superiority claim 均保持关闭。

### Baseline freeze boundary (2026-08-23)

恢复后的四动作实现已冻结为 `arac-recovered-baseline-20260823-v1`，冻结协议和
21 个文件哈希见 `experiments/historical_recovery/recovered_baseline_freeze_protocol_v1.json`。
论文实验必须把该 baseline 作为固定 anchor；后续 shared-variable upgrade 只能
作为新候选报告，并通过 U0 kernel contract、U1 matched-host reachability、U2
nested attribution、U3 AOB preservation 和 U4 production parity 后才可讨论提升。
冻结不等于 B1-Final 全部历史均值已通过，残余 AOR/GCB/CTP attribution 边界仍按
本文件前述证据报告。
