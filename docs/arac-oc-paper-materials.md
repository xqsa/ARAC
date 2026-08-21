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
