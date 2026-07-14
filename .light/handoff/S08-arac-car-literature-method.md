---
session_no: S08
suggested_title: "[ARAC] S08 CAR 文献核验与最终方法冻结"
parent_session: S07
project: arac
date: 2026-07-15
author: Codex
---

## 结论

当前 v36 不是继续加阈值即可修好的实现。held-out 结果中 65 个
case-seed pair 只有 6 个真正改变，且包含 1 个 catastrophic loss；它没有
提供可识别的 long-horizon action utility。

经过 2024-07-15 至 2026-07-15 的 Crossref、Semantic Scholar batch、DBLP
和 arXiv 核验，原始“分组/重叠证据 -> runtime action”主张属于已有动态 CC、
UCB/resource allocation、learned CC、probing trajectory 和 racing 组合的
增量扩展。可守住的最小方法主张是：

> ARAC-CAR 在重叠 CC 的完整 optimizer checkpoint 上，把 graph-conditioned
> backend action 与 native fallback 做 equal-FE/common-RNG 配对探针；所有
> probe/discard FE 计入同一轨迹；只有 fallback-relative lower-tail 风险门
> 通过才原子提交，否则 abstain。

这不是“首个 probing/CRN/LCB/bandit 方法”的声明，文献审计把它标为
incremental extension with a potentially distinctive protocol。

## 文献边界

- OCC/GECCO 2024 和 dynamic/variable-importance CC 已覆盖重叠结构驱动的
  分解或共享变量处理。
- UCB-CC 2025、TEVC 2025 resource allocation、TSC 2025 bandit CC 已覆盖
  贡献/启发式驱动的子问题或资源调度。
- LCC 2025、LH-CC 2026、LightGBM/GNN selection 和 greedy restart 已覆盖
  状态/特征驱动的动态选择。
- arXiv 2501.11414 已覆盖 probing trajectories；arXiv 2604.05792 和
  2603.08493 已覆盖 CRN/racing/不确定性排序。
- 因此 CAR 的五个必需差异是：overlap backend intervention、identical full
  checkpoint、candidate-native-fallback paired delta、single-trajectory
  FE ledger、fallback-relative downside gate + abstain。

证据与 DOI 表见 `docs/literature_review.md`。Semantic Scholar 单篇重试有
HTTP 429，但 batch DOI 查询 HTTP 200；OpenAlex 本轮不可用，不能声称已覆盖。

## 冻结的 CAR 设计

1. Phase-I 先完成 canonical stage 和至少两个完整 overlap-component sweep。
2. W/R/S 三通道固定顺序：writeback、resource allocation、search start；先
   只实现 W。
3. W candidate 复用 v31 proposal，固定 alpha=0.20 和 v33 norm guard；至少
   两个 sweep 的 action family 必须一致，否则 abstain。
4. 在 component barrier 做 K=3 sequential paired probes。每个 candidate 与
   fallback 分支 equal FE、counter-based CRN、独立 evaluator record；前两
   个 pair 主轨迹沿 fallback 前进，最后一个 pair 由 gate 决定采用 candidate
   或 fallback。
5. `d_k=(f_F-f_C)/max(|f_start|,|f_F|,|f_C|,eps)`；`LCB=mean(d)-s/sqrt(3)`；
   lower-tail 为最差一个 d。只有 LCB>eps、tail>=0、state/graph/hash/FE 全部
   合法且 candidate 不劣于自身起点才 commit。
6. 统一 BudgetLedger 计入两臂 FE，probe 总额不超过 6%（建议 W/R/S=3/2/1%）。
   未选分支的 AOB fitness_record、CMA/cache/trust state 不能污染主结果；
   committed-state 与 evaluated-elite 必须分列。
7. R、S 只有 W 通过后才做独立增量消融；不能在同一 probe 叠加多个通道。

## 下一执行顺序

- 先做 CAR-W CLI/5k smoke：snapshot round-trip、branch-order swap、CRN
  replay、equal-FE、AOB hash、payload forbidden-field 和 discarded-branch
  isolation。
- 冻结后用 seeds 9-11 做 6 个 topology-stratified diagnostic cases，包含
  v33、CAR-W、shuffled-W、paired-fallback-probe-control、no-action。
- W gate：至少 6 commits 覆盖 3 cases/2 topology strata，probe-to-3M sign
  agreement >=60%，mean<0、median<=0、zero catastrophic、overhead<=6%。失败
  即停止 R/S。
- W 通过后用 seeds 12-16 做 13-case held-out；只有 mean-case>=7/13、
  worst>=5/13、zero catastrophic、upper-tail CVaR 不恶化且 shuffled 不多数
  优于 CAR，才允许 seeds 17-21 的完整 24-case。

## 当前未完成

CAR 尚未实现，也没有 CAR smoke 或新 held-out 结果。不得把本交接或
`docs/design/core-method.md` 的公式写成已验证性能结论；v33.8 仍是当前
canonical fallback。
