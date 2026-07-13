# 重构后结果与完整 Paper-Best 对照

- 日期：2026-07-12
- 执行者：Codex
- Runtime 结果：`cd69d90`，24-case、`seed=1`、3M-FE、
  `canonical_evidence_controller_v1`
- Paper-best：Two-Phase CC Table 2 每个 case 在所有算法中的最小 reported mean
- Reference：`references/paper_reported_table2_best_by_case.csv`
- Runtime dispatch allowed：`0`

## 汇总

- 可比较：24/24
- 超过 paper-best：5/24
- 胜出：`E3、E4、S2、S3、R1`
- 未胜出：19/24
- 完整生成结果：
  `E:\ARAC\results\refactor_3m_equivalence_20260712\after_vs_complete_paper_best.csv`

| Family | Wins | Cases |
|---|---:|---|
| Elliptic | 2/6 | E3、E4 |
| Schwefel | 2/6 | S2、S3 |
| Rastrigin | 1/6 | R1 |
| Ackley | 0/6 | 无 |

最接近的是 A4：`7.830691e4` 对 `7.830000e4`，仅差 `0.008825%`；A2、
A5、A6、A3、A1 分别低于 paper-best 约 `0.41%`、`0.50%`、`0.80%`、
`0.92%`、`1.12%`。主要差距集中在 E1/E2/E5、S1/S4/S5、R2-R6。

这是单 seed fixed-controller 结果，不是 3-seed best-of-three，也不是 25-run mean。
完整 paper-best 只用于 offline reporting，未进入 selector 或 optimizer。
