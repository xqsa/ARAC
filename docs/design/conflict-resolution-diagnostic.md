# exp_019 冲突取值消解 oracle-ceiling 诊断

Date: 2026-07-19
Executor: Codex
Status: `oracle_no_go`，STOP

## 1. 结论

第 1 步未通过冻结 gate，不进入第 2 步。

在 E3/A4/S5 synthetic conflict、seeds 117-121 的 15 个 fresh case-seed 单位上，
reliability-weighted bridge 相对“可靠度较高 owner，平局选 left”的主 baseline：

- material win 为 0/15；one-sided 95% Wilson `paired_win_lcb = 0`；
- 总体 median delta 为 `-7.307711256377876e-04`，低于要求的
  `log(1.01) = 9.950330853168092e-03`；
- trajectory-level large loss 为 0/15，`large_loss_ucb = 0.1528077046101401`。

因此失败项固定为 `conflict_paired_win_lcb` 和 `conflict_median_delta`。这不是执行
失败，也不能通过调阈值或挑 seeds 修复；它表示该 bridge 在预注册 synthetic
conflict 诊断上没有表现出材料即时收益。

exp_018 conform 阴性对照通过：material win 2/15，`paired_win_lcb =
0.045150886757578446`，median delta `-6.635890363129467e-05`，large loss 0/15。
该结果只能表述为“与无材料收益相容”，不是统计等价性证明。

## 2. 冻结协议

### 2.1 执行矩阵

- smoke：A4 conflict、seed 1、100k FE；结果 `smoke_pass`。
- oracle：E3/A4/S5 conflict、seeds 117-121、3M FE、`jobs=12`，共 15 条 fresh
  trajectories。
- conform：只读 exp_018 mechanism 的 E3/A4/S5、seeds 117-121、
  `b_rddsm_evidence_overlay`，共 15 条 paired-owner trajectories。
- 每条 applicable trajectory 选择四个 relation，执行 x0/left/right/bridge 四点
  probe，共 16 个显式 FE。

private worker 只在自身子进程内把 `scripts.hcc_smoke_runner.Benchmark` 替换为
experiment-local `ConflictBenchmarkFactory`。HCC、RDDSM grouping、relation selection、
四点 probe、FE ledger 和 artifact writer 均复用现有实现；共享 runner、vendor、
exp_018 和 runtime state 未修改。

### 2.2 检验统计量

对每个选中 relation：

```text
baseline = owner with higher reliability; ties choose left
delta = log((f_baseline + 1e-300) / (f_bridge + 1e-300))
```

一个独立单位是同一 trajectory 四个 relation delta 的中位数。material win 定义为
trajectory delta `> log(1.01)`；large loss 定义为 trajectory delta
`<= -log(1.20)`。Wilson 区间使用 one-sided 95% 的 `z = Phi^-1(0.95)`。

## 3. 完整性结果

| 检查 | 结果 |
| --- | --- |
| A4/seed 1/100k smoke | `smoke_pass`；99,985 FE；16 probe FE |
| Fresh conflict trajectories | 15/15 completed，覆盖 3 cases × 5 seeds |
| Conform source trajectories | 15/15 raw manifests；133 个文件 hash 绑定 |
| Conflict FE ledger | A4: 2,999,985-2,999,996；E3: 2,999,987-3,000,000；S5: 2,999,997-3,000,000 |
| Probe bundle | 每条 4 relations、16 probe FE、delayed labels 闭合 |
| Observer boundary | 15/15 `runtime_authorized=0`；15/15 runtime fingerprints 不变 |
| Vendor/AOB inputs | before/after SHA256 一致；synthetic bundle hash 一致 |
| Source drift | smoke/oracle 期间 source bundle 未变化 |

每条 budget 均不超过 3M，并位于 frozen maximum-native-group-population terminal
tolerance 内。所有 raw manifest、plan、probe、delayed、shadow、budget 和 AOB input
artifacts 均在统计前验证；缺失或 hash 不符会产生 `diagnostic_invalid`，不会进入
outcome gate。

## 4. Gate 汇总

| Side | Material wins | Paired-win LCB / UCB | Median delta | Large losses | Large-loss UCB | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Conflict | 0/15 | 0 / 0.152807705 | -7.307711256e-04 | 0/15 | 0.152807705 | FAIL |
| Conform | 2/15 | 0.045150887 / 0.333574763 | -6.635890363e-05 | 0/15 | 0.152807705 | PASS |

| Frozen check | 要求 | 实际 | 结果 |
| --- | --- | ---: | --- |
| Conflict paired-win LCB | `> 0.5` | 0 | FAIL |
| Conflict median delta | `> log(1.01)` | -7.307711256e-04 | FAIL |
| Conflict large-loss count | 0 | 0 | PASS |
| Conform paired-win LCB | `<= 0.5` | 0.045150887 | PASS |
| Conform abs median delta | `<= log(1.01)` | 6.635890363e-05 | PASS |
| Conform large-loss count | 0 | 0 | PASS |

按 case 聚合时，conflict 的 A4/E3/S5 median delta 分别为
`-6.189678851e-08`、`-6.520290482e-03`、`-3.347095567e-03`，三个 case 均为
0/5 material wins。没有单一 case 支持 promotion。

## 5. 逐 case-seed 原始单位

`best-owner delta` 是离线敏感性分析，未进入主 gate。

| Side | Case | Seed | Trajectory delta | Best-owner delta | Material win | Large loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| conflict | A4 | 117 | -1.127685803e-04 | -1.127685803e-04 | 0 | 0 |
| conflict | A4 | 118 | -6.176616641e-09 | -5.995721372e-05 | 0 | 0 |
| conflict | A4 | 119 | -6.189678851e-08 | -6.318435634e-08 | 0 | 0 |
| conflict | A4 | 120 | 7.671188401e-08 | -1.962344262e-08 | 0 | 0 |
| conflict | A4 | 121 | -6.756093071e-06 | -1.025041122e-05 | 0 | 0 |
| conflict | E3 | 117 | 2.484816035e-03 | -1.164800838e-02 | 0 | 0 |
| conflict | E3 | 118 | -1.743172967e-02 | -1.798785202e-02 | 0 | 0 |
| conflict | E3 | 119 | -6.520290482e-03 | -6.520290482e-03 | 0 | 0 |
| conflict | E3 | 120 | -5.244683595e-03 | -2.795625481e-02 | 0 | 0 |
| conflict | E3 | 121 | -1.042102520e-02 | -4.556593931e-02 | 0 | 0 |
| conflict | S5 | 117 | 1.350956542e-03 | -3.782342020e-03 | 0 | 0 |
| conflict | S5 | 118 | -3.347095567e-03 | -3.347095567e-03 | 0 | 0 |
| conflict | S5 | 119 | -7.307711256e-04 | -2.610373983e-03 | 0 | 0 |
| conflict | S5 | 120 | -9.395201225e-03 | -9.395201225e-03 | 0 | 0 |
| conflict | S5 | 121 | -5.737177013e-03 | -5.737177013e-03 | 0 | 0 |
| conform | A4 | 117 | -6.635890363e-05 | -6.635890363e-05 | 0 | 0 |
| conform | A4 | 118 | 8.792473641e-06 | -4.846196900e-06 | 0 | 0 |
| conform | A4 | 119 | -9.243750797e-09 | -4.296590060e-07 | 0 | 0 |
| conform | A4 | 120 | -2.620863751e-06 | -2.621524915e-06 | 0 | 0 |
| conform | A4 | 121 | 1.847832329e-09 | -1.243565929e-07 | 0 | 0 |
| conform | E3 | 117 | -8.496672583e-03 | -1.051147564e-02 | 0 | 0 |
| conform | E3 | 118 | -1.124180333e-02 | -1.124180333e-02 | 0 | 0 |
| conform | E3 | 119 | -4.898422740e-03 | -6.593605765e-03 | 0 | 0 |
| conform | E3 | 120 | -1.365020453e-03 | -1.038066628e-02 | 0 | 0 |
| conform | E3 | 121 | -1.821292575e-03 | -1.821292575e-03 | 0 | 0 |
| conform | S5 | 117 | 1.310421492e-02 | -3.847937528e-02 | 1 | 0 |
| conform | S5 | 118 | -4.626620627e-03 | -9.250378360e-03 | 0 | 0 |
| conform | S5 | 119 | 1.368540196e-03 | -1.669765240e-02 | 0 | 0 |
| conform | S5 | 120 | -5.914430341e-03 | -7.895587378e-02 | 0 | 0 |
| conform | S5 | 121 | 1.278435756e-02 | -5.527329063e-03 | 1 | 0 |

Conflict 的 best-owner sensitivity median delta 为 `-0.0037823420204272564`，
conform 为 `-0.006593605765295856`。换成事后最佳 owner 不会把主结论翻成阳性；
该敏感性结果只作解释，不参与 gate。

relation-level CSV 中两侧各有一个 relation delta 达到 large-loss 阈值，但冻结的
独立单位是四个 relation 的 trajectory median；因此 gate 的 trajectory-level
large-loss count 两侧均为 0。两种粒度不得混用。

## 6. 结果绑定

结果目录 `results/` 按项目规则不提交。以下 SHA256 把本文结论绑定到本地原始
输出：

| Artifact | SHA256 |
| --- | --- |
| `smoke/smoke_gate.json` | `3e4925a7bea9e7b706edbc5db788104c63eee8e56b9438c823995454d4169731` |
| `oracle/oracle_gate.json` | `0f07e033141c5ae6afde6c3fd1fa54c93eba7fa6203617b70c0b72f90f22ca7e` |
| `oracle/trajectory_results.csv` | `780c1687a222d769c2790ca352a576eb1b1ef1a9b91c3203d712fc959f61e4a4` |
| `oracle/relation_results.csv` | `9c9b92c8e6af07cb816cb42455f759cade6413604afe0a8f1095a974c3467bdd` |
| `oracle/conform_source_manifest.json` | `b85d92b78ac43b9e2e7f181a57f9a52d730b6f2e3bcb63140b95cc73e6f71af8` |
| `oracle/execution_records.json` | `ae293562fb5aed95f0148866c89c14efe53a4320e56999db53bcf4c63ac55136` |

`conform_source_manifest.json` 进一步绑定 133 个 exp_018 aggregate/raw files，bundle
SHA256 为 `576a62b00c5b7b3d3080db8b33ca4630fc276e8f7cd26b0dda81c3c2031c5605`。

## 7. Claim boundary 与下一步

本诊断只验证同 checkpoint 的即时 objective value。它没有采用 probe 值、没有
writeback、没有改变 optimizer state，也没有测试终局性能。

冻结 gate 已要求任一 conflict 条件失败即 STOP。因此当前证据不授权：

- 编写第 2 步 conflict score 规格；
- 创建正式 exp_019 online pilot 的 protocol/run 文件；
- 引入 e-process；
- 设计或执行 runtime writeback；
- 声称 bridge 改进 HCC 或 conflict 优化性能。
