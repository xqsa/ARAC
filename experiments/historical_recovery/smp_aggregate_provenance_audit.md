# SMP 历史聚合来源审计

日期：2026-08-12  
执行者：Codex  
性质：只读证据审计；未运行或修改任何优化动作

## 2026-08-12 补充修正

本审计最初只定位到 EXP-028，因此下面关于“截图协议缺失”的结论已经被随后找到的
EXP-029 完整归档取代。保留原文是为了记录调查过程，当前有效结论如下：

- EXP-029 完整目录位于
  `C:/Users/83718/.codex/worktrees/4e56/ARAC/results/exp_029_e_series_smp_seed_replacement_sensitivity/smp-e-series-seed126-to117-sensitivity-v1/`，并已复制到
  `.codex-tasks/smp-table-provenance-recovery/raw/recovered_archives/exp029/`。
- 该目录含 `150/150` 条完整逐 seed 记录：每个 E1-E6 使用 seed117、122-125、
  127-146，全部精确 `3,000,000 FE`；其 144 条 parent 行由 EXP-028 manifest
  SHA-256 绑定，六条 seed117 为 fresh replacement。
- 六个 EXP-029 显示聚合与截图 E-series 列逐行一致，E3 独立复算为
  `1.3413599575351853E+07 +/- 5.199777860974465E+06`，即
  `1.34E+07 +/- 5.20E+06`。
- 所以截图来源已经恢复，但其科学身份必须写成“EXP-029 事后 seed126->117
  敏感性分析”，不能冒充预注册 EXP-028 正式结果。EXP-028 的正式 E3 是
  `1.32E+07 +/- 5.34E+06`。
- 完整哈希和复算结果见
  `.codex-tasks/smp-table-provenance-recovery/raw/recovered_archive_validation.json`。

commit `521f29ec` 的精确源闭包已成功复放 EXP-029 `E3/seed117` 代表轨迹：fresh
执行精确消耗 `3,000,000 FE`，终值 `12206076.633414682`、完整
`run_summary.json` 和 `persistent_phase2_action.json` 均与归档字节哈希一致，严格
验证为 `exact_artifact_reproduction`。下一合法动作是只并行复放 EXP-029 其余五条
fresh seed117；暂不扩到 144 条 retained EXP-028 轨迹，也不追逐 EXP-052 的
`1.39E+07`。

随后 E1/E2/E4/E5/E6 的五条 fresh seed117 也以五路并行完成，五条各自通过
`11/11` 检查；连同 E3，EXP-029 引入的六条 replacement row 已全部达到终值、完整
summary JSON、完整 action JSON 和文件 SHA-256 精确复放。只读总门以 144 条
integrity-passing EXP-028 retained rows 加这 6 条 fresh exact rows 重建 EXP-029，六个
聚合精确等于恢复归档。与已通过的 AOR/CTP/GCB exact lane 合并后，截图 AOB-24
显示均值门为 `24/24`，显示 `mean +/- std` 对也为 `24/24` 精确一致。EXP-052 的
`23/24` 仍是其自身协议的有效失败结果，但已 superseded 为截图来源门。

## 原始中间结论（已被以上修正取代）

当前磁盘上存在三层不同的 E-series/SMP 证据，不能互相替代：

1. 用户截图转录的 ARAC 聚合列；
2. EXP-028 的固定 SMP 25-seed 正式实验；
3. EXP-052 精确 HCC runner 的恢复 lane。

截图列只有均值和样本标准差，没有 seed、runner、命令、checkpoint、配置或 receipt。
EXP-028 和 EXP-052 都有明确协议，但都不能生成截图中的六个 E-series 聚合。因此，
当前 `23/24` 结果只能表述为“对截图显示均值的数值门失败一项”，不能表述为已经
证明或否定同一历史协议下的 SMP 恢复。

## 三层证据

| Case | 截图 ARAC 列 | EXP-028，seeds 122-146 | EXP-052 HCC lane，seeds 117-141 |
|---|---:|---:|---:|
| E1 | `5.69E+05 +/- 1.57E+06` | `5.85E+06 +/- 2.63E+07` | `3.48E+02 +/- 1.74E+03` |
| E2 | `5.62E+06 +/- 3.78E+06` | `5.59E+06 +/- 3.79E+06` | `3.50E+06 +/- 4.05E+06` |
| E3 | `1.34E+07 +/- 5.20E+06` | `1.32E+07 +/- 5.34E+06` | `1.39E+07 +/- 6.75E+06` |
| E4 | `2.61E+07 +/- 9.35E+06` | `2.61E+07 +/- 9.35E+06` | `2.26E+07 +/- 8.62E+06` |
| E5 | `2.98E+07 +/- 9.18E+06` | `2.89E+07 +/- 8.96E+06` | `2.87E+07 +/- 1.13E+07` |
| E6 | `3.19E+07 +/- 6.54E+06` | `3.16E+07 +/- 6.77E+06` | `2.74E+07 +/- 9.06E+06` |

EXP-028 的 E3 与截图接近，但 E1 相差一个数量级，E5/E6 的标准差也不同，所以
“截图来自 EXP-028，只是 seed 窗口写错”这一假设被否定。EXP-052 HCC lane 的 E1
又比截图低三个数量级，其生成协议同样不同。

## 可追溯事实

### 截图聚合列

- 当前转录文件：`output/pdf/aob_arac_method_comparison_corrected.csv`。
- SHA-256：`1b27b193c37bba031f820c62fed3cf01bde600355a093ed34c136ee4dc8735bf`。
- 用户指定旧任务 `019f9dab-0f42-70f2-8f45-5cf51411e668` 中，turn
  `019fa25e-379e-7210-b9dd-09261ee21c29` 表明该列来自用户提供的四张“最终结果”
  截图，并被录入方法对比表。
- 原截图临时文件已不存在；会话和 CSV 都没有记录 E-series 的 seed、runner、命令、
  checkpoint、配置或逐轨迹结果。

### EXP-028

- 原始结果：
  `C:/Users/83718/.codex/worktrees/4e56/ARAC/results/exp_028_e_series_fixed_persistent_phase2_validation/smp-e-series-25run-final-v2/`。
- 配置明确冻结 E1-E6、seeds `122-146`、每条 `3,000,000 FE`、20 workers、
  `smp-e-series-v1` profile。
- `150/150` 完成，manifest 的 `integrity_gate_passed=true`，没有 native rerun。
- config SHA-256：
  `deccaf334a28356851c660baf12087e95c55a5269cd0170171f5ed92832dd9ac`。
- manifest SHA-256：
  `d36e35c5920e6629ee8e0f5aef1c190d445e4f1cc7c04713d00d377d69254148`。
- summary SHA-256：
  `1c44bd79a905545cfc6524f1ba13f3b6a42ddf1f2258f1abd0fd5809b74dd010`。
- EXP-029 只是把 seed126 post-hoc 替换为 fresh seed117 的敏感性分析，另外 144 条
  复用 EXP-028；它不是独立 150-run campaign。

### EXP-052 HCC lane

- 恢复协议明确使用 E1-E6、seeds `117-141`、每条 `3,000,000 FE`。
- 150 条轨迹均合法；只有 E1/E3 的 seeds117-121 有 retained EXP-052 artifact，可做
  逐字段 exact replay，其余轨迹只由冻结 runner、预算、状态和 receipt 完整性约束。
- lane summary SHA-256：
  `2b3d340a730e7a60d196163ab8e136489b3b5a25e6d9db1efbb1019daa70739a`。
- runner/protocol SHA-256：
  `229f89efbd6f11e84178a8552ff6dd567503f7af0b27b2be1d9eb2997c83f4b5`。

## 当前科学判定

- AOR、CTP、GCB 的正式 historical lane 各有逐轨迹/聚合来源，可维持已通过结论。
- SMP 的截图列目前是缺少生成 provenance 的目标值，不足以授权调整 seed、动作或
  optimizer 去追一个显示均值。
- `E3 = 1.39E+07` 高于截图 `1.34E+07` 仍是严格显示值门的真实失败，不能改成通过；
  但该失败不能归因于“历史 SMP 未恢复”，因为两边不是已证明相同的实验协议。
- 在取得原始 E-series 25-run 目录、逐 seed CSV、runner/config/command，或用户明确
  选择一个可追溯协议作为新基准前，不再运行 SMP 调参或 AOB-24 聚合重试。

## 下一合法动作

优先恢复截图 E-series 结果的原始目录或备份。最低可接受证据是 25 个 seed 的逐轨迹
终值和 seed 列表；更完整的证据还应包含 runner/config/command 与预算边界。若原始证据
已经不可恢复，应新建版本化决策，明确选择 EXP-028 或另一套可追溯协议作为“替代基准”，
并将其称为重新绑定后的基准，不能继续称为截图历史协议的精确恢复。
