# ARAC 科研项目结构重构设计

- 日期：2026-07-12
- 执行者：Codex
- 状态：设计已获用户确认，实施计划已编写
- 稳定算法基线：v3.2，提交 `b88a4d9`
- 研究方向：面向大规模重叠全局优化，把协同进化得到的变量分组、重叠关系和运行时证据转化为动态优化动作。

## 1. 目标与非目标

### 目标

将 ARAC 收敛为一个可复现、可审计、可交接的科研项目，清晰分离：

1. 稳定算法库；
2. HCC 第三方/源工程代码；
3. 实验入口与实验配置；
4. 运行结果与离线证据；
5. 方法、协议、审计和论文材料；
6. 失败探索和废弃实现。

重构后的项目必须支持从代码提交、配置、随机种子和 FE 账本追溯到实验产物，并保持 v3.2 的行为和复现入口不变。

### 非目标

本次重构不包含：

- 新增优化策略或改变 ARAC runtime selector；
- 把 v3.3 `late_stagnation_nda_takeover` 合入稳定方法；
- 重跑 25-run final protocol；
- 重跑论文 baseline；
- 使用历史结果、论文数值或 case label 修改 runtime dispatch；
- 删除历史结果或失败实验的可审计证据。

## 2. Canonical 基线

`b88a4d9` 是本项目重构的唯一稳定算法基线，目标是保留其已验证的 `12/13` best-of-three 结果。当前其他 worktree 中未提交的 v3.3 改动不属于稳定 runtime。

v3.3 失败实验必须作为负结果归档，至少保留：

- 设计文档；
- 实验配置；
- 运行结果路径；
- 失败门判定；
- 未通过的原因。

归档实现不得被正式实验入口导入，也不得被默认配置引用。

## 3. 目标目录结构

```text
E:/ARAC/
├── src/arac/
│   ├── evidence/              Phase-I 证据抽取
│   ├── policy/                reference-blind 动作策略
│   ├── actions/               动作合同和数据结构
│   ├── execution/             执行计划、预算和 trace
│   ├── backends/              HCC 等后端适配
│   ├── benchmarks/            AOB/FlyKi 适配
│   ├── evaluation/            FE 账本、比较和统计
│   └── audits/                anti-leakage、语义和风险审计
├── vendor/hcc/                只读 HCC 源码快照
├── experiments/
│   ├── pilots/                 schema、1-run 和 runtime smoke
│   ├── recovery/               历史结果恢复
│   ├── ablations/              消融和机制实验
│   ├── final/                  正式协议入口
│   └── archive/                失败或废弃实验入口
├── configs/                    当前有效配置的唯一来源
├── analysis/                   结果汇总、统计和绘图脚本
├── results/                    本地产物，不进入 Git
├── docs/
│   ├── design/                 方法和架构设计
│   ├── protocols/              实验协议
│   ├── audits/                 结果与合规审计
│   ├── research-log/           研究过程记录
│   └── superpowers/            工程设计和实施计划
├── paper/
│   ├── drafts/                 论文草稿
│   ├── figures/                论文图表源文件
│   └── tables/                 论文表格源文件
├── references/
│   ├── paper/                  论文 reported values
│   ├── historical/             历史运行证据
│   └── source-index.md         外部源代码索引
├── scripts/                    通用运行、审计和构建命令
└── archive/                    不能进入 runtime 的旧材料
```

## 4. 模块边界

### 4.1 稳定库

`src/arac/` 只包含可复用的 ARAC 逻辑。它不能读取论文均值、历史 final outcome、relative gain、problem family 或 case-specific selector 配置。

第一轮迁移只调整路径和 import，不修改算法逻辑。第二轮再拆分两个大模块：

- `policy/relation_policy.py`：拆为证据模型、动作映射和门控；
- `backends/hcc.py`：拆为执行计划、FE 预算、共享变量写回和 trace 审计。

每次拆分必须保持旧测试、FE 账本和 anti-leakage 检查通过。

### 4.2 HCC 源码

`vendor/hcc/` 只保存 HCC/AOB 源码快照和必要说明。ARAC 策略、证据和审计不能反向写入该目录。HCC 修改必须通过显式 patch 或适配器完成，并记录来源。

运行器属于 `src/arac/execution/` 或 `scripts/`，不再与第三方源码混放。

### 4.3 实验

每个实验目录至少有：

```text
README.md
config.yaml
run.py 或 run.ps1
expected_outputs.md
```

正式实验额外需要：

```text
protocol.md
manifest.schema.json
```

实验 README 必须声明 claim level、允许的 runtime 输入、禁止输入、same-budget 状态、backend semantics、negative control 和 catastrophic-loss 状态。

## 5. 结果与证据流

标准科研数据流为：

```text
配置 + 代码提交 + seed
        ↓
实验入口
        ↓
Phase-I trace / evidence profile
        ↓
action decision / execution trace
        ↓
same-budget ledger
        ↓
offline evaluation / audits
        ↓
manifest + analysis tables + figures
```

`results/` 不作为源码事实源。每个实验通过统一 `manifest.csv` 记录：

```text
experiment_id, protocol, git_commit, config_path, seed, case_id,
total_fe, status, claim_level, output_path
```

历史结果和论文表格只能在实验完成后的 offline evaluation 中使用。runtime selector 必须保持 reference-blind。

## 6. 迁移阶段

### 阶段 0：冻结与盘点

- 固定 v3.2 canonical commit；
- 记录当前 worktree、未提交 v3.3 diff 和未跟踪材料；
- 生成旧路径到新路径的迁移清单；
- 确认结果目录不进入 Git。

### 阶段 1：结构迁移

- 创建目标目录；
- 移动方法文档、协议、审计和论文草稿；
- 将 HCC 源码归入 `vendor/hcc/`；
- 将实验按 pilot/recovery/ablation/final/archive 分类；
- 修复 import 和命令路径。

这一阶段不改变运行逻辑。

### 阶段 2：入口与配置收敛

- 每个正式实验只保留一个入口；
- 配置文件成为参数的唯一来源；
- 删除或标记重复 runner；
- 增加 manifest 和结构校验命令。

### 阶段 3：模块拆分

- 拆分大文件；
- 增加明确的 evidence、policy、execution、evaluation 和 audit 接口；
- 通过回归测试证明行为未变。

### 阶段 4：科研产物整理

- 从结果目录生成 manifest；
- 将可复用统计逻辑下沉到 `analysis/`；
- 生成论文图表和表格的可追溯输入；
- 补充复现实验 README 和项目总览。

## 7. 验证门

每个阶段必须通过：

1. `python -m pytest` 相关测试；
2. `git diff --check`；
3. package import 和 CLI smoke；
4. HCC backend smoke；
5. same-budget FE ledger 检查；
6. anti-leakage 检查；
7. 结果 manifest 字段和路径检查；
8. 目录结构审计，确认没有缓存、临时输出或大结果误入 Git。

重构完成后，至少需要重新验证 v3.2 的稳定入口；不能只以“文件移动成功”作为完成标准。

## 8. 风险与处理

| 风险 | 处理 |
| --- | --- |
| 两个 worktree 存在不同算法状态 | 先冻结 `b88a4d9`，再迁移，禁止直接从未提交 v3.3 复制 |
| HCC 相对路径失效 | 用显式 backend root 和 CLI 参数，增加 cwd-equivalence smoke |
| 历史结果被 runtime 读取 | 将历史材料置于 `references/historical`，加 anti-leakage 检查 |
| 大结果移动造成损坏或耗时 | 第一轮只建立索引，不批量重排 1.1 GB 结果 |
| 论文材料和实验协议混淆 | 论文草稿进入 `paper/`，协议进入 `docs/protocols/` |
| 大文件拆分改变行为 | 先保留兼容导入，逐模块做回归测试 |
| 失败实验重新成为默认入口 | archive 目录不被 configs 或正式 runner 引用 |

## 9. 完成标准

本重构只有同时满足以下条件才算完成：

- v3.2 canonical runtime 可运行；
- 正式实验入口和配置唯一且可发现；
- HCC 源码、ARAC 方法和实验 runner 边界清晰；
- 历史/论文结果与 runtime 隔离；
- 结果 manifest 可追溯代码、配置、seed 和 FE；
- 相关测试和结构审计通过；
- v3.3 失败实验被明确归档而非伪装成正式方法；
- Git diff 不包含缓存、临时日志或大结果文件。
