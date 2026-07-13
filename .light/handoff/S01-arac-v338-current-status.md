---
session_no: S01
suggested_title: "[ARAC] S02 恢复 v33.8 证据并验证全 24-case"
parent_session: none
project: arac
date: 2026-07-13
author: Codex
---

## 当前阶段

ARAC 已完成科研项目结构重构、v33.8 风险感知动作保护实现和受保护
8-case 的 3-seed 3M-FE pilot。当前阶段不是继续按 case 调参，而是恢复可审计
原始结果，并用同一个 reference-blind runtime selector 验证完整 24-case 是否达到
best-of-three 至少 13 胜。

## 研究创新想法

研究方向的统一表述：

> 面向大规模重叠全局优化，研究如何把协同进化得到的变量分组和重叠关系转化为
> 运行时动作，动态调整共享变量写回、子问题资源分配和搜索起点，从而提升复杂
> 重叠问题的优化性能。

核心不是只做分组，也不是只换优化器，而是把分组结果从静态结构转化为动态动作
决策依据：

```text
Phase I: 结构识别 / 证据收集
  -> 分组、共享变量、重叠关系、组贡献、冲突、停滞和预算状态
  -> Phase II: evidence-to-action runtime controller
  -> coordinate / isolate / reassign_repair / protect / fallback /
     trajectory or search-state actions
  -> 改变共享变量写回、组间协调、资源分配和搜索起点
  -> 改变后续优化轨迹和最终结果
```

论文贡献应表述为 `grouping evidence -> auditable optimization action` 闭环，不能
表述为按 case 选择历史最优 lane。runtime dispatch 严禁读取 case/function family、
paper-best、历史 final outcome、relative gain 或 oracle 标签。

## 已完成

### 1. v33.8 方法实现

- `src/arac/policy/action_trust_policy.py`：relation/action trust 状态，包含
  probation、trusted、quarantined、cooldown 和 exposure cap。
- `scripts/hcc_smoke_runner.py`：objective-paired credit、bounded writeback、
  no-op 不消耗 trust、search-state 后 pending credit 作废，以及 v33 trace。
- topology-scoped protected fallback：dense overlap 使用
  `dense_preserve_v31`；non-dense overlap 使用 `non_dense_bounded_0_5`。
- v32 legacy trace/schema 保持隔离，v33 为 opt-in lane。
- 主要功能提交 `4498fa8`，已本地合并到 `main`，merge commit 为 `4a72058`。

### 2. 受保护 8-case pilot

协议：E2/E4/E6/S6/R1/R2/A4/A5，seeds 1/2/3，每次 3,000,000 FE，
以 best-of-three 离线比较 `references/paper_reported_table2_best_by_case.csv`。

| Case | v33.8 best | Paper-best | Relative gain |
|---|---:|---:|---:|
| E2 | 3,885,968.0 | 6,870,000.0 | 43.44% |
| E4 | 12,913,830.0 | 19,000,000.0 | 32.03% |
| E6 | 21,431,810.0 | 26,200,000.0 | 18.20% |
| S6 | 12,026.0 | 13,300.0 | 9.58% |
| R1 | 169,214.2 | 174,000.0 | 2.75% |
| R2 | 227,665.2 | 248,000.0 | 8.20% |
| A4 | 78,299.33 | 78,300.0 | <0.01% |
| A5 | 78,149.10 | 78,200.0 | 0.07% |

记录的审计摘要：8/8 best-of-three 胜出；24/24 fresh；same-budget violation
0/24；AOB inputs 237/237 unchanged；anti-leakage 16/16 pass。权威的 tracked
汇总见 `docs/superpowers/specs/2026-07-13-risk-aware-action-guard-design.md`
的 `Topology-Scoped 3M Result`。

该结果只证明 8-case pilot 的“至少一个 seed 胜出”，不等于三个 seed 全胜、
3-seed mean 胜出、25-run 统计结论或完整 24-case 达到 13 胜。A4/A5 是贴线胜出，
必须单列为脆弱结果。

### 3. 合并后验证

- canonical Git-tracked pytest：564 passed，1 skipped。
- `py -3.12 -m compileall -q src scripts experiments`：通过。
- `git diff --cached --check`：通过。
- reviewer 对 v33.8 最终实现未发现 Critical/Important 问题。

## 工作区状态

- 仓库：`E:\ARAC`。
- 分支：`main`；HEAD `4a72058`；相对 `origin/main` ahead 89，尚未 push。
- tracked 工作树无未提交改动；存在用户原有未跟踪论文、FlyKi、exp006-exp008
  和外部源码材料，不得删除、覆盖或顺手纳入 canonical runtime。
- 默认无筛选 `pytest` 会收集未跟踪 exp008 tests；其中 37 项仍导入重构前的
  `experiments.exp_005_hcc_final_protocol_pilot` 路径。canonical tracked suite
  通过，但未跟踪 exp008 尚未迁移。
- `scripts/audit_project_structure.py` 会因现有 `.agents/` 和
  `Large-Scale-Overlapping-Optimization-master/` 顶层目录报错；不要为通过审计
  擅自删除这些用户材料。

## 关键风险

1. 原 v33.8 raw result 目录位于已清理的功能 worktree，当前
   `E:\ARAC\results` 中没有该目录。tracked spec 保留汇总，但逐 seed CSV、trace、
   FE ledger 和 manifest 必须重跑恢复，不能伪造或根据汇总反向生成。
2. 目前没有当前 v33.8 的完整 24-case 结果，因此不能宣称 `13/24`。
3. 3-seed best-of-three 是项目当前 pilot gate，不是论文最终统计口径；最终仍需
   至少 5 seeds 的稳定性检查，正式结论应走 25 runs 或明确标为 pilot。
4. 调整 selector 时必须保住已有胜场，但只能依据 Phase-I/runtime evidence，
   不得用 paper/history/case identity 做 dispatch。

## 下一步

1. 在 `main` 上原样重跑 v33.8 的 8-case x 3-seed x 3M-FE，输出到稳定的
   `E:\ARAC\results\controller_v338_replay_8case_seed123_3m_20260713`，恢复 raw
   CSV、trace、manifest、same-budget 和 anti-leakage 证据。验收要求仍为 8/8、
   24/24 fresh、0 FE violation、AOB inputs unchanged、anti-leakage pass。
2. 8-case replay 一致后，用相同 v33.8 selector 跑完整 E1-E6/S1-S6/R1-R6/A1-A6，
   seeds 1/2/3、3M FE、jobs 24 或机器稳定上限。主要门槛是 best-of-three 至少
   13/24，同时另报 mean、worst seed、每个 case 的 3-seed 胜场和 catastrophic loss。
3. 若不足 13/24，只按 runtime evidence slice 诊断失败，例如 dense/non-dense、
   trust phase、credit、fallback route、冲突和停滞；先做 no-harm 消融，再修改动作
   映射，禁止按 case 追加特殊分支。

## 建议的第一条命令

```powershell
$env:PYTHONPATH='src'
py -3.12 -m experiments.pilots.exp_003_hcc_runtime_consumer_smoke.run `
  --output-dir E:/ARAC/results/controller_v338_replay_8case_seed123_3m_20260713 `
  --seeds 1 2 3 `
  --problems E2 E4 E6 S6 R1 R2 A4 A5 `
  --jobs 24 `
  --max-fes 3000000 `
  --budget-accounting strict `
  --lane-profile evidence_action_controller_v33
```

启动前先运行该入口的 CLI/5k smoke，确认当前环境、路径和输出目录；不要直接覆盖
已有结果目录。

## 必读文件

1. 本卡。
2. `AGENTS.md`。
3. `docs/design/core-method.md` 和 `docs/design/boundaries.md`。
4. `docs/superpowers/specs/2026-07-13-risk-aware-action-guard-design.md`。
5. `docs/superpowers/plans/2026-07-13-risk-aware-action-guard.md`。
6. `src/arac/policy/action_trust_policy.py`、`scripts/hcc_smoke_runner.py` 和
   `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`。

## 禁止

- 不要把 8/8 pilot 写成完整 24-case 或 25-run 结论。
- 不要根据文档中的 best 值伪造 raw runs、均值、标准差或 seed 结果。
- 不要让 paper-best、历史最优、case label 或 function family 进入 runtime dispatch。
- 不要重跑论文 baseline；只跑当前方法并做 offline comparison。
- 不要修改 `E:\HCC-main`，它保持只读证据库。
- 不要删除用户未跟踪材料；接手后先用 `git status` 和 `git log -3` 刷新现实。
