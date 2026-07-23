# AGENTS.md - ARAC 研究与工程规则

本项目研究的是一种两阶段协同进化方法：Phase1 观测优化证据，选择一个显式动作，
在 Phase2 精确执行，以同等函数评价预算超过 native HCC。所有实现、实验和结论都必须
围绕这条因果链展开，不能用复杂度代替有效性。

## 0. 当前阶段锁：Action Validation

当前阶段只验证“动作集合是否在真实 AOB 上存在可重复的正收益”。在动作门通过前：

- 不训练或调优 selector。
- 不继续扩展 `cohen_d`、轨迹特征、软评分等证据特征，除非它们只用于修复数据真实性。
- 不把 oracle/VBS、shadow branch 或 synthetic 结果描述成可运行方法的性能。
- 不启动 25-run final protocol。
- 不以单 case、单 seed 或机械 smoke 宣称动作有效。

当前阶段的退出条件：真实 AOB 上 action ceiling 的 VBS 具有正的 point estimate 和正的
case-seed cluster-bootstrap 95% LCB，material-positive context 不过度稀疏，且没有不可接受的
catastrophic loss。未达到该条件时，优先重新设计动作，不升级 selector。

## 1. 核心研究问题

研究链条固定为：

```text
Phase1 可观测证据 E(s)
        -> selector pi(E)
        -> 显式动作实例 a
        -> Phase2 确定性执行
        -> 同 FE 收益 Delta(s, a)
```

三个模块必须分离：

1. 证据模块只描述 Phase1 已经观测到的状态，不修改优化过程。
2. 选择模块只输出一个显式动作实例或 abstain，不直接写 incumbent 或 optimizer state。
3. 动作模块只执行收到的动作实例，不重新读取证据、重选 owner、重算参数或隐式 fallback。

任何同时读取信号、决定参数并执行变更的 arm 都是“复合策略”，不能直接作为纯动作收益的
证据。当前候选 `efficiency_budget_reallocation`、`delta_priority_scan` 和
`stagnation_cross_group_warm_start` 必须分别拆成候选生成与确定性执行两部分。

## 2. 显式动作契约

动作应采用不可变、可审计的实例，例如：

- `SharedWritebackAction(relation, shared_values)`
- `BudgetAllocationAction(group_budgets)`
- `SweepOrderAction(group_order)`
- `GroupWarmStartAction(group_index, unique_mean_shift)`

动作实例至少绑定：problem、seed、checkpoint、issued/target sweep、作用对象、完整参数、
参数 hash、anchor、FE 预算和随机种子 namespace。

执行约束：

- 相同动作实例在相同 checkpoint 上必须产生相同变更。
- 动作执行器不得根据 Phase2 delta 重新选择动作或参数。
- shared writeback 必须同步相关 owner optimizer 的 context memory。
- 缺失、失配、过期或重复消费必须显式 abstain，并记录原因。
- 不允许 probe、fallback、repair 或 blend 以隐藏分支混入另一个动作。
- 任何动作都必须说明修改了哪些状态、消耗多少 FE、何时开始生效以及何时结束。

### 2.1 已冻结的 R 系列动作

- `full_space_sep_cma` 的动作契约、canonical optimizer 构造和确定性 executor 只允许存在于
  `src/arac/actions/full_space_sep_cma.py`；runner、backend 和实验目录不得复制数值执行逻辑。
- selector 或协议层只能生成 `FullSpaceSepCmaAction`；runtime 必须通过
  `RuntimeActionDispatcher` 按实例类型调用 executor，不得按字符串重新解释动作。
- R1 使用 `phase_boundary` trigger；R2-R6 使用 `relation_dispatch` trigger。trigger adapter 可以
  分开，但两条路径必须调用同一个 executor。
- 当前冻结语义为 1000 维、population 24、canonical PyPop7/Ros-Hansen 参数、无 restart、严格改善
  才接受、下一 sweep 一次性消费；R 系列实验的 sigma 固定为 0.5，burst 预算取前一完整 native
  sweep 的实际 FE，随后恢复 3 个 frozen native sweeps。
- 后续实验需要改变上述任一语义时，必须使用新的 action 名、schema 和 protocol；不得原地修改
  `full_space_sep_cma` 后继续引用已有 R 系列结果。
- 当前证据边界不变：R3-R6 已通过 action-ceiling；R2 因 catastrophic context 被拒绝；R1 仅有
  forced phase-boundary terminal 结果，没有同 seed paired-native 证据。

## 3. Context 定义

一个 context 是目标 relation dispatch 时冻结的一次完整优化现场，也是 action-ceiling 的一个
配对实验样本。它至少包含：

- 完整 incumbent、best-so-far fitness prefix 和当前 FE；
- sweep、group position、RDDSM topology/order；
- relation key、共享变量、owner values/deltas；
- group optimizer mean/state、population、预算和停滞状态；
- Phase1 动作候选及其 hash；
- 后续 native seed schedule 和 branch-local seed namespace。

所有 arm 必须从同一不可变 context clone。不同 branch 不得共享 benchmark、fitness record、
optimizer 或其他可变状态。同一 case-seed 轨迹中的多个 context 是多个比较样本，但不是多个
独立统计样本。

## 4. HCC 与 AOB 真源

- `E:\HCC-main` 是 HCC/AOB 行为真源，默认只读。修改 ARAC 前先检查对应源文件、数据、
  公式和测试；除非用户明确要求，不修改源工程。
- `E:\ARAC\vendor\hcc` 是本项目可运行 vendor 实现，必须保持与真源的 benchmark 和 native
  HCC 语义一致。
- AOB objective、candidate、`Ovector`、`Pvector` 和 optimizer 维度均为 1000。
- 实验 case 使用 `E1/E3/A4/R4/S5` 标识。底层 `F*-info.txt` 是 metadata 文件名，不能把
  “F1-F6”写成实验 case 集。
- `E1` 只作为无 overlap 控制；synthetic conflict 只做压力测试，不与真实 AOB 汇总。
- 不重跑论文 baseline；native HCC 由同一 runner、同一 checkpoint 和同一 FE branch 产生。
  论文 reported values 仅用于离线背景比较。

## 5. Action-Ceiling 协议

每个 context 至少比较：

- `native_eq8`：目标 dispatch 和后续 continuation 均执行原始 HCC。
- `true_no_writeback`：目标位置不修改当前值。
- 每个显式候选动作实例。

`native_eq8` 与 `true_no_writeback` 必须分开，禁止使用含糊的
`conservative_no_action` 名称。

比较必须满足：

- 同 checkpoint、同 absolute FE、同 native seed schedule。
- native branch 在逐 FE objective prefix、incumbent 和 action trace 上与未分支 runner 一致；
  不一致时整个 context 无效。
- 主 horizon 为从目标 dispatch 到下一次同 relation dispatch 的一个完整 native sweep `h`。
- 记录 `t+1`、`t+h` 和 `t+3h`；复杂动作消耗的额外 FE 必须从同一 horizon 内扣除。
- 评价量使用截至目标 FE 的 best-so-far error，而不是自然 endpoint 或即时 candidate fitness。

收益标签固定为：

```text
Delta(s,a) = log((f_native(t+h) + 1e-300) / (f_a(t+h) + 1e-300))
```

因此 `Delta > 0` 表示动作优于 native HCC，`Delta < 0` 表示动作更差。

## 6. SBS、VBS 与决策门

- SBS（Single Best Solver）：在所有真实 AOB context 上始终使用同一个 arm，取 macro mean
  `Delta` 最大者。
- VBS（Virtual Best Solver）：每个 context 事后选择 `Delta` 最大的 arm，是 oracle 上限，
  不是可运行算法。
- native 必须包含在候选中；数值并列优先 native。因此 VBS 不会为负，动作完全无收益时
  VBS 等于 0。
- selector 必须单独报告，不得混入 VBS。

决策顺序固定：

1. VBS point estimate 不为正：`redesign_actions`。
2. VBS point estimate 为正但 95% LCB 不为正：`collect_more_ceiling_contexts`。
3. VBS 显著为正但 material-positive prevalence 过低：动作 headroom 稀疏，先保守 abstain，
   不训练复杂 selector。
4. VBS 显著为正且所有非 native SBS 都不显著：动作有条件价值，才允许进入证据可分性研究。
5. 某个非 native SBS 的 95% LCB 为正且 catastrophic loss 受控：优先验证该固定动作，
   不假设一定需要 selector。
6. 任何 arm 出现不可接受的 catastrophic loss：该动作不得进入 runtime。

默认阈值：

- material positive：`Delta > log(1.01)`。
- catastrophic loss：`Delta <= -log(1.20)`。
- bootstrap：2000 次，固定 seed `2026071901`，按 case-seed 聚类。
- 主集统计使用真实 AOB macro mean；synthetic 不进入 SBS、VBS、bootstrap 或升级判断。

## 7. 实验推进顺序

1. 单 E3、单 seed、300k FE mechanical smoke：只检查执行、FE、parity 和 artifact。
2. E3/S5 action diagnostic：每个 case 至少 300k FE、至少 10 个有效 context；推断性结论
   应尽量覆盖至少 5 个独立 seed，而不是依赖同一轨迹内的相关 context。
3. 真实 AOB pilot：`E1/E3/A4/R4/S5`，固定 seed 集和同 FE 协议。
4. 仅在 action gate 通过后，研究 Phase1 证据能否预测 oracle winner/material-positive context。
5. 仅在证据具有 out-of-sample 可分性后，训练或调整 selector，并报告相对 SBS/VBS regret。
6. 小规模 runtime validation 通过且 catastrophic gate 闭合后，才允许设计 25-run final。

不得跳级。机械 smoke 只证明“能正确运行”，不证明“动作有效”；VBS 只证明动作集合的
oracle headroom，不证明 Phase1 证据包含可学习信号。

## 8. 后续证据与 Selector 规则

Action gate 通过前，本节保持冻结。解锁后仍必须满足：

- runtime 特征只能来自决策时已经可见的 Phase1 数据。
- 禁止使用 future fitness、oracle label、final error、paper result、problem family prior 或
  同一 trajectory 的后续 outcome 作为输入。
- 训练/验证按 case-seed 或 trajectory 切分，不能让同一轨迹的 context 跨训练与测试集。
- 先验证特征对 oracle winner、material win 和 catastrophic loss 的可分性，再训练 selector。
- selector 必须允许 abstain，并与 SBS、VBS、native 分别比较。
- 证据不能预测有益动作时，回到证据设计；不得通过阈值堆叠制造表面提升。

## 9. Artifact 真值契约

- offline branch 一律 `runtime_authorized=0`，使用 `counterfactual_applied` 表示是否真实执行。
- runtime action 的 authorized、consumed、abstained 和 invalidation reason 必须来自真实 ledger，
  不能写固定值。
- context、action instance、checkpoint、seed schedule、artifact 文件都必须有稳定 hash。
- schema/protocol 升级时拒绝旧 artifact 参与新汇总，不能静默兼容旧语义。
- manifest 必须记录 context/arm 数、FE 汇总、parity、完整性 gate 和文件 hash。
- `results/` 是可重生成输出，默认不提交 Git；报告必须给出可定位的结果路径。

## 10. 代码与目录边界

- `src/arac/evidence/`：Phase1 证据抽取，不执行动作。
- `src/arac/policy/`：selector、冻结协议和动作选择契约，不直接操作 benchmark。
- `src/arac/actions/`：确定性动作实例和执行器，不内嵌 selector。
- `src/arac/backends/`：HCC 状态捕获、branch adapter、runtime ledger。
- `experiments/pilots/exp_*`：唯一实验入口；一个研究问题一个目录。
- `configs/` 或实验内唯一 config：当前有效协议，避免第二事实源。
- `docs/`：稳定设计与验证说明；不要为每次尝试新增平行版本。
- `references/`：论文和外部只读证据。
- `.codex/tmp/`：临时材料，不提交。

优先升级现有稳定入口，不新增 `v2/v3/final_new/latest_fixed` 平行文件。版本历史交给 Git；
废弃实现应删除，必须保留的历史证据放入明确的 archive/references 并说明原因。

## 11. 实现与调试规则

- 默认使用简体中文沟通；代码、CSV、Markdown 和配置统一 UTF-8。
- 修改前先阅读入口、相似实现、测试和 HCC 真源；不凭描述重写核心公式。
- 保持小步、局部修改，不顺手重构无关代码，不回滚用户已有改动。
- Debug-first：先复现并定位根因，让 mismatch、FE drift、bad shape 和 stale artifact 显式失败。
- 不允许静默 fallback、伪造 parity、固定 truth 字段或用宽泛异常处理掩盖失败。
- 连续三次遇到同类失败时，停止重复尝试，重新归纳证据和假设。
- 不硬编码密钥，不提交缓存、日志、大型结果或 `.bak` 文件。

## 12. 验证、提交与交付

修改后按风险运行：

1. 聚焦 pytest。
2. Ruff/type/lint（项目已配置时）。
3. 相关 mechanical smoke 或 artifact validator。
4. 全量 pytest。
5. `git diff --check`。
6. `git status --short`，确认没有误提交 `results/`、缓存、临时文件或用户无关改动。

每完成一个可验证版本，创建描述真实目标的本地 Git commit。只暂存当前任务相关文件。
`git push` 属于外部状态变更，必须在用户明确确认后执行；推送失败时报告 commit 和原因。

最终交付必须说明：

- 本次回答了哪个研究问题；
- 修改了哪些稳定接口；
- 运行了哪些验证及其结果；
- 当前证据只支持什么结论、不支持什么结论；
- 下一阶段是否被 action/evidence/selector gate 授权。

## 13. 一句话准则

先证明动作在同 checkpoint、同 FE 下具有真实 headroom，再证明 Phase1 证据能够识别它，
最后才训练 selector；任一前置门不通过，就停在该层重新设计。
