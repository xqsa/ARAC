# 二进制重叠 LSGO F08/F15 诊断实验设计

日期：2026-07-13

执行者：Codex

状态：用户已确认

## 1. 目的

`exp_010` 在 F08 和 F15 上没有稳定触发 ARAC 动作，因此不能仅凭零收益判断
方法无效。本实验用受控消融拆分两个假设：

1. **策略门控受限**：isolate 动作如果被强制执行，能够改善当前搜索，但在线证据
   没有达到策略触发门槛；
2. **底层算子受限**：即使 isolate 被执行，当前单 bit 严格改善搜索也无法跨越
   F08/F15 的 deception valley，必须使用联合变量变化。

实验是机制诊断，不是晋级实验，不修改全局 policy 阈值，也不把 block-flip 结果
包装成 ARAC 正向收益。

## 2. 依据与边界

当前 backend 在 `src/arac/backends/binary_lsgo.py` 中使用单 bit 翻转和严格改善接受。
`isolate_conflicting_relation` 只改变共享变量的 owner/write 权限，不改变候选生成
方式或接受规则。

继承代码 `C:\Users\83718\Desktop\继承\一种高维不可分测试函数构造\程序\program\二进制\SGA_main.m`
使用种群选择、交叉和变异，候选可以同时改变多个 bit。该实现是参考证据，不在本次
直接移植 MATLAB 工具箱依赖；本实验只增加一个最小、可解释的联合变化对照。

本实验不：

- 修改 `E:\HCC-main` 或桌面继承目录；
- 将 final objective、历史结果、problem family label 或 template 信息注入 runtime
  evidence 或 action dispatch；
- 改变现有 ARAC policy 的阈值；
- 宣称 block-flip 是论文 baseline 或 ARAC 的正式动作；
- 取代 `exp_010` 的结果或 promotion gate。

## 3. 实验矩阵

固定问题：`BLSGO-F08`、`BLSGO-F15`。

固定 optimizer seeds：`20260713`、`20260714`、`20260715`、`20260716`、`20260717`。

每次执行总预算：`2000 FE`，Phase I 占 `20%`，即 `400 FE`；Phase II 为 `1600 FE`。

每个问题和 seed 运行四条 lane，共 `2 x 5 x 4 = 40` 条执行：

| Lane | Phase I | Phase II | 作用 |
| --- | --- | --- | --- |
| `native_single_bit` | 原生单 bit | 原生单 bit + 严格改善 | 底层当前基线 |
| `native_group_block` | 原生单 bit | 当前组全部变量同时翻转 + 严格改善 | 联合变化能力对照 |
| `forced_isolate` | 原生单 bit | 强制 `isolate_conflicting_relation` + 单 bit | 策略动作能力对照 |
| `arac_policy` | 原生单 bit | 现有 `decide_action()` + 单 bit | 当前方法行为 |

四条 lane 使用同一问题实例、同一 optimizer seed、同一初始向量和同一 Phase-I
objective。Phase-II 每次候选只计一次目标函数调用，所有 lane 必须正好消费 `2000 FE`。

### 3.1 block 算子定义

`native_group_block` 在 Phase II 按拓扑组轮询。一次 proposal 复制当前全局向量，
将当前组中的所有 eligible variables 同时取反，调用一次完整目标函数；只有目标值
严格改善时才接受。共享变量使用 native 默认写回语义，不使用 ARAC evidence，也不
读取 template。该算子只用于诊断底层“联合变化是否有用”。

当组没有 eligible variable 时，必须抛出可定位异常，不能空转消耗预算或静默退化为
single-bit。

### 3.2 isolate lane 定义

`forced_isolate` 使用与 `exp_010` 相同的初始状态和 Phase-I 统计，但在 Phase II
直接注入合法的 `ActionDecision(ISOLATE, "isolate_conflicting_relation", ...)`。
它只测试 isolate backend binding，不代表 runtime policy 会在真实证据下选择该动作。

## 4. 代码边界

### 4.1 backend

在现有 `BinaryLsgoExecutionRequest` 增加显式的 Phase-II proposal operator 参数，
默认保持 `single_bit`，以保证 `exp_009` 和 `exp_010` 行为不变。允许值只有：

- `single_bit`；
- `group_block`。

在 `BinaryLsgoExecutionResult` 增加结构化 proposal trace，至少包含：operator、
proposed count、accepted count、multi-bit proposed count、multi-bit accepted count、
maximum accepted flip width。trace 只记录优化器在线行为，不包含 final error 或历史
结果。

`group_block` 是 optimizer operator 参数，不扩展 `ActionFamily`，也不改变
`SUPPORTED_ACTIONS`。动作语义和 proposal operator 分离，避免把诊断算子误认为 ARAC
动作。

### 4.2 实验入口

新增 `experiments/exp_011_binary_lsgo_diagnostic/`：

- 固定 F08/F15、五个 seed、四条 lane；
- 复用 `run_binary_lsgo()`，不复制优化逻辑；
- 检查 lane 初始化一致、Phase-I objective 一致、FE 账本一致；
- 写入 `run_results.csv`、`case_summary.csv`、`diagnosis.json`、`manifest.json`；
- manifest 记录协议、固定种子、代码 hash、输入 hash 和 claim boundary。

### 4.3 文档

更新 `experiments/README.md`，加入 `exp_011` 的运行命令和“机制诊断、不可作晋级
结论”的边界说明。除必要的设计、计划和测试外，不新增第二套 backend 或第二个
结果事实源。

## 5. 诊断指标与判定

每个 case 记录四条 lane 相对于 `native_single_bit` 的 offline relative gain，
以及 proposal trace。诊断报告同时给出每个 case 的五 seed 计数和中位数，不以单次
win 作结论。

报告使用以下可解释标签：

- `optimizer_limited`：`native_group_block` 在至少 3/5 个 seed 上严格优于
  `native_single_bit`，且至少 3/5 个 seed 接受过多 bit proposal；同时
  `forced_isolate` 没有达到同样的改善频率；
- `policy_limited`：`forced_isolate` 在至少 3/5 个 seed 上严格优于
  `native_single_bit`，而 `arac_policy` 在至少 3/5 个 seed 没有消费 isolate
  动作；
- `mixed`：上述两个条件同时成立；
- `inconclusive`：两者均不成立，或样本无法区分。

这些标签是本实验的 offline blocker classification，不是总体性能声明。若 block
lane 改善但没有被 ARAC policy 消费，仍只能说明联合变化对当前 scaffold 有帮助，不能
说明完整继承 SGA 已被复现。

## 6. 测试与验证

按 TDD 添加测试，覆盖：

1. proposal operator 参数只接受两个合法值，默认行为与旧 lane 一致；
2. block proposal 一次只消耗 1 FE，并报告正确翻转宽度；
3. `native_group_block` 与 `native_single_bit` 在相同 seed 下初始 hash 和 Phase-I
   objective 一致；
4. 四条 lane 恰好消费 `2000 FE`，无预算违规；
5. forced isolate 的动作和语义 diff 正确，且不改变 proposal operator；
6. runner 固定输出 40 行、四 lane、两个 case、五个 seed，并且重复运行字节一致；
7. diagnosis 分类使用显式输入计数，不读取 forbidden runtime fields；
8. 现有完整测试集不回归。

验证命令至少包括：

```powershell
python -m pytest tests/test_binary_lsgo_backend.py tests/test_exp_011_binary_lsgo_diagnostic.py -q
python -m pytest -q
python -m compileall src experiments/exp_011_binary_lsgo_diagnostic tests
git diff --check
```

## 7. 交付顺序

1. 提交本设计文档；
2. 用户复核设计文档；
3. 写入实现计划；
4. 在隔离 worktree 中按 TDD 修改 backend、实验入口、测试和 README；
5. 运行小预算测试，再运行 canonical `2000 FE` 诊断矩阵；
6. 复核结果和精确 diff 后提交实现版本。推送远程仓库前另行确认。
