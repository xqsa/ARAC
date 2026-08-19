# ARAC 最终方法规范

> **历史状态（2026-08-09）**：本文记录 RF v3 的 AOB 内 proof-of-concept，不再是当前
> 方法规范。当前主线是 `docs/arac-core-method.md` 定义的 ARAC-Core；本文中的冻结
> selector、正式入口和泛化描述只能按历史实验理解。

日期：2026-08-02  
执行者：Codex  
状态：v3 有界标定、选择器冻结与 24 函数 x 25 seed 正式验证已完成

## 1. 方法定义

ARAC 的创新链路是“阶段一证据 -> 一次动作选择 -> 阶段二优化”。AOB 只提供测试目标函数，任何实例名称、函数编号或 A/E/R/S 标签都不会传入 checkpoint、选择器或动作注册表。

```text
AOB numeric objective
  -> Phase I: 180,000 counted black-box FE
  -> 36 identity-blind evidence features + immutable checkpoint
  -> frozen Outcome Selector (one decision)
  -> CTP | SMP | GCB | AOR (one selected action)
  -> Phase II: selected action only
  -> best-so-far error at exactly 3,000,000 FE
```

统一 `EvaluationLedger` 统计所有目标函数调用并维护 strict-best archive。局部动作即使产生较差候选，也不能覆盖已经找到的全局最优记录，因此“局部过程有收益但终值变差”的问题在状态契约层被阻断。

## 2. 阶段一证据

正式协议为 `arac-identity-blind-evidence-v3`。在总预算 3M FE 时，阶段一固定在 180,000 FE 结束：

- 240 FE 景观探针：中心点、随机证据块扰动、关系扰动和四条稠密响应线；
- 最多 174,000 FE 计数结构探测：仅通过黑盒函数值推断变量块和块间关系，并将输出限制为最多 20 个固定宽度结构块；
- 剩余阶段一预算用于 MMES 搜索，形成同一 checkpoint 的 incumbent 和改善特征。

选择器输入共有 36 个数值特征，覆盖误差尺度、探针离散度、非对称性、关系强度、线响应粗糙度、结构块统计和阶段一改善量。特征 schema 不包含 `case_id`、family、series 或人工标签。

当结构探测在预算内无法完成时，协议保留由 240 FE 探针产生的身份盲 fallback blocks/relations，并显式输出 `structural_inference_complete` 特征；这不是按函数类别进行的隐藏回退。

## 3. 离线标签与冻结选择器

训练标签来自真实终值，不来自预设系列动作。对每一个 calibration context：

1. 每个 `case/seed` 只运行一次阶段一，保存并校验共同 checkpoint；
2. 四个动作从同一个不可变 checkpoint 和同一 FE 边界分别执行，不重复阶段一；
3. 每个动作都运行到总计精确 3M FE；
4. 将四个终值转换为相对当次 oracle 的四维 `log10 regret`，作为回归目标；严格最小动作只用于报告分类准确率。

训练 seed 为 `117, 129, 141`，独立 holdout seed 为 `142`。模型选择只依据训练 seed 留一交叉验证，holdout 不参与模型或参数选择。

正式 v3 选择器位于 `artifacts/selectors/outcome_v3/`，为多输出 `RandomForestRegressor`，分别预测四个动作的相对 `log10 regret`，并选择预测 regret 最小的动作。模型参数只依据训练 seed 留一验证确定，holdout 不参与选择。

冻结文件 SHA-256：

- `outcome_selector.joblib`：`45a34c845967a521a8b5b254c667ffa1aff09d386433dc0c757c655263123918`
- `outcome_selector.json`：`ccac2c89694704d81865ea8efd33e085049261a7485c9c9a8f8cb38eb732358f`
- `outcome_selector_evaluation.json`：`acafd720c6af86add8ceab2131e06185903aa4966ec59051dcee5f043cc87fbd`

独立 holdout 结果：

- 严格准确率：`0.6667`
- balanced accuracy：`0.7143`
- 终值在 oracle 1% 范围内：`0.8750`
- 平均 `log10 regret`：`0.01010`
- 最坏 `log10 regret`：`0.13479`
- 最坏 selected/oracle 比值：`1.3639`
- 预设 holdout gate：通过

严格准确率不是唯一验收量，因为多个动作可能终值接近，且逐 seed 的唯一赢家会变化；终值 regret 直接衡量所选动作是否真的造成显著优化损失。

## 4. 四个阶段二动作

- `CTP`（Coverage-to-Polish）：先轮转覆盖全部证据块，再按阶段一关系构造重叠覆盖，进行固定 8 轮块级顺序精修，剩余预算交给终端 MMES。
- `SMP`（State-Memory Persistence）：为每个证据块持续保留 CMA 搜索状态，在多轮块访问间累积方向信息。
- `GCB`（Graph-Conditioned Balancing）：按关系强度与冲突度排序块并协调全空间搜索。
- `AOR`（Adaptive Optimizer Routing）：根据误差尺度与线响应粗糙度选择 Sep-CMA-ES 或 MMES 全空间路线。

四个动作实现相同的 `ActionContext -> ActionResult` 契约，可以从任意受支持问题的同一种 checkpoint 执行。

GCB 对选择器始终只有一个标签。内部只允许以下结构事实分支：

```text
overlap_relation_count == 0  -> zero_relation_global_coordination
overlap_relation_count > 0   -> positive_relation_graph_then_global_coordination
```

内部不读取 R1、R2 或任何 benchmark 实例身份。

## 5. 推理期与预算契约

- 冻结选择器每条轨迹只调用一次；
- 推理期不重新训练、不改阈值、不人工指定动作；
- 推理期不试跑未被选择的动作；
- 阶段一和阶段二共享同一个 FE 账本与 strict-best archive；
- 所有优化器候选在评估前修复到 AOB 的公开搜索边界，账本拒绝任何越界候选；
- 每条轨迹必须在恰好 3,000,000 FE 返回；
- 每条 receipt 固定配置哈希、checkpoint 哈希、选择器哈希、动作结果哈希和终值。

正式入口：

```powershell
.venv\Scripts\python.exe -m experiments.final.run run
.venv\Scripts\python.exe -m experiments.final.run run --resume
```

离线标定入口仅用于从头复现实验，不属于正式推理：

```powershell
.venv\Scripts\python.exe -m experiments.final.run calibrate --resume
```

## 6. 事实源

- 协议、seed 和冻结哈希：`experiments/final/config.json`
- 唯一实验协调器：`experiments/final/run.py`
- 阶段一证据：`src/arac/evidence/phase1.py`
- 结构探测：`src/arac/evidence/structural.py`
- 冻结选择器：`src/arac/analysis/outcome_selector.py`
- 四动作：`src/arac/actions/`
- FE 与动作契约：`src/arac/runtime/`
- AOB 测试集边界：`src/arac/benchmarks/aob.py` 与 `vendor/aob/`
- 正式 v3 有界动作标定：`artifacts/outcome_calibration_v3_bounded/`
- 正式 v3 冻结选择器：`artifacts/selectors/outcome_v3/`
- 正式 24 函数 x 25 seed 结果：`artifacts/final_24x25_v3_bounded/`

## 7. 正式 25-seed 端到端结果

正式评估使用 25 个未参与训练、模型选择或 holdout 的 seed，共 `600/600` 条轨迹，失败 `0`。全部 receipt 通过重新读取验证，全部终止于精确 `3,000,000 FE`，且每条轨迹只执行一个被选动作。动作选择总计为 `AOR=1, CTP=190, GCB=157, SMP=252`。

终值使用“均值 ± 样本标准差”，科学计数法保留两位小数：

| 函数 | 25-seed 终值 | 动作选择次数 |
|---|---:|---|
| A1 | `7.78e+04 ± 1.80e+02` | SMP 25 |
| A2 | `7.86e+04 ± 6.17e+02` | CTP 3, GCB 11, SMP 11 |
| A3 | `7.85e+04 ± 3.90e+02` | CTP 10, GCB 10, SMP 5 |
| A4 | `7.85e+04 ± 4.57e+02` | CTP 11, GCB 10, SMP 4 |
| A5 | `7.85e+04 ± 2.35e+02` | CTP 9, GCB 11, SMP 5 |
| A6 | `7.84e+04 ± 4.65e+02` | AOR 1, CTP 7, GCB 11, SMP 6 |
| E1 | `1.16e+07 ± 9.23e+06` | CTP 24, SMP 1 |
| E2 | `9.52e+06 ± 8.54e+05` | CTP 25 |
| E3 | `1.04e+07 ± 7.40e+05` | CTP 25 |
| E4 | `9.86e+06 ± 8.75e+05` | CTP 25 |
| E5 | `1.20e+07 ± 1.41e+06` | CTP 25 |
| E6 | `1.11e+07 ± 1.16e+06` | CTP 25 |
| R1 | `3.29e+05 ± 4.43e+04` | GCB 5, SMP 20 |
| R2 | `4.98e+05 ± 4.69e+05` | GCB 21, SMP 4 |
| R3 | `6.73e+05 ± 5.39e+05` | GCB 17, SMP 8 |
| R4 | `4.22e+05 ± 1.89e+05` | GCB 24, SMP 1 |
| R5 | `9.33e+05 ± 5.79e+05` | CTP 1, GCB 13, SMP 11 |
| R6 | `4.86e+05 ± 2.68e+05` | GCB 23, SMP 2 |
| S1 | `6.50e+03 ± 2.24e+04` | SMP 25 |
| S2 | `6.95e+03 ± 2.57e+03` | SMP 25 |
| S3 | `7.36e+03 ± 1.23e+03` | SMP 25 |
| S4 | `7.99e+03 ± 1.35e+03` | SMP 25 |
| S5 | `9.28e+03 ± 1.44e+03` | SMP 25 |
| S6 | `1.08e+04 ± 1.26e+04` | GCB 1, SMP 24 |

完整结果位于 `artifacts/final_24x25_v3_bounded/results.csv` 与 `summary.json`。

## 8. 声明边界

正式结果证明了完整的“阶段一证据 -> 冻结选择器 -> 单一阶段二动作”链路可以在同一组 24 个 AOB 函数的新随机轨迹上执行并产生可审计终值。E1、R2、R3、R5、S1 和 S6 的样本标准差仍然较高，因此不能声明所有函数上的稳定性问题已经完全解决。该实验也不能单独证明对未见函数、不同维度或其他 benchmark suite 的泛化能力。
