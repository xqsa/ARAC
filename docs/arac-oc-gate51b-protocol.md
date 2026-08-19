# Gate 51b 协议：v4 冻结 3M 配对——正式性能判定

日期：2026-08-17（夜）
状态：预注册
前置：Gate 51-0 定标冻结（gate51-0-20260817）；Gate 51a 机制
筛选通过（80 项检查全绿）。

## 1. 研究问题

> (a) OC(v4, handoff ON) 是否 4/4 case 非劣于最优 standalone
>     （≤1.05×，消灭灾难性退化）？
> (b) 是否至少一个 case 严格胜（≤0.98×，极小数值差不计胜）？
> (c) 严格胜 case 上组合归因是否成立（≥2 episode material +
>     ON < OFF）？

## 2. 设计

- Case：A3 / R2 / S5 / R6；Phase-I = Gate 50 缓存 checkpoint
  （seed 20260845，哈希核验）；action_seed = 20260845；精确 3M。
- 24 cell = 4 case × 6 臂：
  - 四个 standalone 首次生成后进入稳定缓存
    `artifacts/oc_phase_aware_gate51b_standalone_v1/`（150k 分段
    instrumented，每段误差轨迹入收据——51c anytime/AUC 的对照数据）；
    终值必须与 Gate 50 standalone 收据**逐位对账**；
  - v4 OC handoff ON / OFF（同 checkpoint、同配置）。
- v4 配置 = 冻结定标表（w1=75k、h*=450k、开发上限 0.55、
  保底 0.10、cold_start 0.25、probe_min 20k、r=2、K=3）；
  `scheduler_version = v4.1.1`（v4.1 的运行时性能修订；正文原写
  v4.0，该首跑已失败并归因，见附录修订；产物目录随版本派生）；
  逐 cell manifest 戳。
- 运行纪律：逐臂容错（单臂失败不炸全场）；cell 落盘可续。

## 3. 判定（三层，预注册）

**协议层**（全部 cell）：terminal 3M 精确；standalone 逐位对账；
OC 臂 13 项调度审计全过；manifest 一致；探针 4 先行；FE 对账。

**性能层**：
1. `not_worse_all`：4/4 `OC_ON ≤ 1.05 × best_standalone`；
2. `strict_win_exists`：≥1 case `OC_ON ≤ 0.98 × best_standalone`。

**互补层**（在严格胜 case 上）：
3. `two_episodes_material`：≥2 个不同 episode 在首次采纳 handoff
   后产生 material global gain；
4. `on_off`：该 case `ON < OFF`；且**无任何 case** 上
   `ON > 1.10 × OFF`。

gate_passed = 协议层 + 性能层 + 互补层全部成立。

**归因清单（R6 型回归预注册）**：严格胜 case 上输出——
challenger/escalation 段 material 率与 gain 分解；ON−OFF 差按
段类型（exploit / challenger / ticket / escalation）分解；
600k/1M/2M/3M anytime 表（OC 与 standalone 对照）。
若 R6 型胜利退化为非劣，唯一预注册杠杆 = challenger 窗口
非对称（v4.1，版本隔离流程），不回退 max-2 与强制 challenger。

## 4. 失败条款

性能层 1 失败 → 按收据归因（ticket/递增/轮转各车道的账本与
调度轨迹），不针对 R2/S5 写特判；后备杠杆 BIPOP 尾部预留按
版本隔离流程启用（v4.1）。固定 seed 的 51b 此后只承担冻结
回归对照，确认职责归 fresh seed 的 51c。

## 5. 运行方式

当前 i9-14900HX 工作站推荐只使用 8 个 P-core worker，避免慢 E-core
形成数小时尾波：

```powershell
.venv\Scripts\python.exe -m experiments.oc_phase_aware_gate51b `
  --workers 8 --pin-p-cores
```

若旧 Gate 51b 目录中已有 standalone cell，可在首次建立稳定缓存时导入；
每个 cell 都会重新核验 checkpoint、Gate 50 终值、3M 终端和分段 FE 账：

```powershell
.venv\Scripts\python.exe -m experiments.oc_phase_aware_gate51b `
  --import-standalone-from artifacts/oc_phase_aware_gate51b `
  --import-standalone-from artifacts/oc_phase_aware_gate51b_v4_1 `
  --workers 8 --pin-p-cores
```

`--pin-p-cores` 的逻辑核 `0-15` 映射是本机硬件特定设置；换机器时先确认
CPU 拓扑。稳定缓存预热后，后续 scheduler-only 迭代会自动复用 16 个
standalone cell，只重新运行 8 个 ON/OFF cell。

产出：`artifacts/oc_phase_aware_gate51b_v4_1_1/`（cells/ + confirmation.json）；
standalone 参考臂写入独立稳定缓存目录。

## 附录：v4.1.1 运行时性能修订（2026-08-17）

本修订不改变候选生成、FE 账本、strict-best、调度判定或 standalone 数值
语义，只减少执行与审计开销：OC 状态不再保留调度器不用的逐 FE
`best_trace`，step 已生成的快照在收据边界复用，标量候选复用全空间缓冲区；
24 个 cell 按慢任务优先提交。standalone manifest 只绑定其 runner 和动作
依赖，不再因调度器/报告脚本修改而重复生成。

## 附录：v4.1 修订（2026-08-17 深夜，51b 首跑失败后按版本隔离流程）

v4.0 首跑失败归因（收据证据）：全局 exploit streak 缺失（rate 重选
绕过 max-2，P3 从未触发，AOR 递增窗饿死）+ ticket 大收益污染
leader 选举。v4.1 修订三处，均在确认门前、同版本迭代：

1. exploit_streak 全局计数：任意混合的两连续 exploit 段强制开放
   challenger（规则 3 本意）；
2. rate 历史排除 ticket 段（协议单元不是速率样本）；
3. P3(a) 公平轮转：最少递增授予优先，私有信用降为平局打破
   （防高信用臂耗尽开发上限、低信用晚熟臂永不开梯——R2 复盘
   预判的下一失败模式）。

判据不变。v4.0 收据保留为回归对照；standalone cell 与调度器版本
解耦，首次生成后从稳定缓存直接复用。

**P0 修订（同夜，评审拦截）**：51b 输出目录改为按调度器版本隔离
（`oc_phase_aware_gate51b_v4_1`，随版本常量自动派生），并加 cell
复用强制校验——OC 臂须携带当前 `scheduler_version` 与调度树
manifest，standalone 臂须携带 standalone 依赖 manifest（episode
机器码 + benchmark + standalone runner；不含调度器树）。任何不匹配即重跑：
混合版本的 gate 结果比没有结果更糟。v4.0 的 21 个 cell 留在旧目录
作历史对照；v4.1.1 的 OC 臂写入新版本目录，standalone 只在稳定缓存
缺失或其自身依赖 manifest 漂移时重跑。

## 附录 3：v4.4 预注册（2026-08-18 晨，用户裁决后）

**健康度条件公平（用户评审裁决的正式表述）**：两个调度状态——
MATERIAL_RUN（leader 上一段 exploit material：连续运行，累计
sampling debt，仅 75k 节奏梯档可入）与 DISCOVERY_RUN（leader
上一段非 material：节奏事件走 pending 梯道优先，间隙开放未采样
bootstrap，最后轮转）。门控信号 = leader 上一段 exploit 的全局
strict-best 增益（不得用私有增益）。

**四 overflow cell 参照锚定切换**：R2/R6 的 ctp/gcb 判定的逐位
锚改为当前树全预算确定性值（overflow_reference.json，含出处：
Gate 50 值是 sigma 安全修复前产物；三好一差 0.08%）。基线一致性
问题与调度创新分开处理。

**判定口径**：严格 frozen-version——只有冻结版本的固定结果可作
性能证据；各版本最佳值仅作机制诊断。v4.4 为本预注册下的首个
冻结候选：实现后只重跑 R2/S5/R6/A3 诊断门，不再按单版最好结果
回调策略。
