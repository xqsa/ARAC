# ARAC-OC v4 升级实施计划（修订版）

日期：2026-08-17（第二轮修订）
状态：草案 v2——并入第一轮评审五条修正与第二轮评审六条修正；
Gate 51-0 完成前不得冻结为最终 Gate 协议。
前身：v4 升级方案初稿（2026-08-17 会话）；未注明处与初稿一致。

## 修订说明

| # | 修正 | 依据 | 落点 |
|---|---|---|---|
| 1 | 新增 Gate 51-0 显现 horizon 测量，先测后冻结窗口参数 | R2/aor 282k 探针 global/local 双零；参数不能拍脑袋 | §2 新增；§4 定标；§7 顺序 |
| 2 | R2 饿死修复组合：递增开发窗 + challenger 轮转下限 + 私有轨迹信用（仅升权）；BIPOP 尾部预留降为后备杠杆 | ticket 防淘汰不防饥饿；challenger 优先级原指向"未完成 ticket"臂，完成后即饿死 | §4 第 7-9 条；§5 第 4'/8' 条 |
| 3 | R6 回归风险预注册归因 + challenger 窗口非对称杠杆 | max-2-连续 + 强制 challenger 打断 smp 六连 material 节奏 | §7 Gate 51b 归因清单 |
| 4 | 600k 角落 fallback 显式化；25% cap 预算算术表；ON/OFF 判据措辞修订 | 51a 必然走到全 ticket 不可负担角落；50c 中 ON 仅 R6 胜 OFF | §4 第 9 条与算术表；§7 51b |
| 5 | 极值信用诊断字段（不参与任何排序） | AOS 文献极值信用；为 v5 留数据，零规则成本 | §6 收据字段 |
| 6 | 预算双账本：cold_start_probe_cap + exploration_and_development_cap | 25% 单 cap 与递增窗豁免互相矛盾，探索开销会被藏进 exploitation | §4 |
| 7 | protocol_mature 与 evidence_revealed 概念分离 | 语义成熟 ≠ 信号显现；AOR standalone horizon ≠ handoff 后 contextual horizon | §2、§3、§5 |
| 8 | 定标改累计覆盖 sum(window) ≥ h*，冻结 K | w1 ≥ max(h*) 使首窗即整个显现期，几何递增不可负担 | §2 |
| 9 | 调度裁决优先级表 P0-P5；私有信用带 handoff_epoch | 多规则边界冲突需唯一答案；archive 注入不得计入私有收益 | §5、§6 |
| 10 | 版本与确认隔离：确认后改动 → v4.1，fresh seed 才能确认 | 同批 seed 调参重跑会把 51b 变成开发集 | §7 |
| 11 | 判据消歧：严格胜 ≤0.98×；median ≤1.05 + 最差 seed ≤1.10；新增 Gate 51d | 极小数值差不算胜利；3 seed 不足以支撑普遍性声明 | §7 |
| 12 | 定位表述：block-order scope 的协作超启发式 | 变量级 scope 未完成，不得过度声明 | 总结 |

## 总结

目标是修复 Gate 50c 暴露的"短期收益代表长期价值"错误，同时保留
已经验证的全局 materiality、strict-best、handoff 和收据审计机制。

v4 采用规则型 phase-aware GCB，不立即引入学习模型；先在实验入口
完成验证，双门通过后再接入正式 `run_arac_oc()`。

**定位表述（诚实边界）**：v4 的核心创新是 overlap-aware、
phase-aware 的动态协作超启发式调度（dynamic cooperative
hyper-heuristic）附可审计 handoff；当前 scope 为**块序级**
（block-order），"按变量级冲突范围精确分配预算"是后续升级，
不在本计划宣称范围内。

锁定决策：

- 调度核心：规则型 phase-aware scheduler。
- 晚熟保护（四件套）：maturity ticket（防淘汰）+ 递增开发窗
  （防看不见）+ challenger 轮转下限（防饿死）+ 私有轨迹信用
  仅升权（修 S5 型私有轨迹隐藏）。
- 探针策略：冷启动上限（最小探针 + 首轮 ticket）25%；开发总
  上限 exploration_and_development_cap（覆盖全部非 leader 授予）
  由 Gate 51-0 定标；每个 episode 保证最小可执行探针。
- 参数纪律：窗口基数 w1、递增级数 K、开发上限、exploitation
  保底等数值由 Gate 51-0 实测定标后冻结，初稿数值仅为量级占位。
- 版本纪律：任何确认门之后的机制改动递增版本号（v4.1、v4.2…），
  旧产物保留不覆盖，最终性能确认只允许 fresh seed。
- 阶段信息：由 episode state 显式暴露统一进度契约。
- handoff：继续使用当前角色化 baton；AOR 保持 `fresh_by_design`。
- 验证方式：冻结 seed 4/4 非回归门 + fresh 多 seed 确认门。
- 不新增 top-k elite archive，不按 case ID 写规则，不在 v4 引入
  训练型预测模型或轨迹外推（v5 方向）；BIPOP 式尾部预留为
  预注册后备杠杆，不在 v4 主线。

## 关键改动

### 1. 先冻结实现和实验基线

涉及：

- `E:\ARAC\src\arac\coordination\episodes.py`
- `E:\ARAC\src\arac\actions\phase2_v2.py`
- `E:\ARAC\src\arac\runtime\phase2.py`
- `E:\ARAC\experiments\oc_action_episode_gate50c.py`

执行内容：

- 修复当前 Ruff 未使用 import/变量问题。
- 重新生成 Gate 50c 全部 8 个 OC cell，禁止复用混合版本收据。
- 为每个实验 cell 写入 `implementation_manifest_hash`，覆盖调度器、
  state、协议和配置文件。
- 将 v3 收据和结果保留为历史控制，不覆盖原产物。

### 2. Gate 51-0：显现 horizon 测量（新增，先测后冻结）

**动机（50c 收据实证，`artifacts/oc_action_episode_gate50c/cells/*_on.json`
probes 字段）**：

| case/episode | 282k 窗 global_gain | local_gain | 结论 |
|---|---|---|---|
| R2/aor | 0 | 0 | 双零——任何固定短窗排序（含私有信号）看不见它 |
| S5/ctp | 0.0199 | 1.298（四臂最大） | 晚熟信号在私有轨迹，不在全局边际 |
| R6/aor | 1.013 | 1.013 | 该 case 短窗即可见 |

显现 horizon 因 case/episode 而异，窗口参数必须实测定标。

**执行（两次 3M 分段 instrumented 重跑，用既有 v2 episode 机制，
不依赖进度契约）**：

- aor standalone @ R2 checkpoint（seed 20260845，与 Gate 50 收据
  同源），150k × ~19 段；
- ctp standalone @ S5 checkpoint，同规格；
- 每段落盘 per-segment global/local error 与 incumbent 轨迹；
- 终值与 incumbent 必须与 Gate 50 standalone 收据**逐位对账**
  （分段 ≡ 单发已由 recovered 等价性测试与 Gate 50a 证明；
  对账失败 = gate fail）。

**概念分离（重要，防混用）**：

- `protocol_mature`：episode 完成一个合法语义状态单元（§3 契约
  定义）——协议事实，由 episode state 暴露；
- `evidence_revealed`：该 episode 累计开发性授予预算 ≥ 显现
  horizon——证据事实，由调度器账本维护，episode 不感知。
- 两者独立："语义已成熟而信号未显现"（R2/aor 完成 correction
  仍双零）与"信号先于语义成熟显现"都合法存在，调度规则不得
  混用两个谓词。

**分析产出（写入 `docs/arac-oc-gate51-0-protocol.md` 后冻结）**：

- 每 episode 的 err(FE) 轨迹表；
- 显现 horizon `h*(E)` = 私有轨迹首次越过参考水平 E 的 FE，
  E 取该 case 其它臂探针期末误差与 OC 探针期 archive 水平；
- 定标规则（**累计覆盖**，非单窗覆盖）：冻结 w1、递增因子 r=2
  与级数 K，使几何累计 `w1·(r^K−1)/(r−1) ≥ h*_cal × 安全系数`
  （h*_cal = 跨所测 case 取最大；全局常数，禁止按 case 配置）；
  单窗 w1 不要求等于整个 horizon；
- `exploration_and_development_cap` 与 exploitation 保底比例由
  定标表一并给出；
- 已知局限如实声明：AOR 的 standalone horizon 与 handoff 后
  contextual horizon 不等价（archive 注入改变私有基线）；定标
  采用 standalone 曲线，偏差方向与幅度记录在案；
- 若累计覆盖在剩余预算内不可达 → 如实记录，v4 以 BIPOP 式尾部
  预留作为后备杠杆（见 §7），仍受开发总上限约束。

**约束**：只读既有 checkpoint 与收据 + 两次新跑；不修改调度器。

### 3. 统一 episode 进度契约

在 `E:\ARAC\src\arac\runtime\phase2.py` 增加公共 `EpisodeProgress`
类型，至少包含：

```text
episode
phase
consumed_fes
next_boundary_fes
min_step_fes
maturity_target_fes
protocol_mature
contract
```

所有可调度 state 必须实现 `progress()`（`protocol_mature` 只陈述
协议事实；evidence_revealed 是调度器账本概念，不属于进度契约）：

- CTP：`coverage -> polish`，协议成熟 = 完成 coverage 并执行一个
  polish window。
- GSS：`warmup -> coordination/continuation`，协议成熟 = 完成一个
  语义 sweep 窗口。
- SMP：`block_sweep -> visit`，协议成熟 = 完成一个完整 visit。
- AOR：`global_correction`，协议成熟 = 完成一个独立 correction
  window。

进度契约必须能够回答：

1. 当前是否仍处于冷启动/覆盖阶段；
2. 到下一个合法切换边界还需要多少 FE；
3. 当前请求是否足以完成一个合法状态单元；
4. 在当前总预算下是否可以达到成熟条件。

不得由 `episodes.py` 按 episode 名称维护第二套阶段映射。

### 4. 探针、成熟度与递增窗口预算

**双账本预算（消除口径冲突）**：

- `cold_start_probe_cap = 0.25`：只覆盖最小探针 + 首轮 maturity
  ticket（冷启动税，与 50b 探针税口径可比）；
- `exploration_and_development_cap`（51-0 定标）：覆盖全部
  **非当前 leader** 的授予（challenger、递增开发窗、后续
  ticket）——真正的开发总上限，探索开销不得绕道藏入
  exploitation；
- 计账规则：按"段授予时刻的 leader 身份"归类入账本，
  `grant_kind` 与 leader 标记入收据，账本可重算。

锁定值：`cold_start_probe_cap = 0.25`、`probe_min_fes = 20_000`
（50b 纪律延续）。

定标值（Gate 51-0 后冻结）：maturity 窗基数 w1、递增级数 K、
`exploration_and_development_cap`、exploitation 保底比例、递增
因子（默认 2，Successive Rejects 式几何递增）。初稿占位
`maturity_window_ratio = 0.05`、`[20_000, 64_000]` 仅为量级
参考，**禁止在 51-0 产出前使用**。

600k 算术示例（诚实记录）：Phase-II 按 600k 计，冷启动上限
150k，四个最小探针已耗 80k，余 70k 通常不足以支付四个 ticket
→ `maturity_unaffordable` 与 §4 第 9 条 fallback 是设计内路径，
不是异常。

调度顺序：

1. 每个 episode 先获得一次最小可执行探针；
2. 探针 + 首轮 maturity ticket 合计不得超过
   `cold_start_probe_cap`（25%）；
3. 探针顺序继续由 B/W 证据决定，C 只保留诊断；
4. 探针不足以完成最小窗口时必须显式失败，不能静默缩短；
5. 探针完成后，为 episode 分配不可拆分的 maturity ticket：
   - CTP：coverage 边界 + polish window；
   - GSS：一个完整 sweep/continuation 窗口；
   - SMP：一个完整 visit + maturity window；
   - AOR：至少两个 maturity window，用于保护 late correction；
6. 若总预算不足以支付某个 ticket，收据记录
   `maturity_unaffordable`，该 episode 不得被判定为"早期无价值"；
7. **递增开发窗（新增，修 R2 饿死）**：非当前 exploitation
   leader 的 episode 获得的开发性授予（ticket 后续、challenger、
   验证窗），第 k 次长度 = `w1 × 2^(k-1)`；可续会话使分段与
   单发逐位等价（recovered-aor/smp 等价性已证），跨窗累积等价
   于一段连续长跑；
8. 递增开发窗计入 `exploration_and_development_cap`（非 leader
   授予），且每次授予必须满足：授予后剩余预算 ≥ 终端
   exploitation 保底（保底比例由 51-0 定标）；
9. **600k 角落 fallback（新增，显式定义）**：全部 ticket 不可
   负担且无成熟 episode 时——(a) 可负担 ticket 优先执行；
   (b) 拥有 ≥2 段历史者按 recent_rate 排序；(c) 否则轮转；
   无可执行对象 → 响亮失败收据。

**预算算术表（新增）**：协议文档必须给出 per-case 的四 ticket
预期成本与 25% cap 的可负担性表（由 `progress()` 契约 + 51-0
产出填写）；结构性不可负担如实记录并声明降级顺序（AOR ticket
保底优先）。

600k 实验中，所有 episode 只保证最小探针；3M 实验中必须尽可能
完成四个 episode 的 maturity ticket。

### 5. 替换累计收益排名

删除当前基于全程累计 `gain_per_fe` 的 exploitation 排名。

改为以下确定性规则：

1. 未完成 maturity ticket 的 episode 优先完成自己的 ticket；
2. 已成熟 episode 使用最近两个完整 segment 的全局收益率：

```text
recent_rate =
sum(global_gain in last 2 segments)
/
sum(consumed_fes in last 2 segments)
```

3. 每连续两个 exploitation segment，强制开放一个 challenger；
4. **challenger 优先级链（修订，修"完成 ticket 后被饿死"）**：
   (a) 未完成可负担 ticket 的 episode；
   (b) 轮转指针在"已完成 ticket 且非当前 leader"集合内轮转——
   每 N 个 challenger 段内每个非 leader episode 至少出现一次
   （N = 非 leader episode 数；adaptive pursuit 概率下限的
   确定性版）；
   (c) 无可执行对象 → 响亮失败；
5. 单个 episode 最多连续执行两个 exploitation segment；
6. 切换后进入一个 segment 的回避期，防止立即振荡切回；
7. `materiality` 只负责信用更新、冷却和诊断，不再决定"当前
   episode 是否永久保留调度权"；
8. **私有轨迹信用（新增，仅升权）**：窗口内私有 log-gain
   极值/速率作为确定性排序信号，只允许把 episode 提升进开发/
   验证队列（challenger 优先、ticket 提前），**不得降低全局
   material leader 的 exploitation 权**——Gate 50b "global
   materiality 驱动调度"的设计教训不回退（gcb local 0.85 /
   global 0 被正确降权）。具体形式（极值 vs 速率）由 51-0
   定标，51a 前冻结；轨迹外推属预测类，明确排除到 v5。私有信用
   只在当前 **handoff epoch** 内计——archive 注入引起的基线变化
   不得计入 episode 自身收益（epoch 标记入收据）；
9. 所有排序只允许使用 Phase-I 证据、当前进度、近期收益、私有
   轨迹信用、剩余预算和 episode 状态，不得读取 case 名称或
   最终结果。

**调度裁决优先级（边界冲突的唯一答案）**：

```text
P0 硬约束：预算合法性 / 探针先于 exploit / 最小可执行单元
   （违反 = 响亮失败，不进入以下层级）
P1 未完成且当前可负担的 maturity ticket（含递增后续 ticket，
   按探针序；不可负担 → 记 maturity_unaffordable 并移出本层）
P2 leader exploitation：leader 连续 exploit 段 < 2 时授予
P3 强制 challenger（leader 连续 = 2 或 leader 处于回避期），
   候选 = 非 leader 全集，依序筛：
   (a) evidence_revealed = false 且可负担递增窗者（私有信用
       升权排序）——本类不受 cooldown 约束（证据事件优先）；
   (b) 其余非 leader：cooldown 过滤 + 轮转指针；全部在
       cooldown → 最先到期者；
   (c) 空集 = 响亮失败
P4 排序信号（在 P2/P3 内部决定对象）：recent_rate 决定 leader；
   私有信用只作用于 P3(a) 的升权
P5 fallback：§4 第 9 条三分支（P1-P3 无可执行对象时）
```

每一步裁决必须且只产生一个授予对象；任何规则组合下无解 =
P0 响亮失败，不允许静默降级。

修复对照（诚实版）：

- S5/GSS 短期正收益永久粘住 → max-2-连续 + 强制 challenger；
- S5/CTP 晚熟被全局边际隐藏 → 私有轨迹信用仅升权 + 递增窗；
- R2/AOR 双零饿死 → ticket（防淘汰）+ 递增窗（防看不见）+
  轮转下限（防饿死）；
- CTP 探针止于 coverage 边界 → 进度契约语义边界；
- handoff 后单次边际收益误配立即转向 → recent_rate + 回避期。

### 6. 保留并扩展收据审计

`EpisodeSegmentReceipt` 增加：

```text
progress_before
progress_after
recent_rate
maturity_ticket_id
maturity_committed
challenger
cooldown
remaining_budget
grant_kind            # exploit / ticket / challenger / escalation
grant_index           # 该 episode 第 k 次开发性授予
window_fes
leader                # 该段授予时刻的 leader 身份
handoff_epoch         # 私有信用计账 epoch（archive 注入即换纪）
cumulative_development_fes   # evidence_revealed 的账本依据
evidence_revealed
max_global_log_gain_window   # 极值信用，仅诊断
max_local_log_gain_window    # 极值信用，仅诊断
```

`EpisodeProbeReceipt` 增加：

```text
probe_contract
maturity_target_fes
maturity_unaffordable
max_local_log_gain_window
```

**standalone 收据（新增要求）**：51-0 与 51b 的 standalone 臂
必须输出分段 err(FE) 轨迹（Gate 50/50c 只有终值，51c 的
anytime/AUC 需要对照曲线）。

`schedule_hash` 必须覆盖：

- 调度器版本；
- 配置 hash；
- implementation manifest hash；
- progress 字段；
- ticket/commitment 字段；
- 近期收益与私有信用字段（含 handoff_epoch）；
- grant_kind/grant_index/window/leader 字段；
- 双账本（冷启动/开发）累计数；
- 现有 handoff、state hash、snapshot hash 和 FE 对账字段。

原有保护保持不变：

- global strict-best 单调；
- local/global 双收益分离；
- handoff 不改变私有 FE；
- AOR 状态保持 fresh；
- CTP/GSS OOB baton 拒绝；
- segment 索引连续；
- probe 先于 exploit；
- terminal FE 精确；
- 全局误差单调；
- 收据 hash 可重算。

### 7. 实验和 Gate 顺序

新增协议文档：

- `E:\ARAC\docs\arac-oc-v4-upgrade-plan.md`（本文件）
- `E:\ARAC\docs\arac-oc-gate51-0-protocol.md`（51-0 产出后冻结）
- `E:\ARAC\docs\arac-oc-gate51-protocol.md`

实验分四层：

**Gate 51-0：显现 horizon 测量（新增，前置）**

- 两次 3M 分段 instrumented standalone 重跑（aor/R2、ctp/S5）；
- 产出窗口参数定标表，冻结后 51a 及以后不得改动；
- 只作参数定标，不构成任何性能结论。

**Gate 51a：机制筛选**

- 600k；
- A3/R2/S5/R6；
- 固定 checkpoint；
- 验证 progress、maturity ticket、递增窗、轮转下限、私有信用
  仅升权、challenger、FE 和 handoff 收据；
- 600k 角落 fallback 三分支必须被至少一个 case 实际走到并入收据；
- 只作机制筛选，不宣称性能优越性。

**Gate 51b：冻结 3M 配对**

- 重新生成四个 standalone（含分段轨迹收据）；
- v4 handoff ON/OFF；
- 同 checkpoint、同 seed、同总 FE；
- 要求：
  - 4/4 `OC_ON <= 1.05 * best_standalone`（非劣容差）；
  - 至少一个 case **严格胜**：`OC_ON <= 0.98 * best_standalone`
    （2% 容差，极小数值差不得计入胜利；与非劣容差不混用）；
  - 严格胜利 case 必须有至少两个 material episode；
  - **ON/OFF（修订措辞）**：严格胜 case 上 `ON < OFF`（因果
    归因必要条件）；且无任何 case 上 `ON > 1.10 * OFF`；
  - 全部协议检查通过。
- **归因清单（新增，R6 回归风险预注册）**：严格胜 case 上——
  challenger 段 material 率与 gain 分解；ON−OFF 差按段类型
  （exploit / challenger / ticket）分解。若 R6 型组合胜利退化
  为非劣，唯一预注册杠杆 = **challenger 窗口非对称**（challenger
  段预算 < exploit 段，配置项），不回退 max-2-连续与强制
  challenger 规则本身。

**Gate 51c：fresh 多 seed**

- A3/R2/S5/R6；
- 每个 case 至少 3 个未参与规则设计的 seed；
- 每个 seed 重新生成 Phase-I checkpoint、standalone、OC-ON、
  OC-OFF；
- 要求（消歧后）：
  - 每 case **中位数** ratio ≤ 1.05；
  - 每 case **最差 seed** ratio ≤ 1.10（单 seed 灾难防护；原
    中位数退化条款被中位数条款包含，删除）；
  - 至少两个 case 中位数严格胜（≤ 0.98 × 最优 standalone）；
  - 组合胜例在至少两个 fresh seed 上满足 ON 优于 OFF；
  - 报告 600k、1M、2M、3M 的 anytime 曲线和 AUC（对照 =
    standalone 分段轨迹收据）。
- 51c 定位声明：3 seed/case 是**机制确认**，不支撑"普遍优越"
  声明；普遍性证据归 Gate 51d。

**Gate 51d：泛化验证（新增，论文证据层）**

- 51c 通过后启动，规模与论文声明匹配：
  - 关键 case ≥ 10 seed；
  - 跨拓扑：AOB-24 底函数族 × 重叠度网格（41b 对照口径）；
  - 跨规模抽查（如 500-D / 2000-D）；
- 判据随协议单独预注册；论文普遍性表述必须与 51d 覆盖面一致。

**版本与确认隔离（强制，保护确认门独立性）**：

- 任何确认门（51b/51c）之后的机制改动——包括启用 BIPOP 式
  尾部预留、challenger 窗口非对称等后备杠杆——必须递增版本号
  （v4.1、v4.2 …），`scheduler_version` 与 manifest hash 同步；
- 旧 gate 产物全部保留（artifacts 目录带版本后缀），不覆盖、
  不复用混合版本收据；
- 改动版本必须重过 600k 机制筛选（51a 级），最终性能确认
  **只允许 fresh seed**；固定 seed 的 51b 从此只承担冻结回归
  对照，不承担确认职责，不得在同批 case/seed 上试到通过；
- 连续两轮 fresh 确认失败 → 停止调参，按收据归因，R2/S5 如实
  作为开放边界进入论文收口。

若 Gate 51b 因晚熟排序仍败：后备杠杆 BIPOP 式尾部预留按上述
版本隔离流程启用（v4.1），不针对 R2/S5 编写特判；若 Gate 51c
失败：保持实验入口，不接生产入口。

### 8. 正式入口接线

只有 Gate 51b 和 Gate 51c 均通过后，才修改：

- `E:\ARAC\src\arac\overlap_core.py`
- `E:\ARAC\src\arac\coordination\loop.py`

将正式 `run_arac_oc()` 接入 phase-aware v4 scheduler，并保留显式
配置：

```text
scheduler_policy = "phase_aware_v4"
scheduler_version = "v4.1.1"        # v4.1 机制 + 运行时性能修订
cold_start_probe_cap = 0.25
probe_min_fes = 20000
escalation_factor = <51-0 定标>
escalation_grants_k = <51-0 定标>
maturity_window = <51-0 定标>
exploration_and_development_cap = <51-0 定标>
exploitation_reserve_ratio = <51-0 定标>
```

v3 仅作为历史/消融实现保留，不作为静默 fallback。正式入口仍必须
输出同一套 global/local gain、handoff、progress、state hash 和
FE 对账字段。

## 测试计划

新增或修改定向测试：

- 四个 episode 的 `progress()` 字段和阶段边界；
- CTP 探针不能在 coverage 结束后立即被判定为成熟；
- AOR 在早期零收益后仍能获得 correction ticket；
- GSS 连续收益不能绕过 challenger 和最大连续 segment 限制；
- recent-rate 排名不读取累计历史收益；
- maturity ticket 不可拆分且预算不足时显式记录；
- probe cap、minimum probe 和 FE 对账；
- handoff 角色语义、OOB 拒绝、AOR fresh 独立性；
- schedule hash 在新增字段下可重算；
- frozen code manifest 不一致时实验直接失败；
- 51-0 分段重跑与 Gate 50 standalone 终值逐位对账；
- 递增开发窗几何：第 k 次开发性授予 = `w1×2^(k-1)`，预算不足
  显式失败，不静默缩短；
- 轮转下限：episode 完成 ticket 且 recent_rate 垫底时，N 个
  challenger 段内必须被调度（R2 饿死定向测试）；
- 私有信用仅升权：gcb 型 local>0/global=0 假粘性不得因此获得
  exploitation 权（Gate 50b 教训回归测试）；全局 material
  leader 不因他人私有信用失去调度权；
- 600k 角落 fallback 三分支各有定向测试；
- 极值字段存在于收据但不出现在任何排序输入（断言）；
- ON/OFF 判据措辞与 confirmation.json 输出字段一致；
- 调度裁决优先级边界：leader 达 max-2 + 全部 challenger 在
  cooldown + ticket 不可负担 → P3(b) 最先到期者，唯一解断言；
- 私有信用 handoff epoch：archive 注入前后私有收益分段计账，
  不跨 epoch 累计；
- 双账本计账：leader/非 leader 授予归类可重算，冷启动与开发
  上限各自触发的显式失败路径；
- protocol_mature 与 evidence_revealed 互不推导（各自独立翻转
  的定向用例）；
- 严格胜 0.98 与非劣 1.05 容差在判据实现中不混用；
- scheduler_version / manifest 不一致时 gate 脚本拒绝运行。

## 默认假设

- v4 不引入机器学习训练器、预测模型或轨迹外推；私有轨迹信用
  限确定性统计量（极值/速率）。这些作为 v5 研究方向。
- BIPOP 式尾部预留为预注册后备杠杆（51b 晚熟仍败时启用），
  不在 v4 主线。
- v4 不引入 top-k archive，继续使用当前单一 global strict-best
  baton。
- 当前 block-order scope 仍如实标记为块序级；变量级执行范围
  另立后续任务。
- 目标是总体非回归、降低最坏退化并保留组合胜例，而不是承诺对
  每个实例严格超过事后 oracle standalone。
