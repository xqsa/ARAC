# ARAC-OC v5.3 设计文档：几何验证阶梯 + 收养宽免 + 全验证窗统一阶梯化

日期：2026-08-20。状态：**已冻结（2026-08-20，跑批前；评审修正①-④已并入）**。
承接：`docs/handoff-2026-08-20.md` §四.P0-1（v5.3 两处外科手术）的**升级替代方案**。
本文件即协议附录预注册文本；用户批准后冻结，跑批前不改一字。

---

## 0. 一句话设计

把所有 exploit 类验证窗（adaptive_lock / protected_runway / horizon_promotion 后续）
统一为**逐 episode 的几何阶梯**（75k → 150k → 300k，material 升档、flat 释放），
外加**收养后首窗释放宽免**（事件驱动，每次收养一次）；全部附带
**v5.2 反事实标记**收据，使机制效果本身成为可审计证据。

---

## 1. 动机：v5.2 三个失败模式与一个结构性缺陷

| # | 失败 | 机制根因（源码位置） | 量级 |
|---|---|---|---|
| F1 | S5 健康 seed 灾难回退（4.67→39.65） | 收养重锚后首个 75k 窗是暖机窗（双平），`_execute` 的 `released = not material` 单窗二元判定将其误杀（episodes.py:2126）；v5.1 轨迹证明回报在 (75k, 300k] 到达 | 9.1× OFF |
| F2 | R2/20260901 平坦 lock 烧 300k | adaptive_lock 授予 `segment_window` 全段（episodes.py:1741），未经 w1 钳制 | 300k 纯暴露 |
| F3 | v4.2→v5.2 反复跷跷板 | 单一全局耐心参数（窗口粒度）对各臂**收益到达时间尺度异质**的结构失配；CTP 是脉冲型收益 | 跨版本 |
| F4 | 深层 | 逐窗贡献判定在重叠问题上归因失真（文献 §3.4 负面结果）；单窗 ln(1.01) 判定对脉冲方差无免疫力 | 论文级 |

**设计原则（从 F1–F4 抽出，预注册）**：

1. **非对称损失原则**：误杀一条已显现的 material 链（损失可达 9.1×）远比误容忍
   一个平坦臂（有界 75k）昂贵 → 判定规则应在便宜档位偏向耐心。
2. **尺度匹配原则**：判定窗口必须随臂的已证明产能放大，而不是全局一刀切。
3. **事件驱动原则**：暖机是收养事件的已知伪影，不是臂的证据——豁免挂在
   事件上，不挂在固定窗数上。
4. **机制最小化原则**：阶梯 + 宽免两个机制吸收原 v5.3 两补丁 + F3，机制总数
   不增反减。

---

## 2. 机制规范（精确到伪代码级）

### 2.1 验证阶梯（geometric verification ladder）

每个 episode 维护阶梯档 `rung[e] ∈ {0, 1, 2}`（per run）。窗口：

```
w(rung) = min(w1 * 2^rung, segment_fes) = 75k, 150k, 300k
```

适用于**全部三种 exploit 类授予**：

| 授予点 | v5.2 窗口 | v5.3 窗口 |
|---|---|---|
| adaptive_lock（票据 material 后首次验证） | 整段 ≤300k | rung 0 = 75k |
| horizon_promotion（杠杆 2 显现验证） | 2·w1 = 150k | rung 1 = 150k（语义不变） |
| protected_runway（leader 续跑） | 一律 75k | w(rung[e]) |

状态转移（在 `_execute` 的 exploit 分支，替换 `released = not material`）：

```
flat = (g_gain <= material_log_gain)
if material:
    rung[e] = min(rung[e] + 1, RUNG_MAX)      # 升档，cap 300k
    released[e] = False
    grace_armed[e] = False                    # 修正②：重锚后 material 即自证产能
elif grace_armed[e]:                           # 见 §2.2
    grace_armed[e] = False                     # 消费宽免：不释放、不降档
    receipt.grace_consumed = True
else:
    released[e] = True                         # 释放（同 v5.2 语义）
    rung[e] = 0                                # 降回底层
```

- rung 在**收养重锚时归 0**（状态已变，历史档位失效）；释放时归 0；
  领导权竞争期（未释放）内持续保留。
- promotion material 后进入常规竞争时 rung=1（其验证窗本身就是 150k）。
- 释放重置规则不变：任一 episode 的完整 horizon 预留付清时全体
  `released=False`（episodes.py:2227-2230）。

**暴露上界分析（修正①：区分平坦暴露与 material 累计）**：
- 永久平坦臂：rung 0 一窗即释放，暴露 75k（与 v5.2 相同，S5/20260901 型
  灾难 seed 仍快速释放）。
- 脉冲臂（S5/20260902/03 型）：宽免 + 150k 二档窗把"暖机+回报"捆绑判定
  （v5.1 的 0.3379 到达区间 (75k, 300k] 被 rung-1 窗覆盖）。
- 已证明产能的 leader 在 300k 顶档窗内不再被 75k 级脉冲方差单点处决。
- **material 累计无界且应当无界**——健康任职期合法积累长程打磨
  （v5.1 S5 健康 CTP = 1.22M），这是保护对象不是暴露。有界的是
  **每任职期的平坦暴露**：≤ 宽免窗 75k + 至多一个终态平坦窗 ≤300k =
  **375k**。该界成立的关键：收养重锚将 rung 归 0（§2.1），故宽免窗
  恒为 rung-0 窗 ≤75k；且每一档的到达都由前一档 material 收据背书。

### 2.2 收养宽免（adoption grace，事件驱动）

```
_execute 内 _handoff(departed, episode) 返回 adopted=True（发生重锚）时：
    grace_armed[episode] = True        # 挂在事件上，与 grant_kind 无关
```

- 消费：该 episode **其后第一个 flat 的 exploit 窗**（见 §2.1）；material 窗
  清除 `grace_armed`（重锚后已 material 即自证产能，暖机歧义消除）。
- 上界：每次收养事件至多一次宽免 = 至多 75k 额外暴露（收养将 rung 归 0，
  宽免窗恒为 rung-0 窗）；收养次数本身受 handoff 审计约束。
- 收据：`grace_armed` / `grace_consumed` 布尔字段入段收据。

### 2.3 影子诊断（不改行为，只记录；为 v5.4 决策供数）

1. **速率样本**：每个 exploit 窗记录 `gain_rate = g_gain / consumed`
   （跨窗大小归一），离线复算 CBCC3 式事件驱动让位规则的反事实轨迹。
2. **v5.2 反事实标记**：凡宽免或高档窗阻止了一次"在 v5.2 语义下必然发生
   的释放"，收据打 `would_release_v5_2=True`——机制效果直接可计量。
3. **重叠度入档**：`DO = |shared_variables| / D` 及 per-block 重叠统计写入
   confirmation.json（v5.4 的 DO 参数化设计输入）。

### 2.4 边界情形（逐条预注册）

| 情形 | 规则 |
|---|---|
| 窗 > 剩余预算 | 钳到剩余；低于 `max(min_step_fes, min_window_needed[e])` 则该 episode stuck（现有语义不变） |
| 末端尾巴（generation 对齐的 SMP） | 现有 min_window_needed 机制不变；阶梯窗同样受其下钳 |
| rung-1/2 窗被 challenger 预留节奏打断 | 不打断：runway 段内 reservations 只记账（现有 S5 保护语义保留） |
| released leader 通过 horizon 付清复活 | rung 已为 0，从底层重新自证（与"重锚归 0"一致） |
| promotion pending 与 runway 交错 | 不变（v5.2 杠杆 2 机制原样保留） |
| 宽免与 stuck | stuck episode 不参与授予，宽免状态随 stuck 冻结 |

### 2.5 审计新增四条（跑批前冻结）

1. **阶梯单调性**：同一任职期内 rung 只在 material 时 +1、释放/收养时归 0；
2. **平坦暴露有界（修正①）**：任一 episode 任一任职期（释放到释放）的
   flat exploit 累计 FE ≤ w1 + segment_fes = 375k（material 累计不受此界
   约束——长程打磨是保护对象）；
3. **宽免一次性**：每次 `adopted=True` 事件至多对应一次 `grace_consumed`；
4. **反事实一致性**：`would_release_v5_2=True` 的窗在 v5.2 语义下必满足
   `not material`（确定性重算校验）。

### 2.6 测试（先玩具后跑批）

1. 确定性翻转玩具：脚本化收益到达时刻（100k / 200k 到达的脉冲臂）验证
   阶梯 + 宽免保护链不断；负对照（永久平坦臂）验证 75k 暴露；
2. **lock 期脉冲场景（修正③，v5.2 未测面）**：收益到达点落在 lock 窗
   (75k, 150k] 的玩具，验证 rung-0 lock 不伪释放 S5 型链；**预注册决策
   规则：若玩具显示 lock@rung-0 脆弱，lock 起始档取 rung 1（150k），
   不给 lock 加宽免**；
3. 宽免一次性计数测试（含收养后首窗即 material 的清除路径）；
4. lock 窗 w1 化断言：adaptive_lock 授予 ≤ 75k；
5. v5.1/v5.2 契约测试迁移 + 退役入口断言照旧；
6. 全量测试零新回归。

---

## 3. 文献映射（设计依据；机制并非凭空发明）

### 3.1 几何阶梯 ← IPOP/BIPOP-CMA-ES 的 doubling trick
重启预算/种群逐次翻倍是多模态黑盒优化的金标准重启策略
[Loshchilov, Schoenauer, Sebag — *Alternative Restart Strategies for CMA-ES*,
arXiv:1207.0206；Hansen — BIPOP-CMA-ES, BBOB-2009]。我们把 doubling 从
"重启间"移到"验证窗间"：同一臂的判定粒度随已证明产能翻倍。同时 CMA-ES
传统用**停滞判据组合**（TolHistFun/TolStagnation 等 9 条）而非单窗判决
[Yamaguchi et al. 2017；arXiv:2606.09220 的定量分析]——支持我们用
阶梯+事件替代单点阈值。

### 3.2 事件驱动让位 ← CBCC3（Omidvar, Kazimipour, Li, Yao, CEC 2016）
CBCC3 规则：组件优化持续到其**即时贡献跌破任一其他组件最近记录贡献**；
只用最近贡献、不积累；探索概率 p1（~0.05 最优，0–1 均稳健）。v5.3 以影子
诊断形式预注册该规则的反事实复算（§2.3-1），v5.4 凭收据决定是否转正。
CCFR [Yang et al., IEEE TEVC 2017] 的"平滑贡献 + 停滞归零 + 全体停滞重启"
同样是释放-复活语义的先例。

### 3.3 轨迹判定而非瞬时判定 ← Freeze-Thaw BO（Swersky et al. 2014）
FT-BO 用部分学习曲线外推渐近表现来决定冻结/解冻，而非按当前窗收益
处决慢热臂 [Swersky, Snoek, Adams — *Freeze-Thaw Bayesian Optimization*,
arXiv:1406.3896]。阶梯是同一思想的参数-free 近似：用递增窗口代替
曲线外推，避免在机制内引入代理模型。

### 3.4 负面结果（设计约束）← Sun, Li, Ernst, Omidvar, TEVC 2019
RDG3 论文明确报告：**贡献自适应分配无助于重叠问题**，归因于组件依赖性
使贡献归因失真。推论（预注册为本设计的约束条件）：materiality 必须保持
**全局 strict-best**（现有设计正确，不改），且任何逐窗贡献比较只能作为
影子诊断，不在 v5.3 转正。

### 3.5 非对称暴露 ← Hyperband / successive halving（Li et al., JMLR 2018）
多档耐心对冲（brackets）与"快速淘汰明显差者、对有产者逐级加码"的
预算结构。阶梯 = 单臂时序上的 bracket 序列。

### 3.6 重叠进入调度（v5.4/v5.5 储备）← HCC（Qiu et al., arXiv:2503.21797）
- DO 参数化预算配比：`GloFEs = (0.2 + 4/3·DO)·TFEs`（其 Eq. 6-7）→ v5.4
  用 Phase-I 实测 DO 调制全局臂（AOR/SMP）vs 结构臂（CTP/GSS）的预算基线；
- 贡献加权共享变量仲裁（其 Eq. 8）→ v5.5 以**软收养**替代硬重锚，
  从病因上消除暖机窗（v5.3 的宽免是对症药，软收养是疫苗）。
另：DG2 系（DG2, Omidvar et al. TEVC 2017）与 CBCCO/DOV 的共享变量
策略、OEDG（arXiv:2404.10515）的可调重叠度基准可作 51d 泛化第二测试床。

---

## 4. 验证协议（51c v5.3 复判，沿用版本纪律）

1. 起新版本号 `v5_3`；v5.2 入口退役方式同 v5.1（保签名、调用即报错）；
2. 新建 `experiments/oc_phase_aware_gate51c_v5_3.py`；standalone/Phase-I
   复用冻结 v5.1 产物（锚不变）；只重跑 OC 臂（~2 小时）；
3. 判定三层照旧：中位数/最差门 + ON<OFF + 严格胜 seed 统计；
4. **判定前冻结本文件**；跑批中任何修改即废弃本轮；
5. 预测（预注册，跑批前写下）：
   - S5 健康 seed 回 ~3.7–4.7×；灾难 seed 仍 ~46.6×（快速释放正确）；
   - R2 向 v5.1 回落且保留 promotion 增益；R6 维持双条款达标
     （中位数 ≤1.05 为可证伪检查，余量仅 0.001，最敏感）；
   - `would_release_v5_2` 计数在 S5 健康 seed ≥ 1，在灾难 seed = 0 或
     暴露有界。

---

## 5. 风险与回退

| 风险 | 评估 | 回退 |
|---|---|---|
| 阶梯在 R2 型高方差 case 上过度耐心 | ON<OFF 若恶化（v5.2 已从 3/3 掉到 1/3），反事实标记可定位是哪一档造成 | 阶梯 cap 从 300k 降到 150k 的单参数回退档（预注册为唯一允许的事后档位） |
| **R6 达标余量仅 0.001（修正④）** | promotion material 后 runway 窗 75k→150k，平坦窗每窗多付 75k；R6/20260903 型（0/2 material runway）最敏感 | 预测条款加"R6 中位数维持 ≤1.05"作可证伪检查；回退同上（cap 150k） |
| 宽免被灾难 seed 滥用 | 每次收养 ≤75k，上界硬约束 | 无需回退，收据定量报告 |
| 与杠杆 2 交互 | promotion 语义不变；rung=1 入口已对齐 | 杠杆 2 独立开关保留 |

---

## 6. 后续路线（裁决用，不在本轮）

- **v5.4**（调度器冻结后）：DO 参数化预算配比（§3.6）+ CBCC3 让位规则凭
  v5.3 影子收据转正裁决 + 51d 泛化（24×5 vs HCC + OEDG 可调重叠度基准）。
- **v5.5**：软收养（贡献加权共享变量仲裁）替代硬重锚，消除暖机病因。
- **战略层**：S5 的 1.05 门已被收据证明结构性不可达（CTP standalone 需
  ~2.8M，组合最多 ~1.9M）。建议协议修订：S5 单独口径，exploration tax 作为
  被定价的发现进论文；完成标准 = A3 非劣 + R6 达标 + R2 严格胜 seed +
  机制收据链。**此条独立于 v5.3 技术裁决，需用户单独批准。**
