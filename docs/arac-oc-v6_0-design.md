# ARAC-OC v6.0 设计预注册（双轴状态 + 贡献衰减中断）

日期：2026-08-21
状态：**预注册，未实现**。本文档在 v6.0 任何代码落地之前冻结设计决策。
上游证据：G-ARAC-OC-EVIDENCE-CLOSURE 全部结论（CONCLUSION.md）、
`docs/lit-review-2026-08-21.md`、G11 双轴回放（`artifacts/oc_two_axis_replay_gate/`）。

## 1. 动机与证据边界

- 根因诊断（lit-review §一 + G4c/G5 交叉验证）：v4→v5.3 的跷跷板源于
  **单轴（material/非material）规则链表达不了"难度 × 贡献"双轴状态**。
  v5 层同时携带 R6 的收益（0.836）与 S5 的灾难（73×）。
- G11 回放（离线，gate51c 收据）：
  - **已验证半边**：保护方向成立——S5-CTP 三个 seed 全部落
    difficult/contributing（贡献排名恒为第一）；两轴分配规则给 CTP 的模拟份额
    57.1%，落在获胜的 off 臂实际份额（62–66%）与 v5.3 饥饿份额（20.6%）之间
    的正确一侧；S5-AOR 落 easy/flat（释放），R6 的 aor/ctp 落 easy/contributing
    （正常 exploit，与 v5 层唯一获胜模式一致）。
  - **未解决半边（如实记录）**：喂血/释放判别失败——按预注册的"改进到达间隔"
    难度代理，零全局增益的 episode 无法区分"易且已榨干（释放）"与"未显现
    （供血）"；local progress 粗粒度也不分离（探索性检查，非 gate 结论）。
    R2/A3 的象限跨 seed 不稳定，与"两层调度器都无法赢它们"一致。
  - 预注册判定为 NOT SUPPORTED as specified（P2 操作化歧义：路线图把
    R2-AOR 同时标注为"平坦臂"与"供血象限"，数据把它稳定放在供血象限）。

## 2. v6.0 状态模型（每 episode 连续双轴）

```text
recent_contribution(ep)   CCFR 式最近两窗增益均值，跨 episode 秩归一
                          （AOS rank credit；strict-best archive 保持
                          extreme credit 语义不变）
difficulty(ep)            两个分量的凸组合（权重为版本化配置）：
                          a) 改进到达间隔的滑动中位数（秩化）
                          b) 窗口级贡献衰减斜率（CCFR3）：最近 k 窗的
                             local+global 增益对数斜率；斜率 ≤ 0 持续
                             m 窗 → 衰减确立
```

关键设计决策：**喂血/释放判别必须用分量 (b)（窗口级衰减），不得用分量 (a)
单独判定**——G11 已证明间隔代理在零增益 episode 上无判别力。

## 3. 策略函数（减法目标）

单一两输入分配函数替代以下机制（净减）：

| 被替换机制 | 替代来源 |
|---|---|
| material 单轴状态机 | 双轴象限（protect/exploit/feed/release） |
| w1 固定窗 ladder + rung | CCFR3 贡献衰减中断（无固定窗参数） |
| adoption grace | 衰减检测天然等待（增益到达尺度无关） |
| ticket/challenger/escalation 三车道 | UCB 式选择：score = μ_rank + c·sqrt(ln N / n_ep)（AOS） |

分配：w(ep) = base + α·[difficult ∧ top-contributing] − β·[¬difficult ∧ ¬contributing]，
α=2.0、β=0.75、base=1.0（G11 已用同参数；版本化，校准门独立调）。
预算 pulse 由同一函数驱动，废除独立 pulse 机制。

## 4. 保留不动的东西

- strict-best archive / exact FE / fail-closed / receipt+state hash 契约；
- Phase-I 结构发现与 OverlapCheckpoint 不可变语义；
- 完整候选仲裁（FEA 先例）与 kernel 接口（kernel v3 代码保留可选）；
- production selector（v6.0 只在 gate 中运行，绝不直接接入）。

## 5. Gate 52 协议（判定标准，预注册）

镜像 gate51c：同 cases/seeds/standalone 复用、同 total FE、同 fail-closed 审计。
通过标准（全部同时）：

1. S5 median ratio（对 standalone 最优）≤ 1.10，worst ≤ 2.0；
2. R6 median ratio ≤ 0.90（不得回吐 v5 层在 R6 的收益超过容差）；
3. A3 全部 seed 审计通过且 median ≤ 1.05；
4. R2 median ≤ 1.10（历史最好为 v4/v5.3 的 1.28/1.30）；
5. **fresh-seed 象限复现**：新 seed 上 S5-CTP 的 difficult/contributing 放置
   与贡献第一名复现 ≥ 2/3（G11 的已验证半边必须在未见数据上复现）；
6. 机制数净减（以 receipt 字段与 audit 项计数为准，写进 confirmation）。

失败语义：任一失败 → v6.0 不接入，失败项写入 CONCLUSION 遗留清单；
不许在 gate 内改参数重试。

## 6. 诚实的风险边界

- 难度分量 (a) 在 R2/A3 的跨 seed 不稳定 → (b) 为主的判别 + 跨 seed 平滑
  只是缓解，不保证解决；R2 可能仍然无解（standalone CTP 本身 1.0）；
- 文献机制为机制级借鉴，CCFR/CCFR3 实验均在 disjoint 分解上，贡献定义
  嫁接在 ARAC-OC 的 global materiality 口径；
- UCB 项的 c 与衰减窗口 (k, m) 是新超参，必须走独立离线校准门，
  不得在 gate52 上调参。
