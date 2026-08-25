# shared_transaction_v1 — SCST v3.0 执行候选

基线：`arac-recovered-baseline-20260823-v1`（冻结，不动）。
协议：`docs/arac-oc-scst-upgrade-protocol-v3.0.md`（T0 → T1 → T2 → T3 → T4 → T5 阶梯）。
本目录是该协议的唯一执行实现；不改冻结源，生产默认（patch/soft-routing/selector off）不变。

## 失败案例的教训 → 本候选的设计决策

| 历史失败 | 教训条款化 |
|---|---|
| v5.1 P0 区域合并伪影（粗 RDG 种子序，~8%/对/seed） | T0 前置端点细化拆分修复（见下），Stage 1 本身不动 |
| v5.0 H0 链式三区域拒绝（链变量证据被相邻链接共享变量污染） | T0 内逐对目标验证（chain_pair_isolation 诊断的管线化） |
| Gate 54a 一致性不可辨识 | 内核不做任何分类；median/mean 无权威候选 |
| shared-patch P1 零挂载点（AOB conforming → 永不触发） | T1 先审计"下游确实重读 incumbent"才允许挂载 |
| S1 宿主结构惰性（CTP 纯重排无因果表达点） | 事务只挂在自然阶段边界，绝不插入 coverage 内部 |
| v4–v5.3 在线反馈跷跷板 | 内核无状态、固定 8 FE、零在线反馈信号 |
| Gate 53 派发路径零接受 | AOB 是预注册的"预期静默"（T4），不做 conforming 收益主张 |

## 模块

- `split_repair_discovery.py` — soft-RDDSM 发现管线（复制自冻结仪器）+ 三个插入件：
  1. **端点细化合并修复**：每块细化首/末变量；双侧严格子集（精确覆盖+互不包含）或
     单侧覆盖（合并签名：块首即共享变量）+ 第三种子补集验证 → 拆分。采纳时同时
     替换确认软覆盖。拆分交集归低位块（升序主分配约定）。
  2. **逐对目标验证**：超边目标集用 3 FE 残差探针逐个验证（残差剔除全部共享嫌疑变量）。
  3. **证书派生 relations**：checkpoint 关系 = 认证超边图（strength=证书数），
     解决 soft-DSM 边池在 generator v3 上零跨块边的问题（p0 收据佐证 relation_count=0）。
- `t0_structure_certificate.py` — T0 gate：6 cell × 5 seed（20270501-05），
  P=1.0 / R≥0.9 / merge=0 / 拒绝=0 / 证书图森林且度≤2 / 精确 180k / 双跑重放一致。
  激活 cell 仅 chain4-strong、pairs3-strong（与 H0 同策略；hub3 度 3 按准则不激活）。
- `t0_structure_certificate_protocol_v1.json` — 冻结协议（dsm_budget=110k，
  烟测校准的预算余量，阈值未动）。

## 烟测记录（gate seeds 未触碰，2026-08-24）

| cell/seed | P | R | merge | adopted | 备注 |
|---|---|---|---|---|---|
| chain4-strong/20270601 | 1.0 | 1.0 | 0 | 0 | 修复前 R=0.333（16 条三区域拒绝），目标验证后 24/24 |
| pairs3-strong/20270401 | 1.0 | 1.0 | 0 | 1 | v5.1 P0 的天然合并 seed，被端点修复拆回 10 块 |
| hub3-strong/20270602 | 1.0 | 1.0 | 0 | 0 | 记录不激活 |
| chain4-mild/20270603 | 1.0 | 1.0 | 0 | 1 | mild 也有合并，同样被修复 |

设计迭代（发生在烟测阶段，gate 判据从未回调）：
- 第一版 DSM 强边拆分修复**不可用**——DSM kNN 候选池不采样块内对，强边分量全单例；换成端点细化。
- overlap 判据第一版用 `min_residual_size`（3）——但每链接 shared_width=8，真拆分被误拒；
  改为**互不包含**（overlap < min(两小组)），预算截断的包含式假拆分仍被正确拒绝。
- T1 重锚判据第一版要求"下游首 session 锚 == 边界入口 incumbent 哈希"——过严：SMP rescue
  入口的斜率探针自身可能改进 incumbent（首战 3 臂 reanchor=False 触发，战役即停、产物删除、
  判据修正后全新重跑）。正确判据是**活锚定**：session 构造锚 == 其出生时刻 timeline incumbent
  （这恰是事务写回需要的传播性质——下游拾取任何 live 改进）。timeline 在 install 时播种基行。

## T0 正式判定（2026-08-24，gate_passed=true，result_hash b84e5423…）

6 cell × 5 seed 全部 precision=1.0、recall=1.0、merged_region_count=0、
rejected=0、证书图森林；v5.0 链式失败（recall 0.333）与 v5.1 合并伪影
（recall 0.667）在 gate seeds 上全部消失；5 次天然合并（chain4-mild×2、
pairs3-strong×1、hub3-mild×2）全部被端点细化修复。t1_authorized=true。

## 执行入口

```bash
PYTHONPATH='.;src' ./.venv/Scripts/python -m experiments.upgrade.shared_transaction_v1.t0_structure_certificate
```

解释器纪律：必须 `E:/ARAC/.venv`（libscipy_openblas 栈），U1 事故条款。
