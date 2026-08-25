# shared_transaction_v5 — 认证链接的联合 coverage（早期预算尺度干预）

基线：`arac-recovered-baseline-20260823-v1`（冻结，不动）。
上游证据（只读复用）：T0 认证仪器、T1 CTP 普查、v1–v4 screen 收据。

## 证据链（为什么是 v5、为什么在 coverage 入口）

| 收据来源 | 结论 | v5 的设计回应 |
|---|---|---|
| T1 CTP 普查 | 交错 coverage 的 strict-best 写回集中在 1–4/10 块，**集中锁定发生在 coverage 内部**；任何链接的双 owner 从未同时上写回流 | 在 coverage **入口**（集中锁定之前）给认证链接的合并 scope 一个联合访问单元 |
| v3/v4 联合救援 | 晚期边界（stateful→rescue）无收益空间：1.2M FE 打磨后的 incumbent 对全新会话零写回 | 干预提前到 incumbent 还很弱的位置（coverage 起点误差 ~54k，coverage 自身 853 次写回） |
| v2 重整 | 12 FE 微窗口终值被扰动方差支配（尺度公理第三次复现） | 预算尺度：每链接 20k FE，从 coverage 自身预算划出（≤25%），宿主总 FE 契约不变 |
| v1 仲裁 | 互斥划分不表达共享变量值冲突 | 联合 scope 让共享坐标与双 owner **一起被优化**（架构中第一次在 coverage 尺度发生） |
| G5（已证明的正例） | 拓扑条件化生命周期是本仓库唯一稳定的收益族 | v5 与之同族：结构认证条件化的生命周期修改 |

## 机制契约（冻结于 screen_protocol_v1.json）

在 `run_persistent_blocks` 入口，对每条 T0 认证链接 (A,B)：
以与 coverage 会话**完全相同的配方**（默认 population=BLOCK_POPULATION_SIZE、
默认 σ、全新会话）在合并 scope A∪B 上运行对齐预算（每链接 20k FE，
总量 ≤ coverage 请求的 25%），随后把剩余预算原样转发给冻结的交错 coverage。
选择条件=纯结构（认证链接；coverage 入口按构造不存在 source-phase proposal）。
接受=ledger strict-best；跨 run 无状态。

## 判定

与 v1–v4 同形：A0/A1 配对（同 checkpoint/ledger 边界/action seed），
≥1 cell geoR<0.95 且 CI 上界<1.0；无 cell CI 下界>1.05；reachability
（联合收据非空、joint_fes≥1）。eps_ref 从 T1 patch-off ctp 臂冻结
（chain4=11.198、pairs3=12.095）。confirmation seeds 20270511-15。
