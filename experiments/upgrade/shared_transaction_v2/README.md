# shared_transaction_v2 — 结构认证的共享坐标重整（Re-centering）

基线：`arac-recovered-baseline-20260823-v1`（冻结，不动）。
上游证据（只读复用）：v1 的 T0 结构认证、T1 SMP lane 资格普查、T3 根因收据。

## 为什么有 v2（v1 的裁决链）

1. v1-T0：认证仪器修复了 v5.0 链式三区域与 v5.1 区域合并两个历史失败（P=1.0/R=1.0/零合并）。
2. v1-T1：SMP 边界全合格；CTP 判负（写回集中，机制发现）；仪器在 AOB 生产路径零税（逐位恒等）。
3. v1-T3：median/mean 事务 **0/10 接受**，收据级根因：互斥划分下共享坐标只被主块移动，
   两个 owner 的 proposal 快照在 j 上恒同值 → P_j=[v,v] 退化 → 仲裁无输入。
   **结构性结论：冻结四动作从不表达共享变量的值冲突（FEA 式仲裁在此架构上原理性空转）。**
4. v2 由此导出：不仲裁"冲突值"，而是修复"失配"——非主块移动后，j 的全目标切片最优点
   已偏移，把 j **重新对中**。3 点二次拟合，每坐标 3 FE，≤4 坐标，≤12 FE/边界，
   只信完整目标反馈（比 v1 的 median/mean 更无权威），无状态。

## 机制契约（冻结于 screen_protocol_v1.json）

```text
对每个选中坐标 j（规则：证书链接双 owner 新鲜，升序取前 4）:
    f0     = ledger best（0 FE）
    f_plus = F(x; j+δ),  δ = 0.25×span_j   (1 FE)
    f_minus= F(x; j−δ)                      (1 FE)
    若探针本身严格更优 → strict-best 接受（坐标步进）
    否则曲率>0 且顶点在探针区间内部且异于当前值:
        顶点 = j + δ(f_minus−f_plus)/(2(f_plus+f_minus−2f0))   (1 FE, strict-best 裁决)
```

fail-closed：曲率非正/顶点出界/相同值 → 跳过第三次评估，FE 返还记录。

## 判定（与 SCST T3 同形）

≥1 cell geo R<0.95 且 CI 上界<1.0；无 cell CI 下界>1.05；reachability 全臂；
eps_ref 从 v1 screen 的 smp_a0 臂冻结（同 seeds/action seed，确定性逐位等）。
confirmation seeds 20270511-15（screen 过则跑，认证走 v1 T0 仪器 on-the-fly）。

## 文件

- `recentering_kernel.py` — 内核 + RecenteringMount（继承 v1 TransactionMount，仅换发射动作）。
- `screen.py` — screen/confirmation 战役（A0 vs A1，配对统计，paired bootstrap 10k）。
