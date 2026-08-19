# HCC-ES 源码审计：SOTA 不做黑盒交互检测

日期：2026-08-16
来源：https://github.com/Wukong-SCUT/2025_HCC_GECCO
审计对象：`HCC_SRC/HCC-ES.py`（主入口）+ `HCC_SRC/HCC/RDDSM.py`（分解算法）

## 审计结论

**HCC-ES 的 RDDSM 分解不经过任何黑盒函数评价——设计结构矩阵 Θ 直接从
AOB 构造文件读取。**

## 源码证据

### 1. Θ 的获取（HCC-ES.py）

```python
# 直接从文件加载预计算的 0/1 交互矩阵
design_matrix = np.loadtxt(
    f'HCC_SRC/AOB/AOBG/datafile/F{fun_id}-design.txt', delimiter=','
)
```

这是 AOB 基准构造时生成的真值矩阵——不经过任何函数评价。

### 2. RDDSM 调用（HCC-ES.py）

```python
decomposition = Decomposition(design_matrix)
grouping_result = decomposition.decomposition()
```

`Decomposition` 类从 `HCC.RDDSM` 导入，接受已知的矩阵作为输入。

### 3. RDDSM 的本质（RDDSM.py，~126 行）

- `find_paradigm()`：对矩阵的行做哈希分组（相同模式的行归为一组），
  然后合并子集关系
- `combine()`：合并单元素子列表
- `decomposition()`：调用上述两个方法

**没有递归、没有扰动、没有函数评价、没有探针**——纯矩阵操作。

### 4. 黑盒评价的使用

函数评价仅用于**优化阶段**（fitness 调用 `fun`），不用于分解。

## 对论文定位的影响

| | HCC-ES (SOTA) | ARAC-OC v10 |
|---|---|---|
| Θ 来源 | 文件读取（oracle） | 黑盒检测（180k FE 预算） |
| 分解 FE | 0 | 25-70k |
| 分解精度 | 100%（trivial） | 进行中 |
| 解决的问题 | 给定完美分组的优化 | 结构发现 + 协调优化 |

**论文声明框架**：

> 与 HCC-ES [Qiu et al., 2025] 直接读取基准构造信息不同，ARAC-OC
> 在有限黑盒预算内构建分层交互证据。HCC-ES 报告的 AOB 分解精度
> （100%）是在 oracle 信息条件下的结果；ARAC-OC 评估的是在无先验
> 结构信息条件下的实际发现能力，这是 HCC-ES 未涉及的问题。

## Gate 41 的正确对比框架

Gate 41b 的 AOB-24 结果（动作级派发分支）应标注为：

> 在 Phase-I 证据驱动的动作选择下（无构造信息），对 HCC-ES
> （oracle 结构）的战绩为 21/3/0。这不是同层次对比：HCC-ES
> 拥有完美的分解信息，而 ARAC-OC 的派发基于黑盒证据。

## 已有基线的重新标注

| 基线 | 使用的结构信息 | 标注 |
|---|---|---|
| HCC-ES | oracle Θ | 上界参考（oracle-fed） |
| RDG3-CMAES | 黑盒检测 | 同层次竞争者 |
| DG2-CMAES | 黑盒检测 | 同层次竞争者（在重叠上失败） |
| Sep-CMAES / MM-ES 等 | 无 | 非分解基线 |
