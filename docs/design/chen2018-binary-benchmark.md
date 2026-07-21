# Chen 2018 二进制测试函数移植

- 日期：2026-07-21
- 执行者：Codex
- 状态：独立 benchmark 生成器；未接入 HCC/AOB action-validation 主协议

## 来源与语义

本实现移植陈园等人在《一种高维不可分测试优化问题构造方法》中提出的二进制 LSGO
测试函数发生器。代码依据为
`C:\Users\83718\Desktop\继承\WYQ\test_function.zip` 中的 MATLAB 源码；论文 PDF
用于核对分组、连续性与欺骗性定义。论文给出的官方发布页是 MathWorks File Exchange 的
[Binary test function set](https://www.mathworks.com/matlabcentral/fileexchange/66523-binary-test-function-set)，
2026-07-21 核验页面标题与条目编号可访问。原始代码采用 BSD-3-Clause 许可，许可文本保存在
`src/arac/benchmarks/CHEN2018_LICENSE.txt`。

一个实例由以下内容共同确定：维数、分组上下限、欺骗参数 `alpha`、是否打乱变量顺序、
随机种子、模板串和完整分组。实例生成后这些内容保持不变，并由 `instance_hash` 绑定。
这修复了公开 `test_function.m` 在每次目标评价时重新随机分组/排列、导致目标函数非平稳的
问题；师姐早期实验脚本也明确要求把排列和分组放在评价循环之外。

原 MATLAB 目标以 `-dimension` 为最优值。ARAC 接口将它平移为非负 error，使已知最优值
为 0，与 AOB 的最小化口径一致；`legacy_objective()` 保留原始数值，便于对照。

## 使用

```python
from arac.benchmarks import Chen2018BinaryProblem, Chen2018Spec

spec = Chen2018Spec(
    dimension=1000,
    min_group_size=2,
    max_group_size=5,
    alpha=0.8,
    permuted=True,
    seed=20260721,
)
problem = Chen2018BinaryProblem.generate(spec)
errors = problem([[0] * 1000, list(problem.template)])
manifest = problem.to_manifest()
```

`to_manifest()` 输出完整实例；`from_manifest()` 会核对 schema、字段集合和 SHA-256，拒绝
旧版本或被修改的实例。候选必须是形状为 `(dimension,)` 或 `(n, dimension)` 的二进制
数值数组，连续实值输入不会被静默取整。

## 验证与边界

使用 MATLAB R2024 对公开源码执行 6 维、等分 3 维、`alpha=0.5` 的定向对照，三个候选的
原始目标分别为 `-6`、`-5.4`、`-3.45`；Python 的 `legacy_objective()` 与之相同。
单元测试还覆盖 1000 维不等分/不连续实例、最优值、固定种子复现、manifest hash 和非法
输入失败路径。

该测试函数是二进制问题，不能直接交给当前基于连续 CMA-ES/MMES 的 HCC runner。它目前
只能作为独立 synthetic benchmark 使用，不属于真实 AOB 主集，不得进入 AOB 的 SBS、VBS、
cluster bootstrap 或 action-gate 结论。固定 case catalog、二进制优化器适配和实验协议应在
单独的 benchmark-validation 任务中定义。
