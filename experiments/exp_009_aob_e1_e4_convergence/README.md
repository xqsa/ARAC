# exp_009 AOB E1-E4 收敛曲线

这个实验入口生成与论文风格相同的 E1-E4 best-so-far 收敛曲线。当前协议固定为：

- `seed=1` 的单次真实运行；
- `3,000,000` FEs/算法/问题；
- 四条实线：`ARAC-v33.8`、`HCC_ES`、`MMES`、`RDDSM_CMAES`；
- 不使用历史最终值、论文数值或均值插值恢复曲线；
- 每条曲线的 CSV 是从真实 `fitness_record` 计算得到的 best-so-far 采样。

运行：

```powershell
python -m experiments.exp_009_aob_e1_e4_convergence.run `
  --output-dir results/exp_009_aob_e1_e4_convergence_seed1_3m `
  --cases E1 E2 E3 E4 --seed 1 --max-fes 3000000 --jobs 4
```

只重画已有数据：

```powershell
python -m experiments.exp_009_aob_e1_e4_convergence.run `
  --output-dir results/exp_009_aob_e1_e4_convergence_seed1_3m --plot-only
```

算法定义写入 `figure_metadata.json`，每条运行的原始审计文件位于
`runs/<case>/seed_<seed>/<method>/`。这张图是 single-seed 过程图，不是 25-run mean，
也不用于替代正式统计结论。
