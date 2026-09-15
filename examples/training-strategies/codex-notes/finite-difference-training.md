# 有限差分训练初步探索

更新日期：2026-09-15。原型实现归档于
[`examples/hidden/finite-difference-training`](../../hidden/finite-difference-training/)。

## 方法与规范命名

训练目标仍包含 energy loss，但用少量随机方向上的 directional derivative 近似完整的
逐坐标 conservative-force loss。

| 原型 estimator | 规范名称/用途 |
| --- | --- |
| `exact` | `all-C` |
| `forward-fd` | `all-forward-FD` |
| `central-fd` | `all-central-FD` |
| `ad-directional` | AD-directional estimator control，不作为正式策略名 |

如果有限差分之后再用完整 exact conservative force 修正，完整流程必须命名为
`2stage-forward-FD-C` 或 `2stage-central-FD-C`，而不是 `all-*-FD`。

## Estimator

对结构 `b`，坐标维数 `d_b=3N_b`。每个 batch、每个结构独立采样单位 Rademacher
方向：

```text
v_bi ∈ {-1/sqrt(d_b), +1/sqrt(d_b)}
```

方向监督使用：

```text
L_D = (1/K) sum_k [sum_b d_b (D_v E_theta + v^T F_label)^2 / sum_b d_b].
```

AD control 的期望严格等于逐坐标 force MSE。forward FD 使用
`[E(R+epsilon v)-E(R)]/epsilon`，central FD 使用
`[E(R+epsilon v)-E(R-epsilon v)]/(2 epsilon)`。FD 只修改 `atom_pos`，保留未扰动图的
edge/three-body connectivity；对模型参数保持可微，但不构造 position-force 二阶反传。

所有方法的 validation 都使用完整 exact conservative forces；FD 只替换训练 loss。

## 实验设置

- MatterSim-1M，single-sol 全量 28,279 个结构；seed 42；25,451 train / 2,828 validation。
- energy/force-directional weight = 200/20。
- Muon LR `8e-5`、weight decay `0.1`；ReduceLROnPlateau 监控 exact
  `val/total_loss`；EMA 和 component gradient-norm logging 关闭。
- energy normalization 与 `elec-refact/Single-Sol` 对齐：residual-ridge reference，
  `PerAtomReferencingNormalizerModule -> PerAtomNormalizerModule`。
- 默认 `K=1`；每个 batch 重采样方向；resume 保存并恢复每个 DDP rank 的 RNG state。
- 默认 batch/GPU 与步长：

| 方法 | Batch/GPU | Epsilon |
| --- | ---: | ---: |
| `all-C` | 8 | n/a |
| AD-directional control | 8 | n/a |
| `all-forward-FD` | 16 | 0.01 Å |
| `all-central-FD` | 12 | 0.1 Å |

float32 数值诊断中，`all-central-FD` 默认步长相对 AD directional derivative 的误差约
0.5%；`all-forward-FD` 的最佳区域仍约 19%，所以 forward FD 更适合作为速度/偏差对照。

## 完整单 epoch 结果

结果目录为
`/net/csefiles/coc-fung-cluster/lingyu/ElectrolyteResults/finite-difference-training/single-sol/one-epoch-8gpu-seed42`。
尽管目录名含 `8gpu`，manifest 证明实际实验使用的是 fung-cluster3 上的 **4×A40
(GPU 0–3)**。四个 run 都完整处理同一 train/validation split 并正常结束。

| 方法 | Batch/GPU | Train (s) | Train speedup | Train+exact val (s) | Full speedup | Peak allocated | Val force MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all-C` | 8 | 526.02 | 1.00× | 543.37 | 1.00× | 20.89 GiB | 0.05148 |
| AD-directional control | 8 | 530.64 | 0.99× | 547.93 | 0.99× | 20.89 GiB | 0.07825 |
| `all-forward-FD` | 16 | 298.28 | **1.76×** | 315.87 | **1.72×** | 29.87 GiB | 0.09309 |
| `all-central-FD` | 12 | 449.08 | **1.17×** | 466.67 | **1.16×** | 35.21 GiB | 0.07992 |

四种方法的 exact validation 时间都在 17.29–17.60 s，差异主要来自训练阶段。
`all-forward-FD` 的完整 epoch 时间降低 41.9%，`all-central-FD` 降低 14.1%；AD control
没有加速，符合其仍需高阶 AD 反传的预期。

一轮后的精度不能满足最终成功标准：相对 `all-C`，AD control、`all-forward-FD` 和
`all-central-FD` 的 force MAE 分别高约 52.0%、80.8% 和 55.2%，均远超 1%。这只说明
单 epoch 的 force 学习效率不同，不能说明收敛后或经过最终 C 阶段后的精度。

## 可得结论与下一步

当前证据支持：forward FD 有明显的单 epoch throughput 优势，central FD 有较温和的
优势。它尚未证明 `2stage-forward-FD-C` 或 `2stage-central-FD-C` 能以更少 GPU-hours
达到 `all-C` 精度。

速度结果还包含 batch-size 收益：global batch 分别为 32、64 和 48。若要归因于
estimator 本身，应补充统一 batch=8 的 control；若研究 compute-optimal recipe，则应先
分别校准每种方法（尤其 `all-C`）的最大吞吐 batch。正式下一步应记录 validation MAE 对
累计 GPU-hours 的曲线，并比较 `all-C`、`2stage-forward-FD-C` 和
`2stage-central-FD-C` 达到相同 force MAE 所需的总成本。

## 原型能力

归档代码包含统一 launcher、完整恢复、W&B+CSV logging、best/last checkpoint、逐 epoch
计时、epsilon calibration、batch benchmark、单 epoch 四方法 benchmark 和结果分析。
它是研究原型，不应直接作为稳定公开 example。
