# Scalability 初步探索：`all-E-only`、`all-C` 与 `all-NC`

更新日期：2026-09-15。原型实现归档于
[`examples/hidden/scalability-test`](../../hidden/scalability-test/)。

## 目的与命名映射

这组实验测量 MatterSim-1M 在 single-sol 不同数据规模上完成一个 training epoch 和一个
validation epoch 的时间，用于估计不同 supervision 的单 epoch 成本。

| 旧代码 mode | 规范方法名 | 状态 |
| --- | --- | --- |
| `energy-conservative` | `all-C` | 已完成 |
| `energy-only` | `all-E-only` | 已完成 |
| `energy-direct` | `all-NC` | MatterSim-1M adapter 不支持 |

MatterTune 的 MatterSim/M3GNet adapter 会拒绝
`ForcesPropertyConfig(conservative=False)`，因此这组实验没有产生有效的 `all-NC`
timing。要研究 `all-NC` 或 `2stage-NC-C`，必须先实现 MatterSim direct-force head，或者
换用原生支持 direct force readout 的 backbone；后者不再是严格的同模型比较。

## 实验设置

- 数据：single-sol 的 7 个 XYZ 文件，共 28,279 个结构。
- 规模：512、1,024、2,048、4,096、8,192、16,384 和 28,279。
- split：seed 42 固定 shuffle 后分别从 train/validation pool 取嵌套前缀，保持 90/10。
- 硬件：4×NVIDIA A40，FP32。
- 优化：AdamW，LR `8e-5`，weight decay `0.1`，gradient clip 2.0，EMA 0.99。
- loss：energy weight 200；`all-C` force weight 20。
- batch/GPU：`all-E-only=16`，`all-C=8`。
- 数据读取：byte-offset lazy XYZ index，避免每个 DDP rank 复制约 1.1 GB 数据。
- 计时：CUDA synchronize + DDP barrier；包含训练、构图、optimizer、EMA、metrics 和
  validation，不包含模型/DDP 启动、sanity check 和 checkpoint I/O。

这里的 `all-E-only` validation 只评估 energy；`all-C` validation 同时评估 conservative
forces。因此判断 supervision 本身的吞吐时应优先看 `train_seconds`。

## 数据规模结果

下表来自归档的 `results/results.csv`，每个点只有一次 timing：

| 总结构数 | `all-C` epoch (s) | `all-E-only` epoch (s) | `all-E-only` 加速 |
| ---: | ---: | ---: | ---: |
| 512 | 11.67 | 5.19 | 2.25× |
| 1,024 | 21.13 | 7.91 | 2.67× |
| 2,048 | 39.99 | 13.72 | 2.92× |
| 4,096 | 78.29 | 25.03 | 3.13× |
| 8,192 | 155.29 | 46.61 | 3.33× |
| 16,384 | 306.41 | 89.78 | 3.41× |
| 28,279 | 542.63 | 152.21 | **3.56×** |

全量数据上：

- `all-C`：train 525.64 s，完整 epoch 542.63 s，约 52.11 structures/s；
- `all-E-only`：train 144.79 s，完整 epoch 152.21 s，约 185.79 structures/s；
- 纯训练加速为 3.63×，完整 epoch 加速为 3.56×；
- 4 GPU 的单 epoch 成本约为 0.603 GPU-hours 对 0.169 GPU-hours。

随着数据规模增大，加速从 2.25× 上升到 3.56×，说明小数据点受固定开销影响较大，
全量数据更能反映长期训练成本。

## Batch-size 校准

`all-E-only` 在单张独占 A40、2,048 个结构上的校准结果：

| Batch/GPU | Epoch (s) | Structures/s | Median GPU util | Peak allocated |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 45.26 | 45.25 | 97% | 8.12 GiB |
| 16 | **44.94** | **45.57** | 99% | 15.18 GiB |
| 24 | 45.08 | 45.44 | 99% | 22.01 GiB |
| 32 | 46.01 | 44.51 | 100% | 29.38 GiB |

因此 batch 16 被选为 `all-E-only` 的吞吐膝点。更大的 batch 只增加显存和瞬时
utilization，没有改善有效吞吐。

## 当前结论与限制

这组实验较强地支持“energy supervision 的单 epoch 成本远低于 exact conservative
force supervision”，因此值得研究 `2stage-E-only-C`。它不能证明 `all-E-only` 能达到
`all-C` 的 force 精度，也不能直接证明两阶段流程会节省 time-to-quality。

主要限制：只有一个 timing repeat；`all-E-only` 和 `all-C` 使用不同 global batch；两者
validation 工作量不同；该原型使用 AdamW+EMA，不能与后续 Muon、EMA-off 实验直接混合。
正式报告应在独占 GPU 上至少重复 3 次，并同时提供 method-optimal batch 和统一 batch 的
对照。

## 保留产物

- `results/results.csv`：逐规模原始汇总；
- `results/scalability.png`：规模—epoch 时间曲线；
- `results/raw/`：每个点的 JSON 和日志；
- `results/dataset-index.json`：XYZ byte-offset index。
