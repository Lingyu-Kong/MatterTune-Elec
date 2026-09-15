# 两阶段训练初步探索：`2stage-E-only-C`

更新日期：2026-09-15。原型实现归档于
[`examples/hidden/two-stage-training`](../../hidden/two-stage-training/)。

## 研究问题与命名

当前实现只覆盖一种两阶段策略：

```text
Stage 1: all-E-only warm-up
Stage 2: exact conservative correction
完整方法名: 2stage-E-only-C
```

代码中旧 `variant=baseline` 对应规范名称 `all-C`；旧 `variant=two-stage` 对应
`2stage-E-only-C`。尚未实现 `2stage-NC-C`、`2stage-forward-FD-C` 或
`2stage-central-FD-C`。

实验假设来自 scalability 结果：全量数据上一个 `all-E-only` epoch 约 152 s，而一个
`all-C` epoch 约 543 s（4×A40），前者便宜约 3.56×。如果低成本 Stage 1 能减少 Stage 2
达到同等精度所需的 C epochs，总 GPU-hours 可能低于 `all-C`。

## 当前训练策略

| 设置 | Stage 1 (`E-only`) | Stage 2 / `all-C` control |
| --- | ---: | ---: |
| Batch/GPU | 16 | 8 |
| Energy / force weight | 200 / 0 | 200 / 20 |
| Optimizer | AdamW | 新 AdamW |
| Initial LR | `8e-5` | `8e-5` |
| Scheduler monitor | `val/energy_loss` | `val/total_loss` |
| Scheduler factor / patience | 0.8 / 10 | 0.8 / 10 |
| Early-stop patience | 20 | 100 |
| Max epochs | K，默认 50 | 5000 |

两阶段均关闭 EMA。切换时选择 Stage 1 的最佳 `val/energy_loss` checkpoint，只加载模型
权重；Stage 2 optimizer、scheduler、early-stop、epoch 和 global step 全部重新开始。
Stage 2 与本原型自己的 `all-C` control 应完全一致。阶段内中断则从本阶段
`last.ckpt` 恢复完整训练状态。

## 当前运行快照

结果目录：
`/net/csefiles/coc-fung-cluster/lingyu/ElectrolyteResults/two-stage-training/single-sol/two-stage-k50-seed42`。
硬件为 fung-cluster2 上 4×A40，seed 42。该运行尚未完成，以下只是 2026-09-15 上午的
中间快照。

### Stage 1

- CSV 中有 38 个完整 validation epoch（epoch 0–37）。
- 最佳 checkpoint 在 epoch 17：`val/energy_loss=0.00130468`，energy MAE=0.684996。
- 最佳点之后 20 个 epoch 未改善，Stage 1 在 epoch 37 后停止，符合 patience 20。
- launcher wall time 为 6,249.4 s（约 1.736 h，约 6.94 GPU-hours）。
- Stage 2 正确使用该最佳 checkpoint 做 weight-only initialization。

旧 manifest 报告 `epochs_completed=18` 且 `stop_reason=completed`，这是原型中
`last.ckpt` 只随 top-k checkpoint 刷新的缺陷：它读取的是 epoch 17 的 stale last
checkpoint。真实 epoch 数应以 CSV 和 launcher 最后一行的 epoch 37 为准。该缺陷必须在
后续正式实现中修复后再依赖 resume/stop metadata。

### Stage 2

- 快照时已有至少 564 个完整 validation epoch（epoch 0–563），运行状态仍是 running。
- 截至该快照，最佳点为 epoch 561：`val/total_loss=0.0425103`、energy MAE=0.725981、
  force MAE=0.0330761。
- epoch 563 为 `val/total_loss=0.0427876`、energy MAE=0.720651、force MAE=0.0331951。
- Stage 2 尚未 early-stop，manifest 也尚无最终 wall time；因此现在不能计算
  `2stage-E-only-C` 的最终总 GPU-hours 或成功判定。

## `all-C` 外部参考及不可直接比较处

曾参考成熟运行
`/net/csefiles/coc-fung-cluster/lingyu/ElectrolyteResults/single-sol/20260813-215204-730990-muon`。
它是一个 `all-C` 运行：4×A40、batch/GPU=8、Muon、EMA 0.99；最佳 checkpoint 在
epoch 684，`val/total_loss=0.0177768`，W&B 最后记录的 epoch 784 force MAE 约 0.02107。

当前 `2stage-E-only-C` 使用 AdamW 且关闭 EMA，所以不能把这个 Muon+EMA 运行当作严格
control。正式结论必须新跑与 Stage 2 完全相同 optimizer/scheduler/EMA 配置的 `all-C`
baseline，或者把两阶段实验改成与成熟 Muon recipe 完全一致后重跑。

## 当前结论与后续要求

当前结果只验证了 checkpoint handoff 和长时间 Stage 2 训练可运行；还没有证明
`2stage-E-only-C` 节省计算。Stage 1 自身已消耗约 6.94 GPU-hours，而 Stage 2 当前精度
仍落后上述非严格可比的 Muon `all-C` 参考。

下一轮正式比较应：

1. 先确定统一的 `all-C` control 配置，并保证 Stage 2 与其逐项一致；
2. 修复/验证真正每 epoch 更新的 `last.ckpt` 和实际停止原因；
3. 同时跑 `all-C`、`2stage-E-only-C`，并进一步加入
   `2stage-central-FD-C`/`2stage-forward-FD-C`；
4. 用累计 GPU-hours 对 validation/test MAE 作图，而不是只比较 epoch 数；
5. 在相同硬件上使用至少多个 seed，最终检查 energy/force MAE 1% 容忍度和总 GPU-hours。

## 原型能力

归档代码提供统一 CLI、阶段级 best/last checkpoint、weight-only handoff、阶段内 resume、
CSV 指标解析、硬件/协议 manifest、dry-run、smoke test 和 baseline/two-stage 分析脚本。
