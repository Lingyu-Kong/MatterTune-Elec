# 微调训练策略项目目标

更新日期：2026-09-15。

## 核心问题

本项目研究 MatterSim 等机器学习势函数的 compute-efficient fine-tuning：

1. 在相同 GPU compute time 下，哪种训练策略能得到最好的 energy、force 和（适用时）stress 精度？
2. 在达到相近精度要求时，哪种训练策略消耗的 GPU compute time 最少？

所有面向最终模型的候选流程都必须经过一个使用完整训练集的 conservative-force exact
阶段进行修正。低成本方法既可以作为独立的诊断实验，也可以作为该 exact 阶段之前的
warm-up；只有后者才代表完整的候选生产流程。

## 规范命名

`all-` 表示整个被记录的训练过程只使用一种 supervision；`2stage-` 表示依次执行两种
supervision。名称中的 `C` 是 conservative，`NC` 是 non-conservative/direct，`E-only`
是只使用 energy。

| 规范名称 | 定义 |
| --- | --- |
| `all-C` | 全程 energy + conservative forces，并可选 conservative stress；force/stress 由 energy 对坐标/cell 的自动微分得到。 |
| `all-NC` | 全程 energy + direct forces，并可选 direct stress；force/stress 由 node/edge feature 经 readout MLP 直接输出。 |
| `all-E-only` | 全程只使用 energy supervision。 |
| `all-central-FD` | 全程用 central finite difference directional estimator 代替训练时的完整 conservative-force loss。 |
| `all-forward-FD` | 全程用 forward finite difference directional estimator 代替训练时的完整 conservative-force loss。 |
| `2stage-XXX-C` | Stage 1 使用低成本策略 `XXX`，Stage 2 切换为与 `all-C` 相同的 exact conservative 训练。 |

例如：`2stage-NC-C`、`2stage-E-only-C`、`2stage-central-FD-C` 和
`2stage-forward-FD-C`。不要再用 `baseline`、`energy-conservative`、`energy-direct`
或含糊的 `two-stage` 作为报告中的方法名；代码中的旧 CLI 值可以保留，但必须在分析
输出中映射为上述名称。

AD directional derivative 是有限差分探索中的诊断 estimator，目前不是一个单独约定的
正式训练策略名。需要报告时称为“AD-directional estimator control”。

## 默认比较口径

- 默认模型/数据：MatterSim-1M、single-sol 全量 28,279 个结构、固定 seed 42、固定
  90/10 split（25,451 train，2,828 validation）。
- 精度：至少报告 validation energy MAE、force MAE 和 total loss；含 stress 的实验还需
  报告 stress MAE。最终结论应尽可能使用独立 test split，而不是只复用 early-stop
  validation。
- 成本：`GPU-hours = 各阶段 wall time × 当阶段 GPU 数量 / 3600`，多阶段成本必须求和。
  同时保留训练时间、validation 时间、启动/checkpoint 开销和峰值显存。
- 公平性：主比较要求相同 GPU 型号、GPU 数量、数据划分、seed、模型初始化、精度和
  validation 口径。不同方法允许使用各自经过校准的高吞吐 batch，但要另做相同 batch
  的 controlled comparison，以区分 estimator 收益和 batch-size 收益。
- 随机性：正式结论至少使用 3 个 seed 或 3 次独立 timing repeat，报告均值/中位数和
  波动范围。单次单 epoch 结果只用于筛选，不作为 time-to-quality 的最终证明。
- 成功判定：相对同配置 `all-C`，energy/force（及 stress）MAE 均不恶化超过预先规定的
  容忍度（初始值 1%），同时总 GPU-hours 更低。

## 两阶段切换原则

- Stage 2 必须从 Stage 1 的预先约定指标最佳 checkpoint 加载模型权重。
- 默认只继承模型权重；optimizer、scheduler、early-stop、epoch 和 global step 重新开始。
- Stage 2 的数据、loss、optimizer、scheduler、validation、early stop 和 `all-C` 基线必须
  一致，否则无法把差异归因于 Stage 1。
- 中断恢复只从当前阶段自己的 `last.ckpt` 恢复完整状态，不应把 Stage 1 optimizer 状态
  带进 Stage 2。

## 最终应交付的图表

每一组正式实验至少生成两类图：

1. validation/test MAE 对累计 GPU-hours 的学习曲线；
2. 达到目标精度所需 GPU-hours，或固定 GPU-hours 下各方法精度的 Pareto 图。

当前三组初步探索记录见同目录下的 `scalability-test.md`、
`finite-difference-training.md` 和 `two-stage-training.md`。原型代码已归档到
`examples/hidden/`，不应被误认为稳定的公开 example。
