# ebv-eye saccade SCNN 模型审计

审计日期：2026-09-07。这里的“当前结果”都来自实际执行；没有 Vivado、时序、功耗或上板结果。

## 结论先行

本阶段冻结的是 user 15、eye 0、`corrective_peak` 采样的 8 层分类 SCNN。它有明确的整数化路径：
卷积后的 BatchNorm 可以折叠进权重和偏置，神经元没有 leak/beta，复位是减去阈值，输出是 22 个
类别计数。因此可以做厂商无关 RTL，而且已经能用一个共享 MAC 顺序执行整个网络。

最需要注意的研究口径问题不是 RTL，而是 checkpoint 的选择方式：训练脚本用测试集 loss 挑了
epoch 35 的 `best_checkpoint.pt`。所以 87.139% joint accuracy 是“被用于选模型的测试集”成绩，
不是完全独立的无偏泛化估计。这里仍冻结它，因为它正是现有报告和 Block 1 实验所使用的权重；
后续论文或最终验收应另留一组从未参与选择的数据。

## 冻结产物

- checkpoint：`/data/xuzx/result/02_snn_speck/ebv-eye/snn/user=15/` 下实验目录中的
  `best_checkpoint.pt`
- checkpoint SHA-256：`a2ca40bc306940aafae81e8bda6eb0c64cf81a4f9696c0773de69f0c054dc678`
- 训练配置：classification、10 个时间步、threshold=1.0、min_v_mem=-1.0、weight clip=1.0、
  seed=2021、epoch 35
- 数据缓存：`/data/xuzx/data/ebv-eye/eye_data/user15/0/corrective_peak_start_downsampled128`
- 样本数：train 9,688；test 2,356；每个样本 500 个事件

checkpoint、数据集和原训练结果只读使用，没有修改或覆盖。

## 软件来源与入口

为了避免按常见 SNN 写法猜测，这次冻结行为直接沿着现有代码追踪：

| 职责 | 仓库文件/产物 |
|---|---|
| 训练入口 | `code/ebv-eye/snn/training.py` |
| 评估入口 | `code/ebv-eye/snn/evaluation.py` |
| 网络定义 | `code/ebv-eye/snn/model.py` 中的 `SpikingNet8LayerBN` |
| 输入编码/数据集 | `code/ebv-eye/snn/dataloader.py` 中的 `events_to_time_binned_frames` 与 `EbvEyeSpikeDataset` |
| 冻结配置 | checkpoint 同目录的 `config.json` |
| 训练历史/选择依据 | checkpoint 同目录的 `history.json` |
| 可重复硬件提取 | `code/ebv-eye/snn/hardware_model.py` |

同一实验目录还保留 `final_checkpoint.pt` 和 `train.log`；本阶段明确使用 `best_checkpoint.pt`，没有把
final checkpoint 或别的实验权重混入导出。

## 输入究竟是什么

每个样本先取 500 个事件，再按该样本自己的起止时间均分为 10 个时间区间。每个事件按极性进入
两个 128x128 平面之一，同一像素在同一时间区间内出现多次就累加。因此一个输入字是无符号事件
计数，不是只有 0/1 的布尔值。张量形状为 `(T,C,H,W)=(10,2,128,128)`；RTL 传输顺序也是
时间步、通道、行、列，列变化最快。

## 网络拓扑

“IAF 输出”是神经元刚发出的脉冲；有池化的层随后把 2x2 区域做求和，才成为下一层输入。

| 层 | 卷积 | IAF 输出 shape | 送往下一层的 shape |
|---|---|---:|---:|
| block1 | 2→8, k3, stride2, pad1 | 8x64x64 | 8x64x64 |
| block2 | 8→16, k3, stride2, pad1 | 16x32x32 | 16x32x32 |
| block3 | 16→32, k3, stride1, pad1 | 32x32x32 | 32x16x16（2x2 sum pool） |
| block4 | 32→64, k3, stride1, pad1 | 64x16x16 | 64x8x8（2x2 sum pool） |
| block5 | 64→64, k3, stride1, pad1 | 64x8x8 | 64x4x4（2x2 sum pool） |
| block6 | 64→64, k3, stride1, pad1 | 64x4x4 | 64x4x4 |
| block7 | 64→32, k3, stride1, pad1 | 32x4x4 | 32x4x4 |
| head | 32→22, k4, stride1, pad0 | 22x1x1 | 22x1x1 |

总权重 127,760 个，膜电位状态 103,958 个；最大中间 activation 是 32,768 个字。

## 神经元逐步行为

对每个神经元、每个时间步依次做：

1. 卷积乘加并加入折叠后的 BatchNorm bias，得到 `syn_sum`。
2. `v_pre = 上一步膜电位 + syn_sum`。
3. `v_pre > 0` 时，`spike = floor(v_pre / threshold)`；否则 spike 为 0。
4. 从 `v_pre` 减去完整的 `spike * threshold`，这是 subtractive reset（减法复位）。
5. 若结果低于 `min_v_mem`，钳到 `min_v_mem`，得到下一步膜电位 `v_post`。

没有 leak/beta，也没有在层之间做小数右移。数学上正好等于 threshold 时会发一个 spike，所以也可表述
为正区间上的 `v_pre >= threshold` 比较。每个新样本开始前，所有膜电位清零。

## 输出判决

head 的 22 个通道在 10 个时间步上累加。通道 0..10 是 row 的 11 类，11..21 是 column 的
11 类；两组分别取 argmax。相同计数时选择编号更小的类别。

## 实际准确率

| 路径 | 样本 | row | column | joint | pixel L2 | angular |
|---|---:|---:|---:|---:|---:|---:|
| 原 Exodus CUDA 模型（本次重跑） | 2,356 | 91.596% | 93.548% | 87.139% | 22.315 px | 1.1005° |
| 官方转换后的 CPU FP32 后端 | 2,356 | 91.723% | 93.591% | 87.309% | 22.584 px | 1.1179° |
| 本项目 B16 整数模型 | 2,356 | 91.638% | 93.506% | 86.969% | 23.187 px | 1.1488° |

以实际训练/评估使用的 Exodus CUDA 为基准，整数 joint accuracy 下降 0.170 个百分点。CPU FP32
后端在极少数阈值边界上与 CUDA 后端有差异，因此保留为可复跑的交叉检查，不拿它替代原后端基准。
量化下降很小，本阶段没有擅自启动长时间 QAT。

## RTL 可行性与仍需 Vivado 回答的问题

数值行为适合 RTL；当前实现用一个 MAC 顺序复用，优先保证可检查和 bit-exact。风险在物理实现：

- activation、膜电位、48 位 synapse 累加器和参数合计约 6.3 Mbit，尚未加 BRAM 端口适配和余量；
- 当前通用 memory 写法必须在 Vivado 中确认能否按预期推成 BRAM，特别是 2x2 pool 的多读口；
- 可变 threshold 除法目前是可综合的 `/` 运算，功能正确但可能代价很大；应在 Vivado 报告后决定
  用迭代除法、阈值倒数或逐层常量化；
- 单 MAC 版本吞吐低，最终并行 lane 数必须由目标时钟、DSP/BRAM 端口和实时要求共同决定。

这些都是后续物理实现任务，不影响本阶段对软件行为和 RTL 功能的证明。
