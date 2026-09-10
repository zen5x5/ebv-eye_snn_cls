#!/usr/bin/env python3
"""ebv-eye saccade/pursuit 任务的脉冲卷积网络 (SNN)。

网络按 Speck 的硬件约束设计 (9 个 DYNAPCNN 核、每核的 kernel/neuron 内存预算、
只有 IAF 神经元、卷积 stride 限 {1,2,4,8}、pool kernel 限 {1,2,4}、8 位权重)，
而不是把 ANN 侧的 TanStyleGazeNet128Native 直接照搬过来。结构参考:
VGG 式的"空间减半/通道翻倍"schedule; Amir et al. 2017 (DVS-Gesture) 说明
事件相机分类只需要少数几个 conv+pool 级; Neftci, Mostafa & Zenke 2019 说明
surrogate-gradient BPTT 只需要很少的时间步。

本文件只有一个骨干: 8 层卷积 + BatchNorm, 占 9 个核里的 8 个。

输入布局 (B, T, 2, H, W)。T 是真正的时间轴 (dataloader.py 按固定时长分 bin,
不是固定事件数), 不是同一帧重复 T 次。H=W=128, 与 Speck 原生 128x128 DVS
分辨率以及每层最大输入尺寸限制一致。

每个 IAFSqueeze 都用 MultiSpike + PeriodicExponential + 有限的 min_v_mem +
MembraneSubtract (软复位)。这组合来自 sinabs 自己的 docs/speck/dangers.md 和
faqs/tips_for_training.md 对真机部署的建议; MembraneSubtract 也与 devkit 默认的
硬件复位机制一致。
"""

import torch
import torch.nn as nn
from sinabs.activation import MembraneSubtract, MultiSpike, PeriodicExponential
from sinabs.exodus.layers import IAFSqueeze
from sinabs.layers import SumPool2d

DEFAULT_SPIKE_THRESHOLD = 1.0
DEFAULT_MIN_V_MEM = -1.0  # 必须是有限值; None 会被映射到 -32768, 行为与训练时不符


def _iaf(num_timesteps: int, spike_threshold: float, min_v_mem: float) -> IAFSqueeze:
    return IAFSqueeze(
        num_timesteps=num_timesteps,
        spike_threshold=spike_threshold,
        spike_fn=MultiSpike,
        surrogate_grad_fn=PeriodicExponential(),
        reset_fn=MembraneSubtract(),
        min_v_mem=min_v_mem,
    )


def _conv_bn(in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
    """全程 bias=False -- Speck 的 per-channel leak 寄存器承担 conv bias 的角色,
    而 sinabs 自己的 docs/speck/dangers.md 建议不用 bias (稠密, 硬件开销高)。
    BatchNorm2d 在转成部署网络时会被自动折进前一个卷积的权重。每个卷积后面都用
    BN (不像 ANN 侧只在靠近输出处用一次), 因为逐层归一化对 surrogate-gradient
    SNN 的训练稳定性有明显帮助。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class SpikingNet8LayerBN(nn.Module):
    """8 层卷积 + BatchNorm 的脉冲网络, 占 Speck 9 个核中的 8 个。

    可切换的输出头:

    - ``task='regression'`` 输出 ``2 * num_digits`` 个通道，row/col 每轴各有
      ``num_digits`` 个位值通道，按时间窗内的脉冲计数读出（见 training.py）。
      num_digits=1 时每轴有 11 个可区分档位，与 saccade 标签网格的 11 个
      row/col 取值一一对应。
      这是有意的 rate-coding 设计 (Rueckauer et al. 2017 的"发放率作为连续值的
      无偏估计", 在本任务已经离散的标签结构上特化)。
    - ``task='classification'`` 输出 22 个通道 (11 个 row 类 + 11 个 col 类,
      读出脉冲计数后由调用方切分, 见 training.py), 就是 ANN 侧一直在用的
      saccade 网格分类任务。

    ``forward(x)``, x 形状 (B, T, 2, H, W), 返回 (B, T, C_out) 的逐时间步原始
    脉冲计数, 还没有沿 T 归约。归约由 training.py 负责 (回归取发放率, 分类取
    计数求和后的 argmax), 因为两个任务的读出方式不同。

    ``num_digits`` (只对回归有意义, 默认 1): 不再是每轴 1 个神经元读一个 T 步
    发放率 (只有 11 档: 0.0/0.1/.../1.0), 而是每轴 num_digits 个神经元组成一个
    十进制位值编码。例如 num_digits=3 给出每轴粗/中/细三个位 (共 6 通道),
    还原为 粗 + 中*0.1 + 细*0.01。最终坐标夹到 [0,1] 后，每轴有
    10**3+1=1001 个唯一取值，而不是 11**3；超过 1 的组合都饱和为同一输出。
    通道顺序 [row_digit_0..D-1, col_digit_0..D-1], digit_0 最粗。
    动机见 report/ebv-eye.md: pursuit 的连续标签会暴露单通道
    11 档的输出量化上限。所有位数都在重建后直接与原始连续标签
    计算 loss，不再把标签向输出码本取整，见 training.py。"""

    def __init__(
        self,
        num_timesteps: int = 10,
        task: str = "regression",
        num_digits: int = 1,
        spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
        min_v_mem: float = DEFAULT_MIN_V_MEM,
    ):
        super().__init__()
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
        if task == "classification" and num_digits != 1:
            raise ValueError("num_digits > 1 is only meaningful for task='regression'")
        self.num_timesteps = num_timesteps
        self.task = task
        self.num_digits = num_digits
        out_ch = 2 * num_digits if task == "regression" else 22

        def iaf():
            return _iaf(num_timesteps, spike_threshold, min_v_mem)

        # 1: 128->64, 用 stride 而不是 pool。神经元内存逼的: 128x128 上的
        # stride-1 卷积每个输出通道要 128Ki+ 神经元内存, 超出每个核的预算。
        self.block1 = nn.Sequential(_conv_bn(2, 8, kernel_size=3, stride=2, padding=1), iaf())
        # 2: 64->32, 同样的理由, 64x64 上内存仍然紧张。
        self.block2 = nn.Sequential(_conv_bn(8, 16, kernel_size=3, stride=2, padding=1), iaf())
        # 3-5: 空间尺寸已经负担得起, 改用真正的 SumPool2d (聚合脉冲) 而不是继续用
        # stride 抽取。
        self.block3 = nn.Sequential(_conv_bn(16, 32, kernel_size=3, stride=1, padding=1), iaf(), SumPool2d(2))
        self.block4 = nn.Sequential(_conv_bn(32, 64, kernel_size=3, stride=1, padding=1), iaf(), SumPool2d(2))
        self.block5 = nn.Sequential(_conv_bn(64, 64, kernel_size=3, stride=1, padding=1), iaf(), SumPool2d(2))
        # 6: 在 4x4 瓶颈处多加一层深度。
        self.block6 = nn.Sequential(_conv_bn(64, 64, kernel_size=3, stride=1, padding=1), iaf())
        # 7: 进头部之前先降通道。
        self.block7 = nn.Sequential(_conv_bn(64, 32, kernel_size=3, stride=1, padding=1), iaf())
        # 8: 输出头, 4x4 -> 1x1。sinabs 能把 flatten+Linear 自动转过来 (空间 4x4
        # <= 16x16), 但这里直接写成 Conv2d, 更接近硬件的形式。
        self.head = nn.Sequential(
            nn.Conv2d(32, out_ch, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            iaf(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps = x.shape[0], x.shape[1]
        x = x.reshape(batch * timesteps, *x.shape[2:])
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.head(x)
        x = x.reshape(batch, timesteps, -1)
        return x

    def reset_states(self) -> None:
        for module in self.modules():
            if hasattr(module, "reset_states") and module is not self:
                module.reset_states()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
