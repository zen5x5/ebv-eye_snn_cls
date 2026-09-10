# 硬件数值契约

机器可读版本是相邻目录的 `hardware_numeric.yaml`；本文件解释为什么这样定，以及 RTL 和导出脚本
必须共同遵守什么。改变任何一项都意味着换了模型，必须重新导出并重新跑内部 bit-exact 验证。

## 表示方法

- weight、折叠 bias、threshold、`min_v_mem`：每层各用一个共同 scale，存为 signed 16-bit
  二补码。
- 输入和层间 spike count：unsigned 16-bit，范围 0..65535；超出时饱和，而不是回绕。
- 膜电位：signed 16-bit。减法复位后余数小于正 threshold，负侧又被 `min_v_mem` 钳住，因此合法
  模型中能装下。
- 乘积：unsigned 16-bit activation 先补一个 0 变成正的 signed 17-bit，再乘 signed 16-bit
  weight，数学结果需要 signed 33-bit。
- `syn_sum` 和 `v_pre`：signed 48-bit。最坏卷积有 576 项，保守界
  `576 * 32768 * 65535 < 2^41`，48 位还留出了 bias 和旧膜电位余量。
- 10 步输出类别计数：unsigned 32-bit。

## 量化规则

先用 Sinabs 官方转换把 BatchNorm 折进前面的 convolution。对一层分别求 weight、folded bias、
`[-threshold,+threshold]` 能放入 signed 16-bit 的 scale，取三者中最小值作为共同 scale。浮点数乘
scale 后按 nearest、ties-to-even（正好在两个整数中间时取偶数）舍入。

| 层 | scale | 限制 scale 的量 | threshold / min | weight 地址 | bias 地址 | state 地址 |
|---|---:|---|---:|---:|---:|---:|
| block1 | 408.109772 | weight | 408 / -408 | 0 | 0 | 0 |
| block2 | 32767.000000 | threshold | 32767 / -32767 | 144 | 8 | 32768 |
| block3 | 10856.852539 | bias | 10857 / -10857 | 1296 | 24 | 49152 |
| block4 | 6413.098145 | bias | 6413 / -6413 | 5904 | 56 | 81920 |
| block5 | 17141.611328 | bias | 17142 / -17142 | 24336 | 120 | 98304 |
| block6 | 21993.121094 | bias | 21993 / -21993 | 61200 | 184 | 102400 |
| block7 | 32767.000000 | threshold | 32767 / -32767 | 98064 | 248 | 103424 |
| head | 16835.126953 | bias | 16835 / -16835 | 116496 | 280 | 103936 |

每一层自己的 weight、bias、threshold 和膜电位处于同一整数单位。spike 是“发了多少次”的无量纲
整数，所以层与层之间无需再乘 scale 或右移。

## 更新、溢出和复位

顺序固定为 integrate → threshold/multi-spike → subtractive reset → lower clamp。完整 quotient 用于
复位，即使输出 spike 因 16 位接口而饱和，膜电位也仍使用未饱和的 quotient；不能先截断再复位。

48 位累加器的溢出属于契约违反，不能静默 wrap 或 saturate。16 位 spike 和 2x2 sum pool 明确定义
为 saturate。10 个固定真实样本的 spike/pool saturation 计数均为 0。

模型没有负数右移，也没有 fractional-bit rescale；因此不存在由 HDL `>>>` 与 Python `//` 差异引起
的负数取整问题。将来若加入右移，必须先为负输入单独冻结舍入语义，不能沿用未定义假设。

样本之间是 hard state clear（整块膜电位归零）；单个 spike 后则只允许训练模型的 subtractive reset。
所谓“hard neuron reset（发放后直接把膜电位置零）”不是这个 checkpoint 的行为，不能作为可选开关悄悄
启用。

## 布局和可追踪性

- activation：C,Y,X，X 最快。
- weight：OC,IC,KY,KX，KX 最快。
- `.mem`：每行一个 16-bit 十六进制二补码字，不带 `0x`。
- 输入 stream：T,C,Y,X，X 最快；只在 T=9 的最后一个字断言 `in_last`。
- 输出 stream：先 row class，再 column class，第二个字断言 `out_last`。

`manifest.json` 记录 checkpoint hash、每层 shape/offset/scale 和每个导出文件 SHA-256。真实向量还保存
每层每步 spike；RTL 为每层每步输出一个 64-bit 内部摘要，摘要同时混合 `syn_sum`、`v_pre`、
`spike`、`v_post` 和神经元地址。最终类别碰巧相同不能掩盖中间状态错误。

