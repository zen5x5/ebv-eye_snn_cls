#!/usr/bin/env python3
"""ANN gaze-regression model, following Tan et al.'s described architecture
as closely as the paper's level of detail allows.

Tan et al. (Section IV-C-1, "Network Structure") state: 8 convolution
layers (each 3x3 kernel, stride 1, ReLU), 4 max-pooling layers (following
conv 1/2/4/5, "varying kernel sizes and strides" -- not specified exactly),
one 2D BatchNorm layer after the last conv (stabilizing the FC input), 3
fully-connected layers, ~91k parameters total, for a 2x120x220 input.

This project's ROI is a fixed centered 256x256 square (not Tan's 120x220 --
see .agent/ebv-eye.md), so an identical layer-by-layer reproduction is not
possible; the pooling schedule in TanStyleGazeNet was chosen to reach a
comparably small pre-FC feature map so the parameter count still lands
close to Tan's reported ~91k (this design: ~83k) despite the ~2.5x larger
input area, rather than letting the flatten dimension (and so the first FC
layer) explode. TanStyleGazeNet128 applies the same reasoning to a
128x128 ROI (e.g. a downsampled cache -- see dataset_preprocessing.py's `downsample` subcommand): a milder
first pool keeps it at the same ~83k budget instead of shrinking further.
Where the paper is silent (exact channel counts, exact pooling
kernel/stride per stage, FC widths), the choices below are this project's
own reasonable design, not values taken from the paper.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))


class TanStyleGazeNet(nn.Module):
    """Input: (B, 2, roi_size, roi_size) accumulated event frame.
    Output: (B, 2) normalized [row, col] in [0,1] (see dataloader.py)."""

    def __init__(self, roi_size: int = 256):
        super().__init__()
        if roi_size % 32 != 0:
            raise ValueError(f"roi_size must be a multiple of 32 (4*2*2*2 pooling), got {roi_size}")
        final_spatial = roi_size // 32

        # 8 conv layers; max pooling follows conv 1, 2, 4, 5 (per the paper).
        self.conv1 = _conv_block(2, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # extra-aggressive first pool: our input is much larger than Tan's
        self.conv2 = _conv_block(8, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = _conv_block(16, 16)
        self.conv4 = _conv_block(16, 16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = _conv_block(16, 16)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = _conv_block(16, 16)
        self.conv7 = _conv_block(16, 16)
        self.conv8 = _conv_block(16, 16)

        self.bn = nn.BatchNorm2d(16)
        flat_dim = 16 * final_spatial * final_spatial

        self.fc1 = nn.Sequential(nn.Linear(flat_dim, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.pool3(self.conv4(x))
        x = self.pool4(self.conv5(x))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.bn(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class TanStyleGazeNet128(nn.Module):
    """Same 8-conv/4-pool/BN/3-FC design as TanStyleGazeNet, retuned for a
    128x128 ROI (e.g. dataset_preprocessing.py's `downsample` subcommand's 2x-downsampled ebv-eye cache)
    instead of 256x256.

    MaxPool2d has no learnable parameters, so the total parameter count is
    driven entirely by channel counts and FC width, both of which are
    unchanged from TanStyleGazeNet here -- only the first pool is milder
    (kernel2/stride2 instead of kernel4/stride4, since a 128x128 input is
    already 4x smaller in area than 256x256). Total pooling reduction is
    therefore 16x (2*2*2*2) instead of 32x, landing on the same
    final_spatial=8 pre-FC feature map as the 256x256 model, and so the same
    ~83k parameter count that's close to Tan et al.'s reported ~91k.
    Naively reusing TanStyleGazeNet(roi_size=128) instead would keep the 32x
    schedule, shrink final_spatial to 4, and drop FC1's input 4x to only
    ~34k total parameters -- no longer comparable to the paper's budget."""

    def __init__(self, roi_size: int = 128):
        super().__init__()
        if roi_size % 16 != 0:
            raise ValueError(f"roi_size must be a multiple of 16 (2*2*2*2 pooling), got {roi_size}")
        final_spatial = roi_size // 16

        self.conv1 = _conv_block(2, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # milder than TanStyleGazeNet's k4/s4 -- input already 4x smaller in area
        self.conv2 = _conv_block(8, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = _conv_block(16, 16)
        self.conv4 = _conv_block(16, 16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = _conv_block(16, 16)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = _conv_block(16, 16)
        self.conv7 = _conv_block(16, 16)
        self.conv8 = _conv_block(16, 16)

        self.bn = nn.BatchNorm2d(16)
        flat_dim = 16 * final_spatial * final_spatial

        self.fc1 = nn.Sequential(nn.Linear(flat_dim, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.pool3(self.conv4(x))
        x = self.pool4(self.conv5(x))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.bn(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class TanStyleGazeNet128Native(nn.Module):
    """A capacity budget (80k-100k params) actually designed for 128x128,
    not TanStyleGazeNet128's approach of retrofitting TanStyleGazeNet's
    channel/FC widths (chosen for 256x256) onto a smaller input via a milder
    first pool. That retrofit reproduces TanStyleGazeNet's exact flat_dim
    (1024) and so its exact param count (~83k) -- convenient for an
    apples-to-apples 256-vs-128 comparison, but not a design decision made
    for 128x128 on its own terms.

    Two changes from TanStyleGazeNet128, both aimed at redistributing that
    budget rather than just hitting a similar total:

    1. Pooling schedule: reuses TanStyleGazeNet's own 32x-reduction
       schedule (kernel4/stride4 first pool, then three kernel2/stride2 --
       same positions after conv1/2/4/5 per the paper) instead of 128's
       milder 16x. At 128x128 that lands on a 4x4 (not 8x8) pre-flatten
       feature map -- already the schedule TanStyleGazeNet128Classifier's
       32x sibling (plain TanStyleGazeNet(roi_size=128), the "34k default"
       config in .agent/ebv-eye.md) uses, chosen here specifically because a
       smaller flat multiplier (16 vs 64) leaves far more of the param
       budget available for conv channels before FC1 dominates.
    2. Channel schedule: grows across the network (8/16/16/24/32/36/36/36)
       instead of TanStyleGazeNet's near-flat 8/16/16/16/16/16/16/16 --
       more representational capacity in the conv backbone, which is where
       a CNN's inductive bias (spatial locality/translation-equivariance)
       actually does work, rather than in a single large flatten->FC1
       linear projection.

    Net effect on where the ~87k parameters live: FC1 drops from ~79% of
    the total (TanStyleGazeNet128's 65,600/83,018) to ~42%
    (36,928/87,014) -- the conv backbone becomes the majority contributor
    instead of a single linear layer. Whether this redistribution actually
    improves accuracy over TanStyleGazeNet128 at the same input size is an
    open, untested question -- see .agent/ebv-eye.md's capacity-allocation
    discussion; this class exists to make that experiment possible, not as
    a claim that it will win."""

    def __init__(self, roi_size: int = 128):
        super().__init__()
        if roi_size % 32 != 0:
            raise ValueError(f"roi_size must be a multiple of 32 (4*2*2*2 pooling), got {roi_size}")
        final_spatial = roi_size // 32

        self.conv1 = _conv_block(2, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = _conv_block(8, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = _conv_block(16, 16)
        self.conv4 = _conv_block(16, 24)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = _conv_block(24, 32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = _conv_block(32, 36)
        self.conv7 = _conv_block(36, 36)
        self.conv8 = _conv_block(36, 36)

        self.bn = nn.BatchNorm2d(36)
        flat_dim = 36 * final_spatial * final_spatial

        self.fc1 = nn.Sequential(nn.Linear(flat_dim, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.pool3(self.conv4(x))
        x = self.pool4(self.conv5(x))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.bn(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class TanStyleGazeNet128Classifier(nn.Module):
    """Classification variant of TanStyleGazeNet128, for the ebv-eye saccade
    task: the 11x11 grid has exactly 11 distinct row values and 11 distinct
    col values (see dataloader.ROW_VALUES/COL_VALUES), so row and col can
    each be treated as an independent 11-way categorical target -- trained
    with cross-entropy per head -- instead of continuous [0,1] regression.

    Reuses the identical 8-conv/4-pool/BN/fc1/fc2 backbone by composing a
    TanStyleGazeNet128 instance and dropping its fc3 regression head (rather
    than re-declaring the conv/pool layers a third time), then adds two
    parallel heads, fc_row(32->11) and fc_col(32->11), in its place. Those
    two heads together add ~700 parameters over TanStyleGazeNet128's ~83k
    (vs. fc3's 66 for the single 2-output regression head) -- negligible,
    so this keeps the same parameter budget close to Tan et al.'s ~91k."""

    def __init__(self, roi_size: int = 128, num_row_classes: int = 11, num_col_classes: int = 11):
        super().__init__()
        backbone = TanStyleGazeNet128(roi_size=roi_size)
        del backbone.fc3  # unused -- fc_row/fc_col replace it
        self.backbone = backbone
        self.fc_row = nn.Linear(32, num_row_classes)
        self.fc_col = nn.Linear(32, num_col_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.backbone
        x = b.pool1(b.conv1(x))
        x = b.pool2(b.conv2(x))
        x = b.conv3(x)
        x = b.pool3(b.conv4(x))
        x = b.pool4(b.conv5(x))
        x = b.conv6(x)
        x = b.conv7(x)
        x = b.conv8(x)
        x = b.bn(x)
        x = torch.flatten(x, 1)
        x = b.fc1(x)
        x = b.fc2(x)
        return self.fc_row(x), self.fc_col(x)


class TanStyleGazeNet128NativeClassifier(nn.Module):
    """Classification variant of TanStyleGazeNet128Native -- same
    composition pattern as TanStyleGazeNet128Classifier (compose the
    backbone, drop its fc3, add two 11-way heads in its place), just on the
    "channel-heavy" 87k backbone instead of the "fc1-heavy" 83k retuned one.

    Distinguishing this from TanStyleGazeNet128Classifier matters here more
    than it did for the regression models: unlike the regression backbones
    (which trained fine either way and were always named by --model in
    their run directory), this project's only prior 128x128 classification
    result used the retuned backbone with no explicit "which backbone"
    marker in its run name (--model was silently ignored for --task
    classification) -- see .agent/ebv-eye.md and training.py's
    default_experiment_name(), both fixed 2026-08-15 to name classification
    runs by backbone the same way regression runs already were, precisely
    so a native-backbone classification result can't be confused with the
    existing retuned-backbone one on disk.

    fc_row(32->11)/fc_col(32->11) add the same ~660 parameters as
    TanStyleGazeNet128Classifier does over its regression counterpart, so
    this lands at TanStyleGazeNet128Native's ~87k plus ~660, still in the
    same ballpark."""

    def __init__(self, roi_size: int = 128, num_row_classes: int = 11, num_col_classes: int = 11):
        super().__init__()
        backbone = TanStyleGazeNet128Native(roi_size=roi_size)
        del backbone.fc3  # unused -- fc_row/fc_col replace it
        self.backbone = backbone
        self.fc_row = nn.Linear(32, num_row_classes)
        self.fc_col = nn.Linear(32, num_col_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.backbone
        x = b.pool1(b.conv1(x))
        x = b.pool2(b.conv2(x))
        x = b.conv3(x)
        x = b.pool3(b.conv4(x))
        x = b.pool4(b.conv5(x))
        x = b.conv6(x)
        x = b.conv7(x)
        x = b.conv8(x)
        x = b.bn(x)
        x = torch.flatten(x, 1)
        x = b.fc1(x)
        x = b.fc2(x)
        return self.fc_row(x), self.fc_col(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, cls, roi_size in (("TanStyleGazeNet", TanStyleGazeNet, 256), ("TanStyleGazeNet128", TanStyleGazeNet128, 128)):
        model = cls(roi_size=roi_size)
        n_params = count_parameters(model)
        print(f"{name} parameters: {n_params:,} (Tan et al. report ~91k for their 120x220 input)")
        dummy = torch.zeros(4, 2, roi_size, roi_size)
        out = model(dummy)
        print(f"forward OK: input {tuple(dummy.shape)} -> output {tuple(out.shape)}")

    clf = TanStyleGazeNet128Classifier(roi_size=128)
    n_params = count_parameters(clf)
    print(f"TanStyleGazeNet128Classifier parameters: {n_params:,}")
    dummy = torch.zeros(4, 2, 128, 128)
    row_logits, col_logits = clf(dummy)
    print(f"forward OK: input {tuple(dummy.shape)} -> row_logits {tuple(row_logits.shape)}, col_logits {tuple(col_logits.shape)}")
