"""Price-trend CNN per Jiang–Kelly–Xiu (JF 2023) Appendix architecture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.images import image_shape

KERNEL_HW = (5, 3)
POOL_HW = (2, 1)
LEAKY_RELU_SLOPE = 0.01
FC_DROPOUT = 0.5

_NUM_BLOCKS: dict[int, int] = {5: 2, 20: 3, 60: 4}
_FIRST_STRIDE: dict[int, tuple[int, int]] = {5: (1, 1), 20: (3, 1), 60: (3, 1)}
_FIRST_DILATION: dict[int, tuple[int, int]] = {5: (1, 1), 20: (2, 1), 60: (3, 1)}
_PAPER_FLATTEN: dict[int, int] = {5: 15_360, 20: 46_080, 60: 184_320}


def cnn_num_blocks(window_days: int) -> int:
    return _NUM_BLOCKS[window_days]


def expected_flatten_size(window_days: int) -> int:
    return _PAPER_FLATTEN[window_days]


def _conv_output_size(
    size_in: int,
    kernel: int,
    stride: int,
    dilation: int,
) -> int:
    kernel_eff = dilation * (kernel - 1) + 1
    return math.ceil(size_in / stride)


def _same_pad_before_after(
    size_in: int,
    kernel: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    kernel_eff = dilation * (kernel - 1) + 1
    size_out = _conv_output_size(size_in, kernel, stride, dilation)
    pad_total = max(0, (size_out - 1) * stride + kernel_eff - size_in)
    pad_before = pad_total // 2
    return pad_before, pad_total - pad_before


def _spatial_after_blocks(window_days: int) -> tuple[int, int]:
    height, width = image_shape(window_days)
    n_blocks = cnn_num_blocks(window_days)
    for block_idx in range(n_blocks):
        if block_idx == 0:
            stride = _FIRST_STRIDE[window_days]
            dilation = _FIRST_DILATION[window_days]
        else:
            stride = (1, 1)
            dilation = (1, 1)
        height = _conv_output_size(height, KERNEL_HW[0], stride[0], dilation[0])
        width = _conv_output_size(width, KERNEL_HW[1], stride[1], dilation[1])
        height = math.ceil(height / POOL_HW[0])
        width = math.ceil(width / POOL_HW[1])
    return height, width


def flatten_feature_size(window_days: int) -> int:
    h, w = _spatial_after_blocks(window_days)
    channels = 64 * (2 ** (cnn_num_blocks(window_days) - 1))
    return h * w * channels


class CNNBuildingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=KERNEL_HW,
            stride=stride,
            dilation=dilation,
            padding=0,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(LEAKY_RELU_SLOPE)
        self.pool = nn.MaxPool2d(kernel_size=POOL_HW, stride=POOL_HW)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph_b, ph_a = _same_pad_before_after(h, KERNEL_HW[0], self.stride[0], self.dilation[0])
        pw_b, pw_a = _same_pad_before_after(w, KERNEL_HW[1], self.stride[1], self.dilation[1])
        return F.pad(x, (pw_b, pw_a, ph_b, ph_a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)


class PriceTrendCNN(nn.Module):
    """Binary classifier: grayscale OHLC image -> 2 logits (up / down)."""

    def __init__(self, window_days: int) -> None:
        super().__init__()
        self.window_days = window_days
        n_blocks = cnn_num_blocks(window_days)
        blocks: list[nn.Module] = []
        in_ch = 1
        for block_idx in range(n_blocks):
            out_ch = 64 * (2**block_idx)
            if block_idx == 0:
                stride = _FIRST_STRIDE[window_days]
                dilation = _FIRST_DILATION[window_days]
            else:
                stride = (1, 1)
                dilation = (1, 1)
            blocks.append(
                CNNBuildingBlock(
                    in_ch,
                    out_ch,
                    stride=stride,
                    dilation=dilation,
                )
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        flat = flatten_feature_size(window_days)
        self.dropout = nn.Dropout(FC_DROPOUT)
        self.fc = nn.Linear(flat, 2)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.blocks(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    @torch.no_grad()
    def predict_proba_up(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)[:, 1]
