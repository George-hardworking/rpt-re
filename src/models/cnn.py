"""Price-trend CNN per Jiang–Kelly–Xiu (JF 2023) Appendix architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from data.images import image_shape

KERNEL_HW = (5, 3)
CONV_PADDING = (KERNEL_HW[0] // 2, KERNEL_HW[1] // 2)
POOL_HW = (2, 1)
LEAKY_RELU_SLOPE = 0.01
FC_DROPOUT = 0.5

_NUM_BLOCKS: dict[int, int] = {5: 2, 20: 3, 60: 4}
_FIRST_STRIDE: dict[int, tuple[int, int]] = {5: (1, 1), 20: (3, 1), 60: (3, 1)}
_FIRST_DILATION: dict[int, tuple[int, int]] = {5: (1, 1), 20: (2, 1), 60: (3, 1)}
_PAPER_FLATTEN: dict[int, int] = {5: 15_360, 20: 46_080, 60: 184_320}
_PAPER_PARAMS: dict[int, int] = {5: 155_138, 20: 708_866, 60: 2_952_962}


def cnn_num_blocks(window_days: int) -> int:
    return _NUM_BLOCKS[window_days]


def expected_flatten_size(window_days: int) -> int:
    return _PAPER_FLATTEN[window_days]


def flatten_feature_size(window_days: int) -> int:
    return _PAPER_FLATTEN[window_days]


def _infer_flatten_size(blocks: nn.Sequential, window_days: int) -> int:
    height, width = image_shape(window_days)
    was_training = blocks.training
    blocks.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, height, width)
        flat = int(blocks(dummy).numel())
    if was_training:
        blocks.train()
    return flat


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
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=KERNEL_HW,
            stride=stride,
            dilation=dilation,
            padding=CONV_PADDING,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(LEAKY_RELU_SLOPE)
        self.pool = nn.MaxPool2d(kernel_size=POOL_HW, stride=POOL_HW, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        flat = _infer_flatten_size(self.blocks, window_days)
        assert flat == _PAPER_FLATTEN[window_days], (
            f"I{window_days} flatten={flat} expected {_PAPER_FLATTEN[window_days]}"
        )
        self.dropout = nn.Dropout(FC_DROPOUT)
        self.fc = nn.Linear(flat, 2)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)

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


def copy_conv_blocks_from(
    target: PriceTrendCNN,
    source_state: dict[str, torch.Tensor],
    source_window_days: int,
) -> None:
    """Copy conv+BN weights for all blocks in a shallower source network."""
    n_blocks = cnn_num_blocks(source_window_days)
    target_state = target.state_dict()
    for block_idx in range(n_blocks):
        prefix = f"blocks.{block_idx}"
        for suffix in (
            "conv.weight",
            "conv.bias",
            "bn.weight",
            "bn.bias",
            "bn.running_mean",
            "bn.running_var",
            "bn.num_batches_tracked",
        ):
            key = f"{prefix}.{suffix}"
            if key not in source_state:
                raise KeyError(f"missing {key} in source checkpoint")
            target_state[key] = source_state[key].clone()
    target.load_state_dict(target_state)
