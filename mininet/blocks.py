# Copyright James Clampffer 2025
"""Functional blocks for image processing"""

import torch
import torch.nn as nn


class DepthwiseSeparableConvBlock(nn.Module):
    """
    @brief Depthwise-separable 3x3 convolution

    Reduce parameter count and inference flops by replacing full convs
    with a stacked channel-wise + pointwise conv.
    """

    __slots__ = (
        "_chan_in",
        "_chan_out",
        "_depthwise",
        "_pointwise",
        "_norm",
        "_activation_fn",
    )
    _chan_in: int
    _chan_out: int
    _depthwise: nn.Conv2d
    _pointwise: nn.Conv2d
    _norm: nn.BatchNorm2d
    _activation_fn: nn.Module

    def __init__(
        self, chin: int, chout: int, stride: int = 1, activation: nn.Module = nn.SiLU
    ):
        super().__init__()
        self._chan_in = chin
        self._chan_out = chout

        # Setting groups=chin means each group contains a single channel
        self._depthwise = nn.Conv2d(
            chin, chin, kernel_size=3, stride=stride, padding=1, groups=chin, bias=False
        )
        self._pointwise = nn.Conv2d(chin, chout, kernel_size=1, bias=False)
        self._norm = nn.BatchNorm2d(chout)
        self._activation_fn = activation()

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        fmap = self._depthwise(fmap)
        fmap = self._pointwise(fmap)
        fmap = self._norm(fmap)
        fmap = self._activation_fn(fmap)
        return fmap


class MiniResBlock(nn.Module):
    """
    @brief Lightweight residual block

    Skip layer + a stack of two depthwise separable convs
    """

    # Residuals passed across strided convolutions get downsampled using
    # pooling rather than a convolution - lighter weight. 1x1 conv to handle
    # channel mismatches

    __slots__ = (
        "_chan_in",
        "_chan_out",
        "_activation_fn",
        "_reduce",
        "_conv_stack",
        "_residual_transform",
    )

    # input feature map channels
    _chan_in: int

    # output feature map channels
    _chan_out: int

    # Activation used for residual transforms and convs.
    _activation_fn: nn.Module

    # Indicate if the feature map should be downsampled by two
    _reduce: bool

    # The convolution "stack" used for the core of the block
    _conv_stack: nn.Module

    # The only cool tweak I'm aware of in here - swap residual shape
    # match transforms based on what needs to be done. Plain map size
    # reduction is done by a pooling pass rather than a 1x1 conv. The
    # latter is more expensive and the former preserves spatial info.
    _residual_transform: nn.Module | None

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        reduce: bool = False,
        activation: nn.Module = nn.SiLU,
    ):
        """Set up residual transform and conv stack"""
        super().__init__()
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._reduce = reduce
        self._activation_fn = activation

        # Where the work happens
        self._conv_stack = nn.Sequential(
            DepthwiseSeparableConvBlock(
                chan_in, chan_out, stride=2 if reduce else 1, activation=activation
            ),
            DepthwiseSeparableConvBlock(chan_out, chan_out, activation=activation),
        )

        # 3 options here. Triples is best. Triples makes it safe.
        if self.chan_in == self.chan_out and not self.reduce:
            # Passthrough, no transformation required, here for clarity
            self._residual_transform = None
        elif self.chan_in == self.chan_out and self.reduce:
            # Downsample via pooling
            self._residual_transform = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            # Deal with channel mismatch, downsampled or not.
            self._residual_transform = nn.Sequential(
                nn.Conv2d(
                    self.chan_in,
                    self.chan_out,
                    kernel_size=1,
                    stride=2 if reduce else 1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.chan_out),
            )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        identity = fmap
        fmap = self._conv_stack(fmap)

        if self._residual_transform != None:
            identity = self._residual_transform(identity)

        # shape is now good to go
        return fmap + identity

    @property
    def chan_in(self) -> int:
        """@brief Get number of input channels for the instantiated Module"""
        return self._chan_in

    @property
    def chan_out(self) -> int:
        """@brief Get number of output channels for the instantiated Module"""
        return self._chan_out

    @property
    def reduce(self) -> bool:
        """If this performs a 2x reduction in x and y"""
        return self._reduce

    @property
    def activation_fn(self) -> nn.Module:
        return self._activation_fn
