# James Clampffer - 2025
"""
Resnet-inspired architecture for image classification experiments.
- Depthwise seperable convolutions + SE blocks to reduce parameters.
- configurable activation function (not passed into SE for now)
"""

import torch.nn as nn
import torch.jit
from torchvision.ops import SqueezeExcitation


class DepthwiseSeparableConvBlock(nn.Module):
    """
    Helper to build a depthwise separable convolution block to use in
    place of a typical 3x3xc.
    """

    __slots__ = ("depthwise", "pointwise", "bn", "activation")
    depthwise: nn.Conv2d
    pointwise: nn.Conv2d
    bn: nn.BatchNorm2d
    activation: nn.Module

    def __init__(
        self, chin: int, chout: int, stride: int = 1, activation: nn.Module = nn.SiLU
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            chin, chin, kernel_size=3, stride=stride, padding=1, groups=chin, bias=False
        )
        self.pointwise = nn.Conv2d(chin, chout, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(chout)
        self.activation = activation()

    def forward(self, fmap):
        fmap = self.depthwise(fmap)
        fmap = self.pointwise(fmap)
        fmap = self.bn(fmap)
        fmap = self.activation(fmap)
        return fmap


class MiniResBlock(nn.Module):
    """Residual block with two depthwise separable convolutions and a skip"""

    # Residuals passed across strided convolutions get downsampled using
    # pooling rather than a convolution - lighter weight. 1x1 conv to handle
    # channel mismatches

    __slots__ = (
        "seq",
        "use_residual",
        "reduce",
        "use_projection",
        "activation_fn",
        "downscale",
    )
    seq: nn.Module
    use_residual: bool
    reduce: bool
    activation_fn: nn.Module
    downscale: nn.Module
    projection: nn.Module

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        reduce: bool = False,
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConvBlock(
                chan_in, chan_out, stride=2 if reduce else 1, activation=activation
            ),
            DepthwiseSeparableConvBlock(chan_out, chan_out, activation=activation),
        )

        self.reduce = reduce
        self.use_residual = chan_in == chan_out
        self.downscale = nn.AvgPool2d(kernel_size=2, stride=2)
        self.projection = nn.Sequential(
            nn.Conv2d(
                chan_in, chan_out, kernel_size=1, stride=2 if reduce else 1, bias=False
            ),
            nn.BatchNorm2d(chan_out),
        )

    def forward(self, fmap):
        identity = fmap
        fmap = self.seq(fmap)

        # Add the skip back
        if self.use_residual:
            if self.reduce:
                identity = self.downscale(identity)
            fmap = fmap + identity
        else:
            identity = self.projection(identity)
            fmap = fmap + identity

        return fmap


class MiniNet(nn.Module):
    """
    Shallow resnet-like architecture for image classification.
    """

    __slots__ = "seq", "stem", "num_classes", "activation_fn", "squeeze_factor", 
    seq: nn.Module
    stem: nn.Module
    activation_fn: nn.Module
    num_classes: int
    squeeze_factor: int


    def __init__(self, num_classes, ch, activation_fn=nn.SiLU, squeeze_factor=10):
        super().__init__()
        self.num_classes = num_classes
        self.squeeze_factor = squeeze_factor
        self.activation_fn = activation_fn
        squeeze = lambda ch: int(ch / float(squeeze_factor))

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.seq = nn.Sequential(
            MiniResBlock(32, 64, reduce=False, activation=activation_fn),
            SqueezeExcitation(64, squeeze(64)),
            MiniResBlock(64, 64, reduce=False, activation=activation_fn),
            SqueezeExcitation(64, squeeze(64)),
            MiniResBlock(64, 128, reduce=True, activation=activation_fn),
            SqueezeExcitation(128, squeeze(128)),
            MiniResBlock(128, 256, reduce=True, activation=activation_fn),
            SqueezeExcitation(256, squeeze(256)),
            MiniResBlock(256, 256, reduce=True, activation=activation_fn),
            SqueezeExcitation(256, squeeze(256)),
            MiniResBlock(256, 256, activation=activation_fn),
            # Classifier
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    @torch.jit.export
    def forward_pass(self, fmap):
        fmap = self.stem(fmap)
        return self.seq(fmap)

    def forward(self, fmap):
        return self.forward_pass(fmap)
