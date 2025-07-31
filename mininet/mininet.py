# James Clampffer - 2025
"""
Resnet-inspired architecture to experiment with lightweight classifiers.
- Depthwise separable convolutions + SE blocks to reduce parameters.
- Configurable activation for everything other than the stem.
"""

import torch.nn as nn
import torch.jit
import typing

from blocks import MiniResBlock
from torchvision.ops import SqueezeExcitation


class MiniNet(nn.Module):
    """
    @brief Shallow resnet-like architecture for image classification.
    @note Set up for 3 input channels. Hasn't been tested otherwise.

    Structure:
    5x5 conv -> 3x3 conv -> {MiniResBlock->SqueezeExcite} * 6 -> FC

    Goals:
    - Good accuracy with minimal parameters.
    - Easy to reason about and modify.

    Accuracy per parameter is good, but FLOPs per inference are awful
    compared to architectures like MobileNet and ShuffleNet. That said,
    if the memory hierarchy fits this in LL$, nothing needs to hit the
    bus.
    """

    __slots__ = (
        "_num_classes",
        "_activation_fn",
        "_squeeze_factor",
        "_stem",
        "_backbone",
        "_classifier",
    )
    _num_classes: int
    _activation_fn: type[nn.Module]
    _squeeze_factor: int
    _stem: nn.Module
    _backbone: nn.Module
    _classifier: nn.Module

    def __init__(
        self,
        num_classes: int,
        activation_fn: type[nn.Module] = nn.SiLU,
        squeeze_factor: int = 8,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._activation_fn = activation_fn
        self._squeeze_factor = squeeze_factor

        # Full convs for early feature extraction. ReLU to get some
        # discontinuity early on, use mish or similar for backprop flow
        # through deeper layers.
        self._stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Giving the SE input channels calculate the bottleneck width
        squeeze = lambda ch: int(ch / float(self._squeeze_factor))

        # Alternate res blocks and SE blocks for channel attention.
        # todo: Need to take a look at a skip-layer SE block and pulling the
        # SE into the MiniResBlock
        self._backbone = nn.Sequential(
            # Add channels
            MiniResBlock(32, 64, activation=activation_fn),
            SqueezeExcitation(64, squeeze(64), activation=activation_fn),
            MiniResBlock(64, 64, activation=activation_fn),
            SqueezeExcitation(64, squeeze(64), activation=activation_fn),
            # Quickly expand channels while downsampling x3
            MiniResBlock(64, 128, reduce=True, activation=activation_fn),
            SqueezeExcitation(128, squeeze(128), activation=activation_fn),
            MiniResBlock(128, 256, reduce=True, activation=activation_fn),
            SqueezeExcitation(256, squeeze(256), activation=activation_fn),
            MiniResBlock(256, 256, reduce=True, activation=activation_fn),
            SqueezeExcitation(256, squeeze(256), activation=activation_fn),
            MiniResBlock(256, 256, activation=activation_fn),
            SqueezeExcitation(256, squeeze(256), activation=activation_fn),
        )

        # Simple Pool->FC
        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    @torch.jit.export
    def forward_pass(self, fmap: torch.Tensor) -> torch.Tensor:
        fmap = self._stem(fmap)
        fmap = self._backbone(fmap)
        return self._classifier(fmap)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.forward_pass(fmap)

    @property
    def num_classes(self) -> int:
        """@brief Get the number of output classes"""
        return self._num_classes

    @property
    def activation_fn(self) -> nn.Module:
        """@brief See what activation function the model is using"""
        return self._activation_fn

    @property
    def squeeze_factor(self) -> int:
        """@brief Get channel reduction ratio SE blocks are using"""
        return self._squeeze_factor

    @property
    def arch_name(self) -> str:
        """@brief Get a name suitable for logging"""
        return "mininet"

    def to_file(self, epoch: int, optimizer, scheduler, path: str) -> None:
        """
        @brief Store model and training state to a file.
        @warn  The training params are not restored by default.
        @todo  Track accuracy in the pickled object or write a manifest
               file with info of that nature.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "activation_fn": self._activation_fn.__class__.__name__,
            "squeeze_factor": self._squeeze_factor,
            "num_classes": self._num_classes,
        }
        path = path + "_" + self.arch_name
        torch.save(state, path)

    @classmethod
    def from_file(
        cls, path: str, activation_resolver: dict
    ) -> tuple[nn.Module, typing.Any]:
        """
        @brief Load model with weights that have already been trained.
        @warn  Scheduler and optimizer state are stored in to_file but
               are not used here. Generally using this to fine tune a
               base model.

        """
        checkpoint = torch.load(path)

        activation_cls = activation_resolver(checkpoint["activation_fn"])
        model = cls(
            num_classes=checkpoint["num_classes"],
            activation_fn=activation_cls,
            squeeze_factor=checkpoint["squeeze_factor"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        return model, checkpoint


if __name__ == "__main__":
    pass
