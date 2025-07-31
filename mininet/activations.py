# Copyright James Clampffer - 2025
"""Activation functions that aren't built in"""
import torch
import torch.nn


class SERF(torch.nn.Module):
    """SERF activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
