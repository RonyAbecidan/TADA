"""TADA pipeline emulator.

The paper models image development as a constrained symmetric convolution whose
weights sum to one. ``ConvDeveloper`` implements this emulator with a configurable
odd ``kernel_size`` (valid convolution, no padding). The sharpen toy experiment
uses ``kernel_size: 3``; larger kernels (e.g. 5x5) are used in the operational
experiments described in the paper.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvDeveloper(nn.Module):
    """Learnable TADA pipeline emulator (Section 3.1 / 3.2 of the paper).

    A single valid ``kernel_size x kernel_size`` convolution is applied
    independently to each RGB channel, then combined with learnable convex
    weights ``alpha`` and ``beta``. The kernel size is read from
    ``hyperparameters.developer.args.kernel_size`` (default in this repo: 3).

    The output is min-max normalized to ``[0, 1]``; the running ``x_min`` /
    ``x_max`` values are exported in the hyperparameters so the learned pipeline
    can be replayed after training.

    At the end of each epoch the kernel is projected onto the paper constraints:
    symmetry and unit sum (``norm_constraint`` / ``sym_constraint``).
    """

    def __init__(self, hyperparameters):
        super().__init__()
        args = hyperparameters.developer.args
        self.kernel_size = int(args.kernel_size)
        # Valid convolution (padding=0); no symmetric padding despite the paper's centered kernel view.
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, padding=0, stride=1, bias=False)
        nn.init.dirac_(self.conv.weight)

        self.alpha = nn.Parameter(torch.tensor(0.299, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.587, dtype=torch.float32))
        self.x_min = torch.tensor(float(args.x_min), dtype=torch.float32)
        self.x_max = torch.tensor(float(args.x_max), dtype=torch.float32)

    def norm_constraint(self) -> None:
        """Project kernel weights so they sum to one (paper Section 3.2)."""
        with torch.no_grad():
            weight_sum = self.conv.weight.data.sum()
            if torch.isclose(weight_sum, torch.tensor(0.0, device=weight_sum.device)):
                return
            self.conv.weight.data /= weight_sum

    def sym_constraint(self) -> None:
        """Average kernel weights over the four 90-degree rotations (symmetry)."""
        with torch.no_grad():
            weight = self.conv.weight.data
            self.conv.weight.data = (
                weight
                + torch.rot90(weight, k=1, dims=[2, 3])
                + torch.rot90(weight, k=2, dims=[2, 3])
                + torch.rot90(weight, k=3, dims=[2, 3])
            ) / 4.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Develop a color TIF patch into a normalized grayscale-like tensor."""
        red = self.conv(x[:, 0, :, :].unsqueeze(1))
        green = self.conv(x[:, 1, :, :].unsqueeze(1))
        blue = self.conv(x[:, 2, :, :].unsqueeze(1))
        x = self.alpha * red + self.beta * green + (1.0 - self.alpha - self.beta) * blue

        if self.training:
            with torch.no_grad():
                self.x_min = x.min().detach()
                self.x_max = x.max().detach()

        denom = torch.clamp(self.x_max.to(x.device) - self.x_min.to(x.device), min=1e-8)
        return (x - self.x_min.to(x.device)) / denom
