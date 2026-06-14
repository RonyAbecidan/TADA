"""Noise-residual extractors used as pipeline fingerprints (paper Section 3.3).

The KB high-pass filter highlights inter-pixel noise correlations induced by
image processing while attenuating embedding traces. Residual maps are cropped
so that downstream patch extraction (default ``8 x 16`` from the YAML) stays
aligned with the JPEG grid.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _jpeg_grid_crop(residuals: torch.Tensor, patch_size_rows: int, patch_size_columns: int) -> torch.Tensor:
    """Crop residuals to a JPEG-aligned region usable for patch tiling."""
    width = residuals.shape[-2]
    height = residuals.shape[-1]

    close_multiple_8_w = np.ceil(width / 8)
    gap_w = int(8 * close_multiple_8_w - width)
    max_rows = int((width - 2 * gap_w) // patch_size_rows) * patch_size_rows

    close_multiple_8_h = np.ceil(height / 8)
    gap_h = int(8 * close_multiple_8_h - height)
    max_cols = int((height - 2 * gap_h) // patch_size_columns) * patch_size_columns

    return residuals[:, :, gap_w : gap_w + max_rows, gap_h : gap_h + max_cols]


class _FixedResidualFilter(nn.Module):
    """Base class for fixed 3x3 residual filters with valid convolution."""

    def __init__(self, hyperparameters, kernel: torch.Tensor):
        super().__init__()
        self.patch_size_rows = int(hyperparameters.training.patch_size_rows)
        self.patch_size_columns = int(hyperparameters.training.patch_size_columns)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv.weight.data = kernel.view(1, 1, 3, 3).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = self.conv(x)
        return _jpeg_grid_crop(residuals, self.patch_size_rows, self.patch_size_columns)


class KB2(_FixedResidualFilter):
    """Ker-Bohme (KB) high-pass filter used in the paper (Section 3.3).

    Kernel (up to the paper's 1/4 scaling factor):

    ``[[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]``
    """

    def __init__(self, hyperparameters):
        kernel = torch.tensor(
            [
                [-1.0, 2.0, -1.0],
                [2.0, -4.0, 2.0],
                [-1.0, 2.0, -1.0],
            ]
        )
        super().__init__(hyperparameters, kernel)


class identity(_FixedResidualFilter):
    """Identity filter with the same valid-convolution crop as ``KB2``.

    Used to map UERD probability maps into the same residual geometry as KB
    residuals before patch-wise PMAP filtering on the target set.
    """

    def __init__(self, hyperparameters):
        kernel = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        super().__init__(hyperparameters, kernel)
