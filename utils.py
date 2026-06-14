"""JPEG helpers for the differentiable compression layer used by TADA.

Quantization tables can be built from a standard quality factor, read from a
text file, or extracted from a reference JPEG image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


STANDARD_LUMA_QTABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

STANDARD_CHROMA_QTABLE = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


def diff_round(x: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of JPEG coefficient rounding (DiffJPEG)."""
    return torch.round(x) + (x - torch.round(x)) ** 3


def quality_to_factor(quality: int) -> float:
    """Return the IJG scaling factor used for standard JPEG tables."""
    if not 1 <= int(quality) <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")
    if quality < 50:
        return 5000.0 / quality
    return 200.0 - quality * 2.0


def standard_qtable(quality: int, chroma: bool = False) -> np.ndarray:
    """Build a standard 8x8 JPEG quantization table for a given quality.

    Quality 100 returns an all-ones table, matching the toy sharpen experiment.
    """
    if int(quality) == 100:
        return np.ones((8, 8), dtype=np.float32)

    base = STANDARD_CHROMA_QTABLE if chroma else STANDARD_LUMA_QTABLE
    factor = quality_to_factor(int(quality))
    table = np.floor((base * factor + 50.0) / 100.0)
    table = np.clip(table, 1, 255)
    return table.astype(np.float32)


def read_qtable_txt(path: str | Path) -> np.ndarray:
    """Read an 8x8 quantization table from a whitespace/comma separated text file."""
    text = Path(path).read_text()
    values = np.fromstring(text.replace(",", " "), sep=" ", dtype=np.float32)
    if values.size != 64:
        raise ValueError(f"Expected 64 quantization values in {path}, got {values.size}")
    return values.reshape(8, 8).astype(np.float32)


def read_qtables_from_jpeg(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract luminance and chrominance quantization tables from a JPEG image."""
    from PIL import Image

    with Image.open(path) as image:
        quantization = image.quantization

    if not quantization:
        raise ValueError(f"No JPEG quantization table found in {path}")

    y_table = np.array(quantization[min(quantization.keys())], dtype=np.float32).reshape(8, 8)
    chroma_key = 1 if 1 in quantization else min(quantization.keys())
    c_table = np.array(quantization[chroma_key], dtype=np.float32).reshape(8, 8)
    return y_table, c_table
