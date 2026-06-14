"""Differentiable JPEG compression used after the TADA emulator (Section 3.1).

The emulator output is JPEG-compressed with the target quantization table before
KB residuals are computed. This mirrors the paper pipeline:
TIF -> learned development -> differentiable JPEG -> noise residuals.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from modules import compress_jpeg, decompress_jpeg
from utils import diff_round, read_qtable_txt, read_qtables_from_jpeg, standard_qtable


class DiffJPEG(nn.Module):
    """Differentiable JPEG layer with explicit quantization-table handling.

    Table resolution order:
    1. ``qf_reference_image``: extract tables from a JPEG file.
    2. ``qf_table_txt`` (+ optional chroma table): read 8x8 tables from text.
    3. ``quality``: fall back to a standard IJG table (100 -> all ones).
    """

    def __init__(
        self,
        height: int,
        width: int,
        differentiable: bool = True,
        quality: int = 100,
        qf_table_txt: str | None = None,
        qf_chroma_table_txt: str | None = None,
        qf_reference_image: str | None = None,
    ):
        super().__init__()
        rounding = diff_round if differentiable else torch.round
        y_table, c_table = self._load_tables(
            quality=quality,
            qf_table_txt=qf_table_txt,
            qf_chroma_table_txt=qf_chroma_table_txt,
            qf_reference_image=qf_reference_image,
        )

        self.y_table = nn.Parameter(torch.from_numpy(y_table.astype(np.float32)), requires_grad=False)
        self.c_table = nn.Parameter(torch.from_numpy(c_table.astype(np.float32)), requires_grad=False)
        self.compress = compress_jpeg(rounding=rounding, tables=(self.y_table, self.c_table))
        self.decompress = decompress_jpeg(height, width, rounding=rounding, tables=(self.y_table, self.c_table))

    @staticmethod
    def _load_tables(
        quality: int,
        qf_table_txt: str | None,
        qf_chroma_table_txt: str | None,
        qf_reference_image: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if qf_reference_image:
            return read_qtables_from_jpeg(qf_reference_image)

        if qf_table_txt:
            y_table = read_qtable_txt(qf_table_txt)
            c_table = read_qtable_txt(qf_chroma_table_txt) if qf_chroma_table_txt else np.ones((8, 8), dtype=np.float32)
            return y_table, c_table

        return standard_qtable(quality, chroma=False), standard_qtable(quality, chroma=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress and immediately decompress ``x`` in RGB ``[0, 1]`` format."""
        y, cb, cr = self.compress(x)
        return self.decompress(y, cb, cr)
