"""Train the TADA pipeline emulator (Target Alignment through Data Adaptation).

This script reproduces the sharpen toy experiment from the paper:
  python pipeline_learning_config.py sharpen100_bpnzac.yaml cuda

Data flow (Section 3.1):
  1. Sample color TIF crops from ``color_raws_512.hdf5`` (relatively uniform ALASKA crops).
  2. Develop them with the learnable ``ConvDeveloper`` emulator.
  3. Apply differentiable JPEG with the target quantization table.
  4. Extract KB residuals from source and target patches.
  5. Minimize the TADA loss (Section 3.4 / 4.2):
       - covariance alignment   (``lamb_cov``; geometric matching)
       - correlation alignment  (``lamb_corr``; speeds up training vs. covariance alone)
       - Wasserstein distance   (``lamb_mmd``; distribution matching, complementary to cov/corr)
       - realism (``dev``)          (paper: L2; here MMD on spatial patches for better practical results)

Target-side patch selection uses UERD probability maps to ease the learning of the target processing pipeline.
"""

from __future__ import annotations

import argparse
import os
import pickle
from math import floor
from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import yaml
from geomloss import SamplesLoss
from munch import DefaultMunch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils import data

from developers import ConvDeveloper
from DiffJPEG import DiffJPEG
from residual_extractors import KB2, identity

REPO_ROOT = Path(__file__).resolve().parent
DEVELOPERS = {"ConvDeveloper": ConvDeveloper}
RESIDUAL_EXTRACTORS = {"KB2": KB2, "identity": identity}


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    value = getattr(config, key, default)
    return default if isinstance(value, DefaultMunch) and value == {} else value


def resolve_repo_path(path: str | os.PathLike[str] | None) -> str | None:
    if path in (None, ""):
        return None
    path = Path(path)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def ensure_lightning_mps_support() -> None:
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return

    try:
        from lightning_fabric.accelerators.mps import MPSAccelerator
    except ImportError:
        return

    if not MPSAccelerator.is_available():
        MPSAccelerator.is_available = staticmethod(lambda: torch.backends.mps.is_available())


def resolve_compute_device(device_arg: str = "auto") -> tuple[torch.device, str, int | None]:
    arg = str(device_arg).strip().lower()

    if arg in ("auto", "default"):
        if torch.cuda.is_available():
            return torch.device("cuda:0"), "cuda", 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps"), "mps", None
        return torch.device("cpu"), "cpu", None

    if arg in ("mps", "apple", "mac"):
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available on this system")
        return torch.device("mps"), "mps", None

    if arg == "cpu":
        return torch.device("cpu"), "cpu", None

    if arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({arg}) but not available")
        device = torch.device(arg if ":" in arg else "cuda:0")
        return device, "cuda", device.index if device.index is not None else 0

    if arg.isdigit():
        index = int(arg)
        if torch.cuda.is_available():
            return torch.device(f"cuda:{index}"), "cuda", index
        print(f"Note: CUDA index '{index}' ignored, using CPU")
        return torch.device("cpu"), "cpu", None

    raise ValueError(f"Unknown device '{device_arg}'. Use auto, cpu, mps, cuda, cuda:N, or a GPU index.")


def build_trainer_kwargs(
    device_type: str,
    cuda_index: int | None,
    max_epochs: int,
    callbacks: list[EarlyStopping],
    deterministic: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_epochs": max_epochs,
        "enable_progress_bar": True,
        "callbacks": callbacks,
        "deterministic": deterministic,
    }
    if device_type == "cuda":
        kwargs["accelerator"] = "gpu"
        kwargs["devices"] = [cuda_index]
    elif device_type == "mps":
        kwargs["accelerator"] = "mps"
        kwargs["devices"] = 1
    else:
        kwargs["accelerator"] = "cpu"
        kwargs["devices"] = 1
    return kwargs


class MyEarlyStopping(EarlyStopping):
    """Run custom validation before PyTorch Lightning early-stopping checks."""

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        trainer.model.validation()
        self._run_early_stopping_check(trainer)


class MyDataset(data.Dataset):
    """HDF5 dataset for unsupervised TADA pipeline learning.

    Each sample returns:
      - ``train_datum``: color TIF crop from the source pool.
      - ``ope_datum``: grayscale operational target patch.
      - ``eval_datum``: grayscale eval target patch (disjoint hold-out set).
      - ``pmap_datum``: UERD probability map aligned with the operational patch.

    The paper augments source and target with the same orthogonal rotation. In this
    repository, rotations are derived from the sample index with fixed offsets:
    source ``index % 4``, operational / PMAP ``(index + 1) % 4``, eval ``(index + 2) % 4``.
    """
    @staticmethod
    def _as_2d(array: np.ndarray) -> torch.Tensor:
        tensor = torch.tensor(array).float()
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            return tensor[..., 0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            return tensor[0]
        raise ValueError(f"Expected a 2D image-like array, got shape {tuple(tensor.shape)}")

    @staticmethod
    def _floor_multiple(n: int, base: int = 8) -> int:
        return int(n // base) * base

    @classmethod
    def _align_jpeg_grid(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Crop to the largest top-left region with JPEG-friendly 8x8 alignment."""
        height = cls._floor_multiple(int(tensor.shape[-2]), 8)
        width = cls._floor_multiple(int(tensor.shape[-1]), 8)
        if height <= 0 or width <= 0:
            raise ValueError(f"Image is too small for JPEG grid alignment: {tuple(tensor.shape)}")
        return tensor[:height, :width]

    @classmethod
    def _is_usable(cls, array: np.ndarray, patch_size: int) -> bool:
        try:
            tensor = cls._align_jpeg_grid(cls._as_2d(array))
        except ValueError:
            return False
        return tensor.shape[-2] >= patch_size and tensor.shape[-1] >= patch_size

    @staticmethod
    def _extract_patch_2d(tensor: torch.Tensor, row: int, col: int, patch_size: int) -> torch.Tensor:
        return tensor[row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size]

    @staticmethod
    def _patch_modulo(array: np.ndarray, patch_size: int) -> int:
        spatial_shape = array.shape[:2] if array.ndim == 3 and array.shape[-1] in (1, 3) else array.shape[-2:]
        return min(spatial_shape) // patch_size

    def __init__(self, hyperparameters):
        """Load HDF5 archives and subsample source / operational pools from YAML."""
        seed_everything(hyperparameters.seed)
        self.hyperparameters = hyperparameters
        self.archive_source = h5.File(hyperparameters.training.source, "r")
        self.archive_target = h5.File(hyperparameters.training.target, "r")

        patch_size = int(hyperparameters.im_size_source)
        source_count = len(self.archive_source["train"])
        target_count = len(self.archive_target["operational"])
        source_indices = np.random.choice(
            np.arange(source_count),
            size=min(source_count, hyperparameters.training.n_samples),
            replace=False,
        )
        target_indices = np.random.choice(
            np.arange(target_count),
            size=min(target_count, hyperparameters.operational.n_samples),
            replace=False,
        )

        train = np.array(self.archive_source["train"])[source_indices].astype(np.float32)
        train_mask = np.array([self._is_usable(train[i], patch_size) for i in range(len(train))])
        self.train = train[train_mask]
        if len(self.train) == 0:
            raise ValueError(f"No valid source images found in {hyperparameters.training.source}")

        operational = np.array(self.archive_target["operational"])[target_indices]
        pmap_ope = np.array(self.archive_target["pmap_ope"])[target_indices]
        valid_operational = [
            i for i in range(len(operational)) if self._is_usable(operational[i], patch_size)
        ]
        self.ope = operational[valid_operational]
        self.pmap_ope = pmap_ope[valid_operational]
        if len(self.ope) == 0:
            raise ValueError(f"No valid operational images found in {hyperparameters.training.target}")

        self.eval = self.archive_target["eval"]
        self.eval_indices = [
            i for i in range(len(self.eval)) if self._is_usable(np.array(self.eval[i]), patch_size)
        ]
        if len(self.eval_indices) == 0:
            raise ValueError(f"No valid eval images found in {hyperparameters.training.target}")

        self.modulo_source = max(1, min(self.train.shape[-2], self.train.shape[-3]) // patch_size)

    def _getitem_impl(self, train_idx: int, ope_idx: int, index: int):
        """Build one source/target/eval/pmap tuple with random crops and rotations."""
        size = int(self.hyperparameters.im_size_source)
        train_raw = self.train[train_idx]
        modulo_source = self._patch_modulo(train_raw, size)
        if modulo_source < 1:
            return None

        ope_2d = self._align_jpeg_grid(self._as_2d(self.ope[ope_idx]))
        modulo_target = min(ope_2d.shape[-2] // size, ope_2d.shape[-1] // size)
        if modulo_target < 1:
            return None

        i = int(torch.randint(0, modulo_source, (1,)).item())
        j = int(torch.randint(0, modulo_source, (1,)).item())
        k = int(torch.randint(0, modulo_target, (1,)).item())
        l = int(torch.randint(0, modulo_target, (1,)).item())
        m = int(torch.randint(0, modulo_target, (1,)).item())
        n = int(torch.randint(0, modulo_target, (1,)).item())

        train_datum = torch.tensor(train_raw).permute(2, 0, 1)
        train_datum = train_datum[:, size * i : size * (i + 1), size * j : size * (j + 1)]
        train_datum = torch.rot90(train_datum, k=index % 4, dims=[1, 2])

        ope_datum = self._extract_patch_2d(ope_2d, k, l, size)
        ope_datum = torch.rot90(ope_datum, k=(index + 1) % 4, dims=[0, 1])

        pmap_2d = self._align_jpeg_grid(self._as_2d(self.pmap_ope[ope_idx]))
        pmap_datum = self._extract_patch_2d(pmap_2d, k, l, size)
        pmap_datum = torch.rot90(pmap_datum, k=(index + 1) % 4, dims=[0, 1])

        eval_idx = self.eval_indices[index % len(self.eval_indices)]
        eval_2d = self._align_jpeg_grid(self._as_2d(np.array(self.eval[eval_idx])))
        eval_datum = self._extract_patch_2d(eval_2d, m, n, size)
        eval_datum = torch.rot90(eval_datum, k=(index + 2) % 4, dims=[0, 1])

        return train_datum.float(), ope_datum.float(), eval_datum.float(), pmap_datum.float()

    def __getitem__(self, index: int):
        n_attempts = max(len(self.train), len(self.ope))
        for attempt in range(n_attempts):
            train_idx = (index + attempt) % len(self.train)
            ope_idx = (index + attempt) % len(self.ope)
            sample = self._getitem_impl(train_idx, ope_idx, index)
            if sample is not None:
                return sample
        raise RuntimeError("No usable image pair found after skipping invalid samples")

    def __len__(self) -> int:
        return self.modulo_source * self.modulo_source * len(self.train)

    def close(self) -> None:
        self.archive_source.close()
        self.archive_target.close()


def to_yamlable(value: Any) -> Any:
    if isinstance(value, DefaultMunch):
        return {key: to_yamlable(val) for key, val in value.items()}
    if isinstance(value, dict):
        return {key: to_yamlable(val) for key, val in value.items()}
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def result_folder(hyperparameters) -> Path:
    return REPO_ROOT / "Results" / (
        f"developer={hyperparameters.developer.name}"
        f"-extractor={hyperparameters.residual_extractor.name}"
        f"-qf={hyperparameters.training.qf}"
        f"-{hyperparameters.precisions}"
    )


def save_hyperparameters(hyperparameters) -> None:
    folder_path = result_folder(hyperparameters)
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = (
        folder_path
        / f"hyperparameters-{hyperparameters.residual_extractor.name}"
        f"-{hyperparameters.training.qf}"
        f"-{hyperparameters.operational.n_samples}.yaml"
    )
    file_path.write_text(yaml.safe_dump(to_yamlable(hyperparameters), sort_keys=False))


class PipelineLearner(pl.LightningModule):
    """PyTorch Lightning module implementing TADA pipeline learning.

    The trainable part is the ``ConvDeveloper`` emulator. KB residuals, the
    identity filter, and differentiable JPEG are frozen feature extractors.
    """

    def __init__(self, hyperparameters, dataloader):
        super().__init__()
        seed_everything(hyperparameters.seed)
        self.hyperparameters = hyperparameters
        self.developer = DEVELOPERS[hyperparameters.developer.name](hyperparameters)

        with torch.no_grad():
            developed = self.developer(
                torch.randn(1, 3, hyperparameters.im_size_source, hyperparameters.im_size_source)
            )
            _, _, width, height = developed.size()
            self.W_2 = hyperparameters.training.patch_size_rows * floor(
                width / hyperparameters.training.patch_size_rows
            )
            self.H_2 = hyperparameters.training.patch_size_columns * floor(
                height / hyperparameters.training.patch_size_columns
            )

        self.jpeg_compressor = DiffJPEG(
            height=self.W_2,
            width=self.H_2,
            differentiable=True,
            quality=hyperparameters.training.qf,
            qf_table_txt=cfg_get(hyperparameters.training, "qf_table_txt"),
            qf_chroma_table_txt=cfg_get(hyperparameters.training, "qf_chroma_table_txt"),
            qf_reference_image=cfg_get(hyperparameters.training, "qf_reference_image"),
        )
        self.dataloader = dataloader
        self.noise_residuals = RESIDUAL_EXTRACTORS[hyperparameters.residual_extractor.name](hyperparameters)
        self.id_res = RESIDUAL_EXTRACTORS["identity"](hyperparameters)

        for module in (self.noise_residuals, self.id_res, self.jpeg_compressor):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.epoch_count = 0
        self.cpt = 0
        self.previous_loss = float("inf")
        self.loss_epoch: dict[int, list[list[float]]] = {}
        self.x_min: list[float] = []
        self.x_max: list[float] = []
        self.lamb_mmd = hyperparameters.training.lamb_mmd
        self.lamb_corr = hyperparameters.training.lamb_corr
        self.lamb_cov = hyperparameters.training.lamb_cov
        self.folder_path = result_folder(hyperparameters)
        self.folder_path.mkdir(parents=True, exist_ok=True)
        for checkpoint in self.folder_path.glob("*.pt"):
            checkpoint.unlink()

    def pmap_to_residual_structure(self, pmap: torch.Tensor, like_residuals: torch.Tensor) -> torch.Tensor:
        """Map a UERD probability map into the KB residual geometry."""
        pmap_res = self.id_res(pmap)
        if pmap_res.shape[1] != like_residuals.shape[1]:
            pmap_res = pmap_res.repeat(1, like_residuals.shape[1], 1, 1)
        return pmap_res

    @staticmethod
    def split_residuals(residuals: torch.Tensor, patch_size_rows: int, patch_size_columns: int) -> list[torch.Tensor]:
        """Split residual maps into patches of shape ``patch_size_rows x patch_size_columns``."""
        patches = residuals.unfold(2, patch_size_rows, patch_size_rows).unfold(
            3, patch_size_columns, patch_size_columns
        )
        patches = patches.permute(1, 0, 2, 3, 4, 5).contiguous()
        channels, batch, rows, cols, patch_rows, patch_cols = patches.shape
        return [
            patches[channel].reshape(batch * rows * cols, patch_rows, patch_cols)
            for channel in range(channels)
        ]

    def draw_random_choice(self, tensor: torch.Tensor, size: int) -> torch.Tensor:
        if tensor.size(0) == 0:
            raise ValueError("Cannot sample from an empty tensor")
        size_final = min(size, tensor.size(0))
        idx = torch.randperm(tensor.size(0), device=tensor.device)[:size_final]
        return tensor[idx].to(self.device)

    def select_spatial_split(
        self,
        residuals: torch.Tensor,
        nb_samples: int,
        patch_size_rows: int | None = None,
        patch_size_columns: int | None = None,
    ) -> list[torch.Tensor]:
        patch_size_rows = patch_size_rows or self.hyperparameters.training.patch_size_rows
        patch_size_columns = patch_size_columns or self.hyperparameters.training.patch_size_columns
        return [
            self.draw_random_choice(channel, nb_samples)
            for channel in self.split_residuals(residuals, patch_size_rows, patch_size_columns)
        ]

    def select_residual_split(
        self,
        selection: bool,
        residuals: torch.Tensor,
        spatial: torch.Tensor,
        nb_samples: int,
        patch_size_rows: int | None = None,
        patch_size_columns: int | None = None,
        pmap: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Sample residual patches for covariance / distribution matching.

        When ``selection=True`` (target side), the paper keeps patches with:
          - residual standard deviation > 1 (non-flat regions);
          - low UERD embedding probability (robustness to stego).

        The PMAP threshold starts at 0.1 in this repository 
        and is relaxed adaptively by +0.01 if the filter would otherwise discard every patch in a channel.
        """
        patch_size_rows = patch_size_rows or self.hyperparameters.training.patch_size_rows
        patch_size_columns = patch_size_columns or self.hyperparameters.training.patch_size_columns
        split_res = self.split_residuals(residuals, patch_size_rows, patch_size_columns)
        split_spatial = self.split_residuals(spatial, patch_size_rows, patch_size_columns)
        split_pmap = None
        if pmap is not None:
            if pmap.shape[1] != residuals.shape[1]:
                pmap = pmap.repeat(1, residuals.shape[1], 1, 1)
            split_pmap = self.split_residuals(pmap, patch_size_rows, patch_size_columns)

        selected_channels = []
        for idx, (res_channel, spatial_channel) in enumerate(zip(split_res, split_spatial)):
            variances = res_channel.std(dim=(1, 2)).abs()
            variance_filter = variances > 1.0
            filtered_res = res_channel[variance_filter]
            if filtered_res.size(0) == 0:
                filtered_res = res_channel

            if selection and split_pmap is not None:
                filtered_pmap = split_pmap[idx][variance_filter]
                if filtered_pmap.size(0) == 0:
                    filtered_pmap = split_pmap[idx]
                max_probs = filtered_pmap.amax(dim=(1, 2))
                threshold = 0.1
                pmap_filter = max_probs < threshold
                while pmap_filter.sum().item() == 0 and threshold < 1.0:
                    threshold = min(threshold + 0.01, 1.0)
                    pmap_filter = max_probs < threshold
                selected = filtered_res[pmap_filter] if pmap_filter.sum().item() > 0 else filtered_res
            elif selection:
                stds = filtered_res.std(dim=(1, 2))
                sorted_indices = torch.argsort(stds)
                if sorted_indices.numel() > 2 * nb_samples:
                    sorted_indices = sorted_indices[nb_samples:-nb_samples]
                selected = filtered_res[sorted_indices] if sorted_indices.numel() else filtered_res
            else:
                selected = filtered_res

            selected_channels.append(self.draw_random_choice(selected, nb_samples))

        return selected_channels

    @staticmethod
    def one_to_three(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor.expand(-1, 3, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply emulator + differentiable JPEG and return a grayscale JPEG image."""
        x = self.developer(x)
        x = x[:, :, 0 : self.W_2, 0 : self.H_2]
        x = self.one_to_three(x)
        x = self.jpeg_compressor(x)
        return x[:, 0, :, :].unsqueeze(1) * 255.0

    def cov_loss(self, source_residuals: list[torch.Tensor], target_residuals: list[torch.Tensor]) -> torch.Tensor:
        """Frobenius distance between source and target residual covariances (lambda term)."""
        cov_dist = None
        feature_size = self.hyperparameters.training.patch_size_rows * self.hyperparameters.training.patch_size_columns
        for res_s, res_t in zip(source_residuals, target_residuals):
            res_s = res_s.reshape(-1, feature_size)
            res_t = res_t.reshape(-1, feature_size)
            dist = torch.norm(torch.cov(res_s.T) - torch.cov(res_t.T))
            cov_dist = dist if cov_dist is None else cov_dist + dist
        return torch.nan_to_num(cov_dist, nan=100.0)

    def corr_loss(self, source_residuals: list[torch.Tensor], target_residuals: list[torch.Tensor]) -> torch.Tensor:
        """Frobenius distance between source and target residual correlation matrices."""
        corr_dist = None
        feature_size = self.hyperparameters.training.patch_size_rows * self.hyperparameters.training.patch_size_columns
        for res_s, res_t in zip(source_residuals, target_residuals):
            res_s = res_s.reshape(-1, feature_size)
            res_t = res_t.reshape(-1, feature_size)
            dist = torch.norm(torch.corrcoef(res_s.T) - torch.corrcoef(res_t.T))
            corr_dist = dist if corr_dist is None else corr_dist + dist
        return torch.nan_to_num(corr_dist, nan=100.0)

    def mmd_loss(self, source_residuals: list[torch.Tensor], target_residuals: list[torch.Tensor]) -> torch.Tensor:
        """Wasserstein-1 / Earth Mover distance between residual patch sets (mu term).

        ``geomloss.SamplesLoss(p=1)`` implements the distributional matching term
        described in Section 3.4 of the paper.
        """
        compute_on_cpu = self.device.type == "mps"
        loss = SamplesLoss(p=1, blur=0.01, backend="online")
        mmd_dist = None
        feature_size = self.hyperparameters.training.patch_size_rows * self.hyperparameters.training.patch_size_columns
        for res_s, res_t in zip(source_residuals, target_residuals):
            res_s = res_s.reshape(-1, feature_size)
            res_t = res_t.reshape(-1, feature_size)
            if compute_on_cpu:
                res_s, res_t = res_s.cpu(), res_t.cpu()
            dist = loss(res_s, res_t)
            mmd_dist = dist if mmd_dist is None else mmd_dist + dist
        return mmd_dist.to(self.device) if compute_on_cpu else mmd_dist

    def on_train_epoch_start(self) -> None:
        self.train()
        self.epoch_count += 1

    def train_dataloader(self):
        return self.dataloader

    def _loss_terms(self, source: torch.Tensor, operational: torch.Tensor, eval_tensor: torch.Tensor, pmap: torch.Tensor):
        """Compute all TADA loss components for one mini-batch."""
        row = self.hyperparameters.training.patch_size_rows
        column = self.hyperparameters.training.patch_size_columns
        if self.cpt == 0 and not hasattr(self, "W_T"):
            _, width, height = operational.size()
            self.W_T = row * floor(width / row)
            self.H_T = column * floor(height / column)

        operational = operational[:, 0 : self.W_T, 0 : self.H_T].unsqueeze(1)
        eval_tensor = eval_tensor[:, 0 : self.W_T, 0 : self.H_T].unsqueeze(1)
        pmap = pmap[:, 0 : self.W_T, 0 : self.H_T].unsqueeze(1)

        source_dev = self.forward(source)
        operational_noise_res = self.noise_residuals(operational)
        pmap_adjusted = self.pmap_to_residual_structure(pmap, like_residuals=operational_noise_res)

        source_res = self.select_residual_split(
            False,
            self.noise_residuals(source_dev),
            self.id_res(source_dev),
            nb_samples=self.hyperparameters.training.n_samples_per_batch,
        )
        operational_res = self.select_residual_split(
            True,
            operational_noise_res,
            self.id_res(operational),
            nb_samples=self.hyperparameters.training.n_samples_per_batch,
            pmap=pmap_adjusted,
        )
        eval_res = self.select_residual_split(
            False,
            self.noise_residuals(eval_tensor),
            self.id_res(eval_tensor),
            nb_samples=self.hyperparameters.training.n_samples_per_batch,
        )
        source_orig = self.select_spatial_split(
            source / (2**16 - 1),
            nb_samples=self.hyperparameters.training.n_samples_per_batch,
        )
        source_orig_dev = self.select_spatial_split(
            source_dev / (2**8 - 1),
            nb_samples=self.hyperparameters.training.n_samples_per_batch,
        )

        losses = {
            "mmd": self.mmd_loss(source_res, operational_res),
            "corr": self.corr_loss(source_res, operational_res),
            "cov": self.cov_loss(source_res, operational_res),
            # Realism regularizer gamma: MMD on spatial patches (paper uses L2; MMD worked better in practice).
            "dev": self.mmd_loss(source_orig, source_orig_dev),
            "eval_mmd": self.mmd_loss(source_res, eval_res),
            "eval_cov": self.cov_loss(source_res, eval_res),
        }
        return losses

    def training_step(self, batch, batch_idx):
        source, operational, eval_tensor, pmap = batch
        losses = self._loss_terms(source, operational, eval_tensor, pmap)

        if self.cpt == 0:
            # Normalize each term by its first-batch value (paper Section 4.2).
            self.ref_mmd = losses["mmd"].detach().clamp(min=1e-8)
            self.ref_corr = losses["corr"].detach().clamp(min=1e-8)
            self.ref_cov = losses["cov"].detach().clamp(min=1e-8)
            self.ref_dev = losses["dev"].detach().clamp(min=1e-8)

        total_loss = (
            self.lamb_cov * losses["cov"] / self.ref_cov
            + self.lamb_mmd * losses["mmd"] / self.ref_mmd
            + self.lamb_corr * losses["corr"] / self.ref_corr
            + losses["dev"] / self.ref_dev
        )
        # Auxiliary metric on the disjoint eval split (not used for early stopping).
        eval_loss = losses["eval_cov"] + losses["eval_mmd"]

        self.log("train_loss", total_loss, prog_bar=True, on_epoch=True)
        self.log("eval_loss", eval_loss, prog_bar=True, on_epoch=True)
        self.cpt += 1
        return total_loss

    def on_train_epoch_end(self) -> None:
        # Project the learned kernel onto the paper constraints after each epoch.
        if self.hyperparameters.training.norm_constraint:
            self.developer.norm_constraint()
        if self.hyperparameters.training.sym_constraint:
            self.developer.sym_constraint()
        self.x_min.append(round(float(self.developer.x_min.item()), 3))
        self.x_max.append(round(float(self.developer.x_max.item()), 3))
        self.hyperparameters.developer.args.x_min = self.x_min
        self.hyperparameters.developer.args.x_max = self.x_max
        print(self.developer.state_dict())

    def validation(self) -> None:
        """Custom validation used by early stopping (monitors ``current_loss``)."""
        current_loss = self.eval_pipeline()
        self.log("current_loss", current_loss, prog_bar=True, on_epoch=True)
        with (self.folder_path / "loss_epoch.pickle").open("wb") as handle:
            pickle.dump(self.loss_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if current_loss < self.previous_loss:
            torch.save(self.developer.state_dict(), self.folder_path / f"best_model_epoch_{self.epoch_count}.pt")
            torch.save(self.developer.state_dict(), self.folder_path / "best_model_epoch.pt")
            self.previous_loss = current_loss
            self.hyperparameters.training.best_score = current_loss
            save_hyperparameters(self.hyperparameters)
            print(current_loss)

    def eval_pipeline(self) -> float:
        """Average normalized cov + Wasserstein loss (early-stopping metric).

        Correlation and realism (``dev``) are optimized during training but excluded here.
        """
        self.eval()
        totals = {"mmd": 0.0, "cov": 0.0}
        self.loss_epoch[self.epoch_count] = []
        count = 0
        with torch.no_grad():
            for source, operational, eval_tensor, pmap in self.dataloader:
                source = source.to(self.device)
                operational = operational.to(self.device)
                eval_tensor = eval_tensor.to(self.device)
                pmap = pmap.to(self.device)
                losses = self._loss_terms(source, operational, eval_tensor, pmap)
                totals["mmd"] += float(losses["mmd"].item())
                totals["cov"] += float(losses["cov"].item())
                count += 1

        self.loss_epoch[self.epoch_count].append([count, totals["mmd"], totals["cov"]])
        return (
            totals["mmd"] / float(self.ref_mmd)
            + totals["cov"] / float(self.ref_cov)
        ) / max(count, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hyperparameters.training.lr,
            weight_decay=0,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10000)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


def prepare_hyperparameters(hyperparameters_path: str | os.PathLike[str], device_arg: str):
    """Load YAML config, resolve repo-relative paths, and pick a compute device."""
    hyperparameters = yaml.safe_load(Path(hyperparameters_path).read_text())
    hyperparameters = DefaultMunch.fromDict(hyperparameters)
    hyperparameters.precisions = f"{hyperparameters.scenario}-{hyperparameters.precisions}-{hyperparameters.seed}"
    hyperparameters.training.source = resolve_repo_path(hyperparameters.training.source)
    target = hyperparameters.training.target.replace(".hdf5", f"_{hyperparameters.scenario}.hdf5")
    hyperparameters.training.target = resolve_repo_path(target)
    hyperparameters.training.qf_table_txt = resolve_repo_path(cfg_get(hyperparameters.training, "qf_table_txt"))
    hyperparameters.training.qf_chroma_table_txt = resolve_repo_path(
        cfg_get(hyperparameters.training, "qf_chroma_table_txt")
    )
    hyperparameters.training.qf_reference_image = resolve_repo_path(
        cfg_get(hyperparameters.training, "qf_reference_image")
    )
    hyperparameters.training.earlystop_patience = cfg_get(
        hyperparameters.training,
        "earlystop_patience",
        1000,
    )

    seed_everything(hyperparameters.seed)
    torch_device, device_type, cuda_index = resolve_compute_device(device_arg)
    hyperparameters.device = torch_device
    return hyperparameters, device_type, cuda_index


def simulate(hyperparameters_path: str | os.PathLike[str], device_arg: str = "auto") -> None:
    """Entry point: build dataset, trainer, and fit the TADA emulator."""
    hyperparameters, device_type, cuda_index = prepare_hyperparameters(hyperparameters_path, device_arg)
    print(f"Using device: {hyperparameters.device} (accelerator={device_type})")

    if device_type == "mps":
        ensure_lightning_mps_support()

    save_hyperparameters(hyperparameters)
    dataset = MyDataset(hyperparameters)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hyperparameters.training.batch_size,
        shuffle=True,
    )
    pipeline_learner = PipelineLearner(hyperparameters, train_dataloader)

    early_stop_callback = MyEarlyStopping(
        monitor="current_loss",
        min_delta=0.0,
        patience=hyperparameters.training.earlystop_patience,
        verbose=True,
        mode="min",
    )
    deterministic = device_type != "mps"
    trainer = Trainer(
        **build_trainer_kwargs(
            device_type,
            cuda_index,
            hyperparameters.training.max_epochs,
            [early_stop_callback],
            deterministic,
        )
    )

    try:
        trainer.fit(pipeline_learner)
    finally:
        dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TADA pipeline emulator.")
    parser.add_argument("config", help="YAML configuration file, e.g. sharpen100_bpnzac.yaml")
    parser.add_argument("device", nargs="?", default="auto", help="auto, cpu, mps, cuda, cuda:N, or a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    simulate(args.config, args.device)
