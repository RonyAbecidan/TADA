#!/usr/bin/env python3
"""Print ConvDeveloper weights from training checkpoints (.pt files)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def format_tensor(name: str, value: torch.Tensor) -> str:
    if value.ndim == 4 and value.shape[:2] == (1, 1):
        kernel = value.squeeze().cpu()
        rows = ["  " + "  ".join(f"{x:8.4f}" for x in row) for row in kernel.tolist()]
        return f"{name} ({kernel.shape[0]}x{kernel.shape[1]}):\n" + "\n".join(rows)
    if value.numel() == 1:
        return f"{name}: {float(value.item()):.6f}"
    return f"{name}: shape={tuple(value.shape)}\n{value.cpu()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Display weights saved in TADA .pt checkpoints.")
    parser.add_argument(
        "path",
        nargs="?",
        default="Results",
        help="Checkpoint file or directory to scan (default: Results/)",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*.pt"))
        if not paths:
            raise SystemExit(f"No .pt files found under {root}")

    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        print(f"\n=== {path} ===")
        for key in sorted(state.keys()):
            print(format_tensor(key, state[key]))


if __name__ == "__main__":
    main()
