#!/usr/bin/env python3
"""Extract every frame from a teleoperation HDF5 into per-frame folders."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import imageio.v2 as imageio
import numpy as np


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Convert image data to uint8 for saving."""
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0) * 255.0
    img = np.clip(img, 0, 255).round().astype(np.uint8)
    return img


def validate_shapes(action_ds: h5py.Dataset, cam_ds: h5py.Dataset, qpos_ds: h5py.Dataset) -> int:
    total = cam_ds.shape[0]
    if action_ds.shape[0] != total or qpos_ds.shape[0] != total:
        raise ValueError(
            f"Inconsistent episode length: action={action_ds.shape[0]}, "
            f"cam_high={total}, qpos={qpos_ds.shape[0]}"
        )
    return total


def ensure_indices(total: int, indices: Sequence[int] | None) -> Iterable[int]:
    if indices:
        for idx in indices:
            if idx < 0 or idx >= total:
                raise IndexError(f"Index {idx} out of range (total {total})")
        return indices
    return range(total)


def extract_frames(h5_path: Path, out_dir: Path, indices: Sequence[int] | None, overwrite: bool) -> None:
    with h5py.File(h5_path, "r") as f:
        action_ds = f["action"]
        cam_ds = f["observations/images/cam_high"]
        qpos_ds = f["observations/qpos"]

        total = validate_shapes(action_ds, cam_ds, qpos_ds)
        frame_indices = ensure_indices(total, indices)

        for idx in frame_indices:
            frame_dir = out_dir / f"frame_{idx:06d}"
            if frame_dir.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"{frame_dir} already exists. Use --overwrite to replace existing frames."
                    )
            frame_dir.mkdir(parents=True, exist_ok=True)

            action = action_ds[idx].tolist()
            qpos = qpos_ds[idx].tolist()
            img = normalize_image(np.asarray(cam_ds[idx]))

            json_path = frame_dir / "data.json"
            with json_path.open("w", encoding="utf-8") as fp:
                json.dump({"action": action, "qpos": qpos}, fp, ensure_ascii=False, indent=2)

            img_path = frame_dir / "cam_high.png"
            imageio.imwrite(img_path, img)

            print(f"Saved frame {idx} -> {frame_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-frame action/qpos JSON and cam_high image from a teleoperation HDF5."
    )
    parser.add_argument("input", type=Path, help="Path to teleoperation HDF5 file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: <input>_frames)",
    )
    parser.add_argument(
        "-i",
        "--indices",
        type=int,
        nargs="+",
        help="Specific frame indices to extract (default: all frames)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing frame folders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path: Path = args.input
    out_dir: Path = args.output or h5_path.with_name(f"{h5_path.stem}_frames")

    if not h5_path.exists():
        raise FileNotFoundError(f"Input file '{h5_path}' does not exist")

    out_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(h5_path, out_dir, args.indices, args.overwrite)


if __name__ == "__main__":
    main()
