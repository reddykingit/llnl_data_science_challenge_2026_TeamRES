"""Segment the 9x9x9 octet-lattice CT stack with bounded memory use.

The scan has substantial slice-wise baseline drift.  Each page is therefore
thresholded relative to its own median instead of using one global cutoff.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.morphology import remove_small_objects


def segment_page(page: np.ndarray, offset: float, min_size: int) -> tuple[np.ndarray, float]:
    threshold = float(np.median(page) + offset)
    mask = page >= threshold
    mask = remove_small_objects(mask, min_size=min_size, connectivity=2)
    return mask.astype(np.uint8), threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tif", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=float, default=9000.0)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--evaluation-slice", type=int, default=380)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / "segmented_mask.tif"
    image_path = args.output_dir / f"slice_{args.evaluation_slice}.png"
    report_path = args.output_dir / "segmentation_report.md"

    foreground = 0
    total = 0
    thresholds: list[float] = []
    evaluation_mask = None

    with tifffile.TiffFile(args.input_tif) as source:
        shape = source.series[0].shape
        dtype = str(source.series[0].dtype)
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D TIFF stack, got shape {shape}")
        if not 0 <= args.evaluation_slice < shape[0]:
            raise IndexError("Evaluation slice is outside the TIFF stack")

        with tifffile.TiffWriter(mask_path, bigtiff=True) as destination:
            for index, page in enumerate(source.pages):
                image = page.asarray()
                mask, threshold = segment_page(image, args.offset, args.min_size)
                destination.write(mask, photometric="minisblack", compression=None)
                foreground += int(mask.sum())
                total += int(mask.size)
                thresholds.append(threshold)
                if index == args.evaluation_slice:
                    evaluation_mask = mask.copy()

    if evaluation_mask is None:
        raise RuntimeError("Evaluation slice was not generated")
    plt.imsave(image_path, evaluation_mask, cmap="gray", vmin=0, vmax=1)

    background = total - foreground
    threshold_array = np.asarray(thresholds)
    report = f"""# CT Lattice Segmentation Report

## Inputs and outputs

- Input: `{args.input_tif}`
- Binary mask: `{mask_path}`
- Evaluation image: `{image_path}`
- Volume shape: `{shape}`
- Input dtype: `{dtype}`
- Output dtype: `uint8` (background 0, foreground 1)

## Method

The scan was segmented page-by-page to avoid loading the roughly 1 GB volume
into memory. To accommodate slice-wise baseline drift, the cutoff for each page
was its median intensity plus `{args.offset:g}`. Connected foreground components
smaller than `{args.min_size}` pixels were removed independently on each page
(8-connectivity). No resampling was performed.

## Statistics

- Foreground voxels: `{foreground}`
- Background voxels: `{background}`
- Total voxels: `{total}`
- Foreground fraction: `{foreground / total:.8f}`
- Background fraction: `{background / total:.8f}`
- Per-slice threshold minimum: `{threshold_array.min():.3f}`
- Per-slice threshold median: `{np.median(threshold_array):.3f}`
- Per-slice threshold maximum: `{threshold_array.max():.3f}`
- Evaluation slice: `{args.evaluation_slice}` along axis 0
"""
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
