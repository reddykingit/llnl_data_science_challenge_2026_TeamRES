"""Overlay Step 5 labels on the normal segmented group-8 slice volume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

import generate_group8_all_slice_pngs as group8


OUTPUT_PATH = (
    group8.HIGHLIGHT_DIR
    / "group_008_step5_on_segmented_slice_viewer_rgb.npy"
)
MANIFEST_PATH = OUTPUT_PATH.with_suffix(".json")
SEGMENTED_GRAY = np.uint8(220)
LABEL_COLORS = {
    1: np.array([255, 184, 0], dtype=np.uint8),
    2: np.array([255, 64, 0], dtype=np.uint8),
    3: np.array([255, 13, 13], dtype=np.uint8),
    4: np.array([204, 26, 255], dtype=np.uint8),
}


def main() -> None:
    analyzer = group8.load_analyzer()
    volume = tifffile.memmap(group8.VOLUME_PATH)
    cache = analyzer.MaskCache(
        volume,
        threshold=group8.THRESHOLD,
        foreground="high",
        opening_size=3,
    )
    labels = np.load(
        group8.LABEL_PATH,
        allow_pickle=False,
        mmap_mode="r",
    )
    output = np.lib.format.open_memmap(
        OUTPUT_PATH,
        mode="w+",
        dtype=np.uint8,
        shape=labels.shape + (3,),
    )

    slices_with_tracks = 0
    for local_index, dataset_index in enumerate(
        range(group8.START_INDEX, group8.END_INDEX + 1)
    ):
        segmented = cache.get(dataset_index)
        rgb = np.zeros(segmented.shape + (3,), dtype=np.uint8)
        rgb[segmented] = SEGMENTED_GRAY

        slice_labels = np.asarray(labels[local_index])
        tracked = slice_labels > 0
        if tracked.any():
            slices_with_tracks += 1
        for label_value, color in LABEL_COLORS.items():
            rgb[slice_labels == label_value] = color
        outline = ndimage.binary_dilation(tracked, iterations=2) & ~tracked
        rgb[outline] = np.array([255, 255, 255], dtype=np.uint8)
        output[local_index] = rgb
        print(f"Built segmented overlay {local_index + 1:02d}/78")

    output.flush()
    manifest = {
        "period_group": 8,
        "output_file": str(OUTPUT_PATH.resolve()),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "channel_order": "RGB",
        "slice_axis": 0,
        "dataset_indices_zero_based": [
            group8.START_INDEX,
            group8.END_INDEX,
        ],
        "dataset_slice_numbers_one_based": [
            group8.START_INDEX + 1,
            group8.END_INDEX + 1,
        ],
        "base": {
            "meaning": "normal segmented lattice material",
            "background_rgb": [0, 0, 0],
            "material_rgb": [220, 220, 220],
            "segmentation_threshold": group8.THRESHOLD,
            "morphological_opening_size": 3,
        },
        "step5_overlay_colors": {
            "1_persistent_missing_node": LABEL_COLORS[1].tolist(),
            "2_probable_missing_strut": LABEL_COLORS[2].tolist(),
            "3_confirmed_missing_strut": LABEL_COLORS[3].tolist(),
            "4_persistent_missing_complex_pattern": (
                LABEL_COLORS[4].tolist()
            ),
        },
        "outline_rgb": [255, 255, 255],
        "slices_with_retained_tracks": slices_with_tracks,
        "viewer": str((group8.ROOT / "npy_viewer.py").resolve()),
        "index_mapping": (
            "array[0] is dataset index 546 / dataset slice 547; "
            "array[77] is dataset index 623 / dataset slice 624"
        ),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
