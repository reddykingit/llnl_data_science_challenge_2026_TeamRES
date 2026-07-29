"""Compile group-8 Step 4 and Step 5 masks into analysis-ready NPY volumes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

import generate_group8_all_slice_pngs as group8


OUTPUT_DIR = group8.HIGHLIGHT_DIR / "compiled_step_masks"
STEP4_PATH = OUTPUT_DIR / "group_008_step4_missing_candidate_masks.npy"
STEP5_MASK_PATH = OUTPUT_DIR / "group_008_step5_tracked_masks.npy"
MANIFEST_PATH = OUTPUT_DIR / "group_008_step_masks.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyzer = group8.load_analyzer()
    report = json.loads(group8.REPORT_PATH.read_text(encoding="utf-8"))
    flagged = {
        int(index) for index in report["flagged_indices_zero_based"]
    }
    volume = tifffile.memmap(group8.VOLUME_PATH)
    cache = analyzer.MaskCache(
        volume,
        threshold=group8.THRESHOLD,
        foreground="high",
        opening_size=3,
    )
    step5_labels = np.load(
        group8.LABEL_PATH,
        allow_pickle=False,
        mmap_mode="r",
    )
    step4 = np.zeros(step5_labels.shape, dtype=bool)

    for local_index, dataset_index in enumerate(
        range(group8.START_INDEX, group8.END_INDEX + 1)
    ):
        _, _, missing, _ = analyzer.missing_components(
            index=dataset_index,
            period=group8.PERIOD,
            slice_count=len(volume),
            cache=cache,
            flagged=flagged,
            reference_cycles=2,
            phase_slack=1,
            expected_vote=0.5,
            max_registration=12,
            missing_tolerance=2,
            min_missing_area=20,
            min_foreground_fraction=0.001,
        )
        if missing is not None:
            step4[local_index] = missing
        print(f"Compiled Step 4 slice {local_index + 1:02d}/78")

    step5_mask = np.asarray(step5_labels) > 0
    np.save(STEP4_PATH, step4, allow_pickle=False)
    np.save(STEP5_MASK_PATH, step5_mask, allow_pickle=False)

    manifest = {
        "period_group": 8,
        "shape": list(step4.shape),
        "slice_axis": 0,
        "dtype": "bool",
        "local_slice_numbers_one_based": [1, 78],
        "dataset_indices_zero_based": [
            group8.START_INDEX,
            group8.END_INDEX,
        ],
        "dataset_slice_numbers_one_based": [
            group8.START_INDEX + 1,
            group8.END_INDEX + 1,
        ],
        "step4": {
            "file": STEP4_PATH.name,
            "meaning": (
                "True means registered same-phase material was expected but "
                "absent after spatial tolerance. Includes candidates that may "
                "not persist into Step 5."
            ),
            "true_voxel_count": int(step4.sum()),
        },
        "step5_binary": {
            "file": STEP5_MASK_PATH.name,
            "meaning": (
                "True means the missing component belongs to a retained "
                "parallel track."
            ),
            "true_voxel_count": int(step5_mask.sum()),
        },
        "step5_class_labels": {
            "file": str(group8.LABEL_PATH.resolve()),
            "dtype": str(step5_labels.dtype),
            "labels": {
                "0": "no retained evidence",
                "1": "persistent missing-node evidence",
                "2": "probable missing-strut evidence",
                "3": "confirmed missing-strut evidence",
                "4": "persistent missing-complex-pattern evidence",
            },
        },
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
