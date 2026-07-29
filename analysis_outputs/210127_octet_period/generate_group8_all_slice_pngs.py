"""Render one readable five-panel PNG for every slice in period group 8."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = ROOT / "src/pattern_deformation_analysis.py"
REPORT_PATH = (
    ROOT
    / "src/missing_struts_0point5_axis0_pattern_recognition"
    / "symmetry_report.json"
)
ANALYSIS_PATH = (
    ROOT
    / "src/missing_struts_0point5_axis0_pattern_recognition"
    / "pattern_group_analysis/pattern_analysis.json"
)
VOLUME_PATH = (
    ROOT
    / "data/missing_struts/tif_stacks"
    / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
)
HIGHLIGHT_DIR = (
    Path(__file__).resolve().parent / "period_group_008_highlights"
)
LABEL_PATH = HIGHLIGHT_DIR / "group_008_missing_pattern_labels.npy"
OUTPUT_DIR = HIGHLIGHT_DIR / "all_slice_pipeline_pngs"

PERIOD = 78
START_INDEX = 546
END_INDEX = 623
THRESHOLD = 38257.0

LABEL_COLORS = {
    1: np.array([1.0, 0.72, 0.0]),
    2: np.array([1.0, 0.25, 0.0]),
    3: np.array([1.0, 0.05, 0.05]),
    4: np.array([0.8, 0.1, 1.0]),
}


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "analyze_flagged_groups_all_slices", ANALYZER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import analyzer from {ANALYZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_raw(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1.0, 99.0))
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip(
        (image.astype(np.float32) - low) / (high - low),
        0.0,
        1.0,
    )


def overlay_missing(
    gray: np.ndarray,
    missing: np.ndarray,
    tracked: np.ndarray,
) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[missing] = (
        0.30 * rgb[missing] + 0.70 * np.array([1.0, 0.55, 0.0])
    )
    rgb[tracked] = (
        0.10 * rgb[tracked] + 0.90 * np.array([1.0, 0.0, 0.0])
    )
    return rgb


def overlay_tracks(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for label_value, color in LABEL_COLORS.items():
        mask = labels == label_value
        rgb[mask] = 0.18 * rgb[mask] + 0.82 * color
        outline = ndimage.binary_dilation(mask, iterations=2) & ~mask
        rgb[outline] = np.array([1.0, 1.0, 1.0])
    return rgb


def load_track_ids_by_slice() -> dict[int, list[str]]:
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    result: dict[int, set[str]] = {}
    for group in analysis["groups"]:
        qualifying_ids = {
            int(decision["track_id"])
            for decision in group["conclusion"].get(
                "qualifying_tracks", []
            )
        }
        for track in group["tracks"]:
            track_id = int(track["track_id"])
            if track_id not in qualifying_ids:
                continue
            display_id = f"g{int(group['group_id'])}:t{track_id}"
            for observation in track["observations"]:
                index = int(observation["slice_index_zero_based"])
                if START_INDEX <= index <= END_INDEX:
                    result.setdefault(index, set()).add(display_id)
    return {
        index: sorted(track_ids)
        for index, track_ids in result.items()
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyzer = load_analyzer()
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    flagged = {
        int(index) for index in report["flagged_indices_zero_based"]
    }
    volume = tifffile.memmap(VOLUME_PATH)
    cache = analyzer.MaskCache(
        volume,
        threshold=THRESHOLD,
        foreground="high",
        opening_size=3,
    )
    label_volume = np.load(LABEL_PATH, allow_pickle=False, mmap_mode="r")
    track_ids_by_slice = load_track_ids_by_slice()
    manifest_rows = []

    for local_index, dataset_index in enumerate(
        range(START_INDEX, END_INDEX + 1)
    ):
        raw = np.asarray(volume[dataset_index])
        current = cache.get(dataset_index)
        components, expected, missing, references = analyzer.missing_components(
            index=dataset_index,
            period=PERIOD,
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
        if expected is None:
            expected = np.zeros(current.shape, dtype=bool)
        if missing is None:
            missing = np.zeros(current.shape, dtype=bool)

        labels = np.asarray(label_volume[local_index])
        tracked = labels > 0
        gray = normalize_raw(raw)
        missing_rgb = overlay_missing(gray, missing, tracked)
        tracked_rgb = overlay_tracks(gray, labels)
        track_ids = track_ids_by_slice.get(dataset_index, [])
        local_slice = local_index + 1
        dataset_slice = dataset_index + 1

        panels = [
            (gray, "1. Raw slice", "gray"),
            (current, "2. Segmented material", "gray"),
            (expected, "3. Registered expectation", "gray"),
            (missing_rgb, "4. Missing difference", None),
            (tracked_rgb, "5. Parallel tracked highlight", None),
        ]
        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4.5),
            constrained_layout=True,
        )
        for axis, (image, title, cmap) in zip(axes, panels):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        retained_text = (
            f"{len(track_ids)} retained track(s): "
            + ", ".join(str(track_id) for track_id in track_ids)
            if track_ids
            else "no persistent defect retained"
        )
        fig.suptitle(
            f"Period group 8 Â· local slice {local_slice:02d}/78 Â· "
            f"dataset slice {dataset_slice} Â· {retained_text}",
            fontsize=13,
        )
        filename = (
            f"group_008_local_{local_slice:02d}_"
            f"dataset_slice_{dataset_slice:04d}_pipeline.png"
        )
        fig.savefig(OUTPUT_DIR / filename, dpi=150, facecolor="white")
        plt.close(fig)

        manifest_rows.append(
            {
                "local_slice_one_based": local_slice,
                "dataset_index_zero_based": dataset_index,
                "dataset_slice_one_based": dataset_slice,
                "png_file": filename,
                "candidate_component_count": len(components),
                "retained_track_count": len(track_ids),
                "retained_track_ids": ";".join(track_ids),
                "has_retained_defect": bool(track_ids),
                "reference_indices_zero_based": ";".join(
                    str(index) for index in references
                ),
            }
        )
        print(f"Rendered {local_slice:02d}/78: {filename}")

    manifest_path = OUTPUT_DIR / "slice_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Rendered {len(manifest_rows)} PNGs to {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
