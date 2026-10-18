from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Rectangle
from skimage.filters import threshold_otsu


RUN_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = RUN_DIRECTORY.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.pattern_deformation_analysis import run_grid_pattern_analysis


INPUT_FILE = (
    REPOSITORY_ROOT
    / "data"
    / "missing_struts"
    / "tif_stacks"
    / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
)
GROUPS = (7, 8)
PERIOD_SLICES = 78


def load_volume_and_threshold() -> tuple[np.ndarray, float]:
    try:
        volume = np.asarray(tifffile.memmap(INPUT_FILE))
    except (OSError, ValueError):
        volume = np.asarray(tifffile.imread(INPUT_FILE))
    flat = volume.reshape(-1)
    step = max(1, math.ceil(flat.size / 1_000_000))
    sample = np.asarray(flat[::step])
    sample = sample[np.isfinite(sample)]
    return volume, float(threshold_otsu(sample))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expanded_square(
    bounds: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    scale: float = 1.85,
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = bounds
    center_y = 0.5 * (y0 + y1)
    center_x = 0.5 * (x0 + x1)
    side = max(y1 - y0, x1 - x0) * scale
    crop_y0 = max(0, int(round(center_y - side / 2)))
    crop_y1 = min(image_shape[0], int(round(center_y + side / 2)))
    crop_x0 = max(0, int(round(center_x - side / 2)))
    crop_x1 = min(image_shape[1], int(round(center_x + side / 2)))
    return crop_y0, crop_y1, crop_x0, crop_x1


def make_group_collage(group_number: int, report: dict[str, object]) -> Path:
    group_name = f"group_{group_number:03d}"
    group_directory = RUN_DIRECTORY / group_name
    rows = load_csv_rows(
        group_directory / f"{group_name}_grid_confirmed_patterns.csv"
    )
    viewer = np.load(
        group_directory
        / f"{group_name}_grid_step5_on_segmented_slice_viewer_rgb.npy",
        mmap_mode="r",
    )
    confirmed_mask = np.load(
        group_directory / f"{group_name}_grid_step5_confirmed.npy",
        mmap_mode="r",
    )
    scope = report["scope"]
    scope_start = int(scope["start_index_zero_based"])
    cell_bounds = {
        (int(cell["row"]), int(cell["column"])): (
            int(cell["y_start"]),
            int(cell["y_end_exclusive"]),
            int(cell["x_start"]),
            int(cell["x_end_exclusive"]),
        )
        for cell in report["grid"]["cell_bounds_original_pixels"]
    }
    tracks = {
        int(track["track_id"]): track
        for track in report["confirmed_patterns"]
    }
    rows_by_track: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_track.setdefault(int(row["track_id"]), []).append(row)

    representatives: list[dict[str, object]] = []
    for ordinal, track_id in enumerate(sorted(tracks), start=1):
        track = tracks[track_id]
        track_midpoint = 0.5 * (
            int(track["start_index_zero_based"])
            + int(track["end_index_zero_based"])
        )
        choices: list[tuple[int, float, dict[str, str]]] = []
        for row in rows_by_track.get(track_id, []):
            dataset_index = int(row["dataset_index_zero_based"])
            local_index = dataset_index - scope_start
            grid_cell = (
                int(row["grid_row_zero_based"]),
                int(row["grid_column_zero_based"]),
            )
            y0, y1, x0, x1 = cell_bounds[grid_cell]
            visible_area = int(
                confirmed_mask[local_index, y0:y1, x0:x1].sum()
            )
            midpoint_preference = -abs(dataset_index - track_midpoint)
            choices.append((visible_area, midpoint_preference, row))
        if choices:
            selected = max(choices, key=lambda item: (item[0], item[1]))[2]
            representatives.append(
                {"ordinal": ordinal, "track": track, "row": selected}
            )

    column_count = 4
    row_count = max(1, math.ceil(len(representatives) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15.5, 3.85 * row_count),
        squeeze=False,
    )
    figure.patch.set_facecolor("#10151d")
    plt.subplots_adjust(
        left=0.025,
        right=0.985,
        bottom=0.035,
        top=0.90,
        wspace=0.08,
        hspace=0.22,
    )
    for axis in axes.flat:
        axis.set_facecolor("#10151d")
        axis.axis("off")

    if not representatives:
        axes.flat[0].text(
            0.5,
            0.5,
            "No confirmed missing-strut tracks",
            ha="center",
            va="center",
            color="#f2f5f8",
            fontsize=15,
        )

    for axis, item in zip(axes.flat, representatives):
        row = item["row"]
        track = item["track"]
        dataset_index = int(row["dataset_index_zero_based"])
        local_index = dataset_index - scope_start
        grid_row = int(row["grid_row_zero_based"])
        grid_column = int(row["grid_column_zero_based"])
        y0, y1, x0, x1 = cell_bounds[(grid_row, grid_column)]
        crop_y0, crop_y1, crop_x0, crop_x1 = expanded_square(
            (y0, y1, x0, x1),
            viewer.shape[1:3],
        )
        crop = np.asarray(
            viewer[local_index, crop_y0:crop_y1, crop_x0:crop_x1]
        )
        axis.imshow(crop, interpolation="nearest")
        axis.add_patch(
            Rectangle(
                (x0 - crop_x0, y0 - crop_y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#ffd84d",
                linewidth=1.5,
                linestyle=(0, (4, 2)),
            )
        )
        axis.set_title(
            (
                f"Track {int(item['ordinal']):02d}/{len(representatives)}"
                f" | ID {int(track['track_id'])}"
                f" | {int(track['pattern_instance_count'])} instances\n"
                f"Dataset slice {dataset_index + 1}"
                f" | grid ({grid_row}, {grid_column})"
                f" | {int(track['detected_slice_count'])} detections"
            ),
            color="#f2f5f8",
            fontsize=9.2,
            pad=6,
            loc="left",
        )
        axis.axis("off")

    confirmed_count = int(report["confirmed_missing_strut_count"])
    figure.suptitle(
        (
            f"Group {group_number}: confirmed missing-strut tracks"
            f" | {len(representatives)} tracks"
            f" | {confirmed_count} pattern instances\n"
            "Red = confirmed expected-but-absent segment"
            " | White = segment outline"
            " | Gold = reported grid cell"
        ),
        color="#f5f7fa",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    output_path = (
        group_directory
        / f"{group_name}_confirmed_missing_struts_collage.png"
    )
    figure.savefig(
        output_path,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(figure)
    return output_path


def main() -> None:
    volume, threshold = load_volume_and_threshold()
    summaries: list[dict[str, object]] = []
    for group_number in GROUPS:
        group_directory = RUN_DIRECTORY / f"group_{group_number:03d}"
        report = run_grid_pattern_analysis(
            slices=volume,
            source_path=INPUT_FILE,
            output_directory=group_directory,
            period_slices=PERIOD_SLICES,
            group_number=group_number,
            threshold=threshold,
            foreground="high",
            opening_size=3,
            grid_rows=9,
            grid_columns=9,
            patch_size=48,
            expected_vote=0.5,
            min_reference_fraction=0.02,
            missing_fraction_threshold=0.6,
            max_present_fraction_of_expected=0.45,
            similarity_threshold=0.55,
            spatial_tolerance_pixels=1,
            min_track_slices=3,
            track_radius_cells=1.5,
            confidence_threshold=0.7,
            roi_presence_fraction=0.01,
            geometry_reference_cycles=2,
            comparison_mode="periodic_segments",
            periodic_vector_count=4,
            periodic_min_prediction_votes=3,
            periodic_min_bidirectional_pairs=2,
            min_missing_segment_area=6,
            min_missing_segment_fraction=0.001,
            min_missing_clearance_pixels=4.0,
            min_missing_clearance_fraction=0.04,
            periodic_border_margin_pixels=2,
            periodic_border_margin_fraction=0.10,
            periodic_max_track_gap_fraction=0.35,
            tilt_correction=True,
            tilt_sample_count=16,
            maximum_tilt_shift_per_period_fraction=0.5,
            forward_track_prediction=True,
            forward_track_history=6,
            forward_track_gap_radius_growth=0.08,
            forward_track_maximum_radius_factor=2.0,
        )
        report["inspection_mode"] = "grid"
        report["period_estimation"] = {
            "length_slices": PERIOD_SLICES,
            "confidence": None,
            "source": "established dataset period",
        }
        report["segmentation"]["threshold_source"] = "otsu"
        report_path = Path(report["outputs"]["findings_report_json"])
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        collage_path = make_group_collage(group_number, report)
        summary = {
            "group_number": group_number,
            "dataset_slice_start_one_based": (
                int(report["scope"]["start_index_zero_based"]) + 1
            ),
            "dataset_slice_end_one_based_inclusive": int(
                report["scope"]["end_index_zero_based_exclusive"]
            ),
            "confirmed_missing_strut_count": int(
                report["confirmed_missing_strut_count"]
            ),
            "confirmed_track_count": len(report["confirmed_patterns"]),
            "report": str(report_path),
            "collage": str(collage_path),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    run_summary = {
        "source_file": str(INPUT_FILE),
        "analyzed_groups": list(GROUPS),
        "period_slices": PERIOD_SLICES,
        "otsu_threshold": threshold,
        "groups": summaries,
    }
    (RUN_DIRECTORY / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
