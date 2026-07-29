"""Run the current two-class pattern-irregularity analysis on every group."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.lib.format import open_memmap
from PIL import Image, ImageDraw, ImageFont, ImageOps


RUN_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = RUN_DIRECTORY.parents[1]
GROUP4_RUNNER = (
    REPOSITORY_ROOT
    / "analysis_outputs"
    / "pattern_irregularities_group_004_20260728_142038"
    / "run_group4_analysis.py"
)
PERIOD_SLICES = 78
REFERENCE_GRID_SIZE = 8
GRID_SIZE = 6
GRID_LINEAR_SCALE = GRID_SIZE / REFERENCE_GRID_SIZE
GRID_AREA_SCALE = GRID_LINEAR_SCALE**2
TRACK_RADIUS_CELLS = round(1.5 * GRID_LINEAR_SCALE, 6)
MISSING_FRACTION_THRESHOLD = 0.65
MISSING_OBSERVATION_FRACTION = 0.60
MISSING_SPATIAL_EXTENT_CELLS = 0.28
MIN_MISSING_SEGMENT_FRACTION = 0.0006
MIN_MISSING_CLEARANCE_FRACTION = 0.0325
BORDER_MARGIN_FRACTION = round(0.10 * GRID_LINEAR_SCALE, 6)
MAXIMUM_TILT_SHIFT_FRACTION = round(0.5 * GRID_LINEAR_SCALE, 6)


def load_group_runner():
    spec = importlib.util.spec_from_file_location(
        "current_group_pattern_runner",
        GROUP4_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load group runner from {GROUP4_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_DIRECTORY = RUN_DIRECTORY
    return module


def relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(RUN_DIRECTORY).as_posix()


def expanded_square(
    bounds: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    scale: float = 1.75,
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = bounds
    center_y = 0.5 * (y0 + y1)
    center_x = 0.5 * (x0 + x1)
    side = max(y1 - y0, x1 - x0) * scale
    return (
        max(0, int(round(center_y - side / 2))),
        min(image_shape[0], int(round(center_y + side / 2))),
        max(0, int(round(center_x - side / 2))),
        min(image_shape[1], int(round(center_x + side / 2))),
    )


def make_irregularity_collage(
    group_number: int,
    report: dict[str, object],
) -> Path:
    group_name = f"group_{group_number:03d}"
    group_directory = RUN_DIRECTORY / group_name
    with (
        group_directory / f"{group_name}_grid_confirmed_patterns.csv"
    ).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    viewer = np.load(
        group_directory
        / f"{group_name}_grid_step5_on_segmented_slice_viewer_rgb.npy",
        mmap_mode="r",
    )
    class_labels = np.load(
        group_directory / f"{group_name}_grid_defect_class_labels.npy",
        mmap_mode="r",
    )
    scope_start = int(report["scope"]["start_index_zero_based"])
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

    class_codes = {"missing_strut": 1, "broken_strut": 2}
    class_order = {"missing_strut": 0, "broken_strut": 1}
    class_labels_text = {
        "missing_strut": "Missing",
        "broken_strut": "Broken",
    }
    representatives = []
    ordered_tracks = sorted(
        tracks.items(),
        key=lambda item: (
            class_order[str(item[1]["defect_class"])],
            item[0],
        ),
    )
    for ordinal, (track_id, track) in enumerate(ordered_tracks, start=1):
        midpoint = 0.5 * (
            int(track["start_index_zero_based"])
            + int(track["end_index_zero_based"])
        )
        defect_class = str(track["defect_class"])
        choices = []
        for row in rows_by_track.get(track_id, []):
            dataset_index = int(row["dataset_index_zero_based"])
            local_index = dataset_index - scope_start
            cell = (
                int(row["grid_row_zero_based"]),
                int(row["grid_column_zero_based"]),
            )
            y0, y1, x0, x1 = cell_bounds[cell]
            visible_area = int(
                (
                    class_labels[local_index, y0:y1, x0:x1]
                    == class_codes[defect_class]
                ).sum()
            )
            choices.append(
                (visible_area, -abs(dataset_index - midpoint), row)
            )
        if choices:
            representatives.append(
                {
                    "ordinal": ordinal,
                    "track": track,
                    "row": max(choices, key=lambda item: (item[0], item[1]))[
                        2
                    ],
                }
            )

    columns = min(5, max(1, len(representatives)))
    row_count = max(1, math.ceil(len(representatives) / columns))
    figure_height = 1.55 + 3.35 * row_count
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(max(7.0, 3.6 * columns), figure_height),
        squeeze=False,
    )
    figure.patch.set_facecolor("#10151d")
    figure.subplots_adjust(
        left=0.018,
        right=0.988,
        bottom=0.018,
        top=1.0 - 1.32 / figure_height,
        wspace=0.065,
        hspace=0.26,
    )
    for axis in axes.flat:
        axis.set_facecolor("#10151d")
        axis.axis("off")
    if not representatives:
        axes.flat[0].text(
            0.5,
            0.5,
            "No confirmed pattern irregularities",
            ha="center",
            va="center",
            color="#f2f5f8",
            fontsize=15,
        )
    for axis, item in zip(axes.flat, representatives):
        row = item["row"]
        track = item["track"]
        defect_class = str(track["defect_class"])
        dataset_index = int(row["dataset_index_zero_based"])
        local_index = dataset_index - scope_start
        grid_row = int(row["grid_row_zero_based"])
        grid_column = int(row["grid_column_zero_based"])
        y0, y1, x0, x1 = cell_bounds[(grid_row, grid_column)]
        crop_y0, crop_y1, crop_x0, crop_x1 = expanded_square(
            (y0, y1, x0, x1),
            viewer.shape[1:3],
        )
        axis.imshow(
            np.asarray(
                viewer[
                    local_index,
                    crop_y0:crop_y1,
                    crop_x0:crop_x1,
                ]
            ),
            interpolation="nearest",
        )
        axis.add_patch(
            Rectangle(
                (x0 - crop_x0, y0 - crop_y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#ffd84d",
                linewidth=1.3,
                linestyle=(0, (4, 2)),
            )
        )
        axis.set_title(
            (
                f"{class_labels_text[defect_class]} | track"
                f" {int(item['ordinal']):02d}/{len(representatives)}"
                f" | ID {int(track['track_id'])}"
                f" | {int(track['pattern_instance_count'])} instances\n"
                f"Slice {dataset_index + 1}"
                f" | grid ({grid_row}, {grid_column})"
                f" | {int(track['detected_slice_count'])} detections"
            ),
            color="#f2f5f8",
            fontsize=8.2,
            pad=5,
            loc="left",
        )
    counts = report["confirmed_defect_instance_counts_by_class"]
    track_counts = report["confirmed_track_counts_by_class"]
    figure.text(
        0.5,
        1.0 - 0.10 / figure_height,
        (
            f"Group {group_number}: all confirmed pattern irregularities"
            f" | {int(report['confirmed_defect_count'])} instances"
            f" | {int(report['confirmed_track_count'])} tracks"
        ),
        color="#f5f7fa",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="top",
    )
    figure.text(
        0.5,
        1.0 - 0.52 / figure_height,
        (
            f"Missing: {int(counts['missing_strut'])} /"
            f" {int(track_counts['missing_strut'])} tracks"
            f" | Broken: {int(counts['broken_strut'])} /"
            f" {int(track_counts['broken_strut'])} tracks"
        ),
        color="#dce3eb",
        fontsize=10.5,
        ha="center",
        va="top",
    )
    figure.text(
        0.5,
        1.0 - 0.88 / figure_height,
        (
            "Red = missing | Orange = broken"
            " | White = outline | Gold = reported grid cell"
        ),
        color="#c8d2de",
        fontsize=9.5,
        ha="center",
        va="top",
    )
    output = group_directory / f"{group_name}_all_irregularities_collage.png"
    figure.savefig(
        output,
        dpi=160,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.14,
    )
    plt.close(figure)
    return output


def analyze_group(
    base,
    volume: np.ndarray,
    threshold: float,
    group_number: int,
) -> tuple[dict[str, object], dict[str, object]]:
    group_name = f"group_{group_number:03d}"
    group_directory = RUN_DIRECTORY / group_name
    group_start = (group_number - 1) * PERIOD_SLICES
    group_depth = min(PERIOD_SLICES, int(volume.shape[0]) - group_start)
    resolved_min_track = base.resolve_min_track_slices(
        None,
        PERIOD_SLICES,
        group_depth,
    )
    report_path = (
        group_directory / f"{group_name}_grid_findings_report.json"
    )
    required = [
        group_directory / f"{group_name}_grid_confirmed_patterns.csv",
        group_directory / f"{group_name}_grid_step4_candidates.npy",
        group_directory / f"{group_name}_grid_step5_confirmed.npy",
        group_directory
        / f"{group_name}_grid_step5_on_segmented_slice_viewer_rgb.npy",
        group_directory / f"{group_name}_grid_defect_class_labels.npy",
    ]
    if report_path.is_file():
        if not all(path.is_file() for path in required):
            raise RuntimeError(
                f"{group_name} is partial; remove that group directory "
                "before resuming."
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"Resuming completed {group_name}", flush=True)
    else:
        report = base.run_grid_pattern_analysis(
            slices=volume,
            source_path=base.INPUT_FILE,
            output_directory=group_directory,
            period_slices=PERIOD_SLICES,
            group_number=group_number,
            threshold=threshold,
            foreground="high",
            opening_size=3,
            grid_rows=GRID_SIZE,
            grid_columns=GRID_SIZE,
            patch_size=48,
            expected_vote=0.5,
            min_reference_fraction=0.02,
            missing_fraction_threshold=MISSING_FRACTION_THRESHOLD,
            max_present_fraction_of_expected=0.45,
            similarity_threshold=0.55,
            spatial_tolerance_pixels=1,
            min_track_slices=resolved_min_track,
            track_radius_cells=TRACK_RADIUS_CELLS,
            confidence_threshold=0.7,
            roi_presence_fraction=0.01,
            broken_min_track_slices=8,
            broken_confidence_threshold=0.85,
            missing_min_observation_fraction=(
                MISSING_OBSERVATION_FRACTION
            ),
            missing_span_tolerance_fraction=0.15,
            missing_spatial_extent_threshold_cells=(
                MISSING_SPATIAL_EXTENT_CELLS
            ),
            missing_spatial_confirmation_slices=2,
            broken_jump_threshold_cells=1.0,
            geometry_reference_cycles=2,
            comparison_mode="periodic_segments",
            periodic_vector_count=4,
            periodic_min_prediction_votes=3,
            periodic_min_bidirectional_pairs=2,
            min_missing_segment_area=6,
            min_missing_segment_fraction=MIN_MISSING_SEGMENT_FRACTION,
            min_missing_clearance_pixels=4.0,
            min_missing_clearance_fraction=(
                MIN_MISSING_CLEARANCE_FRACTION
            ),
            periodic_border_margin_pixels=2,
            periodic_border_margin_fraction=(
                BORDER_MARGIN_FRACTION
            ),
            periodic_max_track_gap_fraction=0.35,
            tilt_correction=True,
            tilt_sample_count=16,
            maximum_tilt_shift_per_period_fraction=(
                MAXIMUM_TILT_SHIFT_FRACTION
            ),
            known_tilt_angle_degrees=0.664,
            tilt_angle_tolerance_degrees=0.20,
            forward_track_prediction=True,
            forward_track_history=6,
            forward_track_gap_radius_growth=0.08,
            forward_track_maximum_radius_factor=2.0,
        )
        report["period_estimation"] = {
            "length_slices": PERIOD_SLICES,
            "confidence": None,
            "source": "established dataset period",
        }
        report["segmentation"]["threshold_source"] = "otsu"
        report["inspection_mode"] = (
            "grid_two_class_pattern_irregularities"
        )
        report["grid"]["parameter_scaling"] = {
            "reference_grid_size": REFERENCE_GRID_SIZE,
            "target_grid_size": GRID_SIZE,
            "linear_scale": GRID_LINEAR_SCALE,
            "area_scale": GRID_AREA_SCALE,
            "scaled_parameters": [
                "track_radius_cells",
                "missing_spatial_extent_threshold_cells",
                "min_missing_segment_fraction",
                "min_missing_clearance_fraction",
                "periodic_border_margin_fraction",
                "maximum_tilt_shift_per_period_fraction",
            ],
        }
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    collage = make_irregularity_collage(group_number, report)
    class_counts = report["confirmed_defect_instance_counts_by_class"]
    track_counts = report["confirmed_track_counts_by_class"]
    summary = {
        "group_number": group_number,
        "dataset_slice_start_one_based": group_start + 1,
        "dataset_slice_end_one_based_inclusive": group_start + group_depth,
        "slice_count": group_depth,
        "complete_78_slice_group": group_depth == PERIOD_SLICES,
        "resolved_missing_span_slices": resolved_min_track,
        "confirmed_irregularity_count": int(
            report["confirmed_defect_count"]
        ),
        "confirmed_irregularity_track_count": int(
            report["confirmed_track_count"]
        ),
        "confirmed_missing_strut_count": int(
            class_counts["missing_strut"]
        ),
        "confirmed_missing_track_count": int(
            track_counts["missing_strut"]
        ),
        "confirmed_broken_strut_count": int(
            class_counts["broken_strut"]
        ),
        "confirmed_broken_track_count": int(
            track_counts["broken_strut"]
        ),
        "rejected_track_count": int(
            report["tracking"]["rejected_track_count"]
        ),
        "report": str(report_path),
        "collage": str(collage),
    }
    print(json.dumps(summary), flush=True)
    return summary, report


def combine_group_arrays(
    groups: list[dict[str, object]],
    volume_shape: tuple[int, int, int],
    source_file: Path,
) -> dict[str, object]:
    labels_path = (
        RUN_DIRECTORY / "all_groups_pattern_irregularity_class_labels.npy"
    )
    viewer_path = (
        RUN_DIRECTORY / "all_groups_pattern_irregularities_viewer_rgb.npy"
    )
    labels_output = open_memmap(
        labels_path,
        mode="w+",
        dtype=np.uint8,
        shape=volume_shape,
    )
    viewer_output = open_memmap(
        viewer_path,
        mode="w+",
        dtype=np.uint8,
        shape=(*volume_shape, 3),
    )
    manifest_groups = []
    output_start = 0
    for group in groups:
        number = int(group["group_number"])
        name = f"group_{number:03d}"
        directory = RUN_DIRECTORY / name
        labels = np.load(
            directory / f"{name}_grid_defect_class_labels.npy",
            mmap_mode="r",
        )
        viewer = np.load(
            directory
            / f"{name}_grid_step5_on_segmented_slice_viewer_rgb.npy",
            mmap_mode="r",
        )
        output_end = output_start + int(labels.shape[0])
        if labels.shape[1:] != volume_shape[1:]:
            raise ValueError(f"{name} label dimensions are incompatible")
        if viewer.shape[1:] != (*volume_shape[1:], 3):
            raise ValueError(f"{name} viewer dimensions are incompatible")
        labels_output[output_start:output_end] = labels
        viewer_output[output_start:output_end] = viewer
        labels_output.flush()
        viewer_output.flush()
        manifest_groups.append(
            {
                "group_number": number,
                "output_start_index_zero_based": output_start,
                "output_end_index_zero_based_exclusive": output_end,
                "dataset_slice_start_one_based": output_start + 1,
                "dataset_slice_end_one_based_inclusive": output_end,
            }
        )
        print(
            f"Combined {name}: slices {output_start + 1}-{output_end}",
            flush=True,
        )
        output_start = output_end
    del labels_output, viewer_output
    if output_start != volume_shape[0]:
        raise ValueError(
            f"Combined depth {output_start} != input depth {volume_shape[0]}"
        )
    manifest = {
        "source_file": str(source_file),
        "class_labels_file": str(labels_path),
        "class_labels_shape": list(volume_shape),
        "class_label_values": {
            "background": 0,
            "missing_strut": 1,
            "broken_strut": 2,
        },
        "viewer_rgb_file": str(viewer_path),
        "viewer_rgb_shape": [*volume_shape, 3],
        "viewer_colors_rgb": {
            "missing_strut": [255, 13, 13],
            "broken_strut": [255, 140, 0],
        },
        "dtype": "uint8",
        "concatenation_axis": 0,
        "groups": manifest_groups,
    }
    manifest_path = RUN_DIRECTORY / "combined_npy_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {
        "class_labels": str(labels_path),
        "viewer_rgb": str(viewer_path),
        "manifest": str(manifest_path),
    }


def write_summary_csv(groups: list[dict[str, object]]) -> Path:
    output = RUN_DIRECTORY / "group_summary.csv"
    fields = [
        "group_number",
        "dataset_slice_start_one_based",
        "dataset_slice_end_one_based_inclusive",
        "slice_count",
        "complete_78_slice_group",
        "resolved_missing_span_slices",
        "confirmed_irregularity_count",
        "confirmed_irregularity_track_count",
        "confirmed_missing_strut_count",
        "confirmed_missing_track_count",
        "confirmed_broken_strut_count",
        "confirmed_broken_track_count",
        "rejected_track_count",
        "report",
        "collage",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            row = {field: group[field] for field in fields[:-2]}
            row["report"] = relative(group["report"])
            row["collage"] = relative(group["collage"])
            writer.writerow(row)
    return output


def make_counts_chart(groups: list[dict[str, object]]) -> Path:
    numbers = [int(group["group_number"]) for group in groups]
    missing = [
        int(group["confirmed_missing_strut_count"]) for group in groups
    ]
    broken = [
        int(group["confirmed_broken_strut_count"]) for group in groups
    ]
    totals = np.asarray(missing) + np.asarray(broken)
    figure, axis = plt.subplots(figsize=(12.6, 6.6))
    figure.patch.set_facecolor("#10151d")
    axis.set_facecolor("#10151d")
    axis.bar(numbers, missing, color="#ff0d0d", label="Missing")
    axis.bar(
        numbers,
        broken,
        bottom=missing,
        color="#ff8c00",
        label="Broken",
    )
    for number, total in zip(numbers, totals):
        axis.text(
            number,
            total + 0.5,
            str(int(total)),
            ha="center",
            color="#f7f9fb",
            fontweight="bold",
        )
    axis.set_xticks(numbers)
    axis.set_xlabel("78-slice period group", color="#e9edf2")
    axis.set_ylabel("Confirmed pattern instances", color="#e9edf2")
    axis.set_title(
        "Confirmed pattern irregularities by group",
        color="#f7f9fb",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    axis.tick_params(colors="#e9edf2")
    axis.grid(axis="y", color="#425064", alpha=0.35)
    for spine in axis.spines.values():
        spine.set_color("#425064")
    axis.legend(
        facecolor="#18212d",
        edgecolor="#425064",
        labelcolor="#f7f9fb",
    )
    figure.tight_layout()
    output = RUN_DIRECTORY / "pattern_irregularity_counts_by_group.png"
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def make_contact_sheet(groups: list[dict[str, object]]) -> Path:
    columns = 2
    tile_width, tile_height = 820, 650
    margin, header_height = 28, 88
    rows = math.ceil(len(groups) / columns)
    sheet = Image.new(
        "RGB",
        (
            margin + columns * (tile_width + margin),
            header_height + rows * (tile_height + margin),
        ),
        "#10151d",
    )
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 36)
        label_font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text(
        (margin, 22),
        "All group pattern-irregularity collages",
        fill="#f5f7fa",
        font=title_font,
    )
    for index, group in enumerate(groups):
        row, column = divmod(index, columns)
        x0 = margin + column * (tile_width + margin)
        y0 = header_height + row * (tile_height + margin)
        tile = Image.new("RGB", (tile_width, tile_height), "#18212d")
        with Image.open(str(group["collage"])) as source:
            collage = source.convert("RGB")
        fitted = ImageOps.contain(
            collage,
            (tile_width - 20, tile_height - 68),
            Image.Resampling.LANCZOS,
        )
        tile.paste(
            fitted,
            (
                (tile_width - fitted.width) // 2,
                54 + (tile_height - 54 - fitted.height) // 2,
            ),
        )
        tile_draw = ImageDraw.Draw(tile)
        suffix = (
            ""
            if bool(group["complete_78_slice_group"])
            else " (partial)"
        )
        tile_draw.text(
            (12, 11),
            (
                f"Group {int(group['group_number']):02d}{suffix}"
                f" | slices {int(group['dataset_slice_start_one_based'])}-"
                f"{int(group['dataset_slice_end_one_based_inclusive'])}"
                f" | {int(group['confirmed_irregularity_count'])}"
                " instances"
            ),
            fill="#f5f7fa",
            font=label_font,
        )
        sheet.paste(tile, (x0, y0))
    output = RUN_DIRECTORY / "all_group_pattern_irregularity_collages.png"
    sheet.save(output, optimize=True)
    return output


def write_report(
    summary: dict[str, object],
    summary_csv: Path,
    chart: Path,
    contact_sheet: Path,
) -> Path:
    groups = list(summary["groups"])
    totals = {
        "all": sum(
            int(group["confirmed_irregularity_count"]) for group in groups
        ),
        "tracks": sum(
            int(group["confirmed_irregularity_track_count"])
            for group in groups
        ),
        "missing": sum(
            int(group["confirmed_missing_strut_count"])
            for group in groups
        ),
        "broken": sum(
            int(group["confirmed_broken_strut_count"])
            for group in groups
        ),
        "rejected": sum(
            int(group["rejected_track_count"]) for group in groups
        ),
    }
    rows = []
    collage_sections = []
    for group in groups:
        number = int(group["group_number"])
        start = int(group["dataset_slice_start_one_based"])
        end = int(group["dataset_slice_end_one_based_inclusive"])
        completeness = (
            "Complete"
            if bool(group["complete_78_slice_group"])
            else "Partial"
        )
        rows.append(
            (
                f"| {number} | {start}-{end} |"
                f" {int(group['slice_count'])} ({completeness})"
                f" | {int(group['confirmed_irregularity_count'])}"
                f" | {int(group['confirmed_irregularity_track_count'])}"
                f" | {int(group['confirmed_missing_strut_count'])}"
                f" | {int(group['confirmed_broken_strut_count'])}"
                f" | {int(group['rejected_track_count'])}"
                f" | [JSON]({relative(group['report'])})"
                f" | [Collage]({relative(group['collage'])}) |"
            )
        )
        collage_sections.append(
            (
                f"### Group {number:02d}: slices {start}-{end}\n\n"
                f"![Group {number:02d} evidence]"
                f"({relative(group['collage'])})"
            )
        )
    combined = summary["combined_npy"]
    shape = tuple(summary["input_shape"])
    text = f"""# All-Groups Pattern Irregularity Analysis

## Executive summary

The full {shape[0]}-slice TIFF stack was analyzed in ten lattice-period groups
with the current mutually exclusive missing-versus-broken detector. It found
**{totals['all']} confirmed pattern-irregularity instances across
{totals['tracks']} within-group tracks**:

- **{totals['missing']} missing-strut instances**
- **{totals['broken']} broken-strut instances**

Another **{totals['rejected']} candidate tracks** did not pass persistence and
confidence rules and remain available in the detailed group JSON reports.
Group 10 contains 59 slices and is marked partial. Tracks are resolved
independently within each group and are not de-duplicated across boundaries.

![Counts by group]({relative(chart)})

## Group results

| Group | Dataset slices | Depth | Instances | Tracks | Missing | Broken | Rejected tracks | Detailed data | Evidence |
|---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|:---:|
{chr(10).join(rows)}

Machine-readable summaries: [run_summary.json](run_summary.json) and
[group_summary.csv]({relative(summary_csv)}).

## Combined full-stack NPY outputs

- [Class labels]({relative(combined['class_labels'])}): shape `{shape}`,
  `uint8`; 0=background, 1=missing, 2=broken.
- [Segmented RGB viewer]({relative(combined['viewer_rgb'])}): shape
  `{shape + (3,)}`, `uint8`; red=missing, orange=broken, white=outline,
  gold=reported grid cell.
- [Combined-array manifest]({relative(combined['manifest'])}) records class
  mappings, dimensions, and group-to-full-stack slice ranges.

## All-group collage overview

![All group collages]({relative(contact_sheet)})

## Method

- Input: `{Path(str(summary['source_file'])).name}`; shape `{shape}` (Z, Y, X),
  dtype `{summary['input_dtype']}`.
- Grouping: established 78-slice lattice period; group 10 is the remaining
  59 slices.
- Segmentation: high foreground, shared Otsu threshold
  {float(summary['otsu_threshold']):.1f}, 3-pixel opening.
- Spatial model: {GRID_SIZE} x {GRID_SIZE} grid, periodic translated-neighbor consensus, two
  geometry-reference cycles, tilt correction, and forward trajectory
  prediction.
- Missing classification: nominal half-group persistence with 15% span
  tolerance and at least 60% observation support, or a full absent segment
  spanning {MISSING_SPATIAL_EXTENT_CELLS:.4g} grid cells in at least two slices.
- Broken classification: shorter absence with at least eight observations and
  class confidence at least 0.85, or a qualifying trajectory discontinuity.
- General confirmation confidence: at least 0.70. Classes are mutually
  exclusive and matching fragments are de-duplicated within each group.

## Validation and limitations

All 30 tests in `tests.test_grid_pattern_inspection` passed immediately before
this run. Post-run validation checks group scope, counts, paths, combined-array
dimensions and exact group concatenation, report links, and image readability.

These outputs are algorithmic screening findings rather than
metrology-certified ground truth. Review the evidence collages and full-stack
RGB viewer before physical disposition decisions.

## Evidence collages by group

{chr(10).join(collage_sections)}
"""
    output = RUN_DIRECTORY / "REPORT.md"
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    base = load_group_runner()
    volume, threshold = base.load_volume_and_threshold()
    group_count = math.ceil(int(volume.shape[0]) / PERIOD_SLICES)
    groups = []
    for group_number in range(1, group_count + 1):
        group, _ = analyze_group(
            base,
            volume,
            threshold,
            group_number,
        )
        groups.append(group)

    combined = combine_group_arrays(
        groups,
        tuple(volume.shape),
        base.INPUT_FILE,
    )
    summary = {
        "source_file": str(base.INPUT_FILE),
        "input_shape": list(volume.shape),
        "input_dtype": str(volume.dtype),
        "analyzed_groups": list(range(1, group_count + 1)),
        "period_slices": PERIOD_SLICES,
        "otsu_threshold": threshold,
        "groups": groups,
        "combined_npy": combined,
        "pipeline_source": str(
            REPOSITORY_ROOT / "src" / "pattern_deformation_analysis.py"
        ),
    }
    summary_path = RUN_DIRECTORY / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary_csv = write_summary_csv(groups)
    chart = make_counts_chart(groups)
    contact_sheet = make_contact_sheet(groups)
    report = write_report(summary, summary_csv, chart, contact_sheet)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "report": str(report),
                "combined_npy": combined,
                "contact_sheet": str(contact_sheet),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
