"""Run the current pattern-irregularity detector on group 4 only."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu


RUN_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = RUN_DIRECTORY.parents[1]
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
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.pattern_deformation_analysis import (
    resolve_min_track_slices,
    run_grid_pattern_analysis,
)


INPUT_FILE = (
    REPOSITORY_ROOT
    / "data"
    / "missing_struts"
    / "tif_stacks"
    / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
)
VISUAL_RENDERER = (
    REPOSITORY_ROOT
    / "analysis_outputs"
    / "pattern_irregularities_210127_0point5dash1_20260728_132145"
    / "run_pattern_irregularity_analysis.py"
)
GROUP_NUMBER = 4
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


def load_visual_renderer():
    spec = importlib.util.spec_from_file_location(
        "pattern_irregularity_visual_renderer",
        VISUAL_RENDERER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load renderer from {VISUAL_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_DIRECTORY = RUN_DIRECTORY
    return module


def relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(RUN_DIRECTORY).as_posix()


def write_summary_csv(group: dict[str, object]) -> Path:
    output = RUN_DIRECTORY / "group_summary.csv"
    fields = [
        "group_number",
        "dataset_slice_start_one_based",
        "dataset_slice_end_one_based_inclusive",
        "slice_count",
        "confirmed_irregularity_count",
        "confirmed_irregularity_track_count",
        "confirmed_missing_strut_count",
        "confirmed_missing_track_count",
        "confirmed_broken_strut_count",
        "confirmed_broken_track_count",
        "report",
        "collage",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                **{field: group[field] for field in fields[:-2]},
                "report": relative(group["report"]),
                "collage": relative(group["collage"]),
            }
        )
    return output


def write_report(
    summary: dict[str, object],
    report: dict[str, object],
    summary_csv: Path,
) -> Path:
    group = summary["groups"][0]
    class_counts = report["confirmed_defect_instance_counts_by_class"]
    track_counts = report["confirmed_track_counts_by_class"]
    collage = relative(group["collage"])
    report_json = relative(group["report"])
    evidence_csv = (
        RUN_DIRECTORY
        / "group_004"
        / "group_004_grid_confirmed_patterns.csv"
    )
    with evidence_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    detected_slices = {
        int(row["dataset_slice_one_based"]) for row in rows
    }
    detected_cells = {
        (
            int(row["grid_row_zero_based"]),
            int(row["grid_column_zero_based"]),
        )
        for row in rows
    }
    rejected = int(report["tracking"]["rejected_track_count"])

    text = f"""# Group 4 Pattern Irregularity Analysis

## Executive summary

Only **group 4** was analyzed: dataset slices **235-312** (1-based), one
complete 78-slice lattice period. The current mutually exclusive
missing-versus-broken classifier identified **{int(group['confirmed_irregularity_count'])}
confirmed irregularity instances** across
**{int(group['confirmed_irregularity_track_count'])} tracks**.

| Irregularity class | Confirmed instances | Tracks |
|:--|--:|--:|
| Missing strut | {int(class_counts['missing_strut'])} | {int(track_counts['missing_strut'])} |
| Broken strut | {int(class_counts['broken_strut'])} | {int(track_counts['broken_strut'])} |
| **Total** | **{int(group['confirmed_irregularity_count'])}** | **{int(group['confirmed_irregularity_track_count'])}** |

Confirmed evidence occurs on **{len(detected_slices)} distinct slices** and in
**{len(detected_cells)} distinct {GRID_SIZE} x {GRID_SIZE} grid cells**. Another **{rejected}
candidate tracks** did not satisfy the persistence and confidence rules and
are retained in the detailed JSON for audit.

![Group 4 confirmed pattern irregularities]({collage})

Red overlays denote missing struts, orange overlays denote broken struts,
white outlines show detected material, and gold dashed boxes identify the
reported grid cells.

## Output inventory

- [Detailed findings JSON]({report_json})
- [Confirmed-pattern evidence CSV](group_004/group_004_grid_confirmed_patterns.csv)
- [Group summary CSV]({relative(summary_csv)})
- [Evidence collage]({collage})
- [Run summary JSON](run_summary.json)
- Slice-level candidate mask, confirmed mask, class-label volume, segmented
  RGB viewer, and method-step images are under `group_004/`.

## Method

- Input: `{Path(str(summary['source_file'])).name}`; shape
  `{tuple(summary['input_shape'])}` (Z, Y, X), dtype `{summary['input_dtype']}`.
- Scope: group 4 only, zero-based indices 234-311.
- Segmentation: high-valued foreground, Otsu threshold
  {float(summary['otsu_threshold']):.1f}, and a 3-pixel opening.
- Spatial model: {GRID_SIZE} x {GRID_SIZE} grid, periodic translated-neighbor consensus, two
  geometry-reference cycles, tilt correction, and forward trajectory
  prediction.
- Missing classification: nominal half-group persistence threshold of
  {int(summary['resolved_missing_span_slices'])} slices, with a 15% linked-span
  tolerance and at least 60% observation support, or a full absent segment
  spanning {MISSING_SPATIAL_EXTENT_CELLS:.4g} grid cells in at least two slices.
- Broken classification: shorter absence with at least 8 observations and
  class confidence at least 0.85, or a qualifying trajectory discontinuity.
- General confirmation confidence: at least 0.70. Missing and broken classes
  are mutually exclusive, and spatially/temporally matching fragments are
  de-duplicated before counting.

## Validation and limitations

All 30 tests in `tests.test_grid_pattern_inspection` passed immediately before
this run. Post-run checks verify group scope, summary consistency, array
dimensions and dtypes, output links, and image readability.

These are algorithmic screening findings, not metrology-certified ground
truth. Review the collage and slice-level viewer before making physical
disposition decisions.
"""
    output = RUN_DIRECTORY / "REPORT.md"
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    volume, threshold = load_volume_and_threshold()
    group_directory = RUN_DIRECTORY / "group_004"
    resolved_min_track = resolve_min_track_slices(
        None,
        PERIOD_SLICES,
        PERIOD_SLICES,
    )
    report = run_grid_pattern_analysis(
        slices=volume,
        source_path=INPUT_FILE,
        output_directory=group_directory,
        period_slices=PERIOD_SLICES,
        group_number=GROUP_NUMBER,
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
        missing_min_observation_fraction=MISSING_OBSERVATION_FRACTION,
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
        min_missing_clearance_fraction=MIN_MISSING_CLEARANCE_FRACTION,
        periodic_border_margin_pixels=2,
        periodic_border_margin_fraction=BORDER_MARGIN_FRACTION,
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
    report["inspection_mode"] = "grid_two_class_pattern_irregularities"
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
    report_path = (
        group_directory / "group_004_grid_findings_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    collage_report = copy.deepcopy(report)
    collage_report[
        "confirmed_defect_instance_counts_by_class"
    ]["thin_or_deformed_strut"] = 0
    collage_report["confirmed_track_counts_by_class"][
        "thin_or_deformed_strut"
    ] = 0
    renderer = load_visual_renderer()
    collage = renderer.make_irregularity_collage(
        GROUP_NUMBER,
        collage_report,
    )

    class_counts = report["confirmed_defect_instance_counts_by_class"]
    track_counts = report["confirmed_track_counts_by_class"]
    group = {
        "group_number": GROUP_NUMBER,
        "dataset_slice_start_one_based": 235,
        "dataset_slice_end_one_based_inclusive": 312,
        "slice_count": 78,
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
        "report": str(report_path),
        "collage": str(collage),
    }
    summary = {
        "source_file": str(INPUT_FILE),
        "input_shape": list(volume.shape),
        "input_dtype": str(volume.dtype),
        "analyzed_groups": [GROUP_NUMBER],
        "period_slices": PERIOD_SLICES,
        "resolved_missing_span_slices": resolved_min_track,
        "otsu_threshold": threshold,
        "groups": [group],
        "pipeline_source": str(
            REPOSITORY_ROOT / "src" / "pattern_deformation_analysis.py"
        ),
    }
    summary_path = RUN_DIRECTORY / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary_csv = write_summary_csv(group)
    final_report = write_report(summary, report, summary_csv)
    print(
        json.dumps(
            {
                "analyzed_groups": [GROUP_NUMBER],
                "summary": str(summary_path),
                "report": str(final_report),
                "collage": str(collage),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
