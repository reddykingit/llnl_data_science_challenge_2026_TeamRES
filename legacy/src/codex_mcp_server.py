from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from fastmcp import FastMCP
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from skimage.transform import resize

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")

# Spatial defaults for the current 6x6 volume.  Parameters measured in grid
# cells are scaled from the earlier 8x8 baseline; intensity, confidence, and
# temporal thresholds remain independent of the spatial split.
REFERENCE_GRID_SIZE = 8
DEFAULT_GRID_SIZE = 6
GRID_LINEAR_SCALE = DEFAULT_GRID_SIZE / REFERENCE_GRID_SIZE
GRID_AREA_SCALE = GRID_LINEAR_SCALE**2
DEFAULT_TRACK_RADIUS_CELLS = round(1.5 * GRID_LINEAR_SCALE, 6)
# Slightly stricter missing-strut calibration than pure 8-to-6 scaling.
DEFAULT_GROUP_MISSING_FRACTION_THRESHOLD = 0.45
DEFAULT_MISSING_FRACTION_THRESHOLD = 0.55
DEFAULT_MISSING_OBSERVATION_FRACTION = 0.60
DEFAULT_MISSING_SPATIAL_EXTENT_CELLS = 0.28
DEFAULT_MISSING_SEGMENT_FRACTION = 0.0003
DEFAULT_MISSING_CLEARANCE_FRACTION = 0.025
DEFAULT_BORDER_MARGIN_FRACTION = round(
    0.10 * GRID_LINEAR_SCALE,
    6,
)
DEFAULT_MAXIMUM_TILT_SHIFT_FRACTION = round(
    0.5 * GRID_LINEAR_SCALE,
    6,
)


def _load_volume(input_filepath: str) -> tuple[Path, np.ndarray]:
    """Load a numeric 3-D NPY or TIFF volume without enabling pickle."""
    path = Path(input_filepath).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input volume does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        volume = np.load(path, allow_pickle=False, mmap_mode="r")
    elif suffix in {".tif", ".tiff"}:
        try:
            volume = tifffile.memmap(path)
        except (OSError, ValueError):
            volume = tifffile.imread(path)
    else:
        raise ValueError("input_filepath must end in .npy, .tif, or .tiff")

    volume = np.asarray(volume)
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    is_numeric_or_bool = (
        np.issubdtype(volume.dtype, np.number)
        or np.issubdtype(volume.dtype, np.bool_)
    )
    if volume.ndim != 3 or not is_numeric_or_bool:
        raise ValueError(
            f"Expected a numeric 3-D volume; received shape={volume.shape}, "
            f"dtype={volume.dtype}"
        )
    return path, volume


def _sample_finite_values(volume: np.ndarray, limit: int = 1_000_000) -> np.ndarray:
    """Return a deterministic, bounded sample for input classification/thresholding."""
    flat = volume.reshape(-1)
    step = max(1, math.ceil(flat.size / limit))
    sample = np.asarray(flat[::step])
    return sample[np.isfinite(sample)]


def _resolve_threshold(
    volume: np.ndarray, threshold: float | None
) -> tuple[float, str]:
    """Choose a mask cutoff and identify whether the input is already segmented."""
    sample = _sample_finite_values(volume)
    if sample.size == 0:
        raise ValueError("Input volume contains no finite values")

    unique = np.unique(sample)
    is_segmentation = np.issubdtype(volume.dtype, np.bool_) or unique.size <= 2
    input_kind = "segmentation" if is_segmentation else "raw"

    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        return float(threshold), input_kind
    if is_segmentation:
        if unique.size < 2:
            raise ValueError(
                "A constant segmentation has no foreground/background pattern"
            )
        return float((float(unique[0]) + float(unique[-1])) / 2.0), input_kind
    if unique.size < 2:
        raise ValueError(
            "Automatic thresholding needs at least two intensity values; "
            "provide threshold explicitly"
        )
    return float(threshold_otsu(sample)), input_kind


def _slice_descriptor(
    image: np.ndarray,
    threshold: float,
    foreground: str,
    descriptor_size: int,
    min_foreground_fraction: float,
) -> np.ndarray | None:
    """Build a translation-tolerant descriptor of the nodes/struts in one slice."""
    finite = np.isfinite(image)
    if foreground == "high":
        mask = finite & (image >= threshold)
    else:
        mask = finite & (image <= threshold)
    if float(mask.mean()) < min_foreground_fraction:
        return None

    small = resize(
        mask.astype(np.float32),
        (descriptor_size, descriptor_size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    center_y, center_x = ndimage.center_of_mass(small)
    if np.isfinite(center_y) and np.isfinite(center_x):
        small = ndimage.shift(
            small,
            (
                (descriptor_size - 1) / 2.0 - center_y,
                (descriptor_size - 1) / 2.0 - center_x,
            ),
            order=1,
            mode="constant",
            cval=0.0,
        )

    gradient_y, gradient_x = np.gradient(small)
    descriptor = np.concatenate(
        (small.ravel(), gradient_y.ravel(), gradient_x.ravel())
    )
    descriptor -= descriptor.mean()
    norm = float(np.linalg.norm(descriptor))
    return None if norm <= 1e-12 else descriptor / norm


def _build_descriptors(
    slices: np.ndarray,
    threshold: float,
    foreground: str,
    descriptor_size: int,
    min_foreground_fraction: float,
) -> list[np.ndarray | None]:
    return [
        _slice_descriptor(
            image,
            threshold,
            foreground,
            descriptor_size,
            min_foreground_fraction,
        )
        for image in slices
    ]


def _estimate_repeat_period(
    descriptors: list[np.ndarray | None],
    period_min: int,
    period_max: int,
) -> dict[str, Any]:
    """Score candidate lags using median same-phase descriptor similarity."""
    slice_count = len(descriptors)
    if not 1 <= period_min <= period_max < slice_count:
        raise ValueError(
            "Period bounds must satisfy "
            "1 <= period_min <= period_max < slice count"
        )

    lags: list[int] = []
    scores: list[float] = []
    pair_counts: list[int] = []
    for lag in range(period_min, period_max + 1):
        similarities = [
            float(np.dot(left, right))
            for left, right in zip(descriptors[:-lag], descriptors[lag:])
            if left is not None and right is not None
        ]
        lags.append(lag)
        scores.append(
            float(np.median(similarities)) if similarities else math.nan
        )
        pair_counts.append(len(similarities))

    score_array = np.asarray(scores, dtype=float)
    finite = np.isfinite(score_array)
    if not finite.any():
        raise ValueError(
            "Not enough nonblank slices to estimate an octet repeat length"
        )

    peak_input = np.where(finite, score_array, -2.0)
    detected_peak_indices, properties = find_peaks(
        peak_input,
        prominence=0.01,
        distance=max(2, period_min // 3),
    )
    prominence_by_index = {
        int(index): float(prominence)
        for index, prominence in zip(
            detected_peak_indices, properties["prominences"]
        )
    }
    global_max_index = int(np.nanargmax(score_array))
    peak_indices = np.unique(
        np.append(detected_peak_indices, global_max_index)
    ).astype(int)

    best_peak_score = max(float(score_array[index]) for index in peak_indices)
    # Prefer the fundamental period when a harmonic has effectively the same score.
    eligible = [
        int(index)
        for index in peak_indices
        if score_array[index] >= best_peak_score - 0.02
    ]
    selected_index = min(eligible, key=lambda index: lags[index])

    baseline = float(np.nanmedian(score_array))
    mad = float(np.nanmedian(np.abs(score_array[finite] - baseline)))
    separation = max(0.0, float(score_array[selected_index]) - baseline)
    confidence = min(1.0, separation / max(0.02, 5.0 * 1.4826 * mad))
    confidence *= min(1.0, pair_counts[selected_index] / 20.0)

    ranked_peaks = sorted(
        (int(index) for index in peak_indices),
        key=lambda index: float(score_array[index]),
        reverse=True,
    )[:8]
    candidates = []
    for index in ranked_peaks:
        candidates.append(
            {
                "length_slices": lags[index],
                "similarity": float(score_array[index]),
                "pair_count": pair_counts[index],
                "prominence": prominence_by_index.get(index, 0.0),
            }
        )

    return {
        "length_slices": lags[selected_index],
        "confidence": float(confidence),
        "selected_similarity": float(score_array[selected_index]),
        "baseline_similarity": baseline,
        "candidates": candidates,
    }


def _cycle_similarity(
    descriptors: list[np.ndarray | None],
    previous_start: int,
    candidate_start: int,
    period: int,
) -> float:
    """Compare two complete pattern phases while tolerating missing/blank slices."""
    similarities = []
    for offset in range(period):
        previous_index = previous_start + offset
        candidate_index = candidate_start + offset
        if candidate_index >= len(descriptors):
            break
        left = descriptors[previous_index]
        right = descriptors[candidate_index]
        if left is not None and right is not None:
            similarities.append(float(np.dot(left, right)))
    return float(np.median(similarities)) if similarities else -math.inf


def _find_layer_boundaries(
    descriptors: list[np.ndarray | None], period: int
) -> tuple[list[int], int, int]:
    """Find locally refined layer boundaries within the foreground-bearing range."""
    usable = [
        index
        for index, descriptor in enumerate(descriptors)
        if descriptor is not None
    ]
    if not usable:
        raise ValueError("No foreground-bearing slices were found")

    analysis_start = usable[0]
    analysis_stop = usable[-1] + 1
    boundaries = [analysis_start]
    previous_start = analysis_start
    slack = max(1, min(8, round(period * 0.08)))

    while True:
        expected = previous_start + period
        if expected >= analysis_stop:
            break
        candidates = range(
            max(previous_start + 1, expected - slack),
            min(analysis_stop, expected + slack + 1),
        )
        chosen = max(
            candidates,
            key=lambda candidate: (
                _cycle_similarity(
                    descriptors, previous_start, candidate, period
                ),
                -abs(candidate - expected),
            ),
        )
        boundaries.append(chosen)
        previous_start = chosen

    if boundaries[-1] != analysis_stop:
        boundaries.append(analysis_stop)
    return boundaries, analysis_start, analysis_stop


def _save_layer_volume(
    layer: np.ndarray, output_path: Path, source_suffix: str
) -> None:
    if source_suffix == ".npy":
        np.save(output_path, layer, allow_pickle=False)
    else:
        tifffile.imwrite(output_path, layer)


def _write_layer_files(
    volume: np.ndarray,
    slice_axis: int,
    source_path: Path,
    output_directory: str | None,
    layers: list[dict[str, Any]],
) -> tuple[Path, list[str]]:
    output_dir = (
        Path(output_directory).expanduser().resolve()
        if output_directory
        else source_path.with_name(f"{source_path.stem}_octet_layers")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    slices = np.moveaxis(volume, slice_axis, 0)
    suffix = ".npy" if source_path.suffix.lower() == ".npy" else ".tif"
    saved_paths: list[str] = []

    for layer in layers:
        start = int(layer["start_index_zero_based"])
        stop = int(layer["end_index_zero_based"]) + 1
        layer_in_source_orientation = np.moveaxis(slices[start:stop], 0, slice_axis)
        output_path = output_dir / (
            f"{source_path.stem}_octet_layer_{layer['layer_number']:03d}"
            f"_slices_{start + 1:04d}-{stop:04d}{suffix}"
        )
        _save_layer_volume(
            layer_in_source_orientation, output_path, source_path.suffix.lower()
        )
        layer["file"] = str(output_path)
        saved_paths.append(str(output_path))

    return output_dir, saved_paths


def _resolve_shared_threshold(
    volumes: list[np.ndarray], threshold: float | None
) -> tuple[float, str]:
    """Choose one threshold for every group so their masks remain comparable."""
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        return float(threshold), "user"

    sample_limit = max(10_000, 1_000_000 // len(volumes))
    samples = []
    all_segmentations = True
    for group_index, volume in enumerate(volumes, start=1):
        sample = _sample_finite_values(volume, limit=sample_limit)
        if not sample.size:
            raise ValueError(
                f"Input group {group_index} contains no finite values"
            )
        samples.append(sample)
        all_segmentations = all_segmentations and (
            np.issubdtype(volume.dtype, np.bool_)
            or np.unique(sample).size <= 2
        )
    combined = np.concatenate(samples)
    unique = np.unique(combined)
    if unique.size < 2:
        raise ValueError("Input groups have no foreground/background variation")

    if all_segmentations:
        return (
            float((float(unique[0]) + float(unique[-1])) / 2.0),
            "label_midpoint",
        )
    return float(threshold_otsu(combined)), "otsu"


def _downsample_group_masks(
    slices: np.ndarray,
    threshold: float,
    foreground: str,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Segment and resize a slice group to a bounded CNN-like feature plane."""
    masks = np.empty((len(slices), *output_shape), dtype=bool)
    for index, image in enumerate(slices):
        finite = np.isfinite(image)
        if foreground == "high":
            mask = finite & (image >= threshold)
        else:
            mask = finite & (image <= threshold)
        masks[index] = resize(
            mask,
            output_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(bool)
    return masks


def _foreground_roi(
    group_masks: list[np.ndarray], presence_fraction: float
) -> tuple[int, int, int, int]:
    """Find a common foreground region used by every grid."""
    projection = np.zeros(group_masks[0].shape[1:], dtype=np.float64)
    slice_total = 0
    for masks in group_masks:
        projection += masks.sum(axis=0)
        slice_total += len(masks)
    present = projection / max(1, slice_total) >= presence_fraction
    coordinates = np.argwhere(present)
    if coordinates.size == 0:
        height, width = projection.shape
        return 0, height, 0, width
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    return int(y0), int(y1), int(x0), int(x1)


def _normalized_patch(
    patch: np.ndarray,
    patch_size: int,
    max_registration_pixels: int,
) -> np.ndarray:
    """Resize and center one local receptive field for motif comparison."""
    normalized = resize(
        patch,
        (patch_size, patch_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)
    if not normalized.any():
        return normalized
    center = ndimage.center_of_mass(normalized)
    if not all(np.isfinite(center)):
        return normalized
    target_center = np.array(
        [(patch_size - 1) / 2.0, (patch_size - 1) / 2.0]
    )
    shift = np.clip(
        target_center - np.asarray(center),
        -max_registration_pixels,
        max_registration_pixels,
    )
    return ndimage.shift(
        normalized,
        shift=tuple(float(value) for value in shift),
        order=0,
        mode="constant",
        cval=0,
    ).astype(bool)


def _select_dynamic_box_phase(
    target: np.ndarray,
    roi: tuple[int, int, int, int],
    grid_rows: int,
    grid_columns: int,
    box_overlap: float,
) -> tuple[float, float]:
    """Align box centers with the recurring foreground pattern."""
    y0, y1, x0, x1 = roi
    cell_height = (y1 - y0) / grid_rows
    cell_width = (x1 - x0) / grid_columns
    phase_step = 1.0 - box_overlap
    phase_offsets = [
        index * phase_step
        for index in range(max(1, int(math.ceil(1.0 / phase_step))))
        if index * phase_step < 1.0
    ]
    scores = []
    for row_offset in phase_offsets:
        for column_offset in phase_offsets:
            central_density = []
            for row in range(grid_rows):
                center_y = y0 + (row + 0.5 + row_offset) * cell_height
                if center_y + cell_height / 2.0 > y1 + 0.5:
                    continue
                for column in range(grid_columns):
                    center_x = (
                        x0 + (column + 0.5 + column_offset) * cell_width
                    )
                    if center_x + cell_width / 2.0 > x1 + 0.5:
                        continue
                    half_core_height = cell_height * 0.25
                    half_core_width = cell_width * 0.25
                    core = target[
                        max(y0, int(round(center_y - half_core_height))) :
                        min(y1, int(round(center_y + half_core_height))),
                        max(x0, int(round(center_x - half_core_width))) :
                        min(x1, int(round(center_x + half_core_width))),
                    ]
                    if core.size:
                        central_density.append(float(core.mean()))
            scores.append(
                (
                    float(np.mean(central_density))
                    if central_density
                    else -1.0,
                    row_offset,
                    column_offset,
                )
            )
    _, row_offset, column_offset = max(scores)
    return float(row_offset), float(column_offset)


def _grid_components(
    target: np.ndarray,
    roi: tuple[int, int, int, int],
    grid_rows: int,
    grid_columns: int,
    patch_size: int,
    expected_vote: float,
    missing_fraction_threshold: float,
    similarity_threshold: float,
    min_expected_patch_fraction: float,
    spatial_tolerance_pixels: int,
    max_registration_pixels: int,
    slice_index: int,
    box_mode: str,
    box_overlap: float,
    nms_iou_threshold: float,
    box_phase_offset: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Learn a slice motif and return distinct local pattern violations."""
    y0, y1, x0, x1 = roi
    cell_height = (y1 - y0) / grid_rows
    cell_width = (x1 - x0) / grid_columns

    phase_offset_pairs = (
        [(0.0, 0.0)]
        if box_mode == "fixed"
        else (
            [box_phase_offset]
            if box_phase_offset is not None
            else [
                _select_dynamic_box_phase(
                    target,
                    roi,
                    grid_rows,
                    grid_columns,
                    box_overlap,
                )
            ]
        )
    )
    phase_candidates: list[tuple[float, list[dict[str, Any]]]] = []
    for row_offset, column_offset in phase_offset_pairs:
            phase_windows: list[dict[str, Any]] = []
            central_density = []
            for row in range(grid_rows):
                center_y = y0 + (row + 0.5 + row_offset) * cell_height
                if center_y + cell_height / 2.0 > y1 + 0.5:
                    continue
                for column in range(grid_columns):
                    center_x = (
                        x0 + (column + 0.5 + column_offset) * cell_width
                    )
                    if center_x + cell_width / 2.0 > x1 + 0.5:
                        continue
                    window_y0 = max(
                        y0, int(round(center_y - cell_height / 2.0))
                    )
                    window_y1 = min(
                        y1, int(round(center_y + cell_height / 2.0))
                    )
                    window_x0 = max(
                        x0, int(round(center_x - cell_width / 2.0))
                    )
                    window_x1 = min(
                        x1, int(round(center_x + cell_width / 2.0))
                    )
                    patch = target[
                        window_y0:window_y1, window_x0:window_x1
                    ]
                    motif = _normalized_patch(
                        patch,
                        patch_size,
                        max_registration_pixels,
                    )
                    core_y0 = int(round(patch.shape[0] * 0.25))
                    core_y1 = int(round(patch.shape[0] * 0.75))
                    core_x0 = int(round(patch.shape[1] * 0.25))
                    core_x1 = int(round(patch.shape[1] * 0.75))
                    central_density.append(
                        float(
                            patch[
                                core_y0:core_y1,
                                core_x0:core_x1,
                            ].mean()
                        )
                    )
                    phase_windows.append(
                        {
                            "motif": motif,
                            "centroid_grid_row": float(
                                (center_y - y0) / cell_height - 0.5
                            ),
                            "centroid_grid_column": float(
                                (center_x - x0) / cell_width - 0.5
                            ),
                            "bounds": (
                                window_y0,
                                window_y1,
                                window_x0,
                                window_x1,
                            ),
                            "touches_scan_edge": (
                                window_y0 <= y0
                                or window_y1 >= y1
                                or window_x0 <= x0
                                or window_x1 >= x1
                            ),
                            "phase_offset": {
                                "row": float(row_offset),
                                "column": float(column_offset),
                            },
                        }
                    )
            if phase_windows:
                phase_candidates.append(
                    (float(np.mean(central_density)), phase_windows)
                )

    if not phase_candidates:
        return []
    _, windows = max(phase_candidates, key=lambda item: item[0])
    reference_motifs = [
        window["motif"]
        for window in windows
        if float(window["motif"].mean()) >= min_expected_patch_fraction
    ]

    # A dominant motif needs several spatial examples within this cross section.
    minimum_references = min(
        len(windows),
        max(3, int(math.ceil(len(windows) * 0.1))),
    )
    if len(reference_motifs) < minimum_references:
        return []
    expected_probability = np.mean(reference_motifs, axis=0)
    expected = expected_probability >= expected_vote
    expected_pixels = int(expected.sum())
    if expected_pixels == 0:
        return []

    candidates = []
    for window in windows:
        target_patch = window["motif"]
        target_for_comparison = (
            ndimage.binary_dilation(
                target_patch,
                iterations=spatial_tolerance_pixels,
            )
            if spatial_tolerance_pixels
            else target_patch
        )
        shared_pixels = int(np.logical_and(expected, target_for_comparison).sum())
        missing_fraction = (expected_pixels - shared_pixels) / expected_pixels
        target_pixels = int(target_for_comparison.sum())
        denominator = expected_pixels + target_pixels
        similarity = 2.0 * shared_pixels / denominator if denominator else 1.0
        if (
            missing_fraction < missing_fraction_threshold
            or similarity > similarity_threshold
        ):
            continue
        row = int(
            np.clip(
                round(window["centroid_grid_row"]),
                0,
                grid_rows - 1,
            )
        )
        column = int(
            np.clip(
                round(window["centroid_grid_column"]),
                0,
                grid_columns - 1,
            )
        )
        candidates.append(
            {
                "slice_index_zero_based": slice_index,
                "slice_number_one_based": slice_index + 1,
                "centroid_grid_row": window["centroid_grid_row"],
                "centroid_grid_column": window["centroid_grid_column"],
                "grid_row_min": row,
                "grid_row_max": row,
                "grid_column_min": column,
                "grid_column_max": column,
                "grid_cells": [{"row": row, "column": column}],
                "median_missing_fraction": float(missing_fraction),
                "max_missing_fraction": float(missing_fraction),
                "median_patch_similarity": float(similarity),
                "median_expected_foreground_fraction": float(
                    expected_pixels / max(1, target_patch.size)
                ),
                "prototype_reference_patch_count": len(reference_motifs),
                "box_mode": box_mode,
                "box_phase_offset_cells": window["phase_offset"],
                "box_bounds_analysis_pixels": {
                    "y_start": int(window["bounds"][0]),
                    "y_end_exclusive": int(window["bounds"][1]),
                    "x_start": int(window["bounds"][2]),
                    "x_end_exclusive": int(window["bounds"][3]),
                },
                "touches_scan_edge": bool(window["touches_scan_edge"]),
            }
        )

    def intersection_over_union(
        left: dict[str, Any], right: dict[str, Any]
    ) -> float:
        a = left["box_bounds_analysis_pixels"]
        b = right["box_bounds_analysis_pixels"]
        height = max(
            0,
            min(a["y_end_exclusive"], b["y_end_exclusive"])
            - max(a["y_start"], b["y_start"]),
        )
        width = max(
            0,
            min(a["x_end_exclusive"], b["x_end_exclusive"])
            - max(a["x_start"], b["x_start"]),
        )
        intersection = height * width
        left_area = (
            (a["y_end_exclusive"] - a["y_start"])
            * (a["x_end_exclusive"] - a["x_start"])
        )
        right_area = (
            (b["y_end_exclusive"] - b["y_start"])
            * (b["x_end_exclusive"] - b["x_start"])
        )
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0

    # Overlapping windows create duplicate responses. NMS retains the strongest
    # response at each location while leaving spatially distinct losses intact.
    selected: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["median_missing_fraction"],
            -item["median_patch_similarity"],
        ),
        reverse=True,
    ):
        if all(
            intersection_over_union(candidate, previous) <= nms_iou_threshold
            for previous in selected
        ):
            selected.append(candidate)
    return selected


def _track_grid_components(
    components_by_slice: list[list[dict[str, Any]]],
    track_radius_cells: float,
    max_slice_gap: int,
) -> list[list[dict[str, Any]]]:
    """Track multiple losses with one-to-one assignment, then bridge gaps."""
    tracks: list[list[dict[str, Any]]] = []
    for slice_components in components_by_slice:
        if not slice_components:
            continue
        slice_index = int(slice_components[0]["slice_index_zero_based"])
        active_indices = [
            index
            for index, track in enumerate(tracks)
            if 1
            <= slice_index - int(track[-1]["slice_index_zero_based"])
            <= max_slice_gap + 1
        ]
        matched_components: set[int] = set()
        if active_indices:
            costs = np.full(
                (len(active_indices), len(slice_components)),
                track_radius_cells + 1.0,
                dtype=float,
            )
            for active_row, track_index in enumerate(active_indices):
                previous = tracks[track_index][-1]
                for component_index, component in enumerate(slice_components):
                    costs[active_row, component_index] = math.hypot(
                        float(previous["centroid_grid_row"])
                        - float(component["centroid_grid_row"]),
                        float(previous["centroid_grid_column"])
                        - float(component["centroid_grid_column"]),
                    )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] <= track_radius_cells:
                    tracks[active_indices[int(row)]].append(
                        slice_components[int(column)]
                    )
                    matched_components.add(int(column))
        for component_index, component in enumerate(slice_components):
            if component_index not in matched_components:
                tracks.append([component])

    # Reconnect non-overlapping fragments across the short square/node phase.
    merged = True
    while merged:
        merged = False
        for left_index, left in enumerate(tracks):
            left_end = int(left[-1]["slice_index_zero_based"])
            for right_index in range(left_index + 1, len(tracks)):
                right = tracks[right_index]
                right_start = int(right[0]["slice_index_zero_based"])
                if not 1 <= right_start - left_end <= max_slice_gap + 1:
                    continue
                distance = math.hypot(
                    float(left[-1]["centroid_grid_row"])
                    - float(right[0]["centroid_grid_row"]),
                    float(left[-1]["centroid_grid_column"])
                    - float(right[0]["centroid_grid_column"]),
                )
                if distance <= track_radius_cells:
                    tracks[left_index] = left + right
                    del tracks[right_index]
                    merged = True
                    break
            if merged:
                break
    return tracks


def _longest_persistent_span(
    slice_indices: list[int], max_slice_gap: int
) -> int:
    """Longest cross-section span after tolerating only small detection gaps."""
    if not slice_indices:
        return 0
    ordered = sorted(set(slice_indices))
    longest = current_start = previous = ordered[0]
    for index in ordered[1:]:
        if index - previous <= max_slice_gap + 1:
            previous = index
        else:
            longest = max(longest, previous - current_start + 1)
            current_start = previous = index
    return max(longest, previous - current_start + 1)


def _analyze_group_masks(
    group_masks: list[np.ndarray],
    group_paths: list[Path],
    original_plane_shape: tuple[int, int],
    roi: tuple[int, int, int, int],
    grid_rows: int,
    grid_columns: int,
    patch_size: int,
    expected_vote: float,
    missing_fraction_threshold: float,
    similarity_threshold: float,
    min_expected_patch_fraction: float,
    persistence_fraction: float,
    spatial_tolerance_pixels: int,
    max_registration_pixels: int,
    track_radius_cells: float,
    max_slice_gap: int | None,
    max_edge_touch_fraction: float,
    box_mode: str,
    box_overlap: float,
    nms_iou_threshold: float,
) -> list[dict[str, Any]]:
    """Learn within-slice motifs and report persistent local pattern violations."""
    results = []
    for group_index, target_group in enumerate(group_masks):
        group_box_phase = (
            _select_dynamic_box_phase(
                np.mean(target_group, axis=0),
                roi,
                grid_rows,
                grid_columns,
                box_overlap,
            )
            if box_mode == "dynamic"
            else (0.0, 0.0)
        )
        components_by_slice: list[list[dict[str, Any]]] = []
        for slice_index, target in enumerate(target_group):
            components_by_slice.append(
                _grid_components(
                    target,
                    roi,
                    grid_rows,
                    grid_columns,
                    patch_size,
                    expected_vote,
                    missing_fraction_threshold,
                    similarity_threshold,
                    min_expected_patch_fraction,
                    spatial_tolerance_pixels,
                    max_registration_pixels,
                    slice_index,
                    box_mode,
                    box_overlap,
                    nms_iou_threshold,
                    group_box_phase,
                )
            )

        resolved_slice_gap = (
            max_slice_gap
            if max_slice_gap is not None
            else max(1, int(round(len(target_group) * 0.1)))
        )
        tracks = _track_grid_components(
            components_by_slice,
            track_radius_cells,
            resolved_slice_gap,
        )
        required_slices = max(
            1, int(math.ceil(len(target_group) * persistence_fraction))
        )
        missing_patterns = []
        edge_inconclusive_patterns = []
        for track in tracks:
            slice_indices = [
                int(item["slice_index_zero_based"]) for item in track
            ]
            persistent_span = _longest_persistent_span(
                slice_indices, resolved_slice_gap
            )
            detected_slice_count = len(set(slice_indices))
            if (
                persistent_span < required_slices
                or detected_slice_count < required_slices
            ):
                continue
            unique_cells = sorted(
                {
                    (cell["row"], cell["column"])
                    for observation in track
                    for cell in observation["grid_cells"]
                }
            )
            median_missing = float(
                np.median(
                    [item["median_missing_fraction"] for item in track]
                )
            )
            persistence_ratio = persistent_span / len(target_group)
            detection_coverage = detected_slice_count / persistent_span
            edge_observation_count = sum(
                bool(
                    observation.get(
                        "touches_scan_edge",
                        any(
                            int(cell["row"]) in {0, grid_rows - 1}
                            or int(cell["column"]) in {0, grid_columns - 1}
                            for cell in observation["grid_cells"]
                        ),
                    )
                )
                for observation in track
            )
            edge_touch_fraction = edge_observation_count / len(track)
            is_scan_edge_inconclusive = (
                edge_touch_fraction > max_edge_touch_fraction
            )
            classification = (
                "inconclusive_scan_edge"
                if is_scan_edge_inconclusive
                else "missing_pattern"
            )
            destination = (
                edge_inconclusive_patterns
                if is_scan_edge_inconclusive
                else missing_patterns
            )
            destination.append(
                {
                    "pattern_id": len(destination) + 1,
                    "classification": classification,
                    "cause": "unresolved_physical_or_acquisition_loss",
                    "confidence": float(
                        min(
                            1.0,
                            median_missing
                            * persistence_ratio
                            * detection_coverage,
                        )
                    ),
                    "start_index_zero_based": min(slice_indices),
                    "end_index_zero_based": max(slice_indices),
                    "start_slice_one_based": min(slice_indices) + 1,
                    "end_slice_one_based": max(slice_indices) + 1,
                    "detected_slice_count": detected_slice_count,
                    "persistent_span_slices": persistent_span,
                    "required_persistence_slices": required_slices,
                    "persistence_fraction_of_group": persistence_ratio,
                    "detection_coverage_within_span": detection_coverage,
                    "edge_touch_fraction": edge_touch_fraction,
                    "edge_observation_count": edge_observation_count,
                    "median_missing_fraction": median_missing,
                    "max_missing_fraction": float(
                        max(item["max_missing_fraction"] for item in track)
                    ),
                    "median_patch_similarity": float(
                        np.median(
                            [
                                item["median_patch_similarity"]
                                for item in track
                            ]
                        )
                    ),
                    "grid_cells_zero_based": [
                        {"row": int(row), "column": int(column)}
                        for row, column in unique_cells
                    ],
                    "observations": track,
                }
            )

        results.append(
            {
                "group_number": group_index + 1,
                "input_file": str(group_paths[group_index]),
                "slice_count": int(len(target_group)),
                "prototype_source": "other_grid_boxes_in_same_cross_section",
                "box_phase_offset_cells": {
                    "row": group_box_phase[0],
                    "column": group_box_phase[1],
                },
                "resolved_max_slice_gap": resolved_slice_gap,
                "candidate_track_count": len(tracks),
                "missing_pattern_count": len(missing_patterns),
                "missing_patterns": missing_patterns,
                "edge_inconclusive_pattern_count": len(
                    edge_inconclusive_patterns
                ),
                "edge_inconclusive_patterns": edge_inconclusive_patterns,
            }
        )

    analysis_height, analysis_width = group_masks[0].shape[1:]
    scale_y = original_plane_shape[0] / analysis_height
    scale_x = original_plane_shape[1] / analysis_width
    y0, y1, x0, x1 = roi
    original_roi = {
        "y_start": int(math.floor(y0 * scale_y)),
        "y_end_exclusive": int(math.ceil(y1 * scale_y)),
        "x_start": int(math.floor(x0 * scale_x)),
        "x_end_exclusive": int(math.ceil(x1 * scale_x)),
    }
    for result in results:
        result["analysis_roi_original_pixels"] = original_roi
        for pattern_key in (
            "missing_patterns",
            "edge_inconclusive_patterns",
        ):
            for pattern in result[pattern_key]:
                for observation in pattern["observations"]:
                    bounds = observation.get(
                        "box_bounds_analysis_pixels"
                    )
                    if bounds is None:
                        continue
                    observation["box_bounds_original_pixels"] = {
                        "y_start": int(
                            math.floor(bounds["y_start"] * scale_y)
                        ),
                        "y_end_exclusive": int(
                            math.ceil(bounds["y_end_exclusive"] * scale_y)
                        ),
                        "x_start": int(
                            math.floor(bounds["x_start"] * scale_x)
                        ),
                        "x_end_exclusive": int(
                            math.ceil(bounds["x_end_exclusive"] * scale_x)
                        ),
                    }
    return results


@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.
    
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    pass # Implementation goes here

@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """
    Loads a 3D CT dataset from a .npy file and saves a visualization of a specific slice to an image file.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT data.
        output_filepath: Path indicating where the output image should be saved (e.g., .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    pass # Implementation goes here

@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """
    Creates a skeleton from a 3D segmentation mask.
    
    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    pass # Implementation goes here, calling skeletonize_mask internally


@mcp.tool()
def analyze_octet_layer_length(
    input_filepath: str,
    slice_axis: int = 0,
    threshold: float | None = None,
    foreground: str = "high",
    period_min: int = 8,
    period_max: int | None = None,
    save_layer_files: bool = False,
    output_directory: str | None = None,
) -> dict[str, Any]:
    """
    Find the repeating octet-layer length in a raw CT or segmentation volume.

    The input may be a numeric .npy, .tif, or .tiff 3-D volume. Raw CT data is
    automatically segmented with Otsu's threshold unless ``threshold`` is
    supplied. Binary 0/1 data is recognized as a segmentation. Each slice is
    represented by a centered, downsampled node/strut descriptor, and candidate
    repeat lags are scored across the volume so small scan-position shifts do
    not prevent a match.

    Args:
        input_filepath: Raw CT or segmentation volume (.npy, .tif, or .tiff).
        slice_axis: Axis along which CT slices/layers are ordered (0, 1, or 2).
        threshold: Optional raw-intensity segmentation threshold. When omitted,
            binary/two-label inputs use their midpoint and raw inputs use
            Otsu's method.
        foreground: ``"high"`` when lattice material is above the threshold or
            ``"low"`` when it is below the threshold.
        period_min: Smallest plausible octet-layer length, in slices.
        period_max: Largest plausible length. Defaults to the smaller of 250
            slices and one half of the scan depth.
        save_layer_files: If true, split the detected foreground-bearing range
            into one file per octet layer.
        output_directory: Directory for optional layer files and their JSON
            manifest. Defaults beside the input as ``<stem>_octet_layers``.

    Returns:
        Structured results containing the estimated repeat length, confidence,
        alternative candidate lengths, every layer's slice range and measured
        length, and optional paths to the saved layer files and manifest.
        Indices are reported in both zero-based and one-based form.
    """
    path, volume = _load_volume(input_filepath)
    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, or 2")
    if foreground not in {"high", "low"}:
        raise ValueError("foreground must be 'high' or 'low'")
    if period_min < 2:
        raise ValueError("period_min must be at least 2 slices")

    slices = np.moveaxis(volume, slice_axis, 0)
    slice_count = int(slices.shape[0])
    if slice_count <= period_min:
        raise ValueError(
            f"Scan has {slice_count} slices on axis {slice_axis}; "
            f"period_min must be smaller"
        )
    resolved_period_max = (
        min(250, max(period_min + 1, slice_count // 2))
        if period_max is None
        else period_max
    )
    resolved_period_max = min(resolved_period_max, slice_count - 1)
    if resolved_period_max < period_min:
        raise ValueError("period_max must be greater than or equal to period_min")

    resolved_threshold, input_kind = _resolve_threshold(volume, threshold)
    descriptors = _build_descriptors(
        slices,
        resolved_threshold,
        foreground,
        descriptor_size=48,
        min_foreground_fraction=0.001,
    )
    repeat = _estimate_repeat_period(
        descriptors, period_min, resolved_period_max
    )
    period = int(repeat["length_slices"])
    boundaries, analysis_start, analysis_stop = _find_layer_boundaries(
        descriptors, period
    )

    layers = []
    for layer_number, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:]), start=1
    ):
        length = stop - start
        layers.append(
            {
                "layer_number": layer_number,
                "start_index_zero_based": start,
                "end_index_zero_based": stop - 1,
                "start_slice_one_based": start + 1,
                "end_slice_one_based": stop,
                "length_slices": length,
                "is_complete": abs(length - period)
                <= max(1, round(period * 0.25)),
            }
        )

    result: dict[str, Any] = {
        "input_file": str(path),
        "input_kind": input_kind,
        "volume_shape": [int(size) for size in volume.shape],
        "slice_axis": slice_axis,
        "slice_count": slice_count,
        "threshold_used": resolved_threshold,
        "threshold_source": "user" if threshold is not None else (
            "label_midpoint" if input_kind == "segmentation" else "otsu"
        ),
        "foreground": foreground,
        "estimated_octet_layer_length_slices": period,
        "confidence": repeat["confidence"],
        "selected_similarity": repeat["selected_similarity"],
        "baseline_similarity": repeat["baseline_similarity"],
        "candidate_lengths": repeat["candidates"],
        "analysis_range": {
            "start_index_zero_based": analysis_start,
            "end_index_zero_based": analysis_stop - 1,
            "start_slice_one_based": analysis_start + 1,
            "end_slice_one_based": analysis_stop,
            "excluded_leading_slices": analysis_start,
            "excluded_trailing_slices": slice_count - analysis_stop,
            "boundary_anchor": "first foreground-bearing slice",
        },
        "layer_count": len(layers),
        "layer_lengths_slices": [
            int(layer["length_slices"]) for layer in layers
        ],
        "layers": layers,
        "layer_files_saved": save_layer_files,
    }

    if save_layer_files:
        output_dir, saved_paths = _write_layer_files(
            volume,
            slice_axis,
            path,
            output_directory,
            layers,
        )
        manifest_path = output_dir / "octet_layers.json"
        result["output_directory"] = str(output_dir)
        result["layer_files"] = saved_paths
        result["manifest_file"] = str(manifest_path)
        manifest_path.write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )

    return result


@mcp.tool()
def analyze_slice_groups_for_missing_patterns(
    group_filepaths: list[str],
    segmentation_filepaths: list[str] | None = None,
    slice_axis: int = 0,
    threshold: float | None = None,
    foreground: str = "high",
    grid_rows: int = DEFAULT_GRID_SIZE,
    grid_columns: int = DEFAULT_GRID_SIZE,
    analysis_size: int = 288,
    patch_size: int = 24,
    expected_vote: float = 0.6,
    missing_fraction_threshold: float = (
        DEFAULT_GROUP_MISSING_FRACTION_THRESHOLD
    ),
    similarity_threshold: float = 0.7,
    min_expected_patch_fraction: float = 0.02,
    persistence_fraction: float = 0.5,
    spatial_tolerance_pixels: int = 2,
    max_registration_pixels: int = 8,
    track_radius_cells: float = DEFAULT_TRACK_RADIUS_CELLS,
    max_slice_gap: int | None = None,
    max_edge_touch_fraction: float = 0.5,
    box_mode: str = "fixed",
    box_overlap: float = 0.5,
    nms_iou_threshold: float = 0.3,
    roi_presence_fraction: float = 0.01,
    output_filepath: str | None = None,
) -> dict[str, Any]:
    """
    Find persistent missing local patterns across multiple slice groups.

    This is not a symmetry test and does not require a trained neural network.
    It uses a CNN-like local-receptive-field approach: each cross section is
    segmented and scanned with overlapping, pattern-sized receptive fields.
    The other boxes in the same cross section form its expected dominant-pattern
    prototype after each patch is resized and centered. Windows with substantial
    expected material missing and low similarity become candidates. Overlapping
    duplicate responses are suppressed, then distinct candidates are tracked
    one-to-one across sections, allowing several missing patterns at once. A
    candidate is classified as a missing pattern only when its track spans at
    least ``persistence_fraction`` of that group's slices (one half by default).

    Args:
        group_filepaths: One or more .npy, .tif, or .tiff slice-group volumes.
            Every group is analyzed independently using its repeated spatial
            grid motifs.
        segmentation_filepaths: Optional binary segmentation volume for every
            group, in the same order and shape as ``group_filepaths``. When
            supplied, these masks are authoritative for motif analysis and raw
            intensities (including shadows) are never thresholded into the
            pattern. Nonzero/high labels must represent lattice material.
        slice_axis: Axis along which cross sections are ordered (0, 1, or 2).
        threshold: Optional shared raw-intensity segmentation threshold. When
            omitted, binary groups use their label midpoint and raw groups use
            one shared Otsu threshold.
        foreground: ``"high"`` when material is above the threshold or
            ``"low"`` when it is below the threshold.
        grid_rows: Number of local pattern boxes from top to bottom. Defaults
            to 6 for the current volume.
        grid_columns: Number of local pattern boxes from left to right.
            Defaults to 6 for the current volume.
        analysis_size: Maximum downsampled image dimension used for matching.
        patch_size: Width and height of each normalized local motif descriptor.
        expected_vote: Fraction of valid grid boxes that must contain material
            for a pixel to belong to the dominant expected motif.
        missing_fraction_threshold: Minimum fraction of expected patch material
            that must be absent before the patch is anomalous.
        similarity_threshold: Maximum Dice similarity for an anomalous patch.
        min_expected_patch_fraction: Ignore grid patches containing less
            expected material than this fraction of their area.
        persistence_fraction: Minimum fraction of the group's slice depth that
            an anomaly track must span to be classified as missing.
        spatial_tolerance_pixels: Dilation tolerance in downsampled pixels so
            small strut shifts are not treated as missing.
        max_registration_pixels: Maximum translation used to center a local
            patch before comparing it with the dominant motif.
        track_radius_cells: Maximum grid-cell centroid movement between
            neighboring cross sections for the same anomaly track.
        max_slice_gap: Number of missing detections tolerated inside one 3-D
            strut track. When omitted, use 10% of the group's depth so a track
            can bridge the short square/node phase without being emitted twice.
        max_edge_touch_fraction: Persistent tracks touching the outer scan-grid
            cells more often than this are reported as scan-edge inconclusive,
            not classified as missing.
        box_mode: ``"fixed"`` uses the validated non-overlapping grid.
            ``"dynamic"`` learns a half-cell phase from the recurring pattern
            in the whole group, which can expose seam patterns but should be
            treated as a higher-sensitivity review pass.
        box_overlap: Fractional overlap between dynamic boxes. The default 0.5
            provides a half-cell stride.
        nms_iou_threshold: Maximum box intersection-over-union allowed before
            a weaker response is treated as a duplicate at the same location.
        roi_presence_fraction: Minimum all-group slice frequency used to define
            the common foreground region before gridding.
        output_filepath: Optional .json path for the complete report.

    Returns:
        A structured screening report with shared preprocessing settings,
        per-group candidate counts, and only those missing-pattern tracks that
        satisfy the requested cross-section persistence. Slice indices and grid
        cells are reported as zero-based values plus one-based slice numbers.
    """
    if not group_filepaths:
        raise ValueError("group_filepaths must contain at least one volume")
    if segmentation_filepaths is not None:
        if len(segmentation_filepaths) != len(group_filepaths):
            raise ValueError(
                "segmentation_filepaths must match group_filepaths in length"
            )
        if threshold is not None:
            raise ValueError(
                "threshold cannot be combined with explicit segmentation masks"
            )
    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, or 2")
    if foreground not in {"high", "low"}:
        raise ValueError("foreground must be 'high' or 'low'")
    if grid_rows < 1 or grid_columns < 1:
        raise ValueError("grid_rows and grid_columns must be positive")
    if analysis_size < 4 * max(grid_rows, grid_columns):
        raise ValueError(
            "analysis_size must provide at least four pixels per grid cell"
        )
    if patch_size < 8:
        raise ValueError("patch_size must be at least 8 pixels")
    if not 0 < expected_vote <= 1:
        raise ValueError("expected_vote must be in (0, 1]")
    if not 0 < missing_fraction_threshold <= 1:
        raise ValueError("missing_fraction_threshold must be in (0, 1]")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in [0, 1]")
    if not 0 <= min_expected_patch_fraction < 1:
        raise ValueError("min_expected_patch_fraction must be in [0, 1)")
    if not 0 < persistence_fraction <= 1:
        raise ValueError("persistence_fraction must be in (0, 1]")
    if spatial_tolerance_pixels < 0 or max_registration_pixels < 0:
        raise ValueError("spatial and registration tolerances must be nonnegative")
    if track_radius_cells < 0:
        raise ValueError("track_radius_cells must be nonnegative")
    if max_slice_gap is not None and max_slice_gap < 0:
        raise ValueError("max_slice_gap must be nonnegative")
    if not 0 <= max_edge_touch_fraction <= 1:
        raise ValueError("max_edge_touch_fraction must be in [0, 1]")
    if box_mode not in {"dynamic", "fixed"}:
        raise ValueError("box_mode must be 'dynamic' or 'fixed'")
    if not 0 <= box_overlap < 1:
        raise ValueError("box_overlap must be in [0, 1)")
    if not 0 <= nms_iou_threshold <= 1:
        raise ValueError("nms_iou_threshold must be in [0, 1]")
    if not 0 <= roi_presence_fraction < 1:
        raise ValueError("roi_presence_fraction must be in [0, 1)")

    loaded = [_load_volume(filepath) for filepath in group_filepaths]
    group_paths = [path for path, _ in loaded]
    if len(set(group_paths)) != len(group_paths):
        raise ValueError("group_filepaths must refer to distinct volumes")
    volumes = [volume for _, volume in loaded]
    slice_groups = [np.moveaxis(volume, slice_axis, 0) for volume in volumes]
    if any(len(slices) < 2 for slices in slice_groups):
        raise ValueError("Every slice group must contain at least two slices")

    plane_shapes = {tuple(slices.shape[1:]) for slices in slice_groups}
    if len(plane_shapes) != 1:
        raise ValueError(
            "Every group must have the same in-plane image shape after "
            "applying slice_axis"
        )
    original_plane_shape = tuple(int(value) for value in slice_groups[0].shape[1:])
    scale = min(1.0, analysis_size / max(original_plane_shape))
    analysis_shape = tuple(
        max(1, int(round(value * scale))) for value in original_plane_shape
    )
    if (
        analysis_shape[0] < 4 * grid_rows
        or analysis_shape[1] < 4 * grid_columns
    ):
        raise ValueError(
            "The downsampled plane is too small for the requested grid; "
            "reduce the grid dimensions or increase analysis_size"
        )

    segmentation_paths: list[Path] | None = None
    if segmentation_filepaths is not None:
        loaded_masks = [
            _load_volume(filepath) for filepath in segmentation_filepaths
        ]
        segmentation_paths = [path for path, _ in loaded_masks]
        mask_volumes = [volume for _, volume in loaded_masks]
        for group_index, (raw, mask) in enumerate(
            zip(volumes, mask_volumes), start=1
        ):
            if raw.shape != mask.shape:
                raise ValueError(
                    f"Segmentation group {group_index} shape {mask.shape} "
                    f"does not match raw group shape {raw.shape}"
                )
            sample = _sample_finite_values(mask)
            if not sample.size or np.unique(sample).size > 2:
                raise ValueError(
                    f"Segmentation group {group_index} must contain at most "
                    "two finite labels"
                )
        resolved_threshold, _ = _resolve_shared_threshold(mask_volumes, None)
        mask_slice_groups = [
            np.moveaxis(volume, slice_axis, 0) for volume in mask_volumes
        ]
        group_masks = [
            _downsample_group_masks(
                slices,
                resolved_threshold,
                "high",
                analysis_shape,
            )
            for slices in mask_slice_groups
        ]
        threshold_source = "provided_segmentation_masks"
    else:
        resolved_threshold, threshold_source = _resolve_shared_threshold(
            volumes, threshold
        )
        group_masks = [
            _downsample_group_masks(
                slices,
                resolved_threshold,
                foreground,
                analysis_shape,
            )
            for slices in slice_groups
        ]
    roi = _foreground_roi(group_masks, roi_presence_fraction)
    if roi[1] - roi[0] < grid_rows or roi[3] - roi[2] < grid_columns:
        roi = (0, analysis_shape[0], 0, analysis_shape[1])

    group_results = _analyze_group_masks(
        group_masks,
        group_paths,
        original_plane_shape,
        roi,
        grid_rows,
        grid_columns,
        patch_size,
        expected_vote,
        missing_fraction_threshold,
        similarity_threshold,
        min_expected_patch_fraction,
        persistence_fraction,
        spatial_tolerance_pixels,
        max_registration_pixels,
        track_radius_cells,
        max_slice_gap,
        max_edge_touch_fraction,
        box_mode,
        box_overlap,
        nms_iou_threshold,
    )
    original_roi = group_results[0]["analysis_roi_original_pixels"]
    grid_y_edges = np.linspace(
        original_roi["y_start"],
        original_roi["y_end_exclusive"],
        grid_rows + 1,
        dtype=int,
    )
    grid_x_edges = np.linspace(
        original_roi["x_start"],
        original_roi["x_end_exclusive"],
        grid_columns + 1,
        dtype=int,
    )
    grid_cell_bounds = [
        {
            "row": row,
            "column": column,
            "y_start": int(grid_y_edges[row]),
            "y_end_exclusive": int(grid_y_edges[row + 1]),
            "x_start": int(grid_x_edges[column]),
            "x_end_exclusive": int(grid_x_edges[column + 1]),
        }
        for row in range(grid_rows)
        for column in range(grid_columns)
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "method": (
            "within_slice_dynamic_receptive_field_prototype_tracking"
            if box_mode == "dynamic"
            else "within_slice_fixed_grid_patch_prototype_tracking"
        ),
        "classification_rule": (
            "missing expected material and low patch similarity, tracked across "
            "at least the requested fraction of consecutive cross sections"
        ),
        "group_count": len(group_results),
        "prototype_source": "other_grid_boxes_in_each_cross_section",
        "slice_axis": slice_axis,
        "original_plane_shape": list(original_plane_shape),
        "analysis_plane_shape": list(analysis_shape),
        "threshold_used": resolved_threshold,
        "threshold_source": (
            "user" if threshold is not None else threshold_source
        ),
        "foreground": (
            "high_label_material"
            if segmentation_filepaths is not None
            else foreground
        ),
        "segmentation_files": (
            [str(path) for path in segmentation_paths]
            if segmentation_paths is not None
            else None
        ),
        "grid": {
            "rows": grid_rows,
            "columns": grid_columns,
            "normalized_patch_size": patch_size,
            "box_mode": box_mode,
            "box_overlap": box_overlap if box_mode == "dynamic" else 0.0,
        },
        "grid_cell_bounds_original_pixels": grid_cell_bounds,
        "settings": {
            "expected_vote": expected_vote,
            "missing_fraction_threshold": missing_fraction_threshold,
            "similarity_threshold": similarity_threshold,
            "min_expected_patch_fraction": min_expected_patch_fraction,
            "persistence_fraction": persistence_fraction,
            "spatial_tolerance_pixels_at_analysis_scale": (
                spatial_tolerance_pixels
            ),
            "max_registration_pixels_at_analysis_scale": (
                max_registration_pixels
            ),
            "track_radius_cells": track_radius_cells,
            "max_slice_gap": (
                max_slice_gap
                if max_slice_gap is not None
                else "automatic_10_percent_of_group_depth"
            ),
            "max_edge_touch_fraction": max_edge_touch_fraction,
            "nms_iou_threshold": nms_iou_threshold,
            "roi_presence_fraction": roi_presence_fraction,
        },
        "total_missing_pattern_count": sum(
            group["missing_pattern_count"] for group in group_results
        ),
        "total_edge_inconclusive_pattern_count": sum(
            group["edge_inconclusive_pattern_count"]
            for group in group_results
        ),
        "groups": group_results,
        "interpretation": (
            "Screening evidence only. A missing signal can be caused by a "
            "physical absence or CT acquisition loss. Review persistent grid "
            "tracks against the raw CT volume before calling a strut missing."
        ),
    }

    if output_filepath is not None:
        report_path = Path(output_filepath).expanduser().resolve()
        if report_path.suffix.lower() != ".json":
            raise ValueError("output_filepath must end in .json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        result["report_file"] = str(report_path)

    return result


@mcp.tool()
def analyze_pattern_deformation(
    input_filepath: str,
    output_directory: str,
    group_number: int | None = None,
    slice_axis: int = 0,
    period_slices: int | None = None,
    threshold: float | None = None,
    foreground: str = "high",
    opening_size: int = 3,
    confidence_threshold: float = 0.65,
    min_track_slices: int | None = None,
    inspection_mode: str = "grid",
    grid_size: int = DEFAULT_GRID_SIZE,
    grid_patch_size: int = 48,
    grid_min_reference_fraction: float = 0.02,
    grid_missing_fraction_threshold: float = (
        DEFAULT_MISSING_FRACTION_THRESHOLD
    ),
    grid_max_present_fraction_of_expected: float = 0.55,
    grid_similarity_threshold: float = 0.65,
    grid_spatial_tolerance_pixels: int = 2,
    grid_track_radius_cells: float = DEFAULT_TRACK_RADIUS_CELLS,
    grid_broken_min_track_slices: int = 8,
    grid_broken_confidence_threshold: float = 0.85,
    grid_missing_min_observation_fraction: float = (
        DEFAULT_MISSING_OBSERVATION_FRACTION
    ),
    grid_missing_span_tolerance_fraction: float = 0.15,
    grid_missing_spatial_extent_threshold_cells: float = (
        DEFAULT_MISSING_SPATIAL_EXTENT_CELLS
    ),
    grid_missing_spatial_confirmation_slices: int = 2,
    grid_broken_jump_threshold_cells: float = 1.0,
    grid_roi_presence_fraction: float = 0.01,
    grid_geometry_reference_cycles: int = 2,
    grid_comparison_mode: str = "periodic_segments",
    grid_periodic_vector_count: int = 4,
    grid_periodic_min_prediction_votes: int = 2,
    grid_periodic_min_bidirectional_pairs: int = 2,
    grid_min_missing_segment_area: int = 4,
    grid_min_missing_segment_fraction: float = (
        DEFAULT_MISSING_SEGMENT_FRACTION
    ),
    grid_min_missing_clearance_pixels: float = 3.0,
    grid_min_missing_clearance_fraction: float = (
        DEFAULT_MISSING_CLEARANCE_FRACTION
    ),
    grid_periodic_border_margin_pixels: int = 2,
    grid_periodic_border_margin_fraction: float = (
        DEFAULT_BORDER_MARGIN_FRACTION
    ),
    grid_periodic_max_track_gap_fraction: float = 0.35,
    grid_tilt_correction: bool = True,
    grid_tilt_sample_count: int = 32,
    grid_maximum_tilt_shift_per_period_fraction: float = (
        DEFAULT_MAXIMUM_TILT_SHIFT_FRACTION
    ),
    grid_known_tilt_angle_degrees: float | None = 0.664,
    grid_tilt_angle_tolerance_degrees: float = 0.20,
    grid_slice_spacing_to_pixel_spacing_ratio: float = 1.0,
    grid_forward_track_prediction: bool = True,
    grid_forward_track_history: int = 6,
    grid_forward_track_gap_radius_growth: float = 0.08,
    grid_forward_track_maximum_radius_factor: float = 2.0,
    reference_cycles: int = 2,
    phase_slack: int = 1,
    expected_vote: float = 0.5,
    max_registration_pixels: int = 12,
    missing_tolerance_pixels: int = 2,
    min_missing_area: int = 20,
    track_radius_pixels: float = 15.0,
) -> dict[str, Any]:
    """Run N×N grid-pattern inspection with unlimited parallel tracking.

    Grid inspection is the default. In ``periodic_segments`` mode, the tool
    learns the strongest 2-D translations in every individual cross section.
    Same-phase neighboring copies vote for expected material, so the detector
    handles isolated dots, multi-arm node cross sections, connected diagonal
    struts, alternating motif roles, and struts that cross fixed box edges.
    A contiguous expected-but-absent segment is one candidate. The older
    fixed-cell prototype is available only as ``grid_comparison_mode=
    "fixed_cells"``.

    Every anomalous grid cell is independently tracked across neighboring
    slices. Missing and broken are mutually exclusive outcomes. Spatially and
    temporally matching track fragments are deduplicated before counting, and
    missing takes priority when both temporal rules could apply.
    Set ``inspection_mode="registered"`` only when the older same-phase
    deformation analysis is intentionally desired.

    Persistent absence is a missing strut; only a shorter, high-confidence
    absence or a strong cross-section trajectory jump is a broken strut.
    Outputs include exclusive counts, a uint8 class-label NPY, and a
    color-coded segmented-slice viewer.

    Args:
        input_filepath: Raw CT or segmentation volume (.npy, .tif, or .tiff).
        output_directory: Directory for the NPY, PNG, CSV, and JSON outputs.
        group_number: Optional one-based octet group. Omit for the whole stack.
        slice_axis: Cross-section axis (0, 1, or 2).
        period_slices: Measured octet length. Omit to estimate it from the scan.
        threshold: Raw segmentation threshold. Omit for Otsu/label midpoint.
        foreground: ``"high"`` or ``"low"`` material polarity.
        opening_size: Segmentation opening kernel; use 1 to disable.
        confidence_threshold: Minimum confidence for a confirmed track.
        min_track_slices: Minimum persistent absence span. Omit it to require
            half of the actual target group's slices, rounded up. An explicit
            positive value overrides the automatic half-group threshold.
        inspection_mode: ``"grid"`` (default) or legacy ``"registered"``.
        grid_size: Number of rows and columns in the N×N inspection grid.
            Defaults to 6 for the current volume; spatial defaults are scaled
            from the earlier 8×8 baseline.
        grid_patch_size: Normalized cell size used to learn the common motif.
        grid_min_reference_fraction: Minimum material fraction for a cell to
            help form the expected motif.
        grid_missing_fraction_threshold: Expected-motif fraction that must be
            absent before a grid cell becomes a candidate.
        grid_max_present_fraction_of_expected: Maximum material remaining in a
            candidate. Lower values focus on missing patterns and reject small
            deformations.
        grid_similarity_threshold: Maximum Dice similarity for a missing cell.
        grid_spatial_tolerance_pixels: Normalized-cell tolerance for drift.
        grid_track_radius_cells: Maximum grid-cell movement for one track.
        grid_broken_min_track_slices: Minimum short-lived absence observations
            needed to call a broken strut. The default is eight slices, reduced
            automatically only for groups whose missing threshold is shorter.
        grid_broken_confidence_threshold: Class-specific minimum confidence
            for broken struts. The default 0.85 is stricter than the general
            confidence threshold.
        grid_missing_min_observation_fraction: Fraction of the half-group
            missing threshold that must contain actual detections. The default
            0.60 prevents sparse, short interruptions from being promoted to
            missing only because tracking spans a large phase gap.
        grid_missing_span_tolerance_fraction: Relative tolerance below the
            nominal half-group span. The default 0.15 allows a 39-slice
            nominal threshold to confirm at 34 linked slices.
        grid_missing_spatial_extent_threshold_cells: Normalized spatial length
            of a fully absent strut segment. For the 6×6 volume, a segment
            spanning at least 0.28 grid cells can confirm missing even when
            its cross section is visible in fewer phases.
        grid_missing_spatial_confirmation_slices: Slices that must show the
            full absent spatial segment before the spatial missing rule applies.
        grid_broken_jump_threshold_cells: Abrupt change in corrected trajectory
            or prediction residual that indicates a broken cross section.
        grid_roi_presence_fraction: Slice frequency defining the lattice ROI.
        grid_geometry_reference_cycles: Neighboring octet groups used only to
            anchor the grid boundary. This keeps a completely absent edge motif
            from shrinking and redistributing the target group's grid.
        grid_comparison_mode: ``"periodic_segments"`` (default) preserves
            pattern phase and connectivity; ``"fixed_cells"`` is the legacy
            averaged-cell comparison.
        grid_periodic_vector_count: Strongest undirected lattice translations
            used to predict missing segments. Each is checked in both
            directions.
        grid_periodic_min_prediction_votes: Same-phase periodic neighbors that
            must predict a segment. Three retains edge and partial motifs.
        grid_periodic_min_bidirectional_pairs: Independent lattice directions
            that must predict the candidate from both sides. This rejects
            one-sided material extrapolated beyond a scan or lattice border.
        grid_min_missing_segment_area: Minimum connected expected-but-absent
            area, in original slice pixels.
        grid_min_missing_segment_fraction: Resolution-independent minimum
            missing area as a fraction of one N×N grid cell.
        grid_min_missing_clearance_pixels: Required median empty-space
            clearance from segmented material. Lowering the default to three
            pixels increases sensitivity to smaller missing segments.
        grid_min_missing_clearance_fraction: Resolution-independent clearance
            floor as a fraction of one grid-cell pitch.
        grid_periodic_border_margin_pixels: Suppress predictions that touch the
            outer scan/ROI boundary by this many pixels.
        grid_periodic_border_margin_fraction: Resolution-independent scan-edge
            exclusion as a fraction of one grid-cell pitch.
        grid_periodic_max_track_gap_fraction: Fraction of an octet across which
            the same spatial defect can be linked when it disappears between
            cross-section phases.
        grid_tilt_correction: Estimate depth-varying lateral CT-stack drift
            from same-phase neighboring octets and remove it for tracking.
            Raw coordinates remain unchanged in output overlays.
        grid_tilt_sample_count: Same-phase pairs sampled across the target and
            its neighboring octets for robust piecewise drift estimation.
        grid_maximum_tilt_shift_per_period_fraction: Reject implausible
            registration shifts above this fraction of one grid-cell pitch.
        grid_known_tilt_angle_degrees: Known scan-tilt magnitude. The project
            default is 0.664 degrees; set to null to disable angle-based
            tolerance while retaining measured depth-varying correction.
        grid_tilt_angle_tolerance_degrees: Symmetric uncertainty allowance
            added to the known tilt magnitude. The default is 0.20 degrees.
        grid_slice_spacing_to_pixel_spacing_ratio: Slice spacing divided by
            in-plane pixel spacing, used to convert tilt angle into pixels.
        grid_forward_track_prediction: Follow each flagged pattern forward
            using its recent direction and speed instead of matching only to
            its last observed location.
        grid_forward_track_history: Recent flagged observations used for the
            robust forward-motion estimate.
        grid_forward_track_gap_radius_growth: Increase the assignment gate
            gradually across missing slice phases, using prediction uncertainty
            while retaining a hard maximum.
        grid_forward_track_maximum_radius_factor: Hard cap on the adaptive
            assignment gate as a multiple of ``grid_track_radius_cells``.
        reference_cycles: Same-phase octets sampled in each direction.
        phase_slack: Slice offsets searched around each expected phase.
        expected_vote: Registered-reference vote needed for expected material.
        max_registration_pixels: Maximum same-phase translation.
        missing_tolerance_pixels: Spatial tolerance for normal strut drift.
        min_missing_area: Minimum missing connected-component area.
        track_radius_pixels: Maximum centroid motion for the same 3-D defect.
    """
    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, or 2")
    if foreground not in {"high", "low"}:
        raise ValueError("foreground must be 'high' or 'low'")
    if opening_size < 1:
        raise ValueError("opening_size must be positive")
    if inspection_mode not in {"grid", "registered"}:
        raise ValueError("inspection_mode must be 'grid' or 'registered'")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if grid_patch_size < 8:
        raise ValueError("grid_patch_size must be at least 8")
    if not 0 <= grid_min_reference_fraction < 1:
        raise ValueError("grid_min_reference_fraction must be in [0, 1)")
    if not 0 < grid_missing_fraction_threshold <= 1:
        raise ValueError("grid_missing_fraction_threshold must be in (0, 1]")
    if not 0 <= grid_max_present_fraction_of_expected <= 1:
        raise ValueError(
            "grid_max_present_fraction_of_expected must be in [0, 1]"
        )
    if not 0 <= grid_similarity_threshold <= 1:
        raise ValueError("grid_similarity_threshold must be in [0, 1]")
    if grid_spatial_tolerance_pixels < 0 or grid_track_radius_cells < 0:
        raise ValueError("grid tolerances must be nonnegative")
    if grid_broken_min_track_slices < 2:
        raise ValueError(
            "grid_broken_min_track_slices must be at least 2"
        )
    if not 0 <= grid_broken_confidence_threshold <= 1:
        raise ValueError(
            "grid_broken_confidence_threshold must be in [0, 1]"
        )
    if not 0 < grid_missing_min_observation_fraction <= 1:
        raise ValueError(
            "grid_missing_min_observation_fraction must be in (0, 1]"
        )
    if not 0 <= grid_missing_span_tolerance_fraction < 0.5:
        raise ValueError(
            "grid_missing_span_tolerance_fraction must be in [0, 0.5)"
        )
    if grid_missing_spatial_extent_threshold_cells <= 0:
        raise ValueError(
            "grid_missing_spatial_extent_threshold_cells must be positive"
        )
    if grid_missing_spatial_confirmation_slices < 1:
        raise ValueError(
            "grid_missing_spatial_confirmation_slices must be positive"
        )
    if grid_broken_jump_threshold_cells <= 0:
        raise ValueError(
            "grid_broken_jump_threshold_cells must be positive"
        )
    if not 0 <= grid_roi_presence_fraction < 1:
        raise ValueError("grid_roi_presence_fraction must be in [0, 1)")
    if grid_geometry_reference_cycles < 0:
        raise ValueError("grid_geometry_reference_cycles must be nonnegative")
    if grid_comparison_mode not in {"periodic_segments", "fixed_cells"}:
        raise ValueError(
            "grid_comparison_mode must be 'periodic_segments' or "
            "'fixed_cells'"
        )
    if grid_periodic_vector_count < 2:
        raise ValueError("grid_periodic_vector_count must be at least 2")
    if grid_periodic_min_prediction_votes < 2:
        raise ValueError(
            "grid_periodic_min_prediction_votes must be at least 2"
        )
    if grid_periodic_min_bidirectional_pairs < 1:
        raise ValueError(
            "grid_periodic_min_bidirectional_pairs must be positive"
        )
    if grid_min_missing_segment_area < 1:
        raise ValueError("grid_min_missing_segment_area must be positive")
    if not 0 <= grid_min_missing_segment_fraction <= 1:
        raise ValueError(
            "grid_min_missing_segment_fraction must be in [0, 1]"
        )
    if (
        grid_min_missing_clearance_pixels < 0
        or not 0 <= grid_min_missing_clearance_fraction <= 1
    ):
        raise ValueError("grid missing-clearance thresholds are invalid")
    if grid_periodic_border_margin_pixels < 0:
        raise ValueError(
            "grid_periodic_border_margin_pixels must be nonnegative"
        )
    if not 0 <= grid_periodic_border_margin_fraction <= 1:
        raise ValueError(
            "grid_periodic_border_margin_fraction must be in [0, 1]"
        )
    if not 0 <= grid_periodic_max_track_gap_fraction <= 1:
        raise ValueError(
            "grid_periodic_max_track_gap_fraction must be in [0, 1]"
        )
    if grid_tilt_sample_count < 2:
        raise ValueError("grid_tilt_sample_count must be at least 2")
    if not 0 < grid_maximum_tilt_shift_per_period_fraction <= 1:
        raise ValueError(
            "grid_maximum_tilt_shift_per_period_fraction must be in "
            "(0, 1]"
        )
    if (
        grid_known_tilt_angle_degrees is not None
        and abs(grid_known_tilt_angle_degrees) >= 45
    ):
        raise ValueError(
            "grid_known_tilt_angle_degrees magnitude must be below 45"
        )
    if not 0 <= grid_tilt_angle_tolerance_degrees < 45:
        raise ValueError(
            "grid_tilt_angle_tolerance_degrees must be in [0, 45)"
        )
    if grid_slice_spacing_to_pixel_spacing_ratio <= 0:
        raise ValueError(
            "grid_slice_spacing_to_pixel_spacing_ratio must be positive"
        )
    if grid_forward_track_history < 2:
        raise ValueError(
            "grid_forward_track_history must be at least 2"
        )
    if grid_forward_track_gap_radius_growth < 0:
        raise ValueError(
            "grid_forward_track_gap_radius_growth must be nonnegative"
        )
    if grid_forward_track_maximum_radius_factor < 1:
        raise ValueError(
            "grid_forward_track_maximum_radius_factor must be at least 1"
        )
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if min_track_slices is not None and min_track_slices < 1:
        raise ValueError("min_track_slices must be positive")
    if reference_cycles < 1 or phase_slack < 0:
        raise ValueError("reference_cycles must be positive and phase_slack nonnegative")
    if not 0 < expected_vote <= 1:
        raise ValueError("expected_vote must be in (0, 1]")
    if (
        max_registration_pixels < 0
        or missing_tolerance_pixels < 0
        or min_missing_area < 1
        or track_radius_pixels < 0
    ):
        raise ValueError("registration, tolerance, area, and radius must be valid")

    input_path, volume = _load_volume(input_filepath)
    slices = np.moveaxis(volume, slice_axis, 0)
    resolved_threshold, threshold_input_kind = _resolve_threshold(
        volume,
        threshold,
    )
    period_result: dict[str, Any]
    if period_slices is None:
        descriptors = _build_descriptors(
            slices,
            resolved_threshold,
            foreground,
            descriptor_size=48,
            min_foreground_fraction=0.001,
        )
        period_max = min(150, max(21, len(slices) // 3))
        period_result = _estimate_repeat_period(
            descriptors,
            period_min=20,
            period_max=period_max,
        )
        resolved_period = int(period_result["length_slices"])
    else:
        if not 2 <= period_slices < len(slices):
            raise ValueError(
                "period_slices must be at least 2 and less than the slice count"
            )
        resolved_period = int(period_slices)
        period_result = {
            "length_slices": resolved_period,
            "confidence": None,
            "source": "user",
        }

    try:
        from .pattern_deformation_analysis import (
            resolve_min_track_slices,
            run_grid_pattern_analysis,
            run_pattern_deformation_analysis,
        )
    except ImportError:
        from pattern_deformation_analysis import (
            resolve_min_track_slices,
            run_grid_pattern_analysis,
            run_pattern_deformation_analysis,
        )

    if group_number is None:
        target_group_slice_count = min(resolved_period, len(slices))
    else:
        if group_number < 1:
            raise ValueError("group_number must be one-based and positive")
        target_group_start = (group_number - 1) * resolved_period
        if target_group_start >= len(slices):
            raise ValueError(
                f"group_number {group_number} starts beyond the scan"
            )
        target_group_slice_count = min(
            resolved_period,
            len(slices) - target_group_start,
        )
    resolved_min_track_slices = resolve_min_track_slices(
        min_track_slices,
        resolved_period,
        target_group_slice_count,
    )

    common = {
        "slices": slices,
        "source_path": input_path,
        "output_directory": Path(output_directory).expanduser().resolve(),
        "period_slices": resolved_period,
        "group_number": group_number,
        "threshold": resolved_threshold,
        "foreground": foreground,
        "opening_size": opening_size,
        "confidence_threshold": confidence_threshold,
        "min_track_slices": resolved_min_track_slices,
    }
    if inspection_mode == "grid":
        report = run_grid_pattern_analysis(
            **common,
            grid_rows=grid_size,
            grid_columns=grid_size,
            patch_size=grid_patch_size,
            expected_vote=expected_vote,
            min_reference_fraction=grid_min_reference_fraction,
            missing_fraction_threshold=grid_missing_fraction_threshold,
            max_present_fraction_of_expected=(
                grid_max_present_fraction_of_expected
            ),
            similarity_threshold=grid_similarity_threshold,
            spatial_tolerance_pixels=grid_spatial_tolerance_pixels,
            track_radius_cells=grid_track_radius_cells,
            broken_min_track_slices=grid_broken_min_track_slices,
            broken_confidence_threshold=(
                grid_broken_confidence_threshold
            ),
            missing_min_observation_fraction=(
                grid_missing_min_observation_fraction
            ),
            missing_span_tolerance_fraction=(
                grid_missing_span_tolerance_fraction
            ),
            missing_spatial_extent_threshold_cells=(
                grid_missing_spatial_extent_threshold_cells
            ),
            missing_spatial_confirmation_slices=(
                grid_missing_spatial_confirmation_slices
            ),
            broken_jump_threshold_cells=(
                grid_broken_jump_threshold_cells
            ),
            roi_presence_fraction=grid_roi_presence_fraction,
            geometry_reference_cycles=grid_geometry_reference_cycles,
            comparison_mode=grid_comparison_mode,
            periodic_vector_count=grid_periodic_vector_count,
            periodic_min_prediction_votes=(
                grid_periodic_min_prediction_votes
            ),
            periodic_min_bidirectional_pairs=(
                grid_periodic_min_bidirectional_pairs
            ),
            min_missing_segment_area=grid_min_missing_segment_area,
            min_missing_segment_fraction=(
                grid_min_missing_segment_fraction
            ),
            min_missing_clearance_pixels=(
                grid_min_missing_clearance_pixels
            ),
            min_missing_clearance_fraction=(
                grid_min_missing_clearance_fraction
            ),
            periodic_border_margin_pixels=(
                grid_periodic_border_margin_pixels
            ),
            periodic_border_margin_fraction=(
                grid_periodic_border_margin_fraction
            ),
            periodic_max_track_gap_fraction=(
                grid_periodic_max_track_gap_fraction
            ),
            tilt_correction=grid_tilt_correction,
            tilt_sample_count=grid_tilt_sample_count,
            maximum_tilt_shift_per_period_fraction=(
                grid_maximum_tilt_shift_per_period_fraction
            ),
            known_tilt_angle_degrees=grid_known_tilt_angle_degrees,
            tilt_angle_tolerance_degrees=(
                grid_tilt_angle_tolerance_degrees
            ),
            slice_spacing_to_pixel_spacing_ratio=(
                grid_slice_spacing_to_pixel_spacing_ratio
            ),
            forward_track_prediction=grid_forward_track_prediction,
            forward_track_history=grid_forward_track_history,
            forward_track_gap_radius_growth=(
                grid_forward_track_gap_radius_growth
            ),
            forward_track_maximum_radius_factor=(
                grid_forward_track_maximum_radius_factor
            ),
        )
    else:
        report = run_pattern_deformation_analysis(
            **common,
            reference_cycles=reference_cycles,
            phase_slack=phase_slack,
            expected_vote=expected_vote,
            max_registration=max_registration_pixels,
            missing_tolerance=missing_tolerance_pixels,
            min_missing_area=min_missing_area,
            track_radius_pixels=track_radius_pixels,
        )
    report["inspection_mode"] = inspection_mode
    report["period_estimation"] = period_result
    report["persistence_threshold"] = {
        "requested_min_track_slices": min_track_slices,
        "resolved_min_track_slices": resolved_min_track_slices,
        "target_group_slice_count": target_group_slice_count,
        "source": (
            "user_override"
            if min_track_slices is not None
            else "half_group_rounded_up"
        ),
    }
    if "tracking" in report:
        report["tracking"]["min_track_slices"] = (
            resolved_min_track_slices
        )
    report["segmentation"]["threshold_source"] = (
        "user"
        if threshold is not None
        else (
            "label_midpoint"
            if threshold_input_kind == "segmentation"
            else "otsu"
        )
    )
    report_path = Path(
        report["outputs"]["findings_report_json"]
    )
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
