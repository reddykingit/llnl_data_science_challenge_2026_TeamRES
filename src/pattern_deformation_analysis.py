"""Registered, multi-slice pattern-deformation analysis for lattice CT stacks.

This module restores the workflow used by the original Step 1--5 artifacts:
segment the material, register same-phase slices one octet above and below,
retain every missing-material component, track all components in parallel, and
render the accepted tracks over the segmented CT slices.
"""

from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops
from skimage.registration import phase_cross_correlation


class MaskCache:
    def __init__(
        self,
        slices: np.ndarray,
        threshold: float,
        foreground: str,
        opening_size: int,
        maximum: int = 48,
    ) -> None:
        self.slices = slices
        self.threshold = threshold
        self.foreground = foreground
        self.opening_size = opening_size
        self.maximum = maximum
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, index: int) -> np.ndarray:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        image = np.asarray(self.slices[index])
        finite = np.isfinite(image)
        if self.foreground == "high":
            mask = finite & (image >= self.threshold)
        else:
            mask = finite & (image <= self.threshold)
        if self.opening_size > 1:
            mask = ndimage.binary_opening(
                mask,
                structure=np.ones((self.opening_size,) * 2, dtype=bool),
            )
        self.cache[index] = mask
        if len(self.cache) > self.maximum:
            self.cache.popitem(last=False)
        return mask


def _align_mask(
    moving: np.ndarray,
    reference: np.ndarray,
    max_registration: int,
) -> tuple[np.ndarray, float] | None:
    shift, _, _ = phase_cross_correlation(
        reference.astype(np.float32),
        moving.astype(np.float32),
        upsample_factor=1,
    )
    if np.any(np.abs(shift) > max_registration):
        return None
    aligned = ndimage.shift(
        moving.astype(np.uint8),
        shift=shift,
        order=0,
        mode="constant",
        cval=0,
    ).astype(bool)
    denominator = int(aligned.sum()) + int(reference.sum())
    dice = (
        1.0
        if denominator == 0
        else 2.0 * int((aligned & reference).sum()) / denominator
    )
    return aligned, float(dice)


def registered_missing_components(
    index: int,
    period: int,
    cache: MaskCache,
    reference_cycles: int = 2,
    phase_slack: int = 1,
    expected_vote: float = 0.5,
    max_registration: int = 12,
    missing_tolerance: int = 2,
    min_missing_area: int = 20,
    min_foreground_fraction: float = 0.001,
) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None, list[int]]:
    """Compare one slice with registered same-phase slices above and below."""
    current = cache.get(index)
    if float(current.mean()) < min_foreground_fraction:
        return [], None, None, []

    references: list[np.ndarray] = []
    reference_indices: list[int] = []
    for cycle in range(1, reference_cycles + 1):
        for direction in (-1, 1):
            nominal = index + direction * cycle * period
            best: tuple[float, int, np.ndarray] | None = None
            for delta in range(-phase_slack, phase_slack + 1):
                candidate = nominal + delta
                if not 0 <= candidate < len(cache.slices):
                    continue
                candidate_mask = cache.get(candidate)
                if float(candidate_mask.mean()) < min_foreground_fraction:
                    continue
                aligned = _align_mask(
                    candidate_mask,
                    current,
                    max_registration,
                )
                if aligned is None:
                    continue
                aligned_mask, dice = aligned
                if best is None or dice > best[0]:
                    best = (dice, candidate, aligned_mask)
            if best is not None:
                references.append(best[2])
                reference_indices.append(best[1])

    # Requiring both sides prevents a scan edge from becoming a false defect.
    has_before = any(candidate < index for candidate in reference_indices)
    has_after = any(candidate > index for candidate in reference_indices)
    if not references or not has_before or not has_after:
        return [], None, None, reference_indices

    expected = np.mean(np.stack(references), axis=0) >= expected_vote
    tolerated_current = ndimage.binary_dilation(
        current,
        structure=np.ones((2 * missing_tolerance + 1,) * 2, dtype=bool),
    )
    missing = expected & ~tolerated_current
    missing = ndimage.binary_opening(
        missing,
        structure=np.ones((2, 2), dtype=bool),
    )

    components: list[dict[str, Any]] = []
    for component_id, prop in enumerate(regionprops(label(missing)), start=1):
        if int(prop.area) < min_missing_area:
            continue
        minor = max(float(prop.minor_axis_length), 1.0)
        elongation = float(prop.major_axis_length) / minor
        component_type = (
            "edge_or_strut"
            if elongation >= 2.2 or float(prop.eccentricity) >= 0.85
            else "pattern_region"
        )
        components.append(
            {
                "component_id": component_id,
                "slice_index_zero_based": index,
                "slice_number_one_based": index + 1,
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "area": int(prop.area),
                "bbox": [int(value) for value in prop.bbox],
                "eccentricity": float(prop.eccentricity),
                "elongation": elongation,
                "component_type": component_type,
                "reference_indices_zero_based": reference_indices,
            }
        )
    return components, expected, missing, reference_indices


def missing_components(
    index: int,
    period: int,
    slice_count: int,
    cache: MaskCache,
    flagged: set[int],
    reference_cycles: int,
    phase_slack: int,
    expected_vote: float,
    max_registration: int,
    missing_tolerance: int,
    min_missing_area: int,
    min_foreground_fraction: float,
) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None, list[int]]:
    """Compatibility entry point used by the restored historical renderers."""
    del slice_count, flagged
    return registered_missing_components(
        index=index,
        period=period,
        cache=cache,
        reference_cycles=reference_cycles,
        phase_slack=phase_slack,
        expected_vote=expected_vote,
        max_registration=max_registration,
        missing_tolerance=missing_tolerance,
        min_missing_area=min_missing_area,
        min_foreground_fraction=min_foreground_fraction,
    )


def _predict_track_centroid(
    track: list[dict[str, Any]],
    slice_index: int,
    history: int,
) -> tuple[float, float, float, float]:
    """Predict a flag forward with robust velocity and optional acceleration."""
    prediction = _predict_track_motion(track, slice_index, history)
    return (
        float(prediction["centroid_y"]),
        float(prediction["centroid_x"]),
        float(prediction["velocity_y"]),
        float(prediction["velocity_x"]),
    )


def _predict_track_motion(
    track: list[dict[str, Any]],
    slice_index: int,
    history: int,
) -> dict[str, float | str]:
    """Estimate a robust local trajectory for one independently tracked flag."""
    recent = track[-max(2, history) :]
    last = recent[-1]
    last_y = float(last["centroid_y"])
    last_x = float(last["centroid_x"])
    last_index = int(last["slice_index_zero_based"])
    if len(recent) < 2 or slice_index <= last_index:
        return {
            "centroid_y": last_y,
            "centroid_x": last_x,
            "velocity_y": 0.0,
            "velocity_x": 0.0,
            "acceleration_y": 0.0,
            "acceleration_x": 0.0,
            "uncertainty": 0.0,
            "model": "last_position",
        }
    slopes_y: list[float] = []
    slopes_x: list[float] = []
    for left_position, left in enumerate(recent[:-1]):
        left_index = int(left["slice_index_zero_based"])
        for right in recent[left_position + 1 :]:
            right_index = int(right["slice_index_zero_based"])
            delta = right_index - left_index
            if delta <= 0:
                continue
            slopes_y.append(
                (
                    float(right["centroid_y"])
                    - float(left["centroid_y"])
                )
                / delta
            )
            slopes_x.append(
                (
                    float(right["centroid_x"])
                    - float(left["centroid_x"])
                )
                / delta
            )
    if not slopes_y:
        return {
            "centroid_y": last_y,
            "centroid_x": last_x,
            "velocity_y": 0.0,
            "velocity_x": 0.0,
            "acceleration_y": 0.0,
            "acceleration_x": 0.0,
            "uncertainty": 0.0,
            "model": "last_position",
        }
    velocity_y = float(np.median(slopes_y))
    velocity_x = float(np.median(slopes_x))
    forward = slice_index - last_index
    linear_y = last_y + velocity_y * forward
    linear_x = last_x + velocity_x * forward
    times = np.asarray(
        [
            int(item["slice_index_zero_based"]) - last_index
            for item in recent
        ],
        dtype=float,
    )
    observed_y = np.asarray(
        [float(item["centroid_y"]) for item in recent],
        dtype=float,
    )
    observed_x = np.asarray(
        [float(item["centroid_x"]) for item in recent],
        dtype=float,
    )
    linear_residuals = np.hypot(
        observed_y - (last_y + velocity_y * times),
        observed_x - (last_x + velocity_x * times),
    )
    uncertainty = float(
        1.4826
        * np.median(
            np.abs(linear_residuals - np.median(linear_residuals))
        )
    )
    result: dict[str, float | str] = {
        "centroid_y": linear_y,
        "centroid_x": linear_x,
        "velocity_y": velocity_y,
        "velocity_x": velocity_x,
        "acceleration_y": 0.0,
        "acceleration_x": 0.0,
        "uncertainty": uncertainty,
        "model": "robust_linear_velocity",
    }

    # Four observations are enough to distinguish smooth acceleration from one
    # noisy centroid.  The quadratic is accepted only when it materially
    # improves the in-track fit and predicts a bounded continuation.
    if len(recent) < 4 or len(np.unique(times)) < 4:
        return result
    design = np.column_stack(
        (np.ones_like(times), times, 0.5 * times**2)
    )
    coefficients_y = np.linalg.lstsq(design, observed_y, rcond=None)[0]
    coefficients_x = np.linalg.lstsq(design, observed_x, rcond=None)[0]
    for _ in range(3):
        residual_y = observed_y - design @ coefficients_y
        residual_x = observed_x - design @ coefficients_x
        joint_residual = np.hypot(residual_y, residual_x)
        residual_scale = max(
            1e-4,
            1.4826
            * float(
                np.median(
                    np.abs(
                        joint_residual - np.median(joint_residual)
                    )
                )
            ),
        )
        cutoff = 1.5 * residual_scale
        weights = np.minimum(
            1.0,
            cutoff / np.maximum(joint_residual, 1e-9),
        )
        weighted_design = design * np.sqrt(weights)[:, np.newaxis]
        coefficients_y = np.linalg.lstsq(
            weighted_design,
            observed_y * np.sqrt(weights),
            rcond=None,
        )[0]
        coefficients_x = np.linalg.lstsq(
            weighted_design,
            observed_x * np.sqrt(weights),
            rcond=None,
        )[0]
    quadratic_y = design @ coefficients_y
    quadratic_x = design @ coefficients_x
    quadratic_residuals = np.hypot(
        observed_y - quadratic_y,
        observed_x - quadratic_x,
    )
    linear_rmse = float(np.sqrt(np.mean(linear_residuals**2)))
    quadratic_rmse = float(np.sqrt(np.mean(quadratic_residuals**2)))
    acceleration_y = float(coefficients_y[2])
    acceleration_x = float(coefficients_x[2])
    acceleration = math.hypot(acceleration_y, acceleration_x)
    quadratic_prediction_y = float(
        coefficients_y[0]
        + coefficients_y[1] * forward
        + 0.5 * acceleration_y * forward**2
    )
    quadratic_prediction_x = float(
        coefficients_x[0]
        + coefficients_x[1] * forward
        + 0.5 * acceleration_x * forward**2
    )
    prediction_bend = math.hypot(
        quadratic_prediction_y - linear_y,
        quadratic_prediction_x - linear_x,
    )
    improves_fit = (
        quadratic_rmse <= 0.65 * max(linear_rmse, 1e-6)
        and linear_rmse >= 0.02
    )
    bounded = acceleration <= 0.20 and prediction_bend <= 2.5
    if not improves_fit or not bounded:
        return result
    quadratic_uncertainty = float(
        1.4826
        * np.median(
            np.abs(
                quadratic_residuals - np.median(quadratic_residuals)
            )
        )
    )
    return {
        "centroid_y": quadratic_prediction_y,
        "centroid_x": quadratic_prediction_x,
        "velocity_y": float(
            coefficients_y[1] + acceleration_y * forward
        ),
        "velocity_x": float(
            coefficients_x[1] + acceleration_x * forward
        ),
        "acceleration_y": acceleration_y,
        "acceleration_x": acceleration_x,
        "uncertainty": quadratic_uncertainty,
        "model": "robust_constant_acceleration",
    }


def _track_all(
    observations: dict[int, list[dict[str, Any]]],
    radius: float,
    maximum_gap: int = 2,
    forward_prediction: bool = False,
    prediction_history: int = 6,
    gap_radius_growth: float = 0.08,
    maximum_radius_factor: float = 2.0,
    tilt_tolerance_per_slice_cells: float = 0.0,
) -> list[list[dict[str, Any]]]:
    """Track every component with motion-aware, uncertainty-gated assignment."""
    tracks: list[list[dict[str, Any]]] = []
    for slice_index in sorted(observations):
        components = observations[slice_index]
        active = [
            track_index
            for track_index, track in enumerate(tracks)
            if 0
            < slice_index - int(track[-1]["slice_index_zero_based"])
            <= maximum_gap
        ]
        assigned_components: set[int] = set()
        if active and components:
            costs = np.full(
                (len(active), len(components)),
                radius * maximum_radius_factor + 1.0,
            )
            distances = np.full_like(costs, np.inf)
            gates = np.full(len(active), radius, dtype=float)
            predictions: dict[int, dict[str, float | str]] = {}
            for row, track_index in enumerate(active):
                previous = tracks[track_index][-1]
                if forward_prediction:
                    prediction = _predict_track_motion(
                        tracks[track_index],
                        slice_index,
                        prediction_history,
                    )
                else:
                    prediction = {
                        "centroid_y": float(previous["centroid_y"]),
                        "centroid_x": float(previous["centroid_x"]),
                        "velocity_y": 0.0,
                        "velocity_x": 0.0,
                        "acceleration_y": 0.0,
                        "acceleration_x": 0.0,
                        "uncertainty": 0.0,
                        "model": "last_position",
                    }
                predictions[row] = prediction
                gap = (
                    slice_index
                    - int(previous["slice_index_zero_based"])
                )
                gates[row] = min(
                    radius * maximum_radius_factor,
                    radius
                    * (1.0 + gap_radius_growth * max(0, gap - 1))
                    + tilt_tolerance_per_slice_cells * gap
                    + min(
                        0.5 * radius,
                        2.5 * float(prediction["uncertainty"]),
                    ),
                )
                for column, component in enumerate(components):
                    distance = math.hypot(
                        float(prediction["centroid_y"])
                        - float(component["centroid_y"]),
                        float(prediction["centroid_x"])
                        - float(component["centroid_x"]),
                    )
                    distances[row, column] = distance
                    previous_instances = max(
                        1,
                        int(previous.get("pattern_instance_count", 1)),
                    )
                    current_instances = max(
                        1,
                        int(component.get("pattern_instance_count", 1)),
                    )
                    multiplicity_penalty = 0.10 * abs(
                        current_instances - previous_instances
                    )
                    costs[row, column] = (
                        distance + multiplicity_penalty
                        if distance <= gates[row]
                        else 1_000_000.0 + distance
                    )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if float(distances[row, column]) <= float(gates[row]):
                    component = components[int(column)]
                    prediction = predictions[int(row)]
                    component["tracking_prediction"] = {
                        "predicted_centroid_y": float(
                            prediction["centroid_y"]
                        ),
                        "predicted_centroid_x": float(
                            prediction["centroid_x"]
                        ),
                        "velocity_y_cells_per_slice": float(
                            prediction["velocity_y"]
                        ),
                        "velocity_x_cells_per_slice": float(
                            prediction["velocity_x"]
                        ),
                        "acceleration_y_cells_per_slice_squared": float(
                            prediction["acceleration_y"]
                        ),
                        "acceleration_x_cells_per_slice_squared": float(
                            prediction["acceleration_x"]
                        ),
                        "prediction_uncertainty_cells": float(
                            prediction["uncertainty"]
                        ),
                        "prediction_model": str(prediction["model"]),
                        "assignment_gate_cells": float(gates[row]),
                        "tilt_tolerance_cells": float(
                            tilt_tolerance_per_slice_cells * gap
                        ),
                        "residual_cells": float(
                            distances[row, column]
                        ),
                    }
                    tracks[active[int(row)]].append(component)
                    assigned_components.add(int(column))
        for component_index, component in enumerate(components):
            if component_index not in assigned_components:
                tracks.append([component])
    return tracks


def _longest_run(indices: list[int], maximum_gap: int = 2) -> int:
    if not indices:
        return 0
    longest = current = 1
    for left, right in zip(indices, indices[1:]):
        current = current + 1 if right - left <= maximum_gap else 1
        longest = max(longest, current)
    return longest


def _longest_linked_segment(
    indices: list[int],
    maximum_gap: int,
) -> tuple[int, int]:
    """Return temporal span and observation count for the longest linked run."""
    if not indices:
        return 0, 0
    ordered = sorted(set(indices))
    best_span = best_count = 1
    start = previous = ordered[0]
    count = 1
    for index in ordered[1:]:
        if index - previous <= maximum_gap:
            previous = index
            count += 1
        else:
            span = previous - start + 1
            if (span, count) > (best_span, best_count):
                best_span, best_count = span, count
            start = previous = index
            count = 1
    span = previous - start + 1
    if (span, count) > (best_span, best_count):
        best_span, best_count = span, count
    return int(best_span), int(best_count)


def resolve_min_track_slices(
    requested: int | None,
    period_slices: int,
    group_slice_count: int,
) -> int:
    """Resolve persistence to an override or half the actual target group."""
    if requested is not None:
        if requested < 1:
            raise ValueError("min_track_slices must be positive")
        return int(requested)
    if period_slices < 1 or group_slice_count < 1:
        raise ValueError(
            "period_slices and group_slice_count must be positive"
        )
    actual_group_slices = min(period_slices, group_slice_count)
    return max(1, int(math.ceil(actual_group_slices / 2.0)))


def _summarize_tracks(
    tracks: list[list[dict[str, Any]]],
    min_track_slices: int,
    confidence_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []
    for track_id, track in enumerate(tracks, start=1):
        indices = sorted(
            {int(item["slice_index_zero_based"]) for item in track}
        )
        run = _longest_run(indices)
        strut_fraction = float(
            np.mean(
                [
                    item["component_type"] == "edge_or_strut"
                    for item in track
                ]
            )
        )
        reference_support = min(
            1.0,
            float(
                np.median(
                    [
                        len(item["reference_indices_zero_based"])
                        for item in track
                    ]
                )
            )
            / 4.0,
        )
        persistence = min(1.0, run / max(1, min_track_slices))
        confidence = float(
            min(
                1.0,
                0.50 * persistence
                + 0.30 * strut_fraction
                + 0.20 * reference_support,
            )
        )
        summary = {
            "track_id": track_id,
            "slice_indices_zero_based": indices,
            "slice_numbers_one_based": [index + 1 for index in indices],
            "longest_persistent_span_slices": run,
            "observation_count": len(track),
            "median_area": float(np.median([item["area"] for item in track])),
            "median_elongation": float(
                np.median([item["elongation"] for item in track])
            ),
            "strut_observation_fraction": strut_fraction,
            "confidence": confidence,
            "observations": track,
        }
        summaries.append(summary)
        if (
            run >= min_track_slices
            and strut_fraction >= 0.5
            and confidence >= confidence_threshold
        ):
            confirmed.append(summary)
    return summaries, confirmed


def _normalize_segmented(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask] = np.uint8(220)
    return rgb


def _render_method_steps(
    output_directory: Path,
    scope_name: str,
    dataset_index: int,
    raw: np.ndarray,
    segmented: np.ndarray,
    expected: np.ndarray,
    candidates: np.ndarray,
    retained: np.ndarray,
    viewer_rgb: np.ndarray,
    method_name: str = "Registered pattern-deformation pipeline",
    expected_title: str = "Registered expectation",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_directory = output_directory / "method_steps"
    method_directory.mkdir(exist_ok=True)
    raw_numeric = np.asarray(raw, dtype=np.float32)
    low, high = np.percentile(raw_numeric, (1.0, 99.0))
    raw_gray = (
        np.zeros(raw.shape, dtype=np.float32)
        if high <= low
        else np.clip(
            (raw_numeric - low) / (high - low),
            0.0,
            1.0,
        )
    )
    candidate_rgb = np.repeat(raw_gray[..., None], 3, axis=2)
    candidate_rgb[candidates] = (
        0.25 * candidate_rgb[candidates]
        + 0.75 * np.array([1.0, 0.55, 0.0])
    )
    candidate_rgb[retained] = (
        0.10 * candidate_rgb[retained]
        + 0.90 * np.array([1.0, 0.0, 0.0])
    )
    visible_pattern_count = _count_pattern_instances(retained)
    panels = [
        (raw_gray, "1. Raw CT slice", "gray"),
        (segmented, "2. Segmented material", "gray"),
        (expected, f"3. {expected_title}", "gray"),
        (candidate_rgb, "4. All missing candidates", None),
        (
            viewer_rgb,
            (
                "5. Parallel tracked result "
                f"({visible_pattern_count} visible pattern"
                f"{'' if visible_pattern_count == 1 else 's'})"
            ),
            None,
        ),
    ]
    for position, (image, title, cmap) in enumerate(panels, start=1):
        fig, axis = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
        axis.imshow(image, cmap=cmap)
        axis.set_title(
            f"Step {position} - {title.split('. ', 1)[-1]}\n"
            f"dataset slice {dataset_index + 1}",
        )
        axis.axis("off")
        fig.savefig(
            method_directory / f"method_step_{position}.png",
            dpi=170,
            facecolor="white",
        )
        plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), constrained_layout=True)
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(
        f"{method_name} - {scope_name}, "
        f"dataset slice {dataset_index + 1}",
        fontsize=14,
    )
    overview = method_directory / "method_pipeline_overview.png"
    fig.savefig(overview, dpi=170, facecolor="white")
    plt.close(fig)
    return overview


def run_pattern_deformation_analysis(
    *,
    slices: np.ndarray,
    source_path: Path,
    output_directory: Path,
    period_slices: int,
    group_number: int | None,
    threshold: float,
    foreground: str,
    opening_size: int,
    confidence_threshold: float,
    min_track_slices: int,
    reference_cycles: int,
    phase_slack: int,
    expected_vote: float,
    max_registration: int,
    missing_tolerance: int,
    min_missing_area: int,
    track_radius_pixels: float,
) -> dict[str, Any]:
    if group_number is None:
        scope_start, scope_end = 0, len(slices)
        scope_name = "full_scan"
    else:
        if group_number < 1:
            raise ValueError("group_number must be one-based and positive")
        scope_start = (group_number - 1) * period_slices
        scope_end = min(len(slices), scope_start + period_slices)
        if scope_start >= len(slices):
            raise ValueError(
                f"group_number {group_number} starts beyond the scan"
            )
        scope_name = f"group_{group_number:03d}"

    output_directory.mkdir(parents=True, exist_ok=True)
    cache = MaskCache(
        slices,
        threshold,
        foreground,
        opening_size,
    )
    observations: dict[int, list[dict[str, Any]]] = {}
    step4_by_slice: dict[int, np.ndarray] = {}
    expected_by_slice: dict[int, np.ndarray] = {}
    for index in range(scope_start, scope_end):
        components, expected, missing, _ = registered_missing_components(
            index=index,
            period=period_slices,
            cache=cache,
            reference_cycles=reference_cycles,
            phase_slack=phase_slack,
            expected_vote=expected_vote,
            max_registration=max_registration,
            missing_tolerance=missing_tolerance,
            min_missing_area=min_missing_area,
        )
        observations[index] = components
        if expected is not None:
            expected_by_slice[index] = expected
        if missing is not None:
            step4_by_slice[index] = missing

    tracks = _track_all(observations, track_radius_pixels)
    track_summaries, confirmed = _summarize_tracks(
        tracks,
        min_track_slices,
        confidence_threshold,
    )
    confirmed_ids = {int(track["track_id"]) for track in confirmed}
    retained_by_slice: dict[int, list[dict[str, Any]]] = {}
    for track in confirmed:
        for observation in track["observations"]:
            retained_by_slice.setdefault(
                int(observation["slice_index_zero_based"]),
                [],
            ).append(observation)

    depth = scope_end - scope_start
    plane_shape = tuple(int(value) for value in slices.shape[1:])
    step4_path = output_directory / f"{scope_name}_step4_candidates.npy"
    step5_path = output_directory / f"{scope_name}_step5_confirmed.npy"
    viewer_path = (
        output_directory
        / f"{scope_name}_step5_on_segmented_slice_viewer_rgb.npy"
    )
    step4_volume = np.lib.format.open_memmap(
        step4_path,
        mode="w+",
        dtype=np.bool_,
        shape=(depth,) + plane_shape,
    )
    step5_volume = np.lib.format.open_memmap(
        step5_path,
        mode="w+",
        dtype=np.bool_,
        shape=(depth,) + plane_shape,
    )
    viewer = np.lib.format.open_memmap(
        viewer_path,
        mode="w+",
        dtype=np.uint8,
        shape=(depth,) + plane_shape + (3,),
    )

    csv_rows: list[dict[str, Any]] = []
    method_example: dict[str, Any] | None = None
    method_example_area = -1
    for local_index, index in enumerate(range(scope_start, scope_end)):
        step4 = step4_by_slice.get(
            index,
            np.zeros(plane_shape, dtype=bool),
        )
        retained = np.zeros(plane_shape, dtype=bool)
        labels = label(step4)
        for observation in retained_by_slice.get(index, []):
            retained |= labels == int(observation["component_id"])
            csv_rows.append(
                {
                    "dataset_index_zero_based": index,
                    "dataset_slice_one_based": index + 1,
                    "track_id": next(
                        int(track["track_id"])
                        for track in confirmed
                        if observation in track["observations"]
                    ),
                    "centroid_y": observation["centroid_y"],
                    "centroid_x": observation["centroid_x"],
                    "area": observation["area"],
                    "elongation": observation["elongation"],
                }
            )
        rgb = _normalize_segmented(cache.get(index))
        rgb[retained] = np.array([255, 13, 13], dtype=np.uint8)
        outline = ndimage.binary_dilation(retained, iterations=2) & ~retained
        rgb[outline] = np.array([255, 255, 255], dtype=np.uint8)
        step4_volume[local_index] = step4
        step5_volume[local_index] = retained
        viewer[local_index] = rgb
        retained_area = int(retained.sum())
        candidate_area = int(step4.sum())
        ranking_area = retained_area if retained_area else candidate_area
        if ranking_area > method_example_area:
            method_example_area = ranking_area
            method_example = {
                "dataset_index": index,
                "raw": np.asarray(slices[index]).copy(),
                "segmented": cache.get(index).copy(),
                "expected": expected_by_slice.get(
                    index,
                    np.zeros(plane_shape, dtype=bool),
                ).copy(),
                "candidates": step4.copy(),
                "retained": retained.copy(),
                "viewer_rgb": rgb.copy(),
            }
    step4_volume.flush()
    step5_volume.flush()
    viewer.flush()
    method_overview: Path | None = None
    if method_example is not None:
        method_overview = _render_method_steps(
            output_directory=output_directory,
            scope_name=scope_name,
            dataset_index=int(method_example["dataset_index"]),
            raw=np.asarray(method_example["raw"]),
            segmented=np.asarray(method_example["segmented"]),
            expected=np.asarray(method_example["expected"]),
            candidates=np.asarray(method_example["candidates"]),
            retained=np.asarray(method_example["retained"]),
            viewer_rgb=np.asarray(method_example["viewer_rgb"]),
        )

    csv_path = output_directory / f"{scope_name}_confirmed_tracks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset_index_zero_based",
            "dataset_slice_one_based",
            "track_id",
            "centroid_y",
            "centroid_x",
            "area",
            "elongation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    report = {
        "method": (
            "registered_same_phase_pattern_deformation_with_parallel_tracking"
        ),
        "source_file": str(source_path),
        "group_number": group_number,
        "period_slices": period_slices,
        "scope": {
            "start_index_zero_based": scope_start,
            "end_index_zero_based_exclusive": scope_end,
            "slice_count": depth,
        },
        "segmentation": {
            "threshold": threshold,
            "foreground": foreground,
            "opening_size": opening_size,
        },
        "tracking": {
            "candidate_track_count": len(track_summaries),
            "confirmed_track_ids": sorted(confirmed_ids),
            "min_track_slices": min_track_slices,
            "track_radius_pixels": track_radius_pixels,
            "parallel_track_limit": None,
        },
        "confirmed_missing_strut_count": len(confirmed),
        "confidence_threshold": confidence_threshold,
        "confirmed_tracks": [
            {key: value for key, value in track.items() if key != "observations"}
            for track in confirmed
        ],
        "outputs": {
            "step4_candidate_mask_npy": str(step4_path),
            "step5_confirmed_mask_npy": str(step5_path),
            "segmented_step5_viewer_npy": str(viewer_path),
            "confirmed_tracks_csv": str(csv_path),
            "method_pipeline_overview_png": (
                str(method_overview) if method_overview is not None else None
            ),
        },
    }
    report_path = output_directory / f"{scope_name}_findings_report.json"
    report["outputs"]["findings_report_json"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _grid_roi(
    masks: list[np.ndarray],
    presence_fraction: float,
    grid_rows: int,
    grid_columns: int,
) -> tuple[int, int, int, int]:
    accumulator = np.zeros(masks[0].shape, dtype=np.uint32)
    for mask in masks:
        accumulator += mask
    projection = accumulator / max(1, len(masks)) >= presence_fraction
    return _padded_grid_roi(projection, grid_rows, grid_columns)


def _padded_grid_roi(
    projection: np.ndarray,
    grid_rows: int,
    grid_columns: int,
) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(projection)
    if coordinates.size == 0:
        height, width = projection.shape
        return 0, height, 0, width
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    # The foreground bounding box runs from the first motif to the last motif,
    # not from the outer edge of the first grid cell to the outer edge of the
    # last. Add roughly half a pitch so edge cells are complete inspection
    # boxes instead of being cropped away.
    pad_y = (
        int(round((y1 - y0) / (2 * (grid_rows - 1))))
        if grid_rows > 1
        else 0
    )
    pad_x = (
        int(round((x1 - x0) / (2 * (grid_columns - 1))))
        if grid_columns > 1
        else 0
    )
    return (
        max(0, int(y0) - pad_y),
        min(projection.shape[0], int(y1) + pad_y),
        max(0, int(x0) - pad_x),
        min(projection.shape[1], int(x1) + pad_x),
    )


def _grid_roi_from_indices(
    cache: MaskCache,
    indices: list[int],
    presence_fraction: float,
    grid_rows: int,
    grid_columns: int,
) -> tuple[int, int, int, int]:
    """Measure grid geometry from target plus intact neighboring octets."""
    unique_indices = sorted(set(indices))
    if not unique_indices:
        height, width = cache.slices.shape[1:]
        return 0, int(height), 0, int(width)
    accumulator = np.zeros(cache.slices.shape[1:], dtype=np.uint32)
    for index in unique_indices:
        accumulator += cache.get(index)
    projection = (
        accumulator / max(1, len(unique_indices)) >= presence_fraction
    )
    return _padded_grid_roi(projection, grid_rows, grid_columns)


def _grid_cell_bounds(
    roi: tuple[int, int, int, int],
    grid_rows: int,
    grid_columns: int,
) -> list[tuple[int, int, int, int, int, int]]:
    y0, y1, x0, x1 = roi
    y_edges = np.linspace(y0, y1, grid_rows + 1, dtype=int)
    x_edges = np.linspace(x0, x1, grid_columns + 1, dtype=int)
    return [
        (
            row,
            column,
            int(y_edges[row]),
            int(y_edges[row + 1]),
            int(x_edges[column]),
            int(x_edges[column + 1]),
        )
        for row in range(grid_rows)
        for column in range(grid_columns)
    ]


def _foreground_envelope_center(mask: np.ndarray) -> tuple[float, float] | None:
    coordinates = np.argwhere(mask)
    if len(coordinates) < 8:
        return None
    low = np.percentile(coordinates, 5.0, axis=0)
    high = np.percentile(coordinates, 95.0, axis=0)
    center = 0.5 * (low + high)
    return float(center[0]), float(center[1])


def tilt_tolerance_from_angle(
    angle_degrees: float | None,
    tolerance_degrees: float,
    slice_count: int,
    slice_spacing_to_pixel_spacing_ratio: float,
    cell_pitch_pixels: float,
) -> dict[str, float | int | None]:
    """Convert a known tilt magnitude into detection and tracking allowances."""
    if angle_degrees is not None and not 0 <= abs(angle_degrees) < 45:
        raise ValueError("known tilt angle magnitude must be below 45 degrees")
    if not 0 <= tolerance_degrees < 45:
        raise ValueError("tilt angle tolerance must be in [0, 45) degrees")
    if slice_count < 1:
        raise ValueError("slice_count must be positive")
    if slice_spacing_to_pixel_spacing_ratio <= 0:
        raise ValueError(
            "slice_spacing_to_pixel_spacing_ratio must be positive"
        )
    if cell_pitch_pixels <= 0:
        raise ValueError("cell_pitch_pixels must be positive")
    if angle_degrees is None:
        maximum_angle = 0.0
    else:
        maximum_angle = abs(float(angle_degrees)) + float(
            tolerance_degrees
        )
    maximum_angle = min(maximum_angle, 44.999)
    drift_pixels_per_slice = float(
        math.tan(math.radians(maximum_angle))
        * slice_spacing_to_pixel_spacing_ratio
    )
    group_drift_pixels = drift_pixels_per_slice * max(0, slice_count - 1)
    return {
        "known_angle_degrees": (
            None if angle_degrees is None else float(angle_degrees)
        ),
        "angle_tolerance_degrees": float(tolerance_degrees),
        "maximum_allowed_angle_degrees": float(maximum_angle),
        "slice_spacing_to_pixel_spacing_ratio": float(
            slice_spacing_to_pixel_spacing_ratio
        ),
        "drift_pixels_per_slice": drift_pixels_per_slice,
        "maximum_group_drift_pixels": float(group_drift_pixels),
        "additional_spatial_tolerance_pixels": int(
            math.ceil(group_drift_pixels)
        ),
        "tracking_tolerance_cells_per_slice": float(
            drift_pixels_per_slice / cell_pitch_pixels
        ),
    }


def _estimate_stack_tilt(
    cache: MaskCache,
    scope_start: int,
    scope_end: int,
    period_slices: int,
    roi: tuple[int, int, int, int],
    grid_rows: int,
    grid_columns: int,
    sample_count: int = 16,
    maximum_shift_per_period_fraction: float = 0.5,
) -> dict[str, Any]:
    """Estimate a depth-varying drift field from same-phase octet pairs."""
    y0, y1, x0, x1 = roi
    cell_height = max(1.0, (y1 - y0) / grid_rows)
    cell_width = max(1.0, (x1 - x0) / grid_columns)
    maximum_shift = maximum_shift_per_period_fraction * min(
        cell_height,
        cell_width,
    )
    neighborhood = 2 * period_slices
    estimation_start = max(0, scope_start - neighborhood)
    estimation_end = min(len(cache.slices), scope_end + neighborhood)
    indices = np.linspace(
        estimation_start,
        max(estimation_start, estimation_end - 1),
        num=min(
            sample_count,
            max(1, estimation_end - estimation_start),
        ),
        dtype=int,
    )
    pairs: set[tuple[int, int]] = set()
    for index in indices:
        if index + period_slices < len(cache.slices):
            pairs.add((int(index), int(index + period_slices)))
        elif index - period_slices >= 0:
            pairs.add((int(index - period_slices), int(index)))

    observed_shifts: list[tuple[float, float]] = []
    pair_diagnostics: list[dict[str, Any]] = []
    for reference_index, moving_index in sorted(pairs):
        reference = cache.get(reference_index)[y0:y1, x0:x1]
        moving = cache.get(moving_index)[y0:y1, x0:x1]
        reference_center = _foreground_envelope_center(reference)
        moving_center = _foreground_envelope_center(moving)
        if reference_center is None or moving_center is None:
            continue
        center_shift = np.asarray(moving_center) - np.asarray(
            reference_center
        )

        scale = min(
            1.0,
            256.0 / max(reference.shape),
        )
        if scale < 1.0:
            reference_registration = ndimage.zoom(
                reference.astype(np.float32),
                zoom=(scale, scale),
                order=1,
            )
            moving_registration = ndimage.zoom(
                moving.astype(np.float32),
                zoom=(scale, scale),
                order=1,
            )
        else:
            reference_registration = reference.astype(np.float32)
            moving_registration = moving.astype(np.float32)
        alignment_shift, _, _ = phase_cross_correlation(
            reference_registration,
            moving_registration,
            upsample_factor=1,
        )
        phase_observed_shift = -np.asarray(alignment_shift) / scale
        phase_center_disagreement = float(
            np.linalg.norm(phase_observed_shift - center_shift)
        )
        if phase_center_disagreement <= max(3.0, 0.25 * maximum_shift):
            observed_shift = phase_observed_shift
            source = "phase_registration"
        else:
            observed_shift = center_shift
            source = "robust_envelope_center"
        magnitude = float(np.linalg.norm(observed_shift))
        if magnitude > maximum_shift:
            continue
        observed_shifts.append(
            (float(observed_shift[0]), float(observed_shift[1]))
        )
        pair_diagnostics.append(
            {
                "reference_index_zero_based": reference_index,
                "moving_index_zero_based": moving_index,
                "midpoint_index_zero_based": float(
                    0.5 * (reference_index + moving_index)
                ),
                "observed_shift_y_pixels": float(observed_shift[0]),
                "observed_shift_x_pixels": float(observed_shift[1]),
                "registration_source": source,
                "phase_center_disagreement_pixels": (
                    phase_center_disagreement
                ),
            }
        )

    if not observed_shifts:
        return {
            "applied": False,
            "slope_y_pixels_per_slice": 0.0,
            "slope_x_pixels_per_slice": 0.0,
            "shift_y_pixels_per_period": 0.0,
            "shift_x_pixels_per_period": 0.0,
            "confidence": 0.0,
            "pair_count": 0,
            "pairs": [],
        }

    shifts = np.asarray(observed_shifts, dtype=float)
    midpoints = np.asarray(
        [
            float(item["midpoint_index_zero_based"])
            for item in pair_diagnostics
        ],
        dtype=float,
    )
    order = np.argsort(midpoints)
    midpoints = midpoints[order]
    shifts = shifts[order]
    median_shift = np.median(shifts, axis=0)
    rates = shifts / period_slices
    smoothed_rates = np.empty_like(rates)
    for position in range(len(rates)):
        start = max(0, position - 2)
        stop = min(len(rates), position + 3)
        local = rates[start:stop]
        local_median = np.median(local, axis=0)
        local_deviations = np.linalg.norm(
            local - local_median,
            axis=1,
        )
        local_mad = float(np.median(local_deviations))
        if (
            np.linalg.norm(rates[position] - local_median)
            > max(0.05, 3.5 * 1.4826 * local_mad)
        ):
            smoothed_rates[position] = local_median
        else:
            weights = np.arange(start, stop, dtype=float)
            weights = 1.0 / (
                1.0 + np.abs(weights - float(position))
            )
            smoothed_rates[position] = np.average(
                local,
                axis=0,
                weights=weights,
            )
    predicted_shifts = smoothed_rates * period_slices
    deviations = np.linalg.norm(shifts - predicted_shifts, axis=1)
    median_deviation = float(np.median(deviations))
    confidence = float(
        max(
            0.0,
            min(
                1.0,
                1.0
                - median_deviation
                / max(1.0, 0.20 * maximum_shift),
            ),
        )
    )
    applied = len(observed_shifts) >= 3 and confidence >= 0.25
    rate_knots = [
        {
            "slice_index_zero_based": float(midpoint),
            "rate_y_pixels_per_slice": float(rate[0]),
            "rate_x_pixels_per_slice": float(rate[1]),
        }
        for midpoint, rate in zip(midpoints, smoothed_rates)
    ]
    return {
        "applied": applied,
        "model": "piecewise_depth_varying_rate",
        "slope_y_pixels_per_slice": float(
            median_shift[0] / period_slices
        ),
        "slope_x_pixels_per_slice": float(
            median_shift[1] / period_slices
        ),
        "shift_y_pixels_per_period": float(median_shift[0]),
        "shift_x_pixels_per_period": float(median_shift[1]),
        "confidence": confidence,
        "pair_count": len(observed_shifts),
        "median_pair_deviation_pixels": median_deviation,
        "estimation_range_zero_based": [
            estimation_start,
            estimation_end,
        ],
        "rate_knots": rate_knots,
        "rate_variation_pixels_per_slice": float(
            np.median(
                np.linalg.norm(
                    smoothed_rates - np.median(smoothed_rates, axis=0),
                    axis=1,
                )
            )
        ),
        "pairs": pair_diagnostics,
    }


def _tilt_offset_at_slice(
    tilt: dict[str, Any],
    reference_index: int,
    slice_index: int,
) -> tuple[float, float]:
    """Integrate the locally estimated drift rate between two slice depths."""
    if slice_index == reference_index:
        return 0.0, 0.0
    knots = tilt.get("rate_knots") or []
    if not knots:
        delta = slice_index - reference_index
        return (
            float(tilt.get("slope_y_pixels_per_slice", 0.0)) * delta,
            float(tilt.get("slope_x_pixels_per_slice", 0.0)) * delta,
        )
    depths = np.asarray(
        [float(item["slice_index_zero_based"]) for item in knots],
        dtype=float,
    )
    rates_y = np.asarray(
        [float(item["rate_y_pixels_per_slice"]) for item in knots],
        dtype=float,
    )
    rates_x = np.asarray(
        [float(item["rate_x_pixels_per_slice"]) for item in knots],
        dtype=float,
    )
    lower = float(min(reference_index, slice_index))
    upper = float(max(reference_index, slice_index))
    interior = depths[(depths > lower) & (depths < upper)]
    integration_depths = np.unique(
        np.concatenate(([lower], interior, [upper]))
    )
    interpolated_y = np.interp(
        integration_depths,
        depths,
        rates_y,
    )
    interpolated_x = np.interp(
        integration_depths,
        depths,
        rates_x,
    )
    sign = 1.0 if slice_index > reference_index else -1.0
    return (
        sign * float(np.trapezoid(interpolated_y, integration_depths)),
        sign * float(np.trapezoid(interpolated_x, integration_depths)),
    )


def _apply_tilt_correction(
    candidates: list[dict[str, Any]],
    slice_index: int,
    reference_index: int,
    tilt: dict[str, Any],
    roi: tuple[int, int, int, int],
    grid_rows: int,
    grid_columns: int,
) -> None:
    """Keep overlays in raw coordinates but track in de-tilted coordinates."""
    if not tilt.get("applied"):
        return
    y0, y1, x0, x1 = roi
    cell_height = max(1.0, (y1 - y0) / grid_rows)
    cell_width = max(1.0, (x1 - x0) / grid_columns)
    drift_y, drift_x = _tilt_offset_at_slice(
        tilt,
        reference_index,
        slice_index,
    )
    for candidate in candidates:
        original = candidate.get("centroid_original_pixels")
        if original is None:
            box = candidate["box_bounds_original_pixels"]
            original_y = 0.5 * (
                float(box["y_start"])
                + float(box["y_end_exclusive"])
            )
            original_x = 0.5 * (
                float(box["x_start"])
                + float(box["x_end_exclusive"])
            )
            candidate["centroid_original_pixels"] = {
                "y": original_y,
                "x": original_x,
            }
        else:
            original_y = float(original["y"])
            original_x = float(original["x"])
        candidate["centroid_y_uncorrected"] = float(
            candidate["centroid_y"]
        )
        candidate["centroid_x_uncorrected"] = float(
            candidate["centroid_x"]
        )
        corrected_y = (original_y - drift_y - y0) / cell_height
        corrected_x = (original_x - drift_x - x0) / cell_width
        candidate["centroid_y"] = float(corrected_y)
        candidate["centroid_x"] = float(corrected_x)
        candidate["tilt_correction_pixels"] = {
            "y": float(drift_y),
            "x": float(drift_x),
        }
        candidate["grid_row"] = min(
            grid_rows - 1,
            max(0, int(corrected_y)),
        )
        candidate["grid_column"] = min(
            grid_columns - 1,
            max(0, int(corrected_x)),
        )


def _canonical_translation(dy: int, dx: int) -> tuple[int, int]:
    """Return one stable orientation for an undirected translation."""
    dy, dx = int(dy), int(dx)
    if dy < 0 or (dy == 0 and dx < 0):
        return -dy, -dx
    return dy, dx


def _estimate_periodic_translation_vectors(
    mask: np.ndarray,
    grid_rows: int,
    grid_columns: int,
    vector_count: int = 4,
    minimum_correlation_fraction: float = 0.15,
    maximum_analysis_dimension: int = 384,
) -> list[tuple[int, int]]:
    """Find the strongest repeated 2-D lattice translations in one slice.

    Unlike a fixed cell prototype, an autocorrelation peak preserves the
    spatial phase of every motif. It therefore represents isolated nodes,
    four-arm cross sections, and connected diagonal struts with the same
    mechanism.
    """
    foreground_pixels = int(mask.sum())
    if foreground_pixels == 0 or vector_count < 1:
        return []
    original_height, original_width = mask.shape
    scale = min(
        1.0,
        maximum_analysis_dimension / max(original_height, original_width),
    )
    if scale < 1.0:
        analysis_mask = ndimage.zoom(
            mask.astype(np.float32),
            zoom=(scale, scale),
            order=1,
        ) >= 0.25
    else:
        analysis_mask = mask
    foreground_pixels = int(analysis_mask.sum())
    if foreground_pixels == 0:
        return []
    height, width = analysis_mask.shape
    transform_shape = (2 * height, 2 * width)
    spectrum = np.fft.fft2(
        analysis_mask.astype(np.float32),
        s=transform_shape,
    )
    correlation = np.fft.fftshift(
        np.fft.ifft2(spectrum * np.conj(spectrum)).real
    )
    center_y, center_x = np.asarray(correlation.shape) // 2
    local_maxima = correlation == ndimage.maximum_filter(
        correlation,
        size=7,
        mode="constant",
        cval=0,
    )
    yy, xx = np.indices(correlation.shape)
    shifts_y = yy - center_y
    shifts_x = xx - center_x
    radii = np.hypot(shifts_y, shifts_x)
    grid_scale = max(2, min(grid_rows, grid_columns))
    minimum_radius = max(
        4.0,
        min(height, width) / (2.0 * grid_scale),
    )
    maximum_radius = 0.65 * min(height, width)
    valid = (
        local_maxima
        & (radii >= minimum_radius)
        & (radii <= maximum_radius)
        & (
            correlation
            >= foreground_pixels * minimum_correlation_fraction
        )
        & (np.abs(shifts_y) < height)
        & (np.abs(shifts_x) < width)
    )
    peak_locations = np.argwhere(valid)
    if peak_locations.size == 0:
        return []
    peak_values = correlation[valid]
    order = np.argsort(peak_values)[::-1]

    analysis_vectors: list[tuple[int, int]] = []
    duplicate_tolerance = 3.0
    for peak_index in order:
        y, x = peak_locations[int(peak_index)]
        candidate = _canonical_translation(
            int(y - center_y),
            int(x - center_x),
        )
        if any(
            math.hypot(
                candidate[0] - existing[0],
                candidate[1] - existing[1],
            )
            <= duplicate_tolerance
            for existing in analysis_vectors
        ):
            continue
        analysis_vectors.append(candidate)
        if len(analysis_vectors) >= vector_count:
            break

    if scale == 1.0:
        return analysis_vectors

    scale_y = height / original_height
    scale_x = width / original_width
    foreground_coordinates = np.argwhere(mask)
    maximum_refinement_samples = 20_000
    if len(foreground_coordinates) > maximum_refinement_samples:
        stride = int(
            math.ceil(
                len(foreground_coordinates)
                / maximum_refinement_samples
            )
        )
        foreground_coordinates = foreground_coordinates[::stride]
    refinement_radius = max(
        2,
        int(math.ceil(1.0 / min(scale_y, scale_x))),
    )
    offset_y, offset_x = np.mgrid[
        -refinement_radius : refinement_radius + 1,
        -refinement_radius : refinement_radius + 1,
    ]
    offsets = np.column_stack((offset_y.ravel(), offset_x.ravel()))
    refined: list[tuple[int, int]] = []
    for scaled_dy, scaled_dx in analysis_vectors:
        initial = np.asarray(
            _canonical_translation(
            int(round(scaled_dy / scale_y)),
            int(round(scaled_dx / scale_x)),
            ),
            dtype=int,
        )
        candidates = initial[None, :] + offsets
        target_y = (
            foreground_coordinates[:, 0, None]
            + candidates[None, :, 0]
        )
        target_x = (
            foreground_coordinates[:, 1, None]
            + candidates[None, :, 1]
        )
        valid_targets = (
            (target_y >= 0)
            & (target_y < original_height)
            & (target_x >= 0)
            & (target_x < original_width)
        )
        clipped_y = np.clip(target_y, 0, original_height - 1)
        clipped_x = np.clip(target_x, 0, original_width - 1)
        scores = (
            mask[clipped_y, clipped_x] & valid_targets
        ).sum(axis=0)
        candidate = _canonical_translation(
            *candidates[int(np.argmax(scores))]
        )
        if not any(
            math.hypot(
                candidate[0] - existing[0],
                candidate[1] - existing[1],
            )
            <= duplicate_tolerance / min(scale_y, scale_x)
            for existing in refined
        ):
            refined.append(candidate)
    return refined


def _translate_binary(
    mask: np.ndarray,
    dy: int,
    dx: int,
) -> np.ndarray:
    """Translate without wraparound so scan edges cannot vote across sides."""
    translated = np.zeros_like(mask, dtype=bool)
    output_y0 = max(0, dy)
    output_y1 = min(mask.shape[0], mask.shape[0] + dy)
    input_y0 = max(0, -dy)
    input_y1 = min(mask.shape[0], mask.shape[0] - dy)
    output_x0 = max(0, dx)
    output_x1 = min(mask.shape[1], mask.shape[1] + dx)
    input_x0 = max(0, -dx)
    input_x1 = min(mask.shape[1], mask.shape[1] - dx)
    if output_y0 < output_y1 and output_x0 < output_x1:
        translated[output_y0:output_y1, output_x0:output_x1] = (
            mask[input_y0:input_y1, input_x0:input_x1]
        )
    return translated


def _count_pattern_instances(mask: np.ndarray) -> int:
    """Count disconnected or touching blob-like missing patterns."""
    labels, count = ndimage.label(
        np.asarray(mask, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0:
        return 0
    areas = np.bincount(labels.ravel())[1:]
    total = 0
    for component_label, area in enumerate(areas, start=1):
        if int(area) < 3:
            continue
        component = labels == component_label
        distance = ndimage.distance_transform_edt(component)
        maximum_distance = float(distance.max())
        if maximum_distance < 1.5:
            total += 1
            continue
        peak_window = max(
            3,
            2 * int(math.floor(maximum_distance * 0.75)) + 1,
        )
        peak_mask = (
            (distance >= 0.70 * maximum_distance)
            & (
                distance
                == ndimage.maximum_filter(
                    distance,
                    size=peak_window,
                    mode="constant",
                )
            )
        )
        _, peak_count = ndimage.label(
            peak_mask,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        total += max(1, int(peak_count))
    return max(1, total)


def _confirmed_pattern_instance_count(
    tracks: list[dict[str, Any]],
) -> int:
    return int(
        sum(
            max(1, int(track.get("pattern_instance_count", 1)))
            for track in tracks
        )
    )


def _missing_segment_spatial_metrics(
    missing_mask: np.ndarray,
    cell_height: float,
    cell_width: float,
) -> dict[str, float | int]:
    """Measure whether an absent component spans a substantial lattice cell."""
    labels, component_count = ndimage.label(
        np.asarray(missing_mask, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    cell_pitch = max(1.0, min(float(cell_height), float(cell_width)))
    maximum_span = 0.0
    maximum_aspect_ratio = 1.0
    maximum_area = 0
    for component_label in range(1, component_count + 1):
        coordinates = np.argwhere(labels == component_label).astype(float)
        if not len(coordinates):
            continue
        maximum_area = max(maximum_area, len(coordinates))
        if len(coordinates) == 1:
            major_span = minor_span = 1.0
        else:
            centered = coordinates - coordinates.mean(axis=0)
            covariance = np.cov(centered.T)
            _, eigenvectors = np.linalg.eigh(covariance)
            major_axis = eigenvectors[:, -1]
            minor_axis = eigenvectors[:, 0]
            major_projection = centered @ major_axis
            minor_projection = centered @ minor_axis
            major_span = float(
                major_projection.max() - major_projection.min() + 1.0
            )
            minor_span = float(
                minor_projection.max() - minor_projection.min() + 1.0
            )
        maximum_span = max(maximum_span, major_span)
        maximum_aspect_ratio = max(
            maximum_aspect_ratio,
            major_span / max(1.0, minor_span),
        )
    return {
        "missing_segment_span_pixels": float(maximum_span),
        "missing_segment_span_fraction_cells": float(
            maximum_span / cell_pitch
        ),
        "missing_segment_aspect_ratio": float(maximum_aspect_ratio),
        "largest_missing_component_area_pixels": int(maximum_area),
    }


def _periodic_slice_candidates(
    mask: np.ndarray,
    slice_index: int,
    roi: tuple[int, int, int, int],
    grid_rows: int,
    grid_columns: int,
    expected_vote: float,
    vector_count: int,
    minimum_prediction_votes: int,
    minimum_bidirectional_pairs: int,
    spatial_tolerance_pixels: int,
    min_missing_segment_area: int,
    min_missing_segment_fraction: float,
    min_missing_clearance_pixels: float,
    min_missing_clearance_fraction: float,
    border_margin_pixels: int,
    border_margin_fraction: float,
) -> tuple[
    list[dict[str, Any]],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Find missing motif segments from repeated translations in one slice."""
    y0, y1, x0, x1 = roi
    working = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    empty = np.zeros(mask.shape, dtype=bool)
    vectors = _estimate_periodic_translation_vectors(
        working,
        grid_rows,
        grid_columns,
        vector_count=vector_count,
    )
    signed_vectors = [
        signed
        for dy, dx in vectors
        for signed in ((dy, dx), (-dy, -dx))
    ]
    diagnostics = {
        "translation_vectors_roi_pixels": [
            {"dy": dy, "dx": dx} for dy, dx in vectors
        ],
        "signed_prediction_count": len(signed_vectors),
        "minimum_prediction_votes": None,
        "minimum_bidirectional_pairs": minimum_bidirectional_pairs,
        "raw_missing_component_count": 0,
        "rejected_by_area": 0,
        "rejected_by_clearance": 0,
        "rejected_by_border": 0,
        "rejected_one_sided_pixel_count": 0,
        "rejected_one_sided_component_count": 0,
        "accepted_candidate_count": 0,
    }
    if len(vectors) < 2:
        return [], {}, empty, empty, diagnostics

    votes = np.zeros(working.shape, dtype=np.uint16)
    bidirectional_votes = np.zeros(working.shape, dtype=np.uint8)
    for dy, dx in vectors:
        positive = _translate_binary(working, dy, dx)
        negative = _translate_binary(working, -dy, -dx)
        votes += positive
        votes += negative
        bidirectional_votes += positive & negative
    minimum_votes = min(
        len(signed_vectors),
        max(
            2,
            (
                minimum_prediction_votes
                if minimum_prediction_votes > 0
                else int(
                    math.ceil(
                        expected_vote * len(signed_vectors)
                    )
                )
            ),
        ),
    )
    diagnostics["minimum_prediction_votes"] = minimum_votes
    required_pairs = min(
        len(vectors),
        max(1, minimum_bidirectional_pairs),
    )
    expected_working = (
        (votes >= minimum_votes)
        & (bidirectional_votes >= required_pairs)
    )
    tolerated_observed = (
        ndimage.binary_dilation(
            working,
            iterations=spatial_tolerance_pixels,
        )
        if spatial_tolerance_pixels
        else working
    )
    missing_working = expected_working & ~tolerated_observed
    one_sided_missing = (
        (votes >= minimum_votes)
        & (bidirectional_votes < required_pairs)
        & ~tolerated_observed
    )
    diagnostics["rejected_one_sided_pixel_count"] = int(
        one_sided_missing.sum()
    )
    _, one_sided_component_count = ndimage.label(
        one_sided_missing,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    diagnostics["rejected_one_sided_component_count"] = int(
        one_sided_component_count
    )
    labels, component_count = ndimage.label(
        missing_working,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    diagnostics["raw_missing_component_count"] = int(component_count)
    distance_from_observed = ndimage.distance_transform_edt(~working)

    expected_canvas = np.zeros(mask.shape, dtype=bool)
    expected_canvas[y0:y1, x0:x1] = expected_working
    candidate_canvas = np.zeros(mask.shape, dtype=bool)
    candidates: list[dict[str, Any]] = []
    candidate_masks: dict[int, np.ndarray] = {}
    cell_height = max(1.0, (y1 - y0) / grid_rows)
    cell_width = max(1.0, (x1 - x0) / grid_columns)
    effective_minimum_area = max(
        min_missing_segment_area,
        int(
            math.ceil(
                min_missing_segment_fraction
                * cell_height
                * cell_width
            )
        ),
    )
    diagnostics["effective_minimum_segment_area_pixels"] = (
        effective_minimum_area
    )
    effective_minimum_clearance = max(
        float(min_missing_clearance_pixels),
        float(
            min_missing_clearance_fraction
            * min(cell_height, cell_width)
        ),
    )
    diagnostics["effective_minimum_clearance_pixels"] = (
        effective_minimum_clearance
    )
    margin = max(
        0,
        border_margin_pixels,
        int(
            math.ceil(
                border_margin_fraction
                * min(cell_height, cell_width)
            )
        ),
    )
    diagnostics["effective_border_margin_pixels"] = margin
    candidate_id = 0
    for component_label in range(1, component_count + 1):
        component = labels == component_label
        area = int(component.sum())
        if area < effective_minimum_area:
            diagnostics["rejected_by_area"] += 1
            continue
        coordinates = np.argwhere(component)
        local_y_min, local_x_min = coordinates.min(axis=0)
        local_y_max, local_x_max = coordinates.max(axis=0) + 1
        if (
            local_y_min < margin
            or local_x_min < margin
            or local_y_max > working.shape[0] - margin
            or local_x_max > working.shape[1] - margin
        ):
            diagnostics["rejected_by_border"] += 1
            continue
        median_clearance = float(
            np.median(distance_from_observed[component])
        )
        if median_clearance < effective_minimum_clearance:
            diagnostics["rejected_by_clearance"] += 1
            continue

        center_y, center_x = ndimage.center_of_mass(component)
        median_bidirectional_support = float(
            np.median(bidirectional_votes[component])
        )
        normalized_y = float(center_y) / cell_height
        normalized_x = float(center_x) / cell_width
        row = min(grid_rows - 1, max(0, int(normalized_y)))
        column = min(grid_columns - 1, max(0, int(normalized_x)))
        full_mask = np.zeros(mask.shape, dtype=bool)
        full_mask[y0:y1, x0:x1] = component
        candidate_canvas |= full_mask
        candidate_id += 1
        candidate_masks[candidate_id] = full_mask
        spatial_metrics = _missing_segment_spatial_metrics(
            component,
            cell_height,
            cell_width,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "slice_index_zero_based": slice_index,
                "slice_number_one_based": slice_index + 1,
                "grid_row": row,
                "grid_column": column,
                "centroid_y": normalized_y,
                "centroid_x": normalized_x,
                "centroid_original_pixels": {
                    "y": float(center_y + y0),
                    "x": float(center_x + x0),
                },
                "missing_fraction": 1.0,
                "present_fraction_of_expected": 0.0,
                "complete_absence": True,
                "patch_similarity": 0.0,
                "expected_foreground_pixels": area,
                "missing_segment_area_pixels": area,
                **spatial_metrics,
                "pattern_instance_count": 1,
                "median_clearance_from_observed_pixels": (
                    median_clearance
                ),
                "median_bidirectional_pair_support": (
                    median_bidirectional_support
                ),
                "detection_method": "periodic_translation_consensus",
                "translation_vectors_roi_pixels": diagnostics[
                    "translation_vectors_roi_pixels"
                ],
                "box_bounds_original_pixels": {
                    "y_start": int(local_y_min + y0),
                    "y_end_exclusive": int(local_y_max + y0),
                    "x_start": int(local_x_min + x0),
                    "x_end_exclusive": int(local_x_max + x0),
                },
            }
        )
    diagnostics["accepted_candidate_count"] = len(candidates)
    return (
        candidates,
        candidate_masks,
        expected_canvas,
        candidate_canvas,
        diagnostics,
    )


def _grid_slice_candidates(
    mask: np.ndarray,
    slice_index: int,
    bounds: list[tuple[int, int, int, int, int, int]],
    patch_size: int,
    expected_vote: float,
    min_reference_fraction: float,
    missing_fraction_threshold: float,
    max_present_fraction_of_expected: float,
    similarity_threshold: float,
    spatial_tolerance_pixels: int,
) -> tuple[
    list[dict[str, Any]],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Compare all cells in one cross section and retain every absent motif."""
    motifs: list[np.ndarray] = []
    windows: list[tuple[int, int, int, int, int, int]] = []
    for row, column, y0, y1, x0, x1 in bounds:
        patch = mask[y0:y1, x0:x1]
        motif = ndimage.zoom(
            patch.astype(np.uint8),
            (
                patch_size / max(1, patch.shape[0]),
                patch_size / max(1, patch.shape[1]),
            ),
            order=0,
        ).astype(bool)
        motif = motif[:patch_size, :patch_size]
        if motif.shape != (patch_size, patch_size):
            padded = np.zeros((patch_size, patch_size), dtype=bool)
            padded[: motif.shape[0], : motif.shape[1]] = motif
            motif = padded
        motifs.append(motif)
        windows.append((row, column, y0, y1, x0, x1))

    references = [
        motif
        for motif in motifs
        if float(motif.mean()) >= min_reference_fraction
    ]
    minimum_references = min(
        len(motifs),
        max(3, int(math.ceil(0.1 * len(motifs)))),
    )
    if len(references) < minimum_references:
        empty = np.zeros(mask.shape, dtype=bool)
        return [], {}, empty, empty

    expected_patch = np.mean(np.stack(references), axis=0) >= expected_vote
    expected_pixels = int(expected_patch.sum())
    if expected_pixels == 0:
        empty = np.zeros(mask.shape, dtype=bool)
        return [], {}, empty, empty

    candidates: list[dict[str, Any]] = []
    candidate_masks: dict[int, np.ndarray] = {}
    expected_canvas = np.zeros(mask.shape, dtype=bool)
    candidate_canvas = np.zeros(mask.shape, dtype=bool)
    candidate_id = 0
    for motif, window in zip(motifs, windows):
        row, column, y0, y1, x0, x1 = window
        tolerated = (
            ndimage.binary_dilation(
                motif,
                iterations=spatial_tolerance_pixels,
            )
            if spatial_tolerance_pixels
            else motif
        )
        shared_pixels = int((expected_patch & tolerated).sum())
        missing_patch = expected_patch & ~tolerated
        missing_fraction = (
            expected_pixels - shared_pixels
        ) / expected_pixels
        present_fraction = int(motif.sum()) / expected_pixels
        denominator = expected_pixels + int(tolerated.sum())
        similarity = (
            2.0 * shared_pixels / denominator if denominator else 1.0
        )
        expected_local = ndimage.zoom(
            expected_patch.astype(np.uint8),
            (
                (y1 - y0) / patch_size,
                (x1 - x0) / patch_size,
            ),
            order=0,
        ).astype(bool)[: y1 - y0, : x1 - x0]
        missing_local = ndimage.zoom(
            missing_patch.astype(np.uint8),
            (
                (y1 - y0) / patch_size,
                (x1 - x0) / patch_size,
            ),
            order=0,
        ).astype(bool)[: y1 - y0, : x1 - x0]
        expected_canvas[y0:y1, x0:x1] |= expected_local
        if (
            missing_fraction < missing_fraction_threshold
            or present_fraction > max_present_fraction_of_expected
            or similarity > similarity_threshold
        ):
            continue

        candidate_id += 1
        full_mask = np.zeros(mask.shape, dtype=bool)
        full_mask[y0:y1, x0:x1] = missing_local
        candidate_canvas |= full_mask
        candidate_masks[candidate_id] = full_mask
        spatial_metrics = _missing_segment_spatial_metrics(
            missing_local,
            y1 - y0,
            x1 - x0,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "slice_index_zero_based": slice_index,
                "slice_number_one_based": slice_index + 1,
                "grid_row": row,
                "grid_column": column,
                "centroid_y": float(row),
                "centroid_x": float(column),
                "missing_fraction": float(missing_fraction),
                "present_fraction_of_expected": float(present_fraction),
                "complete_absence": bool(
                    present_fraction <= 0.05
                ),
                "patch_similarity": float(similarity),
                "expected_foreground_pixels": expected_pixels,
                **spatial_metrics,
                "pattern_instance_count": _count_pattern_instances(
                    missing_local
                ),
                "box_bounds_original_pixels": {
                    "y_start": y0,
                    "y_end_exclusive": y1,
                    "x_start": x0,
                    "x_end_exclusive": x1,
                },
            }
        )
    return candidates, candidate_masks, expected_canvas, candidate_canvas


def _track_jump_score(track: list[dict[str, Any]]) -> float:
    """Measure abrupt trajectory change while ignoring steady tilted motion."""
    ordered = sorted(
        track,
        key=lambda item: int(item["slice_index_zero_based"]),
    )
    velocities: list[tuple[float, float]] = []
    prediction_residuals: list[float] = []
    for item in ordered:
        prediction = item.get("tracking_prediction", {})
        if prediction.get("prediction_model") not in {
            None,
            "last_position",
        }:
            residual = prediction.get("residual_cells")
            if residual is not None:
                prediction_residuals.append(float(residual))
    for left, right in zip(ordered, ordered[1:]):
        gap = (
            int(right["slice_index_zero_based"])
            - int(left["slice_index_zero_based"])
        )
        if gap <= 0:
            continue
        velocities.append(
            (
                (
                    float(right["centroid_y"])
                    - float(left["centroid_y"])
                )
                / gap,
                (
                    float(right["centroid_x"])
                    - float(left["centroid_x"])
                )
                / gap,
            )
        )
    velocity_changes = [
        math.hypot(
            right[0] - left[0],
            right[1] - left[1],
        )
        for left, right in zip(velocities, velocities[1:])
    ]
    return max(
        prediction_residuals + velocity_changes,
        default=0.0,
    )


def _summarize_grid_tracks(
    tracks: list[list[dict[str, Any]]],
    min_track_slices: int,
    confidence_threshold: float,
    maximum_gap: int = 2,
    broken_min_track_slices: int = 8,
    broken_confidence_threshold: float = 0.85,
    missing_min_observation_fraction: float = 0.55,
    missing_span_tolerance_fraction: float = 0.15,
    missing_spatial_extent_threshold_cells: float = 0.35,
    missing_spatial_confirmation_slices: int = 2,
    broken_jump_threshold_cells: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify each tracked absence once as either missing or broken."""
    if broken_min_track_slices < 2:
        raise ValueError("broken_min_track_slices must be at least 2")
    if not 0 <= broken_confidence_threshold <= 1:
        raise ValueError(
            "broken_confidence_threshold must be in [0, 1]"
        )
    if not 0 < missing_min_observation_fraction <= 1:
        raise ValueError(
            "missing_min_observation_fraction must be in (0, 1]"
        )
    if not 0 <= missing_span_tolerance_fraction < 0.5:
        raise ValueError(
            "missing_span_tolerance_fraction must be in [0, 0.5)"
        )
    if missing_spatial_extent_threshold_cells <= 0:
        raise ValueError(
            "missing_spatial_extent_threshold_cells must be positive"
        )
    if missing_spatial_confirmation_slices < 1:
        raise ValueError(
            "missing_spatial_confirmation_slices must be positive"
        )
    if broken_jump_threshold_cells <= 0:
        raise ValueError("broken_jump_threshold_cells must be positive")
    summaries: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []
    for track_id, track in enumerate(tracks, start=1):
        indices = sorted(
            {int(item["slice_index_zero_based"]) for item in track}
        )
        persistent_span, persistent_observation_count = (
            _longest_linked_segment(
                indices,
                maximum_gap=maximum_gap,
            )
        )
        missing_strength = float(
            np.median([item["missing_fraction"] for item in track])
        )
        absence_strength = float(
            1.0
            - np.median(
                [
                    item["present_fraction_of_expected"]
                    for item in track
                ]
            )
        )
        effective_missing_min_span = max(
            2,
            int(
                math.ceil(
                    min_track_slices
                    * (1.0 - missing_span_tolerance_fraction)
                )
            ),
        )
        persistence = min(
            1.0,
            persistent_span / max(1, effective_missing_min_span),
        )
        minimum_missing_observations = max(
            2,
            int(
                math.ceil(
                    effective_missing_min_span
                    * missing_min_observation_fraction
                )
            ),
        )
        missing_observation_support = min(
            1.0,
            persistent_observation_count
            / max(1, minimum_missing_observations),
        )
        confidence = float(
            min(
                1.0,
                0.35 * persistence
                + 0.15 * missing_observation_support
                + 0.30 * missing_strength
                + 0.20 * absence_strength,
            )
        )
        median_present_fraction = float(
            np.median(
                [
                    item["present_fraction_of_expected"]
                    for item in track
                ]
            )
        )
        periodic_segment_fraction = float(
            np.mean(
                [
                    item.get("detection_method")
                    == "periodic_translation_consensus"
                    for item in track
                ]
            )
        )
        pattern_instance_count = max(
            max(
                1,
                int(item.get("pattern_instance_count", 1)),
            )
            for item in track
        )
        median_similarity = float(
            np.median(
                [
                    float(item.get("patch_similarity", 0.0))
                    for item in track
                ]
            )
        )
        jump_score = _track_jump_score(track)
        effective_broken_min_track_slices = min(
            broken_min_track_slices,
            max(2, effective_missing_min_span - 1),
        )
        local_continuous_span, local_continuous_observation_count = (
            _longest_linked_segment(
                indices,
                maximum_gap=min(2, maximum_gap),
            )
        )
        spatial_extent_fractions = [
            float(
                item.get(
                    "missing_segment_span_fraction_cells",
                    0.0,
                )
            )
            for item in track
        ]
        spatially_complete_observation_count = int(
            sum(
                extent >= missing_spatial_extent_threshold_cells
                and item.get("detection_method")
                == "periodic_translation_consensus"
                for item, extent in zip(
                    track,
                    spatial_extent_fractions,
                )
            )
        )
        temporal_missing_confirmed = (
            persistent_span >= effective_missing_min_span
            and persistent_observation_count
            >= minimum_missing_observations
            and local_continuous_observation_count
            >= effective_broken_min_track_slices
        )
        spatial_missing_confirmed = (
            spatially_complete_observation_count
            >= missing_spatial_confirmation_slices
            and persistent_observation_count
            >= effective_broken_min_track_slices
        )
        missing_confirmed = (
            temporal_missing_confirmed or spatial_missing_confirmed
        )
        jump_broken = (
            jump_score >= broken_jump_threshold_cells
            and persistent_observation_count >= 3
        )
        transient_broken = (
            persistent_span
            >= effective_broken_min_track_slices
            and persistent_observation_count
            >= effective_broken_min_track_slices
            and not missing_confirmed
        )
        broken_duration = min(
            1.0,
            persistent_observation_count
            / max(1, effective_broken_min_track_slices),
        )
        spatial_missing_support = min(
            1.0,
            spatially_complete_observation_count
            / max(1, missing_spatial_confirmation_slices),
        )
        spatial_missing_confidence = float(
            min(
                1.0,
                0.25 * broken_duration
                + 0.25 * spatial_missing_support
                + 0.25 * missing_strength
                + 0.25 * absence_strength,
            )
        )
        missing_confidence = max(
            confidence,
            (
                spatial_missing_confidence
                if spatial_missing_confirmed
                else 0.0
            ),
        )
        transient_strength = max(
            0.0,
            1.0
            - persistent_span / max(1, effective_missing_min_span),
        )
        jump_strength = min(
            1.0,
            jump_score / broken_jump_threshold_cells,
        )
        broken_confidence = float(
            min(
                1.0,
                0.35 * broken_duration
                + 0.15 * max(transient_strength, jump_strength)
                + 0.25 * missing_strength
                + 0.25 * absence_strength,
            )
        )
        defect_class: str | None = None
        classification_confidence = missing_confidence
        if missing_confirmed:
            defect_class = "missing_strut"
        elif jump_broken:
            defect_class = "broken_strut"
            classification_confidence = broken_confidence
        elif transient_broken:
            defect_class = "broken_strut"
            classification_confidence = broken_confidence
        if classification_confidence < confidence_threshold:
            defect_class = None
        if (
            defect_class == "broken_strut"
            and classification_confidence
            < max(confidence_threshold, broken_confidence_threshold)
        ):
            defect_class = None

        rejection_reasons: list[str] = []
        if defect_class is None:
            if (
                persistent_observation_count
                < effective_broken_min_track_slices
            ):
                rejection_reasons.append(
                    "insufficient_broken_observation_count"
                )
            if persistent_span < effective_broken_min_track_slices:
                rejection_reasons.append(
                    "insufficient_broken_phase_linked_span"
                )
            if classification_confidence < confidence_threshold:
                rejection_reasons.append("below_confidence_threshold")
            if (
                (jump_broken or transient_broken)
                and classification_confidence
                < broken_confidence_threshold
            ):
                rejection_reasons.append(
                    "below_broken_confidence_threshold"
                )
        summary = {
            "track_id": track_id,
            "classification": defect_class or "unclassified_pattern_anomaly",
            "defect_class": defect_class,
            "slice_indices_zero_based": indices,
            "slice_numbers_one_based": [index + 1 for index in indices],
            "start_index_zero_based": min(indices),
            "end_index_zero_based": max(indices),
            "detected_slice_count": len(indices),
            "persistent_span_slices": persistent_span,
            "persistent_observation_count": (
                persistent_observation_count
            ),
            "local_continuous_span_slices": local_continuous_span,
            "local_continuous_observation_count": (
                local_continuous_observation_count
            ),
            "minimum_missing_observations": (
                minimum_missing_observations
            ),
            "requested_missing_min_span_slices": min_track_slices,
            "effective_missing_min_span_slices": (
                effective_missing_min_span
            ),
            "missing_min_observation_fraction": (
                missing_min_observation_fraction
            ),
            "missing_span_tolerance_fraction": (
                missing_span_tolerance_fraction
            ),
            "missing_spatial_extent_threshold_cells": (
                missing_spatial_extent_threshold_cells
            ),
            "missing_spatial_confirmation_slices": (
                missing_spatial_confirmation_slices
            ),
            "spatially_complete_observation_count": (
                spatially_complete_observation_count
            ),
            "maximum_missing_segment_span_fraction_cells": (
                max(spatial_extent_fractions, default=0.0)
            ),
            "median_missing_segment_span_fraction_cells": (
                float(np.median(spatial_extent_fractions))
                if spatial_extent_fractions
                else 0.0
            ),
            "temporal_missing_rule_satisfied": (
                temporal_missing_confirmed
            ),
            "spatial_missing_rule_satisfied": (
                spatial_missing_confirmed
            ),
            "median_missing_fraction": missing_strength,
            "median_present_fraction_of_expected": median_present_fraction,
            "periodic_segment_observation_fraction": (
                periodic_segment_fraction
            ),
            "median_patch_similarity": median_similarity,
            "trajectory_jump_score_cells": jump_score,
            "broken_jump_threshold_cells": broken_jump_threshold_cells,
            "requested_broken_min_track_slices": (
                broken_min_track_slices
            ),
            "effective_broken_min_track_slices": (
                effective_broken_min_track_slices
            ),
            "broken_confidence_threshold": (
                broken_confidence_threshold
            ),
            "pattern_instance_count": pattern_instance_count,
            "complete_absence_observation_fraction": float(
                np.mean([item["complete_absence"] for item in track])
            ),
            "grid_cells_zero_based": sorted(
                {
                    (int(item["grid_row"]), int(item["grid_column"]))
                    for item in track
                }
            ),
            "median_tracking_centroid_y": float(
                np.median([float(item["centroid_y"]) for item in track])
            ),
            "median_tracking_centroid_x": float(
                np.median([float(item["centroid_x"]) for item in track])
            ),
            "confidence": classification_confidence,
            "missing_confidence": missing_confidence,
            "temporal_missing_confidence": confidence,
            "spatial_missing_confidence": spatial_missing_confidence,
            "broken_confidence": broken_confidence,
            "maximum_link_gap_slices": maximum_gap,
            "rejection_reasons": rejection_reasons,
            "observations": track,
        }
        summaries.append(summary)
        if defect_class is not None:
            confirmed.append(summary)
    return summaries, confirmed


def _deduplicate_confirmed_tracks(
    tracks: list[dict[str, Any]],
    maximum_gap: int,
    centroid_tolerance_cells: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Keep one mutually exclusive classification per physical strut."""
    priority = {"missing_strut": 2, "broken_strut": 1}
    ordered = sorted(
        tracks,
        key=lambda track: (
            priority.get(str(track.get("defect_class")), 0),
            float(track.get("confidence", 0.0)),
            int(track.get("persistent_span_slices", 0)),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    duplicate_of: dict[int, int] = {}
    for candidate in ordered:
        candidate_cells = {
            tuple(cell)
            for cell in candidate.get("grid_cells_zero_based", [])
        }
        candidate_start = int(candidate["start_index_zero_based"])
        candidate_end = int(candidate["end_index_zero_based"])
        candidate_y = float(candidate["median_tracking_centroid_y"])
        candidate_x = float(candidate["median_tracking_centroid_x"])
        duplicate_target: dict[str, Any] | None = None
        for existing in kept:
            existing_cells = {
                tuple(cell)
                for cell in existing.get("grid_cells_zero_based", [])
            }
            if candidate_cells and existing_cells:
                if candidate_cells.isdisjoint(existing_cells):
                    continue
            spatial_distance = math.hypot(
                candidate_y
                - float(existing["median_tracking_centroid_y"]),
                candidate_x
                - float(existing["median_tracking_centroid_x"]),
            )
            if spatial_distance > centroid_tolerance_cells:
                continue
            existing_start = int(existing["start_index_zero_based"])
            existing_end = int(existing["end_index_zero_based"])
            temporal_separation = max(
                0,
                candidate_start - existing_end - 1,
                existing_start - candidate_end - 1,
            )
            if temporal_separation > maximum_gap:
                continue
            duplicate_target = existing
            break
        if duplicate_target is None:
            candidate["merged_duplicate_track_ids"] = []
            kept.append(candidate)
            continue
        candidate_id = int(candidate["track_id"])
        target_id = int(duplicate_target["track_id"])
        duplicate_of[candidate_id] = target_id
        candidate["duplicate_of_track_id"] = target_id
        duplicate_target["merged_duplicate_track_ids"].append(candidate_id)
        duplicate_target["pattern_instance_count"] = max(
            int(duplicate_target.get("pattern_instance_count", 1)),
            int(candidate.get("pattern_instance_count", 1)),
        )
    kept.sort(key=lambda track: int(track["track_id"]))
    return kept, duplicate_of


def run_grid_pattern_analysis(
    *,
    slices: np.ndarray,
    source_path: Path,
    output_directory: Path,
    period_slices: int,
    group_number: int | None,
    threshold: float,
    foreground: str,
    opening_size: int,
    grid_rows: int,
    grid_columns: int,
    patch_size: int,
    expected_vote: float,
    min_reference_fraction: float,
    missing_fraction_threshold: float,
    max_present_fraction_of_expected: float,
    similarity_threshold: float,
    spatial_tolerance_pixels: int,
    min_track_slices: int,
    track_radius_cells: float,
    confidence_threshold: float,
    roi_presence_fraction: float,
    broken_min_track_slices: int = 8,
    broken_confidence_threshold: float = 0.85,
    missing_min_observation_fraction: float = 0.55,
    missing_span_tolerance_fraction: float = 0.15,
    missing_spatial_extent_threshold_cells: float = 0.35,
    missing_spatial_confirmation_slices: int = 2,
    broken_jump_threshold_cells: float = 1.0,
    geometry_reference_cycles: int = 2,
    comparison_mode: str = "periodic_segments",
    periodic_vector_count: int = 4,
    periodic_min_prediction_votes: int = 2,
    periodic_min_bidirectional_pairs: int = 2,
    min_missing_segment_area: int = 4,
    min_missing_segment_fraction: float = 0.0005,
    min_missing_clearance_pixels: float = 3.0,
    min_missing_clearance_fraction: float = 0.03,
    periodic_border_margin_pixels: int = 2,
    periodic_border_margin_fraction: float = 0.10,
    periodic_max_track_gap_fraction: float = 0.35,
    tilt_correction: bool = True,
    tilt_sample_count: int = 32,
    maximum_tilt_shift_per_period_fraction: float = 0.5,
    known_tilt_angle_degrees: float | None = 0.664,
    tilt_angle_tolerance_degrees: float = 0.20,
    slice_spacing_to_pixel_spacing_ratio: float = 1.0,
    forward_track_prediction: bool = True,
    forward_track_history: int = 6,
    forward_track_gap_radius_growth: float = 0.08,
    forward_track_maximum_radius_factor: float = 2.0,
) -> dict[str, Any]:
    """Inspect an N×N lattice grid and track every missing cell motif."""
    if comparison_mode not in {"periodic_segments", "fixed_cells"}:
        raise ValueError(
            "comparison_mode must be 'periodic_segments' or 'fixed_cells'"
        )
    if broken_min_track_slices < 2:
        raise ValueError("broken_min_track_slices must be at least 2")
    if not 0 <= broken_confidence_threshold <= 1:
        raise ValueError(
            "broken_confidence_threshold must be in [0, 1]"
        )
    if not 0 < missing_min_observation_fraction <= 1:
        raise ValueError(
            "missing_min_observation_fraction must be in (0, 1]"
        )
    if not 0 <= missing_span_tolerance_fraction < 0.5:
        raise ValueError(
            "missing_span_tolerance_fraction must be in [0, 0.5)"
        )
    if missing_spatial_extent_threshold_cells <= 0:
        raise ValueError(
            "missing_spatial_extent_threshold_cells must be positive"
        )
    if missing_spatial_confirmation_slices < 1:
        raise ValueError(
            "missing_spatial_confirmation_slices must be positive"
        )
    if broken_jump_threshold_cells <= 0:
        raise ValueError("broken_jump_threshold_cells must be positive")
    if not 0 <= min_missing_segment_fraction <= 1:
        raise ValueError(
            "min_missing_segment_fraction must be in [0, 1]"
        )
    if periodic_min_prediction_votes < 2:
        raise ValueError(
            "periodic_min_prediction_votes must be at least 2"
        )
    if periodic_min_bidirectional_pairs < 1:
        raise ValueError(
            "periodic_min_bidirectional_pairs must be positive"
        )
    if (
        min_missing_clearance_pixels < 0
        or not 0 <= min_missing_clearance_fraction <= 1
    ):
        raise ValueError("missing-clearance thresholds are invalid")
    if not 0 <= periodic_max_track_gap_fraction <= 1:
        raise ValueError(
            "periodic_max_track_gap_fraction must be in [0, 1]"
        )
    if not 0 <= periodic_border_margin_fraction <= 1:
        raise ValueError(
            "periodic_border_margin_fraction must be in [0, 1]"
        )
    if tilt_sample_count < 2:
        raise ValueError("tilt_sample_count must be at least 2")
    if not 0 < maximum_tilt_shift_per_period_fraction <= 1:
        raise ValueError(
            "maximum_tilt_shift_per_period_fraction must be in (0, 1]"
        )
    if (
        known_tilt_angle_degrees is not None
        and abs(known_tilt_angle_degrees) >= 45
    ):
        raise ValueError(
            "known_tilt_angle_degrees magnitude must be below 45"
        )
    if not 0 <= tilt_angle_tolerance_degrees < 45:
        raise ValueError(
            "tilt_angle_tolerance_degrees must be in [0, 45)"
        )
    if slice_spacing_to_pixel_spacing_ratio <= 0:
        raise ValueError(
            "slice_spacing_to_pixel_spacing_ratio must be positive"
        )
    if forward_track_history < 2:
        raise ValueError("forward_track_history must be at least 2")
    if forward_track_gap_radius_growth < 0:
        raise ValueError(
            "forward_track_gap_radius_growth must be nonnegative"
        )
    if forward_track_maximum_radius_factor < 1:
        raise ValueError(
            "forward_track_maximum_radius_factor must be at least 1"
        )
    if group_number is None:
        scope_start, scope_end = 0, len(slices)
        scope_name = "full_scan"
    else:
        if group_number < 1:
            raise ValueError("group_number must be one-based and positive")
        scope_start = (group_number - 1) * period_slices
        scope_end = min(len(slices), scope_start + period_slices)
        if scope_start >= len(slices):
            raise ValueError(
                f"group_number {group_number} starts beyond the scan"
            )
        scope_name = f"group_{group_number:03d}"

    output_directory.mkdir(parents=True, exist_ok=True)
    cache = MaskCache(
        slices,
        threshold,
        foreground,
        opening_size,
    )
    geometry_indices = list(range(scope_start, scope_end))
    for cycle in range(1, geometry_reference_cycles + 1):
        offset = cycle * period_slices
        geometry_indices.extend(
            index
            for index in range(scope_start - offset, scope_end - offset)
            if 0 <= index < len(slices)
        )
        geometry_indices.extend(
            index
            for index in range(scope_start + offset, scope_end + offset)
            if 0 <= index < len(slices)
        )
    roi = _grid_roi_from_indices(
        cache,
        geometry_indices,
        roi_presence_fraction,
        grid_rows,
        grid_columns,
    )
    bounds = _grid_cell_bounds(roi, grid_rows, grid_columns)
    cell_height = max(1.0, (roi[1] - roi[0]) / grid_rows)
    cell_width = max(1.0, (roi[3] - roi[2]) / grid_columns)
    angle_tolerance = tilt_tolerance_from_angle(
        known_tilt_angle_degrees,
        tilt_angle_tolerance_degrees,
        scope_end - scope_start,
        slice_spacing_to_pixel_spacing_ratio,
        min(cell_height, cell_width),
    )
    effective_spatial_tolerance_pixels = max(
        spatial_tolerance_pixels,
        int(angle_tolerance["additional_spatial_tolerance_pixels"]),
    )
    if tilt_correction:
        tilt = _estimate_stack_tilt(
            cache,
            scope_start,
            scope_end,
            period_slices,
            roi,
            grid_rows,
            grid_columns,
            sample_count=tilt_sample_count,
            maximum_shift_per_period_fraction=(
                maximum_tilt_shift_per_period_fraction
            ),
        )
    else:
        tilt = {
            "applied": False,
            "slope_y_pixels_per_slice": 0.0,
            "slope_x_pixels_per_slice": 0.0,
            "shift_y_pixels_per_period": 0.0,
            "shift_x_pixels_per_period": 0.0,
            "confidence": 0.0,
            "pair_count": 0,
            "pairs": [],
        }

    observations: dict[int, list[dict[str, Any]]] = {}
    candidate_masks: dict[int, dict[int, np.ndarray]] = {}
    expected_by_slice: dict[int, np.ndarray] = {}
    step4_by_slice: dict[int, np.ndarray] = {}
    periodic_diagnostics: dict[int, dict[str, Any]] = {}
    for index in range(scope_start, scope_end):
        if comparison_mode == "periodic_segments":
            (
                candidates,
                masks_by_id,
                expected,
                step4,
                diagnostics,
            ) = _periodic_slice_candidates(
                cache.get(index),
                index,
                roi,
                grid_rows,
                grid_columns,
                expected_vote,
                periodic_vector_count,
                periodic_min_prediction_votes,
                periodic_min_bidirectional_pairs,
                effective_spatial_tolerance_pixels,
                min_missing_segment_area,
                min_missing_segment_fraction,
                min_missing_clearance_pixels,
                min_missing_clearance_fraction,
                periodic_border_margin_pixels,
                periodic_border_margin_fraction,
            )
            periodic_diagnostics[index] = diagnostics
        else:
            candidates, masks_by_id, expected, step4 = (
                _grid_slice_candidates(
                    cache.get(index),
                    index,
                    bounds,
                    patch_size,
                    expected_vote,
                    min_reference_fraction,
                    missing_fraction_threshold,
                    max_present_fraction_of_expected,
                    similarity_threshold,
                    effective_spatial_tolerance_pixels,
                )
            )
        for candidate in candidates:
            candidate_id = int(candidate["candidate_id"])
            candidate["pattern_instance_count"] = (
                _count_pattern_instances(masks_by_id[candidate_id])
            )
        _apply_tilt_correction(
            candidates,
            index,
            scope_start,
            tilt,
            roi,
            grid_rows,
            grid_columns,
        )
        observations[index] = candidates
        candidate_masks[index] = masks_by_id
        expected_by_slice[index] = expected
        step4_by_slice[index] = step4

    maximum_track_gap = (
        max(
            2,
            int(
                math.ceil(
                    period_slices
                    * periodic_max_track_gap_fraction
                )
            ),
        )
        if comparison_mode == "periodic_segments"
        else 2
    )
    tracks = _track_all(
        observations,
        track_radius_cells,
        maximum_gap=maximum_track_gap,
        forward_prediction=forward_track_prediction,
        prediction_history=forward_track_history,
        gap_radius_growth=forward_track_gap_radius_growth,
        maximum_radius_factor=forward_track_maximum_radius_factor,
        tilt_tolerance_per_slice_cells=float(
            angle_tolerance["tracking_tolerance_cells_per_slice"]
        ),
    )
    track_summaries, confirmed = _summarize_grid_tracks(
        tracks,
        min_track_slices,
        confidence_threshold,
        maximum_gap=maximum_track_gap,
        broken_min_track_slices=broken_min_track_slices,
        broken_confidence_threshold=broken_confidence_threshold,
        missing_min_observation_fraction=(
            missing_min_observation_fraction
        ),
        missing_span_tolerance_fraction=(
            missing_span_tolerance_fraction
        ),
        missing_spatial_extent_threshold_cells=(
            missing_spatial_extent_threshold_cells
        ),
        missing_spatial_confirmation_slices=(
            missing_spatial_confirmation_slices
        ),
        broken_jump_threshold_cells=broken_jump_threshold_cells,
    )
    confirmed_before_deduplication = len(confirmed)
    confirmed, duplicate_track_map = _deduplicate_confirmed_tracks(
        confirmed,
        maximum_gap=maximum_track_gap,
    )
    retained_by_slice: dict[int, list[dict[str, Any]]] = {}
    observation_track_ids: dict[tuple[int, int], int] = {}
    observation_track_classes: dict[tuple[int, int], str] = {}
    observation_track_confidences: dict[tuple[int, int], float] = {}
    for track in confirmed:
        track_id = int(track["track_id"])
        defect_class = str(track["defect_class"])
        for observation in track["observations"]:
            index = int(observation["slice_index_zero_based"])
            retained_by_slice.setdefault(index, []).append(observation)
            observation_key = (
                index,
                int(observation["candidate_id"]),
            )
            observation_track_ids[observation_key] = track_id
            observation_track_classes[observation_key] = defect_class
            observation_track_confidences[observation_key] = float(
                track["confidence"]
            )

    depth = scope_end - scope_start
    plane_shape = tuple(int(value) for value in slices.shape[1:])
    step4_path = output_directory / f"{scope_name}_grid_step4_candidates.npy"
    step5_path = output_directory / f"{scope_name}_grid_step5_confirmed.npy"
    viewer_path = (
        output_directory
        / f"{scope_name}_grid_step5_on_segmented_slice_viewer_rgb.npy"
    )
    classification_path = (
        output_directory
        / f"{scope_name}_grid_defect_class_labels.npy"
    )
    step4_volume = np.lib.format.open_memmap(
        step4_path,
        mode="w+",
        dtype=np.bool_,
        shape=(depth,) + plane_shape,
    )
    step5_volume = np.lib.format.open_memmap(
        step5_path,
        mode="w+",
        dtype=np.bool_,
        shape=(depth,) + plane_shape,
    )
    viewer = np.lib.format.open_memmap(
        viewer_path,
        mode="w+",
        dtype=np.uint8,
        shape=(depth,) + plane_shape + (3,),
    )
    classification_volume = np.lib.format.open_memmap(
        classification_path,
        mode="w+",
        dtype=np.uint8,
        shape=(depth,) + plane_shape,
    )

    csv_rows: list[dict[str, Any]] = []
    defect_class_codes = {
        "missing_strut": 1,
        "broken_strut": 2,
    }
    defect_class_colors = {
        "missing_strut": np.array([255, 13, 13], dtype=np.uint8),
        "broken_strut": np.array([255, 140, 0], dtype=np.uint8),
    }
    rendered_instance_counts_by_class: dict[str, dict[int, int]] = {
        defect_class: {} for defect_class in defect_class_codes
    }
    method_example: dict[str, Any] | None = None
    method_example_area = -1
    for local_index, index in enumerate(range(scope_start, scope_end)):
        step4 = step4_by_slice[index]
        retained = np.zeros(plane_shape, dtype=bool)
        class_labels = np.zeros(plane_shape, dtype=np.uint8)
        rgb = _normalize_segmented(cache.get(index))
        for observation in retained_by_slice.get(index, []):
            candidate_id = int(observation["candidate_id"])
            observation_key = (index, candidate_id)
            defect_class = observation_track_classes[observation_key]
            defect_mask = candidate_masks[index][candidate_id]
            retained |= defect_mask
            defect_code = defect_class_codes[defect_class]
            assignable = defect_mask & (
                (class_labels == 0) | (defect_code < class_labels)
            )
            class_labels[assignable] = defect_code
            box = observation["box_bounds_original_pixels"]
            y0, y1 = int(box["y_start"]), int(box["y_end_exclusive"])
            x0, x1 = int(box["x_start"]), int(box["x_end_exclusive"])
            box_outline = np.zeros(plane_shape, dtype=bool)
            box_outline[y0:y1, x0] = True
            box_outline[y0:y1, max(x0, x1 - 1)] = True
            box_outline[y0, x0:x1] = True
            box_outline[max(y0, y1 - 1), x0:x1] = True
            rgb[box_outline] = np.array([255, 215, 0], dtype=np.uint8)
            original_centroid = observation.get(
                "centroid_original_pixels",
                {},
            )
            tracking_prediction = observation.get(
                "tracking_prediction",
                {},
            )
            csv_rows.append(
                {
                    "dataset_index_zero_based": index,
                    "dataset_slice_one_based": index + 1,
                    "track_id": observation_track_ids[(index, candidate_id)],
                    "defect_class": defect_class,
                    "classification_confidence": (
                        observation_track_confidences[observation_key]
                    ),
                    "grid_row_zero_based": observation["grid_row"],
                    "grid_column_zero_based": observation["grid_column"],
                    "centroid_original_y": original_centroid.get("y"),
                    "centroid_original_x": original_centroid.get("x"),
                    "tracking_centroid_corrected_y": observation[
                        "centroid_y"
                    ],
                    "tracking_centroid_corrected_x": observation[
                        "centroid_x"
                    ],
                    "predicted_centroid_y": tracking_prediction.get(
                        "predicted_centroid_y"
                    ),
                    "predicted_centroid_x": tracking_prediction.get(
                        "predicted_centroid_x"
                    ),
                    "velocity_y_cells_per_slice": tracking_prediction.get(
                        "velocity_y_cells_per_slice"
                    ),
                    "velocity_x_cells_per_slice": tracking_prediction.get(
                        "velocity_x_cells_per_slice"
                    ),
                    "acceleration_y_cells_per_slice_squared": (
                        tracking_prediction.get(
                            "acceleration_y_cells_per_slice_squared"
                        )
                    ),
                    "acceleration_x_cells_per_slice_squared": (
                        tracking_prediction.get(
                            "acceleration_x_cells_per_slice_squared"
                        )
                    ),
                    "prediction_uncertainty_cells": (
                        tracking_prediction.get(
                            "prediction_uncertainty_cells"
                        )
                    ),
                    "prediction_model": tracking_prediction.get(
                        "prediction_model"
                    ),
                    "assignment_gate_cells": tracking_prediction.get(
                        "assignment_gate_cells"
                    ),
                    "tilt_tolerance_cells": tracking_prediction.get(
                        "tilt_tolerance_cells"
                    ),
                    "prediction_residual_cells": tracking_prediction.get(
                        "residual_cells"
                    ),
                    "missing_fraction": observation["missing_fraction"],
                    "present_fraction_of_expected": observation[
                        "present_fraction_of_expected"
                    ],
                    "patch_similarity": observation["patch_similarity"],
                    "missing_segment_span_pixels": observation.get(
                        "missing_segment_span_pixels"
                    ),
                    "missing_segment_span_fraction_cells": observation.get(
                        "missing_segment_span_fraction_cells"
                    ),
                    "missing_segment_aspect_ratio": observation.get(
                        "missing_segment_aspect_ratio"
                    ),
                    "pattern_instance_count": observation.get(
                        "pattern_instance_count",
                        1,
                    ),
                    "detection_method": observation.get(
                        "detection_method",
                        "fixed_cell_prototype",
                    ),
                }
            )
        for defect_class, code in defect_class_codes.items():
            class_mask = class_labels == code
            rgb[class_mask] = defect_class_colors[defect_class]
            rendered_instance_counts_by_class[defect_class][index] = (
                _count_pattern_instances(class_mask)
                if class_mask.any()
                else 0
            )
        outline = ndimage.binary_dilation(retained, iterations=2) & ~retained
        rgb[outline] = np.array([255, 255, 255], dtype=np.uint8)
        step4_volume[local_index] = step4
        step5_volume[local_index] = retained
        viewer[local_index] = rgb
        classification_volume[local_index] = class_labels
        ranking_area = int(retained.sum()) or int(step4.sum())
        if ranking_area > method_example_area:
            method_example_area = ranking_area
            method_example = {
                "dataset_index": index,
                "raw": np.asarray(slices[index]).copy(),
                "segmented": cache.get(index).copy(),
                "expected": expected_by_slice[index].copy(),
                "candidates": step4.copy(),
                "retained": retained.copy(),
                "viewer_rgb": rgb.copy(),
            }
    step4_volume.flush()
    step5_volume.flush()
    viewer.flush()
    classification_volume.flush()

    method_overview: Path | None = None
    if method_example is not None:
        method_overview = _render_method_steps(
            output_directory,
            scope_name,
            int(method_example["dataset_index"]),
            np.asarray(method_example["raw"]),
            np.asarray(method_example["segmented"]),
            np.asarray(method_example["expected"]),
            np.asarray(method_example["candidates"]),
            np.asarray(method_example["retained"]),
            np.asarray(method_example["viewer_rgb"]),
            method_name=(
                "Phase-aware periodic segment inspection"
                if comparison_mode == "periodic_segments"
                else "N×N fixed-cell inspection"
            ),
            expected_title=(
                "Repeated-neighbor consensus"
                if comparison_mode == "periodic_segments"
                else "Within-slice grid prototype"
            ),
        )

    csv_path = output_directory / f"{scope_name}_grid_confirmed_patterns.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset_index_zero_based",
            "dataset_slice_one_based",
            "track_id",
            "defect_class",
            "classification_confidence",
            "grid_row_zero_based",
            "grid_column_zero_based",
            "centroid_original_y",
            "centroid_original_x",
            "tracking_centroid_corrected_y",
            "tracking_centroid_corrected_x",
            "predicted_centroid_y",
            "predicted_centroid_x",
            "velocity_y_cells_per_slice",
            "velocity_x_cells_per_slice",
            "acceleration_y_cells_per_slice_squared",
            "acceleration_x_cells_per_slice_squared",
            "prediction_uncertainty_cells",
            "prediction_model",
            "assignment_gate_cells",
            "tilt_tolerance_cells",
            "prediction_residual_cells",
            "missing_fraction",
            "present_fraction_of_expected",
            "patch_similarity",
            "missing_segment_span_pixels",
            "missing_segment_span_fraction_cells",
            "missing_segment_aspect_ratio",
            "pattern_instance_count",
            "detection_method",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    rejection_reason_counts: dict[str, int] = {}
    for track in track_summaries:
        for reason in track["rejection_reasons"]:
            rejection_reason_counts[reason] = (
                rejection_reason_counts.get(reason, 0) + 1
            )
    periodic_candidate_totals = {
        key: int(
            sum(
                int(diagnostics.get(key, 0))
                for diagnostics in periodic_diagnostics.values()
            )
        )
        for key in [
            "raw_missing_component_count",
            "rejected_by_area",
            "rejected_by_clearance",
            "rejected_by_border",
            "rejected_one_sided_pixel_count",
            "rejected_one_sided_component_count",
            "accepted_candidate_count",
        ]
    }
    confirmed_track_count = len(confirmed)
    confirmed_by_class = {
        defect_class: [
            track
            for track in confirmed
            if track["defect_class"] == defect_class
        ]
        for defect_class in defect_class_codes
    }
    confirmed_instance_counts_by_class = {
        defect_class: max(
            _confirmed_pattern_instance_count(tracks_for_class),
            max(
                rendered_instance_counts_by_class[defect_class].values(),
                default=0,
            ),
        )
        for defect_class, tracks_for_class in confirmed_by_class.items()
    }
    confirmed_track_counts_by_class = {
        defect_class: len(tracks_for_class)
        for defect_class, tracks_for_class in confirmed_by_class.items()
    }
    confirmed_instances_by_slice_and_class = {
        defect_class: {
            str(index): int(count)
            for index, count in counts.items()
            if count > 0
        }
        for defect_class, counts in (
            rendered_instance_counts_by_class.items()
        )
    }
    confirmed_missing_count = confirmed_instance_counts_by_class[
        "missing_strut"
    ]
    confirmed_broken_count = confirmed_instance_counts_by_class[
        "broken_strut"
    ]
    confirmed_defect_count = (
        confirmed_missing_count
        + confirmed_broken_count
    )

    report = {
        "method": (
            "phase_aware_periodic_strut_defect_classification"
            if comparison_mode == "periodic_segments"
            else "within_slice_n_by_n_strut_defect_classification"
        ),
        "source_file": str(source_path),
        "group_number": group_number,
        "period_slices": period_slices,
        "scope": {
            "start_index_zero_based": scope_start,
            "end_index_zero_based_exclusive": scope_end,
            "slice_count": depth,
        },
        "segmentation": {
            "threshold": threshold,
            "foreground": foreground,
            "opening_size": opening_size,
        },
        "tilt_correction": {
            **tilt,
            "requested": tilt_correction,
            "reference_index_zero_based": scope_start,
            "sample_count_requested": tilt_sample_count,
            "maximum_shift_per_period_fraction": (
                maximum_tilt_shift_per_period_fraction
            ),
            "angle_allowance": angle_tolerance,
            "coordinate_policy": (
                "raw coordinates for overlays; de-tilted coordinates "
                "for grid assignment and parallel tracking"
            ),
        },
        "grid": {
            "rows": grid_rows,
            "columns": grid_columns,
            "roi_original_pixels": {
                "y_start": roi[0],
                "y_end_exclusive": roi[1],
                "x_start": roi[2],
                "x_end_exclusive": roi[3],
            },
            "comparison_mode": comparison_mode,
            "prototype_source": (
                "same_phase_periodic_neighbors_in_the_same_slice"
                if comparison_mode == "periodic_segments"
                else "other_grid_cells_in_the_same_slice"
            ),
            "geometry_reference_cycles": geometry_reference_cycles,
            "geometry_reference_slice_count": len(set(geometry_indices)),
            "cell_bounds_original_pixels": [
                {
                    "row": row,
                    "column": column,
                    "y_start": y0,
                    "y_end_exclusive": y1,
                    "x_start": x0,
                    "x_end_exclusive": x1,
                }
                for row, column, y0, y1, x0, x1 in bounds
            ],
        },
        "classification_rule": {
            "missing_fraction_threshold": missing_fraction_threshold,
            "max_present_fraction_of_expected": (
                max_present_fraction_of_expected
            ),
            "similarity_threshold": similarity_threshold,
            "requested_spatial_tolerance_pixels": (
                spatial_tolerance_pixels
            ),
            "effective_spatial_tolerance_pixels": (
                effective_spatial_tolerance_pixels
            ),
            "min_track_slices": min_track_slices,
            "missing_min_observation_fraction": (
                missing_min_observation_fraction
            ),
            "missing_span_tolerance_fraction": (
                missing_span_tolerance_fraction
            ),
            "missing_spatial_extent_threshold_cells": (
                missing_spatial_extent_threshold_cells
            ),
            "missing_spatial_confirmation_slices": (
                missing_spatial_confirmation_slices
            ),
            "broken_min_track_slices": broken_min_track_slices,
            "broken_confidence_threshold": (
                broken_confidence_threshold
            ),
            "broken_jump_threshold_cells": (
                broken_jump_threshold_cells
            ),
            "periodic_vector_count": periodic_vector_count,
            "periodic_min_prediction_votes": (
                periodic_min_prediction_votes
            ),
            "periodic_min_bidirectional_pairs": (
                periodic_min_bidirectional_pairs
            ),
            "min_missing_segment_area": min_missing_segment_area,
            "min_missing_segment_fraction": (
                min_missing_segment_fraction
            ),
            "min_missing_clearance_pixels": (
                min_missing_clearance_pixels
            ),
            "min_missing_clearance_fraction": (
                min_missing_clearance_fraction
            ),
            "periodic_border_margin_pixels": (
                periodic_border_margin_pixels
            ),
            "periodic_border_margin_fraction": (
                periodic_border_margin_fraction
            ),
            "note": (
                "Periodic mode votes from translated copies of the current "
                "slice, preserves alternating motif phase and struts crossing "
                "cell boundaries. Each retained absence receives one mutually "
                "exclusive class. Missing requires either sustained local "
                "temporal support or a full absent spatial segment in multiple "
                "slices; shorter interrupted segments are broken."
            ),
        },
        "periodic_slice_diagnostics": (
            {
                str(index): diagnostics
                for index, diagnostics in periodic_diagnostics.items()
            }
            if comparison_mode == "periodic_segments"
            else None
        ),
        "periodic_candidate_totals": (
            periodic_candidate_totals
            if comparison_mode == "periodic_segments"
            else None
        ),
        "tracking": {
            "candidate_track_count": len(track_summaries),
            "confirmed_before_deduplication": (
                confirmed_before_deduplication
            ),
            "deduplicated_track_count": len(duplicate_track_map),
            "duplicate_track_map": {
                str(duplicate_id): retained_id
                for duplicate_id, retained_id in (
                    duplicate_track_map.items()
                )
            },
            "confirmed_track_ids": [
                int(track["track_id"]) for track in confirmed
            ],
            "confirmed_track_ids_by_class": {
                defect_class: [
                    int(track["track_id"])
                    for track in tracks_for_class
                ]
                for defect_class, tracks_for_class in (
                    confirmed_by_class.items()
                )
            },
            "parallel_track_limit": None,
            "track_radius_cells": track_radius_cells,
            "maximum_phase_gap_slices": maximum_track_gap,
            "maximum_phase_gap_fraction_of_period": (
                periodic_max_track_gap_fraction
            ),
            "forward_prediction": forward_track_prediction,
            "forward_prediction_history": forward_track_history,
            "forward_prediction_gap_radius_growth": (
                forward_track_gap_radius_growth
            ),
            "forward_prediction_maximum_radius_factor": (
                forward_track_maximum_radius_factor
            ),
            "tilt_tolerance_cells_per_slice": angle_tolerance[
                "tracking_tolerance_cells_per_slice"
            ],
            "forward_prediction_model": (
                "robust velocity with bounded constant acceleration, "
                "uncertainty-aware gap gating, and one-to-one parallel "
                "assignment"
            ),
            "rejected_track_count": (
                len(track_summaries)
                - confirmed_before_deduplication
            ),
            "rejection_reason_counts": rejection_reason_counts,
        },
        "confirmed_track_count": confirmed_track_count,
        "confirmed_track_counts_by_class": (
            confirmed_track_counts_by_class
        ),
        "confirmed_defect_count": confirmed_defect_count,
        "confirmed_missing_pattern_count": confirmed_missing_count,
        "confirmed_missing_strut_count": confirmed_missing_count,
        "confirmed_broken_strut_count": confirmed_broken_count,
        "confirmed_defect_instance_counts_by_class": (
            confirmed_instance_counts_by_class
        ),
        "confirmed_pattern_instances_by_slice": (
            confirmed_instances_by_slice_and_class["missing_strut"]
        ),
        "confirmed_instances_by_slice_and_class": (
            confirmed_instances_by_slice_and_class
        ),
        "confidence_threshold": confidence_threshold,
        "candidate_tracks": [
            {
                key: value
                for key, value in track.items()
                if key != "observations"
            }
            for track in track_summaries
        ],
        "confirmed_patterns": [
            {
                **{
                    key: value
                    for key, value in track.items()
                    if key != "observations"
                },
                "grid_cells_zero_based": [
                    {"row": row, "column": column}
                    for row, column in track["grid_cells_zero_based"]
                ],
            }
            for track in confirmed
        ],
        "confirmed_defects_by_class": {
            defect_class: [
                {
                    **{
                        key: value
                        for key, value in track.items()
                        if key != "observations"
                    },
                    "grid_cells_zero_based": [
                        {"row": row, "column": column}
                        for row, column in track[
                            "grid_cells_zero_based"
                        ]
                    ],
                }
                for track in tracks_for_class
            ]
            for defect_class, tracks_for_class in confirmed_by_class.items()
        },
        "outputs": {
            "step4_candidate_mask_npy": str(step4_path),
            "step5_confirmed_mask_npy": str(step5_path),
            "segmented_step5_viewer_npy": str(viewer_path),
            "defect_class_labels_npy": str(classification_path),
            "defect_class_label_values": {
                "background": 0,
                **defect_class_codes,
            },
            "viewer_colors_rgb": {
                defect_class: color.tolist()
                for defect_class, color in defect_class_colors.items()
            },
            "confirmed_patterns_csv": str(csv_path),
            "method_pipeline_overview_png": (
                str(method_overview) if method_overview is not None else None
            ),
        },
    }
    report_path = output_directory / f"{scope_name}_grid_findings_report.json"
    report["outputs"]["findings_report_json"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
