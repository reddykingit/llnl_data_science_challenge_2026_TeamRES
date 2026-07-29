"""Generate registered missing-pattern overlays and method PNGs for one period group."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Patch, Rectangle
from scipy import ndimage
from skimage.measure import label, regionprops


ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = ROOT / "src/pattern_deformation_analysis.py"
REPORT_PATH = (
    ROOT
    / "src/missing_struts_0point5_axis0_pattern_recognition/symmetry_report.json"
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
PERIOD = 78
PERIOD_GROUP = int(os.environ.get("PERIOD_GROUP", "9"))
START_INDEX = (PERIOD_GROUP - 1) * PERIOD
END_INDEX = START_INDEX + PERIOD - 1
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / f"period_group_{PERIOD_GROUP:03d}_highlights"
)
THRESHOLD = 38257.0
FOREGROUND = "high"
OPENING_SIZE = 3
PERIOD_CONFIDENCE = 0.8720156696692944

STATUS_COLORS = {
    "confirmed_missing_strut": np.array([1.0, 0.05, 0.05]),
    "probable_missing_strut": np.array([1.0, 0.25, 0.0]),
    "persistent_missing_node": np.array([1.0, 0.72, 0.0]),
    "persistent_missing_complex_pattern": np.array([0.8, 0.1, 1.0]),
}
STATUS_LABELS = {
    "persistent_missing_node": 1,
    "probable_missing_strut": 2,
    "confirmed_missing_strut": 3,
    "persistent_missing_complex_pattern": 4,
}
SEGMENTED_OVERLAY_COLORS = {
    1: np.array([255, 184, 0], dtype=np.uint8),
    2: np.array([255, 64, 0], dtype=np.uint8),
    3: np.array([255, 13, 13], dtype=np.uint8),
    4: np.array([204, 26, 255], dtype=np.uint8),
}


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_flagged_groups", ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import analyzer from {ANALYZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_raw(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1.0, 99.0))
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def match_component_mask(
    missing: np.ndarray, centroid_y: float, centroid_x: float
) -> tuple[np.ndarray, list[int], int]:
    labels = label(missing)
    props = regionprops(labels)
    if not props:
        raise RuntimeError("No connected missing component was available for highlighting")
    target = min(
        props,
        key=lambda prop: (
            (float(prop.centroid[0]) - centroid_y) ** 2
            + (float(prop.centroid[1]) - centroid_x) ** 2
        ),
    )
    bbox = [int(value) for value in target.bbox]
    return labels == target.label, bbox, int(target.area)


def render_overlay(
    raw: np.ndarray,
    selected_mask: np.ndarray,
    bbox: list[int],
    status: str,
    title: str,
    output_path: Path,
) -> None:
    gray = normalize_raw(raw)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    color = STATUS_COLORS[status]
    rgb[selected_mask] = 0.18 * rgb[selected_mask] + 0.82 * color
    outline = ndimage.binary_dilation(selected_mask, iterations=2) & ~selected_mask
    rgb[outline] = np.array([1.0, 1.0, 1.0])

    min_row, min_col, max_row, max_col = bbox
    pad = 25
    y0 = max(0, min_row - pad)
    y1 = min(raw.shape[0], max_row + pad)
    x0 = max(0, min_col - pad)
    x1 = min(raw.shape[1], max_col + pad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].add_patch(
        Rectangle(
            (min_col - 4, min_row - 4),
            max_col - min_col + 8,
            max_row - min_row + 8,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
        )
    )
    axes[0].set_title("Full CT slice")
    axes[1].imshow(rgb[y0:y1, x0:x1])
    axes[1].set_title("Highlighted missing-pattern evidence")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def save_method_image(
    image: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 7.0), constrained_layout=True)
    axis.imshow(image, cmap=cmap)
    axis.set_title(title, fontsize=13)
    axis.axis("off")
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)


def write_segmented_step5_overlay(
    output_dir: Path,
    cache,
    label_volume: np.ndarray,
) -> tuple[Path, Path, int]:
    """Write the default viewer-ready segmentation plus Step 5 RGB volume."""
    overlay_path = (
        output_dir
        / f"group_{PERIOD_GROUP:03d}_step5_on_segmented_slice_viewer_rgb.npy"
    )
    manifest_path = overlay_path.with_suffix(".json")
    output = np.lib.format.open_memmap(
        overlay_path,
        mode="w+",
        dtype=np.uint8,
        shape=label_volume.shape + (3,),
    )
    slices_with_tracks = 0
    for local_index in range(len(label_volume)):
        segmented = cache.get(START_INDEX + local_index)
        rgb = np.zeros(segmented.shape + (3,), dtype=np.uint8)
        rgb[segmented] = np.uint8(220)
        labels = label_volume[local_index]
        tracked = labels > 0
        slices_with_tracks += int(tracked.any())
        for label_value, color in SEGMENTED_OVERLAY_COLORS.items():
            rgb[labels == label_value] = color
        outline = ndimage.binary_dilation(tracked, iterations=2) & ~tracked
        rgb[outline] = np.array([255, 255, 255], dtype=np.uint8)
        output[local_index] = rgb
    output.flush()

    manifest = {
        "period_group": PERIOD_GROUP,
        "output_file": str(overlay_path.resolve()),
        "shape": list(output.shape),
        "dtype": "uint8",
        "channel_order": "RGB",
        "slice_axis": 0,
        "dataset_indices_zero_based": [START_INDEX, END_INDEX],
        "dataset_slice_numbers_one_based": [
            START_INDEX + 1,
            END_INDEX + 1,
        ],
        "base": {
            "meaning": "normal segmented lattice material",
            "background_rgb": [0, 0, 0],
            "material_rgb": [220, 220, 220],
            "segmentation_threshold": THRESHOLD,
            "morphological_opening_size": OPENING_SIZE,
        },
        "step5_overlay_colors": {
            "1_persistent_missing_node": (
                SEGMENTED_OVERLAY_COLORS[1].tolist()
            ),
            "2_probable_missing_strut": (
                SEGMENTED_OVERLAY_COLORS[2].tolist()
            ),
            "3_confirmed_missing_strut": (
                SEGMENTED_OVERLAY_COLORS[3].tolist()
            ),
            "4_persistent_missing_complex_pattern": (
                SEGMENTED_OVERLAY_COLORS[4].tolist()
            ),
        },
        "outline_rgb": [255, 255, 255],
        "slices_with_retained_tracks": slices_with_tracks,
        "index_mapping": (
            f"array[0] is dataset index {START_INDEX} / dataset slice "
            f"{START_INDEX + 1}; array[{len(label_volume) - 1}] is dataset "
            f"index {END_INDEX} / dataset slice {END_INDEX + 1}"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return overlay_path, manifest_path, slices_with_tracks


def render_method_steps(
    method_dir: Path,
    example: dict[str, object],
) -> None:
    method_dir.mkdir(exist_ok=True)
    raw = np.asarray(example["raw"])
    current = np.asarray(example["current"], dtype=bool)
    expected = np.asarray(example["expected"], dtype=bool)
    missing = np.asarray(example["missing"], dtype=bool)
    selected = np.asarray(example["selected_mask"], dtype=bool)
    bbox = list(example["bbox"])
    status = str(example["status"])
    dataset_slice = int(example["dataset_slice_one_based"])
    local_slice = int(example["local_slice_one_based"])
    references = list(example["references"])
    parallel_track_count = int(example.get("parallel_track_count", 1))
    reference_slices = ", ".join(str(int(index) + 1) for index in references)

    gray = normalize_raw(raw)
    missing_rgb = np.repeat(gray[..., None], 3, axis=2)
    missing_rgb[missing] = (
        0.30 * missing_rgb[missing] + 0.70 * np.array([1.0, 0.55, 0.0])
    )
    missing_rgb[selected] = (
        0.10 * missing_rgb[selected] + 0.90 * np.array([1.0, 0.0, 0.0])
    )

    final_rgb = np.repeat(gray[..., None], 3, axis=2)
    color = STATUS_COLORS[status]
    final_rgb[selected] = 0.18 * final_rgb[selected] + 0.82 * color
    outline = ndimage.binary_dilation(selected, iterations=2) & ~selected
    final_rgb[outline] = np.array([1.0, 1.0, 1.0])

    save_method_image(
        gray,
        method_dir / "method_step_1_raw_slice.png",
        f"Step 1 - raw CT input\n"
        f"period group {PERIOD_GROUP}, local {local_slice}/78, dataset slice {dataset_slice}",
        cmap="gray",
    )
    save_method_image(
        current,
        method_dir / "method_step_2_segmented_material.png",
        f"Step 2 - segment material at raw threshold {THRESHOLD:g}\n"
        f"3x3 morphological opening removes thin background noise",
        cmap="gray",
    )
    save_method_image(
        expected,
        method_dir / "method_step_3_registered_expected_mask.png",
        "Step 3 - registered same-phase expectation\n"
        f"voted from dataset slices {reference_slices}",
        cmap="gray",
    )
    save_method_image(
        missing_rgb,
        method_dir / "method_step_4_missing_difference.png",
        "Step 4 - expected material absent after spatial tolerance\n"
        "orange = candidates; red = component retained by tracking",
    )
    save_method_image(
        final_rgb,
        method_dir / "method_step_5_final_tracked_highlight.png",
        f"Step 5 - {parallel_track_count} parallel tracked pattern(s)\n"
        f"{status.replace('_', ' ')}",
    )

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), constrained_layout=True)
    panels = [
        (gray, "1. Raw slice", "gray"),
        (current, "2. Segmented material", "gray"),
        (expected, "3. Registered expectation", "gray"),
        (missing_rgb, "4. Missing difference", None),
        (final_rgb, "5. Tracked highlight", None),
    ]
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(
        f"Missing-pattern method example - period group {PERIOD_GROUP}, "
        f"dataset slice {dataset_slice}",
        fontsize=14,
    )
    fig.savefig(method_dir / "method_pipeline_overview.png", dpi=170)
    plt.close(fig)

    min_row, min_col, max_row, max_col = bbox
    pad = 35
    y0, y1 = max(0, min_row - pad), min(raw.shape[0], max_row + pad)
    x0, x1 = max(0, min_col - pad), min(raw.shape[1], max_col + pad)
    save_method_image(
        final_rgb[y0:y1, x0:x1],
        method_dir / "method_final_highlight_zoom.png",
        f"Final evidence zoom - dataset slice {dataset_slice}",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir = OUTPUT_DIR / "individual_overlays"
    overlay_dir.mkdir(exist_ok=True)

    analyzer = load_analyzer()
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    flagged = {int(index) for index in report["flagged_indices_zero_based"]}
    volume = tifffile.memmap(VOLUME_PATH)
    cache = analyzer.MaskCache(
        volume,
        threshold=THRESHOLD,
        foreground=FOREGROUND,
        opening_size=OPENING_SIZE,
    )
    label_volume = np.zeros(
        (PERIOD, int(volume.shape[1]), int(volume.shape[2])), dtype=np.uint8
    )

    period_groups = [
        group
        for group in analysis["groups"]
        if group["start_index_zero_based"] <= END_INDEX
        and group["end_index_zero_based"] >= START_INDEX
    ]
    rows: list[dict[str, object]] = []
    representative_images: list[tuple[dict[str, object], np.ndarray, np.ndarray, list[int]]] = []
    method_candidates: list[dict[str, object]] = []

    track_jobs: list[tuple[dict[str, object], dict[str, object]]] = []
    for group in period_groups:
        group_conclusion = group["conclusion"]
        decisions = group_conclusion.get("qualifying_tracks")
        if decisions is None:
            best_track_id = group_conclusion.get("best_track_id")
            decisions = (
                [
                    {
                        "track_id": best_track_id,
                        "status": group_conclusion["status"],
                        "missing_component_type": group_conclusion.get(
                            "missing_component_type"
                        ),
                        "confidence": group_conclusion["confidence"],
                    }
                ]
                if best_track_id is not None
                and group_conclusion["status"] != "inconclusive"
                else []
            )
        track_jobs.extend((group, decision) for decision in decisions)

    for group, track_decision in track_jobs:
        status = str(track_decision["status"])
        best_track_id = int(track_decision["track_id"])
        if status == "inconclusive" or best_track_id is None:
            continue
        track = next(
            item for item in group["tracks"] if item["track_id"] == best_track_id
        )
        flagged_indices = set(group["flagged_indices_zero_based"])
        candidates = [
            observation
            for observation in track["observations"]
            if observation["slice_index_zero_based"] in flagged_indices
            and START_INDEX <= observation["slice_index_zero_based"] <= END_INDEX
        ]
        rendered: list[tuple[dict[str, object], np.ndarray, np.ndarray, list[int]]] = []
        for observation in candidates:
            index = int(observation["slice_index_zero_based"])
            components, expected, missing, references = analyzer.missing_components(
                index=index,
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
            if expected is None or missing is None or not components:
                continue
            selected_mask, bbox, area = match_component_mask(
                missing,
                float(observation["centroid_y"]),
                float(observation["centroid_x"]),
            )
            local_slice = index - START_INDEX + 1
            label_value = STATUS_LABELS[status]
            label_volume[local_slice - 1, selected_mask] = np.maximum(
                label_volume[local_slice - 1, selected_mask], label_value
            )
            one_based_slice = index + 1
            title = (
                f"Period group {PERIOD_GROUP} Ã‚Â· local slice {local_slice}/78 Ã‚Â· "
                f"dataset slice {one_based_slice} Ã‚Â· {status.replace('_', ' ')}"
            )
            output_name = (
                f"local_{local_slice:02d}_dataset_slice_{one_based_slice:04d}"
                f"_analysis_group_{int(group['group_id']):03d}"
                f"_track_{best_track_id:03d}.png"
            )
            render_overlay(
                volume[index],
                selected_mask,
                bbox,
                status,
                title,
                overlay_dir / output_name,
            )
            row: dict[str, object] = {
                "period_group": PERIOD_GROUP,
                "local_slice_one_based": local_slice,
                "dataset_index_zero_based": index,
                "dataset_slice_one_based": one_based_slice,
                "analysis_group_id": int(group["group_id"]),
                "track_id": best_track_id,
                "status": status,
                "label_value": label_value,
                "component_type": track_decision.get(
                    "missing_component_type"
                ),
                "confidence": float(track_decision["confidence"]),
                "centroid_y": float(observation["centroid_y"]),
                "centroid_x": float(observation["centroid_x"]),
                "highlighted_area_pixels": area,
                "bbox_min_row": bbox[0],
                "bbox_min_col": bbox[1],
                "bbox_max_row": bbox[2],
                "bbox_max_col": bbox[3],
                "reference_indices_zero_based": ";".join(str(i) for i in references),
                "overlay_file": f"individual_overlays/{output_name}",
            }
            rows.append(row)
            rendered.append((row, volume[index], selected_mask, bbox))
            method_candidates.append(
                {
                    **row,
                    "raw": np.asarray(volume[index]).copy(),
                    "current": cache.get(index).copy(),
                    "expected": expected.copy(),
                    "missing": missing.copy(),
                    "selected_mask": selected_mask.copy(),
                    "bbox": bbox,
                    "references": references,
                }
            )
        if rendered:
            representative_images.append(
                max(rendered, key=lambda item: int(item[0]["highlighted_area_pixels"]))
            )

    # Step 5 is a slice-level view: combine every independently qualified
    # track on that slice instead of displaying only the group's best track.
    combined_method_candidates: dict[int, dict[str, object]] = {}
    method_status_rank = {
        "confirmed_missing_strut": 3,
        "probable_missing_strut": 2,
        "persistent_missing_node": 1,
        "persistent_missing_complex_pattern": 1,
    }
    preferred_method_index = int(
        max(
            method_candidates,
            key=lambda item: (
                method_status_rank.get(str(item["status"]), 0),
                float(item["confidence"]),
                int(item["highlighted_area_pixels"]),
            ),
        )["dataset_index_zero_based"]
    )
    for candidate in method_candidates:
        index = int(candidate["dataset_index_zero_based"])
        if index not in combined_method_candidates:
            combined = dict(candidate)
            combined["selected_mask"] = np.asarray(
                candidate["selected_mask"], dtype=bool
            ).copy()
            combined["parallel_track_ids"] = [int(candidate["track_id"])]
            combined["parallel_track_count"] = 1
            combined_method_candidates[index] = combined
            continue
        combined = combined_method_candidates[index]
        combined_mask = np.asarray(combined["selected_mask"], dtype=bool)
        combined_mask |= np.asarray(candidate["selected_mask"], dtype=bool)
        combined["selected_mask"] = combined_mask
        combined["highlighted_area_pixels"] = int(combined_mask.sum())
        combined["bbox"] = [
            min(int(combined["bbox"][0]), int(candidate["bbox"][0])),
            min(int(combined["bbox"][1]), int(candidate["bbox"][1])),
            max(int(combined["bbox"][2]), int(candidate["bbox"][2])),
            max(int(combined["bbox"][3]), int(candidate["bbox"][3])),
        ]
        combined["parallel_track_ids"].append(int(candidate["track_id"]))
        combined["parallel_track_count"] = len(
            set(combined["parallel_track_ids"])
        )
        if method_status_rank.get(str(candidate["status"]), 0) > (
            method_status_rank.get(str(combined["status"]), 0)
        ):
            combined["status"] = candidate["status"]
            combined["confidence"] = candidate["confidence"]
    method_candidates = list(combined_method_candidates.values())

    rows.sort(key=lambda row: int(row["dataset_index_zero_based"]))
    np.save(
        OUTPUT_DIR / f"group_{PERIOD_GROUP:03d}_missing_pattern_labels.npy",
        label_volume,
        allow_pickle=False,
    )
    (
        segmented_overlay_path,
        segmented_overlay_manifest,
        segmented_overlay_slice_count,
    ) = write_segmented_step5_overlay(
        OUTPUT_DIR,
        cache,
        label_volume,
    )
    fieldnames = list(rows[0].keys())
    with (OUTPUT_DIR / "missing_pattern_highlights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cols = 4
    rows_count = int(np.ceil(len(representative_images) / cols))
    fig, axes = plt.subplots(
        rows_count,
        cols,
        figsize=(16, 4.3 * rows_count),
        constrained_layout=True,
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, (row, raw, selected_mask, bbox) in zip(
        axes.flat, representative_images
    ):
        gray = normalize_raw(raw)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        color = STATUS_COLORS[str(row["status"])]
        rgb[selected_mask] = 0.18 * rgb[selected_mask] + 0.82 * color
        min_row, min_col, max_row, max_col = bbox
        axis.imshow(rgb)
        axis.add_patch(
            Rectangle(
                (min_col - 4, min_row - 4),
                max_col - min_col + 8,
                max_row - min_row + 8,
                fill=False,
                edgecolor=color,
                linewidth=2.0,
            )
        )
        axis.set_title(
            f"Local {row['local_slice_one_based']}/78 Ã‚Â· dataset {row['dataset_slice_one_based']}\n"
            f"{str(row['status']).replace('_', ' ')} Ã‚Â· confidence {float(row['confidence']):.2f}",
            fontsize=9,
        )
    fig.suptitle(
        f"Period group {PERIOD_GROUP}: registered missing-pattern evidence\n"
        "Red/orange = strut-like loss; amber = persistent node-pattern loss",
        fontsize=14,
    )
    fig.savefig(
        OUTPUT_DIR / f"group_{PERIOD_GROUP:03d}_missing_patterns_overview.png",
        dpi=170,
    )
    plt.close(fig)

    method_example = combined_method_candidates[preferred_method_index]
    render_method_steps(OUTPUT_DIR / "method_steps", method_example)

    inconclusive = [
        {
            "analysis_group_id": int(group["group_id"]),
            "dataset_slice_numbers_one_based": group[
                "flagged_slice_numbers_one_based"
            ],
            "explanation": group["conclusion"]["explanation"],
        }
        for group in period_groups
        if group["conclusion"]["status"] == "inconclusive"
    ]
    summary = {
        "period_group": PERIOD_GROUP,
        "period_slices": PERIOD,
        "dataset_indices_zero_based": [START_INDEX, END_INDEX],
        "dataset_slice_numbers_one_based": [START_INDEX + 1, END_INDEX + 1],
        "source_volume": str(VOLUME_PATH),
        "segmentation_threshold": THRESHOLD,
        "foreground_polarity": FOREGROUND,
        "period_confidence": PERIOD_CONFIDENCE,
        "highlight_color_key": {
            "red": "confirmed missing-strut evidence",
            "orange": "probable missing-strut evidence",
            "amber": "persistent missing-node evidence",
        },
        "highlighted_slice_count": len(rows),
        "highlighted_parallel_track_count": len(
            {
                (int(row["analysis_group_id"]), int(row["track_id"]))
                for row in rows
            }
        ),
        "highlighted_analysis_group_count": len(representative_images),
        "method_example": {
            "dataset_slice_one_based": int(
                method_example["dataset_slice_one_based"]
            ),
            "local_slice_one_based": int(method_example["local_slice_one_based"]),
            "status": str(method_example["status"]),
            "parallel_track_count": int(
                method_example.get("parallel_track_count", 1)
            ),
            "parallel_track_ids": list(
                method_example.get(
                    "parallel_track_ids",
                    [int(method_example["track_id"])],
                )
            ),
            "reference_indices_zero_based": list(method_example["references"]),
            "png_directory": "method_steps",
        },
        "label_volume": {
            "file": f"group_{PERIOD_GROUP:03d}_missing_pattern_labels.npy",
            "shape": list(label_volume.shape),
            "dtype": str(label_volume.dtype),
            "labels": {
                "0": "no highlighted evidence",
                "1": "persistent missing-node evidence",
                "2": "probable missing-strut evidence",
                "3": "confirmed missing-strut evidence",
                "4": "persistent missing-complex-pattern evidence",
            },
        },
        "segmented_step5_overlay": {
            "file": segmented_overlay_path.name,
            "manifest_file": segmented_overlay_manifest.name,
            "shape": list(label_volume.shape) + [3],
            "dtype": "uint8",
            "channel_order": "RGB",
            "slices_with_retained_tracks": (
                segmented_overlay_slice_count
            ),
            "viewer_ready": True,
        },
        "inconclusive_runs_not_highlighted": inconclusive,
        "limitations": (
            "Screening evidence from registered same-phase references; "
            "requires domain review before treating as a confirmed physical defect."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
