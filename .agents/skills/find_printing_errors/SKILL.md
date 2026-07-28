---
name: find-printing-errors
description: Detect missing or disconnected lattice material by rasterizing registered JSON geometry, selecting one CT TIFF segmentation in memory, and comparing the saved segmentation with the ideal raster using the bundled scripts and segmentation MCP server.
---

# Find Printing Errors

Use this skill when the user provides a lattice geometry JSON and a matching CT TIFF stack and wants printing defects, especially missing struts, identified. Follow the stages in order.

## Inputs and outputs

Inputs:

- A JSON containing `junctions` and `struts`.
- The matching 3D CT TIFF stack.
- Optional output directory, threshold candidates, coordinate transform, and defect tolerance.
- Do not modify the JSON or TIFF; they are read-only inputs.
- Do not generate or modify existing python scripts


Default to the repository's `xyz` coordinate convention, unit coordinate scale, zero origin, and a 2-voxel spherical defect tolerance. Confirm the JSON/TIFF are a registered pair before interpreting geometric differences. If they are not registered, stop and report that registration must be resolved; do not silently optimize away a registration error.

Write unique output names derived from the TIFF stem. Do not overwrite an existing result belonging to another input pair. The final artifacts are:

- `<stem>_ideal_raster.tif` and a rasterization report.
- `<stem>_best_segmentation.tif`.
- `<stem>_missing_defects.tif`.
- `<stem>_defect_report.json`.

## Stage 1: Rasterize the JSON

Use `scripts/rasterize_and_align.py` to rasterize the JSON into the exact `(z, y, x)` shape of the CT TIFF. Use `rasterize()` or its CLI with the selected `strut_radius`, `axis_order`, `coordinate_scale`, and `coordinate_origin`.

For a registered pair, use `--registration none`; do not apply COM or correlation registration unless the user explicitly requests it and the resulting transform is recorded. Save the ideal mask as binary uint8 TIFF (`0` and `255`) and save the script's JSON report. Stop if the geometry does not overlap the CT volume or if the raster and CT shapes differ.

## Stage 2: Select and save exactly one segmentation

The MCP server configured as `segmentation-tools` exposes `segment_ct_dataset_tif(input_filepath, output_filepath, threshold)`. It writes its output immediately, so do not call it once per candidate and do not save trial masks.

Instead:

1. Read the raw CT TIFF and ideal raster in memory.
2. Inspect dtype, finite intensity range, percentiles, and Otsu threshold.
3. Build a finite set of candidate thresholds from Otsu plus meaningful histogram/percentile transition values. Adapt the candidates to the observed dtype; never use fixed values that are outside the input range.
4. For each candidate, create an in-memory boolean mask `scan >= threshold` and calculate foreground overlap, Dice, and IoU against the ideal raster. Do not write these masks.
5. Select the candidate with the highest Dice; break ties with the highest IoU, then prefer the lower threshold only if both scores are equal.
6. Call the MCP tool exactly once with the selected threshold and `<stem>_best_segmentation.tif` as its output path.
7. Read back the MCP output and verify it exists, is 3D, has the CT shape, and is nonempty. Record the threshold, candidate scores, selected Dice/IoU, and foreground count in the defect report.

Dice/IoU measure agreement with the rasterized design, not ground-truth accuracy. Inspect the selected mask for obvious noise, disconnected material, clipping, or partial-volume artifacts and record concerns rather than changing the selection after the fact.

## Stage 3: Find missing defects

Run `scripts/detect_missing_with_tolerance.py` with:

- `--ideal <stem>_ideal_raster.tif`
- `--scan <stem>_best_segmentation.tif`
- `--output <stem>_missing_defects.tif`
- `--radius 4.0` unless overridden

The resulting mask is `ideal AND NOT dilated(segmentation)`. Verify the input shapes match and report total ideal voxels, missing voxels, and missing percentage. A tolerance suppresses small registration and partial-volume discrepancies; it does not prove a physical defect.

For geometry-level defect names, run `scripts/identify_json_ids.py` against the selected segmentation using the same coordinate transform and a radius consistent with the rasterization/tolerance. Use its audit output to identify supported struts and junctions, then compute the JSON IDs absent from the supported sets. Preserve clipped geometry and low-support IDs in the report so they are distinguishable from confidently missing defects.

Use `--strictness current` by default. If the user asks for a strict result, use `--strictness strict`. This is strict about reporting a missing defect: it uses the same sampling neighborhoods as `current` but requires less support to classify a strut or junction as present, so its missing ID sets are subsets of (or equal to) `current`. The `loose` and `loosest` presets retain their existing behavior; `loosest` reports the most missing candidates and has the highest false-positive risk. Use `custom` when explicit thresholds are requested. (`white_mode=exact` is a separate mask-selection option and is unchanged.)

The defect report must include input paths, output paths, transform settings, raster radius, segmentation threshold, candidate metric summary, tolerance radius, shape, voxel counts, missing percentage, supported IDs, candidate missing IDs, and any registration or clipping warnings.

- Asking for stricter results in less missing candidates being reported.
- Asking for looser results in more missing candidates being reported, but they may be false positives.

## Failure handling

- Missing files, unreadable TIFFs, non-3D volumes, malformed JSON, empty masks, shape mismatches, and non-overlapping geometry are fatal and must be reported clearly.
- If all candidate masks are empty or all scores are zero, do not invoke MCP; report that threshold selection failed.
- If the MCP call returns an error or does not produce the requested TIFF, stop before defect detection.
- Do not label every unsupported voxel or ID as a confirmed manufacturing defect when registration, clipping, segmentation noise, or partial-volume effects could explain it.
