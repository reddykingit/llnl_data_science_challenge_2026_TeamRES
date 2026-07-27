---
name: rasterize-registered-lattice
description: Find and export the best voxel rasterization of a registered lattice-geometry JSON against its matching CT TIFF stack by searching strut radius and CT intensity thresholds, using a memory-safe coarse evaluation followed by full-resolution export.
---

# Rasterize Registered Lattice

Use this skill when the user asks for the best rasterization, voxelization, or CT-matched mask for a lattice JSON that is already registered to a TIFF stack.

## Inputs

- A registered lattice JSON containing `junctions` and `struts`.
- Its matching 3D CT TIFF stack.
- The repository's `scripts/rasterize_and_align.py`, especially `rasterize()`.

Confirm that the JSON and TIFF are the registered pair. Do not apply translation registration unless the user requests it or the pair is not registered.

## Procedure

1. Read the CT TIFF and inspect its shape, dtype, intensity range, percentiles, and Otsu threshold.
2. Evaluate candidate strut radii from `0.5` through `4.0` voxels in `0.25`-voxel increments. Evaluate CT thresholds around the intensity distribution, normally `35000, 37500, 40049, 42500, 45000, 47500, 50000` for this dataset; adapt these values to the observed dtype and histogram for other data.
3. To avoid exhausting memory on large volumes, evaluate on an 8x coarse grid:
   - CT: `scan[::8, ::8, ::8]`.
   - Geometry: call `rasterize()` with the coarse shape, `strut_radius=radius/8`, and `coordinate_scale=1/8`.
4. For every radius/threshold pair, calculate foreground overlap, Dice, and IoU against `coarse_scan >= threshold`.
5. Select the pair with the highest Dice, breaking ties with IoU. Record the coarse factor and all candidate settings.
6. Rasterize the selected geometry at the original CT shape using the selected full-resolution radius. Save a uint8 TIFF mask with values `0` and `255`.
7. Save a JSON report beside the mask containing input paths, selected radius, evaluation threshold, coarse factor, Dice, IoU, and full-resolution foreground voxel count.

## Output naming

Use:

`outputs/<stem>_best_raster.tif`

and

`outputs/<stem>_best_raster.json`

Do not overwrite an existing result without checking whether it belongs to the same input pair. Prefer a unique suffix for repeated runs.

## Interpretation

Report that the score is a coarse-grid agreement score unless a full-resolution comparison was performed. A low score can reflect CT segmentation ambiguity, partial-volume effects, disconnected struts, or imperfect registration; do not present Dice as ground truth accuracy.

## Failure handling

- If the JSON geometry does not overlap the target volume, verify axis order, coordinate scale, and coordinate origin before changing them.
- If full-resolution rasterization is too memory-intensive, retain the coarse optimization and write the final mask directly, avoiding multiple full-size masks in memory.
- If the CT is not already registered to the JSON, stop and report that registration must be resolved before interpreting the optimization.
