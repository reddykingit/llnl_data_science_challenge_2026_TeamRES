#!/usr/bin/env python3
"""Rasterize strut geometry and optionally register it to a CT volume."""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tf
from scipy.ndimage import binary_dilation, center_of_mass
from skimage.draw import line_nd
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation


def load_geometry(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    junctions = {j["id"]: np.asarray(j["position"], dtype=float) for j in data["junctions"]}
    return junctions, data["struts"]


def _vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.size == 1:
        a = np.repeat(a, 3)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain three finite numbers")
    return a


def _axis_permutation(axis_order):
    order = str(axis_order).lower().replace(",", "").replace(" ", "")
    if sorted(order) != ["x", "y", "z"]:
        raise ValueError("axis_order must be a permutation of xyz (for example xyz or zyx)")
    return [order.index(c) for c in "xyz"]


def _sphere_structure(radius):
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("strut_radius must be a finite non-negative number")
    r = int(np.ceil(radius))
    grid = np.indices((2 * r + 1,) * 3) - r
    return np.sum(grid * grid, axis=0) <= radius * radius + 1e-12


def rasterize(json_path, target_shape, strut_radius=1.0, axis_order="xyz",
              coordinate_scale=1.0, coordinate_origin=0.0):
    """Return a boolean z,y,x raster, safely clipped to ``target_shape``."""
    shape = tuple(int(v) for v in target_shape)
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError("target_shape must contain three positive dimensions")
    perm = _axis_permutation(axis_order)
    scale = _vector(coordinate_scale, "coordinate_scale")
    origin = _vector(coordinate_origin, "coordinate_origin")
    junctions, struts = load_geometry(json_path)
    ideal = np.zeros(shape, dtype=bool)
    for s in struts:
        try:
            p1, p2 = junctions[s["junction0"]], junctions[s["junction1"]]
        except KeyError as exc:
            raise ValueError(f"strut references unknown junction: {exc}") from exc
        p1 = p1[perm] * scale[perm] + origin[perm]
        p2 = p2[perm] * scale[perm] + origin[perm]
        # Convert transformed x,y,z coordinates to volume z,y,x indices.
        start, end = p1[[2, 1, 0]], p2[[2, 1, 0]]
        coords = np.asarray(line_nd(start, end, endpoint=True), dtype=int)
        valid = np.all((coords >= 0) & (coords < np.asarray(shape)[:, None]), axis=0)
        if np.any(valid):
            ideal[tuple(coords[:, valid])] = True
    if strut_radius > 0:
        ideal = binary_dilation(ideal, structure=_sphere_structure(float(strut_radius)))
    if not ideal.any():
        raise ValueError("transformed geometry does not overlap the target volume")
    return ideal


def _threshold_value(scan, threshold):
    if threshold is not None:
        return float(threshold)
    sample = np.asarray(scan)
    if sample.size == 0:
        raise ValueError("scan is empty")
    try:
        return float(threshold_otsu(sample))
    except ValueError:
        return float(np.mean(sample))


def compute_com_shift(scan, ideal, threshold=None):
    binary = np.asarray(scan) > _threshold_value(scan, threshold)
    if not binary.any() or not ideal.any():
        raise ValueError("cannot compute COM for an empty foreground")
    return np.asarray(center_of_mass(binary)) - np.asarray(center_of_mass(ideal))


def apply_integer_shift(volume, offset):
    """Shift with zero fill (never periodic wraparound)."""
    out = np.zeros_like(volume)
    d = np.rint(offset).astype(int)
    # Simplify slices using overlap bounds; empty shifts are valid and remain zero.
    src, dst = [], []
    for size, n in zip(volume.shape, d):
        lo, hi = max(0, -n), min(size, size - n)
        dlo, dhi = max(0, n), min(size, size + n)
        src.append(slice(lo, hi)); dst.append(slice(dlo, dhi))
    if all(a.start < a.stop for a in src):
        out[tuple(dst)] = volume[tuple(src)]
    return out


def register_translation(scan, ideal, threshold=None, downsample=4):
    """Estimate ideal->scan translation using downsampled phase correlation."""
    if int(downsample) < 1:
        raise ValueError("downsample must be a positive integer")
    factor = int(downsample)
    scan_binary = (np.asarray(scan) > _threshold_value(scan, threshold)).astype(np.float32)
    ideal_binary = np.asarray(ideal, dtype=np.float32)
    if not scan_binary.any() or not ideal_binary.any():
        raise ValueError("cannot register empty foreground volumes")
    a, b = scan_binary[::factor, ::factor, ::factor], ideal_binary[::factor, ::factor, ::factor]
    offset_small, error, phasediff = phase_cross_correlation(a, b, upsample_factor=1)
    offset = np.asarray(offset_small) * factor
    if not np.all(np.isfinite(offset)):
        raise ValueError("phase correlation returned an invalid translation")
    return offset, {"phase_error": float(error), "phase_difference": float(phasediff)}


def main(json_path, tif_path, output_tif, threshold=None, align_com=False,
         strut_radius=1.0, axis_order="xyz", coordinate_scale=1.0,
         coordinate_origin=0.0, registration="none", downsample=4,
         report_json=None):
    scan = tf.imread(tif_path)
    print(f"Reference scan shape: {scan.shape}, dtype: {scan.dtype}")
    ideal = rasterize(json_path, scan.shape, strut_radius, axis_order, coordinate_scale, coordinate_origin)
    method = "com" if align_com else registration
    metrics = {}
    offset = np.zeros(3, dtype=float)
    if method == "correlation":
        try:
            offset, metrics = register_translation(scan, ideal, threshold, downsample)
        except ValueError as exc:
            print(f"Correlation registration unavailable ({exc}); falling back to COM.")
            method = "com-fallback"
            offset = compute_com_shift(scan, ideal, threshold)
    elif method in ("com", "com-fallback"):
        offset = compute_com_shift(scan, ideal, threshold)
    elif method != "none":
        raise ValueError("registration must be none, correlation, or com")
    aligned = apply_integer_shift(ideal, offset)
    aligned_uint8 = (aligned * 255).astype(np.uint8)
    Path(output_tif).parent.mkdir(parents=True, exist_ok=True)
    tf.imwrite(output_tif, aligned_uint8)
    print(f"Translation (z, y, x): {np.rint(offset).astype(int).tolist()}")
    print(f"Translation (x, y, z): {np.rint(offset)[::-1].astype(int).tolist()}")
    print(f"Total raster strut voxels: {int(aligned.sum()):,}")
    if report_json:
        report = {"json_path": str(json_path), "tif_path": str(tif_path), "output_tif": str(output_tif),
                  "volume_shape": list(scan.shape), "volume_dtype": str(scan.dtype),
                  "threshold": _threshold_value(scan, threshold), "registration_method": method,
                  "downsample": int(downsample), "axis_order": axis_order,
                  "coordinate_scale": _vector(coordinate_scale, "coordinate_scale").tolist(),
                  "coordinate_origin": _vector(coordinate_origin, "coordinate_origin").tolist(),
                  "strut_radius": float(strut_radius), "translation_zyx": offset.tolist(),
                  "translation_xyz": offset[::-1].tolist(), "ideal_foreground": int(ideal.sum()),
                  "aligned_foreground": int(aligned.sum()), "overlap_foreground": int(np.logical_and(aligned, scan > _threshold_value(scan, threshold)).sum()),
                  "registration_metrics": metrics}
        Path(report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(report_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json-path", required=True); p.add_argument("--tif-path", required=True); p.add_argument("--output-tif", required=True)
    p.add_argument("--threshold", type=float); p.add_argument("--align-com", action="store_true")
    p.add_argument("--strut-radius", type=float, default=1.0); p.add_argument("--axis-order", default="xyz")
    p.add_argument("--coordinate-scale", nargs="+", type=float, default=[1.0]); p.add_argument("--coordinate-origin", nargs="+", type=float, default=[0.0])
    p.add_argument("--registration", choices=["none", "correlation", "com"], default="none")
    p.add_argument("--downsample", type=int, default=4); p.add_argument("--report-json")
    a = p.parse_args()
    if len(a.coordinate_scale) not in (1, 3) or len(a.coordinate_origin) not in (1, 3):
        p.error("coordinate-scale and coordinate-origin require one or three values")
    main(a.json_path, a.tif_path, a.output_tif, a.threshold, a.align_com, a.strut_radius, a.axis_order,
         a.coordinate_scale, a.coordinate_origin, a.registration, a.downsample, a.report_json)
