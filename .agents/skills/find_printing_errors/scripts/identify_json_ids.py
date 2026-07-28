#!/usr/bin/env python3
"""Identify lattice JSON IDs supported by nonzero voxels in a TIFF volume."""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from skimage.draw import line_nd


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.size == 1:
        array = np.repeat(array, 3)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain one or three finite numbers")
    return array


def _axis_permutation(axis_order):
    order = str(axis_order).lower().replace(",", "").replace(" ", "")
    if sorted(order) != ["x", "y", "z"]:
        raise ValueError("axis_order must be a permutation of xyz")
    return [order.index(axis) for axis in "xyz"]


def _sphere_offsets(radius):
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radii must be finite and non-negative")
    limit = int(np.ceil(radius))
    grid = np.indices((2 * limit + 1,) * 3).reshape(3, -1).T - limit
    return grid[np.sum(grid * grid, axis=1) <= radius * radius + 1e-12]


def _volume_coordinates(points_xyz, shape, axis_order, scale, origin):
    """Transform xyz points to integer zyx voxel coordinates."""
    points = np.asarray(points_xyz, dtype=float)
    points = points[:, _axis_permutation(axis_order)] * scale[_axis_permutation(axis_order)]
    points += origin[_axis_permutation(axis_order)]
    return np.rint(points[:, [2, 1, 0]]).astype(int)


def _in_bounds(coords, shape):
    return np.all((coords >= 0) & (coords < np.asarray(shape)), axis=1)


def _strut_voxels(strut, junctions, shape, axis_order, scale, origin, radius):
    try:
        endpoints = np.asarray(
            [junctions[strut["junction0"]], junctions[strut["junction1"]]], dtype=float
        )
    except KeyError as exc:
        raise ValueError(f"strut {strut.get('id', '<unknown>')} references unknown junction {exc}") from exc

    transformed = endpoints[:, _axis_permutation(axis_order)] * scale[_axis_permutation(axis_order)]
    transformed += origin[_axis_permutation(axis_order)]
    line = np.asarray(line_nd(transformed[0][[2, 1, 0]], transformed[1][[2, 1, 0]], endpoint=True)).T
    offsets = _sphere_offsets(radius)
    expanded = (line[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    in_bounds = _in_bounds(expanded, shape)
    voxels = np.unique(expanded[in_bounds], axis=0)
    return voxels, line, offsets, bool(np.all(in_bounds))


def _white_mask(volume, mode, threshold=None):
    if mode == "nonzero":
        return volume != 0
    if mode == "exact":
        return volume == np.iinfo(volume.dtype).max if np.issubdtype(volume.dtype, np.integer) else volume == 1
    if mode == "threshold":
        if threshold is None:
            raise ValueError("white_threshold is required when white_mode='threshold'")
        return volume >= threshold
    raise ValueError("white_mode must be nonzero, exact, or threshold")


def _line_metrics(line, offsets, white, shape):
    """Return centerline coverage, largest uncovered run, and endpoint support."""
    supported = []
    for point in line:
        voxels = point[None, :] + offsets
        voxels = voxels[_in_bounds(voxels, shape)]
        supported.append(bool(len(voxels) and np.any(white[tuple(voxels.T)])))
    supported = np.asarray(supported, dtype=bool)
    if not len(supported):
        return 0.0, 0, (False, False)
    max_gap = 0
    current_gap = 0
    for is_supported in supported:
        if is_supported:
            max_gap = max(max_gap, current_gap)
            current_gap = 0
        else:
            current_gap += 1
    max_gap = max(max_gap, current_gap)
    return float(np.mean(supported)), max_gap, (bool(supported[0]), bool(supported[-1]))


def _preset_values(name):
    presets = {
        "current": {"strut_radius": 1.0, "strut_fraction": 0.5, "junction_radius": 1.0,
                    "junction_min_support": 3, "junction_min_fraction": 0.0, "max_gap": None,
                    "require_endpoints": False, "reject_clipped": False},
        "strict": {"strut_radius": 0.0, "strut_fraction": 0.9, "junction_radius": 0.0,
                    "junction_min_support": 1, "junction_min_fraction": 1.0, "max_gap": 0,
                    "require_endpoints": True, "reject_clipped": False},
        "exact": {"strut_radius": 0.0, "strut_fraction": 1.0, "junction_radius": 0.0,
                   "junction_min_support": 1, "junction_min_fraction": 1.0, "max_gap": 0,
                   "require_endpoints": True, "reject_clipped": True},
    }
    if name not in presets:
        raise ValueError("strictness must be current, strict, exact, or custom")
    return presets[name].copy()


def identify_ids(
    json_path,
    tif_path,
    strut_radius=1.0,
    strut_fraction=0.5,
    junction_radius=1.0,
    junction_min_support=3,
    axis_order="xyz",
    coordinate_scale=1.0,
    coordinate_origin=0.0,
    strictness="custom",
    white_mode="nonzero",
    white_threshold=None,
    max_gap=None,
    require_endpoints=None,
    reject_clipped=None,
    junction_min_fraction=None,
    audit_path=None,
):
    """Return strut and junction IDs supported by a nonzero TIFF mask."""
    preset = _preset_values(strictness) if strictness != "custom" else {}
    if strictness != "custom":
        strut_radius = preset["strut_radius"]
        strut_fraction = preset["strut_fraction"]
        junction_radius = preset["junction_radius"]
        junction_min_support = preset["junction_min_support"]
        if max_gap is None:
            max_gap = preset["max_gap"]
        if require_endpoints is None:
            require_endpoints = preset["require_endpoints"]
        if reject_clipped is None:
            reject_clipped = preset["reject_clipped"]
        if junction_min_fraction is None:
            junction_min_fraction = preset["junction_min_fraction"]
    require_endpoints = bool(require_endpoints)
    reject_clipped = bool(reject_clipped)
    if max_gap is not None and int(max_gap) < 0:
        raise ValueError("max_gap must be non-negative or omitted")
    if junction_min_fraction is None:
        junction_min_fraction = 0.0
    if not 0 <= junction_min_fraction <= 1:
        raise ValueError("junction_min_fraction must be between 0 and 1")
    if not 0 <= strut_fraction <= 1:
        raise ValueError("strut_fraction must be between 0 and 1")
    if int(junction_min_support) < 1:
        raise ValueError("junction_min_support must be at least 1")

    volume = np.asarray(tifffile.imread(tif_path))
    if volume.ndim != 3:
        raise ValueError(f"TIFF must be a 3D volume, got shape {volume.shape}")
    white = _white_mask(volume, white_mode, white_threshold)

    with open(json_path, encoding="utf-8") as handle:
        geometry = json.load(handle)
    if not isinstance(geometry.get("junctions"), list) or not isinstance(geometry.get("struts"), list):
        raise ValueError("JSON must contain 'junctions' and 'struts' arrays")

    scale = _vector(coordinate_scale, "coordinate_scale")
    origin = _vector(coordinate_origin, "coordinate_origin")
    permutation = _axis_permutation(axis_order)
    junctions = {junction["id"]: junction["position"] for junction in geometry["junctions"]}

    strut_ids = []
    strut_audit = {}
    for strut in geometry["struts"]:
        voxels, line, offsets, fully_in_bounds = _strut_voxels(
            strut, junctions, volume.shape, axis_order, scale, origin, float(strut_radius)
        )
        white_voxels = int(np.count_nonzero(white[tuple(voxels.T)])) if len(voxels) else 0
        white_fraction = white_voxels / len(voxels) if len(voxels) else 0.0
        line_fraction, line_max_gap, endpoint_support = _line_metrics(line, offsets, white, volume.shape)
        identified = bool(
            len(voxels) > 0
            and white_fraction >= strut_fraction
            and line_fraction >= strut_fraction
            and (max_gap is None or line_max_gap <= int(max_gap))
            and (not require_endpoints or all(endpoint_support))
            and (not reject_clipped or fully_in_bounds)
        )
        strut_audit[str(strut["id"])] = {
            "identified": identified,
            "white_fraction": white_fraction,
            "white_voxels": white_voxels,
            "total_voxels": int(len(voxels)),
            "line_fraction": line_fraction,
            "max_gap": line_max_gap,
            "endpoint_support": list(endpoint_support),
            "clipped": not fully_in_bounds,
        }
        if identified:
            strut_ids.append(strut["id"])

    offsets = _sphere_offsets(float(junction_radius))
    junction_ids = []
    junction_audit = {}
    for junction in geometry["junctions"]:
        position = np.asarray(junction["position"], dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"junction {junction.get('id', '<unknown>')} has an invalid position")
        transformed = position[permutation] * scale[permutation] + origin[permutation]
        center = np.rint(transformed[[2, 1, 0]]).astype(int)
        voxels = center[None, :] + offsets
        voxels = voxels[_in_bounds(voxels, volume.shape)]
        support = np.count_nonzero(white[tuple(voxels.T)]) if len(voxels) else 0
        support_fraction = support / len(voxels) if len(voxels) else 0.0
        identified = bool(support >= int(junction_min_support) and support_fraction >= junction_min_fraction)
        junction_audit[str(str(junction["id"]))] = {
            "identified": identified,
            "support": int(support),
            "total_voxels": int(len(voxels)),
            "support_fraction": support_fraction,
            "required_support": int(junction_min_support),
            "required_fraction": junction_min_fraction,
        }
        if identified:
            junction_ids.append(junction["id"])

    result = {"strut_ids": strut_ids, "junction_ids": junction_ids}
    if audit_path:
        audit = {
            "settings": {
                "strictness": strictness, "white_mode": white_mode,
                "white_threshold": white_threshold, "strut_radius": strut_radius,
                "strut_fraction": strut_fraction, "junction_radius": junction_radius,
                "junction_min_support": junction_min_support,
                "junction_min_fraction": junction_min_fraction, "max_gap": max_gap,
                "require_endpoints": require_endpoints, "reject_clipped": reject_clipped,
            },
            "struts": strut_audit,
            "junctions": junction_audit,
        }
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_path).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--tif-path", required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--strut-radius", type=float, default=1.0)
    parser.add_argument("--strut-fraction", type=float, default=0.5)
    parser.add_argument("--junction-radius", type=float, default=1.0)
    parser.add_argument("--junction-min-support", type=int, default=3)
    parser.add_argument("--axis-order", default="xyz")
    parser.add_argument("--coordinate-scale", nargs="+", type=float, default=[1.0])
    parser.add_argument("--coordinate-origin", nargs="+", type=float, default=[0.0])
    parser.add_argument("--strictness", choices=["current", "strict", "exact", "custom"], default="current")
    parser.add_argument("--white-mode", choices=["nonzero", "exact", "threshold"], default="nonzero")
    parser.add_argument("--white-threshold", type=float)
    parser.add_argument("--max-gap", type=int)
    parser.add_argument("--require-endpoints", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reject-clipped", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--junction-min-fraction", type=float)
    parser.add_argument("--audit-output", help="Optional detailed per-ID diagnostics JSON path")
    args = parser.parse_args()
    if len(args.coordinate_scale) not in (1, 3) or len(args.coordinate_origin) not in (1, 3):
        parser.error("coordinate-scale and coordinate-origin require one or three values")

    result = identify_ids(
        args.json_path,
        args.tif_path,
        args.strut_radius,
        args.strut_fraction,
        args.junction_radius,
        args.junction_min_support,
        args.axis_order,
        args.coordinate_scale,
        args.coordinate_origin,
        args.strictness,
        args.white_mode,
        args.white_threshold,
        args.max_gap,
        args.require_endpoints,
        args.reject_clipped,
        args.junction_min_fraction,
        args.audit_output,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Identified {len(result['strut_ids'])} struts and {len(result['junction_ids'])} junctions")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
