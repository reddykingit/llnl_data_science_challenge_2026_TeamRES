"""Refine a nominal lattice JSON registration against a 3D CT volume.

The residual transform includes XYZ translation and scale plus X/Y-to-Z
shear.  The shear terms model the common case where a printed lattice layer
is slightly tilted relative to the axial CT slices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import differential_evolution, minimize

from compare_ct_json_defects import estimate_threshold, load_graph


def sample_graph_points(graph: dict, maximum: int = 6000) -> np.ndarray:
    """Return representative XYZ points along nodes and member centerlines."""
    positions = {
        int(node["id"]): np.asarray(node["position"], dtype=float)
        for node in graph["junctions"]
    }
    points = list(positions.values())
    for strut in graph["struts"]:
        p0 = positions[int(strut["junction0"])]
        p1 = positions[int(strut["junction1"])]
        for t in (0.25, 0.5, 0.75):
            points.append(p0 + t * (p1 - p0))
    points = np.asarray(points)
    if len(points) > maximum:
        rng = np.random.default_rng(2026)
        points = points[rng.choice(len(points), maximum, replace=False)]
    return points


def apply_residual(points_xyz: np.ndarray, parameters: np.ndarray,
                   pivot_xyz: np.ndarray) -> np.ndarray:
    """Apply translation, scale, and axial-plane tilt to XYZ coordinates."""
    tx, ty, tz, sx, sy, sz, dzdx, dzdy = parameters
    centred = np.asarray(points_xyz, dtype=float) - pivot_xyz
    transformed = centred * np.asarray((sx, sy, sz))
    transformed[:, 2] += dzdx * centred[:, 0] + dzdy * centred[:, 1]
    return transformed + pivot_xyz + np.asarray((tx, ty, tz))


def ct_scores(volume: np.ndarray, points_xyz: np.ndarray,
              low: float, high: float) -> np.ndarray:
    """Sample normalized CT evidence at continuous registered coordinates."""
    # Manual trilinear sampling avoids scipy converting the entire big-endian
    # TIFF memmap to native byte order on every optimizer evaluation.
    coords = points_xyz[:, (2, 1, 0)]
    base = np.floor(coords).astype(int)
    frac = coords - base
    valid = np.all(base >= 0, axis=1) & np.all(
        base + 1 < np.asarray(volume.shape), axis=1)
    values = np.full(len(points_xyz), low, dtype=float)
    if np.any(valid):
        b = base[valid]
        f = frac[valid]
        interpolated = np.zeros(len(b), dtype=float)
        for oz in (0, 1):
            for oy in (0, 1):
                for ox in (0, 1):
                    weight = (
                        (f[:, 0] if oz else 1 - f[:, 0])
                        * (f[:, 1] if oy else 1 - f[:, 1])
                        * (f[:, 2] if ox else 1 - f[:, 2]))
                    interpolated += weight * np.asarray(
                        volume[b[:, 0] + oz, b[:, 1] + oy, b[:, 2] + ox],
                        dtype=float)
        values[valid] = interpolated
    return np.clip((values - low) / max(high - low, 1.0), 0.0, 1.0)


def refine_registration(volume: np.ndarray, graph: dict,
                        maximum_points: int = 6000) -> tuple[np.ndarray, dict]:
    points = sample_graph_points(graph, maximum_points)
    pivot = np.median(points, axis=0)

    indices = np.linspace(
        0, volume.shape[0] - 1, min(24, volume.shape[0]), dtype=int)
    sampled = np.concatenate([
        np.asarray(volume[z, ::8, ::8]).ravel() for z in indices])
    low, high = np.percentile(sampled, (55, 99.5))

    def objective(parameters):
        moved = apply_residual(points, parameters, pivot)
        scores = ct_scores(volume, moved, low, high)
        # A trimmed average tolerates genuinely missing members while still
        # rewarding a transform that places most design centerlines in metal.
        keep = scores >= np.quantile(scores, 0.08)
        return -float(scores[keep].mean())

    bounds = [
        (-12.0, 12.0), (-12.0, 12.0), (-12.0, 12.0),
        (0.97, 1.03), (0.97, 1.03), (0.97, 1.03),
        (-0.03, 0.03), (-0.03, 0.03),
    ]
    # A modest global pass avoids locking onto the neighboring periodic layer;
    # the local pass then gives sub-voxel refinement.
    coarse = differential_evolution(
        objective, bounds, seed=2026, popsize=7, maxiter=18,
        polish=False, workers=1)
    result = minimize(
        objective, coarse.x, method="Powell", bounds=bounds,
        options={"maxiter": 180, "xtol": 1e-4, "ftol": 1e-5})

    initial_score = -objective(np.asarray(
        (0, 0, 0, 1, 1, 1, 0, 0), dtype=float))
    final_score = -objective(result.x)
    metadata = {
        "method": "CT centerline intensity residual affine",
        "parameters": {
            key: float(value) for key, value in zip(
                ("translation_x", "translation_y", "translation_z",
                 "scale_x", "scale_y", "scale_z",
                 "tilt_dz_dx", "tilt_dz_dy"), result.x)
        },
        "pivot_xyz": pivot.tolist(),
        "sample_points": int(len(points)),
        "initial_alignment_score": initial_score,
        "refined_alignment_score": final_score,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }
    return result.x, metadata


def register(tiff_path: Path, graph_path: Path, output_path: Path,
             maximum_points: int = 6000) -> dict:
    volume = tifffile.memmap(tiff_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF, got {volume.shape}")
    graph = load_graph(graph_path)
    points = np.asarray(
        [node["position"] for node in graph["junctions"]], dtype=float)
    pivot = np.median(sample_graph_points(graph, maximum_points), axis=0)
    parameters, metadata = refine_registration(
        volume, graph, maximum_points)
    moved = apply_residual(points, parameters, pivot)
    for node, position in zip(graph["junctions"], moved):
        node["position"] = position.tolist()
    graph["ct_registration"] = {
        **metadata,
        "source_json": str(graph_path),
        "source_tiff": str(tiff_path),
        "ct_threshold_reference": estimate_threshold(volume),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return graph["ct_registration"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiff", type=Path)
    parser.add_argument("json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-points", type=int, default=6000)
    args = parser.parse_args()
    metadata = register(
        args.tiff, args.json, args.output, args.maximum_points)
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
