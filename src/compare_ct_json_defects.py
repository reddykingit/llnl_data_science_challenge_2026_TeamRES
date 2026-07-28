"""Compare registered CT material evidence with expected JSON lattice struts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu


def load_graph(path: Path) -> dict:
    """Load a registered graph; recover files with a stray preamble."""
    text = path.read_text(encoding="utf-8")
    try:
        graph = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('"junctions"')
        if start < 0:
            raise ValueError(f"Cannot find graph data in {path}")
        graph = json.loads("{" + text[start:])
    needed = {"junctions", "struts", "unit_cells"}
    if not needed.issubset(graph):
        raise ValueError(f"Graph missing: {sorted(needed - set(graph))}")
    return graph


def estimate_threshold(volume: np.ndarray, count: int = 24) -> float:
    """Estimate one global raw CT threshold from decimated volume samples."""
    indices = np.linspace(0, volume.shape[0] - 1, min(count, volume.shape[0]), dtype=int)
    samples = [np.asarray(volume[z, ::4, ::4]).ravel() for z in indices]
    return float(threshold_otsu(np.concatenate(samples)))


def xyz_to_zyx(point: np.ndarray) -> np.ndarray:
    """Registered graph is XYZ; TIFF arrays are ZYX."""
    return np.asarray((point[2], point[1], point[0]), dtype=float)


def material_fraction(volume: np.ndarray, centre_zyx: np.ndarray, radius: int, threshold: float) -> float:
    centre = np.rint(centre_zyx).astype(int)
    lo = np.maximum(centre - radius, 0)
    hi = np.minimum(centre + radius + 1, volume.shape)
    if np.any(lo >= hi):
        return 0.0
    patch = np.asarray(volume[tuple(slice(a, b) for a, b in zip(lo, hi))])
    return float(np.mean(patch >= threshold))


def longest_unsupported_fraction(supported: np.ndarray) -> float:
    changes = np.diff(np.r_[False, ~supported, False].astype(np.int8))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return float(np.max(stops - starts, initial=0) / max(len(supported), 1))


def inspect_strut(volume, start_xyz, end_xyz, threshold, samples, radius,
                  support_occupancy, missing_max_coverage=0.20,
                  broken_min_coverage=0.65,
                  broken_min_gap_fraction=0.20):
    # Ignore end regions: their high occupancy is expected from junctions and
    # should not conceal a break in the middle of a strut.
    t = np.linspace(0.12, 0.88, samples)
    points = start_xyz[None, :] + t[:, None] * (end_xyz - start_xyz)[None, :]
    local = np.asarray([material_fraction(volume, xyz_to_zyx(p), radius, threshold) for p in points])
    # A thin or diagonal cylindrical strut does not fill most of its enclosing
    # voxel cube even when it is intact.  The previous 0.65 cutoff therefore
    # labelled clearly visible members as unsupported.
    supported = local >= support_occupancy
    coverage = float(supported.mean())
    gap = longest_unsupported_fraction(supported)
    if coverage < missing_max_coverage:
        label = "missing_strut"
    elif coverage < broken_min_coverage or gap >= broken_min_gap_fraction:
        label = "broken_strut"
    else:
        label = "intact"
    return {
        "classification": label,
        "coverage": round(coverage, 4),
        "mean_local_occupancy": round(float(local.mean()), 4),
        "min_local_occupancy": round(float(local.min()), 4),
        "longest_unsupported_fraction": round(gap, 4),
    }


def compare(tiff: Path, graph_file: Path, threshold: float | None, samples: int,
            radius: int, node_radius: int, max_struts: int | None,
            support_occupancy: float = 0.20,
            node_occupancy: float = 0.15,
            missing_max_coverage: float = 0.20,
            broken_min_coverage: float = 0.65,
            broken_min_gap_fraction: float = 0.20):
    volume = tifffile.memmap(tiff)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF, got {volume.shape}")
    graph = load_graph(graph_file)
    automatic_threshold = threshold is None
    threshold = estimate_threshold(volume) if threshold is None else threshold
    nodes = {node["id"]: np.asarray(node["position"], dtype=float) for node in graph["junctions"]}

    rows = []
    for strut in graph["struts"][:max_struts]:
        result = inspect_strut(
            volume, nodes[strut["junction0"]], nodes[strut["junction1"]],
            threshold, samples, radius, support_occupancy,
            missing_max_coverage, broken_min_coverage,
            broken_min_gap_fraction)
        rows.append({"strut_id": strut["id"], "junction0": strut["junction0"], "junction1": strut["junction1"], **result})

    bad_nodes = []
    for node in graph["junctions"]:
        occupancy = material_fraction(volume, xyz_to_zyx(np.asarray(node["position"], dtype=float)), node_radius, threshold)
        if occupancy < node_occupancy:
            bad_nodes.append({"junction_id": node["id"], "classification": "missing_or_broken_node", "local_occupancy": round(occupancy, 4), "position_xyz": node["position"]})

    missing = [row for row in rows if row["classification"] == "missing_strut"]
    broken = [row for row in rows if row["classification"] == "broken_strut"]
    report = {
        "inputs": {"tiff": str(tiff), "registered_json": str(graph_file)},
        "ct": {"shape_zyx": list(volume.shape), "dtype": str(volume.dtype), "raw_threshold": threshold, "threshold_source": "Otsu" if automatic_threshold else "user"},
        "parameters": {
            "samples_per_strut": samples,
            "support_radius_voxels": radius,
            "support_occupancy_cutoff": support_occupancy,
            "missing_strut_max_coverage": missing_max_coverage,
            "broken_strut_min_coverage": broken_min_coverage,
            "broken_strut_min_gap_fraction": broken_min_gap_fraction,
            "node_radius_voxels": node_radius,
            "node_occupancy_cutoff": node_occupancy,
        },
        "summary": {"expected_struts_examined": len(rows), "missing_struts": len(missing), "broken_struts": len(broken), "missing_or_broken_nodes": len(bad_nodes)},
        "defects": {"missing_struts": missing, "broken_struts": broken, "missing_or_broken_nodes": bad_nodes},
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiff", type=Path, help="registered 3D CT TIFF")
    parser.add_argument("registered_json", type=Path, help="expected registered strut graph")
    parser.add_argument("--output", type=Path, default=Path("defect_comparison_report.json"))
    parser.add_argument("--ct-threshold", type=float, default=None, help="raw CT material threshold; default: Otsu")
    parser.add_argument("--samples-per-strut", type=int, default=25)
    parser.add_argument("--support-radius", type=int, default=2)
    parser.add_argument("--support-occupancy", type=float, default=0.20)
    parser.add_argument("--node-radius", type=int, default=3)
    parser.add_argument("--node-occupancy", type=float, default=0.15)
    parser.add_argument("--max-struts", type=int, default=None, help="limit work for tuning/testing")
    args = parser.parse_args()
    report, rows = compare(
        args.tiff, args.registered_json, args.ct_threshold,
        args.samples_per_strut, args.support_radius, args.node_radius,
        args.max_struts, args.support_occupancy, args.node_occupancy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["strut_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
