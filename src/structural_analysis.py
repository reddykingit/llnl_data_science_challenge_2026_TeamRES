from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _normalize_plane(plane: str) -> str:
    normalized = plane.strip().lower()
    if normalized not in {"xy", "xz", "yz"}:
        raise ValueError("support_plane and load_plane must be one of: xy, xz, yz")
    return normalized


def _plane_axes(plane: str) -> tuple[int, int]:
    if plane == "xy":
        return 0, 1
    if plane == "xz":
        return 0, 2
    return 1, 2


def _normal_axis(plane: str) -> int:
    return {"xy": 2, "xz": 1, "yz": 0}[plane]


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def _get_graph_data(graph: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], tuple[str, str]]:
    if isinstance(graph.get("junctions"), list) and isinstance(graph.get("struts"), list):
        return "design_graph", graph["junctions"], graph["struts"], ("junction0", "junction1")
    if isinstance(graph.get("nodes"), list) and isinstance(graph.get("members"), list):
        return "skeleton_graph", graph["nodes"], graph["members"], ("a", "b")
    raise ValueError("Registered graph JSON must contain either junctions/struts or nodes/members")


def _get_node_position(node: dict[str, Any]) -> np.ndarray:
    if "position_xyz_voxels" in node:
        return np.asarray(node["position_xyz_voxels"], dtype=float)
    if "position_zyx_voxels" in node:
        return np.asarray(node["position_zyx_voxels"], dtype=float)[::-1]
    if "position_voxels" in node:
        return np.asarray(node["position_voxels"], dtype=float)[::-1]
    if "position" in node:
        return np.asarray(node["position"], dtype=float)
    raise ValueError(f"Node {node.get('id')} has no recognizable position field")


def _collect_strut_labels(report: dict[str, Any]) -> dict[Any, str]:
    labels: dict[Any, str] = {}
    if isinstance(report.get("struts"), list):
        for item in report["struts"]:
            sid = item.get("strut_id", item.get("id"))
            if sid is None:
                continue
            classification = item.get("defect") or item.get("classification") or item.get("label")
            if classification is None:
                continue
            labels[sid] = str(classification).lower()
        return labels

    defects = report.get("defects")
    if isinstance(defects, dict):
        for group in ("missing_struts", "broken_struts", "thinned_struts", "thickened_struts", "curved_struts", "twisted_struts"):
            items = defects.get(group)
            if not isinstance(items, list):
                continue
            for item in items:
                sid = item.get("strut_id", item.get("id"))
                if sid is None:
                    continue
                labels[sid] = group
        return labels

    return labels


def _defect_score(classification: str | None) -> float:
    if classification is None:
        return 0.0
    label = classification.strip().lower()
    if "missing" in label:
        return 1.0
    if "broken" in label:
        return 0.9
    if "thinned" in label or "thin" in label:
        return 0.7
    if "curved" in label or "bent" in label or "twisted" in label:
        return 0.6
    if "thick" in label:
        return 0.2
    if "intact" in label:
        return 0.0
    return 0.3


def _build_heatmap(points: np.ndarray, weights: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0:
        raise ValueError("No projection points were found for the heatmap")
    x = points[:, 0]
    y = points[:, 1]
    if np.ptp(x) == 0:
        x = x + np.linspace(-0.5, 0.5, x.size)
    if np.ptp(y) == 0:
        y = y + np.linspace(-0.5, 0.5, y.size)
    xi = np.linspace(float(x.min()), float(x.max()), resolution + 1)
    yi = np.linspace(float(y.min()), float(y.max()), resolution + 1)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xi, yi), weights=weights)
    return heatmap, xedges, yedges


def _save_heatmap_image(heatmap: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, support_plane: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])]
    im = ax.imshow(
        heatmap.T,
        cmap="inferno",
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax.set_title(f"Structural weak-zone heatmap ({support_plane.upper()} plane)")
    labels = {"xy": ("X", "Y"), "xz": ("X", "Z"), "yz": ("Y", "Z")}
    ax.set_xlabel(labels[support_plane][0])
    ax.set_ylabel(labels[support_plane][1])
    fig.colorbar(im, ax=ax, label="weakness score")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def structural_weakness_analysis(
    defect_report_filepath: str,
    registered_graph_filepath: str,
    support_plane: str = "xy",
    load_plane: str = "xy",
    output_dir: str = "",
    heatmap_resolution: int = 256,
    youngs_modulus: float | None = None,
    yield_strength: float | None = None,
    density: float | None = None,
) -> dict[str, Any]:
    """Estimate weak zones from CT defect findings and lattice geometry.

    Args:
        defect_report_filepath: Path to a JSON defect report produced by the
            CT defect analysis workflow.
        registered_graph_filepath: Path to the registered lattice graph JSON.
        support_plane: Plane that defines the fixed support, one of xy, xz, yz.
        load_plane: Plane normal that defines the applied direction.
        output_dir: Optional directory where outputs are written.
        heatmap_resolution: Resolution for the returned weak-zone heatmap.
        youngs_modulus: Optional elastic modulus in SI units.
        yield_strength: Optional yield strength in SI units.
        density: Optional material density in SI units.

    Returns:
        A dictionary with summary statistics and output file paths.
    """
    defect_path = Path(defect_report_filepath)
    graph_path = Path(registered_graph_filepath)
    if not defect_path.is_file():
        return {"error": f"Defect report not found: {defect_path}"}
    if not graph_path.is_file():
        return {"error": f"Registered graph not found: {graph_path}"}

    try:
        support_plane = _normalize_plane(support_plane)
        load_plane = _normalize_plane(load_plane)
    except ValueError as error:
        return {"error": str(error)}

    if not 32 <= heatmap_resolution <= 1024:
        return {"error": "heatmap_resolution must be between 32 and 1024"}

    report = _load_json(defect_path)
    graph = _load_json(graph_path)
    schema, nodes, struts, strut_keys = _get_graph_data(graph)

    node_positions: dict[Any, np.ndarray] = {}
    for node in nodes:
        node_id = node.get("id")
        if node_id is None:
            continue
        node_positions[node_id] = _get_node_position(node)

    if not node_positions:
        return {"error": "No node positions found in registered graph"}

    defect_labels = _collect_strut_labels(report)
    support_axis = _normal_axis(support_plane)
    load_axis = _normal_axis(load_plane)

    projected_axes = _plane_axes(support_plane)
    support_values = np.asarray([pos[support_axis] for pos in node_positions.values()])
    min_support = float(np.min(support_values))
    max_support = float(np.max(support_values))
    support_range = max(max_support - min_support, 1e-6)

    entries = []
    projection_points = []
    projection_weights = []
    support_node_ids = {nid for nid, pos in node_positions.items() if pos[support_axis] <= min_support + 1e-6}

    for strut in struts:
        a_id = strut.get(strut_keys[0])
        b_id = strut.get(strut_keys[1])
        if a_id not in node_positions or b_id not in node_positions:
            continue
        start = node_positions[a_id]
        end = node_positions[b_id]
        midpoint = 0.5 * (start + end)
        if support_axis == load_axis:
            height = float((midpoint[load_axis] - min_support) / support_range)
        else:
            height = float((midpoint[load_axis] - min_support) / support_range)
        height = np.clip(height, 0.0, 1.0)

        strut_id = strut.get("id", f"{a_id}-{b_id}")
        classification = defect_labels.get(strut_id)
        score = _defect_score(classification)
        score = float(min(max(score + 0.25 * height, 0.0), 1.0))

        projected = np.asarray([midpoint[projected_axes[0]], midpoint[projected_axes[1]]], dtype=float)
        projection_points.append(projected)
        projection_weights.append(score)
        entries.append({
            "strut_id": strut_id,
            "endpoints": [list(start.tolist()), list(end.tolist())],
            "classification": classification or "unknown",
            "weakness_score": round(score, 4),
            "height_fraction": round(height, 4),
            "support_contact": bool(a_id in support_node_ids or b_id in support_node_ids),
        })

    projection_points = np.asarray(projection_points, dtype=float)
    projection_weights = np.asarray(projection_weights, dtype=float)
    try:
        heatmap, xedges, yedges = _build_heatmap(projection_points, projection_weights, heatmap_resolution)
    except ValueError as error:
        return {"error": str(error)}

    output_root = Path(output_dir) if output_dir else defect_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "structural_weakness_analysis.json"
    heatmap_path = output_root / "structural_weakness_heatmap.png"

    summary = {
        "support_plane": support_plane,
        "load_plane": load_plane,
        "support_axis": support_axis,
        "load_axis": load_axis,
        "support_node_count": len(support_node_ids),
        "strut_count": len(entries),
        "weak_strut_count": int(np.count_nonzero(projection_weights >= 0.5)),
        "material_properties": {
            "youngs_modulus": youngs_modulus,
            "yield_strength": yield_strength,
            "density": density,
        },
    }

    output_data = {
        "schema": "structural_weakness_analysis",
        "version": 1,
        "defect_report": str(defect_path),
        "registered_graph": str(graph_path),
        "summary": summary,
        "support_plane": support_plane,
        "load_plane": load_plane,
        "support_axis": support_axis,
        "load_axis": load_axis,
        "projection_plane": support_plane,
        "entries": entries,
        "heatmap_image": str(heatmap_path),
    }
    report_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    _save_heatmap_image(heatmap, xedges, yedges, support_plane, heatmap_path)

    return {
        "output_report": str(report_path),
        "heatmap_image": str(heatmap_path),
        "strut_count": len(entries),
        "weak_strut_count": summary["weak_strut_count"],
    }
