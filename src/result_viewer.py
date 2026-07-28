"""Interactive 3D/2D viewer for registered CT lattice defect findings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


STRUT_CLASSES = (
    "missing_strut", "broken_strut", "thinned_strut",
    "thickened_strut", "curved_strut", "twisted_strut",
)
NODE_CLASSES = ("missing_node", "missing_or_broken_node")
COLORS = {
    "missing_strut": "crimson",
    "broken_strut": "darkorange",
    "thinned_strut": "gold",
    "thickened_strut": "deepskyblue",
    "curved_strut": "mediumorchid",
    "twisted_strut": "limegreen",
    "missing_node": "magenta",
    "missing_or_broken_node": "magenta",
    "intact": "seagreen",
}
DISPLAY_NAMES = {
    "missing_strut": "Missing strut",
    "broken_strut": "Broken strut",
    "thinned_strut": "Thinned strut",
    "thickened_strut": "Thickened strut",
    "curved_strut": "Curved strut",
    "twisted_strut": "Twisted strut",
    "missing_node": "Missing node",
    "missing_or_broken_node": "Missing node",
}


def _read_json(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('"junctions"')
        if start < 0:
            raise
        return json.loads("{" + text[start:])


def _resolve_graph(report, report_path, explicit_graph=None):
    candidate = explicit_graph or report.get("inputs", {}).get("registered_json")
    if not candidate:
        raise ValueError("A registered graph JSON is required for strut endpoints.")
    p = Path(candidate)
    if not p.is_absolute() and not p.exists():
        p = Path(report_path).resolve().parent / p
    if not p.is_file():
        raise ValueError(f"Registered graph not found: {p}")
    return _read_json(p), p


def load_findings(report_path, registered_graph_path=None):
    report = _read_json(report_path)
    graph, graph_path = _resolve_graph(report, report_path, registered_graph_path)
    positions = {int(n["id"]): np.asarray(n["position"], float)
                 for n in graph.get("junctions", [])}
    graph_struts = {int(s["id"]): s for s in graph.get("struts", [])}
    rows = []
    defects = report.get("defects", {})
    defect_keys = {
        "missing_struts": "missing_strut",
        "broken_struts": "broken_strut",
        "thinned_struts": "thinned_strut",
        "thickened_struts": "thickened_strut",
        "curved_struts": "curved_strut",
        "twisted_struts": "twisted_strut",
    }
    for kind, default_class in defect_keys.items():
        for item in defects.get(kind, []):
            sid = int(item["strut_id"])
            source = graph_struts.get(sid, item)
            a, b = int(source["junction0"]), int(source["junction1"])
            if a in positions and b in positions:
                rows.append({**item, "strut_id": sid,
                             "classification": item.get(
                                 "classification", default_class),
                             "p0": positions[a], "p1": positions[b]})
    bad_nodes = []
    for node_key, default_class in (
            ("missing_nodes", "missing_node"),
            ("missing_or_broken_nodes", "missing_or_broken_node")):
        for item in defects.get(node_key, []):
            nid = int(item.get("junction_id", item.get("id", -1)))
            point = np.asarray(
                item.get("position_xyz", positions.get(nid, [])), float)
            if point.shape == (3,):
                bad_nodes.append({
                    **item, "junction_id": nid, "point": point,
                    "classification": item.get(
                        "classification", default_class)})
    return report, graph, rows, bad_nodes, graph_path


def _load_criteria(report, report_path):
    candidate = report.get("inputs", {}).get("criteria")
    if not candidate or candidate == "built-in defaults":
        return {}
    path = Path(candidate)
    if not path.is_absolute() and not path.is_file():
        path = Path(report_path).resolve().parent / path
    try:
        return _read_json(path).get("criteria", {})
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _criterion_summary(criteria):
    if not criteria:
        return ""
    parts = []
    missing = criteria.get("missing_strut", {})
    broken = criteria.get("broken_strut", {})
    thin = criteria.get("thinned_strut", {})
    thick = criteria.get("thickened_strut", {})
    curved = criteria.get("curved_strut", {})
    twisted = criteria.get("twisted_strut", {})
    node = criteria.get("missing_node", {})
    if missing:
        parts.append(f"missing: coverage < {missing.get('max_coverage', 0):.0%}")
    if broken:
        parts.append(
            f"broken: coverage < {broken.get('min_coverage', 0):.0%} "
            f"or gap ≥ {broken.get('min_full_section_gap_fraction', 0):.0%}")
    if thin:
        parts.append(
            f"thin: diameter < {thin.get('max_diameter_ratio', 0):.0%}")
    if thick:
        parts.append(
            f"thick: diameter > {thick.get('min_diameter_ratio', 0):.0%}")
    if curved:
        parts.append(
            "curved: deviation > "
            f"{curved.get('max_lateral_deviation_diameters', 0):g}D")
    if twisted:
        parts.append(
            f"twisted: rotation ≥ "
            f"{twisted.get('min_total_rotation_degrees', 0):g}°")
    if node:
        parts.append(
            f"missing node: occupancy < {node.get('max_occupancy', 0):.0%}")
    return "  |  ".join(parts)


def _clip_strut_to_slice(p0, p1, index, tolerance):
    """Return the part of a 3D strut inside the displayed Z slab.

    Projecting the complete 3D strut onto a 2D slice makes portions that are
    many slices away look like findings on the current image.  Clip to the
    slice thickness first, then project the resulting short segment.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    zmin, zmax = index - tolerance, index + tolerance
    dz = p1[2] - p0[2]

    if np.isclose(dz, 0.0):
        return (p0, p1) if zmin <= p0[2] <= zmax else None

    t0 = (zmin - p0[2]) / dz
    t1 = (zmax - p0[2]) / dz
    enter, leave = sorted((t0, t1))
    enter, leave = max(0.0, enter), min(1.0, leave)
    if enter > leave:
        return None
    direction = p1 - p0
    return p0 + enter * direction, p0 + leave * direction


def _draw_slice(axis, volume, index, rows, bad_nodes, downsample, tolerance,
                visible_classes=None, criteria_text=""):
    axis.clear()
    visible = set(visible_classes or (*STRUT_CLASSES, *NODE_CLASSES))
    image = np.asarray(volume[index, ::downsample, ::downsample])
    axis.imshow(image, cmap="gray", origin="upper")
    counts = {kind: 0 for kind in (*STRUT_CLASSES, *NODE_CLASSES)}
    for row in rows:
        kind = row["classification"]
        if kind not in visible:
            continue
        clipped = _clip_strut_to_slice(
            row["p0"], row["p1"], index, tolerance)
        if clipped is None:
            continue
        p0, p1 = clipped
        axis.plot([p0[0] / downsample, p1[0] / downsample],
                  [p0[1] / downsample, p1[1] / downsample],
                  color=COLORS.get(kind, "yellow"), linewidth=1.4, alpha=0.9)
        counts[kind] = counts.get(kind, 0) + 1
    for node in bad_nodes:
        kind = node.get("classification", "missing_or_broken_node")
        if kind not in visible:
            continue
        p = node["point"]
        if abs(p[2] - index) <= tolerance:
            axis.scatter(p[0] / downsample, p[1] / downsample, s=38,
                         facecolors="none", edgecolors=COLORS[kind],
                         linewidths=1.4)
            counts[kind] = counts.get(kind, 0) + 1
    shown = [
        f"{DISPLAY_NAMES.get(kind, kind)} {count}"
        for kind, count in counts.items()
        if count and kind in visible
    ]
    axis.set_title(f"CT slice {index} | " + (
        " | ".join(shown) if shown else "no selected findings"))
    if criteria_text:
        axis.text(
            0.5, -0.025, criteria_text, transform=axis.transAxes,
            ha="center", va="top", fontsize=6.5, color="0.25",
            wrap=True)
    present = [
        kind for kind, count in counts.items()
        if count and kind in visible
    ]
    if present:
        from matplotlib.lines import Line2D
        handles = [
            Line2D(
                [0], [0], color=COLORS[kind], linewidth=2,
                marker="o" if kind in NODE_CLASSES else None,
                markerfacecolor="none",
                label=DISPLAY_NAMES.get(kind, kind))
            for kind in present
        ]
        axis.legend(
            handles=handles, loc="upper right", fontsize=7,
            framealpha=0.75)
    axis.set_axis_off()


def save_sample_slices(volume, rows, bad_nodes, output_dir, downsample=2,
                       tolerance=2, indices=None, visible_classes=None,
                       criteria_text=""):
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if indices is None:
        indices = np.linspace(0, volume.shape[0] - 1, 5, dtype=int)
    saved = []
    for index in sorted(set(int(np.clip(i, 0, volume.shape[0] - 1)) for i in indices)):
        fig, axis = plt.subplots(figsize=(8, 8))
        _draw_slice(
            axis, volume, index, rows, bad_nodes, downsample, tolerance,
            visible_classes, criteria_text)
        path = out / f"findings_slice_{index:04d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved


def view_results(tiff_path, report_path, registered_graph_path=None,
                 downsample=2, overlay_tolerance=2, sample_output_dir=None,
                 open_window=True):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import Button, CheckButtons, Slider
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    volume = tifffile.memmap(tiff_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF, got {volume.shape}")
    report, graph, rows, bad_nodes, graph_path = load_findings(
        report_path, registered_graph_path)
    criteria = _load_criteria(report, report_path)
    criteria_text = _criterion_summary(criteria)
    if downsample < 1 or overlay_tolerance < 0:
        raise ValueError("downsample must be >= 1 and overlay_tolerance must be >= 0")
    output_dir = sample_output_dir or str(Path(report_path).with_suffix("")) + "_slices"
    defect_z = [int(round((r["p0"][2] + r["p1"][2]) / 2)) for r in rows]
    sample_indices = (np.quantile(defect_z, np.linspace(0, 1, 5)).astype(int)
                      if defect_z else None)
    samples = save_sample_slices(volume, rows, bad_nodes, output_dir,
                                 downsample, overlay_tolerance, sample_indices,
                                 criteria_text=criteria_text)

    if open_window:
        fig = plt.figure(figsize=(15, 9))
        fig.subplots_adjust(bottom=0.23, wspace=0.08)
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        visible_classes = set((*STRUT_CLASSES, *NODE_CLASSES))
        class_artists = {kind: [] for kind in visible_classes}
        positions = {int(n["id"]): np.asarray(n["position"], float)
                     for n in graph.get("junctions", [])}
        base = []
        for s in graph.get("struts", []):
            a, b = int(s["junction0"]), int(s["junction1"])
            if a in positions and b in positions:
                base.append([positions[a], positions[b]])
        if base:
            ax3.add_collection3d(Line3DCollection(base, colors="0.72", linewidths=0.25,
                                                   alpha=0.25))
        for kind in STRUT_CLASSES:
            segments = [[r["p0"], r["p1"]] for r in rows if r["classification"] == kind]
            if segments:
                artist = Line3DCollection(
                    segments, colors=COLORS[kind], linewidths=2.0,
                    label=DISPLAY_NAMES[kind])
                ax3.add_collection3d(artist)
                class_artists[kind].append(artist)
        if bad_nodes:
            for kind in NODE_CLASSES:
                selected = [
                    n["point"] for n in bad_nodes
                    if n.get("classification",
                             "missing_or_broken_node") == kind]
                if selected:
                    p = np.asarray(selected)
                    artist = ax3.scatter(
                        p[:, 0], p[:, 1], p[:, 2], s=16,
                        facecolors="none", edgecolors=COLORS[kind],
                        label=DISPLAY_NAMES[kind])
                    class_artists[kind].append(artist)
        ax3.set_xlim(0, volume.shape[2]); ax3.set_ylim(0, volume.shape[1]); ax3.set_zlim(0, volume.shape[0])
        ax3.set_xlabel("x (voxel)"); ax3.set_ylabel("y (voxel)"); ax3.set_zlabel("z / slice")
        ax3.set_title("3D registered lattice findings")
        if rows or bad_nodes:
            ax3.legend(loc="upper right")

        initial = volume.shape[0] // 2
        _draw_slice(
            ax2, volume, initial, rows, bad_nodes, downsample,
            overlay_tolerance, visible_classes, criteria_text)
        slider_axis = fig.add_axes([0.56, 0.10, 0.34, 0.035])
        slider = Slider(slider_axis, "Slice", 0, volume.shape[0] - 1,
                        valinit=initial, valstep=1)
        save_axis = fig.add_axes([0.40, 0.085, 0.12, 0.055])
        save_button = Button(save_axis, "Save current slice")
        filter_axis = fig.add_axes([0.02, 0.015, 0.20, 0.19])
        filter_axis.set_title("Displayed defects", fontsize=9)
        filter_kinds = (*STRUT_CLASSES, "missing_or_broken_node")
        filter_labels = [DISPLAY_NAMES[kind] for kind in filter_kinds]
        filters = CheckButtons(
            filter_axis, filter_labels, [True] * len(filter_labels))
        for label_artist, kind in zip(filters.labels, filter_kinds):
            label_artist.set_color(COLORS[kind])
            label_artist.set_fontweight("bold")

        def update(value):
            _draw_slice(
                ax2, volume, int(value), rows, bad_nodes,
                downsample, overlay_tolerance, visible_classes,
                criteria_text)
            fig.canvas.draw_idle()

        def save_current(_event):
            save_sample_slices(
                volume, rows, bad_nodes, output_dir, downsample,
                overlay_tolerance, [int(slider.val)], visible_classes,
                criteria_text)

        def toggle(label):
            kind = filter_kinds[filter_labels.index(label)]
            aliases = (
                NODE_CLASSES if kind == "missing_or_broken_node"
                else (kind,))
            turning_on = not any(alias in visible_classes for alias in aliases)
            for alias in aliases:
                if turning_on:
                    visible_classes.add(alias)
                else:
                    visible_classes.discard(alias)
                for artist in class_artists.get(alias, []):
                    artist.set_visible(turning_on)
            update(slider.val)

        slider.on_changed(update)
        save_button.on_clicked(save_current)
        filters.on_clicked(toggle)
        fig._result_widgets = (slider, save_button, filters)
        plt.show()
    return {"tiff": str(tiff_path), "report": str(report_path),
            "registered_graph": str(graph_path), "defect_struts": len(rows),
            "suspect_nodes": len(bad_nodes), "sample_slices": samples}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiff")
    parser.add_argument("findings")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    print(view_results(args.tiff, args.findings, args.graph, args.downsample,
                       args.tolerance, args.samples, not args.no_window))


if __name__ == "__main__":
    main()
