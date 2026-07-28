"""MCP tools for lattice JSON and registered CT comparison."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
import numpy as np
import tifffile

def _load(path):
    text = path.read_text(encoding="utf-8-sig")
    try: data = json.loads(text)
    except json.JSONDecodeError:
        starts = [text.find(f'"{k}"') for k in ("junctions", "nodes")]
        starts = [x for x in starts if x >= 0]
        if not starts: raise ValueError("Invalid JSON; no lattice graph found")
        data = json.loads("{" + text[min(starts):])
    if not isinstance(data, dict): raise ValueError("JSON root must be an object")
    if isinstance(data.get("junctions"), list) and isinstance(data.get("struts"), list):
        return data, "design_graph", data["junctions"], data["struts"]
    if isinstance(data.get("nodes"), list) and isinstance(data.get("members"), list):
        return data, "skeleton_graph", data["nodes"], data["members"]
    raise ValueError("Expected junctions/struts or nodes/members arrays")

def _keys(schema):
    return ("junction0", "junction1") if schema == "design_graph" else ("a", "b")

def read_lattice_json(input_filepath: str, preview_items: int = 5) -> dict[str, Any]:
    """Read, validate, and summarize lattice JSON without returning the entire file."""
    path = Path(input_filepath)
    if not path.is_file() or path.suffix.lower() != ".json": return {"error": f"Invalid path: {path}"}
    if not 0 <= preview_items <= 100: return {"error": "preview_items must be 0..100"}
    try:
        data, schema, nodes, struts = _load(path); ids = {n.get("id") for n in nodes}
        a, b = _keys(schema); degrees = {i: 0 for i in ids}; invalid = []
        for s in struts:
            x, y = s.get(a), s.get(b)
            if x not in ids or y not in ids: invalid.append({"strut_id": s.get("id"), "start": x, "end": y})
            if x in degrees: degrees[x] += 1
            if y in degrees: degrees[y] += 1
        hist = {}
        for degree in degrees.values(): hist[str(degree)] = hist.get(str(degree), 0) + 1
        return {"path": str(path), "schema": schema, "top_level_keys": sorted(data),
                "node_count": len(nodes), "strut_count": len(struts),
                "unit_cell_count": len(data.get("unit_cells", [])), "degree_histogram": hist,
                "invalid_endpoint_count": len(invalid), "invalid_endpoints_preview": invalid[:20],
                "nodes_preview": nodes[:preview_items], "struts_preview": struts[:preview_items]}
    except (OSError, ValueError, json.JSONDecodeError) as e: return {"error": f"Unable to read JSON: {e}"}

def identify_json_nodes(input_filepath: str, node_ids: list[int] | None = None,
                        limit: int = 100) -> dict[str, Any]:
    """Return selected nodes, graph degree, and incident strut IDs."""
    try:
        path = Path(input_filepath); _, schema, nodes, struts = _load(path)
        if not 1 <= limit <= 5000: return {"error": "limit must be 1..5000"}
        wanted = set(node_ids or []); selected = nodes if not wanted else [n for n in nodes if n.get("id") in wanted]
        incident = {n.get("id"): [] for n in selected}; a, b = _keys(schema)
        for s in struts:
            for endpoint in (s.get(a), s.get(b)):
                if endpoint in incident: incident[endpoint].append(s.get("id"))
        output = []
        for node in selected[:limit]:
            item = dict(node); item["degree"] = len(incident[node.get("id")]); item["incident_strut_ids"] = incident[node.get("id")]; output.append(item)
        return {"schema": schema, "matched_count": len(selected), "returned_count": len(output),
                "truncated": len(selected) > limit, "nodes": output}
    except (OSError, ValueError, json.JSONDecodeError) as e: return {"error": f"Unable to identify nodes: {e}"}

def identify_json_struts(input_filepath: str, strut_ids: list[int] | None = None,
                         limit: int = 100) -> dict[str, Any]:
    """Return selected struts with resolved endpoint positions."""
    try:
        path = Path(input_filepath); _, schema, nodes, struts = _load(path)
        if not 1 <= limit <= 5000: return {"error": "limit must be 1..5000"}
        wanted = set(strut_ids or []); selected = struts if not wanted else [s for s in struts if s.get("id") in wanted]
        pos = {n.get("id"): n.get("position", n.get("position_xyz_voxels")) for n in nodes}; a, b = _keys(schema); output = []
        for strut in selected[:limit]:
            item = dict(strut); item["start_position"] = pos.get(strut.get(a)); item["end_position"] = pos.get(strut.get(b)); output.append(item)
        return {"schema": schema, "matched_count": len(selected), "returned_count": len(output),
                "truncated": len(selected) > limit, "struts": output}
    except (OSError, ValueError, json.JSONDecodeError) as e: return {"error": f"Unable to identify struts: {e}"}

def skeleton_to_lattice_json(input_filepath: str, output_filepath: str, voxel_um: float = 10.0,
                             downsample: int = 1, min_strut_um: float = 100.0) -> dict[str, Any]:
    """Convert a segmented or skeletonized 3D NPY/TIFF volume into graph JSON."""
    source, output = Path(input_filepath), Path(output_filepath)
    if not source.is_file() or output.suffix.lower() != ".json": return {"error": "Invalid input or JSON output path"}
    if voxel_um <= 0 or downsample < 1 or min_strut_um < 0: return {"error": "Invalid numeric parameter"}
    try:
        suffix = source.suffix.lower()
        volume = np.load(source, mmap_mode="r") if suffix == ".npy" else tifffile.memmap(source) if suffix in {".tif", ".tiff"} else None
        if volume is None or volume.ndim != 3: return {"error": "Input must be a 3D NPY/TIFF"}
        from strut_graph import build_graph, occupancy_downsample
        mask = np.asarray(volume) != 0
        if downsample > 1: mask = occupancy_downsample(mask, downsample)
        graph = build_graph(mask, voxel_um * downsample, min_strut_um)
        for node in graph["nodes"]:
            if node.get("position_voxels") is not None: node["position_xyz_voxels"] = list(reversed(node["position_voxels"]))
        graph["meta"] = {"source": str(source), "source_shape_zyx": list(volume.shape),
                         "coordinate_order": "position_voxels=ZYX; position_xyz_voxels=XYZ",
                         "voxel_um": voxel_um, "downsample": downsample, "min_strut_um": min_strut_um}
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        return {"output_filepath": str(output), "node_count": len(graph["nodes"]),
                "strut_count": len(graph["members"]), "modal_degree": graph["modal_degree"]}
    except (OSError, ValueError, MemoryError) as e: return {"error": f"Unable to create graph: {e}"}


def refine_lattice_json_registration(
    tiff_filepath: str,
    lattice_json_filepath: str,
    output_filepath: str,
    registration_sample_points: int = 5000,
) -> dict[str, Any]:
    """Detect CT lattice tilt/scale/offset and write a corrected graph JSON.

    Fits a residual 3D transform from the supplied lattice centerlines to CT
    material. The transform corrects XYZ translation, anisotropic scale, and
    X/Y-to-Z tilt, which accounts for lattice layers crossing successive
    axial CT slices from one side of the specimen to the other. The source
    JSON is never overwritten.

    Args:
        tiff_filepath: Raw 3D CT TIFF stack in ZYX array order.
        lattice_json_filepath: Nominal or approximately registered lattice
            graph containing junctions, struts, and unit_cells.
        output_filepath: New JSON path for the tilt-corrected graph. It must
            differ from lattice_json_filepath.
        registration_sample_points: Representative node/member centerline
            samples used during fitting; minimum 500.
    """
    tiff = Path(tiff_filepath)
    graph = Path(lattice_json_filepath)
    output = Path(output_filepath)
    if not tiff.is_file() or tiff.suffix.lower() not in {".tif", ".tiff"}:
        return {"error": f"Invalid CT TIFF path: {tiff}"}
    if not graph.is_file() or graph.suffix.lower() != ".json":
        return {"error": f"Invalid lattice JSON path: {graph}"}
    if output.suffix.lower() != ".json":
        return {"error": "output_filepath must use a .json extension"}
    if graph.resolve() == output.resolve():
        return {"error": "output_filepath must not overwrite the source JSON"}
    if registration_sample_points < 500:
        return {"error": "registration_sample_points must be >= 500"}
    try:
        from register_json_to_ct import register

        registration = register(
            tiff, graph, output, registration_sample_points)
        return {
            "output_filepath": str(output),
            "source_tiff": str(tiff),
            "source_json": str(graph),
            "alignment_score_before": registration[
                "initial_alignment_score"],
            "alignment_score_after": registration[
                "refined_alignment_score"],
            "alignment_improvement": (
                registration["refined_alignment_score"]
                - registration["initial_alignment_score"]),
            "transform": registration["parameters"],
            "pivot_xyz": registration["pivot_xyz"],
            "sample_points": registration["sample_points"],
            "optimizer_success": registration["optimizer_success"],
            "optimizer_message": registration["optimizer_message"],
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            MemoryError) as e:
        return {"error": f"Unable to refine CT registration: {e}"}


def create_lattice_defect_criteria(
    output_filepath: str,
    support_occupancy_cutoff: float = 0.20,
    missing_strut_max_coverage: float = 0.20,
    broken_strut_min_coverage: float = 0.65,
    broken_strut_min_gap_fraction: float = 0.20,
    broken_strut_min_gap_voxels: int = 3,
    thinning_diameter_ratio: float = 0.75,
    thickening_diameter_ratio: float = 1.25,
    diameter_condition_min_length_fraction: float = 0.20,
    curved_max_lateral_deviation_diameters: float = 0.50,
    curved_min_arc_chord_ratio: float = 1.03,
    twisted_min_total_rotation_degrees: float = 20.0,
    twisted_min_cross_section_aspect_ratio: float = 1.15,
    missing_node_max_occupancy: float = 0.15,
    missing_node_max_supported_incident_fraction: float = 0.25,
) -> dict[str, Any]:
    """Create a validated criteria file for CT lattice defect classification.

    The file separates measured features from labels and supports missing,
    broken, thinned, thickened, curved, and twisted struts plus missing nodes.
    Diameter thresholds are ratios to nominal/design diameter. Curvature is
    normalized by nominal diameter. Twist is only considered observable when
    the fitted cross-section is sufficiently non-circular.
    """
    output = Path(output_filepath)
    if output.suffix.lower() != ".json":
        return {"error": "output_filepath must use a .json extension"}
    fractions = {
        "support_occupancy_cutoff": support_occupancy_cutoff,
        "missing_strut_max_coverage": missing_strut_max_coverage,
        "broken_strut_min_coverage": broken_strut_min_coverage,
        "broken_strut_min_gap_fraction": broken_strut_min_gap_fraction,
        "diameter_condition_min_length_fraction":
            diameter_condition_min_length_fraction,
        "missing_node_max_occupancy": missing_node_max_occupancy,
        "missing_node_max_supported_incident_fraction":
            missing_node_max_supported_incident_fraction,
    }
    invalid = [name for name, value in fractions.items()
               if not 0 <= value <= 1]
    if invalid:
        return {"error": f"Criteria must be between 0 and 1: {invalid}"}
    if missing_strut_max_coverage >= broken_strut_min_coverage:
        return {"error": "missing coverage must be below broken coverage"}
    if not 0 < thinning_diameter_ratio < 1:
        return {"error": "thinning_diameter_ratio must be between 0 and 1"}
    if thickening_diameter_ratio <= 1:
        return {"error": "thickening_diameter_ratio must be greater than 1"}
    if broken_strut_min_gap_voxels < 1:
        return {"error": "broken_strut_min_gap_voxels must be positive"}
    if curved_max_lateral_deviation_diameters <= 0:
        return {"error": "curvature deviation threshold must be positive"}
    if curved_min_arc_chord_ratio <= 1:
        return {"error": "curved_min_arc_chord_ratio must be greater than 1"}
    if twisted_min_total_rotation_degrees <= 0:
        return {"error": "twist angle must be positive"}
    if twisted_min_cross_section_aspect_ratio <= 1:
        return {"error": "twist aspect ratio must be greater than 1"}

    criteria = {
        "schema": "ct_lattice_defect_criteria",
        "version": 1,
        "units": {
            "coverage_and_length_fraction": "dimensionless [0,1]",
            "diameter_ratio": "measured / nominal",
            "lateral_deviation": "nominal strut diameters",
            "rotation": "degrees",
            "gap_length": "CT voxels",
        },
        "evaluation_order": [
            "missing_strut", "broken_strut", "curved_strut",
            "thinned_strut", "thickened_strut", "twisted_strut",
            "intact",
        ],
        "allow_multiple_geometric_labels": True,
        "endpoint_exclusion_fraction": 0.12,
        "criteria": {
            "missing_strut": {
                "rule": "supported_length_fraction < max_coverage",
                "max_coverage": missing_strut_max_coverage,
                "support_occupancy_cutoff": support_occupancy_cutoff,
            },
            "broken_strut": {
                "rule": (
                    "not missing AND (supported_length_fraction < "
                    "min_coverage OR full_section_gap meets both limits OR "
                    "end regions are disconnected)"
                ),
                "min_coverage": broken_strut_min_coverage,
                "min_full_section_gap_fraction":
                    broken_strut_min_gap_fraction,
                "min_full_section_gap_voxels":
                    broken_strut_min_gap_voxels,
                "require_connectivity_check": True,
            },
            "thinned_strut": {
                "rule": (
                    "equivalent_diameter_ratio < max_diameter_ratio over "
                    "at least min_length_fraction"
                ),
                "max_diameter_ratio": thinning_diameter_ratio,
                "min_length_fraction":
                    diameter_condition_min_length_fraction,
                "cross_section_normal_to_measured_centerline": True,
            },
            "thickened_strut": {
                "rule": (
                    "equivalent_diameter_ratio > min_diameter_ratio over "
                    "at least min_length_fraction"
                ),
                "min_diameter_ratio": thickening_diameter_ratio,
                "min_length_fraction":
                    diameter_condition_min_length_fraction,
                "cross_section_normal_to_measured_centerline": True,
            },
            "curved_strut": {
                "rule": (
                    "max_lateral_deviation exceeds its limit OR "
                    "centerline_arc_length / endpoint_distance exceeds "
                    "its limit"
                ),
                "max_lateral_deviation_diameters":
                    curved_max_lateral_deviation_diameters,
                "min_arc_chord_ratio": curved_min_arc_chord_ratio,
            },
            "twisted_strut": {
                "rule": (
                    "observable non-circular cross-section rotates by at "
                    "least min_total_rotation_degrees along the member"
                ),
                "min_total_rotation_degrees":
                    twisted_min_total_rotation_degrees,
                "min_cross_section_aspect_ratio":
                    twisted_min_cross_section_aspect_ratio,
                "circular_cross_section_result": "not_observable",
            },
            "missing_node": {
                "rule": (
                    "local_occupancy < max_occupancy AND "
                    "supported_incident_strut_fraction < "
                    "max_supported_incident_fraction"
                ),
                "max_occupancy": missing_node_max_occupancy,
                "max_supported_incident_fraction":
                    missing_node_max_supported_incident_fraction,
                "ignore_nodes_outside_valid_scan_support": True,
            },
            "intact": {
                "rule": (
                    "none of the preceding missing, connectivity, or "
                    "geometric defect criteria are met"
                )
            },
        },
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(criteria, indent=2), encoding="utf-8")
        return {
            "output_filepath": str(output),
            "schema": criteria["schema"],
            "version": criteria["version"],
            "defect_classes": list(criteria["criteria"]),
            "evaluation_order": criteria["evaluation_order"],
        }
    except OSError as e:
        return {"error": f"Unable to write criteria: {e}"}


def compare_registered_json_to_tiff(tiff_filepath: str, registered_json_filepath: str,
                                    output_filepath: str, ct_threshold: float | None = None,
                                    samples_per_strut: int = 25, support_radius_voxels: int = 2,
                                    node_radius_voxels: int = 3, max_struts: int | None = None,
                                    auto_refine_registration: bool = True,
                                    registration_sample_points: int = 5000,
                                    criteria_filepath: str = "") -> dict[str, Any]:
    """Register expected JSON to CT, then compare struts with material evidence.

    Registration refinement corrects residual XYZ translation, anisotropic
    scale, and lattice-layer tilt relative to the CT axial slices before any
    node or member is classified.
    """
    tiff, graph, output = map(Path, (tiff_filepath, registered_json_filepath, output_filepath))
    if not tiff.is_file() or not graph.is_file() or output.suffix.lower() != ".json": return {"error": "Invalid input/output path"}
    if samples_per_strut < 3 or min(support_radius_voxels, node_radius_voxels) < 0: return {"error": "Invalid sampling parameter"}
    if max_struts is not None and max_struts < 1: return {"error": "max_struts must be null or positive"}
    if registration_sample_points < 500: return {"error": "registration_sample_points must be >= 500"}
    try:
        from compare_ct_json_defects import compare
        criterion_values = {
            "support_occupancy": 0.20,
            "node_occupancy": 0.15,
            "missing_max_coverage": 0.20,
            "broken_min_coverage": 0.65,
            "broken_min_gap_fraction": 0.20,
        }
        if criteria_filepath:
            criteria_path = Path(criteria_filepath)
            criteria_document = json.loads(
                criteria_path.read_text(encoding="utf-8"))
            if criteria_document.get("schema") != "ct_lattice_defect_criteria":
                return {"error": "Invalid defect criteria schema"}
            configured = criteria_document["criteria"]
            criterion_values.update({
                "support_occupancy": configured["missing_strut"][
                    "support_occupancy_cutoff"],
                "node_occupancy": configured["missing_node"][
                    "max_occupancy"],
                "missing_max_coverage": configured["missing_strut"][
                    "max_coverage"],
                "broken_min_coverage": configured["broken_strut"][
                    "min_coverage"],
                "broken_min_gap_fraction": configured["broken_strut"][
                    "min_full_section_gap_fraction"],
            })
        registration = None
        comparison_graph = graph
        if auto_refine_registration:
            from register_json_to_ct import register
            comparison_graph = output.with_name(
                output.stem + "_registered.json")
            registration = register(
                tiff, graph, comparison_graph, registration_sample_points)
        report, rows = compare(
            tiff, comparison_graph, ct_threshold, samples_per_strut,
            support_radius_voxels, node_radius_voxels, max_struts,
            **criterion_values)
        report["inputs"]["criteria"] = criteria_filepath or "built-in defaults"
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        csv_path = output.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["strut_id"]); writer.writeheader(); writer.writerows(rows)
        return {"output_filepath": str(output), "csv_filepath": str(csv_path),
                "registered_json_filepath": str(comparison_graph),
                "registration": registration, **report["summary"],
                "raw_threshold": report["ct"]["raw_threshold"], "threshold_source": report["ct"]["threshold_source"]}
    except (OSError, ValueError, KeyError, json.JSONDecodeError, MemoryError) as e: return {"error": f"Unable to compare: {e}"}

def register_lattice_tools(mcp: Any) -> None:
    for tool in (read_lattice_json, identify_json_nodes, identify_json_struts,
                 skeleton_to_lattice_json, refine_lattice_json_registration,
                 create_lattice_defect_criteria,
                 compare_registered_json_to_tiff): mcp.tool()(tool)
