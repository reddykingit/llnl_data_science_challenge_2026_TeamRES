"""MCP server for the Pattern Recognition strut-defect technique (Claude Code: see .mcp.json).

Fair-comparison build: the only things exposed are
  * detect_strut_defects   - missing (>= 85 % absent) and broken (< 85 % absent) struts, counts, 3-D view, marked slices
  * list_defects           - read back the defect table
  * get_marked_slice       - path to the marked slice image for a defect (for visual review)
  * build_reference / score_against_reference - EVALUATION ONLY (use the design STL/JSON; never used by the detector)
All earlier exploratory tools (segmentation, skeletonisation, layer-length, deformation analysis) live in legacy/.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("strut-defects")


@mcp.tool()
def detect_strut_defects(input_filepath: str, output_directory: str, classifier: str = "gbm", specimen: str = "") -> dict[str, Any]:
    """Find missing and broken struts in an octet-lattice CT scan (.tif stack or .npy volume, slices along axis 0).

    Technique: Pattern Recognition. Every voxel is voted on by its 18 lattice-equivalent neighbours
    (lattice vectors measured from the scan itself, so tilt is handled; neighbours outside the specimen abstain,
    so edges are handled). Gaps (where the neighbours agree there should be material but the scan has none) are linked in 3-D and measured:
    a gap spanning >= 85 % of the free strut length is a MISSING strut, a shorter gap that cuts the strut is a
    BROKEN strut. A false-alarm filter then removes bent / displaced / thin-but-present struts.

    Args:
        input_filepath: CT volume (.tif/.tiff stack or .npy), slice index first.
        output_directory: Folder for results (created). Contains summary.json, defects.csv, defects_map.json (lattice-JSON layout linking each
            defect's 3-D position to its marked slice), defects_3d.png/.html,
            marked_slices/*.png, marked_slices_viewer_rgb.npy (open with npy_viewer.py), defect_labels.npy, trace.jsonl.
        classifier: False-alarm filter: "gbm" (gradient boosting, default), "rules", or "none" (85 % rule only).
        specimen: Specimen name; defaults to the output folder name.

    Returns:
        Counts of missing and broken struts, runtime in seconds and the output file paths.
    """
    from src.strut_defects.pipeline import run
    out = Path(output_directory)
    res = run(input_filepath, out, classifier, None, specimen or out.name)
    summary = json.loads((out / "summary.json").read_text())
    return dict(missing_struts=summary["missing_struts"], broken_struts=summary["broken_struts"],
                runtime_seconds=res.get("runtime_seconds"), output_directory=str(out.resolve()),
                files={k: str((out / v).resolve()) for k, v in summary["files"].items()})


@mcp.tool()
def list_defects(output_directory: str, defect_class: str = "") -> list[dict[str, Any]]:
    """Return the detected defects (rows of defects.csv). defect_class: "", "missing" or "broken"."""
    rows = list(csv.DictReader(open(Path(output_directory) / "defects.csv")))
    return [r for r in rows if not defect_class or r["defect_class"] == defect_class]


@mcp.tool()
def get_marked_slice(output_directory: str, defect_id: str = "", slice_index: int = -1) -> str:
    """Path to the marked slice PNG that shows a defect (by defect_id such as "M004"/"B017", or by slice_index).
    Open the returned image to review the detection visually."""
    out = Path(output_directory)
    if defect_id:
        row = next((r for r in csv.DictReader(open(out / "defects.csv")) if r["defect_id"] == defect_id), None)
        if row is None:
            return f"unknown defect_id {defect_id}"
        slice_index = int(row["marked_slice"])
    p = out / "marked_slices" / f"slice_{slice_index:04d}.png"
    return str(p.resolve()) if p.exists() else f"no marked slice for slice {slice_index}"


@mcp.tool()
def build_reference(ct_filepath: str, registered_json: str, output_directory: str, nominal_json: str = "",
                    stl_full: str = "", stl_defect: str = "") -> dict[str, Any]:
    """EVALUATION ONLY. Build the strut-level reference used for scoring.

    Unified DSC input: pass only the CT tif and the registered JSON ("<scan>.ct_registered.json"). The print is
    treated as a 0 % defect design, so every strut the CT shows >= 85 % absent is missing and every strut with no
    connected material between its nodes is broken. nominal_json / stl_full / stl_defect are only for the older
    design-removal mode and should be left empty."""
    from src.evaluation.reference import build
    return build(ct_filepath, registered_json, nominal_json or None, stl_full or None, stl_defect or None, Path(output_directory))


@mcp.tool()
def score_against_reference(output_directory: str, reference_directory: str) -> dict[str, Any]:
    """EVALUATION ONLY. Strut-level precision / recall / F1, per-class recall and IoU/Dice (template Section 4)."""
    from src.evaluation.score import score
    s = score(Path(output_directory), Path(reference_directory))
    return dict(overall=s["overall"], overall_all_struts=s.get("overall_all_struts"), F1_zblock_mean=s["F1_zblock_mean"],
                F1_zblock_std=s["F1_zblock_std"], segmentation_vs_design=s.get("segmentation_vs_design"))


if __name__ == "__main__":
    mcp.run()
