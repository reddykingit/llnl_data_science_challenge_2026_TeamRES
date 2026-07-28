import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")

from lattice_mcp_tools import register_lattice_tools

register_lattice_tools(mcp)


@mcp.tool()
def read_tiff_stack(
    input_filepath: str,
    slice_index: int = -1,
    preview_size: int = 16,
) -> dict[str, Any]:
    """Read and summarize one slice from a TIFF image stack.

    The tool reads TIFF metadata without loading the whole volume, then loads
    one image plane. By default it selects the middle plane of a stack. The
    returned preview is uniformly sampled and deliberately small so large CT
    scans remain usable through MCP.

    Args:
        input_filepath: Path to a .tif or .tiff file.
        slice_index: Zero-based plane to read. Use -1 (the default) for the
            middle plane. Single-image TIFF files use plane 0.
        preview_size: Width and height of the returned sampled preview,
            between 1 and 128 pixels.

    Returns:
        TIFF metadata, statistics for the selected plane, and a compact
        preview as nested lists. Errors are returned in an ``error`` field.
    """
    input_path = Path(input_filepath)
    if not input_path.is_file():
        return {"error": f"Input file not found: {input_path}"}
    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        return {"error": "Input must be a .tif or .tiff file."}
    if not 1 <= preview_size <= 128:
        return {"error": "preview_size must be between 1 and 128."}

    try:
        with tifffile.TiffFile(input_path) as tiff:
            if not tiff.series:
                return {"error": "The TIFF file contains no image series."}

            series = tiff.series[0]
            shape = tuple(int(size) for size in series.shape)
            page_count = len(series.pages)
            plane_count = max(page_count, 1)
            if slice_index == -1:
                selected_index = plane_count // 2
            elif 0 <= slice_index < plane_count:
                selected_index = slice_index
            else:
                return {
                    "error": (
                        f"slice_index must be between 0 and {plane_count - 1}, "
                        "or -1 for the middle plane."
                    )
                }

            plane = series.asarray(key=selected_index)
            if plane.ndim < 2:
                return {"error": f"Selected TIFF plane has unsupported shape {plane.shape}."}
            row_indices = np.linspace(0, plane.shape[0] - 1, min(preview_size, plane.shape[0]), dtype=int)
            column_indices = np.linspace(0, plane.shape[1] - 1, min(preview_size, plane.shape[1]), dtype=int)
            preview = plane[np.ix_(row_indices, column_indices)]

            return {
                "path": str(input_path),
                "series_shape": list(shape),
                "axes": series.axes,
                "dtype": str(series.dtype),
                "page_count": page_count,
                "selected_slice_index": selected_index,
                "selected_slice_shape": list(plane.shape),
                "selected_slice_min": float(np.min(plane)),
                "selected_slice_max": float(np.max(plane)),
                "selected_slice_mean": float(np.mean(plane)),
                "preview": preview.tolist(),
            }
    except (OSError, tifffile.TiffFileError, ValueError) as error:
        return {"error": f"Unable to read TIFF file: {error}"}

@mcp.tool()
def view_tiff_stack_with_slider(input_filepath: str, downsample: int = 1) -> str:
    """Open a local interactive slider for a 3D TIFF stack.

    The viewer memory-maps the TIFF and only reads the slice currently being
    displayed, so it is appropriate for large segmentation masks. This tool
    blocks until the viewer window is closed.

    Args:
        input_filepath: Path to a 3D .tif or .tiff stack.
        downsample: Display every nth pixel in X and Y; use a value greater
            than one to make large images faster to render.
    """
    input_path = Path(input_filepath)
    if not input_path.is_file():
        return f"Error: input file not found: {input_path}"
    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        return "Error: input must be a .tif or .tiff file."
    if downsample < 1:
        return "Error: downsample must be at least 1."

    try:
        volume = tifffile.memmap(input_path)
    except (OSError, tifffile.TiffFileError, ValueError) as error:
        return f"Error: unable to memory-map TIFF file: {error}"
    if volume.ndim != 3:
        return f"Error: expected a 3D TIFF stack, found shape {volume.shape}."

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    def display_slice(index: int) -> np.ndarray:
        return volume[index, ::downsample, ::downsample]

    initial_index = volume.shape[0] // 2
    fig, axis = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(bottom=0.15)
    image = axis.imshow(
        display_slice(initial_index),
        cmap="gray",
        vmin=0,
        vmax=1 if np.issubdtype(volume.dtype, np.bool_) or volume.dtype == np.uint8 else None,
        interpolation="nearest",
    )
    axis.set_title(f"{input_path.name} Ã¢â‚¬â€ slice {initial_index}")
    axis.set_axis_off()
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    slider_axis = fig.add_axes([0.18, 0.06, 0.65, 0.03])
    slider = Slider(
        slider_axis,
        "Slice",
        0,
        volume.shape[0] - 1,
        valinit=initial_index,
        valstep=1,
    )

    def update_slice(value: float) -> None:
        index = int(value)
        image.set_data(display_slice(index))
        axis.set_title(f"{input_path.name} Ã¢â‚¬â€ slice {index}")
        fig.canvas.draw_idle()

    slider.on_changed(update_slice)
    fig._slice_slider = slider
    plt.show()
    return f"Closed TIFF slider viewer for {input_path}."

@mcp.tool()
def determine_otsu_threshold(
    input_filepath: str,
    max_samples: int = 4_000_000,
) -> dict[str, Any]:
    """Estimate a metal-versus-air threshold using Otsu's method.

    A uniformly spaced sample is taken from a TIFF stack to avoid loading a
    large CT volume into memory. The returned ``normalized_threshold`` is in
    the [0, 1] convention used by ``segment_ct_dataset``.
    """
    input_path = Path(input_filepath)
    if not input_path.is_file():
        return {"error": f"Input file not found: {input_path}"}
    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        return {"error": "Input must be a .tif or .tiff file."}
    if max_samples < 1:
        return {"error": "max_samples must be at least 1."}

    try:
        volume = tifffile.memmap(input_path)
    except (OSError, tifffile.TiffFileError, ValueError) as error:
        return {"error": f"Unable to memory-map TIFF file: {error}"}
    if volume.ndim != 3:
        return {"error": f"Expected a 3D TIFF stack, found shape {volume.shape}."}

    from skimage.filters import threshold_otsu

    planes_to_sample = min(volume.shape[0], 64)
    plane_indices = np.linspace(0, volume.shape[0] - 1, planes_to_sample, dtype=int)
    pixels_per_plane = volume.shape[1] * volume.shape[2]
    pixel_stride = max(1, int(np.ceil(np.sqrt((pixels_per_plane * planes_to_sample) / max_samples))))
    samples = np.concatenate(
        [volume[index, ::pixel_stride, ::pixel_stride].ravel() for index in plane_indices]
    )
    threshold = float(threshold_otsu(samples))
    if np.issubdtype(volume.dtype, np.integer):
        normalized_threshold = threshold / float(np.iinfo(volume.dtype).max)
    else:
        normalized_threshold = threshold

    return {
        "path": str(input_path),
        "method": "Otsu",
        "sample_voxels": int(samples.size),
        "raw_threshold": threshold,
        "normalized_threshold": normalized_threshold,
        "metal_rule": f"voxel >= {threshold:g} is metal (white)",
        "air_rule": f"voxel < {threshold:g} is air (black)",
    }

@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.
    
    Args:
        input_filepatht: Path to the input .npy file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.
    
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    if not input_path.is_file():
        return f"Error: input file not found: {input_path}"
    if not 0.0 <= threshold <= 1.0:
        return "Error: threshold must be between 0.0 and 1.0."

    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        volume = np.load(input_path)
    elif suffix in {".tif", ".tiff"}:
        volume = tifffile.memmap(input_path)
    else:
        return "Error: input must be a .npy, .tif, or .tiff file."

    # Integer CT volumes conventionally span their dtype range; interpret the
    # threshold as a normalized [0, 1] density while keeping processing streamed.
    if np.issubdtype(volume.dtype, np.integer):
        cutoff = threshold * np.iinfo(volume.dtype).max
    else:
        cutoff = threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_suffix = output_path.suffix.lower()
    if output_suffix == ".npy":
        mask = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.uint8, shape=volume.shape
        )
    elif output_suffix in {".tif", ".tiff"}:
        mask = tifffile.memmap(
            output_path, shape=volume.shape, dtype=np.uint8, photometric="minisblack"
        )
    else:
        return "Error: output must use a .npy, .tif, or .tiff extension."

    foreground_voxels = 0
    for slice_index in range(volume.shape[0]):
        segmented_slice = (volume[slice_index] >= cutoff).astype(np.uint8)
        mask[slice_index] = segmented_slice
        foreground_voxels += int(segmented_slice.sum())
    mask.flush()

    return (
        f"Saved segmentation to {output_path} "
        f"(threshold={threshold}, foreground_voxels={foreground_voxels}, "
        f"total_voxels={volume.size})."
    )

@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """
    Loads a 3D CT dataset from a .npy file and saves a visualization of a specific slice to an image file.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT data.
        output_filepath: Path indicating where the output image should be saved (e.g., .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    pass # Implementation goes here

@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """
    Creates a skeleton from a 3D segmentation mask.
    
    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    pass # Implementation goes here, calling skeletonize_mask internally


@mcp.tool()
def identify_lattice_nodes(
    input_filepath: str,
    output_filepath: str,
    voxel_um: float = 10.0,
    downsample: int = 1,
    min_member_um: float = 100.0,
) -> dict[str, Any]:
    """Identify lattice junctions and measure their positions and local size.

    The input must be a segmented 3D mask (nonzero means material). The mask
    is skeletonized, junction/end clusters become nodes, and paths between
    nodes become members. Node diameter is reported as the median and maximum
    local inscribed diameter sampled on the node's skeleton cluster. These are
    local junction-size indicators, not fitted spherical diameters.

    Args:
        input_filepath: Segmented 3D .npy, .tif, or .tiff mask.
        output_filepath: Destination .json file for node measurements.
        voxel_um: Isotropic input voxel edge length in micrometres.
        downsample: Optional occupancy downsampling factor before skeletonizing.
        min_member_um: Shorter skeleton paths are treated as junction artifacts.

    Returns:
        Output path and counts, or an ``error`` field.
    """
    input_path, output_path = Path(input_filepath), Path(output_filepath)
    if not input_path.is_file():
        return {"error": f"Input file not found: {input_path}"}
    if output_path.suffix.lower() != ".json":
        return {"error": "output_filepath must use a .json extension."}
    if voxel_um <= 0 or downsample < 1 or min_member_um < 0:
        return {"error": "voxel_um must be positive, downsample >= 1, and min_member_um >= 0."}
    try:
        suffix = input_path.suffix.lower()
        if suffix == ".npy":
            volume = np.load(input_path, mmap_mode="r")
        elif suffix in {".tif", ".tiff"}:
            volume = tifffile.memmap(input_path)
        else:
            return {"error": "Input must be a .npy, .tif, or .tiff file."}
        if volume.ndim != 3:
            return {"error": f"Expected a 3D mask, found shape {volume.shape}."}

        from strut_graph import build_graph, occupancy_downsample

        mask = np.asarray(volume) != 0
        if downsample > 1:
            mask = occupancy_downsample(mask, downsample)
        effective_voxel_um = voxel_um * downsample
        graph = build_graph(mask, effective_voxel_um, min_member_um)
        incident = {node["id"]: [] for node in graph["nodes"]}
        for member in graph["members"]:
            incident[member["a"]].append(member["id"])
            incident[member["b"]].append(member["id"])
        nodes = []
        for node in graph["nodes"]:
            item = dict(node)
            item["incident_member_ids"] = incident[node["id"]]
            nodes.append(item)
        result = {
            "input": str(input_path),
            "voxel_um": voxel_um,
            "downsample": downsample,
            "effective_voxel_um": effective_voxel_um,
            "node_diameter_method": "2x distance-to-surface on node skeleton cluster",
            "node_count": len(nodes),
            "member_count": len(graph["members"]),
            "modal_degree": graph["modal_degree"],
            "suspect_node_count": sum(node["suspect"] for node in nodes),
            "nodes": nodes,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return {"output_filepath": str(output_path),
                "node_count": result["node_count"],
                "member_count": result["member_count"],
                "suspect_node_count": result["suspect_node_count"]}
    except (OSError, ValueError, MemoryError) as error:
        return {"error": f"Unable to identify lattice nodes: {error}"}


@mcp.tool()
def view_ct_defect_results(
    tiff_filepath: str,
    findings_filepath: str,
    registered_graph_filepath: str = "",
    downsample: int = 2,
    overlay_tolerance_slices: int = 2,
    sample_output_dir: str = "",
    open_window: bool = True,
) -> dict[str, Any]:
    """View CT defect results in linked 3D and 2D displays.

    The 3D panel shows the registered lattice, missing/broken struts, and
    suspect nodes. The 2D panel displays the TIFF stack with findings marked
    on the current image and includes a slider spanning every slice. Five
    representative marked slices are also saved as PNG files. The registered
    graph path is normally read from the findings report and only needs to be
    supplied when that metadata is absent or stale.

    Args:
        tiff_filepath: Original registered 3D CT .tif or .tiff stack.
        findings_filepath: Defect comparison report JSON.
        registered_graph_filepath: Optional registered lattice graph JSON.
        downsample: Display stride in X/Y; does not change finding coordinates.
        overlay_tolerance_slices: Show findings within this many Z slices.
        sample_output_dir: Optional directory for marked PNG sample slices.
        open_window: Open the interactive window; false only generates samples.
    """
    tiff_path, findings_path = Path(tiff_filepath), Path(findings_filepath)
    if not tiff_path.is_file():
        return {"error": f"TIFF file not found: {tiff_path}"}
    if tiff_path.suffix.lower() not in {".tif", ".tiff"}:
        return {"error": "tiff_filepath must use a .tif or .tiff extension."}
    if not findings_path.is_file():
        return {"error": f"Findings file not found: {findings_path}"}
    if downsample < 1 or overlay_tolerance_slices < 0:
        return {"error": "downsample must be >= 1 and overlay tolerance >= 0."}
    try:
        from result_viewer import view_results

        return view_results(
            tiff_path, findings_path,
            registered_graph_filepath or None,
            downsample, overlay_tolerance_slices,
            sample_output_dir or None, open_window)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, MemoryError) as error:
        return {"error": f"Unable to view CT defect results: {error}"}

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
