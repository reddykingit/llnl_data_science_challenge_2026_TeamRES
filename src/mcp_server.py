from fastmcp import FastMCP
import matplotlib
import numpy as np
from pathlib import Path
import subprocess
import sys
import tifffile

# MCP servers commonly run without a display, so use a non-interactive backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .skeletonization import skeletonize_mask
    from .visual_engine import render_volume_png, resolve_visual_input
except ImportError:
    # Support running this file directly (``python src/mcp_server.py``).
    from skeletonization import skeletonize_mask
    from visual_engine import render_volume_png, resolve_visual_input

# Initialize the MCP server. The instructions help MCP clients choose this
# purpose-built local viewer instead of attempting a generic visualization skill.
mcp = FastMCP(
    "CT Analysis Tools",
    instructions=(
        "For requests to open, view, render, inspect, rotate, zoom, or interact "
        "with a local TIFF, OBJ, or NPY 3D asset, call visual_engine. Do not use "
        "a visualization skill or generate HTML when visual_engine is available."
    ),
)

@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.
    
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        ct_data = np.load(input_filepath, allow_pickle=False)
        if ct_data.ndim != 3:
            raise ValueError(
                f"Expected a 3D CT dataset, but found a {ct_data.ndim}D array."
            )

        segmentation = (ct_data >= threshold).astype(np.uint8)
        np.save(output_filepath, segmentation)
        return f"Successfully saved segmentation to {output_filepath}"
    except Exception as exc:
        return f"Error segmenting CT dataset: {exc}"

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
    figure = None
    try:
        ct_data = np.load(input_filepath, allow_pickle=False)
        if ct_data.ndim != 3:
            raise ValueError(
                f"Expected a 3D CT dataset, but found a {ct_data.ndim}D array."
            )
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")
        if not 0 <= slice_index < ct_data.shape[axis]:
            raise IndexError(
                f"slice_index {slice_index} is out of bounds for axis {axis} "
                f"with size {ct_data.shape[axis]}."
            )

        image_slice = np.take(ct_data, slice_index, axis=axis)
        figure, axes = plt.subplots()
        axes.imshow(image_slice, cmap="gray")
        axes.axis("off")
        figure.savefig(output_filepath, bbox_inches="tight", pad_inches=0)
        return f"Successfully saved slice visualization to {output_filepath}"
    except Exception as exc:
        return f"Error visualizing CT slice: {exc}"
    finally:
        if figure is not None:
            plt.close(figure)

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
    try:
        skeleton = skeletonize_mask(input_filepath, output_filepath)
        if skeleton is None:
            return f"Error skeletonizing mask: unable to load {input_filepath}"
        return f"Successfully saved skeleton to {output_filepath}"
    except Exception as exc:
        return f"Error skeletonizing mask: {exc}"

@mcp.tool()
def visual_engine(
    input_filepath: str,
    threshold: float | None = None,
    max_dimension: int = 256,
) -> str:
    """Open a local TIFF, OBJ, or NPY 3D asset in interactive PyVista.

    TIFF files and 3D NPY arrays are shown as thresholded surfaces with a live
    threshold slider. NPY arrays shaped (N, 3..16) are shown as point clouds. OBJ
    files are shown as lit meshes. Do not substitute HTML or a visualization skill.

    The viewer runs in a separate process so the MCP request can return while
    the window remains interactive. Drag to rotate, scroll to zoom, right-drag
    to pan, and close the window when finished. No HTML or image is generated.

    Args:
        input_filepath: Path to a ``.tif``, ``.tiff``, ``.npy``, or ``.obj`` file.
        threshold: Volume isosurface density, or omit it for automatic selection.
            Ignored for OBJ meshes and NPY point clouds.
        max_dimension: Maximum size of any displayed volume axis. Larger values
            provide more detail but require more memory and processing time.

    Returns:
        A message containing the viewer process ID, or a validation error.
    """
    try:
        try:
            import pyvista  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyVista is required for interactive viewing. Install the project "
                "dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        input_path = resolve_visual_input(input_filepath)
        if max_dimension < 16:
            raise ValueError("max_dimension must be at least 16.")
        if threshold is not None and not np.isfinite(threshold):
            raise ValueError("threshold must be a finite number.")

        command = [
            sys.executable,
            str(Path(__file__).with_name("visual_engine.py").resolve()),
            str(input_path),
            "--interactive",
            "--max-dimension",
            str(int(max_dimension)),
        ]
        if threshold is not None:
            command.extend(("--threshold", str(float(threshold))))
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        return (
            f"Opened interactive PyVista viewer for {input_path} "
            f"(PID {process.pid}, threshold="
            f"{'auto' if threshold is None else f'{threshold:g}'}, "
            f"max_dimension={max_dimension}). Close the viewer window when finished."
        )
    except Exception as exc:
        return f"Error opening interactive 3D TIFF viewer: {exc}"


def render_tiff_volume(
    input_filepath: str,
    output_filepath: str,
    threshold: float | None = None,
    max_dimension: int = 192,
) -> str:
    """Render a 3D TIFF isosurface to a PNG for internal Python callers.

    This function is intentionally not exposed as an MCP tool; interactive TIFF
    requests should route unambiguously to :func:`visual_engine`.

    Args:
        input_filepath: TIFF file or directory containing exactly one TIFF.
        output_filepath: Destination PNG path.
        threshold: Isosurface density, or omit it to choose one automatically.
        max_dimension: Maximum size of a downsampled volume axis.

    Returns:
        A message describing the saved render and the parameters used.
    """
    try:
        info = render_volume_png(
            input_filepath,
            output_filepath,
            threshold=threshold,
            max_dimension=max_dimension,
        )
        return (
            f"Saved 3D TIFF visualization to {Path(output_filepath).resolve()} "
            f"(source_shape={info.shape}, render_shape={info.render_shape}, "
            f"threshold={info.threshold:g}, stride={info.stride})."
        )
    except Exception as exc:
        return f"Error rendering 3D TIFF visualization: {exc}"


def _run_pyvista_viewer(
    input_filepath: str,
    threshold: float,
    max_dimension: int,
) -> None:
    """Build and display the interactive surface in a child process."""
    import pyvista as pv

    input_path = Path(input_filepath)
    volume = tifffile.memmap(input_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF, found a {volume.ndim}D array.")

    data_min = float(np.min(volume))
    data_max = float(np.max(volume))
    if not data_min < threshold < data_max:
        raise ValueError(
            f"threshold must be between {data_min:g} and {data_max:g}."
        )

    stride = max(1, int(np.ceil(max(volume.shape) / max_dimension)))
    render_volume = np.asarray(volume[::stride, ::stride, ::stride])
    xyz_volume = np.transpose(render_volume, (2, 1, 0))

    grid = pv.ImageData(dimensions=xyz_volume.shape)
    grid.spacing = (float(stride),) * 3
    grid.point_data["density"] = xyz_volume.ravel(order="F")
    surface = grid.contour([float(threshold)], scalars="density")
    if surface.n_points == 0:
        raise ValueError(f"No surface found at threshold {threshold:g}.")

    plotter = pv.Plotter(
        window_size=(1200, 900),
        title=f"{input_path.name} — threshold {threshold:g}",
    )
    plotter.set_background("white")
    plotter.add_mesh(
        surface,
        color="lightsteelblue",
        smooth_shading=True,
        specular=0.25,
    )
    plotter.add_axes()
    plotter.add_text(
        "Drag: rotate | Wheel: zoom | Right-drag: pan | Q: close",
        position="lower_edge",
        font_size=10,
        color="black",
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.15)
    plotter.show()

if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--pyvista-viewer":
        _run_pyvista_viewer(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    else:
        # Run the FastMCP server, exposing the tools over standard I/O (default)
        mcp.run()
