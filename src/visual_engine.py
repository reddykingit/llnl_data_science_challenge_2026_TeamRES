"""Memory-conscious interactive visualization for scientific 3D assets.

The module has two complementary renderers:

* :func:`render_volume_png` creates an image that an agent or report can inspect.
* :func:`open_interactive_viewer` opens TIFF/NPY volumes, NPY point clouds, and
  OBJ meshes in a native PyVista window.

Both operate on a TIFF memory map and downsample before surface extraction, so
the full missing-struts scan does not have to be copied into RAM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "missing_struts" / "tif_stacks"
SUPPORTED_INPUT_SUFFIXES = {".tif", ".tiff", ".npy", ".obj"}


@dataclass(frozen=True)
class VolumeInfo:
    path: Path
    shape: tuple[int, int, int]
    dtype: str
    data_min: float
    data_max: float
    threshold: float
    stride: int
    render_shape: tuple[int, int, int]
    slice_range: tuple[int, int]


def resolve_tiff(input_path: str | Path) -> Path:
    """Resolve a TIFF file, or the only TIFF contained in a directory."""
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        candidates = sorted((*path.glob("*.tif"), *path.glob("*.tiff")))
        if not candidates:
            raise FileNotFoundError(f"No .tif or .tiff files found in {path}")
        if len(candidates) > 1:
            names = ", ".join(candidate.name for candidate in candidates)
            raise ValueError(f"Multiple TIFF files found in {path}: {names}")
        path = candidates[0]
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("input_path must be a TIFF file or a directory containing one")
    if not path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {path}")
    return path


def resolve_visual_input(input_path: str | Path) -> Path:
    """Resolve a supported 3D asset file, or the only one in a directory."""
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        )
        if not candidates:
            raise FileNotFoundError(
                f"No supported .tif, .tiff, .npy, or .obj files found in {path}"
            )
        if len(candidates) > 1:
            names = ", ".join(candidate.name for candidate in candidates)
            raise ValueError(f"Multiple supported 3D files found in {path}: {names}")
        path = candidates[0]
    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError("input must be a .tif, .tiff, .npy, or .obj file")
    if not path.is_file():
        raise FileNotFoundError(f"3D input file not found: {path}")
    return path


def _sample_volume(volume: np.ndarray, target_samples: int = 750_000) -> np.ndarray:
    stride = max(1, int(np.ceil((volume.size / target_samples) ** (1 / 3))))
    return np.asarray(volume[::stride, ::stride, ::stride]).reshape(-1)


def _auto_threshold(volume: np.ndarray) -> float:
    sample = _sample_volume(volume)
    if sample.size == 0 or float(sample.min()) == float(sample.max()):
        raise ValueError("Cannot determine a surface threshold for a constant volume")
    unique = np.unique(sample)
    if unique.size == 2:
        return float(unique[0] + (unique[1] - unique[0]) / 2)
    return float(threshold_otsu(sample))


def prepare_volume(
    input_path: str | Path,
    threshold: float | None = None,
    max_dimension: int = 192,
) -> tuple[np.ndarray, VolumeInfo]:
    """Memory-map, validate, and downsample a 3D TIFF volume."""
    if max_dimension < 32:
        raise ValueError("max_dimension must be at least 32")

    path = resolve_visual_input(input_path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        volume = tifffile.memmap(path)
    elif path.suffix.lower() == ".npy":
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError("Volume preparation supports TIFF and 3D NPY files only")
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, found shape {volume.shape}")
    if not np.issubdtype(volume.dtype, np.number):
        raise TypeError(f"Expected numeric voxel values, found {volume.dtype}")

    sample = _sample_volume(volume)
    data_min = float(sample.min())
    data_max = float(sample.max())
    chosen_threshold = _auto_threshold(volume) if threshold is None else float(threshold)
    if not np.isfinite(chosen_threshold):
        raise ValueError("threshold must be finite")
    if not data_min < chosen_threshold < data_max:
        raise ValueError(
            f"threshold must be between sampled data limits {data_min:g} and {data_max:g}"
        )

    stride = max(1, int(np.ceil(max(volume.shape) / max_dimension)))
    render_volume = np.asarray(volume[::stride, ::stride, ::stride], dtype=np.float32)

    start, stop = 0, render_volume.shape[0]
    if path.suffix.lower() in {".tif", ".tiff"}:
        # Some CT acquisitions contain bright reconstruction end caps that hide
        # the lattice. Apply this acquisition-specific trim only to TIFF stacks.
        plane_fill = np.mean(render_volume >= chosen_threshold, axis=(1, 2))
        usable = np.flatnonzero(plane_fill <= 0.20)
        if usable.size and usable[-1] - usable[0] + 1 >= render_volume.shape[0] * 0.60:
            start, stop = int(usable[0]), int(usable[-1] + 1)
            render_volume = render_volume[start:stop]
    info = VolumeInfo(
        path=path,
        shape=tuple(int(value) for value in volume.shape),
        dtype=str(volume.dtype),
        data_min=data_min,
        data_max=data_max,
        threshold=chosen_threshold,
        stride=stride,
        render_shape=tuple(int(value) for value in render_volume.shape),
        slice_range=(start * stride, min(stop * stride, int(volume.shape[0]))),
    )
    return render_volume, info


def _surface(volume: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces, _, _ = marching_cubes(
        volume,
        level=threshold,
        step_size=2,
        allow_degenerate=False,
    )
    if len(vertices) == 0:
        raise ValueError(f"No surface found at threshold {threshold:g}")
    return vertices, faces


def render_volume_png(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = PROJECT_ROOT / "output" / "missing-struts-volume.png",
    threshold: float | None = None,
    max_dimension: int = 192,
    max_faces: int = 175_000,
) -> VolumeInfo:
    """Render an isometric lattice surface to a transparent-background PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    volume, info = prepare_volume(input_path, threshold, max_dimension)
    vertices, faces = _surface(volume, info.threshold)
    if len(faces) > max_faces:
        keep = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[keep]

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    layout = figure.add_gridspec(3, 4, width_ratios=(1, 1, 1, 0.95))
    axes = figure.add_subplot(layout[:, :3], projection="3d")
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_lengths, 1e-8)
    light = np.asarray((0.35, -0.45, 0.82), dtype=np.float32)
    brightness = np.clip(0.52 + 0.38 * np.abs(normals @ light), 0.42, 0.92)
    base_color = np.asarray((0.23, 0.58, 0.78), dtype=np.float32)
    face_colors = np.column_stack((brightness[:, None] * base_color, np.ones(len(faces))))
    mesh = Poly3DCollection(
        triangles,
        facecolor=face_colors,
        edgecolor="none",
        alpha=1.0,
    )
    axes.add_collection3d(mesh)
    z_size, y_size, x_size = volume.shape
    axes.set_xlim(0, z_size)
    axes.set_ylim(0, y_size)
    axes.set_zlim(0, x_size)
    axes.set_box_aspect((z_size, y_size, x_size))
    axes.view_init(elev=28, azim=-52)
    axes.set_axis_off()
    axes.set_title("Thresholded 3D isosurface", fontsize=11)

    vmin, vmax = np.percentile(volume, (2, 99.5))
    slice_specs = (
        (0, volume.shape[0] // 2, "Axial"),
        (1, volume.shape[1] // 2, "Coronal"),
        (2, volume.shape[2] // 2, "Sagittal"),
    )
    for row, (axis, index, label) in enumerate(slice_specs):
        slice_axes = figure.add_subplot(layout[row, 3])
        slice_axes.imshow(
            np.take(volume, index, axis=axis),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        source_index = index * info.stride
        if axis == 0:
            source_index += info.slice_range[0]
        slice_axes.set_title(f"{label} slice {source_index}", fontsize=10)
        slice_axes.set_axis_off()

    figure.suptitle(
        f"{info.path.name}\nthreshold {info.threshold:.0f} | "
        f"volume {info.shape[0]}×{info.shape[1]}×{info.shape[2]} | "
        f"display stride {info.stride}",
        fontsize=12,
    )
    figure.savefig(output, dpi=160, transparent=True)
    plt.close(figure)
    return info


def open_interactive_viewer(
    input_path: str | Path = DEFAULT_INPUT,
    threshold: float | None = None,
    max_dimension: int = 256,
) -> VolumeInfo | Path:
    """Open a supported volume, point cloud, or mesh in native PyVista."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Interactive viewing requires PyVista. Install it with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    path = resolve_visual_input(input_path)
    suffix = path.suffix.lower()

    if suffix == ".obj":
        mesh = pv.read(path)
        if mesh.n_points == 0:
            raise ValueError(f"OBJ contains no vertices: {path}")
        plotter = pv.Plotter(window_size=(1200, 900), title=path.name)
        plotter.set_background("#15191f", top="#283442")
        plotter.add_mesh(
            mesh,
            color="#82c6e8",
            smooth_shading=True,
            specular=0.3,
            show_edges=True,
            edge_color="#31485a",
        )
        _finish_interactive_plot(plotter, path.name)
        return path

    if suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim == 2 and 3 <= array.shape[1] <= 16:
            points = np.asarray(array[:, :3], dtype=np.float32)
            finite = np.all(np.isfinite(points), axis=1)
            points = points[finite]
            if not len(points):
                raise ValueError(f"NPY contains no finite XYZ points: {path}")
            cloud = pv.PolyData(points)
            plotter = pv.Plotter(window_size=(1200, 900), title=path.name)
            plotter.set_background("#15191f", top="#283442")
            plotter.add_mesh(
                cloud,
                color="#82c6e8",
                point_size=4,
                render_points_as_spheres=True,
            )
            _finish_interactive_plot(plotter, f"{path.name} | {len(points):,} points")
            return path
        if array.ndim != 3:
            raise ValueError(
                f"NPY must contain a 3D volume or an (N, 3..16) point array; "
                f"found shape {array.shape}"
            )

    volume, info = prepare_volume(path, threshold, max_dimension)
    xyz_volume = np.transpose(volume, (2, 1, 0))
    grid = pv.ImageData(dimensions=xyz_volume.shape)
    grid.spacing = (float(info.stride),) * 3
    grid.point_data["density"] = xyz_volume.ravel(order="F")

    plotter = pv.Plotter(window_size=(1200, 900), title=info.path.name)
    plotter.set_background("#15191f", top="#283442")
    actor_name = "lattice-surface"

    def update_surface(value: float) -> None:
        surface = grid.contour([float(value)], scalars="density")
        if surface.n_points:
            plotter.add_mesh(
                surface,
                name=actor_name,
                color="#82c6e8",
                smooth_shading=True,
                specular=0.3,
                reset_camera=False,
            )

    update_surface(info.threshold)
    low = float(np.percentile(_sample_volume(volume), 55))
    high = float(np.percentile(_sample_volume(volume), 99.5))
    plotter.add_slider_widget(
        update_surface,
        rng=(low, high),
        value=info.threshold,
        title="Density threshold",
        pointa=(0.18, 0.08),
        pointb=(0.82, 0.08),
        interaction_event="end",
    )
    _finish_interactive_plot(
        plotter,
        f"{info.path.name} | threshold {info.threshold:.3g}",
    )
    return info


def _finish_interactive_plot(plotter: object, label: str) -> None:
    """Add consistent controls and display a configured PyVista plotter."""
    plotter.add_axes()
    plotter.add_text(label, position="lower_left", font_size=10, color="white")
    plotter.add_text(
        "Drag: rotate  |  Wheel: zoom  |  Right-drag: pan  |  Q: close",
        position="upper_edge",
        font_size=10,
        color="white",
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.15)
    plotter.show()


def _threshold_arg(value: str) -> float | None:
    return None if value.lower() == "auto" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize TIFF/NPY volumes, NPY point clouds, or OBJ meshes"
    )
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--threshold", type=_threshold_arg, default=None)
    parser.add_argument("--max-dimension", type=int, default=192)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "missing-struts-volume.png"),
    )
    args = parser.parse_args()

    if args.interactive:
        info = open_interactive_viewer(args.input, args.threshold, args.max_dimension)
    else:
        info = render_volume_png(args.input, args.output, args.threshold, args.max_dimension)
        print(f"Saved visualization to {Path(args.output).resolve()}")
    if isinstance(info, VolumeInfo):
        print(
            f"Loaded {info.path.name}: shape={info.shape}, dtype={info.dtype}, "
            f"threshold={info.threshold:.3f}, stride={info.stride}"
        )
    else:
        print(f"Loaded {info.name}")


if __name__ == "__main__":
    main()
