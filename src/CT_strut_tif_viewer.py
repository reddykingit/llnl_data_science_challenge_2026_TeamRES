"""
CT_strut_tif_tool.py
====================

Single file tool for reading, calibrating, plotting and exploring X-ray
computed tomography (XCT) data of metallic strut based lattice structures
stored as TIFF.

It replaces the earlier pair of scripts. Everything lives here: the
reader, the segmentation, the batch figure export, the interactive slice
slider and the rotatable 3D surface view. The Matplotlib backend is now
chosen inside main(), after the mode is known, which is what previously
forced the split into two files.

Accepted input layouts, detected automatically
----------------------------------------------
    (A) one multi page TIFF per specimen   sample_001.tif = whole volume
    (B) one single page TIFF per slice     rec0001.tif, rec0002.tif, ...
    (C) one subfolder per specimen, each holding a slice sequence
    (D) a single TIFF file

Modes
-----
    --inspect    report the folder contents and exit
    --export     write the six figures as PNG (default when no mode given)
    --slider     on screen slice slider, XY / XZ / YZ
    --three-d    rotatable 3D surface, written to HTML and opened
    --all        export, then 3D, then slider

Examples
--------
    python CT_strut_tif_tool.py --inspect "D:/data/missing_struts/tif_stacks"
    python CT_strut_tif_tool.py "D:/data/missing_struts/tif_stacks" --voxel 10
    python CT_strut_tif_tool.py "D:/data/missing_struts/tif_stacks" --slider
    python CT_strut_tif_tool.py "D:/data/missing_struts/tif_stacks" --all --which 2

Pressing Run in an editor passes no arguments, in which case the USER
SETTINGS block immediately below is used instead.

License: MIT
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ======================================================================
# USER SETTINGS  -- used only when the script is started with no command
# line arguments, which is what the Run button in an editor does.
# ======================================================================
DEFAULT_PATH = r"C:\Users\Owner\Downloads\DSC\llnl_data_science_challenge_2026_TeamRES\data\missing_struts\tif_stacks"
DEFAULT_VOXEL_UM = 10.0        # isotropic voxel size in micrometres
DEFAULT_WHICH = 0              # specimen index when the folder holds many
DEFAULT_MODE = "all"           # "export", "slider", "3d", or "all"
# ======================================================================

try:
    import tifffile
except ImportError:  # pragma: no cover
    sys.exit("tifffile is required:  pip install tifffile")

import matplotlib                      # note: no matplotlib.use() here, the
import matplotlib.pyplot as plt        # backend is selected inside main()
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons, Slider

try:
    from skimage.filters import median, threshold_otsu
    from skimage.morphology import disk, remove_small_objects
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

TIF_EXT = (".tif", ".tiff")


# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
@dataclass
class ScanConfig:
    """Acquisition and processing metadata.

    voxel_size_um : isotropic edge length of one voxel in micrometres,
                    taken from the CT log file or the simulation settings.
                    Affects axis calibration and the scale bar only.
    p_low, p_high : lower and upper percentiles used for contrast
                    normalisation. 0.5 and 99.5 reject the extreme tails
                    caused by detector noise and by metal induced streaks.
    median_radius : radius in pixels of the disk shaped median kernel
                    applied before thresholding; suppresses salt and
                    pepper noise without blurring strut edges.
    min_object_px : connected components smaller than this are removed
                    from the binary mask as noise speckle.
    """

    voxel_size_um: float = 10.0
    p_low: float = 0.5
    p_high: float = 99.5
    median_radius: int = 1
    min_object_px: int = 32
    cmap: str = "gray"
    outdir: str = "ct_figures"
    dpi: int = 200


# ----------------------------------------------------------------------
# 2. Backend selection
# ----------------------------------------------------------------------
def select_backend(need_window: bool, verbose: bool = True) -> bool:
    """Choose a Matplotlib backend to match the requested mode.

    need_window : True for the slider, False for file export.

    The backend is a process wide setting, so it must be fixed before any
    figure is created. Agg renders into memory and writes files without a
    display, which is what batch export wants; the slider instead needs a
    backend that owns a window and an event loop. Returns True if a window
    capable backend is active.
    """
    if not need_window:
        plt.switch_backend("Agg")
        return False
    if not matplotlib.get_backend().lower().startswith("agg"):
        return True
    for name in ("QtAgg", "TkAgg", "Qt5Agg", "GTK3Agg", "MacOSX"):
        try:
            plt.switch_backend(name)
            if verbose:
                print(f"[backend] {matplotlib.get_backend()}")
            return True
        except Exception:
            continue
    if verbose:
        print("[backend] no interactive backend available; install one with\n"
              "          pip install pyqt5      (or)      pip install tk")
    plt.switch_backend("Agg")
    return False


# ----------------------------------------------------------------------
# 3. Folder inspection and input dispatch
# ----------------------------------------------------------------------
def _natural_key(text) -> list:
    """Sort helper so that slice_2 precedes slice_10."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", str(text))]


def _list_tiffs(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in TIF_EXT]
    return sorted(files, key=lambda p: _natural_key(p.name))


def inspect_folder(path: str, max_report: int = 12) -> dict:
    """Report what is inside a TIFF folder and decide how to read it.

    Returns a dictionary with keys
        layout  : 'stacks' | 'slices' | 'subfolders' | 'single'
        files   : list of Path objects (stacks or slices)
        folders : list of Path objects (only for 'subfolders')
    """
    p = Path(path)
    if p.is_file():
        with tifffile.TiffFile(p) as tf:
            n = len(tf.pages)
            shp, dt = tf.pages[0].shape, tf.pages[0].dtype
        print(f"[inspect] single file: {p.name}, {n} page(s), "
              f"page shape {shp}, dtype {dt}")
        return {"layout": "single", "files": [p], "folders": []}

    if not p.is_dir():
        raise FileNotFoundError(f"path does not exist: {p}")

    files = _list_tiffs(p)
    subdirs = sorted([d for d in p.iterdir() if d.is_dir()],
                     key=lambda d: _natural_key(d.name))

    if not files and subdirs:
        print(f"[inspect] {p}\n          contains {len(subdirs)} subfolder(s); "
              f"treating each as one specimen")
        for d in subdirs[:max_report]:
            print(f"          {d.name}: {len(_list_tiffs(d))} TIFF files")
        return {"layout": "subfolders", "files": [], "folders": subdirs}

    if not files:
        raise FileNotFoundError(f"no TIFF files or subfolders found in {p}")

    with tifffile.TiffFile(files[0]) as tf:
        n_pages = len(tf.pages)
        shp, dt = tf.pages[0].shape, tf.pages[0].dtype

    layout = "stacks" if n_pages > 1 else "slices"
    print(f"[inspect] {p}\n"
          f"          {len(files)} TIFF file(s), first file has {n_pages} page(s)\n"
          f"          page shape {shp}, dtype {dt}\n"
          f"          layout detected: {layout}")

    if layout == "stacks":
        print(f"{'idx':>4}  {'file':<38} {'pages':>6}  {'shape':>14}  {'MB':>7}")
        for i, f in enumerate(files[:max_report]):
            with tifffile.TiffFile(f) as tf:
                npg = len(tf.pages)
                s = tf.pages[0].shape
            print(f"{i:>4}  {f.name[:38]:<38} {npg:>6}  "
                  f"{str(s):>14}  {f.stat().st_size/1e6:>7.1f}")
        if len(files) > max_report:
            print(f"      ... and {len(files) - max_report} more")
    return {"layout": layout, "files": files, "folders": subdirs}


def load_volume(path: str,
                which: int = 0,
                z_range: Optional[Sequence[int]] = None,
                step: int = 1) -> Tuple[np.ndarray, str]:
    """Load one specimen volume as a 3D array of shape (Nz, Ny, Nx).

    which   : index of the specimen when the folder holds several stacks
              or several subfolders; ignored for a plain slice sequence.
    z_range : optional (z_start, z_stop) restricting the slices loaded,
              which keeps memory use manageable for large volumes.
    step    : slice decimation factor; step = 4 loads every fourth slice.
    """
    info = inspect_folder(path)
    layout = info["layout"]

    if layout in ("stacks", "single"):
        files = info["files"]
        which = min(max(which, 0), len(files) - 1)
        target = files[which]
        vol = tifffile.imread(str(target))
        label = target.name
    elif layout == "slices":
        files = info["files"]
        if z_range is not None:
            files = files[z_range[0]:z_range[1]]
        files = files[::step]
        first = tifffile.imread(str(files[0]))
        vol = np.empty((len(files), *first.shape), dtype=first.dtype)
        vol[0] = first
        for k, f in enumerate(files[1:], start=1):
            vol[k] = tifffile.imread(str(f))
        label = Path(path).name
        z_range, step = None, 1          # already applied above
    else:                                 # subfolders
        folders = info["folders"]
        which = min(max(which, 0), len(folders) - 1)
        return load_volume(str(folders[which]), 0, z_range, step)

    vol = np.asarray(vol)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    elif vol.ndim == 4:                  # (Nz, Ny, Nx, C) -> first channel
        vol = vol[..., 0]
    if z_range is not None:
        vol = vol[z_range[0]:z_range[1]]
    if step > 1:
        vol = vol[::step]

    print(f"[load] {label}\n"
          f"       shape (Nz, Ny, Nx) = {vol.shape}, dtype = {vol.dtype}\n"
          f"       grey range = [{vol.min()}, {vol.max()}]")
    return vol, label


# ----------------------------------------------------------------------
# 4. Pre processing and segmentation
# ----------------------------------------------------------------------
def normalise(image: np.ndarray, cfg: ScanConfig) -> np.ndarray:
    """Percentile contrast stretch to [0, 1].

        I_n = clip((I - I_low) / (I_high - I_low), 0, 1)

    I      : raw grey value, a proxy for linear attenuation
    I_low  : grey value at percentile p_low
    I_high : grey value at percentile p_high
    I_n    : normalised, dimensionless value in [0, 1]
    """
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, [cfg.p_low, cfg.p_high])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def _drop_small(mask: np.ndarray, size_px: int) -> np.ndarray:
    """Remove components below size_px, tolerant of the scikit-image
    keyword change (min_size up to 0.25, max_size from 0.26)."""
    try:
        return remove_small_objects(mask, max_size=size_px)
    except TypeError:
        return remove_small_objects(mask, min_size=size_px)


def _otsu_numpy(img: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold using NumPy only (fallback if scikit-image is absent)."""
    counts, edges = np.histogram(img.ravel(), bins=nbins, range=(0.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])
    p = counts.astype(np.float64) / counts.sum()
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    valid = (w0 > 0) & (w1 > 0)
    sig = np.zeros_like(w0)
    sig[valid] = ((mu_t * w0[valid] - mu[valid]) ** 2) / (w0[valid] * w1[valid])
    return float(centres[np.argmax(sig)])


def global_threshold(volume: np.ndarray, cfg: ScanConfig,
                     n_sample: int = 24) -> float:
    """One Otsu threshold for the whole volume, estimated from evenly
    spaced sample slices. A single global value is preferable to per slice
    thresholds because a slice containing almost no metal has no genuine
    second mode, and Otsu applied to it merely splits the noise."""
    nz = volume.shape[0]
    idx = np.linspace(0, nz - 1, min(n_sample, nz)).astype(int)
    sub = normalise(volume[idx], cfg)
    return float(threshold_otsu(sub) if _HAS_SKIMAGE else _otsu_numpy(sub))


def segment_slice(image2d: np.ndarray, cfg: ScanConfig,
                  t: Optional[float] = None):
    """Binarise one slice into metal (True) and air (False)."""
    img = normalise(image2d, cfg)
    if _HAS_SKIMAGE and cfg.median_radius > 0:
        u8 = (img * 255).astype(np.uint8)
        img = median(u8, disk(cfg.median_radius)).astype(np.float32) / 255.0
    if t is None:
        t = threshold_otsu(img) if _HAS_SKIMAGE else _otsu_numpy(img)
    mask = img > t
    if _HAS_SKIMAGE and cfg.min_object_px > 0:
        mask = _drop_small(mask, cfg.min_object_px)
    return mask, float(t)


def density_profile(volume: np.ndarray, cfg: ScanConfig, t: float):
    """Relative density and per slice solid area fraction.

        rho = N_solid / N_total,   A_f(k) = sum_ij B(i,j,k) / (Nx * Ny)

    N_solid : voxels classified as metal in the loaded volume
    N_total : all voxels in the field of view
    B       : binary mask, 1 for metal and 0 for air
    A_f(k)  : solid area fraction of slice k; its periodicity along z
              reflects the unit cell repeat, and a local drop flags a
              missing or thinned strut
    """
    nz, ny, nx = volume.shape
    af = np.empty(nz)
    for k in range(nz):
        m, _ = segment_slice(volume[k], cfg, t)
        af[k] = m.sum() / (nx * ny)
    return float(af.mean()), af


# ----------------------------------------------------------------------
# 5. Static figures
# ----------------------------------------------------------------------
def _add_scalebar(ax, n_px: int, cfg: ScanConfig, frac: float = 0.25):
    target = frac * n_px * cfg.voxel_size_um
    nice = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500,
                     1000, 2000, 5000], dtype=float)
    bar_um = nice[np.argmin(np.abs(nice - target))]
    bar_px = bar_um / cfg.voxel_size_um
    x0, y0 = 0.04 * n_px, 0.94 * n_px
    ax.add_patch(Rectangle((x0, y0), bar_px, 0.012 * n_px,
                           color="white", ec="black", lw=0.4))
    lab = f"{bar_um:.0f} um" if bar_um < 1000 else f"{bar_um/1000:.1f} mm"
    ax.text(x0 + bar_px / 2, y0 - 0.015 * n_px, lab,
            color="white", ha="center", va="bottom", fontsize=8)


def _save(fig, cfg, fname):
    os.makedirs(cfg.outdir, exist_ok=True)
    out = os.path.join(cfg.outdir, fname)
    fig.tight_layout()
    fig.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")
    return out


def plot_single_slice(vol, cfg, index=None, tag="", fname="fig1_single_slice.png"):
    k = vol.shape[0] // 2 if index is None else index
    img = normalise(vol[k], cfg)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    im = ax.imshow(img, cmap=cfg.cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{tag}  slice z = {k} "
                 f"({k * cfg.voxel_size_um / 1000:.2f} mm)", fontsize=10)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    _add_scalebar(ax, img.shape[1], cfg)
    fig.colorbar(im, ax=ax, fraction=0.046, label="normalised attenuation")
    return _save(fig, cfg, fname)


def plot_montage(vol, cfg, n_panels=9, fname="fig2_montage.png"):
    nz = vol.shape[0]
    n_panels = min(n_panels, nz)
    idx = np.linspace(0, nz - 1, n_panels).astype(int)
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, idx):
        ax.imshow(normalise(vol[k], cfg), cmap=cfg.cmap, vmin=0, vmax=1)
        ax.set_title(f"z = {k}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.suptitle("Evenly spaced slices through the strut network", fontsize=10)
    return _save(fig, cfg, fname)


def plot_histogram(vol, cfg, t, fname="fig3_histogram.png"):
    mid = normalise(vol[vol.shape[0] // 2], cfg)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(mid.ravel(), bins=256, color="0.35", log=True)
    ax.axvline(t, color="crimson", lw=1.4, label=f"Otsu threshold = {t:.3f}")
    ax.set_xlabel("normalised grey value")
    ax.set_ylabel("voxel count (log scale)")
    ax.set_title("Grey value distribution: air and metal populations", fontsize=10)
    ax.legend(fontsize=8)
    return _save(fig, cfg, fname)


def plot_orthoviews(vol, cfg, fname="fig4_orthoviews.png"):
    nz, ny, nx = vol.shape
    if nz < 3:
        print("[warn] orthogonal views need a stack; skipping")
        return None
    views = (normalise(vol[nz // 2], cfg),
             normalise(vol[:, ny // 2, :], cfg),
             normalise(vol[:, :, nx // 2], cfg))
    titles = (f"XY plane, z = {nz//2}",
              f"XZ plane, y = {ny//2}",
              f"YZ plane, x = {nx//2}")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, img, ttl in zip(axes, views, titles):
        ax.imshow(img, cmap=cfg.cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")
    fig.suptitle("Orthogonal sections of the reconstructed volume", fontsize=10)
    return _save(fig, cfg, fname)


def plot_segmentation(vol, cfg, t, index=None, fname="fig5_segmentation.png"):
    k = vol.shape[0] // 2 if index is None else index
    mask, _ = segment_slice(vol[k], cfg, t)
    rho, af = density_profile(vol, cfg, t)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    axes[0].imshow(normalise(vol[k], cfg), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"grey level, z = {k}", fontsize=9)
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"binary metal mask (t = {t:.3f})", fontsize=9)
    for ax in axes[:2]:
        ax.axis("off")

    z_mm = np.arange(len(af)) * cfg.voxel_size_um / 1000.0
    axes[2].plot(z_mm, af * 100, lw=1.2, color="navy")
    axes[2].axhline(rho * 100, color="crimson", ls="--", lw=1.0,
                    label=f"mean = {rho*100:.2f} %")
    axes[2].set_xlabel("z position (mm)")
    axes[2].set_ylabel("solid area fraction (%)")
    axes[2].set_title("Through thickness density profile", fontsize=9)
    axes[2].legend(fontsize=8)
    print(f"[metrics] relative density = {rho*100:.2f} %")
    return _save(fig, cfg, fname)


def plot_projections(vol, cfg, fname="fig6_projections.png"):
    """Mean and maximum intensity projections along the three axes.

    A maximum intensity projection collapses the volume onto a plane by
    keeping the largest grey value along the viewing direction. Because an
    intact strut is the brightest object along its own line of sight, a
    strut that is absent leaves an unmistakable gap, which makes this the
    quickest visual check for missing struts.
    """
    v = normalise(vol, cfg)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    for j, axis in enumerate((0, 1, 2)):
        axes[0, j].imshow(v.mean(axis=axis), cmap=cfg.cmap)
        axes[0, j].set_title(f"mean projection along axis {axis}", fontsize=9)
        axes[1, j].imshow(v.max(axis=axis), cmap=cfg.cmap)
        axes[1, j].set_title(f"maximum projection along axis {axis}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Projections: gaps indicate missing or broken struts", fontsize=10)
    return _save(fig, cfg, fname)


def export_figures(vol, cfg, t, label="", index=None, panels=9):
    """Write the full set of six figures."""
    plot_single_slice(vol, cfg, index=index, tag=label)
    plot_montage(vol, cfg, n_panels=panels)
    plot_histogram(vol, cfg, t)
    plot_orthoviews(vol, cfg)
    plot_segmentation(vol, cfg, t, index=index)
    plot_projections(vol, cfg)


# ----------------------------------------------------------------------
# 6. Interactive slice slider
# ----------------------------------------------------------------------
def slice_slider(volume: np.ndarray, cfg: ScanConfig, title: str = ""):
    """Scroll through the stack with a slider, in any of the three planes.

    The volume is normalised once, globally, so the grey scale stays fixed
    as the slider moves. Normalising each slice independently would
    rescale the display continuously and make a genuine change in material
    content indistinguishable from a change in the contrast mapping.
    """
    interactive = select_backend(need_window=True)
    vol = normalise(volume, cfg)
    nz, ny, nx = vol.shape
    state = {"axis": 0, "index": nz // 2}

    def plane(axis, i):
        if axis == 0:
            return vol[i]
        if axis == 1:
            return vol[:, i, :]
        return vol[:, :, i]

    def n_slices(axis):
        return (nz, ny, nx)[axis]

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    fig.subplots_adjust(left=0.26, bottom=0.16)
    im = ax.imshow(plane(0, state["index"]), cmap=cfg.cmap,
                   vmin=0, vmax=1, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("normalised attenuation")

    def refresh():
        a, i = state["axis"], state["index"]
        img = plane(a, i)
        im.set_data(img)
        im.set_extent((-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
        pos_mm = i * cfg.voxel_size_um / 1000.0
        lab = ("XY plane, z", "XZ plane, y", "YZ plane, x")[a]
        ax.set_title(f"{title}   {lab} = {i}  ({pos_mm:.2f} mm)", fontsize=10)
        fig.canvas.draw_idle()

    ax_sl = fig.add_axes([0.26, 0.06, 0.62, 0.03])
    slider = Slider(ax_sl, "slice", 0, nz - 1, valinit=state["index"], valstep=1)

    def on_slide(val):
        state["index"] = int(val)
        refresh()

    slider.on_changed(on_slide)

    ax_radio = fig.add_axes([0.02, 0.62, 0.18, 0.18])
    radio = RadioButtons(ax_radio, ("XY (axial)", "XZ", "YZ"))

    def on_axis(label):
        state["axis"] = {"XY (axial)": 0, "XZ": 1, "YZ": 2}[label]
        n = n_slices(state["axis"])
        state["index"] = n // 2
        slider.valmax = n - 1
        slider.ax.set_xlim(0, n - 1)
        slider.set_val(state["index"])
        refresh()

    radio.on_clicked(on_axis)

    def on_scroll(event):
        n = n_slices(state["axis"])
        step = 1 if event.button == "up" else -1
        slider.set_val(int(np.clip(state["index"] + step, 0, n - 1)))

    def on_key(event):
        n = n_slices(state["axis"])
        step = {"right": 1, "up": 1, "left": -1, "down": -1,
                "pageup": 10, "pagedown": -10}.get(event.key)
        if step:
            slider.set_val(int(np.clip(state["index"] + step, 0, n - 1)))

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Matplotlib widgets are garbage collected if no reference survives the
    # function call, after which the slider silently stops responding.
    fig._widgets = (slider, radio)

    refresh()
    if interactive:
        print("[view] close the window to exit")
        plt.show()
    else:
        os.makedirs(cfg.outdir, exist_ok=True)
        out = os.path.join(cfg.outdir, "slider_fallback.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[save] no window available, mid slice written to "
              f"{os.path.abspath(out)}")
    return fig, slider, radio


# ----------------------------------------------------------------------
# 7. Three dimensional surface rendering
# ----------------------------------------------------------------------
def render_3d(volume: np.ndarray, cfg: ScanConfig, t: Optional[float] = None,
              downsample: int = 2, step_size: int = 2,
              out_html: str = "strut_3d.html", open_browser: bool = True):
    """Extract and display the metal surface as a rotatable 3D mesh.

    t          : iso value at which the surface is extracted, in normalised
                 grey units. Defaults to the global Otsu threshold, so the
                 surface follows the statistically optimal metal to air
                 boundary rather than an arbitrary level.
    downsample : decimation applied to all three axes before extraction. A
                 factor of 2 cuts the voxel count by 8 and the triangle
                 count by roughly 4, usually the difference between a scene
                 that rotates smoothly and one that stalls.
    step_size  : marching cubes stride; larger values give a coarser mesh.

    The marching cubes algorithm visits each cubic cell, classifies its
    eight corners as inside or outside the iso surface, selects the
    matching triangle configuration from a lookup table, and interpolates
    vertex positions along the cell edges, giving sub voxel precision.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[3d] plotly is not installed; skipping the 3D view\n"
              "     install it with:  pip install plotly")
        return None
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("[3d] scikit-image is required for surface extraction; skipping")
        return None

    vol = normalise(volume, cfg)
    if downsample > 1:
        vol = vol[::downsample, ::downsample, ::downsample]
    if t is None:
        t = global_threshold(volume, cfg)

    print(f"[3d] volume {vol.shape}, iso value = {t:.3f}, extracting surface ...")
    verts, faces, _, _ = marching_cubes(vol, level=t, step_size=step_size)

    s = cfg.voxel_size_um * downsample / 1000.0     # mm per decimated voxel
    z, y, x = verts[:, 0] * s, verts[:, 1] * s, verts[:, 2] * s
    print(f"[3d] {len(verts):,} vertices, {len(faces):,} triangles")

    mesh = go.Mesh3d(x=x, y=y, z=z,
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     color="#9aa3ad", opacity=1.0, flatshading=True,
                     lighting=dict(ambient=0.45, diffuse=0.85, specular=0.25,
                                   roughness=0.6, fresnel=0.15),
                     lightposition=dict(x=100, y=200, z=300),
                     name="strut surface")
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=f"Strut network surface, iso value {t:.3f}",
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)",
                   zaxis_title="z (mm)", aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
        margin=dict(l=0, r=0, t=40, b=0), template="plotly_white")

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    full = os.path.abspath(out_html)
    print(f"[save] {full}")
    if open_browser:
        try:
            webbrowser.open(Path(full).as_uri())
        except Exception:
            pass
    return full


# ----------------------------------------------------------------------
# 8. Synthetic phantom (for testing without data)
# ----------------------------------------------------------------------
def synthetic_strut_phantom(n=160, radius_px=9.0, noise=0.06, seed=0):
    """Cubic strut phantom with noise and porosity, for offline testing."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2.0
    p, q, r = x - c, y - c, z - c
    solid = np.zeros((n, n, n), dtype=bool)
    for a, b in ((p, q), (p, r), (q, r)):
        solid |= (a ** 2 + b ** 2) < radius_px ** 2
    for s in (1, -1):
        d = (p + s * q) / np.sqrt(2)
        solid |= (p ** 2 + q ** 2 + r ** 2 - d ** 2) < (0.8 * radius_px) ** 2
    vol = np.where(solid, 0.80, 0.10).astype(np.float32)
    vol[solid & (rng.random((n, n, n)) > 0.9985)] = 0.15
    vol += rng.normal(0.0, noise, vol.shape).astype(np.float32)
    return np.clip(vol * 65535, 0, 65535).astype(np.uint16)


# ----------------------------------------------------------------------
# 9. Command line interface
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Read, plot and explore CT TIFF data of metallic struts.")
    ap.add_argument("path", nargs="?", default=None,
                    help="folder of TIFF stacks or slices, or a single TIFF")
    ap.add_argument("--inspect", action="store_true",
                    help="report the folder contents and exit")
    ap.add_argument("--export", action="store_true", help="write the six figures")
    ap.add_argument("--slider", action="store_true", help="on screen slice slider")
    ap.add_argument("--three-d", dest="three_d", action="store_true",
                    help="rotatable 3D surface in the browser")
    ap.add_argument("--all", action="store_true",
                    help="export, then 3D, then slider")
    ap.add_argument("--which", type=int, default=0, help="specimen index")
    ap.add_argument("--voxel", type=float, default=10.0,
                    help="isotropic voxel size in micrometres")
    ap.add_argument("--zrange", type=int, nargs=2, default=None,
                    metavar=("Z0", "Z1"), help="slice index range to load")
    ap.add_argument("--step", type=int, default=1, help="load every Nth slice")
    ap.add_argument("--slice", type=int, default=None,
                    help="slice index used in the single slice figure")
    ap.add_argument("--panels", type=int, default=9, help="montage panel count")
    ap.add_argument("--downsample", type=int, default=2,
                    help="decimation before surface extraction (default 2)")
    ap.add_argument("--stepsize", type=int, default=2,
                    help="marching cubes stride (default 2)")
    ap.add_argument("--cmap", default="gray", help="matplotlib colour map")
    ap.add_argument("--outdir", default="ct_figures", help="output folder")
    ap.add_argument("--demo", action="store_true", help="synthetic phantom")
    args = ap.parse_args(argv)

    # No command line arguments at all, which is what an editor's Run button
    # produces: fall back to the USER SETTINGS block at the top of the file.
    if argv is None and len(sys.argv) == 1:
        if DEFAULT_PATH and Path(DEFAULT_PATH).exists():
            args.path = DEFAULT_PATH
            args.voxel = DEFAULT_VOXEL_UM
            args.which = DEFAULT_WHICH
            args.export = DEFAULT_MODE in ("export", "all")
            args.slider = DEFAULT_MODE in ("slider", "all")
            args.three_d = DEFAULT_MODE in ("3d", "all")
            print("[settings] using DEFAULT_PATH from the top of this file")
        else:
            print("[settings] DEFAULT_PATH is unset or does not exist, so the\n"
                  "           synthetic phantom is used. Edit DEFAULT_PATH at\n"
                  "           the top of this file, or pass a folder as an\n"
                  "           argument from a terminal.")

    if args.all:
        args.export = args.slider = args.three_d = True
    if not (args.export or args.slider or args.three_d or args.inspect):
        args.export = True                      # sensible default

    if args.inspect:
        if args.path is None:
            sys.exit("--inspect needs a path")
        inspect_folder(args.path)
        return

    # A relative output folder is resolved against this script, not against
    # the shell working directory, which is often somewhere unrelated when
    # the script is launched from an editor.
    if not os.path.isabs(args.outdir):
        args.outdir = str(Path(__file__).resolve().parent / args.outdir)
    print(f"[output] {args.outdir}")

    cfg = ScanConfig(voxel_size_um=args.voxel, cmap=args.cmap, outdir=args.outdir)

    if args.demo or args.path is None:
        print("[demo] synthetic strut phantom (pass a path to use real data)")
        vol, label = synthetic_strut_phantom(), "phantom"
    else:
        vol, label = load_volume(args.path, which=args.which,
                                 z_range=args.zrange, step=args.step)

    t = global_threshold(vol, cfg)
    print(f"[segment] global Otsu threshold = {t:.4f}")

    # File writing first, under Agg, then the window last, because plt.show()
    # blocks until the window is closed.
    if args.export:
        select_backend(need_window=False)
        export_figures(vol, cfg, t, label=label, index=args.slice,
                       panels=args.panels)
    if args.three_d:
        render_3d(vol, cfg, t=t, downsample=args.downsample,
                  step_size=args.stepsize,
                  out_html=os.path.join(cfg.outdir, "strut_3d.html"))
    if args.slider:
        slice_slider(vol, cfg, title=label)

    print("[done]", os.path.abspath(cfg.outdir))


if __name__ == "__main__":
    main()