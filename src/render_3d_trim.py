"""
render_3d_trim.py
=================

Standalone script. Reads an XCT TIFF volume of a metallic strut lattice,
extracts the metal surface by marching cubes, and writes an interactive
HTML page with three sliders (x, y, z) that trim the view from the origin.

Run it directly. There are no imports from any other file of mine.

    python render_3d_trim.py
    python render_3d_trim.py "D:/data/missing_struts/tif_stacks" --voxel 10

Requirements:  pip install tifffile scikit-image plotly numpy
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import webbrowser
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "2026-07-23-f"   # printed at startup, so the output can
                                  # always be traced to the file that made it

# ======================================================================
# USER SETTINGS  -- used when the script is started with no arguments,
# which is what the Run button in VS Code does.
# ======================================================================
DEFAULT_PATH = r"C:\Users\Owner\Downloads\DSC\llnl_data_science_challenge_2026_TeamRES\data\missing_struts\tif_stacks"
DEFAULT_VOXEL_UM = 10.0     # isotropic voxel edge length, micrometres
DEFAULT_WHICH = 0           # specimen index if the folder holds several

# --- what to render -----------------------------------------------------
# "detail"   a small cube at native resolution. Faithful geometry, correct
#            dimensions, but only a few unit cells.
# "overview" the whole specimen, necessarily coarsened. Shows topology and
#            locates missing struts, but thickens every feature slightly, so
#            no dimension should be measured from it.
# "custom"   ignore the presets and use the individual settings below.
DEFAULT_MODE = "detail"

PRESETS = {
    "detail": dict(roicube=300, downsample=1, coarsen=False,
                   maxfaces=6_000_000, sigma=1.0, occ=0.50, median=1),
    # At decimation 3 a 74 um strut spans only 2.5 blocks. Two settings must
    # change with it. The Gaussian is switched off, because a sigma of one
    # block blurs across roughly the whole strut width and smears its
    # occupancy below the contour level, which erases struts while leaving
    # the thicker nodes standing. The contour level is lowered, because
    # blocks straddling so thin a feature rarely reach a metal fraction of
    # one half, so contouring at 0.5 cuts inside the strut. The median filter
    # is skipped only because it is slow on 519 million voxels.
    "overview": dict(roicube=None, downsample=3, coarsen=True,
                     maxfaces=3_000_000, sigma=0.0, occ=0.35, median=0),
}

# --- region of interest -------------------------------------------------
# Used when DEFAULT_MODE is "custom". A centred cube of this many voxels per
# side, or None for the whole specimen.
DEFAULT_ROICUBE = 300

# --- reduction ----------------------------------------------------------
DEFAULT_DOWNSAMPLE = 1          # block averaging factor; 1 means no reduction
DEFAULT_MAXFACES = 6_000_000    # triangle budget
DEFAULT_ALLOW_COARSEN = False   # when the budget is exceeded: False warns and
                                # proceeds, True averages into larger blocks
DEFAULT_STEPSIZE = 1            # marching cubes stride; leave at 1
HARD_FACE_LIMIT = 8_000_000     # absolute ceiling. Past this the mesh is
                                # coarsened whether or not coarsening is
                                # permitted, because a page the browser
                                # cannot open is of less use than a coarse
                                # surface. Roughly 200 MB of payload.

# --- segmentation and surface ------------------------------------------
DEFAULT_CLEAN = True        # False reproduces the original unfiltered path
DEFAULT_OCC = 0.5           # block metal fraction taken as the interface
DEFAULT_BINARISE = False    # True enables morphology but fragments thin struts
DEFAULT_SIGMA = 1.0         # Gaussian smoothing before extraction
DEFAULT_CLOSE = 1           # closing radius, binarised mode only
DEFAULT_MINOBJ = 64         # smallest surviving cluster, binarised mode only
DEFAULT_FILLHOLES = 64      # largest cavity filled, binarised mode only
DEFAULT_LARGEST = False     # keep only the largest connected component

# --- threshold ----------------------------------------------------------
DEFAULT_THRESHOLD = "iso50"     # "otsu", "iso50", or a number in [0, 1].
                                # Otsu maximises the variance between the two
                                # grey populations and is biased when they are
                                # very unequal in size, drifting towards the
                                # air population when metal is only a few
                                # percent of the field. ISO 50 places the
                                # level between the two histogram peaks.
DEFAULT_ISO_FRACTION = 0.45     # where between the air peak and the metal
                                # peak the surface sits. 0.5 is the unbiased
                                # half maximum; lower values sit nearer air
                                # and render thicker struts. 0.40 to 0.50 is
                                # defensible, below 0.35 noise is absorbed.
DEFAULT_MEDIAN = 1              # radius of the 3D median filter on the grey
                                # volume before thresholding, in voxels. A
                                # radius of 1 removes isolated noise voxels
                                # and leaves any feature three voxels or
                                # wider untouched. 0 disables it, which is
                                # advisable on the full volume for speed.

DEFAULT_OUT = "strut_3d_trim.html"
# ======================================================================

P_LOW, P_HIGH = 0.5, 99.5   # percentiles for the contrast stretch
TIF_EXT = (".tif", ".tiff")


# ----------------------------------------------------------------------
# 1. Reading
# ----------------------------------------------------------------------
def _natural_key(text):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", str(text))]


def load_volume(path: str, which: int = 0, step: int = 1):
    """Load one specimen as a 3D array of shape (Nz, Ny, Nx).

    path  : a single TIFF, a folder of multi page TIFF stacks, a folder of
            single page slice files, or a folder of such folders.
    which : specimen index when several are present.
    step  : slice decimation applied at read time, to bound memory use.
    """
    import tifffile

    p = Path(path)
    if p.is_file():
        vol, label = tifffile.imread(str(p)), p.name
    elif p.is_dir():
        files = sorted([f for f in p.iterdir()
                        if f.is_file() and f.suffix.lower() in TIF_EXT],
                       key=lambda f: _natural_key(f.name))
        subdirs = sorted([d for d in p.iterdir() if d.is_dir()],
                         key=lambda d: _natural_key(d.name))
        if not files and subdirs:
            which = min(max(which, 0), len(subdirs) - 1)
            print(f"[load] entering subfolder {subdirs[which].name}")
            return load_volume(str(subdirs[which]), 0, step)
        if not files:
            sys.exit(f"no TIFF files or subfolders in {p}")

        with tifffile.TiffFile(files[0]) as tf:
            n_pages = len(tf.pages)

        if n_pages > 1:                       # one stack per specimen
            which = min(max(which, 0), len(files) - 1)
            vol, label = tifffile.imread(str(files[which])), files[which].name
        else:                                 # one file per slice
            files = files[::step]
            first = tifffile.imread(str(files[0]))
            vol = np.empty((len(files), *first.shape), dtype=first.dtype)
            vol[0] = first
            for k, f in enumerate(files[1:], start=1):
                vol[k] = tifffile.imread(str(f))
            label, step = p.name, 1
    else:
        sys.exit(f"path does not exist: {p}")

    vol = np.asarray(vol)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    elif vol.ndim == 4:
        vol = vol[..., 0]
    if step > 1:
        vol = vol[::step]

    print(f"[load] {label}\n"
          f"       shape (Nz, Ny, Nx) = {vol.shape}, dtype = {vol.dtype}\n"
          f"       grey range = [{vol.min()}, {vol.max()}]")
    return vol, label


# ----------------------------------------------------------------------
# 2. Normalisation and threshold
# ----------------------------------------------------------------------
def normalise(image: np.ndarray) -> np.ndarray:
    """Percentile contrast stretch to the unit interval.

        I_n = clip((I - I_low) / (I_high - I_low), 0, 1)

    I      : raw grey value, a proxy for linear attenuation
    I_low  : grey value at percentile P_LOW
    I_high : grey value at percentile P_HIGH
    I_n    : normalised, dimensionless value in [0, 1]
    """
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, [P_LOW, P_HIGH])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def _otsu_numpy(img: np.ndarray, nbins: int = 256) -> float:
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


def _otsu_from_counts(counts, centres) -> float:
    """Otsu level given a histogram, used only to split the two populations."""
    p = counts.astype(np.float64) / max(counts.sum(), 1)
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centres)
    ok = (w0 > 0) & (w1 > 0)
    sig = np.zeros_like(w0)
    sig[ok] = ((mu[-1] * w0[ok] - mu[ok]) ** 2) / (w0[ok] * w1[ok])
    return float(centres[int(np.argmax(sig))])


def iso_threshold_raw(volume: np.ndarray, fraction: float = 0.5,
                      n_sample: int = 32, nbins: int = 512) -> float:
    """Threshold placed a given fraction of the way from air to metal.

    volume   : raw array; evenly spaced slices are sampled from it.
    fraction : position between the two peaks, 0.5 being the half maximum.
    nbins    : histogram resolution, in raw grey units.

    The peaks are located in the raw histogram rather than in the percentile
    stretched one. This matters when the metal phase is a small fraction of
    the field: the upper percentile then falls inside the metal population
    itself, everything above it is clipped to the top of the scale, and the
    resulting spike is mistaken for the metal mode, which biases the level
    upward and renders every strut too thin.

    The histogram is split at its Otsu level and the tallest bin on each side
    taken as that population's peak, because searching by height alone fails
    whenever the tail of the far larger air distribution outnumbers the metal
    peak. The value returned is in raw grey units.
    """
    nz = volume.shape[0]
    idx = np.linspace(0, nz - 1, min(n_sample, nz)).astype(int)
    sub = volume[idx]
    v0, v1 = float(sub.min()), float(sub.max())
    if v1 <= v0:
        return v0
    counts, edges = np.histogram(sub, bins=nbins, range=(v0, v1))
    centres = 0.5 * (edges[:-1] + edges[1:])

    split = _otsu_from_counts(counts, centres)
    low, high = centres < split, centres >= split
    if not low.any() or not high.any():
        return float(split)

    air = float(centres[low][np.argmax(counts[low])])
    metal = float(centres[high][np.argmax(counts[high])])
    raw_t = air + fraction * (metal - air)
    print(f"[threshold] raw histogram: air peak {air:.0f}, metal peak "
          f"{metal:.0f}, Otsu split {split:.0f}")
    print(f"[threshold] level at {fraction:.2f} of the way from air to "
          f"metal = {raw_t:.0f} raw")
    return float(raw_t)


def global_threshold(volume: np.ndarray, n_sample: int = 24) -> float:
    """One Otsu threshold for the whole stack, from evenly spaced slices.

    A single global value is preferable to per slice thresholds because a
    slice containing almost no metal has no genuine second mode, so Otsu
    applied to it merely partitions the noise.
    """
    nz = volume.shape[0]
    idx = np.linspace(0, nz - 1, min(n_sample, nz)).astype(int)
    return _otsu_numpy(normalise(volume[idx]))


# ----------------------------------------------------------------------
# 2b. Three dimensional merging and cleanup
# ----------------------------------------------------------------------
def percentile_bounds(volume: np.ndarray, n_sample: int = 24):
    """Grey values at the two clipping percentiles, from sampled slices.

    Returning the bounds in raw units lets the threshold be applied to the
    original integer array, so no float copy of the whole volume is ever
    made. For a specimen of 780 cubed voxels that saves close to two
    gigabytes of memory.
    """
    nz = volume.shape[0]
    idx = np.linspace(0, nz - 1, min(n_sample, nz)).astype(int)
    lo, hi = np.percentile(volume[idx].astype(np.float32), [P_LOW, P_HIGH])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def occupancy_downsample(mask: np.ndarray, f: int) -> np.ndarray:
    """Merge each block of f cubed voxels into one local metal fraction.

    mask : boolean array, True where the voxel was classified as metal.
    f    : block edge length in voxels, the decimation factor.

    Each output value is the fraction of the block occupied by metal, so it
    lies between zero and one. This is the merging step: a block spans f
    slices, so information from a slice is combined with the slices above
    and below it, and an isolated noisy voxel is outvoted by its
    neighbours. Averaging also acts as the anti aliasing filter that plain
    striding lacks, which is why a marginally thin strut no longer
    disappears simply because the slice through its narrowest point was the
    one thrown away.
    """
    if f <= 1:
        return mask.astype(np.float32)
    nz, ny, nx = ((s // f) * f for s in mask.shape)
    m = mask[:nz, :ny, :nx]
    occ = m.reshape(nz // f, f, ny // f, f, nx // f, f).sum(axis=(1, 3, 5),
                                                            dtype=np.float32)
    return occ / float(f ** 3)


def clean_field(occ: np.ndarray,
                close_radius: int = 1,
                min_object_vox: int = 64,
                fill_hole_vox: int = 64,
                keep_largest: bool = False,
                smooth_sigma: float = 1.0,
                occ_level: float = 0.5,
                binarise: bool = False):
    """Repair noise artefacts in three dimensions, then smooth.

    occ            : local metal fraction field from occupancy_downsample.
    occ_level      : fraction of a block that must be metal for that block to
                     count as solid. It is also the contour level handed to
                     marching cubes, so a value of 0.5 places the surface
                     where a block is half filled, which on an isotropic grid
                     is the unbiased estimate of the true interface position.
    binarise       : when False, the fractional field is contoured directly,
                     so a block that is thirty percent metal still pulls the
                     surface towards itself and a strut only a few blocks
                     across stays continuous. When True the field is first
                     rounded to zero or one, which permits the morphological
                     repair below but discards that sub block information and
                     will fragment thin struts into disconnected beads. Use
                     it only when noise rather than thin features is the
                     dominant problem.
    close_radius   : radius in decimated voxels of the spherical element used
                     for the morphological closing, a dilation followed by an
                     erosion. It bridges gaps narrower than about twice this
                     radius and leaves everything wider untouched. Binarised
                     mode only.
    min_object_vox : connected clusters smaller than this are deleted, judged
                     by three dimensional connectivity, so a cluster must be
                     isolated in every direction rather than merely within
                     its own slice. Binarised mode only.
    fill_hole_vox  : cavities smaller than this inside the metal are filled.
                     Binarised mode only.
    keep_largest   : retain only the largest connected component. Strips
                     mounting material, but also deletes a strut fragment
                     that has genuinely broken free, so it is off by default.
    smooth_sigma   : standard deviation, in decimated voxels, of the Gaussian
                     applied before extraction.

    Returns the field to contour and the level at which to contour it.
    """
    from scipy import ndimage as ndi

    partial = float(np.mean((occ > 0.0) & (occ < 1.0)))
    mode = "the binarised mask" if binarise else "the fraction directly"
    print(f"[clean] {100 * partial:.1f} % of blocks are partially filled; "
          f"contouring {mode}")

    if not binarise:
        field = occ.astype(np.float32, copy=False)
        if smooth_sigma > 0:
            field = ndi.gaussian_filter(field, smooth_sigma)
        return field, float(occ_level)

    mask = occ >= occ_level
    n_start = int(mask.sum())

    if close_radius > 0:
        try:
            from skimage.morphology import ball
            struct = ball(close_radius)
        except ImportError:
            struct = ndi.generate_binary_structure(3, 1)
        mask = ndi.binary_closing(mask, structure=struct)

    if fill_hole_vox > 0 or min_object_vox > 0:
        try:
            from skimage.morphology import remove_small_holes, remove_small_objects
            # scikit-image 0.26 renamed both size keywords to max_size and
            # deprecated the old ones. Trying the new name first keeps the
            # call correct on either version, whereas guessing wrong here
            # would silently invert the meaning of the size limit.
            def _call(fn, m, size):
                try:
                    return fn(m, max_size=size)
                except TypeError:
                    pass
                try:
                    return fn(m, area_threshold=size)
                except TypeError:
                    return fn(m, min_size=size)

            if fill_hole_vox > 0:
                mask = _call(remove_small_holes, mask, fill_hole_vox)
            if min_object_vox > 0:
                mask = _call(remove_small_objects, mask, min_object_vox)
        except ImportError:
            lab, n = ndi.label(mask)
            if n > 0 and min_object_vox > 0:
                sizes = np.bincount(lab.ravel())
                mask = np.isin(lab, np.flatnonzero(sizes >= min_object_vox)) & (lab > 0)

    if keep_largest:
        lab, n = ndi.label(mask)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            mask = lab == int(sizes.argmax())
            print(f"[clean] {n} components found, largest retained")

    n_end = int(mask.sum())
    print(f"[clean] metal voxels {n_start:,} -> {n_end:,} "
          f"({100.0 * (n_end - n_start) / max(n_start, 1):+.2f} %)")

    field = mask.astype(np.float32)
    if smooth_sigma > 0:
        field = ndi.gaussian_filter(field, smooth_sigma)
    return field, 0.5


# ----------------------------------------------------------------------
# 3. HTML shell
# ----------------------------------------------------------------------
_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --ink:#1c1d1f; --muted:#6b6f76; --line:#d9dbe0; --panel:#f7f8f9; --accent:#2b6d7d; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; color:var(--ink); background:#fff; }
  header { padding:16px 24px 10px; border-bottom:1px solid var(--line); }
  h1 { font-size:16px; font-weight:600; margin:0 0 4px; }
  header p { margin:0; font-size:13px; color:var(--muted); }
  #wrap { display:flex; flex-direction:column; height:calc(100vh - 70px); }
  #plot { flex:1 1 auto; min-height:360px; }
  #panel { flex:0 0 auto; border-top:1px solid var(--line); background:var(--panel);
           padding:14px 24px 18px; display:grid;
           grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px 28px; }
  .ctrl { display:grid; grid-template-columns:22px 1fr 128px; align-items:center; gap:10px; }
  .pair { display:grid; gap:2px; }
  .ctrl label { font-size:13px; font-weight:600; }
  .ctrl output { font-size:12px; color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  #foot { grid-column:1/-1; display:flex; align-items:center; justify-content:space-between;
          gap:16px; font-size:12px; color:var(--muted); border-top:1px dashed var(--line); padding-top:10px; }
  button { font:inherit; font-size:12px; padding:5px 12px; border:1px solid var(--line);
           border-radius:4px; background:#fff; color:var(--ink); cursor:pointer; }
  button:hover { border-color:var(--accent); color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>Each axis has a lower and an upper bound, so the view can be reduced to a thin slab.
     A slab is the only way to follow individual struts, since a thick block of lattice is
     opaque from outside. The cut faces are open, the extracted surface being the metal to
     air interface only.</p>
</header>
<div id="wrap">
  <div id="plot"></div>
  <div id="panel">
    <div class="ctrl"><label for="sx1">x</label>
      <div class="pair">
        <input type="range" id="sx0" min="0" max="__XMAX__" step="__XSTEP__" value="0">
        <input type="range" id="sx1" min="0" max="__XMAX__" step="__XSTEP__" value="__XMAX__">
      </div>
      <output id="ox"></output></div>
    <div class="ctrl"><label for="sy1">y</label>
      <div class="pair">
        <input type="range" id="sy0" min="0" max="__YMAX__" step="__YSTEP__" value="0">
        <input type="range" id="sy1" min="0" max="__YMAX__" step="__YSTEP__" value="__YMAX__">
      </div>
      <output id="oy"></output></div>
    <div class="ctrl"><label for="sz1">z</label>
      <div class="pair">
        <input type="range" id="sz0" min="0" max="__ZMAX__" step="__ZSTEP__" value="0">
        <input type="range" id="sz1" min="0" max="__ZMAX__" step="__ZSTEP__" value="__ZMAX__">
      </div>
      <output id="oz"></output></div>
    <div id="foot">
      <span id="count"></span>
      <span>
        <button id="slabx" type="button">x slab</button>
        <button id="slaby" type="button">y slab</button>
        <button id="slabz" type="button">z slab</button>
        <button id="reset" type="button">Whole volume</button>
      </span>
    </div>
  </div>
</div>
<script>
const LAYOUT = __LAYOUT__;
const LIM = __LIMITS__;
const ARR = __ARRAYS__;

// Base64 to typed array. The mesh is held as 32 bit floats and 32 bit
// integers from end to end. Decoding it into ordinary JavaScript arrays,
// which store eight bytes per entry and are boxed individually, is what
// exhausts the browser heap on a mesh of this size.
function decode(s, T) {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let n = 0; n < bin.length; n++) u8[n] = bin.charCodeAt(n);
  return new T(u8.buffer);
}

const X = decode(ARR.x, Float32Array);
const Y = decode(ARR.y, Float32Array);
const Z = decode(ARR.z, Float32Array);
const F = decode(ARR.f, Uint32Array);
const nF = F.length / 3;

// Per triangle bounding box, computed once, so a slider event costs six
// comparisons per triangle and no allocation at all.
const lx = new Float32Array(nF), hx = new Float32Array(nF);
const ly = new Float32Array(nF), hy = new Float32Array(nF);
const lz = new Float32Array(nF), hz = new Float32Array(nF);
for (let t = 0; t < nF; t++) {
  const a = F[3*t], b = F[3*t+1], c = F[3*t+2];
  lx[t] = Math.min(X[a], X[b], X[c]); hx[t] = Math.max(X[a], X[b], X[c]);
  ly[t] = Math.min(Y[a], Y[b], Y[c]); hy[t] = Math.max(Y[a], Y[b], Y[c]);
  lz[t] = Math.min(Z[a], Z[b], Z[c]); hz[t] = Math.max(Z[a], Z[b], Z[c]);
}

// Output buffers are allocated once at full size and refilled in place.
const I = new Uint32Array(nF), J = new Uint32Array(nF), K = new Uint32Array(nF);

function fill(x0, x1, y0, y1, z0, z1) {
  let n = 0;
  for (let t = 0; t < nF; t++) {
    if (lx[t] >= x0 && hx[t] <= x1 &&
        ly[t] >= y0 && hy[t] <= y1 &&
        lz[t] >= z0 && hz[t] <= z1) {
      I[n] = F[3*t]; J[n] = F[3*t+1]; K[n] = F[3*t+2]; n++;
    }
  }
  return n;
}

const n0 = fill(0, LIM.x, 0, LIM.y, 0, LIM.z);
const trace = {
  type: 'mesh3d', name: 'strut surface',
  x: X, y: Y, z: Z,
  i: I.subarray(0, n0), j: J.subarray(0, n0), k: K.subarray(0, n0),
  color: '#9aa3ad', opacity: 1.0, flatshading: true,
  lighting: {ambient: 0.45, diffuse: 0.85, specular: 0.25, roughness: 0.6, fresnel: 0.15},
  lightposition: {x: 100, y: 200, z: 300}
};

Plotly.newPlot('plot', [trace], LAYOUT, {responsive: true, displaylogo: false}).then(function () {
  const S = {
    x: [document.getElementById('sx0'), document.getElementById('sx1')],
    y: [document.getElementById('sy0'), document.getElementById('sy1')],
    z: [document.getElementById('sz0'), document.getElementById('sz1')]
  };
  const O = {x: document.getElementById('ox'), y: document.getElementById('oy'),
             z: document.getElementById('oz')};
  const cnt = document.getElementById('count');
  let pending = false;

  function apply() {
    pending = false;
    // The lower handle can never pass the upper one, which would otherwise
    // produce an empty range and a blank scene with no obvious cause.
    ['x', 'y', 'z'].forEach(function (ax) {
      if (+S[ax][0].value > +S[ax][1].value) S[ax][0].value = S[ax][1].value;
    });
    const x0 = +S.x[0].value, x1 = +S.x[1].value;
    const y0 = +S.y[0].value, y1 = +S.y[1].value;
    const z0 = +S.z[0].value, z1 = +S.z[1].value;
    const n = fill(x0, x1, y0, y1, z0, z1);
    Plotly.restyle('plot', {i: [I.subarray(0, n)], j: [J.subarray(0, n)],
                            k: [K.subarray(0, n)]}, [0]);
    cnt.textContent = n.toLocaleString() + ' of ' + nF.toLocaleString() + ' triangles shown';
    O.x.textContent = x0.toFixed(2) + ' to ' + x1.toFixed(2) + ' mm';
    O.y.textContent = y0.toFixed(2) + ' to ' + y1.toFixed(2) + ' mm';
    O.z.textContent = z0.toFixed(2) + ' to ' + z1.toFixed(2) + ' mm';
  }
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(apply); } }
  ['x', 'y', 'z'].forEach(function (ax) {
    S[ax][0].addEventListener('input', schedule);
    S[ax][1].addEventListener('input', schedule);
  });

  function whole() {
    ['x', 'y', 'z'].forEach(function (ax) {
      S[ax][0].value = 0; S[ax][1].value = LIM[ax];
    });
  }
  // A slab one tenth of the specimen deep, centred, which is usually thin
  // enough for individual struts to be followed by eye and thick enough to
  // show how they connect.
  function slab(ax) {
    whole();
    const c = LIM[ax] / 2, h = LIM[ax] / 20;
    S[ax][0].value = c - h; S[ax][1].value = c + h;
    apply();
  }
  document.getElementById('slabx').addEventListener('click', function () { slab('x'); });
  document.getElementById('slaby').addEventListener('click', function () { slab('y'); });
  document.getElementById('slabz').addEventListener('click', function () { slab('z'); });
  document.getElementById('reset').addEventListener('click', function () { whole(); apply(); });
  apply();
});
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------
# 4. Surface extraction and page writing
# ----------------------------------------------------------------------
def render_trimmable(volume, voxel_um, t=None, downsample=2, step_size=2,
                     out_html=DEFAULT_OUT, open_browser=True, n_steps=200,
                     clean=True, close_radius=1, min_object_vox=64,
                     fill_hole_vox=64, keep_largest=False, smooth_sigma=1.0,
                     occ_level=0.5, binarise=False, max_faces=6_000_000,
                     allow_coarsen=False, median_radius=0,
                     threshold_method="otsu", iso_fraction=0.5):
    """Extract the metal surface and write the interactive trimming page.

    voxel_um   : isotropic voxel edge length in micrometres; converts vertex
                 indices into millimetres.
    t          : iso value in normalised grey units. None means the global
                 Otsu threshold, so the surface follows the statistically
                 optimal metal to air boundary.
    downsample : decimation on all three axes before extraction. A factor of
                 two cuts the voxel count eightfold and the triangle count by
                 roughly four.
    step_size  : marching cubes stride; larger gives a coarser mesh.
    n_steps    : number of positions on each slider.
    """
    from skimage.measure import marching_cubes

    lo, hi = percentile_bounds(volume)
    raw_t = None
    if t is None:
        if isinstance(threshold_method, (int, float)):
            t = float(threshold_method)
            print(f"[threshold] fixed level {t:.3f} from the settings block")
        elif str(threshold_method).lower() in ("iso50", "iso"):
            raw_t = iso_threshold_raw(volume, fraction=iso_fraction)
            t = (raw_t - lo) / (hi - lo)
        else:
            t = global_threshold(volume)
    if raw_t is None:
        raw_t = lo + t * (hi - lo)

    if median_radius > 0:
        from scipy import ndimage as ndi
        k = 2 * median_radius + 1
        print(f"[3d] 3D median filter, {k} cubed kernel, on "
              f"{volume.size / 1e6:.0f} M voxels ...")
        volume = ndi.median_filter(volume, size=k)

    sens = None
    if clean:
        # The normalised threshold is mapped back into raw grey units so
        # that the comparison is made on the original array, without ever
        # materialising a float copy of the full volume.
        mask = volume > raw_t
        n_at = int(mask.sum())
        print(f"[3d] threshold {t:.3f} normalised = {raw_t:.1f} raw, "
              f"{n_at:,} metal voxels at full resolution")

        # Sensitivity of the segmented volume to the threshold. On a well
        # separated histogram the two populations are far apart, so shifting
        # the level by a tenth of the contrast range changes the metal volume
        # only slightly. A large swing means the level sits on a slope rather
        # than in the valley, and every thickness derived from it inherits
        # that uncertainty.
        span = 0.1 * (hi - lo)
        n_lo = int((volume > raw_t - span).sum())
        n_hi = int((volume > raw_t + span).sum())
        sens = (n_at, n_lo, n_hi, span)

        # Meeting the triangle budget by coarsening the block average, never
        # by raising the marching cubes stride. Block averaging is a low pass
        # filter followed by resampling, so material inside a block still
        # influences the result. The stride simply skips voxels with no
        # filtering, so a strut sampled only three times across its width
        # breaks into disconnected beads while a thicker node survives.
        while True:
            occ = occupancy_downsample(mask, downsample)
            field, level = clean_field(occ, close_radius=close_radius,
                                       min_object_vox=min_object_vox,
                                       fill_hole_vox=fill_hole_vox,
                                       keep_largest=keep_largest,
                                       smooth_sigma=smooth_sigma,
                                       occ_level=occ_level,
                                       binarise=binarise)
            del occ
            print(f"[3d] field {field.shape}, contour level = {level:.3f}, "
                  f"extracting surface ...")
            verts, faces, _, _ = marching_cubes(field, level=level,
                                                step_size=step_size)
            if len(faces) <= max_faces or downsample >= 16:
                break
            if not allow_coarsen and len(faces) < HARD_FACE_LIMIT:
                print(f"[3d] WARNING {len(faces):,} triangles exceeds the "
                      f"budget of {max_faces:,}. Proceeding anyway, because "
                      f"averaging into larger blocks would erase features "
                      f"only a few voxels across. Reduce the region of "
                      f"interest instead if the browser struggles.")
                break
            if not allow_coarsen:
                print(f"[3d] {len(faces):,} triangles is past the hard ceiling "
                      f"of {HARD_FACE_LIMIT:,}, which no browser will open. "
                      f"Coarsening despite the setting. For a faithful "
                      f"surface use a smaller region of interest instead.")
            downsample += 1
            print(f"[3d] {len(faces):,} triangles exceeds the budget of "
                  f"{max_faces:,}; re-averaging at decimation {downsample} "
                  f"(block {voxel_um * downsample:.0f} um)")
            del verts, faces, field
        del mask
    else:
        # Contour the grey data itself. A voxel straddling the boundary of a
        # strut records an intermediate attenuation, so the interpolation
        # places the surface where the attenuation truly crosses the
        # threshold, with sub voxel precision. Rounding to a binary mask
        # first discards exactly that information, which matters when a
        # feature is only a handful of voxels across.
        field = normalise(volume)
        if downsample > 1:
            field = field[::downsample, ::downsample, ::downsample]
        if smooth_sigma > 0:
            from scipy import ndimage as ndi
            field = ndi.gaussian_filter(field, smooth_sigma)
        level = t
        print(f"[3d] field {field.shape}, contour level = {level:.3f}, "
              f"extracting surface ...")
        verts, faces, _, _ = marching_cubes(field, level=level,
                                            step_size=step_size)

    s_mm = voxel_um * downsample / 1000.0        # mm per decimated voxel
    z, y, x = (verts[:, 0] * s_mm).astype(np.float32), \
              (verts[:, 1] * s_mm).astype(np.float32), \
              (verts[:, 2] * s_mm).astype(np.float32)
    tris = np.ascontiguousarray(faces, dtype=np.uint32).ravel()
    print(f"[3d] {len(verts):,} vertices, {len(faces):,} triangles")

    # Mean feature diameter from the surface to volume ratio. For a body of
    # roughly cylindrical cross section V / S = r / 2, so d = 4 V / S. It is
    # an average over everything present, so a specimen mixing fine struts
    # with bulk material returns something between the two, but it is a fast
    # and assumption free check on whether the decimation is admissible.
    _u = voxel_um * downsample
    _tri = verts[faces]
    _area = 0.5 * np.linalg.norm(np.cross(_tri[:, 1] - _tri[:, 0],
                                          _tri[:, 2] - _tri[:, 0]), axis=1).sum()
    _solid = float((field >= level).sum())
    if _area > 0:
        _d = 4.0 * _solid * _u / _area
        print(f"[3d] mean feature diameter from surface to volume: "
              f"{_d:.0f} um = {_d / voxel_um:.1f} voxels. Block averaging is "
              f"safe only while a block stays well below this.")

        # A volume change is not the right measure of threshold quality,
        # because for a thin feature the volume necessarily moves fast: for a
        # cylinder dV / V = 2 dr / r, so a strut three voxels in radius gains
        # a third of its volume from a boundary shift of half a voxel. The
        # meaningful quantity is that shift, obtained as the volume change
        # divided by the surface area, and it should be small compared with
        # one voxel.
        if sens is not None and _area > 0:
            n_at, n_lo, n_hi, span = sens
            area_um2 = _area * _u * _u
            for tag, n in (("down", n_lo), ("up", n_hi)):
                dv = (n - n_at) * (voxel_um ** 3)
                print(f"[3d] shifting the level {tag} by a tenth of the "
                      f"contrast range moves the surface by "
                      f"{abs(dv / area_um2):.1f} um "
                      f"({abs(dv / area_um2) / voxel_um:.2f} voxels)")

    xmax, ymax, zmax = float(x.max()), float(y.max()), float(z.max())

    # The mesh travels as base64 encoded binary rather than as JSON numbers.
    # Decimal text costs roughly five bytes per coordinate and forces the
    # browser to build boxed arrays, which is what exhausted its heap.
    def _b64(arr):
        return base64.b64encode(np.ascontiguousarray(arr)).decode("ascii")

    arrays = {"x": _b64(x), "y": _b64(y), "z": _b64(z), "f": _b64(tris)}
    payload_mb = sum(len(v) for v in arrays.values()) / 1e6
    print(f"[3d] embedded mesh payload {payload_mb:.1f} MB")
    if payload_mb > 200:
        print(f"[3d] WARNING a payload of {payload_mb:.0f} MB is likely to "
              f"exhaust the browser. Reduce the region of interest, or use "
              f"mode 'overview' for the whole specimen.")

    # Ranges and aspect ratio are pinned to the full extent. Left automatic,
    # Plotly would rescale as triangles are removed and the object would
    # appear to grow while being trimmed.
    layout = {
        "scene": {
            "xaxis": {"title": {"text": "x (mm)"}, "range": [0, xmax]},
            "yaxis": {"title": {"text": "y (mm)"}, "range": [0, ymax]},
            "zaxis": {"title": {"text": "z (mm)"}, "range": [0, zmax]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0,
                            "y": ymax / xmax if xmax else 1.0,
                            "z": zmax / xmax if xmax else 1.0},
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.1}},
        },
        "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
        "template": {"layout": {"paper_bgcolor": "#ffffff",
                                "plot_bgcolor": "#ffffff"}},
    }

    html = (_PAGE
            .replace("__TITLE__", f"Strut network surface, iso value {t:.3f}")
            .replace("__LAYOUT__", json.dumps(layout))
            .replace("__ARRAYS__", json.dumps(arrays))
            .replace("__LIMITS__", json.dumps({"x": xmax, "y": ymax, "z": zmax}))
            .replace("__XMAX__", f"{xmax:.6f}").replace("__XSTEP__", f"{xmax/n_steps:.6f}")
            .replace("__YMAX__", f"{ymax:.6f}").replace("__YSTEP__", f"{ymax/n_steps:.6f}")
            .replace("__ZMAX__", f"{zmax:.6f}").replace("__ZSTEP__", f"{zmax/n_steps:.6f}"))

    out_html = os.path.abspath(out_html)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[save] {out_html}")
    if open_browser:
        try:
            webbrowser.open(Path(out_html).as_uri())
        except Exception:
            pass
    return out_html


# ----------------------------------------------------------------------
# 5. Synthetic phantom, so the script still shows something without data
# ----------------------------------------------------------------------
def synthetic_strut_phantom(n=160, radius_px=9.0, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2.0
    p, q, r = x - c, y - c, z - c
    solid = np.zeros((n, n, n), dtype=bool)
    for a, b in ((p, q), (p, r), (q, r)):
        solid |= (a ** 2 + b ** 2) < radius_px ** 2
    for sgn in (1, -1):
        d = (p + sgn * q) / np.sqrt(2)
        solid |= (p ** 2 + q ** 2 + r ** 2 - d ** 2) < (0.8 * radius_px) ** 2
    vol = np.where(solid, 0.80, 0.10).astype(np.float32)
    vol += rng.normal(0.0, noise, vol.shape).astype(np.float32)
    return np.clip(vol * 65535, 0, 65535).astype(np.uint16)


# ----------------------------------------------------------------------
# 6. Entry point
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Trimmable 3D view of a CT strut lattice.")
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_UM)
    ap.add_argument("--which", type=int, default=DEFAULT_WHICH)
    ap.add_argument("--step", type=int, default=1, help="load every Nth slice")
    ap.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    ap.add_argument("--stepsize", type=int, default=DEFAULT_STEPSIZE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--demo", action="store_true", help="synthetic phantom")
    ap.add_argument("--raw", action="store_true",
                    help="skip the 3D merging and cleanup stage")
    ap.add_argument("--close", type=int, default=1,
                    help="closing radius in decimated voxels (0 disables)")
    ap.add_argument("--minobj", type=int, default=64,
                    help="smallest surviving cluster, in decimated voxels")
    ap.add_argument("--fillholes", type=int, default=64,
                    help="largest cavity that is filled, in decimated voxels")
    ap.add_argument("--largest", action="store_true",
                    help="keep only the largest connected component")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                    help="Gaussian smoothing before extraction (0 disables)")
    ap.add_argument("--occ", type=float, default=DEFAULT_OCC,
                    help="block metal fraction taken as the interface (0 to 1)")
    ap.add_argument("--roi", type=int, nargs=6, default=None,
                    metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
                    help="crop a subvolume before rendering, in voxels")
    ap.add_argument("--roicube", type=int, default=None,
                    help="crop a centred cube of this many voxels per side")
    ap.add_argument("--threshold", default=DEFAULT_THRESHOLD,
                    help='"otsu", "iso50", or a number in [0, 1]')
    ap.add_argument("--isofrac", type=float, default=DEFAULT_ISO_FRACTION,
                    help="position between the air and metal peaks, 0 to 1")
    ap.add_argument("--median", type=int, default=DEFAULT_MEDIAN,
                    help="3D median filter radius in voxels (0 disables)")
    ap.add_argument("--coarsen", action="store_true",
                    help="allow block averaging to meet the triangle budget")
    ap.add_argument("--maxfaces", type=int, default=DEFAULT_MAXFACES,
                    help="triangle budget; the stride is raised to meet it")
    ap.add_argument("--binarise", action="store_true",
                    help="round the fraction field before extraction, which "
                         "enables the morphological repair but fragments thin "
                         "struts; leave off unless noise dominates")
    args = ap.parse_args(argv)

    # Pressing Run in an editor supplies no arguments at all, so in that case
    # every option is taken from the USER SETTINGS block at the top of this
    # file. Edit the values there; the command line flags exist only for
    # people launching the script from a terminal.
    print(f"[version] render_3d_trim {SCRIPT_VERSION}")
    if argv is None and len(sys.argv) == 1:
        preset = PRESETS.get(str(DEFAULT_MODE).lower())
        if preset is None and str(DEFAULT_MODE).lower() != "custom":
            print(f"[settings] WARNING DEFAULT_MODE = '{DEFAULT_MODE}' is not "
                  f"one of {list(PRESETS) + ['custom']}; falling back to the "
                  f"individual settings")
        if preset:
            print(f"[settings] mode '{DEFAULT_MODE}': " +
                  ", ".join(f"{k}={v}" for k, v in preset.items()))
        else:
            preset = {}
            print("[settings] mode 'custom': using the individual settings")
        args.roicube = preset.get("roicube", DEFAULT_ROICUBE)
        args.downsample = preset.get("downsample", DEFAULT_DOWNSAMPLE)
        args.maxfaces = preset.get("maxfaces", DEFAULT_MAXFACES)
        args.stepsize = DEFAULT_STEPSIZE
        args.raw = not DEFAULT_CLEAN
        args.occ = preset.get("occ", DEFAULT_OCC)
        args.binarise = DEFAULT_BINARISE
        args.sigma = preset.get("sigma", DEFAULT_SIGMA)
        args.close = DEFAULT_CLOSE
        args.minobj = DEFAULT_MINOBJ
        args.fillholes = DEFAULT_FILLHOLES
        args.largest = DEFAULT_LARGEST
        args.median = preset.get("median", DEFAULT_MEDIAN)
        args.coarsen = preset.get("coarsen", DEFAULT_ALLOW_COARSEN)
        args.threshold = DEFAULT_THRESHOLD
        args.isofrac = DEFAULT_ISO_FRACTION
        args.which = DEFAULT_WHICH
        print("[settings] no arguments given, using the USER SETTINGS block")
        print(f"[settings] roicube={args.roicube}, downsample={args.downsample}, "
              f"maxfaces={args.maxfaces:,}, occ={args.occ}, "
              f"binarise={args.binarise}, sigma={args.sigma}")

    if args.path is None and not args.demo:
        if DEFAULT_PATH and Path(DEFAULT_PATH).exists():
            args.path = DEFAULT_PATH
            print("[settings] using DEFAULT_PATH from the top of this file")
        else:
            print("[settings] DEFAULT_PATH is unset or does not exist, so the "
                  "synthetic phantom is used instead.")
            args.demo = True

    if args.demo:
        vol = synthetic_strut_phantom()
    else:
        vol, _ = load_volume(args.path, which=args.which, step=args.step)

    # A cropped region rendered without decimation is the cleanest test of
    # whether a feature is genuinely absent or merely undersampled, because
    # it removes every reduction step from the pipeline at a cost of a few
    # seconds and a small file.
    if args.roicube and all(args.roicube >= sz for sz in vol.shape):
        print(f"[roi] the requested cube of {args.roicube} voxels is larger "
              f"than the specimen {vol.shape}, so no crop is applied. To view "
              f"the whole specimen properly set DEFAULT_MODE = \"overview\", "
              f"which also adjusts the smoothing and the contour level for "
              f"the coarser sampling it requires.")
    elif args.roicube:
        n = args.roicube
        c = [s // 2 for s in vol.shape]
        sl = tuple(slice(max(0, ci - n // 2), min(si, ci + n // 2))
                   for ci, si in zip(c, vol.shape))
        vol = vol[sl]
        print(f"[roi] centred cube {vol.shape}")
    elif args.roi:
        z0, z1, y0, y1, x0, x1 = args.roi
        vol = vol[z0:z1, y0:y1, x0:x1]
        print(f"[roi] crop {vol.shape}")

    # The page is written next to this script, not to the shell working
    # directory, which is elsewhere when the script is launched from VS Code.
    out = args.out
    if not os.path.isabs(out):
        out = str(Path(__file__).resolve().parent / out)

    render_trimmable(vol, args.voxel, downsample=args.downsample,
                     step_size=args.stepsize, out_html=out,
                     clean=not args.raw, close_radius=args.close,
                     min_object_vox=args.minobj, fill_hole_vox=args.fillholes,
                     keep_largest=args.largest, smooth_sigma=args.sigma,
                     occ_level=args.occ, binarise=args.binarise,
                     max_faces=args.maxfaces, allow_coarsen=args.coarsen,
                     median_radius=args.median,
                     threshold_method=args.threshold,
                     iso_fraction=args.isofrac)
    print("[done]")


if __name__ == "__main__":
    main()
