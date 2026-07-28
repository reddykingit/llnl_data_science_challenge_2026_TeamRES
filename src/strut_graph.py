"""
strut_graph.py
==============

Extracts the node and member graph of a strut lattice from an XCT TIFF
volume and writes two files:

    strut_graph.json   nodes, members and their measured properties
    strut_graph.html   an interactive viewer that opens in Chrome

Method
------
The segmented metal is reduced to a one voxel wide skeleton by medial axis
thinning. Every skeleton voxel is then classified by its number of
neighbours in the 26 voxel neighbourhood: exactly two means the voxel lies
along a path, three or more means a junction, one means a free end. Voxels
that are not path voxels are clustered into nodes; the paths between them
are the members. Each member carries its length, its straight line chord,
its tortuosity and a diameter profile measured along the medial centreline.
The profile excludes junction-influenced ends and reports robust roughness,
local thinning, and local thickening statistics.

A node whose degree falls below the modal degree of the lattice is flagged
as suspect, since that is the signature of a strut that is absent.

Run it directly. Requirements:
    pip install tifffile scikit-image scipy numpy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import webbrowser
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "2026-07-27-diameter-profile"

# ======================================================================
# USER SETTINGS  -- used when the script is started with no arguments.
# ======================================================================
DEFAULT_PATH = r"C:\Users\Owner\Downloads\DSC\llnl_data_science_challenge_2026_TeamRES\data\missing_struts\tif_stacks"
DEFAULT_VOXEL_UM = 10.0     # isotropic voxel edge length, micrometres
DEFAULT_WHICH = 0           # specimen index if the folder holds several

DEFAULT_ROICUBE = None      # centred cube of this many voxels, None for all
DEFAULT_DOWNSAMPLE = 2      # block averaging before skeletonising. Thinning
                            # is the expensive step, and a strut 7 voxels
                            # across still spans 3.5 blocks at a factor of 2,
                            # which is enough to carry a one voxel skeleton.
                            # Use 1 for the most faithful graph on a crop.

DEFAULT_THRESHOLD = "iso50"     # "otsu", "iso50", or a number in [0, 1]
DEFAULT_ISO_FRACTION = 0.45     # position between the air and metal peaks
DEFAULT_MEDIAN = 1              # 3D median radius before thresholding

DEFAULT_MIN_MEMBER_UM = 100.0   # members shorter than this are treated as
                                # skeleton artefacts at junctions rather than
                                # as real struts, and are pruned
DEFAULT_DIAMETER_END_TRIM = 0.10 # fallback for fixed-fraction trim mode
DEFAULT_DIAMETER_TRIM_MODE = "adaptive"
DEFAULT_MAX_ADAPTIVE_TRIM = 0.20 # never hide more than this at either end
DEFAULT_JUNCTION_EXCESS = 0.20   # endpoint must exceed shaft by this much
DEFAULT_TRIM_STABILITY = 0.08    # stable-window relative range tolerance
DEFAULT_THIN_RATIO = 0.70        # local diameter / member median
DEFAULT_THICK_RATIO = 1.30
DEFAULT_PROFILE_POINTS = 64      # compact samples retained in output JSON
DEFAULT_OUT_JSON = "strut_graph.json"
DEFAULT_OUT_HTML = "strut_graph.html"
# ======================================================================

P_LOW, P_HIGH = 0.5, 99.5
TIF_EXT = (".tif", ".tiff")


def _key(t):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(t))]


# ----------------------------------------------------------------------
# 1. Reading and segmentation
# ----------------------------------------------------------------------
def load_volume(path, which=0):
    import tifffile
    p = Path(path)
    if p.is_file():
        vol = tifffile.imread(str(p))
        label = p.name
    elif p.is_dir():
        files = sorted([f for f in p.iterdir()
                        if f.is_file() and f.suffix.lower() in TIF_EXT], key=lambda f: _key(f.name))
        subdirs = sorted([d for d in p.iterdir() if d.is_dir()], key=lambda d: _key(d.name))
        if not files and subdirs:
            which = min(max(which, 0), len(subdirs) - 1)
            return load_volume(str(subdirs[which]), 0)
        if not files:
            sys.exit(f"no TIFF files in {p}")
        with tifffile.TiffFile(files[0]) as tf:
            multi = len(tf.pages) > 1
        if multi:
            which = min(max(which, 0), len(files) - 1)
            vol, label = tifffile.imread(str(files[which])), files[which].name
        else:
            first = tifffile.imread(str(files[0]))
            vol = np.empty((len(files), *first.shape), dtype=first.dtype)
            vol[0] = first
            for k, f in enumerate(files[1:], start=1):
                vol[k] = tifffile.imread(str(f))
            label = p.name
    else:
        sys.exit(f"path does not exist: {p}")

    vol = np.asarray(vol)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    elif vol.ndim == 4:
        vol = vol[..., 0]
    print(f"[load] {label}, shape {vol.shape}, dtype {vol.dtype}")
    return vol, label


def _otsu_from_counts(counts, centres):
    p = counts.astype(np.float64) / max(counts.sum(), 1)
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centres)
    ok = (w0 > 0) & (w1 > 0)
    sig = np.zeros_like(w0)
    sig[ok] = ((mu[-1] * w0[ok] - mu[ok]) ** 2) / (w0[ok] * w1[ok])
    return float(centres[int(np.argmax(sig))])


def raw_threshold(volume, method="iso50", fraction=0.5, n_sample=32, nbins=512):
    """Segmentation level in raw grey units.

    method   : "iso50" places the level a given fraction of the way between
               the air peak and the metal peak, both located in the raw
               histogram where no percentile clipping has occurred. "otsu"
               maximises the variance between the two populations, which is
               biased when one of them is much larger than the other.
    fraction : the position between the peaks, 0.5 being the half maximum.
    """
    nz = volume.shape[0]
    idx = np.linspace(0, nz - 1, min(n_sample, nz)).astype(int)
    sub = volume[idx]
    v0, v1 = float(sub.min()), float(sub.max())
    counts, edges = np.histogram(sub, bins=nbins, range=(v0, v1))
    centres = 0.5 * (edges[:-1] + edges[1:])
    split = _otsu_from_counts(counts, centres)
    if str(method).lower() == "otsu":
        print(f"[threshold] Otsu level {split:.0f} raw")
        return split
    low, high = centres < split, centres >= split
    if not low.any() or not high.any():
        return split
    air = float(centres[low][np.argmax(counts[low])])
    metal = float(centres[high][np.argmax(counts[high])])
    t = air + fraction * (metal - air)
    print(f"[threshold] air peak {air:.0f}, metal peak {metal:.0f}, "
          f"level at {fraction:.2f} = {t:.0f} raw")
    return t


def occupancy_downsample(mask, f):
    """Merge each block of f cubed voxels into its local metal fraction."""
    if f <= 1:
        return mask
    nz, ny, nx = ((s // f) * f for s in mask.shape)
    m = mask[:nz, :ny, :nx]
    occ = m.reshape(nz // f, f, ny // f, f, nx // f, f).sum(axis=(1, 3, 5),
                                                           dtype=np.float32)
    return (occ / float(f ** 3)) >= 0.4


# ----------------------------------------------------------------------
# 2. Skeleton graph
# ----------------------------------------------------------------------
def _ordered_path_coordinates(sub):
    """Return coordinates of a one-voxel-wide path in geodesic order."""
    points = [tuple(int(v) for v in p) for p in np.argwhere(sub)]
    if len(points) < 2:
        return np.asarray(points, dtype=int)
    point_set = set(points)
    neighbours = {}
    for p in points:
        neighbours[p] = [
            (p[0] + dz, p[1] + dy, p[2] + dx)
            for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)
            and (p[0] + dz, p[1] + dy, p[2] + dx) in point_set
        ]
    endpoints = [p for p in points if len(neighbours[p]) <= 1]
    current = endpoints[0] if endpoints else points[0]
    ordered, visited, previous = [], set(), None
    while current is not None:
        ordered.append(current)
        visited.add(current)
        candidates = [q for q in neighbours[current]
                      if q != previous and q not in visited]
        previous, current = current, (candidates[0] if candidates else None)
    ordered.extend(p for p in points if p not in visited)
    return np.asarray(ordered, dtype=int)


def _adaptive_diameter_keep(diameter, max_trim=0.20,
                            junction_excess=0.20, stability=0.08,
                            window=5):
    """Conservatively exclude only demonstrably enlarged junction ends."""
    from scipy.ndimage import median_filter

    n = len(diameter)
    keep = np.ones(n, dtype=bool)
    if n < 2 * window + 3:
        return keep, 0, 0
    smooth = median_filter(np.asarray(diameter, float), size=window, mode="nearest")
    middle = smooth[n // 3:max(n // 3 + 1, 2 * n // 3)]
    shaft = float(np.median(middle))
    if shaft <= 0:
        return keep, 0, 0
    limit = min(int(np.floor(max_trim * n)), n // 2 - 1)

    def trim_count(values):
        # Do not trim a genuine thick region unless the actual endpoint is
        # clearly junction-sized relative to the central shaft estimate.
        if limit < 1 or np.median(values[:window]) <= shaft * (1.0 + junction_excess):
            return 0
        for i in range(1, limit + 1):
            local = values[i:min(n, i + window)]
            if len(local) < 3:
                break
            relative_range = (float(local.max() - local.min())
                              / max(float(np.median(local)), 1e-12))
            no_longer_enlarged = float(np.median(local)) <= shaft * (1.0 + stability)
            no_longer_decreasing = values[min(i + window - 1, n - 1)] >= values[i] * (1.0 - stability)
            if no_longer_enlarged and no_longer_decreasing and relative_range <= 2.0 * stability:
                return i
        # If no stable shaft is found, preserve the profile rather than
        # automatically deleting the maximum allowed region.
        return 0

    left = trim_count(smooth)
    right = trim_count(smooth[::-1])
    if left:
        keep[:left] = False
    if right:
        keep[n - right:] = False
    return keep, left, right


def _diameter_analysis(sub, dist_sub, voxel_um, end_trim=0.10,
                       thin_ratio=0.70, thick_ratio=1.30,
                       profile_points=64, trim_mode="adaptive",
                       max_adaptive_trim=0.20, junction_excess=0.20,
                       trim_stability=0.08):
    """Measure local inscribed diameter along a member centreline."""
    coords = _ordered_path_coordinates(sub)
    if coords.size == 0:
        return {}
    diameter = 2.0 * dist_sub[tuple(coords.T)] * voxel_um
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1) * voxel_um
    station = np.r_[0.0, np.cumsum(steps)]
    total = float(station[-1])
    if trim_mode == "adaptive":
        keep, left_trim, right_trim = _adaptive_diameter_keep(
            diameter, max_adaptive_trim, junction_excess, trim_stability)
    else:
        keep = np.ones(len(diameter), dtype=bool)
        left_trim = right_trim = 0
    if trim_mode == "fraction" and total > 0 and 0.0 <= end_trim < 0.5:
        keep = ((station >= end_trim * total)
                & (station <= (1.0 - end_trim) * total))
        left_trim = int(np.count_nonzero(station < end_trim * total))
        right_trim = int(np.count_nonzero(station > (1.0 - end_trim) * total))
    if keep.sum() < min(5, len(diameter)):
        keep[:] = True
    core_d, core_s = diameter[keep], station[keep]
    median, mean, std = (float(np.median(core_d)), float(np.mean(core_d)),
                         float(np.std(core_d)))
    q05, q25, q75, q95 = np.percentile(core_d, [5, 25, 75, 95])
    thin = core_d < thin_ratio * median if median > 0 else np.zeros_like(core_d, bool)
    thick = core_d > thick_ratio * median if median > 0 else np.zeros_like(core_d, bool)

    def longest_run_fraction(flag):
        longest = current = 0
        for value in flag:
            current = current + 1 if value else 0
            longest = max(longest, current)
        return float(longest / max(len(flag), 1))

    take = np.unique(np.linspace(0, len(diameter) - 1,
                                 min(profile_points, len(diameter))).astype(int))
    profile = [{"s_fraction": round(float(station[i] / total), 4) if total > 0 else 0.0,
                "diameter_um": round(float(diameter[i]), 1),
                "included_in_shaft_stats": bool(keep[i])} for i in take]
    end_window = min(5, len(diameter))
    left_end_ratio = float(np.median(diameter[:end_window]) / median) if median > 0 else None
    right_end_ratio = float(np.median(diameter[-end_window:]) / median) if median > 0 else None
    return {
        "diameter_um": round(median, 1),
        "diameter_mean_um": round(mean, 1),
        "diameter_min_um": round(float(core_d.min()), 1),
        "diameter_p05_um": round(float(q05), 1),
        "diameter_p25_um": round(float(q25), 1),
        "diameter_median_um": round(median, 1),
        "diameter_p75_um": round(float(q75), 1),
        "diameter_p95_um": round(float(q95), 1),
        "diameter_max_um": round(float(core_d.max()), 1),
        "diameter_std_um": round(std, 1),
        "diameter_cv": round(std / mean, 4) if mean > 0 else None,
        "diameter_robust_roughness": round(float((q75 - q25) / median), 4)
                                     if median > 0 else None,
        "thin_fraction": round(float(thin.mean()), 4),
        "longest_thin_run_fraction": round(longest_run_fraction(thin), 4),
        "thick_fraction": round(float(thick.mean()), 4),
        "thin_ratio_threshold": thin_ratio,
        "thick_ratio_threshold": thick_ratio,
        "diameter_end_trim_fraction": end_trim,
        "diameter_trim_mode": trim_mode,
        "adaptive_left_trim_fraction": round(left_trim / len(diameter), 4),
        "adaptive_right_trim_fraction": round(right_trim / len(diameter), 4),
        "left_end_to_shaft_diameter": round(left_end_ratio, 4) if left_end_ratio else None,
        "right_end_to_shaft_diameter": round(right_end_ratio, 4) if right_end_ratio else None,
        "diameter_profile": profile,
    }


def build_graph(mask, voxel_um, min_member_um=0.0,
                diameter_end_trim=DEFAULT_DIAMETER_END_TRIM,
                thin_ratio=DEFAULT_THIN_RATIO,
                thick_ratio=DEFAULT_THICK_RATIO,
                profile_points=DEFAULT_PROFILE_POINTS,
                diameter_trim_mode=DEFAULT_DIAMETER_TRIM_MODE,
                max_adaptive_trim=DEFAULT_MAX_ADAPTIVE_TRIM,
                junction_excess=DEFAULT_JUNCTION_EXCESS,
                trim_stability=DEFAULT_TRIM_STABILITY):
    """Reduce a binary lattice to nodes and members.

    mask          : boolean volume, True for metal.
    voxel_um      : edge length of one voxel of this mask, in micrometres,
                    already including any decimation applied beforehand.
    min_member_um : members shorter than this are pruned. Thinning produces
                    short spurs where several struts meet, and they would
                    otherwise be counted as real members and inflate the
                    degree of every junction.

    Returns a dictionary with node and member lists.

    The classification rests on the neighbour count of each skeleton voxel
    within its 26 voxel neighbourhood. Exactly two neighbours means the
    voxel lies along a path; three or more means several paths meet there;
    one means a free end. Removing the path voxels leaves the junction and
    end clusters, which become the nodes, and the paths between them become
    the members.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    print("[graph] thinning to the medial axis ...")
    skel = skeletonize(mask)
    n_skel = int(skel.sum())
    print(f"[graph] skeleton holds {n_skel:,} voxels")
    if n_skel == 0:
        return {"nodes": [], "members": []}

    struct = np.ones((3, 3, 3), dtype=np.uint8)
    kernel = struct.copy()
    kernel[1, 1, 1] = 0
    deg = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant") * skel

    node_seed = skel & (deg != 2)          # junctions, free ends, isolates
    path = skel & (deg == 2)

    node_lab, n_nodes = ndi.label(node_seed, structure=struct)
    seg_lab, n_seg = ndi.label(path, structure=struct)
    print(f"[graph] {n_nodes:,} node clusters, {n_seg:,} path segments")

    # Distance transform of the solid gives the local radius, so twice the
    # value along the skeleton is the local diameter of that strut.
    dist = ndi.distance_transform_edt(mask)

    # Node centroids, in voxel coordinates.
    centres = np.array(ndi.center_of_mass(node_seed, node_lab,
                                          np.arange(1, n_nodes + 1)))
    if centres.ndim == 1:
        centres = centres.reshape(-1, 3)

    # Which nodes does each segment touch? Walk the node voxels and look at
    # their neighbours, which is far cheaper than dilating every segment.
    offsets = [(dz, dy, dx)
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) != (0, 0, 0)]
    shape = mask.shape
    touch = {}
    nz_idx = np.argwhere(node_seed)
    for z, y, x in nz_idx:
        nid = node_lab[z, y, x]
        for dz, dy, dx in offsets:
            zz, yy, xx = z + dz, y + dy, x + dx
            if 0 <= zz < shape[0] and 0 <= yy < shape[1] and 0 <= xx < shape[2]:
                sid = seg_lab[zz, yy, xx]
                if sid:
                    touch.setdefault(sid, set()).add(int(nid))

    # Path length along a segment. Counting voxels is wrong, because a step
    # to a face neighbour spans one voxel, to an edge neighbour root two and
    # to a corner neighbour root three; a diagonal strut counted by voxels is
    # underestimated by up to seventy percent and returns a tortuosity below
    # one, which is geometrically impossible. Instead every adjacent pair
    # within the segment is counted once, by testing only the thirteen
    # offsets of one half of the neighbourhood, and weighted by its true
    # step length. For a one voxel wide chain the pairs are exactly the
    # links of the chain, so the sum is the arc length.
    half_off = [(0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
                (1, -1, -1), (1, -1, 0), (1, -1, 1),
                (1, 0, -1), (1, 0, 0), (1, 0, 1),
                (1, 1, -1), (1, 1, 0), (1, 1, 1)]
    half_len = [float(np.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)) for o in half_off]

    def _arc_length(sub):
        total = 0.0
        for off, step in zip(half_off, half_len):
            s1 = tuple(slice(max(d, 0), sz + min(d, 0))
                       for d, sz in zip(off, sub.shape))
            s2 = tuple(slice(max(-d, 0), sz + min(-d, 0))
                       for d, sz in zip(off, sub.shape))
            if any(sl.stop <= sl.start for sl in s1):
                continue
            total += step * float(np.logical_and(sub[s1], sub[s2]).sum())
        return total

    # The chord is the straight line between the two node centroids.
    seg_slices = ndi.find_objects(seg_lab)
    members = []
    for sid in range(1, n_seg + 1):
        ends = sorted(touch.get(sid, ()))
        if len(ends) != 2 or ends[0] == ends[1]:
            continue                       # dangling, isolated or a self loop
        sl = seg_slices[sid - 1]
        if sl is None:
            continue
        sub = seg_lab[sl] == sid
        n_vox = int(sub.sum())
        diameter_stats = _diameter_analysis(
            sub, dist[sl], voxel_um, diameter_end_trim,
            thin_ratio, thick_ratio, profile_points, diameter_trim_mode,
            max_adaptive_trim, junction_excess, trim_stability)
        a, b = ends[0] - 1, ends[1] - 1
        chord = float(np.linalg.norm(centres[a] - centres[b]) * voxel_um)
        # One step is added at each end, since the segment stops one voxel
        # short of each node cluster it connects to.
        length = float((_arc_length(sub) + 2.0) * voxel_um)
        if length < min_member_um:
            continue
        members.append({"a": a, "b": b, "voxels": n_vox,
                        "length_um": round(length, 1),
                        "chord_um": round(chord, 1),
                        "tortuosity": round(length / chord, 3) if chord > 0 else None,
                        "continuity": "connected",
                        **diameter_stats})

    # Compare whole-member medians with the lattice population. This catches
    # uniformly thin or thick struts that a within-member relative profile
    # cannot identify by itself.
    nominal_diameter = (float(np.median([m["diameter_median_um"] for m in members]))
                        if members else 0.0)
    for m in members:
        ratio = (m["diameter_median_um"] / nominal_diameter
                 if nominal_diameter > 0 else None)
        m["diameter_to_lattice_median"] = round(ratio, 4) if ratio is not None else None
        m["uniform_thinning_suspect"] = bool(ratio is not None and ratio < thin_ratio)
        m["uniform_thickening_suspect"] = bool(ratio is not None and ratio > thick_ratio)
        m["local_thinning_suspect"] = bool(m["longest_thin_run_fraction"] >= 0.10)
        m["local_thickening_suspect"] = bool(m["thick_fraction"] >= 0.10)

    # Degree, and the flag for a node that has lost a strut.
    degree = np.zeros(n_nodes, dtype=int)
    for m in members:
        degree[m["a"]] += 1
        degree[m["b"]] += 1

    interior = degree > 1
    modal = int(np.bincount(degree[interior]).argmax()) if interior.any() else 0
    node_slices = ndi.find_objects(node_lab)
    nodes = []
    for i in range(n_nodes):
        z, y, x = centres[i]
        node_slice = node_slices[i]
        node_region = node_lab[node_slice] == (i + 1)
        node_radii = dist[node_slice][node_region] * voxel_um
        node_voxels = int(node_region.sum())
        nodes.append({"id": i,
                      "x": round(float(x * voxel_um / 1000.0), 4),
                      "y": round(float(y * voxel_um / 1000.0), 4),
                      "z": round(float(z * voxel_um / 1000.0), 4),
                      "position_zyx_voxels": [round(float(z), 3),
                                               round(float(y), 3),
                                               round(float(x), 3)],
                      "skeleton_voxels": node_voxels,
                      "local_diameter_median_um": round(float(2.0 * np.median(node_radii)), 1),
                      "local_diameter_max_um": round(float(2.0 * node_radii.max()), 1),
                      "degree": int(degree[i]),
                      "suspect": bool(1 < degree[i] < modal)})
    for k, m in enumerate(members):
        m["id"] = k

    n_susp = sum(n["suspect"] for n in nodes)
    print(f"[graph] {len(members):,} members, modal degree {modal}, "
          f"{n_susp:,} nodes below it")
    return {"nodes": nodes, "members": members, "modal_degree": modal,
            "lattice_median_diameter_um": round(nominal_diameter, 1)}


# ----------------------------------------------------------------------
# 3. Viewer
# ----------------------------------------------------------------------
_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strut lattice graph</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --ink:#1c1d1f; --muted:#6b6f76; --line:#d9dbe0; --panel:#f7f8f9; --accent:#2b6d7d; --warn:#c0392b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; color:var(--ink); background:#fff; }
  header { padding:14px 22px 10px; border-bottom:1px solid var(--line); }
  h1 { font-size:16px; font-weight:600; margin:0 0 4px; }
  header p { margin:0; font-size:13px; color:var(--muted); }
  #wrap { display:flex; flex-direction:column; height:calc(100vh - 66px); }
  #plot { flex:1 1 auto; min-height:340px; }
  #panel { flex:0 0 auto; border-top:1px solid var(--line); background:var(--panel);
           padding:12px 22px 16px; display:grid;
           grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:10px 26px; }
  .ctrl { display:grid; grid-template-columns:78px 1fr 116px; align-items:center; gap:10px; }
  .ctrl label { font-size:12px; font-weight:600; }
  .ctrl output { font-size:12px; color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  #foot { grid-column:1/-1; display:flex; align-items:center; justify-content:space-between;
          gap:14px; font-size:12px; color:var(--muted); border-top:1px dashed var(--line); padding-top:9px; }
  button { font:inherit; font-size:12px; padding:5px 11px; border:1px solid var(--line);
           border-radius:4px; background:#fff; color:var(--ink); cursor:pointer; }
  button:hover { border-color:var(--accent); color:var(--accent); }
  button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Strut lattice graph</h1>
  <p>Members are coloured by measured diameter. A node drawn in red has fewer members than the
     modal degree of the lattice, which is where a strut is missing. Hover any node or member
     for its measurements.</p>
</header>
<div id="wrap">
  <div id="plot"></div>
  <div id="panel">
    <div class="ctrl"><label for="dmin">min diameter</label>
      <input type="range" id="dmin" min="0" max="__DMAX__" step="__DSTEP__" value="0">
      <output id="odmin"></output></div>
    <div class="ctrl"><label for="zmax">z above</label>
      <input type="range" id="zmin" min="__ZMIN__" max="__ZMAX__" step="__ZSTEP__" value="__ZMIN__">
      <output id="ozmin"></output></div>
    <div id="foot">
      <span id="count"></span>
      <span>
        <button id="tsusp" type="button">Suspect nodes only</button>
        <button id="tnodes" type="button" class="on">Nodes</button>
        <button id="reset" type="button">Reset</button>
      </span>
    </div>
  </div>
</div>
<script>
const G = __GRAPH__;
const N = G.nodes, M = G.members;

function build(dmin, zmin, suspOnly, showNodes) {
  const lx = [], ly = [], lz = [], lc = [], txt = [];
  let shown = 0;
  for (const m of M) {
    const a = N[m.a], b = N[m.b];
    if (m.diameter_um < dmin) continue;
    if (a.z < zmin && b.z < zmin) continue;
    lx.push(a.x, b.x, null); ly.push(a.y, b.y, null); lz.push(a.z, b.z, null);
    lc.push(m.diameter_um, m.diameter_um, m.diameter_um);
    const t = 'member ' + m.id + '<br>median diameter ' + m.diameter_um
            + ' um<br>5-95% diameter ' + m.diameter_p05_um + '-' + m.diameter_p95_um
            + ' um<br>roughness (IQR/median) ' + m.diameter_robust_roughness
            + '<br>thin fraction ' + (100*m.thin_fraction).toFixed(1) + '%'
            + '<br>longest thin run ' + (100*m.longest_thin_run_fraction).toFixed(1) + '%'
            + '<br>thick fraction ' + (100*m.thick_fraction).toFixed(1) + '%'
            + '<br>length ' + m.length_um + ' um<br>tortuosity ' + m.tortuosity;
    txt.push(t, t, '');
    shown++;
  }
  const edges = {
    type: 'scatter3d', mode: 'lines', name: 'members',
    x: lx, y: ly, z: lz, text: txt, hoverinfo: 'text',
    line: {width: 4, color: lc, colorscale: 'Viridis', cmin: __DLO__, cmax: __DHI__}
  };

  const nx = [], ny = [], nz = [], nc = [], nt = [];
  if (showNodes) {
    for (const n of N) {
      if (n.degree < 2) continue;
      if (suspOnly && !n.suspect) continue;
      if (n.z < zmin) continue;
      nx.push(n.x); ny.push(n.y); nz.push(n.z);
      nc.push(n.suspect ? '#c0392b' : '#2b6d7d');
      nt.push('node ' + n.id + '<br>degree ' + n.degree
              + (n.suspect ? '<br>below modal degree ' + G.modal_degree : ''));
    }
  }
  const pts = {
    type: 'scatter3d', mode: 'markers', name: 'nodes',
    x: nx, y: ny, z: nz, text: nt, hoverinfo: 'text',
    marker: {size: 3.5, color: nc}
  };
  return [[edges, pts], shown, nx.length];
}

const layout = {
  scene: {xaxis: {title: {text: 'x (mm)'}}, yaxis: {title: {text: 'y (mm)'}},
          zaxis: {title: {text: 'z (mm)'}}, aspectmode: 'data',
          camera: {eye: {x: 1.5, y: 1.5, z: 1.1}}},
  margin: {l: 0, r: 0, t: 8, b: 0}, showlegend: false, template: {layout: {}}
};

const dmin = document.getElementById('dmin'), zmin = document.getElementById('zmin');
const odmin = document.getElementById('odmin'), ozmin = document.getElementById('ozmin');
const cnt = document.getElementById('count');
const tsusp = document.getElementById('tsusp'), tnodes = document.getElementById('tnodes');
let suspOnly = false, showNodes = true, started = false, pending = false;

function apply() {
  pending = false;
  const r = build(+dmin.value, +zmin.value, suspOnly, showNodes);
  if (!started) { Plotly.newPlot('plot', r[0], layout, {responsive: true, displaylogo: false}); started = true; }
  else { Plotly.react('plot', r[0], layout); }
  odmin.textContent = (+dmin.value).toFixed(0) + ' um';
  ozmin.textContent = (+zmin.value).toFixed(2) + ' mm';
  cnt.textContent = r[1].toLocaleString() + ' of ' + M.length.toLocaleString()
                  + ' members, ' + r[2].toLocaleString() + ' nodes shown';
}
function schedule() { if (!pending) { pending = true; requestAnimationFrame(apply); } }
dmin.addEventListener('input', schedule);
zmin.addEventListener('input', schedule);
tsusp.addEventListener('click', function () {
  suspOnly = !suspOnly; tsusp.classList.toggle('on', suspOnly); apply();
});
tnodes.addEventListener('click', function () {
  showNodes = !showNodes; tnodes.classList.toggle('on', showNodes); apply();
});
document.getElementById('reset').addEventListener('click', function () {
  dmin.value = 0; zmin.value = __ZMIN__; suspOnly = false; showNodes = true;
  tsusp.classList.remove('on'); tnodes.classList.add('on'); apply();
});
apply();
</script>
</body>
</html>
"""


def write_viewer(graph, out_html, open_browser=True):
    nodes, members = graph["nodes"], graph["members"]
    if not members:
        print("[view] no members to draw")
        return None
    d = np.array([m["diameter_um"] for m in members])
    z = np.array([n["z"] for n in nodes])
    dlo, dhi = float(np.percentile(d, 5)), float(np.percentile(d, 95))
    html = (_PAGE
            .replace("__GRAPH__", json.dumps(graph))
            .replace("__DMAX__", f"{d.max():.1f}")
            .replace("__DSTEP__", f"{max(d.max() / 100.0, 0.1):.2f}")
            .replace("__DLO__", f"{dlo:.2f}").replace("__DHI__", f"{dhi:.2f}")
            .replace("__ZMIN__", f"{z.min():.4f}")
            .replace("__ZMAX__", f"{z.max():.4f}")
            .replace("__ZSTEP__", f"{max((z.max() - z.min()) / 200.0, 1e-4):.5f}"))
    out_html = os.path.abspath(out_html)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[save] {out_html}  ({os.path.getsize(out_html) / 1e6:.1f} MB)")
    if open_browser:
        try:
            webbrowser.open(Path(out_html).as_uri())
        except Exception:
            pass
    return out_html


# ----------------------------------------------------------------------
# 4. Entry point
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract the node and member graph.")
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_UM)
    ap.add_argument("--which", type=int, default=DEFAULT_WHICH)
    ap.add_argument("--roicube", type=int, default=DEFAULT_ROICUBE)
    ap.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    ap.add_argument("--threshold", default=DEFAULT_THRESHOLD)
    ap.add_argument("--isofrac", type=float, default=DEFAULT_ISO_FRACTION)
    ap.add_argument("--median", type=int, default=DEFAULT_MEDIAN)
    ap.add_argument("--minmember", type=float, default=DEFAULT_MIN_MEMBER_UM)
    ap.add_argument("--diameter-end-trim", type=float,
                    default=DEFAULT_DIAMETER_END_TRIM)
    ap.add_argument("--diameter-trim-mode", choices=("adaptive", "fraction", "none"),
                    default=DEFAULT_DIAMETER_TRIM_MODE)
    ap.add_argument("--max-adaptive-trim", type=float,
                    default=DEFAULT_MAX_ADAPTIVE_TRIM)
    ap.add_argument("--junction-excess", type=float, default=DEFAULT_JUNCTION_EXCESS)
    ap.add_argument("--trim-stability", type=float, default=DEFAULT_TRIM_STABILITY)
    ap.add_argument("--thin-ratio", type=float, default=DEFAULT_THIN_RATIO)
    ap.add_argument("--thick-ratio", type=float, default=DEFAULT_THICK_RATIO)
    ap.add_argument("--profile-points", type=int, default=DEFAULT_PROFILE_POINTS)
    ap.add_argument("--json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--html", default=DEFAULT_OUT_HTML)
    args = ap.parse_args(argv)

    print(f"[version] strut_graph {SCRIPT_VERSION}")
    path = args.path or DEFAULT_PATH
    vol, label = load_volume(path, args.which)

    if args.roicube and not all(args.roicube >= s for s in vol.shape):
        n = args.roicube
        c = [s // 2 for s in vol.shape]
        vol = vol[tuple(slice(max(0, ci - n // 2), min(si, ci + n // 2))
                        for ci, si in zip(c, vol.shape))]
        print(f"[roi] centred cube {vol.shape}")

    if args.median > 0:
        from scipy import ndimage as ndi
        k = 2 * args.median + 1
        print(f"[pre] 3D median filter, {k} cubed kernel, "
              f"{vol.size / 1e6:.0f} M voxels ...")
        vol = ndi.median_filter(vol, size=k)

    t_raw = raw_threshold(vol, args.threshold, args.isofrac)
    mask = vol > t_raw
    print(f"[seg] {mask.sum():,} metal voxels, "
          f"{100 * mask.mean():.2f} % of the field")
    del vol

    eff_um = args.voxel * max(args.downsample, 1)
    if args.downsample > 1:
        mask = occupancy_downsample(mask, args.downsample)
        print(f"[pre] decimated to {mask.shape}, block {eff_um:.0f} um")

    graph = build_graph(mask, eff_um, args.minmember,
                        args.diameter_end_trim, args.thin_ratio,
                        args.thick_ratio, args.profile_points,
                        args.diameter_trim_mode, args.max_adaptive_trim,
                        args.junction_excess, args.trim_stability)
    graph["meta"] = {"specimen": label, "voxel_um": args.voxel,
                     "downsample": args.downsample,
                     "effective_voxel_um": eff_um,
                     "threshold_raw": round(float(t_raw), 1),
                     "diameter_method": "2x medial-axis distance transform",
                     "diameter_trim_mode": args.diameter_trim_mode,
                     "diameter_end_trim_fraction": args.diameter_end_trim,
                     "max_adaptive_trim_fraction": args.max_adaptive_trim,
                     "junction_excess": args.junction_excess,
                     "trim_stability": args.trim_stability,
                     "thin_ratio": args.thin_ratio,
                     "thick_ratio": args.thick_ratio,
                     "n_nodes": len(graph["nodes"]),
                     "n_members": len(graph["members"]),
                     "version": SCRIPT_VERSION}

    out_json = args.json
    out_html = args.html
    here = Path(__file__).resolve().parent
    if not os.path.isabs(out_json):
        out_json = str(here / out_json)
    if not os.path.isabs(out_html):
        out_html = str(here / out_html)

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=1)
    print(f"[save] {out_json}  ({os.path.getsize(out_json) / 1e6:.1f} MB)")

    if graph["members"]:
        d = np.array([m["diameter_um"] for m in graph["members"]])
        L = np.array([m["length_um"] for m in graph["members"]])
        print(f"[stats] member diameter, um: median {np.median(d):.0f}, "
              f"quartiles {np.percentile(d, 25):.0f} to {np.percentile(d, 75):.0f}")
        print(f"[stats] member length,   um: median {np.median(L):.0f}, "
              f"quartiles {np.percentile(L, 25):.0f} to {np.percentile(L, 75):.0f}")

    write_viewer(graph, out_html)
    print("[done]")


if __name__ == "__main__":
    main()
