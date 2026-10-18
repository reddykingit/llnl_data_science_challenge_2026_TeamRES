"""Build the strut-level reference used to score the detector.

reference label per design strut:  0 sound | 1 missing | 2 broken | -1 excluded
  missing = CT shows >= 85 % of the strut absent  (OR removed in a defect-design STL, if one is given)
  broken  = not missing, and no connected material path between its two nodes in the CT
  excluded= struts touching the y-max face (that face layer is absent in the specimen: build-plate cut)

Steps
 1. refine the supplied CT<->JSON alignment (the supplied JSON is ~1.5 % too large in x): affine fit of
    design junctions to the CT junction blobs, then per-node snap (<4 voxels)
 2. tube audit: resample a radius-7 tube around every strut, measure absent fraction + end-to-end connectivity
 3. design removals: struts present in 0.stl but not in 0.5.stl (STL frame), mapped to the JSON by the
    cube symmetry that best matches the CT
Unified DSC input (0 % design; every defect visible in the CT counts):
python -m src.evaluation.reference --ct scan.tif --registered-json "<scan>.ct_registered.json" --out reference/<specimen>
Older mode with design removals:
python -m src.evaluation.reference --ct scan.tif --registered-json R.json --nominal-json N.json --stl-full 0.stl --stl-defect 0.5.stl --out reference/

Outputs: reference_labels.npy (build-plate face excluded, used for the headline score), reference_labels_all.npy
(every strut scored), asbuilt_missing_ids.npy (struts absent in the CT, omitted from the voxelised design for IoU).
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

def graph(path):
    R = json.load(open(path))
    P = np.array([j["position"] for j in R["junctions"]], float)          # (x, y, z) ; z = slice
    S = np.array([[s["junction0"], s["junction1"]] for s in R["struts"]])
    return P, S

def global_indices(path):
    """Global lattice index (x, y, z) of every junction: its unit cell's indices + its place inside the cell."""
    R = json.load(open(path)); G = np.full((len(R["junctions"]), 3), np.nan)
    for c in R["unit_cells"]:
        for s in c["struts"]:
            for j in (R["struts"][s]["junction0"], R["struts"][s]["junction1"]):
                G[j] = np.array(c["indices"], float) + np.array(R["junctions"][j]["indices"], float)
    return G

def normalize_for_audit(v, f=4, win=24):
    s = v[::f, ::f, ::f].astype(np.float32)
    bg = ndimage.uniform_filter(ndimage.grey_erosion(s, size=(win // 2,) * 3), size=win // 2)
    hi = ndimage.uniform_filter(ndimage.maximum_filter(s, size=win), size=win // 2)
    out = np.empty(v.shape, np.float32); Z, Y, X = v.shape
    for z in range(Z):
        cz = min(z / f, s.shape[0] - 1); z0 = int(cz); z1 = min(z0 + 1, s.shape[0] - 1); w = cz - z0
        b2 = ndimage.zoom((1 - w) * bg[z0] + w * bg[z1], (Y / bg.shape[1], X / bg.shape[2]), order=1)
        h2 = ndimage.zoom((1 - w) * hi[z0] + w * hi[z1], (Y / hi.shape[1], X / hi.shape[2]), order=1)
        out[z] = np.clip((v[z].astype(np.float32) - b2) / np.maximum(h2 - b2, 1.0), 0, 1.5)
    return out

# ---------------------------------------------------------------- 1. registration
def junction_centroids(v, P, win=10):
    Z, Y, X = v.shape; C = np.full_like(P, np.nan)
    g = np.mgrid[-win:win + 1, -win:win + 1, -win:win + 1]
    for i, (x, y, z) in enumerate(P):
        xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
        if not (win <= xi < X - win and win <= yi < Y - win and win <= zi < Z - win):
            continue
        b = np.clip(v[zi - win:zi + win + 1, yi - win:yi + win + 1, xi - win:xi + win + 1] - 0.5, 0, None); s = b.sum()
        if s >= 5:
            C[i] = [xi + (b * g[2]).sum() / s, yi + (b * g[1]).sum() / s, zi + (b * g[0]).sum() / s]
    return C

def refine_registration(v, P_reg, Q_nom, iters=4):
    P = P_reg.copy(); A = np.c_[Q_nom, np.ones(len(Q_nom))]
    for _ in range(iters):
        C = junction_centroids(v, P); ok = np.isfinite(C[:, 0]); d = np.linalg.norm(C - P, axis=1)
        keep = ok & (d < np.nanpercentile(d[ok], 90) * 1.5)
        M = np.linalg.lstsq(A[keep], C[keep], rcond=None)[0]; P = A @ M
    C = junction_centroids(v, P); d = np.linalg.norm(C - P, axis=1); use = np.isfinite(d) & (d < 4)
    Pl = P.copy(); Pl[use] = C[use]
    return Pl, M

# ---------------------------------------------------------------- 2. tube audit
def tube_audit(v, P, S, RT=7, NT=64, THR=0.3):
    out = np.zeros((len(S), 2))
    ts = np.linspace(0, 1, NT); g = np.arange(-RT, RT + 1)
    T, G1, G2 = np.meshgrid(ts, g, g, indexing="ij"); disk = (G1 ** 2 + G2 ** 2) <= RT ** 2
    mid = (ts >= 0.12) & (ts <= 0.88)
    for i, (j0, j1) in enumerate(S):
        A, B = P[j0], P[j1]; D = B - A; u = D / np.linalg.norm(D)
        tmp = np.array([1., 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1., 0])
        e1 = np.cross(u, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
        pts = A + T[..., None] * D + G1[..., None] * e1 + G2[..., None] * e2
        val = map_coordinates(v, pts[..., ::-1].reshape(-1, 3).T, order=1, mode="nearest").reshape(T.shape)
        m = (val > THR) & disk
        m = ndimage.binary_opening(m, structure=np.ones((1, 2, 2))) | ((val > THR + 0.15) & disk)
        absent = float((~m[mid].any(axis=(1, 2))).mean())
        lab, _ = ndimage.label(m, structure=np.ones((3, 3, 3)))
        a = set(np.unique(lab[ts < 0.1])) - {0}; b = set(np.unique(lab[ts > 0.9])) - {0}
        out[i] = [absent, float(len(a & b) > 0)]
    return out

# ---------------------------------------------------------------- 3. STL design removals
def load_stl(p):
    with open(p, "rb") as f:
        f.read(80); n = np.frombuffer(f.read(4), "<u4")[0]
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        return np.frombuffer(f.read(int(n) * 50), dt)["v"].reshape(-1, 3)

def stl_removed(Q, S, stl_full, stl_defect, mm_per_unit=2.28):
    """struts with surface near their axis in the full design but not in the defect design (STL frame, identity map)."""
    A, B = Q[S[:, 0]], Q[S[:, 1]]; c0 = Q.mean(0); res = {}
    for name, p in [("full", stl_full), ("defect", stl_defect)]:
        v = load_stl(p); tree = cKDTree(v - (v.min(0) + v.max(0)) / 2); cnt = np.zeros(len(S), int)
        for t in (0.35, 0.5, 0.65):
            cnt += np.array([len(x) for x in tree.query_ball_point(((A + t * (B - A)) - c0) * mm_per_unit, r=0.35)])
        res[name] = cnt
    return (res["full"] > 0) & (res["defect"] == 0)

def match_symmetry(Q, S, removed, ct_absent):
    mid = (Q[S[:, 0]] + Q[S[:, 1]]) / 2; c0 = Q.mean(0); tree = cKDTree(mid); best = None
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product([1, -1], repeat=3):
            _, idx = tree.query((mid[removed] - c0)[:, perm] * np.array(flips) + c0)
            hit = int(ct_absent[idx].sum())
            if best is None or hit > best[0]:
                best = (hit, perm, flips, np.sort(idx))
    return best

def build(ct, registered_json, nominal_json=None, stl_full=None, stl_defect=None, out: Path = Path("reference"), exclude_face=("y", "max")):
    import tifffile
    out.mkdir(parents=True, exist_ok=True)
    v = normalize_for_audit(tifffile.imread(ct))
    P0, S = graph(registered_json)
    Q = graph(nominal_json)[0] if nominal_json else P0          # unified input: the registered JSON is its own design
    P, M = refine_registration(v, P0, Q)
    np.save(out / "junctions_refined_xyz.npy", P)
    audit = tube_audit(v, P, S); np.save(out / "tube_audit.npy", audit)
    af, con = audit[:, 0], audit[:, 1] > 0.5
    if stl_defect:
        removed = stl_removed(Q, S, stl_full, stl_defect)
        hit, perm, flips, stl_ids = match_symmetry(Q, S, removed, af >= 0.85)
        sym = dict(perm=list(perm), flips=list(flips), removed_struts_matching_ct_absent=hit)
    else:
        stl_ids = np.zeros(0, int); sym = None
    gi = global_indices(registered_json); ax = "xyz".index(exclude_face[0])
    val = np.nanmax(gi[:, ax]) if exclude_face[1] == "max" else np.nanmin(gi[:, ax])
    excl = (np.abs(gi[S[:, 0], ax] - val) < 1e-6) | (np.abs(gi[S[:, 1], ax] - val) < 1e-6)
    stl = np.zeros(len(S), bool); stl[stl_ids] = True
    miss_ct = af >= 0.85; broken_ct = ~miss_ct & ~con
    ref_all = np.zeros(len(S), np.int8); ref_all[broken_ct] = 2; ref_all[miss_ct | stl] = 1
    ref = ref_all.copy(); ref[excl] = -1
    np.save(out / "reference_labels.npy", ref); np.save(out / "reference_labels_all.npy", ref_all)
    np.save(out / "asbuilt_missing_ids.npy", np.where(ref_all == 1)[0])
    np.save(out / "stl_removed_ids.npy", stl_ids); np.save(out / "struts.npy", S)
    ev = ref >= 0
    summary = dict(inputs=dict(ct=str(ct), registered_json=str(registered_json), nominal_json=nominal_json, stl_full=stl_full, stl_defect=stl_defect),
                   registration_affine_nominal_to_ct_xyz=M.tolist(), stl_symmetry=sym,
                   n_struts=int(len(S)), excluded=int(excl.sum()), evaluated=int(ev.sum()), stl_removed_total=int(stl.sum()),
                   stl_removed_evaluated=int((stl & ev).sum()), stl_removed_ct_missing=int((stl & ev & miss_ct).sum()),
                   stl_removed_ct_broken=int((stl & ev & broken_ct).sum()), stl_removed_ct_intact=int((stl & ev & ~miss_ct & ~broken_ct).sum()),
                   ct_missing_not_in_stl=int((miss_ct & ev & ~stl).sum()), ref_missing=int((ref == 1).sum()), ref_broken=int((ref == 2).sum()), ref_sound=int((ref == 0).sum()),
                   all_struts=dict(missing=int((ref_all == 1).sum()), broken=int((ref_all == 2).sum()), sound=int((ref_all == 0).sum())),
                   excluded_face_breakdown=dict(missing=int((ref_all[excl] == 1).sum()), broken=int((ref_all[excl] == 2).sum()), sound=int((ref_all[excl] == 0).sum())))
    json.dump(summary, open(out / "reference_summary.json", "w"), indent=1)
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ["ct", "registered-json", "out"]:
        ap.add_argument(f"--{k}", required=True)
    for k in ["nominal-json", "stl-full", "stl-defect"]:
        ap.add_argument(f"--{k}", default=None)
    a = ap.parse_args()
    print(json.dumps(build(a.ct, a.registered_json, a.nominal_json, a.stl_full, a.stl_defect, Path(a.out)), indent=1))
