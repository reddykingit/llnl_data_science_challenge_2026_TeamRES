"""Strut-level scoring of a detector run against the reference (template Section 4 definitions).

Primary (strut list): every flagged strut in defects.csv is matched to the design strut whose midpoint is
nearest (must be < 8 voxels, via the refined registration). TP = defective reference strut flagged (any class);
FP = sound strut flagged, or a flagged strut that matches no design strut; FN = defective strut not flagged.
Fallback (voxel overlap, used for runs that only produce a label volume): a design strut
is flagged when >= 20 % of samples along its middle 60 % fall on detected voxels (4 px in-plane dilation).
Struts with reference -1 (absent build-plate face) are not scored.

python -m src.evaluation.score --results results/<specimen> --reference reference/<specimen> [--registered-json R.json]
python -m src.evaluation.score training-set --results ... --reference ... --specimen NAME --csv models/pnv_gbm_training_set.csv
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy import ndimage
from .reference import graph

TS = np.linspace(0.2, 0.8, 25); CMIN = 0.2; DIL = 4; N_BLOCKS = 5

def strut_hits(lab, P, S, dil=DIL):
    yy, xx = np.mgrid[-dil:dil + 1, -dil:dil + 1]; fp = yy ** 2 + xx ** 2 <= dil ** 2
    A, B = P[S[:, 0]], P[S[:, 1]]; pts = A[:, None, :] + TS[None, :, None] * (B - A)[:, None, :]
    z = np.clip(np.round(pts[..., 2]).astype(int), 0, lab.shape[0] - 1)
    y = np.clip(np.round(pts[..., 1]).astype(int), 0, lab.shape[1] - 1)
    x = np.clip(np.round(pts[..., 0]).astype(int), 0, lab.shape[2] - 1)
    hit = np.zeros(z.shape, np.uint8)
    for zz in np.unique(z):
        sl = np.asarray(lab[zz])
        if not sl.any():
            continue
        d = ndimage.grey_dilation(sl, footprint=fp); m = z == zz; hit[m] = d[y[m], x[m]]
    return (hit == 1).mean(1), (hit == 2).mean(1)

def predict(cm, cb, cmin=CMIN):
    pred = np.zeros(len(cm), np.int8); f = (cm + cb) >= cmin; pred[f] = np.where(cm[f] >= cb[f], 1, 2); return pred

def metrics(pred, ref, stl_ids=None):
    ev = ref >= 0; p = pred[ev] > 0; g = ref[ev] > 0
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    P_ = tp / max(tp + fp, 1); R_ = tp / max(tp + fn, 1); F = 2 * P_ * R_ / max(P_ + R_, 1e-9)
    o = dict(TP=tp, FP=fp, FN=fn, precision=round(P_, 4), recall=round(R_, 4), F1=round(F, 4))
    for c, n in [(1, "missing"), (2, "broken")]:
        gc = ref[ev] == c; pc = pred[ev] == c
        o[f"{n}_in_reference"] = int(gc.sum()); o[f"{n}_predicted"] = int(pc.sum())
        o[f"{n}_found_correct_class"] = int((pc & gc).sum()); o[f"{n}_found_any_class"] = int((p & gc).sum())
        o[f"{n}_recall"] = round(float((pc & gc).sum() / max(gc.sum(), 1)), 4)
    if stl_ids is not None:
        s = np.zeros(len(ref), bool); s[stl_ids] = True; s &= ev
        o["stl_removed_found"] = int((pred[s] > 0).sum()); o["stl_removed_n"] = int(s.sum())
    return o

def voxelized_design_iou(seg, P, S, missing_ids, radius=3.0, step=1.0):
    """IoU / Dice of the detector's segmentation vs the voxelised registered design (struts of radius `radius`,
    design-removed struts omitted). Proxy only: no voxel-level ground truth exists for this specimen."""
    G = np.zeros(seg.shape, bool); keep = np.ones(len(S), bool); keep[missing_ids] = False
    r = int(np.ceil(radius)); off = np.argwhere(np.ones((2 * r + 1,) * 3)) - r; off = off[(off ** 2).sum(1) <= radius ** 2]
    for j0, j1 in S[keep]:
        A, B = P[j0][::-1], P[j1][::-1]; n = int(np.linalg.norm(B - A) / step) + 1
        c = np.round(A + np.linspace(0, 1, n)[:, None] * (B - A)).astype(int)
        q = (c[:, None, :] + off[None]).reshape(-1, 3)
        q = q[np.all((q >= 0) & (q < seg.shape), 1)]; G[q[:, 0], q[:, 1], q[:, 2]] = True
    inter = np.logical_and(seg, G).sum(dtype=np.int64); s1 = seg.sum(dtype=np.int64); s2 = G.sum(dtype=np.int64)
    return dict(IoU=round(float(inter / (s1 + s2 - inter)), 4), Dice=round(float(2 * inter / (s1 + s2)), 4))

def predict_from_list(defects_csv: Path, P, S, max_dist=8.0):
    """Map the technique's flagged-strut list onto design struts. Returns per-design-strut prediction and
    the z of every flagged strut that matched no design strut (counted as FP)."""
    from scipy.spatial import cKDTree
    rows = list(csv.DictReader(open(defects_csv)))
    mid = (P[S[:, 0]] + P[S[:, 1]])[:, ::-1] / 2                       # (z,y,x)
    pred = np.zeros(len(S), np.int8); unmatched_z = []
    if rows:
        pm = np.array([(np.array(json.loads(r["node0_zyx"])) + np.array(json.loads(r["node1_zyx"]))) / 2 for r in rows])
        d, i = cKDTree(mid).query(pm)
        for r, dd, ii, p in zip(rows, d, i, pm):
            if dd < max_dist:
                pred[ii] = 1 if r["defect_class"] == "missing" else 2
            else:
                unmatched_z.append(float(p[0]))
    return pred, unmatched_z

def score(results: Path, reference: Path, with_iou=True):
    ref = np.load(reference / "reference_labels.npy"); P = np.load(reference / "junctions_refined_xyz.npy")
    stl_ids = np.load(reference / "stl_removed_ids.npy") if (reference / "stl_removed_ids.npy").exists() else None
    if stl_ids is not None and len(stl_ids) == 0:
        stl_ids = None
    S = np.load(reference / "struts.npy")
    asbuilt_missing = (np.load(reference / "asbuilt_missing_ids.npy") if (reference / "asbuilt_missing_ids.npy").exists()
                       else (stl_ids if stl_ids is not None else np.zeros(0, int)))
    lab = np.load(results / "defect_labels.npy", mmap_mode="r")
    csvp = results / "defects.csv"
    list_mode = csvp.exists() and "node0_zyx" in open(csvp).readline()
    if list_mode:
        pred, unmatched = predict_from_list(csvp, P, S)
    else:
        cm, cb = strut_hits(lab, P, S); pred = predict(cm, cb); unmatched = []
    def add_unmatched(m, zs):
        m = dict(m); m["FP"] += len(zs); m["unmatched_flags"] = len(zs)
        P_ = m["TP"] / max(m["TP"] + m["FP"], 1); R_ = m["recall"]
        m["precision"] = round(P_, 4); m["F1"] = round(2 * P_ * R_ / max(P_ + R_, 1e-9), 4); return m
    out = dict(scoring="strut_list" if list_mode else "voxel_overlap", overall=add_unmatched(metrics(pred, ref, stl_ids), unmatched))
    if (reference / "reference_labels_all.npy").exists():   # same run, every strut scored (build-plate face included)
        out["overall_all_struts"] = add_unmatched(metrics(pred, np.load(reference / "reference_labels_all.npy"), stl_ids), unmatched)
    zmid = (P[S[:, 0], 2] + P[S[:, 1], 2]) / 2; size = lab.shape[0] / N_BLOCKS
    blk = np.minimum((zmid / size).astype(int), N_BLOCKS - 1)
    folds = []
    for b in range(N_BLOCKS):
        r = ref.copy(); r[blk != b] = -1
        folds.append(add_unmatched(metrics(pred, r), [z for z in unmatched if min(int(z / size), N_BLOCKS - 1) == b]))
    out["per_zblock"] = folds
    for k in ["precision", "recall", "F1"]:
        v = np.array([f[k] for f in folds]); out[f"{k}_zblock_mean"] = round(float(v.mean()), 4); out[f"{k}_zblock_std"] = round(float(v.std()), 4)
    if with_iou and (results / "work" / "seg.npy").exists():
        out["segmentation_vs_design"] = voxelized_design_iou(np.load(results / "work" / "seg.npy"), P, S, asbuilt_missing)
    np.save(results / "strut_predictions.npy", pred)
    json.dump(out, open(results / "score.json", "w"), indent=1)
    return out

def training_set(results: Path, reference: Path, specimen: str, csv_path: Path, dil=3, min_hits=3):
    """Label every candidate (1 = overlaps a defective reference strut, else 0) for the GBM false-alarm filter."""
    from src.strut_defects.pnv import FEATURES
    ref = np.load(reference / "reference_labels.npy"); P = np.load(reference / "junctions_refined_xyz.npy"); S = np.load(reference / "struts.npy")
    D = json.load(open(results / "work" / "candidates.json")); lab = np.load(results / "work" / "candidate_labels.npy", mmap_mode="r")
    T = np.linspace(0.15, 0.85, 29); A, B = P[S[:, 0]], P[S[:, 1]]
    zyx = np.round((A[:, None, :] + T[None, :, None] * (B - A)[:, None, :])[..., ::-1]).astype(int)
    Z = lab.shape[0]; rows = []
    for c in D["candidates"]:
        (z0, z1), (y0, y1), (x0, x1) = c["bbox"]; p = dil + 1
        lo = np.array([max(z0 - p, 0), max(y0 - p, 0), max(x0 - p, 0)]); hi = np.minimum([z1 + p, y1 + p, x1 + p], lab.shape)
        m = ndimage.binary_dilation(np.asarray(lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) == c["id"], iterations=dil)
        inb = np.all((zyx >= lo) & (zyx < hi), axis=2); best = (-1, 0)
        for i in np.where(inb.any(1))[0]:
            q = zyx[i][inb[i]] - lo; h = int(m[q[:, 0], q[:, 1], q[:, 2]].sum())
            if h > best[1]:
                best = (i, h)
        i, h = best; r = int(ref[i]) if (h >= min_hits and i >= 0) else 0
        if r < 0:
            continue                                    # excluded face: do not train on it
        rows.append(dict(specimen=specimen, candidate_id=c["id"], z_block=min(int(c["centroid_zyx"][0] / (Z / N_BLOCKS)), N_BLOCKS - 1),
                         **{k: c["features"][k] for k in FEATURES}, matched_strut=int(i), reference_class=r, label=int(r > 0)))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    old = [r for r in csv.DictReader(open(csv_path))] if csv_path.exists() else []
    old = [r for r in old if r["specimen"] != specimen]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(old + rows)
    return dict(rows=len(rows), positives=sum(r["label"] for r in rows))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="score", choices=["score", "training-set"])
    ap.add_argument("--results", required=True); ap.add_argument("--reference", required=True)
    ap.add_argument("--specimen"); ap.add_argument("--csv", default="models/pnv_gbm_training_set.csv"); ap.add_argument("--no-iou", action="store_true")
    a = ap.parse_args()
    if a.mode == "score":
        print(json.dumps(score(Path(a.results), Path(a.reference), not a.no_iou)["overall"], indent=1))
    else:
        print(training_set(Path(a.results), Path(a.reference), a.specimen or Path(a.results).name, Path(a.csv)))
