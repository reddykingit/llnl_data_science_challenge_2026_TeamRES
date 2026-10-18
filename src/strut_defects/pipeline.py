"""End-to-end PNV strut-defect pipeline:  detect -> classify -> render.

    python -m src.strut_defects.pipeline run  --input scan.tif --out results/<specimen>
    python -m src.strut_defects.pipeline detect   --input scan.tif --out DIR
    python -m src.strut_defects.pipeline classify --out DIR --filter gbm|rules|none
    python -m src.strut_defects.pipeline render   --out DIR

Only two defect classes are produced (fair-comparison setting):
  missing strut : >= 85 % of the strut absent
  broken strut  : a void that cuts the strut but covers < 85 % of it
"""
from __future__ import annotations
import argparse, csv, json, os, platform, sys, time
from pathlib import Path
import numpy as np

from . import pnv

SPECIMEN_DEFAULT = "specimen"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_SET = REPO / "models" / "pnv_gbm_training_set.csv"


class Trace:
    """Execution trace (one JSON line per step) for the reporting template."""
    def __init__(self, out: Path):
        self.path = out / "trace.jsonl"; self.t0 = time.time()
    def log(self, step: str, seconds: float, **details):
        rec = dict(step=step, seconds=round(seconds, 3), wall_clock=time.strftime("%Y-%m-%dT%H:%M:%S"), **details)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"[{step}] {seconds:.1f}s", flush=True)


def _load_volume(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p)
    import tifffile
    return tifffile.imread(p)


# ---------------------------------------------------------------- detect
def detect(input_path: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True); work = out / "work"; work.mkdir(exist_ok=True)
    tr = Trace(out); t = time.time()
    vol = _load_volume(input_path); tr.log("load", time.time() - t, shape=list(vol.shape), dtype=str(vol.dtype)); t = time.time()
    norm = pnv.local_normalize(vol); del vol
    np.save(work / "norm.npy", norm); tr.log("normalize", time.time() - t); t = time.time()
    seg, thr = pnv.segment(norm); del norm
    tr.log("segment", time.time() - t, threshold=thr); t = time.time()
    H = pnv.lattice_basis(seg); vecs = pnv.neighbour_vectors(H)
    a_half = float(np.mean(np.linalg.norm(H, axis=0)))
    tr.log("lattice_vectors", time.time() - t, H_zyx=H.tolist(), period_slices=float(2 * abs(H[0]).max())); t = time.time()
    env = pnv.envelope(seg, a_half); tr.log("envelope", time.time() - t); t = time.time()
    expected, votes, valid = pnv.vote(seg, env, vecs)
    for name, arr in [("seg", seg), ("env", env), ("expected", expected), ("votes", votes), ("valid", valid)]:
        np.save(work / f"{name}.npy", arr)
    del env, votes, valid
    tr.log("periodic_neighbour_vote", time.time() - t, n_neighbours=len(vecs)); t = time.time()
    votes = np.load(work / "votes.npy", mmap_mode="r"); valid = np.load(work / "valid.npy", mmap_mode="r")
    lab, comps = pnv.candidates(seg, expected, votes, valid)
    np.save(work / "candidate_labels.npy", lab); del lab
    L_node, d_node = pnv.strut_geometry(H, expected); L_free = L_node - d_node
    pnv.rule_85(comps, L_free)
    tr.log("candidates_and_85_rule", time.time() - t, n_candidates=len(comps), L_node=L_node, d_node=d_node); t = time.time()
    del seg, expected
    lab = np.load(work / "candidate_labels.npy", mmap_mode="r")
    feats = pnv.features(comps, lab, np.load(work / "seg.npy", mmap_mode="r"), np.load(work / "expected.npy", mmap_mode="r"),
                         np.load(work / "norm.npy", mmap_mode="r"), np.load(work / "env.npy", mmap_mode="r"))
    for c, f in zip(comps, feats):
        c["features"] = f
    meta = dict(input=str(input_path), threshold=thr, H_zyx=H.tolist(), L_node=L_node, d_node=d_node, L_free=L_free,
                shape=list(lab.shape), missing_fraction_rule=pnv.MISSING_FRACTION)
    json.dump(dict(meta=meta, candidates=comps), open(work / "candidates.json", "w"), default=float)
    tr.log("features", time.time() - t)
    return meta


# ---------------------------------------------------------------- classify
def _train_gbm(rows, exclude_specimen=None, exclude_block=None, n_blocks=5, depth=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    use = [r for r in rows if not (r["specimen"] == exclude_specimen and int(r["z_block"]) == exclude_block)]
    X = np.array([[float(r[k]) for k in pnv.FEATURES] for r in use]); y = np.array([int(r["label"]) for r in use])
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, class_weight="balanced",
                                          random_state=0).fit(X, y)

def classify(out: Path, method: str = "gbm", training_set: str | None = None, specimen: str = SPECIMEN_DEFAULT) -> dict:
    tr = Trace(out); t = time.time()
    D = json.load(open(out / "work" / "candidates.json")); comps = D["candidates"]; feats = [c["features"] for c in comps]
    if method == "none":
        classes = {c["id"]: c["cls_85"] for c in comps}
    elif method == "rules":
        classes = pnv.filter_rules(comps, feats)
    elif method == "gbm":
        ts = Path(training_set or DEFAULT_TRAINING_SET)
        rows = list(csv.DictReader(open(ts)))
        n_blocks = 5; Z = D["meta"]["shape"][0]
        same = any(r["specimen"] == specimen for r in rows)
        classes = {}
        if same:
            # leave-region-out: a candidate is never classified by a model that saw its own z-block
            for b in range(n_blocks):
                model = _train_gbm(rows, exclude_specimen=specimen, exclude_block=b)
                sub = [(c, f) for c, f in zip(comps, feats) if min(int(c["centroid_zyx"][0] / (Z / n_blocks)), n_blocks - 1) == b]
                if sub:
                    classes.update(pnv.filter_gbm([c for c, _ in sub], [f for _, f in sub], model))
        else:
            classes = pnv.filter_gbm(comps, feats, _train_gbm(rows))
    else:
        raise ValueError(method)
    for c in comps:
        c["cls"] = int(classes.get(c["id"], 0))
    tr.log("classify_voids", time.time() - t, method=method, accepted_voids=sum(c["cls"] > 0 for c in comps)); t = time.time()
    # per-strut decision on the lattice rebuilt from the scan's own periodicity (85 % rule per strut)
    work = out / "work"; m = D["meta"]
    nodes, struts, node_grid = pnv.lattice_graph(np.array(m["H_zyx"]), np.load(work / "seg.npy", mmap_mode="r"),
                                                 np.load(work / "env.npy", mmap_mode="r"), m["d_node"], return_grid=True)
    flagged = pnv.assign_struts(nodes, struts, np.load(work / "candidate_labels.npy"), classes, m["L_node"], m["d_node"])
    prob = {c["id"]: c.get("defect_probability") for c in comps}
    for f in flagged:
        ps = [prob[v] for v in f["voids"] if prob.get(v) is not None]
        f["defect_probability"] = max(ps) if ps else None
        # lattice identity of the strut, for defects_map.json (same layout as the design JSON: junction ids + unit-cell indices)
        j0, j1 = (int(j) for j in struts[f["strut"]])
        f["junction0"], f["junction1"] = j0, j1
        f["node0_grid_zyx"], f["node1_grid_zyx"] = node_grid[j0].tolist(), node_grid[j1].tolist()
    corner, n_cells = pnv.lattice_box(node_grid, nodes, np.load(work / "seg.npy", mmap_mode="r"))
    json.dump(dict(meta=dict(m, n_lattice_nodes=len(nodes), n_lattice_struts=len(struts),
                             lattice_corner_grid_zyx=corner, lattice_unit_cells_zyx=n_cells),
                   method=method, candidates=comps, struts=flagged),
              open(work / "classified.json", "w"), default=float)
    n_m = sum(f["cls"] == 1 for f in flagged); n_b = sum(f["cls"] == 2 for f in flagged)
    tr.log("assign_struts_85_rule", time.time() - t, lattice_struts=len(struts), missing=n_m, broken=n_b)
    return dict(missing=n_m, broken=n_b)


# ---------------------------------------------------------------- render
def render(out: Path, grid: int = 6) -> dict:
    from . import render as R
    tr = Trace(out); t = time.time()
    D = json.load(open(out / "work" / "classified.json"))
    res = R.render_all(out, D, grid=grid)
    tr.log("render", time.time() - t, **{k: v for k, v in res.items() if isinstance(v, (int, float, str))})
    return res


def run(input_path: str, out: Path, method: str = "gbm", training_set: str | None = None, specimen: str = SPECIMEN_DEFAULT) -> dict:
    t0 = time.time(); out.mkdir(parents=True, exist_ok=True)
    (out / "trace.jsonl").unlink(missing_ok=True)
    detect(input_path, out); classify(out, method, training_set, specimen); res = render(out)
    total = time.time() - t0
    Trace(out).log("total", total, host=platform.node(), cpu=platform.processor() or platform.machine(), cpus=os.cpu_count())
    res["runtime_seconds"] = round(total, 1)
    s = json.load(open(out / "summary.json")); s["runtime_seconds"] = res["runtime_seconds"]; json.dump(s, open(out / "summary.json", "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "detect", "classify", "render"])
    ap.add_argument("--input"); ap.add_argument("--out", required=True)
    ap.add_argument("--filter", default="gbm", choices=["gbm", "rules", "none"])
    ap.add_argument("--training-set", default=None)
    ap.add_argument("--specimen", default=None, help="specimen name (used to keep leave-region-out honest)")
    a = ap.parse_args(argv); out = Path(a.out); spec = a.specimen or out.name
    if a.stage == "run":
        print(json.dumps(run(a.input, out, a.filter, a.training_set, spec), indent=1))
    elif a.stage == "detect":
        print(json.dumps(detect(a.input, out), indent=1, default=float))
    elif a.stage == "classify":
        print(classify(out, a.filter, a.training_set, spec))
    else:
        print(json.dumps(render(out), indent=1))

if __name__ == "__main__":
    main()
