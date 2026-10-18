"""Outputs: counts, defect list, 3-D view of defect locations, marked slices (+ slideshow stack)."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {1: (255, 40, 40), 2: (255, 160, 0)}
NAMES = {1: "missing", 2: "broken"}

def _font(size):
    for name in ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

def _grid_bounds(env):
    proj = np.asarray(env).any(axis=0)
    ys, xs = np.where(proj)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

def _to_gray(sl):
    g = np.clip(np.asarray(sl, np.float32), 0, 1) * 255
    return np.dstack([g, g, g]).astype(np.uint8)

def render_all(out: Path, D: dict, grid: int = 6) -> dict:
    work = out / "work"
    lab = np.load(work / "candidate_labels.npy", mmap_mode="r")
    norm = np.load(work / "norm.npy", mmap_mode="r")
    env = np.load(work / "env.npy", mmap_mode="r")
    struts = sorted(D["struts"], key=lambda f: (f["cls"], (f["node0_zyx"][0] + f["node1_zyx"][0]) / 2))
    y0, y1, x0, x1 = _grid_bounds(env)
    gh, gw = (y1 - y0) / grid, (x1 - x0) / grid
    Z = lab.shape[0]
    # ------------------------------------------------ table (one row per flagged strut)
    rows = []; counters = {1: 0, 2: 0}
    void_cls = {}
    for f in struts:
        counters[f["cls"]] += 1
        f["tag"] = ("M" if f["cls"] == 1 else "B") + f"{counters[f['cls']]:03d}"
        a, b = np.array(f["node0_zyx"]), np.array(f["node1_zyx"])
        t = float(np.median(f["hit_t"])) if f.get("hit_t") else 0.5
        f["mark_zyx"] = (a + t * (b - a)).tolist()
        mz, my, mx = f["mark_zyx"]
        for v in f["voids"]:
            void_cls[v] = min(void_cls.get(v, 9), f["cls"])          # a void on a missing strut is drawn red
        rows.append(dict(defect_id=f["tag"], defect_class=NAMES[f["cls"]], absent_fraction_pct=round(100 * f["absent_fraction"], 1),
                         marked_slice=int(round(mz)), mark_y=round(my, 1), mark_x=round(mx, 1),
                         first_slice=int(np.floor(min(a[0], b[0]))), last_slice=int(np.ceil(max(a[0], b[0]))),
                         grid_row=int(np.clip((my - y0) // gh, 0, grid - 1)), grid_col=int(np.clip((mx - x0) // gw, 0, grid - 1)),
                         node0_zyx=json.dumps([round(float(v), 1) for v in a]), node1_zyx=json.dumps([round(float(v), 1) for v in b]),
                         void_ids=";".join(map(str, f["voids"])),
                         defect_probability=None if f.get("defect_probability") is None else round(f["defect_probability"], 3)))
    fields = ["defect_id", "defect_class", "absent_fraction_pct", "marked_slice", "mark_y", "mark_x", "first_slice", "last_slice",
              "grid_row", "grid_col", "node0_zyx", "node1_zyx", "void_ids", "defect_probability"]
    with open(out / "defects.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    # ------------------------------------------------ label volume (missing=1, broken=2)
    n_ids = int(max([c["id"] for c in D["candidates"]] + [0])) + 1
    lut = np.zeros(n_ids, np.uint8)
    for v, k in void_cls.items():
        lut[v] = k
    labels = np.lib.format.open_memmap(out / "defect_labels.npy", mode="w+", dtype=np.uint8, shape=lab.shape)
    for z in range(Z):
        l = np.asarray(lab[z]); labels[z] = lut[np.minimum(l, n_ids - 1)]
    labels.flush()
    # ------------------------------------------------ marked slices
    sdir = out / "marked_slices"; sdir.mkdir(exist_ok=True)
    for old in sdir.glob("*.png"):
        old.unlink()
    by_slice = {}
    for f in struts:
        by_slice.setdefault(int(round(f["mark_zyx"][0])), []).append(f)
    f_big, f_small = _font(18), _font(13)
    stack = []
    for z in sorted(by_slice):
        rgb = _to_gray(norm[z]); l = np.asarray(labels[z])
        for cls, col in COLORS.items():
            rgb[l == cls] = col
        img = Image.fromarray(rgb); d = ImageDraw.Draw(img)
        for i in range(grid + 1):
            yy = y0 + i * gh; xx = x0 + i * gw
            d.line([(x0, yy), (x1, yy)], fill=(70, 170, 255), width=1); d.line([(xx, y0), (xx, y1)], fill=(70, 170, 255), width=1)
        for f in by_slice[z]:
            _, cy, cx = f["mark_zyx"]; col = COLORS[f["cls"]]
            d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], outline=col, width=3)
            d.text((cx + 18, cy - 20), f"{f['tag']} {NAMES[f['cls']]} {100 * f['absent_fraction']:.0f}%", fill=col, font=f_small)
        tags = ", ".join(f["tag"] for f in by_slice[z])
        d.rectangle([0, 0, img.width, 26], fill=(0, 0, 0)); d.text((6, 3), f"slice {z}   {tags}", fill=(255, 255, 255), font=f_big)
        img.save(sdir / f"slice_{z:04d}.png")
        stack.append(np.asarray(img))
    if stack:
        np.save(out / "marked_slices_viewer_rgb.npy", np.stack(stack))
    # ------------------------------------------------ contact sheet (one crop per flagged strut)
    if struts:
        h = 64; tile = 2 * h; ncol = 12; nrow = int(np.ceil(len(struts) / ncol))
        sheet = Image.new("RGB", (ncol * tile, nrow * (tile + 16)), (0, 0, 0)); ds = ImageDraw.Draw(sheet)
        for k, f in enumerate(struts):
            z, y, x = [int(round(v)) for v in f["mark_zyx"]]
            ya, xa = max(y - h, 0), max(x - h, 0)
            crop = _to_gray(norm[z, ya:ya + tile, xa:xa + tile]); l = np.asarray(labels[z, ya:ya + tile, xa:xa + tile])
            for cls, col in COLORS.items():
                crop[l == cls] = col
            r, q = divmod(k, ncol)
            sheet.paste(Image.fromarray(crop), (q * tile, r * (tile + 16) + 16))
            ds.text((q * tile + 3, r * (tile + 16)), f"{f['tag']} z{z}", fill=COLORS[f["cls"]], font=f_small)
        sheet.save(out / "defect_contact_sheet.png")
    # ------------------------------------------------ 3-D view
    _render_3d(out, struts, env, lab.shape)
    n_m = sum(f["cls"] == 1 for f in struts); n_b = sum(f["cls"] == 2 for f in struts)
    summary = dict(missing_struts=n_m, broken_struts=n_b, total_defective_struts=n_m + n_b, classifier=D.get("method"),
                   rule="per strut: missing if >= 85% of its free length is void, broken if a void cuts it but covers < 85%",
                   n_void_candidates=len(D["candidates"]), n_accepted_voids=sum(c["cls"] > 0 for c in D["candidates"]),
                   grid=f"{grid}x{grid}", lattice=D["meta"],
                   files=dict(defects_csv="defects.csv", view_3d_png="defects_3d.png", view_3d_html="defects_3d.html",
                              marked_slices_dir="marked_slices", slideshow_npy="marked_slices_viewer_rgb.npy",
                              contact_sheet="defect_contact_sheet.png", label_volume="defect_labels.npy", trace="trace.jsonl",
                              defects_map_json="defects_map.json"))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    _write_defects_map(out, struts, rows, D, lab.shape, n_m, n_b)
    return dict(missing_struts=n_m, broken_struts=n_b, marked_slices=len(by_slice), out=str(out))

def _write_defects_map(out, struts, rows, D, shape, n_m, n_b):
    """defects_map.json: the flagged struts in the same layout as the repo's lattice JSONs
    (data/octet_truss_8x8x8/octet_truss_8x8x8.json and the registered JSONs):
      junctions  {id, position, indices}   indices = place inside its unit cell (0, 0.5, 1), x/y/z order
      struts     {id, junction0, junction1, ...defect fields}   both junctions belong to the strut's unit cell
      unit_cells {id, struts, indices}     indices = cell number [i, j, k] counted from the lattice corner
    plus a `defect_detection` block (like `ct_registration` in the registered JSON).
    Positions are CT voxels (x, y, z) with z = slice index, as in the registered JSONs. Everything comes from the
    lattice the detector rebuilt from the scan; no design file is read."""
    meta = D["meta"]; corner = meta.get("lattice_corner_grid_zyx"); ncell = meta.get("lattice_unit_cells_zyx")
    junctions, cells, out_struts = [], {}, []
    jkey = {}
    def junction(cell, local_xyz, zyx):
        k = (cell, tuple(local_xyz))
        if k not in jkey:
            jkey[k] = len(junctions)
            junctions.append(dict(id=jkey[k], position=[round(float(v), 2) for v in zyx[::-1]], indices=[float(v) for v in local_xyz]))
        return jkey[k]
    for f, r in zip(struts, rows):
        z = r["marked_slice"]
        s = dict(id=int(f["strut"]), junction0=None, junction1=None,
                 defect_id=r["defect_id"], defect_class=r["defect_class"], absent_fraction_pct=r["absent_fraction_pct"],
                 defect_probability=r["defect_probability"],
                 slices=dict(first=r["first_slice"], last=r["last_slice"], marked=z),
                 marked_slice_png=f"marked_slices/slice_{z:04d}.png",
                 mark_position=[r["mark_x"], r["mark_y"], round(float(f["mark_zyx"][0]), 1)],
                 grid_cell=dict(row=r["grid_row"], col=r["grid_col"]),
                 gap_ids=[int(v) for v in f["voids"]])
        if corner is not None and f.get("node0_grid_zyx") is not None:
            g0 = (np.array(f["node0_grid_zyx"]) - corner)[::-1] / 2.0      # global unit-cell coords, x/y/z
            g1 = (np.array(f["node1_grid_zyx"]) - corner)[::-1] / 2.0
            cell = np.clip(np.floor(np.minimum(g0, g1)), 0, np.array(ncell)[::-1] - 1).astype(int)
            cell = tuple(int(v) for v in cell)
            s["junction0"] = junction(cell, g0 - cell, f["node0_zyx"]); s["junction1"] = junction(cell, g1 - cell, f["node1_zyx"])
            cells.setdefault(cell, []).append(s["id"])
        else:   # run made before lattice numbering existed: endpoints only
            s["junction0"] = junction(("?", r["defect_id"]), (0.0, 0.0, 0.0), f["node0_zyx"])
            s["junction1"] = junction(("?", r["defect_id"]), (1.0, 1.0, 1.0), f["node1_zyx"])
        out_struts.append(s)
    unit_cells = [dict(id=k, struts=sorted(v), indices=list(c)) for k, (c, v) in enumerate(sorted(cells.items()))]
    doc = dict(junctions=junctions, struts=out_struts, unit_cells=unit_cells,
               defect_detection=dict(
                   technique="Pattern Recognition (self-referential; the design STL/JSON is never read)",
                   classifier=D.get("method"), missing_struts=n_m, broken_struts=n_b, total_defective_struts=n_m + n_b,
                   rule="missing if a gap covers >= 85% of the strut's free length, broken if a gap cuts it but covers < 85%",
                   volume_shape_zyx=[int(v) for v in shape],
                   lattice_unit_cells_xyz=None if ncell is None else list(ncell)[::-1],
                   coordinates="position / mark_position: CT voxels (x, y, z), z = slice index. junction indices: place inside "
                               "the unit cell (0, 0.5, 1). unit_cell indices: cell number from the lattice corner, estimated from "
                               "the scan. Strut ids are the detector's own lattice numbering, not the design JSON's ids.",
                   files=dict(marked_slices_dir="marked_slices", view_3d_html="defects_3d.html", defects_csv="defects.csv")))
    json.dump(doc, open(out / "defects_map.json", "w"), indent=1, default=float)


def _box_edges(lo, hi):
    edges = []
    for axis in range(3):
        o = [i for i in range(3) if i != axis]
        for a in (lo, hi):
            for b in (lo, hi):
                p0 = np.zeros(3); p0[o[0]] = a[o[0]]; p0[o[1]] = b[o[1]]
                p1 = p0.copy(); p0[axis] = lo[axis]; p1[axis] = hi[axis]
                edges.append((p0, p1))
    return edges

def _render_3d(out, struts, env, shape):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    e = np.asarray(env[::8, ::8, ::8]); zz, yy, xx = np.where(e)
    lo = np.array([xx.min(), yy.min(), zz.min()]) * 8; hi = np.array([xx.max(), yy.max(), zz.max()]) * 8
    fig = plt.figure(figsize=(15, 7))
    for k, (el, az) in enumerate([(25, -60), (60, 30)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for p0, p1 in _box_edges(lo, hi):
            ax.plot(*zip(p0, p1), color="0.6", lw=0.8)
        for cls, col, name in [(1, "#e8282a", "missing"), (2, "#ff9f00", "broken")]:
            first = True
            for c in struts:
                if c["cls"] != cls:
                    continue
                p0 = np.array(c["node0_zyx"])[::-1]; p1 = np.array(c["node1_zyx"])[::-1]
                ax.plot(*zip(p0, p1), color=col, lw=3 if cls == 1 else 2.2, label=name if first else None); first = False
        ax.set_xlabel("x (voxel)"); ax.set_ylabel("y (voxel)"); ax.set_zlabel("slice (z)")
        ax.view_init(elev=el, azim=az); ax.set_box_aspect(hi - lo)
        if k == 0:
            ax.legend(loc="upper left")
    n_m = sum(c["cls"] == 1 for c in struts); n_b = sum(c["cls"] == 2 for c in struts)
    fig.suptitle(f"Detected strut defects: {n_m} missing (red), {n_b} broken (orange)")
    plt.tight_layout(); fig.savefig(out / "defects_3d.png", dpi=110); plt.close(fig)
    # interactive version (plotly from CDN; open in a browser)
    traces = []
    for cls, col, name in [(1, "#e8282a", "missing"), (2, "#ff9f00", "broken")]:
        xs, ys, zs, txt = [], [], [], []
        for c in struts:
            if c["cls"] != cls:
                continue
            for p in (c["node0_zyx"], c["node1_zyx"]):
                zs.append(round(p[0], 1)); ys.append(round(p[1], 1)); xs.append(round(p[2], 1)); txt.append(c["tag"])
            xs.append(None); ys.append(None); zs.append(None); txt.append(None)
        traces.append(dict(type="scatter3d", mode="lines", x=xs, y=ys, z=zs, text=txt, name=name, line=dict(color=col, width=8 if cls == 1 else 6)))
    box = []
    for p0, p1 in _box_edges(lo, hi):
        box += [p0.tolist(), p1.tolist(), [None, None, None]]
    traces.append(dict(type="scatter3d", mode="lines", x=[p[0] for p in box], y=[p[1] for p in box], z=[p[2] for p in box],
                       name="specimen envelope", line=dict(color="#999", width=2), hoverinfo="skip"))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Strut defects 3D</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script></head>
<body style="margin:0;font-family:sans-serif"><div style="padding:8px">Detected strut defects: <b style="color:#e8282a">{n_m} missing</b>, <b style="color:#ff9f00">{n_b} broken</b>. Drag to rotate; hover for defect IDs (match defects.csv / marked_slices).</div>
<div id="p" style="width:100vw;height:90vh"></div><script>
Plotly.newPlot('p',{json.dumps(traces)},{{scene:{{aspectmode:'data',xaxis:{{title:'x'}},yaxis:{{title:'y'}},zaxis:{{title:'slice'}}}},margin:{{l:0,r:0,t:0,b:0}}}});
</script></body></html>"""
    (out / "defects_3d.html").write_text(html, encoding="utf-8")
