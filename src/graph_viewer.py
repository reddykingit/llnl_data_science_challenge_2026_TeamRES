"""
graph_viewer.py
===============

Reads a strut lattice graph in JSON and writes an interactive HTML page
that opens in Chrome.

It accepts two schemas and detects which it has:

  reference   {"junctions": [{"id", "position": [x, y, z], "indices"}],
               "struts":    [{"id", "junction0", "junction1", "thickness",
                              "unit_cell_edge_idx"}],
               "unit_cells":[{"id", "struts", "indices"}]}

  extracted   the output of strut_graph.py, which carries the same
              junctions and struts plus measured length and diameter

The reference files distributed with this dataset have their first line
overwritten with junk in place of the opening brace. That is repaired
automatically and reported, rather than failing to parse.

Run it directly. Requirements:  pip install numpy
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

SCRIPT_VERSION = "2026-07-23-a"

# ======================================================================
# USER SETTINGS  -- used when the script is started with no arguments.
# ======================================================================
# A single .json file, or a folder holding several. When a folder is given
# its contents are listed and DEFAULT_WHICH selects which one to view.
DEFAULT_JSON = r"C:\Users\Owner\Downloads\DSC\llnl_data_science_challenge_2026_TeamRES\data\missing_struts\registered_jsons"
DEFAULT_WHICH = 0           # index into that listing
DEFAULT_ALL = False         # True writes one page per file in the folder
DEFAULT_VOXEL_UM = 10.0     # to convert positions from voxels to millimetres
DEFAULT_OUT_HTML = "graph_view.html"
# ======================================================================


# ----------------------------------------------------------------------
# 1. Reading, with header repair
# ----------------------------------------------------------------------
def _natural_key(text):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", str(text))]


def list_json_files(path, max_report=40):
    """Return the JSON files at a path, whether it is a file or a folder."""
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        sys.exit(f"path does not exist: {p}")
    files = sorted([f for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() == ".json"],
                   key=lambda f: _natural_key(f.name))
    if not files:
        sys.exit(f"no .json files in {p}")
    print(f"[list] {len(files)} JSON file(s) in {p}")
    for i, f in enumerate(files[:max_report]):
        print(f"       [{i:>3}] {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    if len(files) > max_report:
        print(f"       ... and {len(files) - max_report} more")
    return files


def load_graph_json(path):
    """Parse a graph file, repairing a corrupted opening line if present.

    The first line is replaced by an opening brace and the parse retried.
    The original line is printed rather than discarded silently, since a
    damaged header may itself be information about how the file was
    produced.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        lines = text.split("\n")
        bad = lines[0]
        lines[0] = "{"
        try:
            data = json.loads("\n".join(lines))
        except json.JSONDecodeError as exc:
            sys.exit(f"could not parse {path} even after repairing line 1: {exc}")
        print(f"[repair] line 1 was not valid JSON and has been replaced by an "
              f"opening brace. Original content: {bad!r}")
        return data, bad


def analyse(data, voxel_um):
    """Summarise the lattice and derive the quantities the viewer needs."""
    J = data.get("junctions", [])
    S = data.get("struts", [])
    U = data.get("unit_cells", [])
    if not J or not S:
        sys.exit("the file holds no junctions or no struts")

    P = np.array([j["position"] for j in J], dtype=float)
    a = np.array([s["junction0"] for s in S], dtype=int)
    b = np.array([s["junction1"] for s in S], dtype=int)
    L = np.linalg.norm(P[a] - P[b], axis=1)

    deg = np.bincount(np.concatenate([a, b]), minlength=len(J))
    interior = deg > 2
    modal = int(np.bincount(deg[interior]).argmax()) if interior.any() else 0

    # Cell pitch, from the number of unit cells spanned by the junctions.
    pitch = None
    if U:
        idx = np.array([u["indices"] for u in U], dtype=float)
        n_cells = idx.max(axis=0) - idx.min(axis=0) + 1
        pitch = float(np.median((P.max(axis=0) - P.min(axis=0)) / n_cells))

    print(f"[graph] {len(J):,} junctions, {len(S):,} struts, {len(U):,} unit cells")
    print(f"[graph] strut length, voxels: median {np.median(L):.2f}, "
          f"range {L.min():.2f} to {L.max():.2f}")
    if pitch:
        print(f"[graph] unit cell pitch {pitch:.1f} voxels = "
              f"{pitch * voxel_um:.0f} um")
    print(f"[graph] degree histogram: "
          f"{dict(enumerate(np.bincount(deg).tolist()))}")
    print(f"[graph] modal degree of interior junctions: {modal}")

    # Thickness may be absolute or relative to the cell pitch. A value well
    # below one is relative by construction, since no strut is a fraction of
    # a voxel thick.
    th = np.array([s.get("thickness", np.nan) for s in S], dtype=float)
    if np.isfinite(th).any() and np.nanmedian(th) < 1.0 and pitch:
        diam_um = th * pitch * voxel_um
        print(f"[graph] thickness {np.nanmedian(th):.3f} read as a fraction of "
              f"the pitch, giving {np.nanmedian(diam_um):.0f} um")
    elif "diameter_um" in S[0]:
        diam_um = np.array([s["diameter_um"] for s in S], dtype=float)
    else:
        diam_um = th * voxel_um

    return dict(P=P, a=a, b=b, L=L, deg=deg, modal=modal,
                diam_um=diam_um, pitch=pitch)


# ----------------------------------------------------------------------
# 2. Viewer
# ----------------------------------------------------------------------
_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --ink:#1c1d1f; --muted:#6b6f76; --line:#d9dbe0; --panel:#f7f8f9; --accent:#2b6d7d; --warn:#c0392b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; color:var(--ink); background:#fff; }
  header { padding:14px 22px 10px; border-bottom:1px solid var(--line); }
  h1 { font-size:16px; font-weight:600; margin:0 0 4px; }
  header p { margin:0; font-size:13px; color:var(--muted); }
  #wrap { display:flex; flex-direction:column; height:calc(100vh - 68px); }
  #plot { flex:1 1 auto; min-height:340px; }
  #panel { flex:0 0 auto; border-top:1px solid var(--line); background:var(--panel);
           padding:12px 22px 16px; display:grid;
           grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px 26px; }
  .ctrl { display:grid; grid-template-columns:26px 1fr 132px; align-items:center; gap:10px; }
  .ctrl label { font-size:13px; font-weight:600; }
  .ctrl output { font-size:12px; color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
  .pair { display:grid; gap:2px; }
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
  <h1>__TITLE__</h1>
  <p>__SUBTITLE__</p>
</header>
<div id="wrap">
  <div id="plot"></div>
  <div id="panel">
    <div class="ctrl"><label>x</label>
      <div class="pair">
        <input type="range" id="x0" min="__XMIN__" max="__XMAX__" step="__XSTEP__" value="__XMIN__">
        <input type="range" id="x1" min="__XMIN__" max="__XMAX__" step="__XSTEP__" value="__XMAX__">
      </div><output id="ox"></output></div>
    <div class="ctrl"><label>y</label>
      <div class="pair">
        <input type="range" id="y0" min="__YMIN__" max="__YMAX__" step="__YSTEP__" value="__YMIN__">
        <input type="range" id="y1" min="__YMIN__" max="__YMAX__" step="__YSTEP__" value="__YMAX__">
      </div><output id="oy"></output></div>
    <div class="ctrl"><label>z</label>
      <div class="pair">
        <input type="range" id="z0" min="__ZMIN__" max="__ZMAX__" step="__ZSTEP__" value="__ZMIN__">
        <input type="range" id="z1" min="__ZMIN__" max="__ZMAX__" step="__ZSTEP__" value="__ZMAX__">
      </div><output id="oz"></output></div>
    <div id="foot">
      <span id="count"></span>
      <span>
        <button id="bslab" type="button">z slab</button>
        <button id="bnodes" type="button">Junctions</button>
        <button id="bsusp" type="button">Under connected only</button>
        <button id="breset" type="button">Reset</button>
      </span>
    </div>
  </div>
</div>
<script>
const D = __DATA__;
const PX = D.px, PY = D.py, PZ = D.pz;      // junction coordinates, mm
const A = D.a, B = D.b, DIA = D.dia, DEG = D.deg;
const MODAL = D.modal, LIM = D.lim;
const nS = A.length;

function inside(v, lo, hi) { return v >= lo && v <= hi; }

function build(b, showNodes, suspOnly) {
  const lx = [], ly = [], lz = [], lc = [], lt = [];
  let shown = 0;
  for (let s = 0; s < nS; s++) {
    const i = A[s], j = B[s];
    if (!(inside(PX[i], b.x0, b.x1) && inside(PY[i], b.y0, b.y1) && inside(PZ[i], b.z0, b.z1))) continue;
    if (!(inside(PX[j], b.x0, b.x1) && inside(PY[j], b.y0, b.y1) && inside(PZ[j], b.z0, b.z1))) continue;
    lx.push(PX[i], PX[j], null); ly.push(PY[i], PY[j], null); lz.push(PZ[i], PZ[j], null);
    lc.push(DIA[s], DIA[s], DIA[s]);
    const t = 'strut ' + s + '<br>diameter ' + DIA[s].toFixed(0) + ' um';
    lt.push(t, t, '');
    shown++;
  }
  const edges = {type: 'scatter3d', mode: 'lines', x: lx, y: ly, z: lz,
                 text: lt, hoverinfo: 'text',
                 line: {width: 3, color: lc, colorscale: 'Viridis',
                        cmin: D.dlo, cmax: D.dhi}};

  const nx = [], ny = [], nz = [], nc = [], nt = [];
  if (showNodes) {
    for (let i = 0; i < PX.length; i++) {
      if (DEG[i] < 1) continue;
      const susp = DEG[i] > 2 && DEG[i] < MODAL;
      if (suspOnly && !susp) continue;
      if (!(inside(PX[i], b.x0, b.x1) && inside(PY[i], b.y0, b.y1) && inside(PZ[i], b.z0, b.z1))) continue;
      nx.push(PX[i]); ny.push(PY[i]); nz.push(PZ[i]);
      nc.push(susp ? '#c0392b' : '#2b6d7d');
      nt.push('junction ' + i + '<br>degree ' + DEG[i]
              + (susp ? '<br>below the modal degree of ' + MODAL : ''));
    }
  }
  const pts = {type: 'scatter3d', mode: 'markers', x: nx, y: ny, z: nz,
               text: nt, hoverinfo: 'text', marker: {size: 3, color: nc}};
  return [[edges, pts], shown, nx.length];
}

const layout = {
  scene: {xaxis: {title: {text: 'x (mm)'}}, yaxis: {title: {text: 'y (mm)'}},
          zaxis: {title: {text: 'z (mm)'}}, aspectmode: 'data',
          camera: {eye: {x: 1.5, y: 1.5, z: 1.1}}},
  margin: {l: 0, r: 0, t: 8, b: 0}, showlegend: false
};

const S = {};
['x0','x1','y0','y1','z0','z1'].forEach(function (k) { S[k] = document.getElementById(k); });
const O = {x: document.getElementById('ox'), y: document.getElementById('oy'), z: document.getElementById('oz')};
const cnt = document.getElementById('count');
const bnodes = document.getElementById('bnodes'), bsusp = document.getElementById('bsusp');
let showNodes = false, suspOnly = false, started = false, pending = false;

function apply() {
  pending = false;
  ['x','y','z'].forEach(function (ax) {
    if (+S[ax+'0'].value > +S[ax+'1'].value) S[ax+'0'].value = S[ax+'1'].value;
  });
  const b = {x0:+S.x0.value, x1:+S.x1.value, y0:+S.y0.value, y1:+S.y1.value,
             z0:+S.z0.value, z1:+S.z1.value};
  const r = build(b, showNodes, suspOnly);
  if (!started) { Plotly.newPlot('plot', r[0], layout, {responsive:true, displaylogo:false}); started = true; }
  else { Plotly.react('plot', r[0], layout); }
  O.x.textContent = b.x0.toFixed(2) + ' to ' + b.x1.toFixed(2) + ' mm';
  O.y.textContent = b.y0.toFixed(2) + ' to ' + b.y1.toFixed(2) + ' mm';
  O.z.textContent = b.z0.toFixed(2) + ' to ' + b.z1.toFixed(2) + ' mm';
  cnt.textContent = r[1].toLocaleString() + ' of ' + nS.toLocaleString()
                  + ' struts, ' + r[2].toLocaleString() + ' junctions shown';
}
function schedule() { if (!pending) { pending = true; requestAnimationFrame(apply); } }
Object.values(S).forEach(function (el) { el.addEventListener('input', schedule); });

function whole() {
  S.x0.value = LIM.x0; S.x1.value = LIM.x1;
  S.y0.value = LIM.y0; S.y1.value = LIM.y1;
  S.z0.value = LIM.z0; S.z1.value = LIM.z1;
}
document.getElementById('bslab').addEventListener('click', function () {
  whole();
  const c = (LIM.z0 + LIM.z1) / 2, h = (LIM.z1 - LIM.z0) / 20;
  S.z0.value = c - h; S.z1.value = c + h; apply();
});
bnodes.addEventListener('click', function () {
  showNodes = !showNodes; bnodes.classList.toggle('on', showNodes); apply();
});
bsusp.addEventListener('click', function () {
  suspOnly = !suspOnly; bsusp.classList.toggle('on', suspOnly);
  if (suspOnly && !showNodes) { showNodes = true; bnodes.classList.add('on'); }
  apply();
});
document.getElementById('breset').addEventListener('click', function () {
  whole(); showNodes = false; suspOnly = false;
  bnodes.classList.remove('on'); bsusp.classList.remove('on'); apply();
});
apply();
</script>
</body>
</html>
"""


def write_viewer(info, title, subtitle, out_html, voxel_um, open_browser=True):
    P_mm = info["P"] * voxel_um / 1000.0
    dia = np.nan_to_num(info["diam_um"], nan=0.0)
    lim = {"x0": float(P_mm[:, 0].min()), "x1": float(P_mm[:, 0].max()),
           "y0": float(P_mm[:, 1].min()), "y1": float(P_mm[:, 1].max()),
           "z0": float(P_mm[:, 2].min()), "z1": float(P_mm[:, 2].max())}
    data = {"px": np.round(P_mm[:, 0], 4).tolist(),
            "py": np.round(P_mm[:, 1], 4).tolist(),
            "pz": np.round(P_mm[:, 2], 4).tolist(),
            "a": info["a"].tolist(), "b": info["b"].tolist(),
            "dia": np.round(dia, 1).tolist(),
            "deg": info["deg"].tolist(), "modal": int(info["modal"]),
            "lim": lim,
            "dlo": float(np.percentile(dia, 5)) if dia.size else 0.0,
            "dhi": float(np.percentile(dia, 95)) if dia.size else 1.0}

    html = _PAGE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle)
    html = html.replace("__DATA__", json.dumps(data))
    for ax in "xyz":
        lo, hi = lim[ax + "0"], lim[ax + "1"]
        html = (html.replace(f"__{ax.upper()}MIN__", f"{lo:.4f}")
                    .replace(f"__{ax.upper()}MAX__", f"{hi:.4f}")
                    .replace(f"__{ax.upper()}STEP__", f"{max((hi - lo) / 200, 1e-4):.5f}"))

    out_html = os.path.abspath(out_html)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    Path(out_html).write_text(html, encoding="utf-8")
    print(f"[save] {out_html}  ({os.path.getsize(out_html) / 1e6:.1f} MB)")
    if open_browser:
        try:
            webbrowser.open(Path(out_html).as_uri())
        except Exception:
            pass
    return out_html


# ----------------------------------------------------------------------
# 3. Entry point
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="View a lattice graph JSON in Chrome.")
    ap.add_argument("json", nargs="?", default=None,
                    help="a .json file, or a folder holding several")
    ap.add_argument("--which", type=int, default=DEFAULT_WHICH,
                    help="index into the folder listing")
    ap.add_argument("--all", action="store_true", default=DEFAULT_ALL,
                    help="write one page per file in the folder")
    ap.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_UM)
    ap.add_argument("--html", default=DEFAULT_OUT_HTML)
    ap.add_argument("--repaired", default=None,
                    help="also write the repaired JSON to this path")
    args = ap.parse_args(argv)

    print(f"[version] graph_viewer {SCRIPT_VERSION}")
    path = args.json or DEFAULT_JSON
    files = list_json_files(path)

    if args.all:
        targets = files
    else:
        which = min(max(args.which, 0), len(files) - 1)
        if which != args.which:
            print(f"[list] index {args.which} is out of range, using {which}")
        targets = [files[which]]

    here = Path(__file__).resolve().parent
    for f in targets:
        print("-" * 68)
        print(f"[read] {f.name}")
        data, bad = load_graph_json(f)
        if args.repaired and bad is not None:
            Path(args.repaired).write_text(json.dumps(data), encoding="utf-8")
            print(f"[save] repaired copy at {args.repaired}")

        info = analyse(data, args.voxel)
        sub = (f"{len(info['a']):,} struts and {len(info['P']):,} junctions. "
               f"Struts are coloured by diameter. Use the paired sliders to cut "
               f"a slab; a whole lattice is opaque from outside. Junctions with "
               f"fewer members than the modal degree of {info['modal']} are "
               f"drawn in red, which is where a strut may be absent, though "
               f"truncation at the specimen boundary lowers the degree there "
               f"as well.")

        out = args.html if len(targets) == 1 else f"{f.stem}.html"
        if not os.path.isabs(out):
            out = str(here / out)
        # Only the last page is opened when a whole folder is processed, so
        # that a folder of twenty files does not open twenty browser tabs.
        write_viewer(info, f.stem, sub, out, args.voxel,
                     open_browser=(f is targets[-1]))
    print("[done]")


if __name__ == "__main__":
    main()
