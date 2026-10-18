"""Periodic-Neighbour Voting (PNV): self-referential missing / broken strut detection.

The technique never looks at the design (STL / JSON). It uses only the periodicity of
the lattice in the CT scan itself:

1. local contrast normalisation (removes the brightness gradient across the scan)
2. segmentation (Otsu on the normalised volume)
3. lattice vectors from the 3-D autocorrelation (captures tilt and anisotropic spacing)
4. specimen envelope (so neighbours outside the part abstain instead of voting "absent")
5. KNN-style vote: every voxel is voted on by its 18 lattice-equivalent neighbours
   (12 FCC half-cell + 6 cubic vectors). Material that most neighbours have but the
   voxel lacks is a defect candidate.
6. candidates are linked in 3-D (persistence through the octet) and measured
7. the 85 % rule: a void spanning >= 85 % of the free strut length is a MISSING strut,
   a shorter void that cuts the strut is a BROKEN strut.
8. a false-alarm filter (rules or a gradient-boosted classifier) removes voids that are
   really bent / displaced / thin-but-present struts.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

MISSING_FRACTION = 0.85          # >= 85 % of the strut absent -> missing
MIN_BROKEN_FRACTION = 0.08       # shorter voids are ignored
MIN_CANDIDATE_VOXELS = 40
PRESENCE_TOLERANCE = 2           # voxels: material this close to the prediction counts as present
RULE_MIN_BROKEN_VOXELS = 160     # rule filter: a break must be a real void ...
RULE_MAX_VOID_INTENSITY = 0.12   # ... that is (nearly) empty
EDGE_MAX_WIDTH = 12.0            # blobs wider than this hugging the envelope are edge artefacts
EDGE_DISTANCE = 8.0

# ------------------------------------------------------------------ 1-2
def local_normalize(v: np.ndarray, f: int = 4, win: int = 24) -> np.ndarray:
    """(v - local background) / (local max - local background), computed on a coarse grid."""
    s = v[::f, ::f, ::f].astype(np.float32)
    bg = ndimage.uniform_filter(ndimage.grey_erosion(s, size=(win // 2,) * 3), size=win // 2)
    hi = ndimage.uniform_filter(ndimage.maximum_filter(s, size=win), size=win // 2)
    c = hi - bg
    floor = 0.5 * float(np.median(c[c > np.percentile(c, 50)]))   # do not stretch empty regions into noise
    out = np.empty(v.shape, np.float16)
    Z, Y, X = v.shape
    for z in range(Z):
        cz = min(z / f, s.shape[0] - 1); z0 = int(cz); z1 = min(z0 + 1, s.shape[0] - 1); w = cz - z0
        b2 = ndimage.zoom((1 - w) * bg[z0] + w * bg[z1], (Y / bg.shape[1], X / bg.shape[2]), order=1)
        h2 = ndimage.zoom((1 - w) * hi[z0] + w * hi[z1], (Y / hi.shape[1], X / hi.shape[2]), order=1)
        out[z] = np.clip((v[z].astype(np.float32) - b2) / np.maximum(h2 - b2, floor), 0, 1.5)
    return out

def segment(norm: np.ndarray) -> tuple[np.ndarray, float]:
    t = float(threshold_otsu(norm[::3, ::3, ::3].astype(np.float32).ravel()))
    seg = np.empty(norm.shape, bool)
    for z in range(norm.shape[0]):
        seg[z] = ndimage.binary_opening(norm[z] > t, structure=np.ones((2, 2)))
    return seg, t

# ------------------------------------------------------------------ 3
def _autocorr_peaks(seg, ds):
    s = seg[::ds, ::ds, ::ds].astype(np.float32); s -= s.mean()
    F = np.fft.rfftn(s); ac = np.fft.fftshift(np.fft.irfftn(F * np.conj(F), s=s.shape)); del F
    ac /= ac.max(); c = np.array(ac.shape) // 2
    mx = ndimage.maximum_filter(ac, size=5)
    pk = np.argwhere((ac == mx) & (ac > 0.05))
    d = (pk - c) * float(ds); r = np.linalg.norm(d, axis=1)
    keep = r > 4 * ds
    return d[keep], ac[tuple(pk[keep].T)]

def lattice_basis(seg: np.ndarray, ds: int | None = None) -> np.ndarray:
    """3x3 matrix H whose columns are the half-cell lattice directions (z,y,x voxels).
    FCC nearest-neighbour vectors are H @ (+-1,+-1,0) and permutations."""
    if ds is None:
        ds = 4 if min(seg.shape) > 300 else 1
    d, val = _autocorr_peaks(seg, ds)
    r = np.linalg.norm(d, axis=1)
    near = r < 1.6 * r[np.argmax(val)]            # only the first shells around the origin
    d, val, r = d[near], val[near], r[near]
    # the FCC nearest-neighbour shell = the smallest-radius shell holding >= 8 strong peaks
    strong = val >= 0.3 * val.max()
    best = None
    for ri in np.sort(r[strong]):
        grp = strong & (np.abs(r - ri) < 0.06 * ri)
        if grp.sum() >= 8:
            best = (None, grp); break
    if best is None:
        raise RuntimeError("could not find the FCC autocorrelation shell; is this an octet lattice?")
    fcc = d[best[1]]
    a_half = float(np.median(np.linalg.norm(fcc, axis=1))) / np.sqrt(2)
    H = np.eye(3) * a_half          # assumes lattice axes roughly aligned with scan axes (< ~15 deg)
    for _ in range(5):
        n = np.round(np.linalg.solve(H, fcc.T).T)
        ok = np.all(np.isin(np.abs(n), [0, 1]), 1) & (np.abs(n).sum(1) == 2)
        H = np.linalg.lstsq(n[ok], fcc[ok], rcond=None)[0].T
    return _refine_basis(seg, H)

def _refine_basis(seg, H):
    """Integer/sub-voxel refinement of the three FCC basis vectors by self-overlap."""
    vecs = [np.round(H @ np.array(n, float)).astype(int) for n in [(1, 1, 0), (1, 0, 1), (0, 1, 1)]]
    m = int(max(np.abs(v).max() for v in vecs)) + 3
    half = min(150 + m, min(seg.shape) // 2 - 1)
    c = np.array(seg.shape) // 2
    sub = seg[c[0] - half:c[0] + half, c[1] - half:c[1] + half, c[2] - half:c[2] + half]
    if min(sub.shape) <= 2 * m + 4:
        return H
    a = sub[m:-m, m:-m, m:-m]
    best = []
    for v0 in vecs:
        sc = {}
        for dd in itertools.product(range(-2, 3), repeat=3):
            v = v0 + np.array(dd)
            b = sub[m + v[0]:sub.shape[0] - m + v[0], m + v[1]:sub.shape[1] - m + v[1], m + v[2]:sub.shape[2] - m + v[2]]
            sc[tuple(v)] = int((a & b).sum())
        vb = np.array(max(sc, key=sc.get)); ref = vb.astype(float)
        for ax in range(3):
            e = np.zeros(3, int); e[ax] = 1
            f0, fm, fp = sc[tuple(vb)], sc.get(tuple(vb - e)), sc.get(tuple(vb + e))
            if fm is not None and fp is not None and (fm - 2 * f0 + fp) != 0:
                ref[ax] += 0.5 * (fm - fp) / (fm - 2 * f0 + fp)
        best.append(ref)
    B = np.array(best).T
    N = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], float).T
    return B @ np.linalg.inv(N)

def neighbour_vectors(H: np.ndarray) -> list[np.ndarray]:
    fcc = [p for p in itertools.product([-1, 0, 1], repeat=3) if sum(map(abs, p)) == 2]
    cub = [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)]
    return [H @ np.array(n, float) for n in fcc + cub]

# ------------------------------------------------------------------ 4
def envelope(seg: np.ndarray, a_half: float, f: int = 4) -> np.ndarray:
    """Specimen envelope: closing of the segmentation with a cube of half-width ~0.6 half cells."""
    f = max(1, min(f, int(a_half // 8)))
    s = seg[::f, ::f, ::f]
    r = max(2, int(round(a_half * 0.6 / f)))
    box = np.ones((2 * r + 1,) * 3, bool)
    e = ndimage.binary_closing(np.pad(s, r + 1), structure=box)[r + 1:-r - 1, r + 1:-r - 1, r + 1:-r - 1]
    e = ndimage.binary_fill_holes(ndimage.binary_erosion(ndimage.binary_fill_holes(e), iterations=1) | s)
    out = np.repeat(np.repeat(np.repeat(e, f, 0), f, 1), f, 2)
    return np.ascontiguousarray(out[:seg.shape[0], :seg.shape[1], :seg.shape[2]])

# ------------------------------------------------------------------ 5
def _shift_chunk(a, v, z0, z1):
    dz, dy, dx = v; Z, Y, X = a.shape; out = np.zeros((z1 - z0, Y, X), bool)
    zs0, zs1 = max(z0 + dz, 0), min(z1 + dz, Z)
    if zs0 >= zs1:
        return out
    sy = slice(max(0, -dy), Y - max(0, dy)); ty = slice(max(0, dy), Y - max(0, -dy))
    sx = slice(max(0, -dx), X - max(0, dx)); tx = slice(max(0, dx), X - max(0, -dx))
    out[zs0 - dz - z0:zs1 - dz - z0, sy, sx] = a[zs0:zs1, ty, tx]
    return out

def vote(seg, env, vecs, min_valid=3, chunk=48):
    """Periodic-neighbour (KNN) vote. Neighbours outside the envelope abstain (edge handling).
    expected = at least half of the valid neighbours (and >= 2) have material here."""
    iv = [tuple(int(round(x)) for x in v) for v in vecs]
    Z = seg.shape[0]
    votes = np.zeros(seg.shape, np.uint8); valid = np.zeros(seg.shape, np.uint8); expected = np.zeros(seg.shape, bool)
    for z0 in range(0, Z, chunk):
        z1 = min(Z, z0 + chunk); vo = votes[z0:z1]; va = valid[z0:z1]
        for v in iv:
            e = _shift_chunk(env, v, z0, z1); va += e
            e &= _shift_chunk(seg, v, z0, z1); vo += e
        expected[z0:z1] = (va >= min_valid) & (vo >= 2) & (vo.astype(np.uint16) * 2 >= va) & env[z0:z1]
    return expected, votes, valid

# ------------------------------------------------------------------ 6
def candidates(seg, expected, votes, valid, min_vox=MIN_CANDIDATE_VOXELS, tol=PRESENCE_TOLERANCE):
    miss = ndimage.binary_erosion(expected, iterations=1)
    miss &= ~ndimage.binary_dilation(seg, iterations=tol)
    lab, n = ndimage.label(miss, structure=np.ones((3, 3, 3)), output=np.int32); del miss
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i; nv = int(m.sum())
        if nv < min_vox:
            lab[sl][m] = 0
            continue
        pts = np.argwhere(m) + [s.start for s in sl]
        c = pts.mean(0); X = pts - c
        w, V = np.linalg.eigh(X.T @ X / len(X)); ax = V[:, -1]
        proj = X @ ax; length = float(proj.max() - proj.min() + 1)
        perp = np.linalg.norm(X - np.outer(proj, ax), axis=1)
        vv = np.asarray(votes[sl])[m].astype(float); va = np.asarray(valid[sl])[m].astype(float)
        comps.append(dict(id=i, n_vox=nv, centroid_zyx=c.tolist(), axis_zyx=ax.tolist(),
                          end0_zyx=(c + proj.min() * ax).tolist(), end1_zyx=(c + proj.max() * ax).tolist(),
                          length=length, width=float(2 * np.percentile(perp, 90)),
                          vote_frac=float((vv / va).mean()), valid_mean=float(va.mean()),
                          bbox=[[s.start, s.stop] for s in sl]))
    return lab, comps

def strut_geometry(H, expected, crop=160):
    """Node-to-node strut length and node diameter, both measured from the scan.
    Node diameter: blobs that survive an opening with a ball 1.6x the strut radius (struts vanish, nodes stay)."""
    L_node = float(np.mean([np.linalg.norm(H @ np.array(n, float)) for n in [(1, 1, 0), (1, 0, 1), (0, 1, 1)]]))
    c = np.array(expected.shape) // 2; h = np.minimum(c, crop // 2)
    sub = np.asarray(expected[c[0] - h[0]:c[0] + h[0], c[1] - h[1]:c[1] + h[1], c[2] - h[2]:c[2] + h[2]])
    dt = ndimage.distance_transform_edt(sub)
    r_strut = float(np.median(dt[sub & (dt >= 1)][::7]))            # typical inscribed radius (struts dominate the volume)
    r = max(2, int(np.ceil(1.6 * r_strut)))
    zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    opened = ndimage.binary_opening(sub, structure=(zz ** 2 + yy ** 2 + xx ** 2) <= r * r)
    lab, n = ndimage.label(opened)
    if n == 0:
        return L_node, 0.0
    sizes = np.asarray(ndimage.sum(opened, lab, range(1, n + 1)))
    sizes = sizes[sizes > 0.25 * np.median(sizes)]
    d_node = float(2 * np.median((sizes * 3 / (4 * np.pi)) ** (1 / 3)))
    return L_node, d_node

# ------------------------------------------------------------------ 7-8
def features(comps, lab, seg, expected, norm, env):
    out = []
    for c in comps:
        (z0, z1), (y0, y1), (x0, x1) = c['bbox']; p = 6
        sl = (slice(max(z0 - p, 0), z1 + p), slice(max(y0 - p, 0), y1 + p), slice(max(x0 - p, 0), x1 + p))
        m = np.asarray(lab[sl]) == c['id']
        sh = ndimage.binary_dilation(m, iterations=5) & ~ndimage.binary_dilation(m, iterations=1)
        sg = np.asarray(seg[sl]); ex = np.asarray(expected[sl]); nm = np.asarray(norm[sl]).astype(np.float32)
        r = max(c['width'] / 2, 0.5); ax = np.abs(np.array(c['axis_zyx']))
        out.append(dict(
            n_vox=c['n_vox'], length=c['length'], width=c['width'], absent_fraction=c['absent_fraction'],
            vote_frac=c['vote_frac'], valid_mean=c['valid_mean'],
            displaced=float((sg & ~ex & sh).sum() / max(sh.sum(), 1)),
            shell_present=float((sg & sh).sum() / max(sh.sum(), 1)),
            inten_in=float(nm[m].mean()), inten_p95=float(np.percentile(nm[m], 95)),
            dist_env=float(ndimage.distance_transform_edt(np.asarray(env[sl]))[m].mean()),
            fill=c['n_vox'] / max(c['length'] * np.pi * r * r, 1), axis_z=float(ax[0])))
    return out

FEATURES = ['n_vox', 'length', 'width', 'absent_fraction', 'vote_frac', 'valid_mean', 'displaced',
            'shell_present', 'inten_in', 'inten_p95', 'dist_env', 'fill', 'axis_z']

def rule_85(comps, L_free):
    """The 85 % rule (persistence of the void along the strut)."""
    for c in comps:
        f = c['length'] / L_free; c['absent_fraction'] = float(min(f, 1.5))
        if c['width'] < 3.0 or f < MIN_BROKEN_FRACTION:
            c['cls_85'] = 0
        elif f >= MISSING_FRACTION:
            c['cls_85'] = 1
        else:
            c['cls_85'] = 2
    return comps

def filter_rules(comps, feats):
    """Rule-based false-alarm filter. Returns {candidate id: 0 none / 1 missing / 2 broken}."""
    out = {}
    for c, f in zip(comps, feats):
        k = c['cls_85']
        if k == 2 and (f['n_vox'] < RULE_MIN_BROKEN_VOXELS or f['inten_in'] > RULE_MAX_VOID_INTENSITY):
            k = 0
        if k > 0 and f['width'] > EDGE_MAX_WIDTH and f['dist_env'] < EDGE_DISTANCE:
            k = 0
        out[c['id']] = k
    return out

def filter_gbm(comps, feats, model):
    """Gradient-boosted false-alarm filter: the model only decides defect / not-defect;
    the 85 % rule still decides missing vs broken."""
    if not comps:
        return {}
    X = np.array([[f[k] for k in FEATURES] for f in feats])
    keep = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    out = {}
    for c, k, p in zip(comps, keep, proba):
        out[c['id']] = c['cls_85'] if k == 1 else 0
        c['defect_probability'] = float(p)
    return out

# ------------------------------------------------------------------ 9. self-derived strut list
FCC_STEPS = [n for n in itertools.product([-1, 0, 1], repeat=3) if sum(map(abs, n)) == 2 and n > (0, 0, 0)]

def lattice_graph(H, seg, env, d_node, snap=8, return_grid=False):
    """Node lattice rebuilt from the scan's own periodicity: one node found by blob search, the rest
    generated with the measured lattice vectors (FCC: half-cell coords with even sum) and snapped to the
    local material centroid. No design file is used. Returns node positions (z,y,x) and strut index pairs
    (plus each node's integer half-cell lattice coordinates (z,y,x) when return_grid=True)."""
    Z, Y, X = seg.shape; c = np.array(seg.shape) // 2; h = int(np.abs(H).max() * 1.2)
    crop = np.asarray(seg[c[0] - h:c[0] + h, c[1] - h:c[1] + h, c[2] - h:c[2] + h], np.float32)
    g = ndimage.gaussian_filter(crop, sigma=max(d_node / 4, 1.5))
    o = np.array(np.unravel_index(np.argmax(g), g.shape)) + c - h
    Hi = np.linalg.inv(H)
    corners = np.array(list(itertools.product([0, Z], [0, Y], [0, X]))) - o
    lc = corners @ Hi.T; lo = np.floor(lc.min(0)).astype(int) - 1; hi = np.ceil(lc.max(0)).astype(int) + 1
    grid = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    grid = grid[grid.sum(1) % 2 == 0]
    pos = o + grid @ H.T
    inside = np.all((pos >= snap) & (pos < np.array(seg.shape) - snap), 1)
    grid, pos = grid[inside], pos[inside]
    ip = np.round(pos).astype(int); keep = np.asarray(env)[ip[:, 0], ip[:, 1], ip[:, 2]]
    grid, pos = grid[keep], pos[keep]
    # local deformation: refit the lattice to the node blobs (affine in lattice coords), then snap iteratively
    w = np.mgrid[-snap:snap + 1, -snap:snap + 1, -snap:snap + 1].reshape(3, -1).T
    lim = np.array(seg.shape) - snap - 1
    for it in range(4):
        new = pos.copy(); ok = np.zeros(len(pos), bool)
        for i, p in enumerate(np.clip(np.round(pos).astype(int), snap, lim)):
            b = np.asarray(seg[p[0] - snap:p[0] + snap + 1, p[1] - snap:p[1] + snap + 1, p[2] - snap:p[2] + snap + 1])
            if b.sum() >= 10:
                e = ndimage.distance_transform_edt(b).reshape(-1) ** 4      # node centre = thickest point
                new[i] = p + (w * e[:, None]).sum(0) / e.sum(); ok[i] = True
        if it < 2:   # global affine refit keeps nodes coherent where blobs are weak
            G = np.c_[grid, np.ones(len(grid))]
            M = np.linalg.lstsq(G[ok], new[ok], rcond=None)[0]; pos = G @ M
        else:
            pos = np.where((ok & (np.linalg.norm(new - pos, axis=1) < snap))[:, None], new, pos)
    index = {tuple(n): i for i, n in enumerate(grid)}
    struts = [(i, index[tuple(n + np.array(s))]) for i, n in enumerate(grid) for s in FCC_STEPS if tuple(n + np.array(s)) in index]
    struts = np.array(struts, int).reshape(-1, 2)
    if not return_grid:
        return pos, struts
    # express the half-cell coords along the scan axes (z, y, x), each increasing with position: the measured
    # lattice vectors can come out in any order or sign
    P = np.zeros((3, 3), int)
    for k in range(3):
        ax = int(np.argmax(np.abs(H[:, k]))); P[ax, k] = int(np.sign(H[ax, k]))
    assert sorted(np.abs(P).argmax(0).tolist()) == [0, 1, 2], "lattice vectors not axis-aligned"
    return pos, struts, grid @ P.T

def lattice_box(grid, pos, seg, r=2, min_plane_frac=0.25):
    """Corner and size (in unit cells) of the lattice block, from the scan alone, so unit cells can be numbered
    from the corner like the repo's lattice JSONs. `grid` = half-cell node coords (z,y,x).
    Per axis: planes holding >= 25 % of the median plane count are the well-supported body. A whole octet block spans
    an even number of half-steps, so an odd span is extended by one plane on the better-supported side
    (nodes x material at the node); the corner must also be a lattice node (even coordinate sum).
    Returns (corner_grid_zyx, n_cells_zyx)."""
    grid = np.asarray(grid); ip = np.round(pos).astype(int)
    mat = np.array([np.asarray(seg[max(p[0] - r, 0):p[0] + r + 1, max(p[1] - r, 0):p[1] + r + 1, max(p[2] - r, 0):p[2] + r + 1]).mean()
                    for p in ip])
    opts = []
    for ax in range(3):
        v, c = np.unique(grid[:, ax], return_counts=True); body = v[c >= min_plane_frac * np.median(c)]
        b, B = int(body.min()), int(body.max())
        def support(plane):
            sel = grid[:, ax] == plane
            return float(sel.sum() * mat[sel].mean()) if sel.any() else 0.0
        opts.append([(b, B)] if (B - b) % 2 == 0 else [(b - 1, B), (b, B + 1)])
    best = None
    for lz in opts[0]:
        for ly in opts[1]:
            for lx in opts[2]:
                lo = np.array([lz[0], ly[0], lx[0]]); hi = np.array([lz[1], ly[1], lx[1]])
                if lo.sum() % 2:
                    continue
                score = 0.0
                for ax, (l, o) in enumerate(zip((lz, ly, lx), opts)):
                    if len(o) > 1:   # credit the extra plane this option adds
                        extra = l[0] if l[0] < o[1][0] else l[1]
                        sel = grid[:, ax] == extra; score += float(sel.sum() * mat[sel].mean()) if sel.any() else 0.0
                if best is None or score > best[0]:
                    best = (score, lo, hi)
    _, lo, hi = best
    return lo.tolist(), ((hi - lo) // 2).tolist()

def assign_struts(nodes, struts, lab, classes, L_node, d_node, n_samples=25, min_hits=2):
    """Per self-derived strut: fraction of its free length (node surfaces excluded) covered by accepted voids.
    >= 85 % covered -> missing, otherwise (at least `min_hits` samples) -> broken."""
    t0 = min(0.45, 0.5 * d_node / L_node)
    ts = np.linspace(t0, 1 - t0, n_samples)
    A, B = nodes[struts[:, 0]], nodes[struts[:, 1]]
    lut = np.zeros(int(max(classes) if classes else 0) + 2, np.int32)
    acc = np.zeros(lut.shape, bool)
    for cid, k in classes.items():
        if k > 0 and cid < len(acc):
            acc[cid] = True
    hits = np.zeros((len(struts), n_samples), np.int32)
    offs = np.array([o for o in itertools.product(range(-3, 4), repeat=3) if sum(x * x for x in o) <= 9])   # ~ strut radius
    shp = np.array(lab.shape)
    for j, t in enumerate(ts):
        p = np.round(A + t * (B - A)).astype(int)
        for o in offs:
            q = np.clip(p + o, 0, shp - 1)
            v = np.asarray(lab[q[:, 0], q[:, 1], q[:, 2]]) if not isinstance(lab, np.memmap) else lab[q[:, 0], q[:, 1], q[:, 2]]
            v = np.where(v < len(acc), v, 0); ok = acc[v] & (hits[:, j] == 0)
            hits[ok, j] = v[ok]
    cover = (hits > 0).mean(1)
    out = []
    for s in np.where((hits > 0).sum(1) >= min_hits)[0]:
        ids = sorted(set(int(x) for x in hits[s] if x > 0))
        out.append(dict(strut=int(s), node0_zyx=A[s].tolist(), node1_zyx=B[s].tolist(), absent_fraction=float(cover[s]),
                        cls=1 if cover[s] >= MISSING_FRACTION else 2, voids=ids,
                        hit_t=[float(t) for t, h in zip(ts, hits[s]) if h > 0]))
    return out
