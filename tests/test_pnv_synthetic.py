"""Synthetic octet lattice: one strut removed (missing) and one strut cut (broken).

Run:  python -m pytest tests -q      (or  python -m unittest discover tests)
"""
import itertools, json, tempfile, unittest
from pathlib import Path
import numpy as np
from src.strut_defects import pnv

A = 40          # unit cell (voxels)
N = 4           # cells per side
R = 2.5         # strut radius
M = 14          # margin

def octet_lattice():
    base = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    nodes = np.array([(np.array(b) + c) * A for c in itertools.product(range(N + 1), repeat=3) for b in base], float)
    nodes = nodes[np.all(nodes <= N * A + 1e-6, 1)]
    d = np.linalg.norm(nodes[:, None] - nodes[None], axis=2)
    i, j = np.where(np.triu(np.abs(d - A / np.sqrt(2)) < 1e-6))
    return nodes + M, np.c_[i, j]

def draw(nodes, struts, shape, gap=None):
    vol = np.zeros(shape, np.float32); g = np.indices(shape).reshape(3, -1).T.astype(np.float32)
    for k, (i, j) in enumerate(struts):
        a, b = nodes[i], nodes[j]; lo = np.floor(np.minimum(a, b) - R - 1).astype(int); hi = np.ceil(np.maximum(a, b) + R + 2).astype(int)
        sub = np.indices(hi - lo).reshape(3, -1).T + lo
        d = b - a; t = np.clip(((sub - a) @ d) / (d @ d), 0, 1)
        dist = np.linalg.norm(sub - (a + t[:, None] * d), axis=1)
        keep = dist <= R
        if gap is not None and k == gap[0]:
            keep &= ~((t > gap[1]) & (t < gap[2]))
        p = sub[keep]; vol[p[:, 0], p[:, 1], p[:, 2]] = 1.0
    rr = 6                                            # AM nodes are blobs, like in the CT
    off = np.argwhere(np.ones((2 * rr + 1,) * 3)) - rr; off = off[(off ** 2).sum(1) <= rr * rr]
    for n in nodes:
        q = np.round(n).astype(int) + off; q = q[np.all((q >= 0) & (q < shape), 1)]; vol[q[:, 0], q[:, 1], q[:, 2]] = 1.0
    rng = np.random.default_rng(0)
    return (0.2 + 0.6 * vol + rng.normal(0, 0.03, shape)).astype(np.float32)

class SyntheticOctetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nodes, struts = octet_lattice()
        shape = tuple([N * A + 2 * M] * 3)
        mid = nodes.mean(0)
        dist = np.linalg.norm((nodes[struts[:, 0]] + nodes[struts[:, 1]]) / 2 - mid, axis=1)
        order = np.argsort(dist)
        cls.missing_idx, cls.broken_idx = int(order[0]), int(order[40])
        keep = np.ones(len(struts), bool); keep[cls.missing_idx] = False
        kept = struts[keep]; broken_pos = int(np.where(np.where(keep)[0] == cls.broken_idx)[0][0])
        vol = draw(nodes, kept, shape, gap=(broken_pos, 0.35, 0.65))
        cls.nodes, cls.struts = nodes, struts
        norm = pnv.local_normalize(vol, f=2, win=16)
        seg, _ = pnv.segment(norm)
        H = pnv.lattice_basis(seg)
        cls.H = H
        a_half = float(np.mean(np.linalg.norm(H, axis=0)))
        env = pnv.envelope(seg, a_half)
        exp_, votes, valid = pnv.vote(seg, env, pnv.neighbour_vectors(H))
        lab, comps = pnv.candidates(seg, exp_, votes, valid)
        L_node, d_node = pnv.strut_geometry(H, exp_)
        pnv.rule_85(comps, L_node - d_node)
        cls.comps = comps
        cls.seg, cls.env, cls.d_node = seg, env, d_node

    def _nearest(self, idx):
        a, b = self.nodes[self.struts[idx, 0]], self.nodes[self.struts[idx, 1]]
        best = None
        for c in self.comps:
            q = np.array(c["centroid_zyx"]); d = b - a; t = np.clip((q - a) @ d / (d @ d), 0, 1)
            dist = np.linalg.norm(a + t * d - q)
            if best is None or dist < best[0]:
                best = (dist, c)
        return best

    def test_lattice_vectors_recovered(self):
        self.assertTrue(np.allclose(np.sort(np.abs(np.diag(self.H))), A / 2, atol=1.0))

    def test_removed_strut_is_missing(self):
        dist, c = self._nearest(self.missing_idx)
        self.assertLess(dist, 4.0); self.assertEqual(c["cls_85"], 1)

    def test_cut_strut_is_broken(self):
        dist, c = self._nearest(self.broken_idx)
        self.assertLess(dist, 4.0); self.assertEqual(c["cls_85"], 2)

    def test_no_other_defects(self):
        flagged = [c for c in self.comps if c["cls_85"] > 0]
        self.assertLessEqual(len(flagged), 2, [ (c["centroid_zyx"], c["length"]) for c in flagged])

    def test_lattice_box_and_defects_map(self):
        """Unit cells are numbered from the lattice corner, and defects_map.json follows the repo's lattice JSON layout."""
        from src.strut_defects import render
        pos, struts, grid = pnv.lattice_graph(self.H, self.seg, self.env, self.d_node, return_grid=True)
        corner, n_cells = pnv.lattice_box(grid, pos, self.seg)
        self.assertEqual(n_cells, [N, N, N])
        corner_pos = pos[np.where((grid == corner).all(1))[0][0]]
        self.assertTrue(np.allclose(corner_pos, M, atol=2.0), corner_pos)
        # the removed strut, as the detector's own lattice sees it
        a, b = self.nodes[self.struts[self.missing_idx]]
        cost = [min(np.linalg.norm(pos[i] - a) + np.linalg.norm(pos[j] - b), np.linalg.norm(pos[i] - b) + np.linalg.norm(pos[j] - a))
                for i, j in struts]
        k = int(np.argmin(cost)); i, j = struts[k]
        self.assertLess(cost[k], 4.0)
        f = dict(strut=k, node0_zyx=pos[i].tolist(), node1_zyx=pos[j].tolist(), node0_grid_zyx=grid[i].tolist(),
                 node1_grid_zyx=grid[j].tolist(), mark_zyx=((pos[i] + pos[j]) / 2).tolist(), voids=[1], cls=1)
        row = dict(defect_id="M001", defect_class="missing", absent_fraction_pct=100.0, defect_probability=None,
                   first_slice=0, last_slice=1, marked_slice=int((pos[i][0] + pos[j][0]) / 2), mark_x=0.0, mark_y=0.0,
                   grid_row=0, grid_col=0)
        D = dict(method="none", meta=dict(lattice_corner_grid_zyx=corner, lattice_unit_cells_zyx=n_cells))
        with tempfile.TemporaryDirectory() as tmp:
            render._write_defects_map(Path(tmp), [f], [row], D, self.seg.shape, 1, 0)
            doc = json.load(open(Path(tmp) / "defects_map.json"))
        self.assertEqual(list(doc)[:3], ["junctions", "struts", "unit_cells"])
        for jn in doc["junctions"]:
            self.assertTrue(set(jn["indices"]) <= {0.0, 0.5, 1.0}, jn)
        expected_cell = np.floor((np.minimum(a, b) - M) / A + 1e-6).astype(int)[::-1].tolist()   # x, y, z
        self.assertEqual(doc["unit_cells"][0]["indices"], expected_cell)
        self.assertEqual(doc["unit_cells"][0]["struts"], [k])

if __name__ == "__main__":
    unittest.main()
