# Results: specimen 210127 0.5 % dash-1, unified DSC input (Pattern Recognition)

**Detected:** **87 missing struts** and **74 broken struts**.

| Output | File |
|---|---|
| Counts | `summary.json` |
| 3D view of defect locations | `defects_3d.png`, `defects_3d.html` (rotate, hover for defect ids) |
| Marked slices | `marked_slices/` (114 slices, 6×6 grid). Slideshow: `python npy_viewer.py results/0point5dash1/marked_slices_viewer_rgb.npy` |
| Defect list | `defects.csv` (one row per flagged strut: class, % absent, slice range, grid cell) |
| Defect map | `defects_map.json` (same layout as the repo's lattice JSONs; each strut links to its marked slice) |
| Visual review | `review.md` |
| Execution trace | `trace.jsonl` |
| Scores | `score.json`, `template_numbers.md` |

## Input
- Detector: `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif` only. The STL and JSON are never read.
- Evaluation only: the same CT plus `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.ct_registered.json`.
  The print is treated as a 0 % defect design (`0.stl`), so every strut the CT shows ≥ 85 % absent is missing and every
  strut with no connected material between its two nodes is broken. Reference: `reference/0point5dash1/`.

## Accuracy (strut level)
| Scope | Precision | Recall | F1 | Missing found | Broken found | False alarms |
|---|---|---|---|---|---|---|
| **y-max face excluded (headline)** | **0.944** | **0.784** | **0.856** | **86 / 90** | **64 / 104** | **9** |
| All 18,468 struts | 0.944 | 0.131 | 0.230 | 86 / 414 | 64 / 746 | 9 |

- **Why two scopes:** the specimen's y-max face layer (on the build-plate side) is absent in the CT.
  That accounts for 966 of the 1,160 defective struts when every strut is scored (324 missing, 642 cut off where they
  meet it). The technique compares each strut with its repeating neighbours, and there are none beyond that face, so it
  does not flag it. The headline excludes the 1,008 struts touching that face.
- **Spread across regions:** F1 over the five 152-slice z-blocks is 0.866 ± 0.039 (0.850, 0.871, 0.800, 0.909, 0.900).
- **Determinism:** three repeats gave identical output. Runtime 134.9 ± 1.4 s (2 vCPU, 8 GB, no GPU).
- **Segmentation vs. the voxelised as-built design:** IoU 0.333 / Dice 0.499. A proxy, because no voxel-level ground
  truth exists, and as-built struts are thicker and rougher than the 350 µm design.

## What is missed
- **42 defective struts not flagged:** 3 missing, 39 broken. 32 of them are on an outer face of the lattice, where a
  strut has few neighbours to compare with.
- **9 false alarms,** all flagged broken on struts the reference calls sound.
- **2 with the other class:** M021 is broken in the reference, B072 is missing.
- **Hairline breaks of 1-2 voxels** are hidden by the 2-voxel presence tolerance.
