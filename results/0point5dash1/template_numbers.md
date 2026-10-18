# Template numbers: T2 Pattern Recognition, specimen 210127 0.5 % dash-1 (unified DSC input)

Input: `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif` (detector).
Evaluation reference: CT + `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.ct_registered.json` (0 % design; every defect visible in the CT counts).

## Detection quality (strut level, y-max build-plate face excluded)
| Metric | Value | ± std (5 z-blocks) |
|---|---|---|
| Precision | 0.944 | 0.035 |
| Recall | 0.784 | 0.062 |
| F1 | 0.856 | 0.039 (z-block mean 0.866) |
| IoU (vs voxelised as-built design, proxy) | 0.333 | N/A |
| Dice (same proxy) | 0.499 | N/A |
| Edge F1 | N/A | not a graph technique |

TP 152, FP 9, FN 42. Reference: 90 missing, 104 broken, 17,266 sound (17,460 struts scored).

All 18,468 struts scored (y-max face included): precision 0.944, recall 0.131, F1 0.230.
The y-max face layer is absent in the specimen (324 missing + 642 broken struts there); the technique does not flag it.

## Recall by class
| Class | Recall | In reference | Found (correct class) |
|---|---|---|---|
| Missing | 0.956 | 90 | 86 (87 flagged missing) |
| Broken | 0.615 | 104 | 64 (74 flagged broken) |
| Thin / Bent / Dross | N/A | N/A | not targeted |

## Cost
| Metric | Value |
|---|---|
| Runtime (pipeline) | 134.9 ± 1.4 s (3 repeats, identical output) |
| Hardware | cloud Linux container, 2 vCPU, 8 GB RAM, no GPU |
| Token usage | 21,844k (algorithm run 8,053k + report update 13,790k) |
| Tool calls | 88 (43 + 45) |
| Cost | $6.28 ($2.52 + $3.76), Claude Opus 5.5 rates |
| Cost efficiency (F1 / USD) | 0.136 |

Tokens, tool calls and cost were measured from the session transcript, counting only the algorithm run (detection,
evaluation, slice review) and the report update. Over 99 % of the tokens are cache re-reads of a ~410k-token context
carried over from earlier work in the same session, so a fresh session would use far fewer.
