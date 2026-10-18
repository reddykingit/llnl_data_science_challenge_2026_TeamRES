---
name: strut-defect-run
description: Full fair-comparison run of the Pattern Recognition missing/broken strut technique on one specimen - detect, visually review, score, and collect the reporting-template numbers. Use when asked to run the technique, reproduce the results, or fill the per-technique template.
---
# Pattern Recognition strut-defect run

The technique (Pattern Recognition) finds only two classes:
- **missing strut**: >= 85 % of the strut does not exist
- **broken strut**: a gap cuts the strut but < 85 % of it is absent

Outputs: counts, a 3-D view of defect locations, and marked slices.

## Inputs (unified DSC set, in `data/`)
- CT scan (detector input): `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
- Registered JSON (evaluation only): `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.ct_registered.json`
- `0.stl`: the 0 % defect design. Not needed by either step; the registered JSON is the same design as a graph.

## Steps
1. **Detect.** Delegate to the `strut-defect-detector` subagent with the CT path and
   `results/<specimen>` as the output folder.
2. **Review.** Delegate to the `defect-slice-reviewer` subagent on the same folder. Its verdicts are
   informational. Do not edit detections by hand, because that would invalidate the comparison.
3. **Score.** This step is evaluation only. Delegate to the `technique-reporter` subagent with the
   specimen's reference folder (`reference/<specimen>`), or the CT tif and registered JSON to build it.
4. **Cost.** Run `/cost` in the main session and copy the token count and USD into
   `template_numbers.md`. Count every MCP and tool call made in steps 1 to 3.

## Fair-comparison constraints
- Do not enable or call any other analysis (segmentation-only, skeletonisation, layer length,
  deformation). Those live in `legacy/` and are not exposed.
- The detector must never read the STL or JSON design files. Only `src/evaluation/` may.
- Record any deviation from the frozen model and settings in template Section 7.
