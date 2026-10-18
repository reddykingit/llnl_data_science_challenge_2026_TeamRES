# Team RES · DSC 2026: Pattern Recognition strut-defect technique

This repo holds one technique for the **Agentic Strut-Defect Detection Comparison**: finding
**missing** and **broken** struts in X-ray CT scans of AM octet-truss lattices. It uses
*Pattern Recognition*, a self-referential method that never uses the design.

## Fair-comparison scope (do not widen)
The technique does only three things:
1. find missing struts (>= 85 % of the strut absent)
2. find broken struts (a gap that cuts the strut, < 85 % absent)
3. output the counts, a 3-D view of their locations, and marked slices

Everything else (segmentation tools, skeletonisation, octet-layer length, deformation analysis,
the old Codex skill) is in `legacy/`. It is **not** exposed through MCP and must not be used.

## Layout
- `src/strut_defects/`: the detector (`pnv.py` core, `pipeline.py` CLI, `render.py` outputs). Never reads STL or JSON.
- `src/evaluation/`: reference building and scoring. Uses the STL and the registered JSON. Evaluation only.
- `src/mcp_server.py`: MCP tools (`.mcp.json` registers it as `strut-defects`).
- `.claude/agents/`: subagents `strut-defect-detector`, `defect-slice-reviewer`, `technique-reporter`.
- `.claude/skills/strut-defect-run/`: the end-to-end procedure.
- `models/pnv_gbm_training_set.csv`: candidate features + labels for the GBM false-alarm filter.
- `results/<specimen>/`: outputs. `reference/<specimen>/`: evaluation reference.
- `tests/`: `python -m unittest discover tests` (a synthetic octet lattice with one missing and one broken strut).

## Run
```
python -m src.strut_defects.pipeline run --input "data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" --out results/0point5dash1
```
It needs about 4 GB of RAM and takes ~3-4 min on a laptop CPU. In Claude Code, use the `strut-defect-run` skill.

## Conventions
- Volumes are (z, y, x) with z = slice index. Design JSON positions are (x, y, z).
- The GBM filter is leave-region-out on the training specimen: a candidate is never classified by a
  model that saw its own z-block. Keep it that way when adding specimens.
