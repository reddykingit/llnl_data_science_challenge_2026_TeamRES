# How to run: Team RES, Pattern Recognition (missing / broken struts)

The technique only does three things: find **missing** struts (>= 85 % of the strut absent), find **broken**
struts (< 85 % absent), and output the counts, a 3D view of their locations and marked slices.

## Main file
- **Claude Code:** the `strut-defect-run` skill (`.claude/skills/strut-defect-run/SKILL.md`). It drives three
  subagents (`.claude/agents/`) through the MCP server `src/mcp_server.py`, registered in `.mcp.json` as `strut-defects`.
- **Plain Python:** `src/strut_defects/pipeline.py` (`python -m src.strut_defects.pipeline run ...`).

## Setup (once)
1. Python 3.10+, then `pip install -r requirements.txt` (about 4 GB RAM needed for the run).
2. Put the unified DSC input files in `data/`:
   - `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif` (the only file the detector reads)
   - `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.ct_registered.json` (evaluation only)
   - `data/0.stl` (the 0 % design; not needed by either step)
3. Open the folder in Claude Code and approve the `strut-defects` MCP server. If `python` on your PATH is not the
   environment from step 1, set `command` in `.mcp.json` to that environment's full Python path.

## Prompt (Claude Code)
> Use the strut-defect-run skill on `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif` (specimen 0point5dash1).
> Evaluation reference: the same tif plus `data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.ct_registered.json`.

Then run `/cost` for token usage and cost.

## Run sequence (what the skill does)
1. **Detect:** the `strut-defect-detector` subagent calls `detect_strut_defects` once, writing to `results/0point5dash1/`.
   This takes about 2-3 minutes.
2. **Review:** the `defect-slice-reviewer` subagent looks at the marked slices and writes `review.md`. It is
   informational only and never edits detections.
3. **Score (evaluation only):** the `technique-reporter` subagent calls `build_reference` (tif + registered JSON) and
   `score_against_reference`, and writes `template_numbers.md`.
4. **Cost:** `/cost` in the main session.

## Same run without Claude Code
```
python -m src.strut_defects.pipeline run --input "data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" --out results/0point5dash1
python -m src.evaluation.reference --ct "data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" --registered-json "data/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.ct_registered.json" --out reference/0point5dash1
python -m src.evaluation.score --results results/0point5dash1 --reference reference/0point5dash1
python npy_viewer.py results/0point5dash1/marked_slices_viewer_rgb.npy     # slideshow of the marked slices
python -m unittest discover tests                                         # synthetic-lattice tests
```

## Outputs (`results/0point5dash1/`)
- **Counts:** `summary.json`
- **3D view:** `defects_3d.png`, `defects_3d.html`
- **Marked slices:** `marked_slices/` and the slideshow `marked_slices_viewer_rgb.npy`
- **Defect list:** `defects.csv`
- **Defect map:** `defects_map.json`, in the same layout as the lattice JSONs, with each defect linked to its marked slice
- **Execution trace:** `trace.jsonl`
- **Scores:** `score.json` (evaluation)

Expected result on this scan: **87 missing and 74 broken struts.** The output is identical on every run.

## Fair-comparison notes
- The detector reads only the CT tif. The registered JSON is used only by `src/evaluation/`.
- Earlier exploratory tools are in `legacy/` and are not exposed through MCP.
