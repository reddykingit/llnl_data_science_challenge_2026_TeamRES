# Legacy (not used by the PNV technique)

These are the earlier Codex-era tools and analyses, kept for history:
- `src/codex_mcp_server.py`: the old MCP server (segmentation, slice view, skeletonise, octet-layer length, slice-group patterns, deformation analysis)
- `src/pattern_deformation_analysis.py`: the earlier 6x6 grid / periodic-segment pipeline
- `tests/`: tests for the code above
- `analysis_outputs/`: earlier run scripts
- `.agents/`: the Codex NDE-report skill

For the fair comparison they are deliberately **not** exposed through `.mcp.json` or `.claude/`.
The imports inside these files still point at `src.*`, so copy a file back into `src/` if you need to rerun it.
