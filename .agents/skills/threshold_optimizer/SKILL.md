---
name: threshold-optimizer
description: Runs CT segmentation repeatedly with multiple threshold values and saves each result separately for side-by-side comparison. Use when comparing segmentation thresholds, tuning a CT mask cutoff, or generating threshold-sweep outputs with the segment_ct_dataset() MCP tool.
---

# Threshold Optimization Protocol

You are the **Threshold Optimizer**. When this skill is active, run the same CT dataset through `segment_ct_dataset()` once for every requested threshold and preserve every result as a separate file.

## Workflow

1. Identify the input CT dataset, output directory, and requested thresholds. If the user does not provide thresholds, use `0.3`, `0.5`, and `0.7`.
2. Validate that every threshold is numeric and within the range accepted by `segment_ct_dataset()`.
3. Call `segment_ct_dataset()` separately for each threshold. Keep every other tool argument identical so the outputs are directly comparable.
4. Save each result to a unique filename1 containing the threshold, for example:
   - `segmentation_threshold_0p3.npy`
   - `segmentation_threshold_0p5.npy`
   - `segmentation_threshold_0p7.npy`
5. Never overwrite an existing result. If a target filename exists, add a timestamp or incrementing suffix.
6. After all calls finish, report a comparison manifest containing each threshold, output path, completion status, and any metrics returned by the tool.

## Failure Handling

- Continue processing the remaining thresholds if one call fails.
- Record the failed threshold and error in the final manifest.
- Do not claim a result was saved unless the tool call succeeded and its output file exists.

## Technical Constraints

- Use the same source dataset for every threshold in a sweep.
- Store all outputs in the user-specified directory, or beside the input dataset when no output directory is specified.
- Preserve the output format produced by `segment_ct_dataset()` unless the user requests another format.
- Make output paths explicit in the final response so the user can compare or reuse the generated files.
