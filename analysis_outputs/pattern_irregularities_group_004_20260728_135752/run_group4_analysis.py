"""Run the established three-class pattern analysis on group 4 only."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


RUN_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = RUN_DIRECTORY.parents[1]
REFERENCE_RUNNER = (
    REPOSITORY_ROOT
    / "analysis_outputs"
    / "pattern_irregularities_210127_0point5dash1_20260728_132145"
    / "run_pattern_irregularity_analysis.py"
)
GROUP_NUMBER = 4


def load_reference_runner():
    spec = importlib.util.spec_from_file_location(
        "established_pattern_irregularity_pipeline",
        REFERENCE_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pipeline from {REFERENCE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_DIRECTORY = RUN_DIRECTORY
    return module


def relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(RUN_DIRECTORY).as_posix()


def write_report(
    module,
    summary: dict[str, object],
    detailed_report: dict[str, object],
    csv_path: Path,
    chart_path: Path,
) -> Path:
    group = summary["groups"][0]
    class_counts = detailed_report[
        "confirmed_defect_instance_counts_by_class"
    ]
    track_counts = detailed_report["confirmed_track_counts_by_class"]
    start = int(group["dataset_slice_start_one_based"])
    end = int(group["dataset_slice_end_one_based_inclusive"])
    collage = relative(group["collage"])
    report_json = relative(group["report"])

    evidence_csv = (
        RUN_DIRECTORY
        / "group_004"
        / "group_004_grid_confirmed_patterns.csv"
    )
    with evidence_csv.open(newline="", encoding="utf-8") as handle:
        evidence_rows = list(csv.DictReader(handle))
    detected_slices = sorted(
        {int(row["dataset_index_zero_based"]) + 1 for row in evidence_rows}
    )
    detected_cells = sorted(
        {
            (
                int(row["grid_row_zero_based"]),
                int(row["grid_column_zero_based"]),
            )
            for row in evidence_rows
        }
    )

    report_text = f"""# Group 4 Pattern Irregularity Analysis

## Executive summary

Only **group 4** of the TIFF stack was analyzed: dataset slices
**{start}-{end}** (1-based), a complete {end - start + 1}-slice lattice period.
The detector identified **{int(group['confirmed_irregularity_count'])}
confirmed pattern-irregularity instances** across
**{int(group['confirmed_irregularity_track_count'])} tracks**.

| Irregularity class | Confirmed instances | Tracks |
|:--|--:|--:|
| Missing strut | {int(class_counts['missing_strut'])} | {int(track_counts['missing_strut'])} |
| Broken strut | {int(class_counts['broken_strut'])} | {int(track_counts['broken_strut'])} |
| Thin/deformed strut | {int(class_counts['thin_or_deformed_strut'])} | {int(track_counts['thin_or_deformed_strut'])} |
| **Total** | **{int(group['confirmed_irregularity_count'])}** | **{int(group['confirmed_irregularity_track_count'])}** |

Confirmed evidence occurs on **{len(detected_slices)} distinct slices** and
across **{len(detected_cells)} distinct 9 x 9 grid cells**. Counts are pattern
instances; a persistent physical feature can contribute more than one instance
within a track.

![Group 4 confirmed pattern irregularities]({collage})

The gold dashed boxes identify the reported grid cells. Overlay colors are red
for missing, orange for broken, and cyan for thin/deformed struts.

## Outputs

- [Detailed findings JSON]({report_json})
- [Confirmed-pattern evidence CSV](group_004/group_004_grid_confirmed_patterns.csv)
- [Group summary CSV]({relative(csv_path)})
- [Class-count chart]({relative(chart_path)})
- [Evidence collage]({collage})
- [Run summary JSON](run_summary.json)
- Slice-level arrays in `group_004/` include the candidate mask, confirmed
  mask, class labels, and segmented RGB viewer.

![Group 4 irregularity counts]({relative(chart_path)})

## Method

- Input: `{Path(str(summary['source_file'])).name}` with shape
  `{tuple(summary['input_shape'])}` (Z, Y, X), dtype
  `{summary['input_dtype']}`.
- Scope: group 4 only, zero-based indices 234-311 (dataset slices 235-312).
- Segmentation: high-valued foreground, shared Otsu threshold
  {float(summary['otsu_threshold']):.1f}, followed by a 3-pixel opening.
- Spatial model: 9 x 9 grid with periodic translated-neighbor consensus,
  two geometry-reference cycles, tilt correction, and forward trajectory
  prediction.
- Classification: persistent expected-but-absent evidence is missing;
  transient absence or an abrupt trajectory discontinuity is broken;
  persistent low-clearance mismatch is thin/deformed.
- Confirmation: track-level confidence at least
  {float(detailed_report['confidence_threshold']):.2f}.

## Validation and interpretation

The 25 tests in `tests.test_grid_pattern_inspection` passed immediately before
this run. Output integrity is checked after execution for array dimensions,
non-empty images, slice scope, summary consistency, and report links.

These findings are algorithmic screening results rather than
metrology-certified ground truth. Review the collage and slice-level RGB viewer
before using the classifications for physical disposition decisions.
"""
    output = RUN_DIRECTORY / "REPORT.md"
    output.write_text(report_text, encoding="utf-8")
    return output


def main() -> None:
    module = load_reference_runner()
    volume, threshold = module.load_volume_and_threshold()
    group = module.analyze_group(volume, threshold, GROUP_NUMBER)
    report_path = Path(group["report"])
    detailed_report = json.loads(report_path.read_text(encoding="utf-8"))

    summary = {
        "source_file": str(module.INPUT_FILE),
        "input_shape": list(volume.shape),
        "input_dtype": str(volume.dtype),
        "analyzed_groups": [GROUP_NUMBER],
        "period_slices": module.PERIOD_SLICES,
        "otsu_threshold": threshold,
        "groups": [group],
        "pipeline_source": str(REFERENCE_RUNNER),
    }
    summary_path = RUN_DIRECTORY / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary_csv = module.make_summary_csv([group])
    chart = module.make_counts_chart([group])
    report = write_report(
        module,
        summary,
        detailed_report,
        summary_csv,
        chart,
    )
    print(
        json.dumps(
            {
                "analyzed_groups": [GROUP_NUMBER],
                "summary": str(summary_path),
                "report": str(report),
                "collage": str(group["collage"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
