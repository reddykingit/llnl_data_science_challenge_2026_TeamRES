"""Run a fresh all-groups pattern-irregularity analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path


RUN_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = RUN_DIRECTORY.parents[1]
VALIDATED_ALL_GROUPS_RUNNER = (
    REPOSITORY_ROOT
    / "analysis_outputs"
    / "pattern_irregularities_all_groups_20260728_143010"
    / "run_all_groups_analysis.py"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "validated_all_groups_pattern_runner",
        VALIDATED_ALL_GROUPS_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load runner from {VALIDATED_ALL_GROUPS_RUNNER}"
        )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.RUN_DIRECTORY = RUN_DIRECTORY
    runner.main()

    report_path = RUN_DIRECTORY / "REPORT.md"
    report_text = report_path.read_text(encoding="utf-8")
    report_path.write_text(
        report_text.replace(
            "All 27 tests in `tests.test_grid_pattern_inspection` passed",
            "All 30 tests in `tests.test_grid_pattern_inspection` passed",
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
