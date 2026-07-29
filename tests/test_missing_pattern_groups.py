from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class _FakeMCP:
    def __init__(self, _name: str) -> None:
        pass

    def tool(self):
        return lambda function: function


fastmcp_stub = types.ModuleType("fastmcp")
fastmcp_stub.FastMCP = _FakeMCP
sys.modules.setdefault("fastmcp", fastmcp_stub)

SCRIPT = Path(__file__).parents[1] / "src" / "mcp_server.py"
SPEC = importlib.util.spec_from_file_location("mcp_server_for_tests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def repeated_grid_group(depth: int = 12) -> np.ndarray:
    group = np.zeros((depth, 48, 48), dtype=bool)
    for image in group:
        image[8:16, 8:16] = True
        image[8:16, 32:40] = True
        image[32:40, 8:16] = True
        image[32:40, 32:40] = True
    return group


class MissingPatternGroupTests(unittest.TestCase):
    def _run_analysis(
        self,
        missing_stop: int,
        max_edge_touch_fraction: float = 1.0,
    ):
        normal_a = repeated_grid_group()
        defective = repeated_grid_group()
        normal_b = repeated_grid_group()
        defective[2:missing_stop, 8:16, 8:16] = False
        group_masks = [normal_a, defective, normal_b]
        threshold, source = MODULE._resolve_shared_threshold(group_masks, None)
        self.assertEqual(source, "label_midpoint")
        self.assertEqual(threshold, 0.5)
        roi = MODULE._foreground_roi(group_masks, presence_fraction=0.01)
        return {
            "groups": MODULE._analyze_group_masks(
                group_masks=group_masks,
                group_paths=[
                    Path("normal_a.npy"),
                    Path("defective.npy"),
                    Path("normal_b.npy"),
                ],
                original_plane_shape=(48, 48),
                roi=roi,
                grid_rows=2,
                grid_columns=2,
                patch_size=24,
                expected_vote=0.6,
                missing_fraction_threshold=0.4,
                similarity_threshold=0.7,
                min_expected_patch_fraction=0.05,
                persistence_fraction=0.5,
                spatial_tolerance_pixels=0,
                max_registration_pixels=8,
                track_radius_cells=1.0,
                max_slice_gap=0,
                max_edge_touch_fraction=max_edge_touch_fraction,
                box_mode="fixed",
                box_overlap=0.5,
                nms_iou_threshold=0.3,
            )
        }

    def test_classifies_pattern_missing_for_at_least_half_group(self) -> None:
        result = self._run_analysis(missing_stop=9)
        self.assertEqual(result["groups"][0]["missing_pattern_count"], 0)
        self.assertEqual(result["groups"][2]["missing_pattern_count"], 0)
        self.assertGreaterEqual(result["groups"][1]["missing_pattern_count"], 1)
        missing = result["groups"][1]["missing_patterns"][0]
        self.assertGreaterEqual(missing["persistent_span_slices"], 6)
        self.assertIn(
            {"row": 0, "column": 0},
            missing["grid_cells_zero_based"],
        )

    def test_rejects_pattern_missing_for_less_than_half_group(self) -> None:
        result = self._run_analysis(missing_stop=7)
        self.assertEqual(result["groups"][1]["missing_pattern_count"], 0)
        self.assertGreaterEqual(result["groups"][1]["candidate_track_count"], 1)

    def test_persistent_span_tolerates_only_requested_gap(self) -> None:
        self.assertEqual(
            MODULE._longest_persistent_span([1, 2, 4, 5], max_slice_gap=1),
            5,
        )
        self.assertEqual(
            MODULE._longest_persistent_span([1, 2, 4, 5], max_slice_gap=0),
            2,
        )

    def test_edge_pattern_is_inconclusive_not_missing(self) -> None:
        result = self._run_analysis(
            missing_stop=9,
            max_edge_touch_fraction=0.5,
        )
        defective = result["groups"][1]
        self.assertEqual(defective["missing_pattern_count"], 0)
        self.assertGreaterEqual(
            defective["edge_inconclusive_pattern_count"],
            1,
        )
        self.assertEqual(
            defective["edge_inconclusive_patterns"][0]["classification"],
            "inconclusive_scan_edge",
        )

    def test_connected_tracker_merges_vertical_motion_across_node_gap(self) -> None:
        def component(slice_index: int, row: int, column: int):
            return {
                "slice_index_zero_based": slice_index,
                "slice_number_one_based": slice_index + 1,
                "centroid_grid_row": float(row),
                "centroid_grid_column": float(column),
                "grid_cells": [{"row": row, "column": column}],
                "median_missing_fraction": 0.9,
            }

        observations = [[] for _ in range(12)]
        for slice_index in [1, 2, 3]:
            observations[slice_index] = [component(slice_index, 1, 1)]
        for slice_index in [7, 8, 9]:
            observations[slice_index] = [component(slice_index, 2, 1)]

        tracks = MODULE._track_grid_components(
            observations,
            track_radius_cells=1.5,
            max_slice_gap=3,
        )
        self.assertEqual(len(tracks), 1)
        self.assertEqual(
            [item["slice_index_zero_based"] for item in tracks[0]],
            [1, 2, 3, 7, 8, 9],
        )

    def test_explicit_segmentation_mask_excludes_raw_shadow(self) -> None:
        clean_mask = repeated_grid_group()
        raw_with_shadow = clean_mask.astype(np.uint16) * 50_000
        raw_with_shadow[:, 16:32, :] = 45_000
        path_map = {
            "raw.npy": (Path("raw.npy").resolve(), raw_with_shadow),
            "mask.npy": (Path("mask.npy").resolve(), clean_mask),
        }

        with mock.patch.object(
            MODULE,
            "_load_volume",
            side_effect=lambda filepath: path_map[filepath],
        ):
            result = MODULE.analyze_slice_groups_for_missing_patterns(
                ["raw.npy"],
                segmentation_filepaths=["mask.npy"],
                grid_rows=2,
                grid_columns=2,
                analysis_size=48,
                patch_size=24,
                spatial_tolerance_pixels=0,
                max_registration_pixels=8,
                max_slice_gap=0,
                max_edge_touch_fraction=1.0,
            )

        self.assertEqual(
            result["threshold_source"],
            "provided_segmentation_masks",
        )
        self.assertEqual(result["total_missing_pattern_count"], 0)
        self.assertEqual(
            result["segmentation_files"],
            [str(Path("mask.npy").resolve())],
        )

    def test_tracker_preserves_two_simultaneous_missing_patterns(self) -> None:
        def component(slice_index: int, row: float, column: float):
            return {
                "slice_index_zero_based": slice_index,
                "slice_number_one_based": slice_index + 1,
                "centroid_grid_row": row,
                "centroid_grid_column": column,
                "grid_cells": [
                    {"row": round(row), "column": round(column)}
                ],
            }

        observations = [
            [
                component(slice_index, 2.0, 2.0),
                component(slice_index, 2.0, 3.0),
            ]
            for slice_index in range(8)
        ]
        tracks = MODULE._track_grid_components(
            observations,
            track_radius_cells=1.1,
            max_slice_gap=0,
        )
        self.assertEqual(len(tracks), 2)
        self.assertEqual(sorted(len(track) for track in tracks), [8, 8])

    def test_dynamic_boxes_detect_pattern_on_fixed_grid_seam(self) -> None:
        image = np.zeros((96, 96), dtype=bool)
        for center_y in (24, 48, 72):
            for center_x in (24, 48, 72):
                image[
                    center_y - 5 : center_y + 5,
                    center_x - 5 : center_x + 5,
                ] = True
        image[43:53, 43:53] = False

        candidates = MODULE._grid_components(
            target=image,
            roi=(0, 96, 0, 96),
            grid_rows=4,
            grid_columns=4,
            patch_size=24,
            expected_vote=0.6,
            missing_fraction_threshold=0.4,
            similarity_threshold=0.7,
            min_expected_patch_fraction=0.02,
            spatial_tolerance_pixels=0,
            max_registration_pixels=8,
            slice_index=0,
            box_mode="dynamic",
            box_overlap=0.5,
            nms_iou_threshold=0.3,
        )

        self.assertTrue(
            any(
                abs(item["centroid_grid_row"] - 1.5) <= 0.1
                and abs(item["centroid_grid_column"] - 1.5) <= 0.1
                for item in candidates
            )
        )

    def test_classifies_two_persistent_missing_patterns_independently(self) -> None:
        group = np.zeros((12, 96, 96), dtype=bool)
        for image in group:
            for row in range(4):
                for column in range(4):
                    center_y = row * 24 + 12
                    center_x = column * 24 + 12
                    image[
                        center_y - 5 : center_y + 5,
                        center_x - 5 : center_x + 5,
                    ] = True
        group[2:9, 31:41, 31:41] = False
        group[2:9, 55:65, 55:65] = False

        result = MODULE._analyze_group_masks(
            group_masks=[group],
            group_paths=[Path("two_missing.npy")],
            original_plane_shape=(96, 96),
            roi=(0, 96, 0, 96),
            grid_rows=4,
            grid_columns=4,
            patch_size=24,
            expected_vote=0.6,
            missing_fraction_threshold=0.4,
            similarity_threshold=0.7,
            min_expected_patch_fraction=0.02,
            persistence_fraction=0.5,
            spatial_tolerance_pixels=0,
            max_registration_pixels=8,
            track_radius_cells=0.75,
            max_slice_gap=0,
            max_edge_touch_fraction=0.5,
            box_mode="fixed",
            box_overlap=0.5,
            nms_iou_threshold=0.3,
        )[0]

        self.assertEqual(result["missing_pattern_count"], 2)
        cells = [
            pattern["grid_cells_zero_based"]
            for pattern in result["missing_patterns"]
        ]
        self.assertIn([{"row": 1, "column": 1}], cells)
        self.assertIn([{"row": 2, "column": 2}], cells)


if __name__ == "__main__":
    unittest.main()
