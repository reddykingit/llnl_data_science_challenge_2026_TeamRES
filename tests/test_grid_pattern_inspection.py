from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.pattern_deformation_analysis import (
    MaskCache,
    _apply_tilt_correction,
    _confirmed_pattern_instance_count,
    _count_pattern_instances,
    _deduplicate_confirmed_tracks,
    _estimate_stack_tilt,
    _grid_cell_bounds,
    _grid_roi_from_indices,
    _grid_slice_candidates,
    _missing_segment_spatial_metrics,
    _periodic_slice_candidates,
    _summarize_grid_tracks,
    _track_all,
    resolve_min_track_slices,
    run_grid_pattern_analysis,
    tilt_tolerance_from_angle,
)


def grid_slice() -> np.ndarray:
    image = np.zeros((64, 64), dtype=bool)
    for row in range(4):
        for column in range(4):
            y = row * 16 + 5
            x = column * 16 + 5
            image[y : y + 6, x : x + 6] = True
    return image


def add_disk(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int = 4,
) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    image[
        (yy - center_y) ** 2 + (xx - center_x) ** 2
        <= radius**2
    ] = True


def remove_disk(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int = 5,
) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    image[
        (yy - center_y) ** 2 + (xx - center_x) ** 2
        <= radius**2
    ] = False


def periodic_candidates(image: np.ndarray) -> list[dict]:
    candidates, _, _, _, _ = _periodic_slice_candidates(
        image,
        slice_index=0,
        roi=(0, image.shape[0], 0, image.shape[1]),
        grid_rows=9,
        grid_columns=9,
        expected_vote=0.5,
        vector_count=4,
        minimum_prediction_votes=3,
        minimum_bidirectional_pairs=2,
        spatial_tolerance_pixels=1,
        min_missing_segment_area=6,
        min_missing_segment_fraction=0.001,
        min_missing_clearance_pixels=4.0,
        min_missing_clearance_fraction=0.04,
        border_margin_pixels=2,
        border_margin_fraction=0.10,
    )
    return candidates


class GridPatternInspectionTests(unittest.TestCase):
    def test_missing_cells_are_retained_but_deformation_is_rejected(self):
        image = grid_slice()
        image[21:27, 21:27] = False
        image[37:43, 53:59] = False
        image[5:11, 37:43] = False
        image[6:10, 38:42] = True

        candidates, masks, _, combined = _grid_slice_candidates(
            image,
            slice_index=4,
            bounds=_grid_cell_bounds((0, 64, 0, 64), 4, 4),
            patch_size=32,
            expected_vote=0.6,
            min_reference_fraction=0.01,
            missing_fraction_threshold=0.6,
            max_present_fraction_of_expected=0.45,
            similarity_threshold=0.55,
            spatial_tolerance_pixels=0,
        )

        missing_cells = {
            (candidate["grid_row"], candidate["grid_column"])
            for candidate in candidates
        }
        self.assertEqual(missing_cells, {(1, 1), (2, 3)})
        self.assertEqual(len(masks), 2)
        self.assertTrue(combined.any())

    def test_parallel_tracker_confirms_every_simultaneous_missing_cell(self):
        observations = {}
        bounds = _grid_cell_bounds((0, 64, 0, 64), 4, 4)
        for index in range(3):
            image = grid_slice()
            image[21:27, 21:27] = False
            image[37:43, 53:59] = False
            candidates, _, _, _ = _grid_slice_candidates(
                image,
                slice_index=index,
                bounds=bounds,
                patch_size=32,
                expected_vote=0.6,
                min_reference_fraction=0.01,
                missing_fraction_threshold=0.6,
                max_present_fraction_of_expected=0.45,
                similarity_threshold=0.55,
                spatial_tolerance_pixels=0,
            )
            observations[index] = candidates

        tracks = _track_all(observations, radius=0.25)
        _, confirmed = _summarize_grid_tracks(
            tracks,
            min_track_slices=3,
            confidence_threshold=0.7,
        )
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(confirmed), 2)

    def test_fifteen_slice_confirmation_boundary(self):
        def persistent_track(length: int) -> list[dict]:
            return [
                {
                    "slice_index_zero_based": slice_index,
                    "centroid_y": 2.0,
                    "centroid_x": 3.0,
                    "missing_fraction": 1.0,
                    "present_fraction_of_expected": 0.0,
                    "complete_absence": True,
                    "grid_row": 2,
                    "grid_column": 3,
                    "detection_method": (
                        "periodic_translation_consensus"
                    ),
                }
                for slice_index in range(length)
            ]

        summaries, confirmed = _summarize_grid_tracks(
            [persistent_track(14), persistent_track(15)],
            min_track_slices=15,
            confidence_threshold=0.7,
        )

        self.assertEqual(len(confirmed), 2)
        self.assertEqual(summaries[0]["defect_class"], "missing_strut")
        self.assertEqual(summaries[1]["defect_class"], "missing_strut")

    def test_default_persistence_is_half_the_actual_group(self):
        self.assertEqual(resolve_min_track_slices(None, 78, 78), 39)
        self.assertEqual(resolve_min_track_slices(None, 77, 77), 39)
        self.assertEqual(resolve_min_track_slices(None, 78, 23), 12)
        self.assertEqual(resolve_min_track_slices(15, 78, 78), 15)

    def test_known_tilt_angle_adds_detection_and_tracking_tolerance(self):
        allowance = tilt_tolerance_from_angle(
            angle_degrees=0.664,
            tolerance_degrees=0.15,
            slice_count=78,
            slice_spacing_to_pixel_spacing_ratio=1.0,
            cell_pitch_pixels=20.0,
        )

        self.assertAlmostEqual(
            allowance["maximum_allowed_angle_degrees"],
            0.814,
        )
        self.assertEqual(
            allowance["additional_spatial_tolerance_pixels"],
            2,
        )
        self.assertGreater(
            allowance["tracking_tolerance_cells_per_slice"],
            0.0,
        )

        observations = {
            0: [
                {
                    "slice_index_zero_based": 0,
                    "centroid_y": 1.0,
                    "centroid_x": 1.0,
                }
            ],
            4: [
                {
                    "slice_index_zero_based": 4,
                    "centroid_y": 1.6,
                    "centroid_x": 1.0,
                }
            ],
        }
        without_angle = _track_all(
            observations,
            radius=0.5,
            maximum_gap=4,
            gap_radius_growth=0.0,
            tilt_tolerance_per_slice_cells=0.0,
        )
        with_angle = _track_all(
            observations,
            radius=0.5,
            maximum_gap=4,
            gap_radius_growth=0.0,
            tilt_tolerance_per_slice_cells=0.03,
        )
        self.assertEqual(len(without_angle), 2)
        self.assertEqual(len(with_angle), 1)

    def test_neighboring_octets_preserve_completely_missing_edge_cell(self):
        volume = np.stack([grid_slice() for _ in range(36)])
        # The first motif is completely absent for every slice in group 2.
        volume[12:24, 5:11, 5:11] = False
        cache = MaskCache(
            volume,
            threshold=0.5,
            foreground="high",
            opening_size=1,
        )
        geometry_indices = list(range(0, 36))
        roi = _grid_roi_from_indices(
            cache,
            geometry_indices,
            presence_fraction=0.01,
            grid_rows=4,
            grid_columns=4,
        )
        bounds = _grid_cell_bounds(roi, 4, 4)
        observations = {}
        for index in range(12, 24):
            candidates, _, _, _ = _grid_slice_candidates(
                cache.get(index),
                slice_index=index,
                bounds=bounds,
                patch_size=32,
                expected_vote=0.6,
                min_reference_fraction=0.01,
                missing_fraction_threshold=0.6,
                max_present_fraction_of_expected=0.45,
                similarity_threshold=0.55,
                spatial_tolerance_pixels=0,
            )
            observations[index] = candidates

        tracks = _track_all(observations, radius=0.25)
        _, confirmed = _summarize_grid_tracks(
            tracks,
            min_track_slices=3,
            confidence_threshold=0.7,
        )
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(
            confirmed[0]["classification"],
            "missing_strut",
        )
        self.assertEqual(
            confirmed[0]["grid_cells_zero_based"],
            [(0, 0)],
        )
        self.assertEqual(
            confirmed[0]["complete_absence_observation_fraction"],
            1.0,
        )

    def test_cross_section_with_two_missing_arms_finds_two_segments(self):
        image = np.zeros((206, 210), dtype=bool)
        centers = [
            (20, 60),
            (20, 144),
            (62, 18),
            (62, 102),
            (62, 186),
            (104, 60),
            (104, 144),
            (146, 18),
            (146, 102),
            (146, 186),
            (188, 60),
            (188, 144),
        ]
        for center_y, center_x in centers:
            for dy, dx in [(-12, 0), (12, 0), (0, -12), (0, 12)]:
                add_disk(image, center_y + dy, center_x + dx)
        remove_disk(image, 62, 114)
        remove_disk(image, 92, 144)

        candidates = periodic_candidates(image)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(
            all(
                candidate["detection_method"]
                == "periodic_translation_consensus"
                for candidate in candidates
            )
        )

    def test_connected_diamond_lattice_finds_missing_diagonal_segment(self):
        yy, xx = np.indices((252, 218))
        descending = np.abs(((xx - yy + 42) % 84) - 42) <= 3
        ascending = np.abs(((xx + yy + 42) % 84) - 42) <= 3
        image = descending | ascending
        missing_segment = (
            (np.abs(xx - yy) <= 4)
            & (yy > 87)
            & (yy < 130)
            & (xx > 87)
            & (xx < 130)
        )
        image[missing_segment] = False

        candidates = periodic_candidates(image)

        self.assertEqual(len(candidates), 1)
        self.assertGreater(
            candidates[0]["missing_segment_area_pixels"],
            200,
        )

    def test_staggered_node_lattice_finds_one_wholly_missing_node(self):
        image = np.zeros((199, 194), dtype=bool)
        for row, center_y in enumerate(range(5, 199, 21)):
            start_x = 25 if row % 2 == 0 else 4
            for center_x in range(start_x, 194, 42):
                add_disk(image, center_y, center_x)
        remove_disk(image, 110, 88)

        candidates = periodic_candidates(image)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["complete_absence"])

    def test_small_clear_missing_node_survives_ct_resolution_scaling(self):
        image = np.zeros((400, 400), dtype=bool)
        centers = []
        for row, center_y in enumerate(range(20, 400, 40)):
            start_x = 20 if row % 2 == 0 else 60
            for center_x in range(start_x, 400, 80):
                add_disk(image, center_y, center_x, radius=3)
                centers.append((center_y, center_x))
        missing_y, missing_x = centers[len(centers) // 2]
        remove_disk(image, missing_y, missing_x, radius=4)

        candidates = periodic_candidates(image)

        self.assertEqual(len(candidates), 1)
        self.assertLess(
            candidates[0]["missing_segment_area_pixels"],
            50,
        )

    def test_shifted_node_is_rejected_as_not_missing(self):
        image = np.zeros((400, 400), dtype=bool)
        centers = []
        for row, center_y in enumerate(range(20, 400, 40)):
            start_x = 20 if row % 2 == 0 else 60
            for center_x in range(start_x, 400, 80):
                add_disk(image, center_y, center_x, radius=3)
                centers.append((center_y, center_x))
        shifted_y, shifted_x = centers[len(centers) // 2]
        remove_disk(image, shifted_y, shifted_x, radius=4)
        add_disk(image, shifted_y, shifted_x + 5, radius=3)

        candidates = periodic_candidates(image)

        self.assertEqual(candidates, [])

    def test_scan_border_loss_is_rejected_but_interior_loss_is_kept(self):
        image = np.zeros((400, 400), dtype=bool)
        for row, center_y in enumerate(range(20, 400, 40)):
            start_x = 20 if row % 2 == 0 else 60
            for center_x in range(start_x, 400, 80):
                add_disk(image, center_y, center_x, radius=3)
        remove_disk(image, 180, 180, radius=4)
        # This outer-row site models material cut off by the scan boundary.
        remove_disk(image, 20, 180, radius=4)

        candidates = periodic_candidates(image)

        self.assertEqual(len(candidates), 1)
        center = candidates[0]["centroid_original_pixels"]
        self.assertAlmostEqual(center["y"], 180.0)
        self.assertAlmostEqual(center["x"], 180.0)
        self.assertGreaterEqual(
            candidates[0]["median_bidirectional_pair_support"],
            2.0,
        )

    def test_tiny_missing_arm_inside_larger_cross_pattern_is_kept(self):
        image = np.zeros((206, 210), dtype=bool)
        centers = [
            (20, 60),
            (20, 144),
            (62, 18),
            (62, 102),
            (62, 186),
            (104, 60),
            (104, 144),
            (146, 18),
            (146, 102),
            (146, 186),
            (188, 60),
            (188, 144),
        ]
        for center_y, center_x in centers:
            for dy, dx in [(-12, 0), (12, 0), (0, -12), (0, 12)]:
                add_disk(
                    image,
                    center_y + dy,
                    center_x + dx,
                    radius=2,
                )
        remove_disk(image, 62, 114, radius=3)

        candidates = periodic_candidates(image)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["missing_segment_area_pixels"],
            13,
        )

    def test_phase_gap_tracking_links_multiple_missing_struts(self):
        observations = {}
        for slice_index in [0, 11, 22]:
            observations[slice_index] = [
                {
                    "candidate_id": 1,
                    "slice_index_zero_based": slice_index,
                    "centroid_y": 2.0 + 0.02 * slice_index,
                    "centroid_x": 3.0,
                    "missing_fraction": 1.0,
                    "present_fraction_of_expected": 0.0,
                    "complete_absence": True,
                    "grid_row": 2,
                    "grid_column": 3,
                    "detection_method": (
                        "periodic_translation_consensus"
                    ),
                },
                {
                    "candidate_id": 2,
                    "slice_index_zero_based": slice_index,
                    "centroid_y": 6.0,
                    "centroid_x": 7.0 - 0.02 * slice_index,
                    "missing_fraction": 1.0,
                    "present_fraction_of_expected": 0.0,
                    "complete_absence": True,
                    "grid_row": 6,
                    "grid_column": 7,
                    "detection_method": (
                        "periodic_translation_consensus"
                    ),
                },
            ]

        tracks = _track_all(
            observations,
            radius=1.5,
            maximum_gap=12,
        )
        summaries, confirmed = _summarize_grid_tracks(
            tracks,
            min_track_slices=3,
            confidence_threshold=0.7,
            maximum_gap=12,
        )

        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(confirmed), 2)
        self.assertTrue(
            all(not track["rejection_reasons"] for track in summaries)
        )

    def test_one_track_with_two_patterns_counts_two_missing_struts(self):
        missing_mask = np.zeros((32, 32), dtype=bool)
        missing_mask[5:10, 5:10] = True
        missing_mask[20:26, 21:27] = True
        self.assertEqual(_count_pattern_instances(missing_mask), 2)

        track = []
        for slice_index in range(3):
            track.append(
                {
                    "slice_index_zero_based": slice_index,
                    "centroid_y": 3.0,
                    "centroid_x": 3.0,
                    "missing_fraction": 1.0,
                    "present_fraction_of_expected": 0.0,
                    "complete_absence": True,
                    "grid_row": 3,
                    "grid_column": 3,
                    "pattern_instance_count": 2,
                    "detection_method": "fixed_cell_prototype",
                }
            )

        _, confirmed = _summarize_grid_tracks(
            [track],
            min_track_slices=3,
            confidence_threshold=0.7,
        )

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["pattern_instance_count"], 2)
        self.assertEqual(
            _confirmed_pattern_instance_count(confirmed),
            2,
        )

    def test_tracks_classify_missing_and_broken_exclusively(self):
        def observation(
            slice_index: int,
            centroid_y: float = 2.0,
        ) -> dict:
            return {
                "slice_index_zero_based": slice_index,
                "centroid_y": centroid_y,
                "centroid_x": 3.0,
                "missing_fraction": 1.0,
                "present_fraction_of_expected": 0.0,
                "complete_absence": True,
                "patch_similarity": 0.0,
                "grid_row": 2,
                "grid_column": 3,
                "detection_method": "fixed_cell_prototype",
            }

        persistent_missing = [
            observation(index) for index in range(6)
        ]
        transient_break = [
            observation(index) for index in range(2)
        ]
        trajectory_jump = [
            observation(0, 2.0),
            observation(1, 2.1),
            observation(2, 3.5),
        ]

        _, classified = _summarize_grid_tracks(
            [
                persistent_missing,
                transient_break,
                trajectory_jump,
            ],
            min_track_slices=6,
            confidence_threshold=0.7,
            broken_min_track_slices=2,
            broken_confidence_threshold=0.7,
            missing_min_observation_fraction=0.5,
            missing_span_tolerance_fraction=0.0,
            broken_jump_threshold_cells=0.75,
        )

        self.assertEqual(
            [track["defect_class"] for track in classified],
            [
                "missing_strut",
                "broken_strut",
                "broken_strut",
            ],
        )

    def test_full_spatial_segment_is_missing_despite_sparse_phases(self):
        indices = [0, 1, 2, 3, 4, 13, 24, 25, 26, 27]
        track = []
        for index in indices:
            track.append(
                {
                    "slice_index_zero_based": index,
                    "centroid_y": 2.0,
                    "centroid_x": 3.0,
                    "missing_fraction": 1.0,
                    "present_fraction_of_expected": 0.0,
                    "complete_absence": True,
                    "patch_similarity": 0.0,
                    "missing_segment_span_fraction_cells": (
                        0.49 if index in {25, 26} else 0.10
                    ),
                    "grid_row": 2,
                    "grid_column": 3,
                    "detection_method": (
                        "periodic_translation_consensus"
                    ),
                }
            )

        summaries, confirmed = _summarize_grid_tracks(
            [track],
            min_track_slices=39,
            confidence_threshold=0.65,
            maximum_gap=27,
        )

        self.assertEqual(summaries[0]["defect_class"], "missing_strut")
        self.assertTrue(
            summaries[0]["spatial_missing_rule_satisfied"]
        )
        self.assertFalse(
            summaries[0]["temporal_missing_rule_satisfied"]
        )
        self.assertEqual(len(confirmed), 1)

    def test_sparse_short_interruptions_are_broken_not_missing(self):
        indices = [
            0,
            5,
            6,
            7,
            8,
            9,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            35,
            36,
        ]
        track = [
            {
                "slice_index_zero_based": index,
                "centroid_y": 2.0,
                "centroid_x": 3.0,
                "missing_fraction": 1.0,
                "present_fraction_of_expected": 0.0,
                "complete_absence": True,
                "patch_similarity": 0.0,
                "missing_segment_span_fraction_cells": 0.18,
                "grid_row": 2,
                "grid_column": 3,
                "detection_method": "periodic_translation_consensus",
            }
            for index in indices
        ]

        summaries, confirmed = _summarize_grid_tracks(
            [track],
            min_track_slices=39,
            confidence_threshold=0.65,
            maximum_gap=27,
        )

        self.assertEqual(summaries[0]["persistent_span_slices"], 37)
        self.assertEqual(summaries[0]["defect_class"], "broken_strut")
        self.assertFalse(
            summaries[0]["temporal_missing_rule_satisfied"]
        )
        self.assertFalse(
            summaries[0]["spatial_missing_rule_satisfied"]
        )
        self.assertEqual(len(confirmed), 1)

    def test_segment_geometry_normalizes_length_by_grid_cell(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[9:12, 2:12] = True

        metrics = _missing_segment_spatial_metrics(mask, 20, 20)

        self.assertGreaterEqual(
            metrics["missing_segment_span_fraction_cells"],
            0.45,
        )
        self.assertGreater(
            metrics["missing_segment_aspect_ratio"],
            2.0,
        )

    def test_default_broken_threshold_requires_eight_observations(self):
        def absence_track(length: int) -> list[dict]:
            return [
                {
                    "slice_index_zero_based": index,
                    "centroid_y": 2.0,
                    "centroid_x": 3.0,
                    "missing_fraction": 1.0,
                    "present_fraction_of_expected": 0.0,
                    "complete_absence": True,
                    "patch_similarity": 0.0,
                    "grid_row": 2,
                    "grid_column": 3,
                    "detection_method": "fixed_cell_prototype",
                }
                for index in range(length)
            ]

        summaries, classified = _summarize_grid_tracks(
            [absence_track(7), absence_track(8)],
            min_track_slices=39,
            confidence_threshold=0.65,
        )

        self.assertIsNone(summaries[0]["defect_class"])
        self.assertEqual(summaries[1]["defect_class"], "broken_strut")
        self.assertEqual(len(classified), 1)

    def test_missing_priority_deduplicates_one_physical_strut(self):
        missing = {
            "track_id": 1,
            "defect_class": "missing_strut",
            "confidence": 0.90,
            "persistent_span_slices": 34,
            "start_index_zero_based": 0,
            "end_index_zero_based": 33,
            "grid_cells_zero_based": [(2, 3)],
            "median_tracking_centroid_y": 2.0,
            "median_tracking_centroid_x": 3.0,
            "pattern_instance_count": 1,
        }
        duplicate_broken = {
            "track_id": 2,
            "defect_class": "broken_strut",
            "confidence": 0.95,
            "persistent_span_slices": 8,
            "start_index_zero_based": 30,
            "end_index_zero_based": 37,
            "grid_cells_zero_based": [(2, 3)],
            "median_tracking_centroid_y": 2.1,
            "median_tracking_centroid_x": 3.1,
            "pattern_instance_count": 1,
        }
        independent_broken = {
            "track_id": 3,
            "defect_class": "broken_strut",
            "confidence": 0.90,
            "persistent_span_slices": 8,
            "start_index_zero_based": 30,
            "end_index_zero_based": 37,
            "grid_cells_zero_based": [(5, 6)],
            "median_tracking_centroid_y": 5.0,
            "median_tracking_centroid_x": 6.0,
            "pattern_instance_count": 1,
        }

        kept, duplicate_map = _deduplicate_confirmed_tracks(
            [duplicate_broken, independent_broken, missing],
            maximum_gap=8,
        )

        self.assertEqual(
            [track["track_id"] for track in kept],
            [1, 3],
        )
        self.assertEqual(duplicate_map, {2: 1})
        self.assertEqual(missing["merged_duplicate_track_ids"], [2])

    def test_fragmented_long_absence_is_missing_not_broken(self):
        def absence_track(indices: list[int]) -> list[dict]:
            return [
                {
                    "slice_index_zero_based": slice_index,
                    "centroid_y": 2.0,
                    "centroid_x": 3.0,
                    "missing_fraction": 1.0,
                    "present_fraction_of_expected": 0.0,
                    "complete_absence": True,
                    "patch_similarity": 0.0,
                    "grid_row": 2,
                    "grid_column": 3,
                    "detection_method": "periodic_translation_consensus",
                }
                for slice_index in indices
            ]

        summaries, classified = _summarize_grid_tracks(
            [
                absence_track(list(range(0, 39, 2))),
                absence_track(list(range(10, 15))),
                absence_track([20, 21]),
            ],
            min_track_slices=39,
            confidence_threshold=0.7,
            maximum_gap=2,
            broken_min_track_slices=5,
            broken_confidence_threshold=0.8,
            missing_min_observation_fraction=0.5,
            missing_span_tolerance_fraction=0.1,
        )

        self.assertEqual(summaries[0]["defect_class"], "missing_strut")
        self.assertEqual(
            summaries[0]["effective_missing_min_span_slices"],
            36,
        )
        self.assertEqual(summaries[1]["defect_class"], "broken_strut")
        self.assertIsNone(summaries[2]["defect_class"])
        self.assertEqual(len(classified), 2)

    def test_end_to_end_outputs_separate_missing_and_broken_classes(self):
        volume = np.stack([grid_slice() for _ in range(12)])
        volume[0:4, 21:27, 21:27] = False
        volume[0:2, 37:43, 53:59] = False
        volume[0:3, 5:11, 37:43] = False
        volume[0:3, 6:10, 38:42] = True

        workspace_tmp = Path(__file__).resolve().parents[1] / "tmp"
        workspace_tmp.mkdir(exist_ok=True)
        output_directory = workspace_tmp / "defect_classification_test_outputs"
        output_directory.mkdir(exist_ok=True)
        report = run_grid_pattern_analysis(
            slices=volume,
            source_path=Path("synthetic_two_classes.npy"),
            output_directory=output_directory,
            period_slices=8,
            group_number=1,
            threshold=0.5,
            foreground="high",
            opening_size=1,
            grid_rows=4,
            grid_columns=4,
            patch_size=32,
            expected_vote=0.6,
            min_reference_fraction=0.01,
            missing_fraction_threshold=0.6,
            max_present_fraction_of_expected=0.45,
            similarity_threshold=0.55,
            spatial_tolerance_pixels=0,
            min_track_slices=4,
            track_radius_cells=0.25,
            confidence_threshold=0.7,
            roi_presence_fraction=0.01,
            broken_min_track_slices=2,
            comparison_mode="fixed_cells",
            tilt_correction=False,
            known_tilt_angle_degrees=None,
        )

        self.assertEqual(report["confirmed_missing_strut_count"], 1)
        self.assertEqual(report["confirmed_broken_strut_count"], 1)
        self.assertEqual(report["confirmed_defect_count"], 2)
        labels = np.load(
            report["outputs"]["defect_class_labels_npy"],
            allow_pickle=False,
        )
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertTrue(
            Path(
                report["outputs"]["segmented_step5_viewer_npy"]
            ).is_file()
        )
        self.assertTrue(
            Path(report["outputs"]["confirmed_patterns_csv"]).is_file()
        )

    def test_touching_circle_and_ellipse_patterns_count_separately(self):
        yy, xx = np.ogrid[:64, :64]
        touching_circles = (
            (yy - 32) ** 2 + (xx - 25) ** 2 <= 9**2
        ) | (
            (yy - 32) ** 2 + (xx - 39) ** 2 <= 9**2
        )
        ellipse_pair = (
            ((yy - 31) / 7) ** 2 + ((xx - 23) / 11) ** 2
            <= 1
        ) | (
            ((yy - 33) / 9) ** 2 + ((xx - 40) / 7) ** 2
            <= 1
        )
        straight_line = (
            (yy >= 29)
            & (yy <= 34)
            & (xx >= 5)
            & (xx <= 58)
        )

        self.assertEqual(_count_pattern_instances(touching_circles), 2)
        self.assertEqual(_count_pattern_instances(ellipse_pair), 2)
        self.assertEqual(_count_pattern_instances(straight_line), 1)

    def test_same_phase_registration_estimates_and_removes_stack_tilt(self):
        base = np.zeros((64, 64), dtype=np.uint8)
        base[8:15, 11:20] = 1
        base[27:39, 42:49] = 1
        base[48:54, 18:31] = 1
        period = 4
        volume = []
        for slice_index in range(12):
            cycle = slice_index // period
            shifted = np.zeros_like(base)
            dy = 4 * cycle
            dx = -2 * cycle
            source_y0, source_y1 = 0, base.shape[0] - dy
            target_y0, target_y1 = dy, base.shape[0]
            source_x0, source_x1 = -dx, base.shape[1]
            target_x0, target_x1 = 0, base.shape[1] + dx
            shifted[
                target_y0:target_y1,
                target_x0:target_x1,
            ] = base[
                source_y0:source_y1,
                source_x0:source_x1,
            ]
            volume.append(shifted)
        cache = MaskCache(
            np.stack(volume),
            threshold=0.5,
            foreground="high",
            opening_size=1,
        )

        tilt = _estimate_stack_tilt(
            cache,
            scope_start=0,
            scope_end=4,
            period_slices=period,
            roi=(0, 64, 0, 64),
            grid_rows=4,
            grid_columns=4,
            sample_count=4,
            maximum_shift_per_period_fraction=0.75,
        )

        self.assertTrue(tilt["applied"])
        self.assertAlmostEqual(
            tilt["shift_y_pixels_per_period"],
            4.0,
            delta=0.6,
        )
        self.assertAlmostEqual(
            tilt["shift_x_pixels_per_period"],
            -2.0,
            delta=0.6,
        )
        candidate = {
            "centroid_y": 24.0 / 16.0,
            "centroid_x": 18.0 / 16.0,
            "centroid_original_pixels": {"y": 24.0, "x": 18.0},
            "grid_row": 1,
            "grid_column": 1,
            "box_bounds_original_pixels": {
                "y_start": 20,
                "y_end_exclusive": 28,
                "x_start": 14,
                "x_end_exclusive": 22,
            },
        }
        _apply_tilt_correction(
            [candidate],
            slice_index=4,
            reference_index=0,
            tilt=tilt,
            roi=(0, 64, 0, 64),
            grid_rows=4,
            grid_columns=4,
        )
        self.assertAlmostEqual(candidate["centroid_y"], 20.0 / 16.0)
        self.assertAlmostEqual(candidate["centroid_x"], 20.0 / 16.0)

    def test_forward_prediction_follows_tilted_pattern_across_gap(self):
        observations = {
            0: [
                {
                    "slice_index_zero_based": 0,
                    "centroid_y": 1.0,
                    "centroid_x": 2.0,
                }
            ],
            1: [
                {
                    "slice_index_zero_based": 1,
                    "centroid_y": 1.4,
                    "centroid_x": 2.2,
                }
            ],
            4: [
                {
                    "slice_index_zero_based": 4,
                    "centroid_y": 2.6,
                    "centroid_x": 2.8,
                }
            ],
        }

        last_position_tracks = _track_all(
            observations,
            radius=0.5,
            maximum_gap=4,
            forward_prediction=False,
        )
        predicted_tracks = _track_all(
            observations,
            radius=0.5,
            maximum_gap=4,
            forward_prediction=True,
            prediction_history=4,
        )

        self.assertEqual(len(last_position_tracks), 2)
        self.assertEqual(len(predicted_tracks), 1)
        self.assertAlmostEqual(
            predicted_tracks[0][-1]["tracking_prediction"][
                "residual_cells"
            ],
            0.0,
            places=6,
        )

    def test_depth_varying_registration_removes_nonlinear_tilt(self):
        base = np.zeros((80, 80), dtype=np.uint8)
        base[13:21, 17:28] = 1
        base[34:47, 50:58] = 1
        base[56:63, 25:42] = 1
        period = 4
        true_offsets: list[tuple[int, int]] = []
        volume = []
        for slice_index in range(28):
            dy = int(round(0.08 * slice_index + 0.012 * slice_index**2))
            dx = int(round(-0.04 * slice_index - 0.006 * slice_index**2))
            true_offsets.append((dy, dx))
            volume.append(
                np.roll(
                    base,
                    shift=(dy, dx),
                    axis=(0, 1),
                )
            )
        cache = MaskCache(
            np.stack(volume),
            threshold=0.5,
            foreground="high",
            opening_size=1,
        )
        tilt = _estimate_stack_tilt(
            cache,
            scope_start=8,
            scope_end=20,
            period_slices=period,
            roi=(0, 80, 0, 80),
            grid_rows=4,
            grid_columns=4,
            sample_count=24,
            maximum_shift_per_period_fraction=0.75,
        )

        self.assertTrue(tilt["applied"])
        self.assertEqual(tilt["model"], "piecewise_depth_varying_rate")
        reference_index = 8
        target_index = 19
        reference_offset = true_offsets[reference_index]
        target_offset = true_offsets[target_index]
        original_y = 40.0 + target_offset[0] - reference_offset[0]
        original_x = 40.0 + target_offset[1] - reference_offset[1]
        candidate = {
            "centroid_y": original_y / 20.0,
            "centroid_x": original_x / 20.0,
            "centroid_original_pixels": {
                "y": original_y,
                "x": original_x,
            },
            "grid_row": 2,
            "grid_column": 2,
            "box_bounds_original_pixels": {
                "y_start": 35,
                "y_end_exclusive": 45,
                "x_start": 35,
                "x_end_exclusive": 45,
            },
        }
        _apply_tilt_correction(
            [candidate],
            slice_index=target_index,
            reference_index=reference_index,
            tilt=tilt,
            roi=(0, 80, 0, 80),
            grid_rows=4,
            grid_columns=4,
        )
        self.assertAlmostEqual(candidate["centroid_y"], 2.0, delta=0.08)
        self.assertAlmostEqual(candidate["centroid_x"], 2.0, delta=0.08)

    def test_acceleration_prediction_keeps_curved_track_through_gap(self):
        observations: dict[int, list[dict]] = {}
        for slice_index in (0, 1, 2, 3, 6):
            observations[slice_index] = [
                {
                    "slice_index_zero_based": slice_index,
                    "centroid_y": 0.075 * slice_index**2,
                    "centroid_x": 2.0,
                }
            ]

        tracks = _track_all(
            observations,
            radius=0.3,
            maximum_gap=4,
            forward_prediction=True,
            prediction_history=6,
        )

        self.assertEqual(len(tracks), 1)
        prediction = tracks[0][-1]["tracking_prediction"]
        self.assertEqual(
            prediction["prediction_model"],
            "robust_constant_acceleration",
        )
        self.assertAlmostEqual(prediction["residual_cells"], 0.0, places=6)

    def test_curved_parallel_tracks_remain_independent_through_gap(self):
        observations: dict[int, list[dict]] = {}
        for slice_index in (0, 1, 2, 3, 6):
            curve = 0.075 * slice_index**2
            pair = [
                {
                    "slice_index_zero_based": slice_index,
                    "centroid_y": curve,
                    "centroid_x": 1.0,
                },
                {
                    "slice_index_zero_based": slice_index,
                    "centroid_y": curve,
                    "centroid_x": 3.0,
                },
            ]
            observations[slice_index] = (
                pair if slice_index % 2 == 0 else list(reversed(pair))
            )

        tracks = _track_all(
            observations,
            radius=0.3,
            maximum_gap=4,
            forward_prediction=True,
            prediction_history=6,
        )

        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(len(track) == 5 for track in tracks))
        self.assertEqual(
            sorted(round(track[-1]["centroid_x"]) for track in tracks),
            [1, 3],
        )

    def test_three_ellipse_losses_in_one_grid_count_as_three(self):
        yy, xx = np.ogrid[:160, :160]
        missing = (
            ((yy - 45) / 8) ** 2 + ((xx - 45) / 6) ** 2
            <= 1
        ) | (
            ((yy - 65) / 7) ** 2 + ((xx - 60) / 5) ** 2
            <= 1
        ) | (
            ((yy - 115) / 10) ** 2 + ((xx - 105) / 14) ** 2
            <= 1
        )

        self.assertEqual(_count_pattern_instances(missing), 3)
        confirmed = [{"pattern_instance_count": 3}]
        self.assertEqual(
            _confirmed_pattern_instance_count(confirmed),
            3,
        )


if __name__ == "__main__":
    unittest.main()
