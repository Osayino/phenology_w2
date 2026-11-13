"""Check spatial features against explicit reflected windows, without data writes.

Run with the project's Python interpreter. The checks use only synthetic RGB
images, including an in-memory two-day photograph loader; no labels, ROI masks,
notebooks, real photographs, or saved experiment results are loaded or changed.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.spatial_features import extract_features, extract_image_features


def explicit_windows(rgb, coordinates, window):
    """Independent reference: pad the image, then directly inspect each square."""
    normalized = rgb.astype(np.float64) / 255.0
    radius = window // 2
    # NumPy 'symmetric', unlike NumPy 'reflect', repeats each boundary pixel.
    padded = np.pad(normalized, ((radius, radius), (radius, radius), (0, 0)), mode="symmetric")
    expected = []
    for row, column in coordinates:
        square = padded[row:row + window, column:column + window, :]
        expected.append(np.concatenate((
            normalized[row, column], square.mean(axis=(0, 1)), square.std(axis=(0, 1), ddof=0),
        )))
    return np.asarray(expected, dtype=np.float64).reshape(len(coordinates), 9)


class SpatialFeatureChecks(unittest.TestCase):
    def test_random_windows_edges_duplicates_and_unchanged_inputs(self):
        rgb = np.random.default_rng(72).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
        coordinates = np.array([[5, 7], [0, 0], [2, 3], [0, 7], [5, 0], [2, 3]])
        original_rgb, original_coordinates = rgb.copy(), coordinates.copy()
        rgb.flags.writeable = False
        coordinates.flags.writeable = False
        windows = [1, 3, 5, 25]
        actual = extract_image_features(rgb, coordinates, windows)
        self.assertEqual(set(actual), {"raw_rgb", "context_1", "context_3", "context_5", "context_25"})
        for window in windows:
            with self.subTest(window=window):
                np.testing.assert_allclose(actual[f"context_{window}"], explicit_windows(rgb, coordinates, window),
                                           rtol=2e-7, atol=2e-8)
                self.assertEqual(actual[f"context_{window}"].dtype, np.float32)
                np.testing.assert_array_equal(actual[f"context_{window}"][:, :3], actual["raw_rgb"])
        self.assertEqual(actual["raw_rgb"].dtype, np.float32)
        np.testing.assert_array_equal(rgb, original_rgb)
        np.testing.assert_array_equal(coordinates, original_coordinates)
        self.assertEqual(windows, [1, 3, 5, 25])

    def test_reflection_repeats_boundary_and_uses_population_std(self):
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 90, 18]
        actual = extract_image_features(rgb, [[0, 0]], (3,))["context_3"][0]
        # Reflection places the corner pixel four times in its nine-pixel window.
        amplitudes = np.array([255, 90, 18], dtype=np.float64) / 255
        np.testing.assert_allclose(actual[3:6], amplitudes * 4 / 9, rtol=1e-7)
        np.testing.assert_allclose(actual[6:9], amplitudes * np.sqrt(20) / 9, rtol=1e-7)

    def test_constant_channels_and_single_pixel_image(self):
        for shape in ((7, 9, 3), (1, 1, 3)):
            rgb = np.broadcast_to(np.array([0, 127, 255], dtype=np.uint8), shape)
            coordinates = np.array([[0, 0], [shape[0] - 1, shape[1] - 1]])
            actual = extract_image_features(rgb, coordinates)
            for window in (5, 25):
                values = actual[f"context_{window}"]
                np.testing.assert_array_equal(values[:, :3], values[:, 3:6])
                np.testing.assert_allclose(values[:, 6:9], 0, rtol=0, atol=2e-8)
                self.assertTrue(np.isfinite(values).all())

    def test_temporal_channel_order_and_requested_paths_without_disk_writes(self):
        generator = np.random.default_rng(17)
        images = {day: generator.integers(0, 256, size=(4, 5, 3), dtype=np.uint8) for day in (244, 242)}
        original_images = {day: rgb.copy() for day, rgb in images.items()}
        coordinates = np.array([[3, 4], [0, 0], [2, 1]])
        days = [244, 242]
        expected_paths = [ROOT / "raw" / "images" / "1" / f"2011_{day}_12.jpg" for day in days]

        def open_synthetic(path):
            self.assertIn(path, expected_paths)
            return Image.fromarray(images[days[expected_paths.index(path)]])

        with patch("experiments.spatial_features.Image.open", side_effect=open_synthetic) as loader:
            actual = extract_features(ROOT, coordinates, days)
        self.assertEqual([call.args[0] for call in loader.call_args_list], expected_paths)
        for window in (5, 25):
            by_day = [explicit_windows(images[day], coordinates, window) for day in days]
            expected = np.column_stack([values[:, column] for column in range(9) for values in by_day])
            np.testing.assert_allclose(actual[f"context_{window}"], expected, rtol=2e-7, atol=2e-8)
            self.assertEqual(actual[f"context_{window}"].shape, (3, 18))
            np.testing.assert_array_equal(actual[f"context_{window}"][:, :6], actual["raw_rgb"])
        self.assertEqual(actual["raw_rgb"].shape, (3, 6))
        self.assertTrue(all(values.dtype == np.float32 for values in actual.values()))
        self.assertEqual(days, [244, 242])
        for day in days:
            np.testing.assert_array_equal(images[day], original_images[day])

    def test_invalid_inputs_fail_before_photograph_access(self):
        coordinates = np.array([[0, 0]])
        bad_days = ([], [0], [366], [242, 242], [242.0], [True], [[242]])
        bad_windows = ([], [0], [-3], [2], [5, 5], [3.0], [True], [[3]])
        bad_coordinates = ([0, 0], [[0, 0, 0]], [[-1, 0]], [[0.5, 0]], [[False, True]])
        with patch("experiments.spatial_features.Image.open") as loader:
            for days in bad_days:
                with self.subTest(days=days), self.assertRaises(ValueError):
                    extract_features(ROOT, coordinates, days)
            for windows in bad_windows:
                with self.subTest(windows=windows), self.assertRaises(ValueError):
                    extract_features(ROOT, coordinates, [242], windows)
            for invalid in bad_coordinates:
                with self.subTest(coordinates=invalid), self.assertRaises(ValueError):
                    extract_features(ROOT, invalid, [242])
            loader.assert_not_called()

    def test_image_shape_dtype_and_coordinate_bounds(self):
        valid = np.zeros((4, 5, 3), dtype=np.uint8)
        for invalid in (np.zeros((4, 5), dtype=np.uint8), np.zeros((4, 5, 4), dtype=np.uint8),
                        np.zeros((0, 5, 3), dtype=np.uint8), valid.astype(np.float32)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ValueError):
                extract_image_features(invalid, [[0, 0]])
        for coordinates in ([[4, 0]], [[0, 5]], np.array([[2**64 - 1, 0]], dtype=np.uint64)):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                extract_image_features(valid, coordinates)
        empty = extract_image_features(valid, np.empty((0, 2), dtype=np.int64))
        self.assertEqual(empty["raw_rgb"].shape, (0, 3))
        self.assertEqual(empty["context_25"].shape, (0, 9))

    def test_temporal_dimension_mismatch_fails(self):
        with patch("experiments.spatial_features.Image.open", side_effect=[
            Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)),
            Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)),
        ]), self.assertRaisesRegex(ValueError, "dimensions changed"):
            extract_features(ROOT, [[0, 0]], [242, 243])


if __name__ == "__main__":
    unittest.main(verbosity=2)
