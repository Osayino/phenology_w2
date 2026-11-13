"""Check the notebook's actual interpolation and RGB recurrence functions.

Run from any directory with the project's Python interpreter. These checks use
known signals and independent analytic expectations; they do not run the whole
notebook or modify its data, outputs, or caches.
"""

from pathlib import Path
import ast
import json

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "casework/case_notebook.ipynb").read_bytes())
required = {"upsample_time_series", "compute_recurrence_plot", "create_rgb_recurrence_plot"}
loaded = set()
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.FunctionDef) and node.name in required:
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         "<saved notebook function>", "exec"), globals())
            loaded.add(node.name)
assert loaded == required, f"Missing notebook functions: {required - loaded}"

# Linear interpolation preserves endpoints and reproduces a known straight line.
np.testing.assert_allclose(
    upsample_time_series(np.array([0.2, 0.6]), target_length=5),
    [0.2, 0.3, 0.4, 0.5, 0.6], rtol=0, atol=1e-15,
)
daily_signal = np.linspace(0.1, 0.7, 37)
original = daily_signal.copy()
interpolated = upsample_time_series(daily_signal)
assert interpolated.shape == (140,)
np.testing.assert_allclose(interpolated, np.linspace(0.1, 0.7, 140), rtol=0, atol=1e-15)
np.testing.assert_array_equal(interpolated[[0, -1]], daily_signal[[0, -1]])
np.testing.assert_array_equal(daily_signal, original)
np.testing.assert_array_equal(upsample_time_series(np.full(37, 0.4)), np.full(140, 0.4))
np.testing.assert_allclose(
    upsample_time_series(np.array([0.0, 1.0, 0.0]), target_length=5),
    [0.0, 0.5, 1.0, 0.5, 0.0], rtol=0, atol=0,
)

# A hand-calculated distance matrix detects thresholding and squared distances.
signal = np.array([-2.0, 0.5, 1.75])
expected_distance = np.array([[0, 2.5, 3.75], [2.5, 0, 1.25], [3.75, 1.25, 0]])
distance = compute_recurrence_plot(signal)
np.testing.assert_array_equal(distance, expected_distance)
np.testing.assert_array_equal(distance, distance.T)
np.testing.assert_array_equal(np.diag(distance), np.zeros(3))
np.testing.assert_array_equal(compute_recurrence_plot(np.full(37, 0.4)), np.zeros((37, 37)))

# Distinct analytic channel shapes expose channel swaps and shared-max scaling.
# Red increases by only 0.1, green rises then falls by 0.8, blue is constant.
channels = (np.array([0.2, 0.3]), np.array([0.0, 0.8, 0.0]), np.array([0.4, 0.4]))
unchanged = tuple(channel.copy() for channel in channels)
image = create_rgb_recurrence_plot(*channels)
assert image.shape == (140, 140, 3) and image.dtype == np.uint8
positions = np.linspace(0, 1, 140)
triangle = 1 - np.abs(2 * positions - 1)
expected_red = np.round(255 * np.abs(positions[:, None] - positions[None, :])).astype(np.uint8)
expected_green = np.round(
    255 * np.abs(triangle[:, None] - triangle[None, :]) / triangle.max()
).astype(np.uint8)
np.testing.assert_array_equal(image[:, :, 0], expected_red)
np.testing.assert_array_equal(image[:, :, 1], expected_green)
np.testing.assert_array_equal(image[:, :, 2], np.zeros((140, 140), dtype=np.uint8))
assert image[0, -1, 0] == 255 and image[0, 69, 1] == 255
np.testing.assert_array_equal(image, image.transpose(1, 0, 2))
assert not np.any(np.diagonal(image, axis1=0, axis2=1))
for before, after in zip(unchanged, channels):
    np.testing.assert_array_equal(before, after)
constant_image = create_rgb_recurrence_plot(*[np.full(37, value) for value in (0.1, 0.6, 0.3)])
np.testing.assert_array_equal(constant_image, np.zeros((140, 140, 3), dtype=np.uint8))

print("PASS: actual notebook functions; 37-to-140 interpolation, endpoints, known linear/turning signals,")
print("constant handling, unthresholded Euclidean distances, symmetry, zero diagonals, RGB order,")
print("independent channel normalization, uint8 shape, and preservation of input arrays.")
