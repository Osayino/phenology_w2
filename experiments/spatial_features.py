"""Extract RGB histories and spatial context without reading labels or ROI masks.

Every spatial statistic uses the original photograph. Reflection at the image
edge repeats the edge pixel (SciPy's ``reflect`` mode), and standard deviation
is the population value, with no degrees-of-freedom correction. Only one day's
photograph and one channel's filtering arrays are held at a time.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def _validated_coordinates(coordinates, image_shape=None):
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2): row, column.")
    if coordinates.dtype.kind not in "iu":
        raise ValueError("coordinates must contain integers, not floats or booleans.")
    if np.any(coordinates < 0):
        raise ValueError("coordinates must be non-negative.")
    if image_shape is not None:
        if np.any(coordinates[:, 0] >= image_shape[0]) or np.any(
            coordinates[:, 1] >= image_shape[1]
        ):
            raise ValueError("coordinates fall outside the photograph.")
    return coordinates


def _validated_windows(window_sizes):
    windows = np.asarray(window_sizes)
    if windows.ndim != 1 or len(windows) == 0 or windows.dtype.kind not in "iu":
        raise ValueError("window_sizes must be a non-empty sequence of integers.")
    if np.any(windows < 1) or np.any(windows % 2 != 1):
        raise ValueError("window_sizes must contain positive odd integers.")
    if len(np.unique(windows)) != len(windows):
        raise ValueError("window_sizes must not contain duplicates.")
    return tuple(int(window) for window in windows)


def _validated_days(doys):
    days = np.asarray(doys)
    if days.ndim != 1 or len(days) == 0 or days.dtype.kind not in "iu":
        raise ValueError("doys must be a non-empty sequence of integer days.")
    if np.any(days < 1) or np.any(days > 365):
        raise ValueError("doys must be within 1 through 365 for the source year 2011.")
    if len(np.unique(days)) != len(days):
        raise ValueError("doys must not contain duplicates.")
    return tuple(int(day) for day in days)


def extract_image_features(image_rgb, coordinates, window_sizes=(5, 25)):
    """Return raw RGB and local RGB mean/std features for one uint8 image.

    ``coordinates`` has integer columns (row, column); its order and duplicates
    are preserved. The output contains float32 ``raw_rgb`` of shape (n, 3) and
    ``context_<window>`` of shape (n, 9), ordered as raw R/G/B, mean R/G/B,
    then standard deviation R/G/B. All values refer to RGB intensities / 255.
    The image and coordinate arrays are never modified.
    """
    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("image_rgb must have non-empty shape (height, width, 3).")
    if rgb.dtype != np.uint8:
        raise ValueError("image_rgb must contain uint8 RGB intensities.")
    coordinates = _validated_coordinates(coordinates, rgb.shape)
    windows = _validated_windows(window_sizes)
    rows, columns = coordinates[:, 0], coordinates[:, 1]
    raw = rgb[rows, columns].astype(np.float32) / np.float32(255.0)
    features = {"raw_rgb": raw}
    for window in windows:
        context = np.empty((len(coordinates), 9), dtype=np.float32)
        context[:, :3] = raw
        features[f"context_{window}"] = context

    # Filter full photographs, so context does not depend on a mask or a label.
    # Float64 moments avoid the integer truncation of filtering uint8 directly.
    for channel_index in range(3):
        channel = rgb[:, :, channel_index].astype(np.float64) / 255.0
        squared = channel * channel
        for window in windows:
            mean_image = uniform_filter(channel, size=window, mode="reflect")
            second_moment_image = uniform_filter(squared, size=window, mode="reflect")
            mean = mean_image[rows, columns]
            variance = second_moment_image[rows, columns] - mean * mean
            # Round-off in E[X^2] - E[X]^2 can make a zero variance negative.
            np.maximum(variance, 0.0, out=variance)
            np.sqrt(variance, out=variance)
            context = features[f"context_{window}"]
            context[:, 3 + channel_index] = mean
            context[:, 6 + channel_index] = variance
            del mean_image, second_moment_image
    return features


def extract_features(project_root, coordinates, doys, window_sizes=(5, 25), progress=False):
    """Return channel-major temporal RGB and spatial-context features.

    Photographs are read from ``raw/images/1/2011_<day>_12.jpg`` beneath
    ``project_root``. ``coordinates`` contains (row, column), with no class or
    ROI columns. Days retain their supplied order and must be unique integers.

    For n coordinates and T days, float32 ``raw_rgb`` has shape (n, 3*T): all
    red days, then green, then blue. Each ``context_<window>`` has shape (n, 9*T)
    and concatenates three channel-major blocks: raw RGB, local mean RGB, and
    local population-standard-deviation RGB. The defaults produce context_5
    and context_25. This function neither fits models nor saves any artifacts.
    """
    coordinates = _validated_coordinates(coordinates)
    days = _validated_days(doys)
    windows = _validated_windows(window_sizes)
    root = Path(project_root)
    histories = {"raw_rgb": np.empty((len(coordinates), 3, len(days)), dtype=np.float32)}
    histories.update({
        f"context_{window}": np.empty((len(coordinates), 9, len(days)), dtype=np.float32)
        for window in windows
    })
    image_shape = None
    for day_index, day in enumerate(days):
        path = root / "raw" / "images" / "1" / f"2011_{day}_12.jpg"
        with Image.open(path) as photograph:
            rgb = np.asarray(photograph.convert("RGB"))
        if image_shape is not None and rgb.shape != image_shape:
            raise ValueError(f"Photograph dimensions changed on day {day}.")
        image_shape = rgb.shape
        daily_features = extract_image_features(rgb, coordinates, windows)
        for name, values in daily_features.items():
            histories[name][:, :, day_index] = values
        if progress:
            print(f"Spatial features: day {day} ({day_index + 1}/{len(days)})", flush=True)
        del rgb, daily_features
    return {
        name: values.reshape(len(coordinates), values.shape[1] * len(days))
        for name, values in histories.items()
    }
