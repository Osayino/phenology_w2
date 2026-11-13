"""Pixel datasets and reproducible recurrence-image caches for the notebook.

One row always identifies one camera pixel. Raw features are channel-major
(all red days, all green days, all blue days); flattened images use RGB order.
The notebook supplies the recurrence transformation, keeping its scientific
choices visible beside the exercise rather than duplicating them here.
"""

from pathlib import Path
import hashlib
import json
import os
import time
import uuid

import numpy as np


def extract_pixel_features(chromatic_series, masks):
    """Return float32 features, species labels, and stable pixel identities.

    ``identities`` has columns [species_class, roi_id, row, column]. Mask order
    is preserved; within each mask, np.nonzero visits coordinates row by row.
    No region averaging is performed and no observation day is discarded.
    """
    if not chromatic_series or not masks:
        raise ValueError("Supply at least one observation and one ROI mask.")
    n_days = len(chromatic_series)
    image_shape = np.asarray(chromatic_series[0][0]).shape
    if len(image_shape) != 2:
        raise ValueError("Each chromatic channel must be a 2D image.")
    for channels in chromatic_series:
        if len(channels) != 3 or any(np.shape(c) != image_shape for c in channels):
            raise ValueError("All observations must have three matching image channels.")

    positions = []
    for roi in masks:
        mask = np.asarray(roi["mask"], dtype=bool)
        if mask.shape != image_shape:
            raise ValueError("ROI masks must match the image dimensions.")
        rows, columns = np.nonzero(mask)
        if not len(rows):
            raise ValueError(f"ROI {roi['roi_id']} contains no pixels.")
        positions.append((roi, rows, columns))

    n_pixels = sum(len(rows) for _, rows, _ in positions)
    X = np.empty((n_pixels, 3 * n_days), dtype=np.float32)
    y = np.empty(n_pixels, dtype=np.int64)
    identities = np.empty((n_pixels, 4), dtype=np.int64)
    start = 0
    for roi, rows, columns in positions:
        stop = start + len(rows)
        for day, channels in enumerate(chromatic_series):
            for channel, values in enumerate(channels):
                X[start:stop, channel * n_days + day] = values[rows, columns]
        y[start:stop] = roi["species_class"]
        identities[start:stop, 0] = roi["species_class"]
        identities[start:stop, 1] = roi["roi_id"]
        identities[start:stop, 2] = rows
        identities[start:stop, 3] = columns
        start = stop

    if not np.isfinite(X).all():
        raise ValueError("Pixel features contain non-finite values.")
    if len(np.unique(identities[:, 2:4], axis=0)) != n_pixels:
        raise ValueError("Selected ROI masks contain duplicate pixel coordinates.")
    return X, y, identities


def save_pixel_dataset(path, *, X_trainval, y_trainval, ids_trainval,
                       X_test, y_test, ids_test, train_indices, val_indices,
                       doys, random_state):
    """Validate and atomically save the complete dataset and one shared split.

    Load with ``np.load(path, allow_pickle=False)``. Both representations must
    use the saved train_indices and val_indices against the same row order.
    """
    arrays = {name: np.asarray(value) for name, value in {
        "X_trainval": X_trainval, "y_trainval": y_trainval,
        "ids_trainval": ids_trainval, "X_test": X_test, "y_test": y_test,
        "ids_test": ids_test, "train_indices": train_indices,
        "val_indices": val_indices, "doys": doys,
    }.items()}
    for suffix in ("trainval", "test"):
        X, y, ids = (arrays[f"{prefix}_{suffix}"] for prefix in ("X", "y", "ids"))
        if X.ndim != 2 or X.dtype != np.float32 or not np.isfinite(X).all():
            raise ValueError("Features must be finite 2D float32 arrays.")
        if y.shape != (len(X),) or ids.shape != (len(X), 4):
            raise ValueError("Labels and identities must align with feature rows.")
        if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Labels and identities must be integer arrays.")
        if not np.array_equal(y, ids[:, 0]):
            raise ValueError("Species labels disagree with pixel identities.")
    days = arrays["doys"]
    if days.ndim != 1 or len(days) < 2 or not np.all(np.diff(days) > 0):
        raise ValueError("Observation days must be in strictly increasing order.")
    if arrays["X_trainval"].shape[1] != 3 * len(days) or arrays["X_test"].shape[1] != 3 * len(days):
        raise ValueError("Feature columns must match three complete colour histories.")
    indices = [arrays["train_indices"], arrays["val_indices"]]
    if any(x.ndim != 1 or not np.issubdtype(x.dtype, np.integer) for x in indices):
        raise ValueError("Split indices must be 1D integer arrays.")
    if any(len(x) == 0 for x in indices):
        raise ValueError("Both training and validation need samples.")
    if not np.array_equal(np.sort(np.concatenate(indices)), np.arange(len(arrays["X_trainval"]))):
        raise ValueError("Training/validation indices must partition every training-ROI row once.")
    coordinates = np.concatenate((arrays["ids_trainval"][:, 2:4], arrays["ids_test"][:, 2:4]))
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("A pixel coordinate occurs more than once across the selected ROIs.")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays are not permitted in the saved dataset.")

    metadata = {"schema_version": 1, "random_state": int(random_state),
                "raw_feature_order": "r_cc over days, then g_cc, then b_cc",
                "identity_columns": ["species_class", "roi_id", "row", "column"],
                "split_unit": "individual training-ROI pixel",
                "test_selection": "separate assigned test ROIs"}
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_recurrence_cache(X, cache_dir, name, create_rgb_recurrence_plot, *,
                           transform_signature, target_length=140, progress=False):
    """Return a read-only uint8 memmap and metadata for all pixel images.

    ``transform_signature`` must describe the actual transformation code and
    settings, including its helper functions. The notebook passes their source
    and library versions. Input bytes, input shape/dtype, signature and output
    dimensions jointly determine the cache key. Existing data is checksum-
    verified before reuse. A partial generation never becomes a valid cache.

    Only one RGB image is generated at a time; the full array stays on disk.
    The callback must return (target_length, target_length, 3) uint8 images.
    """
    X = np.asarray(X)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] % 3 or not np.isfinite(X).all():
        raise ValueError("Supply finite pixel features with three channel-major histories.")
    if X.dtype.hasobject or not isinstance(transform_signature, str) or not transform_signature:
        raise ValueError("Supply numeric features and a nonempty transformation signature.")
    if not isinstance(target_length, int) or target_length < 2:
        raise ValueError("target_length must be an integer of at least two.")
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Cache names may contain letters, digits, underscores and hyphens.")
    shape = (len(X), target_length * target_length * 3)
    recipe = {"schema_version": 1, "input_shape": list(X.shape), "input_dtype": str(X.dtype),
              "output_shape": list(shape), "output_dtype": "uint8",
              "target_length": target_length, "flattening_order": "C order: time_i, time_j, RGB",
              "transform_signature": transform_signature}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode("utf-8"))
    for start in range(0, len(X), 1024):
        digest.update(np.ascontiguousarray(X[start:start + 1024]).tobytes())
    key = digest.hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{name}_{key}.npy"
    metadata_path = data_path.with_suffix(".json")
    if data_path.exists() and metadata_path.exists():
        cached = None
        try:
            info = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(data_path, mmap_mode="r", allow_pickle=False)
            valid = (info.get("cache_key") == key and info.get("recipe") == recipe
                     and cached.shape == shape and cached.dtype == np.uint8
                     and info.get("file_sha256") == _file_sha256(data_path))
            if valid:
                return cached, {**info, "path": str(data_path), "reused": True}
        except (ValueError, OSError, KeyError):
            pass
        if cached is not None:
            cached._mmap.close()

    token = uuid.uuid4().hex
    temporary_data = cache_dir / f".{name}_{token}.tmp.npy"
    temporary_metadata = cache_dir / f".{name}_{token}.tmp.json"
    mapped = None
    started = time.perf_counter()
    try:
        mapped = np.lib.format.open_memmap(temporary_data, mode="w+", dtype=np.uint8, shape=shape)
        n_days = X.shape[1] // 3
        for row in range(len(X)):
            channels = X[row].reshape(3, n_days)
            image = np.asarray(create_rgb_recurrence_plot(*channels))
            if image.shape != (target_length, target_length, 3) or image.dtype != np.uint8:
                raise ValueError(f"Recurrence image {row} has an unexpected shape or dtype.")
            if not np.array_equal(image, image.transpose(1, 0, 2)) or np.any(np.diagonal(image, axis1=0, axis2=1)):
                raise ValueError(f"Recurrence image {row} is not symmetric with a zero diagonal.")
            mapped[row] = image.reshape(-1)
            if progress and ((row + 1) % 2000 == 0 or row + 1 == len(X)):
                print(f"{name}: generated {row + 1:,}/{len(X):,} pixel recurrence images", flush=True)
        mapped.flush()
        mapped._mmap.close()
        mapped = None
        info = {"cache_key": key, "recipe": recipe,
                "generation_seconds": time.perf_counter() - started,
                "array_bytes": int(np.prod(shape)), "file_sha256": _file_sha256(temporary_data)}
        temporary_metadata.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_data, data_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        if mapped is not None:
            mapped._mmap.close()
        temporary_data.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return np.load(data_path, mmap_mode="r", allow_pickle=False), {**info, "path": str(data_path), "reused": False}
