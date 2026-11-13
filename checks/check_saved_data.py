"""Audit saved pixel features independently against masks and original JPEGs.

Run after the notebook creates results/pixel_dataset.npz. This check does not
change the dataset or load the large recurrence cache.
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ["A.tomentosum", "C.brasiliensis", "M.guianensis",
            "M.rubiginosa", "P.ramiflora", "P.torta"]
with np.load(ROOT / "results/pixel_dataset.npz", allow_pickle=False) as saved:
    data = {name: saved[name] for name in saved.files}

X, y, ids = (data[name] for name in ("X_trainval", "y_trainval", "ids_trainval"))
Xt, yt, idt = (data[name] for name in ("X_test", "y_test", "ids_test"))
train, validation, days = (data[name] for name in ("train_indices", "val_indices", "doys"))
metadata = json.loads(str(data["metadata_json"]))
assert metadata["random_state"] == 42
assert X.shape == (13745, 111) and Xt.shape == (14105, 111)
assert X.dtype == Xt.dtype == np.float32
assert y.dtype == yt.dtype == ids.dtype == idt.dtype == np.int64
assert np.array_equal(days, np.arange(242, 279))
assert np.array_equal(y, ids[:, 0]) and np.array_equal(yt, idt[:, 0])
assert np.array_equal(np.unique(y), np.arange(6)) and np.array_equal(np.unique(yt), np.arange(6))
assert len(train) == 10996 and len(validation) == 2749
assert np.array_equal(np.sort(np.concatenate([train, validation])), np.arange(len(y)))
coordinates = np.concatenate([ids[:, 2:4], idt[:, 2:4]])
assert len(np.unique(coordinates, axis=0)) == len(coordinates)
for features in (X, Xt):
    assert np.isfinite(features).all() and np.all((features >= 0) & (features <= 1))
    sums = features.reshape(-1, 3, 37).sum(axis=1)
    assert np.all(np.isclose(sums, 1, atol=1e-6) | (sums == 0))

expected_train, expected_validation = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)
np.testing.assert_array_equal(train, expected_train)
np.testing.assert_array_equal(validation, expected_validation)
np.testing.assert_array_equal(np.unique(ids[:, :2], axis=0),
                              [[0, 5], [1, 31], [2, 1], [3, 4], [4, 8], [5, 9]])
np.testing.assert_array_equal(np.unique(idt[:, :2], axis=0),
                              [[0, 7], [1, 29], [2, 21], [3, 32], [4, 10], [5, 3]])
for identities in (ids, idt):
    for species, roi in np.unique(identities[:, :2], axis=0):
        actual = identities[(identities[:, 0] == species) & (identities[:, 1] == roi), 2:4]
        with Image.open(ROOT / "raw/masks" / f"{PREFIXES[species]}_{roi}.pgm") as mask:
            expected = np.argwhere(np.asarray(mask) > 0)
        np.testing.assert_array_equal(actual, expected)

# Choose first/middle/last pixels of every species in each spatial partition.
sample_ids, expected_rows = [], []
for features, labels, identities in ((X, y, ids), (Xt, yt, idt)):
    for species in range(6):
        indices = np.flatnonzero(labels == species)
        for row in indices[[0, len(indices) // 2, len(indices) - 1]]:
            sample_ids.append(identities[row])
            expected_rows.append(features[row])
sample_ids = np.array(sample_ids)
independent = np.empty((len(sample_ids), 3, len(days)), dtype=np.float32)
for time_index, day in enumerate(days):
    with Image.open(ROOT / "raw/images/1" / f"2011_{day}_12.jpg") as photo:
        rgb = np.asarray(photo)[sample_ids[:, 2], sample_ids[:, 3]].astype(np.float32)
    total = rgb[:, 0] + rgb[:, 1] + rgb[:, 2]
    total[total == 0] = 1
    independent[:, :, time_index] = rgb / total[:, None]
np.testing.assert_array_equal(independent.reshape(len(sample_ids), -1), expected_rows)

print("PASS: saved sizes, all 37 days, six species, exact spatial masks and identity/label alignment.")
print("PASS: unique camera coordinates and reproducible stratified 10,996/2,749 split; 14,105 test pixels.")
print("PASS: 36 sampled pixels reconstructed from original JPEGs across all days, exact float32 equality.")
