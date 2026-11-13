"""Small independent checks for pixel order, shared splits and disk caches."""
from pathlib import Path
import ast
import json
import sys
import tempfile

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "casework"))
from phenology_data import extract_pixel_features, save_pixel_dataset, build_recurrence_cache

# Different colours, days and coordinates make accidental axis swaps visible.
chromatic_series = []
for day in range(3):
    r = (0.1 + np.arange(6).reshape(2, 3) * 0.01 + day * 0.02).astype(np.float32)
    g = (0.3 + np.arange(6).reshape(2, 3) * 0.005 - day * 0.01).astype(np.float32)
    chromatic_series.append((r, g, 1 - r - g))

masks = [
    {"species_class": 1, "roi_id": 8, "mask": np.array([[0, 1, 0], [1, 0, 0]], dtype=bool)},
    {"species_class": 0, "roi_id": 5, "mask": np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)},
]
test_masks = [{"species_class": 1, "roi_id": 10,
               "mask": np.array([[0, 0, 1], [0, 0, 1]], dtype=bool)}]
X, y, identities = extract_pixel_features(chromatic_series, masks)
Xt, yt, idt = extract_pixel_features(chromatic_series, test_masks)
np.testing.assert_array_equal(identities, [[1, 8, 0, 1], [1, 8, 1, 0], [0, 5, 0, 0], [0, 5, 1, 1]])
np.testing.assert_array_equal(y, [1, 1, 0, 0])
for i, (_, _, row, column) in enumerate(identities):
    expected = [chromatic_series[day][channel][row, column] for channel in range(3) for day in range(3)]
    np.testing.assert_array_equal(X[i], expected)
assert X.dtype == np.float32 and y.dtype == identities.dtype == np.int64

# Exercise the real saved notebook functions, not a second implementation.
notebook = json.loads((ROOT / "casework/case_notebook.ipynb").read_bytes())
transform_names = {"upsample_time_series", "compute_recurrence_plot", "create_rgb_recurrence_plot"}
transform_sources = []
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in transform_names:
                transform_sources.append(ast.unparse(node))
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<notebook function>", "exec"), globals())

temporary_root = (ROOT / "tmp").resolve()
assert temporary_root.is_relative_to(ROOT)
temporary_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="pixel_data_check_", dir=temporary_root) as directory:
    directory = Path(directory).resolve()
    assert directory.is_relative_to(temporary_root)
    dataset = save_pixel_dataset(directory / "pixels.npz", X_trainval=X, y_trainval=y,
                                ids_trainval=identities, X_test=Xt, y_test=yt, ids_test=idt,
                                train_indices=np.array([0, 2]), val_indices=np.array([1, 3]),
                                doys=[242, 243, 244], random_state=42)
    with np.load(dataset, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["X_trainval"], X)
        np.testing.assert_array_equal(saved["ids_test"], idt)
        np.testing.assert_array_equal(saved["train_indices"], [0, 2])
        assert json.loads(str(saved["metadata_json"]))["random_state"] == 42

    try:
        save_pixel_dataset(directory / "bad.npz", X_trainval=X, y_trainval=y,
                           ids_trainval=identities, X_test=Xt, y_test=yt, ids_test=idt,
                           train_indices=np.array([0, 1, 2]), val_indices=np.array([2, 3]),
                           doys=[242, 243, 244], random_state=42)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlapping train/validation membership was accepted")

    signature = "\n".join(transform_sources)
    mapped, info = build_recurrence_cache(X, directory, "trainval", create_rgb_recurrence_plot,
                                         transform_signature=signature)
    assert mapped.shape == (4, 58800) and mapped.dtype == np.uint8 and not mapped.flags.writeable
    for i in range(len(X)):
        np.testing.assert_array_equal(mapped[i], create_rgb_recurrence_plot(*X[i].reshape(3, 3)).ravel())
    assert not info["reused"]
    mapped._mmap.close()
    mapped, repeated = build_recurrence_cache(X, directory, "trainval", create_rgb_recurrence_plot,
                                             transform_signature=signature)
    assert repeated["reused"] and repeated["cache_key"] == info["cache_key"]
    mapped._mmap.close()

    # A changed input row or a changed transformation produces a different cache.
    changed = X.copy()
    changed[0, 0] += 0.02
    mapped, changed_info = build_recurrence_cache(changed, directory, "trainval", create_rgb_recurrence_plot,
                                                 transform_signature=signature)
    assert changed_info["cache_key"] != info["cache_key"]
    mapped._mmap.close()
    mapped, new_recipe = build_recurrence_cache(X, directory, "trainval", create_rgb_recurrence_plot,
                                               transform_signature=signature + "\nversion 2")
    assert new_recipe["cache_key"] != info["cache_key"]
    mapped._mmap.close()

    # Metadata corruption must trigger regeneration instead of silent reuse.
    metadata_path = Path(info["path"]).with_suffix(".json")
    metadata_path.write_text('{"cache_key": "incorrect"}', encoding="utf-8")
    mapped, regenerated = build_recurrence_cache(X, directory, "trainval", create_rgb_recurrence_plot,
                                                transform_signature=signature)
    assert not regenerated["reused"]
    mapped._mmap.close()

    # A broken transform must not publish a valid-looking partial dataset.
    try:
        build_recurrence_cache(X, directory, "broken", lambda *channels: np.ones((2, 2)),
                               transform_signature="deliberately invalid output")
    except ValueError:
        pass
    else:
        raise AssertionError("An invalid recurrence-image shape was accepted")
    assert not list(directory.glob("broken*"))
    assert not list(directory.glob("*.tmp.*"))

print("PASS: pixel/channel/day order, identity alignment, shared split validation, pickle-free round-trip,")
print("real-function recurrence equality, read-only memmaps, cache reuse, content/signature invalidation,")
print("and invalid-metadata recovery. No full dataset generated.")
