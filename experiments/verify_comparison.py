"""Read-only checks of the optional model comparison and its frozen selection.

Run after the comparison has saved its JSON and prediction arrays:
    .venv/Scripts/python.exe -B experiments/verify_comparison.py

Metrics are calculated directly from confusion-matrix counts, independently of
the sklearn metric functions used to create the notebook tables. This checker
does not train models, change files or require the large fitted-model artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


METRIC_KEYS = ("Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)")
BASELINE_KEYS = {"raw_knn", "rp_knn", "random_forest"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_content_sha256(path):
    """Hash NPZ member names and exact NPY bytes, independent of ZIP encoding.

    No array is decoded. A version tag, member count, and unsigned 8-byte big-
    endian lengths delimit every UTF-8 name and uncompressed member payload.
    Sorting names makes ZIP member order irrelevant; duplicate names are invalid.
    """
    digest = hashlib.sha256(b"phenology-npz-content-v1\0")
    with zipfile.ZipFile(path) as archive:
        members = sorted(archive.infolist(), key=lambda member: member.filename)
        require(len({member.filename for member in members}) == len(members),
                "The dataset archive contains duplicate member names.")
        digest.update(len(members).to_bytes(8, "big"))
        for member in members:
            name = member.filename.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(member.file_size.to_bytes(8, "big"))
            read_size = 0
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    read_size += len(block)
            require(read_size == member.file_size, "An uncompressed dataset member has the wrong size.")
    return digest.hexdigest()


def _verify_dataset_checksums(path, comparison, frozen):
    """Permit archive metadata changes only with exact member-content evidence."""
    require(comparison["dataset_sha256"] == frozen["dataset_sha256"],
            "Comparison and frozen selection disagree on the original dataset file checksum.")
    file_matches = frozen["dataset_sha256"] == sha256(path)
    content_verified = False
    if "dataset_content_sha256" in comparison or "dataset_content_sha256" in frozen:
        expected = frozen.get("dataset_content_sha256")
        require(isinstance(expected, str) and expected == comparison.get("dataset_content_sha256"),
                "Comparison and frozen selection must contain the same dataset content checksum.")
        require(expected == dataset_content_sha256(path),
                "The current dataset has different NPZ member names or uncompressed payload bytes.")
        content_verified = True
    require(file_matches or content_verified,
            "Dataset file bytes differ and no matching content checksum was recorded in both result files.")
    return file_matches, content_verified


def metrics_from_predictions(y_true, prediction, n_classes=6):
    """Build counts and macro scores, assigning zero to undefined class scores."""
    y_true, prediction = np.asarray(y_true), np.asarray(prediction)
    require(y_true.ndim == 1 and prediction.shape == y_true.shape and len(y_true) > 0,
            "Predictions must be a nonempty vector aligned with the true labels.")
    require(np.issubdtype(y_true.dtype, np.integer) and np.issubdtype(prediction.dtype, np.integer),
            "True and predicted labels must be integer class IDs.")
    require(np.isin(y_true, np.arange(n_classes)).all()
            and np.isin(prediction, np.arange(n_classes)).all(), "A label is outside the declared classes.")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, prediction), 1)
    correct = np.diag(matrix).astype(np.float64)
    actual_counts, predicted_counts = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(correct, predicted_counts, out=np.zeros(n_classes), where=predicted_counts > 0)
    recall = np.divide(correct, actual_counts, out=np.zeros(n_classes), where=actual_counts > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes),
                   where=(precision + recall) > 0)
    metrics = {
        "Accuracy": float(correct.sum() / len(y_true)),
        "Precision (macro)": float(precision.mean()),
        "Recall (macro)": float(recall.mean()),
        "F1-score (macro)": float(f1.mean()),
    }
    return metrics, matrix


def check_metric_oracle():
    """Known counts include a class that is never predicted."""
    expected_matrix = np.array([[1, 1, 0], [0, 2, 0], [0, 2, 0]])
    metrics, matrix = metrics_from_predictions(
        np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 1, 1]), n_classes=3,
    )
    np.testing.assert_array_equal(matrix, expected_matrix)
    np.testing.assert_allclose([metrics[k] for k in METRIC_KEYS], [.5, 7 / 15, .5, 26 / 63],
                               rtol=0, atol=1e-14)


def verify_results(project_root):
    """Verify saved metrics, pixel membership and the declared frozen candidates."""
    root = Path(project_root).resolve()
    result_path = root / "results/model_comparison.json"
    prediction_path = root / "results/model_comparison_predictions.npz"
    dataset_path = root / "results/pixel_dataset.npz"
    selection_path = root / "results/model_comparison_selection.json"
    check_metric_oracle()
    comparison = json.loads(result_path.read_text(encoding="utf-8"))
    rows = comparison["rows"]
    require(isinstance(rows, list) and rows, "The comparison must contain at least one model row.")
    names, prediction_keys = [], []
    checked_rows = []
    with np.load(dataset_path, allow_pickle=False) as baseline, \
         np.load(prediction_path, allow_pickle=False) as predictions:
        for key in ("ids_test", "y_test", "train_indices", "val_indices"):
            np.testing.assert_array_equal(predictions[key], baseline[key], err_msg=f"Changed {key} membership/order")
        y_test = baseline["y_test"]
        train, val = baseline["train_indices"], baseline["val_indices"]
        require(len(y_test) == 14105 and (len(train), len(val)) == (10996, 2749),
                "The comparison must retain the full original sample sets.")
        np.testing.assert_array_equal(np.unique(y_test), np.arange(6))
        np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(13745))
        require(not np.intersect1d(train, val).size, "Training and validation indices overlap.")
        np.testing.assert_array_equal(baseline["ids_test"][:, 0], y_test)
        all_coordinates = np.vstack((baseline["ids_trainval"][:, 2:], baseline["ids_test"][:, 2:]))
        require(len(np.unique(all_coordinates, axis=0)) == 27850, "Camera pixel identities overlap.")
        supports = np.bincount(y_test, minlength=6)

        for row in rows:
            name, family, key = row["name"], row["family"], row["prediction_key"]
            require(all(isinstance(v, str) and v for v in (name, family, row["representation"], key)),
                    "Every row needs a name, family, representation and prediction key.")
            names.append(name)
            prediction_keys.append(key)
            require(key in predictions.files, f"Missing prediction array for {name}.")
            accuracy = row["validation_accuracy"]
            require(np.isfinite(accuracy) and 0 <= accuracy <= 1, f"Invalid validation accuracy for {name}.")
            metrics, matrix = metrics_from_predictions(y_test, predictions[key])
            supplied_matrix = np.asarray(row["confusion_matrix"])
            require(supplied_matrix.shape == (6, 6)
                    and np.issubdtype(supplied_matrix.dtype, np.integer)
                    and (supplied_matrix >= 0).all(), f"Invalid confusion matrix for {name}.")
            np.testing.assert_array_equal(supplied_matrix, matrix, err_msg=f"Incorrect confusion matrix for {name}")
            np.testing.assert_array_equal(matrix.sum(axis=1), supports)
            require(int(matrix.sum()) == 14105, f"Incomplete test support for {name}.")
            # Permit typography differences such as 'Recall(macro)' in serialized tables.
            supplied_metrics = {k.replace(" ", ""): v for k, v in row["test_metrics"].items()}
            for metric, expected in metrics.items():
                actual = supplied_metrics[metric.replace(" ", "")]
                require(np.isfinite(actual) and 0 <= actual <= 1, f"Invalid {metric} for {name}.")
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12,
                                           err_msg=f"Incorrect {metric} for {name}")
            if "balanced_accuracy" in row:
                np.testing.assert_allclose(row["balanced_accuracy"], metrics["Recall (macro)"], rtol=0, atol=1e-12,
                                           err_msg=f"Balanced accuracy must equal macro recall for {name}")
            checked_rows.append({"name": name, "prediction_key": key, "test_metrics": metrics,
                                 "balanced_accuracy": metrics["Recall (macro)"]})

    require(len(names) == len(set(names)), "Model names must be unique.")
    require(len(prediction_keys) == len(set(prediction_keys)), "Each model must have its own prediction key.")
    require(BASELINE_KEYS <= set(prediction_keys), "The table omits an original baseline.")

    # The small durable freeze record travels with the submission; fitted models need not.
    require(selection_path.is_file(), "Missing results/model_comparison_selection.json freeze record.")
    require(comparison["selection_lock_sha256"] == sha256(selection_path), "The selection freeze checksum differs.")
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))
    require(frozen["schema_version"] == 1 and frozen["frozen_utc"], "Invalid selection freeze metadata.")
    dataset_file_matches, dataset_content_verified = _verify_dataset_checksums(dataset_path, comparison, frozen)
    protocol_path = root / "experiments/protocol.json"
    protocol_verified = False
    if protocol_path.is_file():
        require(frozen["protocol_sha256"] == sha256(protocol_path), "The experiment protocol changed after freezing.")
        protocol_verified = True
    frozen_models = frozen["models"]
    require(isinstance(frozen_models, list) and frozen_models, "The selection freeze has no models.")
    frozen_by_key = {model["prediction_key"]: model for model in frozen_models}
    require(len(frozen_by_key) == len(frozen_models), "Duplicate prediction keys in the freeze record.")
    new_rows = []
    for row in rows:
        key = row["prediction_key"]
        if row.get("original_baseline", False):
            require(key in BASELINE_KEYS, "An unrecognized model was marked as an original baseline.")
            continue
        require(key not in BASELINE_KEYS, "Original baseline row is missing its original_baseline flag.")
        require(key in frozen_by_key, f"{row['name']} was not in the frozen candidate list.")
        selected = frozen_by_key[key]
        for field in ("name", "family", "representation", "prediction_key"):
            require(row[field] == selected[field], f"Frozen {field} differs for {row['name']}.")
        np.testing.assert_allclose(row["validation_accuracy"], selected["validation_accuracy"], rtol=0, atol=1e-12)
        if "validation_macro_f1" in row:
            np.testing.assert_allclose(row["validation_macro_f1"], selected["validation_macro_f1"], rtol=0, atol=1e-12)
        new_rows.append(row)
    require({row["prediction_key"] for row in new_rows} == set(frozen_by_key),
            "The comparison omitted or added a frozen model.")

    # Optional local development artifacts can be absent from the grading ZIP.
    artifact_lock = root / "experiments/artifacts/selection_lock.json"
    artifact_lock_verified = False
    if artifact_lock.is_file():
        require(sha256(artifact_lock) == sha256(selection_path), "The local and durable freeze records differ.")
        artifact_lock_verified = True
    return {
        "status": "passed", "test_pixels": 14105, "models_checked": len(rows),
        "original_baselines": len(BASELINE_KEYS), "frozen_new_models": len(new_rows),
        "checks": ["Exact original test identities, labels and training/validation index order",
                   "Complete disjoint split and unique camera pixels",
                   "Every confusion matrix and metric independently recomputed from predictions",
                   "All six species and all 14,105 test pixels retained for every model",
                   "Balanced accuracy equals macro recall",
                   "Every new model agrees with the checksummed pre-evaluation selection record",
                   "Every frozen model appears in the final comparison"],
        "protocol_file_verified": protocol_verified, "optional_artifact_lock_verified": artifact_lock_verified,
        "dataset_file_checksum_matches": dataset_file_matches,
        "dataset_content_checksum_verified": dataset_content_verified,
        "large_model_artifacts_required": False, "rows": checked_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_metric_oracle()
        print("PASS: independent confusion-matrix metric oracle, including an unpredicted class.")
    else:
        print(json.dumps(verify_results(args.project_root), indent=2))


if __name__ == "__main__":
    main()
