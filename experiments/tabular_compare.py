"""Validation-only SVM and GLM comparison on two colour representations.

The benchmark reads only the training-region arrays from pixel_dataset.npz.
Run this script directly to fit the fixed shortlist and save its selected models.
The separate test regions are evaluated later, after these choices are fixed.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
from PIL import Image
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
C_VALUES = (0.1, 1.0, 10.0)
REPRESENTATIONS = ("rgbcc", "rgb_intensity")
FAMILIES = ("svm", "glm")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_rgb_histories(project_root, identities, doys):
    """Return channel-major RGB histories for the supplied pixel coordinates."""
    identities = np.asarray(identities)
    rows, columns = identities[:, 2], identities[:, 3]
    histories = np.empty((len(identities), 3, len(doys)), dtype=np.float32)
    image_hashes = {}
    # I retain the same pixel identities and days, but now keep brightness information.
    for day_index, day in enumerate(doys):
        path = Path(project_root) / "raw/images/1" / f"2011_{int(day)}_12.jpg"
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        histories[:, :, day_index] = rgb[rows, columns].astype(np.float32) / 255.0
        image_hashes[path.name] = file_sha256(path)
    return histories.reshape(len(identities), -1), image_hashes


def make_candidate(family, c_value):
    # The pipeline learns its scaling from the fitted training rows only.
    if family == "svm":
        classifier = SVC(C=c_value, kernel="rbf", gamma="scale", class_weight="balanced")
    elif family == "glm":
        classifier = LogisticRegression(
            C=c_value, solver="lbfgs", max_iter=2000, class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
    return make_pipeline(StandardScaler(), classifier)


def json_parameters(model):
    """Keep every estimator parameter, including readable pipeline objects."""
    parameters = {}
    for name, value in model.get_params(deep=True).items():
        # JSON has no numeric NaN or infinity, so I describe those parameter values.
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            parameters[name] = str(float(value))
        elif isinstance(value, (str, int, float, bool, type(None))):
            parameters[name] = value
        else:
            parameters[name] = repr(value)
    return parameters


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fit_validation(project_root=ROOT, artifact_dir=None):
    """Fit the declared shortlist and select one model per family on validation."""
    project_root = Path(project_root).resolve()
    artifact_dir = Path(artifact_dir or project_root / "experiments/artifacts").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = project_root / "results/pixel_dataset.npz"
    # NPZ loading is lazy: only these training-region fields are accessed.
    with np.load(dataset_path, allow_pickle=False) as saved:
        colour_features = saved["X_trainval"]
        labels = saved["y_trainval"]
        identities = saved["ids_trainval"]
        train_indices = saved["train_indices"]
        val_indices = saved["val_indices"]
        doys = saved["doys"]
    assert colour_features.shape == (13745, 111)
    assert (len(train_indices), len(val_indices)) == (10996, 2749)
    np.testing.assert_array_equal(np.sort(np.r_[train_indices, val_indices]), np.arange(13745))
    np.testing.assert_array_equal(labels, identities[:, 0])
    np.testing.assert_array_equal(doys, np.arange(242, 279))

    rgb_features, image_hashes = raw_rgb_histories(project_root, identities, doys)
    representations = {"rgbcc": colour_features, "rgb_intensity": rgb_features}
    # Saving this derived table avoids rereading the photographs during later checks.
    np.save(artifact_dir / "tabular_rgb_intensity_trainval.npy", rgb_features, allow_pickle=False)
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": file_sha256(dataset_path), "input_image_sha256": image_hashes,
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "samples": {"training": len(train_indices), "validation": len(val_indices)},
        "threads": 2, "seed": 42,
        "representations": {
            "rgbcc": "37 red chromatic coordinates, then 37 green, then 37 blue",
            "rgb_intensity": "37 red JPEG intensities / 255, then green, then blue; 111 features",
        },
        "fixed_grid": {"families": list(FAMILIES), "C": list(C_VALUES),
                       "representation_order": list(REPRESENTATIONS)},
        "selection_rule": "Highest validation accuracy per family; first declared configuration breaks a tie",
        "evaluation_scope": "Training-region training/validation only; no held-out labels or predictions",
        "candidates": [], "selected": {},
    }
    report_path = artifact_dir / "tabular_validation.json"
    write_json(report_path, report)

    for family in FAMILIES:
        best_score = -1.0
        best_model = None
        best_record = None
        for representation in REPRESENTATIONS:
            X = representations[representation]
            X_train, X_val = X[train_indices], X[val_indices]
            for c_value in C_VALUES:
                model = make_candidate(family, c_value)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                    warnings.simplefilter("always")
                    model.fit(X_train, labels[train_indices])
                    fit_seconds = time.perf_counter() - started
                    predict_started = time.perf_counter()
                    prediction = model.predict(X_val)
                    prediction_seconds = time.perf_counter() - predict_started
                record = {
                    "family": family, "representation": representation, "C": c_value,
                    "validation_accuracy": float(accuracy_score(labels[val_indices], prediction)),
                    "validation_macro_f1": float(f1_score(labels[val_indices], prediction, average="macro")),
                    "validation_balanced_accuracy": float(recall_score(labels[val_indices], prediction, average="macro")),
                    "fit_seconds": fit_seconds, "validation_prediction_seconds": prediction_seconds,
                    "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
                    "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
                    "n_iter": np.asarray(model.steps[-1][1].n_iter_).tolist(),
                    "parameters": json_parameters(model),
                }
                report["candidates"].append(record)
                if record["validation_accuracy"] > best_score:
                    best_score, best_model, best_record = record["validation_accuracy"], model, record
                write_json(report_path, report)
                print(f"{family:3s} {representation:13s} C={c_value:g}: "
                      f"validation={record['validation_accuracy']:.4f}, "
                      f"macro F1={record['validation_macro_f1']:.4f}, "
                      f"fit={fit_seconds:.1f}s, convergence warning={record['convergence_warning']}", flush=True)

        model_path = artifact_dir / f"tabular_{family}_selected.joblib"
        joblib.dump(best_model, model_path, compress=3)
        report["selected"][family] = {
            **best_record, "model_file": model_path.name, "model_sha256": file_sha256(model_path),
        }
        write_json(report_path, report)
    report["status"] = "completed"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--artifact-dir", type=Path)
    options = parser.parse_args()
    fitted = fit_validation(options.project_root, options.artifact_dir)
    print(json.dumps({family: {key: selected[key] for key in
                              ("representation", "C", "validation_accuracy", "validation_macro_f1", "model_file")}
                      for family, selected in fitted["selected"].items()}, indent=2), flush=True)
