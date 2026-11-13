"""Fixed, validation-only XGBoost comparison on RGBcc and original RGB."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
import xgboost
from xgboost import XGBClassifier

from tabular_compare import file_sha256, raw_rgb_histories, write_json, json_parameters


ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (3, 6)
TREE_COUNTS = (150, 300)
REPRESENTATIONS = ("rgbcc", "rgb_intensity")


def fit_validation(project_root=ROOT, artifact_dir=None):
    """Select one declared configuration without reading any test labels."""
    root = Path(project_root).resolve()
    artifacts = Path(artifact_dir or root / "experiments/artifacts").resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "results/pixel_dataset.npz"
    with np.load(dataset_path, allow_pickle=False) as saved:
        colour_features = saved["X_trainval"]
        labels = saved["y_trainval"]
        identities = saved["ids_trainval"]
        train, val, days = saved["train_indices"], saved["val_indices"], saved["doys"]
    assert colour_features.shape == (13745, 111)
    assert (len(train), len(val)) == (10996, 2749)
    np.testing.assert_array_equal(labels, identities[:, 0])
    np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(13745))
    # I build this independently from the photographs to retain their brightness.
    rgb_features, image_hashes = raw_rgb_histories(root, identities, days)
    features = {"rgbcc": colour_features, "rgb_intensity": rgb_features}
    training_weights = compute_sample_weight(class_weight="balanced", y=labels[train])
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": file_sha256(dataset_path), "input_image_sha256": image_hashes,
        "xgboost_version": xgboost.__version__, "numpy_version": np.__version__,
        "samples": {"training": len(train), "validation": len(val)},
        "fixed_grid": {"representations": list(REPRESENTATIONS), "max_depth": list(DEPTHS),
                       "n_estimators": list(TREE_COUNTS), "learning_rate": 0.05,
                       "min_child_weight": 5, "reg_lambda": 5, "subsample": 0.8,
                       "colsample_bytree": 0.8, "tree_method": "hist", "n_jobs": 2,
                       "objective": "multi:softprob", "num_class": 6, "random_state": 42},
        "training_weights": "inverse class frequency computed from fitted training labels",
        "selection_rule": "Highest validation accuracy; first declared configuration breaks a tie",
        "evaluation_scope": "Only training-region training and validation; no held-out labels or predictions",
        "candidates": [], "selected": None,
    }
    report_path = artifacts / "xgboost_validation.json"
    write_json(report_path, report)
    best_model, best_record = None, None
    for representation in REPRESENTATIONS:
        X = features[representation]
        for depth in DEPTHS:
            for trees in TREE_COUNTS:
                model = XGBClassifier(
                    objective="multi:softprob", num_class=6, eval_metric="mlogloss",
                    tree_method="hist", n_estimators=trees, max_depth=depth,
                    learning_rate=0.05, min_child_weight=5, reg_lambda=5,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2,
                )
                started = time.perf_counter()
                # Validation scores are used for selection only after each fit finishes.
                with threadpool_limits(limits=2):
                    model.fit(X[train], labels[train], sample_weight=training_weights)
                    fit_seconds = time.perf_counter() - started
                    prediction = model.predict(X[val])
                record = {
                    "family": "xgboost", "representation": representation,
                    "max_depth": depth, "n_estimators": trees,
                    "validation_accuracy": float(accuracy_score(labels[val], prediction)),
                    "validation_macro_f1": float(f1_score(labels[val], prediction, average="macro")),
                    "validation_balanced_accuracy": float(recall_score(labels[val], prediction, average="macro")),
                    "fit_seconds": fit_seconds, "fit_and_validation_seconds": time.perf_counter() - started,
                    "parameters": json_parameters(model),
                }
                report["candidates"].append(record)
                if best_record is None or record["validation_accuracy"] > best_record["validation_accuracy"]:
                    best_model, best_record = model, record
                write_json(report_path, report)
                print(f"XGBoost {representation} depth={depth}, trees={trees}: "
                      f"validation={record['validation_accuracy']:.4f}, "
                      f"macro F1={record['validation_macro_f1']:.4f}, fit={fit_seconds:.1f}s", flush=True)
    model_path = artifacts / "xgboost_selected.joblib"
    joblib.dump(best_model, model_path, compress=3)
    report["selected"] = {**best_record, "model_file": model_path.name,
                          "model_sha256": file_sha256(model_path)}
    report["status"] = "completed"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--artifact-dir", type=Path)
    options = parser.parse_args()
    result = fit_validation(options.project_root, options.artifact_dir)
    print(json.dumps({k: result["selected"][k] for k in
                      ("representation", "max_depth", "n_estimators", "validation_accuracy",
                       "validation_macro_f1", "model_file")}, indent=2), flush=True)
