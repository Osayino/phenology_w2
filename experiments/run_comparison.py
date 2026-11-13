"""Freeze validation-selected models, then evaluate the fixed comparison.

Run the main notebook first to prepare its pixel dataset and recurrence cache.
Use --fit to reproduce the declared training/validation searches. Without --fit,
the selected local artifacts must already exist. No original result is replaced.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from tabular_compare import file_sha256, raw_rgb_histories, write_json
from bayesian_compare import predict_artifact
from verify_comparison import dataset_content_sha256

ROOT = SCRIPT_DIR.parent


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fit_candidates(root=ROOT, image_models=("mobilenet_v3_small", "vit_b_32")):
    """Reproduce the small fixed searches using training/validation rows only."""
    from tabular_compare import fit_validation as fit_tabular
    from xgboost_compare import fit_validation as fit_xgboost
    from bayesian_compare import fit_validation as fit_bayesian
    from image_compare import fit_validation as fit_image

    fit_tabular(root)
    fit_xgboost(root)
    fit_bayesian(root / "results/pixel_dataset.npz", root / "experiments/artifacts")
    for model in image_models:
        fit_image(model, root, batch_size=8, threads=2)


def freeze_selection(root=ROOT, image_models=("mobilenet_v3_small", "vit_b_32")):
    """Save all choices and model hashes before the test evaluation starts."""
    root = Path(root).resolve()
    artifacts = root / "experiments/artifacts"
    dataset_hash = file_sha256(root / "results/pixel_dataset.npz")
    tabular_path = artifacts / "tabular_validation.json"
    xgb_path = artifacts / "xgboost_validation.json"
    bayes_path = artifacts / "bayesian_validation.json"
    tabular, xgb, bayes = map(load_json, (tabular_path, xgb_path, bayes_path))
    reports = {"tabular": tabular, "xgboost": xgb, "bayesian": bayes}
    assert tabular["status"] == xgb["status"] == "completed"
    for report in reports.values():
        assert report["dataset_sha256"] == dataset_hash
    records = []

    def add(name, family, input_kind, representation, selected, model_path, report_path, configuration):
        assert model_path.is_file()
        model_hash = file_sha256(model_path)
        if "model_sha256" in selected:
            assert selected["model_sha256"] == model_hash, "Model changed after validation"
        records.append({
            "name": name, "family": family, "prediction_key": family,
            "input_kind": input_kind, "representation": representation,
            "validation_accuracy": selected["validation_accuracy"],
            "validation_macro_f1": selected["validation_macro_f1"],
            "configuration": configuration,
            "model_path": model_path.relative_to(root).as_posix(),
            "model_sha256": model_hash,
            "validation_report": report_path.relative_to(root).as_posix(),
            "validation_report_sha256": file_sha256(report_path),
        })

    for family, name in (("svm", "SVM (RBF)"), ("glm", "Multinomial logistic GLM")):
        selected = tabular["selected"][family]
        winner = max((c for c in tabular["candidates"] if c["family"] == family),
                     key=lambda c: c["validation_accuracy"])
        assert (winner["representation"], winner["C"]) == (selected["representation"], selected["C"])
        representation = "RGB intensities (111)" if selected["representation"] == "rgb_intensity" else "RGBcc (111)"
        add(name, family, selected["representation"], representation, selected,
            artifacts / selected["model_file"], tabular_path, selected["parameters"])
    selected = xgb["selected"]
    winner = max(xgb["candidates"], key=lambda c: c["validation_accuracy"])
    assert all(winner[k] == selected[k] for k in ("representation", "max_depth", "n_estimators"))
    representation = "RGB intensities (111)" if selected["representation"] == "rgb_intensity" else "RGBcc (111)"
    add("XGBoost", "xgboost", selected["representation"], representation, selected,
        artifacts / selected["model_file"], xgb_path, selected["parameters"])

    gaussian = bayes["candidates"][0]
    assert gaussian["name"].startswith("Gaussian naive Bayes")
    add("Gaussian naive Bayes", "gaussian_nb", "bayesian_summary", "RGBcc summaries (9)", gaussian,
        artifacts / gaussian["model_file"], bayes_path, gaussian["parameters"])
    tan_candidates = [c for c in bayes["candidates"] if c["name"].startswith("TAN")]
    selected = max(tan_candidates, key=lambda c: c["validation_accuracy"])
    add("Bayesian network (TAN)", "tan", "bayesian_summary", "Binned RGBcc summaries (9)", selected,
        artifacts / selected["model_file"], bayes_path, selected["parameters"])

    for model_name in image_models:
        report_path = artifacts / f"image_{model_name}_validation.json"
        report = load_json(report_path)
        assert report["dataset_sha256"] == dataset_hash and not report["test_evaluated"]
        selected = report["selected"]
        assert selected == max(report["candidates"], key=lambda c: c["validation_accuracy"])
        family = "cnn_transfer" if model_name == "mobilenet_v3_small" else "vit_transfer"
        name = "CNN transfer (MobileNet)" if family == "cnn_transfer" else "ViT transfer (B/32)"
        feature_info = report["feature_info"]
        width = feature_info["recipe"]["feature_width"]
        configuration = {"backbone": model_name, "head_C": selected["C"],
                         "head": "training-only StandardScaler + balanced multinomial logistic regression",
                         "backbone_recipe": feature_info["recipe"]}
        add(name, family, model_name, f"RP pretrained features ({width})", selected,
            root / report["head_path"], report_path, configuration)
        reports[model_name] = report

    # The freeze record is small enough to accompany the submission. Large fitted
    # models and image-feature caches stay local and can be recreated with --fit.
    frozen = {
        "schema_version": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash,
        "dataset_content_sha256": dataset_content_sha256(root / "results/pixel_dataset.npz"),
        "protocol_sha256": file_sha256(root / "experiments/protocol.json"),
        "selection_rule": "Highest validation accuracy within each declared family; first configuration breaks ties",
        "models": records,
        "baseline_test_results_previously_seen": True,
        "interpretation": "Exploratory fixed-ROI comparison; model families and some representations differ",
    }
    (root / "results").mkdir(exist_ok=True)
    freeze_path = root / "results/model_comparison_selection.json"
    # Strict JSON prevents NaN parameter values from breaking other readers.
    freeze_path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (artifacts / "selection_lock.json").write_bytes(freeze_path.read_bytes())
    (root / "results/model_comparison_validation.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"FROZEN: {len(records)} model configurations before test evaluation", flush=True)
    return frozen


def metric_values(y, prediction):
    return {
        "Accuracy": float(accuracy_score(y, prediction)),
        "Precision (macro)": float(precision_score(y, prediction, labels=np.arange(6), average="macro", zero_division=0)),
        "Recall (macro)": float(recall_score(y, prediction, labels=np.arange(6), average="macro", zero_division=0)),
        "F1-score (macro)": float(f1_score(y, prediction, labels=np.arange(6), average="macro", zero_division=0)),
    }


def evaluate_frozen(root=ROOT, frozen=None):
    """Evaluate every fixed family winner once and report all outcomes together."""
    root = Path(root).resolve()
    freeze_path = root / "results/model_comparison_selection.json"
    frozen = frozen or load_json(freeze_path)
    assert frozen == load_json(freeze_path)
    assert file_sha256(root / "results/pixel_dataset.npz") == frozen["dataset_sha256"]
    assert file_sha256(root / "experiments/protocol.json") == frozen["protocol_sha256"]
    # All hyperparameters and fitted model files are fixed before labels are loaded.
    for record in frozen["models"]:
        assert file_sha256(root / record["model_path"]) == record["model_sha256"]
        assert file_sha256(root / record["validation_report"]) == record["validation_report_sha256"]
    with np.load(root / "results/pixel_dataset.npz", allow_pickle=False) as saved:
        rgbcc, labels, identities = saved["X_test"], saved["y_test"], saved["ids_test"]
        train, val, days = saved["train_indices"], saved["val_indices"], saved["doys"]
    assert rgbcc.shape == (14105, 111) and len(labels) == 14105
    rgb, image_hashes = raw_rgb_histories(root, identities, days)
    tabular_report = load_json(root / "experiments/artifacts/tabular_validation.json")
    assert image_hashes == tabular_report["input_image_sha256"], "Photographs changed after validation"
    prediction_arrays = {"ids_test": identities, "y_test": labels,
                         "train_indices": train, "val_indices": val}
    rows = []

    def add_row(name, family, representation, validation_accuracy, predicted, baseline=False, seconds=None):
        predicted = np.asarray(predicted)
        assert predicted.shape == labels.shape and np.isin(predicted, np.arange(6)).all()
        metrics = metric_values(labels, predicted)
        prediction_arrays[family] = predicted
        rows.append({"name": name, "family": family, "representation": representation,
                     "prediction_key": family, "original_baseline": baseline,
                     "validation_accuracy": float(validation_accuracy), "test_metrics": metrics,
                     "balanced_accuracy": metrics["Recall (macro)"],
                     "confusion_matrix": confusion_matrix(labels, predicted, labels=np.arange(6)).tolist(),
                     "test_prediction_seconds": seconds})

    baseline = load_json(root / "results/experiment_results.json")
    forest = load_json(root / "results/extension_results.json")
    with np.load(root / "results/predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["ids_test"], identities)
        add_row("Raw K-NN", "raw_knn", "RGBcc (111)", max(baseline["raw_validation_accuracy"]), saved["raw_prediction"], True)
        add_row("Recurrence K-NN", "rp_knn", "RP flattened (58,800)", max(baseline["rp_validation_accuracy"]), saved["rp_prediction"], True)
    with np.load(root / "results/extension_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["ids_test"], identities)
        validation_accuracy = max(forest["validation_candidates"], key=lambda r: r["accuracy"])["accuracy"]
        add_row("Random forest", "random_forest", "RGBcc (111)", validation_accuracy, saved["prediction"], True)

    for record in frozen["models"]:
        started = time.perf_counter()
        kind = record["input_kind"]
        model_path = root / record["model_path"]
        if kind == "bayesian_summary":
            predicted = predict_artifact(model_path, rgbcc)
        elif kind in ("rgbcc", "rgb_intensity"):
            model = joblib.load(model_path)
            with threadpool_limits(limits=2):
                predicted = model.predict(rgbcc if kind == "rgbcc" else rgb)
            del model
        else:
            from image_compare import extract_embeddings
            features, feature_info = extract_embeddings(kind, root, split="test", batch_size=8, threads=2)
            training_recipe = record["configuration"]["backbone_recipe"]
            # Dataset membership changes, but the image transformation and weights must match.
            for key in ("model", "weights", "weights_url", "feature_width", "preprocessing",
                        "torch_version", "torchvision_version", "device"):
                assert feature_info["recipe"][key] == training_recipe[key], f"Image recipe changed: {key}"
            model = joblib.load(model_path)
            with threadpool_limits(limits=2):
                predicted = model.predict(features)
            features._mmap.close()
            del model
        add_row(record["name"], record["family"], record["representation"],
                record["validation_accuracy"], predicted, seconds=time.perf_counter() - started)
        # Progress reports completion, not scores: all fixed choices are reported together.
        print(f"EVALUATED: {record['name']}", flush=True)

    results = {
        "schema_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_lock_sha256": file_sha256(freeze_path), "dataset_sha256": frozen["dataset_sha256"],
        "dataset_content_sha256": frozen["dataset_content_sha256"],
        "samples": {"train": len(train), "validation": len(val), "test": len(labels)},
        "rows": rows,
        "limitations": ["Validation shares training ROIs; nearby pixels are correlated",
                        "Previous baseline test results were already observed",
                        "Representations vary between some families; changes are not isolated classifier effects",
                        "Best observed test performance is descriptive and needs new independent regions for confirmation",
                        "The 2026 paper uses a different protocol; matching its metric does not reproduce its experiment"],
    }
    np.savez_compressed(root / "results/model_comparison_predictions.npz", **prediction_arrays)
    (root / "results/model_comparison.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from verify_comparison import verify_results
    verify_results(root)
    from plot_comparison import plot_comparison
    plot_comparison(root, save=True)
    return results


def run_comparison(root=ROOT, *, refit=False, image_models=("mobilenet_v3_small", "vit_b_32")):
    root = Path(root).resolve()
    if refit:
        fit_candidates(root, image_models)
    frozen = freeze_selection(root, image_models)
    return evaluate_frozen(root, frozen)


if __name__ == "__main__":
    # A command-line run writes figures without requiring a desktop GUI toolkit.
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fit", action="store_true", help="Refit all declared candidates before evaluation")
    parser.add_argument("--images", nargs="*", default=["mobilenet_v3_small", "vit_b_32"],
                        choices=["mobilenet_v3_small", "vit_b_32"])
    options = parser.parse_args()
    table = run_comparison(options.root, refit=options.fit, image_models=tuple(options.images))
    print(f"{'Model':<30} {'Validation':>12} {'Test accuracy':>14} {'Balanced acc.':>14} {'Macro F1':>10}")
    for row in table["rows"]:
        metrics = row["test_metrics"]
        print(f"{row['name']:<30} {row['validation_accuracy']:12.4f} {metrics['Accuracy']:14.4f} "
              f"{metrics['Recall (macro)']:14.4f} {metrics['F1-score (macro)']:10.4f}")
