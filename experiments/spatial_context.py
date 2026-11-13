"""A separate, fixed spatial-context experiment; never edits the notebook.

Run with the existing core environment from the project root:
    python -B experiments/spatial_context.py --fit
    python -B experiments/spatial_context.py --evaluate

The first command selects a context window on unused ROIs. The second evaluates
the frozen selection and a matched raw baseline on the original test pixels.
Local fitted models live in ignored experiments/artifacts/spatial_context.
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
from PIL import Image
import scipy
from scipy.spatial import cKDTree
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from spatial_features import extract_features


ROOT = Path(__file__).resolve().parents[1]
SPECIES = ["A.tomentosum", "C.brasiliensis", "M.guianensis", "M.rubiginosa",
           "P.ramiflora", "P.torta"]
PROTOCOL = ROOT / "experiments/spatial_context_protocol.json"
ARTIFACTS = ROOT / "experiments/artifacts/spatial_context"
RESULTS = ROOT / "results"
PREFIX = "spatial_context"
CANDIDATES = ("raw_rgb", "context_5", "context_25")
HASH_POLICY = "SHA256; CRLF normalized to LF for .json/.ipynb/.py; raw bytes otherwise"


def sha256(path):
    path = Path(path)
    if path.suffix in {".json", ".ipynb", ".py"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_npz(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def current_sources():
    paths = [PROTOCOL, Path(__file__), ROOT / "experiments/spatial_features.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in paths}


def preserved_files():
    paths = [ROOT / "casework/case_notebook.ipynb", ROOT / "dist/phenology_submission.zip"]
    paths += sorted((ROOT / "reference").glob("*.ipynb"))
    paths += [p for p in RESULTS.iterdir() if p.is_file() and not p.name.startswith(PREFIX)]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in paths}


def verify_hashes(hashes):
    for relative, expected in hashes.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Recorded input or preserved file changed: {relative}")


def input_hashes(doys):
    paths = sorted((ROOT / "raw/masks").glob("*.pgm"))
    paths += [ROOT / f"raw/images/1/2011_{int(day)}_12.jpg" for day in doys]
    paths += [ROOT / "results/pixel_dataset.npz"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in paths}


def dataset_geometry():
    # Test identity columns are used only for the geometry buffer at this stage.
    # Neither test features nor test predictions are loaded during selection.
    with np.load(RESULTS / "pixel_dataset.npz", allow_pickle=False) as data:
        original_train = data["ids_trainval"][data["train_indices"]]
        test_ids = data["ids_test"]
        used_rois = set(data["ids_trainval"][:, 1]) | set(test_ids[:, 1])
        doys = data["doys"]
    unused = []
    for path in sorted((ROOT / "raw/masks").glob("*.pgm")):
        species, roi_text = path.stem.rsplit("_", 1)
        roi = int(roi_text)
        if roi in used_rois:
            continue
        with Image.open(path) as image:
            rows, columns = np.nonzero(np.asarray(image) > 0)
        unused.append(np.column_stack((np.full(len(rows), SPECIES.index(species)),
                                       np.full(len(rows), roi), rows, columns)))
    original_validation = np.concatenate(unused).astype(np.int64)
    test_tree = cKDTree(test_ids[:, 2:4])
    validation_distance = test_tree.query(original_validation[:, 2:4], p=np.inf)[0]
    validation_ids = original_validation[validation_distance > 24]
    evaluation_tree = cKDTree(np.concatenate((validation_ids[:, 2:4], test_ids[:, 2:4])))
    train_distance = evaluation_tree.query(original_train[:, 2:4], p=np.inf)[0]
    train_ids = original_train[train_distance > 24]
    if (len(train_ids), len(validation_ids), len(test_ids)) != (10113, 30114, 14105):
        raise ValueError("Geometry differs from the declared protocol; stop before fitting.")
    if set(train_ids[:, 0]) != set(range(6)):
        raise ValueError("Every training species must retain pixels.")
    if set(validation_ids[:, 0]) != {0, 1, 3, 5}:
        raise ValueError("Unexpected validation species.")
    if set(validation_ids[:, 1]) != set(original_validation[:, 1]):
        raise ValueError("A validation region was entirely removed by the buffer.")
    if len(np.unique(np.concatenate((train_ids, validation_ids, test_ids))[:, 2:4], axis=0)) != len(train_ids) + len(validation_ids) + len(test_ids):
        raise ValueError("Duplicate pixel coordinates across partitions.")
    summary = {
        "original_train": len(original_train), "train": len(train_ids),
        "original_validation": len(original_validation), "validation": len(validation_ids),
        "test": len(test_ids),
        "train_counts": np.bincount(train_ids[:, 0], minlength=6).tolist(),
        "validation_counts": np.bincount(validation_ids[:, 0], minlength=6).tolist(),
        "minimum_train_evaluation_distance": float(train_distance[train_distance > 24].min()),
        "minimum_validation_test_distance": float(validation_distance[validation_distance > 24].min()),
        "validation_rois": [{"roi_id": int(roi),
                             "original": int(np.sum(original_validation[:, 1] == roi)),
                             "retained": int(np.sum(validation_ids[:, 1] == roi))}
                            for roi in np.unique(original_validation[:, 1])],
    }
    return train_ids, validation_ids, test_ids, doys, summary


def metrics(y_true, predicted):
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (y_true, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    correct = np.diag(confusion)
    recall = np.divide(correct, support, out=np.zeros(6, dtype=float), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted_count, out=np.zeros(6, dtype=float),
                   where=(support + predicted_count) > 0)
    present = support > 0
    return {"accuracy": float(correct.sum() / support.sum()),
            "balanced_accuracy_present_classes": float(recall[present].mean()),
            "macro_f1_present_classes": float(f1[present].mean()),
            "classes_present": np.flatnonzero(present).tolist(),
            "recall": [float(recall[i]) if present[i] else None for i in range(6)],
            "support": support.tolist(), "confusion_matrix": confusion.tolist()}


def region_validation(ids, predicted):
    regions = []
    species_scores = []
    for species in np.unique(ids[:, 0]):
        accuracies = []
        for roi in np.unique(ids[ids[:, 0] == species, 1]):
            selected = ids[:, 1] == roi
            accuracy = float(np.mean(predicted[selected] == species))
            accuracies.append(accuracy)
            regions.append({"species_class": int(species), "roi_id": int(roi),
                            "n_pixels": int(selected.sum()), "accuracy": accuracy})
        species_scores.append({"species_class": int(species),
                               "mean_roi_accuracy": float(np.mean(accuracies))})
    return {"selection_score": float(np.mean([x["mean_roi_accuracy"] for x in species_scores])),
            "per_species": species_scores, "per_roi": regions,
            "pooled_metrics": metrics(ids[:, 0], predicted)}


def fit():
    report_path = RESULTS / f"{PREFIX}_validation.json"
    selection_path = RESULTS / f"{PREFIX}_selection.json"
    if report_path.exists() or selection_path.exists():
        raise FileExistsError("This fixed experiment already has a validation record; refusing to overwrite.")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["window_sizes"] != [5, 25] or protocol["model"]["C"] != 10:
        raise ValueError("Runner and fixed protocol disagree.")
    started = time.perf_counter()
    train_ids, validation_ids, _, doys, geometry = dataset_geometry()
    preserved = preserved_files()
    sources = current_sources()
    inputs = input_hashes(doys)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    coordinates = np.concatenate((train_ids[:, 2:4], validation_ids[:, 2:4]))
    feature_started = time.perf_counter()
    features = extract_features(ROOT, coordinates, doys, window_sizes=(5, 25), progress=True)
    extraction_seconds = time.perf_counter() - feature_started
    validation_predictions = {"ids_train": train_ids, "ids_validation": validation_ids,
                              "y_validation": validation_ids[:, 0], "doys": doys}
    rows = []
    models = {}
    with threadpool_limits(limits=2):
        for name in CANDIDATES:
            print(f"Fitting fixed GLM: {name}", flush=True)
            x_train = features[name][:len(train_ids)]
            x_validation = features[name][len(train_ids):]
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=10.0, class_weight="balanced", solver="lbfgs", max_iter=2000,
                random_state=42))
            fit_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train, train_ids[:, 0])
            fit_seconds = time.perf_counter() - fit_started
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError(f"{name} did not converge within the fixed budget; no test evaluation.")
            predict_started = time.perf_counter()
            predicted = model.predict(x_validation)
            predict_seconds = time.perf_counter() - predict_started
            score = region_validation(validation_ids, predicted)
            validation_predictions[name] = predicted
            model_path = ARTIFACTS / f"{name}.joblib"
            joblib.dump(model, model_path)
            models[name] = {"path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                            "sha256": sha256(model_path)}
            rows.append({"name": name, "n_features": int(x_train.shape[1]),
                         "fit_seconds": fit_seconds, "validation_predict_seconds": predict_seconds,
                         "iterations": model[-1].n_iter_.tolist(), "validation": score})
            print(f"{name}: four-species mean ROI score {score['selection_score']:.4%}; fit {fit_seconds:.2f}s", flush=True)
    del features
    predictions_path = RESULTS / f"{PREFIX}_validation_predictions.npz"
    write_npz(predictions_path, **validation_predictions)
    contextual = [row for row in rows if row["name"].startswith("context_")]
    selected = max(contextual, key=lambda row: row["validation"]["selection_score"])["name"]
    report = {"schema_version": 1, "hash_policy": HASH_POLICY, "completed_utc": now(), "species": SPECIES,
              "geometry": geometry, "sources_sha256": sources, "inputs_sha256": inputs,
              "preserved_files_sha256": preserved,
              "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
              "feature_extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - started,
              "rows": rows, "selected_context": selected,
              "predictions_sha256": sha256(predictions_path),
              "limitations": protocol["interpretation"] + [protocol["validation_limitation"]]}
    write_json(report_path, report)
    # Freeze selection and actual fitted model bytes before reading test outcomes.
    selection = {"schema_version": 1, "hash_policy": HASH_POLICY, "frozen_utc": now(), "selected_context": selected,
                 "test_models": ["raw_rgb", selected], "models": models,
                 "validation_report_sha256": sha256(report_path),
                 "validation_predictions_sha256": sha256(predictions_path),
                 "sources_sha256": sources, "inputs_sha256": inputs}
    write_json(selection_path, selection)
    verify_hashes(preserved)
    print(f"Frozen {selected}; no test predictions computed. Validation record: {report_path}", flush=True)


def evaluate():
    report_path = RESULTS / f"{PREFIX}_results.json"
    if report_path.exists():
        raise FileExistsError("Test results already exist; refusing to rerun or overwrite.")
    started = time.perf_counter()
    selection_path = RESULTS / f"{PREFIX}_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validation_path = RESULTS / f"{PREFIX}_validation.json"
    validation_predictions_path = RESULTS / f"{PREFIX}_validation_predictions.npz"
    if sha256(validation_path) != selection["validation_report_sha256"] or sha256(validation_predictions_path) != selection["validation_predictions_sha256"]:
        raise ValueError("Validation record differs from the frozen selection.")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    verify_hashes(selection["sources_sha256"])
    verify_hashes(selection["inputs_sha256"])
    verify_hashes(validation["preserved_files_sha256"])
    for item in selection["models"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("A fitted model changed after selection.")
    with np.load(RESULTS / "pixel_dataset.npz", allow_pickle=False) as data:
        test_ids, y_test, doys = data["ids_test"], data["y_test"], data["doys"]
    np.testing.assert_array_equal(y_test, test_ids[:, 0])
    selected = selection["selected_context"]
    window = int(selected.rsplit("_", 1)[1])
    feature_started = time.perf_counter()
    features = extract_features(ROOT, test_ids[:, 2:4], doys, window_sizes=(window,), progress=True)
    extraction_seconds = time.perf_counter() - feature_started
    rows = []
    arrays = {"ids_test": test_ids, "y_test": y_test, "doys": doys}
    with threadpool_limits(limits=2):
        for name in selection["test_models"]:
            model = joblib.load(ROOT / selection["models"][name]["path"])
            predict_started = time.perf_counter()
            predicted = model.predict(features[name])
            prediction_seconds = time.perf_counter() - predict_started
            arrays[name] = predicted
            scored = metrics(y_test, predicted)
            rows.append({"name": name, "n_features": int(features[name].shape[1]),
                         "predict_seconds": prediction_seconds, "metrics": scored})
            print(f"{name}: test accuracy {scored['accuracy']:.4%}; balanced accuracy {scored['balanced_accuracy_present_classes']:.4%}", flush=True)
    predictions_path = RESULTS / f"{PREFIX}_predictions.npz"
    write_npz(predictions_path, **arrays)
    raw_metrics, context_metrics = rows[0]["metrics"], rows[1]["metrics"]
    delta = {key: context_metrics[key] - raw_metrics[key]
             for key in ("accuracy", "balanced_accuracy_present_classes", "macro_f1_present_classes")}
    report = {"schema_version": 1, "hash_policy": HASH_POLICY, "completed_utc": now(), "species": SPECIES,
              "selected_context": selected, "samples": validation["geometry"],
              "selection_sha256": sha256(selection_path), "predictions_sha256": sha256(predictions_path),
              "feature_extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - started,
              "rows": rows, "context_minus_raw": delta, "limitations": validation["limitations"]}
    write_json(report_path, report)
    verify_hashes(validation["preserved_files_sha256"])
    print(f"Completed fixed experiment: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--fit", action="store_true")
    stages.add_argument("--evaluate", action="store_true")
    arguments = parser.parse_args()
    fit() if arguments.fit else evaluate()
