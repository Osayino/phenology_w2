"""Independently check the separate spatial-context experiment without fitting.

Use --validation-only before test evaluation, or no switch for the full result.
The checker reads saved predictions, masks and experiment hashes. Fitted model
files are optional; when present, their bytes must match the frozen selection.
Use --audit-preservation separately to compare the earlier notebook, ZIP and
other outputs with the historical pre-integration snapshot. Normal notebook
reruns can legitimately change those outputs; they do not define this experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_cdt, maximum_filter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.verify_comparison import dataset_content_sha256


ROOT = Path(__file__).resolve().parents[1]
SPECIES = ["A.tomentosum", "C.brasiliensis", "M.guianensis", "M.rubiginosa",
           "P.ramiflora", "P.torta"]
TRAIN_ROIS = [5, 31, 1, 4, 8, 9]
TEST_ROIS = [7, 29, 21, 32, 10, 3]
VALIDATION_ROIS = {11, 2, 23, 6, 15, 24, 28, 34, 17, 18}
CANDIDATES = ("raw_rgb", "context_5", "context_25")
FEATURE_COUNTS = {"raw_rgb": 111, "context_5": 333, "context_25": 333}
PREFIX = "spatial_context"
HASH_POLICY = "SHA256; CRLF normalized to LF for .json/.ipynb/.py; raw bytes otherwise"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha256(path):
    path = Path(path)
    if path.suffix in {".json", ".ipynb", ".py"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal(actual, expected, context):
    """Compare serialized values, allowing only rounding error in finite floats."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Different fields: {context}")
        for key, value in expected.items():
            equal(actual[key], value, f"{context}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Different length: {context}")
        for index, value in enumerate(expected):
            equal(actual[index], value, f"{context}[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and np.isfinite(actual), f"Invalid number: {context}")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=context)
    else:
        require(actual == expected, f"Different value: {context}")


def checked_path(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected a relative file path.")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"File is outside the project: {relative}")
    return path


def check_hashes(root, recorded, expected_paths=None):
    require(isinstance(recorded, dict) and recorded, "Missing file checksum manifest.")
    if expected_paths is not None:
        require(set(recorded) == set(expected_paths), "Checksum manifest has missing or unexpected files.")
    for relative, expected in recorded.items():
        require(isinstance(expected, str) and len(expected) == 64, f"Invalid checksum: {relative}")
        path = checked_path(root, relative)
        require(path.is_file(), f"File changed or missing: {relative}")
        if sha256(path) != expected:
            if relative == "results/pixel_dataset.npz":
                verify_dataset_reencoding(root, path, expected)
            else:
                raise AssertionError(f"File changed or missing: {relative}")


def verify_dataset_reencoding(root, path, expected_file_sha256):
    """Accept different ZIP encoding only with matching frozen NPY payloads.

    The pre-existing comparison records must name the same original dataset
    bytes as this experiment. Its frozen selection record is checked before
    using the content digest shared by both records. No array value, NPY header
    or archive member can change under this fallback.
    """
    root, path = Path(root).resolve(), Path(path).resolve()
    require(path == (root / "results/pixel_dataset.npz").resolve(),
            "The encoding fallback is limited to the saved pixel dataset.")
    comparison = read_json(root / "results/model_comparison.json")
    selection_path = root / "results/model_comparison_selection.json"
    frozen = read_json(selection_path)
    require(comparison["dataset_sha256"] == frozen["dataset_sha256"] == expected_file_sha256,
            "Frozen records do not refer to the same original dataset bytes.")
    # This older selection record deliberately uses a raw-byte JSON checksum.
    require(comparison["selection_lock_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "The earlier dataset selection record checksum differs.")
    content_hash = frozen.get("dataset_content_sha256")
    require(isinstance(content_hash, str) and len(content_hash) == 64
            and comparison.get("dataset_content_sha256") == content_hash,
            "The frozen dataset records lack matching content checksums.")
    require(dataset_content_sha256(path) == content_hash,
            "Dataset NPY member names, headers or array payloads changed.")


def reconstruct_geometry(root):
    """Rebuild masks and buffer membership using integer raster distances."""
    masks = []
    occupied = None
    for path in sorted((root / "raw/masks").glob("*.pgm")):
        species, roi_text = path.stem.rsplit("_", 1)
        require(species in SPECIES, f"Unknown species in {path.name}")
        label, roi = SPECIES.index(species), int(roi_text)
        with Image.open(path) as image:
            mask = np.asarray(image) > 0
        require(mask.shape == (960, 1280), f"Unexpected mask dimensions: {path.name}")
        if occupied is None:
            occupied = np.zeros(mask.shape, dtype=bool)
        require(not np.any(occupied & mask), f"ROI masks overlap: {path.name}")
        occupied |= mask
        rows, columns = np.nonzero(mask)
        require(len(rows) > 0, f"Empty ROI: {path.name}")
        ids = np.column_stack((np.full(len(rows), label), np.full(len(rows), roi), rows, columns))
        masks.append((label, roi, ids.astype(np.int64)))
    require(len(masks) == 22 and len({roi for _, roi, _ in masks}) == 22, "Expected 22 distinct ROIs.")
    original_pool = np.concatenate([ids for label, roi, ids in masks if roi == TRAIN_ROIS[label]])
    test_ids = np.concatenate([ids for label, roi, ids in masks if roi == TEST_ROIS[label]])
    unused = [(label, roi, ids) for label, roi, ids in masks
              if roi != TRAIN_ROIS[label] and roi != TEST_ROIS[label]]
    require({roi for _, roi, _ in unused} == VALIDATION_ROIS, "Unexpected validation ROIs.")
    original_validation = np.concatenate([ids for _, _, ids in unused])
    require((len(original_pool), len(test_ids), len(original_validation)) == (13745, 14105, 32112),
            "Original region membership changed.")

    with np.load(root / "results/pixel_dataset.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["ids_trainval"], original_pool)
        np.testing.assert_array_equal(data["y_trainval"], original_pool[:, 0])
        np.testing.assert_array_equal(data["ids_test"], test_ids)
        doys = data["doys"]
        np.testing.assert_array_equal(doys, np.arange(242, 279))
        expected_train, expected_val = train_test_split(
            np.arange(13745), test_size=.2, random_state=42, stratify=original_pool[:, 0])
        np.testing.assert_array_equal(data["train_indices"], expected_train)
        np.testing.assert_array_equal(data["val_indices"], expected_val)
        require((len(expected_train), len(expected_val)) == (10996, 2749), "Changed original split.")
        original_train = original_pool[expected_train]

    def centers(ids):
        mask = np.zeros((960, 1280), dtype=bool)
        mask[ids[:, 2], ids[:, 3]] = True
        return mask

    # Chessboard distances are exact integer Chebyshev distances on this pixel grid.
    test_mask = centers(test_ids)
    distance_to_test = distance_transform_cdt(~test_mask, metric="chessboard")
    validation_distance = distance_to_test[original_validation[:, 2], original_validation[:, 3]]
    validation_ids = original_validation[validation_distance > 24]
    evaluation_mask = test_mask | centers(validation_ids)
    distance_to_evaluation = distance_transform_cdt(~evaluation_mask, metric="chessboard")
    train_distance = distance_to_evaluation[original_train[:, 2], original_train[:, 3]]
    train_ids = original_train[train_distance > 24]
    require((len(train_ids), len(validation_ids), len(test_ids)) == (10113, 30114, 14105),
            "Retained geometry differs from the fixed experiment.")
    require(set(train_ids[:, 0]) == set(range(6)), "A training class disappeared.")
    require(set(validation_ids[:, 0]) == {0, 1, 3, 5}, "Validation classes changed.")
    require(set(validation_ids[:, 1]) == VALIDATION_ROIS, "A validation ROI disappeared.")
    # A separate footprint check catches any mistaken radius or inclusive boundary.
    footprints = [maximum_filter(centers(ids).astype(np.uint8), size=25,
                                 mode="constant", cval=0) > 0
                  for ids in (train_ids, validation_ids, test_ids)]
    for first, second in ((0, 1), (0, 2), (1, 2)):
        require(not np.any(footprints[first] & footprints[second]), "Photograph footprints overlap.")
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
                            for roi in sorted(VALIDATION_ROIS)],
    }
    return train_ids, validation_ids, test_ids, doys, summary


def independent_metrics(labels, predicted):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    require(labels.ndim == 1 and predicted.shape == labels.shape and len(labels), "Misaligned prediction vector.")
    require(np.issubdtype(labels.dtype, np.integer) and np.issubdtype(predicted.dtype, np.integer),
            "Class IDs must be integers.")
    require(np.isin(labels, np.arange(6)).all() and np.isin(predicted, np.arange(6)).all(),
            "Class ID is outside 0 through 5.")
    present = np.unique(labels)
    recall = recall_score(labels, predicted, labels=np.arange(6), average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy_present_classes": float(recall_score(
            labels, predicted, labels=present, average="macro", zero_division=0)),
        "macro_f1_present_classes": float(f1_score(
            labels, predicted, labels=present, average="macro", zero_division=0)),
        "classes_present": present.tolist(),
        "recall": [float(recall[label]) if label in present else None for label in range(6)],
        "support": np.bincount(labels, minlength=6).tolist(),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=np.arange(6)).tolist(),
    }


def region_scores(ids, predicted):
    pooled = independent_metrics(ids[:, 0], predicted)
    regions, species_scores = [], []
    for label in sorted(set(ids[:, 0])):
        scores = []
        for roi in sorted(set(ids[ids[:, 0] == label, 1])):
            selected = (ids[:, 0] == label) & (ids[:, 1] == roi)
            score = float(accuracy_score(ids[selected, 0], predicted[selected]))
            scores.append(score)
            regions.append({"species_class": int(label), "roi_id": int(roi),
                            "n_pixels": int(selected.sum()), "accuracy": score})
        species_scores.append({"species_class": int(label), "mean_roi_accuracy": float(np.mean(scores))})
    return {"selection_score": float(np.mean([row["mean_roi_accuracy"] for row in species_scores])),
            "per_species": species_scores, "per_roi": regions, "pooled_metrics": pooled}


def check_oracles():
    # Unequal ROI sizes distinguish the selection score from pooled pixel accuracy.
    ids = np.array([[0, 10, 0, 0], [0, 11, 0, 1], [0, 11, 0, 2],
                    [0, 11, 0, 3], [1, 20, 0, 4], [1, 20, 0, 5]])
    scored = region_scores(ids, np.array([0, 1, 1, 1, 1, 1]))
    equal(scored["selection_score"], .75, "Known region-balanced score")
    equal(scored["pooled_metrics"]["accuracy"], .5, "Known pixel accuracy")
    equal(scored["pooled_metrics"]["balanced_accuracy_present_classes"], .625, "Known macro recall")
    equal(scored["pooled_metrics"]["macro_f1_present_classes"], 17 / 35, "Known macro F1")
    occupied = np.zeros((61, 61), dtype=bool)
    occupied[30, 30] = True
    distances = distance_transform_cdt(~occupied, metric="chessboard")
    np.testing.assert_array_equal(distances[[30, 30, 5], [54, 55, 5]], [24, 25, 25])


def verify_results(project_root=ROOT, *, validation_only=False, audit_preservation=False):
    root = Path(project_root).resolve()
    results = root / "results"
    check_oracles()
    protocol = read_json(root / "experiments/spatial_context_protocol.json")
    equal(protocol["window_sizes"], [5, 25], "Fixed windows")
    equal(protocol["validation_classes"], [0, 1, 3, 5], "Validation classes")
    equal(protocol["model"], {
        "family": "StandardScaler followed by multinomial LogisticRegression", "C": 10.0,
        "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000,
        "random_state": 42, "threads": 2}, "Fixed model")
    require(protocol["separation"]["max_context_radius"] == 12
            and protocol["separation"]["excluded_center_distance_inclusive"] == 24
            and protocol["separation"]["minimum_retained_center_distance"] == 25,
            "Fixed spatial separation changed.")
    validation_path = results / f"{PREFIX}_validation.json"
    selection_path = results / f"{PREFIX}_selection.json"
    predictions_path = results / f"{PREFIX}_validation_predictions.npz"
    report, selection = read_json(validation_path), read_json(selection_path)
    require(report["schema_version"] == selection["schema_version"] == 1, "Unsupported report schema.")
    require(report["hash_policy"] == selection["hash_policy"] == HASH_POLICY, "Unexpected hash policy.")
    equal(report["species"], SPECIES, "Species order")
    require(sha256(validation_path) == selection["validation_report_sha256"], "Validation report hash differs.")
    require(sha256(predictions_path) == report["predictions_sha256"]
            == selection["validation_predictions_sha256"], "Validation prediction hash differs.")
    require(report["sources_sha256"] == selection["sources_sha256"]
            and report["inputs_sha256"] == selection["inputs_sha256"], "Freeze provenance differs.")
    source_paths = {"experiments/spatial_context_protocol.json", "experiments/spatial_context.py",
                    "experiments/spatial_features.py"}
    check_hashes(root, report["sources_sha256"], source_paths)
    input_paths = {p.relative_to(root).as_posix() for p in (root / "raw/masks").glob("*.pgm")}
    input_paths |= {f"raw/images/1/2011_{day}_12.jpg" for day in range(242, 279)}
    input_paths.add("results/pixel_dataset.npz")
    check_hashes(root, report["inputs_sha256"], input_paths)
    # This is a historical audit, not an input requirement for saved predictions.
    # Later authorized notebook integration and reruns may revise these files.
    preserved = report["preserved_files_sha256"]
    require(isinstance(preserved, dict) and preserved, "Missing historical preservation snapshot.")
    if audit_preservation:
        check_hashes(root, preserved)

    train_ids, validation_ids, test_ids, doys, geometry = reconstruct_geometry(root)
    equal(report["geometry"], geometry, "Retained geometry")
    require([row["name"] for row in report["rows"]] == list(CANDIDATES), "Candidate list/order changed.")
    scores = {}
    with np.load(predictions_path, allow_pickle=False) as saved:
        require(set(saved.files) == {"ids_train", "ids_validation", "y_validation", "doys", *CANDIDATES},
                "Unexpected validation prediction arrays.")
        for key, expected in (("ids_train", train_ids), ("ids_validation", validation_ids),
                              ("y_validation", validation_ids[:, 0]), ("doys", doys)):
            np.testing.assert_array_equal(saved[key], expected, err_msg=f"Changed {key}")
        for row in report["rows"]:
            name = row["name"]
            require(row["n_features"] == FEATURE_COUNTS[name], f"Changed feature count: {name}")
            require(row["iterations"] and all(0 < n <= 2000 for n in row["iterations"]),
                    f"Invalid optimizer iteration count: {name}")
            score = region_scores(validation_ids, saved[name])
            equal(row["validation"], score, f"{name} validation")
            scores[name] = score["selection_score"]
    winner = "context_5" if scores["context_5"] >= scores["context_25"] else "context_25"
    require(report["selected_context"] == selection["selected_context"] == winner, "Wrong validation-selected context.")
    equal(selection["test_models"], ["raw_rgb", winner], "Frozen test candidates")
    require(set(selection["models"]) == set(CANDIDATES), "Frozen model list differs.")
    checked_models, absent_models = [], []
    for name, info in selection["models"].items():
        expected_path = f"experiments/artifacts/spatial_context/{name}.joblib"
        require(info["path"] == expected_path, f"Unexpected model location: {name}")
        path = checked_path(root, info["path"])
        if path.exists():
            require(sha256(path) == info["sha256"], f"Frozen model changed: {name}")
            checked_models.append(name)
        else:
            absent_models.append(name)
    completed, frozen = datetime.fromisoformat(report["completed_utc"]), datetime.fromisoformat(selection["frozen_utc"])
    require(completed.tzinfo is not None and frozen.tzinfo is not None and completed <= frozen,
            "Invalid validation/freeze timestamp sequence.")

    summary = {"status": "passed", "validation_only": validation_only, "geometry": geometry,
               "validation_models_checked": 3, "selected_context": winner,
               "model_files_checked": checked_models, "optional_model_files_absent": absent_models,
               "preserved_files_checked": len(preserved) if audit_preservation else 0,
               "historical_preservation_audit": "passed" if audit_preservation else "not_requested",
               "test_models_checked": 0}
    if validation_only:
        return summary

    final = read_json(results / f"{PREFIX}_results.json")
    require(final["hash_policy"] == HASH_POLICY, "Final hash policy differs.")
    test_predictions = results / f"{PREFIX}_predictions.npz"
    require(final["schema_version"] == 1 and final["selected_context"] == winner, "Final selection differs.")
    equal(final["species"], SPECIES, "Final species order")
    equal(final["samples"], geometry, "Final sample membership")
    require(final["selection_sha256"] == sha256(selection_path), "Final freeze checksum differs.")
    require(final["predictions_sha256"] == sha256(test_predictions), "Final prediction checksum differs.")
    ended = datetime.fromisoformat(final["completed_utc"])
    require(ended.tzinfo is not None and frozen <= ended, "Test completion precedes the recorded freeze.")
    require([row["name"] for row in final["rows"]] == ["raw_rgb", winner], "Final model rows differ.")
    final_metrics = {}
    with np.load(root / "results/pixel_dataset.npz", allow_pickle=False) as baseline, \
         np.load(test_predictions, allow_pickle=False) as saved:
        require(set(saved.files) == {"ids_test", "y_test", "doys", "raw_rgb", winner},
                "Unexpected test prediction arrays.")
        np.testing.assert_array_equal(baseline["y_test"], test_ids[:, 0])
        for key, expected in (("ids_test", test_ids), ("y_test", baseline["y_test"]), ("doys", doys)):
            np.testing.assert_array_equal(saved[key], expected, err_msg=f"Changed test {key}")
        for row in final["rows"]:
            name = row["name"]
            require(row["n_features"] == FEATURE_COUNTS[name], f"Changed test feature count: {name}")
            scored = independent_metrics(baseline["y_test"], saved[name])
            require(scored["classes_present"] == list(range(6)), "Test evaluation omits a species.")
            equal(row["metrics"], scored, f"{name} test metrics")
            final_metrics[name] = scored
    difference = {key: final_metrics[winner][key] - final_metrics["raw_rgb"][key]
                  for key in ("accuracy", "balanced_accuracy_present_classes", "macro_f1_present_classes")}
    equal(final["context_minus_raw"], difference, "Context minus raw differences")
    summary["test_models_checked"] = 2
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--audit-preservation", action="store_true",
                        help="Also compare files with the historical pre-integration snapshot")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validation-only", action="store_true")
    modes.add_argument("--check-only", action="store_true", help="Run tiny independent metric and geometry oracles")
    args = parser.parse_args()
    if args.check_only:
        check_oracles()
        print("PASS: independent region weighting, sklearn metrics and inclusive spatial-buffer oracles.")
        return
    summary = verify_results(args.project_root, validation_only=args.validation_only,
                             audit_preservation=args.audit_preservation)
    geometry = summary["geometry"]
    print(f"PASS: exact geometry: {geometry['train']:,} fitting, {geometry['validation']:,} validation, "
          f"{geometry['test']:,} original test pixels; non-overlapping 25x25 footprints.")
    print(f"PASS: all three validation candidates and selected {summary['selected_context']}; "
          "equal ROI weighting within four species; fixed tie rule.")
    print(f"PASS: experiment source/input/output hashes; "
          f"{len(summary['model_files_checked'])} local model hashes checked.")
    if args.audit_preservation:
        print(f"PASS: {summary['preserved_files_checked']} files match the historical preservation snapshot.")
    else:
        print("Historical notebook/ZIP/output preservation is a separate audit: --audit-preservation.")
    if summary["test_models_checked"]:
        print("PASS: both frozen test models, all six species, sklearn metrics/confusion matrices and paired differences.")
    else:
        print("PASS: validation-only check; test labels and new test predictions were not read.")
    if summary["optional_model_files_absent"]:
        print("Saved predictions verified without optional local fitted models.")


if __name__ == "__main__":
    main()
