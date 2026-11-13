"""Verify saved temporal-CNN results in the core environment, without PyTorch.

No model is fitted or loaded. Geometry and selected-epoch metrics are reconstructed
from masks and saved predictions. Other epoch summaries are checked for internal
consistency; their individual prediction vectors are not part of the saved report.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.check_spatial_results import (
    SPECIES, check_oracles, checked_path, equal, independent_metrics,
    read_json, reconstruct_geometry, region_scores, require, verify_dataset_reencoding,
)


ROOT = Path(__file__).resolve().parents[1]
NAME = "temporal_cnn_context_25"
HASH_POLICY = "SHA-256; .py/.json/.md normalize CRLF to LF; all other files use raw bytes"
CONFIG = {
    "seed": 42, "threads": 2, "epochs": 20, "batch_size": 256,
    "channels": [9, 32, 64], "kernel_sizes": [5, 3], "padding": [2, 1],
    "activation": "GELU", "dropout": .2,
    "pooling": "concatenate temporal global mean and maximum", "head": "Linear(128, 6)",
    "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .01,
    "loss": "cross entropy weighted by training n / (6 * class count)",
    "normalization": "per-channel mean and population std over training samples and time only; std <= 1e-8 replaced with 1",
    "augmentation": "none",
}


def sha256(path):
    path = Path(path)
    if path.suffix.lower() in {".py", ".json", ".md"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hashes(root, recorded, expected_paths):
    require(isinstance(recorded, dict) and set(recorded) == set(expected_paths),
            "Temporal hash manifest has missing or unexpected paths.")
    for relative, expected in recorded.items():
        path = checked_path(root, relative)
        require(path.is_file(), f"Temporal source/input changed: {relative}")
        if sha256(path) != expected:
            if relative == "results/pixel_dataset.npz":
                verify_dataset_reencoding(root, path, expected)
            else:
                raise AssertionError(f"Temporal source/input changed: {relative}")


def check_epoch_summary(scored, ids):
    """Check weighting and metrics from reported counts, without claiming epoch replay."""
    expected_regions = [(int(label), int(roi)) for label in np.unique(ids[:, 0])
                        for roi in np.unique(ids[ids[:, 0] == label, 1])]
    require([(row["species_class"], row["roi_id"]) for row in scored["per_roi"]] == expected_regions,
            "Epoch summary has different validation regions.")
    species_scores, correct_by_species = [], np.zeros(6, dtype=float)
    for label in np.unique(ids[:, 0]):
        accuracies = []
        for row in scored["per_roi"]:
            if row["species_class"] != label:
                continue
            mask = (ids[:, 0] == label) & (ids[:, 1] == row["roi_id"])
            require(row["n_pixels"] == int(mask.sum()), "Epoch ROI support differs.")
            accuracy = row["accuracy"]
            require(np.isfinite(accuracy) and 0 <= accuracy <= 1, "Invalid epoch ROI accuracy.")
            correct = accuracy * row["n_pixels"]
            np.testing.assert_allclose(correct, round(correct), rtol=0, atol=1e-6,
                                       err_msg="ROI accuracy is inconsistent with an integer count")
            correct_by_species[label] += correct
            accuracies.append(accuracy)
        species_scores.append({"species_class": int(label), "mean_roi_accuracy": float(np.mean(accuracies))})
    equal(scored["per_species"], species_scores, "Epoch equal-ROI species means")
    equal(scored["selection_score"], float(np.mean([row["mean_roi_accuracy"] for row in species_scores])),
          "Epoch four-species selection score")

    matrix = np.asarray(scored["pooled_metrics"]["confusion_matrix"])
    require(matrix.shape == (6, 6) and np.issubdtype(matrix.dtype, np.integer)
            and np.all(matrix >= 0), "Invalid epoch confusion matrix.")
    np.testing.assert_array_equal(matrix.sum(axis=1), np.bincount(ids[:, 0], minlength=6))
    np.testing.assert_allclose(np.diag(matrix), correct_by_species, rtol=0, atol=1e-6)
    labels = np.repeat(np.arange(6), matrix.sum(axis=1))
    predicted = np.concatenate([np.repeat(np.arange(6), matrix[label]) for label in range(6)])
    equal(scored["pooled_metrics"], independent_metrics(labels, predicted), "Epoch confusion-derived metrics")


def verify_results(project_root=ROOT, *, validation_only=False):
    root = Path(project_root).resolve()
    results = root / "results"
    check_oracles()
    protocol = read_json(root / "experiments/temporal_context_protocol.json")
    require(protocol["schema_version"] == 1 and protocol["window_size"] == 25, "Unexpected temporal protocol.")
    equal(protocol["model"], CONFIG, "Fixed temporal model configuration")
    validation_path = results / "temporal_context_validation.json"
    selection_path = results / "temporal_context_selection.json"
    predictions_path = results / "temporal_context_validation_predictions.npz"
    report, selection = read_json(validation_path), read_json(selection_path)
    require(report["schema_version"] == selection["schema_version"] == 1, "Unsupported temporal schema.")
    require(report["hash_policy"] == selection["hash_policy"] == HASH_POLICY, "Temporal hash policy differs.")
    require(report["model_name"] == selection["model_name"] == NAME, "Unexpected temporal model.")
    equal(report["species"], SPECIES, "Temporal species order")
    equal(report["model_config"], CONFIG, "Recorded temporal model settings")
    require(report["n_parameters"] == 8454, "Temporal parameter count differs.")
    require(selection["validation_report_sha256"] == sha256(validation_path), "Temporal validation hash differs.")
    require(selection["validation_predictions_sha256"] == report["predictions_sha256"]
            == sha256(predictions_path), "Temporal validation prediction hash differs.")
    require(report["sources_sha256"] == selection["sources_sha256"]
            and report["inputs_sha256"] == selection["inputs_sha256"], "Temporal frozen provenance differs.")
    source_paths = {"experiments/temporal_context.py", "experiments/temporal_context_protocol.json",
                    "experiments/spatial_context.py", "experiments/spatial_features.py",
                    "experiments/spatial_context_protocol.json"}
    check_hashes(root, report["sources_sha256"], source_paths)
    input_paths = {p.relative_to(root).as_posix() for p in (root / "raw/masks").glob("*.pgm")}
    input_paths |= {f"raw/images/1/2011_{day}_12.jpg" for day in range(242, 279)}
    input_paths.add("results/pixel_dataset.npz")
    check_hashes(root, report["inputs_sha256"], input_paths)

    train_ids, validation_ids, test_ids, doys, geometry = reconstruct_geometry(root)
    equal(report["geometry"], geometry, "Temporal exact geometry")
    scaling = report["scaling"]
    for key in ("channel_mean", "channel_std"):
        values = np.asarray(scaling[key], dtype=float)
        require(values.shape == (9,) and np.isfinite(values).all(), f"Invalid temporal {key}.")
        require(np.all(values >= 0) and np.all(values <= 1), f"Invalid RGB-statistic {key} range.")
    require(np.all(np.asarray(scaling["channel_std"]) > 0), "Temporal channel scales must be positive.")
    counts = np.bincount(train_ids[:, 0], minlength=6)
    equal(scaling["training_counts"], counts.tolist(), "Temporal training class counts")
    expected_weights = (len(train_ids) / (6 * counts)).astype(np.float32).tolist()
    equal(scaling["class_weights"], expected_weights, "Training-only balanced class weights")

    history = report["history"]
    require([row["epoch"] for row in history] == list(range(1, 21)), "Expected exactly twenty ordered epochs.")
    for row in history:
        require(np.isfinite(row["train_loss"]) and row["train_loss"] >= 0, "Invalid temporal training loss.")
        check_epoch_summary(row["validation"], validation_ids)
    chosen = max(history, key=lambda row: row["validation"]["selection_score"])
    epoch = chosen["epoch"]
    require(report["selected_epoch"] == selection["selected_epoch"] == epoch,
            "Temporal epoch selection or earliest-epoch tie rule differs.")
    with np.load(predictions_path, allow_pickle=False) as saved:
        require(set(saved.files) == {"ids_train", "ids_validation", "y_validation", "doys", NAME},
                "Unexpected temporal validation arrays.")
        for key, expected in (("ids_train", train_ids), ("ids_validation", validation_ids),
                              ("y_validation", validation_ids[:, 0]), ("doys", doys)):
            np.testing.assert_array_equal(saved[key], expected, err_msg=f"Changed temporal {key}")
        scored = region_scores(validation_ids, saved[NAME])
        equal(report["best_validation"], scored, "Selected temporal validation predictions")
        equal(chosen["validation"], scored, "Selected history epoch and prediction alignment")
    checkpoint = selection["checkpoint"]
    require(checkpoint["path"] == "experiments/artifacts/temporal_context/selected_checkpoint.pt",
            "Unexpected temporal checkpoint path.")
    path = checked_path(root, checkpoint["path"])
    checkpoint_checked = path.is_file()
    if checkpoint_checked:
        require(sha256(path) == checkpoint["sha256"], "Frozen temporal checkpoint changed.")
    completed, frozen = datetime.fromisoformat(report["completed_utc"]), datetime.fromisoformat(selection["frozen_utc"])
    require(completed.tzinfo is not None and frozen.tzinfo is not None and completed <= frozen,
            "Temporal validation/freeze timestamp sequence differs.")
    summary = {"status": "passed", "validation_only": validation_only, "geometry": geometry,
               "history_epochs_checked": 20, "selected_epoch": epoch, "checkpoint_hash_checked": checkpoint_checked,
               "torch_required": False, "test_models_checked": 0,
               "history_scope": "All epoch summaries internally checked; selected epoch predictions independently verified"}
    if validation_only:
        return summary

    final = read_json(results / "temporal_context_results.json")
    test_predictions = results / "temporal_context_predictions.npz"
    require(final["schema_version"] == 1 and final["hash_policy"] == HASH_POLICY, "Final temporal schema/policy differs.")
    require(final["model_name"] == NAME and final["selected_epoch"] == epoch, "Final temporal selection differs.")
    equal(final["species"], SPECIES, "Final temporal species order")
    equal(final["samples"], geometry, "Final temporal sample membership")
    require(final["selection_sha256"] == sha256(selection_path), "Final temporal selection checksum differs.")
    require(final["predictions_sha256"] == sha256(test_predictions), "Final temporal prediction checksum differs.")
    require(final["sources_sha256"] == selection["sources_sha256"]
            and final["inputs_sha256"] == selection["inputs_sha256"], "Final temporal provenance differs.")
    ended = datetime.fromisoformat(final["completed_utc"])
    require(ended.tzinfo is not None and frozen <= ended, "Temporal test completion precedes freezing.")
    require(len(final["rows"]) == 1 and final["rows"][0]["name"] == NAME
            and final["rows"][0]["n_features"] == 333, "Final temporal model rows differ.")
    with np.load(root / "results/pixel_dataset.npz", allow_pickle=False) as baseline, \
         np.load(test_predictions, allow_pickle=False) as saved:
        require(set(saved.files) == {"ids_test", "y_test", "doys", NAME}, "Unexpected temporal test arrays.")
        np.testing.assert_array_equal(baseline["y_test"], test_ids[:, 0])
        for key, expected in (("ids_test", test_ids), ("y_test", baseline["y_test"]), ("doys", doys)):
            np.testing.assert_array_equal(saved[key], expected, err_msg=f"Changed temporal test {key}")
        scored = independent_metrics(baseline["y_test"], saved[NAME])
        require(scored["classes_present"] == list(range(6)), "Temporal test omits a species.")
        equal(final["rows"][0]["metrics"], scored, "Temporal test metrics and confusion matrix")
    summary["test_models_checked"] = 1
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    summary = verify_results(args.project_root, validation_only=args.validation_only)
    print("PASS: original masks, exact buffered pixel membership, 37 days and non-overlapping contexts.")
    print(f"PASS: twenty epoch summaries, earliest-best epoch {summary['selected_epoch']}, "
          "and independently recomputed selected validation predictions.")
    print("PASS: declared model/scaling metadata, training class weights and frozen source/input/output checksums.")
    print("PASS: checkpoint bytes checked." if summary["checkpoint_hash_checked"]
          else "Saved predictions verified without the optional local checkpoint.")
    if summary["test_models_checked"]:
        print("PASS: all 14,105 original test pixels, six classes, sklearn metrics and confusion matrix; no PyTorch needed.")
    else:
        print("PASS: validation-only check; y_test and new test predictions were not read.")


if __name__ == "__main__":
    main()
