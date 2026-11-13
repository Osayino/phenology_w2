"""Fixed temporal CNN experiment, separate from the submitted notebook.

Use the optional experiment environment. Run --benchmark first, then --fit and
--evaluate once each. The notebook may display the saved reports without torch.
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch
from torch import nn

# The existing spatial runner imports its sibling by filename, so retain its
# directory on the search path for both command-line and package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
if __package__ in (None, ""):
    from spatial_context import dataset_geometry, input_hashes, metrics, region_validation, SPECIES
    from spatial_features import extract_features
else:
    from .spatial_context import dataset_geometry, input_hashes, metrics, region_validation, SPECIES
    from .spatial_features import extract_features


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "experiments/artifacts/temporal_context"
PROTOCOL = ROOT / "experiments/temporal_context_protocol.json"
NAME = "temporal_cnn_context_25"
HASH_POLICY = "SHA-256; .py/.json/.md normalize CRLF to LF; all other files use raw bytes"
CONFIG = {
    "seed": 42, "threads": 2, "epochs": 20, "batch_size": 256,
    "channels": [9, 32, 64], "kernel_sizes": [5, 3], "padding": [2, 1],
    "activation": "GELU", "dropout": 0.2,
    "pooling": "concatenate temporal global mean and maximum", "head": "Linear(128, 6)",
    "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .01,
    "loss": "cross entropy weighted by training n / (6 * class count)",
    "normalization": "per-channel mean and population std over training samples and time only; std <= 1e-8 replaced with 1",
    "augmentation": "none",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    path = Path(path)
    if path.suffix.lower() in {".py", ".json", ".md"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def source_hashes():
    paths = [Path(__file__), PROTOCOL, ROOT / "experiments/spatial_context.py",
             ROOT / "experiments/spatial_features.py", ROOT / "experiments/spatial_context_protocol.json"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}


def verify_hashes(recorded):
    for relative, expected in recorded.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen input or source changed: {relative}")


def setup():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["model"] != CONFIG or protocol["window_size"] != 25:
        raise ValueError("Runner and fixed temporal protocol disagree.")
    torch.set_num_threads(CONFIG["threads"])
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.use_deterministic_algorithms(True)
    return protocol


class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=5, padding=2), nn.GELU(), nn.Dropout(.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(.2),
        )
        self.classifier = nn.Linear(128, 6)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (9, 37):
            raise ValueError("Temporal CNN input must have shape (n, 9, 37).")
        hidden = self.convolutions(values)
        pooled = torch.cat((hidden.mean(dim=2), hidden.amax(dim=2)), dim=1)
        return self.classifier(pooled)


def fit_scaling(training):
    if training.ndim != 3 or training.shape[1:] != (9, 37) or not np.isfinite(training).all():
        raise ValueError("Training features must be finite with shape (n, 9, 37).")
    mean = training.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = training.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std[std <= 1e-8] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Channel scaling is not finite.")
    return mean, std


def normalized_tensor(features, mean, std):
    values = np.asarray(features, dtype=np.float32).reshape(-1, 9, 37)
    if np.shape(mean) != (9,) or np.shape(std) != (9,) or np.any(std <= 0):
        raise ValueError("Expected nine finite positive channel scales.")
    scaled = (values - mean[None, :, None]) / std[None, :, None]
    if not np.isfinite(scaled).all():
        raise ValueError("Normalized features are not finite.")
    return torch.from_numpy(np.ascontiguousarray(scaled))


def predict(model, values):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(values), CONFIG["batch_size"]):
            logits = model(values[start:start + CONFIG["batch_size"]])
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite model logits.")
            predictions.append(logits.argmax(dim=1).numpy())
    return np.concatenate(predictions).astype(np.int64, copy=False)


def benchmark():
    """Synthetic timing only; never opens dataset labels or saves fitted weights."""
    setup()
    model = TemporalCNN()
    values = torch.randn(CONFIG["batch_size"], 9, 37)
    labels = torch.arange(CONFIG["batch_size"]) % 6
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    criterion = nn.CrossEntropyLoss(weight=torch.ones(6))
    for step in range(18):
        if step == 2:
            started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        if logits.shape != (CONFIG["batch_size"], 6) or not torch.isfinite(logits).all():
            raise ValueError("Benchmark shape/finite check failed.")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    training_step = (time.perf_counter() - started) / 16
    started = time.perf_counter()
    for _ in range(8):
        predict(model, values)
    prediction_step = (time.perf_counter() - started) / 8
    # A separate constant-channel check guards scaling and source preservation.
    synthetic = np.ones((4, 9, 37), dtype=np.float32)
    before = synthetic.copy()
    mean, std = fit_scaling(synthetic)
    normalized = normalized_tensor(synthetic, mean, std)
    np.testing.assert_array_equal(synthetic, before)
    assert normalized.shape == (4, 9, 37) and torch.isfinite(normalized).all()
    assert torch.count_nonzero(normalized).item() == 0
    report = {"scope": "synthetic benchmark, not a scientific fit", "parameters": sum(p.numel() for p in model.parameters()),
              "training_batch_seconds": training_step, "prediction_batch_seconds": prediction_step,
              "estimated_20_epoch_seconds_excluding_feature_extraction": 20 * (
                  math.ceil(10113 / 256) * training_step + math.ceil(30114 / 256) * prediction_step),
              "shape_and_scaling_checks": "passed", "torch_version": torch.__version__}
    print(json.dumps(report, indent=2), flush=True)


def fit():
    validation_path = RESULTS / "temporal_context_validation.json"
    selection_path = RESULTS / "temporal_context_selection.json"
    if validation_path.exists() or selection_path.exists():
        raise FileExistsError("A temporal validation/selection record already exists; refusing to overwrite.")
    protocol = setup()
    started = time.perf_counter()
    train_ids, validation_ids, _, doys, geometry = dataset_geometry()
    np.testing.assert_array_equal(doys, np.arange(242, 279))
    sources, inputs = source_hashes(), input_hashes(doys)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACTS / "selected_checkpoint.pt"
    feature_started = time.perf_counter()
    coordinates = np.concatenate((train_ids[:, 2:4], validation_ids[:, 2:4]))
    features = extract_features(ROOT, coordinates, doys, window_sizes=(25,), progress=True)["context_25"]
    extraction_seconds = time.perf_counter() - feature_started
    training = features[:len(train_ids)].reshape(-1, 9, 37)
    mean, std = fit_scaling(training)
    train_tensor = normalized_tensor(training, mean, std)
    validation_tensor = normalized_tensor(features[len(train_ids):], mean, std)
    del features, training
    counts = np.bincount(train_ids[:, 0], minlength=6)
    if np.any(counts <= 0):
        raise ValueError("Every training class needs positive support.")
    weights = (len(train_ids) / (6 * counts)).astype(np.float32)
    targets = torch.from_numpy(train_ids[:, 0].astype(np.int64))
    model = TemporalCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    class_weights = torch.from_numpy(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    history, best_score, best_predictions, best_epoch, best_validation = [], -1.0, None, None, None
    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_tensor), generator=generator)
        total_loss, total_weight = 0.0, 0.0
        for start in range(0, len(order), CONFIG["batch_size"]):
            index = order[start:start + CONFIG["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_tensor[index])
            loss = criterion(logits, targets[index])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; no test evaluation.")
            loss.backward()
            optimizer.step()
            batch_weight = float(class_weights[targets[index]].sum())
            total_loss += float(loss.detach()) * batch_weight
            total_weight += batch_weight
        train_seconds = time.perf_counter() - epoch_started
        prediction_started = time.perf_counter()
        predicted = predict(model, validation_tensor)
        predict_seconds = time.perf_counter() - prediction_started
        scored = region_validation(validation_ids, predicted)
        history.append({"epoch": epoch, "train_loss": total_loss / total_weight, "validation": scored,
                        "train_seconds": train_seconds, "validation_predict_seconds": predict_seconds})
        if scored["selection_score"] > best_score:
            best_score, best_epoch, best_validation = scored["selection_score"], epoch, scored
            best_predictions = predicted.copy()
            temporary = checkpoint_path.with_suffix(".tmp.pt")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "channel_mean": torch.from_numpy(mean), "channel_std": torch.from_numpy(std),
                        "class_weights": class_weights}, temporary)
            temporary.replace(checkpoint_path)
        print(f"Epoch {epoch:02d}/20: loss {total_loss / total_weight:.5f}; four-species mean ROI {scored['selection_score']:.4%}; "
              f"train {train_seconds:.2f}s, validation {predict_seconds:.2f}s; best epoch {best_epoch}", flush=True)
    predictions_path = RESULTS / "temporal_context_validation_predictions.npz"
    write_npz(predictions_path, ids_train=train_ids, ids_validation=validation_ids,
              y_validation=validation_ids[:, 0], doys=doys, **{NAME: best_predictions})
    verify_hashes(sources)
    verify_hashes(inputs)
    report = {"schema_version": 1, "hash_policy": HASH_POLICY, "completed_utc": now(), "species": SPECIES,
              "model_name": NAME, "model_config": CONFIG, "n_parameters": sum(p.numel() for p in model.parameters()),
              "geometry": geometry, "selected_epoch": best_epoch, "best_validation": best_validation,
              "history": history, "scaling": {"channel_mean": mean.tolist(), "channel_std": std.tolist(),
                                               "class_weights": weights.tolist(), "training_counts": counts.tolist()},
              "sources_sha256": sources, "inputs_sha256": inputs,
              "predictions_sha256": sha256(predictions_path),
              "feature_extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - started,
              "versions": {"numpy": np.__version__, "torch": torch.__version__},
              "limitations": protocol["limitations"]}
    write_json(validation_path, report)
    selection = {"schema_version": 1, "hash_policy": HASH_POLICY, "frozen_utc": now(),
                 "model_name": NAME, "selected_epoch": best_epoch,
                 "checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
                                "sha256": sha256(checkpoint_path)},
                 "validation_report_sha256": sha256(validation_path),
                 "validation_predictions_sha256": sha256(predictions_path),
                 "sources_sha256": sources, "inputs_sha256": inputs}
    write_json(selection_path, selection)
    print(f"Frozen epoch {best_epoch}; validation score {best_score:.4%}. No test predictions computed.", flush=True)


def evaluate():
    report_path = RESULTS / "temporal_context_results.json"
    if report_path.exists():
        raise FileExistsError("Temporal test results already exist; refusing to rerun or overwrite.")
    setup()
    started = time.perf_counter()
    selection_path = RESULTS / "temporal_context_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validation_path = RESULTS / "temporal_context_validation.json"
    validation_predictions_path = RESULTS / "temporal_context_validation_predictions.npz"
    if selection["hash_policy"] != HASH_POLICY:
        raise ValueError("Unexpected frozen hash policy.")
    if sha256(validation_path) != selection["validation_report_sha256"] or sha256(validation_predictions_path) != selection["validation_predictions_sha256"]:
        raise ValueError("Validation report or selected predictions changed after freezing.")
    verify_hashes(selection["sources_sha256"])
    verify_hashes(selection["inputs_sha256"])
    checkpoint_path = ROOT / selection["checkpoint"]["path"]
    if sha256(checkpoint_path) != selection["checkpoint"]["sha256"]:
        raise ValueError("Selected model changed after freezing.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != selection["selected_epoch"]:
        raise ValueError("Checkpoint epoch differs from the frozen selection.")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    with np.load(RESULTS / "pixel_dataset.npz", allow_pickle=False) as data:
        test_ids, y_test, doys = data["ids_test"], data["y_test"], data["doys"]
    np.testing.assert_array_equal(test_ids[:, 0], y_test)
    np.testing.assert_array_equal(doys, np.arange(242, 279))
    feature_started = time.perf_counter()
    features = extract_features(ROOT, test_ids[:, 2:4], doys, window_sizes=(25,), progress=True)["context_25"]
    tensor = normalized_tensor(features, checkpoint["channel_mean"].numpy(), checkpoint["channel_std"].numpy())
    extraction_seconds = time.perf_counter() - feature_started
    del features
    model = TemporalCNN()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    prediction_started = time.perf_counter()
    predicted = predict(model, tensor)
    predict_seconds = time.perf_counter() - prediction_started
    scored = metrics(y_test, predicted)
    predictions_path = RESULTS / "temporal_context_predictions.npz"
    write_npz(predictions_path, ids_test=test_ids, y_test=y_test, doys=doys, **{NAME: predicted})
    report = {"schema_version": 1, "hash_policy": HASH_POLICY, "completed_utc": now(), "species": SPECIES,
              "model_name": NAME, "selected_epoch": selection["selected_epoch"], "samples": validation["geometry"],
              "selection_sha256": sha256(selection_path), "predictions_sha256": sha256(predictions_path),
              "sources_sha256": selection["sources_sha256"], "inputs_sha256": selection["inputs_sha256"],
              "feature_extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - started,
              "rows": [{"name": NAME, "n_features": 333, "predict_seconds": predict_seconds, "metrics": scored}],
              "limitations": validation["limitations"]}
    write_json(report_path, report)
    print(f"Temporal CNN epoch {selection['selected_epoch']}: accuracy {scored['accuracy']:.4%}; "
          f"balanced accuracy {scored['balanced_accuracy_present_classes']:.4%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--benchmark", action="store_true")
    stages.add_argument("--fit", action="store_true")
    stages.add_argument("--evaluate", action="store_true")
    arguments = parser.parse_args()
    if arguments.benchmark:
        benchmark()
    elif arguments.fit:
        fit()
    else:
        evaluate()
