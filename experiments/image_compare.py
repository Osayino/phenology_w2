"""Frozen ImageNet image features for the optional phenology comparison.

The CLI benchmarks inference or fits heads using training/validation rows only.
It never reads test labels. The required notebook and its saved results are inputs,
not outputs of this experiment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mobilenet_v3_small", "vit_b_32")
HEAD_C_VALUES = (0.1, 1.0, 10.0)
PREPROCESSING = {
    "resize": [224, 224], "interpolation": "bilinear", "antialias": True,
    "crop": None, "augmentation": None,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "adaptation": "Resize the entire RP to preserve both temporal boundaries; no ImageNet centre crop.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_directory(root: Path) -> Path:
    directory = root / "experiments" / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def recurrence_source(root: Path, split: str = "trainval"):
    """Open the verified RP array without decoding any species labels."""
    if split not in ("trainval", "test"):
        raise ValueError("split must be trainval or test")
    experiment = json.loads((root / "results/experiment_results.json").read_text())
    key = experiment["recurrence_cache"][0 if split == "trainval" else 1]
    path = root / "cache" / f"rp_{split}_{key}.npy"
    info = json.loads(path.with_suffix(".json").read_text())
    assert info["cache_key"] == key
    # I check the current raw feature content and row order before reusing images.
    # This loads features only; test species labels are never decoded here.
    with np.load(root / "results/pixel_dataset.npz", allow_pickle=False) as data:
        features = data[f"X_{split}"]
    recipe = info["recipe"]
    if list(features.shape) != recipe["input_shape"] or str(features.dtype) != recipe["input_dtype"]:
        raise ValueError("Current pixel features do not match the recurrence cache input shape/dtype.")
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode("utf-8"))
    for first in range(0, len(features), 1024):
        digest.update(np.ascontiguousarray(features[first:first + 1024]).tobytes())
    if digest.hexdigest() != key:
        raise ValueError("Recurrence cache was built from different pixel features or row order.")
    assert sha256_file(path) == info["file_sha256"], "Recurrence input checksum mismatch"
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    assert values.dtype == np.uint8 and values.ndim == 2 and values.shape[1] == 58800
    return values, {"cache_key": key, "file_sha256": info["file_sha256"], "rows": len(values)}


def load_backbone(model_name: str, root: Path, threads: int = 2):
    """Use official, checksum-checked weights and discard the ImageNet classifier."""
    import torch
    import torchvision
    from torchvision import models

    torch.set_num_threads(threads)
    torch.manual_seed(42)
    torch.hub.set_dir(str(artifact_directory(root) / "image_weights"))
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=None)
        feature_width = 576
    elif model_name == "vit_b_32":
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1
        model = models.vit_b_32(weights=None)
        feature_width = 768
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # These are official pretrained tensors, not code from the archived notebooks.
    state = weights.get_state_dict(progress=True, check_hash=True, weights_only=True)
    model.load_state_dict(state)
    del state
    if model_name == "mobilenet_v3_small":
        model.classifier = torch.nn.Identity()
    else:
        model.heads = torch.nn.Identity()
    model.eval().requires_grad_(False)
    metadata = {
        "model": model_name, "weights": str(weights), "weights_url": weights.url,
        "feature_width": feature_width, "preprocessing": PREPROCESSING,
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "device": "cpu", "threads": threads,
    }
    return model, metadata


def image_batch(rows: np.ndarray):
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform

    # Keep the entire recurrence matrix: a crop would remove early or late positions.
    images = torch.from_numpy(np.array(rows, dtype=np.uint8, copy=True).reshape(-1, 140, 140, 3))
    images = images.permute(0, 3, 1, 2)
    images = transform.resize(images, [224, 224], interpolation=InterpolationMode.BILINEAR,
                              antialias=True)
    images = images.to(torch.float32).div_(255)
    return transform.normalize(images, mean=PREPROCESSING["mean"], std=PREPROCESSING["std"])


def benchmark(model_name: str, root: Path = ROOT, *, batch_size: int = 8,
              threads: int = 2, count: int = 100) -> dict:
    import torch

    source, source_info = recurrence_source(root)
    load_started = time.perf_counter()
    model, metadata = load_backbone(model_name, root, threads)
    model_load_seconds = time.perf_counter() - load_started
    with torch.inference_mode():
        model(image_batch(source[:batch_size]))
        started = time.perf_counter()
        for first in range(0, count, batch_size):
            output = model(image_batch(source[first:min(first + batch_size, count)]))
            assert output.ndim == 2 and output.shape[1] == metadata["feature_width"]
    elapsed = time.perf_counter() - started
    source._mmap.close()
    result = {
        **metadata, "benchmark_images": count, "batch_size": batch_size,
        "model_load_seconds": model_load_seconds, "inference_seconds": elapsed,
        "seconds_per_image": elapsed / count,
        "estimated_trainval_minutes": elapsed / count * source_info["rows"] / 60,
        "estimated_all_pixels_minutes": elapsed / count * 27850 / 60,
        "scope": "Measured on training/validation images only; no test labels accessed.",
    }
    path = artifact_directory(root) / f"image_{model_name}_benchmark.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def extract_embeddings(model_name: str, root: Path = ROOT, *, split: str = "trainval",
                       batch_size: int = 8, threads: int = 2):
    """Return cached frozen image features; this function never reads labels."""
    import torch

    source, source_info = recurrence_source(root, split)
    model, model_info = load_backbone(model_name, root, threads)
    recipe = {**model_info, "source": source_info, "split": split,
              "script_sha256": sha256_file(Path(__file__))}
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    directory = artifact_directory(root)
    path = directory / f"image_{model_name}_{split}_{key[:16]}.npy"
    info_path = path.with_suffix(".json")
    if path.exists() and info_path.exists():
        saved = json.loads(info_path.read_text())
        if saved.get("recipe") == recipe and saved.get("sha256") == sha256_file(path):
            source._mmap.close()
            return np.load(path, mmap_mode="r", allow_pickle=False), saved

    temporary = path.with_suffix(".tmp.npy")
    mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                     shape=(len(source), model_info["feature_width"]))
    started = time.perf_counter()
    last_progress = started
    try:
        with torch.inference_mode():
            for first in range(0, len(source), batch_size):
                stop = min(first + batch_size, len(source))
                features = model(image_batch(source[first:stop])).numpy()
                assert features.shape == (stop - first, model_info["feature_width"])
                assert np.isfinite(features).all()
                mapped[first:stop] = features
                if time.perf_counter() - last_progress >= 30 or stop == len(source):
                    print(f"{model_name} {split}: {stop:,}/{len(source):,} images; "
                          f"{time.perf_counter() - started:.1f}s", flush=True)
                    last_progress = time.perf_counter()
        mapped.flush()
        mapped._mmap.close()
        mapped = None
        os.replace(temporary, path)
        result = {"recipe": recipe, "key": key, "sha256": sha256_file(path),
                  "feature_path": path.relative_to(root).as_posix(),
                  "generation_seconds": time.perf_counter() - started}
        info_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        source._mmap.close()
        if mapped is not None:
            mapped._mmap.close()
        if temporary.exists():
            temporary.unlink()
        del model
        gc.collect()
    return np.load(path, mmap_mode="r", allow_pickle=False), result


def fit_validation(model_name: str, root: Path = ROOT, *, batch_size: int = 8,
                   threads: int = 2) -> dict:
    features, feature_info = extract_embeddings(model_name, root, batch_size=batch_size, threads=threads)
    data_path = root / "results/pixel_dataset.npz"
    with np.load(data_path, allow_pickle=False) as data:
        # Only training-pool labels are decoded. The test arrays stay unopened.
        labels = data["y_trainval"]
        train = data["train_indices"]
        validation = data["val_indices"]
    assert features.shape[0] == len(labels) == 13745
    assert len(train) == 10996 and len(validation) == 2749
    assert not np.intersect1d(train, validation).size
    candidates = []
    fitted_heads = []
    with threadpool_limits(limits=threads):
        for c_value in HEAD_C_VALUES:
            # I learn a small classifier on frozen image features; the backbone stays fixed.
            head = make_pipeline(StandardScaler(), LogisticRegression(
                C=c_value, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=42))
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                head.fit(features[train], labels[train])
            prediction = head.predict(features[validation])
            candidates.append({
                "C": c_value, "validation_accuracy": float(accuracy_score(labels[validation], prediction)),
                "validation_macro_f1": float(f1_score(labels[validation], prediction, average="macro", zero_division=0)),
                "validation_balanced_accuracy": float(recall_score(labels[validation], prediction, average="macro", zero_division=0)),
                "fit_and_validation_seconds": time.perf_counter() - started,
                "convergence_warnings": [str(w.message) for w in captured],
            })
            fitted_heads.append(head)
            print(json.dumps(candidates[-1]), flush=True)
    selected_index = max(range(len(candidates)), key=lambda i: candidates[i]["validation_accuracy"])
    head_path = artifact_directory(root) / f"image_{model_name}_selected_head.joblib"
    joblib.dump(fitted_heads[selected_index], head_path)
    result = {
        "model_family": "frozen ImageNet image features plus multinomial logistic regression",
        "model": model_name, "dataset_sha256": sha256_file(data_path),
        "feature_info": feature_info, "candidates": candidates, "selected": candidates[selected_index],
        "selection_rule": "highest validation accuracy; smaller C wins a tie",
        "head_path": head_path.relative_to(root).as_posix(),
        "train_rows": len(train), "validation_rows": len(validation),
        "test_evaluated": False,
        "limitations": ["ImageNet features are transferred to recurrence images, not trained on phenology",
                        "This is not a reproduction of the 2026 spatio-temporal ViT paper",
                        "Validation pixels share the training regions"],
    }
    features._mmap.close()
    output = artifact_directory(root) / f"image_{model_name}_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {model_name} head: {json.dumps(result['selected'])}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("benchmark", "validation"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.phase == "benchmark":
        benchmark(args.model, args.root.resolve(), batch_size=args.batch_size, threads=args.threads)
    else:
        fit_validation(args.model, args.root.resolve(), batch_size=args.batch_size, threads=args.threads)


if __name__ == "__main__":
    main()
