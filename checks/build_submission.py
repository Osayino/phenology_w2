"""Build the small hand-in ZIP from an explicit set of project files.

Run after verifying the notebook/results. Regenerable caches, environments,
downloaded weights, fitted models and the original reference notebook stay local.
"""
from pathlib import Path
import hashlib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def submission_files(root):
    paths = [root / name for name in (
        "README.md", "START-HERE.txt", "requirements.txt", "requirements-lock.txt",
        "requirements-experiments.txt", "requirements-experiments-lock.txt",
        "casework/case_notebook.ipynb", "casework/phenology_data.py", "casework/knn_utils.py",
        "experiments/protocol.json", "experiments/spatial_context_protocol.json",
        "experiments/temporal_context_protocol.json",
    )]
    paths.extend(sorted((root / "checks").glob("*.py")))
    paths.extend(sorted((root / "experiments").glob("*.py")))
    paths.extend(root / "raw/images/1" / f"2011_{day}_12.jpg" for day in range(242, 279))
    masks = sorted((root / "raw/masks").glob("*.pgm"))
    assert len(masks) == 22, "Expected the 22 original ROI masks"
    paths.extend(masks)
    paths.extend(root / "results" / name for name in (
        "comparison.png", "confusion_matrices.png", "experiment_results.json",
        "extension_comparison.png", "extension_predictions.npz", "extension_results.json",
        "pixel_dataset.npz", "pixel_examples.png", "predictions.npz", "verification.json",
        "model_comparison.json", "model_comparison.png", "model_comparison_predictions.npz",
        "model_comparison_selection.json", "model_comparison_validation.json",
        "spatial_context_comparison.png", "spatial_context_predictions.npz",
        "spatial_context_results.json", "spatial_context_selection.json",
        "spatial_context_validation.json", "spatial_context_validation_predictions.npz",
        "temporal_context_predictions.npz", "temporal_context_results.json",
        "temporal_context_selection.json", "temporal_context_validation.json",
        "temporal_context_validation_predictions.npz", "temporal_context_comparison.png",
        "final_submission_verification.json",
    ))
    assert len(paths) == len(set(paths)) and all(path.is_file() for path in paths)
    return paths


def build_submission(root=ROOT):
    root = Path(root).resolve()
    files = submission_files(root)
    folder = root / "dist"
    folder.mkdir(exist_ok=True)
    destination = folder / "phenology_submission.zip"
    staging = folder / "phenology_submission.zip.tmp"
    with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in files:
            archive.write(source, "phenology_submission/" + source.relative_to(root).as_posix())
    with zipfile.ZipFile(staging) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)
        for source in files:
            name = "phenology_submission/" + source.relative_to(root).as_posix()
            assert hashlib.sha256(archive.read(name)).digest() == hashlib.sha256(source.read_bytes()).digest()
    staging.replace(destination)
    print(f"Verified {len(files)} files: {destination} ({destination.stat().st_size / 2**20:.2f} MiB)")
    return destination


if __name__ == "__main__":
    build_submission()
