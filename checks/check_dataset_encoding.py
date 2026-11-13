"""Check strict NPZ-encoding tolerance using disposable copies, without fitting."""

from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.check_spatial_results import check_hashes as check_spatial, require
from checks.check_temporal_context import check_hashes as check_temporal


ROOT = Path(__file__).resolve().parents[1]


def run_checks(root=ROOT):
    root = Path(root).resolve()
    source = root / "results/pixel_dataset.npz"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    frozen_dataset_hash = json.loads((root / "results/model_comparison.json").read_text(
        encoding="utf-8"))["dataset_sha256"]
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    temporary_parent = root / "tmp"
    temporary_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset-encoding-check-", dir=temporary_parent) as folder:
        project = Path(folder).resolve()
        # Verify the cleanup target before the context manager removes its tree.
        require(project.parent == temporary_parent.resolve() and project.is_relative_to(root),
                "Temporary test directory is outside the intended project location.")
        results = project / "results"
        results.mkdir()
        for name in ("model_comparison.json", "model_comparison_selection.json"):
            shutil.copyfile(root / "results" / name, results / name)
        dataset = results / "pixel_dataset.npz"
        expected = {"results/pixel_dataset.npz": frozen_dataset_hash}

        def check_both():
            check_spatial(project, expected, set(expected))
            check_temporal(project, expected, set(expected))

        def repack(compression, *, change_feature=False):
            with zipfile.ZipFile(dataset, "w", compression=compression,
                                 compresslevel=1 if compression == zipfile.ZIP_DEFLATED else None) as archive:
                for name in reversed(list(members)):
                    payload = members[name]
                    if change_feature and name == "X_trainval.npy":
                        modified = bytearray(payload)
                        modified[-4] ^= 1  # One mantissa bit of the last float32 feature.
                        payload = bytes(modified)
                        require(sum(a != b for a, b in zip(payload, members[name])) == 1,
                                "The negative case must change exactly one feature byte.")
                    info = zipfile.ZipInfo(name, date_time=(2026, 1, 2, 3, 4, 6))
                    info.compress_type = compression
                    archive.writestr(info, payload)
            require(hashlib.sha256(dataset.read_bytes()).hexdigest() != frozen_dataset_hash,
                    "The test did not change the NPZ container encoding.")

        for compression in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
            repack(compression)
            with zipfile.ZipFile(dataset) as archive:
                require({name: archive.read(name) for name in archive.namelist()} == members,
                        "A positive case changed uncompressed NPY bytes.")
            check_both()

        def reject_both():
            for checker in (check_spatial, check_temporal):
                try:
                    checker(project, expected, set(expected))
                except AssertionError:
                    continue
                raise AssertionError("A changed payload or broken freeze record was accepted.")

        repack(zipfile.ZIP_STORED, change_feature=True)
        with zipfile.ZipFile(dataset) as archive:
            for name, original in members.items():
                if name != "X_trainval.npy":
                    require(archive.read(name) == original, "Labels, identities or other members changed.")
        reject_both()

        repack(zipfile.ZIP_DEFLATED)
        comparison_path = results / "model_comparison.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison["selection_lock_sha256"] = "0" * 64
        comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
        reject_both()
    require(hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, "Original dataset changed.")
    print("PASS: compressed/stored NPZ copies with changed timestamps/order retain exact NPY payloads and verify.")
    print("PASS: a one-byte feature change is rejected with all labels/identities unchanged; broken freeze hashes are rejected.")
    print("PASS: both verifiers tested; disposable copies removed; original dataset unchanged. No models fitted.")


if __name__ == "__main__":
    run_checks()
