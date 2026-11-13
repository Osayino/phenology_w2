"""Check exact K-NN correctness; add --benchmark for optional full-width timings.

Run from the project root with:
    ./.venv/Scripts/python.exe checks/check_knn.py
"""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "casework"))
from knn_utils import exact_neighbors, uniform_predictions


def check_against_sklearn():
    rng = np.random.default_rng(20261111)
    train = rng.integers(0, 256, (47, 111), dtype=np.uint8)
    query = rng.integers(0, 256, (19, 111), dtype=np.uint8)
    labels = np.arange(len(train)) % 6
    distances, indices = exact_neighbors(train, query, 15, query_block_size=5, train_block_size=9)
    with threadpool_limits(limits=2):
        model = KNeighborsClassifier(n_neighbors=15, algorithm="brute", metric="euclidean")
        model.fit(train.astype(np.float64), labels)
        expected_d, expected_i = model.kneighbors(query.astype(np.float64))
        np.testing.assert_array_equal(indices, expected_i)
        np.testing.assert_allclose(distances, expected_d, rtol=0, atol=1e-12)
        for k in (1, 3, 5, 7, 9, 11, 15):
            model.set_params(n_neighbors=k)
            np.testing.assert_array_equal(uniform_predictions(indices, labels, k), model.predict(query))

    # Deliberate distance ties that cross reference blocks, using nonconsecutive
    # source indices. A direct full calculation provides an unambiguous oracle.
    tied = np.array([[255, 0], [0, 255], [255, 0], [0, 255], [0, 0], [255, 255]], dtype=np.uint8)
    queries = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    selected = np.array([3, 2, 0, 1])
    d, idx = exact_neighbors(tied, queries, 4, train_indices=selected, train_block_size=2)
    full = euclidean_distances(queries.astype(np.float64), tied[selected].astype(np.float64))
    order = np.lexsort((np.broadcast_to(selected, full.shape), full), axis=1)
    np.testing.assert_array_equal(idx, selected[order])
    np.testing.assert_allclose(d, np.take_along_axis(full, order, axis=1), rtol=0, atol=0)
    np.testing.assert_array_equal(uniform_predictions(idx, np.array([2, 1, 1, 2, 0, 0]), 4), [1, 1])

    # A label-vote tie has a specified outcome in sklearn as well.
    vote_model = KNeighborsClassifier(n_neighbors=4, algorithm="brute").fit(tied[selected].astype(float), [2, 1, 2, 1])
    with threadpool_limits(limits=2):
        np.testing.assert_array_equal(vote_model.predict(queries.astype(float)), [1, 1])

    # Raw float32 chromatic features also use source-row selections, including
    # a deliberately reordered query set. Compare in float64 arithmetic.
    raw = rng.random((64, 111), dtype=np.float32)
    train_rows = np.arange(0, 47, 2)
    query_rows = np.array([63, 48, 52, 61])
    d, idx = exact_neighbors(raw, raw, 5, train_indices=train_rows, query_indices=query_rows, query_block_size=2, train_block_size=7)
    raw_model = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(raw[train_rows].astype(np.float64), np.arange(64)[train_rows] % 6)
    with threadpool_limits(limits=2):
        expected_d, expected_i = raw_model.kneighbors(raw[query_rows].astype(np.float64))
        np.testing.assert_array_equal(idx, train_rows[expected_i])
        np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(uniform_predictions(idx, np.arange(64) % 6, 5), raw_model.predict(raw[query_rows].astype(np.float64)))
    print("PASS: exact distances, candidate-K predictions, raw float32/source-row selections, large uint8 values, cross-block distance ties, and class-vote ties", flush=True)


def check_symmetric_recurrence():
    """Check the exact reduction independently of the optional large benchmark."""
    rng = np.random.default_rng(20261111)
    side = 9
    row, col = np.triu_indices(side, 1)
    images = np.zeros((31, side, side, 3), dtype=np.uint8)
    values = rng.integers(0, 256, (len(images), len(row), 3), dtype=np.uint8)
    images[:, row, col] = values
    images[:, col, row] = values
    images[18] = images[2]  # Equal-distance candidates across reference blocks.
    features = images.reshape(len(images), -1)
    train_rows = np.array([18, 0, 2, 7, 6, 5, 4, 3, 8, 9, 11, 10, 13, 12, 14, 16, 15])
    query_rows = np.array([29, 23, 27, 30, 22])
    results = []
    for symmetric in (False, True):
        results.append(exact_neighbors(
            features, features, 7,
            train_indices=train_rows, query_indices=query_rows,
            query_block_size=3, train_block_size=4,
            symmetric_rgb=symmetric,
        ))
    full_distances, full_indices = results[0]
    reduced_distances, reduced_indices = results[1]
    np.testing.assert_array_equal(reduced_distances, full_distances)
    np.testing.assert_array_equal(reduced_indices, full_indices)

    # Independent integer subtraction avoids the helper's dot-product identity.
    difference = features[query_rows].astype(np.int64)[:, None, :] - features[train_rows].astype(np.int64)[None, :, :]
    squared = np.sum(difference * difference, axis=2)
    order = np.lexsort((np.broadcast_to(train_rows, squared.shape), squared), axis=1)[:, :7]
    np.testing.assert_array_equal(full_indices, train_rows[order])
    np.testing.assert_array_equal(full_distances, np.sqrt(np.take_along_axis(squared, order, axis=1)))

    for corruption in ('asymmetry', 'nonzero diagonal'):
        broken = images[:1].copy()
        if corruption == 'asymmetry':
            broken[0, 0, 1, 0] ^= 1
        else:
            broken[0, 0, 0, 0] = 1
        try:
            exact_neighbors(broken.reshape(1, -1), features[:1], 1, symmetric_rgb=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Symmetric reduction accepted {corruption}')
    print('PASS: symmetric/full distances exactly equal to direct integer oracle; partial blocks and duplicate-image ties; asymmetric and nonzero-diagonal inputs rejected', flush=True)


def benchmark_recurrence():
    rng = np.random.default_rng(42)
    n_train, n_query, side = 1024, 128, 140
    row, col = np.triu_indices(side, 1)
    images = np.zeros((n_train + n_query, side, side, 3), dtype=np.uint8)
    values = rng.integers(0, 256, (len(images), len(row), 3), dtype=np.uint8)
    images[:, row, col] = values
    images[:, col, row] = values
    del values
    features = images.reshape(len(images), -1)
    reference = None
    for symmetric, threads, query_block, train_block in (
        (False, 2, 64, 128),
        (True, 2, 64, 128),
        (True, 2, 128, 256),
        (True, 4, 128, 256),
        (True, 1, 128, 256),
        (False, 4, 64, 128),
        (False, 1, 64, 128),
    ):
        started = perf_counter()
        d, idx = exact_neighbors(features[:n_train], features[n_train:], 15, n_threads=threads, symmetric_rgb=symmetric, query_block_size=query_block, train_block_size=train_block)
        elapsed = perf_counter() - started
        if reference is None:
            reference = d, idx
        else:
            np.testing.assert_array_equal(idx, reference[1])
            np.testing.assert_array_equal(d, reference[0])
        scale = 10996 * (2749 + 14105) / (n_train * n_query)
        print(f"BENCHMARK symmetry={symmetric}, threads={threads}, blocks={query_block}/{train_block}: {elapsed:.3f} s; linear extrapolation for full validation+test={elapsed * scale / 60:.1f} min (estimate, not a guarantee)", flush=True)
    broken = images[:1].copy()
    broken[0, 0, 1, 0] ^= 1
    try:
        exact_neighbors(broken.reshape(1, -1), features[:1], 1, symmetric_rgb=True)
    except ValueError:
        print("PASS: asymmetric inputs rejected; reduced and full-feature results exactly identical", flush=True)
    else:
        raise AssertionError("Asymmetric input accepted")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true', help='Also time representative 140 x 140 RGB inputs.')
    args = parser.parse_args()
    check_against_sklearn()
    check_symmetric_recurrence()
    if args.benchmark:
        benchmark_recurrence()
