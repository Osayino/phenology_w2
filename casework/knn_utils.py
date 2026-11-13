"""Memory-bounded, exact Euclidean nearest neighbours for the phenology notebook.

The recurrence arrays can remain uint8 memory maps on disk. Only small row
blocks become float64 arrays. For uint8 inputs, the products and sums involved
here are exactly representable in float64 at this assignment's dimensions.
No training samples or feature information are discarded.
"""

from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits


def _selected_rows(array, rows, name):
    """Validate an optional selection without copying the feature array."""
    if len(array.shape) != 2:
        raise ValueError(f"{name} must have shape (samples, features).")
    if rows is None:
        return np.arange(len(array), dtype=np.int64)
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(f"{name} row indices must be a one-dimensional integer array.")
    if len(rows) and (rows.min() < 0 or rows.max() >= len(array)):
        raise ValueError(f"{name} row index is out of bounds.")
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"{name} row indices must be unique.")
    return rows.astype(np.int64, copy=False)


def _distance_columns(n_features, symmetric_rgb):
    if not symmetric_rgb:
        return None, None
    side = int(round(np.sqrt(n_features / 3)))
    if side * side * 3 != n_features:
        raise ValueError("Symmetric RGB input must flatten a square, three-channel image.")
    row, col = np.triu_indices(side, k=1)
    columns = ((row * side + col)[:, None] * 3 + np.arange(3)).ravel()
    return columns, side


def _read_block(array, rows, columns, side):
    """Read only requested rows; check the exact symmetry reduction's premise."""
    block = np.asarray(array[rows])
    if columns is not None:
        images = block.reshape(len(rows), side, side, 3)
        if not np.array_equal(images, images.swapaxes(1, 2)):
            raise ValueError("The symmetric RGB reduction requires symmetric channels.")
        if np.any(np.diagonal(images, axis1=1, axis2=2)):
            raise ValueError("The symmetric RGB reduction requires zero diagonals.")
        # np.take keeps C order; advanced column indexing would create a
        # transposed temporary and slow down the following BLAS multiplication.
        block = np.take(block, columns, axis=1)
    block = np.ascontiguousarray(block, dtype=np.float64)
    if not np.issubdtype(array.dtype, np.integer) and not np.isfinite(block).all():
        raise ValueError("Features must be finite.")
    return block


def exact_neighbors(
    X_train,
    X_query,
    n_neighbors=15,
    *,
    train_indices=None,
    query_indices=None,
    query_block_size=64,
    train_block_size=128,
    n_threads=2,
    symmetric_rgb=False,
    progress=False,
):
    """Return exact Euclidean distances and source-row indices of neighbours.

    Parameters
    ----------
    X_train, X_query : 2-D arrays or memory maps
        Feature matrices. They may be the same full dataset, with separate
        ``train_indices`` and ``query_indices`` selecting the two sets.
    train_indices, query_indices : optional integer arrays
        Source rows to use, in the desired order. Returned neighbour indices
        address rows of ``X_train`` itself, NOT positions within train_indices.
    symmetric_rgb : bool
        If true, verify that every image channel is symmetric with a zero
        diagonal, then calculate using each upper-triangle entry once. Each
        off-diagonal squared difference appears twice in the full image, so
        multiplying the squared distances by two recovers FULL-image distances
        exactly. All neighbour rankings and uniform K-NN predictions are
        unchanged. This is a redundancy reduction, not dimensionality learning.
    progress : bool or callable
        True prints progress approximately every ten seconds. A callable is
        called after every query block with (completed, total, elapsed_seconds).

    Notes
    -----
    Rows are ordered by (distance, source row index), making ties deterministic.
    The defaults hold at most 192 feature rows in float64 at a time (about
    86 MiB for full 58,800-feature inputs; about 43 MiB with symmetry).
    The small distance block and source uint8 row copies add modest overhead.
    Float64 arithmetic also supports ordinary floating-point features, subject
    to normal floating-point rounding. The integer recurrence calculation is
    exact at the feature dimensions used by this notebook.
    """
    train_rows = _selected_rows(X_train, train_indices, "X_train")
    query_rows = _selected_rows(X_query, query_indices, "X_query")
    if X_train.shape[1] != X_query.shape[1]:
        raise ValueError("Training and query feature dimensions differ.")
    if not isinstance(n_neighbors, (int, np.integer)) or not 1 <= n_neighbors <= len(train_rows):
        raise ValueError("n_neighbors must be between 1 and the training-set size.")
    if min(query_block_size, train_block_size, n_threads) < 1:
        raise ValueError("Block sizes and n_threads must be positive.")
    columns, side = _distance_columns(X_train.shape[1], symmetric_rgb)
    distances = np.empty((len(query_rows), n_neighbors), dtype=np.float64)
    indices = np.empty((len(query_rows), n_neighbors), dtype=np.int64)
    started = last_report = perf_counter()

    with threadpool_limits(limits=n_threads, user_api="blas"):
        for start in range(0, len(query_rows), query_block_size):
            stop = min(start + query_block_size, len(query_rows))
            query = _read_block(X_query, query_rows[start:stop], columns, side)
            query_norm = np.einsum("ij,ij->i", query, query)
            best_d2 = np.full((len(query), n_neighbors), np.inf)
            best_indices = np.full(best_d2.shape, np.iinfo(np.int64).max, dtype=np.int64)

            for ref_start in range(0, len(train_rows), train_block_size):
                ref_rows = train_rows[ref_start:ref_start + train_block_size]
                reference = _read_block(X_train, ref_rows, columns, side)
                ref_norm = np.einsum("ij,ij->i", reference, reference)
                d2 = query @ reference.T
                # Release this float64 block before allocating the next one.
                del reference
                d2 *= -2
                d2 += query_norm[:, None]
                d2 += ref_norm[None, :]
                np.maximum(d2, 0, out=d2)

                candidate_d2 = np.concatenate((best_d2, d2), axis=1)
                candidate_indices = np.concatenate(
                    (best_indices, np.broadcast_to(ref_rows, d2.shape)), axis=1
                )
                # Sorting these small blocks resolves equal-distance neighbours
                # by original row index, including ties across block boundaries.
                order = np.lexsort((candidate_indices, candidate_d2), axis=1)[:, :n_neighbors]
                best_d2 = np.take_along_axis(candidate_d2, order, axis=1)
                best_indices = np.take_along_axis(candidate_indices, order, axis=1)

            if symmetric_rgb:
                best_d2 *= 2
            distances[start:stop] = np.sqrt(best_d2)
            indices[start:stop] = best_indices
            now = perf_counter()
            if callable(progress):
                progress(stop, len(query_rows), now - started)
            elif progress and (now - last_report >= 10 or stop == len(query_rows)):
                print(f"Neighbours: {stop:,}/{len(query_rows):,} query pixels; {now - started:.1f} s", flush=True)
                last_report = now

    return distances, indices


def uniform_predictions(neighbor_indices, labels, k):
    """Vote among the first k neighbours; class-count ties use the smallest class.

    ``labels`` must be aligned with the complete X_train source array supplied
    to exact_neighbors, because its returned indices address that source.
    """
    neighbor_indices = np.asarray(neighbor_indices)
    labels = np.asarray(labels)
    if neighbor_indices.ndim != 2 or not 1 <= k <= neighbor_indices.shape[1]:
        raise ValueError("k must select an available number of neighbours.")
    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")
    chosen = neighbor_indices[:, :k]
    if chosen.size and (chosen.min() < 0 or chosen.max() >= len(labels)):
        raise ValueError("A neighbour index is outside the label array.")
    classes = np.unique(labels)
    votes = np.stack([(labels[chosen] == cls).sum(axis=1) for cls in classes], axis=1)
    return classes[votes.argmax(axis=1)]
