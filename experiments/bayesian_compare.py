"""Validation-only comparison of two small Bayesian classifiers.

TAN is a Bayesian network with a class parent for every feature and a learned
tree of feature-to-feature edges. It is not a Bayesian neural network.
Reference: Friedman, Geiger and Goldszmidt (1997), Bayesian Network Classifiers,
https://ai.stanford.edu/~nir/Abstracts/FrGG1.html

Run from the project root:
    .venv/Scripts/python.exe -B experiments/bayesian_compare.py
Only the training/validation arrays are read; test evaluation belongs to the
later, fixed comparison stage. No existing notebook or result is overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.naive_bayes import GaussianNB
from threadpoolctl import threadpool_limits


FEATURE_NAMES = [f"{channel}_{statistic}" for channel in ("red", "green", "blue")
                 for statistic in ("mean", "std", "linear_trend")]


def summarize_rgbcc(X):
    """Nine fixed summaries of the 37 red, then green, then blue observations."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 111 or not np.isfinite(X).all():
        raise ValueError("Expected finite arrays of 111 channel-major RGBcc values.")
    series = X.reshape(-1, 3, 37)
    time = np.linspace(0, 1, 37)
    centered_time = time - time.mean()
    # The slope describes the change across the observation window, not per day.
    slopes = (series * centered_time).sum(axis=2) / (centered_time @ centered_time)
    return np.stack((series.mean(axis=2), series.std(axis=2), slopes), axis=2).reshape(-1, 9)


class TreeAugmentedNB:
    """Discrete TAN using training quantiles and symmetric Dirichlet smoothing.

    Conditional mutual information determines a maximum spanning tree. Feature
    zero is its root; every other feature has one feature parent. The class is
    an additional parent of every feature. All conditional tables are learned
    from training rows only, with alpha pseudo-counts per child state.
    """

    def __init__(self, n_bins=3, alpha=1.0):
        if n_bins < 2 or alpha <= 0:
            raise ValueError("Use at least two bins and positive smoothing.")
        self.n_bins = int(n_bins)
        self.alpha = float(alpha)

    def _discretize(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.edges_) or not np.isfinite(X).all():
            raise ValueError("Features must be finite and match the fitted model.")
        return np.column_stack([np.searchsorted(edges, X[:, j], side="right")
                                for j, edges in enumerate(self.edges_)])

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0 or y.shape != (len(X),):
            raise ValueError("Supply a nonempty feature matrix and aligned labels.")
        if not np.isfinite(X).all():
            raise ValueError("Features must be finite.")
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        # All bin boundaries come from training data, including for validation.
        quantiles = np.arange(1, self.n_bins) / self.n_bins
        self.edges_ = [np.unique(np.quantile(X[:, j], quantiles))
                       if np.ptp(X[:, j]) > 0 else np.array([])
                       for j in range(X.shape[1])]
        discrete = self._discretize(X)
        self.cardinalities_ = np.array([len(edges) + 1 for edges in self.edges_])
        n_classes, n_features = len(self.classes_), X.shape[1]
        counts = np.bincount(encoded_y, minlength=n_classes)
        self.prior_ = (counts + self.alpha) / (len(X) + self.alpha * n_classes)

        # Edge weights measure association after conditioning on species.
        weights = np.zeros((n_features, n_features))
        for left in range(n_features):
            for right in range(left + 1, n_features):
                information = 0.0
                for cls, count in enumerate(counts):
                    rows = discrete[encoded_y == cls]
                    table = np.zeros((self.cardinalities_[left], self.cardinalities_[right]))
                    np.add.at(table, (rows[:, left], rows[:, right]), 1)
                    probability = table / count
                    independent = probability.sum(axis=1)[:, None] * probability.sum(axis=0)[None, :]
                    nonzero = probability > 0
                    information += (count / len(X)) * np.sum(
                        probability[nonzero] * np.log(probability[nonzero] / independent[nonzero])
                    )
                weights[left, right] = weights[right, left] = max(0.0, information)
        self.conditional_information_ = weights

        # Prim's maximum-spanning-tree algorithm also works when every weight is zero.
        self.parents_ = np.full(n_features, -1, dtype=int)
        reached = {0}
        while len(reached) < n_features:
            parent, child = max(
                ((p, c) for p in sorted(reached) for c in range(n_features) if c not in reached),
                key=lambda edge: weights[edge[0], edge[1]],
            )
            self.parents_[child] = parent
            reached.add(child)

        self.cpds_ = []
        for feature, parent in enumerate(self.parents_):
            parent_states = 1 if parent == -1 else self.cardinalities_[parent]
            table = np.full((n_classes, parent_states, self.cardinalities_[feature]), self.alpha)
            parent_values = np.zeros(len(X), dtype=int) if parent == -1 else discrete[:, parent]
            np.add.at(table, (encoded_y, parent_values, discrete[:, feature]), 1)
            table /= table.sum(axis=2, keepdims=True)
            self.cpds_.append(table)
        return self

    def _joint_log_probability(self, discrete):
        scores = np.tile(np.log(self.prior_), (len(discrete), 1))
        for feature, parent in enumerate(self.parents_):
            parent_values = np.zeros(len(discrete), dtype=int) if parent == -1 else discrete[:, parent]
            for cls in range(len(self.classes_)):
                scores[:, cls] += np.log(self.cpds_[feature][cls, parent_values, discrete[:, feature]])
        return scores

    def predict_proba(self, X):
        joint = self._joint_log_probability(self._discretize(X))
        return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def to_dict(self):
        return {"kind": "tree_augmented_naive_bayes", "n_bins": self.n_bins, "alpha": self.alpha,
                "classes": self.classes_.tolist(), "quantile_edges": [e.tolist() for e in self.edges_],
                "feature_parents": self.parents_.tolist(), "class_prior": self.prior_.tolist(),
                "conditional_mutual_information": self.conditional_information_.tolist(),
                "conditional_probability_tables": [c.tolist() for c in self.cpds_]}

    @classmethod
    def from_dict(cls, data):
        model = cls(data["n_bins"], data["alpha"])
        model.classes_ = np.asarray(data["classes"])
        model.edges_ = [np.asarray(e, dtype=float) for e in data["quantile_edges"]]
        model.cardinalities_ = np.array([len(e) + 1 for e in model.edges_])
        model.parents_ = np.asarray(data["feature_parents"], dtype=int)
        model.prior_ = np.asarray(data["class_prior"], dtype=float)
        model.conditional_information_ = np.asarray(data["conditional_mutual_information"], dtype=float)
        model.cpds_ = [np.asarray(c, dtype=float) for c in data["conditional_probability_tables"]]
        return model


def check_tan_oracle():
    """An XOR example needs feature dependence; also enumerate the entire joint."""
    time = np.linspace(0, 1, 37)
    known_rgbcc = np.r_[time, 1 - time, np.zeros(37)][None, :]
    expected = [[.5, time.std(), 1, .5, time.std(), -1, 0, 0, 0]]
    np.testing.assert_allclose(summarize_rgbcc(known_rgbcc), expected, atol=1e-14)
    X = np.repeat(np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]]), 2, axis=0)
    y = np.repeat([0, 0, 1, 1], 2)
    model = TreeAugmentedNB(n_bins=2, alpha=1).fit(X, y)
    # Hand calculation: the matching child state has (2+1)/(2+2)=3/4 probability.
    np.testing.assert_allclose(model.predict_proba([[0., 0.], [0., 1.]]), [[.75, .25], [.25, .75]])
    states = np.array(list(itertools.product(range(2), repeat=2)))
    np.testing.assert_allclose(np.exp(model._joint_log_probability(states)).sum(), 1.0)
    np.testing.assert_allclose(model.conditional_information_[0, 1], np.log(2))
    restored = TreeAugmentedNB.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    constant = TreeAugmentedNB(3).fit(np.column_stack((X, np.ones(len(X)))), y)
    assert constant.cardinalities_[-1] == 1
    for table in constant.cpds_:
        np.testing.assert_allclose(table.sum(axis=2), 1.0)


def predict_artifact(path, X_rgbcc, *, probabilities=False):
    """Use a saved candidate on 111-feature arrays; this never reads dataset labels."""
    path = Path(path)
    if path.suffix == ".json":
        model = TreeAugmentedNB.from_dict(json.loads(path.read_text(encoding="utf-8"))["model"])
    else:
        model = joblib.load(path)["model"]
    summary = summarize_rgbcc(X_rgbcc)
    return model.predict_proba(summary) if probabilities else model.predict(summary)


def fit_validation(dataset_path, output_dir):
    """Fit the declared shortlist using only training and validation data."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    check_tan_oracle()
    # np.load is lazy: these are the only dataset members accessed here.
    with np.load(dataset_path, allow_pickle=False) as data:
        X = data["X_trainval"]
        y = data["y_trainval"]
        train_indices, val_indices = data["train_indices"], data["val_indices"]
    assert len(train_indices) == 10996 and len(val_indices) == 2749
    assert not np.intersect1d(train_indices, val_indices).size
    summary = summarize_rgbcc(X)
    candidates = [
        ("Gaussian naive Bayes (9 summaries)", GaussianNB(), "bayesian_gaussian_summary.joblib"),
        ("TAN (2 quantile bins)", TreeAugmentedNB(2), "bayesian_tan_2.json"),
        ("TAN (3 quantile bins)", TreeAugmentedNB(3), "bayesian_tan_3.json"),
    ]
    records = []
    with threadpool_limits(limits=1):
        for name, model, filename in candidates:
            started = perf_counter()
            model.fit(summary[train_indices], y[train_indices])
            prediction = model.predict(summary[val_indices])
            seconds = perf_counter() - started
            payload = {"feature_names": FEATURE_NAMES, "feature_transform": "summarize_rgbcc",
                       "model": model.to_dict() if isinstance(model, TreeAugmentedNB) else model}
            target = output_dir / filename
            if target.suffix == ".json":
                target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            else:
                joblib.dump(payload, target)
            np.testing.assert_array_equal(predict_artifact(target, X[val_indices]), prediction)
            record = {"name": name, "model_file": filename,
                      "validation_accuracy": float(accuracy_score(y[val_indices], prediction)),
                      "validation_macro_f1": float(f1_score(y[val_indices], prediction, average="macro", zero_division=0)),
                      "validation_balanced_accuracy": float(recall_score(y[val_indices], prediction, average="macro", zero_division=0)),
                      "fit_and_validation_seconds": seconds}
            if isinstance(model, TreeAugmentedNB):
                record["parameters"] = {"n_bins": model.n_bins, "alpha": model.alpha,
                                        "class_prior": "smoothed empirical training proportions",
                                        "feature_tree_root": FEATURE_NAMES[0]}
                record["feature_edges"] = [[FEATURE_NAMES[p], FEATURE_NAMES[c]]
                                           for c, p in enumerate(model.parents_) if p != -1]
            else:
                record["parameters"] = model.get_params()
            records.append(record)
            print(json.dumps(record), flush=True)
    winner = max(records, key=lambda record: record["validation_accuracy"])
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "protocol": "Same saved train/validation rows; summaries predefined; quantiles/tree/CPDs fitted on training only; no test arrays accessed",
        "feature_names": FEATURE_NAMES, "samples": {"train": len(train_indices), "validation": len(val_indices)},
        "candidates": records, "selected_model": winner["name"], "selected_model_file": winner["model_file"],
        "selection_rule": "highest validation accuracy; first declared candidate breaks ties",
        "oracle_check": "passed: hand-calculated XOR posterior, exhaustive joint normalization and serialization",
        "limitations": ["These nine summaries discard detailed daily timing",
                        "Quantile discretization loses continuous information",
                        "Validation shares training ROIs",
                        "A TAN graph describes statistical dependence, not proven causal effects",
                        "This is not a Bayesian neural network"],
        "reference": "https://ai.stanford.edu/~nir/Abstracts/FrGG1.html",
    }
    (output_dir / "bayesian_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "results/pixel_dataset.npz")
    parser.add_argument("--output-dir", type=Path, default=root / "experiments/artifacts")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_tan_oracle()
        print("PASS: Bayesian-network XOR oracle, joint normalization, constant feature and save/load checks.")
    else:
        result = fit_validation(args.dataset, args.output_dir)
        print("Selected on validation only:", result["selected_model"])


if __name__ == "__main__":
    main()
