"""Plot saved temporal and matched spatial results with core dependencies only.

Run after training, selection, and test reports have been saved:
    python -B experiments/plot_temporal_context.py

This script does not import torch, fit models, or calculate new predictions.
It writes only results/temporal_context_comparison.png.
"""

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "temporal_cnn_context_25"
COLORS = ("#607889", "#CB772E", "#247EAB")


def _finite(values, description, proportions=False):
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array).all() and (array >= 0).all()
    if proportions:
        valid = valid and (array <= 1).all()
    if not valid:
        raise ValueError(f"Invalid saved {description}.")
    return array


def _style(axis, percent=False, horizontal=False):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x" if horizontal else "y", alpha=.18)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=9)
    if percent:
        if horizontal:
            axis.set_xlim(0, 1)
            axis.set_xticks(np.linspace(0, 1, 6))
            axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        else:
            axis.set_ylim(0, 1)
            axis.set_yticks(np.linspace(0, 1, 6))
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def _label_bars(axis, bars, horizontal=False):
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        inside = value > .90
        offset = -.012 if inside else .012
        if horizontal:
            axis.text(value + offset, bar.get_y() + bar.get_height() / 2,
                      f"{value:.1%}", ha="right" if inside else "left", va="center",
                      fontsize=7.5, color="white" if inside else "#222222")
        else:
            axis.text(bar.get_x() + bar.get_width() / 2, value + offset,
                      f"{value:.2%}", ha="center", va="top" if inside else "bottom",
                      fontsize=8, color="white" if inside else "#222222")


def _figure_from_reports(validation, temporal, spatial):
    if any(report["schema_version"] != 1 for report in (validation, temporal, spatial)):
        raise ValueError("Unsupported experiment report schema.")
    species = temporal["species"]
    if len(species) != 6 or species != validation["species"] or species != spatial["species"]:
        raise ValueError("All reports must describe the same six species.")
    samples = temporal["samples"]
    for partition in ("train", "validation", "test"):
        if samples[partition] != spatial["samples"][partition] or samples[partition] != validation["geometry"][partition]:
            raise ValueError(f"The reports disagree about {partition} sample counts.")
    selected_epoch = temporal["selected_epoch"]
    if selected_epoch != validation["selected_epoch"]:
        raise ValueError("The test and validation reports disagree about the selected epoch.")
    history = validation["history"]
    epochs = np.asarray([row["epoch"] for row in history])
    if not np.array_equal(epochs, np.arange(1, 21)):
        raise ValueError("Expected the fixed 20-epoch training history.")
    loss = _finite([row["train_loss"] for row in history], "training loss")
    scores = _finite([row["validation"]["selection_score"] for row in history], "validation scores", True)
    if selected_epoch != int(epochs[np.argmax(scores)]):
        raise ValueError("Selected epoch must be the earliest highest validation score.")
    spatial_rows = {row["name"]: row for row in spatial["rows"]}
    temporal_rows = {row["name"]: row for row in temporal["rows"]}
    context = spatial["selected_context"]
    if set(spatial_rows) != {"raw_rgb", context} or set(temporal_rows) != {MODEL_NAME}:
        raise ValueError("Expected the matched raw/spatial GLMs and one temporal CNN.")
    window = int(context.rsplit("_", 1)[1])
    labels = ("Raw RGB GLM", f"Spatial GLM ({window} × {window})", "Temporal CNN (25 × 25)")
    rows = (spatial_rows["raw_rgb"], spatial_rows[context], temporal_rows[MODEL_NAME])
    for row in rows:
        support = np.asarray(row["metrics"]["support"])
        if support.shape != (6,) or np.any(support <= 0) or support.sum() != samples["test"]:
            raise ValueError("Each saved model needs all six test species.")

    figure = plt.figure(figsize=(14, 10.5), facecolor="white")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.45), hspace=.48, wspace=.34)
    figure.subplots_adjust(left=.085, right=.97, top=.84, bottom=.18)
    loss_axis = figure.add_subplot(grid[0, 0])
    validation_axis = figure.add_subplot(grid[0, 1])
    overall_axis = figure.add_subplot(grid[1, 0])
    recall_axis = figure.add_subplot(grid[1, 1])
    figure.suptitle("Temporal CNN and matched spatial baselines", fontsize=16, y=.97)
    figure.text(.5, .936, "Saved results from the same buffered training pixels and separate test regions",
                ha="center", fontsize=10, color="#444444")

    loss_axis.plot(epochs, loss, color=COLORS[2], linewidth=2, marker="o", markersize=3)
    loss_axis.axvline(selected_epoch, color="#555555", linestyle=":", linewidth=1)
    loss_axis.set(title="Training: weighted cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    loss_axis.set_xlim(.5, 20.5)
    loss_axis.set_ylim(bottom=0)
    loss_axis.set_xticks([1, 5, 10, 15, 20])
    _style(loss_axis)

    validation_axis.plot(epochs, scores, color=COLORS[2], linewidth=2, marker="o", markersize=3)
    best_score = scores[selected_epoch - 1]
    validation_axis.scatter([selected_epoch], [best_score], color="#222222", marker="*", s=120, zorder=4)
    validation_axis.axvline(selected_epoch, color="#555555", linestyle=":", linewidth=1)
    validation_axis.text(.98, .97, f"Selected epoch {selected_epoch}\n{best_score:.2%}",
                         transform=validation_axis.transAxes, ha="right", va="top", fontsize=9,
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    validation_axis.set(title="Validation: checkpoint selection", xlabel="Epoch",
                        ylabel="Four-species mean ROI accuracy", xlim=(.5, 20.5))
    validation_axis.set_xticks([1, 5, 10, 15, 20])
    _style(validation_axis, percent=True)

    metric_positions = np.arange(2)
    legend_handles = []
    for row, color, offset in zip(rows, COLORS, (-.25, 0, .25)):
        values = _finite([row["metrics"]["accuracy"], row["metrics"]["balanced_accuracy_present_classes"]],
                         "test metrics", True)
        bars = overall_axis.bar(metric_positions + offset, values, width=.22, color=color)
        legend_handles.append(bars[0])
        _label_bars(overall_axis, bars)
    overall_axis.set_xticks(metric_positions, ["Accuracy", "Balanced accuracy\n(all six species)"])
    overall_axis.set_title("Test: overall performance", fontsize=11, pad=12)
    _style(overall_axis, percent=True)
    figure.legend(legend_handles, labels, loc="upper center", bbox_to_anchor=(.5, .91),
                  ncols=3, frameon=False, fontsize=9)

    species_positions = np.arange(6)
    for row, color, offset in zip(rows, COLORS, (-.25, 0, .25)):
        recalls = _finite(row["metrics"]["recall"], "species recalls", True)
        if recalls.shape != (6,):
            raise ValueError("Each model needs six saved recall values.")
        bars = recall_axis.barh(species_positions + offset, recalls, height=.21, color=color)
        _label_bars(recall_axis, bars, horizontal=True)
    recall_axis.set_yticks(species_positions, species)
    recall_axis.invert_yaxis()
    recall_axis.set_title("Test: recall by species", fontsize=11, pad=12)
    recall_axis.set_xlabel("Recall", fontsize=9)
    _style(recall_axis, percent=True, horizontal=True)

    figure.text(.025, .111,
                f"Train: {samples['train']:,} pixels   |   Validation: {samples['validation']:,} pixels"
                f"   |   Test: {samples['test']:,} pixels   |   CNN: one seed, 20 epochs, checkpoint {selected_epoch}",
                fontsize=9, color="#333333")
    figure.text(.025, .083,
                "Validation averages ROI accuracy within each of four species, then equally across species; test balanced accuracy uses six species.",
                fontsize=8.5, color="#444444")
    figure.text(.025, .057,
                "M.guianensis and P.ramiflora have no unused validation ROI. Exploratory follow-up: these test regions were evaluated previously.",
                fontsize=8.5, color="#444444")
    figure.text(.025, .031,
                f"Measured CNN stages, including feature extraction: training/selection {validation['total_seconds']:.1f} s;"
                f" test {temporal['total_seconds']:.1f} s. This temporal CNN is not a paper ViT reproduction.",
                fontsize=8.5, color="#444444")
    return figure


def plot_temporal_context(project_root=ROOT, save=True):
    """Return the report figure and optionally save it; no model code is imported."""
    root = Path(project_root).resolve()
    reports = [json.loads((root / "results" / filename).read_text(encoding="utf-8")) for filename in (
        "temporal_context_validation.json", "temporal_context_results.json", "spatial_context_results.json",
    )]
    figure = _figure_from_reports(*reports)
    if save:
        figure.savefig(root / "results/temporal_context_comparison.png", dpi=160, facecolor="white")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    figure = plot_temporal_context(arguments.project_root)
    plt.close(figure)
    print(f"Saved temporal comparison: {arguments.project_root.resolve() / 'results/temporal_context_comparison.png'}")


if __name__ == "__main__":
    main()
