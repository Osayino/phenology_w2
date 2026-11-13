"""Plot saved spatial-context reports without fitting or evaluating a model.

Run after both reports exist:
    python -B experiments/plot_spatial_context.py

The script reads spatial_context_results.json and spatial_context_validation.json
and writes only results/spatial_context_comparison.png. Importing it creates no
figure or output file.
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
RAW_COLOR = "#346D91"
CONTEXT_COLOR = "#CC702C"
OTHER_CONTEXT_COLOR = "#E7B991"


def _proportions(values, description):
    values = np.asarray(values, dtype=float)
    if not (np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()):
        raise ValueError(f"{description} must be finite proportions between zero and one.")
    return values


def _style_axis(axis, horizontal=False):
    score_axis = axis.xaxis if horizontal else axis.yaxis
    score_axis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if horizontal:
        axis.set_xlim(0, 1)
        axis.set_xticks(np.linspace(0, 1, 6))
    else:
        axis.set_ylim(0, 1)
        axis.set_yticks(np.linspace(0, 1, 6))
    axis.grid(axis="x" if horizontal else "y", alpha=.18)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)


def _bar_values(axis, bars, horizontal=False):
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        inside = value > .90
        offset = -.012 if inside else .012
        if horizontal:
            axis.text(value + offset, bar.get_y() + bar.get_height() / 2,
                      f"{value:.2%}", ha="right" if inside else "left", va="center",
                      fontsize=8, color="white" if inside else "#262626")
        else:
            axis.text(bar.get_x() + bar.get_width() / 2, value + offset,
                      f"{value:.2%}", ha="center", va="top" if inside else "bottom",
                      fontsize=8, color="white" if inside else "#262626")


def _figure_from_reports(results, validation):
    if results["schema_version"] != 1 or validation["schema_version"] != 1:
        raise ValueError("Unsupported spatial-context report schema.")
    selected = results["selected_context"]
    if selected not in ("context_5", "context_25") or selected != validation["selected_context"]:
        raise ValueError("Test and validation reports must name the same selected context.")
    species = results["species"]
    if len(species) != 6 or species != validation["species"]:
        raise ValueError("The reports must describe the same six species.")
    samples = results["samples"]
    for partition in ("train", "validation", "test"):
        if samples[partition] != validation["geometry"][partition]:
            raise ValueError(f"The reports disagree about the {partition} sample count.")
    test_rows = {row["name"]: row for row in results["rows"]}
    validation_rows = {row["name"]: row for row in validation["rows"]}
    if len(results["rows"]) != 2 or set(test_rows) != {"raw_rgb", selected}:
        raise ValueError("Expected the buffered raw baseline and selected context test rows.")
    if len(validation["rows"]) != 3 or set(validation_rows) != {"raw_rgb", "context_5", "context_25"}:
        raise ValueError("Expected raw RGB and both context candidates in validation.")
    validation_counts = np.asarray(validation["geometry"]["validation_counts"])
    if validation_counts.shape != (6,) or np.count_nonzero(validation_counts) != 4:
        raise ValueError("The declared validation score covers four of the six species.")
    unvalidated = [species[index] for index in np.flatnonzero(validation_counts == 0)]
    for row in test_rows.values():
        support = np.asarray(row["metrics"]["support"])
        if support.shape != (6,) or np.any(support <= 0) or support.sum() != samples["test"]:
            raise ValueError("Test metrics must include positive support for all six species.")

    window = int(selected.rsplit("_", 1)[1])
    context_label = f"Selected context ({window} × {window})"
    names = ("raw_rgb", selected)
    colors = (RAW_COLOR, CONTEXT_COLOR)
    labels = ("Buffered raw RGB", context_label)

    figure = plt.figure(figsize=(12.5, 9.5), facecolor="white")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.35), hspace=.50, wspace=.35)
    figure.subplots_adjust(left=.14, right=.97, top=.84, bottom=.18)
    selection_axis = figure.add_subplot(grid[0, 0])
    overall_axis = figure.add_subplot(grid[0, 1])
    recall_axis = figure.add_subplot(grid[1, :])
    figure.suptitle("Spatial context on separate image regions", fontsize=16, x=.54, y=.97)
    figure.text(.54, .93, "Fixed GLM comparison with the same buffered training pixels",
                ha="center", fontsize=10, color="#444444")

    candidate_names = ("raw_rgb", "context_5", "context_25")
    candidate_labels = ["Raw RGB", "5 × 5 context", "25 × 25 context"]
    candidate_labels[candidate_names.index(selected)] += "\n(selected)"
    candidate_scores = _proportions(
        [validation_rows[name]["validation"]["selection_score"] for name in candidate_names],
        "Validation selection scores",
    )
    candidate_colors = [RAW_COLOR] + [
        CONTEXT_COLOR if name == selected else OTHER_CONTEXT_COLOR for name in candidate_names[1:]
    ]
    bars = selection_axis.bar(np.arange(3), candidate_scores, width=.60, color=candidate_colors)
    _bar_values(selection_axis, bars)
    selection_axis.set_xticks(np.arange(3), candidate_labels)
    selection_axis.set_ylabel("Four-species mean ROI accuracy", fontsize=9)
    selection_axis.set_title("Validation: choosing the context window", fontsize=11, pad=12)
    _style_axis(selection_axis)

    metric_keys = ("accuracy", "balanced_accuracy_present_classes")
    metric_labels = ("Accuracy", "Balanced accuracy\n(all six species)")
    positions = np.arange(2)
    legend_handles = []
    for name, color, offset in zip(names, colors, (-.18, .18)):
        scores = _proportions([test_rows[name]["metrics"][key] for key in metric_keys], "Test scores")
        bars = overall_axis.bar(positions + offset, scores, width=.32, color=color)
        legend_handles.append(bars[0])
        _bar_values(overall_axis, bars)
    overall_axis.set_xticks(positions, metric_labels)
    overall_axis.set_title("Test: overall performance", fontsize=11, pad=12)
    _style_axis(overall_axis)
    figure.legend(legend_handles, labels, loc="upper center", bbox_to_anchor=(.54, .905),
                  ncols=2, frameon=False, fontsize=9)

    positions = np.arange(len(species))
    for name, color, offset in zip(names, colors, (-.18, .18)):
        recalls = _proportions(test_rows[name]["metrics"]["recall"], "Species recalls")
        if recalls.shape != (6,):
            raise ValueError("Expected six test recall values.")
        bars = recall_axis.barh(positions + offset, recalls, height=.32, color=color)
        _bar_values(recall_axis, bars, horizontal=True)
    recall_axis.set_yticks(positions, species)
    recall_axis.invert_yaxis()
    recall_axis.set_xlabel("Recall on the separate test regions", fontsize=9)
    recall_axis.set_title("Test: recall by species", fontsize=11, pad=12)
    _style_axis(recall_axis, horizontal=True)

    figure.text(.025, .105,
                f"Train: {samples['train']:,} pixels   |   Validation: {samples['validation']:,} pixels"
                f"   |   Test: {samples['test']:,} pixels   |   Selected window: {window} × {window} pixels",
                fontsize=9, color="#333333")
    figure.text(.025, .078,
                "Validation averages ROI accuracy within each of four species, then averages those species equally.",
                fontsize=8.5, color="#444444")
    figure.text(.025, .051,
                f"No unused validation ROI for {' and '.join(unvalidated)}. Test balanced accuracy covers all six species.",
                fontsize=8.5, color="#444444")
    figure.text(.025, .024,
                "Exploratory follow-up: these test regions were already evaluated in earlier experiments.",
                fontsize=8.5, color="#444444")
    return figure


def plot_spatial_context(project_root=ROOT, save=True):
    """Return a figure of the saved reports and optionally write its PNG."""
    root = Path(project_root).resolve()
    results = json.loads((root / "results/spatial_context_results.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "results/spatial_context_validation.json").read_text(encoding="utf-8"))
    figure = _figure_from_reports(results, validation)
    if save:
        figure.savefig(root / "results/spatial_context_comparison.png", dpi=160, facecolor="white")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    figure = plot_spatial_context(arguments.project_root)
    plt.close(figure)
    print(f"Saved spatial-context figure: {arguments.project_root.resolve() / 'results/spatial_context_comparison.png'}")


if __name__ == "__main__":
    main()
