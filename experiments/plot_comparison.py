"""Plot the verified exploratory comparison without loading or training models.

Importing this module creates no figure or result file. To save the figure:
    .venv/Scripts/python.exe -B experiments/plot_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.verify_comparison import verify_results


def _figure_from_rows(rows):
    """Build the visual from already checked rows; also usable for layout checks."""
    if not rows:
        raise ValueError("The comparison has no model rows to plot.")
    scores = [{key.replace(" ", ""): value for key, value in row["test_metrics"].items()}
              for row in rows]
    accuracy = np.asarray([metrics["Accuracy"] for metrics in scores], dtype=float)
    balanced = np.asarray([metrics["Recall(macro)"] for metrics in scores], dtype=float)
    if not (np.isfinite(accuracy).all() and np.isfinite(balanced).all()
            and ((0 <= accuracy) & (accuracy <= 1)).all()
            and ((0 <= balanced) & (balanced <= 1)).all()):
        raise ValueError("Both scores must be finite proportions between zero and one.")

    # I keep the table's declared order, including the three original baselines.
    labels = [textwrap.fill(row["name"], width=34, break_long_words=False) for row in rows]
    positions = np.arange(len(rows))
    height = max(7.0, .65 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(12, height))
    fig.subplots_adjust(left=.32, right=.97, top=.84, bottom=.13)
    groups = [
        ax.barh(positions - .18, accuracy, height=.32, color="#277DA8",
                label="Accuracy (all test pixels)"),
        ax.barh(positions + .18, balanced, height=.32, color="#D78321",
                label="Balanced accuracy (macro recall)"),
    ]

    # I print percentages so the reader can compare values without estimating bar lengths.
    for bars in groups:
        for bar in bars:
            score = bar.get_width()
            inside = score > .9
            ax.text(score - .012 if inside else score + .012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.2%}", va="center", ha="right" if inside else "left",
                    color="white" if inside else "#222222", fontsize=9)

    ax.set(yticks=positions, yticklabels=labels, xlim=(0, 1), xlabel="Score on the separate test regions")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=10, length=0, pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Exploratory comparison on separate test regions", fontsize=14, pad=47)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncols=2, frameon=False, fontsize=9)
    fig.text(.025, .025,
             "14,105 test pixels from the same separate ROIs.\n"
             "Each model family was configured using validation; this comparison is exploratory.",
             fontsize=9, color="#444444", va="bottom")
    return fig


def plot_comparison(project_root, save=True):
    """Verify the saved comparison, return its figure and optionally save the PNG."""
    root = Path(project_root).resolve()
    verify_results(root)
    comparison = json.loads((root / "results/model_comparison.json").read_text(encoding="utf-8"))
    fig = _figure_from_rows(comparison["rows"])
    if save:
        fig.savefig(root / "results/model_comparison.png", dpi=150, facecolor="white")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    fig = plot_comparison(args.project_root, save=True)
    plt.close(fig)
    print(f"Saved verified comparison figure: {args.project_root / 'results/model_comparison.png'}")


if __name__ == "__main__":
    # Exporting a PNG from a terminal does not need Tk or another GUI toolkit.
    plt.switch_backend("Agg")
    main()
