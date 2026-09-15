# FILE: src/eval/plots.py
"""Report figures. Every plot in the report comes from here."""

import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import CLASSES


def plot_confusion(cm, path, title=""):
    """Row-normalised confusion matrix with counts and percentages."""
    cm = np.asarray(cm)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cmn, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title, fontsize=11)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()