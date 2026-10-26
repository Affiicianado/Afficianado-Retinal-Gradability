"""The two report figures that are drawn the same way every time.

Copied from retina-baseline (plot_confusion) and retina-04-cnn (plot_history).
"""
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["Good", "Usable", "Reject"]


def plot_confusion(cm, path, title="", classes=CLASSES):
    """Row-normalised confusion matrix with counts and percentages.

    Read the bottom-left cell first: Reject predicted as Good is the dangerous error.
    """
    cm = np.asarray(cm)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cmn, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(3), classes)
    ax.set_yticks(range(3), classes)
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
    plt.show()


def plot_history(hists, path, title=""):
    """Loss and validation macro-F1 per epoch, one line per training phase."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in hists.items():
        axes[0].plot(h.history["loss"], label=f"{name} train")
        axes[0].plot(h.history["val_loss"], "--", label=f"{name} val")
        axes[1].plot(h.history["val_macro_f1"], label=f"{name} val macro-F1")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro F1")
    for a in axes:
        a.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
