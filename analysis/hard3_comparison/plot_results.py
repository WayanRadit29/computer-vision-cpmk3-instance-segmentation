"""Buat ulang tabel turunan dan grafik Hard-3 dari artefak yang tersimpan."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = {"yolo11n": "YOLO11n-seg", "yolov8n": "YOLOv8n-seg", "maskrcnn": "Mask R-CNN"}
CLASSES = ["bed", "book", "cabinet-merged"]
BASELINE_COLOR, HARD3_COLOR = "#32628a", "#b85c27"


def save(fig, filename):
    fig.savefig(HERE / filename, dpi=170, facecolor="white")
    plt.close(fig)


def relative_change(table, baseline):
    return 100 * table["delta_pp"] / table[baseline].where(table[baseline] > 0)


def with_relative_change():
    per_class = pd.read_csv(HERE / "per_class_comparison.csv")
    means = pd.read_csv(HERE / "model_comparison.csv")
    per_class["relative_change_pct"] = relative_change(per_class, "all_class_ap_pct")
    means["relative_change_pct"] = relative_change(means, "all_class_mean_ap_pct")
    aggregates = per_class.groupby("model")[["all_class_ap_pct", "hard3_ap_pct", "delta_pp"]].mean()
    assert np.allclose(aggregates.loc[means.model].to_numpy(),
                       means[["all_class_mean_ap_pct", "hard3_mean_ap_pct", "delta_pp"]].to_numpy())
    per_class.to_csv(HERE / "per_class_comparison_with_relative.csv", index=False, na_rep="NA")
    means.to_csv(HERE / "model_comparison_with_relative.csv", index=False, na_rep="NA")
    return per_class


def plot_per_class(per_class):
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 4), sharey=True, layout="constrained")
    x = np.arange(len(CLASSES))
    width = 0.35
    for ax, (key, title) in zip(axes, MODELS.items()):
        rows = per_class.loc[per_class.model.eq(key)].set_index("class_name").loc[CLASSES]
        for offset, name, color, column in (
            (-width / 2, "All-Class", BASELINE_COLOR, "all_class_ap_pct"),
            (width / 2, "Hard-3", HARD3_COLOR, "hard3_ap_pct"),
        ):
            bars = ax.bar(x + offset, rows[column], width, color=color, label=name, zorder=3)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set(xticks=x, xticklabels=["bed", "book", "cabinet-\nmerged"], title=title, ylim=(0, 13))
        ax.grid(axis="y", alpha=0.22, zorder=0)
    axes[0].set_ylabel("Mask AP50–95 (%)")
    axes[-1].legend(loc="upper left", frameon=False)
    fig.suptitle("All-Class dan Hard-3 pada kategori validation yang sama", fontsize=13)
    save(fig, "per_class_comparison.png")


def plot_curves(histories):
    fig_loss, axes_loss = plt.subplots(1, 3, figsize=(11.1, 3.9), layout="constrained")
    fig_ap, axes_ap = plt.subplots(1, 3, figsize=(11.1, 4), sharey=True, layout="constrained")
    for ax_loss, ax_ap, (key, name) in zip(axes_loss, axes_ap, MODELS.items()):
        data = histories[key]
        epoch = data["epoch"]
        maskrcnn = key == "maskrcnn"
        if maskrcnn:
            ax_loss.plot(epoch, data["train_total_loss"], color=HARD3_COLOR, label="Train")
            ax_loss.set_ylabel("Total loss")
            ax_loss.text(0.97, 0.91, "Validation loss\ntidak dicatat",
                         transform=ax_loss.transAxes, ha="right", va="top", fontsize=8)
            mask_ap, box_ap = data["mask_map50_95"], data["box_map50_95"]
        else:
            ax_loss.plot(epoch, data["train/seg_loss"], color=HARD3_COLOR, label="Train")
            ax_loss.plot(epoch, data["val/seg_loss"], color=BASELINE_COLOR, label="Validation")
            ax_loss.set_ylabel("Segmentation loss")
            mask_ap, box_ap = data["metrics/mAP50-95(M)"], data["metrics/mAP50-95(B)"]
        ax_loss.legend(frameon=False, fontsize=8)
        ax_ap.plot(epoch, 100 * mask_ap, color=HARD3_COLOR, label="Mask")
        ax_ap.plot(epoch, 100 * box_ap, color=BASELINE_COLOR, linestyle="--", label="Box")
        for ax in (ax_loss, ax_ap):
            ax.set(xlabel="Epoch", title=name, xlim=(1, len(epoch)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.grid(alpha=0.22)
    axes_ap[0].set_ylabel("mAP50–95 validation (%)")
    axes_ap[-1].legend(loc="upper right", frameon=False, fontsize=8)
    fig_loss.suptitle("Kurva loss Hard-3 menurut definisi masing-masing model", fontsize=13)
    fig_ap.suptitle("Metrik validation per epoch dari evaluator pelatihan", fontsize=13)
    save(fig_loss, "training_loss_curves.png")
    save(fig_ap, "epoch_validation_metrics.png")


def main():
    plot_per_class(with_relative_change())
    histories = {
        key: pd.read_csv(ROOT / "experiments" / f"{key}_hard3" / "metrics" / "results.csv")
        for key in MODELS
    }
    plot_curves(histories)


if __name__ == "__main__":
    main()
