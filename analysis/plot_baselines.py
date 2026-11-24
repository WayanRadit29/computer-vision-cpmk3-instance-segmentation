"""Buat grafik laporan dari tiga CSV baseline tanpa menjalankan model."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "analysis" / "figures"
MODELS = (
    ("YOLO11n-seg", "yolo11n_all_classes", "metrics/mAP50-95(M)", "#17619C"),
    ("YOLOv8n-seg", "yolov8n_all_classes", "metrics/mAP50-95(M)", "#B65300"),
    ("Mask R-CNN", "maskrcnn_all_classes", "mask_map50_95", "#197451"),
)


def read_results(folder):
    path = ROOT / "experiments" / folder / "metrics" / "results.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def series(rows, column, scale=1):
    return [float(row[column]) * scale for row in rows]


def save(figure, name):
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def style_axis(axis, grid_axis="y"):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis=grid_axis, alpha=0.22)


def plot_maskrcnn_loss(rows):
    epochs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
    ax.plot(epochs, series(rows, "train_total_loss"), "o-", color=MODELS[2][3],
            markersize=3, label="Total loss pelatihan")
    ax.set(title="Loss pelatihan Mask R-CNN", xlabel="Epoch", ylabel="Total loss")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    style_axis(ax)
    ax.legend(frameon=False)
    save(fig, "mask_rcnn_training_loss.png")


def plot_validation(runs):
    fig, (curve, peaks) = plt.subplots(
        1, 2, figsize=(11, 4.3), width_ratios=(1.7, 1), layout="constrained"
    )
    styles = ("-", "--", "-.")
    best_scores = []
    for (label, _, key, color), rows, line_style in zip(MODELS, runs, styles):
        epochs = [int(row["epoch"]) for row in rows]
        scores = series(rows, key, 100)
        top = max(range(len(scores)), key=scores.__getitem__)
        best_scores.append((epochs[top], scores[top]))
        curve.plot(epochs, scores, color=color, linestyle=line_style, label=label)
        curve.scatter(epochs[top], scores[top], color=color, s=24, zorder=4)
    curve.set(title="Perkembangan per epoch", xlabel="Epoch", ylabel="Mask mAP50–95 (%)")
    curve.set_ylim(bottom=0)
    curve.legend(loc="lower right", frameon=False)
    style_axis(curve)

    bars = peaks.barh(
        [m[0] for m in MODELS], [s[1] for s in best_scores], color=[m[3] for m in MODELS]
    )
    peaks.invert_yaxis()
    peaks.bar_label(
        bars, labels=[f"{score:.2f}%".replace(".", ",") + f" · e{epoch}"
                      for epoch, score in best_scores], padding=4, fontsize=9
    )
    peaks.set(title="Skor tertinggi", xlabel="Mask mAP50–95 (%)", xlim=(0, 26))
    style_axis(peaks, "x")
    save(fig, "all_class_validation_mask_map.png")


def plot_durations(runs):
    minutes = [float(rows[-1]["time"]) / 60 for rows in runs]
    fig, ax = plt.subplots(figsize=(7.5, 3.5), layout="constrained")
    bars = ax.barh([m[0] for m in MODELS], minutes, color=[m[3] for m in MODELS])
    ax.invert_yaxis()
    ax.bar_label(
        bars, labels=[f"{elapsed:.2f}".replace(".", ",") + f" menit · {len(rows)} epoch"
                      for elapsed, rows in zip(minutes, runs)], padding=5
    )
    ax.set(title="Waktu kumulatif pada log eksperimen", xlabel="Menit",
           xlim=(0, max(minutes) * 1.52))
    style_axis(ax, "x")
    save(fig, "all_class_recorded_duration.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = [read_results(folder) for _, folder, _, _ in MODELS]
    plot_maskrcnn_loss(runs[-1])
    plot_validation(runs)
    plot_durations(runs)


if __name__ == "__main__":
    main()
