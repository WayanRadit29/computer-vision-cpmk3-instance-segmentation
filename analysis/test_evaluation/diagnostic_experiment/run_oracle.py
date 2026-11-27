"""Run the Batch 3.3 frozen-assignment ranking oracle on Run02 personal data."""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "failure_analysis"))
from screening import ROOT, RUN, load_predictions, match, overlap, read_json

MODELS = {"maskrcnn_all": "Mask R-CNN All-Class", "maskrcnn_hard3": "Mask R-CNN Hard-3"}
TARGET = "cabinet-merged"
THRESHOLDS = tuple(round(.50 + i * .05, 2) for i in range(10))
RECALL_GRID = np.linspace(0, 1, 101)


class Assignment(NamedTuple):
    source_index: int
    photo_id: str
    score: float
    status: str
    gt_id: int | None
    iou: float | None
    duplicate: bool


def inputs():
    mapping = read_json(RUN / "image_category_mapping.json")
    photos = {row["id"]: row for row in mapping["images"]["personal"]}
    cabinet = next(row for row in mapping["categories_all_and_hard3"]["classes"]
                   if row["name"] == TARGET)
    annotation_dir = ROOT / "data/custom_test/annotations"
    file_names = ("all_class_master_corrected.coco.json", "hard3_corrected.coco.json")
    datasets = {}
    for model, name, mapping_key in zip(MODELS, file_names, ("all_class_coco", "hard3_coco"), strict=True):
        dataset = read_json(annotation_dir / name)
        names = {row["id"]: row["name"] for row in dataset["categories"]}
        category_id = cabinet[mapping_key]
        assert names[category_id] == TARGET
        assert {row["id"] for row in dataset["images"]} == set(photos)
        gts = defaultdict(list)
        for row in dataset["annotations"]:
            if row["category_id"] == category_id:
                gts[row["image_id"]].append(dict(
                    row, category=TARGET, ignored=bool(row["iscrowd"] or row.get("ignore", False))))
        datasets[model] = names, gts

    def geometry(gts):
        return {g["id"]: (g["image_id"], g["bbox"], g["segmentation"], g["area"], g["iscrowd"], g["ignored"])
                for rows in gts.values() for g in rows}

    assert geometry(datasets["maskrcnn_all"][1]) == geometry(datasets["maskrcnn_hard3"][1])
    assert len(photos) == 40
    with (RUN / "tables/per_class_metrics.csv").open(encoding="utf-8", newline="") as stream:
        official = {
            row["model"]: (float(row["mask_ap50"]), float(row["mask_ap"]))
            for row in csv.DictReader(stream)
            if row["model"] in MODELS and row["class"] == TARGET
            and row["domain"] == "personal" and row["evaluation_scope"] == "primary"
        }
    return photos, datasets, official


def freeze(predictions, gts, photos):
    per_threshold = {t: [] for t in THRESHOLDS}
    for image_id, photo in photos.items():
        detections = [p for p in predictions[image_id] if p["category"] == TARGET and p["seg"] is not None]
        truth = gts[image_id]
        _, ious = overlap(detections, truth, photo)
        valid = range(len(detections))
        for threshold in THRESHOLDS:
            _, assigned = match(detections, truth, ious, threshold, valid)
            for i, pred in enumerate(detections):
                status, j = assigned.get(i, ("FP", None))
                duplicate = status == "FP" and any(
                    not gt["ignored"] and ious[i, k] >= threshold for k, gt in enumerate(truth))
                per_threshold[threshold].append(Assignment(
                    pred["bbox_index"], photo["photo_id"], pred["score"], status,
                    truth[j]["id"] if j is not None else None,
                    float(ious[i, j]) if j is not None else None, duplicate))
    return {t: tuple(sorted(rows, key=lambda a: (-a.score, a.source_index)))
            for t, rows in per_threshold.items()}


def measure(rows, gt_count):
    counts = Counter(row.status for row in rows)
    flags = np.array([row.status == "TP" for row in rows if row.status != "ignored"], dtype=bool)
    tp = np.cumsum(flags, dtype=int)
    recall = tp / gt_count
    precision = tp / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, RECALL_GRID, side="left")
    sampled = np.zeros_like(RECALL_GRID)
    reachable = index < len(recall)
    sampled[reachable] = envelope[index[reachable]]
    return {
        "predictions": len(rows), "tp": counts["TP"], "fp": counts["FP"], "fn": gt_count - counts["TP"],
        "ignored": counts["ignored"], "duplicates": sum(r.duplicate for r in rows),
        "max_recall": counts["TP"] / gt_count, "ap": float(sampled.mean()),
    }, sampled


def run():
    photos, datasets, official = inputs()
    source = read_json(RUN / "integration.json")
    assert source["run_id"] == "20261126T152940Z-d1cca883"
    rows, curves = [], {}
    evidence = {}
    provenance = source["source_archive"]
    with ZipFile(ROOT / provenance["path"]) as archive:
        for model, (categories, gts) in datasets.items():
            gt_count = sum(not g["ignored"] for group in gts.values() for g in group)
            assert gt_count == 12
            predictions = load_predictions(archive, provenance["member_prefix"], model, categories)
            frozen = freeze(predictions, gts, photos)
            ap_by_order = defaultdict(list)
            for threshold, original in frozen.items():
                oracle = tuple(sorted(original, key=lambda row: row.status != "TP"))
                assert Counter(original) == Counter(oracle)
                assert len({r.source_index for r in original}) == len(original)
                for status in ("TP", "FP", "ignored"):
                    assert [r for r in original if r.status == status] == [r for r in oracle if r.status == status]
                base_metrics, base_curve = measure(original, gt_count)
                oracle_metrics, oracle_curve = measure(oracle, gt_count)
                assert all(base_metrics[k] == oracle_metrics[k] for k in base_metrics if k != "ap")
                assert (oracle_metrics["ap"] >= base_metrics["ap"] - 1e-12)
                gap = oracle_metrics["ap"] - base_metrics["ap"]
                if threshold == .50:
                    curves[model] = (base_curve, oracle_curve, base_metrics["max_recall"])
                    expected = {"maskrcnn_all": (116, 10, 106), "maskrcnn_hard3": (1485, 12, 1473)}
                    assert tuple(base_metrics[k] for k in ("predictions", "tp", "fp")) == expected[model]
                    assert oracle_metrics["ap"] == np.mean(RECALL_GRID <= base_metrics["max_recall"])
                    if model == "maskrcnn_hard3":
                        tp = next(r for r in original if r.photo_id == "P018" and r.status == "TP")
                        higher = sum(r.photo_id == "P018" and r.status == "FP" and r.score > tp.score
                                     for r in original)
                        assert (tp.gt_id, tp.source_index, higher) == (124, 1681, 21)
                        assert abs(tp.iou - .933258) < 1e-6 and abs(tp.score - .013883) < 1e-6
                        evidence = {"p018_tp": tp._asdict(), "p018_fp_above_tp": higher}
                for order, metrics in (("original", base_metrics), ("oracle", oracle_metrics)):
                    ap = metrics["ap"]
                    ap_by_order[order].append(ap)
                    rows.append(dict(
                        model=model, iou=f"{threshold:.2f}", ranking=order, images=len(photos),
                        gt_instances=gt_count, **metrics,
                        official_original_ap=official[model][0] if threshold == .50 and order == "original" else "",
                        ranking_headroom_pp=100 * gap if order == "oracle" else ""))
            for order, aps in ap_by_order.items():
                rows.append(dict(
                    model=model, iou="mean_0.50_to_0.95", ranking=order, images=len(photos),
                    gt_instances=gt_count, predictions=len(frozen[.50]), ap=float(np.mean(aps)),
                    official_original_ap=official[model][1] if order == "original" else "",
                    ranking_headroom_pp=100 * (np.mean(aps) - np.mean(ap_by_order["original"]))
                    if order == "oracle" else ""))
            del predictions
    return rows, curves, evidence


def save_figure(curves, rows):
    colors = {"original": "#175F94", "oracle": "#B94D15"}
    ink, muted = "#142939", "#426071"
    size = (1560, 770)
    figure = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(figure)
    fonts = Path(sys.prefix) / "Lib/site-packages/matplotlib/mpl-data/fonts/ttf"

    def text(x, y, value, px, color=ink, bold=False, anchor="lm"):
        path = fonts / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        font = ImageFont.truetype(path, px) if path.is_file() else ImageFont.load_default(size=px)
        draw.text((x, y), value, fill=color, font=font, anchor=anchor)

    text(70, 55, "Precision–recall kelas cabinet-merged", 43, bold=True)
    text(70, 108, "Personal test · mask IoU 0,50 · penugasan TP/FP tetap", 25, muted)
    text(70, 703, "Kurva memakai precision envelope pada 101 titik recall. Garis vertikal putus-putus menandai recall maksimum.", 21, muted)
    text(70, 737, "Oracle memakai label TP/FP dari ground truth untuk memindahkan TP ke depan.", 21, muted)
    ap50 = {(r["model"], r["ranking"]): 100 * r["ap"] for r in rows if r["iou"] == "0.50"}
    for (model, (base, oracle, recall_max)), left in zip(curves.items(), (85, 845), strict=True):
        top, h, w = 222, 400, 618
        bottom = top + h

        def x(v):
            return left + w * float(v)

        def y(v):
            return bottom - h * float(v)

        text(left, 153, MODELS[model], 29, bold=True)
        for order, offset in (("original", 0), ("oracle", 320)):
            color = colors[order]
            draw.line((left + offset, 192, left + offset + 34, 192), fill=color, width=5)
            label = "Asli" if order == "original" else "Oracle"
            score = f"{ap50[model, order]:.2f}".replace(".", ",")
            text(left + offset + 46, 192, f"{label} · AP50 {score}%", 23)
        for tick in np.linspace(0, 1, 6):
            label = f"{tick:.1f}".replace(".", ",")
            draw.line((x(tick), top, x(tick), bottom), fill="#e0e8ed", width=1)
            draw.line((left, y(tick), left + w, y(tick)), fill="#e0e8ed", width=1)
            text(x(tick), bottom + 21, label, 18, muted, anchor="mm")
            text(left - 13, y(tick), label, 18, muted, anchor="rm")
        draw.rectangle((left, top, left + w, bottom), outline="#8da1af", width=2)
        for py in range(top, bottom, 16):
            draw.line((x(recall_max), py, x(recall_max), min(py + 8, bottom)), fill="#758895", width=2)
        for order, precision in (("original", base), ("oracle", oracle)):
            reachable = RECALL_GRID <= recall_max
            xs = np.r_[RECALL_GRID[reachable], recall_max]
            ys = np.r_[precision[reachable], precision[reachable][-1]]
            points = [(x(xs[0]), y(ys[0]))]
            for i in range(1, len(xs)):
                points.extend(((x(xs[i]), y(ys[i - 1])), (x(xs[i]), y(ys[i]))))
            draw.line(points, fill=colors[order], width=5)
        text(left + w / 2, bottom + 48, "Recall", 23, anchor="mm")
        text(left + 15, top + 26, "Precision", 19, muted)
    figure.save(HERE / "precision_recall_oracle.png", dpi=(200, 200))


def main():
    rows, curves, evidence = run()
    with (HERE / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_figure(curves, rows)
    for row in rows:
        if row["iou"] in {"0.50", "mean_0.50_to_0.95"}:
            print(row)
    print(evidence)


if __name__ == "__main__":
    main()
