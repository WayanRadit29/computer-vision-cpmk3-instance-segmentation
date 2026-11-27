"""Screen stored Run02 predictions against frozen personal annotations."""
import argparse
import csv
import gzip
import json
import re
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from pycocotools import mask

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "run02"
IMAGES = ROOT / "data/custom_test/roboflow_raw"
TARGETS = ("bed", "book", "cabinet-merged")
MODELS = tuple(f"{m}_{s}" for m in ("yolo11n", "yolov8n", "maskrcnn") for s in ("all", "hard3"))
THRESHOLDS = (0.50, 0.75)


def read_json(path):
    return json.loads(path.read_bytes())


def as_rle(seg, h, w):
    if isinstance(seg, list):
        return mask.merge(mask.frPyObjects(seg, h, w))
    if isinstance(seg["counts"], list):
        return mask.frPyObjects(seg, h, w)
    return dict(seg, counts=seg["counts"].encode("ascii"))


def archive_rows(archive, prefix, model, kind):
    path = f"{prefix}results/combinations/{model}__personal/predictions_{kind}.json.gz"
    with archive.open(path) as member, gzip.open(member, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def prediction_key(row):
    return row["image_id"], row["category_id"], row["score"], tuple(row["bbox"])


def load_predictions(archive, prefix, model, categories):
    boxes = archive_rows(archive, prefix, model, "bbox")
    masks = defaultdict(deque)
    for index, row in enumerate(archive_rows(archive, prefix, model, "segm")):
        masks[prediction_key(row)].append((index, row["segmentation"]))
    by_image = defaultdict(list)
    for index, row in enumerate(boxes):
        candidates = masks[prediction_key(row)]
        seg_index, encoded = candidates.popleft() if candidates else (None, None)
        by_image[row["image_id"]].append(dict(
            row, id=f"B{index:05}", bbox_index=index, segm_index=seg_index,
            category=categories[row["category_id"]], seg=encoded))
    assert not any(masks.values()), model
    return by_image


def overlap(preds, gts, photo):
    n, k = len(preds), len(gts)
    boxes = np.zeros((n, k))
    masks = np.zeros_like(boxes)
    if n and k:
        crowd = [int(g["iscrowd"]) for g in gts]
        boxes[:] = mask.iou(np.array([p["bbox"] for p in preds], dtype=float),
                            np.array([g["bbox"] for g in gts], dtype=float), crowd)
        h, w = photo["height"], photo["width"]
        valid = [i for i, row in enumerate(preds) if row["seg"] is not None]
        if valid:
            masks[valid] = mask.iou([as_rle(preds[i]["seg"], h, w) for i in valid],
                                    [as_rle(row["segmentation"], h, w) for row in gts], crowd)
    return boxes, masks


def match(preds, gts, ious, threshold, valid):
    """Match confidence-descending, with source-index tie breaks."""
    found, ignored_gt, assigned = {}, set(), {}
    for i in sorted(valid, key=lambda i: (-preds[i]["score"], preds[i]["bbox_index"])):
        compatible = [j for j, gt in enumerate(gts) if gt["category"] == preds[i]["category"]
                      and ious[i, j] >= threshold]
        eligible = [j for j in compatible if j not in found and not gts[j]["ignored"]]
        omitted = [j for j in compatible if gts[j]["ignored"]
                   and (gts[j]["iscrowd"] or j not in ignored_gt)]
        if eligible:
            j = max(eligible, key=lambda j: (ious[i, j], -gts[j]["id"]))
            found[j], assigned[i] = i, ("TP", j)
        elif omitted:
            j = max(omitted, key=lambda j: (ious[i, j], -gts[j]["id"]))
            assigned[i] = ("ignored", j)
            ignored_gt.add(j)
    return found, assigned


def best_index(indices, ious, preds):
    return max(indices, key=lambda i: (ious[i], preds[i]["score"], -preds[i]["bbox_index"]), default=None)


def analyze_photo(preds, gts, photo):
    boxes, masks = overlap(preds, gts, photo)
    target = [j for j, g in enumerate(gts) if g["category"] in TARGETS and not g["ignored"]]
    all_preds = range(len(preds))
    valid_masks = [i for i in all_preds if preds[i]["seg"] is not None]
    matched = {
        f"{kind}{int(t * 100)}": match(preds, gts, metric, t, valid)
        for kind, metric, valid in (("box", boxes, all_preds), ("mask", masks, valid_masks))
        for t in THRESHOLDS
    }

    def evidence(i, j):
        return dict(pred_id=preds[i]["id"], category=preds[i]["category"], score=preds[i]["score"],
                    bbox_iou=round(float(boxes[i, j]), 8), mask_iou=round(float(masks[i, j]), 8),
                    bbox_iou_kind="IoA_crowd" if gts[j]["iscrowd"] else "IoU",
                    segm_index=preds[i]["segm_index"])

    gt_rows = []
    for j in target:
        row = gts[j]
        same = [i for i in all_preds if preds[i]["category"] == row["category"]]
        same_mask = [i for i in same if preds[i]["seg"] is not None]
        wrong = [i for i in valid_masks if preds[i]["category"] != row["category"]]
        indices = {
            "best_same_mask": best_index(same_mask, masks[:, j], preds),
            "best_same_box": best_index(same, boxes[:, j], preds),
            "best_other_mask": best_index(wrong, masks[:, j], preds),
        }
        gt_rows.append({
            "gt": {k: row[k] for k in ("id", "category", "category_id", "bbox", "area", "iscrowd", "ignored")},
            "area_pct": 100 * row["area"] / (photo["width"] * photo["height"]),
            "same_class_preds": len(same), "same_class_segm": len(same_mask),
            "overlap_any_box10": int(sum(boxes[i, j] >= .10 for i in all_preds)),
            "overlap_same_mask50": int(sum(masks[i, j] >= .50 for i in same_mask)),
            "overlap_other_mask50": int(sum(masks[i, j] >= .50 for i in wrong)),
            **{key: evidence(i, j) if i is not None else None for key, i in indices.items()},
            **{key: evidence(found[j], j) if j in found else None for key, (found, _) in matched.items()},
        })

    pred_rows = []
    for i, p in enumerate(preds):
        if p["category"] not in TARGETS and not any(max(boxes[i, j], masks[i, j]) >= .10 for j in target):
            continue
        eligible = [j for j, gt in enumerate(gts) if not gt["ignored"]]
        same = [j for j in eligible if gts[j]["category"] == p["category"]]
        other = [j for j in eligible if gts[j]["category"] != p["category"]]
        best_same = max(same, key=lambda j: masks[i, j], default=None)
        best_other = max(other, key=lambda j: masks[i, j], default=None)
        best_any = max(eligible, key=lambda j: (masks[i, j], boxes[i, j]), default=None)
        best_box = max(eligible, key=lambda j: boxes[i, j], default=None)
        existing = matched["mask50"][1].get(i)
        if p["seg"] is None:
            status, associated = "bbox_only", best_box
        elif existing:
            status, associated = existing
        elif best_same is not None and masks[i, best_same] >= .5:
            status, associated = "duplicate", best_same
        elif best_other is not None and masks[i, best_other] >= .5:
            status, associated = "wrong_class_overlap", best_other
        elif best_any is not None and max(masks[i, best_any], boxes[i, best_box]) >= .10:
            status, associated = "low_overlap", best_any if masks[i, best_any] >= .10 else best_box
        else:
            status, associated = "low_or_no_gt_overlap", best_any
        pred_rows.append({
            "id": p["id"], "category": p["category"], "category_id": p["category_id"], "bbox": p["bbox"],
            "score": p["score"], "bbox_index": p["bbox_index"], "segm_index": p["segm_index"],
            "mask50_status": status, "assigned_gt": None if associated is None else gts[associated]["id"],
            "assigned_category": None if associated is None else gts[associated]["category"],
            "assigned_box_iou": None if associated is None else float(boxes[i, associated]),
            "assigned_mask_iou": None if associated is None or p["seg"] is None else float(masks[i, associated]),
            "matched_mask75_gt": gts[matched["mask75"][1][i][1]]["id"] if i in matched["mask75"][1]
            and matched["mask75"][1][i][0] == "TP" else None,
            "matched_box50_gt": gts[matched["box50"][1][i][1]]["id"] if i in matched["box50"][1]
            and matched["box50"][1][i][0] == "TP" else None,
            "matched_box75_gt": gts[matched["box75"][1][i][1]]["id"] if i in matched["box75"][1]
            and matched["box75"][1][i][0] == "TP" else None,
        })
    return {"bbox_total": len(preds), "segm_total": len(valid_masks), "gt": gt_rows, "pred": pred_rows}


def run_screen():
    provenance = read_json(RUN / "integration.json")
    photos = {row["id"]: row for row in read_json(RUN / "image_category_mapping.json")["images"]["personal"]}
    conditions_doc = (ROOT / "analysis/personal_test_qa/difficult_conditions.md").read_text(encoding="utf-8")
    conditions = dict(re.findall(r"\[\s*(P\d+)\].*? \| ([^|]+?) \|", conditions_doc))
    with (ROOT / "evidence/photo_metadata/photo_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = {row["photo_id"]: row for row in csv.DictReader(f)}
    assert len(photos) == len(manifest) == 40
    all_cases = {p["photo_id"]: {"photo": p, "condition": conditions.get(p["photo_id"]),
                                "models": {}} for p in photos.values()}
    names = {"all": "all_class_master_corrected.coco.json", "hard3": "hard3_corrected.coco.json"}
    datasets = {}
    for scope, name in names.items():
        gt = read_json(ROOT / "data/custom_test/annotations" / name)
        cats = {row["id"]: row["name"] for row in gt["categories"]}
        by_image = defaultdict(list)
        for row in gt["annotations"]:
            by_image[row["image_id"]].append(dict(
                row, category=cats[row["category_id"]], ignored=bool(row["iscrowd"] or row.get("ignore", 0))))
        datasets[scope] = cats, by_image
        assert {i["id"] for i in gt["images"]} == set(photos)
    with zipfile.ZipFile(ROOT / provenance["source_archive"]["path"]) as archive:
        for model in MODELS:
            scope = model.rsplit("_", 1)[1]
            categories, gts = datasets[scope]
            pred = load_predictions(archive, provenance["source_archive"]["member_prefix"], model, categories)
            for photo in photos.values():
                image_id = photo["id"]
                all_cases[photo["photo_id"]]["models"][model] = analyze_photo(pred[image_id], gts[image_id], photo)
            print(model, sum(len(r["gt"]) for p in all_cases.values() for m, r in p["models"].items() if m == model),
                  flush=True)
    return all_cases


def run_summary(screen):
    for target in TARGETS:
        print(f"\n{target}")
        for model in MODELS:
            gt = [g for p in screen.values() for g in p["models"][model]["gt"] if g["gt"]["category"] == target]
            pred = [r for p in screen.values() for r in p["models"][model]["pred"] if r["category"] == target]
            print(model, "GT", len(gt), "TP50/75", sum(bool(g["mask50"]) for g in gt), sum(bool(g["mask75"]) for g in gt),
                  "box50", sum(bool(g["box50"]) for g in gt), dict(Counter(r["mask50_status"] for r in pred)))
    for pid, case in screen.items():
        models = case["models"]
        for g in models[MODELS[0]]["gt"]:
            ident, cat = g["gt"]["id"], g["gt"]["category"]
            evidence = []
            for model, res in models.items():
                gt = next(x for x in res["gt"] if x["gt"]["id"] == ident)
                p = gt["best_same_mask"]
                other = gt["best_other_mask"]
                counters = Counter(x["mask50_status"] for x in res["pred"] if x["category"] == cat)
                evidence.append(
                    f"{p['mask_iou']:.2f}/{p['bbox_iou']:.2f}@{p['score']:.3f}" if p else "-")
                evidence[-1] += f" t{int(bool(gt['mask50']))} b{int(bool(gt['box50']))} n{sum(counters.values())} d{counters['duplicate']}"
                if other and other["mask_iou"] >= .50:
                    evidence[-1] += f" !{other['category']}:{other['mask_iou']:.2f}"
            print(f"{pid}/{case['photo']['id']} #{ident} {cat} {g['area_pct']:.2f}% {case['condition'] or '-'}", " | ".join(evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Optional local JSON derived from the source artifacts")
    args = parser.parse_args()
    screen = run_screen()
    if args.cache:
        args.cache.write_text(json.dumps(screen, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    run_summary(screen)


if __name__ == "__main__":
    main()
