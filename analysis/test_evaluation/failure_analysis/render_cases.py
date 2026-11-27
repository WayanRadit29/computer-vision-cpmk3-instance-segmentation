"""Export auditable rows and the five diagnostic plates from stored Run02 data."""
import argparse
import csv
import sys
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from screening import HERE, IMAGES, MODELS, ROOT, RUN, as_rle, load_predictions, mask, overlap, read_json, run_screen

CASES = (
    ("F01", "P021", 87, "book", "Label buku hilang pada YOLO11n Hard-3", {
        "yolo11n_all": ("B01965", "B01964"), "yolo11n_hard3": ("B00352",),
        "yolov8n_hard3": ("B00182",), "maskrcnn_hard3": ("B01926",),
    }),
    ("F02", "P026", 42, "book", "Mask buku mengikuti objek, tetapi label keliru", {
        "yolov8n_all": ("B02480",), "yolov8n_hard3": ("B00201",),
        "yolo11n_hard3": ("B00381", "B00380"), "maskrcnn_hard3": ("B02405",),
    }),
    ("F03", "P018", 124, "cabinet-merged", "Deteksi lemari benar berada di bawah prediksi keliru", {
        "maskrcnn_all": ("B01718", "B01790"),
        "maskrcnn_hard3": ("B01681", "B01629", "B01631"),
        "yolo11n_hard3": ("B00316",), "yolov8n_hard3": ("B00158",),
    }),
    ("F04", "P006", 23, "bed", "Kasur kecil dan tertutup luput pada kedua YOLO", {
        "maskrcnn_all": ("B00550", "B00584"), "yolo11n_hard3": ("B00080",),
        "yolov8n_hard3": ("B00056",), "maskrcnn_hard3": ("B00558", "B00569"),
    }),
    ("F05", "P012", 77, "bed", "Kotak cocok, mask kasur gagal memenuhi ambang", {
        "yolo11n_all": ("B01105",), "yolo11n_hard3": ("B00195",),
        "yolov8n_hard3": ("B00104", "B00098"), "maskrcnn_hard3": ("B01102", "B01123"),
    }),
)
COLORS = {"bed": "#00B6D4", "book": "#FFB319", "cabinet-merged": "#C78BFF"}
OTHER = "#76DE8A"
MODEL_NAMES = {"yolo11n": "YOLO11n-seg", "yolov8n": "YOLOv8n-seg", "maskrcnn": "Mask R-CNN"}
FONT_DIR = Path(sys.prefix) / "Lib/site-packages/matplotlib/mpl-data/fonts/ttf"
INK, MUTED = "#172936", "#405967"
STATUS_NOTE = {"TP": "TP", "duplicate": "dup.", "low_overlap": "IoU rendah",
               "low_or_no_gt_overlap": "overlap rendah", "wrong_class_overlap": "label berbeda",
               "bbox_only": "bbox saja", "ignored": "ignore"}


@lru_cache(maxsize=20)
def label_font(size, bold=False):
    return ImageFont.truetype(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def model_name(key):
    family, scope = key.rsplit("_", 1)
    return f"{MODEL_NAMES[family]} {'All-Class' if scope == 'all' else 'Hard-3'}"


def paint(canvas, rle, crop, box, category, dashed=False):
    """Decode only the current displayed mask, then scale for rendering."""
    x, y, w, h = crop
    scale_x, scale_y = canvas.width / w, canvas.height / h
    color = COLORS.get(category, OTHER)
    if rle:
        decoded = Image.fromarray(mask.decode(rle) * np.uint8(255))
        cover = decoded.crop((x, y, x + w, y + h)).resize(canvas.size, Image.Resampling.NEAREST)
        canvas.paste(color, (0, 0), cover.point(lambda px: int(0.27 * px)))
        outline = ImageChops.subtract(cover.filter(ImageFilter.MaxFilter(3)), cover.filter(ImageFilter.MinFilter(3)))
        canvas.paste(color, (0, 0), outline)
    x0, y0, bw, bh = box
    boundaries = ((x0-x)*scale_x, (y0-y)*scale_y, (x0+bw-x)*scale_x, (y0+bh-y)*scale_y)
    draw = ImageDraw.Draw(canvas)
    if dashed:
        a,b,c,d = boundaries
        for v in range(int(a), int(c), 13):
            draw.line((v,b,min(v+8,c),b), fill=color, width=2)
            draw.line((v,d,min(v+8,c),d), fill=color, width=2)
        for v in range(int(b), int(d), 13):
            draw.line((a,v,a,min(v+8,d)), fill=color, width=2)
            draw.line((c,v,c,min(v+8,d)), fill=color, width=2)
    else:
        draw.rectangle(boundaries, outline=color, width=2)


def plate(case, item, gt, targets, panels):
    cid, pid, gt_id, category, title, plan = case
    photo = item["photo"]
    w, h = photo["width"], photo["height"]
    with Image.open(IMAGES / photo["file_name"]) as img:
        original = img.convert("RGB")
    assert original.size == (w, h)
    crop = (0, 0, w, h)
    detail = None
    if pid == "P006":
        x, y, bw, bh = gt["bbox"]
        l, t = max(0, int(x - .55*bw)), max(0, int(y - .22*bh))
        r, b = min(w, int(x + 1.15*bw)), min(h, int(y + 1.22*bh))
        detail = (l, t, r-l, b-t)
    panel_w, photo_h, gap, inset_h = 660, 465, 22, (200 if detail else 0)
    row_h = photo_h + inset_h + 180
    top = 167
    figure = Image.new("RGB", (panel_w * 3 + gap * 4, top + row_h * 2 + 45), "#edf2f5")
    draw = ImageDraw.Draw(figure)

    def text(x, y, content, size, fill=INK, bold=False):
        draw.text((x, y), content, font=label_font(size, bold), fill=fill)

    text(gap, 20, f"{cid}  /  {pid}  /  COCO {photo['id']}  /  GT #{gt_id}", 26, MUTED, True)
    text(gap, 57, title, 42, INK, True)
    legend_y = 120
    current_x = gap
    for name, color in (*COLORS.items(), ("kelas lain", OTHER)):
        draw.rectangle((current_x, legend_y, current_x+25, legend_y+23), fill=color, outline=MUTED)
        text(current_x+34, legend_y-3, name, 22)
        current_x += 235 if name == "cabinet-merged" else 156
    text(current_x+12, legend_y-3, "Panel menunjukkan prediksi terpilih. Hitungan memakai seluruh keluaran tersimpan.", 18, MUTED)

    def display(x, y, region, max_height, objects, mark_crop=False):
        l, t, ww, hh = region
        ratio = min((panel_w - 24) / ww, max_height / hh)
        display_img = original.crop((l,t,l+ww,t+hh)).resize((round(ww*ratio),round(hh*ratio)), Image.Resampling.LANCZOS)
        for row, is_gt, dashed in objects:
            segmentation = row["segmentation"] if is_gt else row.get("seg")
            paint(display_img, as_rle(segmentation, h, w) if segmentation else None, region,
                  row["bbox"], row["category"], dashed)
        if mark_crop and detail:
            a,b,ww,hh=detail
            ImageDraw.Draw(display_img).rectangle((a*ratio,b*ratio,(a+ww)*ratio,(b+hh)*ratio), outline="white", width=3)
        figure.paste(display_img, (x+(panel_w-display_img.width)//2,y+(max_height-display_img.height)//2))

    views = [
        ("Foto asli", [], [f"Ekspor personal {w} × {h} piksel."]),
        ("Ground truth beku", [(a, True, a["id"] != gt_id) for a in targets],
         [f"Fokus #{gt_id} {category}. n={sum(a['category']==category for a in targets)} GT kelas fokus.",
          "Kotak putus-putus menandai target Hard-3 lain."]),
    ]
    for model, ids in plan.items():
        data = item["models"][model]
        preds = panels[model]
        diag = {p["id"]: p for p in data["pred"]}
        assert list(preds) == list(ids)
        target = [p for p in data["pred"] if p["category"] == category]
        stats = Counter(p["mask50_status"] for p in target)
        seg = sum(p["segm_index"] is not None for p in target)
        fp = seg - stats["TP"] - stats["ignored"]
        notes = [f"{data['segm_total']} mask total · {seg} mask kelas fokus · TP {stats['TP']} / FP {fp} / dup. {stats['duplicate']}"]
        objects = []
        for p in preds.values():
            status = diag[p["id"]]["mask50_status"]
            objects.append((p, False, p["category"] != category or status == "duplicate"))
            notes.append(f"{p['id']} · {p['category']} · s={p['score']:.4f} · {STATUS_NOTE[status]}")
        views.append((model_name(model), objects, notes))

    for i, (name, objects, notes) in enumerate(views):
        x, y = gap + i%3*(panel_w+gap), top+i//3*row_h
        draw.rounded_rectangle((x,y,x+panel_w,y+row_h-14), radius=10, fill="white")
        text(x+15, y+13, name, 28, INK, True)
        display(x, y+53, crop, photo_h, objects, mark_crop=True)
        if detail:
            text(x+15, y+photo_h+56, "Detail selaras dari kotak putih", 15, MUTED)
            display(x, y+photo_h+80, detail, inset_h-29, objects)
        for k, line in enumerate(notes):
            text(x+15, y+photo_h+inset_h+59+25*k, line, 18, MUTED)
    text(gap, figure.height-39,
         "TP dan FP berdasarkan mask IoU ≥ 0,50, satu-ke-satu dan sadar kelas. Duplikat termasuk FP. s = confidence.",
         18, MUTED)
    dest = HERE / f"figures/{cid}_{pid}.png"
    dest.parent.mkdir(exist_ok=True)
    figure.save(dest, dpi=(220, 220), optimize=True)


def export(screen, keep_figures=False, figure_cases=None):
    provenance = read_json(RUN / "integration.json")["source_archive"]
    annotation_dir = ROOT / "data/custom_test/annotations"
    annotations = read_json(annotation_dir / "all_class_master_corrected.coco.json")
    categories = {r["id"]:r["name"] for r in annotations["categories"]}
    hard_gt = read_json(annotation_dir / "hard3_corrected.coco.json")
    hard_categories = {row["id"]: row["name"] for row in hard_gt["categories"]}
    gt_by_image = defaultdict(list)
    for row in annotations["annotations"]:
        gt_by_image[row["image_id"]].append(dict(row, category=categories[row["category_id"]],
                                                   ignored=bool(row["iscrowd"] or row.get("ignore", 0))))
    cases = [case for case in CASES if not figure_cases or case[0] in figure_cases]
    pids = {case[1] for case in cases}
    archive = zipfile.ZipFile(ROOT / provenance["path"])
    try:
        selected = {}
        for model in MODELS:
            names = categories if model.endswith("_all") else hard_categories
            source = load_predictions(archive, provenance["member_prefix"], model, names)
            selected[model] = {pid: source[screen[pid]["photo"]["id"]] for pid in pids}
            del source
    finally:
        archive.close()

    rows = []
    for case in cases:
        cid, pid, gt_id, category, _, plan = case
        item = screen[pid]
        photo = item["photo"]
        master = gt_by_image[photo["id"]]
        focus = next(r for r in master if r["id"] == gt_id)
        assert focus["category"] == category
        panels = {}
        for model in MODELS:
            data = item["models"][model]
            target = next(r for r in data["gt"] if r["gt"]["id"] == gt_id)
            lookup = {p["id"]:p for p in selected[model][pid]}
            chosen = plan.get(model, ())
            if model in plan:
                panels[model] = {pid:lookup[pid] for pid in chosen}
            if figure_cases:
                continue
            master_valid = [g for g in master if not g["ignored"]]
            master_focus = master_valid.index(focus)
            display_records = [lookup[p["id"]] for p in data["pred"]]
            box, iou = overlap(display_records, master_valid, photo)
            h, w = photo["height"], photo["width"]
            focus_area = int(mask.area(as_rle(focus["segmentation"],h,w)))
            common = dict(case_id=cid, photo_id=pid, coco_image_id=photo["id"], file_name=photo["file_name"],
                          checkpoint=model, condition=item["condition"], focus_gt_id=gt_id, focus_class=category,
                          focus_area_pct=target["area_pct"], all_bbox_count=data["bbox_total"],
                          all_segm_count=data["segm_total"])
            for g in data["gt"]:
                gt = g["gt"]
                for role in ("box50","box75","mask50","mask75","best_same_box","best_same_mask","best_other_mask"):
                    evidence = g[role]
                    pred = lookup[evidence["pred_id"]] if evidence else None
                    rows.append(dict(
                        common, row_type="gt_evidence", evidence_role=role, gt_annotation_id=gt["id"],
                        gt_category_id=gt["category_id"], gt_class=gt["category"], gt_bbox_xywh=gt["bbox"],
                        gt_area_pct=g["area_pct"], same_class_preds=g["same_class_preds"],
                        same_class_segm=g["same_class_segm"], overlap_any_box10=g["overlap_any_box10"],
                        overlap_same_mask50=g["overlap_same_mask50"], overlap_other_mask50=g["overlap_other_mask50"],
                        prediction_id=pred["id"] if pred else None, bbox_source_index=pred["bbox_index"] if pred else None,
                        segm_source_index=pred["segm_index"] if pred else None, pred_class=pred["category"] if pred else None,
                        pred_category_id=pred["category_id"] if pred else None,
                        confidence=pred["score"] if pred else None, bbox_iou=evidence["bbox_iou"] if pred else None,
                        mask_iou=evidence["mask_iou"] if pred and pred["seg"] is not None else None,
                        displayed=int(bool(pred and pred["id"] in chosen))))
            for i, p in enumerate(data["pred"]):
                if p["category"] not in COLORS and p["id"] not in chosen:
                    continue
                original = display_records[i]
                mious, bious = iou[i], box[i]
                j = max(range(len(master_valid)), key=lambda j:(mious[j],bious[j]))
                rle = as_rle(original["seg"], h, w) if original["seg"] is not None else None
                pred_area = int(mask.area(rle)) if rle else None
                focus_iou = float(mious[master_focus]) if rle else None
                intersection = (focus_iou*(pred_area+focus_area)/(1+focus_iou)) if focus_iou is not None else None
                rows.append(dict(
                    common, row_type="prediction", evidence_role=p["mask50_status"],
                    gt_annotation_id=p["assigned_gt"], gt_class=p["assigned_category"], prediction_id=p["id"],
                    bbox_source_index=p["bbox_index"], segm_source_index=p["segm_index"], pred_class=p["category"],
                    pred_category_id=p["category_id"], pred_bbox_xywh=p["bbox"], confidence=p["score"],
                    bbox_iou=p["assigned_box_iou"], mask_iou=p["assigned_mask_iou"], displayed=int(p["id"] in chosen),
                    matched_mask75_gt=p["matched_mask75_gt"], matched_box50_gt=p["matched_box50_gt"],
                    matched_box75_gt=p["matched_box75_gt"], focus_bbox_iou=float(bious[master_focus]),
                    focus_mask_iou=focus_iou, focus_gt_covered=intersection/focus_area if intersection is not None else None,
                    pred_fraction_on_focus=intersection/pred_area if intersection is not None and pred_area else None,
                    best_full_gt_id=master_valid[j]["id"], best_full_gt_class=master_valid[j]["category"],
                    best_full_gt_mask_iou=float(mious[j]) if rle else None, best_full_gt_bbox_iou=float(bious[j])))
        dest = HERE / f"figures/{cid}_{pid}.png"
        if not keep_figures or not dest.is_file():
            plate(case, item, focus, [x for x in master if x["category"] in COLORS], panels)
        print(f"{cid} {pid} COCO={photo['id']} GT={gt_id}", flush=True)
    if figure_cases:
        return
    with (HERE / "case_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows in case_metrics.csv", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Optional local output from screening.py for the same Run02")
    parser.add_argument("--keep-figures", action="store_true", help="Keep existing plates while regenerating the CSV")
    parser.add_argument("--figure-cases", nargs="+", choices=[case[0] for case in CASES],
                        help="Only render the selected plates and leave the CSV in place")
    args=parser.parse_args()
    export(read_json(args.cache) if args.cache else run_screen(), args.keep_figures, args.figure_cases)
