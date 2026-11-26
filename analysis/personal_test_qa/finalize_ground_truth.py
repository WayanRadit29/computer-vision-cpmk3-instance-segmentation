"""Terapkan empat keputusan annotator pada kandidat COCO personal test set."""

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
from importlib.metadata import version
from io import BytesIO, StringIO
import json
from pathlib import Path
import re

from build_candidates import CANDIDATES, HERE, IMAGE_IDS, ORIGINAL, RAW, ROOT, encode, load, sha
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask
from pycocotools.coco import COCO


OUTPUT = HERE / "finalization"
STATUS = "READY FOR FINAL APPROVAL"
REVIEW_DATE = "2026-11-26"
BEFORE_MASTER = "all_class_master_candidate.coco.json"
BEFORE_HARD3 = "hard3_candidate.coco.json"
AFTER_MASTER = "all_class_master_corrected.coco.json"
AFTER_HARD3 = "hard3_corrected.coco.json"
DECISIONS = (
    ("P003", 33, "bed", 32, "pillow"),
    ("P005", 191, "wall-other-merged", 193, "laptop"),
    ("P005", 196, "paper", 195, "mouse"),
    ("P008", 29, "window-other", 27, "cup"),
)


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_sources(provenance, images):
    paths = {ROOT / provenance["zip_name"]: provenance["zip_sha256"]}
    paths.update({RAW / name: digest for name, digest in provenance["raw_files_sha256"].items()})
    paths.update({CANDIDATES / name: digest for name, digest in provenance["candidate_sha256"].items()})
    for row in images:
        paths[ORIGINAL / row["original_name"]] = row["original_sha256"]
        require(paths[RAW / row["exported_file"]] == row["exported_sha256"], "Checksum sumber ekspor berbeda.")
    for subdir in ("photo_metadata", "annotation_screenshots"):
        for path in (ROOT / "evidence" / subdir).iterdir():
            if path.is_file():
                paths[path] = file_hash(path)
    for path, expected in paths.items():
        require(file_hash(path) == expected, f"SHA-256 sumber berbeda pada {path}")
    return paths


def coco_index(data):
    index = COCO()
    index.dataset = data
    index.createIndex()
    for key, by_id in (("images", index.imgs), ("annotations", index.anns), ("categories", index.cats)):
        require(len(data[key]) == len(by_id), f"ID duplikat pada {key}")
        require(all(type(i) is int and i >= 0 for i in by_id), f"ID tidak valid pada {key}")
    return index


def validate(data, photo_rows, labels):
    index = coco_index(data)
    require({i: c["name"] for i, c in index.cats.items()} == labels, "Taksonomi COCO berbeda.")
    photo_ids = set()
    for image_id, image in index.imgs.items():
        row = photo_rows[image_id]
        expected_id = f"P{image_id:03d}"
        require(row["photo_id"] == expected_id, f"Identitas foto berbeda pada {image_id}")
        require(image["file_name"] == row["exported_file"], f"Nama gambar berbeda pada {expected_id}")
        name = Path(image["file_name"]).name
        require(re.match(rf"^{expected_id}(?=[_.])", name), f"Nama ekspor tidak cocok dengan {expected_id}")
        require((image["width"], image["height"]) == (int(row["width"]), int(row["height"])),
                f"Dimensi berbeda pada {expected_id}")
        photo_ids.add(expected_id)
    require(photo_ids == IMAGE_IDS and len(index.imgs) == len(IMAGE_IDS), "Cakupan P001–P040 berbeda.")
    for annotation in index.anns.values():
        note = f"anotasi #{annotation['id']}"
        require(annotation["image_id"] in index.imgs, f"Referensi gambar tidak valid pada {note}")
        require(annotation["category_id"] in index.cats, f"Referensi kelas tidak valid pada {note}")
        require(annotation["iscrowd"] == 0, f"Flag crowd tidak sesuai pada {note}")
        image = index.imgs[annotation["image_id"]]
        try:
            _, area, bbox, _ = encode(annotation["segmentation"], image["height"], image["width"])
        except (AssertionError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Geometri COCO tidak valid pada {note}") from exc
        require(area == annotation["area"] and bbox == annotation["bbox"], f"Area atau bbox berbeda pada {note}")
    return index


def compare_stats(all_class, hard3):
    covered = len({a["image_id"] for a in hard3["annotations"]})
    names = {c["id"]: c["name"] for c in hard3["categories"]}
    return {
        "images": len(all_class["images"]),
        "all_class_instances": len(all_class["annotations"]),
        "represented_classes": len({a["category_id"] for a in all_class["annotations"]}),
        "hard3_instances": len(hard3["annotations"]),
        "hard3_class_counts": dict(sorted(Counter(names[a["category_id"]] for a in hard3["annotations"]).items())),
        "hard3_positive_images": covered,
        "hard3_negative_images": len(hard3["images"]) - covered,
        "invalid_annotations": 0,
        "annotations_using_unknown_categories": 0,
    }


def apply_decisions(before, rows, flags):
    corrected = deepcopy(before.dataset)
    targets = {a["id"]: a for a in corrected["annotations"]}
    photo_images = defaultdict(list)
    log = []
    for photo_id, target_id, target_name, front_id, front_name in DECISIONS:
        target, foreground = before.loadAnns([target_id, front_id])
        image_id = int(photo_id[1:])
        image = before.imgs[image_id]
        require(rows[image_id]["photo_id"] == photo_id and image["file_name"] == rows[image_id]["exported_file"],
                f"Identitas {photo_id} berbeda dari QA awal.")
        require(target["image_id"] == foreground["image_id"] == image_id, f"Referensi {photo_id} berbeda.")
        require([before.cats[a["category_id"]]["name"] for a in (target, foreground)] == [target_name, front_name],
                f"Kelas berbeda pada {photo_id} #{target_id}/#{front_id}.")

        original_mask = before.annToMask(target).astype(bool)
        foreground_mask = before.annToMask(foreground).astype(bool)
        overlap = original_mask & foreground_mask
        removed = int(overlap.sum())
        require(removed > 0 and removed == flags[(photo_id, frozenset((target_id, front_id)))],
                f"Irisan berbeda dari hasil review pada {photo_id} #{target_id}.")
        corrected_mask = original_mask & ~foreground_mask
        require(corrected_mask.any(), f"Koreksi mengosongkan mask {photo_id} #{target_id}.")
        encoded = mask.encode(np.asfortranarray(corrected_mask, dtype=np.uint8))
        require(np.array_equal(mask.decode(encoded), corrected_mask), f"RLE round-trip gagal pada #{target_id}.")
        area_before = int(original_mask.sum())
        area_after = int(mask.area(encoded))
        remaining = int((corrected_mask & foreground_mask).sum())
        require(area_before == target["area"] == area_after + removed and remaining == 0,
                f"Invarian pengurangan mask gagal pada #{target_id}.")

        bbox = mask.toBbox(encoded).tolist()
        targets[target_id].update(segmentation={"size": encoded["size"], "counts": encoded["counts"].decode("ascii")},
                                  area=area_after, bbox=bbox)
        entry = dict(photo_id=photo_id, image_id=image_id, target_annotation_id=target_id,
                     target_class=target_name, target_category_id=target["category_id"],
                     occluding_annotation_id=front_id, occluding_class=front_name,
                     occluding_category_id=foreground["category_id"], area_before_px=area_before,
                     removed_overlap_px=removed, area_after_px=area_after,
                     area_reduction_percent=round(100 * removed / area_before, 6),
                     remaining_intersection_px=remaining, bbox_before=target["bbox"], bbox_after=bbox)
        log.append(entry)
        photo_images[photo_id].append((entry, original_mask, foreground_mask, corrected_mask, overlap))
    return corrected, log, photo_images


def overlay_panel(image, *layers):
    pixels = np.array(image, dtype=np.float32)
    for bitmap, color in layers:
        small = Image.fromarray(bitmap).resize(image.size, Image.Resampling.NEAREST)
        region = np.asarray(small, dtype=bool)
        pixels[region] = .50 * pixels[region] + .50 * np.array(color)
    return pixels.round().astype(np.uint8)


def render_comparison(path, reviews):
    with Image.open(path) as source:
        photo = source.convert("RGB")
        photo.thumbnail((520, 410), Image.Resampling.LANCZOS)
    cyan, orange, magenta = (33, 206, 231), (255, 161, 37), (236, 52, 162)
    fig, axes = plt.subplots(len(reviews), 4, squeeze=False, figsize=(14, 4.45 * len(reviews)),
                             layout="constrained", facecolor="white")
    try:
        for row, (entry, target, foreground, corrected, removed) in zip(axes, reviews):
            labels = ("Foto tanpa overlay", "Anotasi sebelum", "Anotasi sesudah", "Piksel yang dikeluarkan")
            panels = (photo, overlay_panel(photo, (target, cyan), (foreground, orange), (removed, magenta)),
                      overlay_panel(photo, (corrected, cyan), (foreground, orange)),
                      overlay_panel(photo, (removed, magenta)))
            for axis, title, panel in zip(row, labels, panels):
                axis.imshow(panel)
                axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
                axis.set_axis_off()
            target_text = f"{entry['target_class']} #{entry['target_annotation_id']}"
            foreground_text = f"{entry['occluding_class']} #{entry['occluding_annotation_id']}"
            row[0].text(0, -.055, f"Target sian: {target_text}", transform=row[0].transAxes, fontsize=9)
            row[1].text(0, -.055, f"Objek depan jingga: {foreground_text}", transform=row[1].transAxes, fontsize=9)
            row[2].text(0, -.055, "Irisan tersisa: 0 piksel", transform=row[2].transAxes, fontsize=9)
            number = f"{entry['removed_overlap_px']:,}".replace(",", ".")
            row[3].text(0, -.055, f"Magenta: {number} piksel", transform=row[3].transAxes, fontsize=9)
        fig.suptitle(f"{reviews[0][0]['photo_id']} — Koreksi batas mask", fontsize=15, fontweight="semibold")
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=130, pad_inches=.16, bbox_inches="tight")
            return buffer.getvalue()
    finally:
        plt.close(fig)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main():
    require(__debug__, "Jalankan validasi dengan Python tanpa opsi -O.")
    before_provenance_path = HERE / "dataset_provenance.json"
    provenance = load(before_provenance_path)
    mapping_path = HERE / "category_mapping.json"
    mapping = load(mapping_path)["classes"]
    all_labels = {c["all_class_coco"]: c["name"] for c in mapping}
    hard_labels = {c["hard3_coco"]: c["name"] for c in mapping if c["hard3_coco"] is not None}
    hard_map = {c["all_class_coco"]: c["hard3_coco"] for c in mapping if c["hard3_coco"] is not None}
    require(set(all_labels) == set(range(1, 75)), "Mapping All-Class tidak mencakup ID 1–74.")
    require(hard_labels == {1: "bed", 2: "book", 3: "cabinet-merged"}, "Mapping Hard-3 berbeda.")
    with (HERE / "image_mapping.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    photo_rows = {int(row["candidate_image_id"]): row for row in rows}
    require(len(photo_rows) == len(rows) == len(IMAGE_IDS), "ID foto pada mapping tidak unik.")
    sources = verify_sources(provenance, rows)
    flags_path = HERE / "occlusion_flags.json"
    flags = {(row["photo_id"], frozenset(ann["id"] for ann in row["annotations"])): row["overlap_pixels"]
             for row in load(flags_path)["flagged_pairs"]}

    before = coco_index(load(CANDIDATES / BEFORE_MASTER))
    prior_hard = load(CANDIDATES / BEFORE_HARD3)
    require(before.dataset["images"] == prior_hard["images"], "Dua kandidat asal memakai gambar berbeda.")
    require([dict(a, category_id=hard_map[a["category_id"]]) for a in before.dataset["annotations"]
             if a["category_id"] in hard_map] == prior_hard["annotations"], "Turunan Hard-3 asal berbeda dari master.")
    prior_labels = {c["id"]: c["name"] for c in prior_hard["categories"]}
    require(prior_labels == hard_labels, "Kategori Hard-3 asal tidak sesuai mapping.")
    table = before.anns[48]
    require(table["image_id"] == 1 and before.cats[table["category_id"]]["name"] == "table-merged"
            and isinstance(table["segmentation"], dict), "P001 #48 bukan RLE meja yang telah diperiksa.")

    corrected, log, reviews = apply_decisions(before, photo_rows, flags)
    corrected["info"] = {"description": "CPMK3 personal test corrected ground truth candidate",
                         "version": "v1-human-reviewed", "status": STATUS, "ground_truth_frozen": False}
    hard3 = dict(corrected, annotations=[dict(a, category_id=hard_map[a["category_id"]])
                                        for a in corrected["annotations"] if a["category_id"] in hard_map],
                 categories=prior_hard["categories"])
    all_index = validate(corrected, photo_rows, all_labels)
    hard_index = validate(hard3, photo_rows, hard_labels)

    by_id = {row["target_annotation_id"]: row for row in log}
    require(all_index.imgs == before.imgs == hard_index.imgs, "Identitas foto berubah.")
    require(all_index.anns.keys() == before.anns.keys(), "ID instance berubah.")
    for annotation_id, original in before.anns.items():
        updated = all_index.anns[annotation_id]
        expected_changes = {"segmentation", "area"} if annotation_id in by_id else set()
        if annotation_id in by_id and by_id[annotation_id]["bbox_before"] != by_id[annotation_id]["bbox_after"]:
            expected_changes.add("bbox")
        changed = {k for k in original.keys() | updated.keys() if original.get(k) != updated.get(k)}
        require(changed == expected_changes, f"Perubahan di luar keputusan annotator pada #{annotation_id}.")
    require(hard_index.anns[33]["segmentation"] == all_index.anns[33]["segmentation"],
            "Koreksi mask bed P003 berbeda pada kedua skenario.")

    stats_before = compare_stats(before.dataset, prior_hard)
    stats_after = compare_stats(corrected, hard3)
    expected = {
        "images": 40, "all_class_instances": 213, "represented_classes": 38, "hard3_instances": 33,
        "hard3_class_counts": {"bed": 10, "book": 11, "cabinet-merged": 12}, "hard3_positive_images": 25,
        "hard3_negative_images": 15, "invalid_annotations": 0, "annotations_using_unknown_categories": 0,
    }
    require(stats_before == stats_after == expected, f"Statistik menyimpang. Sebelum={stats_before}; sesudah={stats_after}")
    fields = [key for key in log[0] if key not in {"bbox_before", "bbox_after"}]
    output_csv = StringIO(newline="")
    writer = csv.DictWriter(output_csv, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(log)
    files = {
        CANDIDATES / AFTER_MASTER: json_bytes(corrected),
        CANDIDATES / AFTER_HARD3: json_bytes(hard3),
        OUTPUT / "correction_log.json": json_bytes({
            "review_date_wib": REVIEW_DATE, "approval_source": "keputusan eksplisit annotator",
            "operation": "target AND NOT occluding", "corrections": log,
            "P001_unchanged_annotation_id": 48, "P005_paper_label_confirmed": True,
        }),
        OUTPUT / "correction_summary.csv": output_csv.getvalue().encode("utf-8"),
        OUTPUT / "dataset_statistics.json": json_bytes({
            "before": stats_before, "after": stats_after, "corrected_masks_all_class": len(log),
            "corrected_masks_hard3": 1, "occluding_masks_unchanged": len({row["occluding_annotation_id"] for row in log}),
            "total_removed_pixels": sum(row["removed_overlap_px"] for row in log),
        }),
    }
    for photo_id, items in reviews.items():
        image = before.imgs[int(photo_id[1:])]
        files[OUTPUT / "overlays" / f"{photo_id}_before_after.png"] = render_comparison(RAW / image["file_name"], items)
    for path, expected_hash in sources.items():
        require(file_hash(path) == expected_hash, f"Sumber berubah selama finalisasi pada {path}")

    output_provenance = {
        "review_date_wib": REVIEW_DATE, "status": STATUS, "ground_truth_frozen": False,
        "source_candidate_provenance": before_provenance_path.relative_to(ROOT).as_posix(),
        "source_provenance_sha256": file_hash(before_provenance_path),
        "source_candidate_sha256": provenance["candidate_sha256"],
        "category_mapping_sha256": file_hash(mapping_path),
        "review_flags_sha256": file_hash(flags_path),
        "source_zip_sha256": provenance["zip_sha256"],
        "raw_annotation_sha256": provenance["raw_files_sha256"]["test/_annotations.coco.json"],
        "verified_image_hashes": {"original": len(rows), "exported": len(rows)},
        "evidence_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in sources.items()
                            if p.is_relative_to(ROOT / "evidence")},
        "evaluation_configs": {"all_class_master": (CANDIDATES / AFTER_MASTER).relative_to(ROOT).as_posix(),
                               "hard3": (CANDIDATES / AFTER_HARD3).relative_to(ROOT).as_posix(),
                               "shared_image_root": RAW.relative_to(ROOT).as_posix()},
        "reproduction_script_sha256": file_hash(HERE / Path(__file__).name),
        "pycocotools_version": version("pycocotools"),
        "artifacts_sha256": {p.relative_to(ROOT).as_posix(): sha(data) for p, data in files.items()},
    }
    files[OUTPUT / "dataset_provenance.json"] = json_bytes(output_provenance)
    for path, data in files.items():
        # Kandidat historis dan versi yang mungkin telah direview tidak boleh tertimpa.
        require(not path.exists() or path.read_bytes() == data, f"Versi keluaran sudah ada dan berbeda pada {path}")
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        require(file_hash(path) == sha(data), f"Verifikasi hasil simpan gagal pada {path}")
    print(json.dumps({"status": STATUS, **stats_after, "corrections": log}, ensure_ascii=False))


if __name__ == "__main__":
    main()
