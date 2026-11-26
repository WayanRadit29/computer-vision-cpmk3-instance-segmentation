"""Validasi ekspor Roboflow dan buat kandidat COCO untuk review manusia."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PERSONAL = ROOT / "data/custom_test"
RAW = PERSONAL / "roboflow_raw"
CANDIDATES = PERSONAL / "annotations"
ORIGINAL = PERSONAL / "images/CPMK3_Personal_Test_Set"
BASE = ROOT / "data/original_v5/extracted"
IMAGE_IDS = {f"P{i:03d}" for i in range(1, 41)}
PALETTE = ["#56B4E9", "#E69F00", "#009E73", "#F0E442", "#CC79A7", "#E07A54", "#9A9CF3"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def table(path, rows, fields):
    with path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_raw():
    archive, = [p for p in ROOT.iterdir() if p.is_file() and "personal-test-set-v1" in p.name.casefold()
                and zipfile.is_zipfile(p)]
    hashes = {}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member in z.infolist():
            path = PurePosixPath(member.filename)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in member.filename
            assert all(":" not in part for part in path.parts)
            if member.is_dir():
                continue
            assert member.filename not in hashes
            data = z.read(member)
            dst = RAW.joinpath(*path.parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                assert dst.read_bytes() == data, dst
            else:
                dst.write_bytes(data)
            hashes[member.filename] = sha(data)
    return archive, hashes


def image_vector(im):
    return np.asarray(im.convert("RGB").resize((80, 80), Image.Resampling.LANCZOS), dtype=np.float32)


def numeric_ids(records):
    ids = [row["id"] for row in records]
    assert all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == len(ids)


def prepare_images(documents, hashes):
    with (ROOT / "evidence/photo_metadata/photo_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        originals = {Path(row["renamed_filename"]).stem: row for row in csv.DictReader(f)}
    with (ROOT / "evidence/photo_metadata/exif_metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        metadata = {Path(row["renamed_filename"]).stem: row for row in csv.DictReader(f)}
    assert set(originals) == set(metadata) == IMAGE_IDS
    vectors = {}
    original_sizes = {}
    for name, row in originals.items():
        path = ORIGINAL / row["renamed_filename"]
        digest = sha(path.read_bytes())
        assert digest == row["sha256"] == metadata[name]["sha256"]
        with Image.open(path) as source:
            upright = ImageOps.exif_transpose(source)
            original_sizes[name] = upright.size
            vectors[name] = image_vector(upright)
    images, paired, image_lookup = [], [], {}
    for split, document in documents:
        numeric_ids(document["images"])
        for image in document["images"]:
            name = image["file_name"]
            assert PurePosixPath(name).name == name
            ident, = re.findall(r"^P\d{3}(?=[_.])", name)
            assert ident in IMAGE_IDS and ident not in image_lookup.values()
            assert Path(image["extra"]["name"]).stem == ident
            rel = (split / name).as_posix()
            assert rel in hashes
            with Image.open(RAW / rel) as exported:
                exported.load()
                assert exported.getexif().get(274, 1) == 1
                dims = (image["width"], image["height"])
                assert dims == exported.size == original_sizes[ident]
                vector = image_vector(exported)
            distances = {k: float(np.mean(np.abs(vector - v))) for k, v in vectors.items()}
            closest = sorted(distances, key=distances.get)
            assert closest[0] == ident and distances[ident] < 3.0, (ident, closest[:2], distances[ident])
            new_id = int(ident[1:])
            images.append(dict(id=new_id, width=dims[0], height=dims[1], file_name=rel))
            image_lookup[split, image["id"]] = ident
            paired.append(dict(photo_id=ident, source_camera_filename=originals[ident]["original_filename"],
                               original_name=originals[ident]["renamed_filename"], original_sha256=originals[ident]["sha256"],
                               exported_file=rel, exported_sha256=hashes[rel], exported_image_id=image["id"],
                               candidate_image_id=new_id, width=dims[0], height=dims[1],
                               rgb_thumbnail_mae=round(distances[ident], 6),
                               next_closest_mae=round(distances[closest[1]], 6)))
    assert {r["photo_id"] for r in paired} == IMAGE_IDS and len(paired) == 40
    assert len({r["original_sha256"] for r in paired}) == len({r["exported_sha256"] for r in paired}) == 40
    assert {r["exported_file"] for r in paired} == {n for n in hashes if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    paired.sort(key=lambda x: x["photo_id"])
    table(HERE / "image_mapping.csv", paired, list(paired[0]))
    return sorted(images, key=lambda x: x["id"]), image_lookup, paired


def encode(segmentation, height, width):
    outside = 0.0
    if isinstance(segmentation, list):
        assert segmentation
        for ring in segmentation:
            p = np.asarray(ring, dtype=float)
            assert p.ndim == 1 and p.size >= 6 and p.size % 2 == 0 and np.isfinite(p).all()
            x, y = p[::2], p[1::2]
            outside = max(outside, -x.min(), -y.min(), x.max() - width, y.max() - height)
        assert outside <= 1, f"Polygon di luar gambar {outside:.2f} px"
        encoded = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    else:
        assert isinstance(segmentation, dict) and segmentation["size"] == [height, width]
        counts = segmentation["counts"]
        if isinstance(counts, list):
            assert all(type(c) is int and c >= 0 for c in counts) and sum(counts) == width * height
            encoded = coco_mask.frPyObjects(segmentation, height, width)
        else:
            assert isinstance(counts, str) and counts
            encoded = dict(size=segmentation["size"], counts=counts.encode("ascii"))
    bitmap = coco_mask.decode(encoded)
    assert bitmap.shape == (height, width)
    area = int(coco_mask.area(encoded))
    box = coco_mask.toBbox(encoded).tolist()
    assert area == int(bitmap.sum()) and area > 0 and box[2] > 0 and box[3] > 0
    assert min(box[:2]) >= 0 and box[0]+box[2] <= width and box[1]+box[3] <= height
    return encoded, area, box, outside


def render(images, by_image, names):
    overlays = HERE / "overlays"
    overlays.mkdir(exist_ok=True)
    sheets = [Image.new("RGB", (1440, 1080), "white") for _ in range(5)]
    for image in images:
        iid = image["id"]
        with Image.open(RAW / image["file_name"]) as photo:
            original = photo.convert("RGB")
        original.thumbnail((650, 650), Image.Resampling.LANCZOS)
        canvas = original.copy()
        positions = []
        for i, (ann, encoded) in enumerate(by_image[iid]):
            color = PALETTE[i % len(PALETTE)]
            mask = Image.fromarray(coco_mask.decode(encoded) * 255).resize(original.size, Image.Resampling.NEAREST)
            canvas.paste(color, (0, 0, *original.size), mask.point(lambda v: 95 if v else 0))
            x, y, w, h = ann["bbox"]
            sx, sy = original.width / image["width"], original.height / image["height"]
            positions.append(((x+w/2)*sx, (y+h/2)*sy, i+1, ann, color))
        legend_width = 290
        result = Image.new("RGB", (original.width + legend_width, max(original.height, 36+24*len(positions))), "#f5f6f8")
        result.paste(canvas, (0, 0))
        draw = ImageDraw.Draw(result)
        draw.text((original.width+10, 9), f"P{iid:03d}  |  {len(positions)} instance", fill="#111827")
        for x,y,i,ann,color in positions:
            draw.ellipse((x-8,y-8,x+8,y+8), fill="#111827", outline=color)
            draw.text((x,y), str(i), anchor="mm", fill="white")
            draw.rectangle((original.width+10, i*24+8,original.width+22,i*24+20), fill=color)
            draw.text((original.width+28,i*24+7), f"{i:02d}  {names[ann['category_id']]}  [#{ann['id']}]", fill="#111827")
        result.save(overlays / f"P{iid:03d}.jpg", quality=86)
        thumb = canvas.copy()
        thumb.thumbnail((354, 505), Image.Resampling.LANCZOS)
        slot = (iid-1)%8
        x, y = (slot%4)*360, (slot//4)*540
        sheet = sheets[(iid-1)//8]
        sheet.paste(thumb,(x+(360-thumb.width)//2,y+20))
        ImageDraw.Draw(sheet).text((x+7,y+5), f"P{iid:03d} / {len(positions)} instance", fill="black")
    for i, image in enumerate(sheets,1):
        image.save(overlays / f"contact_{i}.jpg", quality=83)


def main():
    archive, hashes = extract_raw()
    docs = [(p.relative_to(RAW).parent, load(p)) for p in sorted(RAW.rglob("_annotations.coco.json"))]
    assert docs
    base = load(BASE/"train/_annotations.coco.json")
    names = {c["id"]: c["name"] for c in base["categories"] if c["id"] != 0}
    config = yaml.safe_load((ROOT/"data/processed/all_classes/data.yaml").read_bytes())
    assert len(names) == 74 and config["names"] == [names[i] for i in range(1,75)]
    assert not any(a["category_id"] == 0 for a in base["annotations"])
    for split in ("valid", "test"):
        assert names == {c["id"]:c["name"] for c in load(BASE/split/"_annotations.coco.json")["categories"] if c["id"] != 0}
    ids = {v:k for k,v in names.items()}
    hard_manifest = load(ROOT/"analysis/hard3_comparison/provenance/dataset_manifest.json")
    hard = {row["name"]: row["hard3_coco"] for row in hard_manifest["class_mapping"]}
    assert hard == {"bed":1,"book":2,"cabinet-merged":3}
    assert all(row["original_coco"]==ids[row["name"]] and row["hard3_yolo"]==row["hard3_coco"]-1
               for row in hard_manifest["class_mapping"])
    assert load(ROOT/"experiments/maskrcnn_all_classes/config.json")["model_classes_including_background"] == 75
    assert load(ROOT/"experiments/maskrcnn_hard3/config.json")["classes_with_background"] == 4
    images, lookup, paired = prepare_images(docs, hashes)
    by_id = {row["id"]:row for row in images}
    rows, corrections, invalid, unknown, overlaps, occlusion_flags = [], [], [], [], [], []
    by_image = defaultdict(list)
    roboflow = {}
    for split, doc in docs:
        numeric_ids(doc["annotations"])
        numeric_ids(doc["categories"])
        category = {c["id"]:c["name"] for c in doc["categories"]}
        assert len(set(category.values())) == len(category)
        for id_, name in category.items():
            roboflow.setdefault(name, {})[split.as_posix()] = id_
            if name not in ids:
                count = sum(a["category_id"] == id_ for a in doc["annotations"])
                unknown.append(dict(split=str(split), id=id_, name=name, annotations=count,
                                    type="project_placeholder" if id_ == 0 and count == 0 else "unexpected"))
        for old in doc["annotations"]:
            iid = int(lookup[split, old["image_id"]][1:])
            image = by_id[iid]
            label = category[old["category_id"]]
            try:
                assert label in ids and old["iscrowd"] in (0, 1)
                rle, area, bbox, outside = encode(old["segmentation"], image["height"], image["width"])
            except (AssertionError, KeyError, TypeError, ValueError, OverflowError) as error:
                invalid.append(dict(annotation_id=old["id"], photo_id=f"P{iid:03d}", error=str(error)))
                continue
            new_id = old["id"] if len(docs) == 1 else len(rows)+1
            new = dict(old, id=new_id, image_id=iid, category_id=ids[label], area=area, bbox=bbox)
            rows.append(new)
            by_image[iid].append((new,rle))
            patch = {key:[old[key], new[key]] for key in ("id","image_id","category_id","bbox","area") if old[key] != new[key]}
            if patch or outside:
                corrections.append(dict(photo_id=f"P{iid:03d}", raw_split=str(split), annotation_id=old["id"],
                                        new_annotation_id=new_id, changes=patch, polygon_outside_max_px=outside))
    for iid, entries in by_image.items():
        for (a,ra),(b,rb) in itertools.combinations(entries,2):
            iou = float(coco_mask.iou([ra],[rb],[0])[0,0])
            same_class = a["category_id"] == b["category_id"]
            if same_class and iou >= .85:
                overlaps.append(dict(photo_id=f"P{iid:03d}", annotation_ids=[a["id"],b["id"]],
                                     name=names[a["category_id"]], mask_iou=iou))
            if not same_class and iou >= .05:
                intersection = int(coco_mask.area(coco_mask.merge([ra, rb], intersect=True)))
                shares = [intersection / row["area"] for row in (a, b)]
                if max(shares) >= .5:
                    occlusion_flags.append(dict(photo_id=f"P{iid:03d}", overlap_pixels=intersection,
                                               annotations=[dict(id=row["id"], label=names[row["category_id"]],
                                                                 overlap_fraction=round(share, 6))
                                                            for row, share in zip((a, b), shares)]))
    counts = Counter(a["category_id"] for a in rows)
    training_counts = Counter(a["category_id"] for a in base["annotations"])
    categories = [c for c in base["categories"] if c["id"] in names]
    dataset = dict(info={"description":"CPMK3 personal test candidate", "status":"REVIEW REQUIRED"},
                   licenses=[], images=images, annotations=rows, categories=categories)
    hard_rows = [dict(a,category_id=hard[names[a["category_id"]]]) for a in rows if names[a["category_id"]] in hard]
    hard_categories = [dict(id=i,name=k,supercategory="object") for k,i in sorted(hard.items(),key=lambda pair:pair[1])]
    output = {"all_class_master_candidate.coco.json":dataset,
              "hard3_candidate.coco.json":dict(dataset,annotations=hard_rows,categories=hard_categories)}
    for filename, data in output.items():
        save(CANDIDATES/filename, data)
        index = COCO(str(CANDIDATES/filename))
        assert len(index.imgs)==40 and len(index.anns)==len(data["annotations"])
        assert set(index.getImgIds())==set(by_id)
    mapping = [dict(name=name, roboflow_ids=roboflow.get(name,{}), all_class_coco=i, all_class_yolo=i-1,
                    all_class_maskrcnn=i, hard3_coco=hard.get(name), hard3_yolo=(hard[name]-1 if name in hard else None),
                    hard3_maskrcnn=hard.get(name), baseline_train_instances=training_counts[i])
               for i,name in sorted(names.items())]
    save(HERE/"category_mapping.json",dict(classes=mapping, unknown_export_categories=unknown, maskrcnn_background_id=0))
    per_class = [dict(name=names[i], coco_id=i, train_instances=training_counts[i], personal_instances=counts[i],
                      personal_images=len({a["image_id"] for a in rows if a["category_id"]==i}))
                 for i in sorted(names)]
    per_image = [dict(photo_id=f"P{i:03d}", image_id=i, instances=len(by_image[i]),
                      hard3_instances=sum(names[a["category_id"]] in hard for a,_ in by_image[i]))
                 for i in sorted(by_id)]
    table(HERE/"instances_per_class.csv",per_class,list(per_class[0]))
    table(HERE/"instances_per_image.csv",per_image,list(per_image[0]))
    save(HERE/"correction_log.json",dict(changes=corrections, invalid_annotations=invalid,
                                        possible_same_class_duplicates=overlaps,
                                        note="Mask dan label semantik pada anotasi valid dipertahankan."))
    save(HERE/"occlusion_flags.json", dict(min_pair_iou=.05, min_overlap_fraction_of_either_mask=.5,
                                          flagged_pairs=occlusion_flags))
    stats = dict(images=len(images), raw_annotations=sum(len(d["annotations"]) for _,d in docs), annotations=len(rows),
                 represented_classes=len(counts), invalid_annotations=invalid, unknown_categories=unknown,
                 hard3_counts={k:counts[ids[k]] for k in hard},
                 hard3_positive_images=sum(r["hard3_instances"]>0 for r in per_image),
                 hard3_negative_images=sum(r["hard3_instances"]==0 for r in per_image),
                 zero_annotation_images=[r["photo_id"] for r in per_image if not r["instances"]],
                 personal_classes_without_train=[names[i] for i in counts if not training_counts[i]],
                 absent_baseline_classes=[names[i] for i in sorted(names) if not counts[i]],
                 correction_counts=dict(Counter(k for r in corrections for k in r["changes"])),
                 max_polygon_outside_px=max((r["polygon_outside_max_px"] for r in corrections),default=0),
                 possible_same_class_duplicates=overlaps,
                 thumbnail_mae_max=max(r["rgb_thumbnail_mae"] for r in paired))
    save(HERE/"dataset_statistics.json",stats)
    with archive.open("rb") as f:
        archive_sha = hashlib.file_digest(f,"sha256").hexdigest()
    save(HERE/"dataset_provenance.json",dict(status="REVIEW REQUIRED", zip_name=archive.name,
         zip_sha256=archive_sha, raw_root=RAW.relative_to(ROOT).as_posix(), raw_files_sha256=hashes,
         configs={key:dict(annotation=(CANDIDATES/path).relative_to(ROOT).as_posix(), image_root=RAW.relative_to(ROOT).as_posix())
                  for key,path in zip(("all_class_master","hard3"),output)},
         candidate_sha256={n:sha((CANDIDATES/n).read_bytes()) for n in output}))
    plot = sorted([r for r in per_class if r["personal_instances"]],key=lambda r:r["personal_instances"])
    fig, ax = plt.subplots(figsize=(9,10),layout="constrained")
    y = np.arange(len(plot))
    ax.barh(y+.17,[r["personal_instances"] for r in plot],.34,label="Instance",color="#356f96")
    ax.barh(y-.17,[r["personal_images"] for r in plot],.34,label="Foto",color="#b76635")
    ax.set(yticks=y,yticklabels=[r["name"] for r in plot],xlabel="Jumlah",title="Distribusi kelas personal test set")
    ax.legend()
    ax.grid(axis="x",alpha=.25)
    fig.savefig(HERE/"class_distribution.png",dpi=145)
    plt.close(fig)
    render(images,by_image,names)
    print(json.dumps(stats,ensure_ascii=False))


if __name__ == "__main__":
    main()
