"""Demo lokal YOLO11n-seg All-Class pada satu gambar."""

import argparse
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "experiments/yolo11n_all_classes/weights/best.pt"
TAXONOMY = ROOT / "data/processed/all_classes/data.yaml"
DEFAULT_IMAGE = ROOT / "data/original_v5/extracted/test/001_jpg.rf.4e68c60b740dfb573ae743bc71df83e2.jpg"
OUTPUT_DIR = ROOT / "demo_outputs"
LOCAL_PACKAGES = ROOT / ".venv/demo-packages"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE, help="Path gambar input.")
    parser.add_argument("--no-show", action="store_true", help="Simpan PNG tanpa membuka jendela.")
    args = parser.parse_args()
    image = args.image.resolve()
    if not image.is_file():
        parser.error(f"Gambar tidak ditemukan: {image}")
    if not CHECKPOINT.is_file():
        parser.error(f"Checkpoint eksperimen tidak ditemukan: {CHECKPOINT}")

    if LOCAL_PACKAGES.is_dir():
        sys.path.insert(0, str(LOCAL_PACKAGES))
    try:
        import matplotlib
        import torch
        import yaml
        from ultralytics import YOLO
    except ImportError as exc:
        parser.exit(1, f"Dependency inference belum tersedia: {exc}\n")

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    expected_names = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))["names"]
    model = YOLO(str(CHECKPOINT))
    names = model.names
    if model.task != "segment" or len(names) != 74 or [names[i] for i in range(74)] != expected_names:
        parser.exit(1, "Checkpoint harus berupa model segmentasi dengan taksonomi 74 kelas Panoptic Indoor v5.\n")

    print("Model      : YOLO11n-seg All-Class (74 kelas)")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Device     : {device}")
    print(f"Input      : {image}")
    if device == "cuda:0":
        torch.cuda.synchronize()
    start = perf_counter()
    result = model.predict(
        source=str(image), imgsz=640, conf=0.5, iou=0.7,
        retina_masks=True, device=device, verbose=False,
    )[0]
    if device == "cuda:0":
        torch.cuda.synchronize()
    seconds = perf_counter() - start

    boxes = result.boxes
    count = len(boxes) if boxes is not None else 0
    print(f"Instance   : {count}")
    if count:
        if result.masks is None or len(result.masks) != count:
            raise RuntimeError("Jumlah instance mask berbeda dari jumlah bounding box.")
        for index, (cls, score) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist()), start=1):
            print(f"  {index:>2}. {names[int(cls)]}  {score:.3f}")
    print(f"Waktu predict: {seconds:.3f} detik (cold run; setup predictor, baca gambar, "
          "preprocessing, forward, dan postprocessing)")

    # Ultralytics menyimpan gambar OpenCV dalam BGR. Matplotlib memakai RGB.
    prediction = result.plot(boxes=True, labels=True, conf=True, masks=True, color_mode="instance")
    fig, (original_ax, prediction_ax) = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    original_ax.imshow(result.orig_img[..., ::-1])
    original_ax.set_title("Gambar input")
    prediction_ax.imshow(prediction[..., ::-1])
    prediction_ax.set_title(f"Prediksi YOLO11n-seg All-Class · {count} instance · conf ≥ 0,25")
    for axis in (original_ax, prediction_ax):
        axis.axis("off")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / f"yolo11n_all_classes__{image.stem}.png"
    fig.savefig(output, dpi=150, facecolor="white")
    print(f"Visualisasi: {output}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
