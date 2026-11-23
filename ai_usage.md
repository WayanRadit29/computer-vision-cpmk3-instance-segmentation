# Pernyataan Penggunaan AI

ChatGPT membantu menyiapkan data, menyusun kode eksperimen YOLO, dan mendokumentasikan hasil ketiga model. Bagian di bawah menjelaskan batas bantuan tersebut dan satu kesalahan kode yang ditemukan.

## Bagian yang dibantu AI

| Tahap | Bentuk bantuan |
| --- | --- |
| Penyiapan data | Menelusuri format COCO dan YOLO, menyusun kode audit dan konversi anotasi, serta membantu menangani perbedaan orientasi gambar. |
| YOLO11n-seg | Menyusun dan menjelaskan konfigurasi notebook serta membantu memperbaiki kode pengambilan metrik setelah pelatihan. |
| YOLOv8n-seg | Membantu menyusun notebook, menyelaraskan konfigurasi dengan YOLO11n-seg, serta menyusun kode pelatihan dan ekspor metrik. [Notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb) |
| Mask R-CNN | Membantu membaca dan mendokumentasikan hasil. Penggunaan AI saat penyusunan notebook pelatihan belum terdokumentasi. [Notebook Mask R-CNN](notebooks/cpmk3-mask-r-cnn-all-class-baseline.ipynb) |
| Dokumentasi | Membantu merangkum metode, metrik, dan keterbatasan eksperimen berdasarkan konfigurasi dan hasil pelatihan. [Logbook](logbook.md) |

## Kesalahan kode yang ditemukan dan cara memperbaikinya

Kode yang disusun dengan bantuan AI pada Cell 8 notebook YOLO11n-seg mencari kolom `metrics/seg(mAP50-95)`. Setelah pelatihan mencapai 60 epoch, sel ekspor melaporkan `AssertionError: Mask mAP column not found.`

CSV hasil pelatihan memakai `metrics/mAP50-95(M)` dan `metrics/mAP50(M)`. Setelah kedua nama itu dipakai di sel ekspor, ringkasan validasi berhasil dihasilkan. Sel ekspor YOLOv8n-seg juga memeriksa keberadaan kolom yang dibutuhkan sebelum menghitung metrik. [CSV YOLO11n-seg](experiments/yolo11n_all_classes/metrics/results.csv), [notebook YOLO11n](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb), dan [notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb)

## Ringkasan permintaan kepada AI

Daftar berikut merangkum permintaan kepada AI selama eksperimen.

1. Periksa anotasi Panoptic Indoor Segmentation dan rancang konversi COCO ke label segmentasi instance YOLO.
2. Telusuri perbedaan ukuran dan orientasi gambar, lalu bantu memeriksa mask setelah koreksi.
3. Susun pelatihan YOLO11n-seg dengan pembagian data dan konfigurasi yang dapat ditelusuri.
4. Ambil metrik validasi dari hasil pelatihan dan jelaskan maknanya.
5. Susun dokumentasi CPMK 3 berdasarkan bukti, termasuk status eksperimen dan penggunaan AI.
