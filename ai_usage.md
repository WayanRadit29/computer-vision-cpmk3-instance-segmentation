# Pernyataan Penggunaan AI

Dokumen ini menjelaskan bagian proyek yang dibantu ChatGPT dan cara keluarannya diperiksa. Pelatihan YOLO11n-seg serta YOLOv8n-seg dijalankan di Kaggle dengan Ultralytics. Peneliti menyediakan arsip hasil YOLOv8n-seg untuk dimasukkan ke proyek.

## Bagian yang dibantu AI

| Tahap | Bentuk bantuan |
| --- | --- |
| Penyiapan data | Menelusuri format COCO dan YOLO, menyusun kode audit dan konversi anotasi, serta membantu menangani perbedaan orientasi gambar. |
| YOLO11n-seg | Menyusun dan menjelaskan konfigurasi notebook serta membantu memperbaiki kode pengambilan metrik setelah pelatihan. |
| YOLOv8n-seg | Berdasarkan riwayat proyek, membantu menyusun notebook, memilih konfigurasi yang sebanding dengan YOLO11n-seg, dan menyusun kode pelatihan serta ekspor. Notebook Kaggle sekarang tersedia. Teks prompt untuk tahap penyusunannya belum tersimpan di proyek. [Notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb) |
| Penataan dan dokumentasi hasil | ChatGPT Work membantu menata berkas, membandingkan dataset dalam arsip dengan salinan lokal, memeriksa parameter dan metrik, serta menyusun dokumentasi proyek. |

## Cara memeriksa hasil

Konfigurasi dan versi pustaka diperiksa dari `args.yaml` serta metadata checkpoint. Jumlah epoch, durasi pelatihan, dan metrik validasi diperiksa dari `results.csv` dan dicocokkan dengan `summary.json`. Untuk YOLOv8n, log notebook mengonfirmasi GPU Tesla T4, dan ringkasan yang dicetak oleh sel ekspor cocok dengan `summary.json` yang disimpan. Sel ekspor juga memeriksa keberadaan kolom metrik sebelum membacanya. Berkas hasil yang disalin diperiksa melalui ukuran dan SHA-256, yaitu sidik digital isi berkas. [Catatan pemeriksaan eksperimen](logbook.md) dan [notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb)

## Kesalahan kode yang ditemukan dan cara memperbaikinya

Kode yang disusun dengan bantuan AI pada Cell 8 notebook YOLO11n-seg mencari kolom `metrics/seg(mAP50-95)`. Setelah pelatihan mencapai 60 epoch, sel ekspor melaporkan `AssertionError: Mask mAP column not found.`

Pemeriksaan CSV menunjukkan bahwa Ultralytics memakai `metrics/mAP50-95(M)` dan `metrics/mAP50(M)`. Kode ekspor disesuaikan dengan nama kolom tersebut. Notebook tersimpan menampilkan ringkasan validasi setelah koreksi. Artefak YOLOv8n-seg yang tersedia tidak menunjukkan kejadian serupa. [CSV YOLO11n-seg](experiments/yolo11n_all_classes/metrics/results.csv) dan [notebook](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb)

## Ringkasan permintaan kepada AI

Transkrip lengkap percakapan awal tidak tersimpan di proyek. Daftar berikut merangkum permintaan yang terkait dengan pekerjaan ini.

1. Periksa anotasi Panoptic Indoor Segmentation dan rancang konversi COCO ke label segmentasi instance YOLO.
2. Telusuri perbedaan ukuran dan orientasi gambar, lalu bantu memeriksa mask setelah koreksi.
3. Susun pelatihan YOLO11n-seg dengan pembagian data dan konfigurasi yang dapat ditelusuri.
4. Ambil metrik dari hasil pelatihan, rangkum hasil, dan rapikan berkas eksperimen.
5. Susun dokumentasi CPMK 3 berdasarkan bukti, termasuk status eksperimen dan penggunaan AI.

Kutipan asli dari permintaan integrasi YOLOv8n-seg

> Tujuan pekerjaan ini adalah mengintegrasikan hasil YOLOv8n-seg all-class yang telah saya unduh ke repository lokal, memverifikasi artefaknya, memperbarui dokumentasi, dan mencatat milestone Git.
