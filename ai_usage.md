# Pernyataan Penggunaan AI

ChatGPT membantu menyiapkan data, menyusun kode eksperimen YOLO, dan mendokumentasikan hasil ketiga model. Bagian di bawah menjelaskan batas bantuan tersebut dan satu kesalahan kode yang ditemukan.

## Bagian yang dibantu AI

| Tahap | Bentuk bantuan |
| --- | --- |
| Penyiapan data | Menelusuri format COCO dan YOLO, menyusun kode audit dan konversi anotasi, serta membantu menangani perbedaan orientasi gambar. |
| YOLO11n-seg | Menyusun dan menjelaskan konfigurasi notebook serta membantu memperbaiki kode pengambilan metrik setelah pelatihan. |
| YOLOv8n-seg | Membantu menyusun notebook, menyelaraskan konfigurasi dengan YOLO11n-seg, serta menyusun kode pelatihan dan ekspor metrik. [Notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb) |
| Mask R-CNN | Membantu membaca dan mendokumentasikan hasil. Penggunaan AI saat penyusunan notebook pelatihan belum terdokumentasi. [Notebook Mask R-CNN](notebooks/cpmk3-mask-r-cnn-all-class-baseline.ipynb) |
| Evaluasi bersama dan Hard-3 | Membantu menelaah rancangan evaluasi, koreksi ground truth, pemetaan COCO–YOLO, serta aturan pemeringkatan. Bantuan pada tahap analisis mencakup penataan artefak untuk reproduksi, penghitungan ulang mean dan SD AP, dan penjelasan keterbatasan. Riwayat bantuan AI untuk penyusunan notebook evaluasi dan rekonstruksi ground truth belum dapat dirinci dari artefak yang ada. [Analisis evaluasi](analysis/all_class_comparison/analysis.md) |
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

### Cuplikan prompt evaluasi bersama

> Hitung ulang mean AP dan ranking dari data aktual sebagai pemeriksaan konsistensi.
>
> Jangan menggunakan original test atau personal test untuk menentukan Hard-3.
>
> Laporkan tiga kandidat Hard-3 untuk mendapatkan konfirmasi pengguna sebelum menetapkannya sebagai keputusan final.

Mean, standar deviasi, dan urutan kandidat dapat ditelusuri ke AP per kelas ketiga model. Peneliti menetapkan keputusan akhir berdasarkan hasil COCOeval pada validation set. [AP per kelas](analysis/all_class_comparison/per_class_ap_comparison.csv) dan [keputusan final](analysis/all_class_comparison/hard3_decision.json).

### Hipotesis sebelum pelatihan Hard-3

AI membantu merumuskan H1, H2, dan H3 sebagai prediksi yang terukur, menelaah dasar empiris dan teknisnya, serta merancang pembandingan pada tiga kategori yang sama. AP baseline digunakan untuk memeriksa dasar H2 dan H3. Penjelasan arsitektur dipakai sebagai dugaan yang perlu diuji. Peneliti menetapkan tiga kelas final dan arah hipotesis. [Hipotesis](logbook.md#hipotesis-hard-3) dan [rancangan pengujian](logbook.md#rancangan-pengujian-hipotesis-hard-3).

### Implementasi notebook Hard-3

AI membantu mengadaptasi notebook baseline menjadi satu alur Kaggle untuk persiapan dataset, pelatihan berurutan, COCOeval pada validation, dan ekspor. Konfigurasi asal, jumlah anotasi target, serta pemetaan kategori diperiksa terhadap artefak penelitian. Pemeriksaan lokal mencakup struktur dan sintaks notebook. Perilaku pelatihan, pemuatan checkpoint hasil run baru, dan COCOeval pada GPU masih memerlukan verifikasi saat notebook dijalankan di Kaggle. [Notebook Hard-3](notebooks/CPMK3_Hard3_All_Models.ipynb).

### Diagnosis numerik dan recovery Hard-3

Berdasarkan traceback run Kaggle, AI membantu merancang percobaan Mask R-CNN dalam FP32 dan menambahkan pemeriksaan input serta diagnostik komponen loss. AI juga membantu memeriksa hubungan antara hash spesifikasi historis, checkpoint, dan metrik validation untuk pemulihan YOLO, serta menyesuaikan pencarian referensi all-class agar salinan identik tidak menimbulkan ambiguitas. Fase diagnosis belum menetapkan penyebab loss nonfinite. Efek FP32 dinilai setelah run recovery di Kaggle. [Logbook](logbook.md) dan [notebook recovery](notebooks/CPMK3_Hard3_Recovery.ipynb).
