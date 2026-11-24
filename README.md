# Computer Vision CPMK 3: Segmentasi Instance

Proyek ini membandingkan model visi komputer yang mengenali objek di dalam ruangan dan menandai piksel yang membentuk setiap objek. Tugas ini disebut *segmentasi instance*. Jika ada dua kursi dalam satu gambar, model diharapkan menghasilkan mask tersendiri untuk setiap kursi.

## Data yang digunakan

Eksperimen memakai Panoptic Indoor Segmentation v5 dengan 74 kelas aktif. Dataset dibagi menjadi 397 gambar untuk melatih model, 166 gambar untuk memantau kinerjanya selama pelatihan, dan 359 gambar uji yang disisihkan untuk evaluasi akhir.

Anotasi objek berbentuk poligon, yaitu rangkaian titik yang mengikuti batas objek pada gambar. Setelah audit awal menemukan sejumlah koordinat di luar batas gambar, label YOLO dibangun ulang dari anotasi COCO. Pemeriksaan terhadap 11.845 anotasi hasil konversi tidak menemukan poligon di luar batas atau ID kelas yang tidak valid. Sembilan kelas tidak memiliki contoh dalam bagian pelatihan. [Catatan konversi](analysis/dataset_audit/conversion_manifest.json) dan [laporan pemeriksaan data](analysis/dataset_audit/qa_report.json)

## Hasil validasi

Pelatihan all-class kedua YOLO selesai dalam 60 epoch dan Mask R-CNN dalam 20 epoch. Checkpoint ketiganya kemudian dinilai dengan COCOeval pada 166 gambar validation dan ground truth COCO yang sama. Mask mAP50–95 merangkum ketepatan mask pada beberapa ambang kecocokan dengan anotasi.

| Model | Mask mAP50–95 (%) | Mask mAP50 (%) | Waktu pipeline (detik) |
| --- | ---: | ---: | ---: |
| YOLO11n-seg | 17,029 | 24,753 | 31,82 |
| YOLOv8n-seg | 16,704 | 25,922 | 19,78 |
| Mask R-CNN ResNet-50 FPN | 15,489 | 23,252 | 40,23 |

YOLO11n memperoleh mask mAP50–95 tertinggi pada validasi. YOLOv8n memperoleh mask mAP50 tertinggi dan durasi pipeline terpendek. Evaluasi memakai evaluator yang sama, dengan resize dan NMS sesuai pipeline masing-masing. Waktu yang dilaporkan adalah waktu pipeline untuk 166 gambar. [Analisis evaluasi bersama](analysis/all_class_comparison/analysis.md).

Hard-3 mencakup `cabinet-merged`, `book`, dan `bed`, tiga kelas dengan rata-rata Mask AP50–95 terendah di antara 39 kelas yang memenuhi syarat. Eksperimen membandingkan pelatihan all-class dan Hard-3 pada tiga kelas yang sama dalam setiap arsitektur. Validation set dipakai untuk pemantauan, sementara original test dan personal test disisihkan untuk penilaian akhir. [Analisis pemilihan](analysis/all_class_comparison/analysis.md#pemilihan-tiga-kelas-tersulit), [hipotesis](logbook.md#hipotesis-hard-3), dan [rancangan pengujian](logbook.md#rancangan-pengujian-hipotesis-hard-3).

## Hasil lengkap dan notebook

| Model | Konfigurasi pelatihan | Riwayat per epoch | Ringkasan dan grafik |
| --- | --- | --- | --- |
| YOLO11n-seg | [args.yaml](experiments/yolo11n_all_classes/args.yaml) | [results.csv](experiments/yolo11n_all_classes/metrics/results.csv) | [Ringkasan](experiments/yolo11n_all_classes/metrics/summary.json), [grafik](experiments/yolo11n_all_classes/metrics/results.png) |
| YOLOv8n-seg | [args.yaml](experiments/yolov8n_all_classes/args.yaml) | [results.csv](experiments/yolov8n_all_classes/metrics/results.csv) | [Ringkasan](experiments/yolov8n_all_classes/metrics/summary.json), [grafik](experiments/yolov8n_all_classes/metrics/results.png) |
| Mask R-CNN | [config.json](experiments/maskrcnn_all_classes/config.json) | [results.csv](experiments/maskrcnn_all_classes/metrics/results.csv) | [Ringkasan](experiments/maskrcnn_all_classes/metrics/summary.json), [grafik](experiments/maskrcnn_all_classes/metrics/validation_metrics.png), [AP per kelas](experiments/maskrcnn_all_classes/metrics/per_class_val_ap.csv) |

Notebook Kaggle memuat penyiapan data, konfigurasi, dan keluaran pelatihan. [Notebook YOLO11n](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb), [notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb), dan [notebook Mask R-CNN](notebooks/cpmk3-mask-r-cnn-all-class-baseline.ipynb)

Artefak evaluasi bersama berada di `analysis/all_class_comparison/` dan grafiknya di `analysis/figures/`. Untuk menelusuri protokol, mulai dari konfigurasi yang memuat versi pustaka, identitas checkpoint, dan aturan prediksi. [Konfigurasi evaluasi](analysis/all_class_comparison/evaluation_config.json), [metrik per model](analysis/all_class_comparison/model_comparison.csv), [AP per kelas](analysis/all_class_comparison/per_class_ap_comparison.csv).

Notebook Hard-3 menyiapkan anotasi, melatih ketiga model secara berurutan, dan membandingkan AP pada tiga kelas yang sama di validation. Untuk menjalankannya di Kaggle, pasang versi dataset COCO yang digunakan pada baseline dan lampirkan `evaluation_config.json` serta `per_class_ap_comparison.csv` dari evaluasi all-class. Pengaturan GPU, direktori hasil, dan prosedur pemulihan tercantum di notebook. [Notebook pelatihan Hard-3](notebooks/CPMK3_Hard3_All_Models.ipynb).

- Lihat contoh mask hasil konversi dan mask setelah gambar dikoreksi. [Inspeksi mask](analysis/dataset_audit/rebuilt_yolo_mask_inspection.png) dan [pemeriksaan orientasi](evidence/orientation_fixes_verification.png)
- Ikuti urutan eksperimen, temuan, dan keputusan teknis. [Logbook](logbook.md)
- Baca analisis komparatif baseline dan rencana pemilihan kelas tersulit. [Laporan penelitian](analysis/report.md)
- Lihat bagian yang dibantu AI dan cara keluarannya diperiksa. [Pernyataan penggunaan AI](ai_usage.md)
