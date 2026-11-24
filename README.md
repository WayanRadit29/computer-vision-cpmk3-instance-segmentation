# Computer Vision CPMK 3: Segmentasi Instance

Proyek ini membandingkan model visi komputer yang mengenali objek di dalam ruangan dan menandai piksel yang membentuk setiap objek. Tugas ini disebut *segmentasi instance*. Jika ada dua kursi dalam satu gambar, model diharapkan menghasilkan mask tersendiri untuk setiap kursi.

## Data yang digunakan

Eksperimen memakai Panoptic Indoor Segmentation v5 dengan 74 kelas aktif. Dataset dibagi menjadi 397 gambar untuk melatih model, 166 gambar untuk memantau kinerjanya selama pelatihan, dan 359 gambar uji yang disisihkan untuk evaluasi akhir.

Anotasi objek berbentuk poligon, yaitu rangkaian titik yang mengikuti batas objek pada gambar. Setelah audit awal menemukan sejumlah koordinat di luar batas gambar, label YOLO dibangun ulang dari anotasi COCO. Pemeriksaan terhadap 11.845 anotasi hasil konversi tidak menemukan poligon di luar batas atau ID kelas yang tidak valid. Sembilan kelas tidak memiliki contoh dalam bagian pelatihan. [Catatan konversi](analysis/dataset_audit/conversion_manifest.json) dan [laporan pemeriksaan data](analysis/dataset_audit/qa_report.json)

## Hasil yang tersedia

Kedua model YOLO dilatih selama 60 epoch dan Mask R-CNN selama 20 epoch. Satu epoch adalah satu putaran melalui data latih. Tabel menunjukkan nilai **mask mAP50-95 tertinggi pada data validasi**. Metrik ini merangkum kualitas prediksi mask pada beberapa ambang kecocokan dengan anotasi.

| Model | Mask mAP50–95 tertinggi | Epoch |
| --- | ---: | ---: |
| YOLO11n-seg | 0,18474 | 45 |
| YOLOv8n-seg | 0,18264 | 48 |
| Mask R-CNN ResNet-50 FPN | 0,14838 | 17 |

Ketiga model menggunakan pembagian gambar validasi yang sama. Mask R-CNN dievaluasi dengan COCOeval, sementara kedua model YOLO memakai evaluator Ultralytics. Protokol evaluasi dan konfigurasi pelatihan yang berbeda membatasi kesimpulan tentang pengaruh arsitektur terhadap skor. Evaluasi pada 359 gambar uji, evaluasi foto pribadi, dan eksperimen tiga kelas tersulit belum dilakukan.

## Hasil lengkap dan notebook

| Model | Konfigurasi pelatihan | Riwayat per epoch | Ringkasan dan grafik |
| --- | --- | --- | --- |
| YOLO11n-seg | [args.yaml](experiments/yolo11n_all_classes/args.yaml) | [results.csv](experiments/yolo11n_all_classes/metrics/results.csv) | [Ringkasan](experiments/yolo11n_all_classes/metrics/summary.json), [grafik](experiments/yolo11n_all_classes/metrics/results.png) |
| YOLOv8n-seg | [args.yaml](experiments/yolov8n_all_classes/args.yaml) | [results.csv](experiments/yolov8n_all_classes/metrics/results.csv) | [Ringkasan](experiments/yolov8n_all_classes/metrics/summary.json), [grafik](experiments/yolov8n_all_classes/metrics/results.png) |
| Mask R-CNN | [config.json](experiments/maskrcnn_all_classes/config.json) | [results.csv](experiments/maskrcnn_all_classes/metrics/results.csv) | [Ringkasan](experiments/maskrcnn_all_classes/metrics/summary.json), [grafik](experiments/maskrcnn_all_classes/metrics/validation_metrics.png), [AP per kelas](experiments/maskrcnn_all_classes/metrics/per_class_val_ap.csv) |

Notebook Kaggle memuat penyiapan data, konfigurasi, dan keluaran pelatihan. [Notebook YOLO11n](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb), [notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb), dan [notebook Mask R-CNN](notebooks/cpmk3-mask-r-cnn-all-class-baseline.ipynb)

- Lihat contoh mask hasil konversi dan mask setelah gambar dikoreksi. [Inspeksi mask](analysis/dataset_audit/rebuilt_yolo_mask_inspection.png) dan [pemeriksaan orientasi](evidence/orientation_fixes_verification.png)
- Ikuti urutan eksperimen, temuan, dan keputusan teknis. [Logbook](logbook.md)
- Baca analisis komparatif baseline dan rencana pemilihan kelas tersulit. [Laporan penelitian](analysis/report.md)
- Lihat bagian yang dibantu AI dan cara keluarannya diperiksa. [Pernyataan penggunaan AI](ai_usage.md)
