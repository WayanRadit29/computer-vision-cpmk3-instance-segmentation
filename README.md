# Computer Vision CPMK 3: Segmentasi Instance

Proyek ini membandingkan model visi komputer yang mengenali objek di dalam ruangan dan menandai piksel yang membentuk setiap objek. Tugas ini disebut *segmentasi instance*. Jika ada dua kursi dalam satu gambar, model diharapkan menghasilkan mask tersendiri untuk setiap kursi.

## Data yang digunakan

Eksperimen memakai Panoptic Indoor Segmentation v5 dengan 74 kelas aktif. Dataset dibagi menjadi 397 gambar untuk melatih model, 166 gambar untuk memantau kinerjanya selama pelatihan, dan 359 gambar uji yang disisihkan untuk evaluasi akhir.

Anotasi objek berbentuk poligon, yaitu rangkaian titik yang mengikuti batas objek pada gambar. Setelah audit awal menemukan sejumlah koordinat di luar batas gambar, label YOLO dibangun ulang dari anotasi COCO. Pemeriksaan terhadap 11.845 anotasi hasil konversi tidak menemukan poligon di luar batas atau ID kelas yang tidak valid. Sembilan kelas tidak memiliki contoh dalam bagian pelatihan. [Catatan konversi](analysis/dataset_audit/conversion_manifest.json) dan [laporan pemeriksaan data](analysis/dataset_audit/qa_report.json)

## Hasil yang tersedia

Kedua model YOLO dilatih selama 60 epoch. Satu epoch adalah satu putaran melalui data latih. Tabel menunjukkan nilai **mask mAP50–95 tertinggi pada data validasi**. Metrik ini merangkum kualitas prediksi mask pada beberapa ambang kecocokan dengan anotasi. Nilai yang lebih tinggi menunjukkan hasil yang lebih baik menurut metrik tersebut.

| Model | Mask mAP50–95 tertinggi | Epoch |
| --- | ---: | ---: |
| YOLO11n-seg | 0,18474 | 45 |
| YOLOv8n-seg | 0,18264 | 48 |

Nomor epoch di tabel berasal dari riwayat validasi. Berkas model `best.pt` pada kedua eksperimen menyimpan `epoch: -1` dan `best_fitness: null`, sehingga epoch pemilihan checkpoint belum diketahui.

Hasil Mask R-CNN belum tersedia di proyek. Eksperimen pada tiga kelas tersulit masih direncanakan untuk ketiga model. Evaluasi pada 359 gambar uji asli belum dilakukan, dan hasil evaluasi foto pribadi belum tersedia. Perbandingan akhir masih menunggu tahap tersebut.

## Cara menemukan hasil dan bukti

| Model | Konfigurasi pelatihan | Riwayat per epoch | Ringkasan dan grafik |
| --- | --- | --- | --- |
| YOLO11n-seg | [args.yaml](experiments/yolo11n_all_classes/args.yaml) | [results.csv](experiments/yolo11n_all_classes/metrics/results.csv) | [Ringkasan](experiments/yolo11n_all_classes/metrics/summary.json), [grafik](experiments/yolo11n_all_classes/metrics/results.png) |
| YOLOv8n-seg | [args.yaml](experiments/yolov8n_all_classes/args.yaml) | [results.csv](experiments/yolov8n_all_classes/metrics/results.csv) | [Ringkasan](experiments/yolov8n_all_classes/metrics/summary.json), [grafik](experiments/yolov8n_all_classes/metrics/results.png) |

Folder setiap eksperimen memuat visualisasi. Berkas bobot model seperti `best.pt`, `last.pt`, dan checkpoint berkala tersimpan secara lokal di subfolder `weights/`. Notebook Kaggle kedua model memuat langkah penyiapan data, konfigurasi, dan keluaran pelatihan. [Notebook YOLO11n](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb) dan [notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb)

Log kedua notebook mencatat Tesla T4 dengan konfigurasi `device=0`. CSV mencatat durasi pelatihan 599,949 detik untuk YOLO11n dan 571,242 detik untuk YOLOv8n. Untuk durasi eksekusi notebook, tampilan Kaggle mencatat 15 menit 26 detik pada YOLO11n, sedangkan metadata Papermill di notebook YOLOv8n mencatat 723,431 detik atau sekitar 12 menit 3 detik. Tampilan Kaggle YOLO11n juga menunjukkan alokasi sesi `GPU T4 x2`. [Tangkapan layar Kaggle YOLO11n](evidence/gpu_usage/YOLO11n-seg_all_class.png) dan [notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb)

- Lihat contoh mask hasil konversi dan mask setelah gambar dikoreksi. [Inspeksi mask](analysis/dataset_audit/rebuilt_yolo_mask_inspection.png) dan [pemeriksaan orientasi](evidence/orientation_fixes_verification.png)
- Ikuti urutan eksperimen, temuan, dan keputusan teknis. [Logbook](logbook.md)
- Lihat bagian yang dibantu AI dan cara keluarannya diperiksa. [Pernyataan penggunaan AI](ai_usage.md)
