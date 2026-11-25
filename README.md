# Computer Vision CPMK 3: Segmentasi Instance

Proyek ini membandingkan model visi komputer yang mengenali objek di dalam ruangan dan menandai piksel yang membentuk setiap objek. Tugas ini disebut *segmentasi instance*. Jika ada dua kursi dalam satu gambar, model diharapkan menghasilkan mask tersendiri untuk setiap kursi.

## Data yang digunakan

Eksperimen memakai Panoptic Indoor Segmentation v5 dengan 74 kelas aktif. Dataset dibagi menjadi 397 gambar untuk melatih model, 166 gambar untuk memantau kinerjanya selama pelatihan, dan 359 gambar uji yang disisihkan untuk evaluasi akhir.

Anotasi objek berbentuk poligon, yaitu rangkaian titik yang mengikuti batas objek pada gambar. Setelah audit awal menemukan sejumlah koordinat di luar batas gambar, label YOLO dibangun ulang dari anotasi COCO. Pemeriksaan terhadap 11.845 anotasi hasil konversi tidak menemukan poligon di luar batas atau ID kelas yang tidak valid. Sembilan kelas tidak memiliki contoh dalam bagian pelatihan. [Catatan konversi](analysis/dataset_audit/conversion_manifest.json) dan [laporan pemeriksaan data](analysis/dataset_audit/qa_report.json)

### Personal test set

Set uji pribadi terdiri dari 40 foto indoor asli di `data/custom_test/images/CPMK3_Personal_Test_Set/`. EXIF dipertahankan, dan SHA-256 foto dicatat pada manifest. Ketentuan mencakup sedikitnya 10 foto kondisi sulit. Daftar terverifikasi untuk kondisi tersebut masih menunggu dokumentasi. [Manifest foto](evidence/photo_metadata/photo_manifest.csv) dan [metadata EXIF](evidence/photo_metadata/exif_metadata.csv).

Anotasi manual seluruh 40 foto selesai di Roboflow berdasarkan konfirmasi peneliti. Pedoman berbasis audit baseline menyelaraskan pemilihan label, pemisahan instance, dan batas mask yang terlihat. Ekspor ground truth serta validasi kelas, instance, dan kecocokan gambar masih menunggu. [Pedoman anotasi](data/custom_test/annotation_guidelines.md), [proses anotasi](evidence/annotation_screenshots/annotation_process.png), dan [tampilan dataset Roboflow](evidence/annotation_screenshots/dataset_overview.png).

Contoh kondisi sulit meliputi pencahayaan rendah pada foto buku dan tempat tidur, serta tempat tidur yang sebagian tertutup lemari. [P026](data/custom_test/images/CPMK3_Personal_Test_Set/P026.jpg), [P027](data/custom_test/images/CPMK3_Personal_Test_Set/P027.jpg), dan [P031](data/custom_test/images/CPMK3_Personal_Test_Set/P031.jpg).

## Progres penelitian

- **Selesai** — pelatihan dan unified validation tiga model pada All-Class dan Hard-3, pemilihan kelas Hard-3, perbandingan hasil, evaluasi hipotesis pada validation, demo inference kepada dosen, pengumpulan 40 foto asli, pedoman anotasi, dan anotasi manual seluruh foto di Roboflow.
- **Menunggu** — ekspor dan validasi personal test set, evaluasi original test dan personal test, analisis pergeseran domain, analisis kegagalan pada lima foto, eksperimen tambahan untuk menguji penjelasan kegagalan, serta laporan penelitian final.

## Hasil validasi All-Class

Pelatihan all-class kedua YOLO selesai dalam 60 epoch dan Mask R-CNN dalam 20 epoch. Checkpoint ketiganya kemudian dinilai dengan COCOeval pada 166 gambar validation dan ground truth COCO yang sama. Mask mAP50–95 merangkum ketepatan mask pada beberapa ambang kecocokan dengan anotasi.

| Model | Mask mAP50–95 (%) | Mask mAP50 (%) | Waktu pipeline (detik) |
| --- | ---: | ---: | ---: |
| YOLO11n-seg | 17,029 | 24,753 | 31,82 |
| YOLOv8n-seg | 16,704 | 25,922 | 19,78 |
| Mask R-CNN ResNet-50 FPN | 15,489 | 23,252 | 40,23 |

YOLO11n memperoleh mask mAP50–95 tertinggi pada validasi. YOLOv8n memperoleh mask mAP50 tertinggi dan durasi pipeline terpendek. Evaluasi memakai evaluator yang sama, dengan resize dan NMS sesuai pipeline masing-masing. Waktu yang dilaporkan adalah waktu pipeline untuk 166 gambar. [Analisis evaluasi bersama](analysis/all_class_comparison/analysis.md).

Hard-3 mencakup `cabinet-merged`, `book`, dan `bed`, tiga kelas dengan rata-rata Mask AP50–95 terendah di antara 39 kelas yang memenuhi syarat. Eksperimen membandingkan pelatihan all-class dan Hard-3 pada tiga kelas yang sama dalam setiap arsitektur. Validation set dipakai untuk pemantauan, sementara original test dan personal test disisihkan untuk penilaian akhir. [Analisis pemilihan](analysis/all_class_comparison/analysis.md#pemilihan-tiga-kelas-tersulit), [hipotesis](logbook.md#hipotesis-hard-3), dan [rancangan pengujian](logbook.md#rancangan-pengujian-hipotesis-hard-3).

## Hasil validasi Hard-3

Ketiga model menyelesaikan pelatihan dan unified COCOeval pada 166 gambar validation. Hasil YOLO11n dan YOLOv8n berasal dari run Hard-3 awal dan dipulihkan tanpa pelatihan ulang. Tabel berikut hanya meringkas tiga kelas target pada kedua kondisi.

| Model | All-Class, rata-rata Mask AP50–95 (%) | Hard-3, rata-rata Mask AP50–95 (%) | Selisih (poin persentase) |
| --- | ---: | ---: | ---: |
| YOLO11n-seg | 1,033 | 4,853 | +3,820 |
| YOLOv8n-seg | 3,672 | 5,581 | +1,909 |
| Mask R-CNN ResNet-50 FPN | 1,454 | 5,738 | +4,284 |

Rata-rata ketiga model meningkat. Mask R-CNN berada 0,157 poin persentase di atas YOLOv8n dan memperoleh AP `book` tertinggi, 10,707%. Kelas `bed` hanya memiliki enam instance validation. Perbandingan Mask R-CNN juga mencakup perubahan precision, laju belajar, dan gradient clipping. Original test dan personal test belum dievaluasi. [Tabel per kelas](analysis/hard3_comparison/per_class_comparison_with_relative.csv), [grafik perbandingan](analysis/hard3_comparison/per_class_comparison.png), dan [analisis Hard-3](analysis/hard3_comparison/analysis.md).

## Hasil lengkap dan notebook

| Model | Konfigurasi pelatihan | Riwayat per epoch | Ringkasan dan grafik |
| --- | --- | --- | --- |
| YOLO11n-seg | [args.yaml](experiments/yolo11n_all_classes/args.yaml) | [results.csv](experiments/yolo11n_all_classes/metrics/results.csv) | [Ringkasan](experiments/yolo11n_all_classes/metrics/summary.json), [grafik](experiments/yolo11n_all_classes/metrics/results.png) |
| YOLOv8n-seg | [args.yaml](experiments/yolov8n_all_classes/args.yaml) | [results.csv](experiments/yolov8n_all_classes/metrics/results.csv) | [Ringkasan](experiments/yolov8n_all_classes/metrics/summary.json), [grafik](experiments/yolov8n_all_classes/metrics/results.png) |
| Mask R-CNN | [config.json](experiments/maskrcnn_all_classes/config.json) | [results.csv](experiments/maskrcnn_all_classes/metrics/results.csv) | [Ringkasan](experiments/maskrcnn_all_classes/metrics/summary.json), [grafik](experiments/maskrcnn_all_classes/metrics/validation_metrics.png), [AP per kelas](experiments/maskrcnn_all_classes/metrics/per_class_val_ap.csv) |

Notebook Kaggle memuat penyiapan data, konfigurasi, dan keluaran pelatihan. [Notebook YOLO11n](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb), [notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb), dan [notebook Mask R-CNN](notebooks/cpmk3-mask-r-cnn-all-class-baseline.ipynb)

Artefak evaluasi bersama berada di `analysis/all_class_comparison/` dan grafiknya di `analysis/figures/`. Untuk menelusuri protokol, mulai dari konfigurasi yang memuat versi pustaka, identitas checkpoint, dan aturan prediksi. [Konfigurasi evaluasi](analysis/all_class_comparison/evaluation_config.json), [metrik per model](analysis/all_class_comparison/model_comparison.csv), [AP per kelas](analysis/all_class_comparison/per_class_ap_comparison.csv).

Notebook Hard-3 menyiapkan anotasi, melatih ketiga model secara berurutan, dan membandingkan AP pada tiga kelas yang sama di validation. Untuk menjalankannya di Kaggle, pasang versi dataset COCO yang digunakan pada baseline dan lampirkan `evaluation_config.json` serta `per_class_ap_comparison.csv` dari evaluasi all-class. Pengaturan GPU, direktori hasil, dan prosedur pemulihan tercantum di notebook. [Notebook pelatihan Hard-3](notebooks/CPMK3_Hard3_All_Models.ipynb).

Recovery V2 menyelesaikan pelatihan Mask R-CNN dalam FP32 dengan laju belajar 0,0005 dan clipping norma L2 5,0. Rekonstruksi notebook memuat cell rebuild ZIP dan dua cell pengganti yang dipakai pada run sukses. Bukti eksekusi berasal dari paket hasil Kaggle. [Notebook final yang direkonstruksi](notebooks/CPMK3_Hard3_Final_Reconstructed.ipynb), [rancangan recovery V2](notebooks/CPMK3_Hard3_Recovery_V2.ipynb), dan [status run](analysis/hard3_comparison/provenance/run_summary.json).

### Reproduksi checkpoint dan demo

Checkpoint disimpan terpisah dari Git. Tempatkan `best.pt` untuk YOLO atau `best.pth` untuk Mask R-CNN di subfolder `weights/` dalam direktori model terkait di `experiments/`. Referensi Kaggle dan SHA-256 untuk ketiga model All-Class tercatat di konfigurasi evaluasi. Bobot Hard-3 berasal dari arsip `CPMK3_Hard3_All_Models.zip` dan `CPMK3_Hard3_Recovery_V2_Complete.zip`. Hash `best` dan `last` Hard-3 tercatat di audit artefak. Akses evaluator ke aset Kaggle dan arsip tersebut belum diverifikasi. [Konfigurasi All-Class](analysis/all_class_comparison/evaluation_config.json) dan [audit Hard-3](analysis/hard3_comparison/provenance/artifact_audit.json).

Demo diuji dengan Python 3.12.4, PyTorch 2.11.0 CPU, Torchvision 0.26.0 CPU, Ultralytics 8.4.216, dan OpenCV 4.13.0.92. PyYAML membaca taksonomi dan Matplotlib menampilkan hasil. Siapkan checkpoint di `experiments/yolo11n_all_classes/weights/best.pt` dan taksonomi hasil konversi di `data/processed/all_classes/data.yaml`. Jalankan `python demo_inference.py <path-gambar>` dari root repositori. Demo menampilkan mask, kotak, dan confidence pada satu gambar dan menyimpan PNG di `demo_outputs/`. [Script inference](demo_inference.py), [contoh 005](demo_outputs/yolo11n_all_classes__005_jpg.rf.f2248d837b7b97b5870fc8796a0eefd8.png), dan [contoh 053](demo_outputs/yolo11n_all_classes__053_jpg.rf.002f0adcbefe28f38a23d14e639c1865.png).

- Lihat contoh mask hasil konversi dan mask setelah gambar dikoreksi. [Inspeksi mask](analysis/dataset_audit/rebuilt_yolo_mask_inspection.png) dan [pemeriksaan orientasi](evidence/orientation_fixes_verification.png)
- Ikuti urutan eksperimen, temuan, dan keputusan teknis. [Logbook](logbook.md)
- Baca hasil baseline, pemilihan kelas, dan penelitian lanjutan. [Laporan penelitian](analysis/report.md)
- Lihat bagian yang dibantu AI dan cara keluarannya diperiksa. [Pernyataan penggunaan AI](ai_usage.md)
