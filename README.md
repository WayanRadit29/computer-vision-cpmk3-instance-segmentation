# Computer Vision CPMK 3: Segmentasi Instance

Proyek ini membandingkan model visi komputer yang mengenali objek di dalam ruangan dan menandai piksel yang membentuk setiap objek. Tugas ini disebut *segmentasi instance*. Jika ada dua kursi dalam satu gambar, model diharapkan menghasilkan mask tersendiri untuk setiap kursi.

## Mulai di sini

Penelitian membandingkan YOLO11n-seg, YOLOv8n-seg, dan Mask R-CNN ResNet-50 FPN dalam dua skenario. **All-Class** melatih model untuk 74 kelas. **Hard-3** melatih model khusus untuk `bed`, `book`, dan `cabinet-merged`, kemudian membandingkannya dengan All-Class pada tiga kelas yang sama.

1. **Periksa hasil utama.** Buka [evaluasi test set dan panduan reproduksi](analysis/test_evaluation/README.md) untuk hasil Run02. Hasil sebelumnya tersedia dalam [evaluasi validation All-Class](analysis/all_class_comparison/analysis.md) dan [perbandingan validation Hard-3](analysis/hard3_comparison/analysis.md), keduanya pada 166 gambar validation.
2. **Telusuri keputusan riset.** Buka [laporan sementara](analysis/report.md) untuk metodologi dan kurva pelatihan, serta [hipotesis sebelum Hard-3](logbook.md#hipotesis-hard-3) untuk melihat rencana yang diuji.
3. **Periksa bukti penilaian.** Mulai dari [finalisasi ground truth personal test set](analysis/personal_test_qa/finalization/report.md), [bukti anotasi Roboflow](evidence/annotation_screenshots/dataset_overview.png), atau [contoh hasil demo](demo_outputs/yolo11n_all_classes__005_jpg.rf.f2248d837b7b97b5870fc8796a0eefd8.png).

## Progres penelitian

- **Selesai** - pelatihan dan unified validation tiga model pada All-Class dan Hard-3, pemilihan kelas Hard-3, perbandingan hasil, evaluasi hipotesis pada validation, demo inference kepada dosen, pengumpulan 40 foto asli, anotasi di Roboflow, serta ekspor dan validasi struktural COCO.
- **Batch 1 selesai** - annotator menyetujui empat koreksi batas mask dan membekukan ground truth All-Class dan Hard-3 pada 26 November 2026 WIB. Kedua versi siap menjadi acuan evaluasi personal test pada Batch 2. [Laporan finalisasi](analysis/personal_test_qa/finalization/report.md) dan [manifest pembekuan](analysis/personal_test_qa/finalization/ground_truth_freeze.json)
- **Batch 2 selesai** - Run02 menilai keenam checkpoint dengan COCOeval pada 359 gambar original test dan 40 foto personal test. Setiap model dinilai pada cakupan kelas pelatihannya. Perbandingan spesialisasi memakai tiga kelas dan ground truth yang sama. [Metrik primer](analysis/test_evaluation/run02/tables/primary_metrics.csv) dan [notebook yang dijalankan](notebooks/CPMK3_Unified_Test_Evaluation_Recovery.ipynb)
- **Menunggu** - analisis pergeseran domain, analisis kegagalan pada lima foto, eksperimen tambahan untuk menguji penjelasan kegagalan, serta laporan penelitian final.

## Struktur repositori

| Lokasi | Isi dan fungsi |
| --- | --- |
| `analysis/dataset_audit/` | Audit dan koreksi dataset baseline. [Manifest konversi](analysis/dataset_audit/conversion_manifest.json), [QA anotasi](analysis/dataset_audit/qa_report.json), dan [contoh overlay](analysis/dataset_audit/rebuilt_yolo_mask_inspection.png). |
| `analysis/all_class_comparison/` | Protokol unified validation, skor semua model, AP per kelas, dan dasar pemilihan Hard-3. [Konfigurasi evaluasi](analysis/all_class_comparison/evaluation_config.json) dan [keputusan Hard-3](analysis/all_class_comparison/hard3_decision.json). |
| `analysis/hard3_comparison/` | Perbandingan pada tiga kelas yang sama, hasil uji hipotesis pada validation, grafik, dan provenance run. [Tabel per kelas](analysis/hard3_comparison/per_class_comparison_with_relative.csv) dan [audit artefak](analysis/hard3_comparison/provenance/artifact_audit.json). |
| `analysis/personal_test_qa/` | Validasi ekspor Roboflow, distribusi kelas, pencocokan foto sumber, keputusan annotator, dan validasi mask sesudah koreksi. [Laporan finalisasi](analysis/personal_test_qa/finalization/report.md) dan [distribusi kelas](analysis/personal_test_qa/class_distribution.png). |
| `analysis/test_evaluation/` | Protokol, tabel, grafik, dan provenance Run02 untuk original dan personal test. Audit Run01 mendokumentasikan kegagalan memori yang memotivasi perubahan implementasi. [Panduan evaluasi](analysis/test_evaluation/README.md) dan [tabel perbandingan tiga kelas](analysis/test_evaluation/run02/tables/shared_hard3_comparison.csv). |
| `analysis/figures/` dan `analysis/report.md` | Grafik baseline serta laporan sementara yang menghubungkan data, metode, hasil, dan pekerjaan tersisa. [Kurva validasi baseline](analysis/figures/all_class_validation_mask_map.png) dan [laporan](analysis/report.md). |
| `experiments/` | Enam folder model per skenario. Konfigurasi berada di tingkat folder model, `metrics/` memuat log atau ringkasan, dan grafik berada di `metrics/` atau `visualizations/`. Lihat [indeks artefak per model](#artefak-per-model-dan-notebook). |
| `notebooks/` | Notebook pelatihan, recovery Mask R-CNN, dan evaluasi test set. [Notebook Hard-3 awal](notebooks/CPMK3_Hard3_All_Models.ipynb), [rekonstruksi run training final](notebooks/CPMK3_Hard3_Final_Reconstructed.ipynb), dan [notebook evaluasi Run02](notebooks/CPMK3_Unified_Test_Evaluation_Recovery.ipynb). |
| `data/custom_test/` | Foto asli, pedoman anotasi, dan ground truth yang telah dibekukan. [Pedoman](data/custom_test/annotation_guidelines.md), [master dan All-Class](data/custom_test/annotations/all_class_master_corrected.coco.json), dan [Hard-3](data/custom_test/annotations/hard3_corrected.coco.json). |
| `evidence/` | Bukti pendukung dataset dan eksperimen. [Screenshot proses anotasi](evidence/annotation_screenshots/annotation_process.png), [metadata EXIF](evidence/photo_metadata/exif_metadata.csv), [bukti GPU](evidence/gpu_usage/YOLO11n-seg_all_class.png), dan [diagnostik recovery](evidence/training_logs/hard3_maskrcnn/fp32_batch43_diagnostic.json). |
| `demo_inference.py` dan `demo_outputs/` | Script inference satu gambar dan dua hasil contoh dengan mask, kotak, dan confidence. [Script](demo_inference.py) dan [contoh 053](demo_outputs/yolo11n_all_classes__053_jpg.rf.002f0adcbefe28f38a23d14e639c1865.png). |
| `logbook.md` dan `ai_usage.md` | Riwayat eksperimen, hipotesis, serta jenis bantuan AI dan cara hasilnya diperiksa. [Logbook](logbook.md) dan [pernyataan penggunaan AI](ai_usage.md). |

Dataset Panoptic asli dan hasil konversi berada di `data/original_v5/` dan `data/processed/` pada instalasi lokal. Bobot model dan arsip run juga disimpan di luar Git. Sumber data dan checksum bobot tersedia pada bagian [reproduksi checkpoint dan demo](#reproduksi-checkpoint-dan-demo).

## Data yang digunakan

Eksperimen memakai Panoptic Indoor Segmentation v5 dengan 74 kelas aktif. Dataset dibagi menjadi 397 gambar untuk melatih model, 166 gambar untuk memantau kinerjanya selama pelatihan, dan 359 gambar uji yang disisihkan untuk evaluasi akhir.

Anotasi objek berbentuk poligon, yaitu rangkaian titik yang mengikuti batas objek pada gambar. Setelah audit awal menemukan sejumlah koordinat di luar batas gambar, label YOLO dibangun ulang dari anotasi COCO. Pemeriksaan terhadap 11.845 anotasi hasil konversi tidak menemukan poligon di luar batas atau ID kelas yang tidak valid. Sembilan kelas tidak memiliki contoh dalam bagian pelatihan. [Catatan konversi](analysis/dataset_audit/conversion_manifest.json) dan [laporan pemeriksaan data](analysis/dataset_audit/qa_report.json)

### Personal test set

Set uji pribadi terdiri dari 40 foto asli di `data/custom_test/images/CPMK3_Personal_Test_Set/`, termasuk 12 contoh kondisi sulit yang ditinjau secara visual. EXIF dipertahankan dan SHA-256 tiap sumber dicatat. [Daftar kondisi sulit](analysis/personal_test_qa/difficult_conditions.md), [manifest foto](evidence/photo_metadata/photo_manifest.csv), dan [metadata EXIF](evidence/photo_metadata/exif_metadata.csv).

Peneliti menganotasi seluruh foto di Roboflow dengan SAM-assisted masking, poligon manual, dan pixel brush, disertai penetapan kelas dan pemeriksaan oleh peneliti. [Pedoman anotasi](data/custom_test/annotation_guidelines.md), [proses anotasi](evidence/annotation_screenshots/annotation_process.png), dan [tampilan dataset Roboflow](evidence/annotation_screenshots/dataset_overview.png).

Ground truth All-Class memuat 213 instance dari 38 kelas. Hard-3 memakai 40 foto yang sama, termasuk 15 foto tanpa instance tiga kelas target untuk mengukur false positives. [Validasi dataset](analysis/personal_test_qa/finalization/report.md)

## Hasil validasi All-Class

Pelatihan all-class kedua YOLO selesai dalam 60 epoch dan Mask R-CNN dalam 20 epoch. Checkpoint ketiganya kemudian dinilai dengan COCOeval pada 166 gambar validation dan ground truth COCO yang sama. Mask mAP50–95 merangkum ketepatan mask pada beberapa ambang kecocokan dengan anotasi.

Tabel di bawah mengambil skor dari evaluasi bersama dengan confidence minimum 0,001 dan maksimum 100 deteksi per gambar. CSV pelatihan per epoch mencatat metrik dari masing-masing run, sehingga angkanya dapat berbeda. [Protokol dan skor sumber](analysis/all_class_comparison/analysis.md#protokol-evaluasi).

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

Rata-rata ketiga model meningkat pada validation. Mask R-CNN berada 0,157 poin persentase di atas YOLOv8n dan memperoleh AP `book` tertinggi, 10,707%. Kelas `bed` hanya memiliki enam instance validation. Perbandingan Mask R-CNN juga mencakup perubahan precision, laju belajar, dan gradient clipping. [Tabel per kelas](analysis/hard3_comparison/per_class_comparison_with_relative.csv), [grafik perbandingan](analysis/hard3_comparison/per_class_comparison.png), dan [analisis Hard-3](analysis/hard3_comparison/analysis.md).

## Evaluasi original dan personal test set

Run02 menggunakan checkpoint yang sama dengan eksperimen penelitian dan ground truth personal yang telah disetujui annotator. COCOeval menghitung Mask AP50–95, AP50, AP75, serta AR100 untuk tiap kombinasi. Tabel lintas domain menyediakan cakupan kelas bersama dengan ground truth yang dapat dinilai di kedua test set. Untuk mereproduksi evaluasi di Kaggle, pasang kedua dataset dan enam checkpoint sesuai manifest, lalu gunakan mode preflight dan uji GPU sebelum menjalankan evaluasi penuh. [Hasil per kelas](analysis/test_evaluation/run02/tables/per_class_metrics.csv) dan [protokol, input, serta panduan reproduksi](analysis/test_evaluation/README.md).

## Artefak per model dan notebook

| Model | Konfigurasi All-Class | Riwayat All-Class | Bukti Hard-3 |
| --- | --- | --- | --- |
| YOLO11n-seg | [args.yaml](experiments/yolo11n_all_classes/args.yaml) | [CSV](experiments/yolo11n_all_classes/metrics/results.csv), [ringkasan](experiments/yolo11n_all_classes/metrics/summary.json), [grafik](experiments/yolo11n_all_classes/metrics/results.png) | [Konfigurasi](experiments/yolo11n_hard3/config.json), [log](experiments/yolo11n_hard3/metrics/training_log.txt), [AP per kelas](experiments/yolo11n_hard3/metrics/unified_validation/per_class_ap.csv) |
| YOLOv8n-seg | [args.yaml](experiments/yolov8n_all_classes/args.yaml) | [CSV](experiments/yolov8n_all_classes/metrics/results.csv), [ringkasan](experiments/yolov8n_all_classes/metrics/summary.json), [grafik](experiments/yolov8n_all_classes/metrics/results.png) | [Konfigurasi](experiments/yolov8n_hard3/config.json), [log](experiments/yolov8n_hard3/metrics/training_log.txt), [AP per kelas](experiments/yolov8n_hard3/metrics/unified_validation/per_class_ap.csv) |
| Mask R-CNN | [config.json](experiments/maskrcnn_all_classes/config.json) | [CSV](experiments/maskrcnn_all_classes/metrics/results.csv), [ringkasan](experiments/maskrcnn_all_classes/metrics/summary.json), [grafik](experiments/maskrcnn_all_classes/metrics/validation_metrics.png) | [Konfigurasi](experiments/maskrcnn_hard3/config.json), [log](experiments/maskrcnn_hard3/metrics/training_log.txt), [AP per kelas](experiments/maskrcnn_hard3/metrics/unified_validation/per_class_ap.csv) |

Notebook Kaggle memuat penyiapan data, konfigurasi, dan keluaran pelatihan. [Notebook YOLO11n](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb), [notebook YOLOv8n](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb), dan [notebook Mask R-CNN](notebooks/cpmk3-mask-r-cnn-all-class-baseline.ipynb)

Bukti evaluasi All-Class tersedia sebagai konfigurasi dan hasil ekspor. Notebook evaluasi All-Class yang dieksekusi tidak tersedia di repo. [Konfigurasi evaluasi](analysis/all_class_comparison/evaluation_config.json), [metrik per model](analysis/all_class_comparison/model_comparison.csv), dan [AP per kelas](analysis/all_class_comparison/per_class_ap_comparison.csv).

Notebook Hard-3 menyiapkan anotasi, melatih ketiga model secara berurutan, dan membandingkan AP pada tiga kelas yang sama di validation. Untuk menjalankannya di Kaggle, pasang versi dataset COCO yang digunakan pada baseline dan lampirkan `evaluation_config.json` serta `per_class_ap_comparison.csv` dari evaluasi all-class. Pengaturan GPU, direktori hasil, dan prosedur pemulihan tercantum di notebook. [Notebook pelatihan Hard-3](notebooks/CPMK3_Hard3_All_Models.ipynb).

Recovery V2 menyelesaikan pelatihan Mask R-CNN dalam FP32 dengan laju belajar 0,0005 dan clipping norma L2 5,0. Rekonstruksi notebook memuat cell rebuild ZIP dan dua cell pengganti yang dipakai pada run sukses. Bukti eksekusi berasal dari paket hasil Kaggle. [Notebook final yang direkonstruksi](notebooks/CPMK3_Hard3_Final_Reconstructed.ipynb), [rancangan recovery V2](notebooks/CPMK3_Hard3_Recovery_V2.ipynb), dan [status run](analysis/hard3_comparison/provenance/run_summary.json).

### Reproduksi checkpoint dan demo

Sumber data baseline yang dipakai notebook adalah dataset Kaggle `wayanradityaputra/panoptic-indoor-segmentation-v5-coco-format`. Cara membangun ulang label dan koreksi orientasi tercatat di [manifest konversi](analysis/dataset_audit/conversion_manifest.json) dan [laporan metodologi](analysis/report.md#3-dataset-dan-persiapan-data).

Checkpoint disimpan terpisah dari Git. Tempatkan `best.pt` untuk YOLO atau `best.pth` untuk Mask R-CNN di subfolder `weights/` dalam direktori model terkait di `experiments/`. Referensi Kaggle dan SHA-256 untuk ketiga model All-Class tercatat di konfigurasi evaluasi. Bobot Hard-3 berasal dari arsip `CPMK3_Hard3_All_Models.zip` dan `CPMK3_Hard3_Recovery_V2_Complete.zip`. Hash `best` dan `last` Hard-3 tercatat di audit artefak. Akses evaluator ke aset Kaggle dan arsip tersebut belum diverifikasi. [Konfigurasi All-Class](analysis/all_class_comparison/evaluation_config.json) dan [audit Hard-3](analysis/hard3_comparison/provenance/artifact_audit.json).

Demo diuji dengan Python 3.12.4, PyTorch 2.11.0 CPU, Torchvision 0.26.0 CPU, Ultralytics 8.4.216, dan OpenCV 4.13.0.92. PyYAML membaca taksonomi dan Matplotlib menampilkan hasil. Siapkan checkpoint di `experiments/yolo11n_all_classes/weights/best.pt` dan taksonomi hasil konversi di `data/processed/all_classes/data.yaml`. Jalankan `python demo_inference.py <path-gambar>` dari root repositori. Demo menampilkan mask, kotak, dan confidence pada satu gambar dan menyimpan PNG di `demo_outputs/`. [Script inference](demo_inference.py), [contoh 005](demo_outputs/yolo11n_all_classes__005_jpg.rf.f2248d837b7b97b5870fc8796a0eefd8.png), dan [contoh 053](demo_outputs/yolo11n_all_classes__053_jpg.rf.002f0adcbefe28f38a23d14e639c1865.png).

