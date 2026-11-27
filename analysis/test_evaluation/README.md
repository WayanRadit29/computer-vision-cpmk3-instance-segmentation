# Evaluasi terpadu original dan personal test set

Saya menggunakan Run02 (`20261126T152940Z-d1cca883`) sebagai sumber hasil akhir. Run ini mencakup enam checkpoint, dua test set, dan 12 kombinasi primer. Ketiga arsitektur memiliki checkpoint All-Class dan Hard-3. Evaluasi berlangsung pada GPU Tesla T4 tanggal 26 November 2026 WIB. [Notebook yang dijalankan](../../notebooks/CPMK3_Unified_Test_Evaluation_Recovery.ipynb) dan [ringkasan Run02](run02/execution_summary.json).

[Analisis kuantitatif Batch 3.1](quantitative_analysis.md) memuat peringkat model, perubahan AP dan recall pada spesialisasi Hard-3, pergeseran domain pada kategori bersama, serta hasil pengujian hipotesis.

[Analisis kegagalan Batch 3.2](failure_analysis/failure_analysis.md) menghubungkan lima foto personal dengan prediksi asli Run02, pencocokan per instance, dan dasar pemilihan eksperimen diagnostik.

[Eksperimen diagnostik Batch 3.3](diagnostic_experiment/diagnostic_experiment.md) mengukur kontribusi urutan confidence terhadap AP50 lemari pada dua checkpoint Mask R-CNN dengan penugasan tetap.

## Ground truth dan cakupan penilaian

| Cakupan | Foto | Instance | Kelas terwakili | Foto positif | Foto negatif |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original All-Class | 359 | 5.344 | 69 | 359 | 0 |
| Original Hard-3 | 359 | 530 | 3 | 137 | 222 |
| Personal All-Class | 40 | 213 | 38 | 40 | 0 |
| Personal Hard-3 | 40 | 33 | 3 | 25 | 15 |

Original test memakai gambar split test Panoptic Indoor Segmentation v5. Audit sebelumnya mencocokkan dimensi foto dan anotasi, memeriksa keanggotaan split, dan tidak menemukan kebutuhan koreksi rotasi pada split test. Hard-3 memproyeksikan COCO ID 2, 4, dan 8 ke ID 1, 2, dan 3 untuk `bed`, `book`, dan `cabinet-merged`. Jumlah instance tiap kelas berturut-turut 71, 349, dan 110 pada original test serta 10, 11, dan 12 pada personal test. Kedua file COCO personal memakai SHA-256 yang ditetapkan saat ground truth dibekukan. Mask R-CNN menggunakan ID 0 untuk background. YOLO mengindeks kelas mulai dari 0. Semua foto dengan dan tanpa objek target ikut dinilai. [Audit dataset v5](../dataset_audit/conversion_manifest.json), [manifest pembekuan](../personal_test_qa/finalization/ground_truth_freeze.json), dan [pemetaan foto dan kategori](run02/image_category_mapping.json).

| Scope | Pertanyaan yang dijawab | Hasil |
| --- | --- | --- |
| Primer All-Class dan Hard-3 | Bagaimana hasil setiap checkpoint pada cakupan kelas pelatihannya? | [Metrik primer](run02/tables/primary_metrics.csv), [AP per kelas](run02/tables/per_class_metrics.csv), dan [grafik](run02/figures/primary.png) |
| Shared Hard-3 | Bagaimana kedua skenario pelatihan dibandingkan pada tiga kelas, foto, dan anotasi yang sama? | [Perbandingan model](run02/tables/shared_hard3_comparison.csv), [perbandingan per kelas](run02/tables/shared_hard3_per_class.csv), dan [grafik](run02/figures/shared_hard3.png) |
| Shared represented | Bagaimana membandingkan original dan personal test dengan cakupan kategori yang sama? Scope ini memakai 37 kelas All-Class, atau tiga kelas Hard-3, dengan ground truth non-crowd di kedua test set. | [Metrik cakupan bersama](run02/tables/shared_represented_metrics.csv), [dukungan ground truth](run02/tables/ground_truth_support.csv), dan [grafik](run02/figures/shared_represented.png) |

## Protokol dan verifikasi

COCOeval menghitung Mask AP pada ambang IoU 0,50 hingga 0,95, AP50, AP75, dan AR100. Metrik bounding box tersedia sebagai hasil sekunder. CSV menyimpan AP dan AR dalam skala 0–1. Grafik memakai persentase dan kolom selisih memakai poin persentase. AP yang tak terdefinisi berupa `null` di JSON dan sel kosong di CSV. Nilai nol berarti AP terdefinisi dan bernilai nol. Proyeksi shared Hard-3 dilakukan setelah NMS dan batas 100 deteksi, sehingga prediksi kelas di luar Hard-3 dapat memengaruhi daftar deteksi All-Class sebelum proyeksi.

Inferensi memakai FP32, confidence minimum 0,001, dan maksimum 100 deteksi per foto. YOLO memproses satu gambar per batch pada kanvas letterbox 640 × 640, dengan retina masks dan NMS IoU 0,70. Run02 memanggil `rect=False` untuk mempertahankan kanvas persegi efektif Run01. Field `rect=True` pada protokol yang diekspor berasal dari konfigurasi sebelumnya. Mask R-CNN memakai sisi minimum 640, sisi maksimum 960, NMS IoU 0,50, dan threshold mask 0,50. Arsitekturnya direkonstruksi dengan FrozenBatchNorm2d sesuai training, dan bobot dimuat secara strict ke 75 atau 4 keluaran termasuk background. Kedua jalur memproses mask per instance dan menyimpan COCO RLE pada koordinat foto asal. [Protokol runtime](run02/evaluation_protocol.json) dan [manifest checkpoint](run02/checkpoint_manifest.json).

Gate GPU memeriksa dua kontrol letterbox, encoding native dan serial pada 806 mask yang berasal dari prediksi yang sama, fixture sintetis 100 mask per keluarga, serta inferensi resolusi tinggi pada keenam checkpoint. Dua mask pada kontrol lintas jalur forward YOLO11n All-Class masing-masing berbeda satu piksel. Run02 menghitung seluruh hasil dari prediksi baru dengan implementasi hemat memori. Fingerprint runtime yang diekspor mendahului revisi final Cell 11. SHA-256 file notebook dan sumber Cell 11 dicatat terpisah untuk mengenali implementasi yang dijalankan. [Gate GPU](run02/gpu_equivalence_report.json), [provenance](run02/integration.json), dan [audit Run01](run01_audit/README.md).

Implementasi validation Hard-3 sebelumnya memakai factory Mask R-CNN `weights=None` tanpa merekonstruksi normalisasi backbone secara eksplisit. Run02 merekonstruksi normalisasi dari arsitektur training sebelum memuat checkpoint. Perbedaan ini relevan jika hasil validation dan test dibandingkan dalam analisis yang sama. [Rekonstruksi notebook validation Hard-3](../../notebooks/CPMK3_Hard3_Final_Reconstructed.ipynb).

## Reproduksi di Kaggle

Gunakan notebook Run02 dan GPU Tesla T4. Cell **00 Install pinned evaluation packages** mencatat Ultralytics 8.4.216 dan pycocotools 2.0.10. Versi Torch, Torchvision, dan CUDA yang digunakan tercantum dalam [lingkungan Run02](run02/environment.json). Pasang input sesuai *dataset manifest* dan *checkpoint manifest*. Tabel berikut memuat nama aset pada run tersebut. Akses ke aset privat mengikuti izin Kaggle. [Manifest dataset](run02/dataset_manifest.json) dan [manifest checkpoint](run02/checkpoint_manifest.json).

| Input | Aset Kaggle yang tercatat | Cadangan lokal |
| --- | --- | --- |
| Original test | `panoptic-indoor-segmentation-v5-coco-format` | `data/original_v5/raw/panoptic-indoor-segmentation.v5i.coco-segmentation.zip` |
| Personal test beku | `cpmk3-personal-test-frozen` | `data/kaggle_inputs/CPMK3_Personal_Test_Frozen.zip` |
| Tiga checkpoint All-Class | `cpmk3-yolo11n-seg`, `cpmk3-yolov8n-seg`, dan `cpmk3-mask-rcnn`, variasi PyTorch `all-classes/1` | Subfolder `weights/` dari eksperimen masing-masing |
| Tiga checkpoint Hard-3 | `hard-3-best-checkpoints` | `data/kaggle_inputs/CPMK3_Hard3_Best_Checkpoints.zip` |

1. Di cell **02 Configuration**, set `MODE = "PREFLIGHT"`, `APPROVE_FULL_RECOVERY = False`, `RESUME_INPUTS = []`, dan `SELECT_PAIRS = []`. Jalankan semua cell dan periksa `PREFLIGHT PASS`. Discovery mencari file terkunci dari ZIP atau folder mount di `INPUT_ROOTS`.
2. Set `MODE = "STRESS_TEST"` lalu jalankan semua cell. Cell **11 Run smoke gate** memeriksa letterbox, encoding mask, integritas prediksi, dan alokasi GPU. Tinjau `GPU STRESS AND EQUIVALENCE PASS (NON-FINAL)` beserta laporan diagnostiknya.
3. Set `MODE = "FULL_EVALUATION"` dan `APPROVE_FULL_RECOVERY = True`. Pertahankan kedua daftar kosong untuk pengulangan fresh. Cell **12 Run full evaluation** menjalankan 12 pasangan setelah preflight dan uji GPU lolos. Unduh ZIP `CPMK3_Unified_Test_Evaluation_Recovery_Complete.zip` saat status `COMPLETE`. Jika ada kegagalan, simpan ZIP Partial beserta diagnostiknya.

## Menelusuri hasil dan prediksi

Tabel dan grafik berada di `analysis/test_evaluation/run02/`. [Manifest status](run02/evaluation_status_manifest.json) mencatat fingerprint, identitas input, dan SHA-256 tiap kombinasi. [Inventaris artefak](run02/artifact_hashes.json) mencakup file di dalam ZIP. [Tabel durasi](run02/tables/timings.csv) memisahkan waktu pipeline, pemuatan model, serialisasi, dan COCOeval. Pipeline meliputi sebagian pemrosesan gambar dan mask. Kolom ini mencatat cakupan waktu eksekusi penelitian. Perbandingan latensi model memerlukan pengukuran terstandar.

Seluruh bbox dan mask tersimpan sebagai `predictions_bbox.json.gz` dan `predictions_segm.json.gz` di `results/combinations/<model>__<domain>/` di dalam arsip Run02. Gunakan awalan anggota arsip yang tercatat di provenance. Setiap hasil dapat dicocokkan dengan foto melalui ID gambar dan kategori. Untuk personal test, pemetaan menghubungkan ID gambar ke P001–P040, sedangkan daftar kondisi sulit menghubungkannya dengan jenis kondisi foto. [Indeks arsip dan checksum](run02/integration.json), [pemetaan foto dan kelas](run02/image_category_mapping.json), dan [bukti kondisi sulit](../personal_test_qa/difficult_conditions.md).

Untuk mereproduksi analisis dari prediksi, siapkan `CPMK3_Unified_Test_Evaluation_Run02_Complete.zip` di root repositori sesuai struktur anggota dan SHA-256 yang tercatat. Arsip disimpan di luar Git. Tabel, gambar, dan hasil diagnostik yang dapat dibaca dari repositori tersedia di [analisis kuantitatif](quantitative_analysis.md), [lima kasus kegagalan](failure_analysis/failure_analysis.md), dan [eksperimen D1](diagnostic_experiment/diagnostic_experiment.md#reproduksi).
