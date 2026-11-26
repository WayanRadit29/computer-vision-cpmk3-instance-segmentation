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
| Pedoman personal test set | Membantu mencocokkan nama dan ID kategori COCO dan YOLO, memeriksa overlay poligon baseline, mengenali label ambigu, serta menyusun usulan aturan label dan mask. [Pedoman anotasi](data/custom_test/annotation_guidelines.md) |
| Demo inference | Membantu menyusun script, memeriksa kecocokan checkpoint dengan hasil unified validation, serta menjalankan uji lokal pada contoh original test. Peneliti mendemonstrasikan hasil kepada dosen. [Script demo](demo_inference.py) dan [contoh visual](demo_outputs/yolo11n_all_classes__005_jpg.rf.f2248d837b7b97b5870fc8796a0eefd8.png). |
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

### Audit konvensi anotasi personal test set

AI membandingkan nama dan ID seluruh 74 kelas antara YAML dan COCO lokal. Inspeksi visual terbatas pada sampel dari 25 kelas. Overlay dapat dipakai mahasiswa untuk memeriksa contoh yang mendasari rekomendasi. Konvensi untuk kelas langka atau batas yang tidak konsisten, seperti `wall` dan pemisahan perlengkapan tidur, memiliki keyakinan terbatas. Mahasiswa memeriksa rekomendasi terhadap contoh baseline saat menemui kasus ambigu dan menggambar sendiri seluruh mask personal test set di Roboflow.

<details>
<summary>Prompt asli audit konvensi anotasi</summary>

```text
# CPMK 3: Baseline Annotation Convention Audit

Act as a Senior Computer Vision Dataset Engineer and Annotation Quality Auditor.

## Context

I am manually annotating a personal test set of 40 indoor photographs in Roboflow for CPMK 3.

My baseline is Panoptic Indoor Segmentation v5, containing 74 classes.

Original dataset reference:
[https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation)

The public Roboflow page may contain a different dataset version and class taxonomy. Therefore, prioritize the actual local dataset used for model training.

## Objective

Perform a focused, read-only audit of the baseline annotations to establish consistent class mapping and masking conventions for my personal test set.

## Tasks

1. Locate the actual baseline dataset and its `data.yaml` within the connected repository or accessible local dataset folders.
2. Verify the 74-class taxonomy and class IDs.
3. Inspect actual annotated images and segmentation polygons, prioritizing:

- paper vs paper-merged
- wall vs wall-other-merged
- wall-wood, wall-brick, wall-tile, wall-stone
- table-merged vs dining table
- box vs cardboard
- cabinet-merged vs shelf
- bottle vs cosmetic tubes and containers
- bed vs blanket vs pillow
- door-stuff
- window-other
- floor-other-merged vs floor-wood
- book
- keyboard vs laptop

4. For each ambiguous class, inspect approximately 3–5 representative annotated examples where available. Increase the sample only when evidence is inconsistent.
5. Determine observed conventions for:

- Class definitions
- Visible versus occluded regions
- Instance separation
- Merged regions
- Background and structural surfaces
- Small or partially visible objects
- Object components

6. Distinguish confirmed observations from assumptions. If annotations are inconsistent, explicitly report the inconsistency rather than inventing a rule.

## Deliverables

Produce a concise annotation reference table containing:

| Class | When to use | When not to use | Masking convention | Evidence | Confidence |

Include representative baseline image filenames and annotation overlays where useful.

Provide a compact decision guide that I can immediately apply while annotating in Roboflow.

Prioritize speed and practical usefulness.

## Constraints

- Do not train or run inference using any model.
- Do not modify baseline data or annotations.
- Do not automatically annotate my personal test set.
- Do not invent class definitions.
- Do not commit or push changes.
- If baseline data is unavailable, report the blocker immediately.
- Return results in chat.
```

</details>

### Stabilisasi numerik Mask R-CNN pada recovery V2

AI membantu menelaah diagnostik Kaggle dari percobaan dengan AMP dan FP32, memeriksa jalur loss RPN dan RoI, serta menerapkan konfigurasi stabilisasi yang ditetapkan peneliti. Bantuan mencakup probe batch tanpa instance dan batch positif, pencatatan gradien serta proposal, dan pemulihan artefak YOLO dari ZIP atau folder ekstraksi. Diagnostik yang tersedia menunjukkan lokasi munculnya loss NaN. Penetapan penyebab dan verifikasi kestabilan memerlukan hasil probe dan run Kaggle. [Catatan penelitian](logbook.md#l-kegagalan-recovery-fp32-dan-rancangan-stabilisasi-v2-24-november-2026-wib)

### Verifikasi runtime dan analisis Hard-3

Instruksi recovery awal dari AI mengasumsikan ZIP dapat dibaca langsung sebagai berkas dan belum mengantisipasi ekstraksi otomatis oleh Kaggle. Peneliti mempertahankan cell rebuild ZIP untuk input yang sudah terpasang. Rekomendasi FP32 pertama masih berakhir dengan NaN pada batch 43. Setelah memeriksa diagnostik, AI membantu menyusun dua cell pengganti yang memisahkan fingerprint kedua YOLO dari fingerprint Mask R-CNN V2 serta menambahkan probe, penurunan laju belajar, clipping, dan pencatatan diagnostik. Peneliti memilih penggantian dua cell dan menjalankan notebook di Kaggle. Pemeriksaan sintaks dan kompatibilitas recovery memberi bukti sebelum eksekusi, sedangkan kestabilan didukung oleh log Kaggle dan penyelesaian 20 epoch. Penyebab NaN tetap belum terisolasi. [Catatan eksperimen](logbook.md#m-recovery-v2-dan-hasil-validation-hard-3-25-november-2026-wib) dan [status Kaggle](analysis/hard3_comparison/provenance/run_summary.json).

AI membantu mencocokkan metadata evaluasi dan ground truth, memeriksa konsistensi angka, menyajikan grafik, dan menilai hasil terhadap hipotesis yang sudah dirumuskan. Kesimpulan didasarkan pada artefak validation dan dibatasi oleh jumlah instance, satu seed, serta perubahan konfigurasi Mask R-CNN. [Analisis hasil dan batas interpretasi](analysis/hard3_comparison/analysis.md).

### Anotasi personal test set dan audit ground truth

Peneliti menganotasi 40 foto di Roboflow dengan bantuan SAM pada sebagian mask, poligon manual, dan pixel brush. Peneliti menetapkan label kelas dan meninjau mask. Penyelesaian anotasi di Roboflow merupakan konfirmasi peneliti, sedangkan cakupan gambar dan jumlah anotasi diperiksa terhadap ekspor COCO. Persetujuan khusus dosen untuk SAM-assisted masking belum terdokumentasi.

AI membantu menyusun skrip validasi untuk mencocokkan foto sumber dan ekspor, memeriksa poligon serta RLE, dan menyelaraskan ID kategori menurut nama kelas baseline. COCO digunakan untuk memverifikasi mask dan menghitung ulang area serta kotak pembatas secara deterministik. AI juga membantu menyiapkan kandidat All-Class dan Hard-3 dari master yang sama dan meninjau overlay untuk merangkum usulan koreksi. Perubahan bentuk mask, kelas, dan pengelompokan instance memerlukan keputusan peneliti. Keterbatasan pemeriksaan visual mencakup objek kecil dan batas tipis pada skala overlay. [Skrip validasi](analysis/personal_test_qa/build_candidates.py), [hasil QA](analysis/personal_test_qa/report.md), dan [antrean keputusan](analysis/personal_test_qa/human_review.md)

### Koreksi batas mask berdasarkan keputusan annotator

AI membantu mengukur irisan mask dan menyajikan bukti visual. Peneliti sebagai annotator memutuskan empat pasangan mask yang perlu dikoreksi serta mengonfirmasi label `paper` pada P005. AI menerapkan operasi pengurangan biner pada pasangan yang disetujui dan memvalidasi RLE, area, kotak pembatas, dan referensi COCO. Perbandingan terhadap kandidat awal memeriksa bahwa setiap perubahan terbatas pada piksel irisan yang disetujui. Keputusan tentang label dan batas anotasi tetap berada pada annotator. Prediksi model tidak digunakan. [Laporan dan log keputusan](analysis/personal_test_qa/finalization/report.md)
