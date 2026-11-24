# Logbook Eksperimen CPMK 3

Catatan ini menjelaskan keputusan penyiapan data, pengaturan pelatihan, dan hasil validasi ketiga model pada Panoptic Indoor Segmentation v5.

## A. Audit anotasi YOLO awal (20 November 2026)

Label YOLO menyimpan titik batas objek dalam koordinat relatif terhadap ukuran gambar. Audit awal melaporkan titik di luar batas gambar pada 3.527 dari 11.845 anotasi (29,8%). Pemeriksaan anotasi COCO terhadap dimensi gambar yang tercatat di JSON tidak menunjukkan masalah serupa pada data yang diperiksa. Temuan ini menjadi dasar keputusan untuk membangun ulang label YOLO dari anotasi COCO.

Pesan audit dari sel Kaggle tidak tersimpan sehingga angka temuan awal belum bisa diperiksa ulang secara independen. Laporan yang tersedia mencatat pemeriksaan setelah label dibangun ulang. [Laporan pemeriksaan](analysis/dataset_audit/qa_report.json)

## B. Koreksi orientasi gambar (20 November 2026)

Empat JPEG memiliki dimensi aktual 480 × 640 piksel, sedangkan metadata COCO mencatat 640 × 480 piksel. Gambar dan koordinat poligon dikoreksi bersama agar mask tetap mengikuti objek. Hasilnya diperiksa dengan menampilkan poligon di atas gambar yang dikoreksi.

| Bagian data | ID gambar                    | Rotasi                          |
| ----------- | ---------------------------- | ------------------------------- |
| Pelatihan   | `182169107`, `182169121` | Berlawanan arah jarum jam (CCW) |
| Validasi    | `182169109`                | Searah jarum jam (CW)           |
| Validasi    | `182169108`                | Berlawanan arah jarum jam (CCW) |

[Catatan konversi](analysis/dataset_audit/conversion_manifest.json) dan [gambar pemeriksaan orientasi](evidence/orientation_fixes_verification.png)

## C. Pembangunan ulang dataset (20 November 2026)

Anotasi COCO dikonversi ke format segmentasi instance YOLO dengan pemetaan `yolo_id = coco_category_id - 1`. Kategori kosong dengan ID 0 dikeluarkan. Hasilnya mencakup 74 kelas aktif.

| Bagian data     |        Gambar |    Anotasi objek |
| --------------- | ------------: | ---------------: |
| Pelatihan       |           397 |            4.479 |
| Validasi        |           166 |            2.022 |
| Uji             |           359 |            5.344 |
| **Total** | **922** | **11.845** |

Pemeriksaan seluruh anotasi hasil konversi menemukan nol poligon di luar batas dan nol ID kelas yang tidak valid. Pemeriksaan visual membantu memastikan posisi mask pada contoh gambar. Sembilan kelas aktif tidak memiliki contoh dalam bagian pelatihan, sehingga model tidak mendapat contoh berlabel untuk kelas tersebut selama pelatihan. [Laporan pemeriksaan](analysis/dataset_audit/qa_report.json) dan [contoh mask](analysis/dataset_audit/rebuilt_yolo_mask_inspection.png)

## D. Pelatihan YOLO11n-seg untuk semua kelas (21 November 2026)

YOLO11n-seg menjadi hasil acuan untuk eksperimen berikutnya. Pelatihan dimulai dari bobot pralatih `yolo11n-seg.pt` dan mencatat 60 epoch dengan `imgsz=640`, `batch=8`, `seed=42`, dan `device=0`. Validasi diaktifkan. Log notebook mengidentifikasi GPU yang dipakai sebagai Tesla T4. Tanggal berasal dari penanda waktu checkpoint (UTC). [Konfigurasi](experiments/yolo11n_all_classes/args.yaml) dan [notebook](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb)

Pada data validasi, mask mAP50–95 tertinggi mencapai 0,18474 pada epoch 45. Pada epoch yang sama, mask mAP50 mencapai 0,27586 dan box mAP50–95 mencapai 0,20845. Metrik mask menilai kecocokan area objek. Metrik box menilai kecocokan kotak yang membatasi objek. Epoch pemilihan `best.pt` belum diketahui karena metadata mencatat `epoch: -1` dan `best_fitness: null`. [Riwayat per epoch](experiments/yolo11n_all_classes/metrics/results.csv) dan [ringkasan](experiments/yolo11n_all_classes/metrics/summary.json)

Kolom `time` mencatat waktu kumulatif. Nilai terakhirnya adalah 599,949 detik atau sekitar 10 menit pelatihan. Secara terpisah, Kaggle mencatat total eksekusi notebook selama 15 menit 26 detik. Label `GPU T4 x2` menunjukkan alokasi sesi, sementara pelatihan dikonfigurasi pada `device=0`. [Tangkapan layar Kaggle](evidence/gpu_usage/YOLO11n-seg_all_class.png)

## E. Perbaikan ekspor metrik YOLO11n-seg (21 November 2026)

Setelah 60 epoch selesai, sel ekspor melaporkan `AssertionError: Mask mAP column not found.` Kode yang disusun dengan bantuan AI mencari `metrics/seg(mAP50-95)`. Pemeriksaan CSV menemukan nama kolom yang dipakai, yaitu `metrics/mAP50-95(M)` dan `metrics/mAP50(M)`. Sel ekspor disesuaikan dengan kedua nama itu, dan notebook yang tersimpan memperlihatkan ringkasan validasi setelah koreksi. Pelajarannya adalah memeriksa nama kolom dari hasil eksekusi sebelum menulis kode yang membacanya. [CSV pelatihan](experiments/yolo11n_all_classes/metrics/results.csv) dan [notebook](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb)

## Catatan retrospektif (23 November 2026 WIB)

Hipotesis untuk perbandingan tiga model pada semua kelas belum dicatat sebelum pelatihan pertama.

## F. Pelatihan YOLOv8n-seg untuk semua kelas (22 November 2026)

YOLOv8n-seg memakai pembagian data, ukuran gambar (`imgsz=640`), ukuran batch (`batch=8`), dan seed (42) yang sama dengan YOLO11n-seg. Pelatihan dimulai dari bobot pralatih `yolov8n-seg.pt` dan menggunakan Ultralytics 8.4.216. [Konfigurasi](experiments/yolov8n_all_classes/args.yaml) dan [ringkasan](experiments/yolov8n_all_classes/metrics/summary.json)

CSV mencatat epoch 1–60. Nilai terakhir pada kolom waktu kumulatif adalah 571,242 detik atau sekitar 9 menit 31 detik pelatihan. Log pelatihan di notebook juga mencatat 60 epoch selesai pada Tesla T4 (`CUDA:0`). Metadata Papermill mencatat 723,431 detik atau sekitar 12 menit 3 detik untuk eksekusi notebook. Semua angka berikut berasal dari epoch 48, saat mask mAP50–95 validasi mencapai puncaknya. [Riwayat pelatihan](experiments/yolov8n_all_classes/metrics/results.csv) dan [notebook Kaggle](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb)

| Metrik validasi pada epoch 48 |   Nilai |
| ----------------------------- | ------: |
| Mask mAP50–95                | 0,18264 |
| Mask mAP50                    | 0,27023 |
| Mask precision                | 0,49762 |
| Mask recall                   | 0,28596 |
| Box mAP50–95                 | 0,19857 |

Precision menggambarkan ketepatan prediksi yang dibuat. Recall menggambarkan cakupan objek beranotasi yang ditemukan. Metadata `best.pt` menyimpan `epoch: -1`, sehingga epoch pemilihan checkpoint belum diketahui.

Mask mAP50–95 validasi naik dari 0,09086 pada epoch 1 ke 0,18264 pada epoch 48, lalu turun ke 0,17153 pada epoch 60. Rata-rata nilai ini turun dari 0,17794 pada epoch 41–50 ke 0,16464 pada epoch 51–60. Dari epoch 1 ke 60, `train/seg_loss` turun dari 3,45219 ke 1,60909 dan `val/seg_loss` turun dari 2,93706 ke 2,09987. Penurunan mAP di akhir belum cukup untuk menyimpulkan overfitting. [Grafik pelatihan](experiments/yolov8n_all_classes/metrics/results.png)

## G. Pelatihan Mask R-CNN untuk semua kelas (23 November 2026 WIB)

Eksperimen ketiga memakai Mask R-CNN dengan ResNet-50 FPN dan bobot pralatih COCO dari Torchvision. Model memiliki 74 kelas objek dan satu kelas latar belakang. Pelatihan memakai Tesla T4 (`cuda:0`), `batch_size=2`, SGD dengan laju belajar awal 0,005, dan seed 42. Notebook mencatat PyTorch `2.11.0+cu130` dan Torchvision `0.26.0+cu130`. [Konfigurasi](experiments/maskrcnn_all_classes/config.json) dan [notebook Kaggle](notebooks/cpmk3-mask-r-cnn-all-class-baseline.ipynb)

Mask R-CNN memakai anotasi COCO dengan koreksi orientasi pada empat gambar yang dijelaskan di bagian B. Pembagian gambar tetap mengikuti dataset awal. Kategori kosong ID 0 dikeluarkan dari daftar kelas objek. [Catatan koreksi](experiments/maskrcnn_all_classes/corrected_annotations/repair_manifest.json) dan [pemeriksaan visual](experiments/maskrcnn_all_classes/visualizations/annotation_sanity_check.png)

Pelatihan menyelesaikan 20 epoch. Kolom waktu kumulatif pada CSV berakhir di 2.238,738 detik, sekitar 37 menit 19 detik untuk putaran pelatihan dan validasi. Dari kolom per epoch, total waktu yang diukur di bagian pelatihan sekitar 23 menit 18 detik, dan di bagian validasi sekitar 13 menit 48 detik. Metadata Papermill mencatat 2.342,766 detik atau sekitar 39 menit 3 detik untuk seluruh eksekusi notebook. [Riwayat pelatihan](experiments/maskrcnn_all_classes/metrics/results.csv) dan [notebook Kaggle](notebooks/cpmk3-mask-r-cnn-all-class-baseline.ipynb)

COCOeval menilai kotak dan mask pada 166 gambar validasi. Mask mAP50-95 meningkat dari 0,03887 pada epoch pertama ke puncak 0,14838 pada epoch 17, lalu berakhir di 0,14693. Notebook menyimpan `best.pth` saat mask mAP50-95 validasi meningkat. Tabel berikut berisi nilai pada epoch 17. [Ringkasan hasil](experiments/maskrcnn_all_classes/metrics/summary.json) dan [grafik validasi](experiments/maskrcnn_all_classes/metrics/validation_metrics.png)

| Metrik validasi pada epoch 17 | Nilai |
| --- | ---: |
| Mask mAP50-95 | 0,14838 |
| Mask mAP50 | 0,22246 |
| Box mAP50-95 | 0,15425 |
| Box mAP50 | 0,24088 |

Tabel AP per kelas berasal dari checkpoint terbaik. Sebanyak 16 dari 74 kelas tidak punya contoh pada data validasi dan nilai AP-nya kosong. Angka kosong perlu dibedakan dari AP nol saat memilih kelas tersulit. [AP validasi per kelas](experiments/maskrcnn_all_classes/metrics/per_class_val_ap.csv)

Ketiga hasil yang dilaporkan berasal dari data validasi. Evaluasi pada set uji asli dan foto pribadi, serta pelatihan khusus tiga kelas tersulit, belum dilakukan.

## H. Evaluasi bersama dan pemeringkatan kandidat Hard-3 (24 November 2026)

Tanggal mengikuti penanda file hasil evaluasi. Ketiga checkpoint `best` dinilai pada 166 gambar validation dengan ground truth COCO yang sama. Ground truth mempertahankan dua koreksi orientasi validation dari audit awal. COCOeval digunakan untuk box dan mask. Konfigurasi menetapkan confidence minimum 0,001, maksimum 100 deteksi per gambar, dan mask threshold 0,50. Resize dan NMS mengikuti pengaturan tiap keluarga model. Penilaian ini memakai bobot yang sudah dilatih. [Konfigurasi](analysis/all_class_comparison/evaluation_config.json).

Mask mAP50–95 tercatat 17,029% untuk YOLO11n, 16,704% untuk YOLOv8n, dan 15,489% untuk Mask R-CNN. YOLOv8n memiliki mask mAP50 tertinggi dan pipeline terpendek. Perbedaan pengaturan inference dan pemilihan checkpoint membatasi penafsiran selisih dari skor puncak pada log pelatihan. [Perbandingan model](analysis/all_class_comparison/model_comparison.csv).

Untuk menghindari pemeringkatan pada kelas yang terlalu jarang atau tidak bisa dinilai, kelas harus memiliki sedikitnya 10 instance train, 5 instance validation, dan AP dari semua model. Sebanyak 39 kelas memenuhi syarat. Mean mask AP50–95 terendah tercatat pada `cabinet-merged`, `book`, dan `bed`. Ketiganya masih kandidat. Tinjauan prediksi validation per kelas dan konfirmasi pilihan dilakukan sebelum pelatihan khusus. Original test dan personal test tidak digunakan untuk seleksi. [Metode dan interpretasi](analysis/all_class_comparison/analysis.md), [aturan seleksi](analysis/all_class_comparison/hard3_selection.json).

## I. Finalisasi kelas dan hipotesis Hard-3 (24 November 2026 WIB)

Hipotesis berikut dirumuskan setelah evaluasi baseline all-class dengan COCOeval dan sebelum pelatihan Hard-3. Hipotesis ini bersifat prospektif untuk eksperimen Hard-3. Catatan retrospektif tentang pelatihan baseline pertama tetap berlaku.

Peneliti menetapkan `cabinet-merged`, `book`, dan `bed` sebagai Hard-3. Ketiganya memiliki rata-rata Mask AP50–95 terendah di antara 39 kelas dengan minimal 10 instance train, 5 instance validation, dan AP yang tersedia dari ketiga model. Pemilihan hanya memakai data validation. Enam instance validation pada `bed` memenuhi ambang, tetapi membuat AP kelas tersebut peka terhadap perubahan beberapa prediksi. [Ranking COCOeval](analysis/all_class_comparison/hard3_ranking.csv) dan [keputusan final](analysis/all_class_comparison/hard3_decision.json).

### Hipotesis Hard-3

**H1. Pelatihan Hard-3 meningkatkan rata-rata Mask AP50–95 pada tiga kelas target untuk setiap arsitektur.** Dengan cakupan kelas yang lebih sempit, kompetisi antarkelas dalam klasifikasi dapat berkurang dan pembelajaran dapat lebih berfokus pada ciri kelas target. Jumlah contoh positif, distribusi data, dan kualitas anotasi tetap dapat membatasi peningkatan.

**H2. YOLOv8n-seg memperoleh rata-rata Mask AP50–95 Hard-3 tertinggi di antara tiga model.** Dasar empirisnya adalah rata-rata **3,672%** pada tiga kelas target dalam evaluasi all-class, dibandingkan **1,033%** untuk YOLO11n dan **1,454%** untuk Mask R-CNN. YOLOv8n juga mencatat AP tertinggi pada `cabinet-merged` dan `bed`. Prediksi satu tahap, fitur multiskala, dan mask berbasis prototipe dapat mendukung pembelajaran yang efisien. Karakteristik ini juga dimiliki YOLO11n dan belum menunjukkan keunggulan arsitektural YOLOv8n. [AP per kelas](analysis/all_class_comparison/per_class_ap_comparison.csv).

**H3. Mask R-CNN memperoleh Mask AP50–95 tertinggi pada kelas `book` setelah pelatihan Hard-3.** Pada baseline all-class, Mask R-CNN memperoleh **3,517%** pada `book`, dibandingkan **2,319%** untuk YOLOv8n dan **0,368%** untuk YOLO11n. Region proposal dan RoIAlign membantu model mengambil fitur dari wilayah kandidat objek. Mekanisme itu mungkin membantu ketika buku kecil, tipis, bertumpuk, atau tertutup sebagian. Hipotesis ini belum menunjukkan bahwa kondisi tersebut menjelaskan skor baseline. Resolusi mask dan kualitas proposal juga dapat membatasi hasil. [AP kelas `book`](analysis/all_class_comparison/per_class_ap_comparison.csv).

### Rancangan pengujian hipotesis Hard-3

Perubahan utama yang diuji adalah cakupan kelas pelatihan, dari 74 kelas menjadi tiga. Versi Hard-3 setiap arsitektur dimulai dari bobot pralatih yang sama dengan baseline pasangannya. Pertahankan pembagian gambar train, validation, dan test, konfigurasi serta anggaran pelatihan, dan protokol evaluasi sejauh memungkinkan. Pada pelatihan Hard-3, gunakan anotasi tiga kelas target dan pertahankan gambar tanpa instance target sebagai sampel negatif. Catat penyimpangan dari rancangan ini beserta alasannya.

Gunakan validation set untuk memantau pelatihan dan memilih checkpoint dengan aturan yang ditetapkan sebelum run. Setelah checkpoint dan pengaturan evaluasi dikunci, nilai seluruh model pada original test set dan personal test set dengan COCOeval, pemetaan kategori konsisten, dan pengaturan evaluasi yang sama. Untuk H1, hitung AP model all-class pada tiga kategori target dari prediksinya di set uji yang sama. mAP agregat seluruh 74 kelas memiliki cakupan berbeda dan tidak dipakai untuk menghitung perubahan.

| Hipotesis | Pembandingan pada setiap set uji | Kriteria dukungan |
| --- | --- | --- |
| H1 | Selisih rata-rata Mask AP50–95 tiga kelas antara versi Hard-3 dan versi all-class untuk setiap arsitektur. | Selisih Hard-3 dikurangi all-class positif pada setiap arsitektur. Laporkan hasil per arsitektur. |
| H2 | Rata-rata Mask AP50–95 tiga kelas dari ketiga model hasil pelatihan Hard-3. | Rata-rata YOLOv8n lebih tinggi daripada kedua model lain. |
| H3 | Mask AP50–95 khusus `book` dari ketiga model hasil pelatihan Hard-3. | AP Mask R-CNN lebih tinggi daripada kedua model YOLO. |

Laporkan nilai AP per kelas dan selisih dalam poin persentase untuk original test dan personal test secara terpisah. Jika suatu kelas tidak memiliki anotasi uji, tandai AP sebagai N/A. Bila AP tiga kelas tidak lengkap, laporkan keterbatasannya dan tunda penilaian H1 atau H2 pada set tersebut. Nilai seri tidak mendukung prediksi keunggulan. Kriteria ini membandingkan hasil secara deskriptif. Klaim keunggulan yang stabil memerlukan pengukuran lintas pengulangan dan perhatian pada jumlah contoh yang terbatas.

## J. Run Hard-3 pertama dan rancangan recovery (24 November 2026 WIB)

YOLO11n dan YOLOv8n menyelesaikan pelatihan Hard-3 serta unified COCOeval pada validation. Mask R-CNN berhasil memuat dataset dan bobot pralatih COCO, lalu berhenti dengan `FloatingPointError: Loss Mask R-CNN nonfinite: epoch 1, batch 32`. Log run awal tidak mencatat ID gambar, jumlah instance, atau komponen loss untuk batch tersebut. Perbandingan gabungan juga belum tersedia karena pencarian menemukan dua lokasi referensi all-class. Kedua masalah dicatat terpisah karena terjadi pada tahap yang berbeda.

Percobaan recovery akan mengulang Mask R-CNN dari bobot pralatih COCO dalam FP32. AMP dinonaktifkan untuk menguji apakah kegagalan numerik berulang. Dataset, seed 42, batch size 2, target 20 epoch, SGD, learning rate 0,005, dan MultiStepLR dipertahankan. Precision FP32 merupakan penyimpangan dari baseline Mask R-CNN yang memakai AMP dan perlu diperhitungkan saat menginterpretasikan H1. AMP belum dapat dinyatakan sebagai penyebab kegagalan. Dampak perubahan precision masih menunggu hasil run Kaggle.

Recovery mempertahankan batch tanpa instance positif dan memeriksa bentuk serta nilai tensor input sebelum forward. Jika loss kembali nonfinite, run dihentikan dengan catatan epoch, batch, ID gambar, jumlah instance, dan nilai tiap komponen loss. Hasil kedua YOLO dipulihkan dengan bobot dan konfigurasi historisnya. Evaluasi validation yang sesuai dengan checkpoint dan protokol digunakan kembali. Referensi all-class dipilih berdasarkan kecocokan isinya, sehingga beberapa salinan identik dapat digunakan. [Notebook recovery](notebooks/CPMK3_Hard3_Recovery.ipynb).
