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

Ketiga hasil yang dilaporkan berasal dari data validasi. Pada tahap ini, saya belum menjalankan evaluasi pada set uji asli dan foto pribadi maupun pelatihan khusus tiga kelas tersulit.

## Catatan retrospektif (23 November 2026 WIB)

Hipotesis untuk perbandingan tiga model pada semua kelas belum dicatat sebelum pelatihan pertama.

## H. Evaluasi bersama dan pemeringkatan kandidat Hard-3 (24 November 2026)

Tanggal mengikuti penanda file hasil evaluasi. Ketiga checkpoint `best` dinilai pada 166 gambar validation dengan ground truth COCO yang sama. Ground truth mempertahankan dua koreksi orientasi validation dari audit awal. COCOeval digunakan untuk box dan mask. Konfigurasi menetapkan confidence minimum 0,001, maksimum 100 deteksi per gambar, dan mask threshold 0,50. Resize dan NMS mengikuti pengaturan tiap keluarga model. Penilaian ini memakai bobot yang sudah dilatih. [Konfigurasi](analysis/all_class_comparison/evaluation_config.json).

Mask mAP50–95 tercatat 17,029% untuk YOLO11n, 16,704% untuk YOLOv8n, dan 15,489% untuk Mask R-CNN. YOLOv8n memiliki mask mAP50 tertinggi dan pipeline terpendek. Perbedaan pengaturan inference dan pemilihan checkpoint membatasi penafsiran selisih dari skor puncak pada log pelatihan. [Perbandingan model](analysis/all_class_comparison/model_comparison.csv).

Untuk menghindari pemeringkatan pada kelas yang terlalu jarang atau tidak bisa dinilai, kelas harus memiliki sedikitnya 10 instance train, 5 instance validation, dan AP dari semua model. Sebanyak 39 kelas memenuhi syarat. Mean mask AP50–95 terendah tercatat pada `cabinet-merged`, `book`, dan `bed`. Ketiganya masih kandidat. Tinjauan prediksi validation per kelas dan konfirmasi pilihan dilakukan sebelum pelatihan khusus. Original test dan personal test tidak digunakan untuk seleksi. [Metode dan interpretasi](analysis/all_class_comparison/analysis.md), [aturan seleksi](analysis/all_class_comparison/hard3_selection.json).

## I. Finalisasi kelas dan hipotesis Hard-3 (24 November 2026 WIB)

Hipotesis berikut dirumuskan setelah evaluasi baseline all-class dengan COCOeval dan sebelum pelatihan Hard-3. Hipotesis ini bersifat prospektif untuk eksperimen Hard-3. Catatan retrospektif tentang pelatihan baseline pertama tetap berlaku.

Saya menetapkan `cabinet-merged`, `book`, dan `bed` sebagai Hard-3. Ketiganya memiliki rata-rata Mask AP50–95 terendah di antara 39 kelas dengan minimal 10 instance train, 5 instance validation, dan AP yang tersedia dari ketiga model. Pemilihan hanya memakai data validation. Enam instance validation pada `bed` memenuhi ambang, tetapi membuat AP kelas tersebut peka terhadap perubahan beberapa prediksi. [Ranking COCOeval](analysis/all_class_comparison/hard3_ranking.csv) dan [keputusan final](analysis/all_class_comparison/hard3_decision.json).

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

## K. Audit konvensi anotasi untuk personal test set (24 November 2026 WIB)

Nama dan ID 74 kelas aktif pada YAML hasil konversi dicocokkan dengan anotasi COCO lokal di ketiga split. Pemeriksaan visual menumpangkan poligon YOLO hasil rekonstruksi pada gambar baseline yang sudah dikoreksi orientasinya. Audit mencakup 25 kelas dan 97 pasangan kelas–foto, dengan empat contoh per kelas kecuali `wall`, yang hanya ditemukan pada satu foto dengan dua mask.

Pemisahan keyboard eksternal dari laptop terlihat konsisten pada contoh yang diperiksa. Batas antarlabel kertas dan material dinding lebih bervariasi. Kotak karton terbuka pernah dilabeli `cardboard`, dan sebagian poligon memasukkan isi wadah atau melintasi penghalang kecil. Pada perlengkapan tidur, selimut atau bantal terkadang tercakup dalam mask `bed`. Untuk foto pribadi, keputusan dibuat berdasarkan bentuk wadah, material yang dapat dikenali, dan batas objek yang terlihat. Seprai yang mengikuti bentuk kasur dicakup dalam `bed`, sementara bantal dan selimut dengan batas jelas diberi mask terpisah.

Aturan *visible-instance masking* dipilih agar batas anotasi dapat diperiksa langsung dari foto. Area tertutup tidak digambar dan instance yang dapat dibedakan diberi mask terpisah. Variasi label dapat membuat prediksi masuk kategori yang berbeda dari ground truth, sedangkan perbedaan batas mask dapat mengubah IoU dan kemungkinan kecocokan pada ambang COCOeval. Keduanya berpotensi memengaruhi AP. Besarnya pengaruh belum diukur. [Pedoman anotasi dan bukti visual](data/custom_test/annotation_guidelines.md)

## L. Kegagalan recovery FP32 dan rancangan stabilisasi V2 (24 November 2026 WIB)

Pada run pertama, pelatihan Mask R-CNN memakai AMP dan laju belajar 0,005. Loss menjadi nonfinite pada epoch 1 batch 32. Recovery pertama memulihkan pelatihan dan evaluasi kedua YOLO tanpa retraining. Mask R-CNN tetap gagal di FP32 dengan laju belajar 0,005 pada epoch 1 batch 43. Gambar train ID 153 dan 205 pada batch tersebut tidak memiliki instance Hard-3. Diagnostik mencatat `loss_classifier` dan `loss_box_reg` bernilai NaN, `loss_objectness` sebesar 933,6926, serta `loss_mask` dan `loss_rpn_box_reg` sama dengan nol.

Pola ini sejalan dengan kemungkinan RoI tanpa sampel setelah pemilihan proposal. Jumlah proposal pada batch yang gagal tidak tercatat. Target kosong valid dan tetap dipertahankan sebagai sampel negatif. Norma gradien, parameter, dan aktivasi pada percobaan sebelumnya belum tercatat, sehingga pembaruan optimizer yang tidak stabil dan masalah pada keluaran RPN masih perlu diperiksa. Kegagalan dalam FP32 menunjukkan bahwa menonaktifkan AMP saja belum cukup pada percobaan tersebut.

Recovery V2 dimulai dari bobot pralatih COCO dalam FP32. Laju belajar diturunkan menjadi 0,0005 dan norma L2 gradien dibatasi pada 5,0 sebelum setiap pembaruan SGD. Dataset, seed, ukuran batch, jadwal penurunan laju belajar, momentum, weight decay, dan jumlah epoch dipertahankan. Sebelum pelatihan, model diagnostik terpisah memeriksa dua gambar yang gagal dan satu batch positif tanpa pembaruan bobot. Pelatihan memerlukan kedua probe lolos. Log awal mencatat loss, jumlah proposal RPN dan RoI, norma gradien, serta kondisi parameter.

Perubahan precision, laju belajar, dan gradient clipping membatasi interpretasi H1 untuk Mask R-CNN. Selisih AP terhadap baseline mencerminkan paket konfigurasi Hard-3 yang digunakan. Protokol evaluasi tetap menggunakan validation, sementara original test dan personal test disisihkan untuk evaluasi akhir. Hasil V2 menunggu eksekusi Kaggle. [Notebook recovery V2](notebooks/CPMK3_Hard3_Recovery_V2.ipynb)

## M. Recovery V2 dan hasil validation Hard-3 (25 November 2026 WIB)

Run Kaggle V2 dimulai sekitar 09.18 WIB sesuai ID run `20261125T021827548702Z`. Percobaan awal dengan AMP gagal pada epoch 1 batch 32. Percobaan FP32 dengan laju belajar 0,005 gagal pada batch 43. Dua gambar tanpa instance Hard-3 (ID 153 dan 205) menghasilkan NaN pada loss classifier dan box regression, dengan RPN objectness loss 933,6926. Angka tersebut mendorong pemeriksaan pembaruan optimizer, keluaran RPN, dan jumlah RoI yang terpilih. Data diagnostik percobaan lama belum cukup untuk menetapkan penyebabnya. [Traceback awal](evidence/training_logs/hard3_maskrcnn/initial_training_error.txt) dan [diagnostik recovery FP32](evidence/training_logs/hard3_maskrcnn/fp32_batch43_diagnostic.json).

Saya mempertahankan notebook Kaggle dengan input yang telah terpasang, termasuk cell rebuild ZIP dari folder ekstraksi. Dua cell diganti. Cell pelatihan menetapkan fingerprint Mask R-CNN V2 setelah recovery YOLO, menjalankan preflight, dan melatih Mask R-CNN dengan FP32, SGD, laju belajar 0,0005, serta clipping norma L2 5,0. Cell ekspor menyimpan hasil dan diagnostik V2. YOLO11n dan YOLOv8n memakai kembali checkpoint, fingerprint, dan evaluasi run awal tanpa pelatihan ulang. [Rekonstruksi kode notebook](notebooks/CPMK3_Hard3_Final_Reconstructed.ipynb) dan [status Kaggle](analysis/hard3_comparison/provenance/run_summary.json).

Kedua probe awal lolos. Pada training batch 43, kedua gambar yang sebelumnya bermasalah kembali muncul dan menghasilkan total loss 0,02873, RPN objectness 0,02298, serta 512 RoI per gambar. Mask R-CNN menyelesaikan 20 epoch dan checkpoint terbaik epoch 5 berhasil dievaluasi dengan unified COCOeval. Keberhasilan konfigurasi gabungan belum mengisolasi faktor stabilisasi atau membuktikan akar kegagalan. [Probe](experiments/maskrcnn_hard3/metrics/preflight.jsonl), [log 100 batch pertama](experiments/maskrcnn_hard3/metrics/early_training_batches.jsonl), dan [ringkasan training](experiments/maskrcnn_hard3/metrics/summary.json).

Pada 166 gambar validation, kenaikan rata-rata Mask AP50–95 tiga kelas adalah +3,820 poin untuk YOLO11n, +1,909 untuk YOLOv8n, dan +4,284 untuk Mask R-CNN. Hasil validation mendukung H1. Mask R-CNN mencapai rata-rata tertinggi (5,738%), sehingga H2 tidak didukung. H3 didukung dengan AP `book` Mask R-CNN 10,707%. Perbandingan ini memakai satu seed dan `bed` hanya memiliki enam instance validation. Langkah berikutnya adalah menyelesaikan anotasi serta bukti personal test set, mengevaluasi model pada kedua set uji setelah protokol dikunci, lalu menganalisis kesalahan dan perbedaan domain. [Analisis dan evaluasi hipotesis](analysis/hard3_comparison/analysis.md#evaluasi-hipotesis).

## N. Validasi ground truth personal test set (26 November 2026 WIB)

Setelah pengumpulan awal di Classroom, ekspor COCO Segmentation dari Roboflow diperiksa sebagai tahap penyelesaian susulan. Ekspor tercatat pada 13.21 WIB dan mencakup 40 foto dalam split test. P001–P040 dicocokkan dengan foto sumber melalui nama, dimensi setelah koreksi orientasi EXIF, dan perbandingan thumbnail RGB. Seluruh foto memiliki anotasi. Bukti foto asli, manifest, metadata, dan screenshot proses tetap menjadi rujukan. Pemeriksaan visual mengidentifikasi 12 foto kondisi sulit yang berbeda. [Pemetaan foto](analysis/personal_test_qa/image_mapping.csv) dan [bukti kondisi sulit](analysis/personal_test_qa/difficult_conditions.md)

Pemeriksaan COCO memvalidasi 213 instance dari 38 kelas. ID kategori diselaraskan berdasarkan kecocokan nama dengan taksonomi 74 kelas. Kategori yang berisi nama proyek tidak memiliki instance sehingga tidak masuk ke master. `mouse`, `keyboard`, dan `toothbrush` dipertahankan meski tidak muncul pada split train. Area piksel dan empat bounding box dihitung kembali dari mask yang valid. Label semantik, ID anotasi, dan geometri mask dipertahankan. Kandidat Hard-3 menyaring anotasi menjadi 33 instance pada 25 foto positif. Lima belas foto tanpa kelas target tetap disertakan agar false positives dapat diukur. [Laporan validasi](analysis/personal_test_qa/report.md) dan [log koreksi teknis](analysis/personal_test_qa/correction_log.json)

Overlay seluruh foto diperiksa untuk melihat kesesuaian kelas, batas dan pemisahan instance, serta bagian yang tertutup. Di P001, satu RLE meja sudah menyimpan empat komponen terpisah. Pada P003, P005, dan P008, irisan mask mencakup seluruh objek depan atau bagian yang besar, sehingga batas mask perlu saya tinjau. Status kandidat masih **REVIEW REQUIRED**. Ground truth perlu disetujui dan dibekukan sebelum prediksi personal test ditinjau. [Antrean review](analysis/personal_test_qa/human_review.md)

## O. Koreksi batas mask berdasarkan peninjauan anotasi (26 November 2026 WIB)

Pada tahap penyelesaian setelah pengumpulan awal di Classroom, saya menyetujui empat koreksi. Pada P003, piksel bantal #32 dikeluarkan dari tempat tidur #33. Pada P005, piksel laptop #193 dikeluarkan dari dinding #191 dan piksel mouse #195 dikeluarkan dari kertas #196. Saya mengonfirmasi bahwa bidang merah muda adalah kertas. Pada P008, piksel cangkir #27 dikeluarkan dari jendela #29. Meja #48 pada P001 tetap satu instance.

Sesuai pedoman visible-instance masking, objek yang berada di depan tidak ikut dihitung sebagai bagian terlihat dari objek di belakangnya. Keputusan diterapkan pada resolusi ekspor dengan operasi `target AND NOT occluding`, kemudian dikodekan sebagai COCO RLE dan dihitung ulang area serta kotak pembatasnya. Keempat pasangan memiliki nol piksel irisan setelah koreksi. Label, ID, referensi gambar, serta seluruh mask selain empat target tetap sesuai kandidat awal. Hard-3 diturunkan dari master hasil koreksi sehingga memakai mask `bed` P003 yang sama.

Validasi ulang menghasilkan 40 foto, 213 instance dari 38 kelas pada All-Class, serta 33 instance pada Hard-3 dengan 25 foto positif dan 15 negatif. Keempat koreksi membuat batas mask lebih sesuai dengan bagian objek yang terlihat. Dampaknya terhadap metrik model belum diukur. Kandidat menunggu pemeriksaan akhir dan persetujuan pembekuan ground truth. [Hasil validasi dan overlay](analysis/personal_test_qa/finalization/report.md)

## P. Persetujuan akhir dan pembekuan ground truth (26 November 2026 WIB)

Saya meninjau overlay sebelum dan sesudah pada P003, P005, dan P008, lalu menyetujui keempat koreksi serta pembekuan kedua versi anotasi. Master COCO dipakai langsung untuk All-Class, sedangkan Hard-3 diturunkan dari master dan memakai 40 foto yang sama. Kedua berkas yang ditinjau ditetapkan sebagai ground truth Batch 2 berdasarkan SHA-256. Status dan tanggal persetujuan dicatat pada manifest agar evaluasi dapat mengacu pada byte anotasi yang disetujui. Batch 1 selesai. [Manifest pembekuan](analysis/personal_test_qa/finalization/ground_truth_freeze.json)

## Q. Persiapan evaluasi test set terpadu (26 November 2026 WIB)

Enam checkpoint dan dua test set disiapkan untuk 12 kombinasi pengujian. Original test berisi 359 foto dan 5.344 anotasi. Proyeksi Hard-3 mempertahankan semua foto dan menyisakan 530 anotasi pada 137 foto positif. Personal test menggunakan kedua COCO JSON beku dengan hash dari Batch 1. Notebook mempertahankan pengaturan confidence, NMS, resize, dan batas deteksi dari konfigurasi unified validation, serta merekonstruksi normalisasi Mask R-CNN sesuai training. Perbandingan tiga kelas memakai proyeksi prediksi yang telah disimpan. Evaluasi penuh menunggu preflight dan smoke test di Kaggle. [Notebook](notebooks/CPMK3_Unified_Test_Evaluation.ipynb) dan [catatan protokol](analysis/test_evaluation/README.md)

## R. Audit Run01 evaluasi test set (26 November 2026 WIB)

Run Tesla T4 lolos preflight dan 12 smoke test, lalu berakhir parsial dengan tujuh hasil primer terverifikasi dan lima kegagalan CUDA OOM. Pada YOLO, loader daftar memasukkan seluruh split ke satu batch meskipun notebook meminta `batch=1`. Traceback original berhenti pada konvolusi mask prototypes. Traceback personal berhenti ketika `Masks.data` membentuk stack mask padat. Mask R-CNN All-Class personal gagal ketika Torchvision menumpuk mask FP32 pada resolusi asli. Recovery yang diusulkan mempertahankan ground truth dan pengaturan penilaian, serta menunggu persetujuan implementasi dan uji kesetaraan pada GPU. [Audit, traceback, dan rencana recovery](analysis/test_evaluation/run01_audit/README.md)

## S. Implementasi recovery Run01 (26 November 2026 WIB)

Setelah persetujuan patch, dibuat notebook recovery terpisah. Jalur YOLO memverifikasi batch singleton FP32 dan mengodekan satu crop mask per instance pada resolusi asli. Jalur Mask R-CNN mempertahankan scaling box dan interpolasi mask Torchvision sambil memproses satu mask tiap langkah. Rancangan saat itu menyiapkan pemeriksaan fingerprint dan checksum untuk mengimpor tujuh hasil Run01, lalu menjalankan lima pasangan yang gagal setelah uji GPU. [Audit Run01](analysis/test_evaluation/run01_audit/README.md)

## T. Evaluasi test set Run02 (26 November 2026 WIB)

Karena pemindahan dan pemasangan artefak parsial menambah langkah reproduksi di Kaggle, saya memilih menjalankan ulang seluruh 12 kombinasi memakai jalur inferensi hemat memori yang sama. Run02 memakai enam checkpoint dan kedua test set yang telah dikunci, dengan `RESUME_INPUTS = []` dan `SELECT_PAIRS = []`.

Cell 11 direvisi untuk memisahkan pemeriksaan encoding dari perbandingan dua jalur forward. Probe membandingkan mask native dan serial dari sumber prediksi yang sama. Di Tesla T4, kedua kontrol letterbox, 806 mask same-source, fixture sintetis 100 mask per keluarga, serta stress test resolusi tinggi untuk keenam checkpoint lulus. Perbandingan jalur forward lama dengan jalur singleton YOLO11n All-Class mencatat dua mask yang masing-masing berbeda satu piksel. Temuan ini dicatat sebagai diagnostik, dan semua metrik Run02 dihitung dari prediksi baru. [Laporan gate GPU](analysis/test_evaluation/run02/gpu_equivalence_report.json)

Run02 berjalan pukul 22.29–23.00 WIB. Kedua domain dinilai untuk setiap checkpoint dan konsolidasi Kaggle mencocokkan ulang metrik dari 12 pasang prediksi yang tersimpan. Prediksi All-Class diproyeksikan ke tiga kategori untuk perbandingan Hard-3 setelah seleksi deteksi per model. [Notebook yang dijalankan](notebooks/CPMK3_Unified_Test_Evaluation_Recovery.ipynb) dan [manifest hasil](analysis/test_evaluation/run02/evaluation_status_manifest.json)

## U. Verifikasi hasil test set (27 November 2026 WIB)

Artefak Run02 ditelusuri ke keenam checkpoint, keempat cakupan ground truth, dan 399 berkas gambar melalui SHA-256. Pemeriksaan menghubungkan cakupan gambar dan hitungan prediksi dengan roster dan catatan runtime, memvalidasi RLE pada resolusi asli, serta mencocokkan metrik JSON dengan tabel. Run02 menjadi sumber hasil test set untuk analisis pergeseran domain dan kegagalan berikutnya. Fingerprint historis runtime dipertahankan, dan hash notebook yang memuat revisi Cell 11 dicatat terpisah. [Provenance Run02](analysis/test_evaluation/run02/integration.json)

## V. Analisis kuantitatif test set (27 November 2026 WIB)

Analisis Run02 membandingkan arsitektur dalam setiap pasangan skenario dan domain. Selisih spesialisasi dihitung pada tiga kelas yang sama. Perubahan domain All-Class dihitung pada 37 kelas dengan ground truth yang dapat dinilai di kedua set uji. Hard-3 menaikkan recall agregat pada semua pasangan, sementara kenaikan AP terjadi pada YOLOv8n di personal test (+8,470 pp). Bukti test tidak mendukung H1 dan H2, serta mendukung H3 sebagian. Analisis berikutnya akan menelaah pola kesalahan pada tingkat instance dan gambar. [Analisis kuantitatif](analysis/test_evaluation/quantitative_analysis.md)

## W. Analisis kegagalan personal test (27 November 2026 WIB)

Prediksi enam checkpoint pada 40 foto disaring menggunakan pencocokan sadar kelas pada IoU 0,50 dan 0,75 serta pemeriksaan overlap agnostik. Lima foto mendokumentasikan konflik label buku, prediksi lemari benar yang skornya berada di bawah FP, dan kasur dengan kotak cocok tetapi mask gagal. Perbandingan memakai prediksi Run02 dan ground truth beku. Eksperimen post-hoc tentang urutan confidence diajukan untuk saya pertimbangkan. [Analisis kegagalan dan bukti per instance](analysis/test_evaluation/failure_analysis/failure_analysis.md)

## X. Eksperimen oracle ranking (27 November 2026 WIB)

Setelah meninjau lima kasus, saya memilih D1. Skrip yang disusun dengan bantuan AI mengurutkan ulang prediksi lemari Mask R-CNN pada 40 foto secara global dengan TP di depan. Assignment dan recall maksimum tetap. Headroom AP50 diagnostik mencapai 45,708 pp untuk All-Class dan 94,396 pp untuk Hard-3. Hasil ini mendukung kontribusi ranking pada AP Hard-3 yang rendah di personal test. [Hasil dan batas interpretasi](analysis/test_evaluation/diagnostic_experiment/diagnostic_experiment.md)

## Y. Finalisasi laporan akademik (27 November 2026 WIB)

Dengan bantuan AI, saya menyusun laporan akhir dari hasil Run02 dan analisis Batch 3 yang telah saya tinjau. Laporan menghubungkan hasil validation dan kedua test set dengan hipotesis yang saya catat sebelum pelatihan Hard-3. AP resmi COCOeval dan AP50 diagnostik D1 disajikan dengan definisi dan batas interpretasi masing-masing. Saya menggunakan log pelatihan, bukti anotasi, dan provenance untuk menelusuri hasil dan keputusan penelitian. Tahap ini berlangsung setelah pengumpulan awal melalui Classroom. [Laporan akademik](analysis/report.md)
