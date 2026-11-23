# Logbook Eksperimen CPMK 3

Catatan ini mengikuti proses penyiapan data dan pelatihan model segmentasi instance untuk Panoptic Indoor Segmentation v5. Setiap tahap mencantumkan sumber yang dapat diperiksa dan keterbatasan bukti yang tersedia.

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

## F. Konflik versi notebook Kaggle (21 November 2026)

Saat notebook hendak disimpan kembali, Kaggle melaporkan `ConcurrencyViolation` dengan `ExpectedSequence=49, ActualSequence=48`. Catatan proyek mengaitkan pesan ini dengan sinkronisasi versi notebook. Langkah pemulihan yang dilakukan belum terdokumentasi.

## Catatan retrospektif (23 November 2026 WIB)

Hipotesis awal belum dicatat sebelum pelatihan pertama (3 model untuk semua kelas). Catatan ini ditulis setelah pelatihan untuk menjelaskan keterbatasan dokumentasi eksperimen pertama.

## G. Pelatihan dan integrasi YOLOv8n-seg (22–23 November 2026 WIB)

YOLOv8n-seg dilatih sebagai model kedua dengan pembagian data yang sama. Penanda waktu checkpoint berasal dari 22 November, dan hasilnya ditata di proyek pada 23 November. Konfigurasi memakai `yolov8n-seg.pt`, `epochs=60`, `imgsz=640`, `batch=8`, `seed=42`, `device=0`, `val=true`, dan `split: val`. Metadata checkpoint mencatat Ultralytics 8.4.216. Ringkasan ekspor mencatat Tesla T4 dan PyTorch `2.11.0+cu130`. [Konfigurasi](experiments/yolov8n_all_classes/args.yaml) dan [ringkasan](experiments/yolov8n_all_classes/metrics/summary.json)

Daftar, ukuran, dan nilai pemeriksaan CRC untuk gambar serta label sama di kedua ZIP eksperimen. Saat ZIP YOLOv8n dibandingkan dengan salinan lokal, 1.845 dari 1.849 entri cocok melalui SHA-256. Empat pengecualian terdiri atas dua cache label dengan byte berbeda, `qa_report.json` dengan isi JSON sama meski byte berbeda, dan satu label validasi kosong yang ada di kedua ZIP tetapi tidak ada di salinan lokal (`182169175_jpg.rf.a8d223203fc5bd311086fdf38e40eb0a.txt`).

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

Sebanyak 42 berkas dari `yolov8n_all_class.zip` disalin ke folder `experiments/yolov8n_all_classes/` dan dicocokkan berdasarkan ukuran dan SHA-256, termasuk konfigurasi, tiga berkas metrik, 23 visualisasi, dan 15 berkas bobot. Empat gambar audit dan keluaran sel yang cocok dengan berkas lama digunakan kembali. Notebook Kaggle ditambahkan secara terpisah pada 23 November. Keluaran sel ekspor memuat ringkasan yang sama dengan `summary.json` di folder eksperimen. Salinan notebook tidak memuat keluaran sel bertipe error. [Notebook Kaggle](notebooks/cpmk3-yolov8n-seg-all-class-baseline.ipynb)

Ringkasan mencatat `test_set_evaluated: false`. Hasil evaluasi pada set uji asli dan foto pribadi, hasil Mask R-CNN, serta eksperimen tiga kelas tersulit belum tersedia.
