# Logbook Eksperimen CPMK 3

## A. Audit anotasi YOLO awal

- **Tanggal:** 20 November 2026.
- **Pengamatan:** Poligon pada label YOLO awal dilaporkan memiliki koordinat ternormalisasi di luar batas pada 3.527 dari 11.845 anotasi (29,8%). Poligon COCO yang diperiksa tidak menunjukkan masalah serupa jika dibandingkan dengan dimensi gambar di berkas JSON.
- **Bukti:** Error message dari code cell kaggle (tidak ada file screenshot atau message aslinya karena tidak disimpan) laporan QA hasil pembangunan ulang tersedia di [`analysis/dataset_audit/qa_report.json`](analysis/dataset_audit/qa_report.json).
- **Keputusan:** Membangun ulang dataset dari anotasi COCO sebagai sumber utama, bukan memakai poligon YOLO awal.
- **Hasil / batasan:** Keputusan ini dicatat berdasarkan konteks historis. Angka audit awal belum dapat dihitung ulang secara independen karena artefak audit aslinya tidak tersimpan.

## B. Ketidaksesuaian orientasi gambar

- **Tanggal:** 20 November 2026.
- **Pengamatan:** Empat berkas JPEG memiliki dimensi aktual 480×640, sedangkan metadata COCO mencatat 640×480.
- **Bukti:** [`analysis/dataset_audit/conversion_manifest.json`](analysis/dataset_audit/conversion_manifest.json) mencatat nama berkas, dimensi, dan rotasi; [`evidence/orientation_fixes_verification.png`](evidence/orientation_fixes_verification.png) menampilkan lapisan mask di atas gambar setelah perbaikan.
- **Keputusan:** Memutar gambar dan koordinat anotasi secara bersamaan: train `182169107` CCW; train `182169121` CCW; validation `182169109` CW; validation `182169108` CCW.
- **Hasil / verifikasi:** Manifest mencatat keempat perbaikan. Gambar verifikasi visual memperlihatkan poligon hasil pembangunan ulang di atas gambar yang sudah dikoreksi.

## C. Pembangunan ulang COCO ke YOLO dan QA

- **Tanggal:** 20 November 2026
- **Kegiatan:** Mengonversi segmentasi instance COCO ke label segmentasi instance YOLO, menggunakan pemetaan `yolo_id = coco_category_id - 1` dan mengecualikan kategori kosong ID 0.
- **Bukti:** [`conversion_manifest.json`](analysis/dataset_audit/conversion_manifest.json), [`qa_report.json`](analysis/dataset_audit/qa_report.json), dan [`rebuilt_yolo_mask_inspection.png`](analysis/dataset_audit/rebuilt_yolo_mask_inspection.png).
- **Hasil:** Sebanyak 922 gambar dan 11.845 instance dipertahankan pada bagian train (397 gambar / 4.479 instance), validation (166 / 2.022), dan test (359 / 5.344); sebanyak 74 kelas aktif dipetakan. QA memvalidasi seluruh 11.845 instance, tanpa poligon di luar batas dan tanpa ID kelas tidak valid. Gambar inspeksi mask menjadi pemeriksaan visual.
- **Batasan:** QA melaporkan sembilan kelas aktif tanpa instance pada data train. Kemunculannya di validation atau test tidak menggantikan ketiadaan contoh pelatihan.

## D. Model dasar YOLO11n-seg untuk semua kelas

- **Tanggal:** 2026-11-21 UTC, berdasarkan penanda waktu di dalam checkpoint.
- **Kegiatan:** Melatih `yolo11n-seg.pt` pralatih pada dataset hasil pembangunan ulang dengan 74 kelas. Ini merupakan model dasar (baseline) semua kelas.
- **Bukti:** [`args.yaml`](experiments/yolo11n_all_classes/args.yaml), [`results.csv`](experiments/yolo11n_all_classes/metrics/results.csv), [`summary.json`](experiments/yolo11n_all_classes/metrics/summary.json), bobot tersimpan, dan notebook yang telah dijalankan.
- **Konfigurasi:** 60 epoch dikonfigurasi dan tercatat; ukuran gambar 640; batch 8; nilai seed 42; `device=0`; validasi aktif. Log pelatihan notebook mengidentifikasi CUDA:0 sebagai Tesla T4, konsisten dengan penggunaan satu GPU.
- **Durasi tercatat:** Nilai kumulatif terakhir pada kolom `time` adalah 599.949 detik. Perhitungan: 599.949 / 60 = 9,99915 menit (sekitar 10,00 menit atau 0,16665 jam). Ini durasi pelatihan, bukan durasi keseluruhan eksekusi notebook.
- **Hasil validasi:** Ringkasan menandai epoch 45 sebagai titik dengan mAP50-95 mask validasi tertinggi, lalu melaporkan mask mAP50-95 0.18474, mask mAP50 0.27586, dan box mAP50-95 0.20845. Semua ini metrik validasi. Bagian test asli belum dievaluasi.
- **Catatan checkpoint:** Checkpoint `best.pt` yang tersimpan berisi `epoch: -1` dan `best_fitness: null`, sehingga tidak menunjukkan epoch pemilihannya. Puncak validasi epoch 45 tidak boleh dianggap sebagai bukti bahwa `best.pt` berasal dari epoch 45.
- **Bukti waktu eksekusi notebook:** Tangkapan layar antarmuka Kaggle (bisa dilihat pada `evidence/gpu_usage/`) yang diberikan menampilkan `15m 26s` dan `GPU T4 x2`. Waktu tersebut dicatat sesuai tampilan, bukan disimpulkan dari durasi pelatihan. Label alokasi sesi Kaggle tidak mengubah konfigurasi `device=0` pada proses pelatihan.

## E. Perbaikan sel ekspor metrik

- **Tanggal:** 21 November 2026.
- **Pengamatan:** Setelah pelatihan pertama menyelesaikan 60 epoch, sel ekspor dilaporkan gagal. Pesan yang dilaporkan adalah `AssertionError: Mask mAP column not found.` Kode berbantuan AI mengharapkan `metrics/seg(mAP50-95)`, sedangkan `results.csv` menggunakan `metrics/mAP50-95(M)`.
- **Bukti:** Error message kaggle ouput cell.
- **Perbaikan:** Mengubah referensi kolom metrik mask agar sesuai dengan nama pada `results.csv` (`metrics/mAP50-95(M)` dan `metrics/mAP50(M)`).
- **Verifikasi / pelajaran:** Notebook yang telah dikoreksi melaporkan 60 epoch dan ringkasan validasi. Periksa nama kolom CSV sebelum menulis kode pengambilan metrik; jangan mengasumsikan konvensi penamaan pustaka.

## F. Konflik versi notebook Kaggle

- **Tanggal:** 21 November 2026.
- **Pengamatan:** Percobaan penyimpanan berikutnya dilaporkan menghasilkan `ConcurrencyViolation` dengan `ExpectedSequence=49, ActualSequence=48`.
- **Bukti:** Notifikasi dari Kaggle.
- **Keputusan / hasil:** Catat sebagai konflik sinkronisasi notebook, terpisah dari implementasi pelatihan model. Langkah pemulihan pastinya belum terverifikasi dan tidak dicatat sebagai fakta.

## Catatan retrospektif 23 November 2026 (WIB)

Hipotesis awal belum didokumentasikan sebelum pelatihan pertama YOLO11n-seg. Catatan ini ditambahkan setelah pelatihan untuk mencatat keterbatasan dokumentasi eksperimen pertama.
