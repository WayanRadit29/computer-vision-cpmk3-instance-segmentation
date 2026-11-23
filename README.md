# Computer Vision CPMK 3 — Segmentasi Instance

Proyek ini mengevaluasi segmentasi instance pada dataset Panoptic Indoor Segmentation v5. Hasil yang saat ini terdokumentasi adalah model dasar (baseline) YOLO11n-seg untuk semua kelas; perbandingan model secara keseluruhan belum selesai.

## Dataset

Dataset model dasar dibangun ulang dari anotasi segmentasi COCO ke format segmentasi instance YOLO. Dataset ini mencakup 74 kelas aktif, dengan 397 gambar pelatihan, 166 gambar validasi, dan 359 gambar uji yang disisihkan. Bagian uji belum dievaluasi.

Manifest konversi dan laporan QA ada di [`analysis/dataset_audit/`](analysis/dataset_audit/). QA melaporkan 11.845 instance tervalidasi, tanpa poligon di luar batas dan tanpa ID kelas yang tidak valid. Sembilan kelas tidak memiliki instance pada data pelatihan; rinciannya ada pada `missing_train_classes` di [`qa_report.json`](analysis/dataset_audit/qa_report.json).

## Status eksperimen

| Model / eksperimen | Status |
| --- | --- |
| YOLO11n-seg, 74 kelas | Model dasar selesai; validasi dilakukan |
| YOLOv8n-seg, semua kelas | Direncanakan |
| Mask R-CNN, semua kelas | Direncanakan |
| Tiga kelas tersulit | Direncanakan |
| Evaluasi pada bagian uji asli | Belum dilakukan |

Konfigurasi model dasar, metrik tiap epoch, ringkasan, berkas model (checkpoint), dan visualisasi tersimpan di [`experiments/yolo11n_all_classes/`](experiments/yolo11n_all_classes/). Ringkasan mencatat mAP50-95 mask validasi tertinggi sebesar 0.18474 pada epoch 45. Ini hasil validasi, bukan hasil pengujian. Metadata checkpoint `best.pt` mencatat epoch `-1`, jadi epoch pemilihan checkpoint belum diketahui dan tidak diasumsikan sebagai epoch 45.

## Reproduksibilitas dan bukti

- Notebook: [`notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb`](notebooks/cpmk3-yolo11n-seg-all-class-baseline.ipynb)
- Konfigurasi dan metrik pelatihan: [`experiments/yolo11n_all_classes/args.yaml`](experiments/yolo11n_all_classes/args.yaml), [`metrics/results.csv`](experiments/yolo11n_all_classes/metrics/results.csv), dan [`metrics/summary.json`](experiments/yolo11n_all_classes/metrics/summary.json)
- Konversi dan validasi dataset: [`analysis/dataset_audit/conversion_manifest.json`](analysis/dataset_audit/conversion_manifest.json), [`analysis/dataset_audit/qa_report.json`](analysis/dataset_audit/qa_report.json), dan [`analysis/dataset_audit/rebuilt_yolo_mask_inspection.png`](analysis/dataset_audit/rebuilt_yolo_mask_inspection.png)
- Verifikasi orientasi: [`evidence/orientation_fixes_verification.png`](evidence/orientation_fixes_verification.png)
- Keputusan dan kendala eksperimen: [`logbook.md`](logbook.md)
- Pernyataan penggunaan AI: [`ai_usage.md`](ai_usage.md)

Durasi pelatihan dihitung dari nilai kumulatif terakhir pada kolom `time` di `results.csv`: 599.949 detik, sekitar 10 menit. Tangkapan layar antarmuka Kaggle yang diberikan menampilkan waktu eksekusi notebook 15m26s; berkasnya masih perlu dipindahkan ke `evidence/gpu_usage/` secara manual oleh pemilik proyek. Kedua durasi ini mengukur hal yang berbeda. Log pelatihan dan konfigurasi `device=0` menunjukkan satu GPU dipakai untuk pelatihan model, sedangkan label `GPU T4 x2` pada tangkapan layar menunjukkan alokasi sesi Kaggle.

## Batasan yang diketahui

- Bagian uji asli belum dievaluasi.
- Eksperimen YOLOv8n-seg, Mask R-CNN, dan tiga kelas tersulit masih direncanakan.
- Sembilan kelas tidak muncul pada data pelatihan, seperti tercantum dalam laporan QA.
- Berkas audit anotasi awal dan salinan lokal tangkapan layar waktu eksekusi Kaggle belum ada di repositori.
- Epoch pemilihan checkpoint tidak dapat dipulihkan dari metadata saat ini (`epoch: -1`, `best_fitness: null`).
