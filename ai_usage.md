# Pernyataan Penggunaan AI

## Alat yang digunakan

ChatGPT digunakan sebagai asisten AI untuk investigasi teknis, penyusunan dan perbaikan kesalahan kode, perencanaan eksperimen, pengorganisasian artefak, serta dokumentasi. Model dasar YOLO11n-seg dijalankan di Kaggle menggunakan notebook tersimpan dan Ultralytics; keduanya merupakan alat eksekusi, bukan asisten AI tambahan.

## Cakupan bantuan

Bantuan AI mencakup:

- Menelusuri format dataset dan mengaudit anotasi segmentasi COCO serta YOLO.
- Menyusun kode konversi COCO ke YOLO dan validasinya, termasuk penanganan orientasi gambar.
- Merancang dan menjelaskan konfigurasi notebook pelatihan model dasar YOLO11n-seg.
- Memeriksa dan memperbaiki pengambilan metrik pada sel terakhir notebook.
- Merencanakan struktur artefak eksperimen dan membantu mengorganisasi hasil secara lokal.
- Mendukung penyusunan logbook, pernyataan penggunaan AI, dan README proyek.

Transkrip prompt historis secara lengkap tidak disimpan di repositori ini. Daftar berikut adalah **ringkasan prompt, bukan prompt verbatim**:

1. Periksa dataset Panoptic Indoor Segmentation dan tentukan cara mengaudit serta membangun ulang label segmentasi instance dari anotasi COCO.
2. Diagnosis ketidaksesuaian ukuran dan orientasi gambar pada proses konversi dataset, lalu verifikasi mask yang sudah dikoreksi.
3. Rancang dan jalankan pelatihan YOLO11n-seg untuk semua kelas yang dapat diulang, dengan bagian dataset, parameter pelatihan, dan keluaran tersimpan yang ditentukan.
4. Ambil metrik pelatihan yang sudah selesai, rangkum proses eksperimen, dan rapikan keluarannya tanpa mengubah eksperimen.
5. Finalisasi dokumentasi CPMK 3 berdasarkan artefak yang ada, bedakan hasil terverifikasi dari rencana, dan ungkapkan penggunaan serta kesalahan AI.

Ringkasan tersebut hanya menjelaskan topik pekerjaan; tidak diklaim sebagai teks prompt asli.

## Contoh kesalahan berbantuan AI yang terverifikasi dan koreksinya

Sel ekspor metrik awal mengharapkan kolom metrik mask `metrics/seg(mAP50-95)`. Kolom yang sebenarnya ada di `results.csv` adalah `metrics/mAP50-95(M)`, sehingga muncul error yang dilaporkan: `AssertionError: Mask mAP column not found.` Sel tersebut kemudian diperbaiki agar memakai nama kolom CSV yang benar. Notebook tersimpan berisi kode pengambilan metrik yang telah dikoreksi beserta ringkasan hasilnya.

Contoh ini menunjukkan batasan konkret kode yang dihasilkan AI: nama API atau metrik yang terlihat masuk akal bisa saja keliru. Koreksi diverifikasi menggunakan keluaran notebook dan nama kolom CSV asli, bukan asumsi yang dihasilkan AI.
