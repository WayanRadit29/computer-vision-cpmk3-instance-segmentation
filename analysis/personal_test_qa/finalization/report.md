# Finalisasi ground truth personal test set

**FROZEN · 26 November 2026 WIB.** Setelah pengumpulan awal di Classroom, annotator meninjau overlay P003, P005, dan P008 serta menyetujui empat koreksi dan pembekuan ground truth All-Class dan Hard-3. Kedua berkas yang disetujui telah lolos validasi COCO dan diidentifikasi dengan SHA-256 pada manifest pembekuan. [Audit sebelum koreksi](../report.md) dan [manifest pembekuan](ground_truth_freeze.json)

## Keputusan dan hasil koreksi

Pedoman visible-instance masking membatasi mask pada bagian objek yang terlihat. Untuk setiap pasangan yang disetujui, piksel objek depan dikeluarkan dari mask target dengan operasi biner `target AND NOT occluding`. Hasil disimpan sebagai COCO RLE pada resolusi ekspor agar komponen terpisah dan lubang pada mask tetap dapat direpresentasikan. Area dan kotak pembatas dihitung kembali dari mask hasil koreksi. Keempat kotak pembatas tetap memiliki koordinat semula.

| Foto | Target | Objek di depan | Area awal (px) | Piksel dikeluarkan | Area akhir (px) | Pengurangan area target | Irisan tersisa (px) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| P003 | `bed` #33 | `pillow` #32 | 2.738.023 | 329.546 | 2.408.477 | 12,04% | 0 |
| P005 | `wall-other-merged` #191 | `laptop` #193 | 3.486.209 | 1.356.283 | 2.129.926 | 38,90% | 0 |
| P005 | `paper` #196 | `mouse` #195 | 1.009.724 | 174.085 | 835.639 | 17,24% | 0 |
| P008 | `window-other` #29 | `cup` #27 | 4.351.551 | 342.444 | 4.009.107 | 7,87% | 0 |

Annotator mengonfirmasi label `paper` pada P005. Meja `table-merged` #48 pada P001 tetap dihitung satu kali sesuai RLE sumber yang memuat beberapa region. Empat mask objek depan dan 205 mask lain yang berada di luar keputusan koreksi tetap identik. [Log koreksi dan metadata bbox](correction_log.json) dan [tabel pengukuran CSV](correction_summary.csv)

## Bukti visual

Tiap baris menampilkan foto ekspor tanpa overlay, anotasi sebelum dan sesudah, serta piksel yang dikeluarkan. Sian menunjukkan target, jingga menunjukkan objek di depan, dan magenta menunjukkan irisan yang dikeluarkan. Ukuran panel diperkecil untuk dibaca, sedangkan pengukuran pada tabel memakai resolusi penuh.

| Foto | Pasangan mask |
| --- | --- |
| [P003 sebelum dan sesudah](overlays/P003_before_after.png) | Tempat tidur dan bantal |
| [P005 sebelum dan sesudah](overlays/P005_before_after.png) | Dinding dan laptop, serta kertas dan mouse |
| [P008 sebelum dan sesudah](overlays/P008_before_after.png) | Jendela dan cangkir |

## Validasi dataset

Pemeriksaan `pycocotools` memastikan P001–P040 tetap unik dan cocok dengan identitas serta dimensi ekspor. Mask dapat didekode, berarea positif, dan sesuai dengan bounding box. ID anotasi, referensi gambar, serta ID dan nama kelas sesuai dengan [pemetaan baseline](../category_mapping.json). All-Class memakai taksonomi 74 kategori dan Hard-3 memakai `bed` = 1, `book` = 2, dan `cabinet-merged` = 3. Hard-3 diturunkan langsung dari master sehingga mask `bed` #33 yang sudah dikoreksi identik pada kedua skenario.

| Indikator | Sebelum | Sesudah |
| --- | ---: | ---: |
| Foto | 40 | 40 |
| Instance All-Class | 213 | 213 |
| Kelas terwakili | 38 | 38 |
| Instance Hard-3 | 33 | 33 |
| `bed` | 10 | 10 |
| `book` | 11 | 11 |
| `cabinet-merged` | 12 | 12 |
| Foto positif Hard-3 | 25 | 25 |
| Foto negatif Hard-3 | 15 | 15 |
| Anotasi invalid | 0 | 0 |
| Anotasi dengan kategori tak dikenal | 0 | 0 |

Empat koreksi membatasi ground truth pada bagian objek yang terlihat. Pengaruhnya terhadap AP belum diukur. Kedua versi beku siap untuk evaluasi Batch 2. [Statistik sebelum dan sesudah](dataset_statistics.json)

## Ground truth beku dan reproduksi

| Artefak | Peran |
| --- | --- |
| [Master COCO dan All-Class](../../../data/custom_test/annotations/all_class_master_corrected.coco.json) | Sumber anotasi untuk 40 foto dan 74 kelas |
| [COCO Hard-3](../../../data/custom_test/annotations/hard3_corrected.coco.json) | Turunan master dengan tiga kelas dan 40 foto yang sama, termasuk foto negatif |
| [Manifest pembekuan](ground_truth_freeze.json) | Persetujuan annotator, status resmi, versi anotasi, SHA-256 final, dan root gambar |
| [Provenance saat review](dataset_provenance.json) | Riwayat sumber, checksum hasil koreksi, dan bukti visual sebelum persetujuan akhir |
| [Skrip finalisasi](../finalize_ground_truth.py) | Pengurangan mask yang disetujui, validasi, dan pembuatan bukti |

Kedua dataset memakai root gambar `data/custom_test/roboflow_raw/`. Simpan arsip Roboflow v1 dengan checksum pada manifest dan ekstrak ke root tersebut dengan struktur aslinya. Untuk mereproduksi JSON, gunakan kedua kandidat sebelum koreksi dan berkas QA yang dirujuk oleh [provenance saat review](dataset_provenance.json), lalu jalankan `python analysis/personal_test_qa/finalize_ground_truth.py` dari root repositori dengan lingkungan Python pada [manifest pembekuan](ground_truth_freeze.json). Skrip memeriksa checksum sumber dan menolak menimpa keluaran versi ini bila isinya berbeda. Kedua JSON menyimpan metadata tahap review sebagai bagian dari byte yang disetujui. Status beku resmi tercatat pada manifest.

Pada uji ulang dari salinan input tanpa keluaran sebelumnya, skrip menghasilkan sembilan artefak yang identik byte demi byte, termasuk kedua JSON evaluasi.
