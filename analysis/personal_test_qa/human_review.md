# Keputusan anotasi personal test set

Saya menggunakan antrean ini untuk meninjau temuan sebelum menyetujui koreksi. Empat usulan telah diputuskan dan diterapkan. [Hasil finalisasi](finalization/report.md)

**Status saat audit kandidat: REVIEW REQUIRED.** Ground truth kandidat mempertahankan seluruh keputusan semantik dari ekspor. Tiga foto di bawah memerlukan pemeriksaan saya sebelum ground truth dibekukan. Persentase merupakan irisan mask yang diukur pada resolusi ekspor.

| Foto dan overlay | Prioritas | Temuan | Usulan untuk ditinjau |
| --- | --- | --- | --- |
| [P003](overlays/P003.jpg) | Penting | Seluruh mask `pillow` #32 ikut tercakup di `bed` #33, yaitu 12,04% dari mask tempat tidur. Bantal terlihat sebagai objek terpisah. | Pertahankan mask bantal. Bila saya mengonfirmasi batasnya, keluarkan bagian kasur yang tertutup bantal dari #33 sesuai pedoman mask bagian terlihat. |
| [P005](overlays/P005.jpg) | Penting | Mask `wall-other-merged` #191 meliputi 50,61% mask `laptop` #193. Seluruh `mouse` #195 juga tercakup di `paper` #196, yaitu 17,24% mask `paper`. | Periksa kembali batas dinding di balik laptop. Konfirmasi apakah lembar merah muda benar-benar kertas. Jika iya, batasi #196 pada area kertas yang terlihat. Jika alas mouse, konfirmasi label yang sesuai atau pengecualian instance berdasarkan taksonomi. |
| [P008](overlays/P008.jpg) | Penting | Seluruh mask `cup` #27 ikut tercakup di `window-other` #29, yaitu 7,87% mask jendela. Cangkir berada di depan bidang yang diberi label jendela. | Periksa batas bidang jendela dan mask cangkir. Jika hubungan oklusi sesuai gambar, keluarkan piksel cangkir dari #29. Keputusan untuk tidak memberi anotasi dinding di balik kaca tetap dicatat sebagai pilihan saya. |

Semua usulan mengikuti [aturan bagian objek yang terlihat](../../data/custom_test/annotation_guidelines.md). Ambang dalam [daftar irisan mask](occlusion_flags.json) dipakai untuk memprioritaskan review, bukan untuk mengoreksi piksel secara otomatis. Pemeriksaan visual pada skala overlay dapat melewatkan objek kecil atau perbedaan batas tipis.

## Keputusan anotasi yang diperiksa

| Foto | Bukti dan keputusan saat ini |
| --- | --- |
| [P001](overlays/P001.jpg) | Satu anotasi `table-merged` #48 sudah menyimpan empat komponen 4-terhubung dalam satu RLE. Bagian meja dipisahkan secara visual oleh benda di atasnya. Instance tetap dihitung satu kali, sehingga penggabungan tidak diperlukan. |
| [P002](overlays/P002.jpg) | Foto koridor semi-indoor dipertahankan. Anotasi `shelf` #174, #175, #177, dan #178 mengikuti bagian rak yang terlihat sesuai keputusan saya. |
| [P007](overlays/P007.jpg) | Bagian rangka kayu yang ambigu tetap tanpa label `bed`, sesuai keputusan saya. |
| [P010](overlays/P010.jpg) | Panel di atas pintu yang ambigu tetap tanpa label tambahan. |
| [P017](overlays/P017.jpg) | Pengecualian benda kecil yang identitasnya ambigu dipertahankan. |

Pada tahap ini saya dapat menyetujui usulan atau mempertahankan anotasi dengan alasan yang dicatat. Apabila geometri berubah setelah review, statistik, overlay, dan checksum kandidat perlu diperbarui sebelum persetujuan akhir.
