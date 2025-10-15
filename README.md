# nlp-kategori
Machine learning NLP mengklasifikasikan produk di marketplace pemerintah berdasarkan nama dan deskripsi, sehingga kategori barang teridentifikasi secara akurat. Ini mendukung pengenaan pajak yang lebih tepat, efisien, dan sesuai regulasi, serta mengurangi kesalahan klasifikasi manual.

## Strategi 3: Klasifikasi Multi-Tahap Hierarkis (Versi Detail)
Strategi ini memecah tugas klasifikasi kategori pajak menjadi beberapa tahap, mengikuti struktur taksonomi produk (misal: Kategori Utama → Subkategori → Kode Pajak). Setiap tahap menggunakan model khusus yang hanya fokus pada ruang label yang relevan, sehingga akurasi meningkat untuk kategori yang sulit dibedakan.

### 1. Arsitektur Sistem
1. **Tahap 1 — Kategori Utama**
   - *Model*: Encoder transformer ringan (mis. IndoBERT lite) dengan kepala klasifikasi softmax.
   - *Input*: Nama produk dan ringkasan deskripsi (maks. 128 token) yang sudah dinormalisasi.
   - *Output*: Probabilitas untuk setiap kategori utama (mis. "Elektronik", "Kesehatan", "Makanan").

2. **Tahap 2 — Subkategori Khusus**
   - *Model*: Satu model per kategori utama, bisa berupa fine-tuned transformer atau kombinasi TF-IDF + LightGBM, tergantung ukuran data.
   - *Input*: Deskripsi penuh (hingga 256–384 token) + embedding dari tahap sebelumnya.
   - *Output*: Label subkategori (mis. "Laptop", "Perangkat Medis", "Makanan Kaleng").

3. **Tahap 3 — Kode Pajak**
   - *Model*: Head klasifikasi khusus yang memetakan subkategori ke kode pajak final. Untuk subkategori dengan data besar, gunakan model transformer kecil; untuk subkategori langka, gunakan rule-based/nearest neighbor dari embedding.
   - *Input*: Representasi gabungan (embedding tahap 2 + fitur domain seperti HS code, kata kunci pajak).
   - *Output*: Kode pajak akhir (mis. "PPN 10%", "PPN 0%", "PPNBM").

4. **Router & Ensemble**
   - Jika kepercayaan tahap 1 < ambang (mis. 0,4), kirim ke *fallback* model generalis (IndoBERT besar) atau voting ensemble untuk menurunkan risiko salah rute.
   - Simpan jejak kepercayaan (logit) sebagai fitur tambahan untuk tahap berikutnya.

### 2. Pipeline Data
1. **Pre-processing Bahasa Indonesia**
   - Normalisasi ejaan, hilangkan HTML/emoji, lakukan stemming ringan atau lemmatization menggunakan *Sastrawi*.
   - Tangani campuran bahasa dengan deteksi bahasa; jika non-Indonesian, terjemahkan otomatis atau rute ke model khusus.

2. **Augmentasi Data**
   - *Back-translation* (ID ↔ EN) untuk memperkaya kosakata.
   - *Synonym replacement* memakai kamus KBBI atau slang marketplace.
   - *Mixup teks* (menggabung dua deskripsi dengan bobot) untuk memperbaiki generalisasi.

3. **Pembagian Dataset**
   - Gunakan stratified split per level hierarki.
   - Sisihkan *validation* khusus tiap tahap agar metrik per level terukur.

### 3. Strategi Pelatihan
1. **Tahap 1**
   - Fine-tune encoder dengan *focal loss* atau *class-weighted cross entropy* jika tidak seimbang.
   - Gunakan *early stopping* berbasis macro-F1 kategori utama.

2. **Tahap 2**
   - Latih model per cabang dengan data yang disaring dari tahap 1 (ground truth saat training, prediksi saat inference).
   - Eksperimen *knowledge distillation* dari model besar ke model kecil agar inference cepat.

3. **Tahap 3**
   - Gunakan *label smoothing* untuk mengurangi overfitting pada kode pajak mirip.
   - Gabungkan fitur numerik (harga, satuan) jika tersedia dengan concatenation sebelum lapisan akhir.

4. **Kalibrasi Probabilitas**
   - Setelah pelatihan, lakukan *temperature scaling* untuk setiap tahap agar probabilitas antar level konsisten.

### 4. Evaluasi & Monitoring
1. **Metrik**
   - Macro-F1 dan weighted-F1 per tahap.
   - *Hierarchical precision/recall*: skor penuh jika semua level benar, skor parsial jika hanya level atas benar.

2. **Analisis Error**
   - Buat confusion matrix per tahap; fokus pada kesalahan silang antar subkategori yang mirip.
   - Telusuri contoh dengan confidence rendah untuk memperbaiki router.

3. **Monitoring Produksi**
   - Log distribusi label per tahap dan bandingkan dengan baseline.
   - Buat *drift detector* (mis. Population Stability Index) untuk mendeteksi perubahan bahasa atau produk baru.

### 5. Deployment & MLOps
1. **Inference Service**
   - Susun pipeline mikroservis: Service A (tahap 1) → Service B (tahap 2) → Service C (tahap 3).
   - Gunakan *async queue* untuk fallback/ensemble agar latensi tetap rendah (<300 ms).

2. **Model Registry & Versioning**
   - Simpan setiap model tahap di registry (MLflow, Vertex AI Model Registry) dengan metadata versi, dataset, dan metrik.
   - Implementasikan *A/B testing* sebelum mengganti model produksi.

3. **Feedback Loop**
   - Integrasikan antarmuka peninjau pajak untuk memberi label koreksi; gunakan untuk *continual learning*.
   - Terapkan *active learning* dengan memilih sampel ber-confidence rendah untuk diberi label manual.

### 6. Roadmap Implementasi Bertahap
1. **Minggu 1–2: Eksplorasi & Desain**
   - Lakukan analisis eksploratif pada distribusi label tiap level untuk memvalidasi asumsi hierarki.
   - Bangun *baseline* cepat (TF-IDF + Logistic Regression) guna mendapatkan tolok ukur awal dan data error.

2. **Minggu 3–4: Prototipe Model Tahap 1**
   - Fine-tune IndoBERT-lite dengan eksperimen hyperparameter (learning rate, warmup, ukuran batch) menggunakan *sweep* otomatis.
   - Ukur metrik latensi pada hardware target (CPU vs GPU) untuk menentukan konfigurasi inference.

3. **Minggu 5–6: Model Tahap 2 Spesifik Cabang**
   - Pilih arsitektur per cabang berdasarkan ukuran data: gunakan transformer untuk cabang besar, gradient boosting untuk cabang kecil.
   - Rancang *data loader* yang otomatis memfilter data berdasarkan label ground truth sehingga pipeline training konsisten.

4. **Minggu 7–8: Integrasi Tahap 3 & Fitur Tambahan**
   - Bangun modul ekstraksi fitur numerik (harga, berat, volume) dan lakukan normalisasi sebelum digabung dengan embedding teks.
   - Implementasikan fallback rule-based untuk cabang dengan jumlah sampel <100 guna menjaga cakupan kode pajak.

5. **Minggu 9–10: MLOps & Deployment**
   - Siapkan *CI/CD* yang menjalankan tes unit pada pipeline preprocessing serta validasi model (mis. minimum F1 per level).
   - Automasi pembuatan paket model (ONNX/TorchScript) dan skenario canary release di lingkungan staging.

### 7. Dokumentasi & Observabilitas
- **Playbook Operasional**: Sediakan panduan langkah demi langkah bagi tim operasional untuk menangani kasus fallback, retraining, dan rollback.
- **Dashboards**: Buat dasbor di Grafana/Looker dengan metrik per tahap (akurasi, latensi, jumlah fallback) serta *alerting* berbasis ambang.
- **Pelacakan Data**: Gunakan alat seperti Great Expectations untuk memvalidasi kualitas data harian (panjang teks, proporsi bahasa, kelengkapan kolom).

### 8. Pertimbangan Tambahan
- **Skalabilitas**: Distribusikan pelatihan per cabang secara paralel; model kecil per cabang memudahkan *horizontal scaling*.
- **Kualitas Data**: Sisipkan modul deteksi anomali teks (kata kasar, huruf random) untuk dibersihkan sebelum inference.
- **Keamanan & Audit**: Simpan audit trail prediksi per tahap untuk kepatuhan regulasi.

Strategi ini memberikan kerangka kerja terukur yang memanfaatkan struktur taksonomi pajak, memungkinkan peningkatan akurasi dan interpretabilitas sekaligus menjaga latensi inference yang dapat diterima.
