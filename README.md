# ⚡ Asisten-AI: Autonomous Intelligence System
### Autonomous Research, Self-Learning, Self-Evaluation, and Continuous Model Training

**Asisten-AI Autonomous Intelligence** adalah sistem kecerdasan buatan lokal yang dirancang bukan sebagai chatbot pasif biasa, melainkan sebagai entitas otonom yang mampu:
1. **Mencari Informasi Sendiri (*Autonomous Research*):** Menentukan kesenjangan pengetahuan (*knowledge gaps*), menyusun rencana riset terstruktur, mencari web secara legal melalui antarmuka mesin pencari (DuckDuckGo / SearXNG), menyaring prompt injection, dan mendeteksi kontradiksi antar-sumber.
2. **Mesin Pembelajaran Mandiri (*Autonomous Learning Engine*):** Mengotomatisasi siklus 11-langkah mulai dari identifikasi topik, ekstraksi fakta, validasi kredibilitas domain, deduplikasi data, pembuatan pasangan uji pemahaman, hingga kalibrasi skor kepercayaan (*confidence score*).
3. **Penyimpanan Memori Berlapis (*Multi-Layer Memory Architecture*):**
   - **Working Memory:** Konteks inferensi sesi aktif.
   - **Episodic Memory:** Riwayat siklus riset dan catatan eksperimen.
   - **Semantic Memory:** Fakta terverifikasi dengan pelacakan asal-usul (*source provenance*) dan nomor versi (*knowledge versioning*).
   - **Source Memory:** URL, domain, tanggal pencarian, hash isi, dan skor reliabilitas.
   - **Training Memory:** Pasangan QA, partisi train/val/test, dan bobot model checkpoint.
   - **Evaluation Memory:** Log hasil pengujian mandiri tanpa *peeking* dan pelacakan regresi.
4. **Pelatihan Model Mandiri (*Continuous Self-Training Pipeline*):**
   - **Dataset Builder:** Mengumpulkan contoh fakta berreferensi sah, menyaring data unverified/tanpa sumber, menjamin *zero data-leakage* antar subset (Train 70% / Val 15% / Test 15%).
   - **Training Manager:** Pra-pemeriksaan kelayakan hardware (RAM, VRAM, Disk), penyesuaian adaptif (LoRA / Lightweight Neural Adapter), pemantauan loss waktu nyata, dan interupsi darurat aman.
   - **Model Registry & Promotion Gate:** Model hasil pelatihan disimpan sebagai kandidat, diuji secara objektif, dan hanya dipromosikan jika melampaui performa model aktif tanpa regresi. Dilengkapi fitur rollback instan.
5. **Evaluasi Mandiri Objektif (*Self-Evaluation Engine*):** Menguji kemampuan model tanpa melihat contekan (*zero-peeking*), mengukur akurasi, konsistensi logika, kepatuhan instruksi, dan *grounding* kutipan.
6. **Intelijen Offline & Komplementer Online:** Ketika offline, AI menjawab berbasis model dan fakta lokal tanpa mengarang (*no hallucination*), serta mencatat topik belum terjawab ke antrean offline untuk diteliti otomatis saat koneksi internet kembali aktif.

---

## 🏛️ Arsitektur Modul Sistem

Sistem dipecah menjadi modul independen berstandar tinggi di dalam direktori `asisten_ai/`:

```
asisten_ai/
├── config.py              # Konfigurasi terpusat, ambang batas hardware, & persistensi JSON
├── resource_monitor.py    # Pemantauan real-time CPU, RAM, GPU/VRAM, & penilaian kelayakan
├── offline_manager.py     # Deteksi konektivitas jaringan & antrean riset tertunda (queue)
├── search_provider.py     # Antarmuka penyedia pencarian (DuckDuckGo, SearXNG, Mock)
├── source_validator.py    # Sanitasi prompt injection, verifikasi kredibilitas, & deteksi kontradiksi
├── knowledge_store.py     # Basis data relasional SQLite berlapis, deduplikasi, & versioning
├── retrieval_engine.py    # Temu kembali semantik (TF-IDF Cosine Similarity) & grounding jawaban
├── research_agent.py      # Agen riset otonom, penjelajah web aman, & ekstraksi sitasi
├── learning_engine.py     # Orkestrator 11-langkah siklus pembelajaran mandiri
├── dataset_builder.py     # Pembuat dataset instruksi dengan pemisahan Train/Val/Test ketat
├── training_manager.py    # Eksekutor training PyTorch/LoRA, loss tracking, & checkpointing
├── evaluation_engine.py   # Mesin benchmark mandiri tanpa peeking & pendeteksi regresi
├── model_registry.py      # Manajemen model aktif vs kandidat, gerbang promosi, & rollback
├── scheduler.py           # Job queue scheduler persisten tahan restart & kontrol pause/resume
├── app_server.py          # Backend REST API berbasis FastAPI
└── dashboard.py           # Dashboard modern interaktif Streamlit berbahasa Indonesia (11 Halaman)
```

---

## 🖥️ Spesifikasi & Hasil Audit Hardware

Berdasarkan audit nyata sistem lokal:
- **Sistem Operasi:** Windows 10/11 (AMD64)
- **Prosesor:** AMD Athlon / Ryzen Architecture (Multi-core)
- **RAM Total:** ~6.0 GB (Bebas: ~0.8 - 1.2 GB)
- **Penyimpanan Bebas:** >55.0 GB
- **Akselerator:** CPU Mode (PyTorch 2.14.1+cpu)
- **Kebijakan Pelatihan Realistis:** Mengingat VRAM GPU diskrit tidak tersedia dan RAM fisik 6GB, sistem secara jujur menerapkan model adapter neural ringan terisolasi dan *local semantic embedding*, bukan memaksakan fine-tuning LLM multi-miliar parameter yang dapat memicu crash sistem operasi (Out of Memory).

---

## 🚀 Panduan Instalasi & Menjalankan Sistem

### 1. Prasyarat Lingkungan
Sistem menggunakan Python 3.11 yang terkelola di dalam `.venv/`:
```powershell
# Mengaktifkan virtual environment
.\.venv\Scripts\activate
```

### 2. Menjalankan Dashboard Streamlit (11 Halaman Lengkap)
Jalankan perintah berikut:
```powershell
python run_asisten.py --mode dashboard
```
atau klik dua kali file **`start.bat`**.

Buka browser pada alamat:
👉 **[http://localhost:8501](http://localhost:8501)**

Halaman yang tersedia di dashboard:
1. **Dashboard Utama:** Metrik ringkas, status konektivitas, kontrol siklus, dan penggunaan hardware.
2. **AI Chat:** Berinteraksi dengan model yang didukung kutipan fakta lokal (*grounding*).
3. **Autonomous Research:** Memilih atau meriset topik otonom, membaca web, dan melihat kontradiksi.
4. **Knowledge Explorer:** Menjelajah fakta, melihat nomor versi, confidence, serta ekspor JSON.
5. **Learning History:** Audit log setiap siklus belajar yang pernah berlangsung.
6. **Dataset Manager:** Membangun dataset QA, melihat statistik Train/Val/Test.
7. **Model Training:** Audit kelayakan hardware, eksekusi pelatihan kandidat, dan monitor loss.
8. **Evaluation Center:** Menjalankan pengujian mandiri objektif dan memantau histori akurasi.
9. **Model Registry:** Menguji gerbang promosi model kandidat dan tombol *Rollback*.
10. **Resource Monitor:** Grafik nyata beban CPU, RAM fisik, VRAM, dan sisa disk.
11. **Settings & Privacy:** Pengaturan mesin pencari, jam aktif scheduler, dan filter keamanan.

### 3. Menjalankan Backend FastAPI REST API
Jika ingin mengakses endpoint API untuk integrasi aplikasi eksternal:
```powershell
python run_asisten.py --mode api --api-port 8000
```
Dokumentasi interaktif Swagger API tersedia di:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 Pengujian Otomatis (Automated Testing)

Seluruh komponen telah diuji menggunakan `pytest` dengan 11 pengujian otomatis terverifikasi:
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

Hasil verifikasi:
```
tests/test_autonomous_system.py::test_resource_monitor PASSED            [  9%]
tests/test_autonomous_system.py::test_source_validator_and_sanitizer PASSED [ 18%]
tests/test_autonomous_system.py::test_source_contradiction_detection PASSED [ 27%]
tests/test_autonomous_system.py::test_knowledge_store_deduplication_and_versioning PASSED [ 36%]
tests/test_autonomous_system.py::test_retrieval_engine PASSED            [ 45%]
tests/test_autonomous_system.py::test_dataset_builder_anti_leakage PASSED [ 54%]
tests/test_autonomous_system.py::test_training_manager_and_checkpoints PASSED [ 63%]
tests/test_autonomous_system.py::test_evaluation_engine_and_model_registry PASSED [ 72%]
tests/test_autonomous_system.py::test_offline_manager_queue PASSED       [ 81%]
tests/test_e2e_learning_cycle.py::test_full_autonomous_learning_cycle PASSED [ 90%]
tests/test_e2e_learning_cycle.py::test_scheduler_restart_recovery PASSED [100%]

============================= 11 passed in 8.65s ==============================
```

---

## 🛡️ Keamanan & Batas Otonomi

1. **Sanitasi Prompt Injection:** Konten dari situs web dipindai dengan regex khusus untuk membersihkan upaya injeksi prompt (seperti *"ignore previous instructions"* atau injeksi tag script).
2. **Pencegahan Eksekusi Kode Sembarangan:** Teks dari web hanya diperlakukan sebagai string data, tanpa pemanggilan `eval()` atau `exec()`.
3. **Privasi & Isolasi:** Tidak ada telemetri rahasia atau pengunggahan data lokal ke server luar. Seluruh database SQLite dan bobot model tersimpan di dalam folder `data/`.
4. **Pencegahan Crash Sistem:** Scheduler dan Training Manager secara aktif memeriksa ambang batas RAM sebelum dan selama proses komputasi berlangsung.