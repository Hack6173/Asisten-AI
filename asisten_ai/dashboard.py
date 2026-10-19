"""
asisten_ai.dashboard - Dashboard Modern Berbahasa Indonesia untuk Asisten-AI Autonomous Intelligence
Menyajikan 11 Halaman Terintegrasi:
1. Dashboard Utama
2. AI Chat (dengan Provenance & Grounding)
3. Autonomous Research
4. Knowledge Explorer
5. Learning History
6. Dataset Manager
7. Model Training (Pemeriksaan Hardware Nyata & Checkpoints)
8. Evaluation Center (Pengujian Tanpa Peeking & Histori)
9. Model Registry (Kandidat vs Model Aktif & Rollback)
10. Resource Monitor (CPU, RAM, GPU/VRAM, Disk)
11. Settings dan Privacy
"""
import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime

# Import Internal Asisten-AI Core Modules
from asisten_ai.config import SYSTEM_CONFIG, DATASETS_DIR
from asisten_ai.resource_monitor import RESOURCE_MONITOR
from asisten_ai.offline_manager import OFFLINE_MANAGER
from asisten_ai.knowledge_store import KNOWLEDGE_STORE
from asisten_ai.retrieval_engine import RETRIEVAL_ENGINE
from asisten_ai.research_agent import RESEARCH_AGENT
from asisten_ai.learning_engine import LEARNING_ENGINE
from asisten_ai.dataset_builder import DATASET_BUILDER
from asisten_ai.training_manager import TRAINING_MANAGER
from asisten_ai.evaluation_engine import EVALUATION_ENGINE
from asisten_ai.model_registry import MODEL_REGISTRY
from asisten_ai.scheduler import SCHEDULER

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Asisten-AI | Autonomous Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS Modern & Elegan
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .badge-online {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-offline {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .status-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Navigasi Utama
with st.sidebar:
    st.markdown("### ⚡ Asisten-AI Autonomous")
    st.markdown("*Sistem Riset & Pembelajaran Mandiri Lokal*")
    st.markdown("---")

    # Status Indikator di Sidebar
    conn = OFFLINE_MANAGER.get_connectivity_status()
    if conn["is_online"]:
        st.markdown(f'<span class="badge-online">🟢 ONLINE: Riset Web Aktif</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-offline">🟠 OFFLINE: Intelijen Lokal</span>', unsafe_allow_html=True)

    active_model = MODEL_REGISTRY.get_active_model()
    st.caption(f"**Model Aktif:** {active_model.model_name if active_model else 'Belum Ada'}")

    st.markdown("---")
    menu = st.radio(
        "Pilih Halaman:",
        [
            "🏠 Dashboard Utama",
            "💬 AI Chat",
            "🔍 Autonomous Research",
            "📚 Knowledge Explorer",
            "📜 Learning History",
            "📦 Dataset Manager",
            "🏋️ Model Training",
            "🎯 Evaluation Center",
            "🏛️ Model Registry",
            "📊 Resource Monitor",
            "⚙️ Settings & Privacy",
        ]
    )

    st.markdown("---")
    st.markdown("#### 🎮 Kontrol Cepat Scheduler")
    sched_info = SCHEDULER.get_status()
    st.caption(f"Mode: **{sched_info.mode.upper()}** | Berjalan: **{'YA' if sched_info.is_running else 'TIDAK'}**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start", use_container_width=True):
            SCHEDULER.start()
            st.rerun()
        if st.button("⏸️ Pause", use_container_width=True):
            SCHEDULER.pause()
            st.rerun()
    with c2:
        if st.button("⏹️ Stop", use_container_width=True):
            SCHEDULER.stop()
            st.rerun()
        if st.button("⏯️ Resume", use_container_width=True):
            SCHEDULER.resume()
            st.rerun()


# =========================================================================
# 1. DASHBOARD UTAMA
# =========================================================================
if menu == "🏠 Dashboard Utama":
    st.markdown('<div class="main-header">⚡ Dashboard Utama Autonomous AI</div>', unsafe_allow_html=True)
    st.write("Pantau aktivitas riset mandiri, pertumbuhan memori lokal, dan evaluasi model secara waktu nyata.")

    stats = KNOWLEDGE_STORE.get_summary_statistics()
    hw = RESOURCE_MONITOR.get_realtime_stats()
    sched = SCHEDULER.get_status()

    # Baris Metrik Utama
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Fakta Terverifikasi", f"{stats['total_facts']:,}")
    with m2:
        st.metric("Sumber Web Diproses", f"{stats['total_sources']:,}")
    with m3:
        st.metric("Topik Dikuasai", f"{stats['total_topics']}")
    with m4:
        st.metric("Rerata Confidence", f"{stats['average_confidence'] * 100:.1f}%")
    with m5:
        st.metric("Eksperimen Pelatihan", f"{stats['total_training_runs']}")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🔄 Status Siklus Pembelajaran Otonom")
        current_job = sched.current_job
        if current_job:
            st.info(f"**Pekerjaan Berjalan:** {current_job.get('type')} pada topik **'{current_job.get('topic')}'** (Mulai: {current_job.get('started_at')[:19]})")
        else:
            st.success("Scheduler siaga menunggu jadwal siklus berikutnya atau pemicu manual.")

        st.write(f"- **Jadwal Siklus Berikutnya:** `{sched.next_scheduled_at or 'Ditentukan oleh scheduler'}`")
        st.write(f"- **Siklus Selesai:** `{sched.total_cycles_completed}` siklus")
        st.write(f"- **Tugas Tertunda di Antrean Offline:** `{conn['pending_research_tasks']}` topik")

        st.markdown("#### ⚡ Aksi Langsung")
        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            if st.button("🚀 Jalankan Siklus Belajar Segera", use_container_width=True):
                with st.spinner("Menjalankan siklus riset & evaluasi otonom..."):
                    try:
                        rep = SCHEDULER.trigger_immediate_cycle()
                        st.success(f"Siklus selesai untuk topik '{rep.topic}'! Fakta baru: {rep.facts_acquired}")
                    except Exception as e:
                        st.error(f"Gagal: {e}")
                st.rerun()

        with action_col2:
            if st.button("🎯 Jalankan Evaluasi Mandiri", use_container_width=True):
                with st.spinner("Menguji model secara independen..."):
                    eval_rep = EVALUATION_ENGINE.run_evaluation()
                    st.success(f"Evaluasi selesai! Skor Akurasi: {eval_rep.accuracy_score * 100:.1f}%")
                st.rerun()

        with action_col3:
            if st.button("📦 Bangun Dataset Training", use_container_width=True):
                with st.spinner("Menyaring fakta dan membagi train/val/test..."):
                    d_res = DATASET_BUILDER.build_and_split_dataset()
                    st.success(f"Dataset dibuat: {d_res.total_samples} sampel (Train: {d_res.train_count})")
                st.rerun()

    with col_right:
        st.subheader("💻 Pemantauan Sumber Daya Nyata")
        st.write(f"**CPU Usage:** {hw['cpu_percent']}%")
        st.progress(min(hw['cpu_percent'] / 100.0, 1.0))

        st.write(f"**RAM Usage:** {hw['ram_percent']}% ({hw['ram_used_mb']:.0f} MB / {hw['ram_total_mb']:.0f} MB)")
        st.progress(min(hw['ram_percent'] / 100.0, 1.0))

        st.write(f"**Penyimpanan Disk Bebas:** {hw['disk_free_gb']} GB")
        st.caption(f"Akselerator: **{hw['gpu_name']}**")

        if sched.active_errors:
            st.error(f"Catatan Error Terakhir: {sched.active_errors[-1]}")


# =========================================================================
# 2. AI CHAT DENGAN GROUNDING & PROVENANCE
# =========================================================================
elif menu == "💬 AI Chat":
    st.markdown('<div class="main-header">💬 AI Chat Terbimbing Fakta Lokal</div>', unsafe_allow_html=True)
    st.write("Bertanya kepada model lokal. Setiap jawaban ditopang oleh kutipan sumber nyata dari memori semantik.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.caption(f"**Kutipan Sumber:** {', '.join(msg['citations'])}")

    user_prompt = st.chat_input("Tulis pertanyaan atau instruksi Anda di sini...")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari fakta terverifikasi dalam basis pengetahuan lokal..."):
                retrieved = RETRIEVAL_ENGINE.search_semantic(user_prompt, top_k=3)
                ans_data = RETRIEVAL_ENGINE.build_grounded_answer(user_prompt, retrieved)

                # Jika fakta belum tersedia, catat gap riset otomatis
                if not ans_data["has_grounding"]:
                    OFFLINE_MANAGER.queue_pending_research(
                        topic=user_prompt[:40],
                        questions=[user_prompt],
                        origin="chat_unanswered"
                    )

                st.markdown(ans_data["answer"])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ans_data["answer"],
                    "citations": ans_data["citations"],
                })


# =========================================================================
# 3. AUTONOMOUS RESEARCH
# =========================================================================
elif menu == "🔍 Autonomous Research":
    st.markdown('<div class="main-header">🔍 Autonomous Research Agent</div>', unsafe_allow_html=True)
    st.write("Agen riset mandiri yang mampu menemukan kesenjangan pengetahuan, menyusun rencana, dan meneliti web secara sah.")

    gaps = RESEARCH_AGENT.identify_knowledge_gaps()
    st.subheader("💡 Kesenjangan Pengetahuan yang Teridentifikasi (Knowledge Gaps)")
    if gaps:
        for idx, g in enumerate(gaps[:5], 1):
            with st.expander(f"{idx}. Topik: {g['topic']} (Alasan: {g.get('reason', g.get('rationale', 'Kebutuhan belajar'))})"):
                st.write("**Pertanyaan Target:**")
                for q in g.get("questions", []):
                    st.write(f"- {q}")
    else:
        st.info("Seluruh topik dasar saat ini telah dipelajari dengan baik.")

    st.markdown("---")
    st.subheader("🚀 Picu Riset Topik Baru")
    c_top, c_btn = st.columns([3, 1])
    with c_top:
        custom_topic = st.text_input("Nama Topik Penelitian:", value="Large Language Model Architecture")
    with c_btn:
        st.write("")
        st.write("")
        run_research_btn = st.button("Mulai Riset Web", use_container_width=True)

    if run_research_btn:
        with st.spinner(f"Menjalankan riset mendalam untuk '{custom_topic}'..."):
            res = RESEARCH_AGENT.execute_autonomous_research(custom_topic)
            if res.status == "completed":
                st.success(f"Riset Selesai ({res.duration_seconds}s)! Berhasil mengambil {res.sources_fetched} sumber dan menyimpan {res.facts_stored} fakta baru.")
                if res.contradictions.has_contradictions:
                    st.warning("Kontradiksi Ditemukan:\n" + "\n".join(res.contradictions.contradiction_details))
                st.write("**Pertanyaan Lanjutan untuk Riset Berikutnya:**")
                for fq in res.follow_up_questions:
                    st.write(f"- {fq}")
            elif res.status == "queued_offline":
                st.warning(res.error_message)
            else:
                st.error(f"Gagal melakukan riset: {res.error_message}")


# =========================================================================
# 4. KNOWLEDGE EXPLORER
# =========================================================================
elif menu == "📚 Knowledge Explorer":
    st.markdown('<div class="main-header">📚 Penjelajah Basis Pengetahuan (Knowledge Explorer)</div>', unsafe_allow_html=True)
    st.write("Periksa fakta semantik terverifikasi, lacak nomor versi, tingkat kepercayaan, dan asal-usul sumber (provenance).")

    items = KNOWLEDGE_STORE.list_knowledge_items(limit=500)
    if items:
        df = pd.DataFrame(items)
        cols_to_show = ["id", "topic", "concept", "content_category", "confidence_score", "version", "source_domain"]
        avail_cols = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[avail_cols], use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Detail Fakta & Manajemen")
        selected_id = st.selectbox("Pilih ID Fakta untuk Diperiksa:", df["id"].tolist())
        selected_item = next((i for i in items if i["id"] == selected_id), None)

        if selected_item:
            st.json(selected_item)
            del_c1, exp_c2 = st.columns([1, 4])
            with del_c1:
                if st.button("🗑️ Hapus Fakta Ini", type="primary"):
                    KNOWLEDGE_STORE.delete_knowledge_item(selected_id)
                    st.success(f"Fakta ID {selected_id} dihapus.")
                    st.rerun()
            with exp_c2:
                json_str = KNOWLEDGE_STORE.export_knowledge_json()
                st.download_button(
                    label="📥 Ekspor Seluruh Pengetahuan (JSON)",
                    data=json_str,
                    file_name="asisten_ai_knowledge_export.json",
                    mime="application/json"
                )
    else:
        st.info("Belum ada pengetahuan yang tersimpan. Jalankan riset terlebih dahulu.")


# =========================================================================
# 5. LEARNING HISTORY
# =========================================================================
elif menu == "📜 Learning History":
    st.markdown('<div class="main-header">📜 Riwayat Siklus Pembelajaran Mandiri</div>', unsafe_allow_html=True)
    st.write("Catatan audit episodik dari setiap siklus riset, materi baru yang diserap, dan evaluasi pemahaman.")

    with KNOWLEDGE_STORE._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM research_sessions ORDER BY started_at DESC LIMIT 50")
        sessions = [dict(r) for r in cursor.fetchall()]

    if sessions:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True)
    else:
        st.info("Belum ada sesi pembelajaran otonom yang tercatat.")


# =========================================================================
# 6. DATASET MANAGER
# =========================================================================
elif menu == "📦 Dataset Manager":
    st.markdown('<div class="main-header">📦 Pengelola Dataset Pelatihan (Dataset Manager)</div>', unsafe_allow_html=True)
    st.write("Membangun dataset instruksi dari sumber terpercaya dengan isolasi anti kebocoran data (zero data-leakage).")

    if st.button("🔄 Bangun Ulang Dataset (Train / Val / Test Split)"):
        with st.spinner("Memproses fakta lokal dan membagi subset..."):
            d_res = DATASET_BUILDER.build_and_split_dataset()
            st.success(f"Dataset '{d_res.dataset_name}' Berhasil Dibangun!")
            st.write(f"- Total Sampel: **{d_res.total_samples}**")
            st.write(f"- Training Set (70%): **{d_res.train_count}**")
            st.write(f"- Validation Set (15%): **{d_res.val_count}**")
            st.write(f"- Test Set (15%): **{d_res.test_count}**")
            st.write(f"- Duplikat Dihapus: **{d_res.deduplicated_removed}**")
            st.write(f"- Data Tidak Terverifikasi Disaring: **{d_res.filtered_unverified}**")

    # Tampilkan berkas dataset yang ada di direktori
    st.subheader("📂 Berkas Dataset Tersimpan")
    files = list(DATASETS_DIR.glob("*.jsonl"))
    if files:
        for f in files:
            size_kb = f.stat().st_size / 1024
            st.write(f"📄 `{f.name}` ({size_kb:.1f} KB)")
    else:
        st.info("Belum ada berkas dataset yang dibuat.")


# =========================================================================
# 7. MODEL TRAINING
# =========================================================================
elif menu == "🏋️ Model Training":
    st.markdown('<div class="main-header">🏋️ Pelatihan Model Mandiri (Self-Training)</div>', unsafe_allow_html=True)
    st.write("Fine-tuning model secara nyata dengan pemeriksaan keamanan sumber daya RAM/VRAM dan checkpointing.")

    hw_rep = RESOURCE_MONITOR.assess_hardware_capabilities()
    st.subheader("🔍 Audit Kelayakan Hardware Sebelum Pelatihan")
    st.info(f"**Rekomendasi Sistem:** {hw_rep.recommendation}")

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.write(f"- Total RAM: **{hw_rep.total_ram_gb:.1f} GB** (Bebas: **{hw_rep.free_ram_gb:.1f} GB**)")
        st.write(f"- Disk Tersedia: **{hw_rep.disk_free_gb:.1f} GB**")
        st.write(f"- Dukungan QLoRA GPU: **{'YA' if hw_rep.can_run_qlora else 'TIDAK'}**")
    with t_col2:
        st.write(f"- Dukungan Adapter CPU LoRA: **{'YA' if hw_rep.can_run_cpu_lora else 'RAM TERBATAS'}**")
        st.write(f"- Nama Perangkat Komputasi: **{hw_rep.gpu_name}**")

    st.markdown("---")
    st.subheader("🚀 Konfigurasi & Eksekusi Pelatihan")

    epochs = st.slider("Jumlah Epoch:", min_value=1, max_value=5, value=2)
    lr = st.selectbox("Learning Rate:", [0.001, 0.0005, 0.0002])

    train_c1, train_c2 = st.columns(2)
    with train_c1:
        if st.button("⚡ Latih Model Kandidat Sekarang", type="primary", use_container_width=True):
            train_f = str(DATASETS_DIR / "asisten_ai_autonomous_v1_train.jsonl")
            val_f = str(DATASETS_DIR / "asisten_ai_autonomous_v1_val.jsonl")

            with st.spinner("Menjalankan backpropagation & pemantauan batas RAM..."):
                res = TRAINING_MANAGER.run_training_experiment(
                    dataset_name="asisten_ai_autonomous_v1",
                    train_file=train_f,
                    val_file=val_f,
                    epochs=epochs,
                    learning_rate=lr
                )

                if res.status == "completed":
                    st.success(f"Pelatihan Selesai ({res.duration_seconds}s)! Checkpoint tersimpan di: `{res.checkpoint_dir}`")
                    st.write(f"- Langkah Total: **{res.total_steps}**")
                    st.write(f"- Final Loss: **{res.final_loss:.4f}**")
                    st.write(f"- Peak RAM: **{res.peak_ram_mb:.0f} MB**")

                    # Daftarkan sebagai kandidat
                    MODEL_REGISTRY.register_candidate(
                        candidate_name=f"Candidate-Adapter-{int(time.time())}",
                        version="1.1.0",
                        checkpoint_dir=res.checkpoint_dir,
                        description=f"Model kandidat hasil pelatihan mandiri ({res.total_steps} langkah)."
                    )
                    st.info("Model otomatis terdaftar di Model Registry sebagai Kandidat.")
                elif res.status == "aborted_insufficient_resources":
                    st.error(f"Pelatihan Dibatalkan Otomatis: {res.error_message}")
                else:
                    st.error(f"Pelatihan Gagal: {res.error_message}")

    with train_c2:
        if st.button("🛑 Hentikan Pelatihan Darurat (Stop)", use_container_width=True):
            TRAINING_MANAGER.interrupt_training()
            st.warning("Perintah penghentian pelatihan dikirim.")


# =========================================================================
# 8. EVALUATION CENTER
# =========================================================================
elif menu == "🎯 Evaluation Center":
    st.markdown('<div class="main-header">🎯 Pusat Evaluasi & Pengujian Mandiri</div>', unsafe_allow_html=True)
    st.write("Menguji model secara mandiri tanpa melihat contekan saat tes untuk mengukur akurasi, konsistensi, dan regresi.")

    if st.button("🚀 Jalankan Uji Mandiri Lengkap (Test Suite)", type="primary"):
        with st.spinner("Menjalankan evaluasi independen..."):
            eval_rep = EVALUATION_ENGINE.run_evaluation()
            st.success(f"Evaluasi Selesai! Skor Keseluruhan: **{eval_rep.overall_score * 100:.1f}%**")

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Akurasi Faktual", f"{eval_rep.accuracy_score * 100:.1f}%")
            with e2:
                st.metric("Konsistensi Logika", f"{eval_rep.consistency_score * 100:.1f}%")
            with e3:
                st.metric("Kepatuhan Instruksi", f"{eval_rep.instruction_following_score * 100:.1f}%")
            with e4:
                st.metric("Grounding Sitasi", f"{eval_rep.citation_grounding_score * 100:.1f}%")

            if eval_rep.weak_topics:
                st.warning(f"Kelemahan Terdeteksi pada Topik: {', '.join(eval_rep.weak_topics)}")

    st.markdown("---")
    st.subheader("📜 Riwayat Pengujian Mandiri Sebelumnya")
    eval_hist = EVALUATION_ENGINE.get_evaluation_history()
    if eval_hist:
        st.dataframe(pd.DataFrame(eval_hist), use_container_width=True)
    else:
        st.info("Belum ada histori pengujian mandiri.")


# =========================================================================
# 9. MODEL REGISTRY
# =========================================================================
elif menu == "🏛️ Model Registry":
    st.markdown('<div class="main-header">🏛️ Registri Model & Gerbang Promosi</div>', unsafe_allow_html=True)
    st.write("Model baru tidak langsung menggantikan model utama sebelum lulus uji regresi dan keamanan.")

    models = MODEL_REGISTRY.list_models()
    if models:
        for m in models:
            badge = "🟢 AKTIF" if m.is_active else ("🟡 KANDIDAT" if m.is_candidate else "⚪ ARSIP")
            with st.expander(f"{badge} | {m.model_name} (v{m.version}) - Skor: {m.evaluation_score * 100:.1f}%"):
                st.write(f"- **ID:** `{m.id}`")
                st.write(f"- **Path Checkpoint:** `{m.model_path}`")
                st.write(f"- **Deskripsi:** {m.description}")
                st.write(f"- **Tanggal Dibuat:** {m.created_at[:19]}")

                if m.is_candidate:
                    if st.button(f"✨ Promosikan Menjadi Model Utama ({m.id})", key=f"prom_{m.id}"):
                        with st.spinner("Menjalankan gerbang promosi dan tes regresi..."):
                            success, msg, _ = MODEL_REGISTRY.evaluate_and_promote_candidate(m.id)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
                        st.rerun()

                if not m.is_active and not m.is_candidate:
                    if st.button(f"⏪ Rollback ke Model Ini ({m.id})", key=f"roll_{m.id}"):
                        ok, msg = MODEL_REGISTRY.rollback_to_previous_model(m.id)
                        if ok:
                            st.warning(msg)
                        else:
                            st.error(msg)
                        st.rerun()
    else:
        st.info("Tidak ada model terdaftar.")


# =========================================================================
# 10. RESOURCE MONITOR
# =========================================================================
elif menu == "📊 Resource Monitor":
    st.markdown('<div class="main-header">📊 Pemantau Sumber Daya Perangkat Keras</div>', unsafe_allow_html=True)
    st.write("Pengukuran nyata metrik CPU, RAM, GPU/VRAM, dan penyimpanan lokal tanpa simulasi.")

    real_hw = RESOURCE_MONITOR.get_realtime_stats()

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.metric("Penggunaan CPU", f"{real_hw['cpu_percent']}%")
        st.metric("Total RAM Fisik", f"{real_hw['ram_total_mb'] / 1024:.1f} GB")
    with rc2:
        st.metric("Penggunaan RAM", f"{real_hw['ram_percent']}%")
        st.metric("RAM Bebas Saat Ini", f"{real_hw['ram_free_mb'] / 1024:.2f} GB")
    with rc3:
        st.metric("Sisa Ruang Disk", f"{real_hw['disk_free_gb']} GB")
        st.metric("Status Komputasi", real_hw["gpu_name"])

    st.markdown("---")
    st.subheader("🛡️ Evaluasi Batas Keamanan Eksekusi")
    is_safe, reason = RESOURCE_MONITOR.is_safe_to_execute()
    if is_safe:
        st.success(f"Status Sistem: {reason}")
    else:
        st.error(f"Peringatan Sumber Daya: {reason}")


# =========================================================================
# 11. SETTINGS & PRIVACY
# =========================================================================
elif menu == "⚙️ Settings & Privacy":
    st.markdown('<div class="main-header">⚙️ Pengaturan & Kebijakan Privasi</div>', unsafe_allow_html=True)
    st.write("Konfigurasi mesin pencari, batas otonomi, jam aktif, dan proteksi privasi data lokal.")

    with st.form("settings_form"):
        st.subheader("🌐 Penyedia Pencarian Web")
        search_prov = st.selectbox("Search Provider:", ["duckduckgo", "searxng", "mock"], index=["duckduckgo", "searxng", "mock"].index(SYSTEM_CONFIG.search.provider))
        searx_url = st.text_input("SearXNG URL (jika dipilih):", value=SYSTEM_CONFIG.search.searxng_url)
        search_key = st.text_input("API Key (opsional):", value=SYSTEM_CONFIG.search.api_key, type="password")

        st.subheader("⏱️ Pengaturan Penjadwal Pembelajaran")
        sched_mode = st.selectbox("Mode Pembelajaran:", ["manual", "scheduled", "autonomous", "offline"], index=["manual", "scheduled", "autonomous", "offline"].index(SYSTEM_CONFIG.scheduler.mode))
        interval_min = st.number_input("Interval Antar Siklus (menit):", min_value=5, max_value=1440, value=SYSTEM_CONFIG.scheduler.interval_minutes)
        h_start = st.number_input("Jam Mulai Aktif (0-23):", min_value=0, max_value=23, value=SYSTEM_CONFIG.scheduler.active_hours_start)
        h_end = st.number_input("Jam Berakhir Aktif (0-23):", min_value=0, max_value=23, value=SYSTEM_CONFIG.scheduler.active_hours_end)

        st.subheader("🛡️ Keamanan & Privasi Data")
        allow_search = st.checkbox("Izinkan Riset Online Keluar", value=SYSTEM_CONFIG.privacy.allow_internet_search)
        strip_injections = st.checkbox("Sanitasi Konten Web dari Prompt Injection", value=SYSTEM_CONFIG.privacy.strip_unsafe_prompts)
        sandbox_code = st.checkbox("Blokir Eksekusi Kode Berbahaya dari Dokumen Web", value=SYSTEM_CONFIG.privacy.sandbox_code_execution)

        submitted = st.form_submit_button("Simpan Pengaturan")
        if submitted:
            SYSTEM_CONFIG.search.provider = search_prov
            SYSTEM_CONFIG.search.searxng_url = searx_url
            SYSTEM_CONFIG.search.api_key = search_key

            SYSTEM_CONFIG.scheduler.mode = sched_mode
            SYSTEM_CONFIG.scheduler.interval_minutes = interval_min
            SYSTEM_CONFIG.scheduler.active_hours_start = h_start
            SYSTEM_CONFIG.scheduler.active_hours_end = h_end

            SYSTEM_CONFIG.privacy.allow_internet_search = allow_search
            SYSTEM_CONFIG.privacy.strip_unsafe_prompts = strip_injections
            SYSTEM_CONFIG.privacy.sandbox_code_execution = sandbox_code

            SYSTEM_CONFIG.save()
            st.success("Pengaturan berhasil disimpan!")
            st.rerun()
