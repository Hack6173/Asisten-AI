"""
asisten_ai.research_agent - Agen Riset Otonom (Autonomous Research Agent)
Mampu menentukan topik yang belum dikuasai, merumuskan rencana riset, mencari web,
mengunduh dan membaca halaman dokumen secara aman, membandingkan sumber independen,
mendeteksi kontradiksi, serta menyimpan pengetahuan baru beserta provenance dan pertanyaan lanjutan.
"""
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from pydantic import BaseModel, Field

from asisten_ai.config import SYSTEM_CONFIG
from asisten_ai.offline_manager import OFFLINE_MANAGER
from asisten_ai.search_provider import get_search_provider, SearchResultItem
from asisten_ai.source_validator import SOURCE_VALIDATOR, ExtractedSource, ContradictionReport
from asisten_ai.knowledge_store import KNOWLEDGE_STORE

logger = logging.getLogger(__name__)


class ResearchPlan(BaseModel):
    topic: str
    target_questions: List[str]
    rationale: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ResearchSessionResult(BaseModel):
    session_id: str
    topic: str
    questions: List[str]
    sources_fetched: int
    facts_stored: int
    contradictions: ContradictionReport
    follow_up_questions: List[str]
    status: str
    duration_seconds: float
    error_message: Optional[str] = None


class AutonomousResearchAgent:
    def __init__(
        self,
        search_provider=None,
        validator=SOURCE_VALIDATOR,
        store=KNOWLEDGE_STORE,
        offline_mgr=OFFLINE_MANAGER,
    ):
        self.search_provider = search_provider or get_search_provider()
        self.validator = validator
        self.store = store
        self.offline_mgr = offline_mgr

    def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """
        Menentukan topik dan pertanyaan yang belum dikuasai atau memiliki skor evaluasi rendah.
        Menganalisis basis pengetahuan lokal dan histori evaluasi sebelumnya.
        """
        gaps = []
        stats = self.store.get_summary_statistics()

        # Daftar topik kurikulum standar kecerdasan AI & sains jika memori masih awal
        default_seed_topics = [
            {
                "topic": "Machine Learning Fundamentals",
                "questions": [
                    "Apa perbedaan supervised, unsupervised, dan reinforcement learning?",
                    "Bagaimana cara kerja regularisasi L1 dan L2 mencegah overfitting?",
                    "Kapan teknik gradient descent membutuhkan penyesuaian learning rate?"
                ],
                "reason": "Topik fundamental pembelajaran mesin yang belum terdokumentasi lengkap."
            },
            {
                "topic": "Large Language Model Architecture",
                "questions": [
                    "Bagaimana mekanisme multi-head attention bekerja pada arsitektur Transformer?",
                    "Apa keuntungan LoRA dan QLoRA dibandingkan full parameter fine-tuning?",
                    "Bagaimana cara kerja Rotary Position Embedding (RoPE)?"
                ],
                "reason": "Topik krusial untuk peningkatan model dan fine-tuning mandiri."
            },
            {
                "topic": "Prompt Engineering & Guardrails",
                "questions": [
                    "Bagaimana teknik Chain-of-Thought (CoT) meningkatkan penalaran LLM?",
                    "Apa metode efektif mendeteksi dan mencegah indirect prompt injection?",
                    "Bagaimana cara memisahkan instruksi sistem dari data eksternal yang tidak tepercaya?"
                ],
                "reason": "Fondasi keamanan otonom dan mitigasi manipulasi konten web."
            },
            {
                "topic": "Vector Database & RAG",
                "questions": [
                    "Bagaimana cosine similarity dihitung pada ruang embedding vektor?",
                    "Apa perbedaan Hierarchical Navigable Small World (HNSW) dan inverted file index?",
                    "Bagaimana cara mengevaluasi retrieval grounding pada RAG?"
                ],
                "reason": "Fondasi penyimpanan pengetahuan semantik lokal."
            }
        ]

        # Periksa topik mana yang belum tersimpan dalam knowledge_items
        existing_items = self.store.list_knowledge_items(limit=1000)
        existing_topics = set(item["topic"].lower() for item in existing_items)

        for seed in default_seed_topics:
            if seed["topic"].lower() not in existing_topics:
                gaps.append(seed)

        # Cek jika ada tugas yang tertunda dari antrean offline
        pending = self.offline_mgr.get_pending_research()
        for p in pending:
            gaps.append({
                "topic": p.get("topic", "Topik Offline"),
                "questions": p.get("questions", []),
                "reason": f"Ditunda dari antrean offline (asal: {p.get('origin', 'manual')})"
            })

        return gaps

    def create_research_plan(self, topic: Optional[str] = None) -> ResearchPlan:
        """Menyusun rencana riset terstruktur."""
        if not topic:
            gaps = self.identify_knowledge_gaps()
            if gaps:
                selected = gaps[0]
                return ResearchPlan(
                    topic=selected["topic"],
                    target_questions=selected["questions"],
                    rationale=selected["reason"],
                )
            topic = "Kecerdasan Buatan Terapan"

        questions = [
            f"Apa prinsip dasar dan definisi utama dari {topic}?",
            f"Bagaimana implementasi praktis dan arsitektur {topic}?",
            f"Apa tantangan utama dan praktik terbaik terkini terkait {topic}?"
        ]
        return ResearchPlan(
            topic=topic,
            target_questions=questions,
            rationale=f"Riset otonom untuk memperdalam penguasaan topik: {topic}",
        )

    def fetch_web_page_safely(self, url: str, timeout: int = 8) -> Tuple[str, Optional[str]]:
        """
        Mengunduh konten halaman web secara aman dengan sandboxing:
        - Mencegah eksekusi script/javascript.
        - Membuang tag script, style, nav, footer iklan.
        - Mengambil teks inti artikel.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 AsistenAI-ResearchBot/2.0"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            html = resp.text

            soup = BeautifulSoup(html, "html.parser")

            # Hapus script, style, form, iklan
            for tag in soup(["script", "style", "nav", "footer", "aside", "form", "iframe"]):
                tag.decompose()

            # Cari tanggal publikasi jika ada pada meta tag
            pub_date = None
            date_meta = soup.find("meta", property=re.compile(r"published_time|date", re.I)) or soup.find("meta", attrs={"name": re.compile(r"date", re.I)})
            if date_meta and date_meta.get("content"):
                pub_date = date_meta["content"][:10]

            # Ambil teks paragraf utama
            paragraphs = [p.get_text().strip() for p in soup.find_all(["p", "h1", "h2", "h3", "li"])]
            content = " ".join([p for p in paragraphs if len(p) > 20])
            return content[:8000], pub_date

        except Exception as e:
            logger.warning(f"Gagal mengunduh halaman {url}: {e}")
            return "", None

    def execute_autonomous_research(
        self,
        topic: str,
        target_questions: Optional[List[str]] = None,
        max_sources: int = 3,
    ) -> ResearchSessionResult:
        """
        Menjalankan satu siklus riset otonom lengkap:
        Pencarian -> Pengunduhan -> Sanitasi & Validasi -> Deteksi Kontradiksi -> Penyimpanan -> QA Lanjutan.
        """
        start_time = time.time()
        session_id = f"res_{int(start_time)}"
        questions = target_questions or [f"Penelitian terstruktur mengenai {topic}"]

        # 1. Periksa konektivitas jaringan
        if not self.offline_mgr.is_online():
            # Jika offline, antrekan ke antrean riset tertunda
            self.offline_mgr.queue_pending_research(topic, questions, origin="autonomous_agent")
            return ResearchSessionResult(
                session_id=session_id,
                topic=topic,
                questions=questions,
                sources_fetched=0,
                facts_stored=0,
                contradictions=ContradictionReport(has_contradictions=False, contradiction_details=[], confidence_impact=0.0),
                follow_up_questions=[],
                status="queued_offline",
                duration_seconds=round(time.time() - start_time, 2),
                error_message="Jaringan offline: Riset ditambahkan ke antrean offline untuk diproses saat internet aktif.",
            )

        extracted_sources: List[ExtractedSource] = []
        all_facts_stored = 0

        try:
            # 2. Lakukan pencarian web untuk setiap pertanyaan
            search_results_map: Dict[str, SearchResultItem] = {}
            for q in questions[:2]:
                query = f"{topic} {q}"
                try:
                    res_items = self.search_provider.search(query, max_results=max_sources)
                    for item in res_items:
                        if item.url not in search_results_map:
                            search_results_map[item.url] = item
                except Exception as err:
                    logger.warning(f"Search provider issue on query '{query}': {err}")

            # 3. Kunjungi dan ekstraksi halaman web
            for url, item in list(search_results_map.items())[:max_sources]:
                raw_page, pub_date = self.fetch_web_page_safely(url)
                if not raw_page:
                    raw_page = item.snippet  # Gunakan snippet jika gagal mengambil halaman penuh

                # Sanitasi dan uji kualitas
                ext_source = self.validator.sanitize_and_validate(
                    url=url,
                    title=item.title,
                    content=raw_page,
                    published_date=pub_date,
                )
                extracted_sources.append(ext_source)

                # Simpan metadata sumber ke SQLite
                src_id = self.store.save_source(
                    url=url,
                    title=item.title,
                    domain=ext_source.domain,
                    reliability_score=ext_source.reliability_score,
                    is_trusted=ext_source.is_trusted_domain,
                    raw_content=ext_source.sanitized_text,
                    published_date=pub_date,
                )

                # Ekstraksi konsep dan fakta utama dari sumber yang kredibel
                if ext_source.reliability_score >= 0.35 and len(ext_source.sanitized_text) > 40:
                    # Ambil beberapa intisari kalimat penting
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ext_source.sanitized_text) if len(s.strip()) > 30]
                    for s in sentences[:3]:
                        # Ekstrak konsep sederhana dari kalimat
                        concept = f"Konsep {topic}: {s[:40]}..."
                        is_new, _, _ = self.store.save_knowledge_item(
                            topic=topic,
                            concept=concept,
                            fact_summary=s,
                            content_category=ext_source.content_category,
                            confidence_score=ext_source.reliability_score,
                            source_id=src_id,
                            source_citation=f"{item.title} ({ext_source.domain})",
                        )
                        if is_new:
                            all_facts_stored += 1

            # 4. Deteksi kontradiksi antar sumber yang didapat
            contradiction_rep = self.validator.compare_sources_for_contradictions(extracted_sources)

            # 5. Susun pertanyaan lanjutan
            follow_ups = [
                f"Bagaimana perbandingan efisiensi {topic} pada skenario dunia nyata?",
                f"Apa standar industri atau implementasi open-source terbaru untuk {topic}?",
                f"Bagaimana metrik evaluasi yang tepat untuk menguji kinerja {topic}?"
            ]

            duration = round(time.time() - start_time, 2)

            return ResearchSessionResult(
                session_id=session_id,
                topic=topic,
                questions=questions,
                sources_fetched=len(extracted_sources),
                facts_stored=all_facts_stored,
                contradictions=contradiction_rep,
                follow_up_questions=follow_ups,
                status="completed",
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Kegagalan pada siklus riset {topic}: {e}")
            return ResearchSessionResult(
                session_id=session_id,
                topic=topic,
                questions=questions,
                sources_fetched=len(extracted_sources),
                facts_stored=all_facts_stored,
                contradictions=ContradictionReport(has_contradictions=False, contradiction_details=[], confidence_impact=0.0),
                follow_up_questions=[],
                status="failed",
                duration_seconds=round(time.time() - start_time, 2),
                error_message=str(e),
            )


# Singleton instance
RESEARCH_AGENT = AutonomousResearchAgent()
