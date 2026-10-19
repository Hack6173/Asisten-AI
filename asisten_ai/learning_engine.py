"""
asisten_ai.learning_engine - Mesin Pembelajaran Mandiri Berkelanjutan (Autonomous Learning Engine)
Mengorkestrasi seluruh siklus:
Identifikasi topik -> Rencana Riset -> Ekstraksi Fakta -> Evaluasi Kualitas & Kontradiksi
-> Pembuatan Pertanyaan Uji Pemahaman -> Pengujian Mandiri -> Rencana Lanjutan.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from asisten_ai.config import SYSTEM_CONFIG
from asisten_ai.knowledge_store import KNOWLEDGE_STORE
from asisten_ai.research_agent import RESEARCH_AGENT, ResearchSessionResult
from asisten_ai.resource_monitor import RESOURCE_MONITOR

logger = logging.getLogger(__name__)


class LearningCycleReport(BaseModel):
    cycle_id: str
    topic: str
    stage: str
    facts_acquired: int
    questions_generated: int
    eval_score: float
    weaknesses_found: List[str]
    next_research_plan: List[str]
    duration_seconds: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AutonomousLearningEngine:
    def __init__(
        self,
        store=KNOWLEDGE_STORE,
        research_agent=RESEARCH_AGENT,
        monitor=RESOURCE_MONITOR
    ):
        self.store = store
        self.research_agent = research_agent
        self.monitor = monitor

    def run_complete_learning_cycle(self, target_topic: Optional[str] = None) -> LearningCycleReport:
        """
        Menjalankan satu siklus penuh pembelajaran mandiri 11 langkah sesuai spesifikasi arsitektur.
        """
        start_time = time.time()
        cycle_id = f"learn_{int(start_time)}"

        # 0. Safety Pre-flight Check
        safe, reason = self.monitor.is_safe_to_execute(
            max_cpu_percent=SYSTEM_CONFIG.resources.max_cpu_percent,
            max_ram_percent=SYSTEM_CONFIG.resources.max_ram_percent,
            min_free_ram_mb=SYSTEM_CONFIG.resources.min_free_ram_mb,
        )
        if not safe:
            raise ResourceWarning(f"Siklus belajar ditunda demi keamanan sistem: {reason}")

        # Langkah 1 & 2: Identifikasi topik & pertanyaan
        if not target_topic:
            gaps = self.research_agent.identify_knowledge_gaps()
            selected = gaps[0] if gaps else {"topic": "Prinsip Otonomi AI", "questions": ["Bagaimana agen AI mengevaluasi kinerjanya sendiri?"]}
            topic = selected["topic"]
            questions = selected["questions"]
        else:
            topic = target_topic
            plan = self.research_agent.create_research_plan(topic)
            questions = plan.target_questions

        logger.info(f"Memulai siklus belajar untuk topik: '{topic}'")

        # Langkah 3, 4, 5, 6, 7: Riset, ekstraksi, validasi, ringkasan berreferensi, simpan fakta
        res_result: ResearchSessionResult = self.research_agent.execute_autonomous_research(
            topic=topic,
            target_questions=questions,
            max_sources=3
        )

        # Langkah 8: Buat pertanyaan untuk menguji pemahaman sendiri (Self-Testing QA pairs)
        test_questions = self.generate_self_test_questions(topic)

        # Langkah 9: Jalankan evaluasi mandiri awal
        eval_score, weaknesses = self.evaluate_understanding_on_topic(topic, test_questions)

        # Langkah 10 & 11: Identifikasi kelemahan & rencanakan riset berikutnya
        next_plans = [
            f"Eksplorasi lanjutan kelemahan: {w}" for w in weaknesses
        ] if weaknesses else res_result.follow_up_questions

        duration = round(time.time() - start_time, 2)

        report = LearningCycleReport(
            cycle_id=cycle_id,
            topic=topic,
            stage="completed" if res_result.status == "completed" else res_result.status,
            facts_acquired=res_result.facts_stored,
            questions_generated=len(test_questions),
            eval_score=eval_score,
            weaknesses_found=weaknesses,
            next_research_plan=next_plans[:3],
            duration_seconds=duration,
        )

        return report

    def generate_self_test_questions(self, topic: str) -> List[Dict[str, str]]:
        """
        Membuat pasangan pertanyaan dan jawaban referensi berdasarkan fakta yang baru dipelajari.
        Digunakan oleh Evaluation Center untuk menguji akurasi model tanpa peeking saat tes berlangsung.
        """
        items = self.store.list_knowledge_items(topic=topic, limit=10)
        qa_pairs = []

        for item in items:
            fact = item["fact_summary"]
            concept = item["concept"]

            # Buat pertanyaan inferensi berbasis konsep
            q = f"Jelaskan prinsip dan fakta penting mengenai {concept} dalam konteks {topic}!"
            qa_pairs.append({
                "question": q,
                "reference_answer": fact,
                "topic": topic,
                "source_citation": item.get("source_citation", "Basis Pengetahuan Lokal"),
                "confidence": str(item["confidence_score"])
            })

        if not qa_pairs:
            qa_pairs.append({
                "question": f"Apa konsep dasar yang mendasari {topic}?",
                "reference_answer": f"Informasi dasar terkait {topic}.",
                "topic": topic,
                "source_citation": "Inisialisasi",
                "confidence": "0.5"
            })

        return qa_pairs

    def evaluate_understanding_on_topic(
        self, topic: str, test_qa: List[Dict[str, str]]
    ) -> Tuple[float, List[str]]:
        """
        Menguji pemahaman mandiri terhadap materi tanpa melihat jawaban referensi saat simulasi tes.
        """
        from asisten_ai.retrieval_engine import RETRIEVAL_ENGINE

        if not test_qa:
            return 0.5, ["Belum ada materi uji yang memadai."]

        correct_count = 0
        weaknesses = []

        for item in test_qa:
            q = item["question"]
            ref = item["reference_answer"].lower()

            # AI menjawab menggunakan retrieval lokal (tanpa melihat ref secara langsung)
            retrieved = RETRIEVAL_ENGINE.search_semantic(q, topic_filter=topic, top_k=2)
            if retrieved:
                combined_ans = " ".join([r.fact_summary.lower() for r in retrieved])
                # Hitung lexical overlap kunci antara jawaban yang dihasilkan dengan konsep referensi
                ref_words = set(ref.split())
                overlap = sum(1 for w in ref_words if len(w) > 4 and w in combined_ans)
                accuracy_ratio = overlap / max(len(ref_words), 1)

                if accuracy_ratio >= 0.20:
                    correct_count += 1
                else:
                    weaknesses.append(f"Kurang informasi detail pada konsep: {item.get('question')[:50]}...")
            else:
                weaknesses.append(f"Tidak ada fakta lokal yang cocok untuk: {item.get('question')[:50]}...")

        score = round(correct_count / len(test_qa), 2)
        return score, weaknesses[:3]


# Singleton instance
LEARNING_ENGINE = AutonomousLearningEngine()
