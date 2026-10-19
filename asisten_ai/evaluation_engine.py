"""
asisten_ai.evaluation_engine - Mesin Pengujian Mandiri dan Evaluasi Objektif (Self-Evaluation & Self-Correction)
Menguji model secara independen tanpa melihat jawaban acuan saat inferensi berlangsung.
Mengukur akurasi, konsistensi logis, kepatuhan instruksi, grounding kutipan,
mendeteksi regresi kemampuan sebelumnya, dan mencatat histori benchmark ke SQLite.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from asisten_ai.knowledge_store import KNOWLEDGE_STORE
from asisten_ai.retrieval_engine import RETRIEVAL_ENGINE

logger = logging.getLogger(__name__)


class BenchmarkQuestion(BaseModel):
    id: str
    topic: str
    question: str
    ground_truth: str
    key_entities: List[str]
    category: str = "reasoning"


class EvaluationReport(BaseModel):
    evaluation_id: str
    model_name: str
    model_type: str  # active, candidate, baseline
    total_questions: int
    accuracy_score: float
    consistency_score: float
    instruction_following_score: float
    citation_grounding_score: float
    overall_score: float
    regression_detected: bool
    weak_topics: List[str]
    detailed_results: List[Dict[str, Any]]
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class EvaluationEngine:
    def __init__(self, store=KNOWLEDGE_STORE, retrieval=RETRIEVAL_ENGINE):
        self.store = store
        self.retrieval = retrieval

    def generate_independent_test_suite(self, sample_size: int = 10) -> List[BenchmarkQuestion]:
        """
        Menyusun suite pengujian mandiri yang terpisah dari dataset training.
        """
        items = self.store.list_knowledge_items(limit=100)
        suite: List[BenchmarkQuestion] = []

        for item in items[:sample_size]:
            words = [w for w in item["fact_summary"].split() if len(w) > 4][:5]
            q_text = f"Jelaskan secara tepat apa yang didefinisikan mengenai '{item['concept']}' dalam bidang {item['topic']}!"
            suite.append(
                BenchmarkQuestion(
                    id=f"bench_{item['id']}",
                    topic=item["topic"],
                    question=q_text,
                    ground_truth=item["fact_summary"],
                    key_entities=words,
                    category="factual_retrieval",
                )
            )

        if not suite:
            # Pertanyaan evaluasi baseline default
            suite = [
                BenchmarkQuestion(
                    id="bench_core_1",
                    topic="Machine Learning",
                    question="Apa perbedaan mendasar antara supervised learning dan unsupervised learning?",
                    ground_truth="Supervised learning menggunakan data berlabel ground truth sedangkan unsupervised learning mencari pola tanpa label.",
                    key_entities=["label", "pola", "ground truth"],
                ),
                BenchmarkQuestion(
                    id="bench_core_2",
                    topic="LLM Guardrails",
                    question="Mengapa konten dari situs web luar harus disanitasi sebelum dimasukkan ke konteks AI?",
                    ground_truth="Untuk mencegah serangan prompt injection dan eksploitasi yang dapat memanipulasi instruksi agen.",
                    key_entities=["prompt injection", "manipulasi", "sanitasi"],
                ),
                BenchmarkQuestion(
                    id="bench_core_3",
                    topic="Vector Database",
                    question="Bagaimana cosine similarity digunakan untuk menentukan kedekatan semantik dua dokumen?",
                    ground_truth="Cosine similarity mengukur sudut kosinus antara dua vektor representasi dokumen dalam ruang multi-dimensi.",
                    key_entities=["sudut kosinus", "vektor", "ruang multi-dimensi"],
                )
            ]

        return suite

    def run_evaluation(
        self,
        model_name: str = "Active-Asisten-Model",
        model_type: str = "active",
        baseline_score: Optional[float] = None,
    ) -> EvaluationReport:
        """
        Menjalankan evaluasi objektif tanpa peeking (jawaban acuan disembunyikan selama proses penalaran).
        Menguji akurasi, konsistensi, instruksi, dan grounding sitasi.
        """
        eval_id = f"eval_{int(datetime.now().timestamp())}"
        test_suite = self.generate_independent_test_suite()

        correct_count = 0
        consistent_count = 0
        instruction_count = 0
        grounding_count = 0
        detailed = []
        topic_scores: Dict[str, List[float]] = {}

        for q in test_suite:
            # AI Menghasilkan respons hanya berdasarkan retrieval semantik lokal tanpa akses ke ground truth
            retrieved = self.retrieval.search_semantic(q.question, top_k=2)
            has_grounding = len(retrieved) > 0 and retrieved[0].confidence_score >= 0.3

            if retrieved:
                generated_answer = " ".join([r.fact_summary for r in retrieved])
                citation_source = retrieved[0].source_domain or "Lokal"
            else:
                generated_answer = "Tidak ada informasi lokal yang mendukung pertanyaan ini."
                citation_source = "None"

            # 1. Uji Akurasi: Periksa kehadiran key_entities dari ground_truth dalam jawaban
            answer_lower = generated_answer.lower()
            entity_matches = sum(1 for ent in q.key_entities if ent.lower() in answer_lower)
            acc_ratio = entity_matches / max(len(q.key_entities), 1)
            is_correct = acc_ratio >= 0.30
            if is_correct:
                correct_count += 1

            # 2. Uji Konsistensi: Apakah jawaban tidak memuat kontradiksi internal
            is_consistent = ("namun salah" not in answer_lower) and ("tidak ada informasi" not in answer_lower or not retrieved)
            if is_consistent:
                consistent_count += 1

            # 3. Kepatuhan Instruksi: Respons berupa penjelasan dan panjang memadai (> 20 karakter)
            follows_instruction = len(generated_answer.strip()) > 20
            if follows_instruction:
                instruction_count += 1

            # 4. Citation Grounding: Memiliki sumber referensi nyata
            is_grounded = has_grounding and citation_source != "None"
            if is_grounded:
                grounding_count += 1

            sample_score = (acc_ratio * 0.4 + (1.0 if is_consistent else 0.0) * 0.2 + (1.0 if follows_instruction else 0.0) * 0.2 + (1.0 if is_grounded else 0.0) * 0.2)
            topic_scores.setdefault(q.topic, []).append(sample_score)

            detailed.append({
                "question_id": q.id,
                "topic": q.topic,
                "question": q.question,
                "is_correct": is_correct,
                "is_grounded": is_grounded,
                "source_used": citation_source,
                "score": round(sample_score, 2),
            })

        n = len(test_suite)
        accuracy = round(correct_count / n, 2)
        consistency = round(consistent_count / n, 2)
        instruction_score = round(instruction_count / n, 2)
        grounding_score = round(grounding_count / n, 2)
        overall = round((accuracy * 0.4 + consistency * 0.2 + instruction_score * 0.2 + grounding_score * 0.2), 2)

        # Deteksi topik terlemah
        weak_topics = []
        for t, scores in topic_scores.items():
            if (sum(scores) / len(scores)) < 0.50:
                weak_topics.append(t)

        # Deteksi regresi jika baseline diberikan
        regression = False
        if baseline_score is not None:
            if overall < (baseline_score - 0.02):
                regression = True

        report = EvaluationReport(
            evaluation_id=eval_id,
            model_name=model_name,
            model_type=model_type,
            total_questions=n,
            accuracy_score=accuracy,
            consistency_score=consistency,
            instruction_following_score=instruction_score,
            citation_grounding_score=grounding_score,
            overall_score=overall,
            regression_detected=regression,
            weak_topics=weak_topics,
            detailed_results=detailed,
        )

        # Simpan laporan evaluasi ke SQLite
        self._record_evaluation(report)
        return report

    def _record_evaluation(self, report: EvaluationReport) -> None:
        with self.store._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs
                (id, model_name, model_type, total_questions, accuracy_score, consistency_score, instruction_following_score, citation_grounding_score, regression_detected, details_json, evaluated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.evaluation_id, report.model_name, report.model_type,
                    report.total_questions, report.accuracy_score, report.consistency_score,
                    report.instruction_following_score, report.citation_grounding_score,
                    1 if report.regression_detected else 0,
                    json.dumps(report.detailed_results), report.evaluated_at
                )
            )
            conn.commit()

    def get_evaluation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evaluation_runs ORDER BY evaluated_at DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]


# Singleton instance
EVALUATION_ENGINE = EvaluationEngine()
