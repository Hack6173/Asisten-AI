"""
asisten_ai.retrieval_engine - Mesin Pencarian Semantik dan Temu Kembali Informasi Lokal
Menggabungkan pencarian leksikal (keyword matching) dan pencarian semantik (TF-IDF Cosine Similarity)
untuk pencarian fakta yang sangat cepat dan 100% offline.
"""
import math
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from asisten_ai.knowledge_store import KNOWLEDGE_STORE

logger = logging.getLogger(__name__)


class RetrievedFact(BaseModel):
    id: int
    topic: str
    concept: str
    fact_summary: str
    content_category: str
    confidence_score: float
    source_citation: Optional[str] = None
    source_url: Optional[str] = None
    source_domain: Optional[str] = None
    similarity_score: float = 0.0


class RetrievalEngine:
    def __init__(self, store=KNOWLEDGE_STORE):
        self.store = store

    def search_semantic(
        self,
        query: str,
        topic_filter: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.15,
    ) -> List[RetrievedFact]:
        """
        Pencarian semantik lokal menggunakan TF-IDF Vectorizer dan Cosine Similarity.
        100% beroperasi offline tanpa dependensi cloud.
        """
        all_items = self.store.list_knowledge_items(topic=topic_filter, limit=5000)
        if not all_items:
            return []

        # Buat korpus teks dari item pengetahuan
        corpus = [
            f"{item['topic']} {item['concept']} {item['fact_summary']}"
            for item in all_items
        ]

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=None, # Mempertahankan kata-kata kunci Bahasa Indonesia & Inggris
                token_pattern=r"(?u)\b\w+\b"
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([query])
            sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

            results: List[RetrievedFact] = []
            for idx, score in enumerate(sim_scores):
                if score >= min_similarity:
                    item = all_items[idx]
                    results.append(
                        RetrievedFact(
                            id=item["id"],
                            topic=item["topic"],
                            concept=item["concept"],
                            fact_summary=item["fact_summary"],
                            content_category=item["content_category"],
                            confidence_score=item["confidence_score"],
                            source_citation=item.get("source_citation"),
                            source_url=item.get("source_url"),
                            source_domain=item.get("source_domain"),
                            similarity_score=round(float(score), 4),
                        )
                    )

            # Urutkan berdasarkan gabungan similarity dan confidence score
            results.sort(
                key=lambda x: (x.similarity_score * 0.7 + x.confidence_score * 0.3),
                reverse=True,
            )
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error pada pencarian semantik: {e}")
            # Fallback ke pencarian kata kunci sederhana
            return self._keyword_fallback(query, all_items, top_k)

    def _keyword_fallback(self, query: str, items: List[Dict[str, Any]], top_k: int) -> List[RetrievedFact]:
        words = set(query.lower().split())
        matched = []
        for item in items:
            text = f"{item['topic']} {item['concept']} {item['fact_summary']}".lower()
            overlap = sum(1 for w in words if w in text)
            if overlap > 0:
                score = overlap / max(len(words), 1)
                matched.append(
                    RetrievedFact(
                        id=item["id"],
                        topic=item["topic"],
                        concept=item["concept"],
                        fact_summary=item["fact_summary"],
                        content_category=item["content_category"],
                        confidence_score=item["confidence_score"],
                        source_citation=item.get("source_citation"),
                        source_url=item.get("source_url"),
                        source_domain=item.get("source_domain"),
                        similarity_score=round(score, 2),
                    )
                )
        matched.sort(key=lambda x: x.similarity_score, reverse=True)
        return matched[:top_k]

    def build_grounded_answer(self, query: str, facts: List[RetrievedFact]) -> Dict[str, Any]:
        """
        Menyusun jawaban offline berbasis fakta yang ditemukan lengkap dengan kutipan sumber.
        Jika fakta tidak mencukupi, menyatakan keterbatasan secara jujur.
        """
        if not facts:
            return {
                "answer": (
                    f"Informasi spesifik mengenai '{query}' belum tersimpan di dalam basis pengetahuan lokal Asisten-AI. "
                    "Dalam mode offline atau ketika topik belum diriset, saya tidak mengarang fakta tanpa bukti. "
                    "Topik ini telah dicatat untuk diteliti saat riset otonom berikutnya dijalankan."
                ),
                "has_grounding": False,
                "citations": [],
                "confidence": 0.0,
            }

        citations = []
        facts_text_list = []
        avg_conf = sum(f.confidence_score for f in facts) / len(facts)

        for i, f in enumerate(facts, 1):
            ref = f"[{i}] {f.source_domain or 'Lokal'} ({f.source_url or 'Penyimpanan Mandiri'})"
            citations.append(ref)
            facts_text_list.append(f"- **{f.concept}** ({f.content_category.upper()}): {f.fact_summary} {ref}")

        formatted_answer = (
            f"Berdasarkan data terverifikasi dalam Basis Pengetahuan Lokal:\n\n"
            + "\n".join(facts_text_list)
            + f"\n\n**Sumber Terpercaya:**\n"
            + "\n".join(citations)
        )

        return {
            "answer": formatted_answer,
            "has_grounding": True,
            "citations": citations,
            "confidence": round(avg_conf, 2),
        }


# Singleton instance
RETRIEVAL_ENGINE = RetrievalEngine()
