"""
asisten_ai.knowledge_store - Basis Pengetahuan Lokal Berlapis, Provenance Sumber, dan Versi Pengetahuan
Mengimplementasikan:
1. Working Memory (memori aktif sesi)
2. Episodic Memory (riwayat riset & eksperimen)
3. Semantic Memory (fakta terverifikasi & konsep terindeks)
4. Source Memory (pelacak asal-usul sumber & kredibilitas)
5. Training Memory (dataset dan checkpoint model)
6. Evaluation Memory (benchmark & histori akurasi)
"""
import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from asisten_ai.config import DB_PATH, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)


class KnowledgeStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_database(self) -> None:
        """Membuat skema database relasional lengkap untuk seluruh lapisan memori."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Source Memory (Tabel Sumber)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                domain TEXT NOT NULL,
                published_date TEXT,
                retrieved_at TEXT NOT NULL,
                reliability_score REAL NOT NULL,
                is_trusted INTEGER DEFAULT 0,
                content_hash TEXT NOT NULL
            )
            """)

            # 2. Semantic Memory (Fakta & Konsep dengan Provenance)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                concept TEXT NOT NULL,
                fact_summary TEXT NOT NULL,
                content_category TEXT NOT NULL, -- fact, opinion, hypothesis, unverified
                confidence_score REAL NOT NULL,
                source_id INTEGER,
                source_citation TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                content_hash TEXT UNIQUE NOT NULL,
                FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE SET NULL
            )
            """)

            # 3. Knowledge Versioning (Riwayat Perubahan Fakta)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                fact_summary TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                changed_at TEXT NOT NULL,
                reason TEXT,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items (id) ON DELETE CASCADE
            )
            """)

            # 4. Episodic Memory (Riwayat Siklus Riset)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                questions_json TEXT NOT NULL,
                sources_found INTEGER DEFAULT 0,
                facts_extracted INTEGER DEFAULT 0,
                contradictions_detected INTEGER DEFAULT 0,
                status TEXT NOT NULL, -- running, completed, failed, paused
                started_at TEXT NOT NULL,
                finished_at TEXT,
                error_message TEXT
            )
            """)

            # 5. Training Memory (Riwayat Pelatihan Model & Checkpoint)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                dataset_size INTEGER NOT NULL,
                train_samples INTEGER NOT NULL,
                val_samples INTEGER NOT NULL,
                test_samples INTEGER NOT NULL,
                base_model TEXT NOT NULL,
                candidate_model_path TEXT,
                loss_history_json TEXT,
                final_val_loss REAL,
                hardware_used TEXT,
                status TEXT NOT NULL, -- completed, failed, aborted
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """)

            # 6. Evaluation Memory (Hasil Uji Mandiri & Riwayat Benchmark)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL, -- active, candidate, baseline
                total_questions INTEGER NOT NULL,
                accuracy_score REAL NOT NULL,
                consistency_score REAL NOT NULL,
                instruction_following_score REAL NOT NULL,
                citation_grounding_score REAL NOT NULL,
                regression_detected INTEGER DEFAULT 0,
                details_json TEXT,
                evaluated_at TEXT NOT NULL
            )
            """)

            # 7. Model Registry (Manajemen Model Aktif vs Kandidat)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_path TEXT NOT NULL,
                is_active INTEGER DEFAULT 0,
                is_candidate INTEGER DEFAULT 0,
                evaluation_score REAL DEFAULT 0.0,
                promoted_at TEXT,
                created_at TEXT NOT NULL,
                description TEXT
            )
            """)

            # 8. Persistent Job Queue (Untuk Scheduler yang tahan restart)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_queue (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL, -- research, learn, evaluate, train
                payload_json TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                status TEXT DEFAULT 'pending', -- pending, running, completed, failed, paused
                created_at TEXT NOT NULL,
                scheduled_for TEXT,
                updated_at TEXT,
                attempts INTEGER DEFAULT 0,
                error_message TEXT
            )
            """)

            # Indeks untuk query cepat
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ki_topic ON knowledge_items(topic)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ki_hash ON knowledge_items(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_src_domain ON sources(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluation_runs(model_name)")

            conn.commit()

    # --- SOURCE MEMORY METHODS ---
    def save_source(
        self,
        url: str,
        title: str,
        domain: str,
        reliability_score: float,
        is_trusted: bool,
        raw_content: str,
        published_date: Optional[str] = None,
    ) -> int:
        content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
        retrieved_at = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sources (url, title, domain, published_date, retrieved_at, reliability_score, is_trusted, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title=excluded.title,
                    reliability_score=excluded.reliability_score,
                    retrieved_at=excluded.retrieved_at,
                    content_hash=excluded.content_hash
                """,
                (url, title, domain, published_date, retrieved_at, reliability_score, 1 if is_trusted else 0, content_hash)
            )
            conn.commit()
            cursor.execute("SELECT id FROM sources WHERE url = ?", (url,))
            row = cursor.fetchone()
            return row["id"] if row else -1

    # --- SEMANTIC KNOWLEDGE METHODS ---
    def save_knowledge_item(
        self,
        topic: str,
        concept: str,
        fact_summary: str,
        content_category: str,
        confidence_score: float,
        source_id: Optional[int],
        source_citation: str,
    ) -> Tuple[bool, int, str]:
        """
        Menyimpan fakta baru dengan deduplikasi otomatis.
        Mengembalikan: (is_new, item_id, message)
        """
        norm_text = f"{topic.lower()}::{concept.lower()}::{fact_summary.strip()}"
        content_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Cek duplikat
            cursor.execute("SELECT id, fact_summary, version, confidence_score FROM knowledge_items WHERE content_hash = ?", (content_hash,))
            existing = cursor.fetchone()

            if existing:
                # Duplikat persis ditemukan
                return False, existing["id"], "Fakta duplikat terdeteksi (sudah tersimpan sebelumnya)."

            # Simpan fakta baru
            cursor.execute(
                """
                INSERT INTO knowledge_items
                (topic, concept, fact_summary, content_category, confidence_score, source_id, source_citation, created_at, updated_at, version, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (topic, concept, fact_summary, content_category, confidence_score, source_id, source_citation, now, now, content_hash)
            )
            item_id = cursor.lastrowid

            # Catat versi 1
            cursor.execute(
                """
                INSERT INTO knowledge_versions (knowledge_id, version, fact_summary, confidence_score, changed_at, reason)
                VALUES (?, 1, ?, ?, ?, 'Inisialisasi fakta pertama kali')
                """,
                (item_id, fact_summary, confidence_score, now)
            )
            conn.commit()
            return True, item_id, "Fakta baru berhasil disimpan ke basis pengetahuan lokal."

    def update_knowledge_item(
        self,
        item_id: int,
        new_summary: str,
        new_confidence: float,
        reason: str = "Pembaruan informasi dari riset lanjutan"
    ) -> bool:
        """Memperbarui fakta yang ada dan mencatat riwayat versinya."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version, topic, concept FROM knowledge_items WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            if not row:
                return False

            new_version = row["version"] + 1
            new_hash = hashlib.sha256(f"{row['topic'].lower()}::{row['concept'].lower()}::{new_summary.strip()}".encode("utf-8")).hexdigest()

            cursor.execute(
                """
                UPDATE knowledge_items
                SET fact_summary = ?, confidence_score = ?, version = ?, updated_at = ?, content_hash = ?
                WHERE id = ?
                """,
                (new_summary, new_confidence, new_version, now, new_hash, item_id)
            )

            cursor.execute(
                """
                INSERT INTO knowledge_versions (knowledge_id, version, fact_summary, confidence_score, changed_at, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, new_version, new_summary, new_confidence, now, reason)
            )
            conn.commit()
            return True

    def list_knowledge_items(
        self,
        topic: Optional[str] = None,
        limit: int = 100,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if topic:
                cursor.execute(
                    """
                    SELECT k.*, s.url as source_url, s.domain as source_domain, s.reliability_score
                    FROM knowledge_items k
                    LEFT JOIN sources s ON k.source_id = s.id
                    WHERE k.topic = ? AND k.confidence_score >= ?
                    ORDER BY k.id DESC LIMIT ?
                    """,
                    (topic, min_confidence, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT k.*, s.url as source_url, s.domain as source_domain, s.reliability_score
                    FROM knowledge_items k
                    LEFT JOIN sources s ON k.source_id = s.id
                    WHERE k.confidence_score >= ?
                    ORDER BY k.id DESC LIMIT ?
                    """,
                    (min_confidence, limit)
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def delete_knowledge_item(self, item_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_items WHERE id = ?", (item_id,))
            conn.commit()
            return cursor.rowcount > 0

    def export_knowledge_json(self) -> str:
        items = self.list_knowledge_items(limit=10000)
        return json.dumps(items, indent=2, ensure_ascii=False)

    def get_summary_statistics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM knowledge_items")
            total_facts = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM sources")
            total_sources = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(DISTINCT topic) as count FROM knowledge_items")
            total_topics = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM research_sessions")
            total_sessions = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM training_runs")
            total_training_runs = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM evaluation_runs")
            total_evaluations = cursor.fetchone()["count"]

            cursor.execute("SELECT AVG(confidence_score) as avg_conf FROM knowledge_items")
            avg_conf = cursor.fetchone()["avg_conf"] or 0.0

            return {
                "total_facts": total_facts,
                "total_sources": total_sources,
                "total_topics": total_topics,
                "total_research_sessions": total_sessions,
                "total_training_runs": total_training_runs,
                "total_evaluations": total_evaluations,
                "average_confidence": round(avg_conf, 2),
            }


# Singleton instance
KNOWLEDGE_STORE = KnowledgeStore()
