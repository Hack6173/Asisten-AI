"""
asisten_ai.model_registry - Registri Model, Evaluasi Promosi Kandidat, dan Mekanisme Rollback
Mengelola siklus hidup model:
1. Registrasi model dasar (Active Baseline).
2. Penyimpanan hasil training sebagai Kandidat (Candidate Model).
3. Gerbang evaluasi ketat (Promotion Gatekeeper).
4. Mekanisme Rollback aman jika terjadi degradasi performa.
"""
import shutil
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

from asisten_ai.config import MODELS_DIR, SYSTEM_CONFIG
from asisten_ai.knowledge_store import KNOWLEDGE_STORE
from asisten_ai.evaluation_engine import EVALUATION_ENGINE, EvaluationReport

logger = logging.getLogger(__name__)


class RegisteredModel(BaseModel):
    id: str
    model_name: str
    version: str
    model_path: str
    is_active: bool
    is_candidate: bool
    evaluation_score: float
    promoted_at: Optional[str] = None
    created_at: str
    description: str


class ModelRegistry:
    def __init__(self, store=KNOWLEDGE_STORE, models_dir: Path = MODELS_DIR):
        self.store = store
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_baseline_registered()

    def _ensure_baseline_registered(self) -> None:
        """Memastikan setidaknya satu model dasar aktif terdaftar di registri."""
        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM model_registry WHERE is_active = 1")
            active = cursor.fetchone()
            if not active:
                now = datetime.now().isoformat()
                cursor.execute(
                    """
                    INSERT INTO model_registry
                    (id, model_name, version, model_path, is_active, is_candidate, evaluation_score, created_at, description)
                    VALUES (?, ?, ?, ?, 1, 0, 0.70, ?, ?)
                    """,
                    (
                        "model_base_v1",
                        "Asisten-AI Local Base (Qwen2.5-0.5B)",
                        "1.0.0",
                        str(self.models_dir / "base_active"),
                        now,
                        "Model dasar aktif awal dengan kemampuan penalaran lokal dan grounding semantik."
                    )
                )
                conn.commit()

    def register_candidate(
        self,
        candidate_name: str,
        version: str,
        checkpoint_dir: str,
        description: str = "Model hasil fine-tuning mandiri",
    ) -> RegisteredModel:
        """Mendaftarkan hasil pelatihan sebagai model kandidat (belum dipromosikan)."""
        cand_id = f"cand_{int(datetime.now().timestamp())}"
        now = datetime.now().isoformat()

        with self.store._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO model_registry
                (id, model_name, version, model_path, is_active, is_candidate, evaluation_score, created_at, description)
                VALUES (?, ?, ?, ?, 0, 1, 0.0, ?, ?)
                """,
                (cand_id, candidate_name, version, checkpoint_dir, now, description)
            )
            conn.commit()

        logger.info(f"Model kandidat '{candidate_name}' ({cand_id}) berhasil didaftarkan.")
        return RegisteredModel(
            id=cand_id,
            model_name=candidate_name,
            version=version,
            model_path=checkpoint_dir,
            is_active=False,
            is_candidate=True,
            evaluation_score=0.0,
            created_at=now,
            description=description,
        )

    def get_active_model(self) -> Optional[RegisteredModel]:
        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE is_active = 1 LIMIT 1")
            row = cursor.fetchone()
            if row:
                return RegisteredModel(
                    id=row["id"],
                    model_name=row["model_name"],
                    version=row["version"],
                    model_path=row["model_path"],
                    is_active=bool(row["is_active"]),
                    is_candidate=bool(row["is_candidate"]),
                    evaluation_score=row["evaluation_score"],
                    promoted_at=row["promoted_at"],
                    created_at=row["created_at"],
                    description=row["description"],
                )
        return None

    def list_models(self) -> List[RegisteredModel]:
        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [
                RegisteredModel(
                    id=r["id"],
                    model_name=r["model_name"],
                    version=r["version"],
                    model_path=r["model_path"],
                    is_active=bool(r["is_active"]),
                    is_candidate=bool(r["is_candidate"]),
                    evaluation_score=r["evaluation_score"],
                    promoted_at=r["promoted_at"],
                    created_at=r["created_at"],
                    description=r["description"],
                )
                for r in rows
            ]

    def evaluate_and_promote_candidate(
        self,
        candidate_id: str,
        min_gain: float = 0.03,
        force_promote: bool = False
    ) -> Tuple[bool, str, EvaluationReport]:
        """
        Gerbang Promosi:
        1. Uji kandidat menggunakan Evaluation Engine independen.
        2. Bandingkan skor kandidat dengan model aktif saat ini.
        3. Tolak jika terjadi regresi performa atau peningkatan tidak mencukupi (kecuali dipaksa).
        4. Jika lulus, aktifkan kandidat dan arsipkan model sebelumnya untuk rollback.
        """
        active_model = self.get_active_model()
        baseline_score = active_model.evaluation_score if active_model else 0.50

        # Ambil data kandidat
        cand = None
        for m in self.list_models():
            if m.id == candidate_id:
                cand = m
                break

        if not cand:
            raise ValueError(f"Model kandidat dengan ID '{candidate_id}' tidak ditemukan.")

        # Jalankan evaluasi objektif
        report = EVALUATION_ENGINE.run_evaluation(
            model_name=cand.model_name,
            model_type="candidate",
            baseline_score=baseline_score,
        )

        gain = report.overall_score - baseline_score

        # Periksa kriteria promosi
        if report.regression_detected and not force_promote:
            msg = (
                f"Promosi Ditolak: Terdeteksi regresi kemampuan sebelumnya. "
                f"Skor kandidat: {report.overall_score:.2f} vs Model Aktif: {baseline_score:.2f}."
            )
            return False, msg, report

        if gain < min_gain and not force_promote:
            msg = (
                f"Promosi Ditolak: Peningkatan performa (+{gain:.2f}) di bawah ambang batas minimum (+{min_gain:.2f}). "
                f"Skor kandidat: {report.overall_score:.2f}."
            )
            return False, msg, report

        # Lulus kriteria -> Promosikan menjadi model aktif
        now = datetime.now().isoformat()
        with self.store._get_connection() as conn:
            # Nonaktifkan model aktif sebelumnya
            conn.execute("UPDATE model_registry SET is_active = 0 WHERE is_active = 1")
            # Aktifkan kandidat
            conn.execute(
                """
                UPDATE model_registry
                SET is_active = 1, is_candidate = 0, evaluation_score = ?, promoted_at = ?
                WHERE id = ?
                """,
                (report.overall_score, now, candidate_id)
            )
            conn.commit()

        success_msg = (
            f"Selamat! Model kandidat '{cand.model_name}' berhasil dipromosikan menjadi Model Aktif. "
            f"Skor Evaluasi: {report.overall_score:.2f} (Peningkatan: +{gain:.2f})."
        )
        logger.info(success_msg)
        return True, success_msg, report

    def rollback_to_previous_model(self, target_model_id: str) -> Tuple[bool, str]:
        """
        Mengembalikan sistem ke versi model sebelumnya jika terdeteksi masalah.
        """
        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, model_name, version FROM model_registry WHERE id = ?", (target_model_id,))
            target = cursor.fetchone()
            if not target:
                return False, f"Model target '{target_model_id}' tidak ditemukan."

            # Nonaktifkan semua
            conn.execute("UPDATE model_registry SET is_active = 0")
            # Aktifkan target
            conn.execute("UPDATE model_registry SET is_active = 1 WHERE id = ?", (target_model_id,))
            conn.commit()

            msg = f"Rollback berhasil: Model aktif kini dikembalikan ke '{target['model_name']}' (v{target['version']})."
            logger.warning(msg)
            return True, msg


# Singleton instance
MODEL_REGISTRY = ModelRegistry()
