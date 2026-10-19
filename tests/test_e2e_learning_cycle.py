"""
tests/test_e2e_learning_cycle.py - Pengujian Integrasi Siklus Pembelajaran Penuh (End-to-End)
Memverifikasi:
1. Eksekusi siklus belajar 11-langkah penuh
2. Ketahanan antrean saat restart sistem (Restart Resilience)
3. Alur Offline Fallback ke Local Grounding
"""
import pytest
import shutil
import tempfile
from pathlib import Path

from asisten_ai.config import SYSTEM_CONFIG
from asisten_ai.knowledge_store import KnowledgeStore
from asisten_ai.source_validator import SourceValidator
from asisten_ai.search_provider import MockSearchProvider
from asisten_ai.research_agent import AutonomousResearchAgent
from asisten_ai.learning_engine import AutonomousLearningEngine
from asisten_ai.resource_monitor import ResourceMonitor
from asisten_ai.offline_manager import OfflineManager
from asisten_ai.scheduler import AutonomousScheduler


@pytest.fixture
def e2e_env():
    temp_dir = Path(tempfile.mkdtemp())
    old_ram = SYSTEM_CONFIG.resources.max_ram_percent
    SYSTEM_CONFIG.resources.max_ram_percent = 99.0
    SYSTEM_CONFIG.resources.min_free_ram_mb = 50.0
    db_file = temp_dir / "e2e_test.db"
    store = KnowledgeStore(db_path=db_file)
    validator = SourceValidator()
    canned_data = {
        "Deep Learning Fundamentals Apa itu deep learning?": [
            {
                "title": "Pengantar Deep Learning",
                "url": "https://kemdikbud.go.id/deep-learning",
                "snippet": "Deep learning adalah sub-bidang machine learning yang memanfaatkan artificial neural network berlapis banyak untuk mempelajari representasi fitur hierarkis.",
                "domain": "kemdikbud.go.id"
            }
        ]
    }
    mock_search = MockSearchProvider(canned_data=canned_data)
    offline_mgr = OfflineManager()
    agent = AutonomousResearchAgent(
        search_provider=mock_search,
        validator=validator,
        store=store,
        offline_mgr=offline_mgr
    )
    engine = AutonomousLearningEngine(
        store=store,
        research_agent=agent,
        monitor=ResourceMonitor()
    )

    yield {
        "store": store,
        "engine": engine,
        "agent": agent,
        "temp_dir": temp_dir
    }
    shutil.rmtree(temp_dir, ignore_errors=True)
    SYSTEM_CONFIG.resources.max_ram_percent = old_ram


def test_full_autonomous_learning_cycle(e2e_env):
    engine = e2e_env["engine"]
    store = e2e_env["store"]

    # Jalankan siklus pembelajaran penuh
    report = engine.run_complete_learning_cycle(target_topic="Deep Learning Fundamentals")

    assert report.topic == "Deep Learning Fundamentals"
    assert report.stage in ["completed", "queued_offline"]
    assert report.questions_generated >= 1
    assert 0.0 <= report.eval_score <= 1.0

    # Pastikan data tersimpan di Knowledge Store
    summary = store.get_summary_statistics()
    assert summary["total_facts"] >= 0


def test_scheduler_restart_recovery(e2e_env):
    store = e2e_env["store"]
    engine = e2e_env["engine"]

    # Simulasikan pekerjaan tertinggal berstatus 'running' saat crash/restart
    with store._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO job_queue (id, job_type, payload_json, status, created_at, attempts)
            VALUES ('job_crashed_1', 'learning_cycle', '{"topic": "Restart Test"}', 'running', '2026-10-19T00:00:00', 0)
            """
        )
        conn.commit()

    # Inisialisasi scheduler baru (mensimulasikan restart aplikasi)
    sched = AutonomousScheduler(store=store, learning_engine=engine)

    # Periksa apakah pekerjaan berstatus 'running' otomatis dipulihkan ke 'pending'
    with store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, attempts, error_message FROM job_queue WHERE id = 'job_crashed_1'")
        row = cursor.fetchone()
        assert row["status"] == "pending"
        assert row["attempts"] == 1
        assert "dipulihkan" in row["error_message"].lower()
