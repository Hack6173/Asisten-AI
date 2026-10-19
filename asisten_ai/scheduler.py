"""
asisten_ai.scheduler - Penjadwal Otonom Berbasis Antrean Pekerjaan Persisten (Job Queue Scheduler)
Mendukung mode: Manual, Scheduled, Autonomous, Offline.
Tahan terhadap restart aplikasi tanpa duplikasi pekerjaan atau kerusakan database.
Dilengkapi kontrol: Start, Pause, Resume, Stop, dan pembatasan sesi/sumber daya.
"""
import time
import json
import logging
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from asisten_ai.config import SYSTEM_CONFIG
from asisten_ai.knowledge_store import KNOWLEDGE_STORE
from asisten_ai.resource_monitor import RESOURCE_MONITOR
from asisten_ai.learning_engine import LEARNING_ENGINE, LearningCycleReport

logger = logging.getLogger(__name__)


class SchedulerStatus(BaseModel):
    is_running: bool
    is_paused: bool
    mode: str  # manual, scheduled, autonomous, offline
    current_job: Optional[Dict[str, Any]] = None
    last_cycle_at: Optional[str] = None
    next_scheduled_at: Optional[str] = None
    total_cycles_completed: int = 0
    active_errors: List[str] = Field(default_factory=list)


class AutonomousScheduler:
    def __init__(
        self,
        store=KNOWLEDGE_STORE,
        learning_engine=LEARNING_ENGINE,
        monitor=RESOURCE_MONITOR
    ):
        self.store = store
        self.learning_engine = learning_engine
        self.monitor = monitor

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._current_job: Optional[Dict[str, Any]] = None
        self._last_cycle_at: Optional[str] = None
        self._next_scheduled_at: Optional[str] = None
        self._total_cycles_completed: int = 0
        self._active_errors: List[str] = []

        # Pulihkan status pekerjaan yang belum selesai akibat restart sebelumnya
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        """Memulihkan pekerjaan yang statusnya 'running' saat aplikasi mati sebelumnya."""
        with self.store._get_connection() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET status = 'pending', attempts = attempts + 1,
                    error_message = 'Dipulihkan secara otomatis setelah sistem di-restart'
                WHERE status = 'running'
                """
            )
            conn.commit()

    def start(self, mode: Optional[str] = None) -> None:
        """Memulai loop background scheduler."""
        if self._thread and self._thread.is_alive():
            logger.info("Scheduler sudah berjalan.")
            return

        if mode:
            SYSTEM_CONFIG.scheduler.mode = mode
            SYSTEM_CONFIG.save()

        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        logger.info(f"Autonomous Scheduler dimulai dalam mode '{SYSTEM_CONFIG.scheduler.mode}'.")

    def pause(self) -> None:
        self._pause_event.set()
        logger.info("Autonomous Scheduler di-pause.")

    def resume(self) -> None:
        self._pause_event.clear()
        logger.info("Autonomous Scheduler di-resume.")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._current_job = None
        logger.info("Autonomous Scheduler dihentikan.")

    def trigger_immediate_cycle(self, topic: Optional[str] = None) -> LearningCycleReport:
        """Memicu satu siklus belajar langsung (Mode Manual atau Eksekusi Segera)."""
        logger.info(f"Memicu siklus belajar mandiri segera (topik: {topic or 'Otonom'})...")
        self._current_job = {
            "type": "immediate_learning_cycle",
            "topic": topic or "Otonom",
            "started_at": datetime.now().isoformat()
        }
        try:
            report = self.learning_engine.run_complete_learning_cycle(target_topic=topic)
            self._last_cycle_at = datetime.now().isoformat()
            self._total_cycles_completed += 1
            return report
        except Exception as e:
            self._active_errors.append(f"Gagal siklus {topic}: {e}")
            raise
        finally:
            self._current_job = None

    def get_status(self) -> SchedulerStatus:
        is_running = self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()
        return SchedulerStatus(
            is_running=is_running,
            is_paused=self._pause_event.is_set(),
            mode=SYSTEM_CONFIG.scheduler.mode,
            current_job=self._current_job,
            last_cycle_at=self._last_cycle_at,
            next_scheduled_at=self._next_scheduled_at,
            total_cycles_completed=self._total_cycles_completed,
            active_errors=self._active_errors[-5:],
        )

    def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(2)
                continue

            mode = SYSTEM_CONFIG.scheduler.mode

            if mode in ["scheduled", "autonomous"]:
                now = datetime.now()
                # Periksa jam aktif
                if SYSTEM_CONFIG.scheduler.active_hours_start <= now.hour < SYSTEM_CONFIG.scheduler.active_hours_end:
                    # Periksa batas sumber daya sebelum memulai
                    safe, reason = self.monitor.is_safe_to_execute()
                    if safe:
                        try:
                            self.trigger_immediate_cycle(topic=None)
                        except Exception as e:
                            logger.error(f"Error pada loop otonom: {e}")
                    else:
                        logger.warning(f"Siklus dilewati karena sumber daya: {reason}")

                # Jadwalkan waktu siklus berikutnya
                interval_secs = max(60, SYSTEM_CONFIG.scheduler.interval_minutes * 60)
                self._next_scheduled_at = (datetime.now() + timedelta(seconds=interval_secs)).isoformat()

                # Tunggu interval dengan pengecekan stop event tiap detik
                for _ in range(interval_secs):
                    if self._stop_event.is_set() or self._pause_event.is_set():
                        break
                    time.sleep(1)

            elif mode == "offline":
                # Mode offline: pelajari materi lokal dan jalankan evaluasi jika ada
                safe, _ = self.monitor.is_safe_to_execute()
                if safe:
                    try:
                        self.trigger_immediate_cycle(topic=None)
                    except Exception as e:
                        logger.warning(f"Siklus offline terhambat: {e}")

                time.sleep(10)
            else:
                # Mode manual: tidak melakukan polling otomatis
                time.sleep(3)


# Singleton instance
SCHEDULER = AutonomousScheduler()
