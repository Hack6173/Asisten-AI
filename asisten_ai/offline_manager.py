"""
asisten_ai.offline_manager - Pengelola Konektivitas dan Operasi Intelijen Offline
Mendeteksi status jaringan, mengelola antrean riset yang tertunda, dan memastikan
seluruh fitur inti dapat beroperasi tanpa internet.
"""
import socket
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from asisten_ai.config import DATA_DIR

logger = logging.getLogger(__name__)

PENDING_RESEARCH_FILE = DATA_DIR / "pending_research_queue.json"


class OfflineManager:
    def __init__(self, check_host: str = "8.8.8.8", check_port: int = 53, timeout: float = 2.0):
        self.check_host = check_host
        self.check_port = check_port
        self.timeout = timeout
        self._last_status: Optional[bool] = None
        self._last_checked: Optional[datetime] = None

    def is_online(self, force_recheck: bool = False) -> bool:
        """
        Memeriksa koneksi internet secara nyata tanpa membuat asumsi.
        Menggunakan socket connect ke DNS publik (timeout 2s).
        """
        if not force_recheck and self._last_status is not None:
            # Cache status selama 10 detik jika tidak dipaksa recheck
            if self._last_checked and (datetime.now() - self._last_checked).total_seconds() < 10:
                return self._last_status

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.check_host, self.check_port))
            sock.close()
            self._last_status = True
        except (socket.timeout, socket.error, OSError):
            self._last_status = False

        self._last_checked = datetime.now()
        return self._last_status

    def get_connectivity_status(self) -> Dict[str, Any]:
        online = self.is_online(force_recheck=True)
        pending_count = len(self.get_pending_research())
        return {
            "is_online": online,
            "status_text": "Online (Riset Web Aktif)" if online else "Offline (Mode Intelijen Lokal)",
            "pending_research_tasks": pending_count,
            "last_checked": self._last_checked.isoformat() if self._last_checked else None,
        }

    def queue_pending_research(self, topic: str, questions: List[str], origin: str = "gap_analysis") -> Dict[str, Any]:
        """Menyimpan topik riset ke antrean saat offline agar dapat diproses saat online."""
        queue = self.get_pending_research()

        # Cek apakah topik sudah ada di antrean
        for item in queue:
            if item.get("topic", "").strip().lower() == topic.strip().lower():
                # Update pertanyaan jika ada yang baru
                existing_q = set(item.get("questions", []))
                for q in questions:
                    if q not in existing_q:
                        item["questions"].append(q)
                self._save_queue(queue)
                return item

        new_entry = {
            "id": f"queue_{int(datetime.now().timestamp())}_{len(queue)}",
            "topic": topic.strip(),
            "questions": questions,
            "origin": origin,
            "queued_at": datetime.now().isoformat(),
            "status": "pending",
        }
        queue.append(new_entry)
        self._save_queue(queue)
        logger.info(f"Topik riset '{topic}' dimasukkan ke antrean offline ({len(queue)} pending).")
        return new_entry

    def get_pending_research(self) -> List[Dict[str, Any]]:
        """Membaca antrean riset tertunda dari disk."""
        if not PENDING_RESEARCH_FILE.exists():
            return []
        try:
            with open(PENDING_RESEARCH_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Gagal membaca antrean riset offline: {e}")
            return []

    def pop_pending_research(self) -> Optional[Dict[str, Any]]:
        """Mengambil dan menghapus item riset pertama yang siap diproses."""
        queue = self.get_pending_research()
        if not queue:
            return None
        # Cari yang statusnya 'pending'
        for i, item in enumerate(queue):
            if item.get("status") == "pending":
                selected = queue.pop(i)
                self._save_queue(queue)
                return selected
        return None

    def clear_completed_or_failed(self) -> None:
        """Membersihkan tugas yang telah selesai."""
        queue = self.get_pending_research()
        active = [item for item in queue if item.get("status") == "pending"]
        self._save_queue(active)

    def _save_queue(self, queue: List[Dict[str, Any]]) -> None:
        try:
            with open(PENDING_RESEARCH_FILE, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Gagal menyimpan antrean riset offline: {e}")


# Singleton instance
OFFLINE_MANAGER = OfflineManager()
