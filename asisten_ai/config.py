"""
asisten_ai.config - Konfigurasi Terpusat untuk Sistem Asisten-AI Autonomous
"""
import os
import json
from pathlib import Path
from pydantic import BaseModel, Field

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = DATA_DIR / "storage"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
MODELS_DIR = DATA_DIR / "models"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
DATASETS_DIR = DATA_DIR / "datasets"
EVALS_DIR = DATA_DIR / "evaluations"
LOGS_DIR = DATA_DIR / "logs"

for p in [DATA_DIR, STORAGE_DIR, KNOWLEDGE_DIR, MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, EVALS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = DATA_DIR / "system_config.json"
DB_PATH = STORAGE_DIR / "asisten_ai_master.db"


class ResourceLimits(BaseModel):
    max_cpu_percent: float = Field(default=95.0, description="Maksimum penggunaan CPU (%)")
    max_ram_percent: float = Field(default=95.0, description="Maksimum penggunaan RAM (%)")
    min_free_ram_mb: float = Field(default=150.0, description="Batas minimum RAM bebas sebelum throttle (MB)")
    max_disk_usage_gb: float = Field(default=20.0, description="Batas maksimum penyimpanan data Asisten-AI (GB)")
    max_sources_per_topic: int = Field(default=5, description="Jumlah sumber web maksimum per siklus riset")
    max_session_duration_minutes: int = Field(default=30, description="Durasi maksimum sesi belajar otonom (menit)")


class SearchSettings(BaseModel):
    provider: str = Field(default="duckduckgo", description="duckduckgo | searxng | mock")
    searxng_url: str = Field(default="http://localhost:8080", description="URL SearXNG jika digunakan")
    api_key: str = Field(default="", description="API key opsional jika menggunakan custom engine")
    timeout_seconds: int = Field(default=10, description="Timeout permintaan HTTP (detik)")
    verify_ssl: bool = Field(default=True, description="Verifikasi sertifikat SSL")
    rate_limit_delay_seconds: float = Field(default=1.5, description="Jeda antar pencarian untuk mencegah rate-limiting")


class SchedulerSettings(BaseModel):
    mode: str = Field(default="manual", description="manual | scheduled | autonomous | offline")
    interval_minutes: int = Field(default=60, description="Interval antar siklus belajar (menit)")
    active_hours_start: int = Field(default=8, description="Jam mulai siklus otomatis (0-23)")
    active_hours_end: int = Field(default=22, description="Jam selesai siklus otomatis (0-23)")
    auto_train_candidates: bool = Field(default=False, description="Otomatis melatih model kandidat jika dataset memenuhi syarat")
    min_dataset_size_for_training: int = Field(default=20, description="Jumlah minimum pasangan QA untuk memicu training")


class TrainingSettings(BaseModel):
    base_model_name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", description="Nama atau path model dasar HuggingFace")
    use_lora: bool = Field(default=True, description="Gunakan PEFT LoRA jika perangkat mendukung")
    lora_r: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA alpha")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    batch_size: int = Field(default=1, description="Batch size untuk fine-tuning")
    epochs: int = Field(default=3, description="Jumlah epoch")
    min_accuracy_gain_for_promotion: float = Field(default=0.03, description="Peningkatan akurasi minimum (+3%) agar kandidat dipromosikan")
    max_regression_tolerance: float = Field(default=0.02, description="Toleransi regresi maksimum (2%) pada kemampuan lama")


class PrivacySettings(BaseModel):
    allow_internet_search: bool = Field(default=True, description="Izinkan riset online keluar")
    allow_local_telemetry: bool = Field(default=False, description="Kirim telemetri keluar (selalu False)")
    sandbox_code_execution: bool = Field(default=True, description="Blokir eksekusi kode acak dari web")
    strip_unsafe_prompts: bool = Field(default=True, description="Filter prompt injection dari teks web")


class SystemConfig(BaseModel):
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    search: SearchSettings = Field(default_factory=SearchSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)

    def save(self, filepath: Path = CONFIG_FILE) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path = CONFIG_FILE) -> "SystemConfig":
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.model_validate(data)
            except Exception:
                pass
        config = cls()
        config.save(filepath)
        return config


# Global Singleton
SYSTEM_CONFIG = SystemConfig.load()
