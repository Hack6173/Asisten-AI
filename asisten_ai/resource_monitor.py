"""
asisten_ai.resource_monitor - Pemantauan Sumber Daya Perangkat Keras Nyata (CPU, RAM, GPU, VRAM, Disk)
Menentukan kelayakan perangkat untuk inferensi lokal, riset, dan pelatihan LoRA/QLoRA.
"""
import shutil
import psutil
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HardwareCapabilityReport:
    def __init__(
        self,
        total_ram_gb: float,
        free_ram_gb: float,
        cpu_count: int,
        disk_free_gb: float,
        has_cuda: bool,
        gpu_name: Optional[str] = None,
        total_vram_gb: float = 0.0,
        free_vram_gb: float = 0.0,
        can_run_qlora: bool = False,
        can_run_cpu_lora: bool = False,
        recommendation: str = ""
    ):
        self.total_ram_gb = total_ram_gb
        self.free_ram_gb = free_ram_gb
        self.cpu_count = cpu_count
        self.disk_free_gb = disk_free_gb
        self.has_cuda = has_cuda
        self.gpu_name = gpu_name
        self.total_vram_gb = total_vram_gb
        self.free_vram_gb = free_vram_gb
        self.can_run_qlora = can_run_qlora
        self.can_run_cpu_lora = can_run_cpu_lora
        self.recommendation = recommendation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ram_gb": round(self.total_ram_gb, 2),
            "free_ram_gb": round(self.free_ram_gb, 2),
            "cpu_count": self.cpu_count,
            "disk_free_gb": round(self.disk_free_gb, 2),
            "has_cuda": self.has_cuda,
            "gpu_name": self.gpu_name or "Tidak Terdeteksi (CPU Only)",
            "total_vram_gb": round(self.total_vram_gb, 2),
            "free_vram_gb": round(self.free_vram_gb, 2),
            "can_run_qlora": self.can_run_qlora,
            "can_run_cpu_lora": self.can_run_cpu_lora,
            "recommendation": self.recommendation,
        }


class ResourceMonitor:
    def __init__(self, data_path: str = "."):
        self.data_path = data_path

    def get_realtime_stats(self) -> Dict[str, Any]:
        """Mengambil data nyata penggunaan hardware tanpa simulasi."""
        vm = psutil.virtual_memory()
        cpu_p = psutil.cpu_percent(interval=None)
        disk_u = shutil.disk_usage(self.data_path)

        has_cuda = False
        gpu_name = None
        vram_total_mb = 0.0
        vram_free_mb = 0.0

        if TORCH_AVAILABLE and torch.cuda.is_available():
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            mem_info = torch.cuda.mem_get_info()
            vram_free_mb = mem_info[0] / (1024 * 1024)
            vram_total_mb = mem_info[1] / (1024 * 1024)

        return {
            "cpu_percent": cpu_p,
            "ram_total_mb": round(vm.total / (1024 * 1024), 1),
            "ram_used_mb": round(vm.used / (1024 * 1024), 1),
            "ram_free_mb": round(vm.available / (1024 * 1024), 1),
            "ram_percent": vm.percent,
            "disk_total_gb": round(disk_u.total / (1024**3), 2),
            "disk_free_gb": round(disk_u.free / (1024**3), 2),
            "disk_used_percent": round((disk_u.used / disk_u.total) * 100, 1),
            "has_cuda": has_cuda,
            "gpu_name": gpu_name or "CPU Mode",
            "vram_total_mb": round(vram_total_mb, 1),
            "vram_free_mb": round(vram_free_mb, 1),
        }

    def assess_hardware_capabilities(self) -> HardwareCapabilityReport:
        """
        Evaluasi kemampuan hardware secara realistis.
        Sesuai arahan: Jika perangkat tidak cukup kuat untuk fine-tuning berat,
        jelaskan keterbatasannya secara jujur dan gunakan jalur pelatihan ringan/pengetahuan lokal.
        """
        stats = self.get_realtime_stats()
        total_ram_gb = stats["ram_total_mb"] / 1024.0
        free_ram_gb = stats["ram_free_mb"] / 1024.0
        cpu_count = psutil.cpu_count(logical=True) or 2
        disk_free_gb = stats["disk_free_gb"]

        has_cuda = stats["has_cuda"]
        total_vram_gb = stats["vram_total_mb"] / 1024.0
        free_vram_gb = stats["vram_free_mb"] / 1024.0

        # Kriteria realistis:
        # GPU QLoRA membutuhkan setidaknya 6GB VRAM CUDA
        can_run_qlora = has_cuda and (total_vram_gb >= 5.5)

        # CPU LoRA / Small Transformer Fine-tuning membutuhkan setidaknya 4GB free RAM
        # Jika RAM bebas < 2GB, model training berat akan memicu OOM pada sistem operasi
        can_run_cpu_lora = free_ram_gb >= 2.5 and disk_free_gb >= 5.0

        if can_run_qlora:
            rec = f"GPU {stats['gpu_name']} terdeteksi dengan {total_vram_gb:.1f}GB VRAM. Mendukung fine-tuning QLoRA dan akselerasi GPU."
        elif can_run_cpu_lora:
            rec = f"Sistem CPU Only ({cpu_count} core, RAM bebas {free_ram_gb:.1f}GB). Mendukung fine-tuning CPU LoRA untuk model berukuran kecil (<1B parameter) dan adapter terisolasi."
        else:
            rec = f"RAM bebas terbatas ({free_ram_gb:.1f}GB dari total {total_ram_gb:.1f}GB). Pembelajaran mandiri difokuskan pada Basis Pengetahuan Lokal, Semantic Indexing, dan evaluasi. Fine-tuning model besar dibatasi untuk mencegah crash sistem."

        return HardwareCapabilityReport(
            total_ram_gb=total_ram_gb,
            free_ram_gb=free_ram_gb,
            cpu_count=cpu_count,
            disk_free_gb=disk_free_gb,
            has_cuda=has_cuda,
            gpu_name=stats["gpu_name"],
            total_vram_gb=total_vram_gb,
            free_vram_gb=free_vram_gb,
            can_run_qlora=can_run_qlora,
            can_run_cpu_lora=can_run_cpu_lora,
            recommendation=rec
        )

    def is_safe_to_execute(
        self,
        max_cpu_percent: float = 95.0,
        max_ram_percent: float = 95.0,
        min_free_ram_mb: float = 150.0,
        min_free_disk_gb: float = 2.0
    ) -> Tuple[bool, str]:
        """Memverifikasi batas keamanan sumber daya sebelum operasi berat dimulai."""
        stats = self.get_realtime_stats()

        if stats["ram_percent"] > max_ram_percent:
            return False, f"Penggunaan RAM ({stats['ram_percent']}%) melebihi batas aman ({max_ram_percent}%)."

        if stats["ram_free_mb"] < min_free_ram_mb:
            return False, f"RAM bebas ({stats['ram_free_mb']} MB) di bawah ambang minimum ({min_free_ram_mb} MB)."

        if stats["disk_free_gb"] < min_free_disk_gb:
            return False, f"Sisa ruang disk ({stats['disk_free_gb']} GB) di bawah ambang batas aman ({min_free_disk_gb} GB)."

        return True, "Sumber daya sistem dalam kondisi aman."


# Singleton instance
RESOURCE_MONITOR = ResourceMonitor()
