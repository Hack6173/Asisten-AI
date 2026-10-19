import os
import platform
import shutil
import json

info = {
    "os": f"{platform.system()} {platform.release()} ({platform.version()})",
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "python_version": platform.python_version(),
}

total, used, free = shutil.disk_usage(".")
info["disk_total_gb"] = round(total / (1024**3), 2)
info["disk_free_gb"] = round(free / (1024**3), 2)

# Check for CUDA / torch if available
try:
    import torch
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
except ImportError:
    info["torch_installed"] = False

print(json.dumps(info, indent=2))
