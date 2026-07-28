"""System resource snapshot (CPU / RAM / disk / GPU-MPS). Fail-safe."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def resource_snapshot(disk_path: str | Path = ".") -> dict[str, Any]:
    snap: dict[str, Any] = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap["cpu_percent"] = psutil.cpu_percent(interval=None)
        snap["cpu_count"] = psutil.cpu_count()
        snap["ram_used_gb"] = round(vm.used / 1e9, 2)
        snap["ram_total_gb"] = round(vm.total / 1e9, 2)
        snap["ram_percent"] = vm.percent
        du = psutil.disk_usage(str(disk_path))
        snap["disk_used_gb"] = round(du.used / 1e9, 2)
        snap["disk_total_gb"] = round(du.total / 1e9, 2)
        snap["disk_percent"] = du.percent
    except Exception as e:  # psutil missing / platform quirk
        snap["error"] = f"psutil unavailable: {e}"

    # Accelerator memory (NVIDIA via torch.cuda, Apple via torch.mps)
    try:
        import torch

        if torch.cuda.is_available():
            snap["device"] = "cuda"
            snap["gpu_mem_alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
            snap["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
        elif torch.backends.mps.is_available():
            snap["device"] = "mps"
            if hasattr(torch.mps, "current_allocated_memory"):
                snap["mps_mem_alloc_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 3)
        else:
            snap["device"] = "cpu"
    except Exception:
        pass
    return snap
