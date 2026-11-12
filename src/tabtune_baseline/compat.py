"""Narrow runtime fixes for verified upstream platform defects."""

from __future__ import annotations

import importlib.metadata
import sys

import psutil


def apply_tabtune_compat(model_name: str) -> str | None:
    """Fix TabTune 0.2.1's vendored TabPFN CPU memory probe on Windows.

    That release calls ``os.sysconf`` and then imports a helper absent from
    modern standalone TabPFN. Patch only the affected class and release; never
    edit TabTune's installed files.
    """
    if sys.platform != "win32" or model_name != "TabPFN":
        return None
    if importlib.metadata.version("tabtune") != "0.2.1":
        return None
    from tabtune.models.tabpfn.architectures.base.memory import MemoryUsageEstimator

    if getattr(MemoryUsageEstimator, "_baseline_windows_patch", False):
        return "Windows TabPFN CPU memory probe compatibility"
    original = MemoryUsageEstimator.get_max_free_memory

    @classmethod
    def patched(cls, device, *, unit="gb", default_gb_cpu_if_failed_to_calculate=8):
        if device.type.startswith("cpu"):
            return cls.convert_bytes_to_unit(psutil.virtual_memory().total, unit)
        return original(device, unit=unit,
                        default_gb_cpu_if_failed_to_calculate=default_gb_cpu_if_failed_to_calculate)

    MemoryUsageEstimator.get_max_free_memory = patched
    MemoryUsageEstimator._baseline_windows_patch = True
    return "Windows TabPFN CPU memory probe compatibility"
