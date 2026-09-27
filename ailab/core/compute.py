"""Compute backend selection.

All GPU work goes through NVIDIA Warp: a kernel is written once in Python and runs on
``cuda:0`` or on ``cpu`` without changes. That gives the CPU fallback for free.

Selection order: ``AILAB_DEVICE`` env var / ``--device`` flag, else CUDA if available,
else CPU.
"""

from __future__ import annotations

import os

from .system import SystemInfo

_device: str | None = None


def init(prefer: str = "auto", info: SystemInfo | None = None) -> str:
    """Initialise Warp and choose the device. Returns ``"cuda:0"`` or ``"cpu"``."""
    global _device
    import warp as wp

    if hasattr(wp, "LOG_WARNING"):
        wp.config.log_level = wp.LOG_WARNING
    else:
        wp.config.quiet = True
    wp.init()
    prefer = os.environ.get("AILAB_DEVICE", prefer) or "auto"

    device = "cpu"
    if prefer in ("auto", "cuda") and wp.is_cuda_available():
        device = "cuda:0"
    elif prefer == "cuda":
        print("[ailab] CUDA requested but not available; using the CPU backend.")

    _device = device
    if info is not None:
        info.compute_device = device
        info.warp_version = wp.config.version
        if device.startswith("cuda"):
            d = wp.get_device(device)
            info.compute_name = d.name.replace("NVIDIA ", "")
            try:
                major, minor = wp.get_cuda_driver_version()
                info.cuda_driver_version = f"CUDA {major}.{minor}"
            except Exception:
                pass
            gpu = info.cuda_gpu
            if gpu and not gpu.compute_capability:
                gpu.compute_capability = f"{d.arch // 10}.{d.arch % 10}"
    return device


def device() -> str:
    if _device is None:
        return init()
    return _device


def set_device(name: str) -> str:
    """Switch backend at runtime (the current chapter must be reloaded)."""
    global _device
    import warp as wp

    if name.startswith("cuda") and not wp.is_cuda_available():
        name = "cpu"
    _device = name
    return name


def synchronize() -> None:
    import warp as wp

    if _device and _device.startswith("cuda"):
        wp.synchronize_device(_device)
