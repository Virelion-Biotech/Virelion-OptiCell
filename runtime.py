"""Runtime and accelerator capability inspection for OptiCell."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os
import platform
import time


@dataclass(frozen=True)
class RuntimeCapabilities:
    python: str
    platform: str
    cpu_count: int
    torch_available: bool
    cuda_available: bool
    cuda_device_count: int
    gpu_names: tuple[str, ...]
    cellpose_available: bool


def capabilities() -> RuntimeCapabilities:
    torch_available = False; cuda_available = False; cuda_count = 0; gpu_names: tuple[str, ...] = ()
    try:
        import torch
        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        gpu_names = tuple(str(torch.cuda.get_device_name(i)) for i in range(cuda_count))
    except Exception:
        pass
    try:
        import cellpose  # noqa: F401
        cellpose_available = True
    except Exception:
        cellpose_available = False
    return RuntimeCapabilities(platform.python_version(), platform.platform(), os.cpu_count() or 1,
                               torch_available, cuda_available, cuda_count, gpu_names, cellpose_available)


def preferred_accelerator() -> str:
    """Return ``cuda`` when a CUDA device is visible, otherwise ``cpu``."""
    return "cuda" if capabilities().cuda_available else "cpu"


def measure(callable_, *args, **kwargs):
    """Execute a callable and return ``(result, elapsed_seconds)``."""
    started = time.perf_counter(); result = callable_(*args, **kwargs)
    return result, time.perf_counter() - started


def capabilities_dict() -> dict:
    """JSON-ready accelerator/runtime capability snapshot."""
    return asdict(capabilities())


__all__ = ["RuntimeCapabilities", "capabilities", "capabilities_dict", "preferred_accelerator", "measure"]
