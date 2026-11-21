"""Where models run, and the computer a run was made on."""

import os
import platform


def pick_device(device: str | None = None) -> str:
    """Return `device` if given, else "cuda" when a GPU is available, else "cpu"."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def computer_info() -> dict:
    """The computer's details, saved next to timings and predictions."""
    import torch

    return {
        "machine": platform.machine(), "system": platform.system(), "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
