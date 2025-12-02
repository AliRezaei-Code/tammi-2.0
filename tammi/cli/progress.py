"""Progress bar and utility classes for CLI."""

from __future__ import annotations

import sys
import time
from typing import Optional


class ProgressBar:
    """Lightweight ASCII progress bar for CLI runs."""

    def __init__(self, total: int, width: int = 30) -> None:
        self.total = total
        self.width = width
        self.enabled = sys.stdout.isatty() and total > 0
        self._start_time: Optional[float] = None

    def update(self, current: int) -> None:
        """Update progress bar display."""
        if not self.enabled:
            return
        
        if self._start_time is None:
            self._start_time = time.perf_counter()
        
        ratio = min(max(current / self.total, 0.0), 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        
        # Calculate ETA
        elapsed = time.perf_counter() - self._start_time
        if current > 0 and ratio < 1.0:
            eta = (elapsed / current) * (self.total - current)
            eta_str = f" ETA: {eta:.0f}s"
        else:
            eta_str = ""
        
        sys.stdout.write(
            f"\rProcessing [{bar}] {current}/{self.total} ({ratio * 100:5.1f}%){eta_str}"
        )
        sys.stdout.flush()

    def close(self) -> None:
        """Close progress bar and print newline."""
        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()


def quick_cpu_probe(sample_size: int = 2000) -> tuple[int, int, float]:
    """
    Run a benchmark to suggest n_process and batch_size.
    
    Returns:
        Tuple of (suggested_n_process, suggested_batch_size, tokens_per_sec)
    """
    import os
    import spacy

    cores = os.cpu_count() or 1
    suggested_n_process = max(1, cores - 1)

    try:
        nlp = spacy.blank("en")
        sample_text = "This is a quick TAMMI benchmark sentence."
        sample = [sample_text] * sample_size
        start = time.perf_counter()
        token_count = 0
        for doc in nlp.pipe(sample, batch_size=100, n_process=1):
            token_count += len(doc)
        duration = max(time.perf_counter() - start, 1e-9)
        tokens_per_sec = token_count / duration
    except Exception:
        tokens_per_sec = 0.0

    if tokens_per_sec >= 40000:
        suggested_batch = 2000
    elif tokens_per_sec >= 20000:
        suggested_batch = 1500
    elif tokens_per_sec >= 10000:
        suggested_batch = 1000
    else:
        suggested_batch = 500

    return suggested_n_process, suggested_batch, tokens_per_sec


def check_gpu_available() -> tuple[bool, str]:
    """
    Check if GPU acceleration is available for spaCy.
    
    Returns:
        Tuple of (is_available, message)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"CUDA GPU available: {device_name}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "Apple MPS (Metal) GPU available"
        else:
            return False, "No GPU detected (torch installed but no CUDA/MPS)"
    except ImportError:
        pass

    try:
        import cupy  # type: ignore[import-not-found]
        return True, "CUDA GPU available via CuPy"
    except ImportError:
        pass

    return False, "No GPU support detected (install torch with CUDA or cupy)"
