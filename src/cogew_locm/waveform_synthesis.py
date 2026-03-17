"""Channel amplitudes -> power / PSD for figures (PRD §4.2, §5.2). No real RF synthesis."""
from __future__ import annotations

import numpy as np


def action_to_power_db(action: np.ndarray) -> float:
    """Total transmitted power in dB relative (log scale)."""
    power = float(np.sum(action ** 2))
    if power <= 0:
        return -np.inf
    return 10.0 * np.log10(power + 1e-12)


def action_to_psd_per_channel(action: np.ndarray) -> np.ndarray:
    """Power per channel (32,) for spectrogram y-axis."""
    return np.square(action).astype(np.float64)
