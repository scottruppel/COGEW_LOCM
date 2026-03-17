"""Conventional hard-switch jammer with 50 μs gap (PRD §3.4)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from locm.utils.logging import get_logger

if TYPE_CHECKING:
    from cogew_locm.threat_model import ThreatState

logger = get_logger(__name__)


class ConventionalJammer:
    """Noise -> (gap) -> DRFM state machine. Returns zeros during transition."""

    def __init__(
        self,
        n_channels: int = 32,
        transition_duration_us: float = 50.0,
        noise_scale: float = 0.8,
        drfm_scale: float = 0.8,
        seed: int | None = None,
    ):
        self.n_channels = n_channels
        self.transition_duration = transition_duration_us * 1e-6  # seconds
        self.noise_scale = noise_scale
        self.drfm_scale = drfm_scale
        self._rng = np.random.default_rng(seed)
        self.technique: str = "noise"
        self._transition_timer = 0.0

    def reset(self) -> None:
        self.technique = "noise"
        self._transition_timer = 0.0

    def step(self, threat_state: "ThreatState", dt: float) -> tuple[np.ndarray, str]:
        """
        Returns (action_32, technique) where technique is 'noise'|'transitioning'|'drfm'.
        During transitioning, action is zeros (gap).
        """
        # Need to switch to DRFM when threat goes STT
        if self.technique == "noise" and threat_state.mode == "STT":
            self.technique = "transitioning"
            self._transition_timer = self.transition_duration

        if self.technique == "transitioning":
            self._transition_timer -= dt
            if self._transition_timer <= 0:
                self.technique = "drfm"
            return np.zeros(self.n_channels, dtype=np.float32), "transitioning"

        if self.technique == "noise":
            action = self._rng.standard_normal(self.n_channels).astype(np.float32) * self.noise_scale
            action = np.clip(action, -1.0, 1.0)
            return action, "noise"
        else:
            # DRFM: coherent copy-like (simplified as correlated amplitudes)
            base = np.linspace(-0.5, 0.5, self.n_channels, dtype=np.float32)
            action = base * self.drfm_scale + self._rng.standard_normal(self.n_channels).astype(np.float32) * 0.1
            action = np.clip(action, -1.0, 1.0)
            return action, "drfm"
