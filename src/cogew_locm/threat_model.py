"""Adaptive pulse-Doppler radar threat model (PRD §3.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from locm.utils.logging import get_logger

logger = get_logger(__name__)

Mode = Literal["search", "TWS", "STT", "off"]
Technique = Literal["noise", "drfm", "gap", "locm_mid"]


@dataclass
class ThreatState:
    """Threat radar state for observation and reward."""
    Q: float  # track quality [0, 1]
    P: float  # track covariance (scalar proxy)
    mode: Mode
    center_freq_ghz: float
    bandwidth_mhz: float
    prf_hz: float
    time_since_mode_change: float
    effectiveness_estimate: float

    def to_obs(self, obs_dim: int = 64, psd_placeholder: np.ndarray | None = None) -> np.ndarray:
        """Build 64-dim observation vector (PRD §4.1)."""
        # 4: freq estimate (center, bw, PRF, PRI)
        pri = 1.0 / self.prf_hz if self.prf_hz > 0 else 0.0
        freq_est = np.array([
            self.center_freq_ghz / 20.0,
            self.bandwidth_mhz / 1000.0,
            self.prf_hz / 1e6,
            min(pri * 1e6, 1.0),
        ], dtype=np.float32)
        # 4: mode one-hot (search, TWS, STT, off)
        mode_idx = {"search": 0, "TWS": 1, "STT": 2, "off": 3}[self.mode]
        mode_onehot = np.zeros(4, dtype=np.float32)
        mode_onehot[mode_idx] = 1.0
        # 32: PSD per channel (placeholder if not provided)
        if psd_placeholder is not None and psd_placeholder.size >= 32:
            psd = np.asarray(psd_placeholder, dtype=np.float32).ravel()[:32]
        else:
            psd = np.zeros(32, dtype=np.float32)
        # 1: time since mode change (normalized)
        t_norm = min(self.time_since_mode_change / 1.0, 1.0)
        # 1: effectiveness estimate
        eff = np.array([self.effectiveness_estimate], dtype=np.float32)
        # 22: padding
        pad = np.zeros(22, dtype=np.float32)
        obs = np.concatenate([freq_est, mode_onehot, psd, [t_norm], eff, pad])
        if len(obs) < obs_dim:
            obs = np.pad(obs, (0, obs_dim - len(obs)))
        return obs[:obs_dim].astype(np.float32)


class AdaptiveRadar:
    """
    Pulse-Doppler track-while-scan radar. Q improves when unjammed,
    degrades when effectively jammed; adapts (TWS -> STT, PRF/freq) when Q low.
    """

    def __init__(
        self,
        alpha_acquire: float = 5.0,
        alpha_degrade: float = 10.0,
        adaptation_threshold: float = 0.3,
        radar_freq_ghz: float = 9.5,
        radar_bandwidth_mhz: float = 50.0,
        prf_hz: float = 10e3,
        seed: int | None = None,
    ):
        self.alpha_acquire = alpha_acquire
        self.alpha_degrade = alpha_degrade
        self.adaptation_threshold = adaptation_threshold
        self.center_freq_ghz = radar_freq_ghz
        self.bandwidth_mhz = radar_bandwidth_mhz
        self.prf_hz = prf_hz
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> ThreatState:
        self.Q = 0.2
        self.P = 1.0
        self.mode: Mode = "TWS"
        self._time_since_mode_change = 0.0
        self.effectiveness_estimate = 0.0
        return self._state()

    def _state(self) -> ThreatState:
        return ThreatState(
            Q=self.Q,
            P=self.P,
            mode=self.mode,
            center_freq_ghz=self.center_freq_ghz,
            bandwidth_mhz=self.bandwidth_mhz,
            prf_hz=self.prf_hz,
            time_since_mode_change=self._time_since_mode_change,
            effectiveness_estimate=self.effectiveness_estimate,
        )

    def _effectiveness(self, technique: Technique) -> float:
        """Jamming effectiveness: rate multiplier for Q (positive = we degrade their track)."""
        if technique == "gap":
            return 0.0
        if self.mode == "TWS":
            if technique == "noise":
                return 1.0
            if technique == "drfm":
                return 0.2
            if technique == "locm_mid":
                return 0.7
        if self.mode == "STT":
            if technique == "noise":
                return 0.2
            if technique == "drfm":
                return 1.0
            if technique == "locm_mid":
                return 0.6
        return 0.5

    def step(
        self,
        action: np.ndarray,
        dt: float,
        technique: Technique = "noise",
    ) -> ThreatState:
        """
        action: 32-dim channel amplitudes (used for power level; technique is separate).
        technique: noise, drfm, gap, or locm_mid (LOCM-EW during transition).
        """
        eff = self._effectiveness(technique)
        self.effectiveness_estimate = eff
        power = float(np.sum(np.square(action)))

        # Interpret alpha_* as per-millisecond rates (more natural for this demo),
        # so dt is converted to ms here.
        dt_ms = dt * 1e3

        if eff <= 0.0:
            # Not jammed or gap: radar acquires
            self.Q = min(1.0, self.Q + self.alpha_acquire * dt_ms)
            self.P = max(0.01, self.P * (1.0 - 0.5 * dt_ms))
        else:
            # Jammed: track degrades
            self.Q = max(0.0, self.Q - self.alpha_degrade * eff * dt_ms)
            self.P = self.P * (1.0 + 0.3 * eff * dt_ms)

        self._time_since_mode_change += dt
        # Adapt: if Q drops below threshold, switch to STT and change PRF/freq
        if self.Q < self.adaptation_threshold and self.mode == "TWS":
            self.mode = "STT"
            self._time_since_mode_change = 0.0
            self.prf_hz *= 0.9 + 0.2 * self._rng.random()
            self.center_freq_ghz += (self._rng.random() - 0.5) * 0.1

        return self._state()
