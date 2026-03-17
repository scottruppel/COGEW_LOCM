"""Generate expert (obs, action) pairs for BC: noise, DRFM, interpolated transition (PRD §4.4)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from cogew_locm.threat_model import AdaptiveRadar, ThreatState
from locm.utils.logging import get_logger

logger = get_logger(__name__)


def _noise_action(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal(n).astype(np.float32) * scale
    return np.clip(a, -1.0, 1.0)


def _drfm_action(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    base = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    a = base * scale + rng.standard_normal(n).astype(np.float32) * 0.1
    return np.clip(a, -1.0, 1.0)


def generate_noise_expert(
    n_channels: int = 32,
    obs_dim: int = 64,
    n_steps: int = 500,
    dt: float = 10e-6,
    alpha_degrade: float = 10.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Radar in TWS; noise jamming actions. Returns (obs_batch, action_batch)."""
    rng = np.random.default_rng(seed)
    radar = AdaptiveRadar(alpha_acquire=0.5, alpha_degrade=alpha_degrade)
    radar.mode = "TWS"
    radar.Q = 0.2
    obs_list, action_list = [], []
    for _ in range(n_steps):
        state = radar._state()
        obs = state.to_obs(obs_dim=obs_dim)
        action = _noise_action(n_channels, 0.8, rng)
        obs_list.append(obs)
        action_list.append(action)
        radar.step(action, dt, technique="noise")
    return np.stack(obs_list), np.stack(action_list)


def generate_drfm_expert(
    n_channels: int = 32,
    obs_dim: int = 64,
    n_steps: int = 500,
    dt: float = 10e-6,
    alpha_degrade: float = 10.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Radar in STT; DRFM actions."""
    rng = np.random.default_rng(seed)
    radar = AdaptiveRadar(alpha_acquire=0.5, alpha_degrade=alpha_degrade)
    radar.mode = "STT"
    radar.Q = 0.2
    obs_list, action_list = [], []
    for _ in range(n_steps):
        state = radar._state()
        obs = state.to_obs(obs_dim=obs_dim)
        action = _drfm_action(n_channels, 0.8, rng)
        obs_list.append(obs)
        action_list.append(action)
        radar.step(action, dt, technique="drfm")
    return np.stack(obs_list), np.stack(action_list)


def generate_transition_expert(
    n_channels: int = 32,
    obs_dim: int = 64,
    n_steps: int = 100,
    dt: float = 1e-6,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear blend from noise to DRFM over n_steps (e.g. 100 μs)."""
    rng = np.random.default_rng(seed)
    radar = AdaptiveRadar()
    radar.mode = "TWS"
    radar.Q = 0.25
    obs_list, action_list = [], []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        state = radar._state()
        obs = state.to_obs(obs_dim=obs_dim)
        a_noise = _noise_action(n_channels, 0.8, rng)
        a_drfm = _drfm_action(n_channels, 0.8, rng)
        action = (1 - t) * a_noise + t * a_drfm
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        obs_list.append(obs)
        action_list.append(action)
        technique = "locm_mid" if 0.2 < t < 0.8 else ("noise" if t <= 0.2 else "drfm")
        radar.step(action, dt, technique=technique)
    return np.stack(obs_list), np.stack(action_list)


def generate_all_experts(
    n_noise: int = 100,
    n_drfm: int = 100,
    n_transition: int = 50,
    steps_per_episode: int = 500,
    obs_dim: int = 64,
    n_channels: int = 32,
    seed: int | None = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate and concatenate all expert demos. Returns (obs_batch, action_batch)."""
    rng = np.random.default_rng(seed)
    obs_parts, action_parts = [], []
    for _ in range(n_noise):
        o, a = generate_noise_expert(
            n_channels=n_channels, obs_dim=obs_dim, n_steps=steps_per_episode, seed=rng.integers(0, 2**31)
        )
        obs_parts.append(o)
        action_parts.append(a)
    for _ in range(n_drfm):
        o, a = generate_drfm_expert(
            n_channels=n_channels, obs_dim=obs_dim, n_steps=steps_per_episode, seed=rng.integers(0, 2**31)
        )
        obs_parts.append(o)
        action_parts.append(a)
    for _ in range(n_transition):
        o, a = generate_transition_expert(
            n_channels=n_channels, obs_dim=obs_dim, n_steps=100, seed=rng.integers(0, 2**31)
        )
        obs_parts.append(o)
        action_parts.append(a)
    obs_batch = np.concatenate(obs_parts, axis=0)
    action_batch = np.concatenate(action_parts, axis=0)
    logger.info("Expert demos: %s (obs %s, action %s)", obs_batch.shape[0], obs_batch.shape, action_batch.shape)
    return obs_batch, action_batch
