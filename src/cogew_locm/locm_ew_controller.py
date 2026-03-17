"""LOCM-EW controller: thin wrapper around LOCM with persistent state (PRD §3.5)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from locm.connectome.spectrum import load_spectrum
from locm.utils.logging import get_logger

from cogew_locm.locm_custom import CogEWLOCM
from cogew_locm.locm_instrumented import InstrumentedCogEWLOCM

logger = get_logger(__name__)


class LOCMEWController:
    """Uses LOCM on spectral coupling graph; output is always defined (no gap)."""

    def __init__(
        self,
        U_r: np.ndarray | jnp.ndarray,
        Lambda_r: np.ndarray | jnp.ndarray,
        obs_dim: int = 64,
        action_dim: int = 32,
        n_blocks: int = 2,
        omega: float = 1.0,
        alpha: float = 0.1,
        gamma_max: float = 2.0,
        dt_default: float = 1e-6,
        key: Optional[jax.random.PRNGKey] = None,
        instrument: bool = False,
    ):
        U_r = jnp.asarray(U_r)
        Lambda_r = jnp.asarray(Lambda_r)
        if key is None:
            key = jax.random.PRNGKey(0)
        if instrument:
            self.model = InstrumentedCogEWLOCM(
                obs_dim,
                action_dim,
                U_r,
                Lambda_r,
                n_blocks=n_blocks,
                omega=omega,
                alpha=alpha,
                gamma_max=gamma_max,
                key=key,
            )
        else:
            self.model = CogEWLOCM(
                obs_dim,
                action_dim,
                U_r,
                Lambda_r,
                n_blocks=n_blocks,
                omega=omega,
                alpha=alpha,
                gamma_max=gamma_max,
                key=key,
            )
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dt_default = dt_default
        self._state: Optional[tuple] = None
        self._instrument = instrument

    def reset(self) -> None:
        self._state = None

    def step(
        self,
        obs: np.ndarray | jnp.ndarray,
        dt: Optional[float] = None,
        *,
        return_debug: bool = False,
    ) -> np.ndarray:
        """Returns 32-dim action (channel amplitudes), always defined."""
        obs = jnp.asarray(obs, dtype=jnp.float32).ravel()[: self.obs_dim]
        if obs.size < self.obs_dim:
            obs = jnp.pad(obs, (0, self.obs_dim - obs.size))
        dt = dt if dt is not None else self.dt_default
        if self._instrument:
            out = self.model(obs, self._state, dt=dt)
            # Instrumented models return (action, state, debug). If a non-instrumented
            # checkpoint was loaded, it may return (action, state) instead.
            if isinstance(out, tuple) and len(out) == 3:
                action, self._state, debug = out
                action = np.clip(np.asarray(action), -1.0, 1.0)
                if return_debug:
                    debug_np = {
                        "y_tilde": np.asarray(debug["y_tilde"]),
                        "y_next_pre_glu": np.asarray(debug["y_next_pre_glu"]),
                        "gamma_tilde": np.asarray(debug["gamma_tilde"]),
                    }
                    return action, debug_np
                return action
            action, self._state = out
            action = np.clip(np.asarray(action), -1.0, 1.0)
            if return_debug:
                return action, {"y_tilde": np.array([]), "y_next_pre_glu": np.array([]), "gamma_tilde": np.array([])}
            return action

        action, self._state = self.model(obs, self._state, dt=dt)
        action = np.clip(np.asarray(action), -1.0, 1.0)
        return action

    @classmethod
    def from_spectrum_dir(
        cls,
        spectrum_dir: str | Path,
        obs_dim: int = 64,
        action_dim: int = 32,
        n_blocks: int = 2,
        omega: float = 1.0,
        alpha: float = 0.1,
        gamma_max: float = 2.0,
        r: Optional[int] = None,
        key: Optional[jax.random.PRNGKey] = None,
        instrument: bool = False,
    ) -> "LOCMEWController":
        """Load U_r, Lambda_r from directory (e.g. data/cogew)."""
        U_r, Lambda_r = load_spectrum(spectrum_dir)
        if r is not None:
            U_r = U_r[:, :r]
            Lambda_r = Lambda_r[:r]
        return cls(
            U_r,
            Lambda_r,
            obs_dim,
            action_dim,
            n_blocks=n_blocks,
            omega=omega,
            alpha=alpha,
            gamma_max=gamma_max,
            key=key,
            instrument=instrument,
        )
