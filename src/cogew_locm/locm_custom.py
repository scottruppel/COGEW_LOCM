"""CogEW-specific LOCM variants.

Why this exists:
- Upstream LOCM constructs LOCMBlock with default omega=1.0 and gamma_max=2.0.
- CogEW PRD requires RF-timescale params (e.g. gamma_max=10, omega_init=1000).
"""

from __future__ import annotations

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from locm.dynamics.block import LOCMBlock


class CogEWLOCM(eqx.Module):
    """LOCM with configurable per-block dynamics parameters."""

    encoder: eqx.nn.Linear
    blocks: list
    C_eff: jnp.ndarray
    D: jnp.ndarray
    obs_dim: int
    action_dim: int
    r: int

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        U_r: jnp.ndarray,
        Lambda_r: jnp.ndarray,
        *,
        n_blocks: int = 2,
        omega: float = 1.0,
        alpha: float = 0.1,
        gamma_max: float = 2.0,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        r = U_r.shape[1]
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.r = r
        self.encoder = eqx.nn.Linear(obs_dim, obs_dim, key=jax.random.fold_in(key, 0))
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(
                LOCMBlock(
                    U_r,
                    Lambda_r,
                    u_dim=obs_dim,
                    omega=omega,
                    alpha=alpha,
                    gamma_max=gamma_max,
                    key=jax.random.fold_in(key, i + 1),
                )
            )
        self.C_eff = (
            jax.random.normal(jax.random.fold_in(key, 100), (action_dim, r)) * 0.1
        )
        self.D = (
            jax.random.normal(jax.random.fold_in(key, 101), (action_dim, obs_dim)) * 0.01
        )

    def __call__(
        self,
        obs: jnp.ndarray,
        state: Optional[tuple] = None,
        dt: float = 0.01,
    ) -> Tuple[jnp.ndarray, tuple]:
        u = self.encoder(jnp.ravel(obs))
        if state is None:
            y_tilde = jnp.zeros(self.r)
            z_tilde = jnp.zeros(self.r)
        else:
            y_tilde, z_tilde = state
        for block in self.blocks:
            y_tilde, z_tilde = block(y_tilde, z_tilde, u, dt)
        action = self.C_eff @ y_tilde + self.D @ u
        return action, (y_tilde, z_tilde)

