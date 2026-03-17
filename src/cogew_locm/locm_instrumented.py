"""Instrumented LOCM blocks for logging y_tilde and gamma_tilde during eval."""

from __future__ import annotations

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from locm.dynamics.spectral_step import spectral_step
from locm.dynamics.damping_mlp import DampingMLP


class InstrumentedLOCMBlock(eqx.Module):
    damping_mlp: DampingMLP
    glu_w1: jnp.ndarray
    glu_w2: jnp.ndarray
    glu_b1: jnp.ndarray
    glu_b2: jnp.ndarray
    omega_sq: jnp.ndarray
    alpha_lambda: jnp.ndarray
    B_tilde: jnp.ndarray

    def __init__(
        self,
        U_r: jnp.ndarray,
        Lambda_r: jnp.ndarray,
        u_dim: int,
        *,
        omega: float = 1.0,
        alpha: float = 0.1,
        gamma_max: float = 2.0,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        r = U_r.shape[1]
        self.omega_sq = jnp.ones(r) * (omega**2)
        self.alpha_lambda = jnp.asarray(Lambda_r, dtype=jnp.float32) * alpha
        self.B_tilde = jax.random.normal(jax.random.fold_in(key, 0), (r, u_dim)) * 0.01
        self.damping_mlp = DampingMLP(
            r, u_dim, gamma_max=gamma_max, key=jax.random.fold_in(key, 1)
        )
        k2 = jax.random.fold_in(key, 2)
        self.glu_w1 = jax.random.normal(k2, (r, r)) * 0.02
        self.glu_w2 = jax.random.normal(jax.random.fold_in(k2, 1), (r, r)) * 0.02
        self.glu_b1 = jnp.zeros(r)
        self.glu_b2 = jnp.zeros(r)

    def __call__(
        self, y_tilde: jnp.ndarray, z_tilde: jnp.ndarray, u: jnp.ndarray, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (y_out, z_next, gamma_tilde, y_next_pre_glu)."""
        B_dot_u = self.B_tilde @ jnp.ravel(u)
        gamma_tilde = self.damping_mlp(y_tilde, z_tilde, u, dt)
        y_next, z_next = spectral_step(
            y_tilde,
            z_tilde,
            u,
            dt,
            self.omega_sq,
            self.alpha_lambda,
            B_dot_u,
            gamma_tilde,
        )
        y_out = jax.nn.sigmoid(self.glu_w1 @ y_next + self.glu_b1) * (
            self.glu_w2 @ y_next + self.glu_b2
        )
        return y_out, z_next, gamma_tilde, y_next


class InstrumentedCogEWLOCM(eqx.Module):
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
                InstrumentedLOCMBlock(
                    U_r,
                    Lambda_r,
                    u_dim=obs_dim,
                    omega=omega,
                    alpha=alpha,
                    gamma_max=gamma_max,
                    key=jax.random.fold_in(key, i + 1),
                )
            )
        self.C_eff = jax.random.normal(jax.random.fold_in(key, 100), (action_dim, r)) * 0.1
        self.D = jax.random.normal(jax.random.fold_in(key, 101), (action_dim, obs_dim)) * 0.01

    def __call__(self, obs: jnp.ndarray, state: Optional[tuple] = None, dt: float = 0.01):
        u = self.encoder(jnp.ravel(obs))
        if state is None:
            y_tilde = jnp.zeros(self.r)
            z_tilde = jnp.zeros(self.r)
        else:
            y_tilde, z_tilde = state

        gammas = []
        y_pre = None
        for block in self.blocks:
            y_tilde, z_tilde, gamma_tilde, y_next_pre_glu = block(y_tilde, z_tilde, u, dt)
            gammas.append(gamma_tilde)
            y_pre = y_next_pre_glu

        action = self.C_eff @ y_tilde + self.D @ u
        debug = {
            "y_tilde": y_tilde,
            "y_next_pre_glu": y_pre if y_pre is not None else y_tilde,
            "gamma_tilde": jnp.stack(gammas) if gammas else jnp.zeros((0, self.r)),
        }
        return action, (y_tilde, z_tilde), debug


def convert_cogew_model_to_instrumented(
    *,
    src_model,
    U_r: jnp.ndarray,
    Lambda_r: jnp.ndarray,
    n_blocks: int,
    omega: float,
    alpha: float,
    gamma_max: float,
    key: jax.random.PRNGKey,
) -> InstrumentedCogEWLOCM:
    """
    Create an InstrumentedCogEWLOCM and copy parameters from a CogEWLOCM-like model.

    This enables logging y_tilde/gamma_tilde during eval while using the same learned weights.
    """
    dst = InstrumentedCogEWLOCM(
        src_model.obs_dim,
        src_model.action_dim,
        U_r,
        Lambda_r,
        n_blocks=n_blocks,
        omega=omega,
        alpha=alpha,
        gamma_max=gamma_max,
        key=key,
    )

    # Copy top-level weights.
    dst = eqx.tree_at(lambda m: m.encoder, dst, src_model.encoder)
    dst = eqx.tree_at(lambda m: m.C_eff, dst, src_model.C_eff)
    dst = eqx.tree_at(lambda m: m.D, dst, src_model.D)

    # Copy per-block weights.
    new_blocks = []
    for b_src, b_dst in zip(src_model.blocks, dst.blocks):
        b_dst = eqx.tree_at(lambda b: b.glu_w1, b_dst, b_src.glu_w1)
        b_dst = eqx.tree_at(lambda b: b.glu_w2, b_dst, b_src.glu_w2)
        b_dst = eqx.tree_at(lambda b: b.glu_b1, b_dst, b_src.glu_b1)
        b_dst = eqx.tree_at(lambda b: b.glu_b2, b_dst, b_src.glu_b2)
        b_dst = eqx.tree_at(lambda b: b.omega_sq, b_dst, b_src.omega_sq)
        b_dst = eqx.tree_at(lambda b: b.alpha_lambda, b_dst, b_src.alpha_lambda)
        b_dst = eqx.tree_at(lambda b: b.B_tilde, b_dst, b_src.B_tilde)
        b_dst = eqx.tree_at(lambda b: b.damping_mlp, b_dst, b_src.damping_mlp)
        new_blocks.append(b_dst)
    dst = eqx.tree_at(lambda m: m.blocks, dst, new_blocks)
    return dst

