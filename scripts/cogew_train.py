"""BC + optional RL training for CogEW-LOCM (PRD Steps 4–5)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cogew_locm.spectral_graph import build_rf_graph, build_and_save_spectrum
from cogew_locm.locm_ew_controller import LOCMEWController
from cogew_locm.expert_demos import generate_all_experts
from cogew_locm.threat_model import AdaptiveRadar
from cogew_locm.reward import compute_reward
from locm.training.bc import run_bc_epoch, bc_loss
from locm.connectome.spectrum import load_spectrum
from locm.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_spectrum(config: dict, data_dir: Path) -> tuple:
    """Build graph and spectrum if not present; return (U_r, Lambda_r)."""
    spec_dir = data_dir / "U_r.npy"
    if spec_dir.parent.exists() and (data_dir / "U_r.npy").exists():
        U_r, Lambda_r = load_spectrum(data_dir)
        return U_r, Lambda_r
    W_rf, _, _ = build_rf_graph(
        n_channels=config["n_channels"],
        freq_range=(config["freq_range_ghz"][0] * 1e9, config["freq_range_ghz"][1] * 1e9),
        spectral_adjacency_weight=config.get("spectral_adjacency_weight", 1.0),
        next_neighbor_weight=config.get("next_neighbor_weight", 0.5),
        hardware_contention_weight=config.get("hardware_contention_weight", -0.5),
        threat_band_weight=config.get("threat_band_weight", 2.0),
        threat_center_ghz=config.get("radar_freq_ghz"),
        threat_bandwidth_ghz=config.get("radar_bandwidth_mhz", 50) / 1000.0,
    )
    U_r, Lambda_r = build_and_save_spectrum(W_rf, config.get("r", 32), out_dir=str(data_dir))
    return U_r, Lambda_r


def run_bc(config: dict, data_dir: Path, exp_dir: Path, seed: int) -> LOCMEWController:
    """Generate experts, run BC, save checkpoint."""
    U_r, Lambda_r = ensure_spectrum(config, data_dir)
    obs_dim = config.get("obs_dim", 64)
    action_dim = config.get("action_dim", 32)
    n_blocks = config.get("n_blocks", 2)
    omega = float(config.get("omega_init", 1.0))
    alpha = float(config.get("alpha", 0.1))
    gamma_max = float(config.get("gamma_max", 2.0))
    bc_epochs = config.get("bc_epochs", 50)
    bc_lr = config.get("bc_lr", 1e-3)
    bc_expert_episodes = config.get("bc_expert_episodes", 200)

    obs_batch, action_batch = generate_all_experts(
        n_noise=bc_expert_episodes // 3,
        n_drfm=bc_expert_episodes // 3,
        n_transition=bc_expert_episodes // 6,
        steps_per_episode=500,
        obs_dim=obs_dim,
        n_channels=action_dim,
        seed=seed,
    )

    key = jax.random.PRNGKey(seed)
    controller = LOCMEWController(
        U_r,
        Lambda_r,
        obs_dim,
        action_dim,
        n_blocks=n_blocks,
        omega=omega,
        alpha=alpha,
        gamma_max=gamma_max,
        key=key,
    )
    model = controller.model

    losses = []
    for epoch in range(bc_epochs):
        loss, model = run_bc_epoch(model, obs_batch, action_batch, lr=bc_lr, key=key)
        controller.model = model
        losses.append(loss)
        if (epoch + 1) % 10 == 0:
            logger.info("BC epoch %s loss %.6f", epoch + 1, loss)

    controller.model = model
    exp_dir.mkdir(parents=True, exist_ok=True)
    # Save model (pickle; equinox Module is picklable)
    import pickle
    with open(exp_dir / "bc_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(exp_dir / "bc_losses.json", "w") as f:
        json.dump(losses, f)
    logger.info("BC done. Final loss %.6f. Model saved to %s", losses[-1], exp_dir / "bc_model.pkl")
    return controller


def _discounted_returns(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def reinforce_update_from_rollout(
    model,
    obs_seq: np.ndarray,
    eps_seq: np.ndarray,
    adv_seq: np.ndarray,
    dt: float,
    sigma: float,
    lr: float,
):
    """One REINFORCE gradient step using stored eps noise for reproducibility."""

    obs_seq_j = jax.device_put(jnp.asarray(obs_seq, dtype=jnp.float32))
    eps_seq_j = jax.device_put(jnp.asarray(eps_seq, dtype=jnp.float32))
    adv_seq_j = jax.device_put(jnp.asarray(adv_seq, dtype=jnp.float32))

    def loss_fn(m):
        init_state = (jnp.zeros(m.r, dtype=jnp.float32), jnp.zeros(m.r, dtype=jnp.float32))
        def step_fn(carry, x):
            state = carry
            obs_t, eps_t, adv_t = x
            mean, state = m(obs_t, state, dt=dt)
            action = mean + sigma * eps_t
            logp = -0.5 * jnp.sum(((action - mean) / sigma) ** 2) - mean.size * jnp.log(
                sigma * jnp.sqrt(2.0 * jnp.pi)
            )
            loss_t = (-adv_t) * logp
            return state, loss_t

        _, losses = jax.lax.scan(
            step_fn,
            init_state,
            (obs_seq_j, eps_seq_j, adv_seq_j),
        )
        return jnp.mean(losses)

    import equinox as eqx

    loss, grad = eqx.filter_value_and_grad(loss_fn)(model)
    model = eqx.apply_updates(model, jax.tree_util.tree_map(lambda g: -lr * g, grad))
    return float(loss), model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "cogew.yaml"))
    parser.add_argument("--bc-only", action="store_true", help="Only run BC, skip RL")
    parser.add_argument("--rl-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    data_dir = ROOT / config.get("data_dir", "data/cogew")
    exp_dir = ROOT / config.get("experiment_dir", "experiments/cogew_demo")
    exp_dir.mkdir(parents=True, exist_ok=True)

    controller = run_bc(config, data_dir, exp_dir, seed)

    if args.bc_only:
        with open(exp_dir / "config.json", "w") as f:
            json.dump({**config, "seed": seed, "phase": "bc_only"}, f, indent=2)
        return

    # RL fine-tuning: REINFORCE with fixed Gaussian exploration around policy mean.
    rl_episodes = args.rl_episodes or config.get("rl_episodes", 200)
    radar = AdaptiveRadar(
        alpha_acquire=config["alpha_acquire"],
        alpha_degrade=config["alpha_degrade"],
        adaptation_threshold=config["adaptation_threshold"],
    )
    sigma = float(config.get("rl_sigma", 0.3))
    lr = float(config.get("rl_lr", 3e-4))
    gamma = float(config.get("rl_gamma", 1.0))
    baseline = 0.0
    baseline_beta = float(config.get("rl_baseline_beta", 0.95))
    losses = []
    returns = []

    duration_ms = float(config.get("rl_window_ms", 30.0))
    macro_step_us = float(config["macro_step_us"])
    dt = macro_step_us * 1e-6

    for ep in range(rl_episodes):
        rng = np.random.default_rng(seed + ep)
        radar.reset()
        controller.reset()

        # Force a single adaptation near center of training window.
        adapt_ms = float(config.get("rl_adapt_ms", 15.0))
        adapt_t = adapt_ms * 1e-3
        steps = int((duration_ms * 1e-3) / dt)

        obs_seq = np.zeros((steps, controller.obs_dim), dtype=np.float32)
        eps_seq = rng.standard_normal((steps, controller.action_dim)).astype(np.float32)
        rew_seq = np.zeros((steps,), dtype=np.float32)

        prev_action = None
        state = None
        for k in range(steps):
            t = k * dt
            if t >= adapt_t and radar.mode == "TWS":
                radar.mode = "STT"
                radar._time_since_mode_change = 0.0
            threat_state = radar._state()
            obs = threat_state.to_obs(controller.obs_dim)
            obs_seq[k] = obs

            mean, state = controller.model(jax.numpy.asarray(obs), state, dt=dt)
            action = np.asarray(mean) + sigma * eps_seq[k]
            action = np.clip(action, -1.0, 1.0)
            technique = "locm_mid" if abs(t - adapt_t) < 0.0002 else (
                "noise" if threat_state.mode == "TWS" else "drfm"
            )
            radar.step(action, dt, technique=technique)

            rew = compute_reward(threat_state, action, prev_action)
            rew_seq[k] = rew
            prev_action = action

        G = _discounted_returns(rew_seq, gamma=gamma)
        ep_return = float(G[0])
        baseline = baseline_beta * baseline + (1.0 - baseline_beta) * ep_return
        adv = (G - baseline).astype(np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        loss, controller.model = reinforce_update_from_rollout(
            controller.model, obs_seq, eps_seq, adv, dt=dt, sigma=sigma, lr=lr
        )
        losses.append(loss)
        returns.append(ep_return)

        if (ep + 1) % 10 == 0:
            logger.info(
                "RL ep %s/%s return=%.3f baseline=%.3f loss=%.4f",
                ep + 1,
                rl_episodes,
                ep_return,
                baseline,
                loss,
            )

    with open(exp_dir / "config.json", "w") as f:
        json.dump({**config, "seed": seed, "phase": "bc_rl_reinforce"}, f, indent=2)
    with open(exp_dir / "rl_returns.json", "w") as f:
        json.dump(returns, f)
    with open(exp_dir / "rl_losses.json", "w") as f:
        json.dump(losses, f)
    logger.info("Training complete. Results in %s", exp_dir)


if __name__ == "__main__":
    main()
