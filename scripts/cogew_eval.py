"""Run evaluation episodes: LOCM-EW vs conventional jammer; record time series (PRD Step 6)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cogew_locm.locm_ew_controller import LOCMEWController
from cogew_locm.conventional_jammer import ConventionalJammer
from cogew_locm.threat_model import AdaptiveRadar, ThreatState
from cogew_locm.waveform_synthesis import action_to_power_db
from cogew_locm.locm_instrumented import convert_cogew_model_to_instrumented
from locm.connectome.spectrum import load_spectrum
from locm.utils.logging import get_logger

logger = get_logger(__name__)


def _window_mask(t_ms: np.ndarray, adapt_time_ms: float, half_width_ms: float) -> np.ndarray:
    return np.abs(t_ms - adapt_time_ms) <= half_width_ms


def transition_metrics(
    t_ms: np.ndarray,
    Q: np.ndarray,
    power_db: np.ndarray,
    *,
    adapt_time_ms: float,
    half_width_ms: float = 0.2,
) -> dict:
    """Compute metrics in a window around adaptation time."""
    m = _window_mask(t_ms, adapt_time_ms, half_width_ms)
    if not np.any(m):
        return {
            "window_half_width_ms": half_width_ms,
            "max_Q_window": float(np.max(Q)),
            "mean_Q_window": float(np.mean(Q)),
            "min_power_db_window": float(np.nanmin(power_db[np.isfinite(power_db)])) if np.any(np.isfinite(power_db)) else float("-inf"),
        }
    Qw = Q[m]
    Pw = power_db[m]
    finite_Pw = Pw[np.isfinite(Pw)]
    return {
        "window_half_width_ms": float(half_width_ms),
        "max_Q_window": float(np.max(Qw)),
        "mean_Q_window": float(np.mean(Qw)),
        "min_power_db_window": float(np.min(finite_Pw)) if len(finite_Pw) else float("-inf"),
    }


def _bootstrap_ci(x: np.ndarray, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n):
        sample = rng.choice(x, size=x.size, replace=True)
        means.append(float(np.mean(sample)))
    means = np.sort(np.asarray(means))
    lo = means[int((alpha / 2) * len(means))]
    hi = means[int((1 - alpha / 2) * len(means)) - 1]
    return lo, hi


def run_episode_locm_ew(
    controller: LOCMEWController,
    radar: AdaptiveRadar,
    duration_ms: float,
    macro_step_us: float,
    adapt_time_ms: float,
    seed: int,
    *,
    log_spectral: bool = False,
) -> dict:
    """Single episode with LOCM-EW; radar adapts at adapt_time_ms."""
    np.random.seed(seed)
    radar.reset()
    controller.reset()
    adapt_t = adapt_time_ms * 1e-3
    dt = macro_step_us * 1e-6
    total_steps = int((duration_ms * 1e-3) / dt)
    t_ms = []
    Q_hist, power_db, actions, techniques = [], [], [], []
    y_tilde_hist = []
    gamma_hist = []
    for step in range(total_steps):
        t = step * dt
        t_ms.append(t * 1e3)
        if t >= adapt_t and radar.mode == "TWS":
            radar.mode = "STT"
            radar._time_since_mode_change = 0.0
        state = radar._state()
        obs = state.to_obs(controller.obs_dim)
        if log_spectral:
            action, dbg = controller.step(obs, dt=dt, return_debug=True)
            if dbg["y_tilde"].size:
                y_tilde_hist.append(dbg["y_tilde"])
            if dbg["gamma_tilde"].size:
                g = dbg["gamma_tilde"]
                gamma_hist.append(g[-1] if g.ndim == 2 and g.shape[0] else g)
        else:
            action = controller.step(obs, dt=dt)
        technique = "locm_mid" if abs(t - adapt_t) < 0.0002 else ("noise" if state.mode == "TWS" else "drfm")
        radar.step(action, dt, technique=technique)
        Q_hist.append(radar.Q)
        power_db.append(action_to_power_db(action))
        actions.append(action.copy())
        techniques.append(technique)
    return {
        "t_ms": np.array(t_ms),
        "Q": np.array(Q_hist),
        "power_db": np.array(power_db),
        "actions": np.array(actions),
        "techniques": techniques,
        "adapt_time_ms": adapt_time_ms,
        "y_tilde": np.array(y_tilde_hist) if (log_spectral and len(y_tilde_hist)) else None,
        "gamma_tilde_last": np.array(gamma_hist) if (log_spectral and len(gamma_hist)) else None,
    }


def run_episode_conventional(
    jammer: ConventionalJammer,
    radar: AdaptiveRadar,
    duration_ms: float,
    macro_step_us: float,
    adapt_time_ms: float,
    seed: int,
) -> dict:
    """Single episode with conventional jammer (gap at transition)."""
    np.random.seed(seed)
    radar.reset()
    jammer.reset()
    adapt_t = adapt_time_ms * 1e-3
    dt = macro_step_us * 1e-6
    total_steps = int((duration_ms * 1e-3) / dt)
    t_ms, Q_hist, power_db, techniques, actions = [], [], [], [], []
    for step in range(total_steps):
        t = step * dt
        t_ms.append(t * 1e3)
        if t >= adapt_t and radar.mode == "TWS":
            radar.mode = "STT"
            radar._time_since_mode_change = 0.0
        state = radar._state()
        action, tech = jammer.step(state, dt)
        radar.step(action, dt, technique="gap" if tech == "transitioning" else ("noise" if tech == "noise" else "drfm"))
        Q_hist.append(radar.Q)
        power_db.append(action_to_power_db(action))
        techniques.append(tech)
        actions.append(action.copy())
    return {
        "t_ms": np.array(t_ms),
        "Q": np.array(Q_hist),
        "power_db": np.array(power_db),
        "techniques": techniques,
        "actions": np.array(actions),
        "adapt_time_ms": adapt_time_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "cogew.yaml"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--adapt-ms", type=float, default=150.0, help="Radar adaptation time (ms)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--log-spectral", action="store_true", help="Log y_tilde and gamma_tilde for S1/S2")
    parser.add_argument("--window-pre-ms", type=float, default=None, help="If set, evaluate only a window before adaptation time")
    parser.add_argument("--window-post-ms", type=float, default=None, help="If set, evaluate only a window after adaptation time")
    parser.add_argument(
        "--save-episode-distribution",
        action="store_true",
        help="Save per-episode transition-window metrics arrays and worst-episode traces.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = ROOT / config.get("data_dir", "data/cogew")
    exp_dir = ROOT / config.get("experiment_dir", "experiments/cogew_demo")
    out_dir = Path(args.out_dir or str(exp_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    U_r, Lambda_r = load_spectrum(data_dir)
    obs_dim = config.get("obs_dim", 64)
    action_dim = config.get("action_dim", 32)
    r = config.get("r", 32)
    if U_r.shape[1] > r:
        U_r = U_r[:, :r]
        Lambda_r = Lambda_r[:r]

    controller = LOCMEWController(
        U_r,
        Lambda_r,
        obs_dim,
        action_dim,
        n_blocks=config.get("n_blocks", 2),
        omega=float(config.get("omega_init", 1.0)),
        alpha=float(config.get("alpha", 0.1)),
        gamma_max=float(config.get("gamma_max", 2.0)),
        key=jax.random.PRNGKey(args.seed),
        instrument=False,
    )
    if (exp_dir / "bc_model.pkl").exists():
        import pickle
        with open(exp_dir / "bc_model.pkl", "rb") as f:
            controller.model = pickle.load(f)

    spectral_controller = None
    if args.log_spectral:
        # Build an instrumented model that shares weights with the loaded policy.
        import jax.numpy as jnp
        U_r_j = jnp.asarray(U_r)
        Lambda_r_j = jnp.asarray(Lambda_r)
        omega = float(config.get("omega_init", 1.0))
        alpha = float(config.get("alpha", 0.1))
        gamma_max = float(config.get("gamma_max", 2.0))
        n_blocks = int(config.get("n_blocks", 2))
        inst_model = convert_cogew_model_to_instrumented(
            src_model=controller.model,
            U_r=U_r_j,
            Lambda_r=Lambda_r_j,
            n_blocks=n_blocks,
            omega=omega,
            alpha=alpha,
            gamma_max=gamma_max,
            key=jax.random.PRNGKey(args.seed + 999),
        )
        spectral_controller = LOCMEWController(
            U_r,
            Lambda_r,
            obs_dim,
            action_dim,
            n_blocks=n_blocks,
            omega=omega,
            alpha=alpha,
            gamma_max=gamma_max,
            key=jax.random.PRNGKey(args.seed + 1000),
            instrument=True,
        )
        spectral_controller.model = inst_model

    jammer = ConventionalJammer(
        n_channels=action_dim,
        transition_duration_us=config.get("conventional_gap_us", 50),
    )
    radar = AdaptiveRadar(
        alpha_acquire=config["alpha_acquire"],
        alpha_degrade=config["alpha_degrade"],
        adaptation_threshold=config["adaptation_threshold"],
    )

    duration_ms_full = float(config.get("episode_duration_ms", 500))
    macro_step_us = config.get("macro_step_us", 10)
    adapt_ms = args.adapt_ms
    window_half_width_ms = float(config.get("transition_window_half_width_ms", 0.2))

    # Optional windowed eval for speed: simulate only around the transition.
    if args.window_pre_ms is not None and args.window_post_ms is not None:
        pre = float(args.window_pre_ms)
        post = float(args.window_post_ms)
        duration_ms = pre + post
        adapt_ms_local = pre  # adaptation happens at the middle split
    else:
        duration_ms = duration_ms_full
        adapt_ms_local = adapt_ms

    # Optional: run a short spectral-logging episode around the transition.
    if spectral_controller is not None:
        spectral_duration_ms = float(config.get("spectral_log_duration_ms", 30.0))
        spectral_adapt_ms = float(config.get("spectral_log_adapt_ms", min(adapt_ms, spectral_duration_ms / 2)))
        spec = run_episode_locm_ew(
            spectral_controller,
            radar,
            spectral_duration_ms,
            macro_step_us,
            spectral_adapt_ms,
            args.seed,
            log_spectral=True,
        )
        np.savez(
            out_dir / "eval_locm_spectral.npz",
            t_ms=spec["t_ms"],
            y_tilde=spec["y_tilde"] if spec["y_tilde"] is not None else np.array([]),
            gamma_tilde_last=spec["gamma_tilde_last"] if spec["gamma_tilde_last"] is not None else np.array([]),
            adapt_time_ms=spec["adapt_time_ms"],
        )
        logger.info("Saved eval_locm_spectral.npz (duration %.1f ms)", spectral_duration_ms)

    locm_results = []
    conv_results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        locm_results.append(
            run_episode_locm_ew(
                controller,
                radar,
                duration_ms,
                macro_step_us,
                adapt_ms_local,
                seed,
                log_spectral=False,
            )
        )
        conv_results.append(run_episode_conventional(jammer, radar, duration_ms, macro_step_us, adapt_ms_local, seed))
        if (ep + 1) % 10 == 0:
            logger.info("Eval progress: %s/%s episodes", ep + 1, args.episodes)

    # Use first episode for money-shot; save all for analysis
    np.savez(
        out_dir / "eval_locm_ew.npz",
        t_ms=locm_results[0]["t_ms"],
        Q=locm_results[0]["Q"],
        power_db=locm_results[0]["power_db"],
        actions=locm_results[0]["actions"],
        techniques=locm_results[0]["techniques"],
        adapt_time_ms=locm_results[0]["adapt_time_ms"],
        y_tilde=np.array([]),
        gamma_tilde_last=np.array([]),
    )
    np.savez(
        out_dir / "eval_conventional.npz",
        t_ms=conv_results[0]["t_ms"],
        Q=conv_results[0]["Q"],
        power_db=conv_results[0]["power_db"],
        techniques=conv_results[0]["techniques"],
        actions=conv_results[0]["actions"],
        adapt_time_ms=conv_results[0]["adapt_time_ms"],
    )
    # Summary metrics
    locm_Q_max = [np.max(r["Q"]) for r in locm_results]
    conv_Q_max = [np.max(r["Q"]) for r in conv_results]
    locm_window = [
        transition_metrics(r["t_ms"], r["Q"], r["power_db"], adapt_time_ms=adapt_ms, half_width_ms=window_half_width_ms)
        for r in locm_results
    ]
    conv_window = [
        transition_metrics(r["t_ms"], r["Q"], r["power_db"], adapt_time_ms=adapt_ms, half_width_ms=window_half_width_ms)
        for r in conv_results
    ]
    locm_max_Q_w = np.array([m["max_Q_window"] for m in locm_window], dtype=np.float64)
    conv_max_Q_w = np.array([m["max_Q_window"] for m in conv_window], dtype=np.float64)
    locm_min_p_w = np.array([m["min_power_db_window"] for m in locm_window], dtype=np.float64)
    conv_min_p_w = np.array([m["min_power_db_window"] for m in conv_window], dtype=np.float64)
    metrics = {
        "locm_mean_max_Q": float(np.mean(locm_Q_max)),
        "conv_mean_max_Q": float(np.mean(conv_Q_max)),
        "transition_window_half_width_ms": window_half_width_ms,
        "locm_mean_max_Q_window": float(np.mean(locm_max_Q_w)),
        "conv_mean_max_Q_window": float(np.mean(conv_max_Q_w)),
        "locm_mean_min_power_db_window": float(np.mean(locm_min_p_w[np.isfinite(locm_min_p_w)])) if np.any(np.isfinite(locm_min_p_w)) else float("-inf"),
        "conv_mean_min_power_db_window": float(np.mean(conv_min_p_w[np.isfinite(conv_min_p_w)])) if np.any(np.isfinite(conv_min_p_w)) else float("-inf"),
        "locm_ci95_max_Q_window": _bootstrap_ci(locm_max_Q_w, seed=args.seed),
        "conv_ci95_max_Q_window": _bootstrap_ci(conv_max_Q_w, seed=args.seed + 1),
        "adapt_time_ms": float(adapt_ms_local),
        "eval_duration_ms": float(duration_ms),
        "n_episodes": args.episodes,
    }
    import json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Eval done. LOCM mean max Q %.4f, Conv mean max Q %.4f", metrics["locm_mean_max_Q"], metrics["conv_mean_max_Q"])

    if args.save_episode_distribution:
        # Save per-episode arrays for histogram/diagnostics.
        np.savez(
            out_dir / "episode_distribution.npz",
            locm_max_Q_window=locm_max_Q_w,
            conv_max_Q_window=conv_max_Q_w,
            locm_min_power_db_window=locm_min_p_w,
            conv_min_power_db_window=conv_min_p_w,
            adapt_time_ms=float(adapt_ms_local),
            window_half_width_ms=float(window_half_width_ms),
            eval_duration_ms=float(duration_ms),
            n_episodes=int(args.episodes),
        )

        # Save the single worst conventional episode (highest max_Q_window) and the corresponding LOCM episode.
        worst_idx = int(np.argmax(conv_max_Q_w))
        conv_worst = conv_results[worst_idx]
        locm_same = locm_results[worst_idx]
        np.savez(
            out_dir / "eval_conventional_worst.npz",
            t_ms=conv_worst["t_ms"],
            Q=conv_worst["Q"],
            power_db=conv_worst["power_db"],
            actions=conv_worst["actions"],
            techniques=conv_worst["techniques"],
            adapt_time_ms=conv_worst["adapt_time_ms"],
            worst_episode_index=worst_idx,
            max_Q_window=float(conv_max_Q_w[worst_idx]),
        )
        np.savez(
            out_dir / "eval_locm_match_worst.npz",
            t_ms=locm_same["t_ms"],
            Q=locm_same["Q"],
            power_db=locm_same["power_db"],
            actions=locm_same["actions"],
            techniques=locm_same["techniques"],
            adapt_time_ms=locm_same["adapt_time_ms"],
            worst_episode_index=worst_idx,
            max_Q_window=float(locm_max_Q_w[worst_idx]),
        )
        logger.info("Saved episode_distribution.npz and worst-episode traces (index=%s).", worst_idx)


if __name__ == "__main__":
    main()
